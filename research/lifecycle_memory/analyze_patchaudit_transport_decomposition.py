#!/usr/bin/env python3
"""Analyze the frozen PatchAudit transport-decomposition follow-up."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory import run_patchaudit_transport_decomposition as runner
from research.lifecycle_memory import run_phase2_reanchored_causal_patch as _base


ANALYSIS_KIND = "patchaudit_transport_decomposition_analysis"
MANIFEST_KIND = "patchaudit_transport_decomposition_analysis_manifest"
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20260726
FORBIDDEN_WORKER_FIELDS = {
    "correct_choice",
    "ego4d_participant_id",
    "participant_cluster_id",
    "correct_margin_bits",
}
OPERATOR_LABELS = {
    "single_full_frame": "single",
    "persistent_full_frame": "persistent clamp",
    "complement_once": "complement",
    "full_sequence_once": "full sequence",
}


class AnalysisError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return runner.read_jsonl(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    runner.atomic_write_json(path, payload)


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AnalysisError("percentile input is empty")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (float(item[1]), item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (family_size - index) * float(value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _bootstrap_ratio(
    cluster_pairs: Mapping[str, tuple[float, float]],
    *,
    draws: int,
    seed: int,
) -> tuple[float, list[float]]:
    clusters = sorted(cluster_pairs)
    if len(clusters) < 2:
        raise AnalysisError("recovery bootstrap requires at least two clusters")
    errors = np.asarray(
        [cluster_pairs[cluster][0] for cluster in clusters], dtype=np.float64
    )
    baselines = np.asarray(
        [cluster_pairs[cluster][1] for cluster in clusters], dtype=np.float64
    )
    denominator = float(baselines.mean())
    if not math.isfinite(denominator) or denominator <= 1e-12:
        raise AnalysisError("recovery denominator is nonpositive")
    estimate = 1.0 - float(errors.mean()) / denominator
    rng = np.random.default_rng(seed)
    bootstrap: list[float] = []
    remaining = draws
    while remaining:
        batch = min(2048, remaining)
        indexes = rng.integers(0, len(clusters), size=(batch, len(clusters)))
        sampled_error = errors[indexes].mean(axis=1)
        sampled_baseline = baselines[indexes].mean(axis=1)
        if np.any(sampled_baseline <= 1e-12):
            raise AnalysisError("bootstrap produced a nonpositive denominator")
        bootstrap.extend((1.0 - sampled_error / sampled_baseline).tolist())
        remaining -= batch
    return estimate, bootstrap


def summarize_recovery(
    cluster_pairs: Mapping[str, tuple[float, float]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    estimate, bootstrap = _bootstrap_ratio(
        cluster_pairs, draws=draws, seed=seed
    )
    clusters = sorted(cluster_pairs)
    return {
        "rho": estimate,
        "ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        "participant_cluster_count": len(clusters),
        "cluster_equal_mean_patched_donor_error_bits": mean(
            cluster_pairs[cluster][0] for cluster in clusters
        ),
        "cluster_equal_mean_recipient_donor_gap_bits": mean(
            cluster_pairs[cluster][1] for cluster in clusters
        ),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def _join_hidden(
    *,
    rows: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    hidden_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    public_map = {str(row["request_id"]): row for row in public_rows}
    hidden_map = {str(row["request_id"]): row for row in hidden_rows}
    if len(public_map) != 160 or len(hidden_map) != 160:
        raise AnalysisError(
            f"expected 160 public and hidden requests, observed "
            f"{len(public_map)} and {len(hidden_map)}"
        )
    if set(public_map) != set(hidden_map):
        raise AnalysisError("public and hidden request inventories differ")
    joined: list[dict[str, Any]] = []
    for raw in rows:
        leaked = FORBIDDEN_WORKER_FIELDS & set(raw)
        if leaked:
            raise AnalysisError(
                f"worker outcome leaked hidden fields: {sorted(leaked)}"
            )
        request_id = str(raw.get("request_id") or "")
        public = public_map.get(request_id)
        hidden = hidden_map.get(request_id)
        if public is None or hidden is None:
            raise AnalysisError(f"cannot join request {request_id}")
        if str(raw.get("frame_unit_id")) != str(public["frame_unit_id"]):
            raise AnalysisError(f"public frame mismatch for {request_id}")
        if str(raw.get("order_id")) != str(public["order_id"]):
            raise AnalysisError(f"public order mismatch for {request_id}")
        correct_choice = str(hidden.get("correct_choice") or "")
        if correct_choice not in {"A", "B"}:
            raise AnalysisError(f"invalid correct choice for {request_id}")
        participant_id = str(hidden.get("ego4d_participant_id") or "")
        if not participant_id:
            raise AnalysisError(f"missing Ego4D participant ID for {request_id}")
        sign = 1.0 if correct_choice == "A" else -1.0
        recipient = sign * float(raw["recipient_margin_a_minus_b_bits"])
        donor = sign * float(raw["donor_margin_a_minus_b_bits"])
        patched = sign * float(raw["patched_margin_a_minus_b_bits"])
        error = abs(patched - donor)
        baseline = abs(recipient - donor)
        if not all(math.isfinite(value) for value in (recipient, donor, patched)):
            raise AnalysisError(f"nonfinite margin for {request_id}")
        if abs(error - float(raw["patched_donor_error_abs_bits"])) > 1e-9:
            raise AnalysisError(f"patched-donor error drift for {request_id}")
        if abs(baseline - float(raw["baseline_donor_gap_abs_bits"])) > 1e-9:
            raise AnalysisError(f"baseline donor gap drift for {request_id}")
        joined.append(
            {
                **raw,
                "correct_choice": correct_choice,
                "ego4d_participant_id": participant_id,
                "recipient_correct_margin_bits": recipient,
                "donor_correct_margin_bits": donor,
                "patched_correct_margin_bits": patched,
                "patched_donor_error_bits": error,
                "recipient_donor_gap_bits": baseline,
            }
        )
    return joined


def _cluster_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows_per_frame: int = 2,
) -> dict[str, tuple[float, float]]:
    by_cluster_frame: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(
        list
    )
    for row in rows:
        key = (
            str(row["ego4d_participant_id"]),
            str(row["frame_unit_id"]),
        )
        by_cluster_frame[key].append(
            (
                float(row["patched_donor_error_bits"]),
                float(row["recipient_donor_gap_bits"]),
            )
        )
    bad = [
        key
        for key, values in by_cluster_frame.items()
        if len(values) != expected_rows_per_frame
    ]
    if bad:
        raise AnalysisError(
            f"endpoint has the wrong rows per frame "
            f"(expected {expected_rows_per_frame}): {bad[:8]}"
        )
    by_cluster: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (cluster, _frame), values in by_cluster_frame.items():
        by_cluster[cluster].append(
            (
                mean(value[0] for value in values),
                mean(value[1] for value in values),
            )
        )
    return {
        cluster: (
            mean(value[0] for value in values),
            mean(value[1] for value in values),
        )
        for cluster, values in by_cluster.items()
    }


def _endpoint_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model_key: str,
    corruption: str,
    direction: str,
    start_layer: int,
    operator: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["model_key"] == model_key
        and row["corruption"] == corruption
        and row["direction"] == direction
        and int(row["start_layer"]) == start_layer
        and row["operator"] == operator
    ]


def endpoint_summaries(
    rows: Sequence[dict[str, Any]],
    *,
    prespec: Mapping[str, Any],
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    counter = 0
    for model in prespec["models"]:
        model_key = str(model["model_key"])
        layers = [
            int(value)
            for value in prespec["depth_plan"][model_key][
                "ordered_post_block_indexes"
            ]
        ]
        for corruption in runner.CORRUPTIONS:
            for direction in ("rescue", "damage"):
                for layer in layers:
                    for operator in runner.OPERATORS:
                        selected = _endpoint_rows(
                            rows,
                            model_key=model_key,
                            corruption=corruption,
                            direction=direction,
                            start_layer=layer,
                            operator=operator,
                        )
                        if len(selected) != 160:
                            raise AnalysisError(
                                f"endpoint has {len(selected)} rows instead of 160: "
                                f"{model_key}/{corruption}/{direction}/{layer}/{operator}"
                            )
                        pairs = _cluster_pairs(selected)
                        if len(pairs) != 80:
                            raise AnalysisError(
                                f"endpoint has {len(pairs)} clusters instead of 80"
                            )
                        summary = summarize_recovery(
                            pairs,
                            draws=draws,
                            seed=seed + counter,
                        )
                        summaries.append(
                            {
                                "model_key": model_key,
                                "corruption": corruption,
                                "direction": direction,
                                "start_layer": layer,
                                "operator": operator,
                                **summary,
                            }
                        )
                        counter += 1
    if len(summaries) != 128:
        raise AnalysisError(
            f"expected 128 endpoint summaries, observed {len(summaries)}"
        )
    return summaries


def _primary_operator_pairs(
    rows: Sequence[dict[str, Any]],
    *,
    model_key: str,
    operator: str,
    central_layers: set[int],
) -> dict[str, tuple[float, float]]:
    selected = [
        row
        for row in rows
        if row["model_key"] == model_key
        and int(row["start_layer"]) in central_layers
        and row["operator"] == operator
        and row["corruption"] in runner.CORRUPTIONS
        and row["direction"] in {"rescue", "damage"}
    ]
    expected = 160 * 2 * 2 * len(central_layers)
    if len(selected) != expected:
        raise AnalysisError(
            f"primary pool {model_key}/{operator} has {len(selected)} rows, "
            f"expected {expected}"
        )
    pairs = _cluster_pairs(
        selected,
        expected_rows_per_frame=2 * 2 * 2 * len(central_layers),
    )
    if len(pairs) != 80:
        raise AnalysisError(
            f"primary pool {model_key}/{operator} has {len(pairs)} clusters"
        )
    return pairs


def _paired_rho_delta(
    comparator: Mapping[str, tuple[float, float]],
    single: Mapping[str, tuple[float, float]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    clusters = sorted(comparator)
    if clusters != sorted(single):
        raise AnalysisError("paired mechanism cells use different cluster inventories")
    comparator_error = np.asarray(
        [comparator[cluster][0] for cluster in clusters], dtype=np.float64
    )
    comparator_baseline = np.asarray(
        [comparator[cluster][1] for cluster in clusters], dtype=np.float64
    )
    single_error = np.asarray(
        [single[cluster][0] for cluster in clusters], dtype=np.float64
    )
    single_baseline = np.asarray(
        [single[cluster][1] for cluster in clusters], dtype=np.float64
    )
    if not np.allclose(
        comparator_baseline, single_baseline, rtol=0.0, atol=1e-12
    ):
        raise AnalysisError("paired operators have different baseline denominators")

    def rho(error: np.ndarray, baseline: np.ndarray) -> float:
        denominator = float(baseline.mean())
        if denominator <= 1e-12:
            raise AnalysisError("primary recovery denominator is nonpositive")
        return 1.0 - float(error.mean()) / denominator

    comparator_rho = rho(comparator_error, comparator_baseline)
    single_rho = rho(single_error, single_baseline)
    estimate = comparator_rho - single_rho
    rng = np.random.default_rng(seed)
    bootstrap: list[float] = []
    remaining = draws
    while remaining:
        batch = min(2048, remaining)
        indexes = rng.integers(0, len(clusters), size=(batch, len(clusters)))
        baseline = comparator_baseline[indexes].mean(axis=1)
        if np.any(baseline <= 1e-12):
            raise AnalysisError("primary bootstrap denominator is nonpositive")
        comparator_boot = 1.0 - comparator_error[indexes].mean(axis=1) / baseline
        single_boot = 1.0 - single_error[indexes].mean(axis=1) / baseline
        bootstrap.extend((comparator_boot - single_boot).tolist())
        remaining -= batch
    pvalue = (1 + sum(value <= 0.0 for value in bootstrap)) / (draws + 1)
    return {
        "estimate_delta_rho": estimate,
        "comparator_rho": comparator_rho,
        "single_rho": single_rho,
        "ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        "one_sided_bootstrap_pvalue": pvalue,
        "participant_cluster_count": len(clusters),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def primary_analysis(
    rows: Sequence[dict[str, Any]],
    *,
    prespec: Mapping[str, Any],
    draws: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    pvalues: dict[str, float] = {}
    for model_index, model in enumerate(prespec["models"]):
        model_key = str(model["model_key"])
        central = {
            int(value)
            for value in prespec["depth_plan"][model_key][
                "central_primary_post_block_indexes"
            ]
        }
        single = _primary_operator_pairs(
            rows,
            model_key=model_key,
            operator="single_full_frame",
            central_layers=central,
        )
        for mechanism_index, (mechanism, operator) in enumerate(
            (
                ("persistent_clamp_minus_single", "persistent_full_frame"),
                ("complement_minus_single", "complement_once"),
            )
        ):
            comparator = _primary_operator_pairs(
                rows,
                model_key=model_key,
                operator=operator,
                central_layers=central,
            )
            cell_id = f"{model_key}:{mechanism}"
            result = _paired_rho_delta(
                comparator,
                single,
                draws=draws,
                seed=seed + 10_000 + model_index * 100 + mechanism_index,
            )
            row = {
                "cell_id": cell_id,
                "model_key": model_key,
                "mechanism": mechanism,
                "comparator_operator": operator,
                "reference_operator": "single_full_frame",
                "central_post_block_indexes": sorted(central),
                **result,
            }
            cells.append(row)
            pvalues[cell_id] = float(result["one_sided_bootstrap_pvalue"])
    if len(cells) != 4:
        raise AnalysisError("primary Holm family must contain exactly four cells")
    adjusted = holm_adjust(pvalues)
    materiality = float(
        prespec["decision_rule"]["materiality_threshold_rho"]
    )
    for cell in cells:
        cell["holm_adjusted_pvalue"] = adjusted[cell["cell_id"]]
        cell["materiality_threshold_rho"] = materiality
        cell["positive"] = bool(
            cell["holm_adjusted_pvalue"] < 0.05
            and cell["estimate_delta_rho"] >= materiality
        )
        cell["null_tested"] = "delta_rho_less_than_or_equal_to_zero"
        cell["materiality_rule"] = (
            "point_estimate_screen_not_a_test_against_the_0.10_threshold"
        )
    by_model = defaultdict(dict)
    for cell in cells:
        by_model[cell["model_key"]][cell["mechanism"]] = cell
    taxonomy: dict[str, Any] = {}
    for model_key, model_cells in sorted(by_model.items()):
        clamp = bool(model_cells["persistent_clamp_minus_single"]["positive"])
        complement = bool(model_cells["complement_minus_single"]["positive"])
        if clamp and complement:
            call = "both_operator_increments_supported"
        elif clamp:
            call = "clamp_increment_supported_complement_unresolved"
        elif complement:
            call = "complement_increment_supported_clamp_unresolved"
        else:
            call = "no_operator_increment_resolved"
        taxonomy[model_key] = {
            "call": call,
            "persistent_clamp_cell_positive": clamp,
            "complement_cell_positive": complement,
            "nonpositive_cell_interpretation": (
                "unresolved_not_evidence_of_absence"
            ),
            "persistent_clamp_interpretation": (
                "operator_identity_not_an_estimand_of_persistence_"
                "recontamination_or_reinjection"
            ),
        }
    return cells, taxonomy


def _write_plot_csv(
    path: Path, endpoints: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_key",
        "corruption",
        "direction",
        "start_layer",
        "operator",
        "operator_label",
        "rho",
        "ci95_low",
        "ci95_high",
        "participant_cluster_count",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in endpoints:
            writer.writerow(
                {
                    "model_key": row["model_key"],
                    "corruption": row["corruption"],
                    "direction": row["direction"],
                    "start_layer": row["start_layer"],
                    "operator": row["operator"],
                    "operator_label": OPERATOR_LABELS[row["operator"]],
                    "rho": f"{float(row['rho']):.9f}",
                    "ci95_low": f"{float(row['ci95'][0]):.9f}",
                    "ci95_high": f"{float(row['ci95'][1]):.9f}",
                    "participant_cluster_count": row[
                        "participant_cluster_count"
                    ],
                }
            )
        handle.flush()
    temporary.replace(path)


def _write_table(path: Path, cells: Sequence[Mapping[str, Any]]) -> None:
    model_labels = {
        "qwen35_27b_hf_bf16": "Qwen3.5-27B",
        "gemma4_31b_hf_bf16": "Gemma4-31B",
    }
    mechanism_labels = {
        "persistent_clamp_minus_single": "Persistent clamp $-$ single",
        "complement_minus_single": "Complement $-$ single",
    }
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Contrast & $\\Delta\\rho$ & 95\\% CI & $p_{1s}$ & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for row in cells:
        low, high = row["ci95"]
        lines.append(
            f"{model_labels[row['model_key']]} & "
            f"{mechanism_labels[row['mechanism']]} & "
            f"{float(row['estimate_delta_rho']):.3f} & "
            f"[{float(low):.3f}, {float(high):.3f}] & "
            f"{float(row['one_sided_bootstrap_pvalue']):.4f} & "
            f"{float(row['holm_adjusted_pvalue']):.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_command(args: argparse.Namespace) -> int:
    repo_dir = args.repo_dir.expanduser().resolve()
    contract, prespec, contract_hashes = runner.verify_contract(
        repo_dir=repo_dir,
        contract_path=args.contract.expanduser().resolve(),
    )
    frozen_draws = int(prespec["statistics"]["bootstrap_replicates"])
    frozen_seed = int(prespec["statistics"]["bootstrap_seed"])
    if args.bootstrap_draws != frozen_draws or args.seed != frozen_seed:
        raise AnalysisError(
            "production analysis must use the frozen bootstrap configuration: "
            f"draws={frozen_draws} seed={frozen_seed}"
        )
    run_dir = args.run_dir.expanduser().resolve()
    aggregate_path = run_dir / "aggregate_summary.json"
    aggregate = read_json(aggregate_path)
    if (
        aggregate.get("kind") != "patchaudit_transport_aggregate"
        or aggregate.get("status") != "pass"
        or aggregate.get("mode") != "full"
    ):
        raise AnalysisError("analysis requires a passing complete full aggregate")
    if aggregate.get("contract") != contract_hashes:
        raise AnalysisError("aggregate contract hashes differ from analysis")
    if aggregate.get("resume_needed"):
        raise AnalysisError("analysis cannot open labels while resume is needed")
    if (
        aggregate.get("exact_cartesian_inventory_status") != "pass"
        or aggregate.get("telemetry", {}).get("status") != "pass"
        or aggregate.get("job_manifest_contract", {}).get("status") != "pass"
        or aggregate.get("production_canary_binding", {}).get("status")
        != "pass"
        or aggregate.get("worker_count") != 4
    ):
        raise AnalysisError(
            "analysis requires passing inventory, telemetry, job-manifest, "
            "and four-worker technical gates"
        )
    if any(
        worker.get("hidden_index_opened") is not False
        for worker in aggregate.get("workers", [])
    ):
        raise AnalysisError("a worker opened the hidden evaluator index")
    expected_rows = int(contract["completion"]["expected_full_outcome_rows"])
    rows: list[dict[str, Any]] = []
    for rank in range(4):
        rows.extend(read_jsonl(run_dir / f"rank_{rank:03d}" / "outcomes.jsonl"))
    identities = [runner.outcome_identity(row) for row in rows]
    if len(rows) != expected_rows or len(set(identities)) != expected_rows:
        raise AnalysisError(
            f"raw outcome contract failed rows={len(rows)} "
            f"unique={len(set(identities))} expected={expected_rows}"
        )
    rank_files = aggregate.get("rank_outcome_files")
    if not isinstance(rank_files, Mapping) or set(rank_files) != {
        "0",
        "1",
        "2",
        "3",
    }:
        raise AnalysisError("aggregate lacks four rank-bound outcome files")
    for rank in range(4):
        rank_dir = run_dir / f"rank_{rank:03d}"
        outcome_path = rank_dir / "outcomes.jsonl"
        completed_path = rank_dir / "completed_requests.jsonl"
        record = rank_files[str(rank)]
        if (
            record.get("outcomes_sha256") != sha256_file(outcome_path)
            or record.get("completed_requests_sha256")
            != sha256_file(completed_path)
            or int(record.get("outcome_rows") or -1)
            != len(read_jsonl(outcome_path))
            or int(record.get("completion_receipt_rows") or -1)
            != len(read_jsonl(completed_path))
        ):
            raise AnalysisError(f"rank {rank} outcome-file binding drifted")
    expected_full_sequence_rows = expected_rows // len(runner.OPERATORS)
    closure_tolerance = float(
        prespec["technical_gates"]["full_sequence_closure_max_abs_error_bits"]
    )
    if (
        aggregate.get("full_sequence_row_count")
        != expected_full_sequence_rows
        or aggregate.get("expected_full_sequence_row_count")
        != expected_full_sequence_rows
        or aggregate.get("full_sequence_max_abs_error_bits") is None
        or float(aggregate["full_sequence_max_abs_error_bits"])
        > closure_tolerance
    ):
        raise AnalysisError("aggregate full-sequence closure contract failed")

    data_dir = args.data_dir.expanduser().resolve()
    mechanism_path = args.mechanism_manifest.expanduser().resolve()
    public_path = args.public_manifest.expanduser().resolve()
    hidden_path = args.hidden_index.expanduser().resolve()
    expected_data = prespec["data"]
    observed_hashes = {
        "mechanism_manifest_sha256": sha256_file(mechanism_path),
        "public_requests_sha256": sha256_file(public_path),
    }
    for key, observed in observed_hashes.items():
        if observed != expected_data[key]:
            raise AnalysisError(
                f"frozen data hash mismatch {key}: "
                f"expected={expected_data[key]} observed={observed}"
            )
    _base.validate_mechanism_manifest(
        data_dir=data_dir,
        manifest_path=mechanism_path,
        public_path=public_path,
    )
    public_rows = read_jsonl(public_path)
    _base.validate_dataset(
        public_rows=public_rows,
        hidden_rows=None,
        data_dir=data_dir,
        expected_frames=80,
        expected_user_prompt=runner.EXPECTED_PROMPT,
    )

    job_manifest_path = run_dir / "job_manifest.json"
    job_manifest = read_json(job_manifest_path)
    aggregate_sha256 = sha256_file(aggregate_path)
    if (
        job_manifest.get("kind") != "patchaudit_transport_tamia_job_manifest"
        or job_manifest.get("mode") != "full"
        or job_manifest.get("worker_hidden_bind") is not False
        or job_manifest.get("hidden_index_staged_before_technical_gate")
        is not False
        or job_manifest.get("technical_gate_passed_before_hidden_stage")
        is not True
        or job_manifest.get("technical_aggregate_sha256") != aggregate_sha256
        or job_manifest.get("hidden_index_staged_after_technical_gate")
        is not True
        or job_manifest.get("hidden_index_outside_worker_data_dir") is not True
        or Path(str(job_manifest.get("hidden_index_stage_path") or "")).resolve()
        != hidden_path
    ):
        raise AnalysisError("launcher label-firewall receipt is incomplete")
    canary_binding = runner.validate_bound_canary(
        job_manifest=job_manifest,
        immutable_hashes=aggregate["immutable_hashes"],
    )
    if canary_binding != aggregate["production_canary_binding"]:
        raise AnalysisError("production canary binding differs from aggregate")
    if (
        hidden_path == data_dir
        or data_dir in hidden_path.parents
        or hidden_path == run_dir
        or run_dir in hidden_path.parents
    ):
        raise AnalysisError(
            "hidden evaluator index must be outside worker data and run trees"
        )

    observed_hidden_sha = sha256_file(hidden_path)
    if observed_hidden_sha != expected_data["hidden_evaluator_index_sha256"]:
        raise AnalysisError(
            "frozen data hash mismatch hidden_evaluator_index_sha256: "
            f"expected={expected_data['hidden_evaluator_index_sha256']} "
            f"observed={observed_hidden_sha}"
        )
    observed_hashes["hidden_evaluator_index_sha256"] = observed_hidden_sha
    mechanism = _base.validate_mechanism_manifest(
        data_dir=data_dir,
        manifest_path=mechanism_path,
        public_path=public_path,
        hidden_path=hidden_path,
        require_hidden_outside_data_dir=True,
    )
    hidden_rows = read_jsonl(hidden_path)
    dataset = _base.validate_dataset(
        public_rows=public_rows,
        hidden_rows=hidden_rows,
        data_dir=data_dir,
        expected_frames=80,
        expected_user_prompt=runner.EXPECTED_PROMPT,
    )
    joined = _join_hidden(
        rows=rows,
        public_rows=public_rows,
        hidden_rows=hidden_rows,
    )
    endpoints = endpoint_summaries(
        joined,
        prespec=prespec,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    primary_cells, taxonomy = primary_analysis(
        joined,
        prespec=prespec,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    full_sequence_rows = [
        row for row in joined if row["operator"] == "full_sequence_once"
    ]
    closure_max = max(
        float(row["patched_donor_error_bits"]) for row in full_sequence_rows
    )
    if closure_max > closure_tolerance:
        raise AnalysisError("full-sequence closure gate failed after label join")
    final_claim_scope = {
        **prespec["claim_scope"],
        "allowed": [
            (
                "persistent_clamp_equality_is_an_operator_identity_not_"
                "evidence_about_persistence_recontamination_or_reinjection"
                if item
                == "persistent_clamp_increment_consistent_with_downstream_"
                "recontamination_or_repeated_native_donor_reinjection"
                else item
            )
            for item in prespec["claim_scope"]["allowed"]
        ],
    }
    output = {
        "kind": ANALYSIS_KIND,
        "status": "pass",
        "claim_scope": final_claim_scope,
        "preregistered_claim_scope": prespec["claim_scope"],
        "contract": contract_hashes,
        "aggregate_sha256": aggregate_sha256,
        "data_hashes": observed_hashes,
        "hidden_index_opened_after_technical_gate_receipt": True,
        "worker_hidden_index_access_receipts_all_false": True,
        "launcher_label_firewall": {
            "worker_hidden_bind": False,
            "hidden_index_staged_before_technical_gate": False,
            "hidden_index_staged_after_technical_gate": True,
            "hidden_index_outside_worker_data_dir": True,
            "job_manifest_path": str(job_manifest_path),
        },
        "production_canary_binding": canary_binding,
        "dataset": dataset,
        "mechanism_validation": mechanism,
        "raw_outcome_row_count": len(rows),
        "joined_outcome_row_count": len(joined),
        "participant_cluster_count": len(
            {row["ego4d_participant_id"] for row in joined}
        ),
        "endpoint_summaries": endpoints,
        "primary_holm_family": {
            "family_size": 4,
            "cells": primary_cells,
            "taxonomy": taxonomy,
            "decision_rule": prespec["decision_rule"],
        },
        "positive_contract_control": {
            "operator": "full_sequence_once",
            "row_count": len(full_sequence_rows),
            "max_abs_patched_donor_error_bits": closure_max,
            "tolerance_bits": closure_tolerance,
            "status": "pass",
        },
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.seed,
            "unit": "ego4d_participant_id",
        },
        "created_at": runner.utc_now(),
    }
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = out_dir / str(contract["outputs"]["analysis"])
    values_path = out_dir / str(contract["outputs"]["paper_values"])
    plot_path = out_dir / str(contract["outputs"]["plot_data"])
    table_path = out_dir / str(contract["outputs"]["paper_table"])
    atomic_write_json(analysis_path, output)
    paper_values = {
        "kind": "patchaudit_transport_paper_values",
        "status": "pass",
        "primary_cells": primary_cells,
        "taxonomy": taxonomy,
        "full_sequence_control": output["positive_contract_control"],
        "claim_scope": final_claim_scope,
        "source_analysis_sha256": sha256_file(analysis_path),
    }
    atomic_write_json(values_path, paper_values)
    _write_plot_csv(plot_path, endpoints)
    _write_table(table_path, primary_cells)
    manifest = {
        "kind": MANIFEST_KIND,
        "status": "pass",
        "inputs": {
            "aggregate_summary": {
                "path": str(aggregate_path),
                "sha256": sha256_file(aggregate_path),
            },
            "mechanism_manifest": {
                "path": str(mechanism_path),
                "sha256": observed_hashes["mechanism_manifest_sha256"],
            },
            "public_manifest": {
                "path": str(public_path),
                "sha256": observed_hashes["public_requests_sha256"],
            },
            "hidden_evaluator_index": {
                "path": str(hidden_path),
                "sha256": observed_hashes["hidden_evaluator_index_sha256"],
            },
        },
        "outputs": {
            path.name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (analysis_path, values_path, plot_path, table_path)
        },
        "created_at": runner.utc_now(),
    }
    atomic_write_json(out_dir / "analysis_manifest.json", manifest)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mechanism-manifest", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--hidden-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--require-hidden-outside-data-dir",
        action="store_true",
        help=(
            "Compatibility flag; production filesystem separation is always "
            "enforced."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap_draws < 100:
        raise SystemExit("--bootstrap-draws must be at least 100")
    try:
        return analyze_command(args)
    except (AnalysisError, runner.GateError, _base.GateError) as exc:
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
