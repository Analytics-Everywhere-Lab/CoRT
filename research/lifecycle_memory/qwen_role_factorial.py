#!/usr/bin/env python3
"""Prospective new-operator followup on the reused held-out Qwen cohort.

Workers never open evaluator labels. Runtime operations and statistical games
reuse the held-out component experiment; this module owns only the followup
design, receipts and inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from research.lifecycle_memory import analyze_patchaudit_heldout_component_closure as analysis
from research.lifecycle_memory import run_patchaudit_heldout_component_closure as old

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_KEY = "qwen35_27b_hf_bf16"
REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
MODEL = {"model_key": MODEL_KEY, "model_id": "Qwen/Qwen3.5-27B",
         "revision": REVISION, "processor_revision": REVISION, "dtype": "bfloat16"}
PUBLIC_SHA256 = "2b0e31236a0ea531a4a9849b00356e2fabfa153084dd12487511b2523d50635f"
FACTORIAL = old.GEMMA_FACTORIAL_OPERATORS
OPERATORS = ("F", *FACTORIAL, "A")
ROWS_PER_REQUEST = 68
CLAIM_SCOPE = (
    "Prospective new-operator followup on the reused 46-frame Job E cohort; "
    "not pristine heldout confirmation and no upgrade of the old Qwen reversal."
)
GateError = old.GateError
read_json = analysis.read_json
read_jsonl = analysis.read_jsonl
sha256_json = old.sha256_json
sha256_file = old.sha256_file


def finite_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise GateError("nonfinite value in artifact")
    if isinstance(value, Mapping):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tree(item)


def reject_labels(value: Any) -> None:
    if isinstance(value, Mapping):
        if set(value) & analysis.FORBIDDEN_WORKER_FIELDS:
            raise GateError("evaluator fields forbidden in worker inputs/outputs")
        for item in value.values():
            reject_labels(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_labels(item)


def contained(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise GateError(f"path escapes its root: {name}")
    return path


def check_hash(path: Path, expected: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
        raise GateError(f"invalid SHA256 for {path}")
    if not path.is_file() or sha256_file(path) != expected:
        raise GateError(f"hash mismatch or missing file: {path}")
    return expected


def make_prespec(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "depth_plan": {MODEL_KEY: {"ordered_post_block_indexes": [36],
                                   "early_post_block_index": 36,
                                   "late_post_block_index": 36}},
        "operator_override": {MODEL_KEY: {"36": list(OPERATORS)}},
        "technical_gates": dict(config["technical_gates"]),
    }


def validate_config(config: Mapping[str, Any]) -> None:
    finite_tree(config)
    reject_labels(config)
    if (config.get("schema_version") != 1 or config.get("kind") != "qwen_role_factorial"
            or config.get("model") != MODEL or config.get("post_block_index") != 36
            or config.get("prompt") != old.EXPECTED_PROMPT):
        raise GateError("frozen Qwen model/prompt/design mismatch")
    if config["public_requests"]["sha256"] != PUBLIC_SHA256:
        raise GateError("not the frozen reused Job E cohort")
    stats = config["analysis"]
    if (stats["min_donor_gap_bits"] != 1e-12 or stats["min_k_recovery"] != .10
            or stats["min_fraction_denominator"] != .10
            or type(stats["bootstrap_draws"]) is not int
            or not 100 <= stats["bootstrap_draws"] <= 100_000
            or type(stats["seed"]) is not int):
        raise GateError("invalid frozen analysis guards/bootstrap settings")
    for field in ("self_patch_max_abs_error_bits", "full_sequence_closure_max_abs_error_bits",
                  "answer_token_equivalence_tolerance_bits"):
        value = config["technical_gates"][field]
        if isinstance(value, bool) or value != .001:
            raise GateError(f"invalid technical tolerance: {field}")
    for field in ("sif_sha256", "slurm_script_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(config["provenance"][field])):
            raise GateError(f"missing provenance {field}")


def shard_requests(rows: Sequence[Mapping[str, Any]], rank: int,
                   world_size: int = 4, mode: str = "full") -> list[dict[str, Any]]:
    if world_size != 4 or not 0 <= rank < 4 or mode not in {"full", "canary"}:
        raise GateError("requires four independent GPU workers and a valid mode/rank")
    frames: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    ids = [row["request_id"] for row in rows]
    if len(ids) != 92 or len(set(ids)) != 92:
        raise GateError("requires exactly 92 unique requests")
    for row in rows:
        frames[str(row["frame_unit_id"])].append(row)
    if len(frames) != 46:
        raise GateError("requires exactly 46 frames")
    for pair in frames.values():
        if len(pair) != 2 or {r["order_id"] for r in pair} != {"order_0", "order_1"}:
            raise GateError("each frame requires both orders")
        if any(r["counterbalanced_request_id"] != pair[1-i]["request_id"] for i, r in enumerate(pair)):
            raise GateError("counterbalanced request pairing mismatch")
    selected = sorted(frames)[rank::world_size]
    if mode == "canary":
        selected = selected[:1]
    return [dict(row) for frame in selected
            for row in sorted(frames[frame], key=lambda row: row["order_id"])]


def load_inputs(config_path: Path, data_dir: Path, *, repo: Path = REPO_ROOT
                ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config = read_json(config_path)
    validate_config(config)
    source_files = config["source_files"]
    required = {"research/lifecycle_memory/" + name for name in (
        "qwen_role_factorial.py", "run_patchaudit_heldout_component_closure.py",
        "analyze_patchaudit_heldout_component_closure.py", "patchaudit_component_partition.py",
        "patchaudit_transport_operators.py", "run_phase2_reanchored_causal_patch.py",
        "run_phase2_reanchored_answer_token_causal_patch.py")}
    # Every already imported local dependency must belong to the frozen release.
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(repo.resolve() / "research"):
            required.add(str(Path(filename).resolve().relative_to(repo.resolve())))
    if not required <= source_files.keys():
        raise GateError(f"unbound source dependencies: {sorted(required - source_files.keys())}")
    for name, digest in source_files.items():
        check_hash(contained(repo, name), digest)
    public = config["public_requests"]
    public_path = contained(repo, public["path"])
    check_hash(public_path, public["sha256"])
    rows = read_jsonl(public_path)
    reject_labels(rows)
    shard_requests(rows, 0)
    assets: dict[str, str] = {}
    for row in rows:
        if row["user_prompt"] != old.EXPECTED_PROMPT:
            raise GateError("public prompt drift")
        if set(row["arms"]) != set(old.ARMS):
            raise GateError("corruption arm drift")
        for arm in old.ARMS:
            for asset in row["arms"][arm]["inputs"]:
                name, digest = asset["path"], asset["sha256"]
                if name in assets and assets[name] != digest:
                    raise GateError("contradictory asset hash")
                assets[name] = digest
    if config["data_files"] != assets:
        raise GateError("data inventory differs from public assets")
    for name, digest in assets.items():
        check_hash(contained(data_dir, name), digest)
    manifest = config["model_manifest"]
    check_hash(contained(repo, manifest["path"]), manifest["sha256"])
    files = read_json(contained(repo, manifest["path"]))["files"]
    if not files or len({entry["path"] for entry in files}) != len(files):
        raise GateError("empty or duplicated model inventory")
    prefix = f"hub/models--Qwen--Qwen3.5-27B/snapshots/{REVISION}/"
    for entry in files:
        if (not entry["path"].startswith(prefix) or ".." in Path(entry["path"]).parts
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                or type(entry["bytes"]) is not int or entry["bytes"] <= 0):
            raise GateError("invalid pinned model inventory")
    binding = {"config_sha256": sha256_file(config_path), "model": MODEL,
               "source_files": source_files, "data_files": assets,
               "public_requests_sha256": public["sha256"],
               "model_manifest_sha256": manifest["sha256"],
               "provenance": config["provenance"]}
    return config, rows, binding


def validate_geometry(partition: Mapping[str, Any]) -> None:
    expected = {
        "active": list(range(926)), "full_frame": list(range(48, 636)),
        "candidate_crops": list(range(638, 738)) + list(range(740, 840)),
        "literal_user_prompt": list(range(841, 914)), "answer_prefix": [923, 924, 925],
        "structural_remainder": list(range(48)) + [636, 637, 738, 739, 840] + list(range(914, 923)),
        "complement": list(range(48)) + list(range(636, 926)), "padding": [],
    }
    if partition.get("sequence_length") != 926:
        raise GateError("Qwen sequence geometry drift")
    for name, positions in expected.items():
        group = partition.get(name, {})
        if list(group.get("positions", ())) != positions:
            raise GateError(f"Qwen frozen {name} geometry drift")
    if partition.get("answer_prefix_start") != 923:
        raise GateError("Qwen answer readout geometry drift")


def validate_rows(rows: Sequence[Mapping[str, Any]], request: Mapping[str, Any],
                  rank: int, config: Mapping[str, Any]) -> None:
    finite_tree(rows)
    reject_labels(rows)
    expected = {(c, d, op) for c in old.CORRUPTIONS for d in ("rescue", "damage") for op in OPERATORS}
    keys = [(r["corruption"], r["direction"], r["operator"]) for r in rows]
    if len(rows) != ROWS_PER_REQUEST or set(keys) != expected:
        raise GateError("missing/duplicate/unexpected factorial rows")
    native: dict[str, tuple[float, float]] = {}
    shared_real = rows[0]["real_margin_a_minus_b_bits"]
    for row in rows:
        old._validate_outcome_row(row, rank=rank, model_key=MODEL_KEY, public_record=request)
        if row["start_layer"] != 36:
            raise GateError("non-L36 outcome")
        margins = (row["real_margin_a_minus_b_bits"], row["corrupt_margin_a_minus_b_bits"])
        if margins[0] != shared_real:
            raise GateError("real native margin differs between corruptions")
        if row["corruption"] in native and native[row["corruption"]] != margins:
            raise GateError("native margin drift across operators/directions")
        native[row["corruption"]] = margins
        if row["operator"] == "A" and row["patched_donor_error_abs_bits"] > config["technical_gates"]["full_sequence_closure_max_abs_error_bits"]:
            raise GateError("all-active closure failed")


def validate_detail_rows(payload: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    detail = payload["detail"]
    frozen = detail["partition"]
    validate_geometry(frozen)
    partition = old.build_component_partition(sequence_length=frozen["sequence_length"],
        active_positions=frozen["active"]["positions"],
        image_slot_positions=[slot["positions"] for slot in frozen["image_slots"]],
        literal_user_prompt_start=frozen["literal_user_prompt_start"],
        literal_user_prompt_count=frozen["literal_user_prompt_count"],
        answer_prefix_token_count=frozen["answer_prefix_token_count"])
    if sha256_json(partition.as_dict()) != sha256_json(frozen):
        raise GateError("partition derived metadata/hash mismatch")
    for row in payload["rows"]:
        positions, receipt = old._operator_positions(operator=row["operator"], model_key=MODEL_KEY,
            layer=36, partition=partition, prespec=make_prespec(config))
        if (row["operator_position_count"] != len(positions)
                or row["operator_positions_sha256"] != sha256_json(list(positions))
                or row["operator_receipt"] != receipt):
            raise GateError("operator geometry receipt mismatch")
        summary = row.get("partition", {})
        expected = {"sequence_length": 926, "active_count": 926, "full_frame_count": 588,
            "complement_count": 338, "padding_count": 0, "active_sha256": partition.active.sha256,
            "full_frame_sha256": partition.full_frame.sha256, "complement_sha256": partition.complement.sha256,
            "component_partition_sha256": sha256_json(partition.as_dict())}
        if summary != expected:
            raise GateError("outcome partition summary differs from derived geometry")


def request_path(rank_dir: Path, request_id: str) -> Path:
    return rank_dir / "requests" / (sha256_json(request_id) + ".json")


def validate_equivalence(equivalence: Mapping[str, Any], tolerance: float) -> None:
    finite_tree(equivalence)
    choices = equivalence.get("choices", {})
    if (equivalence.get("status") != "pass" or set(choices) != {"A", "B"}
            or equivalence.get("label_joined") is not False
            or equivalence.get("construction") != "one_shared_prefix_forward_vs_exact_appended_suffix"):
        raise GateError("answer-token equivalence identity/inventory failed")
    errors, ids = [], []
    for choice in ("A", "B"):
        row = choices[choice]
        tokens = row.get("answer_token_ids", [])
        if len(tokens) != 1 or type(tokens[0]) is not int or tokens[0] < 0:
            raise GateError("equivalence answer must be one nonnegative integer token")
        ids.append(tokens[0])
        values = [row.get(key) for key in ("one_pass_logprob", "suffix_logprob", "absolute_difference_bits")]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise GateError("nonfinite/invalid equivalence probability")
        one_pass, suffix, reported = values
        error = abs(one_pass - suffix) / math.log(2.)
        if one_pass > 0 or suffix > 0 or abs(reported - error) > 1e-12 or error > tolerance:
            raise GateError("answer-token equivalence error algebra/tolerance failed")
        errors.append(error)
    maximum = equivalence.get("maximum_absolute_difference_bits")
    if (len(set(ids)) != 2 or type(maximum) not in (int, float)
            or not math.isfinite(maximum) or abs(maximum - max(errors)) > 1e-12):
        raise GateError("answer-token equivalence maximum/token identity failed")


def save_request(path: Path, payload: Mapping[str, Any]) -> None:
    finite_tree(payload)
    old.atomic_write_json(path, {"payload": payload, "payload_sha256": sha256_json(payload)})


def read_request(path: Path, *, request: Mapping[str, Any], rank: int,
                 binding_sha256: str, config: Mapping[str, Any], mode: str) -> dict[str, Any]:
    envelope = read_json(path)
    payload = envelope["payload"]
    if envelope["payload_sha256"] != sha256_json(payload):
        raise GateError("request atomic receipt hash mismatch")
    if (payload["binding_sha256"] != binding_sha256 or payload["request_id"] != request["request_id"]
            or payload["mode"] != mode or payload["rank"] != rank):
        raise GateError("request atomic receipt binding mismatch")
    finite_tree(payload)
    reject_labels(payload)
    validate_rows(payload["rows"], request, rank, config)
    detail = payload["detail"]
    validate_detail_rows(payload, config)
    if mode == "canary":
        checks = detail["self_checks"]
        if len(checks) != 17 or {c["operator"] for c in checks} != set(OPERATORS):
            raise GateError("incomplete self-patch inventory")
        for check in checks:
            if (check["status"] != "pass" or check["start_layer"] != 36
                    or not 0 <= check["margin_error_bits"] <= config["technical_gates"]["self_patch_max_abs_error_bits"]
                    or not all(check["audit"].get(key) is True for key in ("all_fired_once", "all_removed", "all_finite"))
                    or check["audit"]["status"] != "pass"):
                raise GateError("self-patch check failed")
        equivalence = payload["answer_token_equivalence"]
        validate_equivalence(equivalence, config["technical_gates"]["answer_token_equivalence_tolerance_bits"])
    return payload


def analyze_rows(rows: Sequence[Mapping[str, Any]], public: Sequence[Mapping[str, Any]],
                 hidden: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    joined = analysis._join_hidden(rows=rows, public_rows=public, hidden_rows=hidden)
    # Reused component-cohort absolute-error estimand; labels define clusters, not row ratios.
    for row in joined:
        row["patched_donor_error_bits"] = row["patched_donor_error_abs_bits"]
        row["recipient_donor_gap_bits"] = row["baseline_donor_gap_abs_bits"]
    stats = config["analysis"]
    indexes = analysis._bootstrap_indexes(clusters=46, draws=stats["bootstrap_draws"], seed=stats["seed"])
    grouped = {op: [row for row in joined if row["operator"] == op] for op in OPERATORS}
    points: dict[str, float] = {}
    boot: dict[str, np.ndarray] = {}
    result: dict[str, Any] = {"claim_scope": CLAIM_SCOPE,
        "estimand": "1 - mean(cluster mean absolute donor error) / mean(cluster mean absolute donor gap)",
        "pooling": "both orders, corruptions and directions; 8 rows per cluster/operator",
        "component_order": ["C", "Q", "P", "R"], "K_alias": "CQPR",
        "QPR_description": "non-candidate complement (includes structural special tokens)",
        "bootstrap_draws": stats["bootstrap_draws"], "seed": stats["seed"],
        "family_alpha": .05, "one_sided_confidence": .975, "hypothesis_coefficient": .75,
        "guard_rule": "pooled donor gap > 1e-12 and K one-sided 97.5% lower bound > 0.10; magnitude guard, not inherited proof"}
    # Permit zero-gap individual clusters; only pooled and resampled denominators matter.
    clusters = sorted({r["participant_cluster_id"] for r in joined})
    vectors = {}
    for op, selected in grouped.items():
        by_cluster = {cluster: [r for r in selected if r["participant_cluster_id"] == cluster] for cluster in clusters}
        if any(len(values) != 8 for values in by_cluster.values()):
            raise GateError("analysis cell cluster inventory differs from eight paired observations")
        error = np.asarray([np.mean([r["patched_donor_error_bits"] for r in by_cluster[c]]) for c in clusters])
        baseline = np.asarray([np.mean([r["recipient_donor_gap_bits"] for r in by_cluster[c]]) for c in clusters])
        vectors[op] = analysis.ClusterVector(tuple(clusters), error, baseline, 8)
    baseline = vectors["CQPR"].baseline
    if any(not np.array_equal(vector.baseline, baseline) for vector in vectors.values()):
        raise GateError("unpaired operator denominators")
    gap = float(baseline.mean())
    result["mean_donor_gap_bits"] = gap
    result["raw_cluster_means"] = {op: {"donor_error_bits": float(vector.error.mean()),
        "donor_gap_bits": float(vector.baseline.mean())} for op, vector in vectors.items()}
    if gap <= stats["min_donor_gap_bits"] or np.any(baseline[indexes].mean(axis=1) <= stats["min_donor_gap_bits"]):
        result.update(status="descriptive_guard_failed", guard_status="invalid_donor_gap",
                      operators={op: {"rho": None, "status": "invalid_donor_gap"} for op in OPERATORS},
                      primary=None, factorial=None)
        return result
    for op, vector in vectors.items():
        points[op] = analysis._rho(vector.error, vector.baseline)
        boot[op] = analysis._bootstrap_rho(vector, indexes)
    lower_k = float(np.quantile(boot["CQPR"], .025))
    guard = lower_k > stats["min_k_recovery"]
    result.update(status="interpretable" if guard else "descriptive_guard_failed",
                  guard_status="pass" if guard else "K_lower_bound_not_above_0.10",
                  K_recovery=points["CQPR"], K_lower_97_5=lower_k,
                  operators={op: {"rho": points[op], "lower_97_5": float(np.quantile(boot[op], .025))} for op in OPERATORS})
    contrasts = {
        "noncandidate_below_75pct_K": (.75 * points["CQPR"] - points["QPR"], .75 * boot["CQPR"] - boot["QPR"], ".75*rho(K)-rho(QPR)>0"),
        "prefix_excluded_above_75pct_K": (points["CQR"] - .75 * points["CQPR"], boot["CQR"] - .75 * boot["CQPR"], "rho(CQR)-.75*rho(K)>0"),
    }
    result["primary"] = {}
    for name, (point, samples, hypothesis) in contrasts.items():
        lower = float(np.quantile(samples, .025))
        if not math.isfinite(point) or not math.isfinite(lower):
            raise GateError("nonfinite primary effect or bound")
        result["primary"][name] = {"hypothesis": hypothesis, "signed_contrast": point,
            "lower_97_5": lower, "status": ("supported" if lower > 0 else "not_supported") if guard else "descriptive_guard_failed"}
    values = np.asarray([0.] + [points[op] for op in FACTORIAL])
    bootstrap_values = np.column_stack([np.zeros(len(indexes))] + [boot[op] for op in FACTORIAL])
    shapley = analysis._shapley_from_values(values)
    mobius = analysis._mobius_from_values(values)
    bootstrap_shapley = analysis._shapley_from_values(bootstrap_values)
    bootstrap_mobius = analysis._mobius_from_values(bootstrap_values)
    denominator_valid = (guard and points["CQPR"] > stats["min_fraction_denominator"]
                         and np.all(boot["CQPR"] > stats["min_fraction_denominator"]))
    ci = lambda samples: np.quantile(samples, [.025, .975], axis=0).T.tolist()
    for op in OPERATORS:
        result["operators"][op]["descriptive_ci95"] = ci(boot[op])
    result["factorial"] = {"status": "descriptive", "empty_set_recovery": 0.,
        "shapley": dict(zip(("C", "Q", "P", "R"), shapley.tolist())),
        "mobius": dict(zip(("empty", *FACTORIAL), mobius.tolist())),
        "shapley_descriptive_ci95": dict(zip(("C", "Q", "P", "R"), ci(bootstrap_shapley))),
        "mobius_descriptive_ci95": dict(zip(("empty", *FACTORIAL), ci(bootstrap_mobius))),
        "CQ_recovery_fraction": points["CQ"] / points["CQPR"] if denominator_valid else None,
        "CQ_recovery_fraction_descriptive_ci95": ci(boot["CQ"] / boot["CQPR"]) if denominator_valid else None,
        "CQ_shapley_fraction": float((shapley[0] + shapley[1]) / points["CQPR"]) if denominator_valid else None,
        "CQ_shapley_fraction_descriptive_ci95": ci((bootstrap_shapley[:,0] + bootstrap_shapley[:,1]) / boot["CQPR"]) if denominator_valid else None,
        "CQ_fraction_status": "descriptive_valid_denominator" if denominator_valid else "invalid_denominator_or_guard"}
    finite_tree(result)
    return result


def collect_run(run_dir: Path, mode: str, config: Mapping[str, Any], public: Sequence[Mapping[str, Any]],
                binding: Mapping[str, Any], *, mode_dir: Path | None = None
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    receipts = []
    digest = sha256_json(binding)
    directory = mode_dir or run_dir / mode
    if {p.name for p in directory.glob("rank_*")} != {f"rank_{r:03d}" for r in range(4)}:
        raise GateError("missing/extra rank directories")
    for rank in range(4):
        rank_dir = (mode_dir or run_dir / mode) / f"rank_{rank:03d}"
        manifest = read_json(rank_dir / "worker_manifest.json")
        selected = shard_requests(public, rank, mode=mode)
        if (manifest.get("status") != "pass" or manifest.get("rank") != rank
                or manifest.get("mode") != mode or manifest.get("world_size") != 4
                or manifest.get("binding") != binding or manifest.get("binding_sha256") != digest
                or manifest.get("hidden_index_opened") is not False
                or manifest.get("selected_request_ids") != [r["request_id"] for r in selected]):
            raise GateError(f"rank {rank} incomplete or binding drift")
        expected_paths = {request_path(rank_dir, r["request_id"]) for r in selected}
        if set((rank_dir / "requests").glob("*.json")) != expected_paths:
            raise GateError("missing/extra committed request bundles")
        file_hashes = {}
        for request in selected:
            path = request_path(rank_dir, request["request_id"])
            payload = read_request(path, request=request, rank=rank, binding_sha256=digest, config=config, mode=mode)
            rows.extend(payload["rows"])
            file_hashes[path.name] = sha256_file(path)
        if (manifest.get("request_files") != file_hashes
                or manifest.get("requests_completed") != len(selected)
                or manifest.get("outcome_rows") != len(selected) * ROWS_PER_REQUEST):
            raise GateError("rank manifest output inventory/hash drift")
        finite_tree(manifest)
        for name in ("geometry", "self_patch", "all_active_closure", "answer_token_equivalence"):
            expected = "bound_canary_pass" if mode == "full" and name in {"self_patch", "answer_token_equivalence"} else "pass"
            if manifest.get("checks", {}).get(name) != expected:
                raise GateError(f"rank {rank} technical check mismatch: {name}")
        fp = manifest.get("runtime_fingerprint", {})
        if (fp.get("model_revision") != REVISION or fp.get("processor_revision") != REVISION
                or not fp.get("torch") or not fp.get("transformers")):
            raise GateError("runtime fingerprint absent or revision mismatch")
        for name in ("model_load_elapsed_s", "work_elapsed_s"):
            if not isinstance(manifest.get(name), (int, float)) or manifest[name] < 0:
                raise GateError("invalid elapsed receipt")
        receipts.append(manifest)
    if len({sha256_json(m["runtime_fingerprint"]) for m in receipts}) != 1:
        raise GateError("runtime fingerprint differs across ranks")
    expected = 544 if mode == "canary" else 6256
    if len(rows) != expected:
        raise GateError("full output row inventory mismatch")
    return rows, {"kind": "qwen_role_factorial_aggregate", "status": "pass", "mode": mode,
                  "outcome_rows": len(rows), "ranks": receipts, "binding": binding,
                  "binding_sha256": digest, "technical_only": True, "claim_scope": CLAIM_SCOPE}


def validate_canary(run_dir: Path, config: Mapping[str, Any], public: Sequence[Mapping[str, Any]],
                    binding: Mapping[str, Any]) -> None:
    external = os.environ.get("QRF_CANARY_DIR")
    collect_run(run_dir, "canary", config, public, binding,
                mode_dir=Path(external) if external else None)


def write_progress(rank_dir: Path, *, mode: str, rank: int, done: int,
                   expected: int, elapsed: float, status: str) -> None:
    if not 0 <= done <= expected or not math.isfinite(elapsed) or elapsed < 0:
        raise GateError("invalid rank progress")
    old.atomic_write_json(rank_dir / "rankprogress.json", {
        "mode": mode, "rank": rank, "requests_done": done, "requests_expected": expected,
        "outcomes_done": done * ROWS_PER_REQUEST, "outcomes_expected": expected * ROWS_PER_REQUEST,
        "elapsed_s": elapsed, "status": status, "updated_at": old.utc_now()})


def worker_command(args: argparse.Namespace) -> int:
    config, public, binding = load_inputs(args.config, args.data_dir)
    selected = shard_requests(public, args.rank, args.world_size, args.mode)
    for key, env in (("sif_sha256", "SIF_SHA256"), ("slurm_script_sha256", "SLURM_SCRIPT_SHA256")):
        if old.required_hash_env(env) != config["provenance"][key]:
            raise GateError("runtime environment differs from release provenance")
    if args.mode == "full":
        validate_canary(args.run_dir, config, public, binding)
    rank_dir = args.run_dir / args.mode / f"rank_{args.rank:03d}"
    manifest_path = rank_dir / "worker_manifest.json"
    digest = sha256_json(binding)
    manifest = {"kind": "qwen_role_factorial_worker", "status": "running", "mode": args.mode,
        "rank": args.rank, "world_size": 4, "binding": binding, "binding_sha256": digest,
        "selected_request_ids": [r["request_id"] for r in selected], "hidden_index_opened": False}
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if any(previous.get(key) != manifest[key] for key in ("binding", "binding_sha256", "mode", "rank", "selected_request_ids")):
            raise GateError("resume manifest mismatch")
    elif list((rank_dir / "requests").glob("*.json")):
        raise GateError("orphaned request receipts without manifest")
    for unfinished in (rank_dir / "requests").glob(".*.json.tmp.*"):
        unfinished.unlink()
    expected_paths = {request_path(rank_dir, r["request_id"]) for r in selected}
    if set((rank_dir / "requests").glob("*.json")) - expected_paths:
        raise GateError("unexpected resume requests")
    completed = {}
    for request in selected:
        path = request_path(rank_dir, request["request_id"])
        if path.exists():
            completed[request["request_id"]] = read_request(path, request=request, rank=args.rank,
                binding_sha256=digest, config=config, mode=args.mode)
    old.atomic_write_json(manifest_path, manifest)
    import torch
    import transformers
    torch.manual_seed(config["analysis"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["analysis"]["seed"])
    started = time.perf_counter()
    write_progress(rank_dir, mode=args.mode, rank=args.rank, done=len(completed),
                   expected=len(selected), elapsed=0., status="loading_model")
    runtime, model, processor, tokenizer, device, provenance = old._base._load_anchor_model(
        model_key=MODEL_KEY, dtype="bf16", local_files_only=True)
    load_seconds = time.perf_counter() - started
    snapshot = provenance.get("model_snapshot", {})
    if snapshot.get("status") != "pass" or snapshot.get("resolved_snapshot_revision") != REVISION:
        raise GateError("model snapshot revision unverified")
    for component in (getattr(model, "config", None), processor, tokenizer):
        commit = getattr(component, "_commit_hash", None) or getattr(component, "init_kwargs", {}).get("_commit_hash")
        if commit and commit != REVISION:
            raise GateError("model/processor/tokenizer revision mismatch")
    prespec = make_prespec(config)
    capture_spec = {**prespec, "depth_plan": {MODEL_KEY: {"ordered_post_block_indexes": list(range(64))}}}
    sites, by_layer, architecture = old._decoder_sites(model, model_key=MODEL_KEY, prespec=capture_spec)
    if len(sites) != 64 or set(by_layer) != set(range(64)):
        raise GateError("Qwen native residual probe requires all 64 decoder blocks")
    sites = [by_layer[36]]
    work_started = time.perf_counter()
    write_progress(rank_dir, mode=args.mode, rank=args.rank, done=len(completed),
                   expected=len(selected), elapsed=work_started-started, status="running")
    for request in selected:
        if request["request_id"] in completed:
            continue
        geometry = old.geometry_preflight(runtime=runtime, model=model, processor=processor,
            tokenizer=tokenizer, device=device, model_key=MODEL_KEY, rows=[request],
            data_dir=args.data_dir, prespec=prespec, rank=args.rank)
        validate_geometry(geometry["records"][0]["partition"])
        equivalence = None
        if args.mode == "canary":
            equivalence = old._answer._answer_token_equivalence_audit(runtime=runtime, model=model,
                processor=processor, tokenizer=tokenizer, device=device, model_key=MODEL_KEY,
                request=request, data_dir=args.data_dir, candidate_completions={"A": "A", "B": "B"})
        outcomes, detail = old.evaluate_request(runtime=runtime, model=model, processor=processor,
            tokenizer=tokenizer, device=device, model_key=MODEL_KEY, rank=args.rank, request=request,
            data_dir=args.data_dir, sites=sites, sites_by_layer=by_layer, prespec=prespec,
            canary=args.mode == "canary")
        payload = {"binding_sha256": digest, "request_id": request["request_id"], "mode": args.mode,
                   "rank": args.rank, "rows": outcomes, "detail": detail,
                   "answer_token_equivalence": equivalence}
        path = request_path(rank_dir, request["request_id"])
        # Validate before atomic commit; an invalid committed request must never be resumable.
        validate_rows(outcomes, request, args.rank, config)
        validate_geometry(detail["partition"])
        save_request(path, payload)
        completed[request["request_id"]] = read_request(path, request=request, rank=args.rank,
            binding_sha256=digest, config=config, mode=args.mode)
        write_progress(rank_dir, mode=args.mode, rank=args.rank, done=len(completed),
                       expected=len(selected), elapsed=time.perf_counter()-started, status="running")
    manifest.update(status="pass", model_load_elapsed_s=load_seconds,
        work_elapsed_s=time.perf_counter() - work_started, requests_completed=len(completed),
        outcome_rows=len(completed) * ROWS_PER_REQUEST,
        request_files={path.name: sha256_file(path) for path in sorted(expected_paths)},
        checks={"geometry": "pass", "self_patch": "pass" if args.mode == "canary" else "bound_canary_pass",
                "all_active_closure": "pass", "answer_token_equivalence": "pass" if args.mode == "canary" else "bound_canary_pass"},
        runtime_fingerprint={"torch": torch.__version__, "transformers": transformers.__version__,
            "model_revision": REVISION, "processor_revision": REVISION,
            "processor_revision_evidence": "same pinned offline loader snapshot; cache inventory verified by release harness"},
        architecture=architecture, model_provenance=provenance)
    finite_tree(manifest)
    old.atomic_write_json(manifest_path, manifest)
    write_progress(rank_dir, mode=args.mode, rank=args.rank, done=len(completed),
                   expected=len(selected), elapsed=time.perf_counter()-started, status="done")
    return 0


def results_markdown(result: Mapping[str, Any]) -> str:
    lines = ["# Qwen Role Factorial", "", f"Technical completion: {result['status']}.",
             f"Mode: {result['mode']}; validated outcome rows: {result['outcome_rows']}.",
             "", CLAIM_SCOPE, ""]
    if result["technical_only"]:
        return "\n".join(lines + ["Technical-only summary. Evaluator labels were not opened; no effect estimates or hypothesis decisions.", ""])
    report = result["analysis"]
    lines += [f"Analysis status: {report['status']}.", f"Guard: {report['guard_rule']}",
              f"Guard status: {report['guard_status']}.",
              f"Mean donor gap: {report['mean_donor_gap_bits']:.6g} bits."]
    if "K_recovery" in report:
        lines += [f"K recovery: {report['K_recovery']:.6g}; one-sided 97.5% lower bound: {report['K_lower_97_5']:.6g}."]
    lines += ["", "## Prespecified Contrasts", "", "Bonferroni: two one-sided 97.5% cluster-bootstrap lower bounds; joint family alpha 0.05."]
    for name, item in (report.get("primary") or {}).items():
        lines.append(f"- {name}: {item['hypothesis']}; signed contrast {item['signed_contrast']:.6g}; lower bound {item['lower_97_5']:.6g}; {item['status']}.")
    if report.get("primary") is None:
        lines.append("Not interpretable: invalid donor-gap denominator. No hypothesis claim.")
    lines += ["", "## Descriptive Subsets", "", "Subset | Recovery | Paired-bootstrap 95% interval", "--- | --- | ---"]
    for operator in FACTORIAL:
        item = report["operators"][operator]
        bounds = item.get("descriptive_ci95")
        value = "not interpretable" if item["rho"] is None else f"{item['rho']:.6g}"
        interval = "not interpretable" if bounds is None else f"[{bounds[0]:.6g}, {bounds[1]:.6g}]"
        lines.append(f"{operator} | {value} | {interval}")
    factorial = report.get("factorial")
    if factorial:
        lines += ["", "CQ coalition recovery fraction = rho(CQ)/rho(K): " + str(factorial["CQ_recovery_fraction"]),
                  "CQ Shapley fraction = (phi_C+phi_Q)/rho(K): " + str(factorial["CQ_shapley_fraction"]),
                  "These are distinct quantities. Fraction status: " + factorial["CQ_fraction_status"] + ".",
                  "Descriptive Shapley/Mobius intervals and fraction intervals are in aggregate.json; no multiplicity-adjusted claims."]
    return "\n".join(lines) + "\n"


def aggregate_command(args: argparse.Namespace) -> int:
    if args.mode == "canary" and args.evaluator_index is not None:
        raise GateError("canary is technical-only; evaluator input forbidden")
    config, public, binding = load_inputs(args.config, args.data_dir)
    if args.mode == "full":
        validate_canary(args.run_dir, config, public, binding)
    rows, result = collect_run(args.run_dir, args.mode, config, public, binding)
    if args.mode == "full" and args.evaluator_index is not None:
        result["analysis"] = analyze_rows(rows, public, read_jsonl(args.evaluator_index), config)
        result["evaluator_index_sha256"] = sha256_file(args.evaluator_index)
        result["technical_only"] = False
    old.atomic_write_json(args.run_dir / args.mode / "aggregate.json", result)
    output = args.run_dir / args.mode / "RESULTS.md"
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(results_markdown(result), encoding="utf-8")
    os.replace(temporary, output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("worker", "aggregate"):
        sub = commands.add_parser(command)
        for flag in ("config", "data-dir", "run-dir"):
            sub.add_argument("--" + flag, type=Path, required=True)
        sub.add_argument("--mode", choices=("canary", "full"), required=True)
        if command == "worker":
            sub.add_argument("--rank", type=int, required=True)
            sub.add_argument("--world-size", type=int, required=True)
        else:
            sub.add_argument("--evaluator-index", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return worker_command(args) if args.command == "worker" else aggregate_command(args)
    except Exception as exc:
        path = args.run_dir / args.mode
        if args.command == "worker":
            path = path / f"rank_{args.rank:03d}"
        old.atomic_write_json(path / "failure.json", {"status": "fail", "stage": args.command,
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
