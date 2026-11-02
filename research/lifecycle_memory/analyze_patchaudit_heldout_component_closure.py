#!/usr/bin/env python3
"""Analyze the frozen PatchAudit held-out component-closure experiment.

The analyzer is deliberately independent of model code. It opens evaluator
labels only after the complete worker inventory and every technical receipt
have passed, then performs the preregistered participant-cluster analysis.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
from statistics import mean
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ANALYSIS_KIND = "patchaudit_heldout_component_closure_analysis"
PAPER_VALUES_KIND = "patchaudit_heldout_component_closure_paper_values"
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20260727
MODELS = ("qwen35_27b_hf_bf16", "gemma4_31b_hf_bf16")
CORRUPTIONS = ("blank_full", "wrong_time")
DIRECTIONS = ("rescue", "damage")
COMPONENTS = ("C", "Q", "P", "R")
FACTORIAL_OPERATORS = tuple(
    "".join(subset)
    for width in range(1, len(COMPONENTS) + 1)
    for subset in itertools.combinations(COMPONENTS, width)
)
CARDINALITY_BENCHMARKS = tuple(
    f"B_{component}_{replicate}"
    for component in ("Q", "P", "R")
    for replicate in range(4)
)
FORBIDDEN_WORKER_FIELDS = {
    "correct_choice",
    "correct_margin_bits",
    "donor_id",
    "ego4d_participant_id",
    "frame_id",
    "participant_cluster_id",
    "source_frame_level_unit_id",
    "source_trial_id",
    "source_video_id",
    "target_component_fingerprints",
}
OUTCOME_KINDS = {
    "patchaudit_heldout_component_closure_outcome",
    "patchaudit_component_closure_outcome",
}
AGGREGATE_KINDS = {
    "patchaudit_heldout_component_closure_aggregate",
    "patchaudit_component_closure_aggregate",
}
JOB_MANIFEST_KINDS = {
    "patchaudit_heldout_component_closure_tamia_job_manifest",
    "patchaudit_component_closure_tamia_job_manifest",
}


class AnalysisError(RuntimeError):
    """Raised when a technical or statistical analysis contract fails."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _reject_constant(value: str) -> None:
    raise AnalysisError(f"nonfinite JSON constant is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _strict_json_loads(raw: str, *, source: str) -> Any:
    try:
        return json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, AnalysisError) as exc:
        raise AnalysisError(f"invalid strict JSON in {source}: {exc}") from exc


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise AnalysisError(f"missing JSON file: {path}")
    return _strict_json_loads(path.read_text(encoding="utf-8"), source=str(path))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AnalysisError(f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = _strict_json_loads(
                line, source=f"{path}:{line_number}"
            )
            if not isinstance(value, dict):
                raise AnalysisError(
                    f"JSONL row is not an object: {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"{label} must be finite")
    return result


def percentile(values: np.ndarray | Sequence[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise AnalysisError("percentile input must be finite and nonempty")
    return float(np.quantile(array, probability))


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    if len(pvalues) != 2:
        raise AnalysisError("the frozen Holm family must contain exactly two tests")
    checked = {
        str(name): _finite_float(value, label=f"pvalue:{name}")
        for name, value in pvalues.items()
    }
    if any(value < 0.0 or value > 1.0 for value in checked.values()):
        raise AnalysisError("p-values must lie in [0, 1]")
    ordered = sorted(checked.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (len(ordered) - index) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _operator_alias(raw_operator: Any) -> str:
    raw = str(raw_operator or "")
    aliases = {
        "F": "F",
        "full_frame": "F",
        "full_frame_once": "F",
        "K": "K",
        "complete_complement": "K",
        "complement": "K",
        "complement_once": "K",
        "A": "A",
        "full_active": "A",
        "full_sequence": "A",
        "full_sequence_once": "A",
    }
    if raw in aliases:
        return aliases[raw]
    if raw in FACTORIAL_OPERATORS:
        return raw
    match = re.fullmatch(
        r"(?:B|benchmark|matched)_([QPR])_([0-3])", raw
    )
    if match:
        return f"B_{match.group(1)}_{match.group(2)}"
    raise AnalysisError(f"unregistered Job E operator: {raw!r}")


def expected_operators(
    *, model_key: str, layer: int, prespec: Mapping[str, Any]
) -> tuple[str, ...]:
    plan = prespec["depth_plan"][model_key]
    early = int(plan["early_post_block_index"])
    late = int(plan["late_post_block_index"])
    if layer not in {early, late}:
        raise AnalysisError(f"unregistered layer {model_key}/{layer}")
    if model_key == "qwen35_27b_hf_bf16" or layer == early:
        return ("F", "K", "A")
    return ("F", *FACTORIAL_OPERATORS, *CARDINALITY_BENCHMARKS, "A")


def _expected_outcomes_per_request(
    *, model_key: str, prespec: Mapping[str, Any]
) -> int:
    return len(CORRUPTIONS) * len(DIRECTIONS) * sum(
        len(expected_operators(model_key=model_key, layer=int(layer), prespec=prespec))
        for layer in prespec["depth_plan"][model_key][
            "ordered_post_block_indexes"
        ]
    )


def verify_frozen_contract(
    *, repo_dir: Path, contract_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    contract = read_json(contract_path)
    if (
        contract.get("kind")
        != "patchaudit_heldout_component_closure_job_contract"
        or contract.get("version") != 1
    ):
        raise AnalysisError("invalid Job E contract kind or version")
    prespec_path = (
        repo_dir / Path(str(contract["prespec"]["path"]))
    ).resolve()
    observed_prespec_sha = sha256_file(prespec_path)
    if observed_prespec_sha != str(contract["prespec"]["sha256"]):
        raise AnalysisError("Job E preregistration hash drifted")
    prespec = read_json(prespec_path)
    if (
        prespec.get("kind")
        != "patchaudit_heldout_component_closure_preregistration"
        or prespec.get("version") != 1
        or prespec.get("analysis_status")
        != "prospectively_frozen_before_any_jobE_component_patch_outcome"
    ):
        raise AnalysisError("Job E preregistration is not frozen and valid")
    if tuple(prespec.get("corruptions", ())) != CORRUPTIONS:
        raise AnalysisError("corruption inventory drifted")
    if tuple(row.get("model_key") for row in prespec.get("models", ())) != MODELS:
        raise AnalysisError("model inventory drifted")
    if int(prespec["decision_rule"]["family_size"]) != 2:
        raise AnalysisError("primary family size drifted")
    if len(prespec["decision_rule"]["holm_family"]) != 2:
        raise AnalysisError("Holm family declaration drifted")
    outputs = contract.get("outputs", {})
    required_outputs = {
        "analysis": "component_closure_analysis.json",
        "paper_values": "paper_component_closure_values.json",
        "plot_data": "component_closure_plot.csv",
        "factorial_table": "factorial_component_table.tex",
    }
    if any(outputs.get(key) != value for key, value in required_outputs.items()):
        raise AnalysisError("Job E output filename contract drifted")
    expected_by_model = {
        model: _expected_outcomes_per_request(
            model_key=model, prespec=prespec
        )
        for model in MODELS
    }
    observed_by_model = {
        str(key): int(value)
        for key, value in contract["completion"][
            "expected_outcomes_per_request_by_model"
        ].items()
    }
    if observed_by_model != expected_by_model:
        raise AnalysisError(
            "per-request outcome counts disagree with the frozen design: "
            f"contract={observed_by_model} design={expected_by_model}"
        )
    request_count = int(contract["completion"]["expected_requests_per_model"])
    expected_full = request_count * sum(expected_by_model.values())
    if int(contract["completion"]["expected_full_outcome_rows"]) != expected_full:
        raise AnalysisError("full outcome count disagrees with the frozen design")
    return (
        contract,
        prespec,
        {
            "contract_sha256": sha256_file(contract_path),
            "prespec_sha256": observed_prespec_sha,
        },
    )


def _require_pass_gate(
    container: Mapping[str, Any], *, alternatives: Sequence[str], label: str
) -> Mapping[str, Any]:
    for key in alternatives:
        value = container.get(key)
        if isinstance(value, Mapping):
            if value.get("status") != "pass":
                raise AnalysisError(f"{label} gate did not pass")
            return value
    raise AnalysisError(f"{label} gate receipt is missing")


def _validate_selection(
    *,
    selection_path: Path,
    public_path: Path,
    prespec: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = prespec["data"]
    if sha256_file(selection_path) != str(data["selection_manifest_sha256"]):
        raise AnalysisError("selection manifest hash drifted")
    if sha256_file(public_path) != str(data["public_requests_sha256"]):
        raise AnalysisError("public request manifest hash drifted")
    selection = read_json(selection_path)
    public_rows = read_jsonl(public_path)
    if (
        selection.get("kind") != "patchaudit_jobE_heldout_selection"
        or selection.get("status") != "pass"
        or selection.get("selection_status")
        != "frozen_before_any_jobE_model_outcome"
        or selection.get("selection_uses_correct_choice") is not False
        or selection.get("selection_uses_model_outputs") is not False
    ):
        raise AnalysisError("selection receipt is not frozen, outcome-blind, and valid")
    for field in ("participant_cluster_id", "source_video_id", "frame_id", "donor_id"):
        if int(selection["overlap_audit"][field]["overlap_count"]) != 0:
            raise AnalysisError(f"source identity disjointness failed for {field}")
    request_ids = [str(row.get("request_id") or "") for row in public_rows]
    if (
        len(public_rows) != int(data["request_count"])
        or len(set(request_ids)) != len(request_ids)
        or set(request_ids) != set(selection["selected"]["request_ids"])
    ):
        raise AnalysisError("public request inventory differs from selection")
    frame_counts: dict[str, int] = defaultdict(int)
    public_map = {str(row["request_id"]): row for row in public_rows}
    for row in public_rows:
        request_id = str(row.get("request_id") or "")
        frame_id = str(row.get("frame_unit_id") or "")
        order_id = str(row.get("order_id") or "")
        paired_id = str(row.get("counterbalanced_request_id") or "")
        if not request_id or not frame_id or order_id not in {"order_0", "order_1"}:
            raise AnalysisError("public request identity is incomplete")
        if paired_id not in public_map:
            raise AnalysisError(f"missing counterbalanced request for {request_id}")
        paired = public_map[paired_id]
        if (
            str(paired.get("counterbalanced_request_id")) != request_id
            or str(paired.get("frame_unit_id")) != frame_id
            or str(paired.get("order_id")) == order_id
        ):
            raise AnalysisError(f"counterbalance contract failed for {request_id}")
        frame_counts[frame_id] += 1
    if (
        len(frame_counts) != int(data["frame_count"])
        or set(frame_counts.values()) != {2}
    ):
        raise AnalysisError("public frame/order inventory is not exactly paired")
    return selection, public_rows


def _validate_geometry_partition(record: Mapping[str, Any]) -> None:
    partition = record.get("partition")
    if not isinstance(partition, Mapping):
        raise AnalysisError("geometry audit lacks a component partition")
    if record.get("component_partition_sha256") != canonical_sha256(partition):
        raise AnalysisError("geometry component-partition hash drifted")

    def positions(name: str) -> tuple[int, ...]:
        group = partition.get(name)
        if not isinstance(group, Mapping):
            raise AnalysisError(f"geometry partition lacks {name}")
        raw = group.get("positions")
        if not isinstance(raw, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw
        ):
            raise AnalysisError(f"geometry group {name} has invalid positions")
        result = tuple(raw)
        if result != tuple(sorted(set(result))):
            raise AnalysisError(f"geometry group {name} is not sorted and unique")
        if int(group.get("count", -1)) != len(result):
            raise AnalysisError(f"geometry group {name} count drifted")
        if group.get("sha256") != canonical_sha256(list(result)):
            raise AnalysisError(f"geometry group {name} hash drifted")
        if bool(group.get("structurally_absent")) != (not result):
            raise AnalysisError(f"geometry group {name} absence flag drifted")
        return result

    active = positions("active")
    full_frame = positions("full_frame")
    components = {
        name: positions(name)
        for name in (
            "candidate_crops",
            "literal_user_prompt",
            "answer_prefix",
            "structural_remainder",
        )
    }
    if any(not value for value in components.values()):
        raise AnalysisError("Gemma geometry has a structurally absent component")
    groups = (full_frame, *components.values())
    flattened = [position for group in groups for position in group]
    if len(flattened) != len(set(flattened)):
        raise AnalysisError("Gemma component groups overlap")
    if tuple(sorted(flattened)) != active:
        raise AnalysisError("Gemma component groups do not partition active tokens")
    complement = positions("complement")
    if complement != tuple(
        sorted(position for group in components.values() for position in group)
    ):
        raise AnalysisError("Gemma complement is not C union Q union P union R")


def _validate_aggregate_before_labels(
    *,
    run_dir: Path,
    aggregate: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_hashes: Mapping[str, str],
    prespec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    if (
        aggregate.get("kind") not in AGGREGATE_KINDS
        or aggregate.get("status") != "pass"
        or aggregate.get("mode") != "full"
        or aggregate.get("failures") not in ([], ())
        or aggregate.get("resume_needed") is not False
        or aggregate.get("contract") != contract_hashes
        or int(aggregate.get("worker_count", -1)) != 4
        or aggregate.get("exact_cartesian_inventory_status") != "pass"
    ):
        raise AnalysisError("analysis requires a passing complete full aggregate")
    _require_pass_gate(
        aggregate, alternatives=("telemetry",), label="telemetry"
    )
    _require_pass_gate(
        aggregate,
        alternatives=("job_manifest_contract",),
        label="job manifest",
    )
    _require_pass_gate(
        aggregate,
        alternatives=("production_canary_binding",),
        label="bound canary",
    )
    workers = aggregate.get("workers")
    if (
        not isinstance(workers, list)
        or len(workers) != 4
        or any(
            not isinstance(worker, Mapping)
            or worker.get("status") != "pass"
            or worker.get("hidden_index_opened") is not False
            for worker in workers
        )
    ):
        raise AnalysisError("worker summaries or label-firewall receipts failed")

    rank_files = aggregate.get("rank_outcome_files")
    if not isinstance(rank_files, Mapping) or set(rank_files) != {
        "0",
        "1",
        "2",
        "3",
    }:
        raise AnalysisError("aggregate lacks four rank-bound outcome files")
    all_rows: list[dict[str, Any]] = []
    for rank in range(4):
        rank_dir = run_dir / f"rank_{rank:03d}"
        outcomes_path = rank_dir / "outcomes.jsonl"
        receipts_path = rank_dir / "completed_requests.jsonl"
        rows = read_jsonl(outcomes_path)
        receipts = read_jsonl(receipts_path)
        record = rank_files[str(rank)]
        if (
            record.get("outcomes_sha256") != sha256_file(outcomes_path)
            or record.get("completed_requests_sha256")
            != sha256_file(receipts_path)
            or int(record.get("outcome_rows", -1)) != len(rows)
            or int(record.get("completion_receipt_rows", -1)) != len(receipts)
        ):
            raise AnalysisError(f"rank {rank} file-hash receipt drifted")
        rows_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if int(row.get("rank", -1)) != rank:
                raise AnalysisError(f"rank field drifted in rank {rank} outcomes")
            rows_by_request[str(row.get("request_id") or "")].append(row)
        receipt_ids: list[str] = []
        for receipt in receipts:
            request_id = str(receipt.get("request_id") or "")
            receipt_ids.append(request_id)
            batch = rows_by_request.get(request_id, [])
            if (
                receipt.get("kind")
                != "patchaudit_component_closure_request_receipt"
                or receipt.get("output_schema_version") != 1
                or
                int(receipt.get("result_row_count", -1)) != len(batch)
                or receipt.get("result_batch_sha256") != canonical_sha256(batch)
            ):
                raise AnalysisError(f"request receipt drifted for {request_id}")
        if len(receipt_ids) != len(set(receipt_ids)) or set(receipt_ids) != set(
            rows_by_request
        ):
            raise AnalysisError(f"rank {rank} request receipt inventory drifted")
        all_rows.extend(rows)

    geometry_receipts = aggregate.get("geometry_audits")
    if not isinstance(geometry_receipts, Mapping) or set(geometry_receipts) != {
        "0",
        "1",
        "2",
        "3",
    }:
        raise AnalysisError("aggregate lacks four geometry-audit receipts")
    public_ids = {
        str(value)
        for value in aggregate.get("public_manifest", {}).get(
            "request_ids", []
        )
    }
    if not public_ids:
        public_ids = {
            str(row.get("request_id"))
            for row in all_rows
            if row.get("request_id")
        }
    if len(public_ids) != int(prespec["data"]["request_count"]):
        raise AnalysisError("aggregate public request inventory drifted")
    geometry_ids: dict[int, set[str]] = {}
    gemma_fingerprints: set[str] = set()
    for rank in range(4):
        geometry_path = run_dir / f"rank_{rank:03d}" / "geometry_audit.json"
        geometry = read_json(geometry_path)
        receipt = geometry_receipts[str(rank)]
        ids = [str(value) for value in geometry.get("request_ids", [])]
        records = geometry.get("records")
        if (
            receipt.get("sha256") != sha256_file(geometry_path)
            or int(receipt.get("request_count", -1)) != len(ids)
            or geometry.get("kind")
            != "patchaudit_component_closure_geometry_audit"
            or geometry.get("status") != "pass"
            or int(geometry.get("rank", -1)) != rank
            or geometry.get("hidden_index_opened") is not False
            or int(geometry.get("request_count", -1)) != len(ids)
            or geometry.get("request_ids_sha256")
            != canonical_sha256(sorted(ids))
            or len(ids) != len(set(ids))
            or not set(ids) <= public_ids
            or not isinstance(records, list)
            or len(records) != len(ids)
            or geometry.get("records_sha256") != canonical_sha256(records)
            or {str(row.get("request_id") or "") for row in records}
            != set(ids)
        ):
            raise AnalysisError(f"rank {rank} geometry audit drifted")
        model_key = str(geometry.get("model_key") or "")
        expected_model = (
            "qwen35_27b_hf_bf16"
            if rank == 0
            else "gemma4_31b_hf_bf16"
        )
        if model_key != expected_model:
            raise AnalysisError(f"rank {rank} geometry model drifted")
        geometry_ids[rank] = set(ids)
        if rank > 0:
            fingerprints = {
                str(value)
                for value in geometry.get("component_fingerprints", [])
            }
            if (
                int(geometry.get("component_fingerprint_count", -1)) != 1
                or len(fingerprints) != 1
            ):
                raise AnalysisError(
                    f"rank {rank} Gemma component partition is not invariant"
                )
            gemma_fingerprints.update(fingerprints)
            for record in records:
                _validate_geometry_partition(record)
    if geometry_ids[0] != public_ids:
        raise AnalysisError("Qwen geometry preflight did not cover every request")
    if any(
        geometry_ids[left] & geometry_ids[right]
        for left, right in ((1, 2), (1, 3), (2, 3))
    ):
        raise AnalysisError("Gemma geometry-preflight shards overlap")
    if geometry_ids[1] | geometry_ids[2] | geometry_ids[3] != public_ids:
        raise AnalysisError("Gemma geometry preflight did not cover every request")
    if len(gemma_fingerprints) != 1:
        raise AnalysisError("Gemma component partition differs across ranks")

    expected_rows = int(contract["completion"]["expected_full_outcome_rows"])
    if (
        len(all_rows) != expected_rows
        or int(aggregate.get("outcome_row_count", -1)) != expected_rows
        or int(aggregate.get("unique_outcome_identity_count", -1)) != expected_rows
    ):
        raise AnalysisError("aggregate outcome row count drifted")
    full_active_rows = [
        row for row in all_rows if _operator_alias(row.get("operator")) == "A"
    ]
    expected_full_active = int(prespec["data"]["request_count"]) * 16
    closure_tolerance = float(
        prespec["technical_gates"]["full_sequence_closure_max_abs_error_bits"]
    )
    if (
        len(full_active_rows) != expected_full_active
        or int(aggregate.get("full_sequence_row_count", -1))
        != expected_full_active
        or int(aggregate.get("expected_full_sequence_row_count", -1))
        != expected_full_active
        or _finite_float(
            aggregate.get("full_sequence_max_abs_error_bits"),
            label="aggregate full-sequence error",
        )
        > closure_tolerance
    ):
        raise AnalysisError("full-active closure gate failed")
    return all_rows, sha256_file(run_dir / "aggregate_summary.json")


def _validate_job_manifest_before_labels(
    *,
    run_dir: Path,
    hidden_path: Path,
    data_dir: Path,
    aggregate_sha256: str,
) -> dict[str, Any]:
    manifest_path = run_dir / "job_manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest.get("kind") not in JOB_MANIFEST_KINDS
        or manifest.get("mode") != "full"
        or manifest.get("worker_hidden_bind") is not False
        or manifest.get("hidden_index_staged_before_technical_gate") is not False
        or manifest.get("technical_gate_passed_before_hidden_stage") is not True
        or manifest.get("technical_aggregate_sha256") != aggregate_sha256
        or manifest.get("hidden_index_staged_after_technical_gate") is not True
        or manifest.get("hidden_index_outside_worker_data_dir") is not True
        or manifest.get("worker_visible_hidden_scan_status") != "pass"
        or Path(str(manifest.get("hidden_index_stage_path") or "")).resolve()
        != hidden_path
    ):
        raise AnalysisError("launcher label-firewall receipt is incomplete")
    for forbidden_root in (run_dir, data_dir):
        if hidden_path == forbidden_root or forbidden_root in hidden_path.parents:
            raise AnalysisError(
                "hidden evaluator index is inside a worker-visible tree"
            )
    return manifest


def _validate_and_normalize_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    prespec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    public_map = {str(row["request_id"]): row for row in public_rows}
    allowed_ranks = {
        str(model): {int(rank) for rank in ranks}
        for model, ranks in contract["execution"]["rank_allocation"].items()
    }
    normalized: list[dict[str, Any]] = []
    identities: list[tuple[Any, ...]] = []
    operator_hashes: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for raw in rows:
        leaked = FORBIDDEN_WORKER_FIELDS & set(raw)
        if leaked:
            raise AnalysisError(
                f"worker outcome leaked evaluator fields: {sorted(leaked)}"
            )
        if raw.get("kind") not in OUTCOME_KINDS or raw.get(
            "output_schema_version"
        ) != 1:
            raise AnalysisError("worker outcome kind or schema drifted")
        model_key = str(raw.get("model_key") or "")
        request_id = str(raw.get("request_id") or "")
        public = public_map.get(request_id)
        if model_key not in MODELS or public is None:
            raise AnalysisError("worker outcome model/request inventory drifted")
        rank = int(raw.get("rank", -1))
        if rank not in allowed_ranks.get(model_key, set()):
            raise AnalysisError(f"rank/model allocation drifted for {request_id}")
        for field in ("frame_unit_id", "order_id"):
            if str(raw.get(field)) != str(public.get(field)):
                raise AnalysisError(f"public {field} drifted for {request_id}")
        corruption = str(raw.get("corruption") or "")
        direction = str(raw.get("direction") or "")
        if corruption not in CORRUPTIONS or direction not in DIRECTIONS:
            raise AnalysisError(f"corruption/direction drifted for {request_id}")
        source_arm, recipient_arm = (
            ("real", corruption)
            if direction == "rescue"
            else (corruption, "real")
        )
        if (
            raw.get("source_arm") != source_arm
            or raw.get("recipient_arm") != recipient_arm
        ):
            raise AnalysisError(f"source/recipient arm drifted for {request_id}")
        layer = int(raw.get("start_layer", -1))
        operator = _operator_alias(raw.get("operator"))
        if operator not in expected_operators(
            model_key=model_key, layer=layer, prespec=prespec
        ):
            raise AnalysisError(
                f"operator is not registered at {model_key}/L{layer}: {operator}"
            )
        position_count = int(raw.get("operator_position_count", -1))
        position_sha256 = str(raw.get("operator_positions_sha256") or "")
        operator_receipt = raw.get("operator_receipt")
        if (
            position_count <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", position_sha256)
            or not isinstance(operator_receipt, Mapping)
        ):
            raise AnalysisError(f"operator position receipt failed for {request_id}")
        if operator in {"F", "K", "A"}:
            expected_group = {
                "F": "full_frame",
                "K": "complement",
                "A": "active",
            }[operator]
            if (
                operator_receipt.get("type") != "registered_group"
                or operator_receipt.get("group") != expected_group
            ):
                raise AnalysisError(
                    f"registered-group receipt failed for {request_id}/{operator}"
                )
        elif operator in FACTORIAL_OPERATORS:
            if (
                operator_receipt.get("type")
                != "factorial_component_union"
                or operator_receipt.get("component_codes") != list(operator)
            ):
                raise AnalysisError(
                    f"factorial receipt failed for {request_id}/{operator}"
                )
        else:
            match = re.fullmatch(r"B_([QPR])_([0-3])", operator)
            sample = operator_receipt.get("sample")
            expected_group = {
                "Q": "literal_user_prompt",
                "P": "answer_prefix",
                "R": "structural_remainder",
            }[match.group(1)] if match else None
            if (
                match is None
                or operator_receipt.get("type")
                != "descriptive_crop_cardinality_benchmark"
                or operator_receipt.get("target_group") != expected_group
                or int(operator_receipt.get("replicate", -1))
                != int(match.group(2))
                or not isinstance(sample, Mapping)
                or int(sample.get("count", -1)) != position_count
                or sample.get("sha256") != position_sha256
                or sample.get("positions") is None
                or canonical_sha256(list(sample["positions"])) != position_sha256
            ):
                raise AnalysisError(
                    f"cardinality benchmark receipt failed for "
                    f"{request_id}/{operator}"
                )
        operator_hashes[(model_key, layer, operator)].add(position_sha256)
        recipient = _finite_float(
            raw.get("recipient_margin_a_minus_b_bits"),
            label=f"recipient margin:{request_id}",
        )
        donor = _finite_float(
            raw.get("donor_margin_a_minus_b_bits"),
            label=f"donor margin:{request_id}",
        )
        patched = _finite_float(
            raw.get("patched_margin_a_minus_b_bits"),
            label=f"patched margin:{request_id}",
        )
        baseline = abs(recipient - donor)
        error = abs(patched - donor)
        if (
            abs(
                _finite_float(
                    raw.get("baseline_donor_gap_abs_bits"),
                    label=f"baseline gap:{request_id}",
                )
                - baseline
            )
            > 1e-9
            or abs(
                _finite_float(
                    raw.get("patched_donor_error_abs_bits"),
                    label=f"patched error:{request_id}",
                )
                - error
            )
            > 1e-9
        ):
            raise AnalysisError(f"margin algebra drifted for {request_id}")
        audit = raw.get("audit")
        if (
            not isinstance(audit, Mapping)
            or audit.get("status") != "pass"
            or audit.get("all_fired_once") is not True
            or audit.get("all_removed") is not True
            or audit.get("all_finite") is not True
        ):
            raise AnalysisError(f"hook audit failed for {request_id}")
        identity = (
            model_key,
            request_id,
            corruption,
            direction,
            layer,
            operator,
        )
        identities.append(identity)
        normalized.append(
            {
                **raw,
                "model_key": model_key,
                "request_id": request_id,
                "rank": rank,
                "corruption": corruption,
                "direction": direction,
                "start_layer": layer,
                "operator": operator,
                "recipient_margin_a_minus_b_bits": recipient,
                "donor_margin_a_minus_b_bits": donor,
                "patched_margin_a_minus_b_bits": patched,
                "recipient_donor_gap_bits": baseline,
                "patched_donor_error_bits": error,
            }
        )
    if len(identities) != len(set(identities)):
        raise AnalysisError("duplicate normalized outcome identity")
    expected = {
        (model, request_id, corruption, direction, int(layer), operator)
        for model in MODELS
        for request_id in public_map
        for corruption in CORRUPTIONS
        for direction in DIRECTIONS
        for layer in prespec["depth_plan"][model][
            "ordered_post_block_indexes"
        ]
        for operator in expected_operators(
            model_key=model, layer=int(layer), prespec=prespec
        )
    }
    observed = set(identities)
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise AnalysisError(
            f"exact Cartesian inventory failed: missing={missing} extra={extra}"
        )
    for (model_key, layer, operator), hashes in operator_hashes.items():
        if model_key == "gemma4_31b_hf_bf16" and len(hashes) != 1:
            raise AnalysisError(
                f"Gemma operator geometry drifted at L{layer}/{operator}"
            )
    late = int(
        prespec["depth_plan"]["gemma4_31b_hf_bf16"][
            "late_post_block_index"
        ]
    )
    for component in ("Q", "P", "R"):
        hashes = {
            next(
                iter(
                    operator_hashes[
                        (
                            "gemma4_31b_hf_bf16",
                            late,
                            f"B_{component}_{replicate}",
                        )
                    ]
                )
            )
            for replicate in range(4)
        }
        if len(hashes) != 4:
            raise AnalysisError(
                f"{component} cardinality benchmarks are not four unique sets"
            )
    return normalized


def _join_hidden(
    *,
    rows: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    hidden_rows: Sequence[Mapping[str, Any]],
    expected_request_count: int = 92,
    expected_cluster_count: int = 46,
) -> list[dict[str, Any]]:
    public_ids = [str(row.get("request_id") or "") for row in public_rows]
    hidden_ids = [str(row.get("request_id") or "") for row in hidden_rows]
    if (
        len(public_ids) != expected_request_count
        or len(hidden_ids) != expected_request_count
        or len(set(public_ids)) != expected_request_count
        or len(set(hidden_ids)) != expected_request_count
        or set(public_ids) != set(hidden_ids)
    ):
        raise AnalysisError("public and hidden evaluator inventories differ")
    public_map = {
        str(row["request_id"]): row for row in public_rows
    }
    hidden_map = {
        str(row["request_id"]): row for row in hidden_rows
    }
    clusters: set[str] = set()
    joined: list[dict[str, Any]] = []
    for raw in rows:
        leaked = FORBIDDEN_WORKER_FIELDS & set(raw)
        if leaked:
            raise AnalysisError(
                f"worker outcome leaked evaluator fields: {sorted(leaked)}"
            )
        request_id = str(raw.get("request_id") or "")
        public = public_map.get(request_id)
        hidden = hidden_map.get(request_id)
        if public is None or hidden is None:
            raise AnalysisError(f"cannot join evaluator row {request_id}")
        for field in ("frame_unit_id", "order_id", "counterbalanced_request_id"):
            if field in public and str(hidden.get(field)) != str(public.get(field)):
                raise AnalysisError(
                    f"hidden/public {field} mismatch for {request_id}"
                )
        correct_choice = str(hidden.get("correct_choice") or "")
        cluster = str(hidden.get("participant_cluster_id") or "")
        participant = str(hidden.get("ego4d_participant_id") or "")
        if correct_choice not in {"A", "B"} or not cluster or not participant:
            raise AnalysisError(f"invalid evaluator label row for {request_id}")
        clusters.add(cluster)
        sign = 1.0 if correct_choice == "A" else -1.0
        joined.append(
            {
                **raw,
                "correct_choice": correct_choice,
                "participant_cluster_id": cluster,
                "ego4d_participant_id": participant,
                "recipient_correct_margin_bits": sign
                * float(raw["recipient_margin_a_minus_b_bits"]),
                "donor_correct_margin_bits": sign
                * float(raw["donor_margin_a_minus_b_bits"]),
                "patched_correct_margin_bits": sign
                * float(raw["patched_margin_a_minus_b_bits"]),
            }
        )
    if len(clusters) != expected_cluster_count:
        raise AnalysisError(
            f"expected {expected_cluster_count} participant clusters, "
            f"observed {len(clusters)}"
        )
    return joined


@dataclass(frozen=True)
class ClusterVector:
    clusters: tuple[str, ...]
    error: np.ndarray
    baseline: np.ndarray
    rows_per_cluster: int


def _cluster_vector(
    rows: Sequence[Mapping[str, Any]], *, expected_rows_per_cluster: int
) -> ClusterVector:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        cluster = str(row.get("participant_cluster_id") or "")
        if not cluster:
            raise AnalysisError("joined row lacks participant cluster")
        grouped[cluster].append(
            (
                _finite_float(
                    row.get("patched_donor_error_bits"),
                    label="cluster patched-donor error",
                ),
                _finite_float(
                    row.get("recipient_donor_gap_bits"),
                    label="cluster recipient-donor gap",
                ),
            )
        )
    clusters = tuple(sorted(grouped))
    if len(clusters) != 46:
        raise AnalysisError(f"analysis cell has {len(clusters)} clusters, not 46")
    bad = [
        cluster
        for cluster in clusters
        if len(grouped[cluster]) != expected_rows_per_cluster
    ]
    if bad:
        raise AnalysisError(
            f"analysis cell has wrong within-cluster inventory: {bad[:5]}"
        )
    error = np.asarray(
        [mean(value[0] for value in grouped[cluster]) for cluster in clusters],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [mean(value[1] for value in grouped[cluster]) for cluster in clusters],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(error))
        or not np.all(np.isfinite(baseline))
        or np.any(baseline <= 1e-12)
    ):
        raise AnalysisError("cluster vectors are nonfinite or have zero denominator")
    return ClusterVector(clusters, error, baseline, expected_rows_per_cluster)


def _rho(error: np.ndarray, baseline: np.ndarray) -> float:
    denominator = float(np.mean(baseline))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        raise AnalysisError("recovery denominator is nonpositive")
    value = 1.0 - float(np.mean(error)) / denominator
    if not math.isfinite(value):
        raise AnalysisError("recovery is nonfinite")
    return value


def _bootstrap_rho(
    vector: ClusterVector, indexes: np.ndarray
) -> np.ndarray:
    denominator = vector.baseline[indexes].mean(axis=1)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 1e-12):
        raise AnalysisError("bootstrap recovery denominator is nonpositive")
    values = 1.0 - vector.error[indexes].mean(axis=1) / denominator
    if not np.all(np.isfinite(values)):
        raise AnalysisError("bootstrap recovery is nonfinite")
    return values


def _bootstrap_indexes(*, clusters: int, draws: int, seed: int) -> np.ndarray:
    if clusters < 2 or draws < 100:
        raise AnalysisError("bootstrap requires at least two clusters and 100 draws")
    return np.random.default_rng(seed).integers(
        0, clusters, size=(draws, clusters)
    )


def _select(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    layer: int,
    operator: str,
    corruption: str | None = None,
    direction: str | None = None,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["model_key"] == model_key
        and int(row["start_layer"]) == layer
        and row["operator"] == operator
        and (corruption is None or row["corruption"] == corruption)
        and (direction is None or row["direction"] == direction)
    ]


def _assert_paired_vectors(*vectors: ClusterVector) -> None:
    first = vectors[0]
    for vector in vectors[1:]:
        if vector.clusters != first.clusters:
            raise AnalysisError("paired cells use different participant inventories")
        if not np.allclose(
            vector.baseline, first.baseline, rtol=0.0, atol=1e-12
        ):
            raise AnalysisError("paired cells have different recovery denominators")


def _summary(
    vector: ClusterVector, *, indexes: np.ndarray, seed: int
) -> dict[str, Any]:
    bootstrap = _bootstrap_rho(vector, indexes)
    return {
        "rho": _rho(vector.error, vector.baseline),
        "ci95": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "participant_cluster_count": len(vector.clusters),
        "cluster_equal_mean_patched_donor_error_bits": float(
            np.mean(vector.error)
        ),
        "cluster_equal_mean_recipient_donor_gap_bits": float(
            np.mean(vector.baseline)
        ),
        "bootstrap_draws": int(indexes.shape[0]),
        "bootstrap_seed": seed,
    }


def endpoint_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    prespec: Mapping[str, Any],
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    counter = 0
    for model_key in MODELS:
        for layer_value in prespec["depth_plan"][model_key][
            "ordered_post_block_indexes"
        ]:
            layer = int(layer_value)
            for corruption in CORRUPTIONS:
                for direction in DIRECTIONS:
                    for operator in expected_operators(
                        model_key=model_key, layer=layer, prespec=prespec
                    ):
                        selected = _select(
                            rows,
                            model_key=model_key,
                            layer=layer,
                            operator=operator,
                            corruption=corruption,
                            direction=direction,
                        )
                        vector = _cluster_vector(
                            selected, expected_rows_per_cluster=2
                        )
                        cell_seed = seed + 1000 + counter
                        indexes = _bootstrap_indexes(
                            clusters=46, draws=draws, seed=cell_seed
                        )
                        summaries.append(
                            {
                                "model_key": model_key,
                                "start_layer": layer,
                                "corruption": corruption,
                                "direction": direction,
                                "operator": operator,
                                **_summary(
                                    vector, indexes=indexes, seed=cell_seed
                                ),
                            }
                        )
                        counter += 1
    if len(summaries) != 152:
        raise AnalysisError(
            f"expected 152 endpoint summaries, observed {len(summaries)}"
        )
    return summaries


def primary_gemma_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    prespec: Mapping[str, Any],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    model = "gemma4_31b_hf_bf16"
    early = int(prespec["depth_plan"][model]["early_post_block_index"])
    late = int(prespec["depth_plan"][model]["late_post_block_index"])
    vectors = {
        "early_F": _cluster_vector(
            _select(rows, model_key=model, layer=early, operator="F"),
            expected_rows_per_cluster=8,
        ),
        "early_K": _cluster_vector(
            _select(rows, model_key=model, layer=early, operator="K"),
            expected_rows_per_cluster=8,
        ),
        "late_F": _cluster_vector(
            _select(rows, model_key=model, layer=late, operator="F"),
            expected_rows_per_cluster=8,
        ),
        "late_K": _cluster_vector(
            _select(rows, model_key=model, layer=late, operator="CQPR"),
            expected_rows_per_cluster=8,
        ),
    }
    _assert_paired_vectors(*vectors.values())
    indexes = _bootstrap_indexes(clusters=46, draws=draws, seed=seed)
    boot = {
        name: _bootstrap_rho(vector, indexes)
        for name, vector in vectors.items()
    }
    rho = {
        name: _rho(vector.error, vector.baseline)
        for name, vector in vectors.items()
    }
    late_delta = rho["late_K"] - rho["late_F"]
    early_delta = rho["early_K"] - rho["early_F"]
    late_boot = boot["late_K"] - boot["late_F"]
    early_boot = boot["early_K"] - boot["early_F"]
    interaction_boot = late_boot - early_boot
    threshold = float(
        prespec["decision_rule"]["materiality_threshold_rho"]
    )
    ids = tuple(prespec["decision_rule"]["holm_family"])
    if len(ids) != 2:
        raise AnalysisError("the preregistered Holm family is not exactly two")
    raw_p = {
        ids[0]: (1 + int(np.count_nonzero(late_boot <= threshold)))
        / (draws + 1),
        ids[1]: (1 + int(np.count_nonzero(early_boot >= -threshold)))
        / (draws + 1),
    }
    adjusted = holm_adjust(raw_p)
    cells = [
        {
            "cell_id": ids[0],
            "model_key": model,
            "start_layer": late,
            "contrast": "K_minus_F",
            "estimate_delta_rho": late_delta,
            "comparator_rho": rho["late_K"],
            "reference_rho": rho["late_F"],
            "ci95": [
                percentile(late_boot, 0.025),
                percentile(late_boot, 0.975),
            ],
            "null_hypothesis": "delta_late_less_than_or_equal_to_0.10",
            "materiality_threshold_rho": threshold,
            "one_sided_bootstrap_pvalue": raw_p[ids[0]],
            "holm_adjusted_pvalue": adjusted[ids[0]],
            "passes": adjusted[ids[0]] < 0.05,
        },
        {
            "cell_id": ids[1],
            "model_key": model,
            "start_layer": early,
            "contrast": "K_minus_F",
            "estimate_delta_rho": early_delta,
            "comparator_rho": rho["early_K"],
            "reference_rho": rho["early_F"],
            "ci95": [
                percentile(early_boot, 0.025),
                percentile(early_boot, 0.975),
            ],
            "null_hypothesis": "delta_early_greater_than_or_equal_to_minus_0.10",
            "materiality_threshold_rho": threshold,
            "one_sided_bootstrap_pvalue": raw_p[ids[1]],
            "holm_adjusted_pvalue": adjusted[ids[1]],
            "passes": adjusted[ids[1]] < 0.05,
        },
    ]
    return {
        "family_size": 2,
        "multiplicity": "Holm_over_exactly_two_one_sided_tests",
        "cells": cells,
        "endpoint_interaction": {
            "estimand": "delta_late_minus_delta_early",
            "estimate": late_delta - early_delta,
            "ci95": [
                percentile(interaction_boot, 0.025),
                percentile(interaction_boot, 0.975),
            ],
            "descriptive_only": True,
        },
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def qwen_descriptive_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    prespec: Mapping[str, Any],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    model = "qwen35_27b_hf_bf16"
    early = int(prespec["depth_plan"][model]["early_post_block_index"])
    late = int(prespec["depth_plan"][model]["late_post_block_index"])
    vectors: dict[str, ClusterVector] = {}
    for endpoint, layer in (("early", early), ("late", late)):
        for operator in ("F", "K", "A"):
            vectors[f"{endpoint}_{operator}"] = _cluster_vector(
                _select(rows, model_key=model, layer=layer, operator=operator),
                expected_rows_per_cluster=8,
            )
    _assert_paired_vectors(*vectors.values())
    indexes = _bootstrap_indexes(clusters=46, draws=draws, seed=seed)
    values = {
        name: _rho(vector.error, vector.baseline)
        for name, vector in vectors.items()
    }
    boot = {
        name: _bootstrap_rho(vector, indexes)
        for name, vector in vectors.items()
    }
    endpoints: list[dict[str, Any]] = []
    contrasts: dict[str, tuple[float, np.ndarray]] = {}
    for endpoint, layer in (("early", early), ("late", late)):
        delta = values[f"{endpoint}_K"] - values[f"{endpoint}_F"]
        delta_boot = boot[f"{endpoint}_K"] - boot[f"{endpoint}_F"]
        contrasts[endpoint] = (delta, delta_boot)
        endpoints.append(
            {
                "endpoint": endpoint,
                "start_layer": layer,
                "rho_F": values[f"{endpoint}_F"],
                "rho_K": values[f"{endpoint}_K"],
                "rho_A": values[f"{endpoint}_A"],
                "delta_K_minus_F": delta,
                "delta_K_minus_F_ci95": [
                    percentile(delta_boot, 0.025),
                    percentile(delta_boot, 0.975),
                ],
                "descriptive_only": True,
            }
        )
    interaction = contrasts["late"][0] - contrasts["early"][0]
    interaction_boot = contrasts["late"][1] - contrasts["early"][1]
    return {
        "model_key": model,
        "endpoints": endpoints,
        "endpoint_interaction": {
            "estimate": interaction,
            "ci95": [
                percentile(interaction_boot, 0.025),
                percentile(interaction_boot, 0.975),
            ],
        },
        "multiplicity_adjusted": False,
        "descriptive_only": True,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def _mobius_from_values(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    for bit in range(4):
        for mask in range(16):
            if mask & (1 << bit):
                result[..., mask] -= result[..., mask ^ (1 << bit)]
    return result


def _shapley_from_values(values: np.ndarray) -> np.ndarray:
    result = np.zeros((*values.shape[:-1], 4), dtype=np.float64)
    factorial = math.factorial
    for player in range(4):
        player_bit = 1 << player
        for mask in range(16):
            if mask & player_bit:
                continue
            size = int(mask.bit_count())
            weight = (
                factorial(size)
                * factorial(4 - size - 1)
                / factorial(4)
            )
            result[..., player] += weight * (
                values[..., mask | player_bit] - values[..., mask]
            )
    return result


def factorial_component_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    prespec: Mapping[str, Any],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    model = "gemma4_31b_hf_bf16"
    layer = int(prespec["depth_plan"][model]["late_post_block_index"])
    vectors: dict[str, ClusterVector] = {}
    for operator in FACTORIAL_OPERATORS:
        vectors[operator] = _cluster_vector(
            _select(rows, model_key=model, layer=layer, operator=operator),
            expected_rows_per_cluster=8,
        )
    _assert_paired_vectors(*vectors.values())
    indexes = _bootstrap_indexes(clusters=46, draws=draws, seed=seed)
    point_values = np.zeros(16, dtype=np.float64)
    bootstrap_values = np.zeros((draws, 16), dtype=np.float64)
    operator_by_mask: dict[int, str] = {}
    for operator, vector in vectors.items():
        mask = sum(1 << COMPONENTS.index(code) for code in operator)
        operator_by_mask[mask] = operator
        point_values[mask] = _rho(vector.error, vector.baseline)
        bootstrap_values[:, mask] = _bootstrap_rho(vector, indexes)
    if set(operator_by_mask) != set(range(1, 16)):
        raise AnalysisError("factorial set function is not the complete 15-cell game")
    mobius = _mobius_from_values(point_values)
    mobius_boot = _mobius_from_values(bootstrap_values)
    shapley = _shapley_from_values(point_values)
    shapley_boot = _shapley_from_values(bootstrap_values)
    if (
        abs(float(shapley.sum()) - float(point_values[15])) > 1e-10
        or np.max(np.abs(shapley_boot.sum(axis=1) - bootstrap_values[:, 15]))
        > 1e-10
    ):
        raise AnalysisError("exact Shapley efficiency identity failed")
    reconstructed = np.zeros(16, dtype=np.float64)
    for mask in range(16):
        reconstructed[mask] = sum(
            mobius[submask]
            for submask in range(16)
            if submask & mask == submask
        )
    if not np.allclose(reconstructed, point_values, rtol=0.0, atol=1e-10):
        raise AnalysisError("Mobius inversion identity failed")
    values = [
        {
            "operator": operator_by_mask[mask],
            "included_components": [
                COMPONENTS[index]
                for index in range(4)
                if mask & (1 << index)
            ],
            "rho": float(point_values[mask]),
            "ci95": [
                percentile(bootstrap_values[:, mask], 0.025),
                percentile(bootstrap_values[:, mask], 0.975),
            ],
        }
        for mask in range(1, 16)
    ]
    mobius_rows = [
        {
            "operator": operator_by_mask[mask],
            "included_components": [
                COMPONENTS[index]
                for index in range(4)
                if mask & (1 << index)
            ],
            "coefficient": float(mobius[mask]),
            "ci95": [
                percentile(mobius_boot[:, mask], 0.025),
                percentile(mobius_boot[:, mask], 0.975),
            ],
            "descriptive_only": True,
        }
        for mask in range(1, 16)
    ]
    shapley_rows = [
        {
            "component": component,
            "value": float(shapley[index]),
            "ci95": [
                percentile(shapley_boot[:, index], 0.025),
                percentile(shapley_boot[:, index], 0.975),
            ],
            "descriptive_only": True,
        }
        for index, component in enumerate(COMPONENTS)
    ]
    return {
        "model_key": model,
        "start_layer": layer,
        "empty_set_value": 0.0,
        "empty_set_definition": "unpatched_recipient_recovery",
        "set_values": values,
        "mobius_coefficients": mobius_rows,
        "shapley_values": shapley_rows,
        "shapley_efficiency_residual": float(shapley.sum() - point_values[15]),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "interpretation": (
            "descriptive allocation of the registered set function; "
            "not proof of semantic or mechanistic mediation"
        ),
    }


def cardinality_benchmark_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    prespec: Mapping[str, Any],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    model = "gemma4_31b_hf_bf16"
    layer = int(prespec["depth_plan"][model]["late_post_block_index"])
    result: list[dict[str, Any]] = []
    counter = 0
    for operator in CARDINALITY_BENCHMARKS:
        vector = _cluster_vector(
            _select(rows, model_key=model, layer=layer, operator=operator),
            expected_rows_per_cluster=8,
        )
        cell_seed = seed + counter
        indexes = _bootstrap_indexes(
            clusters=46, draws=draws, seed=cell_seed
        )
        component = operator.split("_")[1]
        result.append(
            {
                "operator": operator,
                "matched_component": component,
                **_summary(vector, indexes=indexes, seed=cell_seed),
                "descriptive_only": True,
                "interpretation": (
                    "same-cardinality candidate-crop benchmark; "
                    "not a negative control or specificity test"
                ),
            }
        )
        counter += 1
    return {
        "model_key": model,
        "start_layer": layer,
        "cells": result,
        "descriptive_only": True,
        "excluded_from_primary_holm_family": True,
    }


def terminal_call(
    *,
    primary: Mapping[str, Any],
    factorial: Mapping[str, Any],
    benchmarks: Mapping[str, Any],
) -> dict[str, Any]:
    cells = {str(row["cell_id"]): row for row in primary["cells"]}
    ordered = list(cells.values())
    late_pass = bool(ordered[0]["passes"])
    early_pass = bool(ordered[1]["passes"])
    concentration: dict[str, Any] = {
        "status": "not_evaluated_unless_reversal_confirmed"
    }
    if not late_pass:
        call = "not_replicated"
    elif not early_pass:
        call = "gemma_late_superiority_only"
    else:
        shapley = list(factorial["shapley_values"])
        positive_total = sum(max(0.0, float(row["value"])) for row in shapley)
        benchmark_by_component: dict[str, list[float]] = defaultdict(list)
        for row in benchmarks["cells"]:
            benchmark_by_component[str(row["matched_component"])].append(
                float(row["rho"])
            )
        set_values = {
            str(row["operator"]): float(row["rho"])
            for row in factorial["set_values"]
        }
        candidates: list[dict[str, Any]] = []
        for row in shapley:
            component = str(row["component"])
            allocation = float(row["value"])
            share = (
                allocation / positive_total if positive_total > 1e-12 else 0.0
            )
            positive_interval = float(row["ci95"][0]) > 0.0
            exceeds_benchmarks = True
            if component in {"Q", "P", "R"}:
                values = benchmark_by_component[component]
                exceeds_benchmarks = (
                    len(values) == 4
                    and set_values[component] > max(values)
                )
            candidates.append(
                {
                    "component": component,
                    "positive_shapley_share": share,
                    "positive_ci95": positive_interval,
                    "exceeds_all_registered_same_size_crop_benchmarks": (
                        exceeds_benchmarks
                    ),
                    "satisfies_registered_concentration_rule": (
                        share >= 0.5
                        and positive_interval
                        and exceeds_benchmarks
                    ),
                }
            )
        concentrated = [
            row
            for row in candidates
            if row["satisfies_registered_concentration_rule"]
        ]
        concentration = {
            "status": "evaluated",
            "positive_shapley_allocation_total": positive_total,
            "components": candidates,
            "concentrated_components": [
                row["component"] for row in concentrated
            ],
            "claim_boundary": (
                "position-set concentration only; not unique mediation"
            ),
        }
        call = (
            "gemma_reversal_confirmed_component_concentrated"
            if concentrated
            else "gemma_reversal_confirmed_distributed_or_interactive"
        )
    return {
        "call": call,
        "G1_late_passes": late_pass,
        "G2_early_passes": early_pass,
        "concentration": concentration,
    }


def _plot_csv(
    *,
    endpoints: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Any],
    qwen: Mapping[str, Any],
    factorial: Mapping[str, Any],
    benchmarks: Mapping[str, Any],
) -> str:
    fields = [
        "analysis_family",
        "model_key",
        "start_layer",
        "corruption",
        "direction",
        "operator",
        "component",
        "estimate",
        "ci95_low",
        "ci95_high",
        "raw_pvalue",
        "holm_pvalue",
        "descriptive_only",
    ]
    rows: list[dict[str, Any]] = []
    for row in endpoints:
        rows.append(
            {
                "analysis_family": "endpoint_recovery",
                **{key: row.get(key, "") for key in fields[1:6]},
                "component": "",
                "estimate": row["rho"],
                "ci95_low": row["ci95"][0],
                "ci95_high": row["ci95"][1],
                "raw_pvalue": "",
                "holm_pvalue": "",
                "descriptive_only": True,
            }
        )
    for row in primary["cells"]:
        rows.append(
            {
                "analysis_family": "primary_gemma_contrast",
                "model_key": row["model_key"],
                "start_layer": row["start_layer"],
                "corruption": "pooled_equal_weight",
                "direction": "pooled_equal_weight",
                "operator": row["contrast"],
                "component": "",
                "estimate": row["estimate_delta_rho"],
                "ci95_low": row["ci95"][0],
                "ci95_high": row["ci95"][1],
                "raw_pvalue": row["one_sided_bootstrap_pvalue"],
                "holm_pvalue": row["holm_adjusted_pvalue"],
                "descriptive_only": False,
            }
        )
    for row in qwen["endpoints"]:
        rows.append(
            {
                "analysis_family": "qwen_descriptive_contrast",
                "model_key": qwen["model_key"],
                "start_layer": row["start_layer"],
                "corruption": "pooled_equal_weight",
                "direction": "pooled_equal_weight",
                "operator": "K_minus_F",
                "component": "",
                "estimate": row["delta_K_minus_F"],
                "ci95_low": row["delta_K_minus_F_ci95"][0],
                "ci95_high": row["delta_K_minus_F_ci95"][1],
                "raw_pvalue": "",
                "holm_pvalue": "",
                "descriptive_only": True,
            }
        )
    for row in factorial["shapley_values"]:
        rows.append(
            {
                "analysis_family": "gemma_shapley",
                "model_key": factorial["model_key"],
                "start_layer": factorial["start_layer"],
                "corruption": "pooled_equal_weight",
                "direction": "pooled_equal_weight",
                "operator": "",
                "component": row["component"],
                "estimate": row["value"],
                "ci95_low": row["ci95"][0],
                "ci95_high": row["ci95"][1],
                "raw_pvalue": "",
                "holm_pvalue": "",
                "descriptive_only": True,
            }
        )
    for row in factorial["mobius_coefficients"]:
        rows.append(
            {
                "analysis_family": "gemma_mobius",
                "model_key": factorial["model_key"],
                "start_layer": factorial["start_layer"],
                "corruption": "pooled_equal_weight",
                "direction": "pooled_equal_weight",
                "operator": row["operator"],
                "component": "",
                "estimate": row["coefficient"],
                "ci95_low": row["ci95"][0],
                "ci95_high": row["ci95"][1],
                "raw_pvalue": "",
                "holm_pvalue": "",
                "descriptive_only": True,
            }
        )
    for row in benchmarks["cells"]:
        rows.append(
            {
                "analysis_family": "cardinality_benchmark",
                "model_key": benchmarks["model_key"],
                "start_layer": benchmarks["start_layer"],
                "corruption": "pooled_equal_weight",
                "direction": "pooled_equal_weight",
                "operator": row["operator"],
                "component": row["matched_component"],
                "estimate": row["rho"],
                "ci95_low": row["ci95"][0],
                "ci95_high": row["ci95"][1],
                "raw_pvalue": "",
                "holm_pvalue": "",
                "descriptive_only": True,
            }
        )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    f"{value:.9f}"
                    if isinstance(value, float)
                    else value
                )
                for key, value in row.items()
            }
        )
    return output.getvalue()


def _factorial_tex(factorial: Mapping[str, Any]) -> str:
    lines = [
        "% Descriptive component allocation; not a mediation certificate.",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Quantity & Component/set & Estimate & 95\\% CI low & 95\\% CI high \\\\",
        "\\midrule",
    ]
    for row in factorial["shapley_values"]:
        lines.append(
            f"Shapley & {row['component']} & {float(row['value']):.3f} & "
            f"{float(row['ci95'][0]):.3f} & {float(row['ci95'][1]):.3f} \\\\"
        )
    lines.append("\\midrule")
    for row in factorial["mobius_coefficients"]:
        lines.append(
            f"Mobius & {row['operator']} & {float(row['coefficient']):.3f} & "
            f"{float(row['ci95'][0]):.3f} & {float(row['ci95'][1]):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _write_outputs(
    *,
    out_dir: Path,
    contract: Mapping[str, Any],
    analysis: Mapping[str, Any],
    paper_values: Mapping[str, Any],
    plot_csv: str,
    factorial_tex: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / f".jobE-analysis-tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    payloads = {
        str(contract["outputs"]["analysis"]): _json_bytes(analysis),
        str(contract["outputs"]["paper_values"]): _json_bytes(paper_values),
        str(contract["outputs"]["plot_data"]): plot_csv.encode("utf-8"),
        str(contract["outputs"]["factorial_table"]): factorial_tex.encode("utf-8"),
    }
    try:
        for name, content in payloads.items():
            path = temporary / name
            path.write_bytes(content)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        for name in payloads:
            os.replace(temporary / name, out_dir / name)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def analyze_command(args: argparse.Namespace) -> int:
    repo_dir = args.repo_dir.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    contract, prespec, contract_hashes = verify_frozen_contract(
        repo_dir=repo_dir, contract_path=contract_path
    )
    frozen_draws = int(prespec["statistics"]["bootstrap_replicates"])
    frozen_seed = int(prespec["statistics"]["bootstrap_seed"])
    if args.bootstrap_draws != frozen_draws or args.seed != frozen_seed:
        raise AnalysisError(
            "production analysis must use the frozen bootstrap configuration: "
            f"draws={frozen_draws} seed={frozen_seed}"
        )
    run_dir = args.run_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    selection_path = args.selection_manifest.expanduser().resolve()
    public_path = args.public_manifest.expanduser().resolve()
    hidden_path = args.hidden_index.expanduser().resolve()

    selection, public_rows = _validate_selection(
        selection_path=selection_path,
        public_path=public_path,
        prespec=prespec,
    )
    aggregate_path = run_dir / "aggregate_summary.json"
    aggregate = read_json(aggregate_path)
    raw_rows, aggregate_sha = _validate_aggregate_before_labels(
        run_dir=run_dir,
        aggregate=aggregate,
        contract=contract,
        contract_hashes=contract_hashes,
        prespec=prespec,
    )
    manifest = _validate_job_manifest_before_labels(
        run_dir=run_dir,
        hidden_path=hidden_path,
        data_dir=data_dir,
        aggregate_sha256=aggregate_sha,
    )
    normalized = _validate_and_normalize_rows(
        rows=raw_rows,
        public_rows=public_rows,
        contract=contract,
        prespec=prespec,
    )

    # The evaluator file is not hashed or parsed until every worker-visible
    # technical and exact-inventory gate above has passed.
    expected_hidden_sha = str(prespec["data"]["selected_hidden_evaluator_sha256"])
    observed_hidden_sha = sha256_file(hidden_path)
    if observed_hidden_sha != expected_hidden_sha:
        raise AnalysisError("selected hidden evaluator index hash drifted")
    hidden_rows = read_jsonl(hidden_path)
    joined = _join_hidden(
        rows=normalized,
        public_rows=public_rows,
        hidden_rows=hidden_rows,
    )
    closure_tolerance = float(
        prespec["technical_gates"]["full_sequence_closure_max_abs_error_bits"]
    )
    closure_rows = [row for row in joined if row["operator"] == "A"]
    closure_max = max(float(row["patched_donor_error_bits"]) for row in closure_rows)
    if closure_max > closure_tolerance:
        raise AnalysisError("full-active closure failed after evaluator join")

    endpoints = endpoint_summaries(
        joined, prespec=prespec, draws=frozen_draws, seed=frozen_seed
    )
    primary = primary_gemma_analysis(
        joined, prespec=prespec, draws=frozen_draws, seed=frozen_seed
    )
    qwen = qwen_descriptive_analysis(
        joined,
        prespec=prespec,
        draws=frozen_draws,
        seed=frozen_seed + 100,
    )
    factorial = factorial_component_analysis(
        joined,
        prespec=prespec,
        draws=frozen_draws,
        seed=frozen_seed + 200,
    )
    benchmarks = cardinality_benchmark_analysis(
        joined,
        prespec=prespec,
        draws=frozen_draws,
        seed=frozen_seed + 300,
    )
    taxonomy = terminal_call(
        primary=primary, factorial=factorial, benchmarks=benchmarks
    )
    analysis = {
        "kind": ANALYSIS_KIND,
        "status": "pass",
        "claim_scope": prespec["claim_scope"],
        "cohort_terminology": prespec["claim_scope"]["cohort_terminology"],
        "contract": contract_hashes,
        "aggregate_sha256": aggregate_sha,
        "data_hashes": {
            "selection_manifest_sha256": sha256_file(selection_path),
            "public_requests_sha256": sha256_file(public_path),
            "selected_hidden_evaluator_sha256": observed_hidden_sha,
        },
        "label_firewall": {
            "hidden_index_opened_only_after_technical_and_inventory_gates": True,
            "worker_hidden_bind": False,
            "worker_hidden_index_access_receipts_all_false": True,
            "hidden_index_outside_worker_data_dir": True,
            "job_manifest_path": str(run_dir / "job_manifest.json"),
            "job_manifest_status": manifest.get("status"),
        },
        "selection": {
            "status": selection["status"],
            "request_count": len(public_rows),
            "frame_count": selection["selected"]["frame_count"],
            "participant_cluster_count": selection["selected"][
                "participant_cluster_count"
            ],
            "jobD_identity_overlap_counts": {
                field: selection["overlap_audit"][field]["overlap_count"]
                for field in (
                    "participant_cluster_id",
                    "source_video_id",
                    "frame_id",
                    "donor_id",
                )
            },
        },
        "raw_outcome_row_count": len(raw_rows),
        "joined_outcome_row_count": len(joined),
        "participant_cluster_count": 46,
        "endpoint_summaries": endpoints,
        "primary_gemma_holm_family": primary,
        "terminal_taxonomy": taxonomy,
        "qwen_descriptive": qwen,
        "gemma_l34_component_game": factorial,
        "cardinality_benchmarks": benchmarks,
        "positive_contract_control": {
            "operator": "A",
            "row_count": len(closure_rows),
            "max_abs_patched_donor_error_bits": closure_max,
            "tolerance_bits": closure_tolerance,
            "status": "pass",
        },
        "bootstrap": {
            "draws": frozen_draws,
            "seed": frozen_seed,
            "unit": "participant_cluster_id",
            "cluster_weighting": "equal",
        },
        "interpretation_limits": {
            "factorial": prespec["statistics"]["shapley_interpretation"],
            "cardinality_benchmarks": (
                "descriptive token-count benchmarks, not negative controls"
            ),
            "qwen": "secondary descriptive analysis outside the Holm family",
        },
    }
    analysis_bytes = _json_bytes(analysis)
    analysis_sha = hashlib.sha256(analysis_bytes).hexdigest()
    paper_values = {
        "kind": PAPER_VALUES_KIND,
        "status": "pass",
        "primary_gemma_holm_family": primary,
        "terminal_taxonomy": taxonomy,
        "qwen_descriptive": qwen,
        "gemma_l34_shapley_values": factorial["shapley_values"],
        "gemma_l34_mobius_coefficients": factorial["mobius_coefficients"],
        "cardinality_benchmarks": benchmarks,
        "claim_scope": prespec["claim_scope"],
        "source_analysis_sha256": analysis_sha,
    }
    _write_outputs(
        out_dir=args.out_dir.expanduser().resolve(),
        contract=contract,
        analysis=analysis,
        paper_values=paper_values,
        plot_csv=_plot_csv(
            endpoints=endpoints,
            primary=primary,
            qwen=qwen,
            factorial=factorial,
            benchmarks=benchmarks,
        ),
        factorial_tex=_factorial_tex(factorial),
    )
    print(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-manifest",
        "--mechanism-manifest",
        dest="selection_manifest",
        type=Path,
        required=True,
        help=(
            "Frozen Job E selection manifest. The mechanism-manifest spelling "
            "is retained only for batch-script compatibility."
        ),
    )
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--hidden-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--require-hidden-outside-data-dir",
        action="store_true",
        help="Compatibility flag; filesystem separation is always enforced.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return analyze_command(args)
    except (AnalysisError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
