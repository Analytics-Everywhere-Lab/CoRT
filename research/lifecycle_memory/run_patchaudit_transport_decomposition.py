#!/usr/bin/env python3
"""Run the PatchAudit persistent-clamp and complement follow-up.

The worker never opens evaluator labels. It emits evaluator-unjoined A-minus-B
answer-token margins with hash-bound, request-atomic resume receipts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory import run_phase2_reanchored_causal_patch as _base
from research.lifecycle_memory import (
    run_phase2_reanchored_answer_token_causal_patch as _answer,
)
from research.lifecycle_memory.patchaudit_transport_operators import (
    PatchInstruction,
    PositionPartition,
    TransportGateError,
    TransportHookAudit,
    TransportSite,
    active_positions_from_inputs,
    assert_complete_capture,
    capture_transport_states,
    patch_transport_outputs,
    position_partition,
    recipient_positions_for_operator,
    selected_from_capture,
    validate_patch_audits,
)


KIND = "patchaudit_transport_decomposition"
OUTPUT_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260726
ARMS = ("real", "blank_full", "wrong_time")
CORRUPTIONS = ("blank_full", "wrong_time")
OPERATORS = (
    "single_full_frame",
    "persistent_full_frame",
    "complement_once",
    "full_sequence_once",
)
EXPECTED_MODEL_BY_RANK = {
    0: "qwen35_27b_hf_bf16",
    1: "qwen35_27b_hf_bf16",
    2: "gemma4_31b_hf_bf16",
    3: "gemma4_31b_hf_bf16",
}
DEFAULT_JOB_DIR = Path(
    "docs/research/lifecycle_memory/phase 2 research/jobs/"
    "patchaudit_jobD_transport_decomposition"
)
EXPECTED_PROMPT = _answer.FROZEN_USER_PROMPT


class GateError(RuntimeError):
    pass


def utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise GateError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl_fsync(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def required_hash_env(name: str) -> str:
    value = str(os.environ.get(name) or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise GateError(f"{name} must be a 64-character SHA-256")
    return value


def verify_contract(
    *, repo_dir: Path, contract_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    contract_path = contract_path.resolve()
    contract = read_json(contract_path)
    if (
        contract.get("kind") != "patchaudit_transport_decomposition_job_contract"
        or contract.get("version") != 1
    ):
        raise GateError("invalid transport job contract kind or version")
    prespec_rel = Path(str(contract["prespec"]["path"]))
    prespec_path = (repo_dir / prespec_rel).resolve()
    expected_prespec_sha = str(contract["prespec"]["sha256"])
    observed_prespec_sha = sha256_file(prespec_path)
    if observed_prespec_sha != expected_prespec_sha:
        raise GateError(
            f"transport prespec drift: expected={expected_prespec_sha} "
            f"observed={observed_prespec_sha}"
        )
    prespec = read_json(prespec_path)
    if (
        prespec.get("kind") != "patchaudit_transport_decomposition_preregistration"
        or prespec.get("version") != 1
        or prespec.get("analysis_status")
        != "prospectively_frozen_before_any_new_transport_decomposition_outcome"
    ):
        raise GateError("invalid transport prespec kind, version, or frozen status")
    if tuple(prespec.get("corruptions", ())) != CORRUPTIONS:
        raise GateError("transport corruption inventory drifted")
    if set(prespec.get("operators", {})) != set(OPERATORS):
        raise GateError("transport operator inventory drifted")
    model_keys = [str(row.get("model_key")) for row in prespec.get("models", [])]
    if model_keys != [
        "qwen35_27b_hf_bf16",
        "gemma4_31b_hf_bf16",
    ]:
        raise GateError("transport model inventory drifted")
    per_request_counts = {
        len(CORRUPTIONS)
        * 2
        * len(
            prespec["depth_plan"][model_key][
                "ordered_post_block_indexes"
            ]
        )
        * len(OPERATORS)
        for model_key in model_keys
    }
    expected_per_request = int(
        contract["completion"]["expected_outcomes_per_request"]
    )
    if per_request_counts != {expected_per_request}:
        raise GateError("transport per-request Cartesian count drifted")
    expected_full = (
        len(model_keys)
        * int(contract["completion"]["expected_requests_per_model"])
        * expected_per_request
    )
    if expected_full != int(
        contract["completion"]["expected_full_outcome_rows"]
    ):
        raise GateError("transport full Cartesian count drifted")
    if (
        float(contract["canary"]["answer_token_equivalence_tolerance_bits"])
        != float(
            prespec["technical_gates"][
                "answer_token_equivalence_tolerance_bits"
            ]
        )
    ):
        raise GateError("answer-token equivalence tolerance drifted")
    return (
        contract,
        prespec,
        {
            "contract_sha256": sha256_file(contract_path),
            "prespec_sha256": observed_prespec_sha,
        },
    )


def load_model_plan(path: Path) -> list[dict[str, Any]]:
    plan = read_json(path.resolve())
    if not isinstance(plan, list) or len(plan) != 2:
        raise GateError("model plan must contain exactly two anchor rows")
    expected = {
        "qwen35_27b_hf_bf16": {
            "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654",
            "ranks": [0, 1],
        },
        "gemma4_31b_hf_bf16": {
            "revision": "3548789868c5356dbf307c98e6f609007b82b3eb",
            "ranks": [2, 3],
        },
    }
    observed = {
        str(row.get("model_key")): {
            "revision": row.get("revision"),
            "ranks": row.get("ranks"),
        }
        for row in plan
    }
    if observed != expected:
        raise GateError(f"model plan drift: {observed}")
    return plan


def model_and_shard(
    plan: Sequence[Mapping[str, Any]], rank: int, world_size: int
) -> tuple[str, int, int]:
    if world_size != 4 or rank not in range(4):
        raise GateError("transport job requires exactly four ranks")
    for row in plan:
        ranks = [int(value) for value in row["ranks"]]
        if rank in ranks:
            return str(row["model_key"]), ranks.index(rank), len(ranks)
    raise GateError(f"rank {rank} is absent from the model plan")


def shard_rows(
    rows: Sequence[dict[str, Any]], *, shard_index: int, shard_count: int
) -> list[dict[str, Any]]:
    frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        frames[str(row["frame_unit_id"])].append(row)
    ordered_frames = sorted(
        frames,
        key=lambda frame_id: hashlib.sha256(
            f"{DEFAULT_SEED}|transport-shard|{frame_id}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    for index, frame_id in enumerate(ordered_frames):
        pair = sorted(frames[frame_id], key=lambda row: str(row["order_id"]))
        if len(pair) != 2:
            raise GateError(f"frame {frame_id} lacks two counterbalanced rows")
        if index % shard_count == shard_index:
            selected.extend(pair)
    return selected


def _prepared_shared(prepared: Any) -> Any:
    return _answer._assert_shared_choice_inputs(prepared)


def _margin_ab(scores: Mapping[str, float]) -> float:
    return _base.a_minus_b_margin_bits(scores)


def _decoder_sites(
    model: Any, *, model_key: str, prespec: Mapping[str, Any]
) -> tuple[list[TransportSite], dict[int, TransportSite], dict[str, Any]]:
    depth = prespec["depth_plan"][model_key]
    starts = [int(value) for value in depth["ordered_post_block_indexes"]]
    last = int(depth["last_post_block_index"])
    all_layers = list(range(min(starts), last + 1))
    sites, receipt = _base.architecture_sites(
        model,
        model_key=model_key,
        phase="refine",
        refinement_layers={
            "decoder_visual_token_residual": all_layers,
            "decoder_final_question_token_residual": [],
        },
    )
    converted = [
        TransportSite(
            site_id=_base.site_state_name(site),
            module_name=site.module_name,
            layer_index=int(site.layer_index),
        )
        for site in sites
    ]
    by_layer = {site.layer_index: site for site in converted}
    if set(by_layer) != set(all_layers):
        raise GateError("architecture adapter did not resolve the full clamp range")
    return converted, by_layer, receipt


def _partition_for_prepared(prepared: Any) -> PositionPartition:
    shared = _prepared_shared(prepared)
    slot_map = shared.slot_map
    full_frame = slot_map.positions((0,))
    active = active_positions_from_inputs(shared.inputs, slot_map.sequence_length)
    return position_partition(
        sequence_length=slot_map.sequence_length,
        active_positions=active,
        full_frame_positions=full_frame,
    )


def _assert_partition_equal(left: PositionPartition, right: PositionPartition) -> None:
    if asdict(left) != asdict(right):
        raise GateError("source and recipient active-position partitions differ")


def native_forward(
    *,
    model: Any,
    prepared: Any,
    sites: Sequence[TransportSite],
    partition: PositionPartition,
    detailed_site_ids: Sequence[str],
) -> tuple[float, dict[str, Any]]:
    import torch

    shared = _prepared_shared(prepared)
    with capture_transport_states(
        model=model,
        sites=sites,
        partition=partition,
        detailed_site_ids=detailed_site_ids,
    ) as captures:
        with torch.inference_mode():
            output = model(**shared.inputs, use_cache=False, return_dict=True)
    assert_complete_capture(captures, sites, detailed_site_ids=detailed_site_ids)
    scores = {
        choice: _answer.score_suffix(output, prepared.choices[choice])
        for choice in _base.CHOICES
    }
    if not all(math.isfinite(float(value)) for value in scores.values()):
        raise GateError("native answer-token score is nonfinite")
    del output
    return _margin_ab(scores), captures


def _instructions(
    *,
    operator: str,
    start_layer: int,
    sites_by_layer: Mapping[int, TransportSite],
    donor_captures: Mapping[str, Any],
    partition: PositionPartition,
) -> list[PatchInstruction]:
    if operator == "persistent_full_frame":
        layers = list(range(start_layer, max(sites_by_layer) + 1))
    else:
        layers = [start_layer]
    instructions: list[PatchInstruction] = []
    for layer in layers:
        site = sites_by_layer[layer]
        capture = donor_captures[site.site_id]
        instructions.append(
            PatchInstruction(
                site=site,
                source_selected=selected_from_capture(
                    capture, operator=operator, partition=partition
                ),
                recipient_positions=recipient_positions_for_operator(
                    operator, partition
                ),
                operator=operator,
            )
        )
    return instructions


def _compact_audit(audits: Sequence[TransportHookAudit]) -> dict[str, Any]:
    payload = [audit.as_json() for audit in audits]
    return {
        "status": "pass",
        "expected_hook_count": len(audits),
        "all_fired_once": all(audit.fire_count == 1 for audit in audits),
        "all_removed": all(audit.hook_removed for audit in audits),
        "all_finite": all(audit.finite for audit in audits),
        "max_abs_change_preserved": max(
            (audit.max_abs_change_preserved for audit in audits), default=0.0
        ),
        "changed_hook_count": sum(
            int(audit.changed_selected_value_count > 0) for audit in audits
        ),
        "first_site_id": audits[0].site_id,
        "last_site_id": audits[-1].site_id,
        "site_ids_sha256": sha256_json([audit.site_id for audit in audits]),
        "full_audit_sha256": sha256_json(payload),
    }


def patched_forward(
    *,
    model: Any,
    prepared: Any,
    instructions: Sequence[PatchInstruction],
    sequence_length: int,
    allow_noop: bool,
) -> tuple[float, list[TransportHookAudit]]:
    import torch

    shared = _prepared_shared(prepared)
    with patch_transport_outputs(
        model=model,
        instructions=instructions,
        sequence_length=sequence_length,
    ) as audits:
        with torch.inference_mode():
            output = model(**shared.inputs, use_cache=False, return_dict=True)
    validate_patch_audits(
        audits,
        expected_site_ids=[instruction.site.site_id for instruction in instructions],
        allow_noop=allow_noop,
    )
    scores = {
        choice: _answer.score_suffix(output, prepared.choices[choice])
        for choice in _base.CHOICES
    }
    if not all(math.isfinite(float(value)) for value in scores.values()):
        raise GateError("patched answer-token score is nonfinite")
    del output
    return _margin_ab(scores), audits


def outcome_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("model_key"),
        row.get("request_id"),
        row.get("corruption"),
        row.get("direction"),
        row.get("start_layer"),
        row.get("operator"),
    )


def expected_outcome_identities(
    *,
    prespec: Mapping[str, Any],
    model_key: str,
    request_ids: Sequence[str],
) -> set[tuple[Any, ...]]:
    layers = [
        int(value)
        for value in prespec["depth_plan"][model_key][
            "ordered_post_block_indexes"
        ]
    ]
    return {
        (model_key, request_id, corruption, direction, layer, operator)
        for request_id in request_ids
        for corruption in CORRUPTIONS
        for direction in ("rescue", "damage")
        for layer in layers
        for operator in OPERATORS
    }


def _validate_outcome_row(
    row: Mapping[str, Any],
    *,
    rank: int,
    model_key: str,
    public_record: Mapping[str, Any],
) -> None:
    if (
        row.get("kind") != "patchaudit_transport_outcome"
        or row.get("output_schema_version") != OUTPUT_SCHEMA_VERSION
        or row.get("rank") != rank
        or row.get("model_key") != model_key
    ):
        raise GateError("outcome kind, schema, rank, or model identity drifted")
    request_id = str(row.get("request_id") or "")
    if request_id != str(public_record["request_id"]):
        raise GateError("outcome request identity differs from its public record")
    if (
        str(row.get("frame_unit_id")) != str(public_record["frame_unit_id"])
        or str(row.get("order_id")) != str(public_record["order_id"])
    ):
        raise GateError(f"outcome public identity drift for {request_id}")
    corruption = str(row.get("corruption") or "")
    direction = str(row.get("direction") or "")
    expected_arms = (
        ("real", corruption) if direction == "rescue" else (corruption, "real")
    )
    if direction not in {"rescue", "damage"} or (
        str(row.get("source_arm") or ""),
        str(row.get("recipient_arm") or ""),
    ) != expected_arms:
        raise GateError(f"outcome source/recipient arm drift for {request_id}")
    numeric_fields = (
        "real_margin_a_minus_b_bits",
        "corrupt_margin_a_minus_b_bits",
        "recipient_margin_a_minus_b_bits",
        "donor_margin_a_minus_b_bits",
        "patched_margin_a_minus_b_bits",
        "baseline_donor_gap_abs_bits",
        "patched_donor_error_abs_bits",
    )
    if not all(
        math.isfinite(float(row[field])) for field in numeric_fields
    ):
        raise GateError(f"outcome contains a nonfinite margin for {request_id}")
    real_margin = float(row["real_margin_a_minus_b_bits"])
    corrupt_margin = float(row["corrupt_margin_a_minus_b_bits"])
    recipient_margin = float(row["recipient_margin_a_minus_b_bits"])
    donor_margin = float(row["donor_margin_a_minus_b_bits"])
    patched_margin = float(row["patched_margin_a_minus_b_bits"])
    expected_recipient = corrupt_margin if direction == "rescue" else real_margin
    expected_donor = real_margin if direction == "rescue" else corrupt_margin
    recomputed_baseline = abs(recipient_margin - donor_margin)
    recomputed_error = abs(patched_margin - donor_margin)
    if (
        abs(recipient_margin - expected_recipient) > 1e-12
        or abs(donor_margin - expected_donor) > 1e-12
        or abs(
            float(row["baseline_donor_gap_abs_bits"]) - recomputed_baseline
        )
        > 1e-12
        or abs(
            float(row["patched_donor_error_abs_bits"]) - recomputed_error
        )
        > 1e-12
    ):
        raise GateError(f"outcome margin algebra drifted for {request_id}")
    audit = row.get("audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("status") != "pass"
        or audit.get("all_fired_once") is not True
        or audit.get("all_removed") is not True
        or audit.get("all_finite") is not True
    ):
        raise GateError(f"outcome hook audit failed for {request_id}")


def validate_rank_artifacts(
    *,
    run_dir: Path,
    rank: int,
    summary: Mapping[str, Any],
    prespec: Mapping[str, Any],
    public_by_id: Mapping[str, Mapping[str, Any]],
    canary: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rank_dir = run_dir / f"rank_{rank:03d}"
    manifest_path = rank_dir / "worker_manifest.json"
    outcomes_path = rank_dir / "outcomes.jsonl"
    completed_path = rank_dir / "completed_requests.jsonl"
    for path in (manifest_path, outcomes_path, completed_path):
        if not path.is_file():
            raise GateError(f"rank {rank} is missing {path.name}")
    manifest = read_json(manifest_path)
    model_key = EXPECTED_MODEL_BY_RANK[rank]
    if (
        summary.get("kind") != "patchaudit_transport_worker_summary"
        or summary.get("output_schema_version") != OUTPUT_SCHEMA_VERSION
        or summary.get("rank") != rank
        or summary.get("model_key") != model_key
        or summary.get("world_size") != 4
    ):
        raise GateError(f"rank {rank} worker summary identity drifted")
    if (
        manifest.get("kind") != "patchaudit_transport_worker_manifest"
        or manifest.get("output_schema_version") != OUTPUT_SCHEMA_VERSION
        or manifest.get("rank") != rank
        or manifest.get("model_key") != model_key
        or manifest.get("world_size") != 4
        or manifest.get("status") != summary.get("status")
        or manifest.get("hidden_index_opened") is not False
    ):
        raise GateError(f"rank {rank} worker manifest identity drifted")
    records = manifest.get("selected_request_records")
    if not isinstance(records, list):
        raise GateError(f"rank {rank} has no selected public request records")
    expected_selected_count = 1 if canary else 80
    if (
        len(records) != expected_selected_count
        or manifest.get("selected_request_count") != expected_selected_count
        or manifest.get("selected_request_records_sha256") != sha256_json(records)
    ):
        raise GateError(f"rank {rank} selected request inventory drifted")
    selected_by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise GateError(f"rank {rank} has a non-object public request record")
        request_id = str(record.get("request_id") or "")
        if request_id in selected_by_id or request_id not in public_by_id:
            raise GateError(f"rank {rank} has an invalid selected request ID")
        public = public_by_id[request_id]
        expected_record = {
            "request_id": request_id,
            "frame_unit_id": str(public["frame_unit_id"]),
            "order_id": str(public["order_id"]),
        }
        if dict(record) != expected_record:
            raise GateError(f"rank {rank} selected public record drifted")
        selected_by_id[request_id] = record
    rows = read_jsonl(outcomes_path)
    receipts = read_jsonl(completed_path)
    if (
        summary.get("outcomes_sha256") != sha256_file(outcomes_path)
        or summary.get("completed_requests_sha256") != sha256_file(completed_path)
        or summary.get("outcome_rows") != len(rows)
        or summary.get("requests_completed") != len(receipts)
        or summary.get("completion_receipt_rows") != len(receipts)
    ):
        raise GateError(f"rank {rank} output file binding drifted")
    receipt_by_id: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        request_id = str(receipt.get("request_id") or "")
        if (
            receipt.get("kind") != "patchaudit_transport_request_receipt"
            or receipt.get("output_schema_version") != OUTPUT_SCHEMA_VERSION
            or request_id in receipt_by_id
            or request_id not in selected_by_id
        ):
            raise GateError(f"rank {rank} request receipt contract drifted")
        receipt_by_id[request_id] = receipt
    expected_all = expected_outcome_identities(
        prespec=prespec,
        model_key=model_key,
        request_ids=list(selected_by_id),
    )
    actual = [outcome_identity(row) for row in rows]
    if len(actual) != len(set(actual)) or not set(actual).issubset(expected_all):
        raise GateError(f"rank {rank} outcome Cartesian inventory drifted")
    rows_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    real_margins_by_request: dict[str, set[float]] = defaultdict(set)
    corrupt_margins_by_request_arm: dict[
        tuple[str, str], set[float]
    ] = defaultdict(set)
    for row in rows:
        request_id = str(row.get("request_id") or "")
        public_record = selected_by_id.get(request_id)
        if public_record is None:
            raise GateError(f"rank {rank} emitted an unassigned request")
        _validate_outcome_row(
            row,
            rank=rank,
            model_key=model_key,
            public_record=public_record,
        )
        rows_by_request[request_id].append(row)
        real_margins_by_request[request_id].add(
            float(row["real_margin_a_minus_b_bits"])
        )
        corrupt_margins_by_request_arm[
            (request_id, str(row["corruption"]))
        ].add(float(row["corrupt_margin_a_minus_b_bits"]))
    if any(len(values) != 1 for values in real_margins_by_request.values()):
        raise GateError(f"rank {rank} native real margins are not invariant")
    if any(
        len(values) != 1
        for values in corrupt_margins_by_request_arm.values()
    ):
        raise GateError(f"rank {rank} native corrupt margins are not invariant")
    expected_rows_per_request = (
        len(CORRUPTIONS)
        * 2
        * len(
            prespec["depth_plan"][model_key][
                "ordered_post_block_indexes"
            ]
        )
        * len(OPERATORS)
    )
    if set(rows_by_request) != set(receipt_by_id):
        raise GateError(f"rank {rank} rows and completion receipts differ")
    for request_id, receipt in receipt_by_id.items():
        request_rows = rows_by_request[request_id]
        expected_request = expected_outcome_identities(
            prespec=prespec,
            model_key=model_key,
            request_ids=[request_id],
        )
        if (
            len(request_rows) != expected_rows_per_request
            or {outcome_identity(row) for row in request_rows} != expected_request
            or receipt.get("result_row_count") != expected_rows_per_request
            or receipt.get("result_batch_sha256") != sha256_json(request_rows)
            or receipt.get("resume_binding_sha256")
            != summary.get("resume_binding_sha256")
        ):
            raise GateError(f"rank {rank} request {request_id} is not atomic")
    expected_completed = set(selected_by_id)
    observed_completed = set(receipt_by_id)
    if summary.get("status") == "pass":
        if observed_completed != expected_completed or set(actual) != expected_all:
            raise GateError(f"rank {rank} passing worker is incomplete")
    elif summary.get("status") == "partial_checkpoint":
        if not observed_completed.issubset(expected_completed):
            raise GateError(f"rank {rank} partial worker inventory is invalid")
    else:
        raise GateError(f"rank {rank} worker status is not aggregatable")
    return rows, manifest


def validate_bound_canary(
    *,
    job_manifest: Mapping[str, Any],
    immutable_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    profile_path = Path(str(job_manifest.get("canary_profile_path") or "")).resolve()
    aggregate_path = Path(
        str(job_manifest.get("canary_aggregate_path") or "")
    ).resolve()
    if not profile_path.is_file() or not aggregate_path.is_file():
        raise GateError("full job has no readable bound canary artifacts")
    profile_sha = sha256_file(profile_path)
    aggregate_sha = sha256_file(aggregate_path)
    if (
        profile_sha != job_manifest.get("canary_profile_sha256")
        or aggregate_sha != job_manifest.get("canary_aggregate_sha256")
    ):
        raise GateError("full job canary artifact hash binding drifted")
    profile = read_json(profile_path)
    aggregate = read_json(aggregate_path)
    if (
        profile.get("kind") != "patchaudit_transport_runtime_profile"
        or profile.get("status") != "pass"
        or aggregate.get("kind") != "patchaudit_transport_aggregate"
        or aggregate.get("status") != "pass"
        or aggregate.get("mode") != "canary"
        or profile.get("canary_aggregate_sha256") != aggregate_sha
        or profile.get("immutable_hashes") != immutable_hashes
        or aggregate.get("immutable_hashes") != immutable_hashes
    ):
        raise GateError("full job is not bound to a passing identical canary")
    return {
        "status": "pass",
        "canary_profile_path": str(profile_path),
        "canary_profile_sha256": profile_sha,
        "canary_aggregate_path": str(aggregate_path),
        "canary_aggregate_sha256": aggregate_sha,
    }


def evaluate_request(
    *,
    runtime: Any,
    model: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    model_key: str,
    rank: int,
    request: dict[str, Any],
    data_dir: Path,
    sites: Sequence[TransportSite],
    sites_by_layer: Mapping[int, TransportSite],
    prespec: Mapping[str, Any],
    canary: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    request_id = str(request["request_id"])
    prepared: dict[str, Any] = {}
    for arm in ARMS:
        prepared[arm] = _answer.prepare_arm(
            runtime=runtime,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            model=model,
            model_key=model_key,
            row=request,
            arm=arm,
            data_dir=data_dir,
            candidate_completions={"A": "A", "B": "B"},
        )
    for corruption in CORRUPTIONS:
        _base.assert_prepared_alignment(prepared["real"], prepared[corruption])
    partitions = {arm: _partition_for_prepared(prepared[arm]) for arm in ARMS}
    for arm in CORRUPTIONS:
        _assert_partition_equal(partitions["real"], partitions[arm])
    partition = partitions["real"]
    starts = [
        int(value)
        for value in prespec["depth_plan"][model_key][
            "ordered_post_block_indexes"
        ]
    ]
    detailed = [sites_by_layer[layer].site_id for layer in starts]
    native_elapsed_s = 0.0
    native_started = time.perf_counter()
    real_margin, real_captures = native_forward(
        model=model,
        prepared=prepared["real"],
        sites=sites,
        partition=partition,
        detailed_site_ids=detailed,
    )
    native_elapsed_s += time.perf_counter() - native_started
    native_forwards = 1
    patch_forwards = 0
    rows: list[dict[str, Any]] = []
    detailed_canary_audits: list[dict[str, Any]] = []
    self_checks: list[dict[str, Any]] = []
    operator_timings: Counter[str] = Counter()

    for corruption in CORRUPTIONS:
        native_started = time.perf_counter()
        corrupt_margin, corrupt_captures = native_forward(
            model=model,
            prepared=prepared[corruption],
            sites=sites,
            partition=partition,
            detailed_site_ids=detailed,
        )
        native_elapsed_s += time.perf_counter() - native_started
        native_forwards += 1
        for direction, donor_arm, recipient_arm, donor_margin, recipient_margin, donor_captures in (
            (
                "rescue",
                "real",
                corruption,
                real_margin,
                corrupt_margin,
                real_captures,
            ),
            (
                "damage",
                corruption,
                "real",
                corrupt_margin,
                real_margin,
                corrupt_captures,
            ),
        ):
            for start_layer in starts:
                for operator in OPERATORS:
                    instructions = _instructions(
                        operator=operator,
                        start_layer=start_layer,
                        sites_by_layer=sites_by_layer,
                        donor_captures=donor_captures,
                        partition=partition,
                    )
                    patch_started = time.perf_counter()
                    patched_margin, audits = patched_forward(
                        model=model,
                        prepared=prepared[recipient_arm],
                        instructions=instructions,
                        sequence_length=partition.sequence_length,
                        allow_noop=False,
                    )
                    operator_timings[operator] += time.perf_counter() - patch_started
                    patch_forwards += 1
                    donor_error = abs(patched_margin - donor_margin)
                    baseline_gap = abs(recipient_margin - donor_margin)
                    full_sequence_tolerance = float(
                        prespec["technical_gates"][
                            "full_sequence_closure_max_abs_error_bits"
                        ]
                    )
                    if (
                        operator == "full_sequence_once"
                        and donor_error > full_sequence_tolerance
                    ):
                        raise GateError(
                            f"full-sequence closure failed model={model_key} "
                            f"request={request_id} corruption={corruption} "
                            f"direction={direction} layer={start_layer} "
                            f"error={donor_error}"
                        )
                    row = {
                        "kind": "patchaudit_transport_outcome",
                        "output_schema_version": OUTPUT_SCHEMA_VERSION,
                        "model_key": model_key,
                        "rank": rank,
                        "request_id": request_id,
                        "frame_unit_id": str(request["frame_unit_id"]),
                        "order_id": str(request["order_id"]),
                        "corruption": corruption,
                        "direction": direction,
                        "source_arm": donor_arm,
                        "recipient_arm": recipient_arm,
                        "start_layer": start_layer,
                        "operator": operator,
                        "real_margin_a_minus_b_bits": real_margin,
                        "corrupt_margin_a_minus_b_bits": corrupt_margin,
                        "recipient_margin_a_minus_b_bits": recipient_margin,
                        "donor_margin_a_minus_b_bits": donor_margin,
                        "patched_margin_a_minus_b_bits": patched_margin,
                        "baseline_donor_gap_abs_bits": baseline_gap,
                        "patched_donor_error_abs_bits": donor_error,
                        "partition": {
                            "sequence_length": partition.sequence_length,
                            "active_count": len(partition.active_positions),
                            "full_frame_count": len(partition.full_frame_positions),
                            "complement_count": len(partition.complement_positions),
                            "padding_count": len(partition.padding_positions),
                            "active_sha256": partition.active_sha256,
                            "full_frame_sha256": partition.full_frame_sha256,
                            "complement_sha256": partition.complement_sha256,
                        },
                        "audit": _compact_audit(audits),
                    }
                    rows.append(row)
                    if canary:
                        detailed_canary_audits.append(
                            {
                                "identity": list(outcome_identity(row)),
                                "audits": [audit.as_json() for audit in audits],
                            }
                        )
        if canary and corruption == "blank_full":
            self_layer = int(
                prespec["depth_plan"][model_key][
                    "central_primary_post_block_indexes"
                ][0]
            )
            for operator in OPERATORS:
                instructions = _instructions(
                    operator=operator,
                    start_layer=self_layer,
                    sites_by_layer=sites_by_layer,
                    donor_captures=real_captures,
                    partition=partition,
                )
                self_margin, audits = patched_forward(
                    model=model,
                    prepared=prepared["real"],
                    instructions=instructions,
                    sequence_length=partition.sequence_length,
                    allow_noop=True,
                )
                patch_forwards += 1
                error = abs(self_margin - real_margin)
                tolerance = float(
                    prespec["technical_gates"]["self_patch_max_abs_error_bits"]
                )
                if error > tolerance:
                    raise GateError(
                        f"self-patch invariance failed {model_key} {operator}: {error}"
                    )
                self_checks.append(
                    {
                        "operator": operator,
                        "start_layer": self_layer,
                        "margin_error_bits": error,
                        "status": "pass",
                        "audit": _compact_audit(audits),
                    }
                )
        del corrupt_captures
    expected = 64
    identities = [outcome_identity(row) for row in rows]
    if len(rows) != expected or len(set(identities)) != expected:
        raise GateError(
            f"request outcome contract failed: rows={len(rows)} "
            f"unique={len(set(identities))} expected={expected}"
        )
    elapsed = time.perf_counter() - started
    return rows, {
        "request_id": request_id,
        "elapsed_s": elapsed,
        "native_capture_elapsed_s": native_elapsed_s,
        "native_forward_count": native_forwards,
        "patch_forward_count": patch_forwards,
        "total_forward_count": native_forwards + patch_forwards,
        "operator_elapsed_s": dict(operator_timings),
        "partition": asdict(partition),
        "self_checks": self_checks,
        "detailed_canary_audits": detailed_canary_audits,
    }


def sanitize_resume(
    *,
    outcomes_path: Path,
    completed_path: Path,
    expected_binding_sha256: str,
    expected_request_ids: set[str],
    expected_rows_per_request: int,
) -> dict[str, Any]:
    outcomes = read_jsonl(outcomes_path)
    receipts = read_jsonl(completed_path)
    receipt_by_request: dict[str, dict[str, Any]] = {}
    duplicate_receipts: set[str] = set()
    for receipt in receipts:
        request_id = str(receipt.get("request_id") or "")
        if request_id in receipt_by_request:
            duplicate_receipts.add(request_id)
        receipt_by_request[request_id] = receipt
    rows_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        rows_by_request[str(row.get("request_id") or "")].append(row)
    kept_rows: list[dict[str, Any]] = []
    kept_receipts: list[dict[str, Any]] = []
    completed: list[str] = []
    dropped_rows = 0
    for request_id in sorted(expected_request_ids):
        request_rows = rows_by_request.get(request_id, [])
        receipt = receipt_by_request.get(request_id)
        valid = (
            request_id not in duplicate_receipts
            and receipt is not None
            and receipt.get("kind") == "patchaudit_transport_request_receipt"
            and receipt.get("resume_binding_sha256") == expected_binding_sha256
            and receipt.get("output_schema_version") == OUTPUT_SCHEMA_VERSION
            and receipt.get("result_row_count") == expected_rows_per_request
            and len(request_rows) == expected_rows_per_request
            and len({outcome_identity(row) for row in request_rows})
            == expected_rows_per_request
            and receipt.get("result_batch_sha256") == sha256_json(request_rows)
        )
        if valid:
            kept_rows.extend(request_rows)
            kept_receipts.append(receipt)
            completed.append(request_id)
        else:
            dropped_rows += len(request_rows)
    extra_rows = sum(
        len(rows)
        for request_id, rows in rows_by_request.items()
        if request_id not in expected_request_ids
    )
    dropped_rows += extra_rows
    if outcomes or receipts:
        atomic_write_jsonl(outcomes_path, kept_rows)
        atomic_write_jsonl(completed_path, kept_receipts)
    return {
        "completed_request_ids": completed,
        "kept_rows": len(kept_rows),
        "dropped_rows": dropped_rows,
        "dropped_receipts": len(receipts) - len(kept_receipts),
    }


def preflight_command(args: argparse.Namespace) -> int:
    repo_dir = args.repo_dir.expanduser().resolve()
    contract, prespec, hashes = verify_contract(
        repo_dir=repo_dir, contract_path=args.contract.expanduser().resolve()
    )
    manifest_path = args.mechanism_manifest.expanduser().resolve()
    public_path = args.public_manifest.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    mechanism = _base.validate_mechanism_manifest(
        data_dir=data_dir, manifest_path=manifest_path, public_path=public_path
    )
    rows = read_jsonl(public_path)
    dataset = _base.validate_dataset(
        public_rows=rows,
        hidden_rows=None,
        data_dir=data_dir,
        expected_frames=80,
        expected_user_prompt=EXPECTED_PROMPT,
    )
    receipt = {
        "kind": "patchaudit_transport_preflight",
        "status": "pass",
        "hidden_index_opened": False,
        "contract": hashes,
        "mechanism": mechanism,
        "dataset": dataset,
        "operator_count": len(OPERATORS),
        "expected_full_outcome_rows": contract["completion"][
            "expected_full_outcome_rows"
        ],
        "models": [row["model_key"] for row in prespec["models"]],
        "created_at": utc_now(),
    }
    if args.out:
        atomic_write_json(args.out.expanduser().resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def worker_command(args: argparse.Namespace) -> int:
    import torch

    worker_started = time.perf_counter()
    repo_dir = args.repo_dir.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    contract, prespec, contract_hashes = verify_contract(
        repo_dir=repo_dir, contract_path=contract_path
    )
    env_hashes = {
        "release_manifest_sha256": required_hash_env("RELEASE_MANIFEST_SHA256"),
        "code_snapshot_sha256": required_hash_env("CODE_SNAPSHOT_SHA256"),
        "contract_sha256": required_hash_env("CONTRACT_SHA256"),
        "prespec_sha256": required_hash_env("PRESPEC_SHA256"),
        "mechanism_manifest_sha256": required_hash_env(
            "MECHANISM_MANIFEST_SHA256"
        ),
        "public_manifest_sha256": required_hash_env("PUBLIC_MANIFEST_SHA256"),
        "model_plan_sha256": required_hash_env("MODEL_PLAN_SHA256"),
        "sif_sha256": required_hash_env("SIF_SHA256"),
    }
    if env_hashes["contract_sha256"] != contract_hashes["contract_sha256"]:
        raise GateError("worker contract hash differs from the bound environment")
    if env_hashes["prespec_sha256"] != contract_hashes["prespec_sha256"]:
        raise GateError("worker prespec hash differs from the bound environment")
    model_plan_path = args.model_plan.expanduser().resolve()
    plan = load_model_plan(model_plan_path)
    if sha256_file(model_plan_path) != env_hashes["model_plan_sha256"]:
        raise GateError("worker model-plan hash mismatch")
    model_key, shard_index, shard_count = model_and_shard(
        plan, args.rank, args.world_size
    )
    data_dir = args.data_dir.expanduser().resolve()
    mechanism_manifest = args.mechanism_manifest.expanduser().resolve()
    public_manifest = args.public_manifest.expanduser().resolve()
    if sha256_file(mechanism_manifest) != env_hashes["mechanism_manifest_sha256"]:
        raise GateError("worker mechanism-manifest hash mismatch")
    if sha256_file(public_manifest) != env_hashes["public_manifest_sha256"]:
        raise GateError("worker public-manifest hash mismatch")
    _base.validate_mechanism_manifest(
        data_dir=data_dir,
        manifest_path=mechanism_manifest,
        public_path=public_manifest,
    )
    public_rows = read_jsonl(public_manifest)
    _base.validate_dataset(
        public_rows=public_rows,
        hidden_rows=None,
        data_dir=data_dir,
        expected_frames=80,
        expected_user_prompt=EXPECTED_PROMPT,
    )
    selected = shard_rows(
        public_rows, shard_index=shard_index, shard_count=shard_count
    )
    if args.canary:
        selected = selected[:1]
    out_dir = args.run_dir.expanduser().resolve() / f"rank_{args.rank:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = out_dir / "outcomes.jsonl"
    completed_path = out_dir / "completed_requests.jsonl"
    manifest_path = out_dir / "worker_manifest.json"
    selected_request_records = [
        {
            "request_id": str(row["request_id"]),
            "frame_unit_id": str(row["frame_unit_id"]),
            "order_id": str(row["order_id"]),
        }
        for row in selected
    ]
    binding = {
        "kind": "patchaudit_transport_resume_binding",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "model_key": model_key,
        "rank": args.rank,
        "world_size": args.world_size,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "immutable_hashes": env_hashes,
    }
    binding_sha256 = sha256_json(binding)
    worker_manifest = {
        "kind": "patchaudit_transport_worker_manifest",
        "status": "initializing",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "model_key": model_key,
        "rank": args.rank,
        "world_size": args.world_size,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_request_count": len(selected),
        "selected_request_records": selected_request_records,
        "selected_request_records_sha256": sha256_json(
            selected_request_records
        ),
        "resume_binding": binding,
        "resume_binding_sha256": binding_sha256,
        "hidden_index_opened": False,
        "created_at": utc_now(),
    }
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if (
            previous.get("resume_binding") != binding
            or previous.get("resume_binding_sha256") != binding_sha256
            or previous.get("selected_request_records")
            != selected_request_records
            or previous.get("selected_request_records_sha256")
            != sha256_json(selected_request_records)
        ):
            raise GateError(
                "resume worker manifest differs from immutable binding or shard"
            )
        if previous.get("status") != "initializing" and (
            not outcomes_path.is_file() or not completed_path.is_file()
        ):
            raise GateError("resume worker files are missing after initialization")
        worker_manifest["created_at"] = previous.get(
            "created_at", worker_manifest["created_at"]
        )
    elif outcomes_path.exists() or completed_path.exists():
        raise GateError("resume rows exist without a worker manifest")
    else:
        atomic_write_json(manifest_path, worker_manifest)
    if not outcomes_path.exists():
        atomic_write_jsonl(outcomes_path, [])
    if not completed_path.exists():
        atomic_write_jsonl(completed_path, [])
    resume = {
        "completed_request_ids": [],
        "kept_rows": 0,
        "dropped_rows": 0,
        "dropped_receipts": 0,
    }
    if not args.canary:
        resume = sanitize_resume(
            outcomes_path=outcomes_path,
            completed_path=completed_path,
            expected_binding_sha256=binding_sha256,
            expected_request_ids={str(row["request_id"]) for row in selected},
            expected_rows_per_request=int(
                contract["completion"]["expected_outcomes_per_request"]
            ),
        )
    completed = set(resume["completed_request_ids"])
    worker_manifest["status"] = "running"
    worker_manifest["resume_sanitization"] = resume
    atomic_write_json(manifest_path, worker_manifest)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    runtime, model, processor, tokenizer, device, provenance = (
        _base._load_anchor_model(
            model_key=model_key,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
    )
    load_elapsed = time.perf_counter() - load_started
    sites, sites_by_layer, architecture = _decoder_sites(
        model, model_key=model_key, prespec=prespec
    )
    compute_started = time.perf_counter()
    answer_token_equivalence_gate: dict[str, Any] | None = None
    if args.canary:
        if not selected:
            raise GateError("canary shard has no public request")
        answer_token_equivalence_gate = _answer._answer_token_equivalence_audit(
            runtime=runtime,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            model_key=model_key,
            request=selected[0],
            data_dir=data_dir,
            candidate_completions={"A": "A", "B": "B"},
        )
        equivalence_tolerance = float(
            prespec["technical_gates"][
                "answer_token_equivalence_tolerance_bits"
            ]
        )
        if (
            answer_token_equivalence_gate.get("status") != "pass"
            or float(
                answer_token_equivalence_gate.get(
                    "maximum_absolute_difference_bits", math.inf
                )
            )
            > equivalence_tolerance
        ):
            raise GateError(
                f"answer-token equivalence failed for {model_key}: "
                f"{answer_token_equivalence_gate}"
            )
    request_timings: list[dict[str, Any]] = []
    errors = 0
    checkpoint_reason: str | None = None
    for request in selected:
        request_id = str(request["request_id"])
        if request_id in completed:
            continue
        if args.stop_file and args.stop_file.expanduser().resolve().exists():
            checkpoint_reason = "stop_file"
            break
        if (
            args.max_runtime_seconds is not None
            and time.perf_counter() - worker_started >= args.max_runtime_seconds
        ):
            checkpoint_reason = "max_runtime_seconds"
            break
        if (
            args.deadline_epoch_seconds is not None
            and time.time() >= args.deadline_epoch_seconds
        ):
            checkpoint_reason = "absolute_deadline"
            break
        try:
            rows, timing = evaluate_request(
                runtime=runtime,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                device=device,
                model_key=model_key,
                rank=args.rank,
                request=request,
                data_dir=data_dir,
                sites=sites,
                sites_by_layer=sites_by_layer,
                prespec=prespec,
                canary=args.canary,
            )
            append_jsonl_fsync(outcomes_path, rows)
            append_jsonl_fsync(
                completed_path,
                [
                    {
                        "kind": "patchaudit_transport_request_receipt",
                        "request_id": request_id,
                        "result_row_count": len(rows),
                        "result_batch_sha256": sha256_json(rows),
                        "output_schema_version": OUTPUT_SCHEMA_VERSION,
                        "resume_binding_sha256": binding_sha256,
                        "completed_at": utc_now(),
                    }
                ],
            )
            completed.add(request_id)
            request_timings.append(timing)
            if args.canary:
                atomic_write_json(
                    out_dir / "canary_audits.json",
                    {
                        "kind": "patchaudit_transport_canary_audits",
                        "request_id": request_id,
                        "answer_token_equivalence_gate": (
                            answer_token_equivalence_gate
                        ),
                        "self_checks": timing["self_checks"],
                        "outcomes": timing["detailed_canary_audits"],
                    },
                )
        except Exception as exc:
            errors += 1
            append_jsonl_fsync(
                out_dir / "errors.jsonl",
                [
                    {
                        "request_id": request_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "created_at": utc_now(),
                    }
                ],
            )
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elapsed = time.perf_counter() - worker_started
        atomic_write_json(
            out_dir / "progress.json",
            {
                "kind": "patchaudit_transport_worker_progress",
                "status": "running",
                "model_key": model_key,
                "rank": args.rank,
                "requests_completed": len(completed),
                "requests_assigned": len(selected),
                "outcome_rows": len(completed)
                * int(contract["completion"]["expected_outcomes_per_request"]),
                "elapsed_s": elapsed,
                "updated_at": utc_now(),
            },
        )
    elapsed = time.perf_counter() - worker_started
    all_complete = len(completed) == len(selected)
    final_outcomes = read_jsonl(outcomes_path)
    final_receipts = read_jsonl(completed_path)
    full_sequence_errors = [
        float(row["patched_donor_error_abs_bits"])
        for row in final_outcomes
        if row.get("operator") == "full_sequence_once"
    ]
    summary = {
        "kind": "patchaudit_transport_worker_summary",
        "status": (
            "pass"
            if all_complete and errors == 0
            else "partial_checkpoint"
            if errors == 0
            else "fail"
        ),
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "model_key": model_key,
        "rank": args.rank,
        "world_size": args.world_size,
        "requests_completed": len(completed),
        "requests_assigned": len(selected),
        "outcome_rows": len(final_outcomes),
        "completion_receipt_rows": len(final_receipts),
        "outcomes_sha256": sha256_file(outcomes_path),
        "completed_requests_sha256": sha256_file(completed_path),
        "expected_outcomes_per_request": int(
            contract["completion"]["expected_outcomes_per_request"]
        ),
        "errors": errors,
        "checkpoint_reason": checkpoint_reason,
        "load_elapsed_s": load_elapsed,
        "run_elapsed_s": elapsed,
        "compute_elapsed_s": time.perf_counter() - compute_started,
        "fresh_request_count": len(request_timings),
        "fresh_request_elapsed_s": [
            float(row["elapsed_s"]) for row in request_timings
        ],
        "fresh_forward_count": sum(
            int(row["total_forward_count"]) for row in request_timings
        ),
        "fresh_operator_elapsed_s": dict(
            sum(
                (
                    Counter(
                        {
                            key: float(value)
                            for key, value in row["operator_elapsed_s"].items()
                        }
                    )
                    for row in request_timings
                ),
                Counter(),
            )
        ),
        "full_sequence_row_count": len(full_sequence_errors),
        "full_sequence_max_abs_error_bits": (
            max(full_sequence_errors) if full_sequence_errors else None
        ),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())
        if torch.cuda.is_available()
        else 0,
        "architecture": architecture,
        "model_provenance": provenance,
        "immutable_hashes": env_hashes,
        "resume_binding_sha256": binding_sha256,
        "hidden_index_opened": False,
        "created_at": utc_now(),
    }
    worker_manifest["status"] = summary["status"]
    worker_manifest["completed_at"] = utc_now()
    atomic_write_json(manifest_path, worker_manifest)
    atomic_write_json(out_dir / "worker_summary.json", summary)
    atomic_write_json(out_dir / "progress.json", summary)
    return 0 if summary["status"] in {"pass", "partial_checkpoint"} else 2


def _round_up(value: float, quantum: int) -> int:
    return int(math.ceil(value / quantum) * quantum)


def aggregate_command(args: argparse.Namespace) -> int:
    repo_dir = args.repo_dir.expanduser().resolve()
    contract, prespec, hashes = verify_contract(
        repo_dir=repo_dir, contract_path=args.contract.expanduser().resolve()
    )
    run_dir = args.run_dir.expanduser().resolve()
    public_path = args.public_manifest.expanduser().resolve()
    expected_public_sha = str(prespec["data"]["public_requests_sha256"])
    if sha256_file(public_path) != expected_public_sha:
        raise GateError("aggregate public manifest differs from the frozen SHA-256")
    public_rows = read_jsonl(public_path)
    public_by_id = {str(row["request_id"]): row for row in public_rows}
    if len(public_rows) != 160 or len(public_by_id) != 160:
        raise GateError("aggregate requires exactly 160 unique public requests")

    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    manifests: dict[int, dict[str, Any]] = {}
    rank_outcome_files: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for rank in range(4):
        rank_dir = run_dir / f"rank_{rank:03d}"
        summary_path = rank_dir / "worker_summary.json"
        if not summary_path.is_file():
            failures.append(f"missing_worker_summary:{rank}")
            continue
        summary = read_json(summary_path)
        summaries.append(summary)
        if summary.get("hidden_index_opened") is not False:
            failures.append(f"worker_hidden_index:{rank}")
        try:
            rank_rows, manifest = validate_rank_artifacts(
                run_dir=run_dir,
                rank=rank,
                summary=summary,
                prespec=prespec,
                public_by_id=public_by_id,
                canary=args.canary,
            )
        except (GateError, KeyError, TypeError, ValueError) as exc:
            failures.append(
                f"worker_artifact_contract:{rank}:{type(exc).__name__}:{exc}"
            )
            continue
        rows.extend(rank_rows)
        manifests[rank] = manifest
        outcomes_path = rank_dir / "outcomes.jsonl"
        completed_path = rank_dir / "completed_requests.jsonl"
        rank_outcome_files[str(rank)] = {
            "outcomes_path": str(outcomes_path),
            "outcomes_sha256": sha256_file(outcomes_path),
            "outcome_rows": len(rank_rows),
            "completed_requests_path": str(completed_path),
            "completed_requests_sha256": sha256_file(completed_path),
            "completion_receipt_rows": len(read_jsonl(completed_path)),
        }

    if len(manifests) == 4:
        rank_request_sets = {
            rank: {
                str(record["request_id"])
                for record in manifest["selected_request_records"]
            }
            for rank, manifest in manifests.items()
        }
        for left, right in ((0, 1), (2, 3)):
            if rank_request_sets[left] & rank_request_sets[right]:
                failures.append(f"shard_request_overlap:{left}:{right}")
        qwen_requests = rank_request_sets[0] | rank_request_sets[1]
        gemma_requests = rank_request_sets[2] | rank_request_sets[3]
        if qwen_requests != gemma_requests:
            failures.append("cross_model_request_inventory_drift")
        if not args.canary and qwen_requests != set(public_by_id):
            failures.append("full_public_request_inventory_drift")

    identities = [outcome_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        failures.append("duplicate_outcome_identity")
    telemetry = _base.telemetry_summary(args.telemetry_csv.expanduser().resolve())
    if telemetry.get("status") != "pass":
        failures.append("telemetry")

    immutable_hashes = [
        summary.get("immutable_hashes") for summary in summaries
    ]
    if immutable_hashes and any(
        value != immutable_hashes[0] for value in immutable_hashes
    ):
        failures.append("worker_immutable_hash_drift")
    aggregate_immutable_hashes = (
        immutable_hashes[0]
        if immutable_hashes
        and isinstance(immutable_hashes[0], dict)
        and len(immutable_hashes) == 4
        else None
    )
    if aggregate_immutable_hashes is None:
        failures.append("missing_worker_immutable_hashes")
    elif (
        aggregate_immutable_hashes.get("contract_sha256")
        != hashes["contract_sha256"]
        or aggregate_immutable_hashes.get("prespec_sha256")
        != hashes["prespec_sha256"]
        or aggregate_immutable_hashes.get("public_manifest_sha256")
        != expected_public_sha
    ):
        failures.append("aggregate_contract_binding")

    job_manifest_path = run_dir / "job_manifest.json"
    job_manifest: dict[str, Any] | None = None
    canary_binding: dict[str, Any] = {
        "status": "not_applicable" if args.canary else "fail"
    }
    if not job_manifest_path.is_file():
        failures.append("missing_job_manifest")
    else:
        job_manifest = read_json(job_manifest_path)
        if (
            job_manifest.get("kind")
            != "patchaudit_transport_tamia_job_manifest"
            or job_manifest.get("mode")
            != ("canary" if args.canary else "full")
            or job_manifest.get("status") != "running"
            or job_manifest.get("worker_hidden_bind") is not False
            or job_manifest.get("hidden_index_staged_before_technical_gate")
            is not False
        ):
            failures.append("job_manifest_label_firewall_contract")
        if aggregate_immutable_hashes is not None:
            for key, value in aggregate_immutable_hashes.items():
                manifest_key = key
                if job_manifest.get(manifest_key) != value:
                    failures.append(f"job_manifest_immutable_binding:{key}")
            if not args.canary:
                try:
                    canary_binding = validate_bound_canary(
                        job_manifest=job_manifest,
                        immutable_hashes=aggregate_immutable_hashes,
                    )
                except (GateError, KeyError, TypeError, ValueError) as exc:
                    failures.append(
                        f"production_canary_binding:{type(exc).__name__}:{exc}"
                    )

    full_sequence_errors = [
        abs(
            float(row["patched_margin_a_minus_b_bits"])
            - float(row["donor_margin_a_minus_b_bits"])
        )
        for row in rows
        if row.get("operator") == "full_sequence_once"
    ]
    expected_full_sequence_rows = sum(
        int(summary.get("requests_completed") or 0)
        * len(CORRUPTIONS)
        * 2
        * len(
            prespec["depth_plan"][
                str(summary.get("model_key"))
            ]["ordered_post_block_indexes"]
        )
        for summary in summaries
        if str(summary.get("model_key")) in prespec["depth_plan"]
    )
    if len(full_sequence_errors) != expected_full_sequence_rows:
        failures.append(
            "full_sequence_row_count:"
            f"{len(full_sequence_errors)}:{expected_full_sequence_rows}"
        )
    full_sequence_max = (
        max(full_sequence_errors) if full_sequence_errors else None
    )
    closure_tolerance = float(
        prespec["technical_gates"]["full_sequence_closure_max_abs_error_bits"]
    )
    if (
        full_sequence_max is not None
        and full_sequence_max > closure_tolerance
    ):
        failures.append("full_sequence_closure")

    all_workers_complete = (
        len(summaries) == 4
        and all(summary.get("status") == "pass" for summary in summaries)
    )
    if args.canary:
        expected_rows = 4 * int(
            contract["completion"]["expected_outcomes_per_request"]
        )
        if len(rows) != expected_rows:
            failures.append(f"canary_row_count:{len(rows)}:{expected_rows}")
        peak_limit = int(contract["canary"]["maximum_peak_cuda_allocated_bytes"])
        for summary in summaries:
            if int(summary.get("peak_cuda_memory_bytes") or 0) > peak_limit:
                failures.append(
                    f"canary_vram:{summary.get('rank')}:"
                    f"{summary.get('peak_cuda_memory_bytes')}"
                )
            canary_path = (
                run_dir / f"rank_{int(summary['rank']):03d}" / "canary_audits.json"
            )
            if not canary_path.is_file():
                failures.append(f"missing_canary_audits:{summary.get('rank')}")
                continue
            canary = read_json(canary_path)
            if (
                canary.get("kind") != "patchaudit_transport_canary_audits"
                or not isinstance(canary.get("answer_token_equivalence_gate"), dict)
            ):
                failures.append(
                    f"canary_answer_token_equivalence:{summary.get('rank')}"
                )
                continue
            equivalence = canary["answer_token_equivalence_gate"]
            equivalence_tolerance = float(
                prespec["technical_gates"][
                    "answer_token_equivalence_tolerance_bits"
                ]
            )
            if (
                equivalence.get("status") != "pass"
                or float(
                    equivalence.get(
                        "maximum_absolute_difference_bits", math.inf
                    )
                )
                > equivalence_tolerance
            ):
                failures.append(
                    f"canary_answer_token_equivalence:{summary.get('rank')}"
                )
            self_rows = canary.get("self_checks", [])
            if (
                {str(row.get("operator")) for row in self_rows}
                != set(OPERATORS)
                or len(self_rows) != len(OPERATORS)
                or any(row.get("status") != "pass" for row in self_rows)
            ):
                failures.append(f"canary_self_patch:{summary.get('rank')}")
        status = "pass" if not failures and all_workers_complete else "fail"
    else:
        expected_rows = int(contract["completion"]["expected_full_outcome_rows"])
        status = (
            "pass"
            if not failures and all_workers_complete and len(rows) == expected_rows
            else "partial_checkpoint"
            if not failures and len(summaries) == 4
            else "fail"
        )
        if all_workers_complete and len(rows) != expected_rows:
            failures.append(f"full_row_count:{len(rows)}:{expected_rows}")
            status = "fail"
    aggregate = {
        "kind": "patchaudit_transport_aggregate",
        "status": status,
        "mode": "canary" if args.canary else "full",
        "failures": failures,
        "contract": hashes,
        "immutable_hashes": aggregate_immutable_hashes,
        "worker_count": len(summaries),
        "workers": summaries,
        "outcome_row_count": len(rows),
        "unique_outcome_identity_count": len(set(identities)),
        "exact_cartesian_inventory_status": (
            "pass"
            if not any(
                "inventory" in failure
                or "worker_artifact_contract" in failure
                or "duplicate_outcome" in failure
                for failure in failures
            )
            else "fail"
        ),
        "rank_outcome_files": rank_outcome_files,
        "public_manifest": {
            "path": str(public_path),
            "sha256": expected_public_sha,
            "rows": len(public_rows),
        },
        "full_sequence_max_abs_error_bits": full_sequence_max,
        "full_sequence_row_count": len(full_sequence_errors),
        "expected_full_sequence_row_count": expected_full_sequence_rows,
        "telemetry": telemetry,
        "job_manifest_contract": {
            "path": str(job_manifest_path),
            "status": "pass"
            if job_manifest is not None
            and "missing_job_manifest" not in failures
            and not any(
                failure.startswith("job_manifest_") for failure in failures
            )
            else "fail",
        },
        "production_canary_binding": canary_binding,
        "resume_needed": status == "partial_checkpoint",
        "created_at": utc_now(),
    }
    atomic_write_json(run_dir / "aggregate_summary.json", aggregate)
    if args.canary and status == "pass":
        request_seconds = [
            float(value)
            for summary in summaries
            for value in summary.get("fresh_request_elapsed_s", [])
        ]
        if not request_seconds:
            raise GateError("canary has no measured request runtimes")
        load_seconds = max(float(row.get("load_elapsed_s") or 0.0) for row in summaries)
        projected_compute = max(request_seconds) * 80
        point_estimate = load_seconds + projected_compute + 600.0
        headroom_fraction = 0.35
        recommended = _round_up(point_estimate * (1 + headroom_fraction), 300)
        chunk_seconds = 2 * 3600 + 55 * 60
        profile = {
            "kind": "patchaudit_transport_runtime_profile",
            "status": "pass",
            "canary_aggregate_sha256": sha256_file(
                run_dir / "aggregate_summary.json"
            ),
            "contract": hashes,
            "immutable_hashes": aggregate_immutable_hashes,
            "request_seconds_by_rank": request_seconds,
            "load_seconds_max": load_seconds,
            "projected_full_compute_seconds": projected_compute,
            "point_estimate_total_seconds": point_estimate,
            "headroom_fraction": headroom_fraction,
            "recommended_total_seconds": recommended,
            "recommended_total_walltime": (
                f"{recommended // 3600:02d}:"
                f"{(recommended % 3600) // 60:02d}:00"
            ),
            "chunk_walltime": "02:55:00",
            "planned_chunk_count": max(1, math.ceil(recommended / chunk_seconds)),
            "created_at": utc_now(),
        }
        atomic_write_json(run_dir / "runtime_profile.json", profile)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0 if status in {"pass", "partial_checkpoint"} else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-dir", type=Path, required=True)
    common.add_argument("--contract", type=Path, required=True)
    preflight = subparsers.add_parser("preflight", parents=[common])
    preflight.add_argument("--data-dir", type=Path, required=True)
    preflight.add_argument("--mechanism-manifest", type=Path, required=True)
    preflight.add_argument("--public-manifest", type=Path, required=True)
    preflight.add_argument("--out", type=Path)
    worker = subparsers.add_parser("run-worker", parents=[common])
    worker.add_argument("--data-dir", type=Path, required=True)
    worker.add_argument("--mechanism-manifest", type=Path, required=True)
    worker.add_argument("--public-manifest", type=Path, required=True)
    worker.add_argument("--model-plan", type=Path, required=True)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--world-size", type=int, default=4)
    worker.add_argument("--dtype", default="bfloat16")
    worker.add_argument("--seed", type=int, default=DEFAULT_SEED)
    worker.add_argument("--canary", action="store_true")
    worker.add_argument("--local-files-only", action="store_true", default=True)
    worker.add_argument("--stop-file", type=Path)
    worker.add_argument("--max-runtime-seconds", type=float)
    worker.add_argument("--deadline-epoch-seconds", type=float)
    aggregate = subparsers.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--run-dir", type=Path, required=True)
    aggregate.add_argument("--public-manifest", type=Path, required=True)
    aggregate.add_argument("--telemetry-csv", type=Path, required=True)
    aggregate.add_argument("--canary", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "preflight":
            return preflight_command(args)
        if args.command == "run-worker":
            return worker_command(args)
        if args.command == "aggregate":
            return aggregate_command(args)
        raise GateError(f"unknown command: {args.command}")
    except (GateError, TransportGateError, _base.GateError) as exc:
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
