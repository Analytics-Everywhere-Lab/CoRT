#!/usr/bin/env python3
"""Behavioral companion runtime and aggregation core.

This module is intentionally fail-closed:
- Worker-side code only reads method-facing request packs + ID-only selection.
- Evaluator-side preflight/aggregate may read hidden indices for gating/scoring.
- Teacher-forced scoring fails if template/token boundaries are ambiguous.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Sequence

import numpy as np

from research.lifecycle_memory import phase2_job2b_paired_analysis as paired_stats
from research.lifecycle_memory import run_phase2_mech_job1_instrument_gates as gates
from research.lifecycle_memory import run_phase2_mech_job2_battery_capture as capture
from research.lifecycle_memory.run_epic_visor_dynamic_llamacpp_metal import _parse_visor_answer


REPO_ROOT = Path(__file__).resolve().parents[2]

ARM_PCS = "plain_contact_sheet"
ARM_MPP = "marked_plain_prompt"
ARM_VP = "visual_prompt"
ARM_ORDER = (ARM_PCS, ARM_MPP, ARM_VP)

DEFAULT_W0_DIR = (
    REPO_ROOT / "runs" / "research" / "lifecycle_memory" / "phase_2" / "w0_geometry_sidecar_2026_07_07"
)
DEFAULT_PRESPEC_PATH = (
    REPO_ROOT
    / "docs"
    / "research"
    / "lifecycle_memory"
    / "phase 2 research"
    / "jobs"
    / "mech_job2c_behavioral_companion"
    / "behavior_prespec.json"
)
DEFAULT_MODEL_PLAN_PATH = (
    REPO_ROOT
    / "docs"
    / "research"
    / "lifecycle_memory"
    / "phase 2 research"
    / "jobs"
    / "mech_job2c_behavioral_companion"
    / "model_plan_default.json"
)

DEFAULT_PRIMARY_QUERY_TYPE = "contact_state_current"

FROZEN_CANDIDATE_COMPLETIONS = {
    "false": '{"answer":false,"abstain":false,"abstention_reason":null,"confidence":1.0}',
    "true": '{"answer":true,"abstain":false,"abstention_reason":null,"confidence":1.0}',
}

CANDIDATE_CONSTRUCTION_NATIVE = "native_assistant_chat_template"
CANDIDATE_CONSTRUCTION_EXACT_APPEND = "base_generation_prompt_exact_token_append"
FROZEN_CANDIDATE_CONSTRUCTION_ORDER = (
    CANDIDATE_CONSTRUCTION_NATIVE,
    CANDIDATE_CONSTRUCTION_EXACT_APPEND,
)
ALLOWED_CANDIDATE_CONSTRUCTIONS = set(FROZEN_CANDIDATE_CONSTRUCTION_ORDER)
FROZEN_MODEL_PLAN = (
    ("qwen35_27b_hf_bf16", "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"),
    ("gemma4_31b_hf_bf16", "google/gemma-4-31B-it", "3548789868c5356dbf307c98e6f609007b82b3eb"),
    ("gemma4_12b_hf_bf16", "google/gemma-4-12B-it", "5926caa4ec0cac5cbfadaf4077420520de1d5205"),
    ("gemma4_26b_a4b_hf_bf16", "google/gemma-4-26B-A4B-it", "20da991ab4afab98e8f910c4a2e8f4fbefc404ad"),
    ("qwen35_4b_hf_bf16", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
)
FROZEN_MODEL_KEYS = tuple(row[0] for row in FROZEN_MODEL_PLAN)

FORBIDDEN_WORKER_FIELDS = {
    "expected_answer",
    "target_label",
    "contact_by_public_frame",
    "visible_by_public_frame",
    "correct",
    "correctness",
    "hidden_label",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _append_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    new_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(path, existing + new_text)


def sanitize_jsonl(path: Path) -> dict[str, Any]:
    """Drop malformed/truncated JSONL lines and keep valid earlier lines."""
    if not path.exists():
        return {"path": str(path), "status": "missing", "dropped_lines": 0}
    good: list[str] = []
    dropped = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            good.append(line.rstrip("\n") + "\n")
    if dropped:
        _atomic_write_text(path, "".join(good))
    return {"path": str(path), "status": "pass", "dropped_lines": int(dropped)}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _resolve_snapshot_revision(summary: dict[str, Any]) -> str | None:
    priority_keys = (
        "resolved_snapshot_revision",
        "commit_hash",
        "requested_revision",
        "revision",
    )
    queue: list[Any] = [summary.get("model_snapshot"), summary]
    seen: set[int] = set()
    while queue:
        value = queue.pop(0)
        if isinstance(value, dict):
            if id(value) in seen:
                continue
            seen.add(id(value))
            for key in priority_keys:
                nested = value.get(key)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
            queue.extend(value.values())
        elif isinstance(value, (list, tuple)):
            queue.extend(value)
    return None


def _candidate_items_from_prespec(prespec: dict[str, Any]) -> list[tuple[str, str]]:
    raw = prespec.get("candidate_completions")
    if not isinstance(raw, dict):
        raise ValueError("prespec candidate_completions must be an object")
    rows: list[tuple[str, str]] = []
    for key, value in raw.items():
        k = str(key)
        if not isinstance(value, str):
            raise ValueError(f"candidate_completions[{k}] must be a string")
        rows.append((k, value))
    if dict(rows) != FROZEN_CANDIDATE_COMPLETIONS:
        raise ValueError("prespec candidate_completions drift from frozen byte strings")
    if [key for key, _ in rows] != list(FROZEN_CANDIDATE_COMPLETIONS.keys()):
        raise ValueError("prespec candidate order drift")
    return rows


def load_prespec(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("kind") != "phase2_job2c_behavioral_companion_prespec":
        raise ValueError(f"invalid prespec kind at {path}")
    arms = tuple(str(value) for value in payload.get("arms", []))
    if arms != ARM_ORDER:
        raise ValueError(f"arm order drift: expected {ARM_ORDER}, got {arms}")
    _candidate_items_from_prespec(payload)
    construction = payload.get("candidate_construction")
    if not isinstance(construction, dict):
        raise ValueError("prespec candidate_construction must be an object")
    allowed_modes = tuple(str(value) for value in construction.get("allowed_modes", []))
    if allowed_modes != FROZEN_CANDIDATE_CONSTRUCTION_ORDER:
        raise ValueError(
            "candidate construction drift: "
            f"expected {FROZEN_CANDIDATE_CONSTRUCTION_ORDER}, got {allowed_modes}"
        )
    return payload


def load_model_plan(path: Path) -> list[dict[str, str]]:
    rows = _read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("model plan must be a non-empty JSON list")
    normalized: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"invalid model plan row: {row!r}")
        model_key = str(row.get("model_key") or "").strip()
        model_id = str(row.get("model_id") or "").strip()
        revision = str(row.get("revision") or "").strip()
        if not model_key or not model_id or not revision:
            raise ValueError(f"model plan row missing key/id/revision: {row!r}")
        normalized.append({"model_key": model_key, "model_id": model_id, "revision": revision})
    return normalized


def _validate_frozen_model_plan(plan: Sequence[dict[str, str]]) -> dict[str, Any]:
    observed = tuple((row["model_key"], row["model_id"], row["revision"]) for row in plan)
    expected = tuple(FROZEN_MODEL_PLAN)
    return {
        "status": "pass" if observed == expected else "fail",
        "expected_rows": [
            {"model_key": key, "model_id": model_id, "revision": revision}
            for key, model_id, revision in expected
        ],
        "observed_rows": [
            {"model_key": key, "model_id": model_id, "revision": revision}
            for key, model_id, revision in observed
        ],
    }


def _pack_path_for_arm(w0_dir: Path, arm: str) -> Path:
    return w0_dir / "derived_pack" / f"order_stripped_{arm}_request_pack.json"


def _hidden_path_for_arm(w0_dir: Path, arm: str) -> Path:
    return w0_dir / "derived_pack" / f"order_stripped_{arm}_hidden_index.json"


def _load_pack_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    rows = [row for row in payload.get("requests", []) if isinstance(row, dict)]
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_id = str(row.get("request_id") or "")
        if request_id:
            out[request_id] = row
    return out


def _load_hidden_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    return [row for row in payload.get("answers", []) if isinstance(row, dict)]


def _selection_entry(source_id: str, request_by_arm: dict[str, str]) -> dict[str, Any]:
    return {
        "source_order_stripped_request_id": source_id,
        "request_id_by_arm": {arm: request_by_arm[arm] for arm in ARM_ORDER},
    }


def _source_row_for_alignment(hidden_row: dict[str, Any], request_row: dict[str, Any]) -> dict[str, Any]:
    query = request_row.get("query") if isinstance(request_row.get("query"), dict) else {}
    frame_count = len(
        [
            frame
            for frame in request_row.get("frames", [])
            if isinstance(frame, dict) and frame.get("path")
        ]
    )
    return {
        "source_order_stripped_request_id": str(hidden_row.get("source_order_stripped_request_id") or ""),
        "request_id": str(hidden_row.get("request_id") or ""),
        "expected_answer": bool(hidden_row.get("expected_answer")),
        "participant": str(hidden_row.get("source_tier1_participant_id") or ""),
        "video_id": str(hidden_row.get("source_video_id") or ""),
        "query_type": str(hidden_row.get("query_type") or query.get("query_type") or ""),
        "public_question": str(query.get("text") or ""),
        "target_label": str(query.get("target_label") or hidden_row.get("target_label") or ""),
        "frame_count": int(frame_count),
    }


def _request_input_identity(request: dict[str, Any], *, pack_path: Path) -> dict[str, Any]:
    prompt = str(request.get("user_prompt") or "")
    raw_paths = [
        str(frame.get("path"))
        for frame in request.get("frames", [])
        if isinstance(frame, dict) and frame.get("path")
    ]
    resolved_paths = [
        path if path.is_absolute() else (pack_path.parent / path).resolve()
        for path in (Path(raw) for raw in raw_paths)
    ]
    return {
        "prompt_sha256": _sha256_text(prompt),
        "prompt_size_bytes": len(prompt.encode("utf-8")),
        "ordered_method_frame_paths": raw_paths,
        "ordered_method_frame_sha256": [_sha256_file(path) for path in resolved_paths],
    }


def _contrast_identity_gate_from_source_maps(
    rows_by_arm: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    prompt_mismatches: list[dict[str, Any]] = []
    frame_mismatches: list[dict[str, Any]] = []
    shared_ids = set(rows_by_arm.get(ARM_PCS, {}))
    shared_ids &= set(rows_by_arm.get(ARM_MPP, {}))
    shared_ids &= set(rows_by_arm.get(ARM_VP, {}))
    for source_id in sorted(shared_ids):
        pcs = rows_by_arm[ARM_PCS][source_id]
        mpp = rows_by_arm[ARM_MPP][source_id]
        vp = rows_by_arm[ARM_VP][source_id]
        prompt_fields = ("prompt_sha256", "prompt_size_bytes")
        if any(pcs.get(field) != mpp.get(field) for field in prompt_fields):
            prompt_mismatches.append(
                {
                    "source_order_stripped_request_id": source_id,
                    "pcs_prompt_sha256": pcs.get("prompt_sha256"),
                    "mpp_prompt_sha256": mpp.get("prompt_sha256"),
                    "pcs_prompt_size_bytes": pcs.get("prompt_size_bytes"),
                    "mpp_prompt_size_bytes": mpp.get("prompt_size_bytes"),
                }
            )
        frame_fields = ("ordered_method_frame_paths", "ordered_method_frame_sha256")
        if any(mpp.get(field) != vp.get(field) for field in frame_fields):
            frame_mismatches.append(
                {
                    "source_order_stripped_request_id": source_id,
                    "mpp_method_frame_paths": mpp.get("ordered_method_frame_paths"),
                    "vp_method_frame_paths": vp.get("ordered_method_frame_paths"),
                    "mpp_method_frame_sha256": mpp.get("ordered_method_frame_sha256"),
                    "vp_method_frame_sha256": vp.get("ordered_method_frame_sha256"),
                }
            )
    return {
        "status": "pass" if not prompt_mismatches and not frame_mismatches else "fail",
        "shared_source_id_count": len(shared_ids),
        "pcs_mpp_prompt_mismatch_count": len(prompt_mismatches),
        "pcs_mpp_prompt_mismatch_examples": prompt_mismatches[:20],
        "mpp_vp_method_frame_mismatch_count": len(frame_mismatches),
        "mpp_vp_method_frame_mismatch_examples": frame_mismatches[:20],
    }


def _alignment_mismatch(
    *,
    source_id: str,
    field: str,
    arm_a: str,
    arm_b: str,
    value_a: Any,
    value_b: Any,
) -> dict[str, Any]:
    mismatch = {
        "source_order_stripped_request_id": source_id,
        "field": field,
        "arm_a": arm_a,
        "arm_b": arm_b,
    }
    if field in FORBIDDEN_WORKER_FIELDS:
        mismatch["values_redacted"] = True
    else:
        mismatch["value_a"] = value_a
        mismatch["value_b"] = value_b
    return mismatch


def validate_three_arm_alignment(
    rows_by_arm: dict[str, Sequence[dict[str, Any]]],
    *,
    compare_fields: Sequence[str] = (
        "expected_answer",
        "participant",
        "video_id",
        "query_type",
        "public_question",
        "target_label",
        "frame_count",
    ),
) -> dict[str, Any]:
    duplicate_ids: dict[str, list[str]] = {}
    source_sets: dict[str, set[str]] = {}
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARM_ORDER:
        rows = list(rows_by_arm.get(arm, []))
        seen: set[str] = set()
        dups: set[str] = set()
        arm_map: dict[str, dict[str, Any]] = {}
        for row in rows:
            source_id = str(row.get("source_order_stripped_request_id") or "")
            if not source_id:
                continue
            if source_id in seen:
                dups.add(source_id)
            seen.add(source_id)
            arm_map[source_id] = dict(row)
        source_sets[arm] = set(arm_map)
        maps[arm] = arm_map
        if dups:
            duplicate_ids[arm] = sorted(dups)

    missing_by_arm: dict[str, list[str]] = {}
    common_ids: set[str] | None = None
    union_ids: set[str] = set()
    for arm in ARM_ORDER:
        union_ids |= source_sets[arm]
        common_ids = source_sets[arm] if common_ids is None else common_ids.intersection(source_sets[arm])
    common_ids = common_ids or set()
    for arm in ARM_ORDER:
        missing = sorted(union_ids - source_sets[arm])
        if missing:
            missing_by_arm[arm] = missing

    mismatches: list[dict[str, Any]] = []
    for source_id in sorted(common_ids):
        base = maps[ARM_ORDER[0]][source_id]
        for arm in ARM_ORDER[1:]:
            other = maps[arm][source_id]
            for field in compare_fields:
                if base.get(field) != other.get(field):
                    mismatches.append(
                        _alignment_mismatch(
                            source_id=source_id,
                            field=field,
                            arm_a=ARM_ORDER[0],
                            arm_b=arm,
                            value_a=base.get(field),
                            value_b=other.get(field),
                        )
                    )
    status = "pass" if not duplicate_ids and not missing_by_arm and not mismatches else "fail"
    return {
        "status": status,
        "duplicate_source_ids_by_arm": duplicate_ids,
        "missing_source_ids_by_arm": missing_by_arm,
        "mismatches": mismatches,
        "shared_source_id_count": int(len(common_ids)),
        "union_source_id_count": int(len(union_ids)),
    }


def _gate(name: str, status: str, *, blocking: bool, detail: Any | None = None) -> dict[str, Any]:
    payload = {
        "gate": name,
        "status": status,
        "blocking": bool(blocking),
    }
    if detail is not None:
        payload["detail"] = detail
    return payload


def _prespec_hash(path: Path) -> str:
    return _sha256_file(path) or ""


def _model_plan_hash(path: Path) -> str:
    return _sha256_file(path) or ""


def preflight(
    *,
    run_dir: Path,
    w0_dir: Path = DEFAULT_W0_DIR,
    prespec_path: Path = DEFAULT_PRESPEC_PATH,
    model_plan_path: Path = DEFAULT_MODEL_PLAN_PATH,
    mode: str = "full",
    query_type: str = DEFAULT_PRIMARY_QUERY_TYPE,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    w0_dir = w0_dir.expanduser().resolve()
    prespec_path = prespec_path.expanduser().resolve()
    model_plan_path = model_plan_path.expanduser().resolve()
    mode = str(mode).strip().lower()
    if mode not in {"full", "canary"}:
        raise ValueError("mode must be 'full' or 'canary'")

    prespec = load_prespec(prespec_path)
    model_plan = load_model_plan(model_plan_path)
    candidate_items = _candidate_items_from_prespec(prespec)

    expected_query = str(
        prespec.get("query_scope", {}).get("primary_query_type") or DEFAULT_PRIMARY_QUERY_TYPE
    )
    selected_query = str(query_type or expected_query)
    if not selected_query:
        selected_query = expected_query

    arm_maps: dict[str, dict[str, dict[str, Any]]] = {}
    arm_hidden_rows: dict[str, list[dict[str, Any]]] = {}
    arm_filtered_rows: dict[str, list[dict[str, Any]]] = {}
    arm_totals: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        pack_path = _pack_path_for_arm(w0_dir, arm)
        hidden_path = _hidden_path_for_arm(w0_dir, arm)
        pack_rows = _load_pack_rows(pack_path)
        hidden_rows = _load_hidden_rows(hidden_path)
        arm_hidden_rows[arm] = hidden_rows

        source_map: dict[str, dict[str, Any]] = {}
        filtered_rows: list[dict[str, Any]] = []
        for hidden in hidden_rows:
            request_id = str(hidden.get("request_id") or "")
            source_id = str(hidden.get("source_order_stripped_request_id") or "")
            if not request_id or not source_id:
                continue
            request = pack_rows.get(request_id)
            if request is None:
                continue
            row = _source_row_for_alignment(hidden, request)
            row.update(_request_input_identity(request, pack_path=pack_path))
            source_map[source_id] = row
            if row["query_type"] == selected_query:
                filtered_rows.append(row)
        arm_maps[arm] = source_map
        arm_filtered_rows[arm] = filtered_rows
        arm_totals.append(
            {
                "arm": arm,
                "pack_path": str(pack_path),
                "hidden_path": str(hidden_path),
                "request_count": len(pack_rows),
                "hidden_count": len(hidden_rows),
                "filtered_query_count": len(filtered_rows),
            }
        )

    alignment = validate_three_arm_alignment(arm_filtered_rows)
    contrast_identity = _contrast_identity_gate_from_source_maps(arm_maps)

    # Use the plain-contact-sheet ordering as canonical request order.
    source_order = [
        str(row.get("source_order_stripped_request_id") or "")
        for row in arm_filtered_rows[ARM_PCS]
        if row.get("source_order_stripped_request_id")
    ]
    source_order = [source_id for source_id in source_order if source_id in arm_maps[ARM_MPP] and source_id in arm_maps[ARM_VP]]
    start = max(0, int(offset))
    rows_after_offset = source_order[start:]
    if limit is not None and int(limit) > 0:
        rows_after_offset = rows_after_offset[: int(limit)]
    selected_source_ids = rows_after_offset

    selection_rows = [
        _selection_entry(
            source_id,
            {
                arm: str(arm_maps[arm][source_id]["request_id"])
                for arm in ARM_ORDER
            },
        )
        for source_id in selected_source_ids
    ]

    participants = sorted(
        {
            str(arm_maps[ARM_PCS][source_id]["participant"])
            for source_id in selected_source_ids
            if source_id in arm_maps[ARM_PCS]
        }
    )

    expected_rows = int(prespec.get("query_scope", {}).get("expected_row_count_per_arm_model") or 0)
    expected_participants = int(
        prespec.get("query_scope", {}).get("expected_participant_count") or 0
    )

    gates_out: list[dict[str, Any]] = []
    model_plan_gate = _validate_frozen_model_plan(model_plan)
    gates_out.append(_gate("frozen_model_plan", model_plan_gate["status"], blocking=True, detail=model_plan_gate))
    gates_out.append(
        _gate(
            "frozen_candidates",
            "pass" if dict(candidate_items) == FROZEN_CANDIDATE_COMPLETIONS else "fail",
            blocking=True,
            detail={"candidate_keys": [key for key, _ in candidate_items]},
        )
    )
    gates_out.append(_gate("three_arm_alignment", alignment["status"], blocking=True, detail=alignment))
    gates_out.append(
        _gate(
            "preflight_contrast_input_identity",
            contrast_identity["status"],
            blocking=True,
            detail=contrast_identity,
        )
    )
    if mode == "full":
        gates_out.append(
            _gate(
                "full_expected_row_count",
                "pass" if len(selection_rows) == expected_rows else "fail",
                blocking=True,
                detail={"expected": expected_rows, "observed": len(selection_rows)},
            )
        )
        gates_out.append(
            _gate(
                "full_expected_participant_count",
                "pass" if len(participants) == expected_participants else "fail",
                blocking=True,
                detail={"expected": expected_participants, "observed": len(participants)},
            )
        )
    else:
        gates_out.append(
            _gate(
                "canary_nonempty_selection",
                "pass" if selection_rows else "fail",
                blocking=True,
                detail={"observed": len(selection_rows)},
            )
        )

    blocking_failures = [gate for gate in gates_out if gate["blocking"] and gate["status"] != "pass"]

    selection_manifest = {
        "kind": "phase2_job2c_behavioral_selection_manifest_id_only",
        "mode": mode,
        "query_type": selected_query,
        "selection_count": len(selection_rows),
        "selection": selection_rows,
        "prespec_sha256": _prespec_hash(prespec_path),
        "model_plan_sha256": _model_plan_hash(model_plan_path),
        "arms": list(ARM_ORDER),
        "generated_at": _utc_now_iso(),
        "claim_boundary": (
            "ID-only evaluator manifest for worker routing. It intentionally omits "
            "expected answers, target labels, contact states, geometry, and any other "
            "answer-bearing hidden fields."
        ),
    }
    input_audit = {
        "kind": "phase2_job2c_behavioral_preflight_audit",
        "status": "pass" if not blocking_failures else "fail",
        "mode": mode,
        "w0_dir": str(w0_dir),
        "prespec_path": str(prespec_path),
        "model_plan_path": str(model_plan_path),
        "prespec_sha256": selection_manifest["prespec_sha256"],
        "model_plan_sha256": selection_manifest["model_plan_sha256"],
        "query_type": selected_query,
        "offset": int(start),
        "limit": int(limit) if limit is not None else None,
        "arm_counts": arm_totals,
        "selected_row_count": len(selection_rows),
        "selected_participant_count": len(participants),
        "participants": participants,
        "gates": gates_out,
        "blocking_failure_count": len(blocking_failures),
        "selection_manifest_path": str(run_dir / "selection_manifest.json"),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "preflight_input_audit.json", input_audit)
    _write_json(run_dir / "selection_manifest.json", selection_manifest)
    return input_audit


def _assert_id_only_selection_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("kind") != "phase2_job2c_behavioral_selection_manifest_id_only":
        raise ValueError("invalid selection manifest kind")
    rows = manifest.get("selection")
    if not isinstance(rows, list):
        raise ValueError("selection manifest missing selection rows")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("selection row must be an object")
        allowed = {"source_order_stripped_request_id", "request_id_by_arm"}
        extra = sorted(set(row.keys()) - allowed)
        if extra:
            raise ValueError(f"selection row contains non-ID fields: {extra}")
        request_map = row.get("request_id_by_arm")
        if not isinstance(request_map, dict):
            raise ValueError("selection row request_id_by_arm must be an object")
        for arm in ARM_ORDER:
            if not str(request_map.get(arm) or ""):
                raise ValueError(f"selection row missing request id for arm {arm}")


def _selection_manifest_provenance_gate(
    manifest: dict[str, Any],
    *,
    prespec_path: Path,
    model_plan_path: Path,
) -> dict[str, Any]:
    expected_prespec = _prespec_hash(prespec_path)
    expected_model_plan = _model_plan_hash(model_plan_path)
    observed_arms = tuple(str(arm) for arm in manifest.get("arms", []))
    selection = [row for row in manifest.get("selection", []) if isinstance(row, dict)]
    source_ids = [str(row.get("source_order_stripped_request_id") or "") for row in selection]
    checks = {
        "prespec_sha256": manifest.get("prespec_sha256") == expected_prespec,
        "model_plan_sha256": manifest.get("model_plan_sha256") == expected_model_plan,
        "arms": observed_arms == ARM_ORDER,
        "selection_count": manifest.get("selection_count") == len(selection),
        "nonempty_unique_source_ids": bool(source_ids)
        and all(source_ids)
        and len(source_ids) == len(set(source_ids)),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "expected_prespec_sha256": expected_prespec,
        "observed_prespec_sha256": manifest.get("prespec_sha256"),
        "expected_model_plan_sha256": expected_model_plan,
        "observed_model_plan_sha256": manifest.get("model_plan_sha256"),
        "expected_arms": list(ARM_ORDER),
        "observed_arms": list(observed_arms),
    }


def _select_model(plan: Sequence[dict[str, str]], *, model_key: str | None, model_index: int | None) -> dict[str, Any]:
    if model_key:
        for idx, row in enumerate(plan):
            if row["model_key"] == model_key:
                return {**row, "model_index": idx}
        raise ValueError(f"model_key not found: {model_key}")
    if model_index is None:
        raise ValueError("one of model_key/model_index is required")
    idx = int(model_index)
    if idx < 0 or idx >= len(plan):
        raise ValueError(f"model_index out of range: {idx}")
    return {**plan[idx], "model_index": idx}


def _load_resume_valid_keys(path: Path, model_key: str) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("model_key") != model_key:
                continue
            if is_valid_completed_worker_row(row):
                arm = str(row.get("arm") or "")
                source_id = str(row.get("source_order_stripped_request_id") or "")
                if arm and source_id:
                    keys.add((arm, source_id))
    return keys


def _prepare_worker_rows_for_run(path: Path, model_key: str, *, resume: bool) -> dict[str, Any]:
    if not resume:
        _write_jsonl_atomic(path, [])
        return {
            "path": str(path),
            "status": "reset",
            "kept_valid_rows": 0,
            "dropped_rows": 0,
            "dropped_malformed_lines": 0,
        }

    sanitize_info = sanitize_jsonl(path)
    valid_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    parsed_count = 0
    for row in _iter_jsonl(path):
        parsed_count += 1
        if row.get("model_key") != model_key or not is_valid_completed_worker_row(row):
            continue
        arm = str(row.get("arm") or "")
        source_id = str(row.get("source_order_stripped_request_id") or "")
        if arm in ARM_ORDER and source_id:
            valid_by_key.setdefault((arm, source_id), row)
    kept_rows = [valid_by_key[key] for key in sorted(valid_by_key)]
    _write_jsonl_atomic(path, kept_rows)
    return {
        "path": str(path),
        "status": "pass",
        "kept_valid_rows": len(kept_rows),
        "dropped_rows": parsed_count - len(kept_rows),
        "dropped_malformed_lines": int(sanitize_info.get("dropped_lines") or 0),
    }


def _input_ids_list(inputs: Any) -> list[int]:
    ids = gates._input_get(inputs, "input_ids")
    if ids is None:
        return []
    if hasattr(ids, "detach"):
        value = ids.detach().cpu()
        if value.dim() == 2:
            value = value[0]
        return [int(item) for item in value.tolist()]
    if isinstance(ids, list):
        if ids and isinstance(ids[0], list):
            return [int(item) for item in ids[0]]
        return [int(item) for item in ids]
    return []


def _encode_candidate_text(tokenizer: Any, text: str) -> list[int]:
    if tokenizer is None:
        return []
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError as exc:
        raise RuntimeError(
            "tokenizer must support encode(..., add_special_tokens=False) for frozen candidates"
        ) from exc
    return [int(token_id) for token_id in ids]


def _visual_signature(inputs: Any) -> dict[str, Any]:
    sig: dict[str, Any] = {}
    for key, value in gates._input_items(inputs):
        lowered = key.lower()
        if "pixel" not in lowered and "image" not in lowered and "video" not in lowered and "grid" not in lowered:
            continue
        row: dict[str, Any] = {"type": type(value).__name__}
        if hasattr(value, "shape"):
            row["shape"] = [int(dim) for dim in value.shape]
        if hasattr(value, "dtype"):
            row["dtype"] = str(value.dtype)
        sig[key] = row
    return sig


def _visual_input_items(inputs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in gates._input_items(inputs):
        lowered = key.lower()
        if any(token in lowered for token in ("pixel", "image", "video", "grid")):
            out[key] = value
    return out


def _visual_identity_audit(base_inputs: Any, candidate_inputs: Any) -> dict[str, Any]:
    base_items = _visual_input_items(base_inputs)
    candidate_items = _visual_input_items(candidate_inputs)
    keys = sorted(set(base_items) | set(candidate_items))
    rows: list[dict[str, Any]] = []
    for key in keys:
        base_value = base_items.get(key)
        candidate_value = candidate_items.get(key)
        row: dict[str, Any] = {
            "key": key,
            "present_in_base": key in base_items,
            "present_in_candidate": key in candidate_items,
        }
        for prefix, value in (("base", base_value), ("candidate", candidate_value)):
            if hasattr(value, "shape"):
                row[f"{prefix}_shape"] = [int(dim) for dim in value.shape]
            if hasattr(value, "dtype"):
                row[f"{prefix}_dtype"] = str(value.dtype)
        exact = False
        if key in base_items and key in candidate_items and type(base_value) is type(candidate_value):
            try:
                import torch

                if isinstance(base_value, torch.Tensor):
                    exact = bool(torch.equal(base_value, candidate_value))
                elif isinstance(base_value, np.ndarray):
                    exact = bool(np.array_equal(base_value, candidate_value))
                else:
                    equality = base_value == candidate_value
                    exact = bool(equality.all()) if hasattr(equality, "all") else bool(equality)
            except Exception:
                exact = False
        row["exact_value_equal"] = exact
        rows.append(row)
    mismatch_keys = [row["key"] for row in rows if not row["exact_value_equal"]]
    exact_identity = bool(keys) and not mismatch_keys
    return {
        "status": "pass" if exact_identity else "fail",
        "visual_keys": keys,
        "tensor_count": len(keys),
        "exact_value_identity": exact_identity,
        "mismatch_keys": mismatch_keys,
        "items": rows,
        "comparison_method": "exact_value_equality_no_cpu_hash",
    }


def _build_user_messages(prompt: str, image_paths: Sequence[Path]) -> list[tuple[str, list[dict[str, Any]]]]:
    image_content_path = [{"type": "image", "image": str(path)} for path in image_paths]
    image_content_url = [{"type": "image", "url": path.resolve().as_uri()} for path in image_paths]
    return [
        (
            "chat_images_path_then_text",
            [{"role": "user", "content": [*image_content_path, {"type": "text", "text": prompt}]}],
        ),
        (
            "chat_text_then_images_path",
            [{"role": "user", "content": [{"type": "text", "text": prompt}, *image_content_path]}],
        ),
        (
            "chat_images_url_then_text",
            [{"role": "user", "content": [*image_content_url, {"type": "text", "text": prompt}]}],
        ),
    ]


def _prepare_base_inputs(
    *,
    runtime: Any,
    processor: Any,
    prompt: str,
    image_paths: Sequence[Path],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    return capture._prepare_inputs_multi(
        runtime=runtime,
        processor=processor,
        prompt=prompt,
        image_paths=[Path(path) for path in image_paths],
        device=device,
    )


def _prepare_candidate_inputs(
    *,
    runtime: Any,
    processor: Any,
    prompt: str,
    image_paths: Sequence[Path],
    candidate_text: str,
    device: Any,
    base_trace: dict[str, Any],
) -> tuple[list[tuple[Any, dict[str, Any]]], dict[str, Any]]:
    base_winner = str(base_trace.get("winner") or "")
    trace: dict[str, Any] = {
        "attempts": [],
        "candidate_sha256": _sha256_text(candidate_text),
        "winner": base_winner,
    }
    message_by_name = dict(_build_user_messages(prompt, image_paths))
    if base_winner not in message_by_name:
        raise RuntimeError(f"base template winner is not a reusable chat message ordering: {base_winner!r}")
    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError("processor has no apply_chat_template for exact candidate continuation")

    prepared: list[tuple[Any, dict[str, Any]]] = []
    user_messages = message_by_name[base_winner]
    messages = [*user_messages, {"role": "assistant", "content": candidate_text}]
    for continue_final in (True, False):
        name = f"{base_winner}__assistant_candidate__continue_{str(continue_final).lower()}"
        attempt = {
            "name": name,
            "message_order_winner": base_winner,
            "continue_final_message": bool(continue_final),
            "candidate_construction": CANDIDATE_CONSTRUCTION_NATIVE,
        }
        try:
            inputs = runtime._apply_chat_template_direct(
                processor,
                messages,
                add_generation_prompt=False,
                continue_final_message=continue_final,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if gates._input_get(inputs, "input_ids") is None:
                trace["attempts"].append({**attempt, "status": "skip", "reason": "missing_input_ids"})
                continue
            trace["attempts"].append({**attempt, "status": "prepared"})
            prepared.append((runtime._to_device(inputs, device), attempt))
        except Exception as exc:
            trace["attempts"].append(
                {**attempt, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}
            )
    if not prepared:
        raise RuntimeError("candidate template preparation failed for all base-winner variants")
    return prepared, trace


def _append_exact_candidate_continuation(
    base_inputs: Any,
    candidate_token_ids: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append a frozen text continuation without rebuilding the validated prompt."""
    import torch

    items = {str(key): value for key, value in gates._input_items(base_inputs)}
    input_ids = items.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2:
        raise RuntimeError("exact continuation requires rank-2 tensor input_ids")
    if int(input_ids.shape[0]) != 1:
        raise RuntimeError("exact continuation requires batch size one")
    suffix_ids = [int(token_id) for token_id in candidate_token_ids]
    if not suffix_ids:
        raise RuntimeError("exact continuation requires at least one candidate token")

    base_len = int(input_ids.shape[-1])
    suffix_len = len(suffix_ids)
    suffix = torch.tensor(
        [suffix_ids],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    out = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in items.items()
    }
    out["input_ids"] = torch.cat([input_ids, suffix], dim=-1)
    extended_fields = ["input_ids"]
    boundary_token_types: dict[str, int] = {}

    for key, fill_value in (
        ("attention_mask", 1),
        ("mm_token_type_ids", 0),
        ("token_type_ids", 0),
    ):
        value = items.get(key)
        if value is None:
            continue
        if (
            not isinstance(value, torch.Tensor)
            or value.dim() != 2
            or int(value.shape[0]) != 1
            or int(value.shape[-1]) != base_len
        ):
            raise RuntimeError(f"cannot safely extend sequence field {key!r}")
        if key != "attention_mask" and not bool(torch.all(value[:, -1] == 0).item()):
            raise RuntimeError(f"base text boundary for {key!r} is not token type zero")
        if key != "attention_mask":
            boundary_token_types[key] = int(value[0, -1].item())
        extension = torch.full(
            (1, suffix_len),
            fill_value,
            dtype=value.dtype,
            device=value.device,
        )
        out[key] = torch.cat([value, extension], dim=-1)
        extended_fields.append(key)

    for key, value in items.items():
        if key in extended_fields:
            continue
        if (
            isinstance(value, torch.Tensor)
            and value.dim() >= 2
            and int(value.shape[-1]) == base_len
        ):
            raise RuntimeError(
                f"unsupported sequence-aligned field {key!r} blocks exact continuation"
            )

    return out, {
        "candidate_construction": CANDIDATE_CONSTRUCTION_EXACT_APPEND,
        "base_input_length": base_len,
        "candidate_token_count": suffix_len,
        "full_input_length": base_len + suffix_len,
        "sequence_fields_extended": extended_fields,
        "base_text_boundary_token_types": boundary_token_types,
        "non_sequence_tensor_fields_cloned": sorted(
            key
            for key, value in items.items()
            if key not in extended_fields and isinstance(value, torch.Tensor)
        ),
        "non_tensor_fields_reused": sorted(
            key
            for key, value in items.items()
            if key not in extended_fields and not isinstance(value, torch.Tensor)
        ),
    }


def score_candidate_suffix_from_logits(
    *,
    base_input_ids: Sequence[int],
    full_input_ids: Sequence[int],
    logits: Any,
) -> dict[str, Any]:
    """Score [base_len, full_len) using shifted causal logits (t-1 -> token[t])."""
    import torch

    base_ids = [int(x) for x in base_input_ids]
    full_ids = [int(x) for x in full_input_ids]
    if not base_ids:
        return {"status": "fail", "reason": "empty_base_input_ids", "prefix_check": False}
    if len(full_ids) <= len(base_ids):
        return {"status": "fail", "reason": "full_length_not_greater_than_base", "prefix_check": False}
    prefix_ok = full_ids[: len(base_ids)] == base_ids
    if not prefix_ok:
        return {"status": "fail", "reason": "base_is_not_prefix_of_full", "prefix_check": False}
    if logits is None or not hasattr(logits, "detach"):
        return {"status": "fail", "reason": "missing_logits", "prefix_check": True}
    matrix = logits.detach()
    if matrix.dim() == 3:
        matrix = matrix[0]
    if matrix.dim() != 2:
        return {"status": "fail", "reason": "invalid_logits_rank", "prefix_check": True}
    full_len = len(full_ids)
    if int(matrix.shape[0]) < full_len:
        return {
            "status": "fail",
            "reason": "logits_sequence_too_short",
            "prefix_check": True,
            "logits_seq_len": int(matrix.shape[0]),
            "full_len": int(full_len),
        }

    base_len = len(base_ids)
    scored_token_ids = [int(token_id) for token_id in full_ids[base_len:]]
    prev_positions = torch.arange(base_len - 1, full_len - 1, dtype=torch.long, device=matrix.device)
    token_ids_tensor = torch.tensor(scored_token_ids, dtype=torch.long, device=matrix.device)
    with torch.inference_mode():
        selected_logits = matrix.float()[prev_positions]
        log_probs = torch.log_softmax(selected_logits, dim=-1)
        token_log_probs = log_probs.gather(1, token_ids_tensor.unsqueeze(1)).squeeze(1)
    if not bool(torch.isfinite(token_log_probs).all().item()):
        return {"status": "fail", "reason": "non_finite_token_logprob", "prefix_check": True}
    summed = float(token_log_probs.sum().item())
    mean = float(token_log_probs.mean().item())
    token_logprob_values = [float(value) for value in token_log_probs.detach().cpu().tolist()]
    if not math.isfinite(summed) or not math.isfinite(mean):
        return {"status": "fail", "reason": "non_finite_aggregate_logprob", "prefix_check": True}
    return {
        "status": "pass",
        "prefix_check": True,
        "base_input_length": int(base_len),
        "full_input_length": int(full_len),
        "token_count": int(len(scored_token_ids)),
        "scored_token_ids": scored_token_ids,
        "token_logprobs": token_logprob_values,
        "sum_logprob": summed,
        "mean_logprob": mean,
    }


def _extract_logits(output: Any) -> Any:
    logits = gates._extract_logits(output)
    if logits is None and hasattr(output, "logits"):
        logits = output.logits
    return logits


def _score_one_candidate(
    *,
    runtime: Any,
    model: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    prompt: str,
    image_paths: Sequence[Path],
    base_inputs: Any,
    base_trace: dict[str, Any],
    candidate_label: str,
    candidate_text: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    base_ids = _input_ids_list(base_inputs)
    base_visual = _visual_signature(base_inputs)
    expected_suffix_ids = _encode_candidate_text(tokenizer, candidate_text)
    try:
        prepared_attempts, prepare_trace = _prepare_candidate_inputs(
            runtime=runtime,
            processor=processor,
            prompt=prompt,
            image_paths=image_paths,
            candidate_text=candidate_text,
            device=device,
            base_trace=base_trace,
        )
    except Exception as exc:
        return {
            "status": "fail",
            "candidate_label": candidate_label,
            "candidate_text": candidate_text,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_s": time.perf_counter() - start,
            "base_template_trace": base_trace,
        }

    evaluated_attempts: list[dict[str, Any]] = []
    accepted: tuple[Any, list[int], list[int], dict[str, Any], dict[str, Any]] | None = None
    for full_inputs, attempt in prepared_attempts:
        full_ids = _input_ids_list(full_inputs)
        prefix_ok = full_ids[: len(base_ids)] == base_ids
        suffix_ids = full_ids[len(base_ids) :] if prefix_ok else []
        suffix_ok = prefix_ok and suffix_ids == expected_suffix_ids
        visual_audit = _visual_identity_audit(base_inputs, full_inputs)
        attempt_audit = {
            **attempt,
            "input_ids_present": bool(full_ids),
            "prefix_check": prefix_ok,
            "exact_candidate_suffix": suffix_ok,
            "visual_exact_identity": visual_audit["status"] == "pass",
            "visual_mismatch_keys": visual_audit["mismatch_keys"],
            "base_input_length": len(base_ids),
            "full_input_length": len(full_ids),
        }
        evaluated_attempts.append(attempt_audit)
        if accepted is None and prefix_ok and suffix_ok and visual_audit["status"] == "pass":
            accepted = (full_inputs, full_ids, suffix_ids, visual_audit, attempt)

    native_prefix_instability = len(evaluated_attempts) == 2 and all(
        bool(attempt.get("input_ids_present"))
        and not bool(attempt.get("prefix_check"))
        and not bool(attempt.get("exact_candidate_suffix"))
        and bool(attempt.get("visual_exact_identity"))
        for attempt in evaluated_attempts
    )
    if accepted is None and native_prefix_instability:
        fallback_attempt: dict[str, Any] = {
            "name": CANDIDATE_CONSTRUCTION_EXACT_APPEND,
            "message_order_winner": str(base_trace.get("winner") or ""),
            "continue_final_message": None,
            "candidate_construction": CANDIDATE_CONSTRUCTION_EXACT_APPEND,
            "activation_reason": "all_native_candidates_rewrote_base_prefix_with_identical_visuals",
        }
        try:
            full_inputs, construction_trace = _append_exact_candidate_continuation(
                base_inputs,
                expected_suffix_ids,
            )
            full_ids = _input_ids_list(full_inputs)
            prefix_ok = full_ids[: len(base_ids)] == base_ids
            suffix_ids = full_ids[len(base_ids) :] if prefix_ok else []
            suffix_ok = prefix_ok and suffix_ids == expected_suffix_ids
            visual_audit = _visual_identity_audit(base_inputs, full_inputs)
            fallback_audit = {
                **fallback_attempt,
                **construction_trace,
                "status": "prepared",
                "input_ids_present": bool(full_ids),
                "prefix_check": prefix_ok,
                "exact_candidate_suffix": suffix_ok,
                "visual_exact_identity": visual_audit["status"] == "pass",
                "visual_mismatch_keys": visual_audit["mismatch_keys"],
            }
            evaluated_attempts.append(fallback_audit)
            if prefix_ok and suffix_ok and visual_audit["status"] == "pass":
                accepted = (full_inputs, full_ids, suffix_ids, visual_audit, fallback_attempt)
        except Exception as exc:
            evaluated_attempts.append(
                {
                    **fallback_attempt,
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    prepare_trace["evaluated_attempts"] = evaluated_attempts
    if accepted is None:
        return {
            "status": "fail",
            "candidate_label": candidate_label,
            "candidate_text": candidate_text,
            "reason": "no_candidate_attempt_passed_exact_prefix_suffix_visual_gates",
            "prefix_check": False,
            "exact_candidate_suffix": False,
            "candidate_token_ids_expected": expected_suffix_ids,
            "base_template_trace": base_trace,
            "candidate_template_trace": prepare_trace,
            "base_visual_signature": base_visual,
            "visual_exact_identity": False,
            "latency_s": time.perf_counter() - start,
        }

    full_inputs, full_ids, suffix_ids, visual_audit, accepted_attempt = accepted
    prepare_trace["accepted_variant"] = dict(accepted_attempt)
    prepare_trace["continue_final_message"] = accepted_attempt.get("continue_final_message")
    candidate_construction = str(
        accepted_attempt.get("candidate_construction") or CANDIDATE_CONSTRUCTION_NATIVE
    )
    prepare_trace["candidate_construction"] = candidate_construction
    full_visual = _visual_signature(full_inputs)

    import torch

    with torch.inference_mode():
        output = model(**full_inputs)
    logits = _extract_logits(output)
    scored = score_candidate_suffix_from_logits(
        base_input_ids=base_ids,
        full_input_ids=full_ids,
        logits=logits,
    )
    return {
        "status": scored.get("status"),
        "candidate_label": candidate_label,
        "candidate_text": candidate_text,
        "candidate_construction": candidate_construction,
        "prefix_check": bool(scored.get("prefix_check")),
        "exact_candidate_suffix": bool(suffix_ids == expected_suffix_ids),
        "base_input_length": int(scored.get("base_input_length") or len(base_ids)),
        "full_input_length": int(scored.get("full_input_length") or len(full_ids)),
        "token_count": int(scored.get("token_count") or 0),
        "scored_token_ids": scored.get("scored_token_ids") or [],
        "token_logprobs": scored.get("token_logprobs") or [],
        "sum_logprob": scored.get("sum_logprob"),
        "mean_logprob": scored.get("mean_logprob"),
        "candidate_token_ids_expected": expected_suffix_ids,
        "candidate_token_ids_observed": suffix_ids,
        "base_template_trace": base_trace,
        "candidate_template_trace": prepare_trace,
        "base_visual_signature": base_visual,
        "candidate_visual_signature": full_visual,
        "visual_signature_match": True,
        "visual_exact_identity": True,
        "visual_identity_audit": visual_audit,
        "latency_s": time.perf_counter() - start,
        "error": scored.get("reason") if scored.get("status") != "pass" else None,
    }


def _decode_generated_tokens(runtime: Any, processor: Any, inputs: Any, output_ids: Any) -> str:
    if hasattr(runtime, "_decode_new_tokens"):
        return str(runtime._decode_new_tokens(processor=processor, inputs=inputs, output_ids=output_ids))
    input_ids = gates._input_get(inputs, "input_ids")
    generated = output_ids
    if input_ids is not None and hasattr(output_ids, "shape") and output_ids.shape[-1] >= input_ids.shape[-1]:
        generated = output_ids[:, input_ids.shape[-1] :]
    if hasattr(processor, "batch_decode"):
        return str(processor.batch_decode(generated, skip_special_tokens=True)[0]).strip()
    if hasattr(processor, "decode"):
        return str(processor.decode(generated[0], skip_special_tokens=True)).strip()
    return ""


def _run_greedy(
    *,
    model: Any,
    runtime: Any,
    processor: Any,
    base_inputs: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **base_inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )
    raw_text = _decode_generated_tokens(runtime, processor, base_inputs, output_ids)
    parsed, parse_error = _parse_visor_answer(raw_text)
    input_ids = gates._input_get(base_inputs, "input_ids")
    generated_tokens = None
    if input_ids is not None and hasattr(output_ids, "shape") and output_ids.shape[-1] >= input_ids.shape[-1]:
        generated_tokens = int(output_ids.shape[-1] - input_ids.shape[-1])
    return {
        "status": "pass",
        "raw_text": raw_text,
        "parsed": parsed,
        "parse_error": parse_error,
        "generated_token_count": generated_tokens,
        "latency_s": time.perf_counter() - start,
    }


def _frame_hash_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": _sha256_file(path) if path.exists() else None,
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    return rows


def _prepare_sample(
    *,
    request: dict[str, Any],
    arm: str,
    pack_path: Path,
    sample_root: Path,
    source_id: str,
) -> dict[str, Any]:
    request_id = str(request.get("request_id") or "")
    frame_paths = capture._request_frame_paths(request, pack_root=pack_path.parent)
    if not frame_paths:
        raise RuntimeError(f"{arm}/{request_id} has no frame paths")

    sample_dir = sample_root / re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    if arm in capture.CONTACT_SHEET_ARMS:
        out_image = sample_dir / "contact_sheet.jpg"
        capture._contact_sheet_layout(
            request=request,
            frame_paths=frame_paths,
            out_path=out_image,
            show_labels=False,
        )
        input_images = [out_image]
        presentation_mode = "contact_sheet"
    elif arm == "single_frame":
        input_images = [frame_paths[0]]
        presentation_mode = "single_image"
    else:
        input_images = frame_paths
        presentation_mode = "multi_image"
    method_frame_paths = [
        str(frame.get("path"))
        for frame in request.get("frames", [])
        if isinstance(frame, dict) and frame.get("path")
    ]
    query = request.get("query") if isinstance(request.get("query"), dict) else {}
    return {
        "arm": arm,
        "source_order_stripped_request_id": source_id,
        "request_id": request_id,
        "prompt": str(request.get("user_prompt") or ""),
        "system_prompt": str(request.get("system_prompt") or ""),
        "query_type": str(query.get("query_type") or ""),
        "public_question": str(query.get("text") or ""),
        "method_facing_pack": str(pack_path),
        "method_frame_paths": method_frame_paths,
        "method_frame_abs_paths": frame_paths,
        "method_frame_count": len(method_frame_paths),
        "input_image_paths": [Path(path).resolve() for path in input_images],
        "presentation_mode": presentation_mode,
        "input_image_count": len(input_images),
    }


def _code_snapshot_sha256() -> str:
    paths = [
        Path(__file__),
        REPO_ROOT / "research" / "lifecycle_memory" / "run_phase2_job2c_behavioral_companion.py",
        REPO_ROOT
        / "docs"
        / "research"
        / "lifecycle_memory"
        / "phase 2 research"
        / "jobs"
        / "job0"
        / "run_phase1_qwen27b_hf.py",
        REPO_ROOT / "research" / "lifecycle_memory" / "run_phase2_mech_job1_instrument_gates.py",
        REPO_ROOT / "research" / "lifecycle_memory" / "run_phase2_mech_job2_battery_capture.py",
        REPO_ROOT / "research" / "lifecycle_memory" / "phase2_job2_probe_analysis.py",
        REPO_ROOT / "research" / "lifecycle_memory" / "phase2_job2b_paired_analysis.py",
        REPO_ROOT / "research" / "lifecycle_memory" / "build_epic_visor_dynamic_smoke.py",
        REPO_ROOT
        / "research"
        / "lifecycle_memory"
        / "run_epic_visor_dynamic_llamacpp_metal.py",
        REPO_ROOT
        / "research"
        / "lifecycle_memory"
        / "run_epic_visor_dynamic_vlm_smoke.py",
        REPO_ROOT
        / "research"
        / "lifecycle_memory"
        / "protocol_v3_3rscan_visual_vlm.py",
    ]
    chunks: list[str] = []
    for path in paths:
        chunks.append(f"### {path.relative_to(REPO_ROOT)}\n")
        try:
            chunks.append(path.read_text(encoding="utf-8"))
            chunks.append("\n")
        except Exception as exc:
            chunks.append(f"<unreadable:{type(exc).__name__}>\n")
    return _sha256_text("".join(chunks))


def run_model(
    *,
    run_dir: Path,
    model_key: str | None = None,
    model_index: int | None = None,
    repo_dir: Path = REPO_ROOT,
    w0_dir: Path = DEFAULT_W0_DIR,
    prespec_path: Path = DEFAULT_PRESPEC_PATH,
    model_plan_path: Path = DEFAULT_MODEL_PLAN_PATH,
    selection_manifest_path: Path | None = None,
    dtype: str = "bf16",
    local_files_only: bool = False,
    resume: bool = True,
    runtime_module: Any | None = None,
    require_single_gpu: bool = True,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    repo_dir = repo_dir.expanduser().resolve()
    w0_dir = w0_dir.expanduser().resolve()
    prespec_path = prespec_path.expanduser().resolve()
    model_plan_path = model_plan_path.expanduser().resolve()
    if selection_manifest_path is None:
        selection_manifest_path = run_dir / "selection_manifest.json"
    else:
        selection_manifest_path = selection_manifest_path.expanduser().resolve()

    prespec = load_prespec(prespec_path)
    candidate_items = _candidate_items_from_prespec(prespec)
    model_plan = load_model_plan(model_plan_path)
    model_plan_gate = _validate_frozen_model_plan(model_plan)
    if model_plan_gate["status"] != "pass":
        raise ValueError("model plan does not match the frozen model ids and revisions")
    model_sel = _select_model(model_plan, model_key=model_key, model_index=model_index)
    model_key_final = str(model_sel["model_key"])
    model_id = str(model_sel["model_id"])
    revision = str(model_sel["revision"])

    manifest = _read_json(selection_manifest_path)
    _assert_id_only_selection_manifest(manifest)
    manifest_gate = _selection_manifest_provenance_gate(
        manifest,
        prespec_path=prespec_path,
        model_plan_path=model_plan_path,
    )
    if manifest_gate["status"] != "pass":
        raise ValueError("selection manifest provenance does not match prespec/model plan")
    selection_rows = [row for row in manifest.get("selection", []) if isinstance(row, dict)]

    model_dir = run_dir / "models" / model_key_final
    model_dir.mkdir(parents=True, exist_ok=True)
    rows_path = model_dir / "worker_rows.jsonl"
    progress_path = model_dir / "progress.json"
    summary_path = model_dir / "worker_summary.json"

    sanitize_info = _prepare_worker_rows_for_run(rows_path, model_key_final, resume=resume)
    existing = _load_resume_valid_keys(rows_path, model_key_final) if resume else set()

    summary: dict[str, Any] = {
        "kind": "phase2_job2c_behavioral_worker_summary",
        "status": "running",
        "model_key": model_key_final,
        "model_id": model_id,
        "model_index": model_sel.get("model_index"),
        "revision": revision,
        "dtype": dtype,
        "repo_dir": str(repo_dir),
        "w0_dir": str(w0_dir),
        "run_dir": str(run_dir),
        "prespec_path": str(prespec_path),
        "model_plan_path": str(model_plan_path),
        "selection_manifest_path": str(selection_manifest_path),
        "selection_count": len(selection_rows),
        "prespec_sha256": _prespec_hash(prespec_path),
        "model_plan_sha256": _model_plan_hash(model_plan_path),
        "code_snapshot_sha256": _code_snapshot_sha256(),
        "candidate_order": [key for key, _ in candidate_items],
        "candidate_strings": {key: value for key, value in candidate_items},
        "runtime_versions": gates._runtime_versions(),
        "rows_sanitize": sanitize_info,
        "started_at": _utc_now_iso(),
    }
    _write_json(summary_path, summary)
    _write_json(progress_path, {"status": "starting", "model_key": model_key_final, "updated_at": _utc_now_iso()})

    runtime = runtime_module or gates._load_job0_runtime(repo_dir)
    if require_single_gpu:
        gpu_snapshot = runtime._assert_single_visible_gpu()
    else:
        gpu_snapshot = {"status": "skipped"}
    summary["gpu_snapshot"] = gpu_snapshot
    summary["model_snapshot"] = gates._snapshot_provenance(
        model_id,
        revision,
        local_files_only=bool(local_files_only),
    )

    load_start = time.perf_counter()
    model, processor, device, adapter, tokenizer = runtime._load_model(
        model_id=model_id,
        dtype_name=dtype,
        trust_remote_code=True,
        local_files_only=bool(local_files_only),
        revision=revision,
        force_single_gpu_device_map=True,
    )
    summary["load_elapsed_s"] = time.perf_counter() - load_start
    summary["adapter"] = adapter
    summary["device"] = str(device)
    summary["processor_class"] = type(processor).__name__ if processor is not None else None
    summary["model_class"] = type(model).__name__
    if tokenizer is None and processor is not None:
        tokenizer = getattr(processor, "tokenizer", None)
    summary["tokenizer_class"] = type(tokenizer).__name__ if tokenizer is not None else None
    _write_json(summary_path, summary)

    pack_maps: dict[str, dict[str, dict[str, Any]]] = {}
    pack_paths: dict[str, Path] = {}
    for arm in ARM_ORDER:
        pack_path = _pack_path_for_arm(w0_dir, arm)
        pack_maps[arm] = _load_pack_rows(pack_path)
        pack_paths[arm] = pack_path

    rows_done = 0
    rows_failed = 0
    rows_skipped = 0
    first_error: str | None = None
    started = time.perf_counter()
    sample_root = model_dir / "input_samples"
    for index, selection in enumerate(selection_rows, start=1):
        source_id = str(selection.get("source_order_stripped_request_id") or "")
        request_map = selection.get("request_id_by_arm") if isinstance(selection.get("request_id_by_arm"), dict) else {}
        for arm in ARM_ORDER:
            if (arm, source_id) in existing:
                rows_skipped += 1
                continue
            row_start = time.perf_counter()
            request_id = str(request_map.get(arm) or "")
            request = pack_maps[arm].get(request_id)
            if request is None:
                rows_failed += 1
                err = f"missing_request_for_selection:{arm}:{request_id}"
                first_error = first_error or err
                row = {
                    "status": "fail",
                    "model_key": model_key_final,
                    "model_id": model_id,
                    "revision": revision,
                    "arm": arm,
                    "source_order_stripped_request_id": source_id,
                    "request_id": request_id,
                    "error": err,
                    "elapsed_s": time.perf_counter() - row_start,
                }
                _append_jsonl_atomic(rows_path, [row])
                continue

            row: dict[str, Any]
            try:
                sample = _prepare_sample(
                    request=request,
                    arm=arm,
                    pack_path=pack_paths[arm],
                    sample_root=sample_root / arm,
                    source_id=source_id,
                )
                prompt = str(sample["prompt"])
                base_inputs, base_trace = _prepare_base_inputs(
                    runtime=runtime,
                    processor=processor,
                    prompt=prompt,
                    image_paths=sample["input_image_paths"],
                    device=device,
                )
                candidate_rows: dict[str, dict[str, Any]] = {}
                candidate_failures: list[str] = []
                for candidate_label, candidate_text in candidate_items:
                    scored = _score_one_candidate(
                        runtime=runtime,
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=prompt,
                        image_paths=sample["input_image_paths"],
                        base_inputs=base_inputs,
                        base_trace=base_trace,
                        candidate_label=candidate_label,
                        candidate_text=candidate_text,
                    )
                    candidate_rows[candidate_label] = scored
                    if scored.get("status") != "pass":
                        candidate_failures.append(candidate_label)
                teacher_forced_status = "pass" if not candidate_failures else "fail"
                try:
                    greedy = _run_greedy(
                        model=model,
                        runtime=runtime,
                        processor=processor,
                        base_inputs=base_inputs,
                        max_new_tokens=int(prespec.get("generation", {}).get("max_new_tokens") or 96),
                    )
                except Exception as exc:
                    greedy = {
                        "status": "fail",
                        "reason": "greedy_generation_failed",
                        "raw_text": "",
                        "parsed": {
                            "answer": None,
                            "abstain": True,
                            "abstention_reason": "greedy_generation_failed",
                            "confidence": 0.0,
                        },
                        "parse_error": f"{type(exc).__name__}: {exc}",
                        "generated_token_count": None,
                        "latency_s": None,
                    }

                row = {
                    "status": "pass" if teacher_forced_status == "pass" else "fail",
                    "model_key": model_key_final,
                    "model_id": model_id,
                    "revision": revision,
                    "arm": arm,
                    "request_id": sample["request_id"],
                    "source_order_stripped_request_id": source_id,
                    "query_type": sample["query_type"],
                    "public_question": sample["public_question"],
                    "presentation_mode": sample["presentation_mode"],
                    "input_image_count": int(sample["input_image_count"]),
                    "method_facing_pack": sample["method_facing_pack"],
                    "method_frame_paths": list(sample["method_frame_paths"]),
                    "method_frame_count": int(sample["method_frame_count"]),
                    "method_frame_hashes": _frame_hash_rows(sample["method_frame_abs_paths"]),
                    "input_image_paths": [str(path) for path in sample["input_image_paths"]],
                    "input_image_hashes": _frame_hash_rows(sample["input_image_paths"]),
                    "prompt_sha256": _sha256_text(prompt),
                    "prompt_char_count": len(prompt),
                    "system_prompt_sha256": _sha256_text(str(sample["system_prompt"])),
                    "candidate_order": [key for key, _ in candidate_items],
                    "teacher_forced": {
                        "status": teacher_forced_status,
                        "candidate_failures": candidate_failures,
                        "candidate_rows": candidate_rows,
                    },
                    "greedy": greedy,
                    "provenance": {
                        "prespec_sha256": summary["prespec_sha256"],
                        "model_plan_sha256": summary["model_plan_sha256"],
                        "code_snapshot_sha256": summary["code_snapshot_sha256"],
                        "runtime_versions": summary["runtime_versions"],
                        "model_snapshot": summary.get("model_snapshot"),
                        "adapter": adapter,
                        "processor_class": summary.get("processor_class"),
                        "model_class": summary.get("model_class"),
                        "tokenizer_class": summary.get("tokenizer_class"),
                        "dtype": dtype,
                    },
                    "elapsed_s": time.perf_counter() - row_start,
                }
                if row["status"] == "pass":
                    rows_done += 1
                    existing.add((arm, source_id))
                else:
                    rows_failed += 1
                    if first_error is None:
                        first_error = f"teacher_forced_failed:{arm}:{source_id}"
            except Exception as exc:
                rows_failed += 1
                row = {
                    "status": "fail",
                    "model_key": model_key_final,
                    "model_id": model_id,
                    "revision": revision,
                    "arm": arm,
                    "request_id": request_id,
                    "source_order_stripped_request_id": source_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": time.perf_counter() - row_start,
                }
                if first_error is None:
                    first_error = row["error"]

            _append_jsonl_atomic(rows_path, [row])
            _write_json(
                progress_path,
                {
                    "status": "running",
                    "model_key": model_key_final,
                    "selection_index": index,
                    "selection_total": len(selection_rows),
                    "rows_done": rows_done,
                    "rows_failed": rows_failed,
                    "rows_skipped": rows_skipped,
                    "rows_seen": rows_done + rows_failed + rows_skipped,
                    "elapsed_s": time.perf_counter() - started,
                    "updated_at": _utc_now_iso(),
                },
            )

    summary.update(
        {
            "status": "pass" if rows_failed == 0 else "fail",
            "rows_done": rows_done,
            "rows_failed": rows_failed,
            "rows_skipped": rows_skipped,
            "rows_seen": rows_done + rows_failed + rows_skipped,
            "first_error": first_error,
            "worker_rows_path": str(rows_path),
            "ended_at": _utc_now_iso(),
            "elapsed_s": time.perf_counter() - started,
        }
    )
    _write_json(summary_path, summary)
    _write_json(progress_path, {"status": summary["status"], "summary_path": str(summary_path), "updated_at": _utc_now_iso()})
    return summary


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _contains_forbidden_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in forbidden:
                return True
            if _contains_forbidden_key(child, forbidden):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_forbidden_key(item, forbidden) for item in value)
    return False


def _candidate_construction_trace_valid(candidate: dict[str, Any]) -> bool:
    mode = candidate.get("candidate_construction")
    if mode not in ALLOWED_CANDIDATE_CONSTRUCTIONS:
        return False
    trace = candidate.get("candidate_template_trace")
    if not isinstance(trace, dict) or trace.get("candidate_construction") != mode:
        return False
    accepted = trace.get("accepted_variant")
    evaluated = trace.get("evaluated_attempts")
    if not isinstance(accepted, dict) or not isinstance(evaluated, list):
        return False
    if accepted.get("candidate_construction") != mode:
        return False
    accepted_name = accepted.get("name")
    matching = [
        row
        for row in evaluated
        if isinstance(row, dict)
        and row.get("name") == accepted_name
        and row.get("candidate_construction") == mode
    ]
    if len(matching) != 1:
        return False
    accepted_audit = matching[0]
    if not (
        bool(accepted_audit.get("input_ids_present"))
        and bool(accepted_audit.get("prefix_check"))
        and bool(accepted_audit.get("exact_candidate_suffix"))
        and bool(accepted_audit.get("visual_exact_identity"))
    ):
        return False

    if mode == CANDIDATE_CONSTRUCTION_EXACT_APPEND:
        activation_reason = "all_native_candidates_rewrote_base_prefix_with_identical_visuals"
        if accepted.get("activation_reason") != activation_reason:
            return False
        native = [
            row
            for row in evaluated
            if isinstance(row, dict)
            and row.get("candidate_construction") == CANDIDATE_CONSTRUCTION_NATIVE
        ]
        if len(native) != 2 or not all(
            bool(row.get("input_ids_present"))
            and not bool(row.get("prefix_check"))
            and not bool(row.get("exact_candidate_suffix"))
            and bool(row.get("visual_exact_identity"))
            for row in native
        ):
            return False
        extended = accepted_audit.get("sequence_fields_extended")
        if not isinstance(extended, list) or "input_ids" not in extended:
            return False
    return True


def is_valid_completed_worker_row(row: dict[str, Any]) -> bool:
    if row.get("status") != "pass":
        return False
    if _contains_forbidden_key(row, FORBIDDEN_WORKER_FIELDS):
        return False
    teacher = row.get("teacher_forced") if isinstance(row.get("teacher_forced"), dict) else {}
    if teacher.get("status") != "pass":
        return False
    candidate_rows = teacher.get("candidate_rows") if isinstance(teacher.get("candidate_rows"), dict) else {}
    if set(candidate_rows) != set(FROZEN_CANDIDATE_COMPLETIONS):
        return False
    for candidate in candidate_rows.values():
        if not isinstance(candidate, dict) or candidate.get("status") != "pass":
            return False
        label = str(candidate.get("candidate_label") or "")
        if label not in FROZEN_CANDIDATE_COMPLETIONS:
            return False
        if candidate.get("candidate_text") != FROZEN_CANDIDATE_COMPLETIONS[label]:
            return False
        if not _candidate_construction_trace_valid(candidate):
            return False
        if not bool(candidate.get("prefix_check")):
            return False
        if not bool(candidate.get("exact_candidate_suffix")):
            return False
        token_count = int(candidate.get("token_count") or 0)
        if token_count <= 0:
            return False
        token_ids = candidate.get("scored_token_ids")
        observed_token_ids = candidate.get("candidate_token_ids_observed")
        expected_token_ids = candidate.get("candidate_token_ids_expected")
        token_logprobs = candidate.get("token_logprobs")
        if (
            not isinstance(token_ids, list)
            or not isinstance(observed_token_ids, list)
            or not isinstance(expected_token_ids, list)
            or not isinstance(token_logprobs, list)
        ):
            return False
        if (
            len(token_ids) != token_count
            or len(token_logprobs) != token_count
            or token_ids != observed_token_ids
            or token_ids != expected_token_ids
        ):
            return False
        if any(_safe_float(value) is None for value in token_logprobs):
            return False
        base_trace = candidate.get("base_template_trace")
        candidate_trace = candidate.get("candidate_template_trace")
        if not isinstance(base_trace, dict) or not isinstance(candidate_trace, dict):
            return False
        if not base_trace.get("winner") or candidate_trace.get("winner") != base_trace.get("winner"):
            return False
        visual_audit = candidate.get("visual_identity_audit")
        if not isinstance(visual_audit, dict) or visual_audit.get("status") != "pass":
            return False
        if not bool(visual_audit.get("exact_value_identity")):
            return False
        summed = _safe_float(candidate.get("sum_logprob"))
        mean = _safe_float(candidate.get("mean_logprob"))
        if summed is None or mean is None:
            return False
        vector_sum = float(sum(float(value) for value in token_logprobs))
        vector_mean = vector_sum / token_count
        if not math.isclose(vector_sum, summed, rel_tol=1e-6, abs_tol=1e-6):
            return False
        if not math.isclose(vector_mean, mean, rel_tol=1e-6, abs_tol=1e-6):
            return False
    return True


def compute_two_candidate_metrics(
    *,
    sum_logprob_true: float,
    sum_logprob_false: float,
    expected_answer: bool,
    mean_logprob_true: float | None = None,
    mean_logprob_false: float | None = None,
) -> dict[str, Any]:
    a = float(sum_logprob_false)
    b = float(sum_logprob_true)
    max_val = max(a, b)
    lse = max_val + math.log(math.exp(a - max_val) + math.exp(b - max_val))
    p_false = math.exp(a - lse)
    p_true = math.exp(b - lse)
    prob_correct = p_true if bool(expected_answer) else p_false
    prob_correct = min(max(prob_correct, 1e-12), 1.0 - 1e-12)
    ce_bits = -math.log(prob_correct, 2.0)
    score_choice = bool(sum_logprob_true >= sum_logprob_false)
    correct_margin = (
        float(sum_logprob_true - sum_logprob_false)
        if expected_answer
        else float(sum_logprob_false - sum_logprob_true)
    )
    out = {
        "score_choice_answer": score_choice,
        "score_choice_accuracy": bool(score_choice == bool(expected_answer)),
        "prob_true": float(p_true),
        "prob_false": float(p_false),
        "prob_correct": float(prob_correct),
        "ce_bits": float(ce_bits),
        "correct_margin": float(correct_margin),
    }
    if mean_logprob_true is not None and mean_logprob_false is not None:
        ma = float(mean_logprob_false)
        mb = float(mean_logprob_true)
        max_m = max(ma, mb)
        lse_m = max_m + math.log(math.exp(ma - max_m) + math.exp(mb - max_m))
        p_false_m = math.exp(ma - lse_m)
        p_true_m = math.exp(mb - lse_m)
        prob_correct_m = p_true_m if bool(expected_answer) else p_false_m
        prob_correct_m = min(max(prob_correct_m, 1e-12), 1.0 - 1e-12)
        out.update(
            {
                "prob_true_per_token_mean": float(p_true_m),
                "prob_false_per_token_mean": float(p_false_m),
                "ce_bits_per_token_mean": float(-math.log(prob_correct_m, 2.0)),
                "correct_margin_per_token_mean": float(mb - ma if expected_answer else ma - mb),
            }
        )
    return out


def candidate_serialization_diagnostics(
    false_candidate: dict[str, Any],
    true_candidate: dict[str, Any],
    *,
    expected_answer: bool | None = None,
) -> dict[str, Any]:
    false_ids = [int(value) for value in false_candidate.get("scored_token_ids", [])]
    true_ids = [int(value) for value in true_candidate.get("scored_token_ids", [])]
    false_logs = [float(value) for value in false_candidate.get("token_logprobs", [])]
    true_logs = [float(value) for value in true_candidate.get("token_logprobs", [])]
    if len(false_ids) != len(false_logs) or len(true_ids) != len(true_logs):
        return {"status": "fail", "reason": "token_id_logprob_length_mismatch"}
    if not false_ids or not true_ids:
        return {"status": "fail", "reason": "empty_candidate_token_vector"}

    prefix = 0
    while prefix < min(len(false_ids), len(true_ids)) and false_ids[prefix] == true_ids[prefix]:
        prefix += 1
    if prefix == len(false_ids) == len(true_ids):
        return {"status": "fail", "reason": "candidate_token_vectors_do_not_diverge"}

    suffix = 0
    false_remaining = len(false_ids) - prefix
    true_remaining = len(true_ids) - prefix
    while (
        suffix < min(false_remaining, true_remaining)
        and false_ids[len(false_ids) - 1 - suffix] == true_ids[len(true_ids) - 1 - suffix]
    ):
        suffix += 1
    false_end = len(false_ids) - suffix
    true_end = len(true_ids) - suffix
    false_span_logs = false_logs[prefix:false_end]
    true_span_logs = true_logs[prefix:true_end]
    first_false_log = false_logs[prefix] if prefix < len(false_logs) else None
    first_true_log = true_logs[prefix] if prefix < len(true_logs) else None
    first_delta = (
        float(first_true_log - first_false_log)
        if first_false_log is not None and first_true_log is not None
        else None
    )
    span_delta = float(sum(true_span_logs) - sum(false_span_logs))
    out: dict[str, Any] = {
        "status": "pass",
        "common_prefix_token_count": prefix,
        "common_suffix_token_count": suffix,
        "first_divergent_token_index": prefix,
        "first_divergent_false_token_id": false_ids[prefix] if prefix < len(false_ids) else None,
        "first_divergent_true_token_id": true_ids[prefix] if prefix < len(true_ids) else None,
        "first_divergent_false_token_logprob": first_false_log,
        "first_divergent_true_token_logprob": first_true_log,
        "first_divergent_true_minus_false_logprob": first_delta,
        "false_divergent_span_start": prefix,
        "false_divergent_span_end_exclusive": false_end,
        "true_divergent_span_start": prefix,
        "true_divergent_span_end_exclusive": true_end,
        "false_divergent_span_token_ids": false_ids[prefix:false_end],
        "true_divergent_span_token_ids": true_ids[prefix:true_end],
        "false_divergent_span_logprob_sum": float(sum(false_span_logs)),
        "true_divergent_span_logprob_sum": float(sum(true_span_logs)),
        "divergent_span_true_minus_false_logprob": span_delta,
        "outside_holm_family": True,
    }
    if expected_answer is not None:
        if type(expected_answer) is not bool:
            return {"status": "fail", "reason": "expected_answer_not_bool"}
        direction = 1.0 if expected_answer else -1.0
        out["first_divergent_correct_minus_incorrect_logprob"] = (
            None if first_delta is None else float(direction * first_delta)
        )
        out["divergent_span_correct_minus_incorrect_logprob"] = float(direction * span_delta)
    return out


def _build_bootstrap_schedule(
    participants: Sequence[str],
    *,
    draws: int,
    seed: int,
    context: Sequence[Any] = (),
) -> list[np.ndarray]:
    participant_list = [str(participant) for participant in participants]
    if not participant_list:
        return []
    seed_material = json.dumps(
        ["job2c_behavioral_cluster_bootstrap_schedule", int(seed), *context],
        sort_keys=False,
    )
    seed_hash = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    rng_seed = int(seed_hash[:16], 16)
    rng = np.random.default_rng(rng_seed)
    schedule: list[np.ndarray] = []
    n = len(participant_list)
    for _ in range(int(draws)):
        chosen = rng.choice(n, size=n, replace=True)
        schedule.append(np.asarray(chosen, dtype=np.int64))
    return schedule


def participant_cluster_bootstrap_mean_delta(
    participant_mean_deltas: dict[str, float],
    *,
    schedule: Sequence[np.ndarray],
) -> dict[str, Any]:
    if not participant_mean_deltas:
        return {
            "point_estimate": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "n_boot_requested": int(len(schedule)),
            "n_boot_valid": 0,
            "method": "participant_cluster_bootstrap_mean_delta_shared_schedule",
        }
    participants = sorted(participant_mean_deltas)
    values = np.asarray([float(participant_mean_deltas[p]) for p in participants], dtype=float)
    point = float(np.mean(values))
    draws: list[float] = []
    for pick in schedule:
        if pick.size == 0:
            continue
        draw_values = values[pick]
        if not np.isfinite(draw_values).all():
            continue
        draws.append(float(np.mean(draw_values)))
    if draws:
        low = float(np.percentile(draws, 2.5))
        high = float(np.percentile(draws, 97.5))
    else:
        low = float("nan")
        high = float("nan")
    return {
        "point_estimate": point,
        "ci95_low": low,
        "ci95_high": high,
        "n_boot_requested": int(len(schedule)),
        "n_boot_valid": int(len(draws)),
        "method": "participant_cluster_bootstrap_mean_delta_shared_schedule",
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    clean = {
        key: float(value)
        for key, value in p_values.items()
        if value is not None and not math.isnan(float(value))
    }
    if not clean:
        return {}
    ordered = sorted(clean.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    prev = 0.0
    m = len(ordered)
    for rank, (key, p_value) in enumerate(ordered, start=1):
        adj = min(1.0, p_value * (m - rank + 1))
        adj = max(adj, prev)
        adjusted[key] = adj
        prev = adj
    return adjusted


def _contrast_arms(contrast: dict[str, Any]) -> tuple[str, str] | None:
    weights = contrast.get("arm_weights")
    if not isinstance(weights, dict):
        return None
    plus = [arm for arm, weight in weights.items() if int(weight) == 1]
    minus = [arm for arm, weight in weights.items() if int(weight) == -1]
    if len(plus) == 1 and len(minus) == 1:
        return plus[0], minus[0]
    return None


def _render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Job2c Behavioral Companion",
        "",
        f"Status: `{summary['overall_status']}`",
        f"Mode: `{summary['mode']}`",
        f"Run dir: `{summary['run_dir']}`",
    ]
    if summary.get("inference_status"):
        lines.append(f"Inference status: `{summary['inference_status']}`")
    lines.extend(["", "| Gate | Status | Blocking |", "| --- | --- | --- |"])
    for gate in summary.get("gates", []):
        lines.append(
            f"| {gate.get('gate')} | {gate.get('status')} | {str(bool(gate.get('blocking'))).lower()} |"
        )
    contrast_rows = summary.get("contrast_summary", [])
    if contrast_rows:
        lines.extend(
            [
                "",
                "## Contrast Summary",
                "",
                str(summary.get("sign_convention") or ""),
                "",
                "| Model | Contrast | Delta CE bits | Sign-flip p | Holm p |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in contrast_rows:
            delta_raw = row.get("participant_equal_mean_delta_ce_bits")
            delta_value = float(delta_raw) if isinstance(delta_raw, (int, float)) else float("nan")
            lines.append(
                "| {model} | {contrast} | {delta:.6f} | {p} | {holm} |".format(
                    model=row.get("model_key"),
                    contrast=row.get("contrast_key"),
                    delta=delta_value,
                    p=row.get("exact_sign_flip_p_two_sided_ce_bits"),
                    holm=row.get("holm_adjusted_p_confirmatory_family"),
                )
            )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            str(summary.get("claim_boundary") or ""),
            "",
        ]
    )
    return "\n".join(lines)


def _worker_provenance_issues(
    row: dict[str, Any],
    *,
    expected_model: dict[str, str],
    expected_prespec_sha256: str,
    expected_model_plan_sha256: str,
    expected_code_snapshot_sha256: str,
) -> list[str]:
    issues: list[str] = []
    if row.get("model_id") != expected_model["model_id"]:
        issues.append("model_id")
    if row.get("revision") != expected_model["revision"]:
        issues.append("revision")
    if not isinstance(row.get("prompt_sha256"), str) or len(row["prompt_sha256"]) != 64:
        issues.append("prompt_sha256")
    if row.get("presentation_mode") not in {"contact_sheet", "single_image", "multi_image"}:
        issues.append("presentation_mode")
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    if provenance.get("prespec_sha256") != expected_prespec_sha256:
        issues.append("prespec_sha256")
    if provenance.get("model_plan_sha256") != expected_model_plan_sha256:
        issues.append("model_plan_sha256")
    code_hash = provenance.get("code_snapshot_sha256")
    if code_hash != expected_code_snapshot_sha256:
        issues.append("code_snapshot_sha256")
    if _resolve_snapshot_revision(provenance) != expected_model["revision"]:
        issues.append("resolved_snapshot_revision")
    for field in ("adapter", "processor_class", "model_class", "dtype"):
        if not provenance.get(field):
            issues.append(field)
    if not isinstance(provenance.get("runtime_versions"), dict):
        issues.append("runtime_versions")
    return issues


def _ordered_worker_hashes(row: dict[str, Any], field: str) -> list[str] | None:
    values = row.get(field)
    if not isinstance(values, list) or not values:
        return None
    hashes: list[str] = []
    for value in values:
        if not isinstance(value, dict):
            return None
        sha = value.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            return None
        hashes.append(sha)
    return hashes


def _candidate_gate_issues(row: dict[str, Any]) -> dict[str, list[str]]:
    out = {"template": [], "visual": [], "token_logprobs": [], "serialization": []}
    teacher = row.get("teacher_forced") if isinstance(row.get("teacher_forced"), dict) else {}
    candidates = teacher.get("candidate_rows") if isinstance(teacher.get("candidate_rows"), dict) else {}
    for label in FROZEN_CANDIDATE_COMPLETIONS:
        candidate = candidates.get(label) if isinstance(candidates.get(label), dict) else {}
        base_trace = candidate.get("base_template_trace") if isinstance(candidate.get("base_template_trace"), dict) else {}
        candidate_trace = (
            candidate.get("candidate_template_trace")
            if isinstance(candidate.get("candidate_template_trace"), dict)
            else {}
        )
        if not base_trace.get("winner") or candidate_trace.get("winner") != base_trace.get("winner"):
            out["template"].append(label)
        if not _candidate_construction_trace_valid(candidate):
            out["template"].append(label)
        visual = candidate.get("visual_identity_audit") if isinstance(candidate.get("visual_identity_audit"), dict) else {}
        if visual.get("status") != "pass" or not bool(visual.get("exact_value_identity")):
            out["visual"].append(label)
        token_count = int(candidate.get("token_count") or 0)
        token_ids = candidate.get("scored_token_ids")
        observed_token_ids = candidate.get("candidate_token_ids_observed")
        expected_token_ids = candidate.get("candidate_token_ids_expected")
        token_logs = candidate.get("token_logprobs")
        if (
            token_count <= 0
            or not isinstance(token_ids, list)
            or not isinstance(observed_token_ids, list)
            or not isinstance(expected_token_ids, list)
            or not isinstance(token_logs, list)
            or len(token_ids) != token_count
            or len(token_logs) != token_count
            or token_ids != observed_token_ids
            or token_ids != expected_token_ids
            or any(_safe_float(value) is None for value in token_logs)
        ):
            out["token_logprobs"].append(label)
        elif (
            _safe_float(candidate.get("sum_logprob")) is None
            or _safe_float(candidate.get("mean_logprob")) is None
            or not math.isclose(
                sum(float(value) for value in token_logs),
                float(candidate["sum_logprob"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            or not math.isclose(
                sum(float(value) for value in token_logs) / token_count,
                float(candidate["mean_logprob"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            out["token_logprobs"].append(label)
    if set(candidates) == set(FROZEN_CANDIDATE_COMPLETIONS):
        diagnostic = candidate_serialization_diagnostics(candidates["false"], candidates["true"])
        if diagnostic.get("status") != "pass":
            out["serialization"].append(str(diagnostic.get("reason") or "failed"))
    else:
        out["serialization"].append("candidate_set_mismatch")
    return out


def _validation_row_from_worker(row: dict[str, Any]) -> dict[str, Any]:
    teacher = row.get("teacher_forced") if isinstance(row.get("teacher_forced"), dict) else {}
    candidates = teacher.get("candidate_rows") if isinstance(teacher.get("candidate_rows"), dict) else {}
    greedy = row.get("greedy") if isinstance(row.get("greedy"), dict) else {}
    parsed = greedy.get("parsed") if isinstance(greedy.get("parsed"), dict) else {}
    candidate_audit: dict[str, Any] = {}
    for label in FROZEN_CANDIDATE_COMPLETIONS:
        candidate = candidates.get(label) if isinstance(candidates.get(label), dict) else {}
        visual = candidate.get("visual_identity_audit") if isinstance(candidate.get("visual_identity_audit"), dict) else {}
        base_trace = candidate.get("base_template_trace") if isinstance(candidate.get("base_template_trace"), dict) else {}
        candidate_trace = (
            candidate.get("candidate_template_trace")
            if isinstance(candidate.get("candidate_template_trace"), dict)
            else {}
        )
        candidate_audit[label] = {
            "status": candidate.get("status"),
            "prefix_check": bool(candidate.get("prefix_check")),
            "exact_candidate_suffix": bool(candidate.get("exact_candidate_suffix")),
            "token_count": int(candidate.get("token_count") or 0),
            "token_logprob_count": len(candidate.get("token_logprobs") or []),
            "finite_scores": _safe_float(candidate.get("sum_logprob")) is not None
            and _safe_float(candidate.get("mean_logprob")) is not None,
            "base_template_winner": base_trace.get("winner"),
            "candidate_template_winner": candidate_trace.get("winner"),
            "candidate_construction": candidate.get("candidate_construction"),
            "visual_exact_identity": visual.get("status") == "pass"
            and bool(visual.get("exact_value_identity")),
        }
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return {
        "model_key": row.get("model_key"),
        "arm": row.get("arm"),
        "source_order_stripped_request_id": row.get("source_order_stripped_request_id"),
        "request_id": row.get("request_id"),
        "query_type": row.get("query_type"),
        "public_question": row.get("public_question"),
        "method_frame_count": row.get("method_frame_count"),
        "prompt_sha256": row.get("prompt_sha256"),
        "ordered_method_frame_sha256": _ordered_worker_hashes(row, "method_frame_hashes"),
        "ordered_input_image_sha256": _ordered_worker_hashes(row, "input_image_hashes"),
        "candidate_audit": candidate_audit,
        "greedy_status": greedy.get("status"),
        "greedy_parse_status": "pass"
        if not greedy.get("parse_error") and isinstance(parsed.get("answer"), bool)
        else "fail",
        "provenance": {
            "prespec_sha256": provenance.get("prespec_sha256"),
            "model_plan_sha256": provenance.get("model_plan_sha256"),
            "code_snapshot_sha256": provenance.get("code_snapshot_sha256"),
            "resolved_snapshot_revision": _resolve_snapshot_revision(provenance),
        },
    }


def _remove_inferential_artifacts(aggregate_dir: Path) -> None:
    for name in ("joined_rows.jsonl", "paired_rows.jsonl", "contrast_summary.json"):
        path = aggregate_dir / name
        if path.exists():
            path.unlink()


def _average_optional(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def _prelabel_aggregate_validation(
    *,
    run_dir: Path,
    prespec: dict[str, Any],
    prespec_path: Path,
    model_plan: Sequence[dict[str, str]],
    model_plan_path: Path,
    manifest: dict[str, Any],
    manifest_gate: dict[str, Any],
    target_models: Sequence[str],
    requested_models: Sequence[str],
    mode: str,
) -> dict[str, Any]:
    plan_by_key = {row["model_key"]: row for row in model_plan}
    expected_model_keys = [row["model_key"] for row in model_plan]
    selection_rows = [row for row in manifest.get("selection", []) if isinstance(row, dict)]
    selected_source_ids = [
        str(row.get("source_order_stripped_request_id") or "")
        for row in selection_rows
        if row.get("source_order_stripped_request_id")
    ]
    preflight_path = run_dir / "preflight_input_audit.json"
    preflight_audit = _read_json(preflight_path) if preflight_path.exists() else {}
    gates_out: list[dict[str, Any]] = [
        _gate(
            "frozen_model_plan",
            _validate_frozen_model_plan(model_plan)["status"],
            blocking=True,
            detail=_validate_frozen_model_plan(model_plan),
        ),
        _gate("selection_manifest_provenance", manifest_gate["status"], blocking=True, detail=manifest_gate),
        _gate(
            "preflight_audit_passed",
            "pass" if preflight_audit.get("status") == "pass" else "fail",
            blocking=True,
            detail={
                "path": str(preflight_path),
                "status": preflight_audit.get("status"),
                "selected_row_count": preflight_audit.get("selected_row_count"),
                "selected_participant_count": preflight_audit.get("selected_participant_count"),
            },
        ),
    ]
    unknown_models = sorted(set(target_models) - set(plan_by_key))
    gates_out.append(
        _gate(
            "selected_models_in_frozen_plan",
            "pass" if not unknown_models else "fail",
            blocking=True,
            detail={"unknown_models": unknown_models},
        )
    )
    gates_out.append(
        _gate(
            "required_frozen_model_set",
            "pass" if list(requested_models) == expected_model_keys else "fail",
            blocking=True,
            detail={
                "expected_model_keys": expected_model_keys,
                "requested_model_keys": list(requested_models),
            },
        )
    )
    if mode == "full":
        expected_rows = int(prespec.get("query_scope", {}).get("expected_row_count_per_arm_model") or 0)
        expected_participants = int(prespec.get("query_scope", {}).get("expected_participant_count") or 0)
        gates_out.extend(
            [
                _gate(
                    "full_selection_row_count",
                    "pass" if len(selected_source_ids) == expected_rows else "fail",
                    blocking=True,
                    detail={"expected": expected_rows, "observed": len(selected_source_ids)},
                ),
                _gate(
                    "full_selection_participant_count",
                    "pass"
                    if preflight_audit.get("selected_participant_count") == expected_participants
                    else "fail",
                    blocking=True,
                    detail={
                        "expected": expected_participants,
                        "observed": preflight_audit.get("selected_participant_count"),
                    },
                ),
            ]
        )
    else:
        gates_out.append(
            _gate(
                "canary_nonempty_selection",
                "pass" if selected_source_ids else "fail",
                blocking=True,
                detail={"observed": len(selected_source_ids)},
            )
        )

    expected_code_hash = _code_snapshot_sha256()
    missing_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    finite_failures: list[dict[str, Any]] = []
    revision_failures: list[dict[str, Any]] = []
    provenance_failures: list[dict[str, Any]] = []
    template_failures: list[dict[str, Any]] = []
    visual_failures: list[dict[str, Any]] = []
    token_vector_failures: list[dict[str, Any]] = []
    serialization_failures: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    accepted_rows: dict[tuple[str, str, str], dict[str, Any]] = {}

    for model_key in target_models:
        model_dir = run_dir / "models" / model_key
        summary_path = model_dir / "worker_summary.json"
        rows_path = model_dir / "worker_rows.jsonl"
        if not summary_path.exists() or not rows_path.exists():
            missing_rows.append({"model_key": model_key, "reason": "missing_model_outputs"})
            continue
        worker_summary = _read_json(summary_path)
        expected_model = plan_by_key.get(model_key)
        resolved_revision = _resolve_snapshot_revision(worker_summary)
        if (
            expected_model is None
            or worker_summary.get("model_key") != model_key
            or worker_summary.get("model_id") != expected_model["model_id"]
            or resolved_revision != expected_model["revision"]
        ):
            revision_failures.append(
                {
                    "model_key": model_key,
                    "expected_model_id": expected_model.get("model_id") if expected_model else None,
                    "observed_model_id": worker_summary.get("model_id"),
                    "expected_revision": expected_model.get("revision") if expected_model else None,
                    "resolved_revision": resolved_revision,
                }
            )
        by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in _iter_jsonl(rows_path):
            arm = str(row.get("arm") or "")
            source_id = str(row.get("source_order_stripped_request_id") or "")
            if arm in ARM_ORDER and source_id in selected_source_ids:
                by_key.setdefault((arm, source_id), []).append(row)
        for source_id in selected_source_ids:
            for arm in ARM_ORDER:
                rows = by_key.get((arm, source_id), [])
                identity = {"model_key": model_key, "arm": arm, "source_order_stripped_request_id": source_id}
                if not rows:
                    missing_rows.append({**identity, "reason": "missing_worker_row"})
                    continue
                if len(rows) != 1:
                    duplicate_rows.append({**identity, "row_count": len(rows)})
                row = rows[-1]
                validation_rows.append(_validation_row_from_worker(row))
                accepted_rows[(model_key, source_id, arm)] = row
                if not is_valid_completed_worker_row(row):
                    invalid_rows.append(identity)
                candidate_issues = _candidate_gate_issues(row)
                if candidate_issues["template"]:
                    template_failures.append({**identity, "candidates": candidate_issues["template"]})
                if candidate_issues["visual"]:
                    visual_failures.append({**identity, "candidates": candidate_issues["visual"]})
                if candidate_issues["token_logprobs"]:
                    token_vector_failures.append({**identity, "candidates": candidate_issues["token_logprobs"]})
                if candidate_issues["serialization"]:
                    serialization_failures.append({**identity, "issues": candidate_issues["serialization"]})
                teacher = row.get("teacher_forced") if isinstance(row.get("teacher_forced"), dict) else {}
                candidates = teacher.get("candidate_rows") if isinstance(teacher.get("candidate_rows"), dict) else {}
                for label in FROZEN_CANDIDATE_COMPLETIONS:
                    candidate = candidates.get(label) if isinstance(candidates.get(label), dict) else {}
                    if _safe_float(candidate.get("sum_logprob")) is None or _safe_float(candidate.get("mean_logprob")) is None:
                        finite_failures.append({**identity, "candidate": label})
                if expected_model is not None:
                    issues = _worker_provenance_issues(
                        row,
                        expected_model=expected_model,
                        expected_prespec_sha256=_prespec_hash(prespec_path),
                        expected_model_plan_sha256=_model_plan_hash(model_plan_path),
                        expected_code_snapshot_sha256=expected_code_hash,
                    )
                    if issues:
                        provenance_failures.append({**identity, "issues": issues})

    def issue_gate(name: str, failures: list[dict[str, Any]]) -> dict[str, Any]:
        return _gate(
            name,
            "pass" if not failures else "fail",
            blocking=True,
            detail={"failure_count": len(failures), "examples": failures[:20]},
        )

    gates_out.extend(
        [
            issue_gate("worker_rows_present", missing_rows),
            issue_gate("worker_row_duplicates", duplicate_rows),
            issue_gate("teacher_forced_completeness", invalid_rows),
            issue_gate("finite_teacher_forced_scores", finite_failures),
            issue_gate("pinned_model_revisions", revision_failures),
            issue_gate("worker_row_provenance", provenance_failures),
            issue_gate("candidate_base_winner_template_identity", template_failures),
            issue_gate("candidate_exact_visual_tensor_identity", visual_failures),
            issue_gate("candidate_per_token_logprob_vectors", token_vector_failures),
            issue_gate("candidate_serialization_diagnostics", serialization_failures),
        ]
    )

    alignment_failures: list[dict[str, Any]] = []
    prompt_failures: list[dict[str, Any]] = []
    marked_input_failures: list[dict[str, Any]] = []
    for model_key in target_models:
        for source_id in selected_source_ids:
            per_arm = {
                arm: accepted_rows.get((model_key, source_id, arm))
                for arm in ARM_ORDER
            }
            if any(row is None for row in per_arm.values()):
                alignment_failures.append(
                    {
                        "model_key": model_key,
                        "source_order_stripped_request_id": source_id,
                        "reason": "missing_arm",
                    }
                )
                continue
            pcs = per_arm[ARM_PCS] or {}
            mpp = per_arm[ARM_MPP] or {}
            vp = per_arm[ARM_VP] or {}
            for arm, other in ((ARM_MPP, mpp), (ARM_VP, vp)):
                for field in ("query_type", "public_question", "method_frame_count"):
                    if pcs.get(field) != other.get(field):
                        alignment_failures.append(
                            {
                                "model_key": model_key,
                                "source_order_stripped_request_id": source_id,
                                "field": field,
                                "arm_a": ARM_PCS,
                                "arm_b": arm,
                            }
                        )
            if pcs.get("prompt_sha256") != mpp.get("prompt_sha256"):
                prompt_failures.append(
                    {
                        "model_key": model_key,
                        "source_order_stripped_request_id": source_id,
                        "pcs_prompt_sha256": pcs.get("prompt_sha256"),
                        "mpp_prompt_sha256": mpp.get("prompt_sha256"),
                    }
                )
            mpp_method = _ordered_worker_hashes(mpp, "method_frame_hashes")
            vp_method = _ordered_worker_hashes(vp, "method_frame_hashes")
            mpp_input = _ordered_worker_hashes(mpp, "input_image_hashes")
            vp_input = _ordered_worker_hashes(vp, "input_image_hashes")
            if mpp_method is None or vp_method is None or mpp_method != vp_method or mpp_input is None or vp_input is None or mpp_input != vp_input:
                marked_input_failures.append(
                    {
                        "model_key": model_key,
                        "source_order_stripped_request_id": source_id,
                        "mpp_method_frame_sha256": mpp_method,
                        "vp_method_frame_sha256": vp_method,
                        "mpp_input_image_sha256": mpp_input,
                        "vp_input_image_sha256": vp_input,
                    }
                )
    gates_out.extend(
        [
            issue_gate("three_arm_worker_pairing", alignment_failures),
            issue_gate("pcs_mpp_prompt_sha256_identity", prompt_failures),
            issue_gate("mpp_vp_method_and_contact_sheet_sha256_identity", marked_input_failures),
        ]
    )
    parse_failures = [
        row
        for row in validation_rows
        if row.get("greedy_parse_status") != "pass"
    ]
    parse_counts_by_model: dict[str, dict[str, int]] = {}
    for row in validation_rows:
        model_key = str(row.get("model_key") or "")
        counts = parse_counts_by_model.setdefault(model_key, {"rows": 0, "failures": 0})
        counts["rows"] += 1
        if row.get("greedy_parse_status") != "pass":
            counts["failures"] += 1
    systematic_models = {
        model_key: counts
        for model_key, counts in parse_counts_by_model.items()
        if counts["rows"] > 0 and counts["failures"] == counts["rows"]
    }
    gates_out.append(
        _gate(
            "systematic_greedy_parse_failure",
            "pass" if not systematic_models else "fail",
            blocking=True,
            detail={
                "definition": "all selected rows for any model fail greedy JSON parsing",
                "models": systematic_models,
            },
        )
    )
    gates_out.append(
        _gate(
            "greedy_parse_failure_claim_gate",
            "pass" if not parse_failures else "fail",
            blocking=False,
            detail={"failure_count": len(parse_failures), "row_count": len(validation_rows)},
        )
    )
    return {
        "gates": gates_out,
        "validation_rows": validation_rows,
        "selected_source_ids": selected_source_ids,
        "accepted_rows": accepted_rows,
        "preflight_audit": preflight_audit,
    }


def _write_withheld_aggregate(
    *,
    run_dir: Path,
    w0_dir: Path,
    prespec: dict[str, Any],
    prespec_path: Path,
    model_plan_path: Path,
    selection_manifest_path: Path,
    mode: str,
    target_models: Sequence[str],
    selected_source_ids: Sequence[str],
    validation_rows: Sequence[dict[str, Any]],
    gates_out: Sequence[dict[str, Any]],
    inference_status: str,
) -> dict[str, Any]:
    blocking_failures = [gate for gate in gates_out if gate["blocking"] and gate["status"] != "pass"]
    overall_status = "pass" if mode == "canary" and not blocking_failures else "fail"
    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    _remove_inferential_artifacts(aggregate_dir)
    _write_jsonl_atomic(aggregate_dir / "validation_rows.jsonl", list(validation_rows))
    _write_json(aggregate_dir / "gate_results.json", {"gates": list(gates_out)})
    summary = {
        "kind": "phase2_job2c_behavioral_aggregate_summary",
        "mode": mode,
        "overall_status": overall_status,
        "inference_status": inference_status,
        "run_dir": str(run_dir),
        "w0_dir": str(w0_dir),
        "prespec_path": str(prespec_path),
        "model_plan_path": str(model_plan_path),
        "selection_manifest_path": str(selection_manifest_path),
        "target_models": list(target_models),
        "selected_row_count": len(selected_source_ids),
        "validation_row_count": len(validation_rows),
        "prespec_sha256": _prespec_hash(prespec_path),
        "model_plan_sha256": _model_plan_hash(model_plan_path),
        "gates": list(gates_out),
        "blocking_failure_count": len(blocking_failures),
        "claim_boundary": str(prespec.get("claim_boundary") or ""),
        "generated_at": _utc_now_iso(),
    }
    _write_json(aggregate_dir / "run_manifest.json", summary)
    _atomic_write_text(aggregate_dir / "report.md", _render_markdown_report(summary))
    _write_json(run_dir / "aggregate_summary.json", summary)
    if blocking_failures:
        _write_json(
            aggregate_dir / "failure_report.json",
            {
                "status": "fail",
                "inference_status": "inference_withheld_due_to_blocking_gates",
                "blocking_failures": blocking_failures,
                "summary_path": str(run_dir / "aggregate_summary.json"),
            },
        )
    return summary


def aggregate(
    *,
    run_dir: Path,
    w0_dir: Path = DEFAULT_W0_DIR,
    prespec_path: Path = DEFAULT_PRESPEC_PATH,
    model_plan_path: Path = DEFAULT_MODEL_PLAN_PATH,
    selection_manifest_path: Path | None = None,
    mode: str = "full",
    models: Sequence[str] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    w0_dir = w0_dir.expanduser().resolve()
    prespec_path = prespec_path.expanduser().resolve()
    model_plan_path = model_plan_path.expanduser().resolve()
    mode = str(mode).strip().lower()
    if mode not in {"full", "canary"}:
        raise ValueError("mode must be 'full' or 'canary'")
    if selection_manifest_path is None:
        selection_manifest_path = run_dir / "selection_manifest.json"
    else:
        selection_manifest_path = selection_manifest_path.expanduser().resolve()

    prespec = load_prespec(prespec_path)
    candidate_items = _candidate_items_from_prespec(prespec)
    candidate_order = [key for key, _ in candidate_items]
    model_plan = load_model_plan(model_plan_path)
    plan_by_key = {row["model_key"]: row for row in model_plan}
    frozen_plan_gate = _validate_frozen_model_plan(model_plan)

    manifest = _read_json(selection_manifest_path)
    _assert_id_only_selection_manifest(manifest)
    manifest_gate = _selection_manifest_provenance_gate(
        manifest,
        prespec_path=prespec_path,
        model_plan_path=model_plan_path,
    )
    selection_rows = [row for row in manifest.get("selection", []) if isinstance(row, dict)]
    selected_source_ids = [
        str(row.get("source_order_stripped_request_id") or "")
        for row in selection_rows
        if row.get("source_order_stripped_request_id")
    ]

    target_models = [row["model_key"] for row in model_plan]
    requested_models = [str(model) for model in models] if models is not None else target_models

    prelabel = _prelabel_aggregate_validation(
        run_dir=run_dir,
        prespec=prespec,
        prespec_path=prespec_path,
        model_plan=model_plan,
        model_plan_path=model_plan_path,
        manifest=manifest,
        manifest_gate=manifest_gate,
        target_models=target_models,
        requested_models=requested_models,
        mode=mode,
    )
    prelabel_gates = list(prelabel["gates"])
    prelabel_blocking_failures = [
        gate for gate in prelabel_gates if gate["blocking"] and gate["status"] != "pass"
    ]
    if mode == "canary":
        return _write_withheld_aggregate(
            run_dir=run_dir,
            w0_dir=w0_dir,
            prespec=prespec,
            prespec_path=prespec_path,
            model_plan_path=model_plan_path,
            selection_manifest_path=selection_manifest_path,
            mode=mode,
            target_models=target_models,
            selected_source_ids=prelabel["selected_source_ids"],
            validation_rows=prelabel["validation_rows"],
            gates_out=prelabel_gates,
            inference_status="label_free_canary_validation_complete"
            if not prelabel_blocking_failures
            else "label_free_canary_validation_failed",
        )
    if prelabel_blocking_failures:
        return _write_withheld_aggregate(
            run_dir=run_dir,
            w0_dir=w0_dir,
            prespec=prespec,
            prespec_path=prespec_path,
            model_plan_path=model_plan_path,
            selection_manifest_path=selection_manifest_path,
            mode=mode,
            target_models=target_models,
            selected_source_ids=prelabel["selected_source_ids"],
            validation_rows=prelabel["validation_rows"],
            gates_out=prelabel_gates,
            inference_status="inference_withheld_due_to_blocking_gates",
        )

    hidden_rows_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    hidden_alignment_input: dict[str, list[dict[str, Any]]] = {}
    hidden_label_type_failures: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        hidden_map: dict[str, dict[str, Any]] = {}
        request_rows = _load_pack_rows(_pack_path_for_arm(w0_dir, arm))
        for hidden in _load_hidden_rows(_hidden_path_for_arm(w0_dir, arm)):
            source_id = str(hidden.get("source_order_stripped_request_id") or "")
            if not source_id or source_id not in selected_source_ids:
                continue
            request_id = str(hidden.get("request_id") or "")
            request = request_rows.get(request_id, {})
            query = request.get("query") if isinstance(request.get("query"), dict) else {}
            expected_answer = hidden.get("expected_answer")
            if type(expected_answer) is not bool:
                hidden_label_type_failures.append(
                    {
                        "arm": arm,
                        "source_order_stripped_request_id": source_id,
                        "observed_type": type(expected_answer).__name__,
                    }
                )
                continue
            hidden_map[source_id] = {
                "source_order_stripped_request_id": source_id,
                "request_id": request_id,
                "expected_answer": expected_answer,
                "participant": str(hidden.get("source_tier1_participant_id") or ""),
                "video_id": str(hidden.get("source_video_id") or ""),
                "query_type": str(hidden.get("query_type") or ""),
                "public_question": str(query.get("text") or ""),
                "target_label": str(hidden.get("target_label") or query.get("target_label") or ""),
                "frame_count": len(
                    [
                        frame
                        for frame in request.get("frames", [])
                        if isinstance(frame, dict) and frame.get("path")
                    ]
                ),
                "source_tier1_participant_id": str(hidden.get("source_tier1_participant_id") or ""),
                "source_video_id": str(hidden.get("source_video_id") or ""),
            }
        hidden_rows_by_arm[arm] = hidden_map
        hidden_alignment_input[arm] = list(hidden_map.values())
    hidden_alignment = validate_three_arm_alignment(hidden_alignment_input)

    expected_rows = int(prespec.get("query_scope", {}).get("expected_row_count_per_arm_model") or 0)
    expected_participants = int(prespec.get("query_scope", {}).get("expected_participant_count") or 0)
    participant_ids = sorted(
        {
            str(hidden_rows_by_arm[ARM_PCS][source_id]["participant"])
            for source_id in selected_source_ids
            if source_id in hidden_rows_by_arm[ARM_PCS]
        }
    )

    gates_out: list[dict[str, Any]] = list(prelabel_gates)
    gates_out.append(_gate("frozen_model_plan", frozen_plan_gate["status"], blocking=True, detail=frozen_plan_gate))
    gates_out.append(
        _gate(
            "selection_manifest_provenance",
            manifest_gate["status"],
            blocking=True,
            detail=manifest_gate,
        )
    )
    unknown_models = sorted(set(target_models) - set(plan_by_key))
    gates_out.append(
        _gate(
            "selected_models_in_frozen_plan",
            "pass" if not unknown_models else "fail",
            blocking=True,
            detail={"unknown_models": unknown_models},
        )
    )
    gates_out.append(_gate("hidden_three_arm_alignment", hidden_alignment["status"], blocking=True, detail=hidden_alignment))
    gates_out.append(
        _gate(
            "hidden_expected_answer_real_bool",
            "pass" if not hidden_label_type_failures else "fail",
            blocking=True,
            detail={
                "failure_count": len(hidden_label_type_failures),
                "examples": hidden_label_type_failures[:20],
            },
        )
    )

    if mode == "full":
        gates_out.append(
            _gate(
                "full_selection_row_count",
                "pass" if len(selected_source_ids) == expected_rows else "fail",
                blocking=True,
                detail={"expected": expected_rows, "observed": len(selected_source_ids)},
            )
        )
        gates_out.append(
            _gate(
                "full_selection_participant_count",
                "pass" if len(participant_ids) == expected_participants else "fail",
                blocking=True,
                detail={"expected": expected_participants, "observed": len(participant_ids)},
            )
        )
    else:
        gates_out.append(
            _gate(
                "canary_nonempty_selection",
                "pass" if selected_source_ids else "fail",
                blocking=True,
                detail={"observed": len(selected_source_ids)},
            )
        )

    hidden_join_blocking_failures = [
        gate for gate in gates_out if gate["blocking"] and gate["status"] != "pass"
    ]
    if hidden_join_blocking_failures:
        return _write_withheld_aggregate(
            run_dir=run_dir,
            w0_dir=w0_dir,
            prespec=prespec,
            prespec_path=prespec_path,
            model_plan_path=model_plan_path,
            selection_manifest_path=selection_manifest_path,
            mode=mode,
            target_models=target_models,
            selected_source_ids=prelabel["selected_source_ids"],
            validation_rows=prelabel["validation_rows"],
            gates_out=gates_out,
            inference_status="inference_withheld_due_to_blocking_gates",
        )

    joined_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    duplicate_keys: list[dict[str, Any]] = []
    teacher_forced_failures: list[dict[str, Any]] = []
    nonfinite_rows: list[dict[str, Any]] = []
    revision_failures: list[dict[str, Any]] = []
    provenance_failures: list[dict[str, Any]] = []

    for model_key in target_models:
        model_dir = run_dir / "models" / model_key
        summary_path = model_dir / "worker_summary.json"
        rows_path = model_dir / "worker_rows.jsonl"
        if not summary_path.exists() or not rows_path.exists():
            missing_rows.append({"model_key": model_key, "reason": "missing_model_outputs"})
            continue

        worker_summary = _read_json(summary_path)
        resolved_revision = _resolve_snapshot_revision(worker_summary)
        expected_model = plan_by_key.get(model_key)
        expected_revision = expected_model.get("revision") if expected_model else None
        if (
            expected_model is None
            or worker_summary.get("model_key") != model_key
            or worker_summary.get("model_id") != expected_model["model_id"]
            or resolved_revision != expected_revision
        ):
            revision_failures.append(
                {
                    "model_key": model_key,
                    "expected_model_id": expected_model.get("model_id") if expected_model else None,
                    "observed_model_id": worker_summary.get("model_id"),
                    "expected_revision": expected_revision,
                    "resolved_revision": resolved_revision,
                }
            )

        by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in _iter_jsonl(rows_path):
            arm = str(row.get("arm") or "")
            source_id = str(row.get("source_order_stripped_request_id") or "")
            if arm not in ARM_ORDER or not source_id:
                continue
            if source_id not in selected_source_ids:
                continue
            by_key.setdefault((arm, source_id), []).append(row)

        for source_id in selected_source_ids:
            for arm in ARM_ORDER:
                rows = by_key.get((arm, source_id), [])
                if not rows:
                    missing_rows.append(
                        {
                            "model_key": model_key,
                            "arm": arm,
                            "source_order_stripped_request_id": source_id,
                            "reason": "missing_worker_row",
                        }
                    )
                    continue
                if len(rows) > 1:
                    duplicate_keys.append(
                        {
                            "model_key": model_key,
                            "arm": arm,
                            "source_order_stripped_request_id": source_id,
                            "row_count": len(rows),
                        }
                    )
                row = rows[-1]
                if not is_valid_completed_worker_row(row):
                    teacher_forced_failures.append(
                        {
                            "model_key": model_key,
                            "arm": arm,
                            "source_order_stripped_request_id": source_id,
                        }
                    )
                    continue
                if expected_model is not None:
                    issues = _worker_provenance_issues(
                        row,
                        expected_model=expected_model,
                        expected_prespec_sha256=_prespec_hash(prespec_path),
                        expected_model_plan_sha256=_model_plan_hash(model_plan_path),
                        expected_code_snapshot_sha256=_code_snapshot_sha256(),
                    )
                    if issues:
                        provenance_failures.append(
                            {
                                "model_key": model_key,
                                "arm": arm,
                                "source_order_stripped_request_id": source_id,
                                "issues": issues,
                            }
                        )
                        continue
                teacher = row["teacher_forced"]
                candidate_rows = teacher["candidate_rows"]
                if list(row.get("candidate_order") or []) != candidate_order:
                    teacher_forced_failures.append(
                        {
                            "model_key": model_key,
                            "arm": arm,
                            "source_order_stripped_request_id": source_id,
                            "reason": "candidate_order_drift",
                        }
                    )
                    continue

                true_row = candidate_rows["true"]
                false_row = candidate_rows["false"]
                s_true = _safe_float(true_row.get("sum_logprob"))
                s_false = _safe_float(false_row.get("sum_logprob"))
                m_true = _safe_float(true_row.get("mean_logprob"))
                m_false = _safe_float(false_row.get("mean_logprob"))
                if s_true is None or s_false is None or m_true is None or m_false is None:
                    nonfinite_rows.append(
                        {
                            "model_key": model_key,
                            "arm": arm,
                            "source_order_stripped_request_id": source_id,
                        }
                    )
                    continue
                hidden = hidden_rows_by_arm[arm].get(source_id)
                if hidden is None:
                    missing_rows.append(
                        {
                            "model_key": model_key,
                            "arm": arm,
                            "source_order_stripped_request_id": source_id,
                            "reason": "missing_hidden_join",
                        }
                    )
                    continue
                expected_answer = hidden["expected_answer"]
                metrics = compute_two_candidate_metrics(
                    sum_logprob_true=s_true,
                    sum_logprob_false=s_false,
                    expected_answer=expected_answer,
                    mean_logprob_true=m_true,
                    mean_logprob_false=m_false,
                )
                serialization_diagnostic = candidate_serialization_diagnostics(
                    false_row,
                    true_row,
                    expected_answer=expected_answer,
                )
                greedy = row.get("greedy") if isinstance(row.get("greedy"), dict) else {}
                parsed = greedy.get("parsed") if isinstance(greedy.get("parsed"), dict) else {}
                greedy_answer = parsed.get("answer")
                greedy_parse_error = greedy.get("parse_error")
                greedy_parse_failure = bool(greedy_parse_error) or not isinstance(greedy_answer, bool)
                greedy_accuracy = (
                    bool(greedy_answer == expected_answer)
                    if isinstance(greedy_answer, bool) and not greedy_parse_failure
                    else None
                )
                joined_rows.append(
                    {
                        "model_key": model_key,
                        "arm": arm,
                        "source_order_stripped_request_id": source_id,
                        "request_id": row.get("request_id"),
                        "query_type": row.get("query_type"),
                        "public_question": row.get("public_question"),
                        "method_frame_count": row.get("method_frame_count"),
                        "expected_answer": expected_answer,
                        "participant": hidden["participant"],
                        "source_video_id": hidden["source_video_id"],
                        "score_choice_answer": metrics["score_choice_answer"],
                        "score_choice_accuracy": metrics["score_choice_accuracy"],
                        "prob_true": metrics["prob_true"],
                        "prob_false": metrics["prob_false"],
                        "prob_correct": metrics["prob_correct"],
                        "ce_bits": metrics["ce_bits"],
                        "correct_margin": metrics["correct_margin"],
                        "prob_true_per_token_mean": metrics.get("prob_true_per_token_mean"),
                        "ce_bits_per_token_mean": metrics.get("ce_bits_per_token_mean"),
                        "correct_margin_per_token_mean": metrics.get("correct_margin_per_token_mean"),
                        "serialization_diagnostic": serialization_diagnostic,
                        "greedy_answer": greedy_answer if isinstance(greedy_answer, bool) else None,
                        "greedy_accuracy": greedy_accuracy,
                        "greedy_abstain": bool(parsed.get("abstain")) if isinstance(parsed, dict) else None,
                        "greedy_parse_failure": greedy_parse_failure,
                    }
                )

    gates_out.append(
        _gate(
            "worker_rows_present",
            "pass" if not missing_rows else "fail",
            blocking=True,
            detail={"missing_count": len(missing_rows), "examples": missing_rows[:20]},
        )
    )
    gates_out.append(
        _gate(
            "worker_row_duplicates",
            "pass" if not duplicate_keys else "fail",
            blocking=True,
            detail={"duplicate_count": len(duplicate_keys), "examples": duplicate_keys[:20]},
        )
    )
    gates_out.append(
        _gate(
            "teacher_forced_completeness",
            "pass" if not teacher_forced_failures else "fail",
            blocking=True,
            detail={"failure_count": len(teacher_forced_failures), "examples": teacher_forced_failures[:20]},
        )
    )
    gates_out.append(
        _gate(
            "finite_teacher_forced_scores",
            "pass" if not nonfinite_rows else "fail",
            blocking=True,
            detail={"failure_count": len(nonfinite_rows), "examples": nonfinite_rows[:20]},
        )
    )
    gates_out.append(
        _gate(
            "pinned_model_revisions",
            "pass" if not revision_failures else "fail",
            blocking=True,
            detail={"failure_count": len(revision_failures), "examples": revision_failures[:20]},
        )
    )
    gates_out.append(
        _gate(
            "worker_row_provenance",
            "pass" if not provenance_failures else "fail",
            blocking=True,
            detail={"failure_count": len(provenance_failures), "examples": provenance_failures[:20]},
        )
    )

    # Build paired rows + three-arm validation over joined worker rows.
    paired_rows: list[dict[str, Any]] = []
    triples_by_model: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for row in joined_rows:
        triples_by_model.setdefault(str(row["model_key"]), {}).setdefault(
            str(row["source_order_stripped_request_id"]),
            {},
        )[str(row["arm"])] = row
    triple_alignment_failures: list[dict[str, Any]] = []
    for model_key, by_source in triples_by_model.items():
        for source_id, per_arm in by_source.items():
            if set(per_arm) != set(ARM_ORDER):
                triple_alignment_failures.append(
                    {
                        "model_key": model_key,
                        "source_order_stripped_request_id": source_id,
                        "reason": "missing_arm",
                        "arms_present": sorted(per_arm),
                    }
                )
                continue
            base = per_arm[ARM_PCS]
            for arm in (ARM_MPP, ARM_VP):
                other = per_arm[arm]
                for field in (
                    "expected_answer",
                    "participant",
                    "source_video_id",
                    "query_type",
                    "public_question",
                    "method_frame_count",
                ):
                    if base.get(field) != other.get(field):
                        triple_alignment_failures.append(
                            {
                                "model_key": model_key,
                                **_alignment_mismatch(
                                    source_id=source_id,
                                    field=field,
                                    arm_a=ARM_PCS,
                                    arm_b=arm,
                                    value_a=base.get(field),
                                    value_b=other.get(field),
                                ),
                            }
                        )
            paired_rows.append(
                {
                    "model_key": model_key,
                    "source_order_stripped_request_id": source_id,
                    "participant": base["participant"],
                    "expected_answer": base["expected_answer"],
                    ARM_PCS: per_arm[ARM_PCS],
                    ARM_MPP: per_arm[ARM_MPP],
                    ARM_VP: per_arm[ARM_VP],
                }
            )
    gates_out.append(
        _gate(
            "three_arm_worker_alignment",
            "pass" if not triple_alignment_failures else "fail",
            blocking=True,
            detail={"failure_count": len(triple_alignment_failures), "examples": triple_alignment_failures[:20]},
        )
    )

    configured_primary_models = list(prespec.get("model_plan", {}).get("primary_model_keys", []))
    configured_confirmatory_contrasts = list(
        prespec.get("multiplicity", {}).get("confirmatory_contrast_keys", [])
    )
    expected_family_size = int(prespec.get("multiplicity", {}).get("family_size") or 0)
    available_contrast_keys = {
        str(contrast.get("key"))
        for contrast in prespec.get("contrasts", [])
        if isinstance(contrast, dict) and _contrast_arms(contrast) is not None
    }
    configured_family = {
        f"{model_key}::{contrast_key}"
        for model_key in configured_primary_models
        for contrast_key in configured_confirmatory_contrasts
        if model_key in target_models and contrast_key in available_contrast_keys
    }
    gates_out.append(
        _gate(
            "confirmatory_family_shape_pre_inference",
            "pass" if len(configured_family) == expected_family_size else "fail",
            blocking=True,
            detail={
                "expected_family_size": expected_family_size,
                "observed_family_size": len(configured_family),
                "tests": sorted(configured_family),
            },
        )
    )
    pre_inference_blocking_failures = [
        gate for gate in gates_out if gate["blocking"] and gate["status"] != "pass"
    ]
    if pre_inference_blocking_failures:
        return _write_withheld_aggregate(
            run_dir=run_dir,
            w0_dir=w0_dir,
            prespec=prespec,
            prespec_path=prespec_path,
            model_plan_path=model_plan_path,
            selection_manifest_path=selection_manifest_path,
            mode=mode,
            target_models=target_models,
            selected_source_ids=prelabel["selected_source_ids"],
            validation_rows=prelabel["validation_rows"],
            gates_out=gates_out,
            inference_status="inference_withheld_due_to_blocking_gates",
        )

    contrasts = [
        contrast
        for contrast in prespec.get("contrasts", [])
        if isinstance(contrast, dict) and _contrast_arms(contrast) is not None
    ]
    contrast_summary: list[dict[str, Any]] = []
    for model_key in target_models:
        model_pairs = [row for row in paired_rows if row["model_key"] == model_key]
        participant_order = sorted({str(row["participant"]) for row in model_pairs})
        draws = int(prespec.get("bootstrap", {}).get("draws") or 5000)
        seed = int(prespec.get("bootstrap", {}).get("seed") or 20260713)
        schedule = (
            _build_bootstrap_schedule(participant_order, draws=draws, seed=seed, context=(model_key, "job2c_behavioral"))
            if mode == "full"
            else []
        )
        for contrast in contrasts:
            pair = _contrast_arms(contrast)
            if pair is None:
                continue
            arm_a, arm_b = pair
            rows_for_contrast = []
            by_participant: dict[str, list[float]] = {}
            by_participant_per_token: dict[str, list[float]] = {}
            by_participant_first_divergence: dict[str, list[float]] = {}
            by_participant_divergent_span: dict[str, list[float]] = {}
            for row in model_pairs:
                a = row[arm_a]
                b = row[arm_b]
                delta_ce = float(a["ce_bits"] - b["ce_bits"])
                participant = str(row["participant"])
                by_participant.setdefault(participant, []).append(delta_ce)
                by_participant_per_token.setdefault(participant, []).append(
                    float(a["ce_bits_per_token_mean"] - b["ce_bits_per_token_mean"])
                )
                diag_a = a.get("serialization_diagnostic", {})
                diag_b = b.get("serialization_diagnostic", {})
                first_a = diag_a.get("first_divergent_correct_minus_incorrect_logprob")
                first_b = diag_b.get("first_divergent_correct_minus_incorrect_logprob")
                if first_a is not None and first_b is not None:
                    by_participant_first_divergence.setdefault(participant, []).append(
                        float(first_a - first_b)
                    )
                span_a = diag_a.get("divergent_span_correct_minus_incorrect_logprob")
                span_b = diag_b.get("divergent_span_correct_minus_incorrect_logprob")
                if span_a is not None and span_b is not None:
                    by_participant_divergent_span.setdefault(participant, []).append(
                        float(span_a - span_b)
                    )
                rows_for_contrast.append((a, b))
            participant_mean = {
                participant: float(np.mean(values))
                for participant, values in by_participant.items()
            }
            participant_mean_per_token = {
                participant: float(np.mean(values))
                for participant, values in by_participant_per_token.items()
            }
            participant_mean_first_divergence = {
                participant: float(np.mean(values))
                for participant, values in by_participant_first_divergence.items()
            }
            participant_mean_divergent_span = {
                participant: float(np.mean(values))
                for participant, values in by_participant_divergent_span.items()
            }
            point = float(np.mean(list(participant_mean.values()))) if participant_mean else float("nan")
            if mode == "full":
                boot = participant_cluster_bootstrap_mean_delta(participant_mean, schedule=schedule)
                boot_per_token = participant_cluster_bootstrap_mean_delta(
                    participant_mean_per_token,
                    schedule=schedule,
                )
                sign_flip = paired_stats.exact_participant_sign_flip_test(participant_mean)
                p_two_sided = float(sign_flip.get("p_value_two_sided"))
            else:
                boot = {
                    "point_estimate": point,
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                    "n_boot_requested": 0,
                    "n_boot_valid": 0,
                    "method": "skipped_canary_mode",
                }
                sign_flip = {
                    "n_participants": len(participant_mean),
                    "observed_mean_delta_ce_bits": point,
                    "p_value_two_sided": float("nan"),
                    "n_assignments": 0,
                    "method": "skipped_canary_mode",
                }
                p_two_sided = float("nan")
                boot_per_token = {
                    "point_estimate": float("nan"),
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                    "n_boot_requested": 0,
                    "n_boot_valid": 0,
                }

            def _avg(values: list[float | None]) -> float | None:
                clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
                return float(np.mean(clean)) if clean else None

            score_acc_a = _avg([float(pair_a["score_choice_accuracy"]) for pair_a, _ in rows_for_contrast])
            score_acc_b = _avg([float(pair_b["score_choice_accuracy"]) for _, pair_b in rows_for_contrast])
            margin_a = _avg([float(pair_a["correct_margin"]) for pair_a, _ in rows_for_contrast])
            margin_b = _avg([float(pair_b["correct_margin"]) for _, pair_b in rows_for_contrast])
            greedy_acc_a = _avg(
                [
                    float(pair_a["greedy_accuracy"])
                    for pair_a, _ in rows_for_contrast
                    if isinstance(pair_a.get("greedy_accuracy"), bool)
                ]
            )
            greedy_acc_b = _avg(
                [
                    float(pair_b["greedy_accuracy"])
                    for _, pair_b in rows_for_contrast
                    if isinstance(pair_b.get("greedy_accuracy"), bool)
                ]
            )
            abst_a = _avg(
                [
                    float(bool(pair_a.get("greedy_abstain")))
                    for pair_a, _ in rows_for_contrast
                    if pair_a.get("greedy_abstain") is not None
                ]
            )
            abst_b = _avg(
                [
                    float(bool(pair_b.get("greedy_abstain")))
                    for _, pair_b in rows_for_contrast
                    if pair_b.get("greedy_abstain") is not None
                ]
            )
            parse_fail_a = _avg([float(bool(pair_a.get("greedy_parse_failure"))) for pair_a, _ in rows_for_contrast])
            parse_fail_b = _avg([float(bool(pair_b.get("greedy_parse_failure"))) for _, pair_b in rows_for_contrast])
            contrast_summary.append(
                {
                    "model_key": model_key,
                    "contrast_key": str(contrast.get("key")),
                    "contrast_label": str(contrast.get("label")),
                    "confirmatory": bool(contrast.get("confirmatory")),
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "n_aligned_rows": len(rows_for_contrast),
                    "n_participants": len(participant_mean),
                    "participant_equal_mean_delta_ce_bits": point,
                    "bootstrap_ci95_low_ce_bits": float(boot["ci95_low"]),
                    "bootstrap_ci95_high_ce_bits": float(boot["ci95_high"]),
                    "bootstrap_n_requested": int(boot["n_boot_requested"]),
                    "bootstrap_n_valid": int(boot["n_boot_valid"]),
                    "participant_equal_mean_delta_ce_bits_per_token_mean": float(
                        boot_per_token["point_estimate"]
                    ),
                    "bootstrap_ci95_low_ce_bits_per_token_mean": float(boot_per_token["ci95_low"]),
                    "bootstrap_ci95_high_ce_bits_per_token_mean": float(boot_per_token["ci95_high"]),
                    "bootstrap_n_requested_per_token_mean": int(boot_per_token["n_boot_requested"]),
                    "bootstrap_n_valid_per_token_mean": int(boot_per_token["n_boot_valid"]),
                    "participant_equal_mean_delta_correct_minus_incorrect_first_divergent_logprob": (
                        float(np.mean(list(participant_mean_first_divergence.values())))
                        if participant_mean_first_divergence
                        else None
                    ),
                    "participant_equal_mean_delta_correct_minus_incorrect_divergent_span_logprob": (
                        float(np.mean(list(participant_mean_divergent_span.values())))
                        if participant_mean_divergent_span
                        else None
                    ),
                    "serialization_diagnostics_outside_holm": True,
                    "exact_sign_flip_p_two_sided_ce_bits": p_two_sided,
                    "exact_sign_flip_n_assignments": int(sign_flip.get("n_assignments") or 0),
                    "score_choice_accuracy_a": score_acc_a,
                    "score_choice_accuracy_b": score_acc_b,
                    "delta_score_choice_accuracy": (
                        None if score_acc_a is None or score_acc_b is None else float(score_acc_a - score_acc_b)
                    ),
                    "correct_margin_a": margin_a,
                    "correct_margin_b": margin_b,
                    "delta_correct_margin": None
                    if margin_a is None or margin_b is None
                    else float(margin_a - margin_b),
                    "greedy_accuracy_a": greedy_acc_a,
                    "greedy_accuracy_b": greedy_acc_b,
                    "delta_greedy_accuracy": None
                    if greedy_acc_a is None or greedy_acc_b is None
                    else float(greedy_acc_a - greedy_acc_b),
                    "greedy_abstention_rate_a": abst_a,
                    "greedy_abstention_rate_b": abst_b,
                    "delta_greedy_abstention_rate": None
                    if abst_a is None or abst_b is None
                    else float(abst_a - abst_b),
                    "greedy_parse_failure_rate_a": parse_fail_a,
                    "greedy_parse_failure_rate_b": parse_fail_b,
                    "delta_greedy_parse_failure_rate": None
                    if parse_fail_a is None or parse_fail_b is None
                    else float(parse_fail_a - parse_fail_b),
                    "holm_family_member": False,
                    "holm_adjusted_p_confirmatory_family": float("nan"),
                }
            )

    confirmatory_family = {
        "status": "skipped_canary_mode" if mode == "canary" else "pending",
        "method": "holm_over_exactly_four_primary_tests",
    }
    if mode == "full":
        primary_models = list(prespec.get("model_plan", {}).get("primary_model_keys", []))
        confirmatory_contrasts = list(prespec.get("multiplicity", {}).get("confirmatory_contrast_keys", []))
        p_values: dict[str, float] = {}
        for row in contrast_summary:
            if row["model_key"] in primary_models and row["contrast_key"] in confirmatory_contrasts:
                test_id = f"{row['model_key']}::{row['contrast_key']}"
                p_values[test_id] = float(row["exact_sign_flip_p_two_sided_ce_bits"])
        expected_family_size = int(prespec.get("multiplicity", {}).get("family_size") or 0)
        if len(p_values) != expected_family_size:
            confirmatory_family = {
                "status": "fail",
                "reason": "confirmatory_family_size_mismatch",
                "observed_family_size": len(p_values),
                "expected_family_size": expected_family_size,
                "tests": sorted(p_values),
            }
        else:
            adjusted = holm_adjust(p_values)
            for row in contrast_summary:
                test_id = f"{row['model_key']}::{row['contrast_key']}"
                if test_id in adjusted:
                    row["holm_family_member"] = True
                    row["holm_adjusted_p_confirmatory_family"] = float(adjusted[test_id])
            confirmatory_family = {
                "status": "pass",
                "expected_family_size": expected_family_size,
                "observed_family_size": len(p_values),
                "tests": sorted(p_values),
                "adjusted_p_values": {key: float(value) for key, value in adjusted.items()},
                "method": "holm_over_exactly_four_primary_tests",
            }
    gates_out.append(
        _gate(
            "confirmatory_holm_family",
            "pass" if confirmatory_family["status"] in {"pass", "skipped_canary_mode"} else "fail",
            blocking=(mode == "full"),
            detail=confirmatory_family,
        )
    )

    parse_fail_count = sum(1 for row in joined_rows if bool(row.get("greedy_parse_failure")))
    parse_fail_rate = float(parse_fail_count / max(len(joined_rows), 1))
    gates_out.append(
        _gate(
            "greedy_parse_failure_claim_gate",
            "pass" if parse_fail_count == 0 else "fail",
            blocking=False,
            detail={
                "parse_failure_count": parse_fail_count,
                "row_count": len(joined_rows),
                "parse_failure_rate": parse_fail_rate,
                "claim": "Complete greedy-accuracy claim is blocked when parse failures are nonzero",
            },
        )
    )

    blocking_failures = [gate for gate in gates_out if gate["blocking"] and gate["status"] != "pass"]
    overall_status = "pass" if not blocking_failures else "fail"

    if blocking_failures:
        return _write_withheld_aggregate(
            run_dir=run_dir,
            w0_dir=w0_dir,
            prespec=prespec,
            prespec_path=prespec_path,
            model_plan_path=model_plan_path,
            selection_manifest_path=selection_manifest_path,
            mode=mode,
            target_models=target_models,
            selected_source_ids=prelabel["selected_source_ids"],
            validation_rows=prelabel["validation_rows"],
            gates_out=gates_out,
            inference_status="inference_withheld_due_to_blocking_gates",
        )

    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    failure_report_path = aggregate_dir / "failure_report.json"
    if failure_report_path.exists():
        failure_report_path.unlink()
    _write_jsonl_atomic(aggregate_dir / "joined_rows.jsonl", joined_rows)
    _write_jsonl_atomic(aggregate_dir / "paired_rows.jsonl", paired_rows)
    _write_json(aggregate_dir / "contrast_summary.json", {"rows": contrast_summary})
    _write_json(aggregate_dir / "gate_results.json", {"gates": gates_out})

    summary = {
        "kind": "phase2_job2c_behavioral_aggregate_summary",
        "mode": mode,
        "overall_status": overall_status,
        "inference_status": "complete",
        "run_dir": str(run_dir),
        "w0_dir": str(w0_dir),
        "prespec_path": str(prespec_path),
        "model_plan_path": str(model_plan_path),
        "selection_manifest_path": str(selection_manifest_path),
        "target_models": target_models,
        "selected_row_count": len(selected_source_ids),
        "selected_participant_count": len(participant_ids),
        "joined_row_count": len(joined_rows),
        "paired_row_count": len(paired_rows),
        "prespec_sha256": _prespec_hash(prespec_path),
        "model_plan_sha256": _model_plan_hash(model_plan_path),
        "gates": gates_out,
        "blocking_failure_count": len(blocking_failures),
        "confirmatory_inference": confirmatory_family,
        "contrast_summary": contrast_summary,
        "sign_convention": str(prespec.get("sign_convention") or ""),
        "claim_boundary": str(prespec.get("claim_boundary") or ""),
        "generated_at": _utc_now_iso(),
    }
    _write_json(aggregate_dir / "run_manifest.json", summary)
    report_text = _render_markdown_report(summary)
    _atomic_write_text(aggregate_dir / "report.md", report_text)
    _write_json(run_dir / "aggregate_summary.json", summary)

    if blocking_failures:
        _write_json(
            aggregate_dir / "failure_report.json",
            {
                "status": "fail",
                "blocking_failures": blocking_failures,
                "summary_path": str(run_dir / "aggregate_summary.json"),
            },
        )
    return summary
