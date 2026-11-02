#!/usr/bin/env python3
"""Burned-data EgoHOS construction pilot item builder.

CPU-only construction tool. It emits method-facing public request packs/assets,
hidden evaluator records, pending-negative review queue artifacts, and strict
machine-readable audits for burned-data construction.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from research.lifecycle_memory import phase2_job3_egohos_census as census
from research.lifecycle_memory import phase2_job3_marker_renderer as marker_renderer
from research.lifecycle_memory import phase2_job3_negative_review_export as review_validator
from research.lifecycle_memory import (
    phase2_job3_negative_replacement_review_export as cumulative_review_validator,
)


SOURCE_FAMILY_LOCK = "epic"
PROVENANCE_STATUS_LOCK = "excluded_epic"
CLASS_IDS = (3, 4, 5, 6, 7, 8)
PILOT_POSITIVE_CLASSES = frozenset({3, 4, 5})
PENDING_NEGATIVE_CLASSES = frozenset({6, 7, 8})
DEFAULT_SELECTION_TARGET = 120
DEFAULT_PER_CLASS_TARGET = 20
DEFAULT_MIN_COMPONENT_AREA = 64
REFERENCE_CROP_EROSION_RADIUS = 1
REFERENCE_CROP_RETAINED_AREA_FLOOR = 64
REFERENCE_CROP_MARGIN = 4
DEFAULT_PAGE_SIZE = 12
DEFAULT_CANDIDATE_SIDECAR_REL = Path(
    "derived/job3_stage_a2_census_2026_07_14/hidden/egohos_candidate_sidecar.jsonl"
)

ARM_PCS_E = "pcs_e"
ARM_MPP_E = "mpp_e"
ARM_VP_E = "vp_e"
ARM_ALT_MARKED = "mpp_e_alternate_descriptive_nonconfirmatory"
ARM_ORDER = (ARM_PCS_E, ARM_MPP_E, ARM_VP_E, ARM_ALT_MARKED)

SYSTEM_PROMPT = (
    "Answer the binary current-contact question using only the provided RGB inputs. "
    "Return exactly one JSON object with keys: answer, abstain, abstention_reason, confidence."
)
USER_PROMPT_BASE = (
    "Input 1 is a full frame. Input 2 is an unmarked target-only reference crop.\n"
    "Question: Is a hand currently in direct contact with the referenced target object?"
)
VP_FROZEN_INSTRUCTION_PACKAGE = (
    "Frozen outline instruction package:\n"
    "- Treat colored overlays as localization hints only.\n"
    "- Green marks indicate target localization.\n"
    "- Blue marks indicate visible hands.\n"
    "- Do not infer answer labels from overlay presence alone."
)
CANONICAL_TRUE_JSON = (
    '{"answer":true,"abstain":false,"abstention_reason":null,"confidence":1.0}'
)
CANONICAL_FALSE_JSON = (
    '{"answer":false,"abstain":false,"abstention_reason":null,"confidence":1.0}'
)
NEGATIVE_APPROVAL_DECISION = "approve_no_direct_hand_contact"
NEGATIVE_REJECTION_DECISIONS = frozenset(
    {
        "reject_direct_or_ambiguous_contact",
        "reject_target_not_visually_identifiable",
        "reject_render_or_crop_failure",
    }
)

CLAIM_BOUNDARY = (
    "Stage D burned-data construction pilot for Phase 2 Job3 EgoHOS. "
    "All outputs are burned, construction_pilot, non-confirmatory, rows_frozen=false."
)

FORBIDDEN_PUBLIC_KEYWORDS = (
    "source_video_id",
    "source_subject_id",
    "source_frame_or_timestamp",
    "image_relpath",
    "label_relpath",
    "mask_class",
    "class_id",
    "canonical_answer_json",
    "contact_role",
    "label_role",
    "scoring_eligible",
    "candidate_id",
    "min_target_hand_pixel_distance",
    "target_hand_8_neighbor_adjacent",
    "raw/",
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    split: str
    frame_id: str
    source_video_id: str
    source_subject_id: str
    source_frame_or_timestamp: str
    image_relpath: str
    label_relpath: str
    class_id: int
    target_area: int
    reference_retained_area: int
    target_component_mask: np.ndarray
    hand_mask: np.ndarray
    source_frame_key: str
    subject_key: str
    video_key: str
    sidecar_label_role: str | None
    hand_overlap_px: int
    hand_adjacent_px: int
    min_target_hand_pixel_distance_euclidean: float
    min_target_hand_pixel_distance_chebyshev: int
    target_hand_8_neighbor_adjacent: bool


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque_id(namespace: str, raw_value: str) -> str:
    digest = hashlib.sha256(f"phase2_job3_burned|{namespace}|{raw_value}".encode("utf-8")).hexdigest()
    return f"{namespace}_{digest[:24]}"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(path, body)


def _write_tsv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    temporary.replace(path)


def _png_bytes(image_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    image = Image.fromarray(image_rgb, mode="RGB")
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _save_png_atomic(path: Path, image_rgb: np.ndarray) -> bytes:
    data = _png_bytes(image_rgb)
    _atomic_write_bytes(path, data)
    return data


def _prepare_fresh_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"out-dir exists but is not a directory: {resolved}")
        first_entry = next(resolved.iterdir(), None)
        if first_entry is not None:
            raise ValueError(
                "out-dir must be absent or empty so prior review-round artifacts "
                f"cannot survive a rebuild: {resolved} (found {first_entry.name})"
            )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_optional_sidecar(path: Path | None) -> dict[tuple[str, str, int], dict[str, Any]]:
    if path is None:
        return {}
    rows = _read_jsonl(path.expanduser())
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        split = str(row.get("split") or "")
        frame_id = str(row.get("frame_id") or "")
        class_id = row.get("mask_class")
        if not isinstance(class_id, int):
            continue
        key = (split, frame_id, class_id)
        if key in indexed:
            raise ValueError(f"duplicate candidate sidecar key: {key}")
        indexed[key] = row
    return indexed


def _load_negative_audit(
    path: Path | None,
    *,
    data: bytes | None = None,
) -> dict[str, dict[str, str]]:
    if path is None:
        if data is not None:
            raise ValueError("negative audit bytes require a source path")
        return {}
    path = path.expanduser()
    if data is None:
        data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("negative audit sidecar is not valid UTF-8") from exc
    if path.suffix.lower() == ".tsv":
        rows = [
            dict(row)
            for row in csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
        ]
    else:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"negative audit sidecar line {line_number} is not valid JSON"
                ) from exc
            if isinstance(payload, dict):
                rows.append(payload)
    decisions: dict[str, dict[str, str]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        decision = str(row.get("decision") or "").strip()
        if not candidate_id and not decision:
            continue
        if not candidate_id:
            raise ValueError("negative audit row is missing candidate_id")
        if decision not in {NEGATIVE_APPROVAL_DECISION, *NEGATIVE_REJECTION_DECISIONS}:
            raise ValueError(
                f"negative audit row {candidate_id} has invalid decision {decision!r}"
            )
        reason = str(row.get("reason") or "").strip()
        reviewer_id = str(row.get("reviewer_id") or "").strip()
        reviewed_at = str(row.get("reviewed_at") or "").strip()
        review_asset_sha256 = str(row.get("review_asset_sha256") or "").strip()
        if not reason or not reviewer_id or not reviewed_at or not re.fullmatch(
            r"[0-9a-f]{64}", review_asset_sha256
        ):
            raise ValueError(
                f"negative audit row {candidate_id} requires reason, reviewer_id, "
                "reviewed_at, and a lowercase SHA-256 review_asset_sha256"
            )
        if candidate_id in decisions:
            raise ValueError(f"duplicate negative audit candidate_id: {candidate_id}")
        decisions[candidate_id] = {
            "decision": decision,
            "reason": reason,
            "reviewer_id": reviewer_id,
            "reviewed_at": reviewed_at,
            "review_asset_sha256": review_asset_sha256,
        }
    return decisions


def _verify_receipt_file_record(
    record: Any,
    *,
    label: str,
    current_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"negative audit receipt input {label} must be an object")
    recorded_path = record.get("path")
    expected_sha256 = record.get("sha256")
    if not isinstance(recorded_path, str) or not recorded_path.strip():
        raise ValueError(f"negative audit receipt input {label} has no path")
    if not isinstance(expected_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ) is None:
        raise ValueError(
            f"negative audit receipt input {label} has an invalid SHA-256"
        )
    path = (
        current_path.expanduser().resolve()
        if current_path is not None
        else Path(recorded_path).expanduser().resolve()
    )
    if not path.is_file():
        raise ValueError(f"negative audit receipt input {label} is missing: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"negative audit receipt input {label} hash mismatch: {path}"
        )
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and expected_bytes != path.stat().st_size:
        raise ValueError(
            f"negative audit receipt input {label} byte count mismatch: {path}"
        )
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "bytes": path.stat().st_size,
    }


def _infer_cumulative_repo_root(
    *,
    replacement_manifest_path: Path,
    replacement_schema_path: Path,
) -> Path:
    try:
        manifest = json.loads(replacement_manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "cumulative replacement manifest is not valid JSON"
        ) from exc
    bindings = manifest.get("bindings") if isinstance(manifest, dict) else None
    schema_binding = bindings.get("schema") if isinstance(bindings, dict) else None
    relative_value = (
        schema_binding.get("path") if isinstance(schema_binding, dict) else None
    )
    if not isinstance(relative_value, str) or not relative_value.strip():
        raise ValueError("cumulative replacement manifest has no schema path binding")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "cumulative replacement manifest schema path must be repository-relative"
        )
    schema_path = replacement_schema_path.expanduser().resolve()
    repo_root = schema_path
    for _ in relative.parts:
        repo_root = repo_root.parent
    if (repo_root / relative).resolve() != schema_path:
        raise ValueError(
            "cumulative replacement schema path cannot identify its repository root"
        )
    return repo_root


def _without_cumulative_issuance_fields(payload: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(payload))
    copied.pop("validated_at_utc", None)
    copied.pop("inputs", None)
    copied.pop("receipt_filename", None)
    return copied


def _verify_cumulative_negative_audit_receipt(
    *,
    sidecar_path: Path,
    receipt_path: Path,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if receipt.get("version") != cumulative_review_validator.VALIDATION_VERSION:
        raise ValueError("cumulative negative audit receipt version is invalid")
    if receipt.get("status") != "pass" or receipt.get("rows_frozen") is not False:
        raise ValueError("cumulative negative audit receipt is not a pass")
    candidate_count = receipt.get("candidate_count")
    candidate_order_sha256 = receipt.get("candidate_order_sha256")
    decision_counts = receipt.get("decision_counts")
    if (
        not isinstance(candidate_count, int)
        or isinstance(candidate_count, bool)
        or candidate_count <= 0
        or not isinstance(candidate_order_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", candidate_order_sha256) is None
        or not isinstance(decision_counts, dict)
        or any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in decision_counts.items()
        )
        or sum(decision_counts.values()) != candidate_count
    ):
        raise ValueError("cumulative negative audit receipt summary is invalid")

    top_inputs = receipt.get("inputs")
    if not isinstance(top_inputs, dict) or set(top_inputs) != {"cumulative_export"}:
        raise ValueError("cumulative negative audit receipt inputs are incomplete")
    cumulative_input = top_inputs["cumulative_export"]
    if not isinstance(cumulative_input, dict):
        raise ValueError("cumulative negative audit export input is invalid")
    verified_export = _verify_receipt_file_record(
        cumulative_input,
        label="cumulative_export",
        current_path=sidecar_path,
    )
    cumulative_sha256 = verified_export["sha256"]
    expected_export_name = (
        f"{cumulative_review_validator.CUMULATIVE_PREFIX}"
        f"{cumulative_sha256}.jsonl"
    )
    if (
        sidecar_path.expanduser().resolve().name != expected_export_name
        or cumulative_input.get("filename") != expected_export_name
    ):
        raise ValueError("cumulative negative audit sidecar filename is invalid")
    expected_receipt_name = (
        f"{cumulative_review_validator.RECEIPT_PREFIX}"
        f"{cumulative_sha256}.json"
    )
    if (
        receipt.get("receipt_filename") != expected_receipt_name
        or receipt_path.name != expected_receipt_name
    ):
        raise ValueError("cumulative negative audit receipt is not content-addressed")

    rounds = receipt.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 2:
        raise ValueError("cumulative negative audit receipt rounds are invalid")
    expected_input_keys = {
        "parent": {
            "export",
            "validation_receipt",
            "queue",
            "schema",
            "manifest",
            "generator",
            "tool",
            "images",
        },
        "replacement": {
            "export",
            "queue",
            "schema",
            "manifest",
            "generator",
            "base_generator",
            "tool",
            "parent_validation_receipt",
            "images",
            "source_construction_manifest",
            "construction_audit",
            "replacement_ledger",
            "human_rejections",
        },
    }
    verified_rounds: list[dict[str, Any]] = []
    verified_images: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    aggregate_decisions: Counter[str] = Counter()
    aggregate_classes: Counter[str] = Counter()
    aggregate_reviewers: set[str] = set()
    round_candidate_count = 0
    for round_index, (expected_round, summary) in enumerate(
        zip(("parent", "replacement"), rounds)
    ):
        if not isinstance(summary, dict) or summary.get("round") != expected_round:
            raise ValueError("cumulative negative audit round order is invalid")
        count = summary.get("candidate_count")
        round_order_sha256 = summary.get("candidate_order_sha256")
        round_decisions = summary.get("decision_counts")
        class_counts = summary.get("class_counts")
        reviewers = summary.get("reviewer_ids")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or not isinstance(round_order_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", round_order_sha256) is None
            or not isinstance(round_decisions, dict)
            or any(
                not isinstance(key, str)
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for key, value in round_decisions.items()
            )
            or sum(round_decisions.values()) != count
            or not isinstance(class_counts, dict)
            or any(
                not isinstance(key, str)
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for key, value in class_counts.items()
            )
            or sum(class_counts.values()) != count
            or not isinstance(reviewers, list)
            or len(reviewers) != 1
            or not isinstance(reviewers[0], str)
            or not reviewers[0].strip()
        ):
            raise ValueError(
                f"cumulative negative audit {expected_round} summary is invalid"
            )
        inputs = summary.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != expected_input_keys[expected_round]:
            raise ValueError(
                f"cumulative negative audit {expected_round} inputs are incomplete"
            )
        verified_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            records = value if key == "images" else [value]
            if not isinstance(records, list) or not records:
                raise ValueError(
                    f"cumulative negative audit {expected_round} {key} is invalid"
                )
            verified_records: list[dict[str, Any]] = []
            for record_index, record in enumerate(records):
                verified = _verify_receipt_file_record(
                    record,
                    label=f"{expected_round}.{key}[{record_index}]",
                )
                if key == "images":
                    candidate_id = (
                        record.get("candidate_id")
                        if isinstance(record, dict)
                        else None
                    )
                    if (
                        not isinstance(candidate_id, str)
                        or not candidate_id.strip()
                        or candidate_id in seen_candidates
                    ):
                        raise ValueError(
                            "cumulative negative audit image candidate is invalid or duplicated"
                        )
                    seen_candidates.add(candidate_id)
                    verified["candidate_id"] = candidate_id
                    verified_images.append(verified)
                verified_records.append(verified)
            verified_inputs[key] = (
                verified_records if key == "images" else verified_records[0]
            )
        round_images = verified_inputs["images"]
        if len(round_images) != count:
            raise ValueError(
                f"cumulative negative audit {expected_round} image count is invalid"
            )
        image_order_sha256 = _sha256_text(
            "\n".join(row["candidate_id"] for row in round_images) + "\n"
        )
        if image_order_sha256 != round_order_sha256:
            raise ValueError(
                f"cumulative negative audit {expected_round} image order is invalid"
            )
        verified_summary = dict(summary)
        verified_summary["inputs"] = verified_inputs
        verified_rounds.append(verified_summary)
        round_candidate_count += count
        aggregate_decisions.update(round_decisions)
        aggregate_classes.update(class_counts)
        aggregate_reviewers.update(reviewers)

    if (
        round_candidate_count != candidate_count
        or dict(sorted(aggregate_decisions.items()))
        != dict(sorted(decision_counts.items()))
        or receipt.get("class_counts") != dict(sorted(aggregate_classes.items()))
        or receipt.get("reviewer_ids") != sorted(aggregate_reviewers)
        or _sha256_text(
            "\n".join(row["candidate_id"] for row in verified_images) + "\n"
        )
        != candidate_order_sha256
    ):
        raise ValueError("cumulative negative audit round summaries do not aggregate")

    parent_inputs = verified_rounds[0]["inputs"]
    replacement_inputs = verified_rounds[1]["inputs"]
    replacement_manifest_path = Path(replacement_inputs["manifest"]["path"])
    replacement_schema_path = Path(replacement_inputs["schema"]["path"])
    repo_root = _infer_cumulative_repo_root(
        replacement_manifest_path=replacement_manifest_path,
        replacement_schema_path=replacement_schema_path,
    )
    try:
        fresh = cumulative_review_validator.validate_replacement_review_export(
            parent_export_path=Path(parent_inputs["export"]["path"]),
            parent_receipt_path=Path(parent_inputs["validation_receipt"]["path"]),
            replacement_export_path=Path(replacement_inputs["export"]["path"]),
            replacement_queue_path=Path(replacement_inputs["queue"]["path"]),
            replacement_schema_path=replacement_schema_path,
            replacement_manifest_path=replacement_manifest_path,
            repo_root=repo_root,
        )
    except (
        KeyError,
        TypeError,
        OSError,
        cumulative_review_validator.ReplacementReviewExportError,
    ) as exc:
        raise ValueError(f"fresh cumulative validation failed: {exc}") from exc
    if fresh.cumulative_bytes != sidecar_path.read_bytes():
        raise ValueError(
            "cumulative negative audit sidecar differs from fresh validation"
        )
    if _without_cumulative_issuance_fields(receipt) != (
        _without_cumulative_issuance_fields(fresh.payload_without_cumulative)
    ):
        raise ValueError(
            "cumulative negative audit receipt differs from fresh validation"
        )

    return {
        "kind": receipt["kind"],
        "version": receipt["version"],
        "status": receipt["status"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "receipt_filename": expected_receipt_name,
        "candidate_count": candidate_count,
        "candidate_order_sha256": candidate_order_sha256,
        "class_counts": dict(sorted(aggregate_classes.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "reviewer_ids": sorted(aggregate_reviewers),
        "rounds": verified_rounds,
        "inputs": {
            "export": verified_export,
            "images": verified_images,
        },
    }


def _verify_negative_audit_receipt(
    *,
    sidecar_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    if not receipt_path.is_file():
        raise ValueError(
            f"negative audit validation receipt is missing: {receipt_path}"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("negative audit validation receipt is not valid JSON") from exc
    if not isinstance(receipt, dict):
        raise ValueError("negative audit validation receipt must be a JSON object")
    if receipt.get("kind") == cumulative_review_validator.VALIDATION_KIND:
        return _verify_cumulative_negative_audit_receipt(
            sidecar_path=sidecar_path,
            receipt_path=receipt_path,
            receipt=receipt,
        )
    if receipt.get("kind") != review_validator.VALIDATION_KIND:
        raise ValueError("negative audit validation receipt kind is invalid")
    if receipt.get("version") != review_validator.VALIDATION_VERSION:
        raise ValueError("negative audit validation receipt version is invalid")
    if receipt.get("status") != "pass":
        raise ValueError("negative audit validation receipt status is not pass")
    if receipt.get("rows_frozen") is not False:
        raise ValueError("negative audit validation receipt rows_frozen must be false")
    candidate_count = receipt.get("candidate_count")
    if not isinstance(candidate_count, int) or candidate_count <= 0:
        raise ValueError("negative audit validation receipt candidate_count is invalid")
    candidate_order_sha256 = receipt.get("candidate_order_sha256")
    if not isinstance(candidate_order_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", candidate_order_sha256
    ) is None:
        raise ValueError(
            "negative audit validation receipt candidate_order_sha256 is invalid"
        )
    decision_counts = receipt.get("decision_counts")
    if (
        not isinstance(decision_counts, dict)
        or any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or value < 0
            for key, value in decision_counts.items()
        )
        or sum(decision_counts.values()) != candidate_count
    ):
        raise ValueError(
            "negative audit validation receipt decision_counts is invalid"
        )

    inputs = receipt.get("inputs")
    required_inputs = {
        "export",
        "queue",
        "schema",
        "manifest",
        "generator",
        "tool",
        "images",
    }
    if not isinstance(inputs, dict) or set(inputs) != required_inputs:
        raise ValueError("negative audit validation receipt inputs are incomplete")
    export_input = inputs["export"]
    if not isinstance(export_input, dict):
        raise ValueError("negative audit validation receipt export input is invalid")
    recorded_export_filename = export_input.get("filename")
    expected_browser_filename = export_input.get("expected_browser_filename")
    if (
        not isinstance(recorded_export_filename, str)
        or not recorded_export_filename
        or not isinstance(expected_browser_filename, str)
        or not expected_browser_filename
        or sidecar_path.expanduser().resolve().name != recorded_export_filename
    ):
        raise ValueError(
            "negative audit sidecar filename does not match its validation receipt"
        )
    expected_stem = Path(expected_browser_filename).stem
    filename_pattern = re.compile(
        rf"{re.escape(expected_stem)}(?: \([1-9][0-9]*\))?\.jsonl\Z"
    )
    if filename_pattern.fullmatch(recorded_export_filename) is None:
        raise ValueError(
            "negative audit validation receipt browser filename binding is invalid"
        )
    export_record = _verify_receipt_file_record(
        export_input,
        label="export",
        current_path=sidecar_path,
    )
    expected_receipt_name = (
        f"{review_validator.RECEIPT_PREFIX}{export_record['sha256']}.json"
    )
    if receipt.get("receipt_filename") != expected_receipt_name:
        raise ValueError("negative audit validation receipt is not content-addressed")
    if receipt_path.name != expected_receipt_name:
        raise ValueError("negative audit validation receipt filename is invalid")

    verified_inputs: dict[str, Any] = {"export": export_record}
    for key in ("queue", "schema", "manifest", "generator", "tool"):
        verified_inputs[key] = _verify_receipt_file_record(inputs[key], label=key)
    image_rows = inputs["images"]
    if not isinstance(image_rows, list) or len(image_rows) != candidate_count:
        raise ValueError(
            "negative audit validation receipt image inputs are incomplete"
        )
    verified_images: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for index, row in enumerate(image_rows):
        candidate_id = row.get("candidate_id") if isinstance(row, dict) else None
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or candidate_id in seen_candidates
        ):
            raise ValueError(
                "negative audit validation receipt image "
                f"{index} has an invalid candidate_id"
            )
        seen_candidates.add(candidate_id)
        verified = _verify_receipt_file_record(
            row,
            label=f"image[{candidate_id}]",
        )
        verified["candidate_id"] = candidate_id
        verified_images.append(verified)
    verified_inputs["images"] = verified_images

    return {
        "kind": receipt["kind"],
        "version": receipt["version"],
        "status": receipt["status"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "receipt_filename": expected_receipt_name,
        "candidate_count": candidate_count,
        "candidate_order_sha256": candidate_order_sha256,
        "decision_counts": dict(sorted(decision_counts.items())),
        "inputs": verified_inputs,
    }


def _cross_check_negative_audit_binding(
    *,
    decisions: dict[str, dict[str, str]],
    validation: dict[str, Any],
) -> None:
    if len(decisions) != validation["candidate_count"]:
        raise ValueError(
            "negative audit sidecar row count does not match its validation receipt"
        )
    decision_order = list(decisions)
    decision_order_sha256 = _sha256_text("\n".join(decision_order) + "\n")
    if decision_order_sha256 != validation["candidate_order_sha256"]:
        raise ValueError(
            "negative audit sidecar order does not match its validation receipt"
        )
    receipt_candidate_ids = {
        row["candidate_id"] for row in validation["inputs"]["images"]
    }
    if set(decision_order) != receipt_candidate_ids:
        raise ValueError(
            "negative audit sidecar candidates do not match receipt image bindings"
        )
    decision_counts = Counter(row["decision"] for row in decisions.values())
    if dict(sorted(decision_counts.items())) != validation["decision_counts"]:
        raise ValueError(
            "negative audit sidecar decisions do not match receipt decision counts"
        )


def _verify_and_load_negative_audit(
    *,
    sidecar_path: Path,
    receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    resolved_sidecar = sidecar_path.expanduser().resolve()
    sidecar_bytes = resolved_sidecar.read_bytes()
    validation = _verify_negative_audit_receipt(
        sidecar_path=resolved_sidecar,
        receipt_path=receipt_path,
    )
    export_record = validation.get("inputs", {}).get("export")
    if (
        not isinstance(export_record, dict)
        or export_record.get("sha256") != _sha256_bytes(sidecar_bytes)
        or export_record.get("bytes") != len(sidecar_bytes)
    ):
        raise ValueError(
            "negative audit sidecar changed while its receipt was being validated"
        )
    decisions = _load_negative_audit(resolved_sidecar, data=sidecar_bytes)
    _cross_check_negative_audit_binding(
        decisions=decisions,
        validation=validation,
    )
    return validation, decisions


def _extract_single_component(
    *,
    mask_array: np.ndarray,
    class_id: int,
    min_component_area: int,
) -> tuple[np.ndarray | None, int, int]:
    class_mask = mask_array == class_id
    if not np.any(class_mask):
        return None, 0, 0
    structure = np.ones((3, 3), dtype=int)
    labeled, count = ndimage.label(class_mask, structure=structure)
    if count == 0:
        return None, 0, 0
    counts = np.bincount(labeled.reshape(-1), minlength=count + 1)
    qualifying_labels = [idx for idx in range(1, count + 1) if int(counts[idx]) >= min_component_area]
    if len(qualifying_labels) != 1:
        return None, int(np.sum(class_mask)), len(qualifying_labels)
    selected_label = qualifying_labels[0]
    component_mask = labeled == selected_label
    return component_mask, int(np.sum(component_mask)), len(qualifying_labels)


def _compute_contact_stats(
    target_component_mask: np.ndarray,
    hand_mask: np.ndarray,
) -> dict[str, Any]:
    if target_component_mask.shape != hand_mask.shape:
        raise ValueError("target/hand masks must have identical dimensions")
    if not np.any(target_component_mask) or not np.any(hand_mask):
        raise ValueError("target/hand masks must both be nonempty")
    overlap_px = int(np.sum(target_component_mask & hand_mask))
    structure = np.ones((3, 3), dtype=bool)
    adjacent_mask = (
        ndimage.binary_dilation(target_component_mask, structure=structure, iterations=1)
        & hand_mask
    )
    adjacent_px = int(np.sum(adjacent_mask))
    euclidean_distance = ndimage.distance_transform_edt(~hand_mask)
    chebyshev_distance = ndimage.distance_transform_cdt(~hand_mask, metric="chessboard")
    minimum_euclidean = float(np.min(euclidean_distance[target_component_mask]))
    minimum_chebyshev = int(np.min(chebyshev_distance[target_component_mask]))
    return {
        "hand_overlap_px": overlap_px,
        "hand_adjacent_px": adjacent_px,
        "min_target_hand_pixel_distance_euclidean": round(minimum_euclidean, 6),
        "min_target_hand_pixel_distance_chebyshev": minimum_chebyshev,
        "target_hand_8_neighbor_adjacent": minimum_chebyshev <= 1,
    }


def _reference_interior_mask(target_component_mask: np.ndarray) -> np.ndarray:
    if target_component_mask.ndim != 2:
        raise ValueError("reference target mask must be 2-D")
    target_mask = np.asarray(target_component_mask, dtype=bool)
    if not np.any(target_mask):
        raise ValueError("reference target mask must be nonempty")
    structure = np.ones((3, 3), dtype=bool)
    return np.asarray(
        ndimage.binary_erosion(
            target_mask,
            structure=structure,
            iterations=REFERENCE_CROP_EROSION_RADIUS,
            border_value=0,
        ),
        dtype=bool,
    )


def _crop_bbox(
    mask: np.ndarray,
    *,
    margin: int = REFERENCE_CROP_MARGIN,
) -> tuple[int, int, int, int]:
    if not np.any(mask):
        raise ValueError("cannot compute crop bbox from an empty mask")
    ys, xs = np.where(mask)
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(mask.shape[0] - 1, y1 + margin)
    x1 = min(mask.shape[1] - 1, x1 + margin)
    return x0, y0, x1 + 1, y1 + 1


def _build_reference_crop(
    *,
    image_rgb: np.ndarray,
    target_component_mask: np.ndarray,
    hand_mask: np.ndarray,
    matte_rgb: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, dict[str, Any]]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("reference image must be uint8 HxWx3")
    if target_component_mask.shape != image_rgb.shape[:2]:
        raise ValueError("reference target mask/image dimensions differ")
    if hand_mask.shape != image_rgb.shape[:2]:
        raise ValueError("reference hand mask/image dimensions differ")

    target_mask = np.asarray(target_component_mask, dtype=bool)
    hand_mask_bool = np.asarray(hand_mask, dtype=bool)
    retained_mask = _reference_interior_mask(target_mask)
    retained_pixel_count = int(np.sum(retained_mask))
    if retained_pixel_count < REFERENCE_CROP_RETAINED_AREA_FLOOR:
        raise ValueError(
            "reference crop retained area below frozen floor "
            f"({retained_pixel_count} < {REFERENCE_CROP_RETAINED_AREA_FLOOR})"
        )

    non_target_mask = ~target_mask
    hand_leakage = int(np.sum(retained_mask & hand_mask_bool))
    non_target_leakage = int(np.sum(retained_mask & non_target_mask))
    if hand_leakage or non_target_leakage:
        raise ValueError(
            "reference crop leakage "
            f"(hand={hand_leakage}, non_target={non_target_leakage})"
        )

    retained_bbox = _crop_bbox(retained_mask, margin=0)
    x0, y0, x1, y1 = _crop_bbox(retained_mask)
    matte = np.zeros_like(image_rgb, dtype=np.uint8)
    matte[:, :] = np.asarray(matte_rgb, dtype=np.uint8)
    matte[retained_mask] = image_rgb[retained_mask]
    crop_rgb = matte[y0:y1, x0:x1, :]
    crop_retained = retained_mask[y0:y1, x0:x1]
    crop_non_target = non_target_mask[y0:y1, x0:x1]
    crop_hand = hand_mask_bool[y0:y1, x0:x1]
    target_pixel_count = int(np.sum(target_mask))
    audit = {
        "interior_rule": "uniform_square_3x3_binary_erosion",
        "erosion_radius_pixels": REFERENCE_CROP_EROSION_RADIUS,
        "minimum_retained_pixel_count": REFERENCE_CROP_RETAINED_AREA_FLOOR,
        "target_pixel_count": target_pixel_count,
        "retained_pixel_count": int(np.sum(crop_retained)),
        "retained_area_fraction": round(retained_pixel_count / target_pixel_count, 8),
        "retained_bbox_xyxy": list(retained_bbox),
        "crop_bbox_xyxy": [x0, y0, x1, y1],
        "crop_margin_pixels": REFERENCE_CROP_MARGIN,
        "bbox_source": "retained_mask",
        "hand_leakage_pixel_count": int(np.sum(crop_retained & crop_hand)),
        "non_target_leakage_pixel_count": int(np.sum(crop_retained & crop_non_target)),
        "crop_shape_hw": [int(crop_rgb.shape[0]), int(crop_rgb.shape[1])],
    }
    if audit["retained_pixel_count"] < REFERENCE_CROP_RETAINED_AREA_FLOOR:
        raise ValueError("reference crop retained pixel count drifted below floor")
    if (
        audit["hand_leakage_pixel_count"]
        or audit["non_target_leakage_pixel_count"]
    ):
        raise ValueError(f"reference crop leakage audit failed: {audit}")
    return crop_rgb, audit


def _candidate_sort_key(candidate: Candidate) -> tuple[str, str, str]:
    stable_rank = hashlib.sha256(
        f"phase2_job3_selection_v1|{candidate.class_id}|{candidate.source_frame_key}".encode(
            "utf-8"
        )
    ).hexdigest()
    return (candidate.subject_key, candidate.video_key, stable_rank)


def _selection_quotas(target_total: int) -> dict[int, int]:
    if target_total < 1 or target_total > DEFAULT_SELECTION_TARGET:
        raise ValueError(
            f"target_total must be in [1, {DEFAULT_SELECTION_TARGET}], got {target_total}"
        )
    base, remainder = divmod(target_total, len(CLASS_IDS))
    quotas = {
        class_id: base + (1 if index < remainder else 0)
        for index, class_id in enumerate(CLASS_IDS)
    }
    if any(quota > DEFAULT_PER_CLASS_TARGET for quota in quotas.values()):
        raise ValueError("selection quota exceeds the frozen 20-per-class construction target")
    return quotas


def _select_balanced_candidates(
    *,
    candidates: list[Candidate],
    target_total: int,
) -> tuple[list[Candidate], dict[int, deque[Candidate]]]:
    quotas = _selection_quotas(target_total)
    by_class: dict[int, list[Candidate]] = {class_id: [] for class_id in CLASS_IDS}
    for candidate in candidates:
        by_class[candidate.class_id].append(candidate)
    for class_id in CLASS_IDS:
        by_class[class_id].sort(key=_candidate_sort_key)

    selected: list[Candidate] = []
    used_frames: set[str] = set()
    per_subject: Counter[str] = Counter()
    per_video: Counter[str] = Counter()
    per_class: Counter[int] = Counter()
    per_class_subject: Counter[tuple[int, str]] = Counter()
    per_class_video: Counter[tuple[int, str]] = Counter()

    def choose_next(class_id: int) -> Candidate | None:
        if per_class[class_id] >= quotas[class_id]:
            return None
        pool = by_class[class_id]
        if not pool:
            return None
        best_index: int | None = None
        best_score: tuple[int, int, int, int, str, str, str] | None = None
        for index, candidate in enumerate(pool):
            if candidate.source_frame_key in used_frames:
                continue
            score = (
                int(per_class_subject[(class_id, candidate.subject_key)]),
                int(per_class_video[(class_id, candidate.video_key)]),
                int(per_subject[candidate.subject_key]),
                int(per_video[candidate.video_key]),
                candidate.subject_key,
                candidate.video_key,
                _candidate_sort_key(candidate)[2],
            )
            if best_score is None or score < best_score:
                best_score = score
                best_index = index
        if best_index is None:
            return None
        candidate = pool.pop(best_index)
        return candidate

    while len(selected) < target_total:
        progressed = False
        for class_id in CLASS_IDS:
            if len(selected) >= target_total:
                break
            candidate = choose_next(class_id)
            if candidate is None:
                continue
            selected.append(candidate)
            used_frames.add(candidate.source_frame_key)
            per_class[class_id] += 1
            per_class_subject[(class_id, candidate.subject_key)] += 1
            per_class_video[(class_id, candidate.video_key)] += 1
            per_subject[candidate.subject_key] += 1
            per_video[candidate.video_key] += 1
            progressed = True
        if not progressed:
            break

    def frozen_spare_score(candidate: Candidate) -> tuple[int, int, int, int, str, str, str]:
        return (
            int(per_class_subject[(candidate.class_id, candidate.subject_key)]),
            int(per_class_video[(candidate.class_id, candidate.video_key)]),
            int(per_subject[candidate.subject_key]),
            int(per_video[candidate.video_key]),
            candidate.subject_key,
            candidate.video_key,
            _candidate_sort_key(candidate)[2],
        )

    ranked_spares = {
        class_id: deque(sorted(by_class[class_id], key=frozen_spare_score))
        for class_id in CLASS_IDS
    }
    return selected, ranked_spares


def _take_ranked_spare(
    *,
    class_id: int,
    ranked_spares: dict[int, deque[Candidate]],
    occupied_source_frames: set[str],
    scheduled_candidate_ids: set[str],
    excluded_candidate_ids: set[str],
) -> Candidate | None:
    pool = ranked_spares[class_id]
    deferred: deque[Candidate] = deque()
    selected: Candidate | None = None
    while pool:
        candidate = pool.popleft()
        if (
            candidate.candidate_id in scheduled_candidate_ids
            or candidate.candidate_id in excluded_candidate_ids
        ):
            continue
        if candidate.source_frame_key in occupied_source_frames:
            deferred.append(candidate)
            continue
        selected = candidate
        break
    pool.extendleft(reversed(deferred))
    return selected


def _load_candidates(
    *,
    data_root: Path,
    inventory_rows: list[dict[str, str]],
    sidecar_index: dict[tuple[str, str, int], dict[str, Any]],
    min_component_area: int,
) -> tuple[
    list[Candidate],
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    candidates: list[Candidate] = []
    structural_failures: list[str] = []
    pre_review_rejections: list[dict[str, Any]] = []
    crop_nonviable_rejections: list[dict[str, Any]] = []
    for row in inventory_rows:
        source_family = str(row.get("source_family") or "").strip()
        provenance_status = str(row.get("confirmation_provenance_status") or "").strip()
        if source_family != SOURCE_FAMILY_LOCK or provenance_status != PROVENANCE_STATUS_LOCK:
            continue

        image_path = data_root / "raw" / str(row.get("image_relpath") or "")
        mask_path = data_root / "raw" / str(row.get("label_relpath") or "")
        try:
            _, mask_array = census.decode_image_mask_pair(image_path, mask_path)
        except Exception as exc:  # noqa: BLE001
            structural_failures.append(f"decode_failure:{row.get('frame_id')}:{type(exc).__name__}:{exc}")
            continue
        if mask_array.ndim != 2:
            structural_failures.append(f"invalid_mask_dim:{row.get('frame_id')}:{mask_array.ndim}")
            continue
        hand_mask_all = (mask_array == census.MASK_CLASS_LEFT_HAND) | (
            mask_array == census.MASK_CLASS_RIGHT_HAND
        )
        for class_id in CLASS_IDS:
            target_component_mask, target_area, qualifying_count = _extract_single_component(
                mask_array=mask_array,
                class_id=class_id,
                min_component_area=min_component_area,
            )
            if target_component_mask is None:
                continue
            if target_area < min_component_area or qualifying_count != 1:
                continue
            required_hands = census.HAND_REQUIRED_BY_CLASS[class_id]
            hand_required_ok = all(np.any(mask_array == hand_class) for hand_class in required_hands)
            if not hand_required_ok:
                continue
            if not np.any(hand_mask_all):
                continue
            split = str(row.get("split") or "")
            frame_id = str(row.get("frame_id") or "")
            sidecar = sidecar_index.get((split, frame_id, class_id))
            sidecar_label_role = None
            candidate_id = _sha256_text(
                f"phase2_job3_burned|candidate|{split}|{frame_id}|{class_id}"
            )
            if sidecar is not None:
                sidecar_source = str(sidecar.get("source_family") or "").strip()
                if sidecar_source and sidecar_source != SOURCE_FAMILY_LOCK:
                    structural_failures.append(
                        f"sidecar_firewall_violation:{split}:{frame_id}:class{class_id}:{sidecar_source}"
                    )
                    continue
                sidecar_label_role = str(sidecar.get("label_role") or "") or None
                candidate_id = str(sidecar.get("candidate_id") or "").strip()
                if re.fullmatch(r"[0-9a-f]{64}", candidate_id) is None:
                    structural_failures.append(
                        f"sidecar_candidate_id_invalid:{split}:{frame_id}:class{class_id}"
                    )
                    continue
            retained_mask = _reference_interior_mask(target_component_mask)
            retained_area = int(np.sum(retained_mask))
            retained_hand_overlap = int(np.sum(retained_mask & hand_mask_all))
            if (
                retained_area < REFERENCE_CROP_RETAINED_AREA_FLOOR
                or retained_hand_overlap > 0
            ):
                rejection_reason = (
                    "rejected_before_selection_retained_hand_overlap"
                    if retained_hand_overlap
                    else "rejected_before_selection_nonviable_uniform_eroded_interior"
                )
                crop_nonviable_rejections.append(
                    {
                        "candidate_id": candidate_id,
                        "split": split,
                        "frame_id": frame_id,
                        "source_video_id": str(row.get("source_video_id") or ""),
                        "source_subject_id": str(row.get("source_subject_id") or ""),
                        "mask_class": class_id,
                        "target_area": target_area,
                        "retained_area": retained_area,
                        "retained_hand_overlap_pixel_count": retained_hand_overlap,
                        "erosion_radius_pixels": REFERENCE_CROP_EROSION_RADIUS,
                        "minimum_retained_pixel_count": (
                            REFERENCE_CROP_RETAINED_AREA_FLOOR
                        ),
                        "rejection_reason": rejection_reason,
                        "hidden": True,
                        "rows_frozen": False,
                    }
                )
                continue
            contact_stats = _compute_contact_stats(target_component_mask, hand_mask_all)
            if not isinstance(contact_stats["hand_overlap_px"], int) or not isinstance(
                contact_stats["hand_adjacent_px"], int
            ):
                raise AssertionError("contact pixel counts must be integers")
            if not isinstance(
                contact_stats["min_target_hand_pixel_distance_euclidean"], float
            ) or not isinstance(
                contact_stats["min_target_hand_pixel_distance_chebyshev"], int
            ):
                raise AssertionError("minimum target/hand distances must be numeric")
            if class_id in PENDING_NEGATIVE_CLASSES and contact_stats[
                "target_hand_8_neighbor_adjacent"
            ]:
                pre_review_rejections.append(
                    {
                        "candidate_id": candidate_id,
                        "split": split,
                        "frame_id": frame_id,
                        "source_video_id": str(row.get("source_video_id") or ""),
                        "source_subject_id": str(row.get("source_subject_id") or ""),
                        "mask_class": class_id,
                        "target_area": target_area,
                        **contact_stats,
                        "rejection_reason": (
                            "rejected_before_human_review_target_hand_8_neighbor_adjacency"
                        ),
                        "hidden": True,
                        "rows_frozen": False,
                    }
                )
                continue
            candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    split=split,
                    frame_id=frame_id,
                    source_video_id=str(row.get("source_video_id") or ""),
                    source_subject_id=str(row.get("source_subject_id") or ""),
                    source_frame_or_timestamp=str(row.get("source_frame_or_timestamp") or ""),
                    image_relpath=str(row.get("image_relpath") or ""),
                    label_relpath=str(row.get("label_relpath") or ""),
                    class_id=class_id,
                    target_area=target_area,
                    reference_retained_area=retained_area,
                    target_component_mask=target_component_mask,
                    hand_mask=hand_mask_all.copy(),
                    source_frame_key=f"{split}|{frame_id}",
                    subject_key=str(row.get("source_subject_id") or "") or "unknown_subject",
                    video_key=str(row.get("source_video_id") or "") or "unknown_video",
                    sidecar_label_role=sidecar_label_role,
                    hand_overlap_px=contact_stats["hand_overlap_px"],
                    hand_adjacent_px=contact_stats["hand_adjacent_px"],
                    min_target_hand_pixel_distance_euclidean=contact_stats[
                        "min_target_hand_pixel_distance_euclidean"
                    ],
                    min_target_hand_pixel_distance_chebyshev=contact_stats[
                        "min_target_hand_pixel_distance_chebyshev"
                    ],
                    target_hand_8_neighbor_adjacent=contact_stats[
                        "target_hand_8_neighbor_adjacent"
                    ],
                )
            )
    return (
        candidates,
        structural_failures,
        pre_review_rejections,
        crop_nonviable_rejections,
    )


def _public_request_record(
    *,
    arm: str,
    request_id: str,
    item_id: str,
    full_frame_path: str,
    reference_crop_path: str,
    width: int,
    height: int,
    crop_width: int,
    crop_height: int,
) -> dict[str, Any]:
    if arm == ARM_PCS_E:
        user_prompt = USER_PROMPT_BASE
        full_frame_description = "unmarked_full_frame"
    elif arm == ARM_MPP_E:
        user_prompt = USER_PROMPT_BASE
        full_frame_description = "egohos_mask_marked_full_frame"
    elif arm == ARM_VP_E:
        user_prompt = USER_PROMPT_BASE + "\n" + VP_FROZEN_INSTRUCTION_PACKAGE
        full_frame_description = "egohos_mask_marked_full_frame"
    elif arm == ARM_ALT_MARKED:
        user_prompt = USER_PROMPT_BASE
        full_frame_description = "alternate_marked_full_frame"
    else:
        raise ValueError(f"unknown arm: {arm}")
    return {
        "request_id": request_id,
        "item_id": item_id,
        "arm": arm,
        "benchmark_split": f"phase2_job3_egohos_{arm}_construction_pilot",
        "claim_boundary": CLAIM_BOUNDARY,
        "dataset_role": "construction_pilot",
        "burned": True,
        "non_confirmatory": True,
        "analysis_role": (
            "descriptive_renderer_robustness" if arm == ARM_ALT_MARKED else "primary_construction"
        ),
        "export_role": "construction_review_only",
        "scoring_permitted": False,
        "rows_frozen": False,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "inputs": [
            {
                "kind": "full_frame",
                "description": full_frame_description,
                "path": full_frame_path,
                "width": width,
                "height": height,
            },
            {
                "kind": "reference_crop",
                "description": "unmarked_target_only_matted_crop",
                "path": reference_crop_path,
                "width": crop_width,
                "height": crop_height,
            },
        ],
    }


def _scan_public_leakage(
    *,
    public_payloads: dict[str, Any],
    selected_candidates: list[Candidate],
) -> list[str]:
    serialized = json.dumps(public_payloads, sort_keys=True)
    failures: list[str] = []
    for keyword in FORBIDDEN_PUBLIC_KEYWORDS:
        if keyword in serialized:
            failures.append(f"public_keyword_leakage:{keyword}")
    dynamic_tokens: set[str] = set()
    for candidate in selected_candidates:
        dynamic_tokens.update(
            token
            for token in (
                candidate.frame_id,
                candidate.source_video_id,
                candidate.source_subject_id,
                candidate.image_relpath,
                candidate.label_relpath,
                candidate.source_frame_or_timestamp,
            )
            if token and len(token) >= 4
        )
    for token in sorted(dynamic_tokens):
        if token in serialized:
            failures.append(f"public_dynamic_leakage:{token}")
    return failures


def _build_negative_review_asset(
    *,
    unmarked_rgb: np.ndarray,
    marked_rgb: np.ndarray,
    reference_crop_rgb: np.ndarray,
) -> np.ndarray:
    full_height, full_width = unmarked_rgb.shape[:2]
    crop_image = Image.fromarray(reference_crop_rgb, mode="RGB")
    crop_image.thumbnail((full_width, full_height), Image.Resampling.NEAREST)
    label_height = 22
    canvas = Image.new(
        "RGB",
        (full_width * 3, full_height + label_height),
        color=(235, 235, 235),
    )
    canvas.paste(Image.fromarray(unmarked_rgb, mode="RGB"), (0, label_height))
    canvas.paste(Image.fromarray(marked_rgb, mode="RGB"), (full_width, label_height))
    crop_x = full_width * 2 + (full_width - crop_image.width) // 2
    crop_y = label_height + (full_height - crop_image.height) // 2
    canvas.paste(crop_image, (crop_x, crop_y))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), "unmarked full frame", fill=(10, 10, 10))
    draw.text((full_width + 4, 4), "target/hand overlay", fill=(10, 10, 10))
    draw.text((full_width * 2 + 4, 4), "target-only crop", fill=(10, 10, 10))
    return np.asarray(canvas, dtype=np.uint8)


def _build_pending_contact_sheets(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
    page_size: int,
) -> list[str]:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if not rows:
        return []
    page_paths: list[str] = []
    page_count = (len(rows) + page_size - 1) // page_size
    for page_index in range(page_count):
        page_rows = rows[page_index * page_size : (page_index + 1) * page_size]
        thumbs: list[Image.Image] = []
        labels: list[str] = []
        for row in page_rows:
            review_path = out_dir / str(row["review_asset_path"])
            with Image.open(review_path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((960, 220), Image.Resampling.BILINEAR)
            thumbs.append(thumb)
            labels.append(
                f"candidate {str(row['candidate_id'])[:16]} | class {row['class_id']} | pending visual audit"
            )
        row_heights = [thumb.height + 36 for thumb in thumbs]
        canvas = Image.new("RGB", (1000, sum(row_heights)), color=(245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        y0 = 0
        for idx, thumb in enumerate(thumbs):
            draw.text((12, y0 + 4), labels[idx], fill=(10, 10, 10))
            canvas.paste(thumb, ((canvas.width - thumb.width) // 2, y0 + 24))
            y0 += row_heights[idx]
        rel = f"hidden/pending_negative_review/contact_sheets/page_{page_index:03d}.png"
        _save_png_atomic(out_dir / rel, np.asarray(canvas, dtype=np.uint8))
        page_paths.append(rel)
    return page_paths


def build_burned_items(
    *,
    data_root: Path,
    inventory_path: Path,
    out_dir: Path,
    candidate_sidecar_path: Path | None = None,
    negative_audit_sidecar_path: Path | None = None,
    negative_audit_receipt_path: Path | None = None,
    selection_target: int = DEFAULT_SELECTION_TARGET,
    limit: int | None = None,
    min_component_area: int = DEFAULT_MIN_COMPONENT_AREA,
    pending_contact_sheet_page_size: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    if selection_target <= 0:
        raise ValueError("selection_target must be positive")
    if selection_target > DEFAULT_SELECTION_TARGET:
        raise ValueError(
            f"selection_target cannot exceed frozen target {DEFAULT_SELECTION_TARGET}"
        )
    if min_component_area < 1:
        raise ValueError("min_component_area must be >= 1")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    inventory_rows = census.load_frame_inventory(inventory_path.expanduser())
    sidecar_index = _load_optional_sidecar(candidate_sidecar_path)
    if (negative_audit_sidecar_path is None) != (negative_audit_receipt_path is None):
        raise ValueError(
            "negative-audit-sidecar and negative-audit-validation-receipt "
            "must be supplied together"
        )
    negative_audit_validation = None
    negative_decisions: dict[str, dict[str, str]] = {}
    if negative_audit_sidecar_path is not None and negative_audit_receipt_path is not None:
        negative_audit_validation, negative_decisions = _verify_and_load_negative_audit(
            sidecar_path=negative_audit_sidecar_path,
            receipt_path=negative_audit_receipt_path,
        )
    (
        candidates,
        structural_failures,
        pre_review_rejections,
        crop_nonviable_rejections,
    ) = _load_candidates(
        data_root=data_root.expanduser(),
        inventory_rows=inventory_rows,
        sidecar_index=sidecar_index,
        min_component_area=min_component_area,
    )

    target_total = min(selection_target, limit) if limit is not None else selection_target
    selected, ranked_spares = _select_balanced_candidates(
        candidates=candidates,
        target_total=target_total,
    )
    if not selected:
        structural_failures.append("no_candidates_selected")
    if len(selected) != target_total:
        structural_failures.append(
            f"selection_underfilled:expected={target_total}:actual={len(selected)}"
        )

    out_dir = _prepare_fresh_output_dir(out_dir)
    (out_dir / "public").mkdir(parents=True, exist_ok=True)
    (out_dir / "hidden").mkdir(parents=True, exist_ok=True)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    (out_dir / "audits").mkdir(parents=True, exist_ok=True)

    public_rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_ORDER}
    hidden_rows: list[dict[str, Any]] = []
    emitted_candidates: list[Candidate] = []
    pending_rows: list[dict[str, Any]] = []
    human_rejection_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    source_before_sha: dict[str, str] = {}
    source_after_sha: dict[str, str] = {}
    intended_confinement_failures = 0
    crop_audits: list[dict[str, Any]] = []
    prompt_audit_rows: list[dict[str, Any]] = []
    item_audit_rows: list[dict[str, Any]] = []
    mpp_vp_equal_count = 0
    mpp_vp_not_equal_count = 0
    vp_instruction_delta_only_count = 0
    duplicate_source_frames = 0
    used_source_frames: set[str] = set()
    item_ids: set[str] = set()
    request_ids: set[str] = set()
    asset_paths: set[str] = set()
    class_counts: Counter[int] = Counter()
    subject_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()

    candidate_worklist: deque[Candidate] = deque(selected)
    scheduled_candidate_ids = {candidate.candidate_id for candidate in selected}
    scheduled_source_frames = {candidate.source_frame_key for candidate in selected}
    excluded_candidate_ids: set[str] = set()
    replacement_deficits: Counter[int] = Counter()
    rejected_slots_by_class: dict[int, deque[str]] = {
        class_id: deque() for class_id in CLASS_IDS
    }

    while candidate_worklist or any(replacement_deficits.values()):
        if not candidate_worklist:
            for class_id in CLASS_IDS:
                while replacement_deficits[class_id] > 0:
                    replacement = _take_ranked_spare(
                        class_id=class_id,
                        ranked_spares=ranked_spares,
                        occupied_source_frames=(used_source_frames | scheduled_source_frames),
                        scheduled_candidate_ids=scheduled_candidate_ids,
                        excluded_candidate_ids=excluded_candidate_ids,
                    )
                    rejected_candidate_id = rejected_slots_by_class[class_id].popleft()
                    if replacement is None:
                        structural_failures.append(
                            "human_rejection_spare_exhausted:"
                            f"class{class_id}:{rejected_candidate_id}"
                        )
                        replacement_rows.append(
                            {
                                "class_id": class_id,
                                "rejected_candidate_id": rejected_candidate_id,
                                "replacement_candidate_id": None,
                                "replacement_status": "spare_pool_exhausted",
                                "rows_frozen": False,
                            }
                        )
                        replacement_deficits[class_id] -= 1
                        continue
                    scheduled_candidate_ids.add(replacement.candidate_id)
                    scheduled_source_frames.add(replacement.source_frame_key)
                    candidate_worklist.append(replacement)
                    replacement_rows.append(
                        {
                            "class_id": class_id,
                            "rejected_candidate_id": rejected_candidate_id,
                            "replacement_candidate_id": replacement.candidate_id,
                            "replacement_status": "scheduled_from_frozen_ranked_spare_pool",
                            "rows_frozen": False,
                        }
                    )
                    replacement_deficits[class_id] -= 1
            if not candidate_worklist:
                break

        candidate = candidate_worklist.popleft()
        scheduled_source_frames.discard(candidate.source_frame_key)

        raw_identity = "|".join([candidate.split, candidate.frame_id, str(candidate.class_id)])
        item_id = _opaque_id("item", raw_identity)

        image_path = data_root / "raw" / candidate.image_relpath
        mask_path = data_root / "raw" / candidate.label_relpath
        image_before_sha = _sha256_file(image_path)
        mask_before_sha = _sha256_file(mask_path)
        with Image.open(image_path) as image:
            image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        _, decoded_mask = census.decode_image_mask_pair(image_path, mask_path)
        if decoded_mask.ndim != 2:
            structural_failures.append(f"selected_mask_not_2d:{candidate.frame_id}")
            continue
        if not np.any(candidate.target_component_mask):
            structural_failures.append(f"empty_target_component:{candidate.frame_id}:class{candidate.class_id}")
            continue
        if image_rgb.shape[:2] != decoded_mask.shape:
            structural_failures.append(
                f"selected_dimension_mismatch:{candidate.frame_id}:{image_rgb.shape[:2]}:{decoded_mask.shape}"
            )
            continue

        hand_mask = (decoded_mask == census.MASK_CLASS_LEFT_HAND) | (
            decoded_mask == census.MASK_CLASS_RIGHT_HAND
        )
        contact_stats = _compute_contact_stats(candidate.target_component_mask, hand_mask)
        overlap_px = contact_stats["hand_overlap_px"]
        adjacent_px = contact_stats["hand_adjacent_px"]
        if not isinstance(overlap_px, int) or not isinstance(adjacent_px, int):
            raise AssertionError("decoded contact pixel counts must be integers")
        if not isinstance(
            contact_stats["min_target_hand_pixel_distance_euclidean"], float
        ) or not isinstance(
            contact_stats["min_target_hand_pixel_distance_chebyshev"], int
        ):
            raise AssertionError("decoded target/hand distances must be numeric")
        if contact_stats != {
            "hand_overlap_px": candidate.hand_overlap_px,
            "hand_adjacent_px": candidate.hand_adjacent_px,
            "min_target_hand_pixel_distance_euclidean": (
                candidate.min_target_hand_pixel_distance_euclidean
            ),
            "min_target_hand_pixel_distance_chebyshev": (
                candidate.min_target_hand_pixel_distance_chebyshev
            ),
            "target_hand_8_neighbor_adjacent": candidate.target_hand_8_neighbor_adjacent,
        }:
            structural_failures.append(f"selected_contact_stat_drift:{candidate.candidate_id}")
            continue
        if candidate.class_id in PENDING_NEGATIVE_CLASSES and contact_stats[
            "target_hand_8_neighbor_adjacent"
        ]:
            structural_failures.append(
                f"pending_negative_adjacency_reached_builder:{candidate.candidate_id}"
            )
            continue

        try:
            marked_rgb, intended_mask = marker_renderer.render_marked_frame(
                image_rgb=image_rgb,
                target_mask=candidate.target_component_mask,
                hand_mask=hand_mask,
                style=marker_renderer.DISCOVERY_STYLE,
            )
            alt_marked_rgb, alt_intended_mask = marker_renderer.render_marked_frame(
                image_rgb=image_rgb,
                target_mask=candidate.target_component_mask,
                hand_mask=hand_mask,
                style=marker_renderer.ALTERNATE_STYLE,
            )
        except Exception as exc:  # noqa: BLE001
            structural_failures.append(f"render_failure:{candidate.frame_id}:class{candidate.class_id}:{exc}")
            continue

        changed = np.any(marked_rgb != image_rgb, axis=2)
        confinement_failed = False
        if np.any(changed & ~intended_mask):
            intended_confinement_failures += 1
            structural_failures.append(f"intended_pixel_confinement_failure:{item_id}")
            confinement_failed = True
        alt_changed = np.any(alt_marked_rgb != image_rgb, axis=2)
        if np.any(alt_changed & ~alt_intended_mask):
            intended_confinement_failures += 1
            structural_failures.append(f"alt_intended_pixel_confinement_failure:{item_id}")
            confinement_failed = True
        if confinement_failed:
            continue

        try:
            crop_rgb, crop_audit = _build_reference_crop(
                image_rgb=image_rgb,
                target_component_mask=candidate.target_component_mask,
                hand_mask=hand_mask,
            )
        except Exception as exc:  # noqa: BLE001
            structural_failures.append(f"crop_failure:{candidate.frame_id}:class{candidate.class_id}:{exc}")
            continue
        if crop_audit["retained_pixel_count"] != candidate.reference_retained_area:
            structural_failures.append(
                f"reference_retained_area_drift:{candidate.candidate_id}"
            )
            continue

        base_asset_key = _opaque_id("asset", item_id)
        full_unmarked_rel = f"assets/full/{base_asset_key}.png"
        full_marked_rel = f"assets/marked/{base_asset_key}.png"
        full_alt_marked_rel = f"assets/marked_alt/{base_asset_key}.png"
        crop_rel = f"assets/reference_crop/{base_asset_key}.png"
        review_asset_rel = (
            f"hidden/pending_negative_review/assets/{base_asset_key}.png"
        )
        review_asset_rgb = _build_negative_review_asset(
            unmarked_rgb=image_rgb,
            marked_rgb=marked_rgb,
            reference_crop_rgb=crop_rgb,
        )
        review_asset_bytes = _png_bytes(review_asset_rgb)
        review_asset_sha256 = _sha256_bytes(review_asset_bytes)

        scoring_eligible = candidate.class_id in PILOT_POSITIVE_CLASSES
        label_role = "positive"
        canonical_answer_json: str | None = CANONICAL_TRUE_JSON if scoring_eligible else None
        negative_review: dict[str, str] | None = None
        if candidate.class_id in PENDING_NEGATIVE_CLASSES:
            negative_review = negative_decisions.get(candidate.candidate_id)
            if negative_review and negative_review["review_asset_sha256"] != review_asset_sha256:
                structural_failures.append(
                    f"negative_review_asset_hash_mismatch:{candidate.candidate_id}"
                )
                negative_review = None
            if negative_review and negative_review["decision"] == NEGATIVE_APPROVAL_DECISION:
                scoring_eligible = True
                label_role = "negative_human_audited_no_direct_contact"
                canonical_answer_json = CANONICAL_FALSE_JSON
            elif negative_review and negative_review["decision"] in NEGATIVE_REJECTION_DECISIONS:
                excluded_candidate_ids.add(candidate.candidate_id)
                replacement_deficits[candidate.class_id] += 1
                rejected_slots_by_class[candidate.class_id].append(candidate.candidate_id)
                human_rejection_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "item_id": item_id,
                        "class_id": candidate.class_id,
                        "source_split": candidate.split,
                        "source_frame_id": candidate.frame_id,
                        "source_video_id": candidate.source_video_id,
                        "source_subject_id": candidate.source_subject_id,
                        "decision": negative_review["decision"],
                        "reason": negative_review["reason"],
                        "reviewer_id": negative_review["reviewer_id"],
                        "reviewed_at": negative_review["reviewed_at"],
                        "review_asset_sha256": review_asset_sha256,
                        "selection_action": "excluded_and_refill_from_frozen_ranked_spare_pool",
                        "rows_frozen": False,
                    }
                )
                if (
                    _sha256_file(image_path) != image_before_sha
                    or _sha256_file(mask_path) != mask_before_sha
                ):
                    structural_failures.append(
                        f"raw_input_mutation_detected_while_rejecting:{candidate.candidate_id}"
                    )
                continue
            else:
                scoring_eligible = False
                label_role = "negative_candidate_pending_no_direct_contact_audit"
                canonical_answer_json = None

        if item_id in item_ids:
            structural_failures.append(f"duplicate_item_id:{item_id}")
        item_ids.add(item_id)
        crop_audits.append({"item_id": item_id, **crop_audit})
        for rel in (
            full_unmarked_rel,
            full_marked_rel,
            full_alt_marked_rel,
            crop_rel,
            review_asset_rel,
        ):
            if rel in asset_paths:
                structural_failures.append(f"duplicate_asset_path:{rel}")
            asset_paths.add(rel)

        full_unmarked_bytes = _save_png_atomic(out_dir / full_unmarked_rel, image_rgb)
        full_marked_bytes = _save_png_atomic(out_dir / full_marked_rel, marked_rgb)
        full_alt_marked_bytes = _save_png_atomic(out_dir / full_alt_marked_rel, alt_marked_rgb)
        crop_bytes = _save_png_atomic(out_dir / crop_rel, crop_rgb)
        _atomic_write_bytes(out_dir / review_asset_rel, review_asset_bytes)
        full_unmarked_sha = _sha256_bytes(full_unmarked_bytes)
        full_marked_sha = _sha256_bytes(full_marked_bytes)
        full_alt_marked_sha = _sha256_bytes(full_alt_marked_bytes)
        crop_sha = _sha256_bytes(crop_bytes)
        if label_role == "negative_candidate_pending_no_direct_contact_audit":
            pending_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "item_id": item_id,
                    "class_id": candidate.class_id,
                    "source_split": candidate.split,
                    "source_frame_id": candidate.frame_id,
                    "source_video_id": candidate.source_video_id,
                    "source_subject_id": candidate.source_subject_id,
                    "target_area": candidate.target_area,
                    "hand_overlap_px": overlap_px,
                    "hand_adjacent_px": adjacent_px,
                    "min_target_hand_pixel_distance_euclidean": contact_stats[
                        "min_target_hand_pixel_distance_euclidean"
                    ],
                    "min_target_hand_pixel_distance_chebyshev": contact_stats[
                        "min_target_hand_pixel_distance_chebyshev"
                    ],
                    "target_hand_8_neighbor_adjacent": contact_stats[
                        "target_hand_8_neighbor_adjacent"
                    ],
                    "review_asset_path": review_asset_rel,
                    "review_asset_sha256": review_asset_sha256,
                    "decision": "",
                    "reason": "",
                    "reviewer_id": "",
                    "reviewed_at": "",
                    "rows_frozen": False,
                }
            )

        arm_asset_map = {
            ARM_PCS_E: full_unmarked_rel,
            ARM_MPP_E: full_marked_rel,
            ARM_VP_E: full_marked_rel,
            ARM_ALT_MARKED: full_alt_marked_rel,
        }
        per_arm_request_ids: dict[str, str] = {}
        for arm in ARM_ORDER:
            request_id = _opaque_id("request", f"{arm}|{item_id}")
            per_arm_request_ids[arm] = request_id
            if request_id in request_ids:
                structural_failures.append(f"duplicate_request_id:{request_id}")
            request_ids.add(request_id)
            public_record = _public_request_record(
                arm=arm,
                request_id=request_id,
                item_id=item_id,
                full_frame_path=arm_asset_map[arm],
                reference_crop_path=crop_rel,
                width=int(image_rgb.shape[1]),
                height=int(image_rgb.shape[0]),
                crop_width=int(crop_rgb.shape[1]),
                crop_height=int(crop_rgb.shape[0]),
            )
            public_rows_by_arm[arm].append(public_record)

        mpp_payload = public_rows_by_arm[ARM_MPP_E][-1]
        vp_payload = public_rows_by_arm[ARM_VP_E][-1]
        pcs_payload = public_rows_by_arm[ARM_PCS_E][-1]
        if (
            mpp_payload["system_prompt"].encode("utf-8") == pcs_payload["system_prompt"].encode("utf-8")
            and mpp_payload["user_prompt"].encode("utf-8") == pcs_payload["user_prompt"].encode("utf-8")
        ):
            prompt_equal = True
        else:
            prompt_equal = False
            structural_failures.append(f"pcs_mpp_prompt_byte_mismatch:{item_id}")
        if mpp_payload["inputs"][0]["path"] == vp_payload["inputs"][0]["path"]:
            mpp_vp_equal_count += 1
        else:
            mpp_vp_not_equal_count += 1
            structural_failures.append(f"mpp_vp_full_frame_path_mismatch:{item_id}")
        vp_expected = USER_PROMPT_BASE + "\n" + VP_FROZEN_INSTRUCTION_PACKAGE
        if (
            vp_payload["system_prompt"] == mpp_payload["system_prompt"]
            and vp_payload["user_prompt"] == vp_expected
            and mpp_payload["user_prompt"] == USER_PROMPT_BASE
        ):
            vp_instruction_delta_only_count += 1
        else:
            structural_failures.append(f"vp_instruction_delta_violation:{item_id}")
        if (
            pcs_payload["inputs"][1]["path"] == mpp_payload["inputs"][1]["path"]
            and mpp_payload["inputs"][1]["path"] == vp_payload["inputs"][1]["path"]
        ):
            pass
        else:
            structural_failures.append(f"reference_crop_path_mismatch:{item_id}")

        prompt_audit_rows.append(
            {
                "item_id": item_id,
                "pcs_system_sha256": _sha256_text(pcs_payload["system_prompt"]),
                "pcs_user_sha256": _sha256_text(pcs_payload["user_prompt"]),
                "mpp_system_sha256": _sha256_text(mpp_payload["system_prompt"]),
                "mpp_user_sha256": _sha256_text(mpp_payload["user_prompt"]),
                "vp_system_sha256": _sha256_text(vp_payload["system_prompt"]),
                "vp_user_sha256": _sha256_text(vp_payload["user_prompt"]),
                "pcs_mpp_prompt_bytes_equal": prompt_equal,
                "mpp_vp_full_frame_sha256_equal": (
                    _sha256_file(out_dir / mpp_payload["inputs"][0]["path"])
                    == _sha256_file(out_dir / vp_payload["inputs"][0]["path"])
                ),
                "vp_delta_text_sha256": _sha256_text(VP_FROZEN_INSTRUCTION_PACKAGE),
                "reference_crop_sha256": crop_sha,
                "full_unmarked_sha256": full_unmarked_sha,
                "full_marked_sha256": full_marked_sha,
                "full_alt_marked_sha256": full_alt_marked_sha,
            }
        )
        item_audit_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "item_id": item_id,
                "mask_class": candidate.class_id,
                "rows_frozen": False,
                "full_frame_shape_hw": [int(image_rgb.shape[0]), int(image_rgb.shape[1])],
                "reference_crop_shape_hw": [int(crop_rgb.shape[0]), int(crop_rgb.shape[1])],
                "target_area_px": candidate.target_area,
                **contact_stats,
                "crop_leakage": crop_audit,
                "discovery_changed_pixel_count": int(np.sum(changed)),
                "discovery_unintended_changed_pixel_count": int(
                    np.sum(changed & ~intended_mask)
                ),
                "alternate_changed_pixel_count": int(np.sum(alt_changed)),
                "alternate_unintended_changed_pixel_count": int(
                    np.sum(alt_changed & ~alt_intended_mask)
                ),
                "pcs_mpp_prompt_bytes_equal": prompt_equal,
                "mpp_vp_png_sha256_equal": (
                    _sha256_file(out_dir / mpp_payload["inputs"][0]["path"])
                    == _sha256_file(out_dir / vp_payload["inputs"][0]["path"])
                ),
                "reference_crop_sha256": crop_sha,
                "review_asset_sha256": review_asset_sha256,
            }
        )

        hidden_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "item_id": item_id,
                "request_ids": per_arm_request_ids,
                "dataset_role": "construction_pilot",
                "burned": True,
                "non_confirmatory": True,
                "rows_frozen": False,
                "source_family": SOURCE_FAMILY_LOCK,
                "confirmation_provenance_status": PROVENANCE_STATUS_LOCK,
                "split": candidate.split,
                "frame_id": candidate.frame_id,
                "source_video_id": candidate.source_video_id,
                "source_subject_id": candidate.source_subject_id,
                "source_frame_or_timestamp": candidate.source_frame_or_timestamp,
                "image_relpath": candidate.image_relpath,
                "label_relpath": candidate.label_relpath,
                "class_id": candidate.class_id,
                "label_role": label_role,
                "target_area": candidate.target_area,
                "hand_overlap_px": overlap_px,
                "hand_adjacent_px": adjacent_px,
                "min_target_hand_pixel_distance_euclidean": contact_stats[
                    "min_target_hand_pixel_distance_euclidean"
                ],
                "min_target_hand_pixel_distance_chebyshev": contact_stats[
                    "min_target_hand_pixel_distance_chebyshev"
                ],
                "target_hand_8_neighbor_adjacent": contact_stats[
                    "target_hand_8_neighbor_adjacent"
                ],
                "scoring_eligible": scoring_eligible,
                "canonical_answer_json": canonical_answer_json if scoring_eligible else None,
                "negative_review": negative_review,
                "review_asset_path": review_asset_rel,
                "review_asset_sha256": review_asset_sha256,
                "reference_crop_asset_path": crop_rel,
                "full_unmarked_asset_path": full_unmarked_rel,
                "full_marked_asset_path": full_marked_rel,
                "full_alt_marked_asset_path": full_alt_marked_rel,
                "crop_audit": crop_audit,
            }
        )
        emitted_candidates.append(candidate)
        class_counts[candidate.class_id] += 1
        subject_counts[candidate.subject_key] += 1
        video_counts[candidate.video_key] += 1
        if candidate.source_frame_key in used_source_frames:
            duplicate_source_frames += 1
        used_source_frames.add(candidate.source_frame_key)
        source_before_sha[str(image_path)] = image_before_sha
        source_before_sha[str(mask_path)] = mask_before_sha
        source_after_sha[str(image_path)] = _sha256_file(image_path)
        source_after_sha[str(mask_path)] = _sha256_file(mask_path)

    if duplicate_source_frames:
        structural_failures.append(f"duplicate_source_frame_count:{duplicate_source_frames}")

    for source_path, before_sha in source_before_sha.items():
        after_sha = source_after_sha.get(source_path)
        if after_sha != before_sha:
            structural_failures.append(f"raw_input_mutation_detected:{source_path}")

    expected_class_quotas = _selection_quotas(target_total)
    if len(hidden_rows) != target_total:
        structural_failures.append(
            f"materialized_cohort_underfilled:expected={target_total}:actual={len(hidden_rows)}"
        )
    for class_id, expected_count in expected_class_quotas.items():
        actual_count = int(class_counts.get(class_id, 0))
        if actual_count != expected_count:
            structural_failures.append(
                "materialized_class_quota_mismatch:"
                f"class{class_id}:expected={expected_count}:actual={actual_count}"
            )

    construction_pack_payloads: dict[str, Any] = {}
    construction_pack_paths: dict[str, str] = {}
    for arm in ARM_ORDER:
        payload = {
            "kind": "phase2_job3_egohos_burned_construction_request_pack",
            "arm": arm,
            "dataset_role": "construction_pilot",
            "export_role": "construction_review_only",
            "scoring_permitted": False,
            "burned": True,
            "non_confirmatory": True,
            "rows_frozen": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "request_count": len(public_rows_by_arm[arm]),
            "requests": public_rows_by_arm[arm],
        }
        construction_pack_payloads[arm] = payload
        pack_rel = f"public/construction/{arm}_request_pack.json"
        construction_pack_paths[arm] = pack_rel
        pack_path = out_dir / pack_rel
        _write_json(pack_path, payload)
        _write_jsonl(pack_path.with_suffix(".jsonl"), payload["requests"])

    hidden_index = {
        "kind": "phase2_job3_egohos_burned_hidden_evaluator_index",
        "dataset_role": "construction_pilot",
        "burned": True,
        "non_confirmatory": True,
        "rows_frozen": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "answer_count": len(hidden_rows),
        "answers": hidden_rows,
    }
    _write_json(out_dir / "hidden" / "evaluator_index.json", hidden_index)
    _write_jsonl(out_dir / "hidden" / "evaluator_index.jsonl", hidden_rows)
    _write_jsonl(
        out_dir / "hidden" / "pending_negative_pre_review_rejections.jsonl",
        pre_review_rejections,
    )
    _write_jsonl(
        out_dir / "hidden" / "reference_crop_nonviable_rejections.jsonl",
        crop_nonviable_rejections,
    )
    _write_jsonl(
        out_dir / "hidden" / "pending_negative_human_rejections.jsonl",
        human_rejection_rows,
    )
    _write_jsonl(
        out_dir / "hidden" / "pending_negative_replacements.jsonl",
        replacement_rows,
    )

    pending_tsv_columns = [
        "candidate_id",
        "item_id",
        "class_id",
        "source_split",
        "source_frame_id",
        "source_video_id",
        "source_subject_id",
        "target_area",
        "hand_overlap_px",
        "hand_adjacent_px",
        "min_target_hand_pixel_distance_euclidean",
        "min_target_hand_pixel_distance_chebyshev",
        "target_hand_8_neighbor_adjacent",
        "review_asset_path",
        "review_asset_sha256",
        "decision",
        "reason",
        "reviewer_id",
        "reviewed_at",
        "rows_frozen",
    ]
    pending_queue_rel = "hidden/pending_negative_review/queue.tsv"
    _write_tsv(out_dir / pending_queue_rel, pending_tsv_columns, pending_rows)
    pending_sheet_paths = _build_pending_contact_sheets(
        rows=pending_rows,
        out_dir=out_dir,
        page_size=pending_contact_sheet_page_size,
    )

    public_leakage_failures = _scan_public_leakage(
        public_payloads=construction_pack_payloads,
        selected_candidates=emitted_candidates,
    )
    structural_failures.extend(public_leakage_failures)

    class_balance = {str(class_id): int(class_counts.get(class_id, 0)) for class_id in CLASS_IDS}
    unique_subject_count = len([key for key, count in subject_counts.items() if count > 0])
    unique_video_count = len([key for key, count in video_counts.items() if count > 0])
    prompt_checks_pass = all(
        row["pcs_system_sha256"] == row["mpp_system_sha256"]
        and row["pcs_user_sha256"] == row["mpp_user_sha256"]
        and row["vp_user_sha256"] == _sha256_text(USER_PROMPT_BASE + "\n" + VP_FROZEN_INSTRUCTION_PACKAGE)
        for row in prompt_audit_rows
    )
    rows_frozen_public_ok = all(not row["rows_frozen"] for row in hidden_rows)
    rows_frozen_public_ok = rows_frozen_public_ok and all(
        not request["rows_frozen"]
        for arm in ARM_ORDER
        for request in public_rows_by_arm[arm]
    )
    if not rows_frozen_public_ok:
        structural_failures.append("rows_frozen_not_false_everywhere")

    scoring_pack_payloads: dict[str, Any] = {}
    scoring_pack_paths: dict[str, str] = {}
    scoring_ready = (
        bool(hidden_rows)
        and len(hidden_rows) == target_total
        and all(bool(row["scoring_eligible"]) for row in hidden_rows)
        and len(structural_failures) == 0
    )
    if scoring_ready:
        for arm in ARM_ORDER:
            scoring_requests = [
                {
                    **request,
                    "export_role": "scoring",
                    "scoring_permitted": True,
                }
                for request in public_rows_by_arm[arm]
            ]
            scoring_pack_payloads[arm] = {
                "kind": "phase2_job3_egohos_burned_scoring_request_pack",
                "arm": arm,
                "dataset_role": "construction_pilot",
                "export_role": "scoring",
                "scoring_permitted": True,
                "eligibility_gate": "entire_construction_cohort_approved_for_scoring",
                "burned": True,
                "non_confirmatory": True,
                "rows_frozen": False,
                "claim_boundary": CLAIM_BOUNDARY,
                "request_count": len(scoring_requests),
                "requests": scoring_requests,
            }
        scoring_leakage_failures = _scan_public_leakage(
            public_payloads=scoring_pack_payloads,
            selected_candidates=emitted_candidates,
        )
        if scoring_leakage_failures:
            public_leakage_failures.extend(scoring_leakage_failures)
            structural_failures.extend(scoring_leakage_failures)
            scoring_pack_payloads = {}
            scoring_ready = False

    if scoring_ready:
        for arm in ARM_ORDER:
            pack_rel = f"public/scoring/{arm}_scoring_request_pack.json"
            scoring_pack_paths[arm] = pack_rel
            pack_path = out_dir / pack_rel
            _write_json(pack_path, scoring_pack_payloads[arm])
            _write_jsonl(
                pack_path.with_suffix(".jsonl"),
                scoring_pack_payloads[arm]["requests"],
            )

    if scoring_ready:
        scoring_export_status = "emitted_all_rows_eligible"
        scoring_block_reason = None
    elif structural_failures:
        scoring_export_status = "blocked"
        scoring_block_reason = "structural_failure"
    elif pending_rows:
        scoring_export_status = "blocked"
        scoring_block_reason = "pending_negative_human_audit"
    elif not hidden_rows:
        scoring_export_status = "blocked"
        scoring_block_reason = "empty_cohort"
    else:
        scoring_export_status = "blocked"
        scoring_block_reason = "ineligible_cohort"

    if structural_failures:
        construction_status = "structural_blocked"
    elif pending_rows:
        construction_status = "construction_pass_scoring_blocked_pending_negative_human_audit"
    elif not scoring_ready:
        construction_status = "construction_pass_scoring_blocked_ineligible_cohort"
    else:
        construction_status = "construction_pass_scoring_ready"

    audit = {
        "kind": "phase2_job3_egohos_burned_construction_audit",
        "created_at": _now(),
        "claim_boundary": CLAIM_BOUNDARY,
        "rows_frozen": False,
        "renderer_parameters": {
            "egohos_mask_renderer_primary": {
                "target_rgb": list(marker_renderer.DISCOVERY_STYLE.target_rgb),
                "hand_rgb": list(marker_renderer.DISCOVERY_STYLE.hand_rgb),
                "fill_alpha": marker_renderer.DISCOVERY_STYLE.fill_alpha,
                "mask_outline_width": marker_renderer.DISCOVERY_STYLE.mask_outline_width,
                "bbox_width": marker_renderer.DISCOVERY_STYLE.bbox_width,
                "labels": [
                    marker_renderer.DISCOVERY_STYLE.target_label,
                    marker_renderer.DISCOVERY_STYLE.hand_label,
                ],
            },
            "alternate_descriptive": {
                "target_rgb": list(marker_renderer.ALTERNATE_STYLE.target_rgb),
                "hand_rgb": list(marker_renderer.ALTERNATE_STYLE.hand_rgb),
                "fill_alpha": marker_renderer.ALTERNATE_STYLE.fill_alpha,
                "mask_outline_width": marker_renderer.ALTERNATE_STYLE.mask_outline_width,
                "bbox_width": marker_renderer.ALTERNATE_STYLE.bbox_width,
            },
            "cross_dataset_byte_equivalence_claimed": False,
            "visor_polygon_equivalence_claimed": False,
        },
        "selection_target": selection_target,
        "effective_target_total": target_total,
        "selected_count": len(hidden_rows),
        "candidate_pool_count": len(candidates),
        "source_lock": {
            "required_source_family": SOURCE_FAMILY_LOCK,
            "required_confirmation_provenance_status": PROVENANCE_STATUS_LOCK,
            "selected_rows_all_locked": all(
                row["source_family"] == SOURCE_FAMILY_LOCK
                and row["confirmation_provenance_status"] == PROVENANCE_STATUS_LOCK
                for row in hidden_rows
            ),
        },
        "prompt_bytes": {
            "pcs_mpp_identical": prompt_checks_pass,
            "vp_instruction_package_sha256": _sha256_text(VP_FROZEN_INSTRUCTION_PACKAGE),
            "vp_delta_only_count": vp_instruction_delta_only_count,
        },
        "pixel_invariants": {
            "mpp_vp_full_frame_equal_count": mpp_vp_equal_count,
            "mpp_vp_full_frame_not_equal_count": mpp_vp_not_equal_count,
            "intended_pixel_confinement_failure_count": intended_confinement_failures,
            "alternate_marker_generated_count": len(hidden_rows),
        },
        "crop_invariants": {
            "interior_rule": "uniform_square_3x3_binary_erosion",
            "erosion_radius_pixels": REFERENCE_CROP_EROSION_RADIUS,
            "minimum_retained_pixel_count": REFERENCE_CROP_RETAINED_AREA_FLOOR,
            "bbox_source": "retained_mask",
            "shared_reference_crop_count": len(hidden_rows),
            "nonviable_preselection_rejection_count": len(crop_nonviable_rejections),
            "crop_audits": crop_audits,
        },
        "distribution": {
            "class_counts": class_balance,
            "subject_count": unique_subject_count,
            "video_count": unique_video_count,
            "subject_counts": dict(sorted(subject_counts.items())),
            "video_counts": dict(sorted(video_counts.items())),
        },
        "duplicates": {
            "duplicate_item_id_count": len(item_ids) - len(set(item_ids)),
            "duplicate_request_id_count": len(request_ids) - len(set(request_ids)),
            "duplicate_asset_path_count": len(asset_paths) - len(set(asset_paths)),
            "duplicate_source_frame_count": duplicate_source_frames,
        },
        "pending_negative": {
            "pending_count": len(pending_rows),
            "pre_review_adjacency_rejection_count": len(pre_review_rejections),
            "human_rejection_count": len(human_rejection_rows),
            "replacement_event_count": len(replacement_rows),
            "replacement_spare_exhaustion_count": sum(
                row["replacement_status"] == "spare_pool_exhausted"
                for row in replacement_rows
            ),
            "queue_tsv": pending_queue_rel,
            "contact_sheet_pages": pending_sheet_paths,
        },
        "pack_exports": {
            "construction_role": "construction_review_only",
            "construction_paths": construction_pack_paths,
            "scoring_status": scoring_export_status,
            "scoring_block_reason": scoring_block_reason,
            "scoring_paths": scoring_pack_paths,
            "entire_cohort_gate": True,
        },
        "public_leakage_failure_count": len(public_leakage_failures),
        "raw_input_immutability_ok": all(
            source_after_sha.get(path) == before for path, before in source_before_sha.items()
        ),
        "rows_frozen_false_everywhere": rows_frozen_public_ok,
        "structural_failure_count": len(structural_failures),
        "structural_failures": sorted(structural_failures),
    }
    _write_json(out_dir / "audits" / "construction_audit.json", audit)
    _write_jsonl(out_dir / "audits" / "prompt_audit.jsonl", prompt_audit_rows)
    _write_jsonl(out_dir / "audits" / "item_audit.jsonl", item_audit_rows)

    manifest = {
        "kind": "phase2_job3_egohos_burned_construction_manifest",
        "created_at": _now(),
        "claim_boundary": CLAIM_BOUNDARY,
        "data_root": str(data_root),
        "inventory_path": str(inventory_path),
        "candidate_sidecar_path": str(candidate_sidecar_path) if candidate_sidecar_path else None,
        "negative_audit_sidecar_path": (
            str(negative_audit_sidecar_path) if negative_audit_sidecar_path else None
        ),
        "negative_audit_validation_receipt_path": (
            str(negative_audit_receipt_path) if negative_audit_receipt_path else None
        ),
        "negative_audit_validation": negative_audit_validation,
        "selection_target": selection_target,
        "effective_target_total": target_total,
        "selected_count": len(hidden_rows),
        "class_counts": class_balance,
        "outputs": {
            "public": {
                "construction": construction_pack_paths,
                "scoring": scoring_pack_paths,
            },
            "hidden_evaluator_index": "hidden/evaluator_index.json",
            "pending_negative_queue": pending_queue_rel,
            "pending_negative_pre_review_rejections": (
                "hidden/pending_negative_pre_review_rejections.jsonl"
            ),
            "reference_crop_nonviable_rejections": (
                "hidden/reference_crop_nonviable_rejections.jsonl"
            ),
            "pending_negative_human_rejections": (
                "hidden/pending_negative_human_rejections.jsonl"
            ),
            "pending_negative_replacements": (
                "hidden/pending_negative_replacements.jsonl"
            ),
            "pending_negative_contact_sheets": pending_sheet_paths,
            "construction_audit": "audits/construction_audit.json",
            "prompt_audit_jsonl": "audits/prompt_audit.jsonl",
            "item_audit_jsonl": "audits/item_audit.jsonl",
        },
        "scoring_export_status": scoring_export_status,
        "scoring_export_block_reason": scoring_block_reason,
        "construction_status": construction_status,
        "rows_frozen": False,
    }
    _write_json(out_dir / "manifest.json", manifest)

    return {
        "manifest": manifest,
        "audit": audit,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/external/egohos"),
        help="EgoHOS root containing raw/ and manifests/.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="Frame inventory TSV (default: <data-root>/manifests/frame_inventory.tsv).",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--candidate-sidecar",
        type=Path,
        default=None,
        help="Optional Stage A candidate sidecar JSONL for cross-checking.",
    )
    parser.add_argument(
        "--negative-audit-sidecar",
        type=Path,
        default=None,
        help="Optional human audit JSONL with no_direct_contact_approved rows.",
    )
    parser.add_argument(
        "--negative-audit-validation-receipt",
        type=Path,
        default=None,
        help=(
            "Immutable content-addressed pass receipt required whenever "
            "--negative-audit-sidecar is supplied."
        ),
    )
    parser.add_argument(
        "--selection-target",
        type=int,
        default=DEFAULT_SELECTION_TARGET,
        help=f"Default selection target (default: {DEFAULT_SELECTION_TARGET}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke cap while preserving class round-robin behavior.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=DEFAULT_MIN_COMPONENT_AREA,
        help=f"Minimum target component area in pixels (default: {DEFAULT_MIN_COMPONENT_AREA}).",
    )
    parser.add_argument(
        "--pending-contact-sheet-page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"Pending queue contact sheet rows per page (default: {DEFAULT_PAGE_SIZE}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    inventory = (
        args.inventory.expanduser().resolve()
        if args.inventory is not None
        else (data_root / "manifests" / "frame_inventory.tsv")
    )
    report = build_burned_items(
        data_root=data_root,
        inventory_path=inventory,
        out_dir=args.out_dir.expanduser().resolve(),
        candidate_sidecar_path=args.candidate_sidecar.expanduser().resolve()
        if args.candidate_sidecar
        else None,
        negative_audit_sidecar_path=args.negative_audit_sidecar.expanduser().resolve()
        if args.negative_audit_sidecar
        else None,
        negative_audit_receipt_path=(
            args.negative_audit_validation_receipt.expanduser().resolve()
            if args.negative_audit_validation_receipt
            else None
        ),
        selection_target=args.selection_target,
        limit=args.limit,
        min_component_area=args.min_component_area,
        pending_contact_sheet_page_size=args.pending_contact_sheet_page_size,
    )
    print(json.dumps(report["manifest"], indent=2, sort_keys=True))
    status = report["manifest"]["construction_status"]
    return 1 if status == "structural_blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
