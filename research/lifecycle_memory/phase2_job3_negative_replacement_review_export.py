#!/usr/bin/env python3
"""Validate a burned-data replacement review and issue a cumulative audit receipt."""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence

from research.lifecycle_memory import phase2_job3_negative_review_export as parent_validator


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MANIFEST_KIND = (
    "phase2_job3_pending_negative_replacement_review_tool_manifest"
)
EXPECTED_MANIFEST_VERSION = 2
EXPECTED_MANIFEST_STATUS = "construction_pilot_replacement_review_only"
EXPECTED_MANIFEST_FILENAME = "replacement_review_tool_manifest_v2.json"
EXPECTED_TOOL_FILENAME = "replacement_review_tool_v2.html"
EXPECTED_CONSTRUCTION_MANIFEST_KIND = (
    "phase2_job3_egohos_burned_construction_manifest"
)
EXPECTED_CONSTRUCTION_AUDIT_KIND = (
    "phase2_job3_egohos_burned_construction_audit"
)
EXPECTED_REPLACEMENT_BINDINGS = {
    "queue",
    "schema",
    "generator",
    "base_generator",
    "tool",
    "parent_validation_receipt",
    "source_construction_manifest",
    "construction_audit",
    "replacement_ledger",
    "human_rejections",
    "images",
}
VALIDATION_KIND = (
    "phase2_job3_pending_negative_replacement_cumulative_validation"
)
VALIDATION_VERSION = 1
CUMULATIVE_PREFIX = "pending_negative_review_CUMULATIVE_"
RECEIPT_PREFIX = "pending_negative_replacement_cumulative_validation_"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_FUTURE_TIMESTAMP_SKEW = timedelta(minutes=5)


class ReplacementReviewExportError(ValueError):
    """Raised when replacement or parent review provenance fails validation."""


@dataclass(frozen=True)
class PreparedCumulativeValidation:
    """Fully validated material awaiting immutable cumulative-file issuance."""

    cumulative_bytes: bytes
    payload_without_cumulative: dict[str, Any]


@dataclass(frozen=True)
class IssuedCumulativeValidation:
    """Paths and final payload produced by immutable issuance."""

    cumulative_path: Path
    receipt_path: Path
    payload: dict[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path, *, label: str) -> bytes:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReplacementReviewExportError(f"{label} does not exist: {resolved}")
    return resolved.read_bytes()


def _load_json(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplacementReviewExportError(
            f"{label} is not valid UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ReplacementReviewExportError(f"{label} must contain a JSON object")
    return value


def _load_jsonl(data: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplacementReviewExportError(f"{label} is not valid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ReplacementReviewExportError(
                f"{label} contains blank line {line_number}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplacementReviewExportError(
                f"{label} line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ReplacementReviewExportError(
                f"{label} line {line_number} must contain a JSON object"
            )
        rows.append(row)
    return rows


def _canonical_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("ascii")
        for row in rows
    )


def _candidate_order_sha256(candidate_ids: list[str]) -> str:
    return _sha256_bytes(("\n".join(candidate_ids) + "\n").encode("utf-8"))


def _canonical_object_sha256(value: dict[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )


def _load_embedded_review_data(tool_bytes: bytes) -> dict[str, Any]:
    try:
        text = tool_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplacementReviewExportError(
            "replacement tool is not valid UTF-8"
        ) from exc
    matches = re.findall(
        r'<script id="review-data" type="application/json">(.*?)</script>',
        text,
        flags=re.DOTALL,
    )
    if len(matches) != 1:
        raise ReplacementReviewExportError(
            "replacement tool must contain exactly one review-data payload"
        )
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise ReplacementReviewExportError(
            "replacement tool review-data payload is invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ReplacementReviewExportError(
            "replacement tool review-data payload must be an object"
        )
    return payload


def _file_record(path: Path, data: bytes | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    content = _read_bytes(resolved, label=str(resolved)) if data is None else data
    return {
        "path": str(resolved),
        "sha256": _sha256_bytes(content),
        "bytes": len(content),
    }


def _verify_record(
    record: Any,
    *,
    label: str,
    current_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ReplacementReviewExportError(f"{label} record must be an object")
    recorded_path = record.get("path")
    expected_sha256 = record.get("sha256")
    if not isinstance(recorded_path, str) or not recorded_path.strip():
        raise ReplacementReviewExportError(f"{label} record has no path")
    if not isinstance(expected_sha256, str) or SHA256_RE.fullmatch(expected_sha256) is None:
        raise ReplacementReviewExportError(f"{label} record has an invalid SHA-256")
    path = (
        current_path.expanduser().resolve()
        if current_path is not None
        else Path(recorded_path).expanduser().resolve()
    )
    data = _read_bytes(path, label=label)
    actual_sha256 = _sha256_bytes(data)
    if actual_sha256 != expected_sha256:
        raise ReplacementReviewExportError(f"{label} hash mismatch: {path}")
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and expected_bytes != len(data):
        raise ReplacementReviewExportError(f"{label} byte count mismatch: {path}")
    return _file_record(path, data)


def _binding_record(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    bindings = manifest.get("bindings")
    record = bindings.get(key) if isinstance(bindings, dict) else None
    if not isinstance(record, dict):
        raise ReplacementReviewExportError(f"manifest is missing bindings.{key}")
    path = record.get("path")
    sha256 = record.get("sha256")
    byte_count = record.get("bytes")
    if not isinstance(path, str) or not path.strip():
        raise ReplacementReviewExportError(f"manifest bindings.{key}.path is invalid")
    if not isinstance(sha256, str) or SHA256_RE.fullmatch(sha256) is None:
        raise ReplacementReviewExportError(
            f"manifest bindings.{key}.sha256 is invalid"
        )
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count <= 0
    ):
        raise ReplacementReviewExportError(
            f"manifest bindings.{key}.bytes is invalid"
        )
    return record


def _resolve_relative(*, base: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ReplacementReviewExportError(f"manifest {label} path must be relative")
    resolved_base = base.resolve()
    resolved = (resolved_base / relative).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ReplacementReviewExportError(
            f"manifest {label} path escapes its allowed root"
        ) from exc
    return resolved


def _verify_bound_file(
    *,
    manifest: dict[str, Any],
    key: str,
    base: Path,
    label: str,
    current_path: Path | None = None,
) -> dict[str, Any]:
    binding = _binding_record(manifest, key)
    bound_path = _resolve_relative(
        base=base,
        value=str(binding["path"]),
        label=label,
    )
    if current_path is not None and bound_path != current_path.expanduser().resolve():
        raise ReplacementReviewExportError(
            f"manifest {label} path does not identify the supplied file"
        )
    return _verify_record(binding, label=label, current_path=bound_path)


def _parse_timestamp(value: Any, *, candidate_id: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ReplacementReviewExportError(f"candidate {candidate_id} has no reviewed_at")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ReplacementReviewExportError(
            f"candidate {candidate_id} has invalid reviewed_at"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReplacementReviewExportError(
            f"candidate {candidate_id} reviewed_at must include a timezone"
        )
    parsed_utc = parsed.astimezone(timezone.utc)
    if parsed_utc > datetime.now(timezone.utc) + MAX_FUTURE_TIMESTAMP_SKEW:
        raise ReplacementReviewExportError(
            f"candidate {candidate_id} reviewed_at is implausibly in the future"
        )
    return parsed_utc


def _without_validation_time(payload: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(payload))
    copied.pop("validated_at_utc", None)
    return copied


def _verify_parent_round(
    *,
    export_path: Path,
    receipt_path: Path,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    export_path = export_path.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    receipt_bytes = _read_bytes(receipt_path, label="parent validation receipt")
    receipt = _load_json(receipt_bytes, label="parent validation receipt")
    if receipt.get("kind") != parent_validator.VALIDATION_KIND:
        raise ReplacementReviewExportError("parent validation receipt kind is invalid")
    if receipt.get("version") != parent_validator.VALIDATION_VERSION:
        raise ReplacementReviewExportError("parent validation receipt version is invalid")
    if receipt.get("status") != "pass":
        raise ReplacementReviewExportError("parent validation receipt status is not pass")
    _parse_timestamp(
        receipt.get("validated_at_utc"),
        candidate_id="parent-validation-receipt",
    )
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict):
        raise ReplacementReviewExportError("parent validation receipt inputs are invalid")
    required = {"export", "queue", "schema", "manifest", "generator", "tool", "images"}
    if set(inputs) != required:
        raise ReplacementReviewExportError("parent validation receipt inputs are incomplete")

    export_record = _verify_record(
        inputs["export"],
        label="parent export",
        current_path=export_path,
    )
    expected_receipt_name = (
        f"{parent_validator.RECEIPT_PREFIX}{export_record['sha256']}.json"
    )
    if receipt.get("receipt_filename") != expected_receipt_name or receipt_path.name != expected_receipt_name:
        raise ReplacementReviewExportError(
            "parent validation receipt is not correctly content-addressed"
        )

    try:
        fresh = parent_validator.validate_review_export(
            export_path=export_path,
            queue_path=Path(inputs["queue"]["path"]),
            schema_path=Path(inputs["schema"]["path"]),
            manifest_path=Path(inputs["manifest"]["path"]),
            repo_root=repo_root,
        )
    except (KeyError, TypeError, parent_validator.ReviewExportError) as exc:
        raise ReplacementReviewExportError(
            f"parent review bundle failed fresh validation: {exc}"
        ) from exc
    if _without_validation_time(receipt) != _without_validation_time(fresh):
        raise ReplacementReviewExportError(
            "parent validation receipt does not match a fresh validation"
        )

    parent_rows = _load_jsonl(_read_bytes(export_path, label="parent export"), label="parent export")
    parent_inputs: dict[str, Any] = {
        "export": {**export_record, "filename": export_path.name},
        "validation_receipt": _file_record(receipt_path, receipt_bytes),
    }
    for key in ("queue", "schema", "manifest", "generator", "tool"):
        parent_inputs[key] = _verify_record(inputs[key], label=f"parent {key}")
    parent_images: list[dict[str, Any]] = []
    image_rows = inputs["images"]
    if not isinstance(image_rows, list) or len(image_rows) != len(parent_rows):
        raise ReplacementReviewExportError("parent receipt image bindings are incomplete")
    for index, image in enumerate(image_rows):
        candidate_id = image.get("candidate_id") if isinstance(image, dict) else None
        if not isinstance(candidate_id, str):
            raise ReplacementReviewExportError(
                f"parent image binding {index} has no candidate_id"
            )
        verified = _verify_record(image, label=f"parent image {candidate_id}")
        verified["candidate_id"] = candidate_id
        parent_images.append(verified)
    parent_inputs["images"] = parent_images
    summary = {
        "round": "parent",
        "claim_boundary": receipt["claim_boundary"],
        "candidate_count": receipt["candidate_count"],
        "candidate_order_sha256": receipt["candidate_order_sha256"],
        "class_counts": receipt["class_counts"],
        "decision_counts": receipt["decision_counts"],
        "class_decision_counts": receipt["class_decision_counts"],
        "reviewer_ids": receipt["reviewer_ids"],
        "inputs": parent_inputs,
    }
    return parent_rows, summary


def _parse_expected_class_counts(manifest: dict[str, Any]) -> dict[int, int]:
    raw = manifest.get("class_counts")
    if not isinstance(raw, dict) or not raw:
        raise ReplacementReviewExportError("replacement manifest class_counts is invalid")
    parsed: dict[int, int] = {}
    for key, count in raw.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ReplacementReviewExportError(
                "replacement manifest class_counts has a noninteger key"
            ) from exc
        if str(class_id) != str(key) or not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ReplacementReviewExportError(
                "replacement manifest class_counts values must be positive integers"
            )
        parsed[class_id] = count
    return parsed


def _load_replacement_queue(data: bytes) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplacementReviewExportError("replacement queue is not valid UTF-8") from exc
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    required = {"candidate_id", "class_id", "review_asset_sha256", "rows_frozen"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ReplacementReviewExportError("replacement queue is missing required columns")
    return list(reader)


def _validate_replacement_round(
    *,
    export_path: Path,
    queue_path: Path,
    schema_path: Path,
    manifest_path: Path,
    parent_summary: dict[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    export_path = export_path.expanduser().resolve()
    queue_path = queue_path.expanduser().resolve()
    schema_path = schema_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    export_bytes = _read_bytes(export_path, label="replacement export")
    queue_bytes = _read_bytes(queue_path, label="replacement queue")
    schema_bytes = _read_bytes(schema_path, label="replacement schema")
    manifest_bytes = _read_bytes(manifest_path, label="replacement manifest")
    export_rows = _load_jsonl(export_bytes, label="replacement export")
    queue_rows = _load_replacement_queue(queue_bytes)
    schema = _load_json(schema_bytes, label="replacement schema")
    manifest = _load_json(manifest_bytes, label="replacement manifest")

    if manifest_path.name != EXPECTED_MANIFEST_FILENAME:
        raise ReplacementReviewExportError("replacement manifest filename is invalid")

    if schema.get("kind") != parent_validator.EXPECTED_SCHEMA_KIND:
        raise ReplacementReviewExportError("replacement schema kind is invalid")
    if schema.get("version") != parent_validator.EXPECTED_SCHEMA_VERSION:
        raise ReplacementReviewExportError("replacement schema version is invalid")
    if schema.get("status") != parent_validator.EXPECTED_SCHEMA_STATUS:
        raise ReplacementReviewExportError("replacement schema status is invalid")
    allowed_classes = schema.get("allowed_mask_classes")
    if not isinstance(allowed_classes, list) or not all(isinstance(value, int) for value in allowed_classes):
        raise ReplacementReviewExportError("replacement schema allowed_mask_classes is invalid")
    if schema.get("rows_frozen") is not False:
        raise ReplacementReviewExportError("replacement schema rows_frozen must be false")
    claim_boundary = schema.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise ReplacementReviewExportError("replacement schema claim_boundary is invalid")

    if manifest.get("kind") != EXPECTED_MANIFEST_KIND:
        raise ReplacementReviewExportError("replacement manifest kind is invalid")
    if manifest.get("version") != EXPECTED_MANIFEST_VERSION:
        raise ReplacementReviewExportError("replacement manifest version is invalid")
    if manifest.get("status") != EXPECTED_MANIFEST_STATUS:
        raise ReplacementReviewExportError("replacement manifest status is invalid")
    if manifest.get("claim_boundary") != claim_boundary:
        raise ReplacementReviewExportError("replacement manifest claim_boundary is invalid")
    if claim_boundary != parent_summary["claim_boundary"]:
        raise ReplacementReviewExportError(
            "replacement claim_boundary does not match the parent review"
        )
    if manifest.get("rows_frozen") is not False:
        raise ReplacementReviewExportError("replacement manifest rows_frozen must be false")
    round_index = manifest.get("round_index")
    if (
        not isinstance(round_index, int)
        or isinstance(round_index, bool)
        or round_index <= 0
        or manifest.get("replacement_round") != round_index
    ):
        raise ReplacementReviewExportError("replacement manifest replacement_round is invalid")
    required_reviewer_id = manifest.get("required_reviewer_id")
    parent_reviewer_ids = parent_summary.get("reviewer_ids")
    if (
        not isinstance(required_reviewer_id, str)
        or not required_reviewer_id.strip()
        or parent_reviewer_ids != [required_reviewer_id]
    ):
        raise ReplacementReviewExportError(
            "replacement manifest required_reviewer_id does not match the parent reviewer"
        )
    candidate_count = manifest.get("candidate_count")
    if not isinstance(candidate_count, int) or isinstance(candidate_count, bool) or candidate_count <= 0:
        raise ReplacementReviewExportError("replacement manifest candidate_count is invalid")
    expected_class_counts = _parse_expected_class_counts(manifest)
    if sum(expected_class_counts.values()) != candidate_count:
        raise ReplacementReviewExportError(
            "replacement manifest class_counts do not sum to candidate_count"
        )
    if not set(expected_class_counts).issubset(set(allowed_classes)):
        raise ReplacementReviewExportError(
            "replacement manifest class_counts include a disallowed class"
        )
    if len(queue_rows) != candidate_count or len(export_rows) != candidate_count:
        raise ReplacementReviewExportError(
            "replacement queue/export count does not match the manifest"
        )

    parent_binding = manifest.get("parent_validation")
    parent_receipt = parent_summary["inputs"]["validation_receipt"]
    parent_export = parent_summary["inputs"]["export"]
    expected_parent_binding = {
        "receipt_sha256": parent_receipt["sha256"],
        "receipt_filename": Path(parent_receipt["path"]).name,
        "export_sha256": parent_export["sha256"],
        "candidate_order_sha256": parent_summary["candidate_order_sha256"],
    }
    if parent_binding != expected_parent_binding:
        raise ReplacementReviewExportError(
            "replacement manifest parent_validation binding is invalid"
        )
    parent_receipt_payload = _load_json(
        _read_bytes(
            Path(parent_receipt["path"]), label="parent validation receipt"
        ),
        label="parent validation receipt",
    )
    parent_validated_at = _parse_timestamp(
        parent_receipt_payload.get("validated_at_utc"),
        candidate_id="parent-validation-receipt",
    )

    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != EXPECTED_REPLACEMENT_BINDINGS:
        raise ReplacementReviewExportError("replacement manifest bindings are incomplete")
    queue_record = _verify_bound_file(
        manifest=manifest,
        key="queue",
        base=manifest_path.parent,
        label="replacement queue",
        current_path=queue_path,
    )
    schema_record = _verify_bound_file(
        manifest=manifest,
        key="schema",
        base=repo_root,
        label="replacement schema",
        current_path=schema_path,
    )
    generator_record = _verify_bound_file(
        manifest=manifest,
        key="generator",
        base=repo_root,
        label="replacement generator",
    )
    base_generator_record = _verify_bound_file(
        manifest=manifest,
        key="base_generator",
        base=repo_root,
        label="replacement base generator",
    )
    tool_record = _verify_bound_file(
        manifest=manifest,
        key="tool",
        base=manifest_path.parent,
        label="replacement tool",
    )
    if Path(tool_record["path"]).name != EXPECTED_TOOL_FILENAME:
        raise ReplacementReviewExportError("replacement tool filename is invalid")
    parent_receipt_record = _verify_bound_file(
        manifest=manifest,
        key="parent_validation_receipt",
        base=repo_root,
        label="parent validation receipt",
        current_path=Path(parent_receipt["path"]),
    )
    if parent_receipt_record != parent_receipt:
        raise ReplacementReviewExportError(
            "replacement manifest parent receipt record is inconsistent"
        )
    construction_record = _verify_bound_file(
        manifest=manifest,
        key="source_construction_manifest",
        base=repo_root,
        label="source construction manifest",
    )
    audit_record = _verify_bound_file(
        manifest=manifest,
        key="construction_audit",
        base=repo_root,
        label="construction audit",
    )
    replacement_ledger_record = _verify_bound_file(
        manifest=manifest,
        key="replacement_ledger",
        base=repo_root,
        label="replacement ledger",
    )
    human_rejections_record = _verify_bound_file(
        manifest=manifest,
        key="human_rejections",
        base=repo_root,
        label="human rejection ledger",
    )
    construction = _load_json(
        _read_bytes(Path(construction_record["path"]), label="source construction manifest"),
        label="source construction manifest",
    )
    if construction.get("kind") != EXPECTED_CONSTRUCTION_MANIFEST_KIND:
        raise ReplacementReviewExportError("source construction manifest kind is invalid")
    if (
        construction.get("rows_frozen") is not False
        or construction.get("scoring_export_status") != "blocked"
        or construction.get("scoring_export_block_reason")
        != "pending_negative_human_audit"
    ):
        raise ReplacementReviewExportError(
            "source construction manifest is not a blocked, unfrozen review build"
        )
    embedded_parent = construction.get("negative_audit_validation")
    if (
        not isinstance(embedded_parent, dict)
        or embedded_parent.get("receipt_sha256") != parent_receipt["sha256"]
        or embedded_parent.get("receipt_filename") != Path(parent_receipt["path"]).name
    ):
        raise ReplacementReviewExportError(
            "source construction manifest does not bind the parent receipt"
        )
    outputs = construction.get("outputs")
    if not isinstance(outputs, dict):
        raise ReplacementReviewExportError(
            "source construction manifest outputs are invalid"
        )
    construction_root = Path(construction_record["path"]).parent
    expected_outputs = {
        "pending_negative_queue": queue_path,
        "construction_audit": Path(audit_record["path"]),
        "pending_negative_replacements": Path(replacement_ledger_record["path"]),
        "pending_negative_human_rejections": Path(human_rejections_record["path"]),
    }
    for output_key, expected_path in expected_outputs.items():
        relative = outputs.get(output_key)
        if (
            not isinstance(relative, str)
            or _resolve_relative(
                base=construction_root,
                value=relative,
                label=f"construction output {output_key}",
            )
            != expected_path.resolve()
        ):
            raise ReplacementReviewExportError(
                f"source construction manifest does not identify {output_key}"
            )
    if outputs.get("pending_negative_queue") != "hidden/pending_negative_review/queue.tsv":
        raise ReplacementReviewExportError(
            "source construction manifest does not identify the replacement queue"
        )

    audit = _load_json(
        _read_bytes(Path(audit_record["path"]), label="construction audit"),
        label="construction audit",
    )
    if audit.get("kind") != EXPECTED_CONSTRUCTION_AUDIT_KIND:
        raise ReplacementReviewExportError("construction audit kind is invalid")
    pending = audit.get("pending_negative")
    if (
        not isinstance(pending, dict)
        or pending.get("pending_count") != candidate_count
        or pending.get("replacement_event_count") != candidate_count
        or pending.get("replacement_spare_exhaustion_count") != 0
    ):
        raise ReplacementReviewExportError(
            "construction audit replacement counts are inconsistent"
        )
    replacement_ledger_rows = _load_jsonl(
        _read_bytes(Path(replacement_ledger_record["path"]), label="replacement ledger"),
        label="replacement ledger",
    )
    human_rejection_rows = _load_jsonl(
        _read_bytes(Path(human_rejections_record["path"]), label="human rejection ledger"),
        label="human rejection ledger",
    )
    if len(replacement_ledger_rows) != candidate_count or not human_rejection_rows:
        raise ReplacementReviewExportError(
            "replacement or human-rejection ledger count is invalid"
        )

    local_binding = manifest.get("local_storage_binding")
    binding_material = {
        "round_index": round_index,
        "queue_sha256": queue_record["sha256"],
        "schema_sha256": schema_record["sha256"],
        "parent_receipt_sha256": parent_receipt_record["sha256"],
        "construction_manifest_sha256": construction_record["sha256"],
        "construction_audit_sha256": audit_record["sha256"],
        "replacement_ledger_sha256": replacement_ledger_record["sha256"],
        "human_rejections_sha256": human_rejections_record["sha256"],
        "generator_sha256": generator_record["sha256"],
        "base_generator_sha256": base_generator_record["sha256"],
    }
    expected_local_binding = {
        **binding_material,
        "storage_binding_sha256": _canonical_object_sha256(binding_material),
    }
    if local_binding != expected_local_binding:
        raise ReplacementReviewExportError(
            "replacement manifest local_storage_binding is invalid"
        )

    expected_filename = (
        f"pending_negative_replacement_round{round_index}_FINAL_"
        f"{queue_record['sha256'][:12]}.jsonl"
    )
    if manifest.get("export_filename") != expected_filename:
        raise ReplacementReviewExportError(
            "replacement manifest export_filename is invalid"
        )
    expected_stem = Path(expected_filename).stem
    if re.fullmatch(
        rf"{re.escape(expected_stem)}(?: \([1-9][0-9]*\))?\.jsonl",
        export_path.name,
    ) is None:
        raise ReplacementReviewExportError(
            "replacement export filename is not final and queue-bound"
        )

    allowed_decisions = schema.get("allowed_decisions")
    required_fields = schema.get("decision_record_required_fields")
    if not isinstance(allowed_decisions, list) or not all(isinstance(value, str) for value in allowed_decisions):
        raise ReplacementReviewExportError("replacement schema allowed_decisions is invalid")
    if not isinstance(required_fields, list) or not all(isinstance(value, str) for value in required_fields):
        raise ReplacementReviewExportError(
            "replacement schema decision_record_required_fields is invalid"
        )
    expected_fields = set(required_fields) | {"rows_frozen"}

    queue_by_id: dict[str, dict[str, str]] = {}
    queue_order: list[str] = []
    queue_class_counts: Counter[int] = Counter()
    for row in queue_rows:
        candidate_id = row["candidate_id"].strip()
        asset_sha256 = row["review_asset_sha256"].strip()
        if SHA256_RE.fullmatch(candidate_id) is None or candidate_id in queue_by_id:
            raise ReplacementReviewExportError(
                f"replacement queue candidate_id is invalid or duplicated: {candidate_id!r}"
            )
        if SHA256_RE.fullmatch(asset_sha256) is None:
            raise ReplacementReviewExportError(
                f"replacement queue asset hash is invalid for {candidate_id}"
            )
        if row["rows_frozen"].strip().lower() != "false":
            raise ReplacementReviewExportError(
                f"replacement queue row {candidate_id} is unexpectedly frozen"
            )
        try:
            class_id = int(row["class_id"])
        except ValueError as exc:
            raise ReplacementReviewExportError(
                f"replacement queue class_id is invalid for {candidate_id}"
            ) from exc
        queue_by_id[candidate_id] = row
        queue_order.append(candidate_id)
        queue_class_counts[class_id] += 1
    if dict(queue_class_counts) != expected_class_counts:
        raise ReplacementReviewExportError(
            "replacement queue class counts do not match the manifest"
        )
    if manifest.get("candidate_order_sha256") != _candidate_order_sha256(queue_order):
        raise ReplacementReviewExportError(
            "replacement manifest candidate order does not match the queue"
        )

    parent_export_rows = _load_jsonl(
        _read_bytes(
            Path(parent_summary["inputs"]["export"]["path"]),
            label="parent export",
        ),
        label="parent export",
    )
    parent_queue_rows = _load_replacement_queue(
        _read_bytes(
            Path(parent_summary["inputs"]["queue"]["path"]),
            label="parent queue",
        )
    )
    parent_class_by_id = {
        row["candidate_id"].strip(): int(row["class_id"])
        for row in parent_queue_rows
    }
    parent_rejections = {
        str(row["candidate_id"]): row
        for row in parent_export_rows
        if str(row.get("decision") or "").startswith("reject_")
    }
    if len(parent_rejections) != len(human_rejection_rows):
        raise ReplacementReviewExportError(
            "human rejection ledger count does not match the parent export"
        )

    human_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(human_rejection_rows, 1):
        candidate_id = row.get("candidate_id")
        parent_row = parent_rejections.get(candidate_id)
        if (
            not isinstance(candidate_id, str)
            or parent_row is None
            or candidate_id in human_by_id
            or row.get("class_id") != parent_class_by_id.get(candidate_id)
            or row.get("rows_frozen") is not False
        ):
            raise ReplacementReviewExportError(
                f"human rejection ledger row {index} does not match the parent audit"
            )
        for field in (
            "candidate_id",
            "decision",
            "reason",
            "reviewer_id",
            "reviewed_at",
            "review_asset_sha256",
            "rows_frozen",
        ):
            if row.get(field) != parent_row.get(field):
                raise ReplacementReviewExportError(
                    f"human rejection ledger row {index} differs from the parent export"
                )
        human_by_id[candidate_id] = row
    if set(human_by_id) != set(parent_rejections):
        raise ReplacementReviewExportError(
            "human rejection ledger does not contain every parent rejection"
        )

    replacement_by_id: dict[str, dict[str, Any]] = {}
    replaced_parent_ids: set[str] = set()
    for index, row in enumerate(replacement_ledger_rows, 1):
        replacement_id = row.get("replacement_candidate_id")
        rejected_id = row.get("rejected_candidate_id")
        class_id = row.get("class_id")
        if (
            not isinstance(replacement_id, str)
            or replacement_id not in queue_by_id
            or replacement_id in replacement_by_id
            or class_id != int(queue_by_id[replacement_id]["class_id"])
            or not isinstance(rejected_id, str)
            or rejected_id not in human_by_id
            or rejected_id in replaced_parent_ids
            or human_by_id[rejected_id].get("class_id") != class_id
            or row.get("replacement_status")
            != "scheduled_from_frozen_ranked_spare_pool"
            or row.get("rows_frozen") is not False
        ):
            raise ReplacementReviewExportError(
                f"replacement ledger row {index} does not match the queue and parent rejection"
            )
        replacement_by_id[replacement_id] = row
        replaced_parent_ids.add(rejected_id)
    if set(replacement_by_id) != set(queue_by_id) or replaced_parent_ids != set(human_by_id):
        raise ReplacementReviewExportError(
            "replacement ledger does not close the queue and parent rejections"
        )

    image_bindings = bindings["images"]
    if not isinstance(image_bindings, list) or len(image_bindings) != candidate_count:
        raise ReplacementReviewExportError(
            "replacement manifest image bindings have an invalid count"
        )
    image_by_id = {
        row.get("candidate_id"): row for row in image_bindings if isinstance(row, dict)
    }
    if len(image_by_id) != candidate_count:
        raise ReplacementReviewExportError(
            "replacement manifest image bindings are invalid or duplicated"
        )
    image_records: list[dict[str, Any]] = []
    for candidate_id in queue_order:
        binding = image_by_id.get(candidate_id)
        if not isinstance(binding, dict):
            raise ReplacementReviewExportError(
                f"replacement image binding is missing for {candidate_id}"
            )
        if binding.get("sha256") != queue_by_id[candidate_id]["review_asset_sha256"]:
            raise ReplacementReviewExportError(
                f"replacement image hash does not match queue for {candidate_id}"
            )
        if (
            not isinstance(binding.get("bytes"), int)
            or isinstance(binding.get("bytes"), bool)
            or binding["bytes"] <= 0
        ):
            raise ReplacementReviewExportError(
                f"replacement image byte count is invalid for {candidate_id}"
            )
        image_path = _resolve_relative(
            base=manifest_path.parent,
            value=str(binding.get("path") or ""),
            label=f"replacement image {candidate_id}",
        )
        verified = _verify_record(
            binding,
            label=f"replacement image {candidate_id}",
            current_path=image_path,
        )
        verified["candidate_id"] = candidate_id
        image_records.append(verified)

    tool_payload = _load_embedded_review_data(
        _read_bytes(Path(tool_record["path"]), label="replacement tool")
    )
    expected_tool_header = {
        "queue_sha256": queue_record["sha256"],
        "schema_sha256": schema_record["sha256"],
        "storage_binding_sha256": expected_local_binding["storage_binding_sha256"],
        "round_index": round_index,
        "required_reviewer_id": required_reviewer_id,
        "candidate_count": candidate_count,
        "class_counts": {
            str(class_id): queue_class_counts.get(class_id, 0)
            for class_id in allowed_classes
        },
        "allowed_decisions": allowed_decisions,
        "required_fields": required_fields,
        "claim_boundary": claim_boundary,
        "review_instructions": schema.get("review_instructions"),
    }
    for key, expected_value in expected_tool_header.items():
        if tool_payload.get(key) != expected_value:
            raise ReplacementReviewExportError(
                f"replacement tool review-data {key} is inconsistent"
            )
    embedded_rows = tool_payload.get("rows")
    if not isinstance(embedded_rows, list) or len(embedded_rows) != candidate_count:
        raise ReplacementReviewExportError(
            "replacement tool review-data rows are invalid"
        )
    for expected_id, embedded, image_record in zip(
        queue_order, embedded_rows, image_records
    ):
        if not isinstance(embedded, dict):
            raise ReplacementReviewExportError(
                "replacement tool contains a non-object review row"
            )
        if (
            embedded.get("candidate_id") != expected_id
            or embedded.get("class_id") != int(queue_by_id[expected_id]["class_id"])
            or embedded.get("review_asset_sha256") != image_record["sha256"]
        ):
            raise ReplacementReviewExportError(
                f"replacement tool row does not match the queue for {expected_id}"
            )
        image_uri = embedded.get("image_path")
        if not isinstance(image_uri, str) or not image_uri.startswith(
            "data:image/png;base64,"
        ):
            raise ReplacementReviewExportError(
                f"replacement tool does not embed the PNG for {expected_id}"
            )
        try:
            embedded_image = base64.b64decode(
                image_uri.removeprefix("data:image/png;base64,"), validate=True
            )
        except (ValueError, binascii.Error) as exc:
            raise ReplacementReviewExportError(
                f"replacement tool has invalid embedded PNG data for {expected_id}"
            ) from exc
        if (
            not embedded_image.startswith(b"\x89PNG\r\n\x1a\n")
            or _sha256_bytes(embedded_image) != image_record["sha256"]
        ):
            raise ReplacementReviewExportError(
                f"replacement tool embedded PNG bytes changed for {expected_id}"
            )

    export_order: list[str] = []
    seen: set[str] = set()
    reviewer_ids: set[str] = set()
    decision_counts: Counter[str] = Counter()
    class_decision_counts: dict[int, Counter[str]] = {
        class_id: Counter() for class_id in expected_class_counts
    }
    for row in export_rows:
        if set(row) != expected_fields:
            raise ReplacementReviewExportError(
                "replacement export fields do not exactly match the schema"
            )
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in queue_by_id:
            raise ReplacementReviewExportError(
                f"foreign candidate_id in replacement export: {candidate_id!r}"
            )
        if candidate_id in seen:
            raise ReplacementReviewExportError(
                f"duplicate candidate_id in replacement export: {candidate_id}"
            )
        seen.add(candidate_id)
        export_order.append(candidate_id)
        decision = row.get("decision")
        if decision not in allowed_decisions:
            raise ReplacementReviewExportError(
                f"candidate {candidate_id} has invalid decision {decision!r}"
            )
        reason = row.get("reason")
        reviewer_id = row.get("reviewer_id")
        if not isinstance(reason, str) or not reason.strip():
            raise ReplacementReviewExportError(f"candidate {candidate_id} has no reason")
        if (
            decision == parent_validator.APPROVAL_DECISION
            and reason.strip() != parent_validator.APPROVAL_REASON
        ):
            raise ReplacementReviewExportError(
                f"candidate {candidate_id} approval reason is not canonical"
            )
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise ReplacementReviewExportError(
                f"candidate {candidate_id} has no reviewer_id"
            )
        reviewer_ids.add(reviewer_id.strip())
        reviewed_at = _parse_timestamp(
            row.get("reviewed_at"), candidate_id=candidate_id
        )
        if reviewed_at < parent_validated_at:
            raise ReplacementReviewExportError(
                f"candidate {candidate_id} was reviewed before the parent validation"
            )
        if row.get("review_asset_sha256") != queue_by_id[candidate_id]["review_asset_sha256"]:
            raise ReplacementReviewExportError(
                f"candidate {candidate_id} asset hash does not match the queue"
            )
        if row.get("rows_frozen") is not False:
            raise ReplacementReviewExportError(
                f"candidate {candidate_id} rows_frozen must be false"
            )
        class_id = int(queue_by_id[candidate_id]["class_id"])
        decision_counts[str(decision)] += 1
        class_decision_counts[class_id][str(decision)] += 1
    if export_order != queue_order:
        raise ReplacementReviewExportError(
            "replacement export candidate order does not match the queue"
        )
    if len(reviewer_ids) != 1:
        raise ReplacementReviewExportError(
            "replacement export must contain exactly one reviewer_id"
        )
    if reviewer_ids != {required_reviewer_id}:
        raise ReplacementReviewExportError(
            "replacement export reviewer_id does not match the parent reviewer"
        )

    export_record = _file_record(export_path, export_bytes)
    export_record.update(
        {
            "filename": export_path.name,
            "expected_browser_filename": expected_filename,
        }
    )
    summary = {
        "round": "replacement",
        "claim_boundary": claim_boundary,
        "replacement_round": round_index,
        "candidate_count": candidate_count,
        "candidate_order_sha256": _candidate_order_sha256(queue_order),
        "class_counts": {str(key): value for key, value in expected_class_counts.items()},
        "decision_counts": dict(sorted(decision_counts.items())),
        "class_decision_counts": {
            str(class_id): dict(sorted(counts.items()))
            for class_id, counts in class_decision_counts.items()
        },
        "reviewer_ids": sorted(reviewer_ids),
        "inputs": {
            "export": export_record,
            "queue": queue_record,
            "schema": schema_record,
            "manifest": _file_record(manifest_path, manifest_bytes),
            "generator": generator_record,
            "base_generator": base_generator_record,
            "tool": tool_record,
            "parent_validation_receipt": parent_receipt_record,
            "images": image_records,
            "source_construction_manifest": construction_record,
            "construction_audit": audit_record,
            "replacement_ledger": replacement_ledger_record,
            "human_rejections": human_rejections_record,
        },
    }
    return export_rows, summary


def validate_replacement_review_export(
    *,
    parent_export_path: Path,
    parent_receipt_path: Path,
    replacement_export_path: Path,
    replacement_queue_path: Path,
    replacement_schema_path: Path,
    replacement_manifest_path: Path,
    repo_root: Path = REPO_ROOT,
) -> PreparedCumulativeValidation:
    """Validate both rounds and prepare a parent-first cumulative JSONL."""

    repo_root = repo_root.expanduser().resolve()
    parent_rows, parent_summary = _verify_parent_round(
        export_path=parent_export_path,
        receipt_path=parent_receipt_path,
        repo_root=repo_root,
    )
    replacement_rows, replacement_summary = _validate_replacement_round(
        export_path=replacement_export_path,
        queue_path=replacement_queue_path,
        schema_path=replacement_schema_path,
        manifest_path=replacement_manifest_path,
        parent_summary=parent_summary,
        repo_root=repo_root,
    )

    parent_ids = [str(row["candidate_id"]) for row in parent_rows]
    replacement_ids = [str(row["candidate_id"]) for row in replacement_rows]
    duplicates = set(parent_ids).intersection(replacement_ids)
    if duplicates:
        raise ReplacementReviewExportError(
            "replacement candidates overlap the parent audit: "
            + ", ".join(sorted(duplicates))
        )
    cumulative_rows = parent_rows + replacement_rows
    cumulative_ids = parent_ids + replacement_ids
    cumulative_bytes = _canonical_jsonl(cumulative_rows)

    decision_counts = Counter(str(row["decision"]) for row in cumulative_rows)
    class_counts: Counter[str] = Counter(parent_summary["class_counts"])
    class_counts.update(replacement_summary["class_counts"])
    class_decision_counts: dict[str, Counter[str]] = {}
    for summary in (parent_summary, replacement_summary):
        for class_id, counts in summary["class_decision_counts"].items():
            class_decision_counts.setdefault(class_id, Counter()).update(counts)
    reviewer_ids = sorted(
        set(parent_summary["reviewer_ids"]).union(replacement_summary["reviewer_ids"])
    )
    payload = {
        "kind": VALIDATION_KIND,
        "version": VALIDATION_VERSION,
        "status": "pass",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim_boundary": parent_summary["claim_boundary"],
        "rows_frozen": False,
        "rounds": [parent_summary, replacement_summary],
        "candidate_count": len(cumulative_rows),
        "candidate_order_sha256": _candidate_order_sha256(cumulative_ids),
        "class_counts": dict(sorted(class_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "class_decision_counts": {
            class_id: dict(sorted(counts.items()))
            for class_id, counts in sorted(class_decision_counts.items())
        },
        "reviewer_ids": reviewer_ids,
        "checks": {
            "parent_receipt_and_bundle_fresh_validation": "pass",
            "replacement_manifest_dynamic_counts": "pass",
            "replacement_queue_schema_manifest_hash_binding": "pass",
            "replacement_generator_tool_image_bytes": "pass",
            "replacement_v2_embedded_png_and_reviewer_binding": "pass",
            "replacement_ledger_and_human_rejection_semantics": "pass",
            "source_construction_and_parent_lineage": "pass",
            "exact_record_fields_membership_and_order": "pass",
            "single_reviewer_per_round": "pass",
            "timezone_aware_timestamps": "pass",
            "parent_first_no_candidate_overlap": "pass",
            "cumulative_canonical_jsonl": "pass",
        },
        "next_permitted_action": (
            "rebuild_fresh_and_require_zero_pending_negative_rows_before_scoring"
        ),
    }
    return PreparedCumulativeValidation(
        cumulative_bytes=cumulative_bytes,
        payload_without_cumulative=payload,
    )


def _all_input_paths(payload: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        raise ReplacementReviewExportError("validation payload rounds are invalid")
    for round_index, round_summary in enumerate(rounds):
        inputs = round_summary.get("inputs") if isinstance(round_summary, dict) else None
        if not isinstance(inputs, dict):
            raise ReplacementReviewExportError(
                f"validation payload round {round_index} inputs are invalid"
            )
        for key, value in inputs.items():
            records = value if key == "images" else [value]
            if not isinstance(records, list):
                raise ReplacementReviewExportError(
                    f"validation payload round {round_index} {key} is invalid"
                )
            for record in records:
                path = record.get("path") if isinstance(record, dict) else None
                if not isinstance(path, str):
                    raise ReplacementReviewExportError(
                        f"validation payload round {round_index} {key} path is invalid"
                    )
                paths.append(Path(path).expanduser().resolve())
    return paths


def _write_once(*, target: Path, data: bytes, label: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ReplacementReviewExportError(
            f"{label} already exists; refusing to overwrite: {target}"
        )
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    try:
        try:
            os.link(temporary_path, target)
        except FileExistsError as exc:
            raise ReplacementReviewExportError(
                f"{label} already exists; refusing to overwrite: {target}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _verify_prepared_consistency(
    *,
    cumulative_bytes: bytes,
    payload: dict[str, Any],
) -> None:
    rows = _load_jsonl(cumulative_bytes, label="prepared cumulative export")
    candidate_ids = [str(row.get("candidate_id") or "") for row in rows]
    if any(SHA256_RE.fullmatch(candidate_id) is None for candidate_id in candidate_ids):
        raise ReplacementReviewExportError(
            "prepared cumulative export contains an invalid candidate_id"
        )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ReplacementReviewExportError(
            "prepared cumulative export contains duplicate candidate_ids"
        )
    if payload.get("candidate_count") != len(rows):
        raise ReplacementReviewExportError(
            "prepared cumulative candidate_count is inconsistent"
        )
    if payload.get("candidate_order_sha256") != _candidate_order_sha256(candidate_ids):
        raise ReplacementReviewExportError(
            "prepared cumulative candidate order is inconsistent"
        )
    decisions = Counter(str(row.get("decision") or "") for row in rows)
    if payload.get("decision_counts") != dict(sorted(decisions.items())):
        raise ReplacementReviewExportError(
            "prepared cumulative decision counts are inconsistent"
        )

    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 2:
        raise ReplacementReviewExportError("prepared cumulative rounds are invalid")
    offset = 0
    combined_classes: Counter[str] = Counter()
    combined_class_decisions: dict[str, Counter[str]] = {}
    combined_reviewers: set[str] = set()
    for expected_round, summary in zip(("parent", "replacement"), rounds):
        if not isinstance(summary, dict) or summary.get("round") != expected_round:
            raise ReplacementReviewExportError(
                "prepared cumulative round ordering is invalid"
            )
        count = summary.get("candidate_count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ReplacementReviewExportError(
                f"prepared {expected_round} candidate_count is invalid"
            )
        round_ids = candidate_ids[offset : offset + count]
        if summary.get("candidate_order_sha256") != _candidate_order_sha256(round_ids):
            raise ReplacementReviewExportError(
                f"prepared {expected_round} candidate order is inconsistent"
            )
        inputs = summary.get("inputs")
        bound_export = inputs.get("export") if isinstance(inputs, dict) else None
        bound_export_path = (
            bound_export.get("path") if isinstance(bound_export, dict) else None
        )
        if not isinstance(bound_export_path, str):
            raise ReplacementReviewExportError(
                f"prepared {expected_round} export binding is invalid"
            )
        bound_rows = _load_jsonl(
            _read_bytes(
                Path(bound_export_path),
                label=f"prepared {expected_round} bound export",
            ),
            label=f"prepared {expected_round} bound export",
        )
        cumulative_slice = rows[offset : offset + count]
        if bound_rows != cumulative_slice:
            raise ReplacementReviewExportError(
                f"prepared {expected_round} cumulative rows differ from the bound export"
            )
        round_decisions = Counter(
            str(row.get("decision") or "") for row in cumulative_slice
        )
        if summary.get("decision_counts") != dict(sorted(round_decisions.items())):
            raise ReplacementReviewExportError(
                f"prepared {expected_round} decision counts are inconsistent"
            )
        class_counts = summary.get("class_counts")
        class_decisions = summary.get("class_decision_counts")
        reviewers = summary.get("reviewer_ids")
        if (
            not isinstance(class_counts, dict)
            or any(
                not isinstance(class_id, str)
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for class_id, value in class_counts.items()
            )
            or sum(class_counts.values()) != count
        ):
            raise ReplacementReviewExportError(
                f"prepared {expected_round} class counts are inconsistent"
            )
        if (
            not isinstance(class_decisions, dict)
            or set(class_decisions) != set(class_counts)
            or not isinstance(reviewers, list)
            or len(reviewers) != 1
            or not isinstance(reviewers[0], str)
            or not reviewers[0].strip()
        ):
            raise ReplacementReviewExportError(
                f"prepared {expected_round} summary is incomplete"
            )
        combined_classes.update(class_counts)
        decisions_from_classes: Counter[str] = Counter()
        for class_id, counts in class_decisions.items():
            if (
                not isinstance(counts, dict)
                or any(
                    not isinstance(decision, str)
                    or not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for decision, value in counts.items()
                )
                or sum(counts.values()) != class_counts[class_id]
            ):
                raise ReplacementReviewExportError(
                    f"prepared {expected_round} class decision counts are invalid"
                )
            combined_class_decisions.setdefault(class_id, Counter()).update(counts)
            decisions_from_classes.update(counts)
        if decisions_from_classes != round_decisions:
            raise ReplacementReviewExportError(
                f"prepared {expected_round} class decisions are inconsistent"
            )
        combined_reviewers.update(str(value) for value in reviewers)
        offset += count
    if offset != len(rows):
        raise ReplacementReviewExportError(
            "prepared cumulative round counts do not cover the export"
        )
    if payload.get("class_counts") != dict(sorted(combined_classes.items())):
        raise ReplacementReviewExportError(
            "prepared cumulative class counts are inconsistent"
        )
    expected_class_decisions = {
        class_id: dict(sorted(counts.items()))
        for class_id, counts in sorted(combined_class_decisions.items())
    }
    if payload.get("class_decision_counts") != expected_class_decisions:
        raise ReplacementReviewExportError(
            "prepared cumulative class decision counts are inconsistent"
        )
    if payload.get("reviewer_ids") != sorted(combined_reviewers):
        raise ReplacementReviewExportError(
            "prepared cumulative reviewer_ids are inconsistent"
        )


def _rehash_prepared_inputs(payload: dict[str, Any]) -> None:
    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        raise ReplacementReviewExportError("validation payload rounds are invalid")
    for round_index, summary in enumerate(rounds):
        inputs = summary.get("inputs") if isinstance(summary, dict) else None
        if not isinstance(inputs, dict):
            raise ReplacementReviewExportError(
                f"validation payload round {round_index} inputs are invalid"
            )
        for key, value in inputs.items():
            records = value if key == "images" else [value]
            if not isinstance(records, list):
                raise ReplacementReviewExportError(
                    f"validation payload round {round_index} {key} is invalid"
                )
            for record_index, record in enumerate(records):
                _verify_record(
                    record,
                    label=(
                        f"prepared round {round_index} {key}[{record_index}]"
                    ),
                )


def issue_cumulative_validation(
    *,
    cumulative_dir: Path,
    receipt_dir: Path,
    prepared: PreparedCumulativeValidation,
) -> IssuedCumulativeValidation:
    """Write immutable cumulative JSONL and its content-addressed receipt."""

    payload = json.loads(json.dumps(prepared.payload_without_cumulative))
    if payload.get("kind") != VALIDATION_KIND or payload.get("version") != VALIDATION_VERSION:
        raise ReplacementReviewExportError("refusing to issue an unknown validation payload")
    if payload.get("status") != "pass":
        raise ReplacementReviewExportError("refusing to issue a non-pass validation payload")
    if payload.get("rows_frozen") is not False:
        raise ReplacementReviewExportError("prepared payload rows_frozen must be false")
    _parse_timestamp(
        payload.get("validated_at_utc"),
        candidate_id="cumulative-validation-payload",
    )
    _verify_prepared_consistency(
        cumulative_bytes=prepared.cumulative_bytes,
        payload=payload,
    )
    _rehash_prepared_inputs(payload)
    cumulative_sha256 = _sha256_bytes(prepared.cumulative_bytes)
    cumulative_filename = f"{CUMULATIVE_PREFIX}{cumulative_sha256}.jsonl"
    receipt_filename = f"{RECEIPT_PREFIX}{cumulative_sha256}.json"
    cumulative_path = cumulative_dir.expanduser().resolve() / cumulative_filename
    receipt_path = receipt_dir.expanduser().resolve() / receipt_filename
    input_paths = _all_input_paths(payload)
    if cumulative_path in input_paths or receipt_path in input_paths:
        raise ReplacementReviewExportError(
            "cumulative output or receipt path collides with a validation input"
        )
    if cumulative_path == receipt_path:
        raise ReplacementReviewExportError("cumulative output and receipt paths collide")
    if cumulative_path.exists():
        raise ReplacementReviewExportError(
            f"cumulative export already exists; refusing to overwrite: {cumulative_path}"
        )
    if receipt_path.exists():
        raise ReplacementReviewExportError(
            f"receipt already exists; refusing to overwrite: {receipt_path}"
        )

    payload["inputs"] = {
        "cumulative_export": {
            "path": str(cumulative_path),
            "sha256": cumulative_sha256,
            "bytes": len(prepared.cumulative_bytes),
            "filename": cumulative_filename,
        }
    }
    payload["receipt_filename"] = receipt_filename
    receipt_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    _write_once(
        target=cumulative_path,
        data=prepared.cumulative_bytes,
        label="cumulative export",
    )
    try:
        _write_once(target=receipt_path, data=receipt_bytes, label="receipt")
    except Exception:
        cumulative_path.unlink(missing_ok=True)
        raise
    return IssuedCumulativeValidation(
        cumulative_path=cumulative_path,
        receipt_path=receipt_path,
        payload=payload,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-export", type=Path, required=True)
    parser.add_argument("--parent-validation-receipt", type=Path, required=True)
    parser.add_argument("--replacement-export", type=Path, required=True)
    parser.add_argument("--replacement-queue", type=Path, required=True)
    parser.add_argument("--replacement-schema", type=Path, required=True)
    parser.add_argument("--replacement-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--cumulative-dir", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        prepared = validate_replacement_review_export(
            parent_export_path=args.parent_export,
            parent_receipt_path=args.parent_validation_receipt,
            replacement_export_path=args.replacement_export,
            replacement_queue_path=args.replacement_queue,
            replacement_schema_path=args.replacement_schema,
            replacement_manifest_path=args.replacement_manifest,
            repo_root=args.repo_root,
        )
        issued = issue_cumulative_validation(
            cumulative_dir=args.cumulative_dir,
            receipt_dir=args.receipt_dir,
            prepared=prepared,
        )
    except ReplacementReviewExportError as exc:
        print(f"replacement review validation failed: {exc}")
        return 2
    print(f"cumulative_path={issued.cumulative_path}")
    print(f"receipt_path={issued.receipt_path}")
    print(json.dumps(issued.payload, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
