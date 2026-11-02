#!/usr/bin/env python3
"""Validate the frozen burned-data pending-negative browser export."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_ROOT = REPO_ROOT / Path(
    "runs/research/lifecycle_memory/phase_2/"
    "job3_burned_egohos_construction_2026_07_15/hidden/pending_negative_review"
)
DEFAULT_QUEUE = DEFAULT_REVIEW_ROOT / "queue.tsv"
DEFAULT_MANIFEST = DEFAULT_REVIEW_ROOT / "review_tool_manifest.json"
DEFAULT_SCHEMA = REPO_ROOT / Path(
    "docs/research/lifecycle_memory/phase 2 research/preregistration/"
    "job3_pending_negative_review_schema_v0.json"
)

EXPECTED_SCHEMA_KIND = "phase2_job3_pending_negative_review_schema"
EXPECTED_SCHEMA_VERSION = 0
EXPECTED_SCHEMA_STATUS = "draft_before_burned_review"
EXPECTED_MANIFEST_KIND = "phase2_job3_pending_negative_review_tool_manifest"
EXPECTED_MANIFEST_VERSION = 1
EXPECTED_MANIFEST_STATUS = "construction_pilot_local_review_only"
EXPECTED_CANDIDATE_COUNT = 60
EXPECTED_CLASS_COUNTS = {6: 20, 7: 20, 8: 20}
APPROVAL_DECISION = "approve_no_direct_hand_contact"
APPROVAL_REASON = "reviewer_confirmed_no_direct_hand_contact"
VALIDATION_KIND = "phase2_job3_pending_negative_review_export_validation"
VALIDATION_VERSION = 2
RECEIPT_PREFIX = "pending_negative_review_export_validation_"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_FUTURE_TIMESTAMP_SKEW = timedelta(minutes=5)


class ReviewExportError(ValueError):
    """Raised when a review export fails a frozen validation gate."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path, *, label: str) -> bytes:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReviewExportError(f"{label} does not exist: {resolved}")
    return resolved.read_bytes()


def _load_json(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewExportError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewExportError(f"{label} must contain a JSON object")
    return value


def _load_queue(data: bytes) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewExportError("queue is not valid UTF-8") from exc
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    required = {"candidate_id", "class_id", "review_asset_sha256", "rows_frozen"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ReviewExportError("queue is missing required columns")
    rows = list(reader)
    if len(rows) != EXPECTED_CANDIDATE_COUNT:
        raise ReviewExportError(
            f"queue must contain {EXPECTED_CANDIDATE_COUNT} rows, got {len(rows)}"
        )
    return rows


def _load_export(data: bytes) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewExportError("export is not valid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ReviewExportError(f"export contains blank line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReviewExportError(
                f"export line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ReviewExportError(
                f"export line {line_number} must contain a JSON object"
            )
        rows.append(row)
    if len(rows) != EXPECTED_CANDIDATE_COUNT:
        raise ReviewExportError(
            f"export must contain {EXPECTED_CANDIDATE_COUNT} rows, got {len(rows)}"
        )
    return rows


def _parse_timestamp(value: Any, *, candidate_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewExportError(f"candidate {candidate_id} has no reviewed_at")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ReviewExportError(
            f"candidate {candidate_id} has invalid reviewed_at"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReviewExportError(
            f"candidate {candidate_id} reviewed_at must include a timezone"
        )
    if (
        parsed.astimezone(timezone.utc)
        > datetime.now(timezone.utc) + MAX_FUTURE_TIMESTAMP_SKEW
    ):
        raise ReviewExportError(
            f"candidate {candidate_id} reviewed_at is implausibly in the future"
        )
    return value.strip()


def _binding_record(manifest: dict[str, Any], key: str) -> dict[str, str]:
    try:
        record = manifest["bindings"][key]
    except (KeyError, TypeError) as exc:
        raise ReviewExportError(f"manifest is missing bindings.{key}") from exc
    if not isinstance(record, dict):
        raise ReviewExportError(f"manifest bindings.{key} must be an object")
    path = record.get("path")
    value = record.get("sha256")
    if not isinstance(path, str) or not path.strip():
        raise ReviewExportError(f"manifest bindings.{key}.path is invalid")
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ReviewExportError(f"manifest bindings.{key}.sha256 is invalid")
    return {"path": path, "sha256": value}


def _binding_hash(manifest: dict[str, Any], key: str) -> str:
    return _binding_record(manifest, key)["sha256"]


def _resolve_bound_file(*, base: Path, relative_path: str, label: str) -> Path:
    raw = Path(relative_path)
    if raw.is_absolute():
        raise ReviewExportError(f"manifest {label} path must be relative")
    resolved_base = base.resolve()
    resolved = (resolved_base / raw).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ReviewExportError(
            f"manifest {label} path escapes its allowed root"
        ) from exc
    if not resolved.is_file():
        raise ReviewExportError(f"manifest-bound {label} does not exist: {resolved}")
    return resolved


def _verified_file_record(
    path: Path, expected_sha256: str, *, label: str
) -> dict[str, Any]:
    data = _read_bytes(path, label=label)
    actual_sha256 = _sha256_bytes(data)
    if actual_sha256 != expected_sha256:
        raise ReviewExportError(
            f"{label} hash does not match the review-tool manifest"
        )
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "bytes": len(data),
    }


def validate_review_export(
    *,
    export_path: Path,
    queue_path: Path = DEFAULT_QUEUE,
    schema_path: Path = DEFAULT_SCHEMA,
    manifest_path: Path = DEFAULT_MANIFEST,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    export_path = export_path.expanduser().resolve()
    queue_path = queue_path.expanduser().resolve()
    schema_path = schema_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()

    export_bytes = _read_bytes(export_path, label="export")
    queue_bytes = _read_bytes(queue_path, label="queue")
    schema_bytes = _read_bytes(schema_path, label="schema")
    manifest_bytes = _read_bytes(manifest_path, label="manifest")

    schema = _load_json(schema_bytes, label="schema")
    manifest = _load_json(manifest_bytes, label="manifest")
    queue_rows = _load_queue(queue_bytes)
    export_rows = _load_export(export_bytes)

    if schema.get("kind") != EXPECTED_SCHEMA_KIND:
        raise ReviewExportError(
            "schema kind does not match the frozen review schema"
        )
    if schema.get("version") != EXPECTED_SCHEMA_VERSION:
        raise ReviewExportError(
            "schema version does not match the frozen review schema"
        )
    if schema.get("status") != EXPECTED_SCHEMA_STATUS:
        raise ReviewExportError(
            "schema status does not match the frozen review schema"
        )
    if schema.get("allowed_mask_classes") != list(EXPECTED_CLASS_COUNTS):
        raise ReviewExportError("schema allowed_mask_classes must be [6, 7, 8]")
    if schema.get("rows_frozen") is not False:
        raise ReviewExportError("schema rows_frozen must be false")
    claim_boundary = schema.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise ReviewExportError("schema claim_boundary must be a nonempty string")
    if manifest.get("kind") != EXPECTED_MANIFEST_KIND:
        raise ReviewExportError("review-tool manifest kind is invalid")
    if manifest.get("version") != EXPECTED_MANIFEST_VERSION:
        raise ReviewExportError("review-tool manifest version is invalid")
    if manifest.get("status") != EXPECTED_MANIFEST_STATUS:
        raise ReviewExportError("review-tool manifest status is invalid")
    if manifest.get("claim_boundary") != claim_boundary:
        raise ReviewExportError("review-tool manifest claim_boundary is invalid")
    if manifest.get("candidate_count") != EXPECTED_CANDIDATE_COUNT:
        raise ReviewExportError("review-tool manifest candidate_count is invalid")
    if manifest.get("rows_frozen") is not False:
        raise ReviewExportError("review-tool manifest rows_frozen must be false")

    queue_sha256 = _sha256_bytes(queue_bytes)
    schema_sha256 = _sha256_bytes(schema_bytes)
    queue_binding = _binding_record(manifest, "queue")
    schema_binding = _binding_record(manifest, "schema")
    if Path(queue_binding["path"]).name != queue_path.name:
        raise ReviewExportError(
            "manifest queue path does not identify the validated queue"
        )
    if Path(schema_binding["path"]).name != schema_path.name:
        raise ReviewExportError(
            "manifest schema path does not identify the validated schema"
        )
    if _binding_hash(manifest, "queue") != queue_sha256:
        raise ReviewExportError("queue hash does not match the review-tool manifest")
    if _binding_hash(manifest, "schema") != schema_sha256:
        raise ReviewExportError("schema hash does not match the review-tool manifest")
    if manifest.get("local_storage_binding") != {
        "queue_sha256": queue_sha256,
        "schema_sha256": schema_sha256,
    }:
        raise ReviewExportError(
            "review-tool local_storage_binding does not match queue and schema"
        )

    expected_filename = f"pending_negative_review_FINAL_{queue_sha256[:12]}.jsonl"
    expected_name_pattern = re.compile(
        rf"{re.escape(expected_filename[:-6])}(?: \([1-9][0-9]*\))?\.jsonl\Z"
    )
    if expected_name_pattern.fullmatch(export_path.name) is None:
        raise ReviewExportError(
            "export filename is not the final browser-export name bound to this queue"
        )

    generator_binding = _binding_record(manifest, "generator")
    generator_path = _resolve_bound_file(
        base=repo_root,
        relative_path=generator_binding["path"],
        label="generator",
    )
    generator_record = _verified_file_record(
        generator_path,
        generator_binding["sha256"],
        label="generator",
    )
    tool_binding = _binding_record(manifest, "tool")
    tool_path = _resolve_bound_file(
        base=manifest_path.parent,
        relative_path=tool_binding["path"],
        label="tool",
    )
    tool_record = _verified_file_record(
        tool_path,
        tool_binding["sha256"],
        label="review tool",
    )

    allowed_decisions = schema.get("allowed_decisions")
    required_fields = schema.get("decision_record_required_fields")
    if not isinstance(allowed_decisions, list) or not all(
        isinstance(value, str) for value in allowed_decisions
    ):
        raise ReviewExportError("schema allowed_decisions is invalid")
    if not isinstance(required_fields, list) or not all(
        isinstance(value, str) for value in required_fields
    ):
        raise ReviewExportError("schema decision_record_required_fields is invalid")
    expected_record_fields = set(required_fields) | {"rows_frozen"}

    queue_by_id: dict[str, dict[str, str]] = {}
    queue_order: list[str] = []
    queue_class_counts: Counter[int] = Counter()
    for row in queue_rows:
        candidate_id = row["candidate_id"].strip()
        asset_sha256 = row["review_asset_sha256"].strip()
        if SHA256_RE.fullmatch(candidate_id) is None:
            raise ReviewExportError(f"queue candidate_id is invalid: {candidate_id!r}")
        if candidate_id in queue_by_id:
            raise ReviewExportError(f"queue candidate_id is duplicated: {candidate_id}")
        if SHA256_RE.fullmatch(asset_sha256) is None:
            raise ReviewExportError(
                f"queue review_asset_sha256 is invalid for {candidate_id}"
            )
        if row["rows_frozen"].strip().lower() != "false":
            raise ReviewExportError(f"queue row {candidate_id} is unexpectedly frozen")
        try:
            class_id = int(row["class_id"])
        except ValueError as exc:
            raise ReviewExportError(
                f"queue class_id is invalid for {candidate_id}"
            ) from exc
        queue_by_id[candidate_id] = row
        queue_order.append(candidate_id)
        queue_class_counts[class_id] += 1
    if dict(queue_class_counts) != EXPECTED_CLASS_COUNTS:
        raise ReviewExportError(
            f"queue class counts are invalid: {dict(queue_class_counts)}"
        )

    expected_manifest_counts = {
        str(class_id): count for class_id, count in EXPECTED_CLASS_COUNTS.items()
    }
    if manifest.get("class_counts") != expected_manifest_counts:
        raise ReviewExportError("review-tool manifest class_counts is invalid")

    try:
        image_rows = manifest["bindings"]["images"]
    except (KeyError, TypeError) as exc:
        raise ReviewExportError("manifest is missing bindings.images") from exc
    if not isinstance(image_rows, list) or len(image_rows) != EXPECTED_CANDIDATE_COUNT:
        raise ReviewExportError("manifest bindings.images has an invalid row count")
    manifest_images = {
        row.get("candidate_id"): row
        for row in image_rows
        if isinstance(row, dict)
    }
    if len(manifest_images) != EXPECTED_CANDIDATE_COUNT:
        raise ReviewExportError(
            "manifest bindings.images has duplicate or invalid rows"
        )
    image_records: list[dict[str, Any]] = []
    for candidate_id, queue_row in queue_by_id.items():
        image_binding = manifest_images.get(candidate_id)
        if not isinstance(image_binding, dict):
            raise ReviewExportError(
                f"manifest image binding is missing for candidate {candidate_id}"
            )
        image_path_value = image_binding.get("path")
        image_sha256 = image_binding.get("sha256")
        if not isinstance(image_path_value, str) or not image_path_value.strip():
            raise ReviewExportError(
                f"manifest image path is invalid for candidate {candidate_id}"
            )
        if image_sha256 != queue_row["review_asset_sha256"]:
            raise ReviewExportError(
                f"manifest image binding mismatch for candidate {candidate_id}"
            )
        image_path = _resolve_bound_file(
            base=manifest_path.parent,
            relative_path=image_path_value,
            label=f"image {candidate_id}",
        )
        record = _verified_file_record(
            image_path,
            image_sha256,
            label=f"review image {candidate_id}",
        )
        record["candidate_id"] = candidate_id
        image_records.append(record)

    export_order: list[str] = []
    reviewer_ids: set[str] = set()
    decision_counts: Counter[str] = Counter()
    class_decision_counts: dict[int, Counter[str]] = {
        class_id: Counter() for class_id in EXPECTED_CLASS_COUNTS
    }
    seen: set[str] = set()
    for row in export_rows:
        if set(row) != expected_record_fields:
            raise ReviewExportError(
                "export record fields do not exactly match the frozen schema"
            )
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in queue_by_id:
            raise ReviewExportError(f"foreign candidate_id in export: {candidate_id!r}")
        if candidate_id in seen:
            raise ReviewExportError(f"duplicate candidate_id in export: {candidate_id}")
        seen.add(candidate_id)
        export_order.append(candidate_id)

        decision = row.get("decision")
        if decision not in allowed_decisions:
            raise ReviewExportError(
                f"candidate {candidate_id} has invalid decision {decision!r}"
            )
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ReviewExportError(f"candidate {candidate_id} has no reason")
        reviewer_id = row.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise ReviewExportError(f"candidate {candidate_id} has no reviewer_id")
        reviewer_ids.add(reviewer_id.strip())
        _parse_timestamp(row.get("reviewed_at"), candidate_id=candidate_id)

        asset_sha256 = row.get("review_asset_sha256")
        if asset_sha256 != queue_by_id[candidate_id]["review_asset_sha256"]:
            raise ReviewExportError(
                f"candidate {candidate_id} review asset hash does not match the queue"
            )
        if row.get("rows_frozen") is not False:
            raise ReviewExportError(
                f"candidate {candidate_id} rows_frozen must be false"
            )

        class_id = int(queue_by_id[candidate_id]["class_id"])
        decision_counts[decision] += 1
        class_decision_counts[class_id][decision] += 1

    if export_order != queue_order:
        raise ReviewExportError(
            "export candidate order does not match the frozen queue"
        )
    if len(reviewer_ids) != 1:
        raise ReviewExportError(
            "export must contain exactly one nonempty reviewer_id"
        )

    export_sha256 = _sha256_bytes(export_bytes)
    receipt_filename = f"{RECEIPT_PREFIX}{export_sha256}.json"
    candidate_order_sha256 = _sha256_bytes(
        ("\n".join(queue_order) + "\n").encode("utf-8")
    )
    return {
        "kind": VALIDATION_KIND,
        "version": VALIDATION_VERSION,
        "status": "pass",
        "receipt_filename": receipt_filename,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim_boundary": schema.get("claim_boundary"),
        "rows_frozen": False,
        "inputs": {
            "export": {
                "path": str(export_path),
                "sha256": export_sha256,
                "bytes": len(export_bytes),
                "filename": export_path.name,
                "expected_browser_filename": expected_filename,
            },
            "queue": {"path": str(queue_path), "sha256": queue_sha256},
            "schema": {"path": str(schema_path), "sha256": schema_sha256},
            "manifest": {
                "path": str(manifest_path),
                "sha256": _sha256_bytes(manifest_bytes),
            },
            "generator": generator_record,
            "tool": tool_record,
            "images": image_records,
        },
        "candidate_count": len(export_rows),
        "candidate_order_sha256": candidate_order_sha256,
        "class_counts": {str(key): value for key, value in EXPECTED_CLASS_COUNTS.items()},
        "decision_counts": dict(sorted(decision_counts.items())),
        "class_decision_counts": {
            str(class_id): dict(sorted(counts.items()))
            for class_id, counts in class_decision_counts.items()
        },
        "reviewer_ids": sorted(reviewer_ids),
        "checks": {
            "exact_record_fields": "pass",
            "exact_queue_membership_and_order": "pass",
            "queue_schema_manifest_hash_binding": "pass",
            "manifest_version_status_claim_boundary": "pass",
            "final_export_filename_pattern": "pass",
            "generator_tool_and_image_bytes": "pass",
            "review_asset_hash_binding": "pass",
            "single_reviewer": "pass",
            "timezone_aware_timestamps": "pass",
            "required_reasons": "pass",
            "approval_default_or_typed_note": "pass",
            "rows_frozen_false": "pass",
        },
        "next_permitted_action": (
            "rebuild_burned_construction_in_a_fresh_round_directory_with_this_export"
        ),
    }


def _receipt_input_paths(payload: dict[str, Any]) -> list[Path]:
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ReviewExportError("validation payload has no inputs object")
    paths: list[Path] = []
    for key in ("export", "queue", "schema", "manifest", "generator", "tool"):
        record = inputs.get(key)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ReviewExportError(f"validation payload input {key} is invalid")
        paths.append(Path(record["path"]).expanduser().resolve())
    images = inputs.get("images")
    if not isinstance(images, list):
        raise ReviewExportError("validation payload input images is invalid")
    for index, record in enumerate(images):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ReviewExportError(
                f"validation payload image input {index} is invalid"
            )
        paths.append(Path(record["path"]).expanduser().resolve())
    return paths


def issue_validation_receipt(
    *, receipt_dir: Path, payload: dict[str, Any]
) -> Path:
    if payload.get("kind") != VALIDATION_KIND or payload.get("version") != VALIDATION_VERSION:
        raise ReviewExportError("refusing to issue a receipt for an unknown validation payload")
    if payload.get("status") != "pass":
        raise ReviewExportError("refusing to issue a non-pass validation receipt")
    filename = payload.get("receipt_filename")
    inputs = payload.get("inputs")
    export_record = inputs.get("export") if isinstance(inputs, dict) else None
    export_sha256 = export_record.get("sha256") if isinstance(export_record, dict) else None
    if (
        not isinstance(export_sha256, str)
        or SHA256_RE.fullmatch(export_sha256) is None
        or filename != f"{RECEIPT_PREFIX}{export_sha256}.json"
    ):
        raise ReviewExportError("validation payload receipt filename is not content-addressed")

    directory = receipt_dir.expanduser().resolve()
    if directory.exists() and not directory.is_dir():
        raise ReviewExportError(f"receipt directory is not a directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    input_paths = _receipt_input_paths(payload)
    if target in input_paths:
        raise ReviewExportError("receipt path collides with a validation input")
    if target.exists():
        raise ReviewExportError(f"receipt already exists; refusing to overwrite: {target}")

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=f".{filename}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary_path, target)
        except FileExistsError as exc:
            raise ReviewExportError(
                f"receipt already exists; refusing to overwrite: {target}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        required=True,
        help="Directory for the immutable content-addressed pass receipt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        receipt = validate_review_export(
            export_path=args.export,
        )
        receipt_path = issue_validation_receipt(
            receipt_dir=args.receipt_dir,
            payload=receipt,
        )
    except ReviewExportError as exc:
        print(f"review export validation failed: {exc}")
        return 2
    print(f"receipt_path={receipt_path}")
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
