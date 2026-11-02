#!/usr/bin/env python3
"""Freeze the evaluator-blind, identity-disjoint held-out request cohort."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


KIND = "patchaudit_jobE_heldout_selection"
VERSION = 1
EXPECTED_SOURCE = {
    "mechanism_manifest_sha256": (
        "a87aa16daa4d944882b0ba1a3891b639c71a83ec090aa29b0f5ac6c3f26586be"
    ),
    "public_sha256": (
        "d8c92857c0f2bbb11de41b1b92853bf208b816d4640d21fadbfb9e6cf9be6d97"
    ),
    "hidden_sha256": (
        "7ca499317bf2eb6b7e88a2b851fe467d72c603abd2ddee4c825b8443228f3011"
    ),
}
EXPECTED_EXCLUSION = {
    "mechanism_manifest_sha256": (
        "c219a8f6746382edbb3e6a304fe85e750755e719874cf5775a2a6ef48a3fca88"
    ),
    "public_sha256": (
        "a6b9d166597efae8a26a65e2c654367da99fddef7ccaab6698a1646930083ced"
    ),
    "hidden_sha256": (
        "7a764586ac27a05fbe0d02151a632e91674d4b6c5721f4adba6bdae79a9725f1"
    ),
}
IDENTITY_FIELDS = (
    "participant_cluster_id",
    "source_video_id",
    "frame_id",
    "donor_id",
)
EXPECTED_FRAME_COUNT = 46
EXPECTED_REQUEST_COUNT = 92


class SelectionError(RuntimeError):
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
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SelectionError(f"invalid JSONL {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise SelectionError(f"non-object JSONL row {path}:{line_number}")
            rows.append(value)
    return rows


def _mechanism_paths(root: Path) -> dict[str, Path]:
    mechanism = root / "mechanism"
    return {
        "manifest": mechanism / "manifest.json",
        "public": mechanism / "public/mechanism_requests.jsonl",
        "hidden": mechanism / "hidden/mechanism_evaluator_index.jsonl",
    }


def _verify_source(root: Path, expected: Mapping[str, str]) -> dict[str, Any]:
    paths = _mechanism_paths(root)
    for path in paths.values():
        if not path.is_file():
            raise SelectionError(f"missing frozen mechanism input: {path}")
    observed = {
        "mechanism_manifest_sha256": sha256_file(paths["manifest"]),
        "public_sha256": sha256_file(paths["public"]),
        "hidden_sha256": sha256_file(paths["hidden"]),
    }
    if observed != dict(expected):
        raise SelectionError(
            f"frozen mechanism hash mismatch at {root}: "
            f"expected={dict(expected)} observed={observed}"
        )
    manifest = read_json(paths["manifest"])
    if (
        manifest.get("kind")
        != "phase2_reanchored_egohos_mechanism_subset_manifest"
        or manifest.get("status") != "complete_production"
        or manifest.get("production_eligible") is not True
    ):
        raise SelectionError(f"invalid mechanism manifest status at {root}")
    return {"root": root, "paths": paths, "manifest": manifest, "hashes": observed}


def _unique_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in result:
            raise SelectionError(f"missing or duplicate {key}: {value!r}")
        result[value] = dict(row)
    return result


def _pair_rows(
    public_rows: Iterable[Mapping[str, Any]],
    hidden_by_request: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    frames: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for public in public_rows:
        request_id = str(public.get("request_id") or "")
        hidden = hidden_by_request.get(request_id)
        if hidden is None:
            raise SelectionError(f"public request lacks evaluator join: {request_id}")
        if public.get("frame_unit_id") != hidden.get("frame_unit_id"):
            raise SelectionError(f"frame-unit join mismatch: {request_id}")
        frames[str(public["frame_unit_id"])].append((dict(public), dict(hidden)))
    for frame_id, pair in frames.items():
        if len(pair) != 2:
            raise SelectionError(f"frame {frame_id} does not have two orders")
        orders = {str(hidden.get("order_id")) for _public, hidden in pair}
        if orders != {"order_0", "order_1"}:
            raise SelectionError(f"frame {frame_id} has invalid order pair: {orders}")
        identities = {
            field: {str(hidden.get(field)) for _public, hidden in pair}
            for field in IDENTITY_FIELDS
        }
        if any(len(values) != 1 or "" in values for values in identities.values()):
            raise SelectionError(f"frame {frame_id} identity fields disagree")
    return frames


def _hashed_values(values: Iterable[str]) -> str:
    return canonical_sha256(sorted(set(str(value) for value in values)))


def jsonl_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def build_selection(source_root: Path, exclusion_root: Path) -> dict[str, Any]:
    source = _verify_source(source_root.resolve(), EXPECTED_SOURCE)
    exclusion = _verify_source(exclusion_root.resolve(), EXPECTED_EXCLUSION)

    source_public = read_jsonl(source["paths"]["public"])
    source_hidden = read_jsonl(source["paths"]["hidden"])
    exclusion_hidden = read_jsonl(exclusion["paths"]["hidden"])
    source_hidden_by_request = _unique_by(source_hidden, "request_id")
    if set(source_hidden_by_request) != {
        str(row.get("request_id") or "") for row in source_public
    }:
        raise SelectionError("source public/evaluator request inventory differs")
    source_frames = _pair_rows(source_public, source_hidden_by_request)

    excluded = {
        field: {str(row.get(field) or "") for row in exclusion_hidden}
        for field in IDENTITY_FIELDS
    }
    if any("" in values for values in excluded.values()):
        raise SelectionError("exclusion identity field is missing")

    selected: list[tuple[str, list[tuple[dict[str, Any], dict[str, Any]]]]] = []
    rejected_counts: dict[str, int] = defaultdict(int)
    for frame_unit_id, pair in source_frames.items():
        hidden = pair[0][1]
        overlaps = [
            field
            for field in IDENTITY_FIELDS
            if str(hidden[field]) in excluded[field]
        ]
        if overlaps:
            for field in overlaps:
                rejected_counts[field] += 1
            continue
        selected.append((frame_unit_id, pair))

    selected.sort(
        key=lambda item: hashlib.sha256(
            f"patchaudit-jobE-heldout-v1|{item[0]}".encode("utf-8")
        ).hexdigest()
    )
    selected_public: list[dict[str, Any]] = []
    selected_hidden: list[dict[str, Any]] = []
    for _frame_id, pair in selected:
        for public, hidden in sorted(pair, key=lambda row: str(row[0]["order_id"])):
            selected_public.append(public)
            selected_hidden.append(hidden)

    if len(selected) != EXPECTED_FRAME_COUNT:
        raise SelectionError(
            f"held-out frame count drift: {len(selected)} != {EXPECTED_FRAME_COUNT}"
        )
    if len(selected_public) != EXPECTED_REQUEST_COUNT:
        raise SelectionError(
            "held-out request count drift: "
            f"{len(selected_public)} != {EXPECTED_REQUEST_COUNT}"
        )
    request_ids = [str(row["request_id"]) for row in selected_public]
    frame_ids = [str(row["frame_unit_id"]) for row in selected_public[::2]]
    if len(set(request_ids)) != EXPECTED_REQUEST_COUNT:
        raise SelectionError("selected request IDs are not unique")
    if len(set(frame_ids)) != EXPECTED_FRAME_COUNT:
        raise SelectionError("selected frame-unit IDs are not unique")

    overlap_audit: dict[str, Any] = {}
    for field in IDENTITY_FIELDS:
        source_values = {str(row[field]) for row in selected_hidden}
        overlap = source_values & excluded[field]
        if overlap:
            raise SelectionError(f"selected {field} overlaps exclusion cohort")
        overlap_audit[field] = {
            "selected_count": len(source_values),
            "selected_values_sha256": _hashed_values(source_values),
            "exclusion_count": len(excluded[field]),
            "exclusion_values_sha256": _hashed_values(excluded[field]),
            "overlap_count": 0,
        }

    return {
        "kind": KIND,
        "version": VERSION,
        "status": "pass",
        "selection_status": "frozen_before_any_jobE_model_outcome",
        "selection_rule": (
            "retain every source frame pair whose participant cluster, source "
            "video, source frame, and donor identities are all absent from the "
            "Job D cohort; sort only for deterministic execution"
        ),
        "selection_uses_correct_choice": False,
        "selection_uses_model_outputs": False,
        "source": {
            **source["hashes"],
            "frame_count": int(source["manifest"]["frame_count"]),
            "request_count": int(source["manifest"]["trial_count"]),
        },
        "exclusion": {
            **exclusion["hashes"],
            "frame_count": int(exclusion["manifest"]["frame_count"]),
            "request_count": int(exclusion["manifest"]["trial_count"]),
            "role": "complete Job D cohort",
        },
        "selected": {
            "frame_count": len(selected),
            "participant_cluster_count": len(
                {str(row["participant_cluster_id"]) for row in selected_hidden}
            ),
            "source_video_count": len(
                {str(row["source_video_id"]) for row in selected_hidden}
            ),
            "request_count": len(selected_public),
            "frame_unit_ids": frame_ids,
            "frame_unit_ids_sha256": canonical_sha256(frame_ids),
            "request_ids": request_ids,
            "request_ids_sha256": canonical_sha256(request_ids),
            "public_rows_sha256": canonical_sha256(selected_public),
            "public_jsonl_sha256": jsonl_sha256(selected_public),
            "hidden_jsonl_sha256": jsonl_sha256(selected_hidden),
        },
        "overlap_audit": overlap_audit,
        "rejected_source_frame_counts_by_overlap_field": dict(
            sorted(rejected_counts.items())
        ),
        "_selected_public_rows": selected_public,
        "_selected_hidden_rows": selected_hidden,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--exclusion-data-dir", type=Path, required=True)
    parser.add_argument("--out-selection", type=Path)
    parser.add_argument("--out-public", type=Path)
    parser.add_argument("--out-hidden", type=Path)
    parser.add_argument("--verify-selection", type=Path)
    parser.add_argument("--verify-public", type=Path)
    args = parser.parse_args()

    payload = build_selection(args.source_data_dir, args.exclusion_data_dir)
    public_rows = payload.pop("_selected_public_rows")
    hidden_rows = payload.pop("_selected_hidden_rows")
    if args.verify_selection:
        observed = read_json(args.verify_selection)
        if observed != payload:
            raise SelectionError("frozen held-out selection manifest drifted")
    if args.verify_public:
        if read_jsonl(args.verify_public) != public_rows:
            raise SelectionError("frozen held-out public request pack drifted")
    if args.out_selection:
        _write_json(args.out_selection, payload)
    if args.out_public:
        _write_jsonl(args.out_public, public_rows)
    if args.out_hidden:
        _write_jsonl(args.out_hidden, hidden_rows)
    print(
        json.dumps(
            {
                "status": "pass",
                "frame_count": payload["selected"]["frame_count"],
                "request_count": payload["selected"]["request_count"],
                "selection_sha256": canonical_sha256(payload),
                "public_rows_sha256": canonical_sha256(public_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
