#!/usr/bin/env python3
"""EgoHOS provenance and mask-capacity census for burned-data construction.

Annotation-only census over local EgoHOS frame inventory and segmentation masks.
Does not freeze usable model rows or emit method-facing packs.

Connected-component geometry uses an absolute minimum pixel area threshold
(default ``min_component_area=64``). Components below that threshold are retained
as satellites but do not qualify for single-target geometry.

CLI::

    python -m research.lifecycle_memory.phase2_job3_egohos_census \\
        --data-root data/external/egohos \\
        --out-dir data/external/egohos/derived/job3_stage_a2_census \\
        --ego4d-metadata /path/to/ego4d.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

MASK_CLASS_BACKGROUND = 0
MASK_CLASS_LEFT_HAND = 1
MASK_CLASS_RIGHT_HAND = 2
MASK_CLASS_FO_LEFT = 3
MASK_CLASS_FO_RIGHT = 4
MASK_CLASS_FO_BOTH = 5
MASK_CLASS_SO_LEFT = 6
MASK_CLASS_SO_RIGHT = 7
MASK_CLASS_SO_BOTH = 8

FIRST_ORDER_CLASSES = frozenset({3, 4, 5})
SECOND_ORDER_CLASSES = frozenset({6, 7, 8})
CONTACT_CLASSES = frozenset({3, 4, 5, 6, 7, 8})
ALL_MASK_CLASSES = frozenset(range(9))

HAND_REQUIRED_BY_CLASS: dict[int, frozenset[int]] = {
    3: frozenset({MASK_CLASS_LEFT_HAND}),
    4: frozenset({MASK_CLASS_RIGHT_HAND}),
    5: frozenset({MASK_CLASS_LEFT_HAND, MASK_CLASS_RIGHT_HAND}),
    6: frozenset({MASK_CLASS_LEFT_HAND}),
    7: frozenset({MASK_CLASS_RIGHT_HAND}),
    8: frozenset({MASK_CLASS_LEFT_HAND, MASK_CLASS_RIGHT_HAND}),
}

SOURCE_FAMILY_EGO4D = "ego4d"
SOURCE_FAMILY_EPIC = "epic"
SOURCE_FAMILY_THU_READ = "thu_read"
SOURCE_FAMILY_ESCAPE = "escape"
SOURCE_FAMILY_YOUTUBE_OOD = "youtube_ood"

GATE_PREFERRED_MIN_CLUSTERS = 55
GATE_PREFERRED_MAX_CLUSTERS = 60
GATE_MINIMUM_CLUSTERS = 40
DEFAULT_MIN_COMPONENT_AREA = 64

CLAIM_BOUNDARY = (
    "Stage A2 EgoHOS provenance + mask-capacity census. Candidate rows are raw "
    "mask-derived provenance only (rows_frozen=false). They are NOT final usable "
    "model rows. Second-order classes (6/7/8) are pending negatives only."
)
CANDIDATE_CLAIM = (
    "Hidden/provenance-facing candidate sidecar. Not for method-facing model input. "
    "Not frozen usable model rows."
)

INVENTORY_COLUMNS = [
    "split",
    "frame_id",
    "source_family",
    "source_video_id",
    "source_subject_id",
    "source_frame_or_timestamp",
    "confirmation_provenance_status",
    "image_relpath",
    "label_relpath",
    "image_bytes",
    "label_bytes",
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_frame_inventory(path: Path) -> list[dict[str, str]]:
    path = path.expanduser()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing TSV header")
        missing = [col for col in INVENTORY_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing inventory columns: {missing}")
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({col: str(row.get(col) or "") for col in INVENTORY_COLUMNS})
    return rows


def load_ego4d_participant_map(path: Path) -> dict[str, Any]:
    path = path.expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError(f"{path}: expected videos list")

    video_to_participant: dict[str, str] = {}
    null_participant_video_uids: list[str] = []
    all_video_uids: list[str] = []
    seen: set[str] = set()

    for entry in videos:
        if not isinstance(entry, dict):
            continue
        video_uid = str(entry.get("video_uid") or "").strip()
        if not video_uid:
            continue
        if video_uid in seen:
            raise ValueError(f"{path}: duplicate video_uid {video_uid!r}")
        seen.add(video_uid)
        all_video_uids.append(video_uid)
        participant = entry.get("fb_participant_id")
        if participant is None or str(participant).strip() == "":
            null_participant_video_uids.append(video_uid)
            continue
        video_to_participant[video_uid] = str(participant)

    return {
        "video_to_participant": video_to_participant,
        "null_participant_video_uids": sorted(null_participant_video_uids),
        "all_video_uids": sorted(all_video_uids),
        "n_videos": len(all_video_uids),
        "n_mapped": len(video_to_participant),
        "n_null_participant": len(null_participant_video_uids),
    }


def _validate_mask_values(mask_array: np.ndarray) -> None:
    if mask_array.size == 0:
        return
    flat = mask_array.reshape(-1)
    if np.any(flat < 0) or np.any(flat > 8):
        invalid = sorted({int(v) for v in np.unique(flat) if int(v) < 0 or int(v) > 8})
        raise ValueError(f"mask contains invalid class values: {invalid}")


def decode_image_mask_pair(
    image_path: Path,
    mask_path: Path,
) -> tuple[tuple[int, int], np.ndarray]:
    image_path = image_path.expanduser()
    mask_path = mask_path.expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"missing image: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"missing mask: {mask_path}")

    with Image.open(image_path) as image:
        image.load()
        image_hw = (image.height, image.width)
    with Image.open(mask_path) as mask_image:
        mask_array = np.asarray(mask_image)
        if mask_array.ndim == 3:
            if mask_array.shape[2] == 1:
                mask_array = mask_array[:, :, 0]
            else:
                raise ValueError(f"mask must be single-channel: {mask_path}")

    mask_hw = mask_array.shape[:2]
    if image_hw != mask_hw:
        raise ValueError(
            f"image/mask dimension mismatch for {image_path.name}: "
            f"image={image_hw}, mask={mask_hw}"
        )

    mask_array = mask_array.astype(np.int64, copy=False)
    _validate_mask_values(mask_array)
    return image_hw, mask_array


def _decode_failure_kind(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "missing_file"
    message = str(exc)
    if "dimension mismatch" in message:
        return "dimension_mismatch"
    if "invalid class values" in message or "mask must be single-channel" in message:
        return "invalid_mask"
    if isinstance(exc, ValueError):
        return "invalid_data"
    return "decode_error"


def analyze_mask(mask_array: np.ndarray, *, min_component_area: int) -> dict[str, Any]:
    if min_component_area < 1:
        raise ValueError(f"min_component_area must be >= 1, got {min_component_area}")
    _validate_mask_values(mask_array)

    flat = mask_array.reshape(-1)
    class_counts = np.bincount(flat, minlength=max(ALL_MASK_CLASSES) + 1)
    labels_present = [
        class_id
        for class_id, count in enumerate(class_counts[: max(ALL_MASK_CLASSES) + 1])
        if count > 0
    ]
    pixel_counts = {class_id: int(class_counts[class_id]) for class_id in labels_present}

    left_hand_area = int(pixel_counts.get(MASK_CLASS_LEFT_HAND, 0))
    right_hand_area = int(pixel_counts.get(MASK_CLASS_RIGHT_HAND, 0))
    left_hand_present = left_hand_area > 0
    right_hand_present = right_hand_area > 0

    structure = np.ones((3, 3), dtype=int)
    components: dict[str, dict[str, Any]] = {}
    required_hand_ok: dict[str, bool] = {}
    single_target_geometry: dict[str, dict[str, Any]] = {}

    for class_id in labels_present:
        class_mask = mask_array == class_id
        labeled, count = ndimage.label(class_mask, structure=structure)
        component_counts = np.bincount(labeled.reshape(-1), minlength=count + 1)
        areas = [int(area) for area in component_counts[1 : count + 1]]
        areas_sorted = sorted(areas, reverse=True)
        areas_ge_min = [area for area in areas_sorted if area >= min_component_area]
        satellite_areas = [area for area in areas_sorted if area < min_component_area]
        components[str(class_id)] = {
            "count": int(count),
            "areas_sorted": areas_sorted,
            "areas_ge_min": areas_ge_min,
            "satellite_areas": satellite_areas,
        }

    for class_id in sorted(CONTACT_CLASSES):
        required = HAND_REQUIRED_BY_CLASS[class_id]
        ok = all(pixel_counts.get(hand, 0) > 0 for hand in required)
        required_hand_ok[str(class_id)] = ok
        class_key = str(class_id)
        class_components = components.get(class_key, {})
        qualifying = list(class_components.get("areas_ge_min", []))
        single_target_geometry[class_key] = {
            "is_single_target": len(qualifying) == 1,
            "qualifying_count": len(qualifying),
            "satellite_areas": list(class_components.get("satellite_areas", [])),
        }

    first_order = sorted(label for label in labels_present if label in FIRST_ORDER_CLASSES)
    second_order = sorted(label for label in labels_present if label in SECOND_ORDER_CLASSES)

    return {
        "labels_present": labels_present,
        "pixel_counts": pixel_counts,
        "left_hand_present": left_hand_present,
        "right_hand_present": right_hand_present,
        "left_hand_area": left_hand_area,
        "right_hand_area": right_hand_area,
        "first_order_classes_present": first_order,
        "second_order_classes_present": second_order,
        "components": components,
        "required_hand_ok": required_hand_ok,
        "single_target_geometry": single_target_geometry,
    }


def resolve_row_provenance(
    row: dict[str, str],
    ego4d_map_or_none: dict[str, Any] | None,
) -> dict[str, Any]:
    source_family = str(row.get("source_family") or "").strip()
    source_video_id = str(row.get("source_video_id") or "").strip()
    source_subject_id = str(row.get("source_subject_id") or "").strip()

    base = {
        "source_family": source_family,
        "provenance_eligible": False,
        "exclusion_reason": None,
        "cluster_kind": None,
        "cluster_id": None,
        "fb_participant_id": None,
    }

    if source_family == SOURCE_FAMILY_EPIC:
        base["exclusion_reason"] = "epic_overlap_excluded"
        return base

    if source_family == SOURCE_FAMILY_EGO4D:
        if ego4d_map_or_none is None:
            base["exclusion_reason"] = "ego4d_participant_metadata_missing"
            return base
        if not source_video_id:
            base["exclusion_reason"] = "ego4d_video_uid_missing"
            return base
        video_to_participant = ego4d_map_or_none.get("video_to_participant", {})
        null_uids = set(ego4d_map_or_none.get("null_participant_video_uids", []))
        if source_video_id in null_uids:
            base["exclusion_reason"] = "ego4d_participant_null"
            return base
        participant = video_to_participant.get(source_video_id)
        if participant is None:
            base["exclusion_reason"] = "ego4d_participant_unmapped"
            return base
        base.update(
            {
                "provenance_eligible": True,
                "cluster_kind": "ego4d_participant",
                "cluster_id": f"ego4d_participant:{participant}",
                "fb_participant_id": str(participant),
            }
        )
        return base

    if source_family == SOURCE_FAMILY_THU_READ:
        if not source_subject_id:
            base["exclusion_reason"] = "thu_subject_id_missing"
            return base
        base.update(
            {
                "provenance_eligible": True,
                "cluster_kind": "thu_subject",
                "cluster_id": f"thu_subject:{source_subject_id}",
            }
        )
        return base

    if source_family in {SOURCE_FAMILY_ESCAPE, SOURCE_FAMILY_YOUTUBE_OOD}:
        base["exclusion_reason"] = "subject_map_required"
        return base

    base["exclusion_reason"] = "unknown_source_family"
    return base


def deterministic_candidate_id(
    *,
    split: str,
    frame_id: str,
    source_family: str,
    source_video_id: str,
    mask_class: int,
    label_role: str,
) -> str:
    key = "|".join(
        [
            split,
            frame_id,
            source_family,
            source_video_id,
            str(mask_class),
            label_role,
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _opaque_key(*, namespace: str, raw_value: str) -> str:
    digest = hashlib.sha256(
        f"phase2_job3_egohos|{namespace}|{raw_value}".encode("utf-8")
    ).hexdigest()
    return f"{namespace}_{digest[:24]}"


def _public_cluster_key(cluster_id: str) -> str:
    return _opaque_key(namespace="cluster", raw_value=cluster_id)


def _public_row_key(row: dict[str, str]) -> str:
    raw_value = "|".join(str(row.get(column) or "") for column in INVENTORY_COLUMNS)
    return _opaque_key(namespace="row", raw_value=raw_value)


def build_candidate_records_for_row(
    *,
    row: dict[str, str],
    provenance: dict[str, Any],
    mask_analysis: dict[str, Any],
    image_path: str,
    mask_path: str,
    decode_ok: bool,
) -> list[dict[str, Any]]:
    split = str(row.get("split") or "")
    frame_id = str(row.get("frame_id") or "")
    source_family = str(row.get("source_family") or "")
    source_video_id = str(row.get("source_video_id") or "")
    source_subject_id = str(row.get("source_subject_id") or "")
    pixel_counts = mask_analysis.get("pixel_counts", {})
    required_hand_ok = mask_analysis.get("required_hand_ok", {})
    single_target_geometry = mask_analysis.get("single_target_geometry", {})

    candidates: list[dict[str, Any]] = []

    def _append_candidate(class_id: int, label_role: str) -> None:
        class_key = str(class_id)
        pixel_count = int(pixel_counts.get(class_id, 0))
        if pixel_count <= 0:
            return
        hand_ok = bool(required_hand_ok.get(class_key, False))
        if not hand_ok:
            return
        geometry = single_target_geometry.get(class_key, {})
        mask_eligible = bool(decode_ok and hand_ok)
        confirmation_candidate = bool(
            label_role == "positive"
            and class_id in FIRST_ORDER_CLASSES
            and mask_eligible
            and provenance.get("provenance_eligible")
        )
        candidates.append(
            {
                "split": split,
                "frame_id": frame_id,
                "source_family": source_family,
                "source_video_id": source_video_id,
                "source_subject_id": source_subject_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "mask_class": class_id,
                "label_role": label_role,
                "pixel_count": pixel_count,
                "required_hand_ok": hand_ok,
                "mask_eligible": mask_eligible,
                "provenance_eligible": bool(provenance.get("provenance_eligible")),
                "confirmation_candidate": confirmation_candidate,
                "is_single_target": bool(geometry.get("is_single_target", False)),
                "qualifying_component_count": int(geometry.get("qualifying_count", 0)),
                "satellite_areas": list(geometry.get("satellite_areas", [])),
                "cluster_kind": provenance.get("cluster_kind"),
                "cluster_id": provenance.get("cluster_id"),
                "fb_participant_id": provenance.get("fb_participant_id"),
                "exclusion_reason": provenance.get("exclusion_reason"),
                "candidate_id": deterministic_candidate_id(
                    split=split,
                    frame_id=frame_id,
                    source_family=source_family,
                    source_video_id=source_video_id,
                    mask_class=class_id,
                    label_role=label_role,
                ),
                "hidden": True,
                "rows_frozen": False,
                "claim_boundary": CANDIDATE_CLAIM,
            }
        )

    for class_id in sorted(FIRST_ORDER_CLASSES):
        _append_candidate(class_id, "positive")

    for class_id in sorted(SECOND_ORDER_CLASSES):
        _append_candidate(
            class_id,
            "negative_candidate_pending_no_direct_contact_audit",
        )

    return candidates


def _inventory_has_ego4d(rows: list[dict[str, str]]) -> bool:
    return any(str(row.get("source_family") or "") == SOURCE_FAMILY_EGO4D for row in rows)


def _ego4d_mapping_status(
    inventory_rows: list[dict[str, str]],
    ego4d_map_or_none: dict[str, Any] | None,
) -> str:
    """Return not_applicable, missing, incomplete, or complete for inventory rows."""
    has_ego4d = _inventory_has_ego4d(inventory_rows)
    if not has_ego4d:
        return "not_applicable"
    if ego4d_map_or_none is None:
        return "missing"
    video_to_participant = ego4d_map_or_none.get("video_to_participant", {})
    null_uids = set(ego4d_map_or_none.get("null_participant_video_uids", []))
    for row in inventory_rows:
        if str(row.get("source_family") or "") != SOURCE_FAMILY_EGO4D:
            continue
        video_uid = str(row.get("source_video_id") or "").strip()
        if not video_uid:
            return "incomplete"
        if video_uid in null_uids:
            return "incomplete"
        participant = video_to_participant.get(video_uid)
        if participant is None or not str(participant).strip():
            return "incomplete"
    return "complete"


def evaluate_gate_verdict(
    *,
    inventory_rows: list[dict[str, str]],
    ego4d_map_or_none: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    decode_failure_count: int = 0,
) -> dict[str, Any]:
    has_ego4d = _inventory_has_ego4d(inventory_rows)
    ego4d_mapping_status = _ego4d_mapping_status(inventory_rows, ego4d_map_or_none)
    blocked_missing_metadata = ego4d_mapping_status == "missing"
    blocked_incomplete_mapping = ego4d_mapping_status == "incomplete"

    qualifying_clusters: set[str] = set()

    for candidate in candidates:
        cluster_id = candidate.get("cluster_id")
        if not cluster_id:
            continue
        if (
            candidate.get("mask_class") in FIRST_ORDER_CLASSES
            and candidate.get("label_role") == "positive"
            and candidate.get("provenance_eligible")
            and candidate.get("mask_eligible")
            and candidate.get("confirmation_candidate")
        ):
            qualifying_clusters.add(str(cluster_id))

    cluster_count = len(qualifying_clusters)

    if blocked_missing_metadata:
        status = "blocked_missing_ego4d_participant_metadata"
        verdict = "blocked"
    elif blocked_incomplete_mapping:
        status = "blocked_incomplete_ego4d_participant_mapping"
        verdict = "blocked"
    elif decode_failure_count > 0:
        status = "decode_closure_incomplete"
        verdict = "blocked"
    elif cluster_count < GATE_MINIMUM_CLUSTERS:
        status = "below_minimum_clusters"
        verdict = "fail"
    elif cluster_count < GATE_PREFERRED_MIN_CLUSTERS or cluster_count > GATE_PREFERRED_MAX_CLUSTERS:
        status = "outside_preferred_cluster_band"
        verdict = "pass_with_note"
    else:
        status = "within_preferred_cluster_band"
        verdict = "pass"

    return {
        "kind": "phase2_job3_egohos_gate_verdict",
        "created_at": _now(),
        "verdict": verdict,
        "status": status,
        "blocked_missing_ego4d_participant_metadata": blocked_missing_metadata,
        "blocked_incomplete_ego4d_participant_mapping": blocked_incomplete_mapping,
        "ego4d_mapping_status": ego4d_mapping_status,
        "inventory_has_ego4d_rows": has_ego4d,
        "decode_failure_count": decode_failure_count,
        "decode_closure_complete": decode_failure_count == 0,
        "independent_cluster_count": cluster_count,
        "preferred_cluster_band": [GATE_PREFERRED_MIN_CLUSTERS, GATE_PREFERRED_MAX_CLUSTERS],
        "minimum_cluster_count": GATE_MINIMUM_CLUSTERS,
        "qualifying_cluster_keys": sorted(
            _public_cluster_key(cluster_id) for cluster_id in qualifying_clusters
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _aggregate_source_split_counts(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_split: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_family_split: Counter[tuple[str, str]] = Counter()
    for row in rows:
        split = str(row.get("split") or "")
        family = str(row.get("source_family") or "")
        by_split[split] += 1
        by_family[family] += 1
        by_family_split[(family, split)] += 1
    return {
        "kind": "phase2_job3_egohos_source_split_counts",
        "created_at": _now(),
        "row_count": len(rows),
        "by_split": dict(sorted(by_split.items())),
        "by_source_family": dict(sorted(by_family.items())),
        "by_source_family_split": {
            f"{family}\t{split}": count
            for (family, split), count in sorted(by_family_split.items())
        },
    }


def _aggregate_mask_class_capacity(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: Counter[str] = Counter()
    mask_eligible_by_class: Counter[str] = Counter()
    confirmation_by_class: Counter[str] = Counter()
    for candidate in candidates:
        class_key = str(candidate.get("mask_class"))
        by_class[class_key] += 1
        if candidate.get("mask_eligible"):
            mask_eligible_by_class[class_key] += 1
        if candidate.get("confirmation_candidate"):
            confirmation_by_class[class_key] += 1
    return {
        "kind": "phase2_job3_egohos_mask_class_capacity",
        "created_at": _now(),
        "candidate_count_by_class": dict(sorted(by_class.items())),
        "mask_eligible_count_by_class": dict(sorted(mask_eligible_by_class.items())),
        "confirmation_candidate_count_by_class": dict(sorted(confirmation_by_class.items())),
    }


def _aggregate_cluster_capacity(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_cluster: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "candidate_count": 0,
            "mask_eligible_count": 0,
            "confirmation_candidate_count": 0,
            "positive_count": 0,
            "pending_negative_count": 0,
        }
    )
    unclustered = {
        "candidate_count": 0,
        "mask_eligible_count": 0,
        "confirmation_candidate_count": 0,
    }
    for candidate in candidates:
        cluster_id = candidate.get("cluster_id")
        if cluster_id:
            bucket = by_cluster[_public_cluster_key(str(cluster_id))]
        else:
            bucket = unclustered
        bucket["candidate_count"] += 1
        if candidate.get("mask_eligible"):
            bucket["mask_eligible_count"] += 1
        if candidate.get("confirmation_candidate"):
            bucket["confirmation_candidate_count"] += 1
        if cluster_id:
            if candidate.get("label_role") == "positive":
                bucket["positive_count"] += 1
            elif (
                candidate.get("label_role")
                == "negative_candidate_pending_no_direct_contact_audit"
            ):
                bucket["pending_negative_count"] += 1

    return {
        "kind": "phase2_job3_egohos_cluster_capacity",
        "created_at": _now(),
        "cluster_count": len(by_cluster),
        "qualifying_positive_cluster_count": sum(
            1
            for bucket in by_cluster.values()
            if bucket["confirmation_candidate_count"] > 0
        ),
        "by_cluster": {key: dict(value) for key, value in sorted(by_cluster.items())},
        "unclustered": dict(unclustered),
    }


def _write_source_split_tsv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source_family", "split", "row_count"])
        for key, count in sorted(payload.get("by_source_family_split", {}).items()):
            family, split = key.split("\t", 1)
            writer.writerow([family, split, count])


def _write_ego4d_hidden_map(path: Path, ego4d_map: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    video_to_participant = ego4d_map.get("video_to_participant", {})
    null_uids = set(ego4d_map.get("null_participant_video_uids", []))
    all_uids = list(ego4d_map.get("all_video_uids", []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["video_uid", "fb_participant_id", "participant_status"])
        for video_uid in sorted(all_uids):
            if video_uid in video_to_participant:
                writer.writerow([video_uid, video_to_participant[video_uid], "mapped"])
            elif video_uid in null_uids:
                writer.writerow([video_uid, "", "null_participant"])
            else:
                writer.writerow([video_uid, "", "unmapped"])


def _markdown_report(
    *,
    summary: dict[str, Any],
    gate: dict[str, Any],
    mask_capacity: dict[str, Any],
    cluster_capacity: dict[str, Any],
) -> str:
    lines = [
        "# Phase 2 Job3 EgoHOS Census Report (Stage A2)",
        "",
        f"- Gate verdict: **{gate['verdict']}** ({gate['status']})",
        f"- Independent clusters: {gate['independent_cluster_count']}",
        f"- Inventory rows processed: {summary['inventory_row_count']}",
        f"- Decode failures: {summary['decode_failure_count']}",
        f"- Candidate rows: {summary['candidate_row_count']}",
        f"- Confirmation candidates: {summary['confirmation_candidate_count']}",
        f"- Min component area: {summary['min_component_area']}",
        "",
        "## Mask Class Capacity",
    ]
    for class_id, count in sorted(mask_capacity.get("candidate_count_by_class", {}).items()):
        lines.append(f"- class {class_id}: {count} candidates")
    lines.extend(
        [
            "",
            "## Cluster Capacity",
            f"- Provenance clusters with candidates: {cluster_capacity.get('cluster_count', 0)}",
            "",
            "## Claim Boundary",
            summary["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def write_census_outputs(
    *,
    out_dir: Path,
    summary: dict[str, Any],
    gate: dict[str, Any],
    source_split_counts: dict[str, Any],
    mask_capacity: dict[str, Any],
    cluster_capacity: dict[str, Any],
    candidates: list[dict[str, Any]],
    decode_failures_hidden: list[dict[str, Any]],
    ego4d_map_or_none: dict[str, Any] | None,
) -> None:
    out_dir = out_dir.expanduser()
    _write_json(out_dir / "egohos_census_summary.json", summary)
    _write_json(out_dir / "egohos_gate_verdict.json", gate)
    _write_json(out_dir / "egohos_source_split_counts.json", source_split_counts)
    _write_json(out_dir / "egohos_mask_class_capacity.json", mask_capacity)
    _write_json(out_dir / "egohos_cluster_capacity.json", cluster_capacity)
    _write_source_split_tsv(out_dir / "egohos_source_split_counts.tsv", source_split_counts)
    _write_jsonl(out_dir / "hidden" / "egohos_candidate_sidecar.jsonl", candidates)
    _write_jsonl(
        out_dir / "hidden" / "egohos_decode_failures.jsonl",
        decode_failures_hidden,
    )
    (out_dir / "egohos_census_report.md").write_text(
        _markdown_report(
            summary=summary,
            gate=gate,
            mask_capacity=mask_capacity,
            cluster_capacity=cluster_capacity,
        ),
        encoding="utf-8",
    )
    if ego4d_map_or_none is not None:
        _write_ego4d_hidden_map(
            out_dir / "hidden" / "ego4d_video_participant_map.tsv",
            ego4d_map_or_none,
        )


def run_census(
    *,
    data_root: Path,
    inventory_path: Path,
    out_dir: Path,
    ego4d_metadata_path: Path | None = None,
    min_component_area: int = DEFAULT_MIN_COMPONENT_AREA,
    limit: int | None = None,
) -> dict[str, Any]:
    data_root = data_root.expanduser()
    inventory_rows = load_frame_inventory(inventory_path)
    if limit is not None:
        inventory_rows = inventory_rows[:limit]

    ego4d_map: dict[str, Any] | None = None
    if ego4d_metadata_path is not None:
        ego4d_map = load_ego4d_participant_map(ego4d_metadata_path)

    all_candidates: list[dict[str, Any]] = []
    decode_failures_public: list[dict[str, str]] = []
    decode_failures_hidden: list[dict[str, Any]] = []
    decode_failure_kind_counts: Counter[str] = Counter()
    exclusion_reason_counts: Counter[str] = Counter()
    provenance_eligible_rows = 0
    mask_eligible_rows = 0

    for row in inventory_rows:
        provenance = resolve_row_provenance(row, ego4d_map)
        if provenance.get("exclusion_reason"):
            exclusion_reason_counts[str(provenance["exclusion_reason"])] += 1
        if provenance.get("provenance_eligible"):
            provenance_eligible_rows += 1

        image_relpath = str(row.get("image_relpath") or "")
        label_relpath = str(row.get("label_relpath") or "")
        image_path = data_root / "raw" / image_relpath
        mask_path = data_root / "raw" / label_relpath

        try:
            _, mask_array = decode_image_mask_pair(image_path, mask_path)
            mask_analysis = analyze_mask(mask_array, min_component_area=min_component_area)
            decode_ok = True
        except (OSError, ValueError) as exc:
            failure_kind = _decode_failure_kind(exc)
            decode_failure_kind_counts[failure_kind] += 1
            decode_failures_public.append(
                {
                    "row_key": _public_row_key(row),
                    "failure_kind": failure_kind,
                }
            )
            decode_failures_hidden.append(
                {
                    "split": str(row.get("split") or ""),
                    "frame_id": str(row.get("frame_id") or ""),
                    "source_family": str(row.get("source_family") or ""),
                    "source_video_id": str(row.get("source_video_id") or ""),
                    "source_subject_id": str(row.get("source_subject_id") or ""),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "failure_kind": failure_kind,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            mask_analysis = {
                "labels_present": [],
                "pixel_counts": {},
                "left_hand_present": False,
                "right_hand_present": False,
                "left_hand_area": 0,
                "right_hand_area": 0,
                "first_order_classes_present": [],
                "second_order_classes_present": [],
                "components": {},
                "required_hand_ok": {str(c): False for c in sorted(CONTACT_CLASSES)},
                "single_target_geometry": {},
            }
            decode_ok = False

        row_candidates = build_candidate_records_for_row(
            row=row,
            provenance=provenance,
            mask_analysis=mask_analysis,
            image_path=str(image_path),
            mask_path=str(mask_path),
            decode_ok=decode_ok,
        )
        if decode_ok and any(candidate.get("mask_eligible") for candidate in row_candidates):
            mask_eligible_rows += 1
        all_candidates.extend(row_candidates)

    gate = evaluate_gate_verdict(
        inventory_rows=inventory_rows,
        ego4d_map_or_none=ego4d_map,
        candidates=all_candidates,
        decode_failure_count=len(decode_failures_hidden),
    )
    source_split_counts = _aggregate_source_split_counts(inventory_rows)
    mask_capacity = _aggregate_mask_class_capacity(all_candidates)
    cluster_capacity = _aggregate_cluster_capacity(all_candidates)

    summary = {
        "kind": "phase2_job3_egohos_census_summary",
        "created_at": _now(),
        "inventory_path": str(inventory_path),
        "data_root": str(data_root),
        "inventory_row_count": len(inventory_rows),
        "decode_failure_count": len(decode_failures_hidden),
        "decode_failure_count_by_kind": dict(sorted(decode_failure_kind_counts.items())),
        "decode_failures": decode_failures_public,
        "provenance_eligible_row_count": provenance_eligible_rows,
        "mask_eligible_row_count": mask_eligible_rows,
        "candidate_row_count": len(all_candidates),
        "positive_candidate_count": sum(
            1 for row in all_candidates if row.get("label_role") == "positive"
        ),
        "pending_negative_candidate_count": sum(
            1
            for row in all_candidates
            if row.get("label_role") == "negative_candidate_pending_no_direct_contact_audit"
        ),
        "confirmation_candidate_count": sum(
            1 for row in all_candidates if row.get("confirmation_candidate")
        ),
        "exclusion_reason_counts": dict(sorted(exclusion_reason_counts.items())),
        "min_component_area": min_component_area,
        "ego4d_metadata_supplied": ego4d_map is not None,
        "gate_status": gate["status"],
        "gate_verdict": gate["verdict"],
        "independent_cluster_count": gate["independent_cluster_count"],
        "claim_boundary": CLAIM_BOUNDARY,
        "rows_frozen": False,
    }

    write_census_outputs(
        out_dir=out_dir,
        summary=summary,
        gate=gate,
        source_split_counts=source_split_counts,
        mask_capacity=mask_capacity,
        cluster_capacity=cluster_capacity,
        candidates=all_candidates,
        decode_failures_hidden=decode_failures_hidden,
        ego4d_map_or_none=ego4d_map,
    )

    return {
        "summary": summary,
        "gate": gate,
        "candidates": all_candidates,
        "source_split_counts": source_split_counts,
        "mask_capacity": mask_capacity,
        "cluster_capacity": cluster_capacity,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage A2 EgoHOS provenance + mask-capacity census for Phase 2 Job3."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/external/egohos"),
        help="EgoHOS data root containing raw/ and manifests/.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="Frame inventory TSV (default: <data-root>/manifests/frame_inventory.tsv).",
    )
    parser.add_argument(
        "--ego4d-metadata",
        type=Path,
        default=None,
        help="Optional licensed local ego4d.json for participant collapse.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory (expected under data/external/egohos/derived/).",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=DEFAULT_MIN_COMPONENT_AREA,
        help="Absolute minimum connected-component area in pixels (default: 64).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke/debug runs only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser()
    inventory_path = (
        args.inventory.expanduser()
        if args.inventory is not None
        else data_root / "manifests" / "frame_inventory.tsv"
    )
    ego4d_metadata_path = (
        args.ego4d_metadata.expanduser() if args.ego4d_metadata is not None else None
    )
    report = run_census(
        data_root=data_root,
        inventory_path=inventory_path,
        out_dir=args.out_dir.expanduser(),
        ego4d_metadata_path=ego4d_metadata_path,
        min_component_area=args.min_component_area,
        limit=args.limit,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
