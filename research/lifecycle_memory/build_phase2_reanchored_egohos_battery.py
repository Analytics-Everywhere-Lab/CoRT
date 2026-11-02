#!/usr/bin/env python3
"""Build the outcome-blind re-anchored EgoHOS 2AFC battery.

Each selected frame emits both counterbalanced A/B orders for one first-order
and one second-order interacting-object crop. Roles, classes, A/B answers,
provenance, frame-level IDs, and wrong-time donor identity remain evaluator-only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory import phase2_job3_egohos_burned_items as burned_builder
from research.lifecycle_memory import phase2_job3_egohos_census as census


PRESPEC_REL = Path(
    "docs/research/lifecycle_memory/phase 2 research/preregistration/"
    "reanchored_evidence_failure_localization_v1_2026_07_21.json"
)
PRESPEC_SHA256 = "bd0f021fd510d873f388075a3d4c0e0d138da07fc6da8215cc0ff16b1d09279f"
PROMPT_PRESPEC_REL = Path(
    "docs/research/lifecycle_memory/phase 2 research/jobs/"
    "mech_job3_heldout_confirmation/scoring_prespec_prompt_repair_round1_final.json"
)
PROMPT_PRESPEC_SHA256 = "19cbe5775b122dba9a02ced1f62b2c29c12e35d07e4f646cc6af7fdf20c42dc8"

DEFAULT_DATA_ROOT = Path("data/external/egohos")
DEFAULT_INVENTORY_REL = Path(
    "derived/job3_stage_a2_ego4d_mapped_subset_2026_07_17/hidden/"
    "frame_inventory_mapped_subset.tsv"
)
DEFAULT_CANDIDATE_SIDECAR_REL = Path(
    "derived/job3_stage_a2_census_mapped_subset_v2_2026_07_17/hidden/"
    "egohos_candidate_sidecar.jsonl"
)
DEFAULT_CANDIDATE_SIDECAR_SHA256 = (
    "89ff17f0d788f337c659c7a4cec086abcb4fa0ea9a1a7b48d30221eef5c848bd"
)
DEFAULT_BURNED_EVALUATOR_REL = Path(
    "runs/research/lifecycle_memory/phase_2/"
    "job3_burned_egohos_prompt_repair_round1_final_2026_07_21/hidden/"
    "evaluator_index.json"
)
DEFAULT_BURNED_EVALUATOR_SHA256 = (
    "58ac9787a84af6f54cc8608fe13fbd3f783cf1815385db6780b69a036f64a3aa"
)

FRAME_TARGET = 800
TRIALS_PER_FRAME = 2
TRIAL_TARGET = FRAME_TARGET * TRIALS_PER_FRAME
MIN_PARTICIPANT_CLUSTERS = 100
MECHANISM_FRAME_TARGET = 96
MECHANISM_TRIAL_TARGET = MECHANISM_FRAME_TARGET * TRIALS_PER_FRAME
MAX_FRAMES_PER_PARTICIPANT = 10
MIN_DONOR_FRAME_GAP = 900
TARGET_COMPONENT_MIN_AREA = 64
FIRST_ORDER_CLASSES = (3, 4, 5)
SECOND_ORDER_CLASSES = (6, 7, 8)
ARMS = ("real", "blank_full", "wrong_time")
TARGET_COMPONENT_RULE_TEXT = (
    "within the chosen class mask, select the largest 8-connected component with "
    "area at least 64 pixels; break exact area ties by lexicographically smallest "
    "top,left,bottom,right bounding box; render only that component and record its "
    "hidden fingerprint"
)
MECHANISM_SELECTION_NAMESPACE = "phase2_reanchored_egohos_mechanism_subset_v1"
MECHANISM_SUBSET_RULE = (
    "outcome_blind_participant_first_hash_order_one_frame_per_participant_before_fill"
)

# Both sizes are multiples of 28, which keeps the source pixels fixed while
# leaving processor-specific token-shape validation to the model runtime.
FULL_CANVAS_WIDTH = 896
FULL_CANVAS_HEIGHT = 672
CROP_CANVAS_WIDTH = 336
CROP_CANVAS_HEIGHT = 336
FULL_CANVAS_SIZE = (FULL_CANVAS_WIDTH, FULL_CANVAS_HEIGHT)
CROP_CANVAS_SIZE = (CROP_CANVAS_WIDTH, CROP_CANVAS_HEIGHT)
CANVAS_MATTE_RGB = (0, 0, 0)

SELECTION_NAMESPACE = "phase2_reanchored_egohos_2afc_selection_v1"
ID_NAMESPACE = "phase2_reanchored_egohos_2afc_ids_v1"
PROMPT_SOURCE_ID = "reanchored_egohos_first_vs_second_order_2afc_v1"
SYSTEM_PROMPT = (
    "Answer the first-versus-second-order target question using only the provided "
    "RGB inputs. Return exactly one JSON object with keys: answer, abstain, "
    "abstention_reason, confidence. The answer value must be A or B."
)
EXPECTED_USER_PROMPT = (
    "Input 1 is a full frame. Inputs 2 and 3 are unmarked target-only reference "
    "crops A and B. Which referenced target is the first-order interacting object, "
    "meaning the object directly manipulated by a visible hand rather than a "
    "second-order object interacting through it? Return exactly one JSON object "
    "with answer A or B plus the frozen abstention and confidence keys."
)
CANONICAL_A_JSON = (
    '{"answer":"A","abstain":false,"abstention_reason":null,"confidence":1.0}'
)
CANONICAL_B_JSON = (
    '{"answer":"B","abstain":false,"abstention_reason":null,"confidence":1.0}'
)

CLAIM_BOUNDARY = (
    "Annotation-native paired 2AFC construction. Class 6-8 rows are used only as "
    "second-order interacting-object annotations. The two counterbalanced orders "
    "are repeated trials within one frame-level unit and are not independent frames."
)

FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_json",
        "canonical_answer_json",
        "candidate_id",
        "class_id",
        "cluster_id",
        "component_area",
        "component_bbox_convention",
        "component_bbox_tlbr",
        "contact_role",
        "crop_a_selected_component_fingerprint",
        "crop_b_selected_component_fingerprint",
        "donor_id",
        "donor_frame_id",
        "donor_frame_unit",
        "donor_video_id",
        "fb_participant_id",
        "first_order_candidate_id",
        "first_order_class",
        "first_order_component",
        "frame_id",
        "frame_level_unit_id",
        "frame_pair_id",
        "image_relpath",
        "label_relpath",
        "label_role",
        "mask_class",
        "multi_component_indicator",
        "participant_id",
        "provenance",
        "qualifying_component_count",
        "second_order_candidate_id",
        "second_order_class",
        "second_order_component",
        "selected_component_fingerprint",
        "source_frame_or_timestamp",
        "source_video_id",
        "split",
        "target_role",
    }
)
MECHANISM_FORBIDDEN_PUBLIC_KEYS = FORBIDDEN_PUBLIC_KEYS | frozenset(
    {
        "class",
        "correct_choice",
        "ego4d_participant_id",
        "first_order",
        "label",
        "participant",
        "participant_cluster_id",
        "role",
        "second_order",
        "source_frame_unit",
        "source_frame_level_unit_id",
        "source_trial_id",
        "target_component_fingerprints",
        "target_component_rule",
    }
)


class ConstructionError(ValueError):
    """The requested cohort or output violates a frozen construction rule."""


class CapacityError(ConstructionError):
    """The requested frame or participant capacity is unavailable."""


@dataclass(frozen=True)
class InventoryFrame:
    split: str
    frame_id: str
    source_video_id: str
    frame_unit: int
    image_relpath: str
    label_relpath: str

    @property
    def key(self) -> str:
        return f"{self.split}|{self.frame_id}"


@dataclass(frozen=True)
class TargetCandidate:
    candidate_id: str
    class_id: int
    component_area: int | None = None
    component_bbox_tlbr: tuple[int, int, int, int] | None = None
    qualifying_component_count: int | None = None
    selected_component_fingerprint: str | None = None
    multi_component_indicator: bool | None = None


@dataclass(frozen=True)
class SelectedTargetComponent:
    mask: np.ndarray
    area: int
    bbox_tlbr: tuple[int, int, int, int]
    qualifying_component_count: int
    fingerprint: str

    @property
    def multi_component_indicator(self) -> bool:
        return self.qualifying_component_count > 1


@dataclass(frozen=True)
class FramePool:
    frame: InventoryFrame
    participant_id: str
    cluster_id: str
    targets_by_class: Mapping[int, TargetCandidate]
    donor: InventoryFrame

    @property
    def first_classes(self) -> tuple[int, ...]:
        return tuple(class_id for class_id in FIRST_ORDER_CLASSES if class_id in self.targets_by_class)

    @property
    def second_classes(self) -> tuple[int, ...]:
        return tuple(class_id for class_id in SECOND_ORDER_CLASSES if class_id in self.targets_by_class)


@dataclass(frozen=True)
class SelectedFrame:
    pool: FramePool
    first_order: TargetCandidate
    second_order: TargetCandidate
    frame_level_unit_id: str
    base_first_order_is_a: bool


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[SelectedFrame, ...]
    audit: dict[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _component_fingerprint(
    *,
    component_mask: np.ndarray,
    class_id: int,
    area: int,
    bbox_tlbr: tuple[int, int, int, int],
) -> str:
    canonical_header = json.dumps(
        {
            "area": area,
            "bbox_tlbr": list(bbox_tlbr),
            "class_id": class_id,
            "mask_shape_hw": list(component_mask.shape),
            "version": "selected_component_mask_sha256_v1",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    packed_mask = np.packbits(
        np.asarray(component_mask, dtype=bool).reshape(-1),
        bitorder="big",
    ).tobytes()
    return _sha256_bytes(canonical_header + b"\n" + packed_mask)


def _select_target_component(
    mask_array: np.ndarray,
    class_id: int,
    *,
    min_area: int = TARGET_COMPONENT_MIN_AREA,
) -> SelectedTargetComponent | None:
    """Return the frozen largest qualifying 8-connected class component."""
    if min_area < 1:
        raise ValueError("min_area must be >= 1")
    mask = np.asarray(mask_array)
    if mask.ndim != 2:
        raise ValueError("mask_array must be two-dimensional")
    class_mask = mask == int(class_id)
    if not np.any(class_mask):
        return None

    labeled, component_count = ndimage.label(
        class_mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    if component_count == 0:
        return None
    component_areas = np.bincount(
        labeled.reshape(-1),
        minlength=component_count + 1,
    )
    component_slices = ndimage.find_objects(labeled)
    qualifying: list[tuple[int, tuple[int, int, int, int], int]] = []
    for label_id, component_slice in enumerate(component_slices, start=1):
        area = int(component_areas[label_id])
        if area < min_area or component_slice is None:
            continue
        row_slice, column_slice = component_slice
        bbox_tlbr = (
            int(row_slice.start),
            int(column_slice.start),
            int(row_slice.stop),
            int(column_slice.stop),
        )
        qualifying.append((area, bbox_tlbr, label_id))
    if not qualifying:
        return None

    area, bbox_tlbr, label_id = min(
        qualifying,
        key=lambda item: (-item[0], item[1]),
    )
    component_mask = labeled == label_id
    return SelectedTargetComponent(
        mask=component_mask,
        area=area,
        bbox_tlbr=bbox_tlbr,
        qualifying_component_count=len(qualifying),
        fingerprint=_component_fingerprint(
            component_mask=component_mask,
            class_id=int(class_id),
            area=area,
            bbox_tlbr=bbox_tlbr,
        ),
    )


def _target_component_contract() -> dict[str, Any]:
    return {
        "connectivity": 8,
        "minimum_area_pixels": TARGET_COMPONENT_MIN_AREA,
        "selection": "largest_area_then_lexicographic_bbox_tlbr",
        "bbox_convention": "top_left_inclusive_bottom_right_exclusive",
        "render_scope": "selected_component_only",
        "fingerprint": "sha256_canonical_header_and_packed_selected_mask_v1",
        "required_sidecar_gate": "required_hand_ok_is_true",
        "prespec_rule": TARGET_COMPONENT_RULE_TEXT,
    }


def _rank(*parts: object) -> str:
    joined = "|".join(str(part) for part in parts)
    return _sha256_text(f"{SELECTION_NAMESPACE}|{joined}")


def _opaque_id(kind: str, *parts: object) -> str:
    joined = "|".join(str(part) for part in parts)
    return f"{kind}_{_sha256_text(f'{ID_NAMESPACE}|{kind}|{joined}')[:24]}"


def _mechanism_rank(*parts: object) -> str:
    joined = "|".join(str(part) for part in parts)
    return _sha256_text(f"{MECHANISM_SELECTION_NAMESPACE}|{joined}")


def select_mechanism_subset(
    selected: Sequence[SelectedFrame],
    *,
    frame_target: int = MECHANISM_FRAME_TARGET,
) -> tuple[SelectedFrame, ...]:
    """Select one outcome-blind frame from each of distinct participant clusters."""
    if frame_target < 1:
        raise ValueError("mechanism frame_target must be positive")
    by_participant: dict[str, list[SelectedFrame]] = defaultdict(list)
    for selected_frame in selected:
        by_participant[selected_frame.pool.participant_id].append(selected_frame)
    if len(by_participant) < frame_target:
        raise CapacityError(
            "mechanism participant-cluster capacity below target: "
            f"{len(by_participant)} < {frame_target}"
        )

    participants = sorted(
        by_participant,
        key=lambda participant: (
            _mechanism_rank("participant", participant),
            participant,
        ),
    )
    chosen: list[SelectedFrame] = []
    for participant in participants[:frame_target]:
        participant_frames = by_participant[participant]
        chosen.append(
            min(
                participant_frames,
                key=lambda row: (
                    _mechanism_rank(
                        "frame",
                        participant,
                        row.frame_level_unit_id,
                    ),
                    row.frame_level_unit_id,
                ),
            )
        )
    if len(chosen) != frame_target:
        raise CapacityError("mechanism subset selection underfilled")
    if len({row.pool.participant_id for row in chosen}) != frame_target:
        raise ConstructionError("mechanism subset participant uniqueness failed")
    if len({row.frame_level_unit_id for row in chosen}) != frame_target:
        raise ConstructionError("mechanism subset frame uniqueness failed")
    return tuple(chosen)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    body = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    _atomic_write_bytes(path, body.encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConstructionError(f"{path}: invalid JSON on line {line_number}") from exc
            if not isinstance(row, dict):
                raise ConstructionError(f"{path}: line {line_number} is not an object")
            rows.append(row)
    return rows


def _load_burned_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConstructionError(f"cannot read burned evaluator index {path}: {exc}") from exc
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("answers", payload.get("rows", payload.get("items")))
    else:
        rows = None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ConstructionError(f"{path}: burned evaluator index has no object row list")
    return [dict(row) for row in rows]


def _burned_identifiers(
    rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, set[str]]:
    identifiers = {
        "candidate_ids": set(),
        "frame_keys": set(),
        "frame_ids": set(),
        "participant_ids": set(),
        "video_ids": set(),
    }
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        frame_id = str(row.get("frame_id") or "").strip()
        split = str(row.get("split") or "").strip()
        participant = str(
            row.get("fb_participant_id")
            or row.get("participant_id")
            or row.get("ego4d_participant_id")
            or row.get("source_subject_id")
            or ""
        ).strip()
        video = str(row.get("source_video_id") or "").strip()
        if candidate_id:
            identifiers["candidate_ids"].add(candidate_id)
        if frame_id:
            identifiers["frame_ids"].add(frame_id)
            if split:
                identifiers["frame_keys"].add(f"{split}|{frame_id}")
        if participant:
            identifiers["participant_ids"].add(participant)
        if video:
            identifiers["video_ids"].add(video)
    # Candidate-only evaluator rows still burn the candidate's complete frame.
    for row in candidate_rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id not in identifiers["candidate_ids"]:
            continue
        frame_id = str(row.get("frame_id") or "").strip()
        split = str(row.get("split") or "").strip()
        if frame_id:
            identifiers["frame_ids"].add(frame_id)
            if split:
                identifiers["frame_keys"].add(f"{split}|{frame_id}")
    return identifiers


def _load_inventory(path: Path) -> list[dict[str, str]]:
    return census.load_frame_inventory(path)


def _inventory_index(rows: Sequence[Mapping[str, str]]) -> dict[str, InventoryFrame]:
    indexed: dict[str, InventoryFrame] = {}
    for row in rows:
        if str(row.get("source_family") or "").strip() != census.SOURCE_FAMILY_EGO4D:
            continue
        split = str(row.get("split") or "").strip()
        frame_id = str(row.get("frame_id") or "").strip()
        video_id = str(row.get("source_video_id") or "").strip()
        frame_value = str(row.get("source_frame_or_timestamp") or "").strip()
        image_relpath = str(row.get("image_relpath") or "").strip()
        label_relpath = str(row.get("label_relpath") or "").strip()
        if not split or not frame_id or not video_id or not image_relpath:
            continue
        try:
            frame_unit = int(frame_value)
        except ValueError:
            continue
        frame = InventoryFrame(
            split=split,
            frame_id=frame_id,
            source_video_id=video_id,
            frame_unit=frame_unit,
            image_relpath=image_relpath,
            label_relpath=label_relpath,
        )
        if frame.key in indexed:
            raise ConstructionError(f"duplicate Ego4D inventory frame key: {frame.key}")
        indexed[frame.key] = frame
    return indexed


def _candidate_is_mapped_ego4d(row: Mapping[str, Any]) -> bool:
    participant = str(row.get("fb_participant_id") or "").strip()
    cluster_id = str(row.get("cluster_id") or "").strip()
    return bool(
        row.get("source_family") == census.SOURCE_FAMILY_EGO4D
        and row.get("provenance_eligible") is True
        and row.get("mask_eligible") is True
        and row.get("required_hand_ok") is True
        and row.get("cluster_kind") == "ego4d_participant"
        and participant
        and cluster_id == f"ego4d_participant:{participant}"
    )


def _eligible_candidate_groups(
    candidate_rows: Sequence[Mapping[str, Any]],
    burned_ids: Mapping[str, set[str]],
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    seen_candidate_ids: set[str] = set()
    for source_row in candidate_rows:
        row = dict(source_row)
        class_id = row.get("mask_class")
        candidate_id = str(row.get("candidate_id") or "").strip()
        split = str(row.get("split") or "").strip()
        frame_id = str(row.get("frame_id") or "").strip()
        participant = str(row.get("fb_participant_id") or "").strip()
        video_id = str(row.get("source_video_id") or "").strip()
        if class_id not in {*FIRST_ORDER_CLASSES, *SECOND_ORDER_CLASSES}:
            rejected["outside_target_classes"] += 1
            continue
        if row.get("required_hand_ok") is not True:
            rejected["required_hand_not_ok"] += 1
            continue
        if not _candidate_is_mapped_ego4d(row):
            rejected["not_mapped_eligible_ego4d"] += 1
            continue
        if not candidate_id or re.fullmatch(r"[0-9a-f]{64}", candidate_id) is None:
            raise ConstructionError("mapped candidate has an invalid candidate_id")
        if candidate_id in seen_candidate_ids:
            raise ConstructionError(f"duplicate candidate_id: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        frame_key = f"{split}|{frame_id}"
        if candidate_id in burned_ids["candidate_ids"]:
            rejected["burned_candidate"] += 1
            continue
        if frame_key in burned_ids["frame_keys"] or frame_id in burned_ids["frame_ids"]:
            rejected["burned_frame"] += 1
            continue
        if participant in burned_ids["participant_ids"]:
            rejected["burned_participant"] += 1
            continue
        if video_id in burned_ids["video_ids"]:
            rejected["burned_video"] += 1
            continue
        grouped[frame_key].append(row)
    return dict(grouped), rejected


def _geometry_viability_filter(
    *,
    data_root: Path,
    inventory: Mapping[str, InventoryFrame],
    grouped_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    checked_frames = 0
    for frame_key in sorted(grouped_candidates, key=lambda key: _rank("geometry", key)):
        rows = grouped_candidates[frame_key]
        classes = {int(row["mask_class"]) for row in rows}
        if not classes.intersection(FIRST_ORDER_CLASSES) or not classes.intersection(
            SECOND_ORDER_CLASSES
        ):
            rejection_counts["frame_missing_role_before_geometry"] += 1
            continue
        frame = inventory.get(frame_key)
        if frame is None:
            rejection_counts["frame_missing_inventory"] += 1
            continue
        image_path = data_root / "raw" / frame.image_relpath
        mask_path = data_root / "raw" / frame.label_relpath
        try:
            _, mask_array = census.decode_image_mask_pair(image_path, mask_path)
        except (OSError, ValueError) as exc:
            raise ConstructionError(
                f"geometry preflight decode failed for {frame_key}: {type(exc).__name__}: {exc}"
            ) from exc
        checked_frames += 1
        viable_rows: list[dict[str, Any]] = []
        for row in rows:
            class_id = int(row["mask_class"])
            component = _select_target_component(mask_array, class_id)
            if component is None:
                rejection_counts[f"class_{class_id}_no_qualifying_component"] += 1
                continue
            retained = burned_builder._reference_interior_mask(component.mask)
            retained_area = int(np.sum(retained))
            if retained_area < burned_builder.REFERENCE_CROP_RETAINED_AREA_FLOOR:
                rejection_counts[f"class_{class_id}_crop_nonviable"] += 1
                continue
            viable_row = dict(row)
            viable_row.update(
                {
                    "_selected_component_area": component.area,
                    "_selected_component_bbox_tlbr": list(component.bbox_tlbr),
                    "_selected_component_qualifying_count": (
                        component.qualifying_component_count
                    ),
                    "_selected_component_fingerprint": component.fingerprint,
                    "_selected_component_multi": component.multi_component_indicator,
                }
            )
            viable_rows.append(viable_row)
        viable_classes = {int(row["mask_class"]) for row in viable_rows}
        if not viable_classes.intersection(FIRST_ORDER_CLASSES) or not viable_classes.intersection(
            SECOND_ORDER_CLASSES
        ):
            rejection_counts["frame_missing_role_after_geometry"] += 1
            continue
        kept.extend(viable_rows)
    viable_component_counts = Counter(
        "multi_component" if row["_selected_component_multi"] else "single_component"
        for row in kept
    )
    return kept, {
        "checked_frame_count": checked_frames,
        "viable_candidate_count": len(kept),
        "viable_single_component_candidate_count": viable_component_counts[
            "single_component"
        ],
        "viable_multi_component_candidate_count": viable_component_counts[
            "multi_component"
        ],
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "reference_crop_interior_rule": "uniform_square_3x3_binary_erosion",
        "reference_crop_erosion_radius_pixels": burned_builder.REFERENCE_CROP_EROSION_RADIUS,
        "reference_crop_minimum_retained_pixels": (
            burned_builder.REFERENCE_CROP_RETAINED_AREA_FLOOR
        ),
        "target_component_rule": _target_component_contract(),
    }


def _balanced_targets(total: int, classes: Sequence[int]) -> dict[int, int]:
    base, remainder = divmod(total, len(classes))
    return {
        class_id: base + (1 if index < remainder else 0)
        for index, class_id in enumerate(classes)
    }


def _pick_donor(
    source: InventoryFrame,
    candidates: Sequence[InventoryFrame],
) -> InventoryFrame | None:
    eligible = [
        donor
        for donor in candidates
        if donor.key != source.key
        and abs(donor.frame_unit - source.frame_unit) >= MIN_DONOR_FRAME_GAP
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda donor: (_rank("donor", source.key, donor.key), donor.key))


def _target_candidate_from_row(row: Mapping[str, Any]) -> TargetCandidate:
    metadata_keys = (
        "_selected_component_area",
        "_selected_component_bbox_tlbr",
        "_selected_component_qualifying_count",
        "_selected_component_fingerprint",
        "_selected_component_multi",
    )
    present = [key in row for key in metadata_keys]
    if not any(present):
        return TargetCandidate(
            candidate_id=str(row["candidate_id"]),
            class_id=int(row["mask_class"]),
        )
    if not all(present):
        raise ConstructionError("candidate has incomplete selected-component metadata")

    area = row["_selected_component_area"]
    bbox = row["_selected_component_bbox_tlbr"]
    qualifying_count = row["_selected_component_qualifying_count"]
    fingerprint = row["_selected_component_fingerprint"]
    multi = row["_selected_component_multi"]
    if not isinstance(area, int) or isinstance(area, bool) or area < TARGET_COMPONENT_MIN_AREA:
        raise ConstructionError("candidate selected-component area is invalid")
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(not isinstance(value, int) or isinstance(value, bool) for value in bbox)
    ):
        raise ConstructionError("candidate selected-component bbox is invalid")
    bbox_tlbr = tuple(bbox)
    top, left, bottom, right = bbox_tlbr
    if top < 0 or left < 0 or bottom <= top or right <= left:
        raise ConstructionError("candidate selected-component bbox is empty or inverted")
    if area > (bottom - top) * (right - left):
        raise ConstructionError("candidate selected-component area exceeds its bbox")
    if (
        not isinstance(qualifying_count, int)
        or isinstance(qualifying_count, bool)
        or qualifying_count < 1
    ):
        raise ConstructionError("candidate qualifying-component count is invalid")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ConstructionError("candidate selected-component fingerprint is invalid")
    if not isinstance(multi, bool) or multi != (qualifying_count > 1):
        raise ConstructionError("candidate multi-component indicator is invalid")
    return TargetCandidate(
        candidate_id=str(row["candidate_id"]),
        class_id=int(row["mask_class"]),
        component_area=area,
        component_bbox_tlbr=bbox_tlbr,
        qualifying_component_count=qualifying_count,
        selected_component_fingerprint=fingerprint,
        multi_component_indicator=multi,
    )


def _component_provenance(target: TargetCandidate) -> dict[str, Any]:
    if (
        target.component_area is None
        or target.component_bbox_tlbr is None
        or target.qualifying_component_count is None
        or target.selected_component_fingerprint is None
        or target.multi_component_indicator is None
    ):
        raise ConstructionError(
            f"selected candidate lacks component provenance: {target.candidate_id}"
        )
    return {
        "component_area": target.component_area,
        "component_bbox_tlbr": list(target.component_bbox_tlbr),
        "component_bbox_convention": "top_left_inclusive_bottom_right_exclusive",
        "qualifying_component_count": target.qualifying_component_count,
        "selected_component_fingerprint": target.selected_component_fingerprint,
        "multi_component_indicator": target.multi_component_indicator,
    }


def _assert_component_matches_preflight(
    *,
    target: TargetCandidate,
    rendered: SelectedTargetComponent,
    frame_key: str,
) -> None:
    expected = _component_provenance(target)
    actual = {
        "component_area": rendered.area,
        "component_bbox_tlbr": list(rendered.bbox_tlbr),
        "component_bbox_convention": "top_left_inclusive_bottom_right_exclusive",
        "qualifying_component_count": rendered.qualifying_component_count,
        "selected_component_fingerprint": rendered.fingerprint,
        "multi_component_indicator": rendered.multi_component_indicator,
    }
    if actual != expected:
        raise ConstructionError(
            "rendered target component disagrees with geometry preflight: "
            f"{frame_key} candidate={target.candidate_id}"
        )


def _selected_component_audit(
    selected: Sequence[SelectedFrame],
) -> dict[str, Any]:
    overall: Counter[str] = Counter()
    by_role: dict[str, Counter[str]] = {
        "first_order": Counter(),
        "second_order": Counter(),
    }
    for selected_frame in selected:
        for role, target in (
            ("first_order", selected_frame.first_order),
            ("second_order", selected_frame.second_order),
        ):
            if target.multi_component_indicator is None:
                category = "unknown"
            elif target.multi_component_indicator:
                category = "multi_component"
            else:
                category = "single_component"
            overall[category] += 1
            by_role[role][category] += 1
    return {
        "selected_target_count": len(selected) * 2,
        "single_component_selected_target_count": overall["single_component"],
        "multi_component_selected_target_count": overall["multi_component"],
        "unknown_component_metadata_count": overall["unknown"],
        "by_role": {
            role: {
                "single_component_selected_target_count": counts["single_component"],
                "multi_component_selected_target_count": counts["multi_component"],
                "unknown_component_metadata_count": counts["unknown"],
            }
            for role, counts in by_role.items()
        },
    }


def _build_frame_pools(
    *,
    inventory_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, Any]],
    burned_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[FramePool], dict[str, Any]]:
    inventory = _inventory_index(inventory_rows)
    burned_ids = _burned_identifiers(burned_rows, candidate_rows)
    grouped, candidate_rejections = _eligible_candidate_groups(candidate_rows, burned_ids)

    donors_by_video: dict[str, list[InventoryFrame]] = defaultdict(list)
    for frame in inventory.values():
        if frame.key in burned_ids["frame_keys"] or frame.frame_id in burned_ids["frame_ids"]:
            continue
        if frame.source_video_id in burned_ids["video_ids"]:
            continue
        donors_by_video[frame.source_video_id].append(frame)
    for video_id in donors_by_video:
        donors_by_video[video_id].sort(key=lambda frame: (frame.frame_unit, frame.key))

    pools: list[FramePool] = []
    frame_rejections: Counter[str] = Counter()
    for frame_key, rows in grouped.items():
        frame = inventory.get(frame_key)
        if frame is None:
            frame_rejections["missing_inventory"] += 1
            continue
        participant_ids = {str(row.get("fb_participant_id") or "") for row in rows}
        cluster_ids = {str(row.get("cluster_id") or "") for row in rows}
        video_ids = {str(row.get("source_video_id") or "") for row in rows}
        if len(participant_ids) != 1 or len(cluster_ids) != 1 or video_ids != {
            frame.source_video_id
        }:
            raise ConstructionError(f"candidate provenance disagrees within {frame_key}")
        targets: dict[int, TargetCandidate] = {}
        for row in rows:
            class_id = int(row["mask_class"])
            if class_id in targets:
                raise ConstructionError(f"duplicate class {class_id} candidate for {frame_key}")
            targets[class_id] = _target_candidate_from_row(row)
        if not set(targets).intersection(FIRST_ORDER_CLASSES):
            frame_rejections["missing_first_order_target"] += 1
            continue
        if not set(targets).intersection(SECOND_ORDER_CLASSES):
            frame_rejections["missing_second_order_target"] += 1
            continue
        donor = _pick_donor(frame, donors_by_video.get(frame.source_video_id, ()))
        if donor is None:
            frame_rejections["missing_wrong_time_donor"] += 1
            continue
        pools.append(
            FramePool(
                frame=frame,
                participant_id=next(iter(participant_ids)),
                cluster_id=next(iter(cluster_ids)),
                targets_by_class=targets,
                donor=donor,
            )
        )
    pools.sort(key=lambda pool: (_rank("frame", pool.frame.key), pool.frame.key))
    return pools, {
        "candidate_rejection_counts": dict(sorted(candidate_rejections.items())),
        "frame_rejection_counts": dict(sorted(frame_rejections.items())),
        "burned_identifier_counts": {
            key: len(values) for key, values in sorted(burned_ids.items())
        },
        "mapped_inventory_frame_count": len(inventory),
        "eligible_frame_pool_count": len(pools),
    }


def _choice_score(
    *,
    pool: FramePool,
    first_class: int,
    second_class: int,
    first_counts: Counter[int],
    second_counts: Counter[int],
    pair_counts: Counter[tuple[int, int]],
    participant_counts: Counter[str],
    first_targets: Mapping[int, int],
    second_targets: Mapping[int, int],
    class_availability: Mapping[int, int],
) -> tuple[Any, ...]:
    first_deficit = first_targets[first_class] - first_counts[first_class]
    second_deficit = second_targets[second_class] - second_counts[second_class]
    return (
        int(first_deficit <= 0) + int(second_deficit <= 0),
        -max(first_deficit, 0) - max(second_deficit, 0),
        -max(first_deficit, 0),
        -max(second_deficit, 0),
        class_availability[first_class],
        class_availability[second_class],
        pair_counts[(first_class, second_class)],
        participant_counts[pool.participant_id],
        _rank("choice", pool.frame.key, first_class, second_class),
    )


def _select_pools_greedily(
    pools: Sequence[FramePool],
    *,
    target_frames: int,
    maximum_frames_per_participant: int = MAX_FRAMES_PER_PARTICIPANT,
) -> tuple[list[FramePool], dict[str, Any]]:
    first_targets = _balanced_targets(target_frames, FIRST_ORDER_CLASSES)
    second_targets = _balanced_targets(target_frames, SECOND_ORDER_CLASSES)
    class_availability = Counter(
        class_id for pool in pools for class_id in {*pool.first_classes, *pool.second_classes}
    )
    first_counts: Counter[int] = Counter()
    second_counts: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    participant_counts: Counter[str] = Counter()
    selected_keys: set[str] = set()
    selected: list[FramePool] = []

    def choose(pool_subset: Sequence[FramePool]) -> tuple[FramePool, int, int] | None:
        best: tuple[tuple[Any, ...], FramePool, int, int] | None = None
        for pool in pool_subset:
            if pool.frame.key in selected_keys:
                continue
            if participant_counts[pool.participant_id] >= maximum_frames_per_participant:
                continue
            for first_class in pool.first_classes:
                for second_class in pool.second_classes:
                    score = _choice_score(
                        pool=pool,
                        first_class=first_class,
                        second_class=second_class,
                        first_counts=first_counts,
                        second_counts=second_counts,
                        pair_counts=pair_counts,
                        participant_counts=participant_counts,
                        first_targets=first_targets,
                        second_targets=second_targets,
                        class_availability=class_availability,
                    )
                    candidate = (score, pool, first_class, second_class)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
        if best is None:
            return None
        return best[1], best[2], best[3]

    def commit(choice: tuple[FramePool, int, int]) -> None:
        pool, first_class, second_class = choice
        selected.append(pool)
        selected_keys.add(pool.frame.key)
        participant_counts[pool.participant_id] += 1
        first_counts[first_class] += 1
        second_counts[second_class] += 1
        pair_counts[(first_class, second_class)] += 1

    by_participant: dict[str, list[FramePool]] = defaultdict(list)
    for pool in pools:
        by_participant[pool.participant_id].append(pool)
    participant_order = sorted(
        by_participant,
        key=lambda participant: (_rank("participant", participant), participant),
    )
    for participant in participant_order[:target_frames]:
        choice = choose(by_participant[participant])
        if choice is None:
            raise ConstructionError(f"eligible participant has no selectable frame: {participant}")
        commit(choice)

    while len(selected) < target_frames:
        choice = choose(pools)
        if choice is None:
            break
        commit(choice)

    return selected, {
        "greedy_first_order_counts": {
            str(class_id): first_counts[class_id] for class_id in FIRST_ORDER_CLASSES
        },
        "greedy_second_order_counts": {
            str(class_id): second_counts[class_id] for class_id in SECOND_ORDER_CLASSES
        },
        "greedy_pair_counts": {
            f"{first_class}:{second_class}": pair_counts[(first_class, second_class)]
            for first_class in FIRST_ORDER_CLASSES
            for second_class in SECOND_ORDER_CLASSES
        },
    }


def _select_balanced_options(
    pools: Sequence[FramePool],
    *,
    target_frames: int,
    minimum_participant_clusters: int,
    maximum_frames_per_participant: int = MAX_FRAMES_PER_PARTICIPANT,
) -> tuple[list[tuple[FramePool, int, int]], dict[str, Any]]:
    """Solve the frozen frame/class allocation with exact hard constraints."""
    options = sorted(
        (
            (pool, first_class, second_class)
            for pool in pools
            for first_class in pool.first_classes
            for second_class in pool.second_classes
        ),
        key=lambda value: (
            _rank("milp-option", value[0].frame.key, value[1], value[2]),
            value[0].frame.key,
            value[1],
            value[2],
        ),
    )
    participants = sorted(
        {pool.participant_id for pool in pools},
        key=lambda value: (_rank("milp-participant", value), value),
    )
    first_targets = _balanced_targets(target_frames, FIRST_ORDER_CLASSES)
    second_targets = _balanced_targets(target_frames, SECOND_ORDER_CLASSES)
    class_targets = {**first_targets, **second_targets}
    pairs = tuple(
        (first_class, second_class)
        for first_class in FIRST_ORDER_CLASSES
        for second_class in SECOND_ORDER_CLASSES
    )
    pair_target_values = _balanced_targets(target_frames, tuple(range(len(pairs))))
    pair_targets = {
        pair_value: pair_target_values[index] for index, pair_value in enumerate(pairs)
    }

    n_options = len(options)
    participant_offset = n_options
    class_deviation_offset = participant_offset + len(participants)
    pair_deviation_offset = class_deviation_offset + 2 * len(class_targets)
    n_variables = pair_deviation_offset + 2 * len(pairs)

    option_indexes_by_frame: dict[str, list[int]] = defaultdict(list)
    option_indexes_by_participant: dict[str, list[int]] = defaultdict(list)
    option_indexes_by_class: dict[int, list[int]] = defaultdict(list)
    option_indexes_by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
    for option_index, (pool, first_class, second_class) in enumerate(options):
        option_indexes_by_frame[pool.frame.key].append(option_index)
        option_indexes_by_participant[pool.participant_id].append(option_index)
        option_indexes_by_class[first_class].append(option_index)
        option_indexes_by_class[second_class].append(option_index)
        option_indexes_by_pair[(first_class, second_class)].append(option_index)

    constraint_specs: list[tuple[dict[int, float], float, float]] = []
    constraint_specs.append(({index: 1.0 for index in range(n_options)}, target_frames, target_frames))
    for frame_key in sorted(option_indexes_by_frame):
        constraint_specs.append(
            ({index: 1.0 for index in option_indexes_by_frame[frame_key]}, 0.0, 1.0)
        )
    participant_index = {value: index for index, value in enumerate(participants)}
    for participant in participants:
        indexes = option_indexes_by_participant[participant]
        y_index = participant_offset + participant_index[participant]
        cap_row = {index: 1.0 for index in indexes}
        cap_row[y_index] = -float(maximum_frames_per_participant)
        constraint_specs.append((cap_row, -np.inf, 0.0))
        selected_row = {index: -1.0 for index in indexes}
        selected_row[y_index] = 1.0
        constraint_specs.append((selected_row, -np.inf, 0.0))
    constraint_specs.append(
        (
            {
                participant_offset + index: 1.0
                for index in range(len(participants))
            },
            minimum_participant_clusters,
            np.inf,
        )
    )

    class_order = tuple((*FIRST_ORDER_CLASSES, *SECOND_ORDER_CLASSES))
    for class_index, class_id in enumerate(class_order):
        row = {index: 1.0 for index in option_indexes_by_class[class_id]}
        row[class_deviation_offset + 2 * class_index] = -1.0
        row[class_deviation_offset + 2 * class_index + 1] = 1.0
        constraint_specs.append((row, class_targets[class_id], class_targets[class_id]))
    for pair_index, pair_value in enumerate(pairs):
        row = {index: 1.0 for index in option_indexes_by_pair[pair_value]}
        row[pair_deviation_offset + 2 * pair_index] = -1.0
        row[pair_deviation_offset + 2 * pair_index + 1] = 1.0
        constraint_specs.append((row, pair_targets[pair_value], pair_targets[pair_value]))

    matrix = lil_matrix((len(constraint_specs), n_variables), dtype=np.float64)
    lower = np.empty(len(constraint_specs), dtype=np.float64)
    upper = np.empty(len(constraint_specs), dtype=np.float64)
    for row_index, (coefficients, row_lower, row_upper) in enumerate(constraint_specs):
        for column_index, coefficient in coefficients.items():
            matrix[row_index, column_index] = coefficient
        lower[row_index] = row_lower
        upper[row_index] = row_upper

    objective = np.zeros(n_variables, dtype=np.float64)
    for option_index, (pool, first_class, second_class) in enumerate(options):
        rank_value = int(
            _rank("milp-tiebreak", pool.frame.key, first_class, second_class)[:13], 16
        ) / float(16**13)
        objective[option_index] = rank_value * 1e-3
    objective[participant_offset:class_deviation_offset] = -10.0
    objective[class_deviation_offset:pair_deviation_offset] = 1_000_000.0
    objective[pair_deviation_offset:] = 100.0

    integrality = np.zeros(n_variables, dtype=np.int8)
    integrality[:class_deviation_offset] = 1
    variable_lower = np.zeros(n_variables, dtype=np.float64)
    variable_upper = np.full(n_variables, np.inf, dtype=np.float64)
    variable_upper[:class_deviation_offset] = 1.0
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"mip_rel_gap": 0.0, "time_limit": 120.0, "presolve": True},
    )
    if not result.success or result.x is None:
        raise CapacityError(
            "balanced frame/class allocation did not solve to optimality: "
            f"status={result.status} message={result.message}"
        )
    chosen_indexes = [
        index for index, value in enumerate(result.x[:n_options]) if value > 0.5
    ]
    if len(chosen_indexes) != target_frames:
        raise ConstructionError(
            f"balanced optimizer selected {len(chosen_indexes)} frames, expected {target_frames}"
        )
    chosen = [options[index] for index in chosen_indexes]
    if len({value[0].frame.key for value in chosen}) != target_frames:
        raise ConstructionError("balanced optimizer selected duplicate source frames")
    first_counts = Counter(value[1] for value in chosen)
    second_counts = Counter(value[2] for value in chosen)
    pair_counts = Counter((value[1], value[2]) for value in chosen)
    participant_counts = Counter(value[0].participant_id for value in chosen)
    chosen.sort(
        key=lambda value: (
            _rank("selected-order", value[0].frame.key),
            value[0].frame.key,
        )
    )
    return chosen, {
        "method": "scipy_milp_exact_hard_constraints_lexicographic_weighted_objective",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective_value": float(result.fun),
        "option_count": n_options,
        "selected_participant_count": len(participant_counts),
        "first_order_target_counts": {
            str(class_id): first_targets[class_id] for class_id in FIRST_ORDER_CLASSES
        },
        "first_order_actual_counts": {
            str(class_id): first_counts[class_id] for class_id in FIRST_ORDER_CLASSES
        },
        "second_order_target_counts": {
            str(class_id): second_targets[class_id] for class_id in SECOND_ORDER_CLASSES
        },
        "second_order_actual_counts": {
            str(class_id): second_counts[class_id] for class_id in SECOND_ORDER_CLASSES
        },
        "pair_actual_counts": {
            f"{first_class}:{second_class}": pair_counts[(first_class, second_class)]
            for first_class, second_class in pairs
        },
    }


def _match_class_quotas(
    pools: Sequence[FramePool],
    *,
    role: str,
    quotas: Mapping[int, int],
) -> dict[str, int] | None:
    if role == "first_order":
        allowed_for = lambda pool: pool.first_classes
    elif role == "second_order":
        allowed_for = lambda pool: pool.second_classes
    else:
        raise ValueError(f"unknown role: {role}")
    slots = [
        (class_id, slot_index)
        for class_id in sorted(quotas)
        for slot_index in range(quotas[class_id])
    ]
    slots_by_class: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for slot in slots:
        slots_by_class[slot[0]].append(slot)
    owner: dict[tuple[int, int], str] = {}
    pools_by_key = {pool.frame.key: pool for pool in pools}
    assignment_slot: dict[str, tuple[int, int]] = {}

    def augment(frame_key: str, seen_slots: set[tuple[int, int]]) -> bool:
        pool = pools_by_key[frame_key]
        class_order = sorted(
            allowed_for(pool),
            key=lambda class_id: (_rank("match-class", role, frame_key, class_id), class_id),
        )
        for class_id in class_order:
            for slot in slots_by_class[class_id]:
                if slot in seen_slots:
                    continue
                seen_slots.add(slot)
                previous = owner.get(slot)
                if previous is None or augment(previous, seen_slots):
                    owner[slot] = frame_key
                    assignment_slot[frame_key] = slot
                    return True
        return False

    ordered = sorted(
        pools,
        key=lambda pool: (
            len(allowed_for(pool)),
            _rank("match-frame", role, pool.frame.key),
            pool.frame.key,
        ),
    )
    for pool in ordered:
        if not augment(pool.frame.key, set()):
            return None
    return {frame_key: slot[0] for frame_key, slot in assignment_slot.items()}


def _balanced_class_assignment(
    pools: Sequence[FramePool],
    *,
    role: str,
    classes: Sequence[int],
) -> tuple[dict[str, int], dict[str, Any]]:
    target = _balanced_targets(len(pools), classes)
    exact = _match_class_quotas(pools, role=role, quotas=target)
    if exact is not None:
        return exact, {
            "exact_equal_quota_feasible": True,
            "target_counts": {str(key): value for key, value in target.items()},
            "actual_counts": {str(key): value for key, value in target.items()},
        }

    allowed_for = (
        (lambda pool: pool.first_classes)
        if role == "first_order"
        else (lambda pool: pool.second_classes)
    )
    counts: Counter[int] = Counter()
    assignment: dict[str, int] = {}
    ordered = sorted(
        pools,
        key=lambda pool: (
            len(allowed_for(pool)),
            _rank("fallback-frame", role, pool.frame.key),
        ),
    )
    for pool in ordered:
        class_id = min(
            allowed_for(pool),
            key=lambda value: (
                counts[value],
                _rank("fallback-class", role, pool.frame.key, value),
            ),
        )
        assignment[pool.frame.key] = class_id
        counts[class_id] += 1

    improved = True
    while improved:
        improved = False
        baseline = (
            max(counts[class_id] for class_id in classes)
            - min(counts[class_id] for class_id in classes),
            sum(abs(counts[class_id] - len(pools) / len(classes)) for class_id in classes),
            tuple(counts[class_id] for class_id in classes),
        )
        for pool in ordered:
            current = assignment[pool.frame.key]
            for alternative in allowed_for(pool):
                if alternative == current:
                    continue
                counts[current] -= 1
                counts[alternative] += 1
                objective = (
                    max(counts[class_id] for class_id in classes)
                    - min(counts[class_id] for class_id in classes),
                    sum(
                        abs(counts[class_id] - len(pools) / len(classes))
                        for class_id in classes
                    ),
                    tuple(counts[class_id] for class_id in classes),
                )
                if objective < baseline:
                    assignment[pool.frame.key] = alternative
                    improved = True
                    baseline = objective
                    break
                counts[alternative] -= 1
                counts[current] += 1
            if improved:
                break

    return assignment, {
        "exact_equal_quota_feasible": False,
        "target_counts": {str(key): value for key, value in target.items()},
        "actual_counts": {str(class_id): counts[class_id] for class_id in classes},
    }


def select_frame_pairs(
    *,
    inventory_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, Any]],
    burned_rows: Sequence[Mapping[str, Any]],
    target_frames: int = FRAME_TARGET,
    minimum_participant_clusters: int = MIN_PARTICIPANT_CLUSTERS,
    maximum_frames_per_participant: int = MAX_FRAMES_PER_PARTICIPANT,
    governing_prespec_sha256: str = PRESPEC_SHA256,
) -> SelectionResult:
    """Select frame pairs without using model outputs or evaluator outcomes."""
    if target_frames < 1 or target_frames > FRAME_TARGET:
        raise ValueError(f"target_frames must be in [1, {FRAME_TARGET}]")
    if minimum_participant_clusters < 1 or minimum_participant_clusters > target_frames:
        raise ValueError("minimum_participant_clusters must be in [1, target_frames]")
    if maximum_frames_per_participant < 1:
        raise ValueError("maximum_frames_per_participant must be positive")

    pools, pool_audit = _build_frame_pools(
        inventory_rows=inventory_rows,
        candidate_rows=candidate_rows,
        burned_rows=burned_rows,
    )
    by_participant = Counter(pool.participant_id for pool in pools)
    capped_capacity = sum(
        min(maximum_frames_per_participant, count) for count in by_participant.values()
    )
    if capped_capacity < target_frames:
        raise CapacityError(
            f"frame capacity below target after participant cap: {capped_capacity} < {target_frames}"
        )
    if len(by_participant) < minimum_participant_clusters:
        raise CapacityError(
            "participant-cluster capacity below target: "
            f"{len(by_participant)} < {minimum_participant_clusters}"
        )

    selected_options, optimization_audit = _select_balanced_options(
        pools,
        target_frames=target_frames,
        minimum_participant_clusters=minimum_participant_clusters,
        maximum_frames_per_participant=maximum_frames_per_participant,
    )
    if len(selected_options) != target_frames:
        raise CapacityError(
            f"selection underfilled: {len(selected_options)} < {target_frames}"
        )

    selected: list[SelectedFrame] = []
    participant_counts: Counter[str] = Counter()
    first_counts: Counter[int] = Counter()
    second_counts: Counter[int] = Counter()
    donor_gap_minimum: int | None = None
    for pool, first_class, second_class in selected_options:
        first = pool.targets_by_class[first_class]
        second = pool.targets_by_class[second_class]
        unit_id = _opaque_id(
            "frameunit",
            pool.frame.key,
            first.candidate_id,
            first.selected_component_fingerprint or "component-metadata-unavailable",
            second.candidate_id,
            second.selected_component_fingerprint or "component-metadata-unavailable",
        )
        base_first_is_a = int(_rank("ab-order", unit_id), 16) % 2 == 0
        selected.append(
            SelectedFrame(
                pool=pool,
                first_order=first,
                second_order=second,
                frame_level_unit_id=unit_id,
                base_first_order_is_a=base_first_is_a,
            )
        )
        participant_counts[pool.participant_id] += 1
        first_counts[first_class] += 1
        second_counts[second_class] += 1
        gap = abs(pool.frame.frame_unit - pool.donor.frame_unit)
        donor_gap_minimum = gap if donor_gap_minimum is None else min(donor_gap_minimum, gap)

    if len({row.pool.frame.key for row in selected}) != target_frames:
        raise ConstructionError("selected source frames are not distinct")
    if max(participant_counts.values(), default=0) > maximum_frames_per_participant:
        raise ConstructionError("participant frame cap was exceeded")
    if len(participant_counts) < minimum_participant_clusters:
        raise ConstructionError("selected participant cluster count is below target")
    if donor_gap_minimum is None or donor_gap_minimum < MIN_DONOR_FRAME_GAP:
        raise ConstructionError("wrong-time donor gap invariant failed")

    selection_rows = [
        {
            "frame_level_unit_id": row.frame_level_unit_id,
            "split": row.pool.frame.split,
            "frame_id": row.pool.frame.frame_id,
            "source_video_id": row.pool.frame.source_video_id,
            "source_frame_unit": row.pool.frame.frame_unit,
            "participant_id": row.pool.participant_id,
            "cluster_id": row.pool.cluster_id,
            "first_order_candidate_id": row.first_order.candidate_id,
            "first_order_class": row.first_order.class_id,
            "first_order_component": (
                _component_provenance(row.first_order)
                if row.first_order.selected_component_fingerprint is not None
                else None
            ),
            "second_order_candidate_id": row.second_order.candidate_id,
            "second_order_class": row.second_order.class_id,
            "second_order_component": (
                _component_provenance(row.second_order)
                if row.second_order.selected_component_fingerprint is not None
                else None
            ),
            "donor_id": _opaque_id("donor", row.pool.donor.key),
            "donor_frame_id": row.pool.donor.frame_id,
            "donor_video_id": row.pool.donor.source_video_id,
            "donor_frame_unit": row.pool.donor.frame_unit,
            "donor_gap_frame_units": abs(
                row.pool.frame.frame_unit - row.pool.donor.frame_unit
            ),
            "base_first_order_is_a": row.base_first_order_is_a,
        }
        for row in selected
    ]
    audit = {
        "kind": "phase2_reanchored_egohos_job_a_2afc_selection_audit",
        "selection_namespace": SELECTION_NAMESPACE,
        "governing_prespec_sha256": governing_prespec_sha256,
        "outcome_blind": True,
        "target_frame_count": target_frames,
        "selected_frame_count": len(selected),
        "counterbalanced_trials_per_frame": TRIALS_PER_FRAME,
        "trial_count": len(selected) * TRIALS_PER_FRAME,
        "minimum_participant_clusters": minimum_participant_clusters,
        "selected_participant_clusters": len(participant_counts),
        "max_frames_per_participant": maximum_frames_per_participant,
        "maximum_observed_frames_per_participant": max(participant_counts.values()),
        "eligible_capped_frame_capacity": capped_capacity,
        "eligible_participant_cluster_count": len(by_participant),
        "first_order_class_counts": {
            str(class_id): first_counts[class_id] for class_id in FIRST_ORDER_CLASSES
        },
        "second_order_class_counts": {
            str(class_id): second_counts[class_id] for class_id in SECOND_ORDER_CLASSES
        },
        "first_order_balance": {
            "target_counts": optimization_audit["first_order_target_counts"],
            "actual_counts": optimization_audit["first_order_actual_counts"],
            "globally_optimized": True,
        },
        "second_order_balance": {
            "target_counts": optimization_audit["second_order_target_counts"],
            "actual_counts": optimization_audit["second_order_actual_counts"],
            "globally_optimized": True,
        },
        "minimum_wrong_time_gap_observed": donor_gap_minimum,
        "same_video_donor_all": all(
            row.pool.frame.source_video_id == row.pool.donor.source_video_id
            for row in selected
        ),
        "participant_frame_counts": dict(sorted(participant_counts.items())),
        "selected_target_components": _selected_component_audit(selected),
        "target_component_contract": _target_component_contract(),
        "pool_audit": pool_audit,
        "balanced_selection_optimization": optimization_audit,
        "selected_frames": selection_rows,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return SelectionResult(selected=tuple(selected), audit=audit)


def _fit_fixed_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    transposed = ImageOps.exif_transpose(image).convert("RGB")
    contained = ImageOps.contain(transposed, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, CANVAS_MATTE_RGB)
    x = (size[0] - contained.width) // 2
    y = (size[1] - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _save_fixed_asset(path: Path, image: Image.Image, size: tuple[int, int]) -> dict[str, Any]:
    fixed = _fit_fixed_canvas(image, size)
    data = _png_bytes(fixed)
    _atomic_write_bytes(path, data)
    return {
        "path": path,
        "sha256": _sha256_bytes(data),
        "width": fixed.width,
        "height": fixed.height,
        "bytes": len(data),
    }


def _public_asset(record: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    return {
        "path": str(Path(record["path"]).relative_to(out_dir)),
        "sha256": record["sha256"],
        "width": record["width"],
        "height": record["height"],
    }


def _prepare_out_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise ConstructionError(f"out-dir is not a directory: {resolved}")
        entry = next(resolved.iterdir(), None)
        if entry is not None:
            raise ConstructionError(f"out-dir must be absent or empty: found {entry.name}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _render_selected(
    *,
    data_root: Path,
    out_dir: Path,
    selected: Sequence[SelectedFrame],
    user_prompt: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    public_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    hidden_rows: list[dict[str, Any]] = []
    source_assets: dict[str, dict[str, Any]] = {}
    donor_assets: dict[str, dict[str, Any]] = {}
    crop_assets: dict[str, dict[str, Any]] = {}
    crop_audits: dict[str, dict[str, Any]] = {}
    rendered_component_counts: Counter[str] = Counter()
    component_fingerprint_match_count = 0

    blank_path = out_dir / "assets" / "full" / "blank_full.png"
    blank_record = _save_fixed_asset(
        blank_path,
        Image.new("RGB", FULL_CANVAS_SIZE, CANVAS_MATTE_RGB),
        FULL_CANVAS_SIZE,
    )

    for selected_frame in selected:
        pool = selected_frame.pool
        source_asset_key = pool.frame.key
        if source_asset_key not in source_assets:
            source_path = data_root / "raw" / pool.frame.image_relpath
            with Image.open(source_path) as source_image:
                source_assets[source_asset_key] = _save_fixed_asset(
                    out_dir
                    / "assets"
                    / "full"
                    / f"{_opaque_id('asset', 'source', pool.frame.key)}.png",
                    source_image,
                    FULL_CANVAS_SIZE,
                )
        donor_asset_key = pool.donor.key
        if donor_asset_key not in donor_assets:
            donor_path = data_root / "raw" / pool.donor.image_relpath
            with Image.open(donor_path) as donor_image:
                donor_assets[donor_asset_key] = _save_fixed_asset(
                    out_dir
                    / "assets"
                    / "full"
                    / f"{_opaque_id('asset', 'donor', pool.donor.key)}.png",
                    donor_image,
                    FULL_CANVAS_SIZE,
                )

        source_path = data_root / "raw" / pool.frame.image_relpath
        mask_path = data_root / "raw" / pool.frame.label_relpath
        _, mask_array = census.decode_image_mask_pair(source_path, mask_path)
        with Image.open(source_path) as source_image:
            image_rgb = np.asarray(ImageOps.exif_transpose(source_image).convert("RGB"))
        if image_rgb.shape[:2] != mask_array.shape:
            raise ConstructionError(f"EXIF-normalized source/mask shape mismatch: {pool.frame.key}")
        hand_mask = (mask_array == census.MASK_CLASS_LEFT_HAND) | (
            mask_array == census.MASK_CLASS_RIGHT_HAND
        )
        for target in (selected_frame.first_order, selected_frame.second_order):
            if target.candidate_id in crop_assets:
                continue
            component = _select_target_component(mask_array, target.class_id)
            if component is None:
                raise ConstructionError(
                    "selected target has no qualifying component during render: "
                    f"{pool.frame.key} class={target.class_id}"
                )
            _assert_component_matches_preflight(
                target=target,
                rendered=component,
                frame_key=pool.frame.key,
            )
            component_fingerprint_match_count += 1
            rendered_component_counts[
                "multi_component"
                if component.multi_component_indicator
                else "single_component"
            ] += 1
            crop_rgb, crop_audit = burned_builder._build_reference_crop(
                image_rgb=image_rgb,
                target_component_mask=component.mask,
                hand_mask=hand_mask,
            )
            crop_record = _save_fixed_asset(
                out_dir
                / "assets"
                / "reference_crop"
                / f"{_opaque_id('asset', 'crop', target.candidate_id)}.png",
                Image.fromarray(crop_rgb, mode="RGB"),
                CROP_CANVAS_SIZE,
            )
            crop_assets[target.candidate_id] = crop_record
            crop_audits[target.candidate_id] = crop_audit

    for selected_frame in selected:
        pool = selected_frame.pool
        for order_index in range(TRIALS_PER_FRAME):
            first_is_a = (
                selected_frame.base_first_order_is_a
                if order_index == 0
                else not selected_frame.base_first_order_is_a
            )
            target_a = (
                selected_frame.first_order if first_is_a else selected_frame.second_order
            )
            target_b = (
                selected_frame.second_order if first_is_a else selected_frame.first_order
            )
            answer = "A" if first_is_a else "B"
            trial_id = _opaque_id(
                "trial", selected_frame.frame_level_unit_id, order_index
            )
            item_id = _opaque_id("item", trial_id)
            request_ids: dict[str, str] = {}
            for arm in ARMS:
                request_id = _opaque_id("request", trial_id, arm)
                request_ids[arm] = request_id
                if arm == "real":
                    full_record = source_assets[pool.frame.key]
                    description = "fixed_canvas_full_frame"
                elif arm == "blank_full":
                    full_record = blank_record
                    description = "fixed_canvas_blank_full_frame"
                else:
                    full_record = donor_assets[pool.donor.key]
                    description = "fixed_canvas_wrong_time_full_frame"
                public_by_arm[arm].append(
                    {
                        "request_id": request_id,
                        "item_id": item_id,
                        "arm": arm,
                        "benchmark_split": "phase2_reanchored_egohos_job_a_2afc",
                        "system_prompt": SYSTEM_PROMPT,
                        "user_prompt": user_prompt,
                        "inputs": [
                            {
                                "kind": "full_frame",
                                "description": description,
                                **_public_asset(full_record, out_dir),
                            },
                            {
                                "kind": "reference_crop_a",
                                "description": "fixed_canvas_unmarked_target_crop_a",
                                **_public_asset(crop_assets[target_a.candidate_id], out_dir),
                            },
                            {
                                "kind": "reference_crop_b",
                                "description": "fixed_canvas_unmarked_target_crop_b",
                                **_public_asset(crop_assets[target_b.candidate_id], out_dir),
                            },
                        ],
                    }
                )
            hidden_rows.append(
                {
                    "trial_id": trial_id,
                    "item_id": item_id,
                    "frame_level_unit_id": selected_frame.frame_level_unit_id,
                    "counterbalance_order_index": order_index,
                    "answer": answer,
                    "answer_json": CANONICAL_A_JSON if answer == "A" else CANONICAL_B_JSON,
                    "request_ids": request_ids,
                    "split": pool.frame.split,
                    "frame_id": pool.frame.frame_id,
                    "source_video_id": pool.frame.source_video_id,
                    "source_frame_unit": pool.frame.frame_unit,
                    "participant_id": pool.participant_id,
                    "cluster_id": pool.cluster_id,
                    "first_order_candidate_id": selected_frame.first_order.candidate_id,
                    "first_order_class": selected_frame.first_order.class_id,
                    "first_order_component": _component_provenance(
                        selected_frame.first_order
                    ),
                    "second_order_candidate_id": selected_frame.second_order.candidate_id,
                    "second_order_class": selected_frame.second_order.class_id,
                    "second_order_component": _component_provenance(
                        selected_frame.second_order
                    ),
                    "crop_a_candidate_id": target_a.candidate_id,
                    "crop_a_selected_component_fingerprint": (
                        target_a.selected_component_fingerprint
                    ),
                    "crop_a_target_role": (
                        "first_order_interacting_object"
                        if first_is_a
                        else "second_order_interacting_object"
                    ),
                    "crop_b_candidate_id": target_b.candidate_id,
                    "crop_b_selected_component_fingerprint": (
                        target_b.selected_component_fingerprint
                    ),
                    "crop_b_target_role": (
                        "second_order_interacting_object"
                        if first_is_a
                        else "first_order_interacting_object"
                    ),
                    "donor_id": _opaque_id("donor", pool.donor.key),
                    "donor_frame_id": pool.donor.frame_id,
                    "donor_video_id": pool.donor.source_video_id,
                    "donor_frame_unit": pool.donor.frame_unit,
                    "donor_gap_frame_units": abs(
                        pool.frame.frame_unit - pool.donor.frame_unit
                    ),
                    "source_image_relpath": pool.frame.image_relpath,
                    "source_label_relpath": pool.frame.label_relpath,
                    "donor_image_relpath": pool.donor.image_relpath,
                    "assets": {
                        "real_full": _public_asset(source_assets[pool.frame.key], out_dir),
                        "blank_full": _public_asset(blank_record, out_dir),
                        "wrong_time_full": _public_asset(donor_assets[pool.donor.key], out_dir),
                        "crop_a": _public_asset(crop_assets[target_a.candidate_id], out_dir),
                        "crop_b": _public_asset(crop_assets[target_b.candidate_id], out_dir),
                    },
                    "crop_audit": {
                        "first_order": crop_audits[selected_frame.first_order.candidate_id],
                        "second_order": crop_audits[selected_frame.second_order.candidate_id],
                    },
                }
            )

    render_audit = {
        "source_full_asset_count": len(source_assets),
        "wrong_time_full_asset_count": len(donor_assets),
        "reference_crop_asset_count": len(crop_assets),
        "component_fingerprint_match_count": component_fingerprint_match_count,
        "component_fingerprint_mismatch_count": 0,
        "single_component_selected_target_count": rendered_component_counts[
            "single_component"
        ],
        "multi_component_selected_target_count": rendered_component_counts[
            "multi_component"
        ],
        "blank_full_sha256": blank_record["sha256"],
        "full_frame_dimensions": {
            "width": FULL_CANVAS_WIDTH,
            "height": FULL_CANVAS_HEIGHT,
        },
        "reference_crop_dimensions": {
            "width": CROP_CANVAS_WIDTH,
            "height": CROP_CANVAS_HEIGHT,
        },
    }
    return public_by_arm, hidden_rows, render_audit


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _walk_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_string_values(child)
    elif isinstance(value, str):
        yield value


def _public_leakage_failures(
    public_payloads: Mapping[str, Any],
    selected: Sequence[SelectedFrame],
) -> list[str]:
    failures: list[str] = []
    keys = set(_walk_keys(public_payloads))
    for forbidden in sorted(FORBIDDEN_PUBLIC_KEYS.intersection(keys)):
        failures.append(f"forbidden_public_key:{forbidden}")
    serialized = json.dumps(public_payloads, sort_keys=True)
    public_string_values = set(_walk_string_values(public_payloads))
    dynamic_values: set[str] = set()
    participant_aliases: set[str] = set()
    for row in selected:
        dynamic_values.update(
            {
                row.pool.frame.frame_id,
                row.pool.frame.source_video_id,
                row.pool.frame.image_relpath,
                row.pool.frame.label_relpath,
                row.pool.donor.frame_id,
                row.pool.donor.image_relpath,
                row.first_order.candidate_id,
                row.second_order.candidate_id,
                row.pool.participant_id,
                row.pool.cluster_id,
            }
        )
        participant_aliases.update(
            {
                row.pool.participant_id,
                row.pool.cluster_id,
                f"ego4d_participant:{row.pool.participant_id}",
            }
        )
    for value in sorted(participant_aliases.intersection(public_string_values)):
        failures.append(f"participant_value_public_leakage:{value}")
    for alias in sorted(participant_aliases):
        boundary_pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
        )
        if any(boundary_pattern.search(value) for value in public_string_values):
            failure = f"participant_value_public_leakage:{alias}"
            if failure not in failures:
                failures.append(failure)
    for value in sorted(dynamic_values):
        if value and len(value) >= 8 and value in serialized:
            failures.append(f"dynamic_public_leakage:{value}")
    return failures


def _asset_identity(asset: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(asset.get("path") or ""),
        str(asset.get("sha256") or ""),
        int(asset.get("width") or 0),
        int(asset.get("height") or 0),
    )


def _write_mechanism_subset(
    *,
    out_dir: Path,
    selected: Sequence[SelectedFrame],
    public_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    hidden_rows: Sequence[Mapping[str, Any]],
    source_frame_count: int,
    production_eligible: bool,
    selection_audit_sha256: str,
    governing_prespec_sha256: str = PRESPEC_SHA256,
    mechanism_frame_target: int = MECHANISM_FRAME_TARGET,
) -> dict[str, Any]:
    mechanism_trial_target = mechanism_frame_target * TRIALS_PER_FRAME
    mechanism_selected = select_mechanism_subset(
        selected, frame_target=mechanism_frame_target
    )
    public_by_item = {
        arm: {str(row["item_id"]): dict(row) for row in public_by_arm[arm]}
        for arm in ARMS
    }
    hidden_by_frame_order = {
        (str(row["frame_level_unit_id"]), int(row["counterbalance_order_index"])): dict(
            row
        )
        for row in hidden_rows
    }

    mechanism_public_rows: list[dict[str, Any]] = []
    mechanism_hidden_rows: list[dict[str, Any]] = []
    public_frame_ids: list[str] = []
    for selected_frame in mechanism_selected:
        public_frame_id = _opaque_id(
            "mechanismframe",
            selected_frame.frame_level_unit_id,
        )
        request_ids = [
            _opaque_id(
                "mechanismrequest",
                selected_frame.frame_level_unit_id,
                order_index,
            )
            for order_index in range(TRIALS_PER_FRAME)
        ]
        public_frame_ids.append(public_frame_id)
        for order_index in range(TRIALS_PER_FRAME):
            source_hidden = hidden_by_frame_order.get(
                (selected_frame.frame_level_unit_id, order_index)
            )
            if source_hidden is None:
                raise ConstructionError(
                    "mechanism subset cannot join a selected frame to both hidden orders"
                )
            item_id = str(source_hidden["item_id"])
            source_public = {
                arm: public_by_item[arm].get(item_id) for arm in ARMS
            }
            if any(row is None for row in source_public.values()):
                raise ConstructionError(
                    f"mechanism subset cannot join public arms for item {item_id}"
                )
            real_row = source_public["real"]
            assert real_row is not None
            for arm, arm_row in source_public.items():
                assert arm_row is not None
                if arm_row["system_prompt"] != real_row["system_prompt"]:
                    raise ConstructionError(
                        f"mechanism cross-arm system prompt mismatch for {item_id}:{arm}"
                    )
                if arm_row["user_prompt"] != real_row["user_prompt"]:
                    raise ConstructionError(
                        f"mechanism cross-arm user prompt mismatch for {item_id}:{arm}"
                    )
            request_id = request_ids[order_index]
            counterpart_request_id = request_ids[1 - order_index]
            mechanism_public_rows.append(
                {
                    "schema_version": 1,
                    "request_id": request_id,
                    "frame_unit_id": public_frame_id,
                    "order_id": f"order_{order_index}",
                    "counterbalanced_request_id": counterpart_request_id,
                    "system_prompt": str(real_row["system_prompt"]),
                    "user_prompt": str(real_row["user_prompt"]),
                    "arms": {
                        arm: {
                            "inputs": [dict(asset) for asset in source_public[arm]["inputs"]]
                        }
                        for arm in ARMS
                    },
                }
            )
            mechanism_hidden_rows.append(
                {
                    "schema_version": 1,
                    "request_id": request_id,
                    "frame_unit_id": public_frame_id,
                    "order_id": f"order_{order_index}",
                    "counterbalanced_request_id": counterpart_request_id,
                    "correct_choice": str(source_hidden["answer"]),
                    "ego4d_participant_id": selected_frame.pool.participant_id,
                    "participant_cluster_id": selected_frame.pool.cluster_id,
                    "frame_id": selected_frame.pool.frame.frame_id,
                    "source_video_id": selected_frame.pool.frame.source_video_id,
                    "source_frame_unit": selected_frame.pool.frame.frame_unit,
                    "source_trial_id": str(source_hidden["trial_id"]),
                    "source_frame_level_unit_id": selected_frame.frame_level_unit_id,
                    "donor_id": str(source_hidden["donor_id"]),
                    "target_component_rule": TARGET_COMPONENT_RULE_TEXT,
                    "target_component_fingerprints": {
                        "A": str(
                            source_hidden["crop_a_selected_component_fingerprint"]
                        ),
                        "B": str(
                            source_hidden["crop_b_selected_component_fingerprint"]
                        ),
                    },
                }
            )

    public_keys = set(_walk_keys(mechanism_public_rows))
    leaked_keys = sorted(MECHANISM_FORBIDDEN_PUBLIC_KEYS.intersection(public_keys))
    leakage_failures = _public_leakage_failures(
        {"mechanism_requests": mechanism_public_rows},
        mechanism_selected,
    )
    if leaked_keys or leakage_failures:
        raise ConstructionError(
            "mechanism public-pack leakage detected: "
            f"keys={leaked_keys}, dynamic={leakage_failures}"
        )

    public_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hidden_by_request = {
        str(row["request_id"]): row for row in mechanism_hidden_rows
    }
    for row in mechanism_public_rows:
        public_by_frame[str(row["frame_unit_id"])].append(row)
    if len(public_by_frame) != mechanism_frame_target:
        raise ConstructionError("mechanism public frame count drift")
    if len(mechanism_public_rows) != mechanism_trial_target:
        raise ConstructionError("mechanism public trial count drift")
    if len(hidden_by_request) != mechanism_trial_target:
        raise ConstructionError("mechanism hidden request IDs are not unique")
    if {str(row["request_id"]) for row in mechanism_public_rows} != set(
        hidden_by_request
    ):
        raise ConstructionError("mechanism public/hidden request IDs differ")

    full_identity_checks = 0
    crop_permutation_checks = 0
    cross_arm_crop_identity_checks = 0
    full_identity_rows: list[dict[str, Any]] = []
    crop_identity_rows: list[dict[str, Any]] = []
    for public_frame_id in public_frame_ids:
        pair = sorted(
            public_by_frame[public_frame_id],
            key=lambda row: str(row["order_id"]),
        )
        if len(pair) != 2 or {str(row["order_id"]) for row in pair} != {
            "order_0",
            "order_1",
        }:
            raise ConstructionError(
                f"mechanism frame {public_frame_id} lacks two counterbalanced orders"
            )
        first, second = pair
        if (
            first["counterbalanced_request_id"] != second["request_id"]
            or second["counterbalanced_request_id"] != first["request_id"]
        ):
            raise ConstructionError(
                f"mechanism frame {public_frame_id} has a nonreciprocal order join"
            )
        answers = {
            str(hidden_by_request[str(row["request_id"])]["correct_choice"])
            for row in pair
        }
        if answers != {"A", "B"}:
            raise ConstructionError(
                f"mechanism frame {public_frame_id} is not answer-counterbalanced"
            )
        for arm in ARMS:
            first_inputs = first["arms"][arm]["inputs"]
            second_inputs = second["arms"][arm]["inputs"]
            if len(first_inputs) != 3 or len(second_inputs) != 3:
                raise ConstructionError("mechanism arm does not contain exactly three images")
            first_full = _asset_identity(first_inputs[0])
            second_full = _asset_identity(second_inputs[0])
            if first_full != second_full:
                raise ConstructionError(
                    f"mechanism full-frame identity failed for {public_frame_id}:{arm}"
                )
            full_identity_checks += 1
            first_crops = [_asset_identity(asset) for asset in first_inputs[1:]]
            second_crops = [_asset_identity(asset) for asset in second_inputs[1:]]
            if first_crops != list(reversed(second_crops)):
                raise ConstructionError(
                    f"mechanism crop permutation failed for {public_frame_id}:{arm}"
                )
            crop_permutation_checks += 1
            full_identity_rows.append(
                {
                    "frame_unit_id": public_frame_id,
                    "arm": arm,
                    "full": first_full,
                }
            )
            crop_identity_rows.append(
                {
                    "frame_unit_id": public_frame_id,
                    "arm": arm,
                    "order_0": first_crops,
                    "order_1": second_crops,
                }
            )
        for row in pair:
            real_crops = [
                _asset_identity(asset) for asset in row["arms"]["real"]["inputs"][1:]
            ]
            for arm in ("blank_full", "wrong_time"):
                arm_crops = [
                    _asset_identity(asset) for asset in row["arms"][arm]["inputs"][1:]
                ]
                if arm_crops != real_crops:
                    raise ConstructionError(
                        "mechanism reference crops changed across full-frame arms: "
                        f"{public_frame_id}:{row['order_id']}:{arm}"
                    )
                cross_arm_crop_identity_checks += 1

    participant_ids = {
        str(row["ego4d_participant_id"]) for row in mechanism_hidden_rows
    }
    hidden_frame_ids = {str(row["frame_id"]) for row in mechanism_hidden_rows}
    if len(participant_ids) != mechanism_frame_target:
        raise ConstructionError("mechanism hidden participant count drift")
    if len(hidden_frame_ids) != mechanism_frame_target:
        raise ConstructionError("mechanism hidden frame count drift")

    public_relpath = Path("mechanism/public/mechanism_requests.jsonl")
    hidden_relpath = Path("mechanism/hidden/mechanism_evaluator_index.jsonl")
    manifest_relpath = Path("mechanism/manifest.json")
    public_path = out_dir / public_relpath
    hidden_path = out_dir / hidden_relpath
    _write_jsonl(public_path, mechanism_public_rows)
    _write_jsonl(hidden_path, mechanism_hidden_rows)
    pair_audit = {
        "status": "pass",
        "full_frame_identity_check_count": full_identity_checks,
        "crop_ab_exact_permutation_check_count": crop_permutation_checks,
        "cross_arm_crop_identity_check_count": cross_arm_crop_identity_checks,
        "full_frame_identity_sha256": _sha256_text(
            json.dumps(full_identity_rows, separators=(",", ":"), sort_keys=True)
        ),
        "crop_permutation_identity_sha256": _sha256_text(
            json.dumps(crop_identity_rows, separators=(",", ":"), sort_keys=True)
        ),
    }
    mechanism_manifest = {
        "kind": "phase2_reanchored_egohos_mechanism_subset_manifest",
        "schema_version": 1,
        "status": (
            "complete_production" if production_eligible else "complete_nonproduction"
        ),
        "production_eligible": production_eligible,
        "governing_prespec_sha256": governing_prespec_sha256,
        "selection_namespace": MECHANISM_SELECTION_NAMESPACE,
        "selection_rule": MECHANISM_SUBSET_RULE,
        "selection_uses_model_outputs": False,
        "selection_uses_hidden_answers_or_roles": False,
        "source_battery_frame_count": source_frame_count,
        "frame_count": mechanism_frame_target,
        "trial_count": mechanism_trial_target,
        "participant_cluster_count": mechanism_frame_target,
        "counterbalanced_trials_per_frame": TRIALS_PER_FRAME,
        "arms": list(ARMS),
        "images_per_arm": 3,
        "public_frame_order_sha256": _sha256_text("\n".join(public_frame_ids)),
        "pair_audit": pair_audit,
        "fixed_geometry": {
            "full_frame": {
                "width": FULL_CANVAS_WIDTH,
                "height": FULL_CANVAS_HEIGHT,
            },
            "reference_crop": {
                "width": CROP_CANVAS_WIDTH,
                "height": CROP_CANVAS_HEIGHT,
            },
        },
        "public_schema": {
            "request_id": "opaque unique trial ID",
            "frame_unit_id": "opaque public-neutral frame-pair ID",
            "order_id": ["order_0", "order_1"],
            "counterbalanced_request_id": "reciprocal request ID",
            "arms": {arm: "three ordered image references" for arm in ARMS},
            "evaluator_fields_present": False,
        },
        "hidden_schema": {
            "join_key": "request_id",
            "correct_choice": ["A", "B"],
            "participant_field": "ego4d_participant_id",
            "frame_field": "frame_id",
            "component_fingerprint_field": "target_component_fingerprints",
        },
        "consumer_contract": {
            "job_b": {
                "public_requests": str(public_relpath),
                "subset_selection_already_frozen": True,
                "must_not_take_first_rows_from_full_battery": True,
            },
            "job_c": {
                "public_manifest_argument": str(public_relpath),
                "hidden_index_argument": str(hidden_relpath),
                "data_dir_argument": ".",
            },
        },
        "files": {
            "public_requests": {
                "path": str(public_relpath),
                "sha256": _sha256_file(public_path),
                "rows": len(mechanism_public_rows),
            },
            "hidden_evaluator_index": {
                "path": str(hidden_relpath),
                "sha256": _sha256_file(hidden_path),
                "rows": len(mechanism_hidden_rows),
            },
            "source_selection_audit_sha256": selection_audit_sha256,
        },
    }
    manifest_path = out_dir / manifest_relpath
    _write_json(manifest_path, mechanism_manifest)
    return {
        "status": mechanism_manifest["status"],
        "production_eligible": production_eligible,
        "schema_version": 1,
        "selection_rule": MECHANISM_SUBSET_RULE,
        "frame_count": mechanism_frame_target,
        "trial_count": mechanism_trial_target,
        "participant_cluster_count": mechanism_frame_target,
        "manifest": str(manifest_relpath),
        "manifest_sha256": _sha256_file(manifest_path),
        "public_requests": str(public_relpath),
        "public_requests_sha256": _sha256_file(public_path),
        "hidden_evaluator_index": str(hidden_relpath),
        "hidden_evaluator_index_sha256": _sha256_file(hidden_path),
        "pair_audit": pair_audit,
        "consumer_contract": mechanism_manifest["consumer_contract"],
    }


def _input_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _validate_prespec(
    *,
    prespec_path: Path,
    candidate_sidecar_path: Path,
    burned_evaluator_path: Path,
    prompt_prespec_path: Path,
    enforce_frozen_bindings: bool,
    expected_prespec_sha256: str = PRESPEC_SHA256,
    expected_frame_count: int = FRAME_TARGET,
    expected_minimum_participant_clusters: int = MIN_PARTICIPANT_CLUSTERS,
    expected_maximum_frames_per_participant: int = MAX_FRAMES_PER_PARTICIPANT,
    expected_mechanism_frame_count: int = MECHANISM_FRAME_TARGET,
    additional_exclusion_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prespec_hash = _sha256_file(prespec_path)
    if prespec_hash != expected_prespec_sha256:
        raise ConstructionError(
            "governing prespec SHA-256 mismatch: "
            f"{prespec_hash} != {expected_prespec_sha256}"
        )
    try:
        prespec = json.loads(prespec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConstructionError("governing prespec is not valid JSON") from exc
    data = prespec.get("data", {})
    selection_contract = data.get("selection", {})
    generation = prespec.get("generation", {})
    job_b = prespec.get("job_b", {})
    for key, expected in {
        "frame_count": expected_frame_count,
        "minimum_participant_clusters": expected_minimum_participant_clusters,
        "arms": list(ARMS),
    }.items():
        if data.get(key) != expected:
            raise ConstructionError(f"governing prespec {key} drifted from builder freeze")
    for key, expected in {
        "exact_targets_per_frame": 2,
        "first_order_classes": list(FIRST_ORDER_CLASSES),
        "max_frames_per_participant": expected_maximum_frames_per_participant,
        "one_target_per_role": True,
        "outcome_blind_hash_order": True,
        "second_order_classes": list(SECOND_ORDER_CLASSES),
        "target_component_rule": TARGET_COMPONENT_RULE_TEXT,
        "wrong_time_minimum_source_frame_gap": MIN_DONOR_FRAME_GAP,
        "wrong_time_same_video": True,
    }.items():
        if selection_contract.get(key) != expected:
            raise ConstructionError(
                f"governing prespec selection.{key} drifted from builder freeze"
            )
    if generation.get("prompt_source") != PROMPT_SOURCE_ID:
        raise ConstructionError("governing prespec prompt source is not the frozen 2AFC prompt")
    if generation.get("prompt_text") != EXPECTED_USER_PROMPT:
        raise ConstructionError("governing prespec prompt text drifted")
    for key, expected in {
        "arms": list(ARMS),
        "diagnostic_frame_count": expected_mechanism_frame_count,
        "diagnostic_trial_count": expected_mechanism_frame_count * TRIALS_PER_FRAME,
        "subset_selection": MECHANISM_SUBSET_RULE,
    }.items():
        if job_b.get(key) != expected:
            raise ConstructionError(f"governing prespec job_b.{key} drifted")
    if _sha256_file(prompt_prespec_path) != PROMPT_PRESPEC_SHA256:
        raise ConstructionError("prompt-source prespec SHA-256 mismatch")
    bindings = {
        "governing_prespec": _input_record(prespec_path),
        "candidate_sidecar": _input_record(candidate_sidecar_path),
        "burned_evaluator_index": _input_record(burned_evaluator_path),
        "prompt_source_prespec": _input_record(prompt_prespec_path),
        "frozen_binding_enforcement": enforce_frozen_bindings,
    }
    additional_contract = selection_contract.get("additional_exclusion")
    if additional_exclusion_path is not None:
        if not isinstance(additional_contract, dict):
            raise ConstructionError("governing prespec lacks additional exclusion binding")
        additional_binding = _input_record(additional_exclusion_path)
        if additional_binding["sha256"] != additional_contract.get("sha256"):
            raise ConstructionError("additional exclusion SHA-256 mismatch")
        if additional_contract.get("scope") != "source_and_donor_frames_only":
            raise ConstructionError("additional exclusion scope drift")
        bindings["additional_exclusion_evaluator_index"] = additional_binding
    elif additional_contract is not None:
        raise ConstructionError("prespec requires an additional exclusion index")
    if enforce_frozen_bindings:
        if bindings["candidate_sidecar"]["sha256"] != DEFAULT_CANDIDATE_SIDECAR_SHA256:
            raise ConstructionError("candidate sidecar SHA-256 does not match the prespec")
        if bindings["burned_evaluator_index"]["sha256"] != DEFAULT_BURNED_EVALUATOR_SHA256:
            raise ConstructionError("burned evaluator index SHA-256 does not match the prespec")
    return prespec, bindings


def _write_hash_manifest(out_dir: Path) -> dict[str, Any]:
    forbidden_metadata = sorted(
        str(path.relative_to(out_dir))
        for path in out_dir.rglob("*")
        if path.is_file() and (path.name == ".DS_Store" or path.name.startswith("._"))
    )
    if forbidden_metadata:
        raise ConstructionError(
            "output contains forbidden untracked filesystem metadata: "
            f"count={len(forbidden_metadata)} sample={forbidden_metadata[:8]}"
        )
    entries = []
    for path in sorted(out_dir.rglob("*")):
        if (
            not path.is_file()
            or path.name == "hash_manifest.json"
        ):
            continue
        entries.append(
            {
                "path": str(path.relative_to(out_dir)),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    payload = {
        "kind": "phase2_reanchored_egohos_job_a_2afc_hash_manifest",
        "algorithm": "sha256",
        "entry_count": len(entries),
        "entries": entries,
    }
    _write_json(out_dir / "hash_manifest.json", payload)
    return payload


def build_reanchored_egohos_battery(
    *,
    repo_root: Path,
    data_root: Path,
    inventory_path: Path,
    candidate_sidecar_path: Path,
    burned_evaluator_path: Path | None = None,
    out_dir: Path,
    burned_evaluator_index_path: Path | None = None,
    prespec_path: Path | None = None,
    prompt_prespec_path: Path | None = None,
    selection_only: bool = False,
    limit_frames: int | None = None,
    enforce_frozen_bindings: bool = True,
    additional_exclusion_path: Path | None = None,
    expected_prespec_sha256: str = PRESPEC_SHA256,
    frame_target: int = FRAME_TARGET,
    minimum_participant_clusters: int = MIN_PARTICIPANT_CLUSTERS,
    maximum_frames_per_participant: int = MAX_FRAMES_PER_PARTICIPANT,
    mechanism_frame_target: int = MECHANISM_FRAME_TARGET,
) -> dict[str, Any]:
    """Build the complete battery or a non-production limited/selection-only package."""
    repo_root = repo_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    inventory_path = inventory_path.expanduser().resolve()
    candidate_sidecar_path = candidate_sidecar_path.expanduser().resolve()
    if burned_evaluator_path is not None and burned_evaluator_index_path is not None:
        raise ValueError(
            "supply only one of burned_evaluator_path and burned_evaluator_index_path"
        )
    burned_evaluator_path = burned_evaluator_path or burned_evaluator_index_path
    if burned_evaluator_path is None:
        raise ValueError("a burned evaluator index path is required")
    burned_evaluator_path = burned_evaluator_path.expanduser().resolve()
    prespec_path = (prespec_path or (repo_root / PRESPEC_REL)).expanduser().resolve()
    prompt_prespec_path = (
        prompt_prespec_path or (repo_root / PROMPT_PRESPEC_REL)
    ).expanduser().resolve()
    additional_exclusion_path = (
        additional_exclusion_path.expanduser().resolve()
        if additional_exclusion_path is not None
        else None
    )
    if frame_target < 1 or frame_target > FRAME_TARGET:
        raise ValueError(f"frame_target must be in [1, {FRAME_TARGET}]")
    if limit_frames is not None and not 1 <= limit_frames <= frame_target:
        raise ValueError(f"limit_frames must be in [1, {frame_target}]")
    target_frames = limit_frames if limit_frames is not None else frame_target
    required_clusters = (
        min(minimum_participant_clusters, target_frames)
        if limit_frames is not None
        else minimum_participant_clusters
    )

    prespec, source_bindings = _validate_prespec(
        prespec_path=prespec_path,
        candidate_sidecar_path=candidate_sidecar_path,
        burned_evaluator_path=burned_evaluator_path,
        prompt_prespec_path=prompt_prespec_path,
        enforce_frozen_bindings=enforce_frozen_bindings,
        expected_prespec_sha256=expected_prespec_sha256,
        expected_frame_count=frame_target,
        expected_minimum_participant_clusters=minimum_participant_clusters,
        expected_maximum_frames_per_participant=maximum_frames_per_participant,
        expected_mechanism_frame_count=mechanism_frame_target,
        additional_exclusion_path=additional_exclusion_path,
    )
    inventory_rows = _load_inventory(inventory_path)
    candidate_rows = _read_jsonl(candidate_sidecar_path)
    burned_rows = _load_burned_rows(burned_evaluator_path)
    additional_rows: list[dict[str, Any]] = []
    if additional_exclusion_path is not None:
        additional_rows = _load_burned_rows(additional_exclusion_path)
        for row in additional_rows:
            burned_rows.append(
                {
                    "split": row.get("split"),
                    "frame_id": row.get("frame_id"),
                }
            )
            donor_relpath = str(row.get("donor_image_relpath") or "")
            donor_split = donor_relpath.split("/", 1)[0] if "/" in donor_relpath else ""
            burned_rows.append(
                {
                    "split": donor_split,
                    "frame_id": row.get("donor_frame_id"),
                }
            )
    grouped, _ = _eligible_candidate_groups(
        candidate_rows,
        _burned_identifiers(burned_rows, candidate_rows),
    )
    inventory_index = _inventory_index(inventory_rows)
    geometry_rows, geometry_audit = _geometry_viability_filter(
        data_root=data_root,
        inventory=inventory_index,
        grouped_candidates=grouped,
    )
    selection = select_frame_pairs(
        inventory_rows=inventory_rows,
        candidate_rows=geometry_rows,
        burned_rows=burned_rows,
        target_frames=target_frames,
        minimum_participant_clusters=required_clusters,
        maximum_frames_per_participant=maximum_frames_per_participant,
        governing_prespec_sha256=expected_prespec_sha256,
    )
    selection_audit = dict(selection.audit)
    selection_audit["geometry_preflight"] = geometry_audit
    selection_audit["limited_nonproduction_mode"] = limit_frames is not None
    if selection_audit["selected_target_components"]["unknown_component_metadata_count"]:
        raise ConstructionError("geometry-preflight component provenance was not preserved")
    selected_participant_count = int(selection_audit["selected_participant_clusters"])
    if selected_participant_count >= mechanism_frame_target:
        mechanism_selection = select_mechanism_subset(
            selection.selected, frame_target=mechanism_frame_target
        )
        selection_audit["mechanism_subset_selection"] = {
            "status": "frozen",
            "selection_namespace": MECHANISM_SELECTION_NAMESPACE,
            "selection_rule": MECHANISM_SUBSET_RULE,
            "frame_count": len(mechanism_selection),
            "participant_cluster_count": len(
                {row.pool.participant_id for row in mechanism_selection}
            ),
            "frame_level_unit_order_sha256": _sha256_text(
                "\n".join(row.frame_level_unit_id for row in mechanism_selection)
            ),
            "selected_frames": [
                {
                    "frame_level_unit_id": row.frame_level_unit_id,
                    "participant_id": row.pool.participant_id,
                    "cluster_id": row.pool.cluster_id,
                    "frame_id": row.pool.frame.frame_id,
                }
                for row in mechanism_selection
            ],
        }
    else:
        selection_audit["mechanism_subset_selection"] = {
            "status": "unavailable_in_limited_nonproduction_selection",
            "selection_namespace": MECHANISM_SELECTION_NAMESPACE,
            "selection_rule": MECHANISM_SUBSET_RULE,
            "required_participant_cluster_count": mechanism_frame_target,
            "available_participant_cluster_count": selected_participant_count,
        }

    out_dir = _prepare_out_dir(out_dir)
    _write_json(out_dir / "audits" / "selection_audit.json", selection_audit)

    public_pack_paths: dict[str, str] = {}
    public_payloads: dict[str, Any] = {}
    render_audit: dict[str, Any] | None = None
    if selection_only:
        hidden_rows = []
        for selected_frame in selection.selected:
            for order_index in range(TRIALS_PER_FRAME):
                first_is_a = (
                    selected_frame.base_first_order_is_a
                    if order_index == 0
                    else not selected_frame.base_first_order_is_a
                )
                target_a = (
                    selected_frame.first_order
                    if first_is_a
                    else selected_frame.second_order
                )
                target_b = (
                    selected_frame.second_order
                    if first_is_a
                    else selected_frame.first_order
                )
                hidden_rows.append(
                    {
                        "trial_id": _opaque_id(
                            "trial",
                            selected_frame.frame_level_unit_id,
                            order_index,
                        ),
                        "frame_level_unit_id": selected_frame.frame_level_unit_id,
                        "counterbalance_order_index": order_index,
                        "answer": "A" if first_is_a else "B",
                        "split": selected_frame.pool.frame.split,
                        "frame_id": selected_frame.pool.frame.frame_id,
                        "source_video_id": selected_frame.pool.frame.source_video_id,
                        "source_frame_unit": selected_frame.pool.frame.frame_unit,
                        "participant_id": selected_frame.pool.participant_id,
                        "cluster_id": selected_frame.pool.cluster_id,
                        "first_order_candidate_id": selected_frame.first_order.candidate_id,
                        "first_order_class": selected_frame.first_order.class_id,
                        "first_order_component": _component_provenance(
                            selected_frame.first_order
                        ),
                        "second_order_candidate_id": selected_frame.second_order.candidate_id,
                        "second_order_class": selected_frame.second_order.class_id,
                        "second_order_component": _component_provenance(
                            selected_frame.second_order
                        ),
                        "crop_a_candidate_id": target_a.candidate_id,
                        "crop_a_selected_component_fingerprint": (
                            target_a.selected_component_fingerprint
                        ),
                        "crop_a_target_role": (
                            "first_order_interacting_object"
                            if first_is_a
                            else "second_order_interacting_object"
                        ),
                        "crop_b_candidate_id": target_b.candidate_id,
                        "crop_b_selected_component_fingerprint": (
                            target_b.selected_component_fingerprint
                        ),
                        "crop_b_target_role": (
                            "second_order_interacting_object"
                            if first_is_a
                            else "first_order_interacting_object"
                        ),
                        "donor_id": _opaque_id("donor", selected_frame.pool.donor.key),
                        "donor_frame_id": selected_frame.pool.donor.frame_id,
                        "donor_video_id": selected_frame.pool.donor.source_video_id,
                        "donor_frame_unit": selected_frame.pool.donor.frame_unit,
                        "donor_gap_frame_units": abs(
                            selected_frame.pool.frame.frame_unit
                            - selected_frame.pool.donor.frame_unit
                        ),
                    }
                )
    else:
        public_by_arm, hidden_rows, render_audit = _render_selected(
            data_root=data_root,
            out_dir=out_dir,
            selected=selection.selected,
            user_prompt=str(prespec["generation"]["prompt_text"]),
        )
        selected_component_audit = selection_audit["selected_target_components"]
        for key in (
            "single_component_selected_target_count",
            "multi_component_selected_target_count",
        ):
            if render_audit[key] != selected_component_audit[key]:
                raise ConstructionError(
                    f"render component audit disagrees with selection audit for {key}"
                )
        rendering_contract = {
            "full_frame": {
                "width": FULL_CANVAS_WIDTH,
                "height": FULL_CANVAS_HEIGHT,
                "mode": "RGB",
                "resize": "contain_lanczos_centered_black_canvas",
            },
            "reference_crop": {
                "width": CROP_CANVAS_WIDTH,
                "height": CROP_CANVAS_HEIGHT,
                "mode": "RGB",
                "resize": "contain_lanczos_centered_black_canvas",
            },
        }
        for arm in ARMS:
            payload = {
                "kind": "phase2_reanchored_egohos_job_a_2afc_request_pack",
                "arm": arm,
                "request_count": len(public_by_arm[arm]),
                "frame_level_unit_count": target_frames,
                "counterbalanced_trials_per_frame": TRIALS_PER_FRAME,
                "prompt_id": PROMPT_SOURCE_ID,
                "system_prompt_sha256": _sha256_text(SYSTEM_PROMPT),
                "user_prompt_sha256": _sha256_text(EXPECTED_USER_PROMPT),
                "rendering_contract": rendering_contract,
                "requests": public_by_arm[arm],
            }
            public_payloads[arm] = payload
        leakage_failures = _public_leakage_failures(public_payloads, selection.selected)
        if leakage_failures:
            raise ConstructionError(f"public-pack leakage detected: {leakage_failures}")
        for arm in ARMS:
            relpath = Path("public") / f"{arm}_request_pack.json"
            public_pack_paths[arm] = str(relpath)
            _write_json(out_dir / relpath, public_payloads[arm])
            _write_jsonl(out_dir / relpath.with_suffix(".jsonl"), public_by_arm[arm])

    answer_counts = Counter(str(row["answer"]) for row in hidden_rows)
    if answer_counts != Counter({"A": target_frames, "B": target_frames}):
        raise ConstructionError(f"counterbalanced answer counts failed: {answer_counts}")
    hidden_index = {
        "kind": "phase2_reanchored_egohos_job_a_2afc_hidden_evaluator_index",
        "frame_level_unit_count": target_frames,
        "trial_count": len(hidden_rows),
        "governing_prespec_sha256": expected_prespec_sha256,
        "target_component_contract": _target_component_contract(),
        "answers": hidden_rows,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(out_dir / "hidden" / "evaluator_index.json", hidden_index)
    _write_jsonl(out_dir / "hidden" / "evaluator_index.jsonl", hidden_rows)

    additional_exclusion_audit: dict[str, Any] | None = None
    if additional_rows:
        def identities(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
            values = {
                "source_frame_ids": set(),
                "donor_frame_ids": set(),
                "candidate_ids": set(),
                "component_fingerprints": set(),
                "nonblank_asset_sha256": set(),
            }
            for row in rows:
                for field, bucket in (
                    ("frame_id", "source_frame_ids"),
                    ("donor_frame_id", "donor_frame_ids"),
                    ("first_order_candidate_id", "candidate_ids"),
                    ("second_order_candidate_id", "candidate_ids"),
                    ("crop_a_candidate_id", "candidate_ids"),
                    ("crop_b_candidate_id", "candidate_ids"),
                    ("crop_a_selected_component_fingerprint", "component_fingerprints"),
                    ("crop_b_selected_component_fingerprint", "component_fingerprints"),
                ):
                    value = str(row.get(field) or "")
                    if value:
                        values[bucket].add(value)
                assets = row.get("assets")
                if isinstance(assets, Mapping):
                    for name, asset in assets.items():
                        if name == "blank_full" or not isinstance(asset, Mapping):
                            continue
                        value = str(asset.get("sha256") or "")
                        if value:
                            values["nonblank_asset_sha256"].add(value)
            return values

        old_identities = identities(additional_rows)
        new_identities = identities(hidden_rows)
        overlaps = {
            key: sorted(old_identities[key].intersection(new_identities[key]))
            for key in old_identities
        }
        if any(overlaps.values()):
            raise ConstructionError(
                "additional exclusion identity overlap survived: "
                + ", ".join(
                    f"{key}={len(values)}" for key, values in overlaps.items() if values
                )
            )
        additional_exclusion_audit = {
            "status": "pass",
            "source_evaluator_sha256": _sha256_file(additional_exclusion_path),
            "scope": "source_and_donor_frames_plus_candidate_component_and_nonblank_asset_identity",
            "old_identity_counts": {
                key: len(values) for key, values in old_identities.items()
            },
            "new_identity_counts": {
                key: len(values) for key, values in new_identities.items()
            },
            "overlap_counts": {key: len(values) for key, values in overlaps.items()},
        }
        _write_json(
            out_dir / "audits" / "additional_exclusion_audit.json",
            additional_exclusion_audit,
        )

    source_bindings["inventory"] = _input_record(inventory_path)
    selection_audit_path = out_dir / "audits" / "selection_audit.json"
    production_eligible = (
        not selection_only and limit_frames is None and enforce_frozen_bindings
    )
    if selection_only:
        mechanism_subset = {
            "status": "omitted_selection_only",
            "production_eligible": False,
            "required_for_production": True,
            "selection_rule": MECHANISM_SUBSET_RULE,
            "required_frame_count": mechanism_frame_target,
            "required_trial_count": mechanism_frame_target * TRIALS_PER_FRAME,
        }
    elif target_frames < mechanism_frame_target:
        mechanism_subset = {
            "status": "omitted_limited_canary_below_96_frames",
            "production_eligible": False,
            "required_for_production": True,
            "selection_rule": MECHANISM_SUBSET_RULE,
            "required_frame_count": mechanism_frame_target,
            "required_trial_count": mechanism_frame_target * TRIALS_PER_FRAME,
        }
    else:
        mechanism_subset = _write_mechanism_subset(
            out_dir=out_dir,
            selected=selection.selected,
            public_by_arm=public_by_arm,
            hidden_rows=hidden_rows,
            source_frame_count=target_frames,
            production_eligible=production_eligible,
            selection_audit_sha256=_sha256_file(selection_audit_path),
            governing_prespec_sha256=expected_prespec_sha256,
            mechanism_frame_target=mechanism_frame_target,
        )
        mechanism_subset["required_for_production"] = True
    if production_eligible and mechanism_subset["status"] != "complete_production":
        raise ConstructionError("production battery lacks the frozen mechanism subset")

    manifest = {
        "kind": "phase2_reanchored_egohos_job_a_2afc_manifest",
        "status": (
            "selection_only_nonproduction"
            if selection_only
            else "limited_canary_nonproduction"
            if limit_frames is not None
            else "production_construction_complete"
        ),
        "production_eligible": production_eligible,
        "governing_prespec_sha256": expected_prespec_sha256,
        "prompt_id": PROMPT_SOURCE_ID,
        "prompt": {
            "system_sha256": _sha256_text(SYSTEM_PROMPT),
            "user_sha256": _sha256_text(EXPECTED_USER_PROMPT),
        },
        "source_bindings": source_bindings,
        "selection_only": selection_only,
        "limited_nonproduction_mode": limit_frames is not None,
        "frame_level_unit_count": target_frames,
        "counterbalanced_trials_per_frame": TRIALS_PER_FRAME,
        "trial_count": len(hidden_rows),
        "arm_count": len(ARMS),
        "request_count_all_arms": 0 if selection_only else len(hidden_rows) * len(ARMS),
        "participant_cluster_count": selection_audit["selected_participant_clusters"],
        "class_allocation": {
            "first_order": selection_audit["first_order_class_counts"],
            "second_order": selection_audit["second_order_class_counts"],
        },
        "answer_counts_hidden": dict(sorted(answer_counts.items())),
        "target_component_contract": _target_component_contract(),
        "selected_target_components": selection_audit["selected_target_components"],
        "mechanism_subset": mechanism_subset,
        "rendering_contract": {
            "full_frame": {
                "width": FULL_CANVAS_WIDTH,
                "height": FULL_CANVAS_HEIGHT,
                "mode": "RGB",
            },
            "reference_crop": {
                "width": CROP_CANVAS_WIDTH,
                "height": CROP_CANVAS_HEIGHT,
                "mode": "RGB",
            },
            "processor_token_shape_identity_required_for_patching": True,
        },
        "analysis_unit_contract": {
            "primary_unit": "frame_level_unit",
            "counterbalanced_orders_are_repeated_trials": True,
            "orders_must_not_be_counted_as_independent_frames": True,
            "primary_evidence_estimand": (
                "intersection_union_test_requiring_positive_participant_equal_"
                "real_minus_blank_full_and_real_minus_wrong_time_components"
            ),
            "primary_evidence_effect_summary": (
                "minimum_of_the_two_observed_component_effects_descriptive_only;_"
                "no_direct_nonsmooth_minimum_confidence_interval"
            ),
            "required_counterbalanced_pair_summaries": [
                "frame_pair_both_orders_correct",
                "order_consistency",
                "pair_correct",
            ],
            "job_c_coarse_decoder_layer_fractions": [
                0.0,
                0.125,
                0.25,
                0.375,
                0.5,
                0.625,
                0.75,
                0.875,
                1.0,
            ],
            "job_c_refinement_boundary": (
                "plus_or_minus_two_layer_refinement_is_descriptive_unless_the_complete_"
                "selection_is_replayed_inside_every_inferential_resample"
            ),
        },
        "ontology_contract": {
            "primary_task": "annotation_native_first_vs_second_order_interacting_object_2afc",
            "single_target_binary_direct_contact_primary": False,
            "second_order_role": "second_order_interacting_object",
            "old_burned_binary_pilot_role": "burned_construct_validity_reference_only_not_a_gate",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "public_request_packs": public_pack_paths,
            "hidden_evaluator_index": "hidden/evaluator_index.json",
            "selection_audit": "audits/selection_audit.json",
            "mechanism_manifest": mechanism_subset.get("manifest"),
            "mechanism_public_requests": mechanism_subset.get("public_requests"),
            "mechanism_hidden_evaluator_index": mechanism_subset.get(
                "hidden_evaluator_index"
            ),
            "hash_manifest": "hash_manifest.json",
        },
        "selection_audit_sha256": _sha256_file(selection_audit_path),
        "render_audit": render_audit,
        "additional_exclusion_audit": additional_exclusion_audit,
    }
    _write_json(out_dir / "manifest.json", manifest)
    hash_manifest = _write_hash_manifest(out_dir)
    return {
        "manifest": manifest,
        "selection_audit": selection_audit,
        "hidden_evaluator_index": hidden_index,
        "public_packs": public_payloads,
        "mechanism_subset": mechanism_subset,
        "hash_manifest": hash_manifest,
    }


build_battery = build_reanchored_egohos_battery


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--candidate-sidecar", type=Path, default=None)
    parser.add_argument("--burned-evaluator-index", type=Path, default=None)
    parser.add_argument("--prespec", type=Path, default=None)
    parser.add_argument("--prompt-source-prespec", type=Path, default=None)
    parser.add_argument("--additional-exclusion-evaluator-index", type=Path, default=None)
    parser.add_argument("--expected-prespec-sha256", default=PRESPEC_SHA256)
    parser.add_argument("--frame-target", type=int, default=FRAME_TARGET)
    parser.add_argument(
        "--minimum-participant-clusters",
        type=int,
        default=MIN_PARTICIPANT_CLUSTERS,
    )
    parser.add_argument(
        "--maximum-frames-per-participant",
        type=int,
        default=MAX_FRAMES_PER_PARTICIPANT,
    )
    parser.add_argument(
        "--mechanism-frame-target",
        type=int,
        default=MECHANISM_FRAME_TARGET,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=None,
        help="Non-production frame cap for tests/canaries; emits two target questions per frame.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    inventory_path = (
        args.inventory.expanduser().resolve()
        if args.inventory is not None
        else data_root / DEFAULT_INVENTORY_REL
    )
    candidate_sidecar_path = (
        args.candidate_sidecar.expanduser().resolve()
        if args.candidate_sidecar is not None
        else data_root / DEFAULT_CANDIDATE_SIDECAR_REL
    )
    burned_evaluator_path = (
        args.burned_evaluator_index.expanduser().resolve()
        if args.burned_evaluator_index is not None
        else repo_root / DEFAULT_BURNED_EVALUATOR_REL
    )
    report = build_reanchored_egohos_battery(
        repo_root=repo_root,
        data_root=data_root,
        inventory_path=inventory_path,
        candidate_sidecar_path=candidate_sidecar_path,
        burned_evaluator_path=burned_evaluator_path,
        out_dir=args.out_dir,
        prespec_path=args.prespec,
        prompt_prespec_path=args.prompt_source_prespec,
        selection_only=args.selection_only,
        limit_frames=args.limit_frames,
        enforce_frozen_bindings=True,
        additional_exclusion_path=args.additional_exclusion_evaluator_index,
        expected_prespec_sha256=args.expected_prespec_sha256,
        frame_target=args.frame_target,
        minimum_participant_clusters=args.minimum_participant_clusters,
        maximum_frames_per_participant=args.maximum_frames_per_participant,
        mechanism_frame_target=args.mechanism_frame_target,
    )
    print(json.dumps(report["manifest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
