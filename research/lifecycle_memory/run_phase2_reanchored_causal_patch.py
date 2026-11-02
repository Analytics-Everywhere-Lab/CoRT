#!/usr/bin/env python3
"""Fail-closed bidirectional causal patching for the re-anchored EgoHOS 2AFC study.

The runtime is deliberately conservative.  It supports only the two frozen
anchor models, proves three image-slot boundaries before patching, keeps at
most one trial's detached activations resident, and writes scalar outcomes and
audits only. Sample-specific unrepresentable interventions are explicit rows;
frozen architecture drift aborts the job before inference can authorize a call.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
from typing import Any, Iterable, Iterator, Mapping, Sequence

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory import phase2_job2c_behavioral_companion as job2c
from research.lifecycle_memory import phase2_job3_heldout_confirmation as job3
from research.lifecycle_memory import run_phase2_mech_job1_instrument_gates as gates
from research.lifecycle_memory.reanchored_token_positions import (
    QUESTION_TOKEN_CONTRACT,
    resolve_final_question_token,
)


PRESPEC_REL = Path(
    "docs/research/lifecycle_memory/phase 2 research/preregistration/"
    "reanchored_evidence_failure_localization_v1_2026_07_21.json"
)
PRESPEC_SHA256 = "bd0f021fd510d873f388075a3d4c0e0d138da07fc6da8215cc0ff16b1d09279f"
MECHANISM_GOVERNING_PRESPEC_SHA256 = PRESPEC_SHA256
LEDGER_REL = Path(
    "docs/research/lifecycle_memory/phase 2 research/"
    "job3_stage_a_rights_and_provenance_ledger_2026_07_14.md"
)
CHOICES = ("A", "B")
ARMS = ("real", "blank_full", "wrong_time")
CORRUPT_ARMS = ("blank_full", "wrong_time")
IMAGE_SLOT_NAMES = ("full_frame", "crop_A", "crop_B")
OUTPUT_SCHEMA_VERSION = 5
RESULT_STATUSES = ("pass", "unsupported", "not_applicable")
QUESTION_TOKEN_TRANSFER_NOT_APPLICABLE_REASON = (
    "question-token residual has no exact separate crop-slot slice"
)
PREFUSION_TRANSFER_NOT_APPLICABLE_REASON = (
    "content-aligned crop transfer is a freshly audited exact activation identity at "
    "the captured pre-decoder image stage: byte-identical donor crop assets map to "
    "their identical recipient crop assets"
)
PREFUSION_TRANSFER_FAMILIES = (
    "vision_tower_final",
    "projector_or_merger_output",
)
DEFAULT_SEED = 20260721
PROJECTION_FORMULA_VERSION = 4
REQUIRED_CANARY_ANSWER_TOKEN_EQUIVALENCE = False
CANARY_ANSWER_TOKEN_EQUIVALENCE_TOLERANCE_BITS = 1e-3
SELF_MARGIN_TOLERANCE_BITS = 1e-5
SELF_ACTIVATION_TOLERANCE = 1e-6
NORM_TOLERANCE_REL = 1e-5
# One BF16 spacing at unit scale. The realized perturbation is measured after
# quantization, so this is a dtype-derived bound rather than a run-tuned value.
BF16_NORM_TOLERANCE_REL = 2.0**-7
MECHANISM_MANIFEST_KIND = "phase2_reanchored_egohos_mechanism_subset_manifest"
MECHANISM_MANIFEST_REL = Path("mechanism/manifest.json")
MECHANISM_PUBLIC_REL = Path("mechanism/public/mechanism_requests.jsonl")
MECHANISM_HIDDEN_REL = Path("mechanism/hidden/mechanism_evaluator_index.jsonl")
EXPECTED_MECHANISM_MANIFEST_SHA256: str | None = None
EXPECTED_MECHANISM_PUBLIC_SHA256: str | None = None
EXPECTED_MECHANISM_HIDDEN_SHA256: str | None = None
MECHANISM_SELECTION_RULE = (
    "outcome_blind_participant_first_hash_order_one_frame_per_participant_before_fill"
)
MECHANISM_SELECTION_NAMESPACE = "phase2_reanchored_egohos_mechanism_subset_v1"
SOURCE_BATTERY_FRAME_COUNT = 800
MECHANISM_FRAME_COUNT = 96
MECHANISM_TRIAL_COUNT = 192
MECHANISM_PARTICIPANT_COUNT = 96
EXPECTED_GPU_IDS = ("0", "1", "2", "3")
MIN_TELEMETRY_SAMPLES_PER_GPU = 4
TARGET_COMPONENT_RULE = (
    "within the chosen class mask, select the largest 8-connected component with area at least "
    "64 pixels; break exact area ties by lexicographically smallest top,left,bottom,right bounding "
    "box; render only that component and record its hidden fingerprint"
)

PATCH_DIRECTIONS = (
    "clean_to_corrupt_rescue",
    "corrupt_to_clean_damage",
    "identity_control",
    "counterbalanced_crop_transfer",
)
OUTCOME_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "phase",
        "inferential_scope",
        "model_key",
        "rank",
        "request_id",
        "frame_unit_id",
        "order_id",
        "public_request_sha256",
        "label_joined",
        "corruption",
        "site_id",
        "stage_family",
        "layer_index",
        "state_name",
        "layer_crosswalk_rule",
        "patch_direction",
        "intervention",
        "source_request_id",
        "source_arm",
        "recipient_arm",
        "status",
        "reason",
        "recipient_baseline_a_minus_b_margin_bits",
        "real_baseline_a_minus_b_margin_bits",
        "patched_a_minus_b_margin_bits",
        "patch_audits",
        "slot_mapping",
        "created_at",
    }
)
RESUME_COMPLETION_FIELDS = frozenset(
    {
        "request_id",
        "result_row_count",
        "result_batch_sha256",
        "output_schema_version",
        "resume_binding_sha256",
        "model_key",
        "rank",
        "phase",
        "completed_at",
    }
)


MODEL_ADAPTERS: dict[str, dict[str, Any]] = {
    "qwen35_27b_hf_bf16": {
        "model_id": "Qwen/Qwen3.5-27B",
        "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654",
        "family": "qwen35",
        "image_token_id": 248056,
        "vision_module": "model.visual.blocks.26",
        "projector_candidates": ["model.visual.merger"],
        "decoder_container": "model.language_model.layers",
        "expected_decoder_layers": 64,
        "expected_vision_layers": 27,
    },
    "gemma4_31b_hf_bf16": {
        "model_id": "google/gemma-4-31B-it",
        "revision": "3548789868c5356dbf307c98e6f609007b82b3eb",
        "family": "gemma4",
        "image_token_id": 258880,
        "vision_module": "model.vision_tower.encoder.layers.26",
        "projector_candidates": [
            "model.embed_vision.embedding_projection",
            "model.embed_vision.multimodal_embedder.embedding_projection",
        ],
        "decoder_container": "model.language_model.layers",
        "expected_decoder_layers": 60,
        "expected_vision_layers": 27,
    },
}


class GateError(RuntimeError):
    """A protocol or exactness gate failed; inference must not continue."""


class UnsupportedSite(RuntimeError):
    """A requested intervention cannot be represented exactly."""


class DegenerateIntervention(UnsupportedSite):
    """A sample-specific intervention has no nonzero representable effect."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def required_sha256_env(name: str) -> str:
    value = str(os.environ.get(name) or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise GateError(f"worker requires a lowercase SHA-256 binding in {name}")
    return value


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise GateError(f"JSONL row is not an object: {path}:{line_no}")
            rows.append(value)
    return rows


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def append_jsonl_fsync(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_governing_prespec(repo_dir: Path) -> dict[str, Any]:
    path = (repo_dir / PRESPEC_REL).resolve()
    if not path.is_file():
        raise GateError(f"governing prespec is missing: {path}")
    observed = sha256_file(path)
    if observed != PRESPEC_SHA256:
        raise GateError(
            "governing prespec hash mismatch: "
            f"expected {PRESPEC_SHA256}, observed {observed}"
        )
    payload = read_json(path)
    candidates = payload.get("candidate_completions")
    if not isinstance(candidates, dict) or set(candidates) != set(CHOICES):
        raise GateError("governing prespec is not the frozen A/B task")
    fractions = payload.get("job_c", {}).get("coarse_decoder_layer_fractions")
    expected = [index / 8 for index in range(9)]
    if fractions != expected:
        raise GateError(f"coarse layer fractions drifted: {fractions!r}")
    return {"path": str(path), "sha256": observed, "payload": payload}


def module_by_path(model: Any, path: str) -> Any:
    current = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            if not hasattr(current, part):
                raise UnsupportedSite(f"module path does not exist: {path}")
            current = getattr(current, part)
    return current


def resolve_exactly_one_module(model: Any, candidates: Sequence[str]) -> tuple[str, Any]:
    found: list[tuple[str, Any]] = []
    for path in candidates:
        try:
            found.append((path, module_by_path(model, path)))
        except (UnsupportedSite, IndexError, KeyError):
            continue
    if len(found) != 1:
        raise UnsupportedSite(
            f"expected exactly one architecture module from {list(candidates)}, found "
            f"{[path for path, _ in found]}"
        )
    return found[0]


def coarse_layer_indexes(layer_count: int) -> list[int]:
    if layer_count < 2:
        raise GateError(f"decoder layer count must be >=2, got {layer_count}")
    indexes = [int(math.floor((layer_count - 1) * (step / 8) + 0.5)) for step in range(9)]
    unique = sorted(set(indexes))
    if len(unique) != 9 or unique[0] != 0 or unique[-1] != layer_count - 1:
        raise GateError(f"coarse fraction mapping is not nine unique endpoints: {indexes}")
    return unique


def site_state_name(site: "PatchSite") -> str:
    """Return architecture-neutral decoder-state names for capture/patch crosswalks."""
    if site.layer_index is not None:
        return f"post_block_{site.layer_index}"
    if site.family == "vision_tower_final":
        return "vision_tower_final"
    if site.family == "projector_or_merger_output":
        return "projector_or_merger_output"
    raise GateError(f"cannot name state for site {site.site_id}")


def layer_crosswalk(model_key: str) -> dict[str, Any]:
    layer_count = int(MODEL_ADAPTERS[model_key]["expected_decoder_layers"])
    return {
        "model_key": model_key,
        "lm_entry": {
            "job_b_capture_state_index": 0,
            "job_c_patch_site": None,
            "meaning": "directly hooked decoder input before block 0",
        },
        "post_block_rule": (
            "Job B direct-hook capture state i+1 is raw post_block_i and corresponds "
            "to Job C decoder block i; final normalization is excluded"
        ),
        "post_blocks": [
            {
                "block_index": index,
                "state_name": f"post_block_{index}",
                "job_b_capture_state_index": index + 1,
                "job_c_block_index": index,
            }
            for index in range(layer_count)
        ],
    }


@dataclass(frozen=True)
class PatchSite:
    site_id: str
    family: str
    module_name: str
    tensor_selector: str
    layer_index: int | None = None
    inferential_scope: str = "coarse_primary"


@dataclass(frozen=True)
class SlotBoundary:
    slot_index: int
    slot_name: str
    decoder_positions: tuple[int, ...]
    decoder_start: int
    decoder_stop: int
    decoder_count: int
    premerge_count: int | None


@dataclass(frozen=True)
class ImageSlotMap:
    image_token_id: int
    sequence_length: int
    slots: tuple[SlotBoundary, SlotBoundary, SlotBoundary]
    grid_triplets: tuple[tuple[int, int, int], ...]
    processor_signature_sha256: str

    def positions(self, indexes: Sequence[int]) -> tuple[int, ...]:
        merged: list[int] = []
        for index in indexes:
            merged.extend(self.slots[index].decoder_positions)
        return tuple(merged)


@dataclass
class HookAudit:
    site_id: str
    mode: str
    status: str = "pending"
    source_shape: list[int] | None = None
    recipient_shape: list[int] | None = None
    patched_shape: list[int] | None = None
    intended_count: int = 0
    changed_intended_count: int = 0
    max_abs_change_intended: float = 0.0
    max_abs_change_outside: float = 0.0
    source_delta_norm: float | None = None
    applied_delta_norm: float | None = None
    relative_norm_error: float | None = None
    norm_tolerance_rel: float | None = None
    hook_count_before: int | None = None
    hook_count_after: int | None = None
    hook_removed: bool = False
    finite: bool = False
    error: str | None = None


@dataclass(frozen=True)
class TensorLocation:
    kind: str
    key: Any = None


def extract_patch_tensor(output: Any) -> tuple[Any, TensorLocation]:
    """Return the one explicitly supported tensor from a module output."""
    import torch

    if torch.is_tensor(output):
        return output, TensorLocation("tensor")
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0], TensorLocation("tuple", 0)
    if isinstance(output, list) and output and torch.is_tensor(output[0]):
        return output[0], TensorLocation("list", 0)
    for key in ("last_hidden_state", "hidden_state"):
        if isinstance(output, Mapping) and torch.is_tensor(output.get(key)):
            return output[key], TensorLocation("mapping", key)
        value = getattr(output, key, None)
        if torch.is_tensor(value):
            return value, TensorLocation("attribute", key)
    raise UnsupportedSite(f"unsupported module output type: {type(output).__name__}")


def replace_patch_tensor(output: Any, location: TensorLocation, tensor: Any) -> Any:
    if location.kind == "tensor":
        return tensor
    if location.kind == "tuple":
        values = list(output)
        values[int(location.key)] = tensor
        if hasattr(output, "_fields") and hasattr(output, "_make"):
            return type(output)._make(values)
        return tuple(values)
    if location.kind == "list":
        values = list(output)
        values[int(location.key)] = tensor
        return values
    if location.kind in {"mapping", "attribute"}:
        try:
            cloned = output.copy()
        except Exception:
            import copy

            cloned = copy.copy(output)
        if isinstance(cloned, Mapping):
            cloned[location.key] = tensor
        else:
            setattr(cloned, location.key, tensor)
        return cloned
    raise UnsupportedSite(f"unknown tensor location: {location.kind}")


def contiguous_runs(indexes: Sequence[int]) -> list[tuple[int, ...]]:
    if not indexes:
        return []
    runs: list[list[int]] = [[int(indexes[0])]]
    for raw in indexes[1:]:
        value = int(raw)
        if value == runs[-1][-1] + 1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return [tuple(run) for run in runs]


def _grid_triplets_from_inputs(inputs: Any) -> tuple[tuple[int, int, int], ...]:
    raw = gates._input_get(inputs, "image_grid_thw")
    if raw is None:
        return ()
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().tolist()
    if not isinstance(raw, list):
        return ()
    if raw and not isinstance(raw[0], list):
        raw = [raw]
    triplets: list[tuple[int, int, int]] = []
    for row in raw:
        if isinstance(row, list) and len(row) == 3:
            triplets.append(tuple(int(value) for value in row))
    return tuple(triplets)


def _merge_size(model: Any, processor: Any) -> int | None:
    value = gates._merge_size(getattr(model, "config", None), processor)
    return int(value) if value else None


def build_image_slot_map(
    *, inputs: Any, model: Any, processor: Any, adapter: Mapping[str, Any]
) -> ImageSlotMap:
    import torch

    input_ids = gates._input_get(inputs, "input_ids")
    if input_ids is None or not torch.is_tensor(input_ids):
        raise GateError("processor input has no tensor input_ids")
    ids = input_ids.detach()
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise GateError(f"only batch-size-one inputs are supported, got {list(ids.shape)}")
    flat = ids[0]
    image_token_id = int(adapter["image_token_id"])
    indexes = torch.nonzero(flat == image_token_id, as_tuple=False).reshape(-1).cpu().tolist()
    runs = contiguous_runs(indexes)
    if len(runs) != 3:
        raise GateError(
            f"expected exactly three contiguous image-token runs for slots 0/1/2, got "
            f"{len(runs)} with counts {[len(run) for run in runs]}"
        )
    grid = _grid_triplets_from_inputs(inputs)
    premerge: list[int | None] = [None, None, None]
    if adapter["family"] == "qwen35":
        if len(grid) != 3:
            raise GateError(f"Qwen requires three image_grid_thw rows, got {len(grid)}")
        merge = _merge_size(model, processor)
        if merge is None:
            raise GateError("Qwen merge size is unavailable")
        expected_post: list[int] = []
        for index, (t, height, width) in enumerate(grid):
            raw_count = t * height * width
            if height % merge or width % merge:
                raise GateError(f"slot {index} grid is not divisible by merge size {merge}")
            post = t * (height // merge) * (width // merge)
            premerge[index] = raw_count
            expected_post.append(post)
        observed = [len(run) for run in runs]
        if observed != expected_post:
            raise GateError(
                f"Qwen slot token counts disagree with image grid: observed={observed}, "
                f"expected={expected_post}"
            )
    else:
        pixels = gates._input_get(inputs, "pixel_values")
        if pixels is None or not torch.is_tensor(pixels):
            raise GateError("Gemma processor input has no pixel_values")
        if pixels.ndim < 2 or int(pixels.shape[0]) != 3:
            raise GateError(
                "Gemma three-slot mapping requires pixel_values first dimension 3; "
                f"got {list(pixels.shape)}"
            )
        if pixels.ndim >= 3:
            premerge = [int(pixels.shape[1])] * 3
    boundaries: list[SlotBoundary] = []
    for index, run in enumerate(runs):
        boundaries.append(
            SlotBoundary(
                slot_index=index,
                slot_name=IMAGE_SLOT_NAMES[index],
                decoder_positions=run,
                decoder_start=run[0],
                decoder_stop=run[-1] + 1,
                decoder_count=len(run),
                premerge_count=premerge[index],
            )
        )
    signature = {
        "input_ids": flat.cpu().tolist(),
        "slot_counts": [len(run) for run in runs],
        "grid": grid,
        "pixel_shape": list(getattr(gates._input_get(inputs, "pixel_values"), "shape", [])),
    }
    return ImageSlotMap(
        image_token_id=image_token_id,
        sequence_length=int(flat.shape[0]),
        slots=tuple(boundaries),  # type: ignore[arg-type]
        grid_triplets=grid,
        processor_signature_sha256=sha256_json(signature),
    )


def assert_slot_maps_aligned(
    source: ImageSlotMap,
    recipient: ImageSlotMap,
    *,
    allow_suffix_extension: bool = False,
) -> None:
    if source.image_token_id != recipient.image_token_id:
        raise GateError("source and recipient image token IDs differ")
    if source.sequence_length != recipient.sequence_length and not (
        allow_suffix_extension and recipient.sequence_length > source.sequence_length
    ):
        raise GateError("source and recipient sequence lengths differ")
    if source.grid_triplets != recipient.grid_triplets:
        raise GateError("source and recipient image grids differ")
    for source_slot, recipient_slot in zip(source.slots, recipient.slots):
        if source_slot.decoder_positions != recipient_slot.decoder_positions:
            raise GateError(
                f"slot {source_slot.slot_index} decoder positions differ between source and recipient"
            )
        if source_slot.premerge_count != recipient_slot.premerge_count:
            raise GateError(
                f"slot {source_slot.slot_index} premerge counts differ between source and recipient"
            )


def _sequence_axis(tensor: Any, sequence_length: int) -> int:
    shape = list(tensor.shape)
    candidates = [axis for axis, size in enumerate(shape[:-1]) if int(size) == sequence_length]
    if len(candidates) != 1:
        raise UnsupportedSite(
            f"cannot uniquely identify decoder sequence axis {sequence_length} in shape {shape}"
        )
    return candidates[0]


def _stage_slot_slices(
    tensor: Any, *, stage: str, slot_map: ImageSlotMap
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Map processor slots to a vision/projector output axis, exactly."""
    shape = list(tensor.shape)
    if not shape:
        raise UnsupportedSite("scalar module output cannot carry image slots")
    if stage == "vision_tower_final":
        counts = [slot.premerge_count for slot in slot_map.slots]
    elif stage == "projector_or_merger_output":
        counts = [slot.decoder_count for slot in slot_map.slots]
    else:
        raise UnsupportedSite(f"stage slot slices are undefined for {stage}")
    if any(count is None for count in counts):
        raise UnsupportedSite(f"{stage} slot counts are unavailable")
    exact_counts = [int(count) for count in counts if count is not None]
    total = sum(exact_counts)
    axes = [axis for axis, size in enumerate(shape[:-1]) if int(size) == total]
    if len(axes) == 1:
        axis = axes[0]
        ranges: list[tuple[int, int]] = []
        offset = 0
        for count in exact_counts:
            ranges.append((offset, offset + count))
            offset += count
        return axis, tuple(ranges)
    slot_axes = [axis for axis, size in enumerate(shape[:-1]) if int(size) == 3]
    token_axes = [
        axis
        for axis, size in enumerate(shape[:-1])
        if all(count == int(size) for count in exact_counts)
    ]
    if len(slot_axes) == 1:
        # Encode each slot as one slice on the explicit slot axis.
        return slot_axes[0], ((0, 1), (1, 2), (2, 3))
    raise UnsupportedSite(
        f"cannot prove {stage} slot slices for tensor shape {shape} and counts {exact_counts}"
    )


def tensor_selection(
    tensor: Any,
    *,
    site: PatchSite,
    slot_map: ImageSlotMap,
    slot_indexes: Sequence[int],
    question_position: int,
) -> tuple[int, Any, tuple[int, ...]]:
    """Return (axis, index tensor/slice tuple, flattened logical positions)."""
    import torch

    if site.tensor_selector == "stage_slots":
        axis, ranges = _stage_slot_slices(tensor, stage=site.family, slot_map=slot_map)
        pieces: list[int] = []
        for slot_index in slot_indexes:
            start, stop = ranges[slot_index]
            pieces.extend(range(start, stop))
        index = torch.tensor(pieces, dtype=torch.long, device=tensor.device)
        return axis, index, tuple(pieces)
    if site.tensor_selector == "decoder_slots":
        axis = _sequence_axis(tensor, slot_map.sequence_length)
        pieces = list(slot_map.positions(slot_indexes))
        index = torch.tensor(pieces, dtype=torch.long, device=tensor.device)
        return axis, index, tuple(pieces)
    if site.tensor_selector == "question_token":
        axis = _sequence_axis(tensor, slot_map.sequence_length)
        if question_position < 0 or question_position >= slot_map.sequence_length:
            raise GateError(f"question token position is out of range: {question_position}")
        index = torch.tensor([question_position], dtype=torch.long, device=tensor.device)
        return axis, index, (question_position,)
    raise UnsupportedSite(f"unknown tensor selector: {site.tensor_selector}")


def _index_select(tensor: Any, axis: int, index: Any) -> Any:
    return tensor.index_select(axis, index)


def _index_copy(tensor: Any, axis: int, index: Any, values: Any) -> Any:
    cloned = tensor.clone()
    cloned.index_copy_(axis, index, values)
    return cloned


def _outside_max_abs(before: Any, after: Any, axis: int, index: Any) -> float:
    import torch

    mask = torch.ones(before.shape[axis], dtype=torch.bool, device=before.device)
    mask[index] = False
    outside = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if outside.numel() == 0:
        return 0.0
    delta = _index_select(after - before, axis, outside)
    return float(delta.abs().max().item()) if delta.numel() else 0.0


def deterministic_norm_matched_replacement(
    recipient: Any,
    source_delta: Any,
    seed: int,
) -> Any:
    """Construct deterministic random replacement and correct its realized norm.

    Candidate replacements are quantized to the recipient dtype before their
    norm is measured. A bounded multiplicative correction loop then selects the
    closest representable perturbation. This avoids validating an FP32 norm and
    only later discovering that BF16 casting/addition moved it outside budget.
    """
    import torch

    recipient_fp32 = recipient.detach().float()
    delta_fp32 = source_delta.detach().float()
    target_norm = delta_fp32.norm()
    if float(target_norm.item()) == 0.0:
        raise DegenerateIntervention(
            "norm-matched random control is undefined for an exactly zero donor delta"
        )
    target_value = float(target_norm.item())
    tolerance = (
        BF16_NORM_TOLERANCE_REL
        if recipient.dtype == torch.bfloat16
        else NORM_TOLERANCE_REL
    )
    best: tuple[float, Any, float] | None = None
    seed_mask = (1 << 63) - 1
    for attempt in range(8):
        generator = torch.Generator(device=delta_fp32.device)
        attempt_seed = int(seed) ^ ((attempt + 1) * 0x5DEECE66D)
        generator.manual_seed(attempt_seed & seed_mask)
        noise = torch.randn(
            delta_fp32.shape,
            generator=generator,
            device=delta_fp32.device,
            dtype=torch.float32,
        )
        noise_norm = noise.norm()
        if float(noise_norm.item()) == 0.0:
            continue
        direction = noise / noise_norm

        def candidate_at(scale: float) -> tuple[float, Any, float]:
            candidate = (recipient_fp32 + direction * scale).to(dtype=recipient.dtype)
            realized = (candidate.detach().float() - recipient_fp32).norm()
            realized_value = float(realized.item())
            if not math.isfinite(realized_value):
                raise GateError("norm-matched random control produced a nonfinite norm")
            relative_error = abs(realized_value - target_value) / max(target_value, 1e-12)
            return relative_error, candidate, realized_value

        lower_scale = 0.0
        upper_scale = target_value
        upper_candidate = candidate_at(upper_scale)
        if best is None or upper_candidate[0] < best[0]:
            best = upper_candidate
        if upper_candidate[0] <= tolerance:
            return upper_candidate[1]
        for _ in range(12):
            if upper_candidate[2] >= target_value:
                break
            lower_scale = upper_scale
            upper_scale *= 2.0
            upper_candidate = candidate_at(upper_scale)
            if upper_candidate[0] < best[0]:
                best = upper_candidate
            if upper_candidate[0] <= tolerance:
                return upper_candidate[1]
        if upper_candidate[2] < target_value:
            continue
        for _ in range(24):
            midpoint = (lower_scale + upper_scale) / 2.0
            candidate = candidate_at(midpoint)
            if candidate[0] < best[0]:
                best = candidate
            if candidate[0] <= tolerance:
                return candidate[1]
            if candidate[2] < target_value:
                lower_scale = midpoint
            else:
                upper_scale = midpoint
    if best is None:
        raise GateError("deterministic random control produced no usable direction")
    raise DegenerateIntervention(
        "no random perturbation was representable within the dtype-aware norm "
        f"tolerance; closest relative error={best[0]} tolerance={tolerance}"
    )


@contextmanager
def patch_module_output(
    *,
    module: Any,
    site: PatchSite,
    source_selected: Any,
    recipient_slot_map: ImageSlotMap,
    recipient_slot_indexes: Sequence[int],
    question_position: int,
    mode: str,
    seed: int,
) -> Iterator[HookAudit]:
    """Install one exact patch hook and always remove it."""
    import torch

    audit = HookAudit(site_id=site.site_id, mode=mode)
    before_count = len(getattr(module, "_forward_hooks", {}))
    audit.hook_count_before = before_count

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        try:
            tensor, location = extract_patch_tensor(output)
            axis, index, _ = tensor_selection(
                tensor,
                site=site,
                slot_map=recipient_slot_map,
                slot_indexes=recipient_slot_indexes,
                question_position=question_position,
            )
            recipient_selected = _index_select(tensor, axis, index)
            source = source_selected.to(device=tensor.device, dtype=tensor.dtype)
            audit.source_shape = list(source.shape)
            audit.recipient_shape = list(recipient_selected.shape)
            if source.shape != recipient_selected.shape:
                raise GateError(
                    f"patch shape mismatch at {site.site_id}: source={list(source.shape)} "
                    f"recipient={list(recipient_selected.shape)}"
                )
            if mode in {"matched", "unrelated", "self", "counterbalanced_order"}:
                replacement = source
            elif mode == "norm_matched_random":
                delta = source.float() - recipient_selected.detach().float()
                replacement = deterministic_norm_matched_replacement(
                    recipient_selected,
                    delta,
                    seed,
                )
            else:
                raise GateError(f"unknown patch mode: {mode}")
            replacement = replacement.to(device=tensor.device, dtype=tensor.dtype)
            before = tensor.detach().clone()
            patched = _index_copy(tensor, axis, index, replacement)
            intended_delta = _index_select(patched - before, axis, index)
            source_delta = source.float() - recipient_selected.detach().float()
            applied_delta = replacement.detach().float() - recipient_selected.detach().float()
            source_norm = float(source_delta.norm().item())
            applied_norm = float(applied_delta.norm().item())
            rel_error = abs(applied_norm - source_norm) / max(source_norm, 1e-12)
            norm_tolerance = (
                BF16_NORM_TOLERANCE_REL
                if tensor.dtype == torch.bfloat16
                else NORM_TOLERANCE_REL
            )
            audit.patched_shape = list(patched.shape)
            audit.intended_count = int(index.numel())
            audit.changed_intended_count = int(
                (intended_delta.detach().float().abs().reshape(-1) > 0).sum().item()
            )
            audit.max_abs_change_intended = (
                float(intended_delta.detach().float().abs().max().item())
                if intended_delta.numel()
                else 0.0
            )
            audit.max_abs_change_outside = _outside_max_abs(before, patched, axis, index)
            audit.source_delta_norm = source_norm
            audit.applied_delta_norm = applied_norm
            audit.relative_norm_error = rel_error
            audit.norm_tolerance_rel = norm_tolerance
            audit.finite = bool(torch.isfinite(patched).all().item())
            if audit.max_abs_change_outside != 0.0:
                raise GateError("patch modified an unselected tensor position")
            if mode != "self" and audit.intended_count > 0 and audit.changed_intended_count == 0:
                raise DegenerateIntervention(
                    f"{mode} intervention was an exact no-op at {site.site_id}"
                )
            if mode == "norm_matched_random" and rel_error > norm_tolerance:
                raise GateError(
                    f"norm-matched random control at {site.site_id} mode={mode} relative error "
                    f"{rel_error} exceeds dtype-aware tolerance {norm_tolerance}"
                )
            audit.status = "pass"
            return replace_patch_tensor(output, location, patched)
        except Exception as exc:
            audit.status = "fail"
            audit.error = f"{type(exc).__name__}: {exc}"
            raise

    handle = module.register_forward_hook(hook)
    try:
        yield audit
    finally:
        handle.remove()
        after_count = len(getattr(module, "_forward_hooks", {}))
        audit.hook_count_after = after_count
        audit.hook_removed = after_count == before_count
        if not audit.hook_removed and audit.status != "fail":
            audit.status = "fail"
            audit.error = "forward hook was not removed"


@contextmanager
def capture_module_outputs(
    *,
    model: Any,
    sites: Sequence[PatchSite],
    slot_map: ImageSlotMap,
    slot_indexes_by_site: Mapping[str, Sequence[int]],
    question_position: int,
) -> Iterator[dict[str, Any]]:
    """Capture detached selected activations for one forward only."""
    captures: dict[str, Any] = {}
    handles: list[Any] = []
    grouped: dict[str, list[PatchSite]] = defaultdict(list)
    for site in sites:
        grouped[site.module_name].append(site)

    for module_name, module_sites in grouped.items():
        module = module_by_path(model, module_name)

        def make_hook(current_sites: Sequence[PatchSite]):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                tensor, _ = extract_patch_tensor(output)
                for current in current_sites:
                    indexes = slot_indexes_by_site[current.site_id]
                    axis, selection, _ = tensor_selection(
                        tensor,
                        site=current,
                        slot_map=slot_map,
                        slot_indexes=indexes,
                        question_position=question_position,
                    )
                    captures[current.site_id] = (
                        _index_select(tensor, axis, selection).detach().clone()
                    )

            return hook

        handles.append(module.register_forward_hook(make_hook(tuple(module_sites))))
    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()


def architecture_sites(
    model: Any,
    *,
    model_key: str,
    phase: str,
    refinement_layers: Mapping[str, Sequence[int]] | None = None,
) -> tuple[list[PatchSite], dict[str, Any]]:
    if model_key not in MODEL_ADAPTERS:
        raise GateError(f"unsupported model key: {model_key}")
    adapter = MODEL_ADAPTERS[model_key]
    vision_module = str(adapter["vision_module"])
    module_by_path(model, vision_module)
    projector_name, _ = resolve_exactly_one_module(model, adapter["projector_candidates"])
    layers = module_by_path(model, str(adapter["decoder_container"]))
    layer_count = len(layers)
    if layer_count != int(adapter["expected_decoder_layers"]):
        raise GateError(
            f"{model_key} decoder layer count drift: expected "
            f"{adapter['expected_decoder_layers']}, got {layer_count}"
        )
    if phase == "coarse":
        selected_by_family = {
            "decoder_visual_token_residual": coarse_layer_indexes(layer_count),
            "decoder_final_question_token_residual": coarse_layer_indexes(layer_count),
        }
    elif phase == "refine":
        if not isinstance(refinement_layers, Mapping):
            raise GateError("refinement layers must be keyed by decoder patch family")
        selected_by_family = {}
        for family in (
            "decoder_visual_token_residual",
            "decoder_final_question_token_residual",
        ):
            selected = sorted(set(int(value) for value in refinement_layers.get(family, ())))
            if any(value < 0 or value >= layer_count for value in selected):
                raise GateError(f"refinement layer outside decoder for {family}: {selected}")
            selected_by_family[family] = selected
    else:
        raise GateError(f"unknown phase: {phase}")
    sites: list[PatchSite] = []
    if phase == "coarse":
        sites.extend(
            [
                PatchSite(
                    "vision_tower_final",
                    "vision_tower_final",
                    vision_module,
                    "stage_slots",
                ),
                PatchSite(
                    "projector_or_merger_output",
                    "projector_or_merger_output",
                    projector_name,
                    "stage_slots",
                ),
            ]
        )
    scope = "coarse_primary" if phase == "coarse" else "refinement_descriptive"
    for index in selected_by_family["decoder_visual_token_residual"]:
        module_name = f"{adapter['decoder_container']}.{index}"
        sites.append(
            PatchSite(
                f"decoder_visual_token_residual.layer_{index:03d}",
                "decoder_visual_token_residual",
                module_name,
                "decoder_slots",
                index,
                scope,
            )
        )
    for index in selected_by_family["decoder_final_question_token_residual"]:
        module_name = f"{adapter['decoder_container']}.{index}"
        sites.append(
            PatchSite(
                f"decoder_final_question_token_residual.layer_{index:03d}",
                "decoder_final_question_token_residual",
                module_name,
                "question_token",
                index,
                scope,
            )
        )
    receipt = {
        "model_key": model_key,
        "adapter": adapter,
        "vision_module": vision_module,
        "projector_module": projector_name,
        "decoder_layer_count": layer_count,
        "selected_decoder_layers_by_family": selected_by_family,
        "layer_crosswalk": layer_crosswalk(model_key),
        "sites": [asdict(site) for site in sites],
    }
    return sites, receipt


def _resolve_asset(data_dir: Path, raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        spec = {"path": raw}
    elif isinstance(raw, dict):
        spec = dict(raw)
    else:
        raise GateError(f"asset spec is neither path nor object: {raw!r}")
    value = str(spec.get("path") or "").strip()
    if not value:
        raise GateError("asset spec has no path")
    path = Path(value)
    if not path.is_absolute():
        path = data_dir / path
    path = path.resolve()
    if not path.is_file():
        raise GateError(f"asset is missing: {path}")
    expected_hash = str(spec.get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise GateError(f"asset has no valid frozen sha256: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise GateError(f"asset hash mismatch: {path}")
    with Image.open(path) as image:
        width, height = image.size
    if spec.get("width") is not None and int(spec["width"]) != width:
        raise GateError(f"asset width mismatch: {path}")
    if spec.get("height") is not None and int(spec["height"]) != height:
        raise GateError(f"asset height mismatch: {path}")
    return {
        "path": path,
        "sha256": observed_hash,
        "width": width,
        "height": height,
    }


def arm_assets(row: Mapping[str, Any], arm: str, data_dir: Path) -> list[dict[str, Any]]:
    arms = row.get("arms")
    if not isinstance(arms, Mapping) or arm not in arms:
        raise GateError(f"public row lacks arm {arm}")
    arm_value = arms[arm]
    if isinstance(arm_value, Mapping):
        raw_inputs = arm_value.get("inputs")
    else:
        raw_inputs = arm_value
    if not isinstance(raw_inputs, list) or len(raw_inputs) != 3:
        raise GateError(f"arm {arm} must contain exactly three ordered inputs")
    return [_resolve_asset(data_dir, raw) for raw in raw_inputs]


def validate_mechanism_manifest(
    *,
    data_dir: Path,
    manifest_path: Path,
    public_path: Path,
    hidden_path: Path | None = None,
    require_hidden_outside_data_dir: bool = False,
) -> dict[str, Any]:
    """Validate the builder-emitted frozen mechanism subset without reselection."""
    data_dir = data_dir.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    public_path = public_path.expanduser().resolve()
    expected_manifest = (data_dir / MECHANISM_MANIFEST_REL).resolve()
    expected_public = (data_dir / MECHANISM_PUBLIC_REL).resolve()
    expected_hidden = (data_dir / MECHANISM_HIDDEN_REL).resolve()
    if manifest_path != expected_manifest:
        raise GateError(f"mechanism manifest must be {expected_manifest}, got {manifest_path}")
    if public_path != expected_public:
        raise GateError(f"mechanism requests must be {expected_public}, got {public_path}")
    if hidden_path is not None:
        resolved_hidden = hidden_path.expanduser().resolve()
        if require_hidden_outside_data_dir:
            try:
                resolved_hidden.relative_to(data_dir)
            except ValueError:
                pass
            else:
                raise GateError(
                    "coordinator evaluator index must be outside the worker-visible data root"
                )
        elif resolved_hidden != expected_hidden:
            raise GateError(
                f"mechanism evaluator index must be {expected_hidden}, got {hidden_path}"
            )
    if not manifest_path.is_file() or not public_path.is_file():
        raise GateError("mechanism manifest or public requests file is missing")

    manifest_sha = sha256_file(manifest_path)
    public_sha = sha256_file(public_path)
    if (
        EXPECTED_MECHANISM_MANIFEST_SHA256 is not None
        and manifest_sha != EXPECTED_MECHANISM_MANIFEST_SHA256
    ):
        raise GateError("mechanism manifest differs from the preregistered SHA-256")
    if (
        EXPECTED_MECHANISM_PUBLIC_SHA256 is not None
        and public_sha != EXPECTED_MECHANISM_PUBLIC_SHA256
    ):
        raise GateError("mechanism public rows differ from the preregistered SHA-256")

    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise GateError("mechanism manifest is not an object")
    exact = {
        "kind": MECHANISM_MANIFEST_KIND,
        "schema_version": 1,
        "status": "complete_production",
        "production_eligible": True,
        "governing_prespec_sha256": MECHANISM_GOVERNING_PRESPEC_SHA256,
        "selection_namespace": MECHANISM_SELECTION_NAMESPACE,
        "selection_rule": MECHANISM_SELECTION_RULE,
        "selection_uses_model_outputs": False,
        "selection_uses_hidden_answers_or_roles": False,
        "source_battery_frame_count": SOURCE_BATTERY_FRAME_COUNT,
        "frame_count": MECHANISM_FRAME_COUNT,
        "trial_count": MECHANISM_TRIAL_COUNT,
        "participant_cluster_count": MECHANISM_PARTICIPANT_COUNT,
        "counterbalanced_trials_per_frame": 2,
        "arms": list(ARMS),
        "images_per_arm": 3,
    }
    drift = {
        key: {"expected": expected, "observed": manifest.get(key)}
        for key, expected in exact.items()
        if manifest.get(key) != expected
    }
    if drift:
        raise GateError(f"mechanism manifest contract drift: {drift}")
    pair_audit = manifest.get("pair_audit")
    if not isinstance(pair_audit, Mapping) or pair_audit.get("status") != "pass":
        raise GateError("mechanism manifest pair audit is not pass")
    expected_pair_counts = {
        "full_frame_identity_check_count": MECHANISM_FRAME_COUNT * len(ARMS),
        "crop_ab_exact_permutation_check_count": MECHANISM_FRAME_COUNT * len(ARMS),
        "cross_arm_crop_identity_check_count": MECHANISM_TRIAL_COUNT * 2,
    }
    for key, expected in expected_pair_counts.items():
        if int(pair_audit.get(key) or 0) != expected:
            raise GateError(f"mechanism manifest pair audit drift for {key}")
    geometry = manifest.get("fixed_geometry")
    if geometry != {
        "full_frame": {"width": 896, "height": 672},
        "reference_crop": {"width": 336, "height": 336},
    }:
        raise GateError(f"mechanism fixed geometry drift: {geometry!r}")
    public_schema = manifest.get("public_schema")
    if not isinstance(public_schema, Mapping) or public_schema.get(
        "evaluator_fields_present"
    ) is not False:
        raise GateError("mechanism public schema does not preserve the label firewall")
    hidden_schema = manifest.get("hidden_schema")
    if not isinstance(hidden_schema, Mapping) or hidden_schema.get("join_key") != "request_id":
        raise GateError("mechanism hidden join schema drifted")
    if hidden_schema.get("correct_choice") != list(CHOICES):
        raise GateError("mechanism hidden choice schema drifted")
    if hidden_schema.get("participant_field") != "ego4d_participant_id":
        raise GateError("mechanism hidden participant schema drifted")
    consumer = manifest.get("consumer_contract", {}).get("job_c", {})
    if consumer != {
        "public_manifest_argument": str(MECHANISM_PUBLIC_REL),
        "hidden_index_argument": str(MECHANISM_HIDDEN_REL),
        "data_dir_argument": ".",
    }:
        raise GateError(f"mechanism Job C consumer contract drift: {consumer!r}")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise GateError("mechanism manifest has no files table")
    public_record = files.get("public_requests")
    if not isinstance(public_record, Mapping):
        raise GateError("mechanism manifest has no public request record")
    if public_record.get("path") != str(MECHANISM_PUBLIC_REL):
        raise GateError("mechanism public path declaration drifted")
    if int(public_record.get("rows") or 0) != MECHANISM_TRIAL_COUNT:
        raise GateError("mechanism public row declaration drifted")
    if public_record.get("sha256") != public_sha:
        raise GateError("mechanism public requests SHA-256 mismatch")
    hidden_sha: str | None = None
    if hidden_path is not None:
        hidden_path = hidden_path.expanduser().resolve()
        if not hidden_path.is_file():
            raise GateError(f"mechanism evaluator index is missing: {hidden_path}")
        hidden_record = files.get("hidden_evaluator_index")
        if not isinstance(hidden_record, Mapping):
            raise GateError("mechanism manifest has no hidden evaluator record")
        if hidden_record.get("path") != str(MECHANISM_HIDDEN_REL):
            raise GateError("mechanism hidden path declaration drifted")
        if int(hidden_record.get("rows") or 0) != MECHANISM_TRIAL_COUNT:
            raise GateError("mechanism hidden row declaration drifted")
        hidden_sha = sha256_file(hidden_path)
        if hidden_record.get("sha256") != hidden_sha:
            raise GateError("mechanism evaluator SHA-256 mismatch")
        if (
            EXPECTED_MECHANISM_HIDDEN_SHA256 is not None
            and hidden_sha != EXPECTED_MECHANISM_HIDDEN_SHA256
        ):
            raise GateError(
                "mechanism evaluator rows differ from the preregistered SHA-256"
            )
    return {
        "status": "pass",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "public_path": str(public_path),
        "public_sha256": public_sha,
        "hidden_path": str(hidden_path) if hidden_path is not None else None,
        "hidden_sha256": hidden_sha,
        "frame_count": MECHANISM_FRAME_COUNT,
        "trial_count": MECHANISM_TRIAL_COUNT,
        "participant_cluster_count": MECHANISM_PARTICIPANT_COUNT,
        "subset_selection_already_frozen": True,
    }


def _public_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "request_id": row.get("request_id"),
        "frame_unit_id": row.get("frame_unit_id"),
        "order_id": row.get("order_id"),
        "counterbalanced_request_id": row.get("counterbalanced_request_id"),
        "system_prompt": row.get("system_prompt"),
        "user_prompt": row.get("user_prompt"),
        "arms": row.get("arms"),
    }
    return allowed


def validate_dataset(
    *,
    public_rows: Sequence[dict[str, Any]],
    hidden_rows: Sequence[dict[str, Any]] | None,
    data_dir: Path,
    expected_frames: int | None,
    expected_user_prompt: str | None = None,
) -> dict[str, Any]:
    if not public_rows:
        raise GateError("public manifest is empty")
    request_ids = [str(row.get("request_id") or "") for row in public_rows]
    if any(not value for value in request_ids) or len(set(request_ids)) != len(request_ids):
        raise GateError("public request IDs are missing or duplicated")
    forbidden_public_keys = {
        "answer",
        "correct_choice",
        "label",
        "class_id",
        "target_role",
        "first_order",
        "second_order",
    }
    for row in public_rows:
        nested_keys: set[str] = set()
        pending: list[Any] = [row]
        while pending:
            value = pending.pop()
            if isinstance(value, Mapping):
                nested_keys.update(str(key) for key in value)
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        leaked = forbidden_public_keys.intersection(nested_keys)
        if leaked:
            raise GateError(f"public row leaks evaluator fields: {sorted(leaked)}")
        if row.get("schema_version") != 1:
            raise GateError("public mechanism row schema_version must be 1")
        if expected_user_prompt is not None and row.get("user_prompt") != expected_user_prompt:
            raise GateError("public mechanism row prompt differs from the governing prompt")
        if str(row.get("order_id") or "") not in {"order_0", "order_1"}:
            raise GateError("order_id must be public-neutral order_0 or order_1")
        if not str(row.get("frame_unit_id") or ""):
            raise GateError("public row lacks frame_unit_id")
        if not str(row.get("counterbalanced_request_id") or ""):
            raise GateError("public row lacks counterbalanced_request_id")
        for arm in ARMS:
            arm_assets(row, arm, data_dir)
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    public_map = {str(row["request_id"]): row for row in public_rows}
    for row in public_rows:
        by_frame[str(row["frame_unit_id"])].append(row)
    if expected_frames is not None and len(by_frame) != expected_frames:
        raise GateError(f"expected {expected_frames} frame units, found {len(by_frame)}")
    for frame_id, pair in by_frame.items():
        if len(pair) != 2 or {str(row["order_id"]) for row in pair} != {"order_0", "order_1"}:
            raise GateError(f"frame {frame_id} is not one counterbalanced pair")
        first, second = sorted(pair, key=lambda row: str(row["order_id"]))
        if str(first["counterbalanced_request_id"]) != str(second["request_id"]):
            raise GateError(f"frame {frame_id} counterbalanced join is not reciprocal")
        if str(second["counterbalanced_request_id"]) != str(first["request_id"]):
            raise GateError(f"frame {frame_id} counterbalanced join is not reciprocal")
        for arm in ARMS:
            assets_first = arm_assets(first, arm, data_dir)
            assets_second = arm_assets(second, arm, data_dir)
            if assets_first[0]["sha256"] != assets_second[0]["sha256"]:
                raise GateError(f"frame {frame_id} full-frame asset differs across crop orders")
            if [assets_first[1]["sha256"], assets_first[2]["sha256"]] != [
                assets_second[2]["sha256"],
                assets_second[1]["sha256"],
            ]:
                raise GateError(f"frame {frame_id} crop order is not an exact A/B permutation")
        real_first = arm_assets(first, "real", data_dir)
        real_second = arm_assets(second, "real", data_dir)
        for row, real_assets in ((first, real_first), (second, real_second)):
            for arm in CORRUPT_ARMS:
                corrupt_assets = arm_assets(row, arm, data_dir)
                if [asset["sha256"] for asset in corrupt_assets[1:]] != [
                    asset["sha256"] for asset in real_assets[1:]
                ]:
                    raise GateError(
                        f"frame {frame_id} crop assets changed across arm {arm}"
                    )
    hidden_map: dict[str, dict[str, Any]] = {}
    if hidden_rows is not None:
        if len(hidden_rows) != len(public_rows):
            raise GateError("hidden/public row counts differ")
        hidden_map = {str(row.get("request_id") or ""): row for row in hidden_rows}
        if len(hidden_map) != len(hidden_rows) or "" in hidden_map:
            raise GateError("hidden request IDs are missing or duplicated")
        if set(hidden_map) != set(public_map):
            raise GateError("hidden/public request ID sets differ")
        for frame_id, pair in by_frame.items():
            choices = {str(hidden_map[str(row["request_id"])].get("correct_choice")) for row in pair}
            if choices != set(CHOICES):
                raise GateError(f"frame {frame_id} hidden answers are not counterbalanced A/B")
            clusters = {
                str(hidden_map[str(row["request_id"])].get("ego4d_participant_id") or "")
                for row in pair
            }
            if len(clusters) != 1 or "" in clusters:
                raise GateError(f"frame {frame_id} has invalid participant cluster join")
            for row in pair:
                hidden = hidden_map[str(row["request_id"])]
                if hidden.get("schema_version") != 1:
                    raise GateError("hidden mechanism row schema_version must be 1")
                for key in (
                    "frame_unit_id",
                    "order_id",
                    "counterbalanced_request_id",
                ):
                    if hidden.get(key) != row.get(key):
                        raise GateError(f"hidden/public {key} mismatch for {row['request_id']}")
            first, second = sorted(pair, key=lambda row: str(row["order_id"]))
            first_fingerprints = hidden_map[str(first["request_id"])].get(
                "target_component_fingerprints"
            )
            second_fingerprints = hidden_map[str(second["request_id"])].get(
                "target_component_fingerprints"
            )
            if not isinstance(first_fingerprints, Mapping) or not isinstance(
                second_fingerprints, Mapping
            ):
                raise GateError(f"frame {frame_id} lacks target component fingerprints")
            if [first_fingerprints.get("A"), first_fingerprints.get("B")] != [
                second_fingerprints.get("B"),
                second_fingerprints.get("A"),
            ]:
                raise GateError(
                    f"frame {frame_id} component fingerprints are not an exact A/B permutation"
                )
        for request_id, hidden in hidden_map.items():
            if hidden.get("target_component_rule") != TARGET_COMPONENT_RULE:
                raise GateError(f"hidden row {request_id} lacks the frozen largest-component rule")
            fingerprints = hidden.get("target_component_fingerprints")
            if not isinstance(fingerprints, Mapping) or set(fingerprints) != {"A", "B"}:
                raise GateError(f"hidden row {request_id} lacks A/B component fingerprints")
            for choice, value in fingerprints.items():
                if not re.fullmatch(r"[0-9a-f]{64}", str(value).lower()):
                    raise GateError(
                        f"hidden row {request_id} has invalid component fingerprint for {choice}"
                    )
        participants = {
            str(row.get("ego4d_participant_id") or "") for row in hidden_rows
        }
        if "" in participants:
            raise GateError("hidden participant IDs are invalid")
        if expected_frames == MECHANISM_FRAME_COUNT and len(participants) != MECHANISM_PARTICIPANT_COUNT:
            raise GateError(
                "mechanism subset must contain "
                f"{MECHANISM_PARTICIPANT_COUNT} distinct participant clusters"
            )
    dimensions = Counter()
    for row in public_rows:
        assets = arm_assets(row, "real", data_dir)
        dimensions[("full", assets[0]["width"], assets[0]["height"])] += 1
        dimensions[("crop", assets[1]["width"], assets[1]["height"])] += 1
        dimensions[("crop", assets[2]["width"], assets[2]["height"])] += 1
    full_sizes = {(width, height) for kind, width, height in dimensions if kind == "full"}
    crop_sizes = {(width, height) for kind, width, height in dimensions if kind == "crop"}
    if len(full_sizes) != 1 or len(crop_sizes) != 1:
        raise GateError(f"fixed canvas geometry failed: full={full_sizes}, crop={crop_sizes}")
    if expected_frames == MECHANISM_FRAME_COUNT:
        if full_sizes != {(896, 672)} or crop_sizes != {(336, 336)}:
            raise GateError(
                f"mechanism canvas geometry drift: full={full_sizes}, crop={crop_sizes}"
            )
    return {
        "status": "pass",
        "trial_count": len(public_rows),
        "frame_unit_count": len(by_frame),
        "public_sha256": sha256_json([_public_identity(row) for row in public_rows]),
        "hidden_sha256": sha256_json(hidden_rows) if hidden_rows is not None else None,
        "full_canvas_sizes": sorted(full_sizes),
        "crop_canvas_sizes": sorted(crop_sizes),
    }


def _input_ids(inputs: Any) -> list[int]:
    value = gates._input_get(inputs, "input_ids")
    if value is None or not hasattr(value, "detach"):
        raise GateError("prepared inputs have no input_ids")
    rows = value.detach().cpu().tolist()
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise GateError("prepared input_ids must have batch size one")
    return [int(item) for item in rows[0]]


def _tensor_fingerprint(value: Any, *, include_values: bool) -> dict[str, Any]:
    import torch

    if not torch.is_tensor(value):
        return {"type": type(value).__name__, "repr": repr(value)[:200]}
    detached = value.detach().cpu().contiguous()
    payload: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
    }
    if include_values:
        payload["sha256"] = hashlib.sha256(detached.numpy().tobytes()).hexdigest()
    return payload


def processor_input_signature(inputs: Any) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        try:
            items = dict(inputs)
        except Exception as exc:
            raise GateError(f"prepared inputs are not mapping-like: {type(inputs).__name__}") from exc
    else:
        items = dict(inputs)
    value_keys = {
        "input_ids",
        "attention_mask",
        "image_grid_thw",
        "image_position_ids",
        "mm_token_type_ids",
        "token_type_ids",
        "position_ids",
    }
    return {
        key: _tensor_fingerprint(value, include_values=key in value_keys)
        for key, value in sorted(items.items())
    }


def assert_processor_alignment(source_inputs: Any, recipient_inputs: Any) -> dict[str, Any]:
    source = processor_input_signature(source_inputs)
    recipient = processor_input_signature(recipient_inputs)
    if set(source) != set(recipient):
        raise GateError(
            f"processor input key mismatch: source={sorted(source)}, recipient={sorted(recipient)}"
        )
    pixel_keys = {key for key in source if "pixel" in key or "image" in key and "ids" not in key}
    for key in source:
        source_row = source[key]
        recipient_row = recipient[key]
        if source_row.get("shape") != recipient_row.get("shape"):
            raise GateError(f"processor tensor shape mismatch at {key}")
        if key not in pixel_keys and source_row != recipient_row:
            raise GateError(f"processor nonvisual tensor identity mismatch at {key}")
    if _input_ids(source_inputs) != _input_ids(recipient_inputs):
        raise GateError("source and recipient token sequences differ")
    return {
        "status": "pass",
        "source_signature_sha256": sha256_json(source),
        "recipient_signature_sha256": sha256_json(recipient),
        "visual_value_keys_allowed_to_differ": sorted(pixel_keys),
    }


@dataclass
class PreparedChoice:
    choice: str
    inputs: Any
    base_length: int
    full_ids: list[int]
    candidate_ids: list[int]
    slot_map: ImageSlotMap
    construction: str
    policy_trace_sha256: str


@dataclass
class PreparedArm:
    arm: str
    base_inputs: Any
    base_messages: list[dict[str, Any]]
    base_ids: list[int]
    question_position: int
    question_token_audit: dict[str, Any]
    slot_map: ImageSlotMap
    choices: dict[str, PreparedChoice]
    serialization: dict[str, Any]
    asset_receipts: list[dict[str, Any]]


def assert_prepared_alignment(source: PreparedArm, recipient: PreparedArm) -> None:
    assert_processor_alignment(source.base_inputs, recipient.base_inputs)
    assert_slot_maps_aligned(source.slot_map, recipient.slot_map)
    for choice in CHOICES:
        assert_processor_alignment(
            source.choices[choice].inputs,
            recipient.choices[choice].inputs,
        )
        assert_slot_maps_aligned(
            source.choices[choice].slot_map,
            recipient.choices[choice].slot_map,
            allow_suffix_extension=True,
        )


def prepare_arm(
    *,
    runtime: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    model: Any,
    model_key: str,
    row: Mapping[str, Any],
    arm: str,
    data_dir: Path,
    candidate_completions: Mapping[str, str],
) -> PreparedArm:
    assets = arm_assets(row, arm, data_dir)
    image_paths = [Path(asset["path"]) for asset in assets]
    system_prompt = str(row.get("system_prompt") or "")
    user_prompt = str(row.get("user_prompt") or "")
    if not user_prompt:
        raise GateError("public row has no user_prompt")
    base_inputs, serialization = job3._prepare_universal_base(
        runtime=runtime,
        processor=processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=image_paths,
        device=device,
    )
    base_messages = job3._universal_user_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=image_paths,
    )
    base_ids = _input_ids(base_inputs)
    if not base_ids:
        raise GateError("base prompt has no tokens")
    question_token_audit = resolve_final_question_token(
        base_ids, tokenizer, user_prompt
    )
    if question_token_audit.get("status") != "pass":
        raise GateError(
            "literal user-question token boundary could not be resolved: "
            f"{question_token_audit.get('reason')}"
        )
    adapter = MODEL_ADAPTERS[model_key]
    slot_map = build_image_slot_map(
        inputs=base_inputs,
        model=model,
        processor=processor,
        adapter=adapter,
    )
    choices: dict[str, PreparedChoice] = {}
    for choice in CHOICES:
        candidate_text = str(candidate_completions[choice])
        (
            full_inputs,
            full_ids,
            candidate_ids,
            visual_audit,
            construction,
            policy_trace,
        ) = job3._prepare_candidate_with_policy(
            runtime=runtime,
            processor=processor,
            device=device,
            model_key=model_key,
            base_messages=base_messages,
            base_inputs=base_inputs,
            tokenizer=tokenizer,
            candidate_text=candidate_text,
        )
        if visual_audit.get("status") != "pass":
            raise GateError(f"candidate {choice} visual identity audit failed")
        if full_ids[: len(base_ids)] != base_ids or full_ids[len(base_ids) :] != candidate_ids:
            raise GateError(f"candidate {choice} exact suffix audit failed")
        candidate_slot_map = build_image_slot_map(
            inputs=full_inputs,
            model=model,
            processor=processor,
            adapter=adapter,
        )
        assert_slot_maps_aligned(slot_map, candidate_slot_map, allow_suffix_extension=True)
        choices[choice] = PreparedChoice(
            choice=choice,
            inputs=full_inputs,
            base_length=len(base_ids),
            full_ids=[int(value) for value in full_ids],
            candidate_ids=[int(value) for value in candidate_ids],
            slot_map=candidate_slot_map,
            construction=str(construction),
            policy_trace_sha256=sha256_json(policy_trace),
        )
    return PreparedArm(
        arm=arm,
        base_inputs=base_inputs,
        base_messages=base_messages,
        base_ids=base_ids,
        question_position=int(question_token_audit["index"]),
        question_token_audit=question_token_audit,
        slot_map=slot_map,
        choices=choices,
        serialization=serialization,
        asset_receipts=[
            {
                "slot_index": index,
                "slot_name": IMAGE_SLOT_NAMES[index],
                "path": str(asset["path"]),
                "sha256": asset["sha256"],
                "width": asset["width"],
                "height": asset["height"],
            }
            for index, asset in enumerate(assets)
        ],
    )


def score_suffix(output: Any, prepared: PreparedChoice) -> float:
    logits = job2c._extract_logits(output)
    scored = job2c.score_candidate_suffix_from_logits(
        base_input_ids=prepared.full_ids[: prepared.base_length],
        full_input_ids=prepared.full_ids,
        logits=logits,
    )
    if scored.get("status") != "pass":
        raise GateError(f"candidate score failed: {scored.get('reason')}")
    value = float(scored["sum_logprob"])
    if not math.isfinite(value):
        raise GateError("candidate score is nonfinite")
    return value


def margin_bits(scores: Mapping[str, float], correct_choice: str) -> float:
    if correct_choice not in CHOICES:
        raise GateError(f"invalid correct choice: {correct_choice!r}")
    other = "B" if correct_choice == "A" else "A"
    value = (float(scores[correct_choice]) - float(scores[other])) / math.log(2.0)
    if not math.isfinite(value):
        raise GateError("correct-choice margin is nonfinite")
    return value


def a_minus_b_margin_bits(scores: Mapping[str, float]) -> float:
    value = (float(scores["A"]) - float(scores["B"])) / math.log(2.0)
    if not math.isfinite(value):
        raise GateError("evaluator-unjoined A-minus-B margin is nonfinite")
    return value


def forward_capture_scores(
    *,
    model: Any,
    prepared: PreparedArm,
    sites: Sequence[PatchSite],
    slot_indexes_by_site: Mapping[str, Sequence[int]],
) -> tuple[dict[str, float], dict[str, dict[str, Any]], int]:
    import torch

    scores: dict[str, float] = {}
    captures: dict[str, dict[str, Any]] = {}
    forwards = 0
    for choice, candidate in prepared.choices.items():
        with capture_module_outputs(
            model=model,
            sites=sites,
            slot_map=candidate.slot_map,
            slot_indexes_by_site=slot_indexes_by_site,
            question_position=prepared.question_position,
        ) as captured:
            with torch.inference_mode():
                output = model(**candidate.inputs, use_cache=False, return_dict=True)
            forwards += 1
        missing = [site.site_id for site in sites if site.site_id not in captured]
        if missing:
            raise GateError(f"capture hooks did not fire: {missing}")
        scores[choice] = score_suffix(output, candidate)
        captures[choice] = captured
        del output
    return scores, captures, forwards


def forward_plain_scores(model: Any, prepared: PreparedArm) -> tuple[dict[str, float], int]:
    import torch

    scores: dict[str, float] = {}
    for choice, candidate in prepared.choices.items():
        with torch.inference_mode():
            output = model(**candidate.inputs, use_cache=False, return_dict=True)
        scores[choice] = score_suffix(output, candidate)
        del output
    return scores, len(CHOICES)


def forward_patched_scores(
    *,
    model: Any,
    prepared: PreparedArm,
    site: PatchSite,
    source_by_choice: Mapping[str, Any],
    recipient_slot_indexes: Sequence[int],
    mode: str,
    seed: int,
) -> tuple[dict[str, float] | None, list[dict[str, Any]], int, str | None]:
    import torch

    module = module_by_path(model, site.module_name)
    scores: dict[str, float] = {}
    audits: list[dict[str, Any]] = []
    for choice, candidate in prepared.choices.items():
        audit: HookAudit | None = None
        try:
            with patch_module_output(
                module=module,
                site=site,
                source_selected=source_by_choice[choice],
                recipient_slot_map=candidate.slot_map,
                recipient_slot_indexes=recipient_slot_indexes,
                question_position=prepared.question_position,
                mode=mode,
                seed=seed ^ (0 if choice == "A" else 0x5A5A5A5A),
            ) as audit:
                with torch.inference_mode():
                    output = model(**candidate.inputs, use_cache=False, return_dict=True)
        except DegenerateIntervention as exc:
            if audit is not None:
                audits.append({"candidate_choice": choice, **asdict(audit)})
            return None, audits, len(scores), f"degenerate_intervention:{type(exc).__name__}:{exc}"
        except GateError as exc:
            raise GateError(
                f"patch failure site={site.site_id} mode={mode} choice={choice}: {exc}"
            ) from exc
        assert audit is not None
        if audit.status != "pass" or not audit.hook_removed or not audit.finite:
            raise GateError(f"patch audit failed at {site.site_id}: {asdict(audit)}")
        scores[choice] = score_suffix(output, candidate)
        audits.append({"candidate_choice": choice, **asdict(audit)})
        del output
    return scores, audits, len(CHOICES), None


def _deterministic_int(*parts: str, seed: int = DEFAULT_SEED) -> int:
    raw = "\x1f".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def select_unrelated_donor_pairs(
    public_rows: Sequence[dict[str, Any]],
) -> dict[str, tuple[str, str]]:
    """Select one unrelated frame and retain both blinded order candidates.

    The mechanism manifest proves one frame per participant.  Keeping both
    counterbalanced orders lets the analyzer choose the same-label donor only
    after all technical gates pass and the evaluator index is opened.
    """
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in public_rows:
        by_frame[str(row["frame_unit_id"])].append(row)
    ordered_pairs: dict[str, tuple[str, str]] = {}
    for frame_id, pair in by_frame.items():
        ordered = sorted(pair, key=lambda row: str(row["order_id"]))
        if len(ordered) != 2 or [str(row["order_id"]) for row in ordered] != [
            "order_0",
            "order_1",
        ]:
            raise GateError(f"unrelated donor frame {frame_id} is not a complete pair")
        ordered_pairs[frame_id] = (
            str(ordered[0]["request_id"]),
            str(ordered[1]["request_id"]),
        )
    result: dict[str, tuple[str, str]] = {}
    frame_ids = sorted(by_frame)
    if len(frame_ids) < 2:
        raise GateError("unrelated donor control requires at least two frames")
    for row in public_rows:
        request_id = str(row["request_id"])
        own_frame = str(row["frame_unit_id"])
        eligible = [frame_id for frame_id in frame_ids if frame_id != own_frame]
        donor_frame = min(
            eligible,
            key=lambda frame_id: (
                hashlib.sha256(
                    f"{DEFAULT_SEED}|unrelated-frame|{request_id}|{frame_id}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                frame_id,
            ),
        )
        result[request_id] = ordered_pairs[donor_frame]
    return result


def _site_source_slots(site: PatchSite) -> tuple[int, ...]:
    if site.tensor_selector in {"stage_slots", "decoder_slots"}:
        return (0,)
    return ()


def _site_recipient_slots(site: PatchSite) -> tuple[int, ...]:
    return _site_source_slots(site)


def transfer_site_applicability(site: PatchSite) -> tuple[bool, str | None]:
    """Classify the frozen content-aligned crop transfer without approximating it."""
    if (
        site.family in PREFUSION_TRANSFER_FAMILIES
        and site.site_id == site.family
        and site.tensor_selector == "stage_slots"
        and site.layer_index is None
    ):
        return False, PREFUSION_TRANSFER_NOT_APPLICABLE_REASON
    if (
        site.family == "decoder_visual_token_residual"
        and site.site_id.startswith("decoder_visual_token_residual.layer_")
        and site.tensor_selector == "decoder_slots"
        and site.layer_index is not None
    ):
        return True, None
    if (
        site.family == "decoder_final_question_token_residual"
        and site.site_id.startswith("decoder_final_question_token_residual.layer_")
        and site.tensor_selector == "question_token"
        and site.layer_index is not None
    ):
        return False, QUESTION_TOKEN_TRANSFER_NOT_APPLICABLE_REASON
    raise GateError(
        "crop-transfer applicability is undefined for architecture site "
        f"{site.site_id}:{site.family}:{site.tensor_selector}:{site.layer_index}"
    )


def validate_content_aligned_identity_audit(
    audit: Mapping[str, Any],
    *,
    site: PatchSite,
    expected_choices: Sequence[str],
) -> bool:
    """Validate the scalar receipt proving a pre-fusion transfer is identity."""
    expected_choice_list = list(expected_choices)
    if (
        not isinstance(audit, Mapping)
        or not expected_choice_list
        or any(choice not in CHOICES for choice in expected_choice_list)
        or len(set(expected_choice_list)) != len(expected_choice_list)
        or set(audit)
        != {
            "kind",
            "status",
            "site_id",
            "stage_family",
            "comparison",
            "source_slots",
            "recipient_slots",
            "candidate_choices",
            "candidate_audits",
        }
        or audit.get("kind") != "content_aligned_prefusion_activation_identity_audit"
        or audit.get("status") != "pass"
        or audit.get("site_id") != site.site_id
        or audit.get("stage_family") != site.family
        or audit.get("comparison") != "torch.equal"
        or audit.get("source_slots") != [1, 2]
        or audit.get("recipient_slots") != [2, 1]
        or audit.get("candidate_choices") != expected_choice_list
    ):
        return False
    candidate_audits = audit.get("candidate_audits")
    if not isinstance(candidate_audits, list) or len(candidate_audits) != len(
        expected_choice_list
    ):
        return False
    for choice, row in zip(expected_choice_list, candidate_audits):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "candidate_choice",
                "source_shape",
                "recipient_shape",
                "source_dtype",
                "recipient_dtype",
                "source_finite",
                "recipient_finite",
                "exact_equal",
                "max_abs_difference",
            }
            or row.get("candidate_choice") != choice
            or not isinstance(row.get("source_shape"), list)
            or row.get("source_shape") != row.get("recipient_shape")
            or not isinstance(row.get("source_dtype"), str)
            or row.get("source_dtype") != row.get("recipient_dtype")
            or row.get("source_finite") is not True
            or row.get("recipient_finite") is not True
            or row.get("exact_equal") is not True
            or type(row.get("max_abs_difference")) not in {int, float}
            or float(row["max_abs_difference"]) != 0.0
        ):
            return False
    return True


def content_aligned_identity_audit(
    *,
    site: PatchSite,
    source_by_choice: Mapping[str, Any],
    recipient_by_choice: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove exact pre-fusion equality without persisting activation tensors."""
    import torch

    supported, reason = transfer_site_applicability(site)
    if supported or reason != PREFUSION_TRANSFER_NOT_APPLICABLE_REASON:
        raise GateError(f"identity audit is undefined for {site.site_id}")
    source_choices = set(source_by_choice)
    recipient_choices = set(recipient_by_choice)
    if (
        not source_choices
        or source_choices != recipient_choices
        or not source_choices.issubset(set(CHOICES))
    ):
        raise GateError(
            f"identity audit candidate inventory mismatch at {site.site_id}: "
            f"source={sorted(source_choices)} recipient={sorted(recipient_choices)}"
        )
    choices = [choice for choice in CHOICES if choice in source_choices]
    candidate_audits: list[dict[str, Any]] = []
    for choice in choices:
        source = source_by_choice[choice]
        recipient = recipient_by_choice[choice]
        if not hasattr(source, "shape") or not hasattr(recipient, "shape"):
            raise GateError(f"identity audit received a non-tensor at {site.site_id}:{choice}")
        source_shape = list(source.shape)
        recipient_shape = list(recipient.shape)
        source_dtype = str(source.dtype)
        recipient_dtype = str(recipient.dtype)
        if source_shape != recipient_shape or source_dtype != recipient_dtype:
            raise GateError(
                f"content-aligned activation shape/dtype mismatch at {site.site_id}:{choice}: "
                f"{source_shape}/{source_dtype} != {recipient_shape}/{recipient_dtype}"
            )
        source_finite = bool(torch.isfinite(source).all().item())
        recipient_finite = bool(torch.isfinite(recipient).all().item())
        exact_equal = bool(torch.equal(source, recipient))
        max_abs_difference = (
            float((source.detach().float() - recipient.detach().float()).abs().max().item())
            if source.numel()
            else 0.0
        )
        candidate_audits.append(
            {
                "candidate_choice": choice,
                "source_shape": source_shape,
                "recipient_shape": recipient_shape,
                "source_dtype": source_dtype,
                "recipient_dtype": recipient_dtype,
                "source_finite": source_finite,
                "recipient_finite": recipient_finite,
                "exact_equal": exact_equal,
                "max_abs_difference": max_abs_difference,
            }
        )
        if (
            not source_finite
            or not recipient_finite
            or not exact_equal
            or max_abs_difference != 0.0
        ):
            raise GateError(
                f"content-aligned pre-fusion activation is not exact identity at "
                f"{site.site_id}:{choice}"
            )
    audit = {
        "kind": "content_aligned_prefusion_activation_identity_audit",
        "status": "pass",
        "site_id": site.site_id,
        "stage_family": site.family,
        "comparison": "torch.equal",
        "source_slots": [1, 2],
        "recipient_slots": [2, 1],
        "candidate_choices": choices,
        "candidate_audits": candidate_audits,
    }
    if not validate_content_aligned_identity_audit(
        audit, site=site, expected_choices=choices
    ):
        raise GateError(f"identity audit receipt construction failed at {site.site_id}")
    return audit


def structural_transfer_slot_mapping(
    site: PatchSite,
    *,
    identity_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    supported, reason = transfer_site_applicability(site)
    if supported or reason is None:
        raise GateError(f"site is not structurally absent for crop transfer: {site.site_id}")
    if site.family in PREFUSION_TRANSFER_FAMILIES:
        choices = (
            identity_audit.get("candidate_choices")
            if isinstance(identity_audit, Mapping)
            else None
        )
        if not isinstance(choices, list) or not validate_content_aligned_identity_audit(
            identity_audit, site=site, expected_choices=choices
        ):
            raise GateError(
                f"pre-fusion structural absence lacks an exact activation identity audit: "
                f"{site.site_id}"
            )
        return {
            "source_slots": [1, 2],
            "recipient_slots": [2, 1],
            "mapping": {
                "donor_crop_A": "recipient_crop_B",
                "donor_crop_B": "recipient_crop_A",
            },
            "full_frame_slot_unchanged": True,
            "byte_identical_content_alignment": True,
            "cross_image_mixing_assumption_required": False,
            "content_aligned_activation_identity_audit": dict(identity_audit),
        }
    if identity_audit is not None:
        raise GateError(
            f"question-token structural absence cannot carry a crop identity audit: {site.site_id}"
        )
    return {
        "source_slots": [],
        "recipient_slots": [],
        "full_frame_slot_unchanged": True,
        "separate_crop_slot_slice_present": False,
    }


def structural_transfer_record_is_valid(
    record: Mapping[str, Any],
    *,
    expected_choices: Sequence[str],
) -> bool:
    """Validate one evaluator-unjoined canary structural-site receipt exactly."""
    if (
        not isinstance(record, Mapping)
        or set(record) != {"site_id", "stage_family", "reason", "slot_mapping"}
        or not isinstance(record.get("slot_mapping"), Mapping)
    ):
        return False
    site_id = str(record.get("site_id") or "")
    family = str(record.get("stage_family") or "")
    if family in PREFUSION_TRANSFER_FAMILIES:
        if site_id != family or record.get("reason") != (
            PREFUSION_TRANSFER_NOT_APPLICABLE_REASON
        ):
            return False
        site = PatchSite(
            site_id=site_id,
            family=family,
            module_name="structural-receipt-only",
            tensor_selector="stage_slots",
        )
        audit = record["slot_mapping"].get(
            "content_aligned_activation_identity_audit"
        )
        if not isinstance(audit, Mapping) or not validate_content_aligned_identity_audit(
            audit,
            site=site,
            expected_choices=expected_choices,
        ):
            return False
        try:
            expected_mapping = structural_transfer_slot_mapping(
                site, identity_audit=audit
            )
        except GateError:
            return False
        return dict(record["slot_mapping"]) == expected_mapping
    match = re.fullmatch(
        r"decoder_final_question_token_residual\.layer_(\d{3})", site_id
    )
    if (
        family != "decoder_final_question_token_residual"
        or match is None
        or record.get("reason") != QUESTION_TOKEN_TRANSFER_NOT_APPLICABLE_REASON
    ):
        return False
    site = PatchSite(
        site_id=site_id,
        family=family,
        module_name="structural-receipt-only",
        tensor_selector="question_token",
        layer_index=int(match.group(1)),
    )
    try:
        expected_mapping = structural_transfer_slot_mapping(site)
    except GateError:
        return False
    return dict(record["slot_mapping"]) == expected_mapping


def structural_transfer_gate_violations(
    gate: Mapping[str, Any],
    *,
    expected_site_ids: Sequence[str],
    expected_choices: Sequence[str],
) -> list[str]:
    """Require the exact structural site inventory, not only its total count."""
    violations: list[str] = []
    expected_ids = list(expected_site_ids)
    if not isinstance(gate, Mapping):
        return ["gate_not_mapping"]
    if set(gate) != {
        "status",
        "not_applicable_count",
        "site_ids",
        "intervention",
        "sites",
        "reason_counts",
    }:
        violations.append("gate_schema")
    if gate.get("status") != "pass":
        violations.append("gate_status")
    if gate.get("not_applicable_count") != len(expected_ids):
        violations.append("not_applicable_count")
    if gate.get("site_ids") != expected_ids:
        violations.append("site_ids")
    if gate.get("intervention") != "counterbalanced_content_aligned_crop_transfer":
        violations.append("intervention")
    sites = gate.get("sites")
    if not isinstance(sites, list) or len(sites) != len(expected_ids):
        violations.append("sites")
    else:
        observed_ids = [str(row.get("site_id") or "") for row in sites]
        if observed_ids != expected_ids:
            violations.append("site_record_ids")
        if not all(
            structural_transfer_record_is_valid(
                row, expected_choices=expected_choices
            )
            for row in sites
        ):
            violations.append("site_records")
    expected_question_count = sum(
        site_id.startswith("decoder_final_question_token_residual.layer_")
        for site_id in expected_ids
    )
    expected_reason_counts = {
        PREFUSION_TRANSFER_NOT_APPLICABLE_REASON: len(PREFUSION_TRANSFER_FAMILIES),
        QUESTION_TOKEN_TRANSFER_NOT_APPLICABLE_REASON: expected_question_count,
    }
    if gate.get("reason_counts") != dict(sorted(expected_reason_counts.items())):
        violations.append("reason_counts")
    return violations


def is_structural_not_applicable_row(row: Mapping[str, Any]) -> bool:
    """Recognize only the two exact frozen crop-transfer absence mechanisms."""
    audits = row.get("patch_audits")
    slot_mapping = row.get("slot_mapping")
    common = bool(
        row.get("status") == "not_applicable"
        and row.get("intervention")
        == "counterbalanced_content_aligned_crop_transfer"
        and row.get("patch_direction") == "counterbalanced_crop_transfer"
        and row.get("corruption") == "counterbalanced_order"
        and row.get("source_arm") == "real"
        and row.get("recipient_arm") == "real"
        and row.get("patched_a_minus_b_margin_bits") is None
        and audits == []
        and isinstance(slot_mapping, Mapping)
        and slot_mapping.get("full_frame_slot_unchanged") is True
    )
    if not common:
        return False
    family = str(row.get("stage_family") or "")
    site_id = str(row.get("site_id") or "")
    if family in PREFUSION_TRANSFER_FAMILIES:
        identity_audit = slot_mapping.get(
            "content_aligned_activation_identity_audit"
        )
        site = PatchSite(
            site_id=site_id,
            family=family,
            module_name="structural-receipt-only",
            tensor_selector="stage_slots",
        )
        return bool(
            set(slot_mapping)
            == {
                "source_slots",
                "recipient_slots",
                "mapping",
                "full_frame_slot_unchanged",
                "byte_identical_content_alignment",
                "cross_image_mixing_assumption_required",
                "content_aligned_activation_identity_audit",
            }
            and site_id == family
            and row.get("layer_index") is None
            and row.get("reason") == PREFUSION_TRANSFER_NOT_APPLICABLE_REASON
            and slot_mapping.get("source_slots") == [1, 2]
            and slot_mapping.get("recipient_slots") == [2, 1]
            and slot_mapping.get("mapping")
            == {
                "donor_crop_A": "recipient_crop_B",
                "donor_crop_B": "recipient_crop_A",
            }
            and slot_mapping.get("byte_identical_content_alignment") is True
            and slot_mapping.get("cross_image_mixing_assumption_required") is False
            and isinstance(identity_audit, Mapping)
            and validate_content_aligned_identity_audit(
                identity_audit,
                site=site,
                expected_choices=CHOICES,
            )
        )
    return bool(
        set(slot_mapping)
        == {
            "source_slots",
            "recipient_slots",
            "full_frame_slot_unchanged",
            "separate_crop_slot_slice_present",
        }
        and family == "decoder_final_question_token_residual"
        and site_id.startswith("decoder_final_question_token_residual.layer_")
        and isinstance(row.get("layer_index"), int)
        and row.get("reason") == QUESTION_TOKEN_TRANSFER_NOT_APPLICABLE_REASON
        and slot_mapping.get("source_slots") == []
        and slot_mapping.get("recipient_slots") == []
        and slot_mapping.get("separate_crop_slot_slice_present") is False
    )


def transfer_status_contract_violation(row: Mapping[str, Any]) -> str | None:
    """Return a reason when crop-transfer status contradicts its frozen site class."""
    if row.get("intervention") != "counterbalanced_content_aligned_crop_transfer":
        return None
    family = str(row.get("stage_family") or "")
    status = str(row.get("status") or "")
    if family in PREFUSION_TRANSFER_FAMILIES or family == (
        "decoder_final_question_token_residual"
    ):
        if status != "not_applicable":
            return f"structural_site_status_must_be_not_applicable:{family}:{status}"
        if not is_structural_not_applicable_row(row):
            return f"structural_site_receipt_invalid:{family}"
        return None
    if family == "decoder_visual_token_residual":
        if status == "not_applicable":
            return "decoder_visual_transfer_cannot_be_not_applicable"
        if status not in {"pass", "unsupported"}:
            return f"decoder_visual_transfer_status_invalid:{status}"
        return None
    return f"unknown_crop_transfer_family:{family}"


def result_representability_gate(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail on applicable representability loss, not frozen absent endpoints."""
    status_counts = Counter(str(row.get("status") or "missing") for row in rows)
    structurally_not_applicable_rows = [
        row for row in rows if row.get("status") == "not_applicable"
    ]
    invalid_not_applicable_count = sum(
        not is_structural_not_applicable_row(row)
        for row in structurally_not_applicable_rows
    )
    invalid_status_count = sum(
        count
        for status, count in status_counts.items()
        if status not in RESULT_STATUSES
    )
    transfer_status_violations = [
        reason
        for row in rows
        for reason in [transfer_status_contract_violation(row)]
        if reason is not None
    ]
    unsupported_count = int(status_counts.get("unsupported", 0))
    return {
        "status": (
            "pass"
            if (
                unsupported_count == 0
                and invalid_not_applicable_count == 0
                and invalid_status_count == 0
                and not transfer_status_violations
            )
            else "fail"
        ),
        "unsupported_count": unsupported_count,
        "not_applicable_count": len(structurally_not_applicable_rows),
        "invalid_not_applicable_count": invalid_not_applicable_count,
        "invalid_status_count": invalid_status_count,
        "transfer_status_violation_count": len(transfer_status_violations),
        "transfer_status_violations": transfer_status_violations[:20],
        "result_status_counts": dict(sorted(status_counts.items())),
        "rule": (
            "all applicable confirmatory and descriptive intervention rows must be "
            "representable; content-aligned crop transfer is structurally not applicable "
            "where fresh exact activation equality makes it an identity, or at a "
            "question-token scalar with no crop slice; unsupported rows cannot authorize "
            "later gates"
        ),
    }


def _slot_audit_payload(slot_map: ImageSlotMap) -> dict[str, Any]:
    return {
        "image_token_id": slot_map.image_token_id,
        "sequence_length": slot_map.sequence_length,
        "processor_signature_sha256": slot_map.processor_signature_sha256,
        "grid_triplets": [list(row) for row in slot_map.grid_triplets],
        "slots": [asdict(slot) for slot in slot_map.slots],
    }


def _outcome_row(
    *,
    phase: str,
    model_key: str,
    rank: int,
    request: Mapping[str, Any],
    corruption: str,
    site: PatchSite,
    patch_direction: str,
    intervention: str,
    source_request_id: str,
    source_arm: str,
    recipient_arm: str,
    recipient_a_minus_b_margin: float,
    real_a_minus_b_margin: float,
    patched_a_minus_b_margin: float | None,
    audits: Sequence[Mapping[str, Any]],
    status: str,
    reason: str | None = None,
    slot_mapping: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if patch_direction not in PATCH_DIRECTIONS:
        raise GateError(f"unknown patch direction: {patch_direction}")
    if status not in RESULT_STATUSES:
        raise GateError(f"unknown outcome status: {status}")
    row = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "kind": "phase2_reanchored_causal_patch_outcome",
        "phase": phase,
        "inferential_scope": site.inferential_scope,
        "model_key": model_key,
        "rank": rank,
        "request_id": str(request["request_id"]),
        "frame_unit_id": str(request["frame_unit_id"]),
        "order_id": str(request["order_id"]),
        "public_request_sha256": sha256_json(_public_identity(request)),
        "label_joined": False,
        "corruption": corruption,
        "site_id": site.site_id,
        "stage_family": site.family,
        "layer_index": site.layer_index,
        "state_name": site_state_name(site),
        "layer_crosswalk_rule": (
            "Job B direct-hook capture state i+1 corresponds to Job C raw post_block_i"
            if site.layer_index is not None
            else None
        ),
        "patch_direction": patch_direction,
        "intervention": intervention,
        "source_request_id": source_request_id,
        "source_arm": source_arm,
        "recipient_arm": recipient_arm,
        "status": status,
        "reason": reason,
        "recipient_baseline_a_minus_b_margin_bits": recipient_a_minus_b_margin,
        "real_baseline_a_minus_b_margin_bits": real_a_minus_b_margin,
        "patched_a_minus_b_margin_bits": patched_a_minus_b_margin,
        "patch_audits": list(audits),
        "slot_mapping": dict(slot_mapping or {}),
        "created_at": utc_now(),
    }
    if set(row) != OUTCOME_ROW_FIELDS:
        raise GateError(
            "internal outcome schema drift: "
            f"missing={sorted(OUTCOME_ROW_FIELDS - set(row))} "
            f"unexpected={sorted(set(row) - OUTCOME_ROW_FIELDS)}"
        )
    return row


def evaluate_full_trial(
    *,
    runtime: Any,
    model: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    model_key: str,
    rank: int,
    phase: str,
    request: dict[str, Any],
    public_map: Mapping[str, dict[str, Any]],
    unrelated_map: Mapping[str, tuple[str, str]],
    sites: Sequence[PatchSite],
    data_dir: Path,
    candidate_completions: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one trial with matched rescue and reverse damage interventions."""
    request_id = str(request["request_id"])
    timings: dict[str, float] = defaultdict(float)
    forward_count = 0
    started = time.perf_counter()
    prepared: dict[str, PreparedArm] = {}
    prepare_started = time.perf_counter()
    for arm in ARMS:
        prepared[arm] = prepare_arm(
            runtime=runtime,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            model=model,
            model_key=model_key,
            row=request,
            arm=arm,
            data_dir=data_dir,
            candidate_completions=candidate_completions,
        )
    timings["prepare_inputs_s"] += time.perf_counter() - prepare_started
    for corrupt in CORRUPT_ARMS:
        assert_prepared_alignment(prepared["real"], prepared[corrupt])

    unrelated_ids = unrelated_map[request_id]
    unrelated_requests = [public_map[unrelated_id] for unrelated_id in unrelated_ids]
    unrelated_frame_ids = {
        str(unrelated_request["frame_unit_id"]) for unrelated_request in unrelated_requests
    }
    unrelated_orders = {
        str(unrelated_request["order_id"]) for unrelated_request in unrelated_requests
    }
    if len(unrelated_frame_ids) != 1 or str(request["frame_unit_id"]) in unrelated_frame_ids:
        raise GateError("unrelated donor candidates are not one distinct frame")
    if unrelated_orders != {"order_0", "order_1"}:
        raise GateError("unrelated donor candidates are not both counterbalanced orders")
    if {
        str(public_map[unrelated_id]["counterbalanced_request_id"])
        for unrelated_id in unrelated_ids
    } != set(unrelated_ids):
        raise GateError("unrelated donor candidate pair is not reciprocal")

    source_slots = {site.site_id: _site_source_slots(site) for site in sites}
    capture_started = time.perf_counter()
    real_scores, real_captures, count = forward_capture_scores(
        model=model,
        prepared=prepared["real"],
        sites=sites,
        slot_indexes_by_site=source_slots,
    )
    forward_count += count
    real_margin = a_minus_b_margin_bits(real_scores)
    corrupt_margins: dict[str, float] = {}
    corrupt_captures: dict[str, dict[str, dict[str, Any]]] = {}
    for corrupt in CORRUPT_ARMS:
        scores, captures, count = forward_capture_scores(
            model=model,
            prepared=prepared[corrupt],
            sites=sites,
            slot_indexes_by_site=source_slots,
        )
        forward_count += count
        corrupt_margins[corrupt] = a_minus_b_margin_bits(scores)
        corrupt_captures[corrupt] = captures
    timings["source_capture_s"] += time.perf_counter() - capture_started

    rows: list[dict[str, Any]] = []
    patch_started = time.perf_counter()
    for site in sites:
        slots = _site_recipient_slots(site)
        clean_source = {choice: real_captures[choice][site.site_id] for choice in CHOICES}
        for corrupt in CORRUPT_ARMS:
            corrupt_source = {
                choice: corrupt_captures[corrupt][choice][site.site_id] for choice in CHOICES
            }
            for intervention, direction, source, recipient_arm, mode in (
                (
                    "matched_clean_source_rescue",
                    "clean_to_corrupt_rescue",
                    clean_source,
                    corrupt,
                    "matched",
                ),
                (
                    "rescue_norm_matched_random",
                    "clean_to_corrupt_rescue",
                    clean_source,
                    corrupt,
                    "norm_matched_random",
                ),
                (
                    "matched_corrupt_source_damage",
                    "corrupt_to_clean_damage",
                    corrupt_source,
                    "real",
                    "matched",
                ),
                (
                    "damage_norm_matched_random",
                    "corrupt_to_clean_damage",
                    corrupt_source,
                    "real",
                    "norm_matched_random",
                ),
            ):
                patched_scores, audits, count, unsupported_reason = forward_patched_scores(
                    model=model,
                    prepared=prepared[recipient_arm],
                    site=site,
                    source_by_choice=source,
                    recipient_slot_indexes=slots,
                    mode=mode,
                    seed=_deterministic_int(
                        model_key, request_id, corrupt, site.site_id, intervention
                    ),
                )
                forward_count += count
                patched_margin = (
                    a_minus_b_margin_bits(patched_scores)
                    if patched_scores is not None
                    else None
                )
                rows.append(
                    _outcome_row(
                        phase=phase,
                        model_key=model_key,
                        rank=rank,
                        request=request,
                        corruption=corrupt,
                        site=site,
                        patch_direction=direction,
                        intervention=intervention,
                        source_request_id=request_id,
                        source_arm="real" if direction == "clean_to_corrupt_rescue" else corrupt,
                        recipient_arm=recipient_arm,
                        recipient_a_minus_b_margin=(
                            corrupt_margins[corrupt] if recipient_arm == corrupt else real_margin
                        ),
                        real_a_minus_b_margin=real_margin,
                        patched_a_minus_b_margin=patched_margin,
                        audits=audits,
                        status="unsupported" if unsupported_reason else "pass",
                        reason=unsupported_reason,
                        slot_mapping={
                            "source_slots": list(_site_source_slots(site)),
                            "recipient_slots": list(slots),
                            "full_frame_only": site.tensor_selector != "question_token",
                            "crop_slots_unchanged": True,
                        },
                    )
                )

            corrupt_self_scores, audits, count, unsupported_reason = forward_patched_scores(
                model=model,
                prepared=prepared[corrupt],
                site=site,
                source_by_choice=corrupt_source,
                recipient_slot_indexes=slots,
                mode="self",
                seed=_deterministic_int(
                    model_key, request_id, corrupt, site.site_id, "corrupt-self"
                ),
            )
            if unsupported_reason or corrupt_self_scores is None:
                raise GateError("self-patch unexpectedly became a degenerate intervention")
            forward_count += count
            corrupt_self_margin = a_minus_b_margin_bits(corrupt_self_scores)
            corrupt_self_status = (
                "pass"
                if abs(corrupt_self_margin - corrupt_margins[corrupt])
                <= SELF_MARGIN_TOLERANCE_BITS
                else "fail"
            )
            rows.append(
                _outcome_row(
                    phase=phase,
                    model_key=model_key,
                    rank=rank,
                    request=request,
                    corruption=corrupt,
                    site=site,
                    patch_direction="identity_control",
                    intervention="corrupt_to_corrupt_self_patch",
                    source_request_id=request_id,
                    source_arm=corrupt,
                    recipient_arm=corrupt,
                    recipient_a_minus_b_margin=corrupt_margins[corrupt],
                    real_a_minus_b_margin=real_margin,
                    patched_a_minus_b_margin=corrupt_self_margin,
                    audits=audits,
                    status=corrupt_self_status,
                    reason=(
                        None
                        if corrupt_self_status == "pass"
                        else "corrupt_self_patch_margin_invariance_failed"
                    ),
                    slot_mapping={
                        "source_slots": list(_site_source_slots(site)),
                        "recipient_slots": list(slots),
                        "crop_slots_unchanged": True,
                    },
                )
            )
            if corrupt_self_status != "pass":
                raise GateError(f"corrupt self-patch invariance failed at {site.site_id}")

        self_scores, audits, count, unsupported_reason = forward_patched_scores(
            model=model,
            prepared=prepared["real"],
            site=site,
            source_by_choice=clean_source,
            recipient_slot_indexes=slots,
            mode="self",
            seed=_deterministic_int(model_key, request_id, site.site_id, "clean-self"),
        )
        if unsupported_reason or self_scores is None:
            raise GateError("self-patch unexpectedly became a degenerate intervention")
        forward_count += count
        self_margin = a_minus_b_margin_bits(self_scores)
        self_status = (
            "pass"
            if abs(self_margin - real_margin) <= SELF_MARGIN_TOLERANCE_BITS
            else "fail"
        )
        rows.append(
            _outcome_row(
                phase=phase,
                model_key=model_key,
                rank=rank,
                request=request,
                corruption="real",
                site=site,
                patch_direction="identity_control",
                intervention="clean_to_clean_self_patch",
                source_request_id=request_id,
                source_arm="real",
                recipient_arm="real",
                recipient_a_minus_b_margin=real_margin,
                real_a_minus_b_margin=real_margin,
                patched_a_minus_b_margin=self_margin,
                audits=audits,
                status=self_status,
                reason=None if self_status == "pass" else "clean_self_patch_margin_invariance_failed",
                slot_mapping={
                    "source_slots": list(_site_source_slots(site)),
                    "recipient_slots": list(slots),
                    "crop_slots_unchanged": True,
                },
            )
        )
        if self_status != "pass":
            raise GateError(f"clean self-patch invariance failed at {site.site_id}")
    del real_captures, corrupt_captures

    # Blinded unrelated controls are mirrored in both directions. The analyzer
    # retains exactly one same-label order after the technical gate.
    for unrelated_id, unrelated_request in zip(unrelated_ids, unrelated_requests):
        for donor_arm, direction, intervention, recipient_arm in (
            (
                "real",
                "clean_to_corrupt_rescue",
                "rescue_unrelated_donor_candidate",
                None,
            ),
            (
                "blank_full",
                "corrupt_to_clean_damage",
                "damage_unrelated_donor_candidate",
                "real",
            ),
            (
                "wrong_time",
                "corrupt_to_clean_damage",
                "damage_unrelated_donor_candidate",
                "real",
            ),
        ):
            prepare_started = time.perf_counter()
            donor_prepared = prepare_arm(
                runtime=runtime,
                processor=processor,
                tokenizer=tokenizer,
                device=device,
                model=model,
                model_key=model_key,
                row=unrelated_request,
                arm=donor_arm,
                data_dir=data_dir,
                candidate_completions=candidate_completions,
            )
            timings["prepare_inputs_s"] += time.perf_counter() - prepare_started
            assert_prepared_alignment(prepared[donor_arm], donor_prepared)
            capture_started = time.perf_counter()
            donor_scores, donor_captures, count = forward_capture_scores(
                model=model,
                prepared=donor_prepared,
                sites=sites,
                slot_indexes_by_site=source_slots,
            )
            forward_count += count
            timings["source_capture_s"] += time.perf_counter() - capture_started
            del donor_scores
            target_corruptions = CORRUPT_ARMS if donor_arm == "real" else (donor_arm,)
            for site in sites:
                slots = _site_recipient_slots(site)
                donor_source = {
                    choice: donor_captures[choice][site.site_id] for choice in CHOICES
                }
                for corrupt in target_corruptions:
                    target_arm = corrupt if recipient_arm is None else recipient_arm
                    patched_scores, audits, count, unsupported_reason = forward_patched_scores(
                        model=model,
                        prepared=prepared[target_arm],
                        site=site,
                        source_by_choice=donor_source,
                        recipient_slot_indexes=slots,
                        mode="unrelated",
                        seed=_deterministic_int(
                            model_key,
                            request_id,
                            corrupt,
                            site.site_id,
                            intervention,
                            unrelated_id,
                        ),
                    )
                    forward_count += count
                    patched_margin = (
                        a_minus_b_margin_bits(patched_scores)
                        if patched_scores is not None
                        else None
                    )
                    rows.append(
                        _outcome_row(
                            phase=phase,
                            model_key=model_key,
                            rank=rank,
                            request=request,
                            corruption=corrupt,
                            site=site,
                            patch_direction=direction,
                            intervention=intervention,
                            source_request_id=unrelated_id,
                            source_arm=donor_arm,
                            recipient_arm=target_arm,
                            recipient_a_minus_b_margin=(
                                corrupt_margins[corrupt]
                                if target_arm == corrupt
                                else real_margin
                            ),
                            real_a_minus_b_margin=real_margin,
                            patched_a_minus_b_margin=patched_margin,
                            audits=audits,
                            status="unsupported" if unsupported_reason else "pass",
                            reason=unsupported_reason,
                            slot_mapping={
                                "source_slots": list(_site_source_slots(site)),
                                "recipient_slots": list(slots),
                                "full_frame_only": site.tensor_selector != "question_token",
                                "crop_slots_unchanged": True,
                                "same_label_status": "blinded_pending_analyzer_join",
                            },
                        )
                    )
            del donor_captures, donor_prepared

    # Finally consume the paired crop-order source; it never modifies slot 0.
    partner_id = str(request["counterbalanced_request_id"])
    partner_request = public_map[partner_id]
    prepare_started = time.perf_counter()
    partner_prepared = prepare_arm(
        runtime=runtime,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        model=model,
        model_key=model_key,
        row=partner_request,
        arm="real",
        data_dir=data_dir,
        candidate_completions=candidate_completions,
    )
    timings["prepare_inputs_s"] += time.perf_counter() - prepare_started
    assert_prepared_alignment(prepared["real"], partner_prepared)
    current_assets = [row["sha256"] for row in prepared["real"].asset_receipts]
    partner_assets = [row["sha256"] for row in partner_prepared.asset_receipts]
    if current_assets != [partner_assets[0], partner_assets[2], partner_assets[1]]:
        raise GateError("counterbalanced partner is not the exact slot-1/slot-2 permutation")
    transfer_sites = [
        site for site in sites if transfer_site_applicability(site)[0]
    ]
    prefusion_sites = [
        site for site in sites if site.family in PREFUSION_TRANSFER_FAMILIES
    ]
    partner_capture_sites = [*transfer_sites, *prefusion_sites]
    capture_started = time.perf_counter()
    partner_scores, partner_captures, count = forward_capture_scores(
        model=model,
        prepared=partner_prepared,
        sites=partner_capture_sites,
        slot_indexes_by_site={
            site.site_id: (1, 2) for site in partner_capture_sites
        },
    )
    forward_count += count
    timings["source_capture_s"] += time.perf_counter() - capture_started
    del partner_scores
    recipient_prefusion_captures: dict[str, dict[str, Any]] = {}
    prefusion_identity_audits: dict[str, dict[str, Any]] = {}
    if prefusion_sites:
        recipient_capture_started = time.perf_counter()
        recipient_scores, recipient_prefusion_captures, count = forward_capture_scores(
            model=model,
            prepared=prepared["real"],
            sites=prefusion_sites,
            slot_indexes_by_site={
                site.site_id: (2, 1) for site in prefusion_sites
            },
        )
        forward_count += count
        timings["source_capture_s"] += (
            time.perf_counter() - recipient_capture_started
        )
        del recipient_scores
        for prefusion_site in prefusion_sites:
            prefusion_identity_audits[prefusion_site.site_id] = (
                content_aligned_identity_audit(
                    site=prefusion_site,
                    source_by_choice={
                        choice: partner_captures[choice][prefusion_site.site_id]
                        for choice in CHOICES
                    },
                    recipient_by_choice={
                        choice: recipient_prefusion_captures[choice][
                            prefusion_site.site_id
                        ]
                        for choice in CHOICES
                    },
                )
            )
    for site in sites:

        transfer_supported, structural_reason = transfer_site_applicability(site)
        if transfer_supported:
            transfer_source = {
                choice: partner_captures[choice][site.site_id] for choice in CHOICES
            }
            transfer_scores, audits, count, unsupported_reason = forward_patched_scores(
                model=model,
                prepared=prepared["real"],
                site=site,
                source_by_choice=transfer_source,
                recipient_slot_indexes=(2, 1),
                mode="counterbalanced_order",
                seed=_deterministic_int(model_key, request_id, site.site_id, "counterbalanced"),
            )
            forward_count += count
            transfer_margin = (
                a_minus_b_margin_bits(transfer_scores)
                if transfer_scores is not None
                else None
            )
            rows.append(
                _outcome_row(
                    phase=phase,
                    model_key=model_key,
                    rank=rank,
                    request=request,
                    corruption="counterbalanced_order",
                    site=site,
                    patch_direction="counterbalanced_crop_transfer",
                    intervention="counterbalanced_content_aligned_crop_transfer",
                    source_request_id=partner_id,
                    source_arm="real",
                    recipient_arm="real",
                    recipient_a_minus_b_margin=real_margin,
                    real_a_minus_b_margin=real_margin,
                    patched_a_minus_b_margin=transfer_margin,
                    audits=audits,
                    status="unsupported" if unsupported_reason else "pass",
                    reason=unsupported_reason,
                    slot_mapping={
                        "source_slots": [1, 2],
                        "recipient_slots": [2, 1],
                        "mapping": {"donor_crop_A": "recipient_crop_B", "donor_crop_B": "recipient_crop_A"},
                        "full_frame_slot_unchanged": True,
                    },
                )
            )
        else:
            if structural_reason is None:
                raise GateError(
                    f"missing structural crop-transfer reason for {site.site_id}"
                )
            rows.append(
                _outcome_row(
                    phase=phase,
                    model_key=model_key,
                    rank=rank,
                    request=request,
                    corruption="counterbalanced_order",
                    site=site,
                    patch_direction="counterbalanced_crop_transfer",
                    intervention="counterbalanced_content_aligned_crop_transfer",
                    source_request_id=partner_id,
                    source_arm="real",
                    recipient_arm="real",
                    recipient_a_minus_b_margin=real_margin,
                    real_a_minus_b_margin=real_margin,
                    patched_a_minus_b_margin=None,
                    audits=[],
                    status="not_applicable",
                    reason=structural_reason,
                    slot_mapping=structural_transfer_slot_mapping(
                        site,
                        identity_audit=prefusion_identity_audits.get(site.site_id),
                    ),
                )
            )
    del partner_captures, recipient_prefusion_captures, partner_prepared
    timings["patch_forwards_s"] += time.perf_counter() - patch_started
    elapsed = time.perf_counter() - started
    return rows, {
        "request_id": request_id,
        "elapsed_s": elapsed,
        "patch_forward_count": forward_count,
        "patch_forwards_per_s": forward_count / elapsed if elapsed > 0 else None,
        "timings_s": dict(timings),
        "slot_audit": _slot_audit_payload(prepared["real"].slot_map),
        "question_token_audit": prepared["real"].question_token_audit,
        "peak_simultaneous_source_activation_rows": 1,
        "bidirectional_intervention": True,
    }


def _result_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("phase")),
        str(row.get("model_key")),
        str(row.get("request_id")),
        str(row.get("corruption")),
        str(row.get("site_id")),
        str(row.get("intervention")),
        str(row.get("source_request_id")),
    )


def sanitize_resume_files(
    results_path: Path,
    complete_path: Path,
    *,
    expected_resume_binding_sha256: str,
    expected_model_key: str,
    expected_rank: int,
    expected_phase: str,
    expected_request_ids: set[str],
) -> dict[str, Any]:
    """Retain only complete rows bound to this exact release/schema worker."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_resume_binding_sha256):
        raise GateError("resume sanitizer requires an exact binding SHA-256")
    completion_receipts: dict[str, dict[str, Any]] = {}
    if complete_path.exists():
        for row in read_jsonl(complete_path):
            request_id = str(row.get("request_id") or "")
            if (
                set(row) != RESUME_COMPLETION_FIELDS
                or request_id not in expected_request_ids
                or row.get("output_schema_version") != OUTPUT_SCHEMA_VERSION
                or row.get("resume_binding_sha256")
                != expected_resume_binding_sha256
                or row.get("model_key") != expected_model_key
                or row.get("rank") != expected_rank
                or row.get("phase") != expected_phase
                or not isinstance(row.get("result_row_count"), int)
                or int(row["result_row_count"]) < 1
                or not isinstance(row.get("result_batch_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["result_batch_sha256"])
                or not isinstance(row.get("completed_at"), str)
                or not row["completed_at"]
            ):
                raise GateError(
                    "resume completion receipt differs from current release/schema "
                    f"binding: {request_id or '<missing>'}"
                )
            if request_id in completion_receipts and completion_receipts[request_id] != row:
                raise GateError(f"conflicting completion receipts for {request_id}")
            completion_receipts[request_id] = row
    completed = set(completion_receipts)
    kept: list[dict[str, Any]] = []
    dropped = 0
    duplicate = 0
    seen: set[tuple[str, ...]] = set()
    if results_path.exists():
        try:
            rows = read_jsonl(results_path)
        except GateError:
            rows = []
            with results_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        dropped += 1
                        continue
                    if isinstance(value, dict):
                        rows.append(value)
        for row in rows:
            request_id = str(row.get("request_id") or "")
            if (
                set(row) != OUTCOME_ROW_FIELDS
                or row.get("schema_version") != OUTPUT_SCHEMA_VERSION
                or row.get("model_key") != expected_model_key
                or row.get("rank") != expected_rank
                or row.get("phase") != expected_phase
                or request_id not in expected_request_ids
            ):
                raise GateError(
                    "resume outcome row differs from current release/schema worker "
                    f"binding: {request_id or '<missing>'}"
                )
            if request_id not in completed:
                dropped += 1
                continue
            key = _result_identity(row)
            if key in seen:
                duplicate += 1
                continue
            seen.add(key)
            kept.append(row)
        by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in kept:
            by_request[str(row["request_id"])].append(row)
        invalid_completed: set[str] = set()
        for request_id, receipt in completion_receipts.items():
            request_rows = by_request.get(request_id, [])
            expected_count = receipt.get("result_row_count")
            expected_hash = receipt.get("result_batch_sha256")
            if (
                not isinstance(expected_count, int)
                or expected_count != len(request_rows)
                or not isinstance(expected_hash, str)
                or expected_hash != sha256_json(request_rows)
            ):
                invalid_completed.add(request_id)
        if invalid_completed:
            retained: list[dict[str, Any]] = []
            for row in kept:
                if str(row["request_id"]) in invalid_completed:
                    dropped += 1
                else:
                    retained.append(row)
            kept = retained
            completed.difference_update(invalid_completed)
            completion_receipts = {
                request_id: row
                for request_id, row in completion_receipts.items()
                if request_id not in invalid_completed
            }
        atomic_write_text(
            results_path,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in kept),
        )
        atomic_write_text(
            complete_path,
            "".join(
                json.dumps(completion_receipts[request_id], sort_keys=True) + "\n"
                for request_id in sorted(completion_receipts)
            ),
        )
    elif completion_receipts:
        completed.clear()
        completion_receipts.clear()
        atomic_write_text(complete_path, "")
    return {
        "completed_request_ids": sorted(completed),
        "kept_rows": len(kept),
        "dropped_incomplete_or_invalid_rows": dropped,
        "dropped_duplicate_rows": duplicate,
    }


def combined_source_activation_peak(
    trial_receipts: Sequence[Mapping[str, Any]],
    *,
    resumed_kept_rows: int,
) -> int:
    """Combine fresh and exact-release resumed activation-cache evidence."""
    if resumed_kept_rows < 0:
        raise GateError("resumed kept-row count cannot be negative")
    fresh_peak = max(
        [
            int(row.get("peak_simultaneous_source_activation_rows") or 0)
            for row in trial_receipts
        ]
        or [0]
    )
    # Sanitized resumed rows are bound to this exact worker implementation and
    # could only have been committed after a one-row trial completed.
    resumed_peak = 1 if resumed_kept_rows > 0 else 0
    return max(fresh_peak, resumed_peak)


def model_and_shard_for_rank(rank: int, world_size: int) -> tuple[str, int, int]:
    if world_size != 4 or rank not in range(4):
        raise GateError("Job C requires exactly four ranks")
    if rank < 2:
        return "qwen35_27b_hf_bf16", rank, 2
    return "gemma4_31b_hf_bf16", rank - 2, 2


def shard_public_rows(
    rows: Sequence[dict[str, Any]], *, shard_index: int, shard_count: int
) -> list[dict[str, Any]]:
    frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        frames[str(row["frame_unit_id"])].append(row)
    ordered_frames = sorted(
        frames,
        key=lambda frame_id: hashlib.sha256(
            f"{DEFAULT_SEED}|mechanism-shard|{frame_id}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    for index, frame_id in enumerate(ordered_frames):
        if index % shard_count == shard_index:
            selected.extend(sorted(frames[frame_id], key=lambda row: str(row["order_id"])))
    return selected


def refinement_layers_for_model(path: Path, model_key: str) -> dict[str, list[int]]:
    payload = read_json(path)
    if payload.get("governing_prespec_sha256") != PRESPEC_SHA256:
        raise GateError("refinement plan prespec hash mismatch")
    rows = payload.get("models", {}).get(model_key, {})
    if not isinstance(rows, Mapping):
        raise GateError(f"refinement plan lacks model {model_key}")
    layers: dict[str, list[int]] = {}
    for family in ("decoder_visual_token_residual", "decoder_final_question_token_residual"):
        values = rows.get(family, {}).get("refinement_layers", [])
        if not isinstance(values, list):
            raise GateError(f"invalid refinement layer list for {model_key}:{family}")
        layers[family] = sorted(set(int(value) for value in values))
    return layers


def _load_anchor_model(
    *, model_key: str, dtype: str, local_files_only: bool
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    adapter = MODEL_ADAPTERS[model_key]
    runtime, model, processor, tokenizer, device, provenance = job3._load_model_bundle(
        model_row={
            "model_key": model_key,
            "model_id": str(adapter["model_id"]),
            "revision": str(adapter["revision"]),
        },
        dtype=dtype,
        local_files_only=local_files_only,
    )
    snapshot = provenance.get("model_snapshot", {})
    resolved = str(snapshot.get("resolved_snapshot_revision") or "")
    if resolved and resolved != str(adapter["revision"]):
        raise GateError(
            f"model revision mismatch for {model_key}: expected {adapter['revision']}, got {resolved}"
        )
    return runtime, model, processor, tokenizer, device, provenance


def _run_technical_canary_case(
    *,
    runtime: Any,
    model: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    model_key: str,
    rank: int,
    choice: str,
    request: dict[str, Any],
    public_map: Mapping[str, dict[str, Any]],
    sites: Sequence[PatchSite],
    data_dir: Path,
    candidate_completions: Mapping[str, str],
) -> dict[str, Any]:
    """Exercise one candidate path without labels or outcome directions."""
    import torch

    started = time.perf_counter()
    real = prepare_arm(
        runtime=runtime,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        model=model,
        model_key=model_key,
        row=request,
        arm="real",
        data_dir=data_dir,
        candidate_completions=candidate_completions,
    )
    corrupt_prepared = {
        arm: prepare_arm(
            runtime=runtime,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            model=model,
            model_key=model_key,
            row=request,
            arm=arm,
            data_dir=data_dir,
            candidate_completions=candidate_completions,
        )
        for arm in CORRUPT_ARMS
    }
    partner_request = public_map[str(request["counterbalanced_request_id"])]
    partner = prepare_arm(
        runtime=runtime,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        model=model,
        model_key=model_key,
        row=partner_request,
        arm="real",
        data_dir=data_dir,
        candidate_completions=candidate_completions,
    )
    for arm in CORRUPT_ARMS:
        assert_prepared_alignment(real, corrupt_prepared[arm])
    assert_prepared_alignment(real, partner)
    real_assets = [row["sha256"] for row in real.asset_receipts]
    partner_assets = [row["sha256"] for row in partner.asset_receipts]
    if real_assets != [partner_assets[0], partner_assets[2], partner_assets[1]]:
        raise GateError("canary counterbalanced partner is not an exact crop permutation")
    canary_sites = list(sites)
    if choice not in CHOICES:
        raise GateError(f"unknown technical-canary candidate choice: {choice}")
    source_slots = {site.site_id: _site_source_slots(site) for site in canary_sites}
    with capture_module_outputs(
        model=model,
        sites=canary_sites,
        slot_map=real.choices[choice].slot_map,
        slot_indexes_by_site=source_slots,
        question_position=real.question_position,
    ) as source_captures:
        with torch.inference_mode():
            output = model(**real.choices[choice].inputs, use_cache=False, return_dict=True)
    if gates._extract_logits(output) is None:
        raise GateError("canary real forward produced no logits")
    del output
    audits: list[dict[str, Any]] = []
    unsupported_events: list[dict[str, Any]] = []
    forwards = 1
    corrupt_source_captures: dict[str, dict[str, Any]] = {}
    for arm in CORRUPT_ARMS:
        with capture_module_outputs(
            model=model,
            sites=canary_sites,
            slot_map=corrupt_prepared[arm].choices[choice].slot_map,
            slot_indexes_by_site=source_slots,
            question_position=corrupt_prepared[arm].question_position,
        ) as captured:
            with torch.inference_mode():
                output = model(
                    **corrupt_prepared[arm].choices[choice].inputs,
                    use_cache=False,
                    return_dict=True,
                )
        if gates._extract_logits(output) is None:
            raise GateError(f"canary {arm} source forward produced no logits")
        corrupt_source_captures[arm] = captured
        forwards += 1
        del output

    def exercise_patch(
        *,
        site: PatchSite,
        source_selected: Any,
        recipient: PreparedArm,
        mode: str,
        direction: str,
        source_arm: str,
        recipient_arm: str,
        recipient_slot_indexes: Sequence[int] | None = None,
    ) -> None:
        nonlocal forwards
        module = module_by_path(model, site.module_name)
        audit: HookAudit | None = None
        try:
            with patch_module_output(
                module=module,
                site=site,
                source_selected=source_selected,
                recipient_slot_map=recipient.choices[choice].slot_map,
                recipient_slot_indexes=(
                    tuple(recipient_slot_indexes)
                    if recipient_slot_indexes is not None
                    else _site_recipient_slots(site)
                ),
                question_position=recipient.question_position,
                mode=mode,
                seed=_deterministic_int(
                    model_key,
                    str(request["request_id"]),
                    site.site_id,
                    direction,
                    mode,
                    source_arm,
                    recipient_arm,
                    choice,
                ),
            ) as audit:
                with torch.inference_mode():
                    patched = model(
                        **recipient.choices[choice].inputs,
                        use_cache=False,
                        return_dict=True,
                    )
        except DegenerateIntervention as exc:
            event = {
                "candidate_choice": choice,
                "site_id": site.site_id,
                "mode": mode,
                "patch_direction": direction,
                "source_arm": source_arm,
                "recipient_arm": recipient_arm,
                "status": "unsupported",
                "reason": f"{type(exc).__name__}: {exc}",
                "audit": asdict(audit) if audit is not None else None,
            }
            audits.append(event)
            unsupported_events.append(event)
            return
        assert audit is not None
        if gates._extract_logits(patched) is None or audit.status != "pass" or not audit.hook_removed:
            raise GateError(f"canary bidirectional patch failed: {asdict(audit)}")
        audits.append(
            {
                "candidate_choice": choice,
                "site_id": site.site_id,
                "mode": mode,
                "patch_direction": direction,
                "source_arm": source_arm,
                "recipient_arm": recipient_arm,
                "audit": asdict(audit),
            }
        )
        forwards += 1
        del patched

    for site in canary_sites:
        exercise_patch(
            site=site,
            source_selected=source_captures[site.site_id],
            recipient=real,
            mode="self",
            direction="identity_control",
            source_arm="real",
            recipient_arm="real",
        )
        for arm in CORRUPT_ARMS:
            for mode in ("matched", "norm_matched_random"):
                exercise_patch(
                    site=site,
                    source_selected=source_captures[site.site_id],
                    recipient=corrupt_prepared[arm],
                    mode=mode,
                    direction="clean_to_corrupt_rescue",
                    source_arm="real",
                    recipient_arm=arm,
                )
                exercise_patch(
                    site=site,
                    source_selected=corrupt_source_captures[arm][site.site_id],
                    recipient=real,
                    mode=mode,
                    direction="corrupt_to_clean_damage",
                    source_arm=arm,
                    recipient_arm="real",
                )
            exercise_patch(
                site=site,
                source_selected=corrupt_source_captures[arm][site.site_id],
                recipient=corrupt_prepared[arm],
                mode="self",
                direction="identity_control",
                source_arm=arm,
                recipient_arm=arm,
            )
    del source_captures, corrupt_source_captures

    donor_id = select_unrelated_donor_pairs(list(public_map.values()))[
        str(request["request_id"])
    ][0]
    for donor_arm in ARMS:
        donor = prepare_arm(
            runtime=runtime,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            model=model,
            model_key=model_key,
            row=public_map[donor_id],
            arm=donor_arm,
            data_dir=data_dir,
            candidate_completions=candidate_completions,
        )
        assert_prepared_alignment(real if donor_arm == "real" else corrupt_prepared[donor_arm], donor)
        with capture_module_outputs(
            model=model,
            sites=canary_sites,
            slot_map=donor.choices[choice].slot_map,
            slot_indexes_by_site=source_slots,
            question_position=donor.question_position,
        ) as donor_captures:
            with torch.inference_mode():
                donor_output = model(
                    **donor.choices[choice].inputs,
                    use_cache=False,
                    return_dict=True,
                )
        if gates._extract_logits(donor_output) is None:
            raise GateError("canary unrelated donor forward produced no logits")
        forwards += 1
        del donor_output
        for site in canary_sites:
            target_arms = CORRUPT_ARMS if donor_arm == "real" else ("real",)
            for target_arm in target_arms:
                corrupt = target_arm if donor_arm == "real" else donor_arm
                exercise_patch(
                    site=site,
                    source_selected=donor_captures[site.site_id],
                    recipient=(real if target_arm == "real" else corrupt_prepared[target_arm]),
                    mode="unrelated",
                    direction=(
                        "clean_to_corrupt_rescue"
                        if donor_arm == "real"
                        else "corrupt_to_clean_damage"
                    ),
                    source_arm=donor_arm,
                    recipient_arm=target_arm,
                )
        del donor_captures, donor

    transfer_applicability = {
        site.site_id: transfer_site_applicability(site) for site in canary_sites
    }
    transfer_sites = [
        site for site in canary_sites if transfer_applicability[site.site_id][0]
    ]
    prefusion_sites = [
        site for site in canary_sites if site.family in PREFUSION_TRANSFER_FAMILIES
    ]
    structurally_not_applicable_sites = [
        site for site in canary_sites if not transfer_applicability[site.site_id][0]
    ]
    partner_capture_sites = [*transfer_sites, *prefusion_sites]
    with capture_module_outputs(
        model=model,
        sites=partner_capture_sites,
        slot_map=partner.choices[choice].slot_map,
        slot_indexes_by_site={
            site.site_id: (1, 2) for site in partner_capture_sites
        },
        question_position=partner.question_position,
    ) as partner_captures:
        with torch.inference_mode():
            partner_output = model(
                **partner.choices[choice].inputs,
                use_cache=False,
                return_dict=True,
            )
    if gates._extract_logits(partner_output) is None:
        raise GateError("canary partner forward produced no logits")
    del partner_output
    forwards += 1
    recipient_prefusion_captures: dict[str, Any] = {}
    if prefusion_sites:
        with capture_module_outputs(
            model=model,
            sites=prefusion_sites,
            slot_map=real.choices[choice].slot_map,
            slot_indexes_by_site={
                site.site_id: (2, 1) for site in prefusion_sites
            },
            question_position=real.question_position,
        ) as recipient_prefusion_captures:
            with torch.inference_mode():
                recipient_prefusion_output = model(
                    **real.choices[choice].inputs,
                    use_cache=False,
                    return_dict=True,
                )
        if gates._extract_logits(recipient_prefusion_output) is None:
            raise GateError("canary recipient pre-fusion audit forward produced no logits")
        del recipient_prefusion_output
        forwards += 1
    prefusion_identity_audits = {
        site.site_id: content_aligned_identity_audit(
            site=site,
            source_by_choice={choice: partner_captures[site.site_id]},
            recipient_by_choice={choice: recipient_prefusion_captures[site.site_id]},
        )
        for site in prefusion_sites
    }
    structural_site_records = [
        {
            "site_id": site.site_id,
            "stage_family": site.family,
            "reason": transfer_applicability[site.site_id][1],
            "slot_mapping": structural_transfer_slot_mapping(
                site,
                identity_audit=prefusion_identity_audits.get(site.site_id),
            ),
        }
        for site in structurally_not_applicable_sites
    ]
    structural_not_applicable_gate = {
        "status": (
            "pass"
            if structurally_not_applicable_sites
            and all(
                structural_transfer_record_is_valid(
                    row, expected_choices=(choice,)
                )
                for row in structural_site_records
            )
            else "fail"
        ),
        "not_applicable_count": len(structurally_not_applicable_sites),
        "site_ids": [site.site_id for site in structurally_not_applicable_sites],
        "intervention": "counterbalanced_content_aligned_crop_transfer",
        "sites": structural_site_records,
        "reason_counts": dict(
            sorted(
                Counter(str(row["reason"]) for row in structural_site_records).items()
            )
        ),
    }
    for site in transfer_sites:
        exercise_patch(
            site=site,
            source_selected=partner_captures[site.site_id],
            recipient=real,
            recipient_slot_indexes=(2, 1),
            mode="counterbalanced_order",
            direction="counterbalanced_crop_transfer",
            source_arm="real_counterbalanced_partner",
            recipient_arm="real",
        )
    del partner_captures, recipient_prefusion_captures
    elapsed = time.perf_counter() - started
    observed_rate = forwards / elapsed if elapsed > 0 else None
    coarse_site_count = len(canary_sites)
    coarse_transfer_site_count = len(transfer_sites)
    expected_canary_forward_count = (
        8 + 15 * coarse_site_count + coarse_transfer_site_count
    )
    expected_coarse_forwards_per_trial = (
        22 + 38 * coarse_site_count + 2 * coarse_transfer_site_count
    )
    # Each decoder family can add at most four untested +/-2 neighbors.
    expected_refine_upper_forwards_per_trial = 20 + 38 * 8 + 2 * 4
    full_trials_per_rank = MECHANISM_TRIAL_COUNT // 2
    projected_seconds = (
        full_trials_per_rank
        * (expected_coarse_forwards_per_trial + expected_refine_upper_forwards_per_trial)
        / observed_rate
        if observed_rate
        else None
    )
    return {
        "kind": "phase2_reanchored_causal_patch_evaluator_unjoined_canary",
        "status": (
            "pass"
            if not unsupported_events
            and structural_not_applicable_gate["status"] == "pass"
            else "fail"
        ),
        "model_key": model_key,
        "rank": rank,
        "candidate_choice": choice,
        "request_id_sha256": hashlib.sha256(str(request["request_id"]).encode()).hexdigest(),
        "label_joined": False,
        "outcome_metrics_emitted": False,
        "bidirectional_interventions_exercised": (
            not unsupported_events
            and structural_not_applicable_gate["status"] == "pass"
        ),
        "representable_intervention_gate": {
            "status": (
                "pass"
                if not unsupported_events
                and structural_not_applicable_gate["status"] == "pass"
                else "fail"
            ),
            "unsupported_count": len(unsupported_events),
            "unsupported_events": unsupported_events,
            "not_applicable_count": len(structurally_not_applicable_sites),
            "invalid_not_applicable_count": 0,
            "structural_not_applicable_gate": structural_not_applicable_gate,
            "rule": (
                "every applicable evaluator-unjoined canary intervention must be "
                "representable; content-aligned crop transfer is structurally absent only "
                "after fresh exact activation equality, or at question-token sites without "
                "crop slices"
            ),
        },
        "slot_audit": _slot_audit_payload(real.slot_map),
        "question_token_audit": real.question_token_audit,
        "hook_audits": audits,
        "forward_count": forwards,
        "expected_canary_forward_count": expected_canary_forward_count,
        "observed_forwards_per_s": observed_rate,
        "expected_coarse_forwards_per_trial": expected_coarse_forwards_per_trial,
        "expected_refine_upper_forwards_per_trial": expected_refine_upper_forwards_per_trial,
        "full_trials_per_rank": full_trials_per_rank,
        "projected_full_inference_seconds_per_rank_upper": projected_seconds,
        "projected_full_inference_hours_per_rank_upper": (
            projected_seconds / 3600.0 if projected_seconds is not None else None
        ),
        "projection_formula": {
            "version": 4,
            "coarse": "22 + 38*coarse_site_count + 2*transfer_site_count",
            "refinement_upper": "20 + 38*8 + 2*4",
            "includes_fresh_prefusion_identity_audit": True,
            "includes_clean_to_corrupt_rescue": True,
            "includes_corrupt_to_clean_damage": True,
        },
        "peak_simultaneous_source_activation_rows": 1,
        "elapsed_s": elapsed,
    }


def run_technical_canary(
    *,
    runtime: Any,
    model: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    model_key: str,
    rank: int,
    request: dict[str, Any],
    public_map: Mapping[str, dict[str, Any]],
    sites: Sequence[PatchSite],
    data_dir: Path,
    candidate_completions: Mapping[str, str],
) -> dict[str, Any]:
    """Exercise both frozen A/B candidate paths and combine fail-closed receipts."""

    cases = [
        _run_technical_canary_case(
            runtime=runtime,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            model_key=model_key,
            rank=rank,
            choice=choice,
            request=request,
            public_map=public_map,
            sites=sites,
            data_dir=data_dir,
            candidate_completions=candidate_completions,
        )
        for choice in CHOICES
    ]
    unsupported_events = [
        event
        for case in cases
        for event in case["representable_intervention_gate"]["unsupported_events"]
    ]
    question_audits = [
        {
            "candidate_choice": case["candidate_choice"],
            "audit": case["question_token_audit"],
        }
        for case in cases
    ]
    question_gate_pass = all(
        row["audit"].get("status") == "pass"
        and row["audit"].get("contract") == QUESTION_TOKEN_CONTRACT
        for row in question_audits
    )
    structural_cases = [
        {
            "candidate_choice": case["candidate_choice"],
            **case["representable_intervention_gate"][
                "structural_not_applicable_gate"
            ],
        }
        for case in cases
    ]
    structural_pass = all(row.get("status") == "pass" for row in structural_cases)
    status = (
        "pass"
        if all(case.get("status") == "pass" for case in cases)
        and not unsupported_events
        and question_gate_pass
        and structural_pass
        else "fail"
    )
    elapsed = sum(float(case.get("elapsed_s") or 0.0) for case in cases)
    forwards = sum(int(case.get("forward_count") or 0) for case in cases)
    return {
        "kind": "phase2_reanchored_causal_patch_evaluator_unjoined_canary",
        "status": status,
        "model_key": model_key,
        "rank": rank,
        "request_id_sha256": cases[0]["request_id_sha256"],
        "candidate_choices_exercised": list(CHOICES),
        "canary_case_count": len(cases),
        "label_joined": False,
        "outcome_metrics_emitted": False,
        "bidirectional_interventions_exercised": status == "pass",
        "representable_intervention_gate": {
            "status": "pass" if not unsupported_events and structural_pass else "fail",
            "unsupported_count": len(unsupported_events),
            "unsupported_events": unsupported_events,
            "not_applicable_count": sum(
                int(
                    case["representable_intervention_gate"].get(
                        "not_applicable_count", 0
                    )
                )
                for case in cases
            ),
            "invalid_not_applicable_count": 0,
            "structural_not_applicable_gate": {
                "status": "pass" if structural_pass else "fail",
                "cases": structural_cases,
            },
            "rule": (
                "both frozen candidate paths must represent every applicable "
                "evaluator-unjoined intervention; content-aligned crop transfer remains "
                "structurally absent after exact activation identity is proven, and at "
                "question-token sites"
            ),
        },
        "slot_audits": [
            {
                "candidate_choice": case["candidate_choice"],
                "audit": case["slot_audit"],
            }
            for case in cases
        ],
        "question_token_audit": {
            "status": "pass" if question_gate_pass else "fail",
            "contract": QUESTION_TOKEN_CONTRACT,
            "candidate_choice_audits": question_audits,
        },
        "hook_audits": [
            audit for case in cases for audit in case["hook_audits"]
        ],
        "forward_count": forwards,
        "expected_canary_forward_count": sum(
            int(case["expected_canary_forward_count"]) for case in cases
        ),
        "observed_forwards_per_s": forwards / elapsed if elapsed > 0 else None,
        "expected_coarse_forwards_per_trial": max(
            int(case["expected_coarse_forwards_per_trial"]) for case in cases
        ),
        "expected_refine_upper_forwards_per_trial": max(
            int(case["expected_refine_upper_forwards_per_trial"]) for case in cases
        ),
        "full_trials_per_rank": cases[0]["full_trials_per_rank"],
        "projected_full_inference_seconds_per_rank_upper": max(
            float(case["projected_full_inference_seconds_per_rank_upper"])
            for case in cases
        ),
        "projected_full_inference_hours_per_rank_upper": max(
            float(case["projected_full_inference_hours_per_rank_upper"])
            for case in cases
        ),
        "projection_formula": {
            **cases[0]["projection_formula"],
            "canary_candidate_paths_exercised": list(CHOICES),
        },
        "peak_simultaneous_source_activation_rows": max(
            int(case["peak_simultaneous_source_activation_rows"])
            for case in cases
        ),
        "elapsed_s": elapsed,
        "canary_cases": cases,
    }


def preflight_command(args: argparse.Namespace) -> int:
    repo_dir = args.repo_dir.expanduser().resolve()
    prespec = verify_governing_prespec(repo_dir)
    manifest_path = args.mechanism_manifest.expanduser().resolve()
    public_path = args.public_manifest.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    mechanism_receipt = validate_mechanism_manifest(
        data_dir=data_dir,
        manifest_path=manifest_path,
        public_path=public_path,
    )
    public_rows = read_jsonl(public_path)
    receipt = validate_dataset(
        public_rows=public_rows,
        hidden_rows=None,
        data_dir=data_dir,
        expected_frames=args.expected_frames,
        expected_user_prompt=str(prespec["payload"]["generation"]["prompt_text"]),
    )
    receipt.update(
        {
            "kind": "phase2_reanchored_causal_patch_preflight",
            "governing_prespec": {key: value for key, value in prespec.items() if key != "payload"},
            "public_manifest": str(public_path),
            "public_manifest_sha256": sha256_file(public_path),
            "mechanism_manifest": mechanism_receipt,
            "hidden_index_opened": False,
            "model_adapters_sha256": sha256_json(MODEL_ADAPTERS),
            "created_at": utc_now(),
        }
    )
    if args.out:
        atomic_write_json(args.out.expanduser().resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def worker_command(args: argparse.Namespace) -> int:
    import torch

    sif_sha256 = required_sha256_env("SIF_SHA256")
    release_manifest_sha256 = required_sha256_env("RELEASE_MANIFEST_SHA256")
    model_plan_sha256 = required_sha256_env("MODEL_PLAN_SHA256")
    repo_dir = args.repo_dir.expanduser().resolve()
    prespec = verify_governing_prespec(repo_dir)
    manifest_path = args.mechanism_manifest.expanduser().resolve()
    public_path = args.public_manifest.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    code_snapshot_sha256 = str(os.environ.get("CODE_SNAPSHOT_SHA256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", code_snapshot_sha256):
        raise GateError("worker requires the node-local CODE_SNAPSHOT_SHA256")
    model_key, shard_index, shard_count = model_and_shard_for_rank(args.rank, args.world_size)
    out_dir = run_dir / f"rank_{args.rank:03d}" / args.phase
    out_dir.mkdir(parents=True, exist_ok=True)
    mechanism_receipt = validate_mechanism_manifest(
        data_dir=data_dir,
        manifest_path=manifest_path,
        public_path=public_path,
    )
    public_rows = read_jsonl(public_path)
    validate_dataset(
        public_rows=public_rows,
        hidden_rows=None,
        data_dir=data_dir,
        expected_frames=args.expected_frames,
        expected_user_prompt=str(prespec["payload"]["generation"]["prompt_text"]),
    )
    public_map = {str(row["request_id"]): row for row in public_rows}
    selected_rows = shard_public_rows(
        public_rows, shard_index=shard_index, shard_count=shard_count
    )
    if args.canary:
        selected_rows = selected_rows[:1]
    refinement_layers: dict[str, list[int]] | None = None
    if args.phase == "refine":
        if not args.refinement_plan:
            raise GateError("refine phase requires --refinement-plan")
        refinement_layers = refinement_layers_for_model(
            args.refinement_plan.expanduser().resolve(), model_key
        )
    refinement_plan_sha256 = (
        sha256_file(args.refinement_plan.expanduser().resolve())
        if args.refinement_plan is not None
        else None
    )
    results_path = out_dir / "patch_outcomes.jsonl"
    complete_path = out_dir / "completed_requests.jsonl"
    existing_manifest_path = out_dir / "worker_manifest.json"
    preload_manifest_hashes = {
        "governing_prespec_sha256": PRESPEC_SHA256,
        "code_snapshot_sha256": code_snapshot_sha256,
        "mechanism_manifest_sha256": mechanism_receipt["manifest_sha256"],
        "public_manifest_sha256": sha256_file(public_path),
        "model_plan_sha256": model_plan_sha256,
        "question_token_contract": QUESTION_TOKEN_CONTRACT,
        "sif_sha256": sif_sha256,
        "release_manifest_sha256": release_manifest_sha256,
        "refinement_plan_sha256": refinement_plan_sha256,
        "phase": args.phase,
    }
    resume_binding = {
        "kind": "phase2_reanchored_causal_patch_resume_binding",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "model_key": model_key,
        "rank": args.rank,
        "world_size": args.world_size,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "immutable_hashes": preload_manifest_hashes,
    }
    resume_binding_sha256 = sha256_json(resume_binding)
    resume_artifacts_exist = results_path.exists() or complete_path.exists()
    if (
        not args.canary
        and resume_artifacts_exist
        and not existing_manifest_path.exists()
    ):
        raise GateError("resume result files exist without a worker manifest")
    if not args.canary and existing_manifest_path.exists():
        existing = read_json(existing_manifest_path)
        existing_hashes = existing.get("immutable_hashes")
        preload_drift = {
            key: {
                "expected": value,
                "observed": (
                    existing_hashes.get(key)
                    if isinstance(existing_hashes, Mapping)
                    else None
                ),
            }
            for key, value in preload_manifest_hashes.items()
            if not isinstance(existing_hashes, Mapping)
            or existing_hashes.get(key) != value
        }
        identity_drift = {
            key: {"expected": value, "observed": existing.get(key)}
            for key, value in {
                "kind": "phase2_reanchored_causal_patch_worker_manifest",
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "model_key": model_key,
                "rank": args.rank,
                "world_size": args.world_size,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "resume_binding": resume_binding,
                "resume_binding_sha256": resume_binding_sha256,
            }.items()
            if existing.get(key) != value
        }
        if preload_drift or identity_drift:
            raise GateError(
                "resume worker manifest differs from current immutable inputs before "
                f"model load: hashes={preload_drift} identity={identity_drift}"
            )
    resume = {
        "completed_request_ids": [],
        "kept_rows": 0,
        "dropped_incomplete_or_invalid_rows": 0,
        "dropped_duplicate_rows": 0,
    }
    if not args.canary:
        resume = sanitize_resume_files(
            results_path,
            complete_path,
            expected_resume_binding_sha256=resume_binding_sha256,
            expected_model_key=model_key,
            expected_rank=args.rank,
            expected_phase=args.phase,
            expected_request_ids={str(row["request_id"]) for row in selected_rows},
        )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    runtime, model, processor, tokenizer, device, provenance = _load_anchor_model(
        model_key=model_key,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    load_elapsed = time.perf_counter() - load_started
    sites, adapter_receipt = architecture_sites(
        model,
        model_key=model_key,
        phase=args.phase,
        refinement_layers=refinement_layers,
    )
    if args.canary:
        receipt = run_technical_canary(
            runtime=runtime,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            model_key=model_key,
            rank=args.rank,
            request=selected_rows[0],
            public_map=public_map,
            sites=sites,
            data_dir=data_dir,
            candidate_completions=prespec["payload"]["candidate_completions"],
        )
        receipt.update(
            {
                "phase": args.phase,
                "load_elapsed_s": load_elapsed,
                "architecture": adapter_receipt,
                "layer_crosswalk": layer_crosswalk(model_key),
                "governing_prespec_sha256": PRESPEC_SHA256,
                "code_snapshot_sha256": code_snapshot_sha256,
                "mechanism_manifest_sha256": mechanism_receipt["manifest_sha256"],
                "public_manifest_sha256": sha256_file(public_path),
                "model_plan_sha256": model_plan_sha256,
                "sif_path": str(os.environ.get("SIF_PATH") or ""),
                "sif_sha256": sif_sha256,
                "release_manifest_sha256": release_manifest_sha256,
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                "fixed_microbatch_size": 1,
            }
        )
        atomic_write_json(out_dir / "canary_summary.json", receipt)
        return 0

    unrelated = select_unrelated_donor_pairs(public_rows)
    manifest_hashes = {
        **preload_manifest_hashes,
        "architecture_receipt_sha256": sha256_json(adapter_receipt),
    }
    if existing_manifest_path.exists():
        existing = read_json(existing_manifest_path)
        if existing.get("immutable_hashes") != manifest_hashes:
            raise GateError("resume manifest differs from current immutable inputs")
    completed = set(resume["completed_request_ids"])
    resumed_request_count = len(completed)
    worker_manifest = {
        "kind": "phase2_reanchored_causal_patch_worker_manifest",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "model_key": model_key,
        "rank": args.rank,
        "world_size": args.world_size,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "resume_binding": resume_binding,
        "resume_binding_sha256": resume_binding_sha256,
        "selected_trial_count": len(selected_rows),
        "selected_frame_count": len({str(row["frame_unit_id"]) for row in selected_rows}),
        "immutable_hashes": manifest_hashes,
        "architecture": adapter_receipt,
        "model_provenance": provenance,
        "fixed_microbatch_size": 1,
        "hidden_index_opened": False,
        "resume_sanitization": resume,
        "created_at": utc_now(),
    }
    atomic_write_json(existing_manifest_path, worker_manifest)
    started = time.perf_counter()
    trial_receipts: list[dict[str, Any]] = []
    rows_written = resume["kept_rows"]
    errors = 0
    for trial_index, request in enumerate(selected_rows):
        request_id = str(request["request_id"])
        if request_id in completed:
            continue
        if args.stop_file and args.stop_file.expanduser().resolve().exists():
            break
        try:
            outcomes, timing = evaluate_full_trial(
                runtime=runtime,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                device=device,
                model_key=model_key,
                rank=args.rank,
                phase=args.phase,
                request=request,
                public_map=public_map,
                unrelated_map=unrelated,
                sites=sites,
                data_dir=data_dir,
                candidate_completions=prespec["payload"]["candidate_completions"],
            )
            append_jsonl_fsync(results_path, outcomes)
            append_jsonl_fsync(
                complete_path,
                [
                    {
                        "request_id": request_id,
                        "result_row_count": len(outcomes),
                        "result_batch_sha256": sha256_json(outcomes),
                        "output_schema_version": OUTPUT_SCHEMA_VERSION,
                        "resume_binding_sha256": resume_binding_sha256,
                        "model_key": model_key,
                        "rank": args.rank,
                        "phase": args.phase,
                        "completed_at": utc_now(),
                    }
                ],
            )
            rows_written += len(outcomes)
            completed.add(request_id)
            trial_receipts.append(timing)
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
            if torch.cuda.is_available() and trial_index % 8 == 7:
                torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        atomic_write_json(
            out_dir / "progress.json",
            {
                "status": "running",
                "model_key": model_key,
                "rank": args.rank,
                "phase": args.phase,
                "trials_completed": len(completed),
                "trials_assigned": len(selected_rows),
                "rows_written": rows_written,
                "errors": errors,
                "elapsed_s": elapsed,
                "updated_at": utc_now(),
            },
        )
    elapsed = time.perf_counter() - started
    total_forwards = sum(int(row.get("patch_forward_count") or 0) for row in trial_receipts)
    peak_source_rows = combined_source_activation_peak(
        trial_receipts,
        resumed_kept_rows=int(resume["kept_rows"]),
    )
    if peak_source_rows > 1:
        raise GateError("source activation cache exceeded one row")
    stage_timings: Counter[str] = Counter()
    for row in trial_receipts:
        stage_timings.update(
            {key: float(value) for key, value in row.get("timings_s", {}).items()}
        )
    question_token_audits = [
        row.get("question_token_audit") for row in trial_receipts
    ]
    expected_new_question_audits = len(selected_rows) - resumed_request_count
    question_token_gate = {
        "status": "pass"
        if (
            len(question_token_audits) == expected_new_question_audits
            and all(
                isinstance(audit, Mapping)
                and audit.get("status") == "pass"
                and audit.get("contract") == QUESTION_TOKEN_CONTRACT
                for audit in question_token_audits
            )
        )
        else "fail",
        "contract": QUESTION_TOKEN_CONTRACT,
        "new_request_audit_count": len(question_token_audits),
        "expected_new_request_audit_count": expected_new_question_audits,
        "resumed_request_count": resumed_request_count,
        "resume_rows_are_bound_by_worker_manifest": True,
    }
    representable_intervention_gate = result_representability_gate(
        read_jsonl(results_path)
    )
    summary = {
        "kind": "phase2_reanchored_causal_patch_worker_summary",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "pass"
        if (
            len(completed) == len(selected_rows)
            and errors == 0
            and question_token_gate["status"] == "pass"
            and representable_intervention_gate["status"] == "pass"
        )
        else "partial",
        "code_snapshot_sha256": code_snapshot_sha256,
        "governing_prespec_sha256": PRESPEC_SHA256,
        "mechanism_manifest_sha256": mechanism_receipt["manifest_sha256"],
        "public_manifest_sha256": sha256_file(public_path),
        "model_plan_sha256": model_plan_sha256,
        "refinement_plan_sha256": manifest_hashes["refinement_plan_sha256"],
        "sif_path": str(os.environ.get("SIF_PATH") or ""),
        "sif_sha256": sif_sha256,
        "release_manifest_sha256": release_manifest_sha256,
        "model_key": model_key,
        "rank": args.rank,
        "phase": args.phase,
        "trials_completed": len(completed),
        "trials_assigned": len(selected_rows),
        "rows_written": rows_written,
        "errors": errors,
        "load_elapsed_s": load_elapsed,
        "run_elapsed_s": elapsed,
        "patch_forward_count": total_forwards,
        "patch_forwards_per_s": total_forwards / elapsed if elapsed > 0 else None,
        "stage_timings_s": dict(stage_timings),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        "fixed_microbatch_size": 1,
        "one_trial_activation_cache": True,
        "question_token_gate": question_token_gate,
        "representable_intervention_gate": representable_intervention_gate,
        "peak_simultaneous_source_activation_rows": peak_source_rows,
        "source_activation_tensors_persisted": False,
        "architecture": adapter_receipt,
        "created_at": utc_now(),
    }
    atomic_write_json(out_dir / "worker_summary.json", summary)
    atomic_write_json(out_dir / "progress.json", summary)
    return 0 if summary["status"] == "pass" else 2


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def telemetry_summary(path: Path) -> dict[str, Any]:
    import csv

    if not path.exists():
        return {"status": "missing", "path": str(path)}
    by_gpu: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_timestamp: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gpu = str(row.get("index") or row.get("gpu") or "unknown").strip()
            timestamp = str(row.get("timestamp") or "").strip()
            parsed: dict[str, float] = {}
            for field, aliases in {
                "utilization_gpu_pct": ("utilization.gpu [%]", "utilization_gpu_pct"),
                "memory_used_mib": ("memory.used [MiB]", "memory_used_mib"),
                "power_w": ("power.draw [W]", "power_w"),
            }.items():
                raw = next((row.get(alias) for alias in aliases if row.get(alias) is not None), None)
                if raw is None:
                    continue
                match = re.search(r"[-+]?\d+(?:\.\d+)?", str(raw))
                if match:
                    value = float(match.group(0))
                    by_gpu[gpu][field].append(value)
                    parsed[field] = value
            if timestamp and parsed:
                by_timestamp[timestamp][gpu] = parsed
    observed_gpus = set(by_gpu)
    expected_gpus = set(EXPECTED_GPU_IDS)
    issues: list[str] = []
    if observed_gpus != expected_gpus:
        issues.append("gpu_inventory")
    for gpu in EXPECTED_GPU_IDS:
        metrics = by_gpu.get(gpu, {})
        for field in ("utilization_gpu_pct", "memory_used_mib"):
            values = metrics.get(field, [])
            if len(values) < MIN_TELEMETRY_SAMPLES_PER_GPU:
                issues.append(f"{gpu}:{field}:sample_count")
            elif max(values) <= 0.0:
                issues.append(f"{gpu}:{field}:no_activity")
    concurrent_active_samples = sum(
        1
        for sample in by_timestamp.values()
        if all(
            float(sample.get(gpu, {}).get("memory_used_mib", 0.0)) > 0.0
            for gpu in EXPECTED_GPU_IDS
        )
    )
    if concurrent_active_samples < 1:
        issues.append("no_all_gpu_concurrent_residency_sample")
    return {
        "status": "pass" if not issues else "fail",
        "path": str(path),
        "expected_gpu_ids": list(EXPECTED_GPU_IDS),
        "observed_gpu_ids": sorted(observed_gpus),
        "minimum_samples_per_gpu": MIN_TELEMETRY_SAMPLES_PER_GPU,
        "concurrent_all_gpu_residency_sample_count": concurrent_active_samples,
        "issues": issues,
        "gpus": {
            gpu: {
                field: {
                    "n": len(values),
                    "mean": sum(values) / len(values) if values else None,
                    "p05": _percentile(values, 0.05),
                    "p50": _percentile(values, 0.50),
                    "p95": _percentile(values, 0.95),
                    "max": max(values) if values else None,
                }
                for field, values in metrics.items()
            }
            for gpu, metrics in sorted(by_gpu.items())
        },
    }


def expected_coarse_site_ids(model_key: str) -> list[str]:
    layer_count = int(MODEL_ADAPTERS[model_key]["expected_decoder_layers"])
    layers = coarse_layer_indexes(layer_count)
    return [
        "vision_tower_final",
        "projector_or_merger_output",
        *(f"decoder_visual_token_residual.layer_{index:03d}" for index in layers),
        *(
            f"decoder_final_question_token_residual.layer_{index:03d}"
            for index in layers
        ),
    ]


def expected_structural_transfer_site_ids(model_key: str) -> list[str]:
    return [
        site_id
        for site_id in expected_coarse_site_ids(model_key)
        if not site_id.startswith("decoder_visual_token_residual.layer_")
    ]


def worker_receipt_contract_violations(
    payload: Mapping[str, Any],
    *,
    rank: int,
    phase: str,
    canary: bool,
) -> list[str]:
    expected_model, _, _ = model_and_shard_for_rank(rank, 4)
    violations: list[str] = []
    expected_kind = (
        "phase2_reanchored_causal_patch_evaluator_unjoined_canary"
        if canary
        else "phase2_reanchored_causal_patch_worker_summary"
    )
    for condition, reason in (
        (payload.get("kind") == expected_kind, "kind"),
        (payload.get("rank") == rank, "rank"),
        (payload.get("model_key") == expected_model, "model_key"),
        (payload.get("phase") == phase, "phase"),
        (
            payload.get("governing_prespec_sha256") == PRESPEC_SHA256,
            "governing_prespec_sha256",
        ),
        (payload.get("fixed_microbatch_size") == 1, "fixed_microbatch_size"),
        (
            payload.get("output_schema_version") == OUTPUT_SCHEMA_VERSION,
            "output_schema_version",
        ),
    ):
        if not condition:
            violations.append(reason)

    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        violations.append("architecture")
        architecture = {}
    expected_adapter = MODEL_ADAPTERS[expected_model]
    expected_layers = coarse_layer_indexes(
        int(expected_adapter["expected_decoder_layers"])
    )
    if architecture.get("model_key") != expected_model:
        violations.append("architecture.model_key")
    if architecture.get("adapter") != expected_adapter:
        violations.append("architecture.adapter")
    if architecture.get("decoder_layer_count") != int(
        expected_adapter["expected_decoder_layers"]
    ):
        violations.append("architecture.decoder_layer_count")
    sites = architecture.get("sites")
    if not isinstance(sites, list):
        violations.append("architecture.sites")
        sites = []

    support_gate = payload.get("representable_intervention_gate")
    if not isinstance(support_gate, Mapping):
        violations.append("representable_intervention_gate")
        support_gate = {}

    if canary:
        expected_site_ids = expected_coarse_site_ids(expected_model)
        observed_site_ids = [
            str(row.get("site_id") or "") for row in sites if isinstance(row, Mapping)
        ]
        if observed_site_ids != expected_site_ids:
            violations.append("architecture.coarse_site_inventory")
        if architecture.get("selected_decoder_layers_by_family") != {
            "decoder_visual_token_residual": expected_layers,
            "decoder_final_question_token_residual": expected_layers,
        }:
            violations.append("architecture.coarse_layer_inventory")
        expected_transfer_sites = len(expected_layers)
        expected_structural_site_ids = expected_structural_transfer_site_ids(
            expected_model
        )
        expected_structural_not_applicable_sites = (
            len(expected_structural_site_ids)
        )
        expected_per_case_forwards = (
            8 + 15 * len(expected_site_ids) + expected_transfer_sites
        )
        expected_total_forwards = expected_per_case_forwards * len(CHOICES)
        expected_not_applicable = (
            expected_structural_not_applicable_sites * len(CHOICES)
        )
        cases = payload.get("canary_cases")
        case_choices = (
            [str(row.get("candidate_choice") or "") for row in cases]
            if isinstance(cases, list)
            and all(isinstance(row, Mapping) for row in cases)
            else []
        )
        for condition, reason in (
            (payload.get("candidate_choices_exercised") == list(CHOICES), "candidate_choices"),
            (payload.get("canary_case_count") == len(CHOICES), "canary_case_count"),
            (case_choices == list(CHOICES), "canary_case_inventory"),
            (payload.get("label_joined") is False, "label_joined"),
            (payload.get("outcome_metrics_emitted") is False, "outcome_metrics_emitted"),
            (
                payload.get("bidirectional_interventions_exercised") is True,
                "bidirectional_interventions_exercised",
            ),
            (
                payload.get("expected_canary_forward_count")
                == expected_total_forwards,
                "expected_canary_forward_count",
            ),
            (payload.get("forward_count") == expected_total_forwards, "forward_count"),
            (
                support_gate.get("not_applicable_count") == expected_not_applicable,
                "not_applicable_count",
            ),
            (
                payload.get("full_trials_per_rank") == MECHANISM_TRIAL_COUNT // 2,
                "full_trials_per_rank",
            ),
            (
                payload.get("peak_simultaneous_source_activation_rows") == 1,
                "peak_simultaneous_source_activation_rows",
            ),
        ):
            if not condition:
                violations.append(reason)
        if REQUIRED_CANARY_ANSWER_TOKEN_EQUIVALENCE:
            equivalence = payload.get("answer_token_equivalence_gate")
            if not isinstance(equivalence, Mapping):
                violations.append("answer_token_equivalence_gate")
            else:
                tolerance = equivalence.get("tolerance_bits")
                maximum = equivalence.get("maximum_absolute_difference_bits")
                choices = equivalence.get("choices")
                for condition, reason in (
                    (
                        equivalence.get("status") == "pass",
                        "answer_token_equivalence_gate.status",
                    ),
                    (
                        equivalence.get("construction")
                        == "one_shared_prefix_forward_vs_exact_appended_suffix",
                        "answer_token_equivalence_gate.construction",
                    ),
                    (
                        isinstance(tolerance, (int, float))
                        and 0.0 < float(tolerance)
                        <= CANARY_ANSWER_TOKEN_EQUIVALENCE_TOLERANCE_BITS,
                        "answer_token_equivalence_gate.tolerance_bits",
                    ),
                    (
                        isinstance(maximum, (int, float))
                        and math.isfinite(float(maximum))
                        and isinstance(tolerance, (int, float))
                        and float(maximum) <= float(tolerance),
                        "answer_token_equivalence_gate.maximum_absolute_difference_bits",
                    ),
                    (
                        isinstance(choices, Mapping)
                        and set(choices) == set(CHOICES),
                        "answer_token_equivalence_gate.choices",
                    ),
                    (
                        equivalence.get("forward_count") == 1 + len(CHOICES),
                        "answer_token_equivalence_gate.forward_count",
                    ),
                    (
                        equivalence.get("label_joined") is False,
                        "answer_token_equivalence_gate.label_joined",
                    ),
                ):
                    if not condition:
                        violations.append(reason)
                if isinstance(choices, Mapping):
                    for choice in CHOICES:
                        row = choices.get(choice)
                        if not isinstance(row, Mapping):
                            violations.append(
                                f"answer_token_equivalence_gate.choices.{choice}"
                            )
                            continue
                        row_delta = row.get("absolute_difference_bits")
                        token_ids = row.get("answer_token_ids")
                        if not (
                            isinstance(row_delta, (int, float))
                            and math.isfinite(float(row_delta))
                            and isinstance(tolerance, (int, float))
                            and float(row_delta) <= float(tolerance)
                        ):
                            violations.append(
                                "answer_token_equivalence_gate."
                                f"choices.{choice}.absolute_difference_bits"
                            )
                        if not (
                            isinstance(token_ids, list)
                            and len(token_ids) == 1
                            and type(token_ids[0]) is int
                            and token_ids[0] >= 0
                        ):
                            violations.append(
                                f"answer_token_equivalence_gate.choices.{choice}.answer_token_ids"
                            )
        if isinstance(cases, list):
            for case_index, case in enumerate(cases):
                if not isinstance(case, Mapping):
                    continue
                case_gate = case.get("representable_intervention_gate")
                structural_gate = (
                    case_gate.get("structural_not_applicable_gate")
                    if isinstance(case_gate, Mapping)
                    else None
                )
                structural_violations = structural_transfer_gate_violations(
                    structural_gate,
                    expected_site_ids=expected_structural_site_ids,
                    expected_choices=(str(case.get("candidate_choice") or ""),),
                )
                if (
                    case.get("status") != "pass"
                    or case.get("expected_canary_forward_count")
                    != expected_per_case_forwards
                    or case.get("forward_count") != expected_per_case_forwards
                    or not isinstance(case_gate, Mapping)
                    or case_gate.get("unsupported_count") != 0
                    or case_gate.get("not_applicable_count")
                    != expected_structural_not_applicable_sites
                    or structural_violations
                ):
                    violations.append(f"canary_case_{case_index}")
                if structural_violations:
                    violations.extend(
                        f"canary_case_{case_index}.structural.{reason}"
                        for reason in structural_violations
                    )
    else:
        expected_trials = MECHANISM_TRIAL_COUNT // 2
        for condition, reason in (
            (payload.get("trials_assigned") == expected_trials, "trials_assigned"),
            (payload.get("trials_completed") == expected_trials, "trials_completed"),
            (payload.get("errors") == 0, "errors"),
            (payload.get("one_trial_activation_cache") is True, "one_trial_activation_cache"),
            (
                payload.get("peak_simultaneous_source_activation_rows") == 1,
                "peak_simultaneous_source_activation_rows",
            ),
            (
                payload.get("source_activation_tensors_persisted") is False,
                "source_activation_tensors_persisted",
            ),
        ):
            if not condition:
                violations.append(reason)
    return violations


def aggregate_command(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    verify_governing_prespec(args.repo_dir.expanduser().resolve())
    if args.canary and args.coarse_only:
        raise GateError("aggregate cannot be both canary and coarse-only")
    phases = ["coarse"] if args.canary or args.coarse_only else ["coarse", "refine"]
    worker_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    worker_contract_violations: list[dict[str, Any]] = []
    for rank in range(4):
        for phase in phases:
            base = run_dir / f"rank_{rank:03d}" / phase
            name = "canary_summary.json" if args.canary else "worker_summary.json"
            path = base / name
            if not path.exists():
                failures.append(f"missing:{path}")
                continue
            payload = read_json(path)
            worker_rows.append(payload)
            contract_violations = worker_receipt_contract_violations(
                payload,
                rank=rank,
                phase=phase,
                canary=bool(args.canary),
            )
            if contract_violations:
                failures.append(f"worker_receipt_contract:{rank}:{phase}")
                worker_contract_violations.append(
                    {
                        "rank": rank,
                        "phase": phase,
                        "violations": contract_violations,
                    }
                )
            if payload.get("status") != "pass":
                failures.append(f"worker_not_pass:{rank}:{phase}:{payload.get('status')}")
            support_gate = payload.get("representable_intervention_gate")
            unsupported_count = (
                support_gate.get("unsupported_count")
                if isinstance(support_gate, Mapping)
                else None
            )
            not_applicable_count = (
                support_gate.get("not_applicable_count")
                if isinstance(support_gate, Mapping)
                else None
            )
            invalid_not_applicable_count = (
                support_gate.get("invalid_not_applicable_count")
                if isinstance(support_gate, Mapping)
                else None
            )
            if (
                not isinstance(support_gate, Mapping)
                or support_gate.get("status") != "pass"
                or type(unsupported_count) is not int
                or unsupported_count != 0
                or type(not_applicable_count) is not int
                or not_applicable_count < 0
                or type(invalid_not_applicable_count) is not int
                or invalid_not_applicable_count != 0
            ):
                failures.append(f"worker_representable_intervention_gate:{rank}:{phase}")
    telemetry = telemetry_summary(args.telemetry_csv.expanduser().resolve()) if args.telemetry_csv else {"status": "not_requested"}
    if telemetry.get("status") != "pass":
        failures.append(f"telemetry_{telemetry.get('status')}")
    code_hashes = {
        str(row.get("code_snapshot_sha256") or "") for row in worker_rows
    }
    if len(code_hashes) != 1 or not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in code_hashes):
        failures.append("worker_code_snapshot_hash_mismatch")
    binding_fields = (
        "mechanism_manifest_sha256",
        "public_manifest_sha256",
        "model_plan_sha256",
        "sif_path",
        "sif_sha256",
        "release_manifest_sha256",
    )
    binding: dict[str, str | None] = {}
    for field in binding_fields:
        values = {str(row.get(field) or "") for row in worker_rows}
        valid = len(values) == 1 and next(iter(values), "") != ""
        if field.endswith("sha256"):
            valid = valid and bool(re.fullmatch(r"[0-9a-f]{64}", next(iter(values), "")))
        if not valid:
            failures.append(f"worker_{field}_binding_mismatch")
            binding[field] = None
        else:
            binding[field] = next(iter(values))
    refinement_rows = [row for row in worker_rows if row.get("phase") == "refine"]
    if args.canary or args.coarse_only:
        binding["refinement_plan_sha256"] = None
    else:
        refinement_hashes = {
            str(row.get("refinement_plan_sha256") or "")
            for row in refinement_rows
        }
        valid_refinement_binding = (
            len(refinement_rows) == 4
            and len(refinement_hashes) == 1
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{64}", next(iter(refinement_hashes), "")
                )
            )
        )
        if not valid_refinement_binding:
            failures.append("worker_refinement_plan_sha256_binding_mismatch")
            binding["refinement_plan_sha256"] = None
        else:
            binding["refinement_plan_sha256"] = next(iter(refinement_hashes))
    for index, row in enumerate(worker_rows):
        if args.canary:
            audit = row.get("question_token_audit")
            valid_question_gate = (
                isinstance(audit, Mapping)
                and audit.get("status") == "pass"
                and audit.get("contract") == QUESTION_TOKEN_CONTRACT
            )
        else:
            gate = row.get("question_token_gate")
            valid_question_gate = (
                isinstance(gate, Mapping)
                and gate.get("status") == "pass"
                and gate.get("contract") == QUESTION_TOKEN_CONTRACT
            )
        if not valid_question_gate:
            failures.append(f"worker_question_token_gate:{index}")
    payload = {
        "kind": "phase2_reanchored_causal_patch_aggregate",
        "status": "pass" if not failures else "fail",
        "mode": "canary" if args.canary else "coarse_gate" if args.coarse_only else "full",
        "governing_prespec_sha256": PRESPEC_SHA256,
        "workers": worker_rows,
        "failures": failures,
        "worker_contract_violations": worker_contract_violations,
        "telemetry": telemetry,
        "code_snapshot_sha256": next(iter(code_hashes)) if len(code_hashes) == 1 else None,
        "immutable_input_bindings": binding,
        "total_patch_forwards": sum(int(row.get("patch_forward_count") or row.get("forward_count") or 0) for row in worker_rows),
        "unsupported_intervention_count": sum(
            count
            for row in worker_rows
            for count in [
                (row.get("representable_intervention_gate") or {}).get(
                    "unsupported_count", 0
                )
            ]
            if type(count) is int and count >= 0
        ),
        "structural_not_applicable_intervention_count": sum(
            count
            for row in worker_rows
            for count in [
                (row.get("representable_intervention_gate") or {}).get(
                    "not_applicable_count", 0
                )
            ]
            if type(count) is int and count >= 0
        ),
        "peak_cuda_memory_bytes": max([int(row.get("peak_cuda_memory_bytes") or 0) for row in worker_rows] or [0]),
        "projected_full_inference_hours_per_rank_upper": max(
            [
                float(row["projected_full_inference_hours_per_rank_upper"])
                for row in worker_rows
                if isinstance(
                    row.get("projected_full_inference_hours_per_rank_upper"),
                    (int, float),
                )
            ]
            or [0.0]
        ),
        "created_at": utc_now(),
    }
    output_name = "coarse_aggregate_summary.json" if args.coarse_only else "aggregate_summary.json"
    output_path = run_dir / output_name
    atomic_write_json(output_path, payload)
    if args.canary:
        max_projection = float(payload["projected_full_inference_hours_per_rank_upper"]) * 3600.0
        max_load = max([float(row.get("load_elapsed_s") or 0.0) for row in worker_rows] or [0.0])
        fixed_overhead_s = 1800.0
        headroom_fraction = 0.25
        projected_total_s = max_projection + 2.0 * max_load + fixed_overhead_s
        recommended_s = int(math.ceil((projected_total_s * (1.0 + headroom_fraction)) / 300.0) * 300)
        recommended_s = max(1800, recommended_s)
        chunk_limit_s = 2 * 3600 + 55 * 60

        def hhmmss(seconds: int) -> str:
            return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

        profile = {
            "kind": "phase2_reanchored_causal_patch_runtime_profile",
            "status": payload["status"],
            "governing_prespec_sha256": PRESPEC_SHA256,
            "canary_aggregate_path": str(output_path),
            "canary_aggregate_sha256": sha256_file(output_path),
            "code_snapshot_sha256": payload.get("code_snapshot_sha256"),
            **binding,
            "projection_formula_version": PROJECTION_FORMULA_VERSION,
            "max_projected_bidirectional_inference_seconds_per_rank": max_projection,
            "max_observed_model_load_seconds": max_load,
            "full_model_load_count_per_rank": 2,
            "fixed_analysis_and_copyback_overhead_seconds": fixed_overhead_s,
            "headroom_fraction": headroom_fraction,
            "unclipped_recommended_walltime_seconds": recommended_s,
            "unclipped_recommended_walltime": hhmmss(recommended_s),
            "resumable_chunk_limit_seconds": chunk_limit_s,
            "projection_fits_single_chunk": recommended_s <= chunk_limit_s,
            "submission_walltime_seconds": min(recommended_s, chunk_limit_s),
            "submission_walltime": hhmmss(min(recommended_s, chunk_limit_s)),
            "resume_required_if_not_single_chunk": recommended_s > chunk_limit_s,
            "created_at": utc_now(),
        }
        atomic_write_json(run_dir / "runtime_profile.json", profile)
        payload["runtime_profile"] = profile
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
        subparser.add_argument("--data-dir", type=Path, required=True)
        subparser.add_argument("--mechanism-manifest", type=Path, required=True)
        subparser.add_argument("--public-manifest", type=Path, required=True)
        subparser.add_argument("--expected-frames", type=int, default=96)

    preflight = subparsers.add_parser("preflight", help="Validate frozen inputs without loading a model")
    add_common(preflight)
    preflight.add_argument("--out", type=Path)
    preflight.set_defaults(func=preflight_command)

    worker = subparsers.add_parser("run-worker", help="Run one static GPU worker")
    add_common(worker)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--world-size", type=int, default=4)
    worker.add_argument("--phase", choices=("coarse", "refine"), required=True)
    worker.add_argument("--refinement-plan", type=Path)
    worker.add_argument("--canary", action="store_true")
    worker.add_argument("--dtype", default="bfloat16")
    worker.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    worker.add_argument("--seed", type=int, default=DEFAULT_SEED)
    worker.add_argument("--stop-file", type=Path)
    worker.set_defaults(func=worker_command)

    aggregate = subparsers.add_parser("aggregate", help="Fail-closed worker and telemetry aggregation")
    aggregate.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    aggregate.add_argument("--run-dir", type=Path, required=True)
    aggregate.add_argument("--telemetry-csv", type=Path)
    aggregate.add_argument("--canary", action="store_true")
    aggregate.add_argument("--coarse-only", action="store_true")
    aggregate.set_defaults(func=aggregate_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except (GateError, UnsupportedSite) as exc:
        traceback.print_exc()
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
