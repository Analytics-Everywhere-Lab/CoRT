#!/usr/bin/env python3
"""Analyze the four-model three-image 2AFC layer capture.

Canary analysis is strictly technical and evaluator-unjoined. Full analysis
joins the hidden A/B evaluator only after capture completeness, provenance,
tensor, and three-slot gates pass.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory import run_phase2_reanchored_layer_capture as capture


ARMS = capture.ARMS
SLOTS = capture.SLOT_NAMES
MODEL_KEYS = tuple(row[0] for row in capture.FROZEN_MODEL_IDENTITIES)
CORRUPT_ARMS = ("blank_full", "wrong_time")
STAGE_FAMILIES = (
    "vision_tower_final",
    "final_pre_projector_vision_representation",
    "projector_or_merger_output",
    "decoder_visual_token_residual",
    "decoder_final_question_token_residual",
)
SELECTIVE_STAGE_FAMILIES = (
    "vision_tower_final",
    "final_pre_projector_vision_representation",
    "projector_or_merger_output",
    "decoder_visual_token_residual",
)
CLAIM_BOUNDARY = capture.CLAIM_BOUNDARY
_FEATURE_MATRIX_CACHE: dict[tuple[Any, ...], np.ndarray] = {}


def _vision_family_for_model(model_key: str) -> str:
    return (
        "final_pre_projector_vision_representation"
        if model_key == "gemma4_12b_hf_bf16"
        else "vision_tower_final"
    )


def _stage_families_for_model(model_key: str) -> tuple[str, ...]:
    return (
        _vision_family_for_model(model_key),
        "projector_or_merger_output",
        "decoder_visual_token_residual",
        "decoder_final_question_token_residual",
    )


def _selective_stage_families_for_model(model_key: str) -> tuple[str, ...]:
    return (
        _vision_family_for_model(model_key),
        "projector_or_merger_output",
        "decoder_visual_token_residual",
    )


@dataclass(frozen=True)
class Record:
    model_key: str
    arm: str
    item_id: str
    request_id: str
    public_pair_id: str
    counterbalance_pair_index: int
    public_order_index: int
    tensor_prefix: str
    tensor_path: Path
    metadata: dict[str, Any]
    answer: str | None = None
    frame_unit_id: str | None = None
    participant_id: str | None = None
    video_id: str | None = None
    hidden_order_index: int | None = None
    margin_a_minus_b_bits: float | None = None
    correct_margin_bits: float | None = None
    predicted_choice: str | None = None
    predicted_role: str | None = None
    first_order_nuisance: tuple[float, ...] | None = None
    second_order_nuisance: tuple[float, ...] | None = None
    answer_position_nuisance: tuple[float, ...] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield row


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _capture_root(source_run: Path, model_key: str) -> Path:
    return source_run / model_key / "capture"


def _validate_capture_row(
    row: Mapping[str, Any], *, capture_root: Path, tensor_hashes: dict[Path, str]
) -> Record:
    if row.get("status") != "pass":
        raise ValueError(
            f"capture row failed closed: {row.get('item_id') or row.get('trial_id')}"
        )
    item_id = str(row.get("item_id") or row.get("trial_id") or "")
    if not item_id:
        raise ValueError("capture row lacks item_id")
    arm = str(row.get("arm") or "")
    if arm not in ARMS:
        raise ValueError(f"invalid capture arm: {arm}")
    tensor_rel = str(row.get("tensor_path") or "")
    tensor_path = (capture_root / tensor_rel).resolve()
    capture_root = capture_root.resolve()
    if tensor_path != capture_root and capture_root not in tensor_path.parents:
        raise ValueError(f"capture tensor path escapes capture root: {tensor_rel}")
    if not tensor_rel or not tensor_path.is_file():
        raise ValueError(f"capture tensor file is missing for {item_id}/{arm}")
    expected_hash = str(row.get("tensor_file_sha256") or "")
    observed_hash = tensor_hashes.get(tensor_path)
    if observed_hash is None:
        observed_hash = _sha256_file(tensor_path)
        tensor_hashes[tensor_path] = observed_hash
    if expected_hash != observed_hash:
        raise ValueError(f"capture tensor hash mismatch for {tensor_path}")
    slot_audit = row.get("image_slot_audit")
    if not isinstance(slot_audit, dict) or slot_audit.get("status") != "pass":
        raise ValueError(f"decoder three-slot audit failed for {item_id}/{arm}")
    checks = slot_audit.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError(f"decoder three-slot checks are incomplete for {item_id}/{arm}")
    stage_audits = row.get("stage_slot_audits")
    if not isinstance(stage_audits, dict) or not stage_audits:
        raise ValueError(f"stage slot audits are absent for {item_id}/{arm}")
    if any(
        not isinstance(audit, dict) or audit.get("status") != "pass"
        for audit in stage_audits.values()
    ):
        raise ValueError(f"stage slot audit failed for {item_id}/{arm}")
    decoder = row.get("decoder_capture")
    if not isinstance(decoder, dict) or decoder.get("status") != "pass":
        raise ValueError(f"decoder capture audit failed for {item_id}/{arm}")
    teacher = row.get("teacher_forced")
    if not isinstance(teacher, dict) or teacher.get("status") != "pass":
        raise ValueError(f"teacher-forced A/B scoring failed for {item_id}/{arm}")
    if teacher.get("hidden_answer_opened") is not False:
        raise ValueError("capture worker crossed the hidden-label firewall")
    margin = float(teacher.get("margin_A_minus_B_bits"))
    if not math.isfinite(margin):
        raise ValueError(f"nonfinite A-minus-B margin for {item_id}/{arm}")
    return Record(
        model_key="",
        arm=arm,
        item_id=item_id,
        request_id=str(row.get("request_id") or ""),
        public_pair_id=str(row.get("public_pair_id") or ""),
        counterbalance_pair_index=int(row.get("counterbalance_pair_index")),
        public_order_index=int(row.get("public_order_index")),
        tensor_prefix=str(row.get("tensor_prefix") or ""),
        tensor_path=tensor_path,
        metadata=dict(row),
        margin_a_minus_b_bits=margin,
    )


def load_records(source_run: Path, model_key: str) -> list[Record]:
    source_run = source_run.expanduser().resolve()
    model_root = source_run / model_key
    summary_path = model_root / "capture_summary.json"
    summary = _read_json(summary_path)
    if summary.get("overall_status") != "pass":
        raise ValueError(f"capture summary is not passing for {model_key}")
    if summary.get("governing_prespec_sha256") != capture.GOVERNING_PRESPEC_SHA256:
        raise ValueError(f"capture prespec hash drift for {model_key}")
    tensor_hashes: dict[Path, str] = {}
    shard_rows = summary.get("capture_shards")
    if shard_rows is None:
        capture_roots = [_capture_root(source_run, model_key)]
    else:
        if not isinstance(shard_rows, list) or not shard_rows:
            raise ValueError(f"capture shard manifest is empty for {model_key}")
        capture_roots = []
        seen_roots: set[Path] = set()
        resolved_model_root = model_root.resolve()
        for shard in shard_rows:
            if not isinstance(shard, Mapping):
                raise ValueError(f"capture shard manifest is malformed for {model_key}")
            capture_root = (model_root / str(shard.get("capture_root") or "")).resolve()
            if (
                capture_root == resolved_model_root
                or resolved_model_root not in capture_root.parents
                or capture_root in seen_roots
            ):
                raise ValueError(f"capture shard root is unsafe or duplicated for {model_key}")
            expected_index = (model_root / str(shard.get("capture_index") or "")).resolve()
            if expected_index != capture_root / "capture_index.jsonl":
                raise ValueError(f"capture shard index binding drifted for {model_key}")
            seen_roots.add(capture_root)
            capture_roots.append(capture_root)
    rows = []
    for capture_root in capture_roots:
        index_path = capture_root / "capture_index.jsonl"
        rows.extend(
            replace(
                _validate_capture_row(
                    row, capture_root=capture_root, tensor_hashes=tensor_hashes
                ),
                model_key=model_key,
            )
            for row in _iter_jsonl(index_path)
        )
    keys = [(row.item_id, row.arm) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate item/arm capture rows for {model_key}")
    return rows


def _tensor_keys(path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        return list(handle.keys())


def common_features(records: Sequence[Record]) -> list[str]:
    if not records:
        return []
    features: set[str] | None = None
    by_file: dict[Path, list[Record]] = defaultdict(list)
    for row in records:
        by_file[row.tensor_path].append(row)
    for path, file_rows in by_file.items():
        keys = _tensor_keys(path)
        key_set = set(keys)
        for row in file_rows:
            prefix = row.tensor_prefix + "__"
            current = {key[len(prefix) :] for key in keys if key.startswith(prefix)}
            if not current:
                raise ValueError(f"tensor prefix has no vectors: {row.tensor_prefix}")
            features = current if features is None else features & current
            expected = {
                row.tensor_prefix + "__" + feature for feature in current
            }
            if not expected.issubset(key_set):
                raise ValueError("tensor key discovery became inconsistent")
    return sorted(features or set())


def load_feature_matrix(records: Sequence[Record], feature: str) -> np.ndarray:
    from safetensors import safe_open

    if not records:
        raise ValueError("cannot load a feature for zero records")
    cache_key = (
        feature,
        tuple((str(row.tensor_path), row.tensor_prefix) for row in records),
    )
    cached = _FEATURE_MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    output: list[np.ndarray | None] = [None] * len(records)
    by_file: dict[Path, list[tuple[int, Record]]] = defaultdict(list)
    for index, row in enumerate(records):
        by_file[row.tensor_path].append((index, row))
    for path, file_rows in by_file.items():
        with safe_open(path, framework="np") as handle:
            available = set(handle.keys())
            for index, row in file_rows:
                key = (
                    row.tensor_prefix
                    + "__"
                    + capture.legacy._safe_tensor_key(feature)
                )
                if key not in available:
                    raise ValueError(f"missing tensor key {key} in {path}")
                value = np.asarray(handle.get_tensor(key), dtype=np.float32)
                if value.ndim != 1 or not np.isfinite(value).all():
                    raise ValueError(f"invalid pooled vector {key}")
                output[index] = value
    if any(value is None for value in output):
        raise RuntimeError("feature loader underfilled its output")
    shapes = {tuple(value.shape) for value in output if value is not None}
    if len(shapes) != 1:
        raise ValueError(f"feature dimension drift for {feature}: {shapes}")
    matrix = np.stack([value for value in output if value is not None], axis=0)
    _FEATURE_MATRIX_CACHE[cache_key] = matrix
    return matrix


def _hidden_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return list(_iter_jsonl(path))
    payload = _read_json(path)
    rows = (
        payload.get("answers") or payload.get("rows") or payload.get("trials")
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("hidden evaluator index must contain object rows")
    return list(rows)


def normalize_hidden(
    evaluator_path: Path, selected_item_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    selected = set(selected_item_ids)
    result: dict[str, dict[str, Any]] = {}
    for row in _hidden_rows(evaluator_path):
        item_id = str(
            row.get("request_id") or row.get("item_id") or row.get("trial_id") or ""
        )
        if item_id not in selected:
            continue
        if item_id in result:
            raise ValueError(f"duplicate hidden item ID: {item_id}")
        answer = str(row.get("correct_choice") or row.get("answer") or "")
        if answer not in {"A", "B"}:
            raise ValueError(f"hidden answer is not A/B for {item_id}")
        frame_id = str(
            row.get("frame_level_unit_id")
            or row.get("frame_unit_id")
            or row.get("counterbalance_group_id")
            or ""
        )
        participant = str(
            row.get("participant_id")
            or row.get("ego4d_participant_id")
            or row.get("source_participant_id")
            or ""
        )
        video_id = str(row.get("source_video_id") or row.get("video_id") or "")
        order_index = row.get("counterbalance_order_index")
        if not frame_id or not participant:
            raise ValueError(f"hidden identity metadata is incomplete for {item_id}")
        if order_index is not None and (
            type(order_index) is not int or order_index not in {0, 1}
        ):
            raise ValueError(f"hidden order is not 0/1 for {item_id}")
        if row.get("target_component_rule") != capture.TARGET_COMPONENT_RULE:
            raise ValueError(f"hidden component rule drift for {item_id}")
        fingerprints = row.get("target_component_fingerprints")
        if not isinstance(fingerprints, Mapping) or set(fingerprints) != {"A", "B"}:
            raise ValueError(f"hidden A/B component fingerprints missing for {item_id}")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
            for value in fingerprints.values()
        ):
            raise ValueError(f"hidden component fingerprint malformed for {item_id}")
        request_ids = row.get("request_ids")
        if request_ids is not None and not isinstance(request_ids, dict):
            raise ValueError(f"hidden request map is malformed for {item_id}")
        def component_nuisance(prefix: str) -> tuple[float, ...] | None:
            class_id = row.get(f"{prefix}_class")
            component = row.get(f"{prefix}_component")
            if not isinstance(class_id, int) or not isinstance(component, Mapping):
                return None
            bbox = component.get("component_bbox_tlbr")
            required = (
                component.get("component_area"),
                component.get("qualifying_component_count"),
                component.get("multi_component_indicator"),
            )
            if (
                not isinstance(bbox, Sequence)
                or isinstance(bbox, (str, bytes))
                or len(bbox) != 4
                or any(not isinstance(value, (int, float)) for value in bbox)
                or any(not isinstance(value, (int, float, bool)) for value in required)
            ):
                return None
            return tuple(
                float(value)
                for value in (
                    class_id,
                    required[0],
                    *bbox,
                    required[1],
                    int(bool(required[2])),
                )
            )

        first_nuisance = component_nuisance("first_order")
        second_nuisance = component_nuisance("second_order")
        if (first_nuisance is None) != (second_nuisance is None):
            raise ValueError(f"hidden nuisance fields are partially available for {item_id}")
        result[item_id] = {
            "answer": answer,
            "frame_unit_id": frame_id,
            "participant_id": participant,
            "video_id": video_id,
            "counterbalance_order_index": order_index,
            "request_ids": request_ids,
            "first_order_nuisance": first_nuisance,
            "second_order_nuisance": second_nuisance,
        }
    if set(result) != selected:
        raise ValueError("hidden evaluator does not map exactly to selected capture items")
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result.values():
        by_frame[row["frame_unit_id"]].append(row)
    if len(by_frame) * 2 != len(result):
        raise ValueError("hidden evaluator does not contain exactly two trials per frame")
    for frame_id, pair in by_frame.items():
        if len(pair) != 2:
            raise ValueError(f"hidden counterbalance group is not a pair: {frame_id}")
        if {row["answer"] for row in pair} != {"A", "B"}:
            raise ValueError(f"hidden answers do not counterbalance A/B: {frame_id}")
        observed_orders = {row["counterbalance_order_index"] for row in pair}
        if observed_orders != {None} and observed_orders != {0, 1}:
            raise ValueError(f"hidden order does not counterbalance 0/1: {frame_id}")
        if len({row["participant_id"] for row in pair}) != 1:
            raise ValueError(f"hidden participant drift inside frame: {frame_id}")
    return result


def join_hidden(
    records: Sequence[Record], hidden: Mapping[str, Mapping[str, Any]]
) -> list[Record]:
    joined = []
    for row in records:
        label = hidden[row.item_id]
        request_ids = label.get("request_ids")
        if (
            isinstance(request_ids, dict)
            and request_ids.get(row.arm) not in {None, row.request_id}
        ):
            raise ValueError(f"hidden/public request mismatch: {row.item_id}/{row.arm}")
        answer = str(label["answer"])
        raw_margin = float(row.margin_a_minus_b_bits)
        correct_margin = raw_margin if answer == "A" else -raw_margin
        predicted = "A" if raw_margin >= 0.0 else "B"
        predicted_role = (
            "first_order_interacting_object"
            if predicted == answer
            else "second_order_interacting_object"
        )
        first_nuisance = label.get("first_order_nuisance")
        second_nuisance = label.get("second_order_nuisance")
        answer_position_nuisance = None
        if first_nuisance is not None and second_nuisance is not None:
            ordered = (
                (first_nuisance, second_nuisance)
                if answer == "A"
                else (second_nuisance, first_nuisance)
            )
            answer_position_nuisance = tuple(
                float(value) for part in ordered for value in part
            )
        joined.append(
            replace(
                row,
                answer=answer,
                frame_unit_id=str(label["frame_unit_id"]),
                participant_id=str(label["participant_id"]),
                video_id=str(label.get("video_id") or ""),
                hidden_order_index=(
                    row.public_order_index
                    if label["counterbalance_order_index"] is None
                    else int(label["counterbalance_order_index"])
                ),
                correct_margin_bits=correct_margin,
                predicted_choice=predicted,
                predicted_role=predicted_role,
                first_order_nuisance=first_nuisance,
                second_order_nuisance=second_nuisance,
                answer_position_nuisance=answer_position_nuisance,
            )
        )
    return joined


def cluster_bootstrap_metric(
    values: Sequence[float] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    if vector.ndim != 1 or len(vector) != len(group_array) or len(vector) == 0:
        raise ValueError("cluster bootstrap inputs are invalid")
    unique = sorted({str(value) for value in group_array})
    cluster_means = np.asarray(
        [vector[group_array == group].mean() for group in unique], dtype=np.float64
    )
    estimate = float(cluster_means.mean())
    if n_boot <= 0:
        return {
            "estimate": estimate,
            "ci_low": None,
            "ci_high": None,
            "n_participant_clusters": len(unique),
            "bootstrap_draws": 0,
        }
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    draws = cluster_means[sampled].mean(axis=1)
    return {
        "estimate": estimate,
        "ci_low": _quantile(draws, 0.025),
        "ci_high": _quantile(draws, 0.975),
        "n_participant_clusters": len(unique),
        "bootstrap_draws": int(n_boot),
    }


def _distance_rows(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("distance matrices must be aligned two-dimensional arrays")
    rms = np.sqrt(np.mean((left - right) ** 2, axis=1))
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = 1.0 - np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return rms, cosine


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda row: (float(row[1]), row[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * float(value)))
        adjusted[key] = running
    return adjusted


def _model_plan_row(model_key: str) -> dict[str, Any]:
    for row in capture.DEFAULT_MODEL_PLAN:
        if row["model_key"] == model_key:
            return dict(row)
    raise KeyError(model_key)


def feature_inventory(
    source_run: Path, model_key: str, records: Sequence[Record]
) -> dict[str, dict[str, Any]]:
    plan = _read_json(source_run / model_key / "capture_plan.json")
    if plan.get("status") != "pass":
        raise ValueError(f"capture plan is not passing for {model_key}")
    roles = {
        capture.legacy._safe_tensor_key(str(name)): str(role)
        for name, role in plan.get("hook_roles", {}).items()
    }
    features = common_features(records)
    inventory: dict[str, dict[str, Any]] = {}
    expected_layers = int(_model_plan_row(model_key)["num_hidden_layers"]) + 1
    decoder_seen: dict[int, set[str]] = defaultdict(set)
    hook_seen: dict[str, set[str]] = defaultdict(set)
    for feature in features:
        if feature.startswith("decoder_layer_"):
            parts = feature.split("__")
            if len(parts) != 2:
                raise ValueError(f"malformed decoder feature: {feature}")
            layer = int(parts[0].removeprefix("decoder_layer_"))
            suffix = parts[1]
            if suffix == "final_question_token":
                family = "decoder_final_question_token_residual"
                slot = "final_question_token"
            elif suffix in {*SLOTS, "all_visual"}:
                family = "decoder_visual_token_residual"
                slot = suffix
            else:
                raise ValueError(f"unknown decoder pooled site: {feature}")
            decoder_seen[layer].add(slot)
            inventory[feature] = {
                "family": family,
                "layer_index": layer,
                "normalized_depth": layer / max(1, expected_layers - 1),
                "slot": slot,
                "state_name": (
                    "lm_entry_pre_block_0"
                    if layer == 0
                    else f"raw_post_block_{layer - 1}"
                ),
                "job_c_decoder_block_index": None if layer == 0 else layer - 1,
            }
            continue
        if not feature.startswith("hook__"):
            raise ValueError(f"unknown tensor feature: {feature}")
        matched = False
        for module, role in roles.items():
            prefix = f"hook__{module}__"
            if not feature.startswith(prefix):
                continue
            slot = feature[len(prefix) :]
            if slot not in {*SLOTS, "all_visual"}:
                raise ValueError(f"unknown hook slot: {feature}")
            family = (
                "vision_tower_final"
                if role == "vision_block"
                else "final_pre_projector_vision_representation"
                if role == "final_pre_projector_vision_representation"
                else "projector_or_merger_output"
                if role == "projector_or_merger"
                else None
            )
            if family is None:
                raise ValueError(f"unknown hook role: {role}")
            hook_seen[family].add(slot)
            inventory[feature] = {
                "family": family,
                "layer_index": None,
                "normalized_depth": None,
                "slot": slot,
                "module": module,
            }
            matched = True
            break
        if not matched:
            raise ValueError(f"hook feature is not bound to the capture plan: {feature}")
    if set(decoder_seen) != set(range(expected_layers)):
        raise ValueError(
            f"decoder layer coverage drift for {model_key}: "
            f"{len(decoder_seen)} versus {expected_layers}"
        )
    required_decoder = {*SLOTS, "all_visual", "final_question_token"}
    for layer, slots in decoder_seen.items():
        if slots != required_decoder:
            raise ValueError(f"decoder layer {layer} slot coverage drift: {slots}")
    required_hook = {*SLOTS, "all_visual"}
    for family in (_vision_family_for_model(model_key), "projector_or_merger_output"):
        if hook_seen.get(family) != required_hook:
            raise ValueError(f"{family} slot coverage drift: {hook_seen.get(family)}")
    return inventory


def _selection(source_run: Path) -> dict[str, Any]:
    selection = _read_json(source_run / "selection_manifest.json")
    if selection.get("governing_prespec_sha256") != capture.GOVERNING_PRESPEC_SHA256:
        raise ValueError("selection manifest prespec hash drift")
    if selection.get("selection_uses_hidden_answers_or_roles") is not False:
        raise ValueError("selection manifest crossed the hidden-label firewall")
    if selection.get("hidden_labels_opened") is not False:
        raise ValueError("selection manifest says hidden labels were opened")
    return selection


def validate_capture_matrix(
    source_run: Path, records_by_model: Mapping[str, Sequence[Record]]
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    selection = _selection(source_run)
    selected = [str(value) for value in selection.get("selected_item_ids") or []]
    expected = {(item_id, arm) for item_id in selected for arm in ARMS}
    inventories: dict[str, dict[str, dict[str, Any]]] = {}
    model_checks = {}
    for model_key in MODEL_KEYS:
        rows = list(records_by_model.get(model_key) or [])
        observed = {(row.item_id, row.arm) for row in rows}
        if observed != expected:
            missing = sorted(expected - observed)[:5]
            extra = sorted(observed - expected)[:5]
            raise ValueError(
                f"capture matrix mismatch for {model_key}: missing={missing} extra={extra}"
            )
        inventory = feature_inventory(source_run, model_key, rows)
        inventories[model_key] = inventory
        model_checks[model_key] = {
            "row_count": len(rows),
            "feature_count": len(inventory),
            "tensor_file_count": len({row.tensor_path for row in rows}),
            "slot_audit_status": "pass",
            "feature_schema_status": "pass",
        }
    return {
        "status": "pass",
        "mode": selection.get("mode"),
        "frame_count": int(selection.get("frame_count") or 0),
        "trial_count": len(selected),
        "expected_rows_per_model": len(expected),
        "models": model_checks,
    }, inventories


def _technical_summary(
    source_run: Path,
    records_by_model: Mapping[str, Sequence[Record]],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    models = []
    for model_key in MODEL_KEYS:
        summary = _read_json(source_run / model_key / "capture_summary.json")
        plan = _read_json(source_run / model_key / "capture_plan.json")
        rows = list(records_by_model[model_key])
        stage_methods = sorted(
            {
                str(audit.get("method"))
                for row in rows
                for audit in row.metadata.get("stage_slot_audits", {}).values()
                if isinstance(audit, dict)
            }
        )
        decoder_methods = sorted(
            {
                str(row.metadata["image_slot_audit"].get("method"))
                for row in rows
            }
        )
        models.append(
            {
                "model_key": model_key,
                "row_count": len(rows),
                "tensor_file_count": len({row.tensor_path for row in rows}),
                "tensor_bytes": sum(
                    path.stat().st_size for path in {row.tensor_path for row in rows}
                ),
                "load_elapsed_s": summary.get("load_elapsed_s"),
                "capture_elapsed_s": summary.get("capture_elapsed_s"),
                "rows_per_second": summary.get("rows_per_second"),
                "peak_gpu_allocated_bytes": summary.get("peak_gpu_allocated_bytes"),
                "peak_gpu_reserved_bytes": summary.get("peak_gpu_reserved_bytes"),
                "effective_microbatch_size": summary.get("effective_microbatch_size"),
                "slot_boundary_methods": decoder_methods,
                "stage_slot_methods": stage_methods,
                "unsupported_semantics": plan.get("unsupported_semantics", {}),
                "full_token_tensors_persisted": False,
            }
        )
    return {
        "kind": "phase2_reanchored_jobB_technical_evaluator_unjoined_summary",
        "created_at": _utc_now(),
        "status": "pass",
        "analysis_visibility": "technical_evaluator_unjoined",
        "hidden_evaluator_opened": False,
        "contains_effect_estimates": False,
        "contains_p_values_or_intervals": False,
        "contains_layer_rankings": False,
        "governing_prespec_sha256": capture.GOVERNING_PRESPEC_SHA256,
        "coverage": dict(coverage),
        "models": models,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _validate_public_hidden_pair_alignment(
    records: Sequence[Record], hidden: Mapping[str, Mapping[str, Any]]
) -> None:
    item_rows: dict[str, Record] = {}
    for row in records:
        item_rows.setdefault(row.item_id, row)
        if str(hidden[row.item_id]["frame_unit_id"]) != row.public_pair_id:
            raise ValueError(
                f"public/hidden frame join mismatch for {row.item_id}"
            )
    by_public_pair: dict[int, list[str]] = defaultdict(list)
    for row in item_rows.values():
        by_public_pair[row.counterbalance_pair_index].append(row.item_id)
    for pair_index, item_ids in by_public_pair.items():
        if len(item_ids) != 2:
            raise ValueError(f"public pair is not a pair: {pair_index}")
        frame_ids = {str(hidden[item_id]["frame_unit_id"]) for item_id in item_ids}
        if len(frame_ids) != 1:
            raise ValueError(f"public pair crosses hidden frame units: {pair_index}")
    participant_frames: dict[str, set[str]] = defaultdict(set)
    for item_id, label in hidden.items():
        participant_frames[str(label["participant_id"])].add(
            str(label["frame_unit_id"])
        )
    if len(participant_frames) != capture.MECHANISM_PARTICIPANT_COUNT:
        raise ValueError("hidden mechanism subset does not contain 96 participants")
    if any(len(frames) != 1 for frames in participant_frames.values()):
        raise ValueError("hidden mechanism subset is not one frame per participant")


def _stable_seed(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256(
        json.dumps([seed, *(str(part) for part in parts)], separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return int(digest[:8], 16)


def _frame_distance_values(
    records: Sequence[Record], feature: str, corrupt_arm: str
) -> dict[str, dict[str, Any]]:
    by_key = {(row.item_id, row.arm): row for row in records}
    item_ids = sorted({row.item_id for row in records})
    if any((item_id, arm) not in by_key for item_id in item_ids for arm in ARMS):
        raise ValueError("item/arm matrix is incomplete")
    real_rows = [by_key[(item_id, "real")] for item_id in item_ids]
    corrupt_rows = [by_key[(item_id, corrupt_arm)] for item_id in item_ids]
    real = load_feature_matrix(real_rows, feature)
    corrupt = load_feature_matrix(corrupt_rows, feature)
    rms, cosine = _distance_rows(real, corrupt)
    grouped: dict[str, list[tuple[Record, float, float]]] = defaultdict(list)
    for row, rms_value, cosine_value in zip(real_rows, rms, cosine):
        if row.frame_unit_id is None or row.participant_id is None:
            raise ValueError("frame distances require joined hidden identities")
        grouped[row.frame_unit_id].append(
            (row, float(rms_value), float(cosine_value))
        )
    output = {}
    for frame_id, pair in grouped.items():
        if len(pair) != 2:
            raise ValueError(f"frame does not have two repeated orders: {frame_id}")
        participants = {row.participant_id for row, _, _ in pair}
        if len(participants) != 1:
            raise ValueError(f"participant drift while collapsing frame: {frame_id}")
        output[frame_id] = {
            "participant_id": next(iter(participants)),
            "rms": float(np.mean([value for _, value, _ in pair])),
            "cosine": float(np.mean([value for _, _, value in pair])),
        }
    return output


def _participant_matrix(
    frame_ids: Sequence[str],
    frame_groups: Sequence[str],
    frame_matrix: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    frame_ids = list(frame_ids)
    groups = np.asarray(frame_groups, dtype=object)
    matrix = np.asarray(frame_matrix, dtype=np.float64)
    if matrix.shape[0] != len(frame_ids) or matrix.shape[0] != len(groups):
        raise ValueError("participant matrix inputs are misaligned")
    participants = sorted({str(value) for value in groups})
    collapsed = np.stack(
        [matrix[groups == participant].mean(axis=0) for participant in participants],
        axis=0,
    )
    return participants, collapsed


def joint_intersection_cluster_bands(
    blank_matrix: np.ndarray,
    wrong_time_matrix: np.ndarray,
    frame_groups: Sequence[str],
    *,
    n_boot: int,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    blank = np.asarray(blank_matrix, dtype=np.float64)
    wrong = np.asarray(wrong_time_matrix, dtype=np.float64)
    if blank.shape != wrong.shape or blank.ndim != 2 or blank.shape[0] == 0:
        raise ValueError("joint intersection matrices are not aligned")
    participants_blank, participant_blank = _participant_matrix(
        [str(index) for index in range(blank.shape[0])],
        frame_groups,
        blank,
    )
    participants_wrong, participant_wrong = _participant_matrix(
        [str(index) for index in range(wrong.shape[0])],
        frame_groups,
        wrong,
    )
    if participants_blank != participants_wrong:
        raise ValueError("joint intersection participant order drifted")
    blank_estimate = participant_blank.mean(axis=0)
    wrong_estimate = participant_wrong.mean(axis=0)
    estimate = np.minimum(blank_estimate, wrong_estimate)
    rng = np.random.default_rng(seed)
    if n_boot > 0:
        sampled = rng.integers(
            0,
            participant_blank.shape[0],
            size=(n_boot, participant_blank.shape[0]),
        )
        blank_draws = participant_blank[sampled].mean(axis=1)
        wrong_draws = participant_wrong[sampled].mean(axis=1)
        intersection_draws = np.minimum(blank_draws, wrong_draws)
        point_low = np.quantile(
            intersection_draws, 0.025, axis=0, method="linear"
        )
        point_high = np.quantile(
            intersection_draws, 0.975, axis=0, method="linear"
        )
        max_deviation = np.max(
            np.abs(intersection_draws - estimate[None, :]), axis=1
        )
        radius = _quantile(max_deviation, 0.95)
        simultaneous_low = estimate - radius
        simultaneous_high = estimate + radius
    else:
        point_low = np.full_like(estimate, np.nan)
        point_high = np.full_like(estimate, np.nan)
        simultaneous_low = np.full_like(estimate, np.nan)
        simultaneous_high = np.full_like(estimate, np.nan)
        radius = None
    observed = float(np.max(np.abs(estimate)))
    if n_permutations > 0:
        extreme = 0
        for _ in range(n_permutations):
            signs = rng.choice((-1.0, 1.0), size=participant_blank.shape[0])
            null_blank = (participant_blank * signs[:, None]).mean(axis=0)
            null_wrong = (participant_wrong * signs[:, None]).mean(axis=0)
            null_intersection = np.minimum(null_blank, null_wrong)
            extreme += float(np.max(np.abs(null_intersection))) >= observed - 1e-15
        p_value = (extreme + 1) / (n_permutations + 1)
    else:
        p_value = None
    return {
        "blank_full_estimate": blank_estimate,
        "wrong_time_estimate": wrong_estimate,
        "estimate": estimate,
        "pointwise_low": point_low,
        "pointwise_high": point_high,
        "simultaneous_low": simultaneous_low,
        "simultaneous_high": simultaneous_high,
        "simultaneous_radius": radius,
        "family_p_value": p_value,
        "participant_clusters": int(participant_blank.shape[0]),
        "bootstrap_draws": int(n_boot),
        "sign_flip_draws": int(n_permutations),
        "shared_cluster_draws": True,
        "arm_minimum_recomputed_inside_each_draw": True,
    }


def _slot_triplets(
    inventory: Mapping[str, Mapping[str, Any]], family: str
) -> list[tuple[str, str, str, dict[str, Any]]]:
    triplets = []
    for feature, info in inventory.items():
        if info["family"] != family or info.get("slot") != "full_frame":
            continue
        stem = feature[: -len("full_frame")]
        crop_a = stem + "crop_A"
        crop_b = stem + "crop_B"
        if crop_a not in inventory or crop_b not in inventory:
            raise ValueError(f"full-frame feature lacks exact crop counterparts: {feature}")
        for counterpart in (crop_a, crop_b):
            other = inventory[counterpart]
            if (
                other["family"] != family
                or other.get("layer_index") != info.get("layer_index")
            ):
                raise ValueError(f"slot counterpart semantic drift: {counterpart}")
        triplets.append((feature, crop_a, crop_b, dict(info)))
    return sorted(
        triplets,
        key=lambda row: (
            -1 if row[3].get("layer_index") is None else int(row[3]["layer_index"]),
            row[0],
        ),
    )


def selective_distance_contrast(
    full: float | np.ndarray,
    crop_a: float | np.ndarray,
    crop_b: float | np.ndarray,
) -> np.ndarray:
    full_values = np.asarray(full, dtype=np.float64)
    crop_a_values = np.asarray(crop_a, dtype=np.float64)
    crop_b_values = np.asarray(crop_b, dtype=np.float64)
    if full_values.shape != crop_a_values.shape or full_values.shape != crop_b_values.shape:
        raise ValueError("slot-selective distance arrays are misaligned")
    return full_values - 0.5 * (crop_a_values + crop_b_values)


def _selective_frame_values(
    records: Sequence[Record],
    full_feature: str,
    crop_a_feature: str,
    crop_b_feature: str,
    corrupt_arm: str,
) -> dict[str, dict[str, Any]]:
    maps = {
        "full": _frame_distance_values(records, full_feature, corrupt_arm),
        "crop_A": _frame_distance_values(records, crop_a_feature, corrupt_arm),
        "crop_B": _frame_distance_values(records, crop_b_feature, corrupt_arm),
    }
    frame_ids = sorted(maps["full"])
    if any(set(rows) != set(frame_ids) for rows in maps.values()):
        raise ValueError("slot-selective distances do not share frame units")
    output = {}
    for frame_id in frame_ids:
        participants = {
            str(rows[frame_id]["participant_id"]) for rows in maps.values()
        }
        if len(participants) != 1:
            raise ValueError("participant drift across slot-selective distances")
        output[frame_id] = {
            "participant_id": next(iter(participants)),
            "rms": float(
                selective_distance_contrast(
                    maps["full"][frame_id]["rms"],
                    maps["crop_A"][frame_id]["rms"],
                    maps["crop_B"][frame_id]["rms"],
                )
            ),
            "cosine": float(
                selective_distance_contrast(
                    maps["full"][frame_id]["cosine"],
                    maps["crop_A"][frame_id]["cosine"],
                    maps["crop_B"][frame_id]["cosine"],
                )
            ),
        }
    return output


def centered_cluster_max_bands(
    frame_matrix: np.ndarray,
    frame_groups: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    matrix = np.asarray(frame_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("selective divergence matrix must be nonempty and 2D")
    _, participant_values = _participant_matrix(
        [str(index) for index in range(matrix.shape[0])], frame_groups, matrix
    )
    estimate = participant_values.mean(axis=0)
    participant_sd = participant_values.std(axis=0, ddof=1)
    standardized = np.divide(
        estimate,
        participant_sd,
        out=np.full_like(estimate, np.nan),
        where=participant_sd > 0,
    )
    if n_boot <= 0:
        nan = np.full_like(estimate, np.nan)
        return {
            "estimate": estimate,
            "within_model_standardized_effect": standardized,
            "pointwise_low": nan,
            "pointwise_high": nan,
            "simultaneous_low": nan,
            "simultaneous_high": nan,
            "family_p_value": None,
            "participant_clusters": int(participant_values.shape[0]),
            "bootstrap_draws": 0,
        }
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        participant_values.shape[0],
        size=(n_boot, participant_values.shape[0]),
    )
    draws = participant_values[sampled].mean(axis=1)
    point_low = np.quantile(draws, 0.025, axis=0, method="linear")
    point_high = np.quantile(draws, 0.975, axis=0, method="linear")
    radius = _quantile(np.max(np.abs(draws - estimate[None, :]), axis=1), 0.95)
    centered = participant_values - estimate[None, :]
    null_draws = centered[sampled].mean(axis=1)
    observed = float(np.max(estimate))
    null_max = np.max(null_draws, axis=1)
    p_value = (1 + int(np.sum(null_max >= observed - 1e-15))) / (n_boot + 1)
    return {
        "estimate": estimate,
        "within_model_standardized_effect": standardized,
        "pointwise_low": point_low,
        "pointwise_high": point_high,
        "simultaneous_low": estimate - radius,
        "simultaneous_high": estimate + radius,
        "simultaneous_radius": radius,
        "family_p_value": p_value,
        "participant_clusters": int(participant_values.shape[0]),
        "bootstrap_draws": int(n_boot),
        "null_method": (
            "participant-cluster bootstrap of layerwise contrasts centered at "
            "their observed participant-equal means; one-sided max-layer statistic"
        ),
        "simultaneous_layer_control": True,
    }


def matched_arm_layer_analysis(
    source_run: Path,
    records_by_model: Mapping[str, Sequence[Record]],
    inventories: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    del permutations
    curve_rows: list[dict[str, Any]] = []
    selective_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    complete_family_p: dict[str, float] = {}
    for model_key in MODEL_KEYS:
        records = list(records_by_model[model_key])
        inventory = inventories[model_key]
        for corrupt_arm in CORRUPT_ARMS:
            # Raw nonnegative movement is diagnostic only. It has descriptive
            # participant-bootstrap uncertainty and never drives localization.
            for family in _stage_families_for_model(model_key):
                features = [
                    feature
                    for feature, info in inventory.items()
                    if info["family"] == family
                ]
                features.sort(
                    key=lambda feature: (
                        inventory[feature].get("slot") or "",
                        -1
                        if inventory[feature].get("layer_index") is None
                        else int(inventory[feature]["layer_index"]),
                        feature,
                    )
                )
                if not features:
                    raise ValueError(f"stage family is empty: {model_key}/{family}")
                frame_maps = [
                    _frame_distance_values(records, feature, corrupt_arm)
                    for feature in features
                ]
                frame_ids = sorted(frame_maps[0])
                if any(set(frame_map) != set(frame_ids) for frame_map in frame_maps):
                    raise ValueError("feature trajectories do not share frame units")
                participants = [
                    str(frame_maps[0][frame_id]["participant_id"])
                    for frame_id in frame_ids
                ]
                rms_matrix = np.asarray(
                    [
                        [frame_map[frame_id]["rms"] for frame_map in frame_maps]
                        for frame_id in frame_ids
                    ],
                    dtype=np.float64,
                )
                cosine_matrix = np.asarray(
                    [
                        [frame_map[frame_id]["cosine"] for frame_map in frame_maps]
                        for frame_id in frame_ids
                    ],
                    dtype=np.float64,
                )
                for index, feature in enumerate(features):
                    info = inventory[feature]
                    rms_stats = cluster_bootstrap_metric(
                        rms_matrix[:, index],
                        participants,
                        n_boot=bootstrap,
                        seed=_stable_seed(
                            seed, model_key, corrupt_arm, family, feature, "rms"
                        ),
                    )
                    cosine_stats = cluster_bootstrap_metric(
                        cosine_matrix[:, index],
                        participants,
                        n_boot=bootstrap,
                        seed=_stable_seed(
                            seed, model_key, corrupt_arm, family, feature, "cosine"
                        ),
                    )
                    curve_rows.append(
                        {
                            "model_key": model_key,
                            "corrupt_arm": corrupt_arm,
                            "stage_family": family,
                            "feature": feature,
                            "slot": info.get("slot"),
                            "layer_index": info.get("layer_index"),
                            "state_name": info.get("state_name"),
                            "job_c_decoder_block_index": info.get(
                                "job_c_decoder_block_index"
                            ),
                            "normalized_depth": info.get("normalized_depth"),
                            "frame_units": len(frame_ids),
                            "participant_clusters": rms_stats[
                                "n_participant_clusters"
                            ],
                            "rms_estimate": rms_stats["estimate"],
                            "rms_pointwise_ci_low": rms_stats["ci_low"],
                            "rms_pointwise_ci_high": rms_stats["ci_high"],
                            "cosine_estimate": cosine_stats["estimate"],
                            "cosine_ci_low": cosine_stats["ci_low"],
                            "cosine_ci_high": cosine_stats["ci_high"],
                            "bootstrap_draws": bootstrap,
                            "analysis_status": "descriptive_only",
                            "inferential_p_value": None,
                            "holm_adjusted_p": None,
                            "localization_call_allowed": False,
                            "reason": (
                                "RMS and cosine distances are nonnegative movement "
                                "diagnostics, not signed null contrasts"
                            ),
                            "inference_unit": (
                                "counterbalanced_trials_collapsed_to_frame_then_"
                                "participant_cluster"
                            ),
                        }
                    )

            for family in _selective_stage_families_for_model(model_key):
                triplets = _slot_triplets(inventory, family)
                if not triplets:
                    raise ValueError(f"selective stage family is empty: {model_key}/{family}")
                frame_maps = [
                    _selective_frame_values(records, full, crop_a, crop_b, corrupt_arm)
                    for full, crop_a, crop_b, _ in triplets
                ]
                frame_ids = sorted(frame_maps[0])
                if any(set(rows) != set(frame_ids) for rows in frame_maps):
                    raise ValueError("selective trajectories do not share frame units")
                participants = [
                    str(frame_maps[0][frame_id]["participant_id"])
                    for frame_id in frame_ids
                ]
                metric_results = {}
                for metric in ("rms", "cosine"):
                    matrix = np.asarray(
                        [
                            [rows[frame_id][metric] for rows in frame_maps]
                            for frame_id in frame_ids
                        ],
                        dtype=np.float64,
                    )
                    metric_results[metric] = centered_cluster_max_bands(
                        matrix,
                        participants,
                        n_boot=bootstrap,
                        seed=_stable_seed(
                            seed, model_key, corrupt_arm, family, metric
                        ),
                    )
                for metric, result in metric_results.items():
                    family_key = f"{model_key}:{corrupt_arm}:{family}:{metric}"
                    complete_family_p[family_key] = float(result["family_p_value"])
                    family_rows.append(
                        {
                            "model_key": model_key,
                            "corrupt_arm": corrupt_arm,
                            "stage_family": family,
                            "distance_metric": metric,
                            "feature_count": len(triplets),
                            "frame_units": len(frame_ids),
                            "participant_clusters": result["participant_clusters"],
                            "centered_cluster_max_layer_p": result[
                                "family_p_value"
                            ],
                            "bootstrap_draws": bootstrap,
                            "simultaneous_layer_control": True,
                            "signed_contrast": (
                                "real_corrupt_distance_full_frame_minus_mean_"
                                "real_corrupt_distance_unchanged_crop_A_crop_B"
                            ),
                            "interpretation": (
                                "full-frame-selective propagation; not failure origin"
                            ),
                            "inference_unit": (
                                "counterbalanced_orders_collapsed_within_frame_then_"
                                "participant_equal"
                            ),
                        }
                    )
                for index, (full, crop_a, crop_b, info) in enumerate(triplets):
                    rms = metric_results["rms"]
                    cosine = metric_results["cosine"]
                    selective_rows.append(
                        {
                            "model_key": model_key,
                            "corrupt_arm": corrupt_arm,
                            "stage_family": family,
                            "full_feature": full,
                            "crop_A_feature": crop_a,
                            "crop_B_feature": crop_b,
                            "capture_state_index": info.get("layer_index"),
                            "state_name": info.get("state_name"),
                            "job_c_decoder_block_index": info.get(
                                "job_c_decoder_block_index"
                            ),
                            "normalized_depth": info.get("normalized_depth"),
                            "rms_selective_estimate": float(rms["estimate"][index]),
                            "rms_pointwise_ci_low": float(rms["pointwise_low"][index]),
                            "rms_pointwise_ci_high": float(rms["pointwise_high"][index]),
                            "rms_simultaneous_ci_low": float(
                                rms["simultaneous_low"][index]
                            ),
                            "rms_simultaneous_ci_high": float(
                                rms["simultaneous_high"][index]
                            ),
                            "rms_within_model_standardized_effect": _finite_or_none(
                                rms["within_model_standardized_effect"][index]
                            ),
                            "cosine_selective_estimate": float(
                                cosine["estimate"][index]
                            ),
                            "cosine_simultaneous_ci_low": float(
                                cosine["simultaneous_low"][index]
                            ),
                            "cosine_simultaneous_ci_high": float(
                                cosine["simultaneous_high"][index]
                            ),
                            "frame_units": len(frame_ids),
                            "participant_clusters": rms["participant_clusters"],
                            "bootstrap_draws": bootstrap,
                            "cross_model_magnitude_comparison_allowed": False,
                            "interpretation": (
                                "selective propagation at this configuration, not origin"
                            ),
                        }
                    )
    adjusted = holm_adjust(complete_family_p)
    for row in family_rows:
        key = (
            f"{row['model_key']}:{row['corrupt_arm']}:{row['stage_family']}:"
            f"{row['distance_metric']}"
        )
        row["holm_adjusted_p_complete_selective_family"] = adjusted[key]
        row["selective_propagation_call"] = (
            float(row["holm_adjusted_p_complete_selective_family"]) < 0.05
        )
    return curve_rows, selective_rows, family_rows


def counterbalance_metrics(
    records_by_model: Mapping[str, Sequence[Record]],
    *,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        for arm in ARMS:
            rows = [row for row in records_by_model[model_key] if row.arm == arm]
            grouped: dict[str, list[Record]] = defaultdict(list)
            for row in rows:
                if row.frame_unit_id is None or row.participant_id is None:
                    raise ValueError("counterbalance metrics require hidden joins")
                grouped[row.frame_unit_id].append(row)
            frame_rows = []
            for frame_id, pair in sorted(grouped.items()):
                if len(pair) != 2 or {row.answer for row in pair} != {"A", "B"}:
                    raise ValueError(f"invalid counterbalance pair: {frame_id}/{arm}")
                participants = {row.participant_id for row in pair}
                if len(participants) != 1:
                    raise ValueError("participant drift in counterbalance collapse")
                frame_rows.append(
                    {
                        "participant_id": next(iter(participants)),
                        "pair_correct": float(
                            all(row.predicted_choice == row.answer for row in pair)
                        ),
                        "order_consistent": float(
                            pair[0].predicted_role is not None
                            and pair[0].predicted_role == pair[1].predicted_role
                        ),
                        "mean_correct_margin_bits": float(
                            np.mean([row.correct_margin_bits for row in pair])
                        ),
                    }
                )
            groups = [str(row["participant_id"]) for row in frame_rows]
            for metric in (
                "pair_correct",
                "order_consistent",
                "mean_correct_margin_bits",
            ):
                stats = cluster_bootstrap_metric(
                    [float(row[metric]) for row in frame_rows],
                    groups,
                    n_boot=bootstrap,
                    seed=_stable_seed(seed, model_key, arm, metric),
                )
                output.append(
                    {
                        "model_key": model_key,
                        "arm": arm,
                        "metric": metric,
                        "estimate": stats["estimate"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "frame_units": len(frame_rows),
                        "participant_clusters": stats["n_participant_clusters"],
                        "bootstrap_draws": bootstrap,
                        "analysis_unit": (
                            "frame_pair_then_participant_cluster_not_independent_trials"
                        ),
                    }
                )
    return output


def evidence_intersection_metrics(
    records_by_model: Mapping[str, Sequence[Record]],
    *,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for model_key in MODEL_KEYS:
        records = list(records_by_model[model_key])
        by_item_arm = {(row.item_id, row.arm): row for row in records}
        real_rows = [row for row in records if row.arm == "real"]
        grouped: dict[str, list[Record]] = defaultdict(list)
        for row in real_rows:
            if row.frame_unit_id is None or row.participant_id is None:
                raise ValueError("evidence intersection requires hidden frame identities")
            grouped[row.frame_unit_id].append(row)
        frame_rows = []
        for frame_id, pair in sorted(grouped.items()):
            if len(pair) != 2:
                raise ValueError(f"evidence intersection frame is not paired: {frame_id}")
            participants = {str(row.participant_id) for row in pair}
            if len(participants) != 1:
                raise ValueError("participant drift in evidence intersection")
            arm_accuracy: dict[str, float] = {}
            arm_margin: dict[str, float] = {}
            for arm in ARMS:
                arm_rows = [by_item_arm[(row.item_id, arm)] for row in pair]
                arm_accuracy[arm] = float(
                    np.mean(
                        [
                            float(row.predicted_choice == row.answer)
                            for row in arm_rows
                        ]
                    )
                )
                arm_margin[arm] = float(
                    np.mean([float(row.correct_margin_bits) for row in arm_rows])
                )
            frame_rows.append(
                {
                    "participant_id": next(iter(participants)),
                    "accuracy_blank_delta": arm_accuracy["real"]
                    - arm_accuracy["blank_full"],
                    "accuracy_wrong_delta": arm_accuracy["real"]
                    - arm_accuracy["wrong_time"],
                    "margin_blank_delta": arm_margin["real"]
                    - arm_margin["blank_full"],
                    "margin_wrong_delta": arm_margin["real"]
                    - arm_margin["wrong_time"],
                }
            )
        participants = [str(row["participant_id"]) for row in frame_rows]
        for metric, blank_key, wrong_key in (
            (
                "teacher_forced_pairwise_choice_accuracy_delta_evidence",
                "accuracy_blank_delta",
                "accuracy_wrong_delta",
            ),
            (
                "teacher_forced_correct_margin_bits_delta_evidence",
                "margin_blank_delta",
                "margin_wrong_delta",
            ),
        ):
            joint = joint_intersection_cluster_bands(
                np.asarray(
                    [[float(row[blank_key])] for row in frame_rows],
                    dtype=np.float64,
                ),
                np.asarray(
                    [[float(row[wrong_key])] for row in frame_rows],
                    dtype=np.float64,
                ),
                participants,
                n_boot=bootstrap,
                n_permutations=permutations,
                seed=_stable_seed(seed, model_key, metric),
            )
            key = f"{model_key}:{metric}"
            p_values[key] = float(joint["family_p_value"])
            output.append(
                {
                    "model_key": model_key,
                    "metric": metric,
                    "intersection_estimate": float(joint["estimate"][0]),
                    "real_minus_blank_full_estimate": float(
                        joint["blank_full_estimate"][0]
                    ),
                    "real_minus_wrong_time_estimate": float(
                        joint["wrong_time_estimate"][0]
                    ),
                    "intersection_ci_low": float(joint["pointwise_low"][0]),
                    "intersection_ci_high": float(joint["pointwise_high"][0]),
                    "sign_flip_p": joint["family_p_value"],
                    "frame_units": len(frame_rows),
                    "participant_clusters": joint["participant_clusters"],
                    "bootstrap_draws": bootstrap,
                    "permutation_draws": permutations,
                    "arm_minimum_recomputed_inside_each_shared_draw": True,
                    "counterbalanced_orders_collapsed_before_inference": True,
                    "strict_greedy_generation_outcome": False,
                    "interpretation": (
                        "Job B descriptive teacher-forced counterpart to the frozen "
                        "behavioral intersection criterion; it is not the prespecified "
                        "strict-greedy primary outcome."
                    ),
                }
            )
    for metric in {
        "teacher_forced_pairwise_choice_accuracy_delta_evidence",
        "teacher_forced_correct_margin_bits_delta_evidence",
    }:
        family = {
            key: value for key, value in p_values.items() if key.endswith(f":{metric}")
        }
        adjusted = holm_adjust(family)
        for row in output:
            if row["metric"] == metric:
                key = f"{row['model_key']}:{metric}"
                row["holm_adjusted_p_across_four_models"] = adjusted[key]
    return output


def _slot_pair_features(
    inventory: Mapping[str, Mapping[str, Any]]
) -> list[tuple[str, str, dict[str, Any]]]:
    pairs = []
    for feature, info in inventory.items():
        if info.get("slot") != "crop_A":
            continue
        counterpart = feature[: -len("crop_A")] + "crop_B"
        if counterpart not in inventory:
            raise ValueError(f"crop A feature lacks crop B counterpart: {feature}")
        pairs.append((feature, counterpart, dict(info)))
    return sorted(
        pairs,
        key=lambda row: (
            row[2]["family"],
            -1 if row[2].get("layer_index") is None else row[2]["layer_index"],
            row[0],
        ),
    )


def _role_vectors_by_frame(
    records: Sequence[Record],
    arm: str,
    feature_a: str,
    feature_b: str,
) -> dict[str, dict[str, Any]]:
    rows = sorted(
        [row for row in records if row.arm == arm],
        key=lambda row: row.item_id,
    )
    vectors_a = load_feature_matrix(rows, feature_a)
    vectors_b = load_feature_matrix(rows, feature_b)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, vector_a, vector_b in zip(rows, vectors_a, vectors_b):
        if row.answer not in {"A", "B"} or row.frame_unit_id is None:
            raise ValueError("role-oriented vectors require joined A/B labels")
        first, second = (
            (vector_a, vector_b) if row.answer == "A" else (vector_b, vector_a)
        )
        grouped[row.frame_unit_id].append(
            {
                "participant_id": row.participant_id,
                "order_index": row.hidden_order_index,
                "first": first,
                "second": second,
                "first_nuisance": row.first_order_nuisance,
                "second_nuisance": row.second_order_nuisance,
            }
        )
    output = {}
    for frame_id, pair in grouped.items():
        if len(pair) != 2 or {row["order_index"] for row in pair} != {0, 1}:
            raise ValueError(f"role vectors do not form a counterbalanced pair: {frame_id}")
        pair.sort(key=lambda row: row["order_index"])
        if len({row["participant_id"] for row in pair}) != 1:
            raise ValueError("participant drift in role-vector frame collapse")
        if any(
            pair[0][field] != pair[1][field]
            for field in ("first_nuisance", "second_nuisance")
        ):
            raise ValueError("nuisance geometry drift in role-vector frame collapse")
        first = np.stack([row["first"] for row in pair]).mean(axis=0)
        second = np.stack([row["second"] for row in pair]).mean(axis=0)
        role_rms, role_cosine = _distance_rows(first[None, :], second[None, :])
        first_order_rms, _ = _distance_rows(
            pair[0]["first"][None, :], pair[1]["first"][None, :]
        )
        second_order_rms, _ = _distance_rows(
            pair[0]["second"][None, :], pair[1]["second"][None, :]
        )
        output[frame_id] = {
            "participant_id": pair[0]["participant_id"],
            "first": first,
            "second": second,
            "first_nuisance": pair[0]["first_nuisance"],
            "second_nuisance": pair[0]["second_nuisance"],
            "role_rms": float(role_rms[0]),
            "role_cosine": float(role_cosine[0]),
            "order_stability_rms": float(
                (first_order_rms[0] + second_order_rms[0]) / 2.0
            ),
        }
    return output


def target_role_contrasts(
    records_by_model: Mapping[str, Sequence[Record]],
    inventories: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        records = records_by_model[model_key]
        for feature_a, feature_b, info in _slot_pair_features(
            inventories[model_key]
        ):
            for arm in ARMS:
                frame_map = _role_vectors_by_frame(
                    records, arm, feature_a, feature_b
                )
                frame_ids = sorted(frame_map)
                groups = [
                    str(frame_map[frame_id]["participant_id"])
                    for frame_id in frame_ids
                ]
                for metric in ("role_rms", "role_cosine", "order_stability_rms"):
                    stats = cluster_bootstrap_metric(
                        [float(frame_map[frame_id][metric]) for frame_id in frame_ids],
                        groups,
                        n_boot=bootstrap,
                        seed=_stable_seed(
                            seed, model_key, arm, feature_a, metric
                        ),
                    )
                    output.append(
                        {
                            "model_key": model_key,
                            "arm": arm,
                            "stage_family": info["family"],
                            "feature_A": feature_a,
                            "feature_B": feature_b,
                            "layer_index": info.get("layer_index"),
                            "normalized_depth": info.get("normalized_depth"),
                            "metric": metric,
                            "estimate": stats["estimate"],
                            "ci_low": stats["ci_low"],
                            "ci_high": stats["ci_high"],
                            "frame_units": len(frame_ids),
                            "participant_clusters": stats[
                                "n_participant_clusters"
                            ],
                            "bootstrap_draws": bootstrap,
                            "analysis_status": "exploratory_descriptive",
                            "binding_claim_allowed": False,
                            "nuisance_adjusted": False,
                            "contrast": (
                                "first_order_vs_second_order_role_oriented_"
                                "counterbalance_collapsed"
                            ),
                            "interpretation": (
                                "hidden-label descriptive role geometry only; raw "
                                "separation is not evidence of scene binding"
                            ),
                        }
                    )
    return output


def _nested_group_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    outer_splits: int,
    inner_splits: int,
    c_grid: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    del seed
    unique_groups = sorted({str(value) for value in groups})
    outer_count = min(int(outer_splits), len(unique_groups))
    if outer_count < 2:
        raise ValueError("grouped CV requires at least two groups")
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows = []
    outer = GroupKFold(n_splits=outer_count)
    for fold_index, (train, test) in enumerate(outer.split(x, y, groups)):
        train_groups = groups[train]
        unique_train = sorted({str(value) for value in train_groups})
        inner_count = min(int(inner_splits), len(unique_train))
        scores = []
        for c_value in c_grid:
            fold_scores = []
            if inner_count >= 2:
                inner = GroupKFold(n_splits=inner_count)
                for inner_train, inner_valid in inner.split(
                    x[train], y[train], train_groups
                ):
                    if len(set(y[train][inner_train])) < 2:
                        continue
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=float(c_value),
                            solver="liblinear",
                            max_iter=2000,
                        ),
                    )
                    model.fit(x[train][inner_train], y[train][inner_train])
                    probability = model.predict_proba(x[train][inner_valid])[:, 1]
                    if len(set(y[train][inner_valid])) == 2:
                        fold_scores.append(
                            roc_auc_score(y[train][inner_valid], probability)
                        )
            scores.append(
                (
                    float(np.mean(fold_scores)) if fold_scores else 0.5,
                    -float(c_value),
                    float(c_value),
                )
            )
        selected_c = max(scores)[2]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=selected_c,
                solver="liblinear",
                max_iter=2000,
            ),
        )
        model.fit(x[train], y[train])
        predictions[test] = model.predict_proba(x[test])[:, 1]
        train_group_set = {str(value) for value in groups[train]}
        test_group_set = {str(value) for value in groups[test]}
        if train_group_set & test_group_set:
            raise RuntimeError("participant/video leakage across grouped CV folds")
        fold_rows.append(
            {
                "fold": fold_index,
                "selected_C": selected_c,
                "train_groups": len(train_group_set),
                "test_groups": len(test_group_set),
            }
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("grouped CV predictions are incomplete")
    return predictions, fold_rows


def grouped_predictive_probe(
    x: np.ndarray,
    y: Sequence[int],
    groups: Sequence[str],
    pair_ids: Sequence[str],
    *,
    nuisance_x: np.ndarray | None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0),
    bootstrap: int = 2000,
    permutations: int = 0,
    seed: int = 20260721,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    matrix = np.asarray(x, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    group_array = np.asarray(groups, dtype=object)
    pair_array = np.asarray(pair_ids, dtype=object)
    if (
        matrix.ndim != 2
        or len(matrix) != len(labels)
        or len(labels) != len(group_array)
        or len(labels) != len(pair_array)
    ):
        raise ValueError("probe arrays are misaligned")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("probe labels must contain both 0/1 classes")
    if permutations != 0:
        raise ValueError("exploratory probes forbid sequential permutation refits")
    if nuisance_x is None:
        return {
            "status": "not_run_nuisance_baseline_unavailable",
            "nuisance_baseline_required": True,
            "activation_auroc": None,
            "nuisance_baseline_auroc": None,
            "incremental_grouped_cv_auroc": None,
            "incremental_cluster_bootstrap_p": None,
            "bootstrap_draws": 0,
            "participant_or_video_grouped_cv": True,
            "participant_leakage_free": True,
            "probe_interpretation": (
                "exploratory probe failed closed because frozen class/component-"
                "geometry nuisance fields were unavailable; raw AUROC is not emitted"
            ),
            "outer_folds": [],
        }
    nuisance = np.asarray(nuisance_x, dtype=np.float32)
    if nuisance.ndim != 2 or nuisance.shape[0] != matrix.shape[0]:
        raise ValueError("probe nuisance baseline is misaligned")
    if not np.isfinite(nuisance).all():
        raise ValueError("probe nuisance baseline contains nonfinite values")
    for pair_id in sorted({str(value) for value in pair_array}):
        pair_labels = labels[pair_array == pair_id]
        if len(pair_labels) != 2 or set(pair_labels.tolist()) != {0, 1}:
            raise ValueError(f"probe pair is not counterbalanced: {pair_id}")
    baseline_predictions, baseline_folds = _nested_group_predictions(
        nuisance,
        labels,
        group_array,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        c_grid=c_grid,
        seed=seed,
    )
    full_predictions, full_folds = _nested_group_predictions(
        np.concatenate([nuisance, matrix], axis=1),
        labels,
        group_array,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        c_grid=c_grid,
        seed=seed,
    )
    baseline_auc = float(roc_auc_score(labels, baseline_predictions))
    activation_auc = float(roc_auc_score(labels, full_predictions))
    observed = activation_auc - baseline_auc
    unique_groups = sorted({str(value) for value in group_array})
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(bootstrap)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indexes = np.concatenate(
            [np.flatnonzero(group_array == group) for group in sampled_groups]
        )
        draws.append(
            float(roc_auc_score(labels[indexes], full_predictions[indexes]))
            - float(roc_auc_score(labels[indexes], baseline_predictions[indexes]))
        )
    draw_array = np.asarray(draws, dtype=np.float64)
    centered_null = draw_array - observed
    p_value = (
        (1 + int(np.sum(centered_null >= observed - 1e-15)))
        / (len(draw_array) + 1)
        if len(draw_array)
        else None
    )
    return {
        "status": "exploratory_complete",
        "activation_auroc": activation_auc,
        "nuisance_baseline_auroc": baseline_auc,
        "incremental_grouped_cv_auroc": observed,
        "incremental_ci_low": _quantile(draw_array, 0.025) if len(draw_array) else None,
        "incremental_ci_high": _quantile(draw_array, 0.975) if len(draw_array) else None,
        "incremental_cluster_bootstrap_p": p_value,
        "bootstrap_draws": int(bootstrap),
        "permutation_count": 0,
        "participant_or_video_grouped_cv": True,
        "participant_leakage_free": True,
        "nuisance_baseline_required": True,
        "nuisance_features": (
            "ontology_class, component_area, bbox_tlbr, qualifying_component_count, "
            "multi_component_indicator"
        ),
        "outer_folds": {"baseline": baseline_folds, "activation_plus_baseline": full_folds},
        "probe_interpretation": (
            "exploratory incremental grouped-CV decodability beyond frozen class/"
            "component-geometry nuisance features; not binding and not causal use"
        ),
    }


def _representative_probe_sites(
    inventory: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    sites = []
    vision_families = {
        str(info["family"])
        for info in inventory.values()
        if info.get("family")
        in {"vision_tower_final", "final_pre_projector_vision_representation"}
    }
    if len(vision_families) != 1:
        raise ValueError(f"probe vision-family count drift: {sorted(vision_families)}")
    for family in (next(iter(vision_families)), "projector_or_merger_output"):
        crop_a = [
            feature
            for feature, info in inventory.items()
            if info["family"] == family and info.get("slot") == "crop_A"
        ]
        if len(crop_a) != 1:
            raise ValueError(f"probe site count drift for {family}")
        sites.append(
            {
                "kind": "target_role",
                "family": family,
                "feature_a": crop_a[0],
                "feature_b": crop_a[0][: -len("crop_A")] + "crop_B",
                "layer_index": None,
                "normalized_depth": None,
            }
        )
    decoder_layers = sorted(
        {
            int(info["layer_index"])
            for info in inventory.values()
            if info["family"] == "decoder_visual_token_residual"
        }
    )
    final_layer = decoder_layers[-1]
    visual_a = next(
        feature
        for feature, info in inventory.items()
        if info["family"] == "decoder_visual_token_residual"
        and info["layer_index"] == final_layer
        and info["slot"] == "crop_A"
    )
    question = next(
        feature
        for feature, info in inventory.items()
        if info["family"] == "decoder_final_question_token_residual"
        and info["layer_index"] == final_layer
    )
    sites.extend(
        [
            {
                "kind": "target_role",
                "family": "decoder_visual_token_residual",
                "feature_a": visual_a,
                "feature_b": visual_a[: -len("crop_A")] + "crop_B",
                "layer_index": final_layer,
                "normalized_depth": 1.0,
            },
            {
                "kind": "answer_position",
                "family": "decoder_final_question_token_residual",
                "feature": question,
                "layer_index": final_layer,
                "normalized_depth": 1.0,
            },
        ]
    )
    return sites


def predictive_probes(
    records_by_model: Mapping[str, Sequence[Record]],
    inventories: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    permutations: int,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for model_key in MODEL_KEYS:
        records = list(records_by_model[model_key])
        for site in _representative_probe_sites(inventories[model_key]):
            for arm in ARMS:
                if site["kind"] == "target_role":
                    frame_map = _role_vectors_by_frame(
                        records, arm, site["feature_a"], site["feature_b"]
                    )
                    frame_ids = sorted(frame_map)
                    x = np.concatenate(
                        [
                            np.stack(
                                [frame_map[frame_id]["first"] for frame_id in frame_ids]
                            ),
                            np.stack(
                                [frame_map[frame_id]["second"] for frame_id in frame_ids]
                            ),
                        ],
                        axis=0,
                    )
                    y = [1] * len(frame_ids) + [0] * len(frame_ids)
                    groups = [
                        str(frame_map[frame_id]["participant_id"])
                        for frame_id in frame_ids
                    ] * 2
                    pair_ids = frame_ids + frame_ids
                    nuisance_available = all(
                        frame_map[frame_id]["first_nuisance"] is not None
                        and frame_map[frame_id]["second_nuisance"] is not None
                        for frame_id in frame_ids
                    )
                    nuisance_x = (
                        np.asarray(
                            [frame_map[frame_id]["first_nuisance"] for frame_id in frame_ids]
                            + [frame_map[frame_id]["second_nuisance"] for frame_id in frame_ids],
                            dtype=np.float32,
                        )
                        if nuisance_available
                        else None
                    )
                    feature_name = site["feature_a"] + "::" + site["feature_b"]
                else:
                    rows = sorted(
                        [row for row in records if row.arm == arm],
                        key=lambda row: (str(row.frame_unit_id), int(row.hidden_order_index)),
                    )
                    x = load_feature_matrix(rows, site["feature"])
                    y = [1 if row.answer == "A" else 0 for row in rows]
                    groups = [
                        str(row.participant_id or row.video_id) for row in rows
                    ]
                    pair_ids = [str(row.frame_unit_id) for row in rows]
                    nuisance_available = all(
                        row.answer_position_nuisance is not None for row in rows
                    )
                    nuisance_x = (
                        np.asarray(
                            [row.answer_position_nuisance for row in rows],
                            dtype=np.float32,
                        )
                        if nuisance_available
                        else None
                    )
                    feature_name = site["feature"]
                result = grouped_predictive_probe(
                    x,
                    y,
                    groups,
                    pair_ids,
                    nuisance_x=nuisance_x,
                    bootstrap=bootstrap,
                    permutations=permutations,
                    seed=_stable_seed(
                        seed, model_key, arm, site["family"], site["kind"]
                    ),
                )
                output.append(
                    {
                        "model_key": model_key,
                        "arm": arm,
                        "stage_family": site["family"],
                        "probe_target": site["kind"],
                        "feature": feature_name,
                        "layer_index": site.get("layer_index"),
                        "normalized_depth": site.get("normalized_depth"),
                        **{
                            key: value
                            for key, value in result.items()
                            if key != "outer_folds"
                        },
                        "outer_folds_json": json.dumps(
                            result["outer_folds"], sort_keys=True
                        ),
                    }
                )
    available_p = {
        str(index): float(row["incremental_cluster_bootstrap_p"])
        for index, row in enumerate(output)
        if row.get("incremental_cluster_bootstrap_p") is not None
    }
    complete_family = len(available_p) == len(output) and bool(output)
    adjusted = holm_adjust(available_p) if complete_family else {}
    for index, row in enumerate(output):
        row["complete_probe_family_size"] = len(output)
        row["complete_probe_family_status"] = (
            "holm_corrected_exploratory"
            if complete_family
            else "fail_closed_nuisance_baseline_unavailable"
        )
        row["holm_adjusted_p_complete_probe_family"] = adjusted.get(str(index))
        row["binding_claim_allowed"] = False
    return output


def _write_paper_tables(
    out_dir: Path,
    family_rows: Sequence[Mapping[str, Any]],
    counterbalance_rows: Sequence[Mapping[str, Any]],
    intersection_rows: Sequence[Mapping[str, Any]],
) -> None:
    family_lines = [
        "# Full-frame-selective propagation tests",
        "",
        "| Model | Corruption | Stage family | Distance | Centered cluster max-layer p | Complete-family Holm p |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in family_rows:
        family_lines.append(
            "| {model_key} | {corrupt_arm} | {stage_family} | {metric} | {p:.4g} | {holm:.4g} |".format(
                model_key=row["model_key"],
                corrupt_arm=row["corrupt_arm"],
                stage_family=row["stage_family"],
                metric=row["distance_metric"],
                p=float(row["centered_cluster_max_layer_p"]),
                holm=float(row["holm_adjusted_p_complete_selective_family"]),
            )
        )
    (out_dir / "paper_table_stage_family.md").write_text(
        "\n".join(family_lines) + "\n", encoding="utf-8"
    )
    counter_lines = [
        "# Counterbalance summaries",
        "",
        "| Model | Arm | Metric | Estimate | 95% cluster CI |",
        "|---|---|---|---:|---:|",
    ]
    for row in counterbalance_rows:
        counter_lines.append(
            "| {model_key} | {arm} | {metric} | {estimate:.4f} | [{low:.4f}, {high:.4f}] |".format(
                model_key=row["model_key"],
                arm=row["arm"],
                metric=row["metric"],
                estimate=float(row["estimate"]),
                low=float(row["ci_low"]),
                high=float(row["ci_high"]),
            )
        )
    (out_dir / "paper_table_counterbalance.md").write_text(
        "\n".join(counter_lines) + "\n", encoding="utf-8"
    )
    intersection_lines = [
        "# Evidence intersection summaries",
        "",
        "| Model | Teacher-forced outcome | Joint minimum | Real-blank | Real-wrong-time | 95% cluster CI | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in intersection_rows:
        intersection_lines.append(
            "| {model_key} | {metric} | {estimate:.4f} | {blank:.4f} | {wrong:.4f} | [{low:.4f}, {high:.4f}] | {holm:.4g} |".format(
                model_key=row["model_key"],
                metric=row["metric"],
                estimate=float(row["intersection_estimate"]),
                blank=float(row["real_minus_blank_full_estimate"]),
                wrong=float(row["real_minus_wrong_time_estimate"]),
                low=float(row["intersection_ci_low"]),
                high=float(row["intersection_ci_high"]),
                holm=float(row["holm_adjusted_p_across_four_models"]),
            )
        )
    (out_dir / "paper_table_evidence_intersection.md").write_text(
        "\n".join(intersection_lines) + "\n", encoding="utf-8"
    )


def _write_plots(
    out_dir: Path, selective_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        _write_json(
            out_dir / "plot_status.json",
            {
                "kind": "phase2_reanchored_jobB_plot_status",
                "status": "not_rendered_missing_optional_dependency",
                "dependency": "matplotlib",
                "required_for_numerical_analysis": False,
                "plots": [],
            },
        )
        return []

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for corrupt_arm in CORRUPT_ARMS:
        for model_key in MODEL_KEYS:
            figure, axis = plt.subplots(1, 1, figsize=(7, 4))
            rows = sorted(
                [
                    row
                    for row in selective_rows
                    if row["model_key"] == model_key
                    and row["corrupt_arm"] == corrupt_arm
                    and row["stage_family"] == "decoder_visual_token_residual"
                    and row["rms_within_model_standardized_effect"] is not None
                ],
                key=lambda row: float(row["normalized_depth"]),
            )
            axis.plot(
                [float(row["normalized_depth"]) for row in rows],
                [float(row["rms_within_model_standardized_effect"]) for row in rows],
                label=model_key,
                linewidth=1.5,
            )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_title(f"{model_key}: {corrupt_arm}")
            axis.set_xlabel("Normalized decoder depth")
            axis.set_ylabel("Within-model standardized selective RMS contrast")
            axis.grid(alpha=0.25)
            figure.tight_layout()
            path = plot_dir / f"selective_propagation_{model_key}_{corrupt_arm}.png"
            figure.savefig(path, dpi=220)
            plt.close(figure)
            written.append(str(path.relative_to(out_dir)))
    _write_json(
        out_dir / "plot_status.json",
        {
            "kind": "phase2_reanchored_jobB_plot_status",
            "status": "pass",
            "dependency": "matplotlib",
            "required_for_numerical_analysis": False,
            "plots": written,
        },
    )
    return written


def run_analysis(args: argparse.Namespace) -> int:
    source_run = args.source_run.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    capture.assert_governing_prespec(args.prespec_path)
    records_by_model = {
        model_key: load_records(source_run, model_key) for model_key in MODEL_KEYS
    }
    coverage, inventories = validate_capture_matrix(source_run, records_by_model)
    technical = _technical_summary(source_run, records_by_model, coverage)
    _write_json(out_dir / "technical_summary.json", technical)
    selection = _selection(source_run)
    if selection.get("mode") == "canary":
        if args.evaluator_index is not None:
            raise ValueError("canary analysis forbids a hidden evaluator path")
        print(json.dumps(technical, indent=2, sort_keys=True))
        return 0
    if selection.get("mode") != "full":
        raise ValueError(f"unknown selection mode: {selection.get('mode')}")
    if args.evaluator_index is None:
        raise ValueError("full analysis requires --evaluator-index")
    if int(args.bootstrap) <= 0 or int(args.max_stat_permutations) <= 0:
        raise ValueError("full analysis requires positive confirmatory resampling counts")
    if int(args.probe_permutations) != 0:
        raise ValueError("exploratory probes forbid sequential permutation refits")
    evaluator_path = args.evaluator_index.expanduser().resolve()
    declared_hidden_hash = selection.get("source_mechanism_subset", {}).get(
        "hidden_sha256_declared_only"
    )
    if _sha256_file(evaluator_path) != declared_hidden_hash:
        raise ValueError("hidden mechanism evaluator hash does not match its manifest")
    selected_ids = [str(value) for value in selection["selected_item_ids"]]
    hidden = normalize_hidden(evaluator_path, selected_ids)
    _validate_public_hidden_pair_alignment(records_by_model[MODEL_KEYS[0]], hidden)
    joined_by_model = {
        model_key: join_hidden(records, hidden)
        for model_key, records in records_by_model.items()
    }
    curve_rows, selective_rows, family_rows = matched_arm_layer_analysis(
        source_run,
        joined_by_model,
        inventories,
        bootstrap=int(args.bootstrap),
        permutations=int(args.max_stat_permutations),
        seed=int(args.seed),
    )
    counter_rows = counterbalance_metrics(
        joined_by_model,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    intersection_rows = evidence_intersection_metrics(
        joined_by_model,
        bootstrap=int(args.bootstrap),
        permutations=int(args.max_stat_permutations),
        seed=int(args.seed),
    )
    target_rows = target_role_contrasts(
        joined_by_model,
        inventories,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    probe_rows: list[dict[str, Any]] = []
    probe_status = {
        "kind": "phase2_reanchored_jobB_predictive_probe_status",
        "status": "not_run",
        "reason": (
            "The registered exploratory probe null requires full-refit grouped "
            "counterbalance-pair permutation. This bounded TamIA job does not run "
            "or replace that null with a cheaper bootstrap test."
        ),
        "registered_null": "full_refit_grouped_counterbalance_pair_permutation",
        "substitute_inference_used": False,
        "probe_rows_emitted": 0,
    }
    _write_csv(out_dir / "matched_arm_layer_deltas.csv", curve_rows)
    _write_csv(out_dir / "full_frame_selective_divergence.csv", selective_rows)
    _write_csv(out_dir / "selective_propagation_family_tests.csv", family_rows)
    _write_csv(out_dir / "counterbalance_metrics.csv", counter_rows)
    _write_csv(out_dir / "evidence_intersection_metrics.csv", intersection_rows)
    _write_csv(out_dir / "target_role_contrasts.csv", target_rows)
    _write_json(out_dir / "predictive_probe_status.json", probe_status)
    crosswalk = {
        model_key: [
            {
                "capture_state_index": index,
                "state_name": (
                    "lm_entry_pre_block_0"
                    if index == 0
                    else f"raw_post_block_{index - 1}"
                ),
                "job_c_decoder_block_index": None if index == 0 else index - 1,
            }
            for index in range(int(_model_plan_row(model_key)["num_hidden_layers"]) + 1)
        ]
        for model_key in MODEL_KEYS
    }
    _write_json(
        out_dir / "decoder_state_crosswalk.json",
        {
            "kind": "jobB_jobC_decoder_state_crosswalk",
            "semantics": (
                "Job B capture state 0 is the direct-hook lm_entry before block 0; "
                "capture state i+1 is the raw direct-hook post_block_i output and "
                "corresponds to Job C decoder block i; final normalization is excluded"
            ),
            "capture_source": "direct_decoder_module_hooks",
            "models": crosswalk,
        },
    )
    _write_paper_tables(out_dir, family_rows, counter_rows, intersection_rows)
    if args.skip_plots:
        plots: list[str] = []
        _write_json(
            out_dir / "plot_status.json",
            {
                "kind": "phase2_reanchored_jobB_plot_status",
                "status": "not_rendered_by_explicit_request",
                "dependency": "matplotlib",
                "required_for_numerical_analysis": False,
                "plots": [],
            },
        )
    else:
        plots = _write_plots(out_dir, selective_rows)
    summary = {
        "kind": "phase2_reanchored_jobB_full_layerwise_analysis",
        "created_at": _utc_now(),
        "status": "pass",
        "governing_prespec_sha256": capture.GOVERNING_PRESPEC_SHA256,
        "target_component_rule": capture.TARGET_COMPONENT_RULE,
        "selection_manifest_sha256": _sha256_file(
            source_run / "selection_manifest.json"
        ),
        "hidden_evaluator_sha256": _sha256_file(
            evaluator_path
        ),
        "frame_units": len({row["frame_unit_id"] for row in hidden.values()}),
        "participant_clusters": len(
            {row["participant_id"] for row in hidden.values()}
        ),
        "counterbalanced_trials_are_repeated_not_independent": True,
        "evidence_intersection_criterion": (
            "minimum of real-minus-blank_full and real-minus-wrong_time is "
            "recomputed jointly inside every shared participant-cluster draw"
        ),
        "matched_arm_layer_rows": len(curve_rows),
        "raw_distance_claim": "descriptive_only_no_p_values_no_localization_calls",
        "selective_divergence_rows": len(selective_rows),
        "selective_family_test_rows": len(family_rows),
        "selective_propagation_claim": (
            "signed full-slot distance minus mean unchanged-crop-slot distance; "
            "localizes selective propagation, not failure origin"
        ),
        "cross_model_activation_magnitude_comparison_allowed": False,
        "cross_model_comparison_scope": "normalized_depth_and_configuration_only",
        "counterbalance_metric_rows": len(counter_rows),
        "evidence_intersection_metric_rows": len(intersection_rows),
        "target_role_contrast_rows": len(target_rows),
        "predictive_probe_rows": len(probe_rows),
        "predictive_probe_family_status": probe_status["status"],
        "bootstrap_draws": int(args.bootstrap),
        "max_stat_permutations": int(args.max_stat_permutations),
        "probe_permutations": 0,
        "probe_bootstrap_draws": 0,
        "probe_claim": (
            "No predictive-probe result is emitted. The registered full-refit grouped "
            "counterbalance-pair permutation null was not run, and no cheaper "
            "substitute inference is used."
        ),
        "job_c_slot_contract": {
            "evidence_restoration": (
                "modify slot 0 full-frame only; crop A/B tensors and positions must "
                "remain exactly equal to the recipient"
            ),
            "counterbalanced_crop_order": (
                "map crop A/B under the exact permutation; never modify full-frame "
                "tokens; audit all unpatched positions for exact equality"
            ),
            "coarse_decoder_fractions": [
                index / 8 for index in range(9)
            ],
            "refinement": (
                "plus or minus two layers is descriptive unless selection is replayed "
                "inside every inferential resample"
            ),
        },
        "outputs": {
            "technical_summary": "technical_summary.json",
            "matched_arm_layer_deltas": "matched_arm_layer_deltas.csv",
            "full_frame_selective_divergence": "full_frame_selective_divergence.csv",
            "selective_propagation_family_tests": (
                "selective_propagation_family_tests.csv"
            ),
            "decoder_state_crosswalk": "decoder_state_crosswalk.json",
            "counterbalance_metrics": "counterbalance_metrics.csv",
            "evidence_intersection_metrics": "evidence_intersection_metrics.csv",
            "target_role_contrasts": "target_role_contrasts.csv",
            "predictive_probe_status": "predictive_probe_status.json",
            "plot_status": "plot_status.json",
            "paper_stage_table": "paper_table_stage_family.md",
            "paper_counterbalance_table": "paper_table_counterbalance.md",
            "paper_evidence_intersection_table": (
                "paper_table_evidence_intersection.md"
            ),
            "plots": plots,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(out_dir / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prespec-path", type=Path)
    parser.add_argument("--evaluator-index", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--max-stat-permutations", type=int, default=10000)
    parser.add_argument("--probe-permutations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help=(
            "Skip optional PNG rendering while preserving all registered numerical "
            "outputs and paper tables."
        ),
    )
    args = parser.parse_args(argv)
    if min(
        int(args.bootstrap),
        int(args.max_stat_permutations),
        int(args.probe_permutations),
    ) < 0:
        parser.error("resampling counts must be nonnegative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run_analysis(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
