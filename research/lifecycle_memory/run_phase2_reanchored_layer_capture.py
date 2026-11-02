#!/usr/bin/env python3
"""Run the prespecified four-model layerwise 2AFC capture.

Workers consume public three-image packs only. Hidden A/B answers are joined
later by the full-analysis command. Every saved activation is a pooled float16
vector; full token tensors are never persisted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory import phase2_job2c_behavioral_companion as scoring
from research.lifecycle_memory import phase2_job3_heldout_confirmation as job3
from research.lifecycle_memory import run_phase2_mech_job1_instrument_gates as gates
from research.lifecycle_memory.reanchored_token_positions import (
    QUESTION_TOKEN_CONTRACT,
    resolve_final_question_token,
)
from research.lifecycle_memory import run_phase2_mech_job2_battery_capture as legacy


ARMS = ("real", "blank_full", "wrong_time")
SLOT_NAMES = ("full_frame", "crop_A", "crop_B")
FULL_FRAME_COUNT = 96
CANARY_FRAME_COUNT = 4
MECHANISM_FRAME_COUNT = 96
MECHANISM_TRIAL_COUNT = 192
MECHANISM_PARTICIPANT_COUNT = 96
MECHANISM_SELECTION_NAMESPACE = "phase2_reanchored_egohos_mechanism_subset_v1"
MECHANISM_SELECTION_RULE = (
    "outcome_blind_participant_first_hash_order_one_frame_per_participant_before_fill"
)
FULL_CANVAS_SIZE = (896, 672)
CROP_CANVAS_SIZE = (336, 336)
SELECTION_NAMESPACE = "phase2-reanchored-jobB-layerwise-2afc-v2"
GOVERNING_PRESPEC_REL = (
    "docs/research/lifecycle_memory/phase 2 research/preregistration/"
    "reanchored_evidence_failure_localization_v1_2026_07_21.json"
)
GOVERNING_PRESPEC_SHA256 = "bd0f021fd510d873f388075a3d4c0e0d138da07fc6da8215cc0ff16b1d09279f"
TARGET_COMPONENT_RULE = (
    "within the chosen class mask, select the largest 8-connected component with area "
    "at least 64 pixels; break exact area ties by lexicographically smallest "
    "top,left,bottom,right bounding box; render only that component and record its "
    "hidden fingerprint"
)
CANDIDATE_COMPLETIONS = {
    "A": '{"answer":"A","abstain":false,"abstention_reason":null,"confidence":1.0}',
    "B": '{"answer":"B","abstain":false,"abstention_reason":null,"confidence":1.0}',
}
TARGET_STORAGE_GIB = 6.0
HARD_STORAGE_GIB = 8.0
FLOAT_BYTES = 2
COUNTERBALANCE_PAIR_SIZE = 2
FULL_WALLTIME_SECONDS = 2 * 60 * 60
ANALYSIS_CLEANUP_RESERVE_SECONDS = 20 * 60
FULL_WALLTIME_HEADROOM_FRACTION = 0.25
MIN_FULL_WALLTIME_SECONDS = 60 * 60
WALLTIME_ROUND_SECONDS = 5 * 60
MIN_PROJECTED_AGGREGATE_BUSY_FRACTION = 0.70
BALANCED_SCHEDULE_KIND = "phase2_reanchored_jobB_balanced_full_schedule_v1"
FROZEN_BATCH_PLAN_KIND = "phase2_reanchored_jobB_canary_frozen_batch_plan_v3"
DECODER_STATE_SEMANTICS = (
    "direct_hook_state_0=lm_entry_pre_block_0; "
    "direct_hook_state_i+1=raw_post_block_i=jobC_block_i; final_norm_excluded"
)
CLAIM_BOUNDARY = (
    "Observational layerwise capture only. Signed slot-selective contrasts can "
    "localize selective propagation, not the origin of failure. Hidden-label "
    "probes are exploratory and require nuisance baselines; positive probes or "
    "representation differences do not establish causal use."
)
# Explicitly rejected evaluator/leakage keys. Capture workers never consume them.
FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "canonical_answer_json",
    "class_id",
    "correct_answer",
    "correct_choice",
    "component_fingerprint",
    "expected_answer",
    "ego4d_participant_id",
    "first_order_class",
    "first_order_target",
    "label",
    "label_role",
    "participant_cluster_id",
    "participant_id",
    "second_order_class",
    "second_order_target",
    "target_component_fingerprint",
    "target_component_fingerprints",
    "target_role",
}
FROZEN_MODEL_IDENTITIES = (
    ("qwen35_4b_hf_bf16", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ("qwen35_27b_hf_bf16", "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"),
    ("gemma4_12b_hf_bf16", "google/gemma-4-12B-it", "5926caa4ec0cac5cbfadaf4077420520de1d5205"),
    ("gemma4_31b_hf_bf16", "google/gemma-4-31B-it", "3548789868c5356dbf307c98e6f609007b82b3eb"),
)
MODEL_ARCHITECTURE_SITES = {
    "qwen35_4b_hf_bf16": {
        "vision_final_module": "model.visual.blocks.23",
        "vision_capture_role": "vision_block",
        "projector_candidates": ["model.visual.merger"],
        "decoder_container": "model.language_model.layers",
    },
    "qwen35_27b_hf_bf16": {
        "vision_final_module": "model.visual.blocks.26",
        "vision_capture_role": "vision_block",
        "projector_candidates": ["model.visual.merger"],
        "decoder_container": "model.language_model.layers",
    },
    "gemma4_12b_hf_bf16": {
        "vision_final_module": (
            "model.embed_vision.multimodal_embedder.embedding_pre_projection_norm"
        ),
        "vision_capture_role": "final_pre_projector_vision_representation",
        "projector_candidates": [
            "model.embed_vision.multimodal_embedder.embedding_projection"
        ],
        "decoder_container": "model.language_model.layers",
    },
    "gemma4_31b_hf_bf16": {
        "vision_final_module": "model.vision_tower.encoder.layers.26",
        "vision_capture_role": "vision_block",
        "projector_candidates": [
            "model.embed_vision.embedding_projection",
            "model.embed_vision.multimodal_embedder.embedding_projection",
        ],
        "decoder_container": "model.language_model.layers",
    },
}
DEFAULT_MODEL_PLAN = [
    {
        "canonical_rank": canonical_rank,
        "model_key": key,
        "model_id": model_id,
        "revision": revision,
        "hidden_size": hidden_size,
        "num_hidden_layers": layers,
        "vision_hidden_size": vision_hidden,
        "canary_microbatch_size": batch,
        "full_microbatch_hard_cap": batch,
        "decoder_state_semantics": DECODER_STATE_SEMANTICS,
        "governing_prespec_sha256": GOVERNING_PRESPEC_SHA256,
        **MODEL_ARCHITECTURE_SITES[key],
    }
    for canonical_rank, (
        (key, model_id, revision),
        hidden_size,
        layers,
        vision_hidden,
        batch,
    ) in enumerate(
        zip(
            FROZEN_MODEL_IDENTITIES,
            (2560, 5120, 3840, 5376),
            (32, 64, 48, 60),
            (1024, 1152, 3840, 1152),
            (4, 1, 2, 1),
        )
    )
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    legacy._write_json(path, payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_sha256_env(name: str) -> str:
    value = str(os.environ.get(name) or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"worker requires a lowercase SHA-256 binding in {name}")
    return value


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _code_bundle_binding(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    relative_paths = (
        "research/lifecycle_memory/run_phase2_reanchored_layer_capture.py",
        "research/lifecycle_memory/analyze_phase2_reanchored_layer_capture.py",
        "research/lifecycle_memory/phase2_job2c_behavioral_companion.py",
        "research/lifecycle_memory/phase2_job3_heldout_confirmation.py",
        "research/lifecycle_memory/run_phase2_mech_job1_instrument_gates.py",
        "research/lifecycle_memory/run_phase2_mech_job2_battery_capture.py",
        "docs/research/lifecycle_memory/phase 2 research/jobs/"
        "reanchored_jobB_layerwise/job_tamia_reanchored_jobB_layerwise_h100.sh",
        "docs/research/lifecycle_memory/phase 2 research/jobs/job0/"
        "run_phase1_qwen27b_hf.py",
    )
    files = {}
    for relative in relative_paths:
        path = (repo_root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Job B code-bundle file is missing: {relative}")
        files[relative] = _sha256_file(path)
    return {
        "files": files,
        "sha256": _canonical_sha256(files),
    }


def _stable_key(*parts: Any) -> str:
    return _canonical_sha256([SELECTION_NAMESPACE, *(str(part) for part in parts)])


def assert_governing_prespec(path: Path | None = None) -> dict[str, Any]:
    resolved = (path or (REPO_ROOT / GOVERNING_PRESPEC_REL)).expanduser().resolve()
    observed = _sha256_file(resolved)
    if observed != GOVERNING_PRESPEC_SHA256:
        raise ValueError(
            f"governing prespec sha256 mismatch: expected {GOVERNING_PRESPEC_SHA256}, "
            f"observed {observed}"
        )
    payload = _read_json(resolved)
    if payload.get("candidate_completions") != CANDIDATE_COMPLETIONS:
        raise ValueError("governing prespec A/B candidate completions drifted")
    job_b = payload.get("job_b", {})
    if job_b.get("diagnostic_frame_count") != FULL_FRAME_COUNT:
        raise ValueError("governing prespec Job B frame count drifted")
    observed_rule = payload.get("data", {}).get("selection", {}).get("target_component_rule")
    if observed_rule != TARGET_COMPONENT_RULE:
        raise ValueError("governing prespec target-component rule drifted")
    return {"path": str(resolved), "sha256": observed, "payload": payload}


def _assert_public_firewall(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).strip().lower() in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"public pack contains hidden key at {path}.{key}")
            _assert_public_firewall(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_public_firewall(child, f"{path}[{index}]")


def _image_ref(value: Any, fallback: Mapping[str, str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a path/hash object")
    path = str(value.get("path") or value.get("file") or "")
    digest = str(value.get("sha256") or fallback.get(path) or "").lower()
    if not path or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{field} lacks an exact path/sha256")
    row = {"path": path, "sha256": digest}
    for key in ("width", "height", "kind", "description", "slot"):
        if value.get(key) is not None:
            row[key] = value[key]
    return row


def _normalize_request(
    row: dict[str, Any], *, arm: str
) -> dict[str, Any]:
    _assert_public_firewall(row)
    request_id = str(row.get("request_id") or "")
    item_id = request_id
    public_pair_id = str(row.get("frame_unit_id") or "")
    order_id = str(row.get("order_id") or "")
    if order_id not in {"order_0", "order_1"}:
        raise ValueError(f"request {request_id} has invalid public-neutral order_id")
    order_index = int(order_id.removeprefix("order_"))
    system_prompt = str(row.get("system_prompt") or "")
    user_prompt = str(row.get("user_prompt") or row.get("prompt") or "")
    values = {
        "item_id": item_id,
        "request_id": request_id,
        "user_prompt": user_prompt,
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError(f"public 2AFC request missing fields: {missing}")
    prompt = user_prompt.lower()
    if not all(term in prompt for term in ("first-order interacting object", "inputs 2 and 3", "answer a or b")):
        raise ValueError(f"item {item_id} does not use the frozen 2AFC prompt")
    if "no direct contact" in prompt:
        raise ValueError(f"item {item_id} uses forbidden legacy wording")
    arms = row.get("arms")
    arm_row = arms.get(arm) if isinstance(arms, Mapping) else None
    raw_inputs = arm_row.get("inputs") if isinstance(arm_row, Mapping) else arm_row
    if not isinstance(raw_inputs, list) or len(raw_inputs) != 3:
        raise ValueError(f"item {item_id} must have exactly three image inputs")
    inputs = [
        _image_ref(value, {}, f"{arm}.inputs[{index}]")
        for index, value in enumerate(raw_inputs)
    ]
    return {
        **values,
        "trial_id": item_id,
        "public_pair_id": public_pair_id,
        "frame_id": public_pair_id,
        "counterbalance_pair_index": -1,
        "public_order_index": order_index,
        "crop_order": order_id,
        "counterbalanced_request_id": str(
            row.get("counterbalanced_request_id") or ""
        ),
        "arm": arm,
        "system_prompt": system_prompt,
        "inputs": inputs,
    }


def _resolve_asset(asset_root: Path, ref: Mapping[str, Any]) -> Path:
    path = Path(str(ref["path"]))
    return path if path.is_absolute() else asset_root / path


def _builder_file_record(
    manifest: Mapping[str, Any], key: str, expected_path: str
) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("mechanism manifest lacks the builder files object")
    record = files.get(key)
    if not isinstance(record, Mapping):
        raise ValueError(f"mechanism manifest lacks files.{key}")
    if str(record.get("path") or "") != expected_path:
        raise ValueError(f"mechanism manifest files.{key}.path drifted")
    digest = str(record.get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"mechanism manifest files.{key}.sha256 is invalid")
    if int(record.get("rows") or 0) != MECHANISM_TRIAL_COUNT:
        raise ValueError(f"mechanism manifest files.{key}.rows is not 192")
    return {"path": expected_path, "sha256": digest, "rows": MECHANISM_TRIAL_COUNT}


def load_and_validate_mechanism_subset(
    mechanism_dir: Path,
    *,
    verify_files: bool,
    prespec_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    mechanism_dir = mechanism_dir.expanduser().resolve()
    prespec = assert_governing_prespec(prespec_path)
    manifest_path = mechanism_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("governing_prespec_sha256") != GOVERNING_PRESPEC_SHA256:
        raise ValueError("mechanism manifest is not bound to the governing prespec")
    if manifest.get("kind") != "phase2_reanchored_egohos_mechanism_subset_manifest":
        raise ValueError("mechanism manifest kind is not the builder mechanism subset")
    if int(manifest.get("schema_version") or 0) != 1:
        raise ValueError("mechanism manifest schema_version is not 1")
    if manifest.get("status") != "complete_production" or manifest.get(
        "production_eligible"
    ) is not True:
        raise ValueError("mechanism subset is not production eligible")
    frame_count = int(manifest.get("frame_count") or 0)
    trial_count = int(manifest.get("trial_count") or 0)
    participant_count = int(manifest.get("participant_cluster_count") or 0)
    if (
        frame_count != MECHANISM_FRAME_COUNT
        or trial_count != MECHANISM_TRIAL_COUNT
        or participant_count != MECHANISM_PARTICIPANT_COUNT
    ):
        raise ValueError(
            "mechanism subset must declare exactly 96 frames, 192 trials, "
            "and 96 distinct participants"
        )
    if manifest.get("selection_namespace") != MECHANISM_SELECTION_NAMESPACE:
        raise ValueError("mechanism selection namespace drifted")
    if manifest.get("selection_rule") != MECHANISM_SELECTION_RULE:
        raise ValueError("mechanism participant-first selection rule drifted")
    if manifest.get("selection_uses_model_outputs") is not False or manifest.get(
        "selection_uses_hidden_answers_or_roles"
    ) is not False:
        raise ValueError("mechanism selection is not outcome blind")
    if int(manifest.get("counterbalanced_trials_per_frame") or 0) != 2:
        raise ValueError("mechanism manifest does not declare two orders per frame")
    if manifest.get("arms") != list(ARMS) or int(manifest.get("images_per_arm") or 0) != 3:
        raise ValueError("mechanism manifest arm or image-slot contract drifted")
    pair_audit = manifest.get("pair_audit")
    if not isinstance(pair_audit, Mapping) or pair_audit.get("status") != "pass":
        raise ValueError("mechanism builder pair audit did not pass")
    expected_pair_checks = {
        "full_frame_identity_check_count": MECHANISM_FRAME_COUNT * len(ARMS),
        "crop_ab_exact_permutation_check_count": MECHANISM_FRAME_COUNT * len(ARMS),
        "cross_arm_crop_identity_check_count": MECHANISM_TRIAL_COUNT * 2,
    }
    for key, expected in expected_pair_checks.items():
        if int(pair_audit.get(key) or 0) != expected:
            raise ValueError(f"mechanism pair audit count drifted at {key}")
    for key in (
        "public_frame_order_sha256",
        "full_frame_identity_sha256",
        "crop_permutation_identity_sha256",
    ):
        value = (
            manifest.get(key)
            if key == "public_frame_order_sha256"
            else pair_audit.get(key)
        )
        if not re.fullmatch(r"[0-9a-f]{64}", str(value or "").lower()):
            raise ValueError(f"mechanism manifest has invalid {key}")
    geometry = manifest.get("fixed_geometry")
    expected_geometry = {
        "full_frame": {"width": FULL_CANVAS_SIZE[0], "height": FULL_CANVAS_SIZE[1]},
        "reference_crop": {"width": CROP_CANVAS_SIZE[0], "height": CROP_CANVAS_SIZE[1]},
    }
    if geometry != expected_geometry:
        raise ValueError("mechanism manifest fixed geometry drifted")
    public_schema = manifest.get("public_schema")
    if not isinstance(public_schema, Mapping) or public_schema.get(
        "evaluator_fields_present"
    ) is not False:
        raise ValueError("mechanism public schema does not declare the label firewall")
    hidden_schema = manifest.get("hidden_schema")
    if not isinstance(hidden_schema, Mapping) or {
        "join_key": hidden_schema.get("join_key"),
        "correct_choice": hidden_schema.get("correct_choice"),
        "participant_field": hidden_schema.get("participant_field"),
        "frame_field": hidden_schema.get("frame_field"),
        "component_fingerprint_field": hidden_schema.get(
            "component_fingerprint_field"
        ),
    } != {
        "join_key": "request_id",
        "correct_choice": ["A", "B"],
        "participant_field": "ego4d_participant_id",
        "frame_field": "frame_id",
        "component_fingerprint_field": "target_component_fingerprints",
    }:
        raise ValueError("mechanism hidden join schema drifted")
    job_b_contract = (
        manifest.get("consumer_contract", {}).get("job_b", {})
        if isinstance(manifest.get("consumer_contract"), Mapping)
        else {}
    )
    if job_b_contract != {
        "public_requests": "mechanism/public/mechanism_requests.jsonl",
        "subset_selection_already_frozen": True,
        "must_not_take_first_rows_from_full_battery": True,
    }:
        raise ValueError("mechanism Job B consumer contract drifted")

    public_rel = "mechanism/public/mechanism_requests.jsonl"
    hidden_rel = "mechanism/hidden/mechanism_evaluator_index.jsonl"
    public_record = _builder_file_record(manifest, "public_requests", public_rel)
    hidden_record = _builder_file_record(
        manifest, "hidden_evaluator_index", hidden_rel
    )
    public_path = mechanism_dir.parent / public_record["path"]
    public_hash = public_record["sha256"]
    hidden_hash = hidden_record["sha256"]
    if _sha256_file(public_path) != public_hash:
        raise ValueError("mechanism public request file hash mismatch")
    public_rows = list(_iter_jsonl(public_path))
    if len(public_rows) != MECHANISM_TRIAL_COUNT:
        raise ValueError("mechanism public request count is not 192")
    expected_prompt = str(
        prespec["payload"].get("generation", {}).get("prompt_text") or ""
    )
    request_ids = [str(row.get("request_id") or "") for row in public_rows]
    if any(not value for value in request_ids) or len(set(request_ids)) != len(
        request_ids
    ):
        raise ValueError("mechanism public request IDs are missing or duplicated")
    frame_order: list[str] = []
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    verified_hashes: dict[Path, str] = {}
    verified_sizes: dict[Path, tuple[int, int]] = {}
    full_sizes: set[tuple[int, int]] = set()
    crop_sizes: set[tuple[int, int]] = set()
    asset_root_raw = str(manifest.get("asset_root") or "..")
    asset_root = (
        Path(asset_root_raw)
        if Path(asset_root_raw).is_absolute()
        else mechanism_dir / asset_root_raw
    ).resolve()
    for source_index, public_row in enumerate(public_rows):
        _assert_public_firewall(public_row)
        if str(public_row.get("user_prompt") or "") != expected_prompt:
            raise ValueError("mechanism request prompt drifted")
        frame_id = str(public_row.get("frame_unit_id") or "")
        if not frame_id:
            raise ValueError("mechanism public row lacks frame_unit_id")
        if frame_id not in by_frame:
            frame_order.append(frame_id)
        by_frame[frame_id].append(public_row)
        pair_index = frame_order.index(frame_id)
        normalized_by_arm = {}
        for arm in ARMS:
            normalized = _normalize_request(public_row, arm=arm)
            normalized["counterbalance_pair_index"] = pair_index
            normalized_by_arm[arm] = normalized
            rows_by_arm[arm].append(normalized)
            for slot_index, ref in enumerate(normalized["inputs"]):
                declared_size = (
                    int(ref.get("width") or 0),
                    int(ref.get("height") or 0),
                )
                if min(declared_size) <= 0:
                    raise ValueError("mechanism asset lacks declared fixed geometry")
                (full_sizes if slot_index == 0 else crop_sizes).add(declared_size)
                if verify_files:
                    asset = _resolve_asset(asset_root, ref)
                    if not asset.is_file():
                        raise FileNotFoundError(asset)
                    observed = verified_hashes.get(asset)
                    if observed is None:
                        observed = _sha256_file(asset)
                        verified_hashes[asset] = observed
                    if observed != ref["sha256"]:
                        raise ValueError(f"mechanism asset hash mismatch: {asset}")
                    size = verified_sizes.get(asset)
                    if size is None:
                        with Image.open(asset) as image:
                            size = (int(image.width), int(image.height))
                        verified_sizes[asset] = size
                    if size != declared_size:
                        raise ValueError(f"mechanism asset geometry mismatch: {asset}")
        crop_hashes = {
            arm: [ref["sha256"] for ref in normalized_by_arm[arm]["inputs"][1:]]
            for arm in ARMS
        }
        if any(crop_hashes[arm] != crop_hashes["real"] for arm in ARMS):
            raise ValueError(
                f"crop A/B bytes changed across evidence arms: {public_row['request_id']}"
            )
    if len(by_frame) != MECHANISM_FRAME_COUNT:
        raise ValueError("mechanism public rows do not form exactly 96 frame pairs")
    if len(full_sizes) != 1 or len(crop_sizes) != 1:
        raise ValueError(
            f"fixed mechanism geometry failed: full={full_sizes} crops={crop_sizes}"
        )
    for frame_id, pair in by_frame.items():
        if len(pair) != 2 or {str(row.get("order_id")) for row in pair} != {
            "order_0",
            "order_1",
        }:
            raise ValueError(f"mechanism frame is not one order pair: {frame_id}")
        first, second = sorted(pair, key=lambda row: str(row["order_id"]))
        if (
            first.get("counterbalanced_request_id") != second.get("request_id")
            or second.get("counterbalanced_request_id") != first.get("request_id")
        ):
            raise ValueError(f"counterbalance join is not reciprocal: {frame_id}")
        for arm in ARMS:
            first_row = next(
                row
                for row in rows_by_arm[arm]
                if row["request_id"] == first["request_id"]
            )
            second_row = next(
                row
                for row in rows_by_arm[arm]
                if row["request_id"] == second["request_id"]
            )
            first_hashes = [ref["sha256"] for ref in first_row["inputs"]]
            second_hashes = [ref["sha256"] for ref in second_row["inputs"]]
            if first_hashes[0] != second_hashes[0]:
                raise ValueError(f"full frame changed across orders: {frame_id}/{arm}")
            if first_hashes[1:] != list(reversed(second_hashes[1:])):
                raise ValueError(
                    f"crop A/B is not an exact permutation: {frame_id}/{arm}"
                )
    audit = {
        "status": "pass",
        "mechanism_dir": str(mechanism_dir),
        "asset_root": str(asset_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "public_path": str(public_path),
        "public_sha256": public_hash,
        "hidden_relative_path": hidden_rel,
        "hidden_sha256_declared_only": hidden_hash,
        "hidden_evaluator_opened": False,
        "frame_count": len(by_frame),
        "trial_count": len(public_rows),
        "participant_count_declared": participant_count,
        "one_frame_per_participant": True,
        "outcome_blind": True,
        "full_canvas_sizes": sorted(full_sizes),
        "crop_canvas_sizes": sorted(crop_sizes),
        "verified_unique_assets": len(verified_hashes),
        "slot_order": list(SLOT_NAMES),
        "counterbalance_permutation_audit": "pass",
        "cross_arm_crop_identity_audit": "pass",
        "public_hidden_label_firewall": "pass",
        "target_component_rule": TARGET_COMPONENT_RULE,
        "governing_prespec_sha256": GOVERNING_PRESPEC_SHA256,
    }
    return manifest, rows_by_arm, audit


def _parse_model_plan(path: Path | None) -> list[dict[str, Any]]:
    raw = _read_json(path.expanduser().resolve()) if path else DEFAULT_MODEL_PLAN
    if not isinstance(raw, list):
        raise ValueError("model plan must be a list")
    plan = [dict(row) for row in raw if isinstance(row, dict)]
    identities = tuple(
        (str(row.get("model_key")), str(row.get("model_id")), str(row.get("revision")))
        for row in plan
    )
    if identities != FROZEN_MODEL_IDENTITIES:
        raise ValueError("model plan identities/revisions do not match the frozen four-model order")
    required = ("hidden_size", "num_hidden_layers", "vision_hidden_size", "canary_microbatch_size")
    for canonical_rank, row in enumerate(plan):
        if any(int(row.get(key) or 0) <= 0 for key in required):
            raise ValueError(f"model plan lacks positive sizing fields: {row.get('model_key')}")
        if int(row.get("canonical_rank", -1)) != canonical_rank:
            raise ValueError(f"model plan canonical rank drift: {row.get('model_key')}")
        if row.get("governing_prespec_sha256") != GOVERNING_PRESPEC_SHA256:
            raise ValueError(f"model plan prespec hash drift: {row.get('model_key')}")
        if row.get("vision_final_module") != MODEL_ARCHITECTURE_SITES[row["model_key"]][
            "vision_final_module"
        ]:
            raise ValueError(f"model plan vision-final site drift: {row.get('model_key')}")
        if row.get("vision_capture_role") != MODEL_ARCHITECTURE_SITES[row["model_key"]][
            "vision_capture_role"
        ]:
            raise ValueError(f"model plan vision capture role drift: {row.get('model_key')}")
        if row.get("projector_candidates") != MODEL_ARCHITECTURE_SITES[row["model_key"]][
            "projector_candidates"
        ]:
            raise ValueError(f"model plan projector candidates drift: {row.get('model_key')}")
        if row.get("decoder_container") != MODEL_ARCHITECTURE_SITES[row["model_key"]][
            "decoder_container"
        ]:
            raise ValueError(f"model plan decoder container drift: {row.get('model_key')}")
        if int(row.get("full_microbatch_hard_cap") or 0) != int(
            row["canary_microbatch_size"]
        ):
            raise ValueError(f"model plan full batch cap drift: {row.get('model_key')}")
        if row.get("decoder_state_semantics") != DECODER_STATE_SEMANTICS:
            raise ValueError(f"model plan decoder-state semantics drift: {row.get('model_key')}")
    return plan


def _observed_language_dimensions(config_summary: Mapping[str, Any]) -> tuple[int, int]:
    """Read decoder dimensions from either a text config or a text-only config."""
    nested = config_summary.get("text_config")
    candidates = [config_summary]
    if isinstance(nested, Mapping):
        candidates.insert(0, nested)
    for candidate in candidates:
        hidden = candidate.get("hidden_size")
        layers = candidate.get("num_hidden_layers")
        if type(hidden) is int and hidden > 0 and type(layers) is int and layers > 0:
            return hidden, layers
    return -1, -1


def _observed_vision_hidden_size(
    config_summary: Mapping[str, Any],
    model: Any,
    model_row: Mapping[str, Any],
) -> tuple[int, str]:
    """Resolve the live pre-projector vision width without architecture guessing."""
    modules = dict(model.named_modules())
    for module_name in model_row.get("projector_candidates", []):
        module = modules.get(str(module_name))
        if module is None:
            continue
        in_features = getattr(module, "in_features", None)
        if type(in_features) is int and in_features > 0:
            return in_features, f"{module_name}.in_features"
    vision_config = config_summary.get("vision_config")
    if isinstance(vision_config, Mapping):
        hidden = vision_config.get("hidden_size")
        if type(hidden) is int and hidden > 0:
            return hidden, "vision_config.hidden_size"
    return -1, "unresolved"


def estimate_storage(plan: Sequence[Mapping[str, Any]], frame_count: int) -> dict[str, Any]:
    rows_per_model = int(frame_count) * 2 * len(ARMS)
    models = []
    total = 0
    for row in plan:
        hidden = int(row["hidden_size"])
        hidden_states = int(row["num_hidden_layers"]) + 1
        vision = int(row["vision_hidden_size"])
        decoder_values = hidden_states * 5 * hidden
        hook_values = (1 * 4 * vision) + (1 * 4 * hidden)
        payload_bytes = rows_per_model * (decoder_values + hook_values) * FLOAT_BYTES
        total += payload_bytes
        models.append(
            {
                "model_key": row["model_key"],
                "rows": rows_per_model,
                "projected_payload_bytes": payload_bytes,
                "projected_payload_gib": payload_bytes / (1024**3),
            }
        )
    conservative = int(math.ceil(total * 1.10))
    return {
        "kind": "phase2_reanchored_jobB_storage_projection",
        "float_format": "float16",
        "bytes_per_value": FLOAT_BYTES,
        "full_token_tensors_persisted": False,
        "frame_count": int(frame_count),
        "rows_per_model": rows_per_model,
        "models": models,
        "raw_payload_bytes": total,
        "raw_payload_gib": total / (1024**3),
        "conservative_bytes_with_10pct_overhead": conservative,
        "conservative_gib_with_10pct_overhead": conservative / (1024**3),
        "target_gib": TARGET_STORAGE_GIB,
        "hard_ceiling_gib": HARD_STORAGE_GIB,
        "target_status": "pass" if conservative <= TARGET_STORAGE_GIB * 1024**3 else "warn",
        "hard_ceiling_status": "pass" if conservative <= HARD_STORAGE_GIB * 1024**3 else "fail",
    }


def prepare_subset(args: argparse.Namespace) -> int:
    frame_count = FULL_FRAME_COUNT if args.mode == "full" else CANARY_FRAME_COUNT
    _, rows_by_arm, audit = load_and_validate_mechanism_subset(
        args.mechanism_dir, verify_files=True, prespec_path=args.prespec_path
    )
    plan = _parse_model_plan(args.model_plan_file)
    storage = estimate_storage(plan, frame_count)
    if storage["hard_ceiling_status"] != "pass" and not args.allow_storage_over_ceiling:
        raise RuntimeError("projected tensor payload exceeds the hard 8 GiB ceiling")
    real_rows = rows_by_arm["real"]
    selected = list(real_rows[: frame_count * 2])
    selected_frames = list(dict.fromkeys(row["public_pair_id"] for row in selected))
    if len(selected_frames) != frame_count:
        raise RuntimeError("mechanism canary/full selection split a counterbalanced pair")
    if [row["public_pair_id"] for row in selected] != [
        pair for pair in selected_frames for _ in range(2)
    ]:
        raise RuntimeError("mechanism request file does not keep paired orders adjacent")
    manifest = {
        "kind": "phase2_reanchored_jobB_subset_manifest",
        "schema_version": 2,
        "created_at": _utc_now(),
        "mode": args.mode,
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_uses_fields": [
            "builder_frozen_mechanism_request_order",
            "frame_unit_id",
            "order_id",
            "request_id",
        ],
        "selection_rule": (
            "full mode consumes all 192 builder-frozen mechanism trials; canary mode "
            "uses four complete public pairs only for technical validation"
        ),
        "full_mode_reselects_rows": False,
        "selection_uses_model_outputs": False,
        "selection_uses_hidden_answers_or_roles": False,
        "hidden_labels_opened": False,
        "source_mechanism_subset": audit,
        "governing_prespec_sha256": GOVERNING_PRESPEC_SHA256,
        "frame_count": len(selected_frames),
        "trial_count": len(selected),
        "selected_public_pair_ids": selected_frames,
        "selected_item_ids": [row["item_id"] for row in selected],
        "selected_trial_ids": [row["item_id"] for row in selected],
        "arms": list(ARMS),
        "storage_projection": storage,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "selection_manifest.json"
    if path.exists():
        previous = _read_json(path)
        for field in (
            "mode",
            "selection_namespace",
            "selected_public_pair_ids",
            "selected_item_ids",
        ):
            if previous.get(field) != manifest.get(field):
                raise RuntimeError(f"existing selection manifest drifted at {field}")
    _write_json(path, manifest)
    _write_json(run_dir / "storage_projection.json", storage)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _evenly_selected(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return list(rows)
    indexes = sorted({round(index * (len(rows) - 1) / (count - 1)) for index in range(count)})
    return [rows[index] for index in indexes]


def build_capture_plan(
    module_rows: Sequence[dict[str, Any]],
    model_family: str,
    model_plan_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pattern = re.compile(r"(?:layers|blocks)\.(\d+)$")
    vision = []
    for row in module_rows:
        name = str(row.get("name") or "")
        match = pattern.search(name)
        if match and any(token in name.lower() for token in ("visual", "vision", "vit", "siglip")):
            vision.append({**row, "layer_index": int(match.group(1))})
    vision.sort(key=lambda row: (row["layer_index"], str(row["name"])))
    selected_vision = vision[-1:]
    projectors = []
    for row in module_rows:
        name = str(row.get("name") or "")
        text = f"{name}.{row.get('class_name', '')}".lower()
        if name and any(token in text for token in ("projector", "merger", "mm_projector", "multi_modal")):
            projectors.append(row)
    projectors.sort(
        key=lambda row: (
            0 if str(row["name"]).lower().endswith(("projector", "merger")) else 1,
            str(row["name"]).count("."),
            str(row["name"]),
        )
    )
    selected_projector = projectors[:1]
    frozen_vision_is_last_observed = True
    frozen_vision_layer_index: int | None = None
    observed_final_vision_layer_index: int | None = None
    if model_plan_row is not None:
        by_name = {str(row.get("name") or ""): row for row in module_rows}
        vision_name = str(model_plan_row.get("vision_final_module") or "")
        selected_vision = [by_name[vision_name]] if vision_name in by_name else []
        declared_match = pattern.search(vision_name)
        if declared_match is not None:
            frozen_vision_layer_index = int(declared_match.group(1))
            vision_container = vision_name[: declared_match.start(1)]
            peer_indexes = []
            for row in module_rows:
                candidate_name = str(row.get("name") or "")
                candidate_match = pattern.search(candidate_name)
                if (
                    candidate_match is not None
                    and candidate_name[: candidate_match.start(1)] == vision_container
                ):
                    peer_indexes.append(int(candidate_match.group(1)))
            if peer_indexes:
                observed_final_vision_layer_index = max(peer_indexes)
                frozen_vision_is_last_observed = (
                    frozen_vision_layer_index == observed_final_vision_layer_index
                )
        projector_names = [
            str(value) for value in model_plan_row.get("projector_candidates", [])
        ]
        selected_projector = [
            by_name[name] for name in projector_names if name in by_name
        ]
    hook_names = [str(row["name"]) for row in [*selected_vision, *selected_projector]]
    vision_role = (
        str(model_plan_row.get("vision_capture_role"))
        if model_plan_row is not None
        else "vision_block"
    )
    roles = {str(row["name"]): vision_role for row in selected_vision}
    roles.update({str(row["name"]): "projector_or_merger" for row in selected_projector})
    missing = []
    if len(selected_vision) != 1:
        missing.append("vision_tower_final")
    elif not frozen_vision_is_last_observed:
        missing.append("vision_tower_final_not_last_observed")
    if len(selected_projector) != 1:
        missing.append("projector_or_merger")
    return {
        "status": "pass" if not missing else "fail",
        "model_family": model_family,
        "hook_names": hook_names,
        "hook_roles": roles,
        "vision_block_count_available": len(vision),
        "architecture_sites_source": (
            "frozen_model_plan" if model_plan_row is not None else "synthetic_discovery"
        ),
        "vision_capture_semantics": (
            "frozen_architecture_site_per_image_slot_mean; Gemma4-12B uses its "
            "final pre-projector representation rather than a raw tower block"
        ),
        "selected_vision_blocks": [str(row["name"]) for row in selected_vision],
        "frozen_vision_layer_index": frozen_vision_layer_index,
        "observed_final_vision_layer_index": observed_final_vision_layer_index,
        "frozen_vision_is_last_observed": frozen_vision_is_last_observed,
        "projector_or_merger": [str(row["name"]) for row in selected_projector],
        "decoder_container": (
            str(model_plan_row.get("decoder_container"))
            if model_plan_row is not None
            else None
        ),
        "expected_decoder_layers": (
            int(model_plan_row.get("num_hidden_layers"))
            if model_plan_row is not None
            else None
        ),
        "decoder_state_semantics": DECODER_STATE_SEMANTICS,
        "unsupported_semantics": (
            {"within_slot_spatial_subregions": "unsupported_without_architecture_exact_spatial_map"}
            if "gemma" in model_family.lower()
            else {}
        ),
        "missing_required_sites": missing,
        "coarse_decoder_fractions_for_job_c": [index / 8 for index in range(9)],
        "job_c_refinement_boundary": (
            "plus_or_minus_two_layer refinement is descriptive unless selection is replayed "
            "inside every inferential resample"
        ),
    }


def find_final_question_token(
    input_ids: Sequence[int], tokenizer: Any, prompt: str
) -> dict[str, Any]:
    """Compatibility wrapper around the shared capture/patch token contract."""

    return resolve_final_question_token(input_ids, tokenizer, prompt)


def _require_runtime_tokenizer(processor: Any, tokenizer: Any) -> Any:
    """Recover the tokenizer exposed by generic multimodal processors."""

    resolved = tokenizer if tokenizer is not None else getattr(processor, "tokenizer", None)
    if resolved is None or not callable(getattr(resolved, "encode", None)):
        raise RuntimeError("model runtime did not provide an encode-capable tokenizer")
    return resolved


def _contiguous_runs(positions: Any) -> list[Any]:
    import torch

    if positions is None or int(positions.numel()) == 0:
        return []
    values = [int(value) for value in positions.detach().cpu().tolist()]
    runs: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value == runs[-1][-1] + 1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return [torch.tensor(run, dtype=torch.long, device=positions.device) for run in runs]


def derive_image_slot_masks(
    *, inputs: Any, model: Any, processor: Any, image_count: int = 3
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    input_ids = gates._input_get(inputs, "input_ids")
    if input_ids is None or not hasattr(input_ids, "detach"):
        return {}, {"status": "fail", "reason": "input_ids_unavailable"}
    ids = input_ids.detach()[0] if input_ids.detach().dim() == 2 else input_ids.detach()
    mask = torch.zeros_like(ids, dtype=torch.bool)
    token_ids = gates._image_token_candidates(getattr(model, "config", None), processor)
    for token_id in token_ids:
        mask |= ids == int(token_id)
    positions = torch.nonzero(mask, as_tuple=False).reshape(-1)
    runs = _contiguous_runs(positions)
    grid = gates._grid_from_inputs(inputs)
    key = str(grid.get("grid_key") or "")
    triplets = gates._grid_triplets(grid.get(key, [])) if key else []
    merge = gates._merge_size(getattr(model, "config", None), processor)
    postmerge_counts = []
    premerge_counts = []
    for t, height, width in triplets:
        premerge_counts.append(t * height * width)
        if merge and (t * height * width) % (merge * merge) == 0:
            postmerge_counts.append((t * height * width) // (merge * merge))
    method = None
    slot_positions: list[Any] = []
    if len(runs) == image_count and sum(int(run.numel()) for run in runs) == int(positions.numel()):
        slot_positions = runs
        method = "three_exact_contiguous_image_token_runs"
    elif len(postmerge_counts) == image_count and sum(postmerge_counts) == int(positions.numel()):
        offset = 0
        for count in postmerge_counts:
            slot_positions.append(positions[offset : offset + count])
            offset += count
        method = "processor_grid_exact_ordered_postmerge_counts"
    if len(slot_positions) != image_count or any(int(row.numel()) <= 0 for row in slot_positions):
        return {}, {
            "status": "fail",
            "reason": "three_image_slot_boundaries_not_established",
            "image_token_ids": token_ids,
            "image_token_count": int(positions.numel()),
            "contiguous_run_counts": [int(run.numel()) for run in runs],
            "premerge_counts": premerge_counts,
            "postmerge_counts": postmerge_counts,
        }
    masks = {
        "full_frame": slot_positions[0],
        "crop_A": slot_positions[1],
        "crop_B": slot_positions[2],
        "all_visual": positions,
    }
    ranges = {}
    for name, values in masks.items():
        cpu = [int(value) for value in values.detach().cpu().tolist()]
        ranges[name] = {
            "count": len(cpu),
            "first": min(cpu),
            "last": max(cpu),
            "contiguous": cpu == list(range(min(cpu), max(cpu) + 1)) if cpu else False,
        }
    slot_sets = [set(int(value) for value in masks[name].detach().cpu().tolist()) for name in SLOT_NAMES]
    union = set().union(*slot_sets)
    checks = {
        "three_nonempty_slots": all(slot_sets),
        "pairwise_disjoint": all(not (slot_sets[i] & slot_sets[j]) for i in range(3) for j in range(i + 1, 3)),
        "union_equals_all_visual": union
        == set(int(value) for value in masks["all_visual"].detach().cpu().tolist()),
        "ordered_slots": max(slot_sets[0]) < min(slot_sets[1]) < min(slot_sets[2]),
    }
    return masks, {
        "status": "pass" if all(checks.values()) else "fail",
        "method": method,
        "slot_order": list(SLOT_NAMES),
        "image_token_ids": token_ids,
        "decoder_slot_ranges": ranges,
        "premerge_counts": premerge_counts,
        "postmerge_counts": postmerge_counts,
        "checks": checks,
    }


def _pool_flat_slices(flat: Any, counts: Sequence[int]) -> dict[str, Any] | None:
    if len(counts) != 3 or sum(int(value) for value in counts) != int(flat.shape[0]):
        return None
    result = {}
    offset = 0
    for name, count in zip(SLOT_NAMES, counts):
        value = flat[offset : offset + int(count)]
        if int(value.shape[0]) <= 0:
            return None
        result[name] = value.mean(dim=0)
        offset += int(count)
    result["all_visual"] = flat.mean(dim=0)
    return result


def pool_stage_output(
    tensor: Any, *, batch_size: int, slot_audits: Sequence[Mapping[str, Any]], role: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    if tensor is None or not hasattr(tensor, "detach"):
        return [], {"status": "fail", "reason": "hook_output_not_tensor"}
    value = tensor.detach()
    hidden = int(value.shape[-1])
    candidates = []
    premerge_roles = {
        "vision_block",
        "final_pre_projector_vision_representation",
    }
    for audit in slot_audits:
        preferred = (
            audit.get("premerge_counts")
            if role in premerge_roles
            else audit.get("postmerge_counts")
        )
        fallback = [audit["decoder_slot_ranges"][name]["count"] for name in SLOT_NAMES]
        candidates.append(list(preferred) if isinstance(preferred, list) and len(preferred) == 3 else fallback)
    pooled: list[dict[str, Any]] = []
    method = None
    if value.dim() >= 3 and int(value.shape[0]) == batch_size * 3:
        for batch_index in range(batch_size):
            slots = {}
            chunks = []
            for slot_index, name in enumerate(SLOT_NAMES):
                chunk = value[batch_index * 3 + slot_index].reshape(-1, hidden)
                chunks.append(chunk)
                slots[name] = chunk.mean(dim=0)
            slots["all_visual"] = torch.cat(chunks, dim=0).mean(dim=0)
            pooled.append(slots)
        method = "leading_axis_batch_times_three_images"
    elif value.dim() >= 3 and int(value.shape[0]) == batch_size:
        for batch_index in range(batch_size):
            flat = value[batch_index].reshape(-1, hidden)
            row = _pool_flat_slices(flat, candidates[batch_index])
            if row is None:
                return [], {"status": "fail", "reason": "per_batch_stage_token_count_mismatch"}
            pooled.append(row)
        method = "leading_batch_axis_then_exact_slot_counts"
    else:
        flat = value.reshape(-1, hidden)
        expected = sum(sum(row) for row in candidates)
        if int(flat.shape[0]) != expected:
            return [], {
                "status": "fail",
                "reason": "flattened_stage_token_count_mismatch",
                "shape": [int(dim) for dim in value.shape],
                "expected_tokens": expected,
            }
        offset = 0
        for counts in candidates:
            count = sum(counts)
            row = _pool_flat_slices(flat[offset : offset + count], counts)
            if row is None:
                return [], {"status": "fail", "reason": "flattened_slot_split_failed"}
            pooled.append(row)
            offset += count
        method = "flattened_batch_exact_slot_counts"
    return pooled, {
        "status": "pass",
        "method": method,
        "role": role,
        "input_shape": [int(dim) for dim in value.shape],
        "batch_size": batch_size,
        "slot_order": list(SLOT_NAMES),
        "pooled_only": True,
    }


class SlotHookCollector:
    def __init__(
        self,
        model: Any,
        capture_plan: Mapping[str, Any],
        *,
        batch_size: int,
        slot_audits: Sequence[Mapping[str, Any]],
    ) -> None:
        import torch

        self.handles: list[Any] = []
        self.records: dict[str, Any] = {}
        self.tensors_by_sample: list[dict[str, Any]] = [dict() for _ in range(batch_size)]
        modules = dict(model.named_modules())
        for name in capture_plan["hook_names"]:
            module = modules.get(name)
            if module is None:
                continue
            role = capture_plan["hook_roles"][name]

            def hook(_module: Any, _inputs: Any, output: Any, module_name: str = name, module_role: str = role) -> None:
                if module_name in self.records:
                    self.records[module_name] = {
                        "status": "fail",
                        "reason": "hook_fired_more_than_once_slot_order_ambiguous",
                    }
                    return
                tensor = gates._first_tensor(output)
                pooled, audit = pool_stage_output(
                    tensor,
                    batch_size=batch_size,
                    slot_audits=slot_audits,
                    role=module_role,
                )
                self.records[module_name] = audit
                if audit.get("status") != "pass":
                    return
                safe = legacy._safe_tensor_key(module_name)
                for sample_index, slots in enumerate(pooled):
                    for slot, vector in slots.items():
                        self.tensors_by_sample[sample_index][
                            f"hook__{safe}__{slot}"
                        ] = vector.to(device="cpu", dtype=torch.float16).contiguous()

            self.handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class DecoderStateCollector:
    """Pool the decoder entry and raw block outputs directly from module hooks."""

    def __init__(
        self,
        model: Any,
        *,
        decoder_container: str,
        expected_layers: int,
        batch_size: int,
        slot_masks: Sequence[Mapping[str, Any]],
        question_tokens: Sequence[Mapping[str, Any]],
    ) -> None:
        import torch

        modules = dict(model.named_modules())
        container = modules.get(decoder_container)
        if container is None:
            raise RuntimeError(f"decoder container is absent: {decoder_container}")
        try:
            layers = list(container)
        except TypeError as exc:
            raise RuntimeError(f"decoder container is not iterable: {decoder_container}") from exc
        if len(layers) != expected_layers:
            raise RuntimeError(
                f"decoder layer count drift: expected={expected_layers} observed={len(layers)}"
            )
        if len(slot_masks) != batch_size or len(question_tokens) != batch_size:
            raise RuntimeError("decoder collector boundary-audit batch size drift")

        self._torch = torch
        self.batch_size = batch_size
        self.expected_layers = expected_layers
        self.slot_masks = slot_masks
        self.question_tokens = question_tokens
        self.handles: list[Any] = []
        self.records: dict[int, dict[str, Any]] = {}
        self.tensors_by_sample: list[dict[str, Any]] = [dict() for _ in range(batch_size)]

        def entry_hook(_module: Any, inputs: Any) -> None:
            self._capture_state(0, gates._first_tensor(inputs), "lm_entry_pre_block_0")

        self.handles.append(layers[0].register_forward_pre_hook(entry_hook))
        for block_index, layer in enumerate(layers):

            def block_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                index: int = block_index,
            ) -> None:
                self._capture_state(
                    index + 1,
                    gates._first_tensor(output),
                    f"raw_post_block_{index}",
                )

            self.handles.append(layer.register_forward_hook(block_hook))

    def _capture_state(self, state_index: int, hidden: Any, state_name: str) -> None:
        torch = self._torch
        if state_index in self.records:
            self.records[state_index] = {
                "status": "fail",
                "reason": "decoder_state_hook_fired_more_than_once",
                "state_name": state_name,
            }
            return
        if not torch.is_tensor(hidden) or hidden.dim() != 3:
            self.records[state_index] = {
                "status": "fail",
                "reason": "decoder_state_shape_invalid",
                "state_name": state_name,
            }
            return
        if int(hidden.shape[0]) != self.batch_size:
            self.records[state_index] = {
                "status": "fail",
                "reason": "decoder_state_batch_size_mismatch",
                "state_name": state_name,
                "shape": [int(dim) for dim in hidden.shape],
            }
            return
        prefix = f"decoder_layer_{state_index:03d}"
        for batch_index in range(self.batch_size):
            sequence = hidden.detach()[batch_index]
            question_index = int(self.question_tokens[batch_index]["index"])
            if not 0 <= question_index < int(sequence.shape[0]):
                self.records[state_index] = {
                    "status": "fail",
                    "reason": "question_token_out_of_bounds",
                    "state_name": state_name,
                }
                return
            sample = self.tensors_by_sample[batch_index]
            sample[f"{prefix}__final_question_token"] = (
                sequence[question_index].to(device="cpu", dtype=torch.float16).contiguous()
            )
            for slot in (*SLOT_NAMES, "all_visual"):
                positions = self.slot_masks[batch_index][slot]
                sample[f"{prefix}__{slot}"] = (
                    sequence[positions]
                    .mean(dim=0)
                    .to(device="cpu", dtype=torch.float16)
                    .contiguous()
                )
        self.records[state_index] = {
            "status": "pass",
            "state_name": state_name,
            "shape": [int(dim) for dim in hidden.shape],
            "capture_source": "direct_decoder_module_hook",
        }

    def audit(self) -> dict[str, Any]:
        expected = set(range(self.expected_layers + 1))
        observed = set(self.records)
        failures = {
            str(index): row
            for index, row in sorted(self.records.items())
            if row.get("status") != "pass"
        }
        status = "pass" if observed == expected and not failures else "fail"
        crosswalk = [
            {
                "capture_state_index": index,
                "capture_feature_prefix": f"decoder_layer_{index:03d}",
                "state_name": "lm_entry_pre_block_0" if index == 0 else f"raw_post_block_{index - 1}",
                "job_c_decoder_block_index": None if index == 0 else index - 1,
            }
            for index in range(self.expected_layers + 1)
        ]
        return {
            "status": status,
            "capture_source": "direct_decoder_module_hooks",
            "state_count": len(observed),
            "expected_state_count": self.expected_layers + 1,
            "state_indexes": sorted(observed),
            "missing_state_indexes": sorted(expected - observed),
            "unexpected_state_indexes": sorted(observed - expected),
            "failures": failures,
            "state_crosswalk": crosswalk,
            "slot_pools": [*SLOT_NAMES, "all_visual"],
            "final_normalization_excluded": True,
            "numbering_semantics": DECODER_STATE_SEMANTICS,
            "persisted_dtype": "float16",
            "full_token_tensors_persisted": False,
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def capture_decoder_summaries(
    *,
    hidden_states: Sequence[Any],
    batch_index: int,
    question_index: int,
    slot_masks: Mapping[str, Any],
    expected_count: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if not hidden_states:
        return {}, {"status": "fail", "reason": "hidden_states_missing"}
    if expected_count is not None and len(hidden_states) != expected_count:
        return {}, {
            "status": "fail",
            "reason": "hidden_state_count_mismatch",
            "expected_count": expected_count,
            "observed_count": len(hidden_states),
        }
    tensors = {}
    state_crosswalk = []
    for layer_index, hidden in enumerate(hidden_states):
        if not hasattr(hidden, "detach") or hidden.dim() != 3:
            return {}, {"status": "fail", "reason": f"layer_{layer_index}_shape_invalid"}
        sequence = hidden.detach()[batch_index]
        if not 0 <= question_index < int(sequence.shape[0]):
            return {}, {"status": "fail", "reason": "question_token_out_of_bounds"}
        prefix = f"decoder_layer_{layer_index:03d}"
        state_name = (
            "lm_entry_pre_block_0"
            if layer_index == 0
            else f"raw_post_block_{layer_index - 1}"
        )
        state_crosswalk.append(
            {
                "capture_state_index": layer_index,
                "capture_feature_prefix": prefix,
                "state_name": state_name,
                "job_c_decoder_block_index": None if layer_index == 0 else layer_index - 1,
            }
        )
        tensors[f"{prefix}__final_question_token"] = (
            sequence[question_index].to(device="cpu", dtype=torch.float16).contiguous()
        )
        for slot in (*SLOT_NAMES, "all_visual"):
            positions = slot_masks[slot]
            tensors[f"{prefix}__{slot}"] = (
                sequence[positions].mean(dim=0).to(device="cpu", dtype=torch.float16).contiguous()
            )
    return tensors, {
        "status": "pass",
        "hidden_state_count": len(hidden_states),
        "layer_indexes": list(range(len(hidden_states))),
        "state_names": [row["state_name"] for row in state_crosswalk],
        "state_crosswalk": state_crosswalk,
        "slot_pools": [*SLOT_NAMES, "all_visual"],
        "lm_entry_alias": "decoder_layer_000",
        "capture_source": "caller_supplied_explicit_decoder_states",
        "numbering_semantics": DECODER_STATE_SEMANTICS,
        "persisted_dtype": "float16",
        "full_token_tensors_persisted": False,
    }


def _collate_inputs(inputs_list: Sequence[Any]) -> Any:
    import torch

    if len(inputs_list) == 1:
        return inputs_list[0]
    keys = [set(key for key, _ in gates._input_items(inputs)) for inputs in inputs_list]
    if any(current != keys[0] for current in keys[1:]):
        raise ValueError("microbatch input keys differ")
    result = {}
    for key in sorted(keys[0]):
        values = [gates._input_get(inputs, key) for inputs in inputs_list]
        if all(isinstance(value, torch.Tensor) for value in values):
            try:
                result[key] = torch.cat(values, dim=0)
            except RuntimeError as exc:
                raise ValueError(f"microbatch tensor shape mismatch for {key}") from exc
        else:
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"microbatch non-tensor input differs for {key}")
            result[key] = values[0]
    return result


def _prepare_base(
    *, runtime: Any, processor: Any, request: Mapping[str, Any], image_paths: Sequence[Path], device: Any
) -> tuple[Any, dict[str, Any]]:
    return job3._prepare_universal_base(
        runtime=runtime,
        processor=processor,
        system_prompt=str(request["system_prompt"]),
        user_prompt=str(request["user_prompt"]),
        image_paths=image_paths,
        device=device,
    )


def _candidate_inputs(base_inputs: Any, tokenizer: Any, text: str) -> tuple[Any, list[int], dict[str, Any]]:
    candidate_ids = scoring._encode_candidate_text(tokenizer, text)
    full_inputs, trace = scoring._append_exact_candidate_continuation(base_inputs, candidate_ids)
    audit = scoring._visual_identity_audit(base_inputs, full_inputs)
    if audit.get("status") != "pass":
        raise RuntimeError("candidate append changed visual tensors")
    return full_inputs, candidate_ids, {"append": trace, "visual_identity": audit}


def _score_candidate_batch(
    *, model: Any, base_inputs: Sequence[Any], tokenizer: Any, candidate_label: str
) -> list[dict[str, Any]]:
    import torch

    full_inputs = []
    traces = []
    candidate_ids = []
    for base in base_inputs:
        full, ids, trace = _candidate_inputs(base, tokenizer, CANDIDATE_COMPLETIONS[candidate_label])
        full_inputs.append(full)
        candidate_ids.append(ids)
        traces.append(trace)
    collated = _collate_inputs(full_inputs)
    with torch.inference_mode():
        output = model(**collated)
    logits = gates._extract_logits(output)
    rows = []
    for index, (base, full, ids, trace) in enumerate(zip(base_inputs, full_inputs, candidate_ids, traces)):
        scored = scoring.score_candidate_suffix_from_logits(
            base_input_ids=scoring._input_ids_list(base),
            full_input_ids=scoring._input_ids_list(full),
            logits=logits[index : index + 1],
        )
        rows.append(
            {
                "status": scored.get("status"),
                "candidate_label": candidate_label,
                "candidate_text_sha256": hashlib.sha256(
                    CANDIDATE_COMPLETIONS[candidate_label].encode("utf-8")
                ).hexdigest(),
                "token_count": scored.get("token_count"),
                "sum_logprob": scored.get("sum_logprob"),
                "mean_logprob": scored.get("mean_logprob"),
                "exact_candidate_suffix": scored.get("status") == "pass",
                "candidate_trace": trace,
            }
        )
    return rows


def _geometry_features(paths: Sequence[Path]) -> dict[str, float]:
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append((float(image.width), float(image.height)))
    full_w, full_h = sizes[0]
    a_w, a_h = sizes[1]
    b_w, b_h = sizes[2]
    return {
        "full_aspect_ratio": full_w / full_h,
        "crop_A_aspect_ratio": a_w / a_h,
        "crop_B_aspect_ratio": b_w / b_h,
        "crop_A_to_full_pixel_ratio": (a_w * a_h) / (full_w * full_h),
        "crop_B_to_full_pixel_ratio": (b_w * b_h) / (full_w * full_h),
    }


def _prefetch_assets(paths: Sequence[Path], workers: int = 4) -> dict[str, Any]:
    unique = sorted(set(path.resolve() for path in paths))

    def read(path: Path) -> int:
        with path.open("rb") as handle:
            return len(handle.read())

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        sizes = list(pool.map(read, unique))
    return {
        "file_count": len(unique),
        "bytes_read": sum(sizes),
        "elapsed_s": time.perf_counter() - started,
        "workers": max(1, workers),
    }


def capture_microbatch(
    *,
    runtime: Any,
    model: Any,
    model_id: str,
    processor: Any,
    tokenizer: Any,
    device: Any,
    requests: Sequence[dict[str, Any]],
    battery_dir: Path,
    capture_plan: Mapping[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    import torch

    base_inputs = []
    traces = []
    image_paths_by_row = []
    slot_masks_by_row = []
    slot_audits = []
    question_tokens = []
    for request in requests:
        paths = [_resolve_asset(battery_dir, ref) for ref in request["inputs"]]
        inputs, trace = _prepare_base(
            runtime=runtime, processor=processor, request=request, image_paths=paths, device=device
        )
        masks, slot_audit = derive_image_slot_masks(inputs=inputs, model=model, processor=processor)
        token = find_final_question_token(
            scoring._input_ids_list(inputs), tokenizer, str(request["user_prompt"])
        )
        if slot_audit.get("status") != "pass" or token.get("status") != "pass":
            raise RuntimeError(
                f"slot/question boundary failure: slots={slot_audit.get('reason')} "
                f"question={token.get('reason')}"
            )
        base_inputs.append(inputs)
        traces.append(trace)
        image_paths_by_row.append(paths)
        slot_masks_by_row.append(masks)
        slot_audits.append(slot_audit)
        question_tokens.append(token)
    collated = _collate_inputs(base_inputs)
    collector = SlotHookCollector(
        model, capture_plan, batch_size=len(requests), slot_audits=slot_audits
    )
    decoder_collector = DecoderStateCollector(
        model,
        decoder_container=str(capture_plan["decoder_container"]),
        expected_layers=int(capture_plan["expected_decoder_layers"]),
        batch_size=len(requests),
        slot_masks=slot_masks_by_row,
        question_tokens=question_tokens,
    )
    gates._reset_cuda_peak_memory()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = gates._forward_model(model, collated)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        collector.close()
        decoder_collector.close()
    forward_elapsed = time.perf_counter() - started
    if set(collector.records) != set(capture_plan["hook_names"]):
        raise RuntimeError("one or more required architecture hooks did not fire")
    if any(row.get("status") != "pass" for row in collector.records.values()):
        raise RuntimeError(f"stage slot mapping failed: {collector.records}")
    decoder_audit = decoder_collector.audit()
    if decoder_audit.get("status") != "pass":
        raise RuntimeError(f"direct decoder-state capture failed: {decoder_audit}")
    del output
    candidates = {
        label: _score_candidate_batch(
            model=model, base_inputs=base_inputs, tokenizer=tokenizer, candidate_label=label
        )
        for label in ("A", "B")
    }
    peak = gates._cuda_peak_memory()
    results = []
    for index, request in enumerate(requests):
        decoder = decoder_collector.tensors_by_sample[index]
        a_row = candidates["A"][index]
        b_row = candidates["B"][index]
        if a_row.get("status") != "pass" or b_row.get("status") != "pass":
            raise RuntimeError("A/B teacher-forced candidate score failed")
        a_minus_b = (float(a_row["sum_logprob"]) - float(b_row["sum_logprob"])) / math.log(2.0)
        tensors = {**decoder, **collector.tensors_by_sample[index]}
        metadata = {
            "status": "pass",
            "arm": request["arm"],
            "trial_id": request["trial_id"],
            "item_id": request["item_id"],
            "request_id": request["request_id"],
            "public_pair_id": request["public_pair_id"],
            "counterbalance_pair_index": request["counterbalance_pair_index"],
            "public_order_index": request["public_order_index"],
            "crop_order": request["crop_order"],
            "prompt_sha256": hashlib.sha256(request["user_prompt"].encode("utf-8")).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(request["system_prompt"].encode("utf-8")).hexdigest(),
            "image_refs": request["inputs"],
            "input_preparation_trace": traces[index],
            "question_token": question_tokens[index],
            "image_slot_audit": slot_audits[index],
            "stage_slot_audits": collector.records,
            "decoder_capture": decoder_audit,
            "teacher_forced": {
                "status": "pass",
                "margin_A_minus_B_bits": a_minus_b,
                "candidates": {"A": a_row, "B": b_row},
                "hidden_answer_opened": False,
            },
            "geometry_features": _geometry_features(image_paths_by_row[index]),
            "forward_elapsed_s_per_microbatch": forward_elapsed,
            "effective_microbatch_size": len(requests),
            "cuda_peak": peak,
            "tensor_count": len(tensors),
            "persisted_dtype": "float16",
            "full_token_tensors_persisted": False,
            "unpatched_slot_identity_contract_for_job_c": (
                "Job C evidence restoration may change slot 0 only; crop-order patches may "
                "change mapped crop slots only. All other positions require exact equality audits."
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        }
        results.append((metadata, tensors))
    del collated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def _completed_keys(capture_root: Path) -> set[tuple[str, str]]:
    result = set()
    seen = set()
    for row in _iter_jsonl(capture_root / "capture_index.jsonl"):
        key = (str(row.get("arm")), str(row.get("trial_id")))
        if key in seen:
            raise ValueError(f"duplicate capture index key: {key}")
        seen.add(key)
        tensor_path = row.get("tensor_path")
        if (
            row.get("status") == "pass"
            and tensor_path
            and (capture_root / str(tensor_path)).is_file()
        ):
            result.add(key)
    return result


def _flush_chunk(
    *,
    capture_root: Path,
    model_key: str,
    chunk_index: int,
    tensors: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    row_keys = [(str(row.get("arm")), str(row.get("trial_id"))) for row in rows]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("capture chunk contains duplicate arm/trial keys")
    existing_keys = {
        (str(row.get("arm")), str(row.get("trial_id")))
        for row in _iter_jsonl(capture_root / "capture_index.jsonl")
    }
    collisions = sorted(set(row_keys).intersection(existing_keys))
    if collisions:
        raise ValueError(f"capture chunk would overwrite result keys: {collisions[:5]}")
    base = capture_root / "chunks" / f"chunk_{chunk_index:05d}"
    tensor_path = legacy._save_tensor_chunk(
        chunk_path=base, tensors=tensors, tensor_format="safetensors"
    )
    tensor_sha = _sha256_file(Path(tensor_path))
    indexed = []
    for row in rows:
        copied = dict(row)
        copied["tensor_path"] = os.path.relpath(tensor_path, capture_root)
        copied["tensor_file_sha256"] = tensor_sha
        indexed.append(copied)
    legacy._append_jsonl(capture_root / "capture_index.jsonl", indexed)
    _write_json(
        capture_root / "chunks" / f"chunk_{chunk_index:05d}.manifest.json",
        {
            "kind": "phase2_reanchored_jobB_slot_pooled_capture_chunk",
            "model_key": model_key,
            "chunk_index": chunk_index,
            "tensor_path": os.path.relpath(tensor_path, capture_root),
            "tensor_file_sha256": tensor_sha,
            "row_count": len(rows),
            "tensor_count": len(tensors),
            "persisted_dtype": "float16",
            "full_token_tensors_persisted": False,
            "rows": [
                {
                    "arm": row["arm"],
                    "trial_id": row["trial_id"],
                    "tensor_prefix": row["tensor_prefix"],
                }
                for row in rows
            ],
        },
    )


def _schedule_profile_rows(
    model_reports: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if len(model_reports) != len(FROZEN_MODEL_IDENTITIES):
        raise ValueError("balanced scheduling requires exactly four model reports")
    profiles: dict[str, dict[str, Any]] = {}
    ranks: set[int] = set()
    for report in model_reports:
        model_key = str(report.get("model_key") or "")
        rank = int(report.get("canonical_rank", report.get("rank", -1)))
        rate = float(report.get("rows_per_second") or 0.0)
        load = float(report.get("load_elapsed_s") or 0.0)
        microbatch = int(report.get("effective_microbatch_size") or 0)
        if not model_key or model_key in profiles:
            raise ValueError("balanced scheduling model keys are missing or duplicated")
        if rank not in range(4) or rank in ranks:
            raise ValueError("balanced scheduling canonical ranks must be 0..3 exactly once")
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"balanced scheduling requires positive throughput for {model_key}")
        if not math.isfinite(load) or load < 0.0:
            raise ValueError(f"balanced scheduling requires nonnegative load time for {model_key}")
        if microbatch <= 0:
            raise ValueError(f"balanced scheduling requires a tested microbatch for {model_key}")
        profiles[model_key] = {
            "model_key": model_key,
            "canonical_rank": rank,
            "rows_per_second": rate,
            "load_elapsed_s": load,
            "effective_microbatch_size": microbatch,
        }
        ranks.add(rank)
    expected_keys = {row[0] for row in FROZEN_MODEL_IDENTITIES}
    if set(profiles) != expected_keys or ranks != set(range(4)):
        raise ValueError("balanced scheduling model/rank topology drifted")
    return profiles


def _projected_assignment_seconds(
    profile: Mapping[str, Any], *, selection_count: int, arm_count: int
) -> tuple[float, float, float]:
    load_s = float(profile["load_elapsed_s"])
    capture_s = selection_count * arm_count / float(profile["rows_per_second"])
    return load_s, capture_s, load_s + capture_s


def build_balanced_full_schedule(
    model_reports: Sequence[Mapping[str, Any]],
    *,
    selection_count: int = MECHANISM_TRIAL_COUNT,
    arm_count: int = len(ARMS),
) -> dict[str, Any]:
    """Freeze the best exact two-wave schedule from canary-only measurements.

    Each model begins on its canonical rank. A rank that finishes its canonical
    model early may then start a fresh process, load one paired slower model, and
    capture a contiguous tail aligned to a complete two-order frame pair.
    """
    if selection_count < COUNTERBALANCE_PAIR_SIZE or (
        selection_count % COUNTERBALANCE_PAIR_SIZE
    ):
        raise ValueError("balanced scheduling requires an even trial count")
    if arm_count <= 0:
        raise ValueError("balanced scheduling requires at least one arm")
    profiles = _schedule_profile_rows(model_reports)
    by_rank = {row["canonical_rank"]: row for row in profiles.values()}
    rank_pairings = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    pair_count = selection_count // COUNTERBALANCE_PAIR_SIZE

    def full_seconds(profile: Mapping[str, Any]) -> float:
        return _projected_assignment_seconds(
            profile, selection_count=selection_count, arm_count=arm_count
        )[2]

    best: tuple[tuple[Any, ...], tuple[tuple[int, int, int], ...]] | None = None
    for pairing in rank_pairings:
        pair_options: list[list[tuple[int, int, int]]] = []
        for left_rank, right_rank in pairing:
            ordered = sorted(
                (left_rank, right_rank),
                key=lambda rank: (full_seconds(by_rank[rank]), rank),
            )
            fast_rank, slow_rank = ordered
            options = [
                (fast_rank, slow_rank, helper_pair_count)
                for helper_pair_count in range(pair_count)
            ]
            pair_options.append(options)
        for selected_options in itertools.product(*pair_options):
            rank_runtime = {
                rank: full_seconds(by_rank[rank]) for rank in range(4)
            }
            for fast_rank, slow_rank, helper_pair_count in selected_options:
                if helper_pair_count == 0:
                    continue
                helper_count = helper_pair_count * COUNTERBALANCE_PAIR_SIZE
                slow_prefix_count = selection_count - helper_count
                rank_runtime[slow_rank] = _projected_assignment_seconds(
                    by_rank[slow_rank],
                    selection_count=slow_prefix_count,
                    arm_count=arm_count,
                )[2]
                rank_runtime[fast_rank] += _projected_assignment_seconds(
                    by_rank[slow_rank],
                    selection_count=helper_count,
                    arm_count=arm_count,
                )[2]
            makespan = max(rank_runtime.values())
            busy_fraction = sum(rank_runtime.values()) / (4.0 * makespan)
            split_pairs = sum(option[2] for option in selected_options)
            canonical = tuple(sorted(selected_options))
            objective = (makespan, -busy_fraction, split_pairs, canonical)
            if best is None or objective < best[0]:
                best = (objective, canonical)
    if best is None:
        raise RuntimeError("balanced schedule optimizer produced no candidate")

    selected_options = best[1]
    split_by_slow_rank = {
        slow_rank: helper_pair_count * COUNTERBALANCE_PAIR_SIZE
        for _, slow_rank, helper_pair_count in selected_options
        if helper_pair_count > 0
    }
    helper_by_slow_rank = {
        slow_rank: fast_rank
        for fast_rank, slow_rank, helper_pair_count in selected_options
        if helper_pair_count > 0
    }
    assignments_by_rank: dict[int, list[dict[str, Any]]] = {
        rank: [] for rank in range(4)
    }
    for rank in range(4):
        profile = by_rank[rank]
        helper_count = split_by_slow_rank.get(rank, 0)
        stop = selection_count - helper_count
        shard_count = 2 if helper_count else 1
        assignment_id = f"{profile['model_key']}.shard-0-of-{shard_count}"
        assignments_by_rank[rank].append(
            {
                "assignment_id": assignment_id,
                "rank": rank,
                "wave_index": 0,
                "model_key": profile["model_key"],
                "canonical_rank": rank,
                "shard_index": 0,
                "shard_count": shard_count,
                "selection_start": 0,
                "selection_stop": stop,
                "counterbalance_pair_size": COUNTERBALANCE_PAIR_SIZE,
                "effective_microbatch_size": profile["effective_microbatch_size"],
                "output_namespace": f"{profile['model_key']}/shards/{assignment_id}",
                "fresh_process_required": True,
            }
        )
    for slow_rank, helper_count in split_by_slow_rank.items():
        helper_rank = helper_by_slow_rank[slow_rank]
        profile = by_rank[slow_rank]
        assignment_id = f"{profile['model_key']}.shard-1-of-2"
        assignments_by_rank[helper_rank].append(
            {
                "assignment_id": assignment_id,
                "rank": helper_rank,
                "wave_index": 1,
                "model_key": profile["model_key"],
                "canonical_rank": slow_rank,
                "shard_index": 1,
                "shard_count": 2,
                "selection_start": selection_count - helper_count,
                "selection_stop": selection_count,
                "counterbalance_pair_size": COUNTERBALANCE_PAIR_SIZE,
                "effective_microbatch_size": profile["effective_microbatch_size"],
                "output_namespace": f"{profile['model_key']}/shards/{assignment_id}",
                "fresh_process_required": True,
            }
        )

    rank_rows = []
    for rank in range(4):
        elapsed_s = 0.0
        projected_assignments = []
        for assignment in sorted(
            assignments_by_rank[rank], key=lambda row: int(row["wave_index"])
        ):
            profile = profiles[str(assignment["model_key"])]
            count = int(assignment["selection_stop"]) - int(
                assignment["selection_start"]
            )
            load_s, capture_s, duration_s = _projected_assignment_seconds(
                profile, selection_count=count, arm_count=arm_count
            )
            projected_assignments.append(
                {
                    **assignment,
                    "projected_capture_rows": count * arm_count,
                    "canary_rows_per_second": profile["rows_per_second"],
                    "projected_load_s": load_s,
                    "projected_capture_s": capture_s,
                    "projected_duration_s": duration_s,
                    "projected_start_s": elapsed_s,
                    "projected_finish_s": elapsed_s + duration_s,
                }
            )
            elapsed_s += duration_s
        rank_rows.append(
            {
                "rank": rank,
                "assignments": projected_assignments,
                "projected_runtime_s": elapsed_s,
            }
        )
    makespan_s = max(float(row["projected_runtime_s"]) for row in rank_rows)
    busy_s = sum(float(row["projected_runtime_s"]) for row in rank_rows)
    busy_fraction = busy_s / (4.0 * makespan_s)
    recommended_walltime_s = int(
        math.ceil(
            (
                makespan_s * (1.0 + FULL_WALLTIME_HEADROOM_FRACTION)
                + ANALYSIS_CLEANUP_RESERVE_SECONDS
            )
            / WALLTIME_ROUND_SECONDS
        )
        * WALLTIME_ROUND_SECONDS
    )
    recommended_walltime_s = max(MIN_FULL_WALLTIME_SECONDS, recommended_walltime_s)
    fits_walltime = (
        recommended_walltime_s <= FULL_WALLTIME_SECONDS
    )
    threshold_pass = (
        busy_fraction >= MIN_PROJECTED_AGGREGATE_BUSY_FRACTION and fits_walltime
    )
    payload = {
        "kind": BALANCED_SCHEDULE_KIND,
        "selection_count": selection_count,
        "arm_count": arm_count,
        "counterbalance_pair_size": COUNTERBALANCE_PAIR_SIZE,
        "construction": "two_wave_fast_rank_takes_contiguous_pair_aligned_slow_model_tail",
        "optimizer": "exact_three_pairings_all_pair_aligned_tail_splits_minimum_makespan",
        "adaptation_during_full_run": False,
        "every_model_starts_on_canonical_rank": True,
        "fresh_process_per_assignment": True,
        "exact_disjoint_union_required": True,
        "duplicate_capture_rows": 0,
        "storage_projection_multiplier": 1.0,
        "ranks": rank_rows,
        "projected_makespan_s": makespan_s,
        "projected_total_rank_busy_s": busy_s,
        "projected_aggregate_busy_fraction": busy_fraction,
        "projected_max_tail_idle_fraction": max(
            (makespan_s - float(row["projected_runtime_s"])) / makespan_s
            for row in rank_rows
        ),
        "minimum_required_projected_aggregate_busy_fraction": (
            MIN_PROJECTED_AGGREGATE_BUSY_FRACTION
        ),
        "configured_full_walltime_s": FULL_WALLTIME_SECONDS,
        "analysis_cleanup_reserve_s": ANALYSIS_CLEANUP_RESERVE_SECONDS,
        "walltime_headroom_fraction": FULL_WALLTIME_HEADROOM_FRACTION,
        "recommended_full_walltime_s_rounded_to_5_minutes": recommended_walltime_s,
        "fits_two_hour_walltime_with_reserve": fits_walltime,
        "threshold_status": "pass" if threshold_pass else "fail",
        "failure_action": (
            None
            if threshold_pass
            else "fail_closed_another_node_allocation_or_schedule_is_required"
        ),
    }
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def _schedule_assignments(schedule: Mapping[str, Any]) -> list[dict[str, Any]]:
    ranks = schedule.get("ranks")
    if not isinstance(ranks, list):
        raise ValueError("balanced schedule lacks rank rows")
    return [
        dict(assignment)
        for rank_row in ranks
        if isinstance(rank_row, Mapping)
        for assignment in rank_row.get("assignments", [])
        if isinstance(assignment, Mapping)
    ]


def _validate_balanced_full_schedule(
    schedule: Mapping[str, Any],
    *,
    model_reports: Sequence[Mapping[str, Any]],
    model_plan: Sequence[Mapping[str, Any]],
    require_passing: bool,
) -> dict[str, Any]:
    if schedule.get("kind") != BALANCED_SCHEDULE_KIND:
        raise ValueError("balanced schedule kind mismatch")
    observed_hash = str(schedule.get("sha256") or "")
    expected_hash = _canonical_sha256(
        {key: value for key, value in schedule.items() if key != "sha256"}
    )
    if observed_hash != expected_hash:
        raise ValueError("balanced schedule hash mismatch")
    if int(schedule.get("selection_count", -1)) != MECHANISM_TRIAL_COUNT:
        raise ValueError("balanced schedule selection count mismatch")
    if int(schedule.get("arm_count", -1)) != len(ARMS):
        raise ValueError("balanced schedule arm count mismatch")
    if int(schedule.get("counterbalance_pair_size", -1)) != COUNTERBALANCE_PAIR_SIZE:
        raise ValueError("balanced schedule pair alignment mismatch")
    ranks = schedule.get("ranks")
    if not isinstance(ranks, list) or {
        int(row.get("rank", -1)) for row in ranks if isinstance(row, Mapping)
    } != set(range(4)) or len(ranks) != 4:
        raise ValueError("balanced schedule must contain ranks 0..3 exactly once")
    plan_by_key = {str(row["model_key"]): dict(row) for row in model_plan}
    profiles = _schedule_profile_rows(model_reports)
    assignments = _schedule_assignments(schedule)
    assignment_ids = [str(row.get("assignment_id") or "") for row in assignments]
    namespaces = [str(row.get("output_namespace") or "") for row in assignments]
    if (
        not assignment_ids
        or len(assignment_ids) != len(set(assignment_ids))
        or len(namespaces) != len(set(namespaces))
    ):
        raise ValueError("balanced schedule has assignment/output collisions")
    for rank_row in ranks:
        rank = int(rank_row["rank"])
        rank_assignments = [
            row for row in assignments if int(row.get("rank", -1)) == rank
        ]
        waves = [int(row.get("wave_index", -1)) for row in rank_assignments]
        if waves not in ([0], [0, 1]):
            raise ValueError("balanced schedule rank waves are not sequential 0/1")
        canonical = rank_assignments[0]
        expected_model = next(
            str(row["model_key"])
            for row in model_plan
            if int(row["canonical_rank"]) == rank
        )
        if (
            canonical.get("model_key") != expected_model
            or int(canonical.get("selection_start", -1)) != 0
            or int(canonical.get("canonical_rank", -1)) != rank
        ):
            raise ValueError("balanced schedule canonical wave drifted")
    for assignment in assignments:
        model_key = str(assignment.get("model_key") or "")
        if model_key not in plan_by_key or model_key not in profiles:
            raise ValueError("balanced schedule references an unknown model")
        start = int(assignment.get("selection_start", -1))
        stop = int(assignment.get("selection_stop", -1))
        if not 0 <= start < stop <= MECHANISM_TRIAL_COUNT:
            raise ValueError("balanced schedule contains an invalid interval")
        if start % COUNTERBALANCE_PAIR_SIZE or stop % COUNTERBALANCE_PAIR_SIZE:
            raise ValueError("balanced schedule split is not counterbalance-pair aligned")
        expected_namespace = f"{model_key}/shards/{assignment['assignment_id']}"
        if assignment.get("output_namespace") != expected_namespace:
            raise ValueError("balanced schedule output namespace drifted")
        if assignment.get("fresh_process_required") is not True:
            raise ValueError("balanced schedule does not require fresh assignment processes")
        if int(assignment.get("effective_microbatch_size", 0)) != int(
            profiles[model_key]["effective_microbatch_size"]
        ):
            raise ValueError("balanced schedule microbatch binding drifted")
    for model_key in plan_by_key:
        intervals = sorted(
            (
                int(row["selection_start"]),
                int(row["selection_stop"]),
            )
            for row in assignments
            if row["model_key"] == model_key
        )
        cursor = 0
        for start, stop in intervals:
            if start != cursor:
                raise ValueError(f"balanced schedule overlap/gap for {model_key}")
            cursor = stop
        if cursor != MECHANISM_TRIAL_COUNT:
            raise ValueError(f"balanced schedule incomplete for {model_key}")
    rebuilt = build_balanced_full_schedule(model_reports)
    if schedule != rebuilt:
        raise ValueError("balanced schedule is not the deterministic canary-rate optimum")
    if require_passing and (
        schedule.get("threshold_status") != "pass"
        or float(schedule.get("projected_aggregate_busy_fraction") or 0.0)
        < MIN_PROJECTED_AGGREGATE_BUSY_FRACTION
        or schedule.get("fits_two_hour_walltime_with_reserve") is not True
    ):
        raise ValueError(
            "balanced schedule utilization/walltime gate failed; another node allocation "
            "or schedule is required"
        )
    return {str(row["assignment_id"]): row for row in assignments}


def _load_frozen_batch_plan(
    path: Path,
    model_key: str,
    *,
    model_plan_path: Path,
    rank: int,
    assignment_id: str,
    selection_start: int,
    selection_stop: int,
    wave_index: int,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    payload = _read_json(path.expanduser().resolve())
    if payload.get("kind") != FROZEN_BATCH_PLAN_KIND:
        raise ValueError("batch plan kind mismatch")
    if payload.get("status") != "pass":
        raise ValueError("batch plan did not pass its frozen canary gates")
    if payload.get("governing_prespec_sha256") != GOVERNING_PRESPEC_SHA256:
        raise ValueError("batch plan prespec hash mismatch")
    if payload.get("technical_only_selection") is not True:
        raise ValueError("batch plan is not declared technical-only")
    expected_bindings = {
        "model_plan_sha256": _sha256_file(model_plan_path.expanduser().resolve()),
        "runner_code_sha256": _sha256_file(Path(__file__).resolve()),
        "sif_sha256": os.environ.get("SIF_SHA256"),
        "release_manifest_sha256": os.environ.get("RELEASE_MANIFEST_SHA256"),
        "mechanism_manifest_sha256": os.environ.get("MECHANISM_MANIFEST_SHA256"),
        "public_manifest_sha256": os.environ.get("PUBLIC_MANIFEST_SHA256"),
    }
    for field, expected in expected_bindings.items():
        observed = payload.get(field)
        if not expected or observed != expected:
            raise ValueError(f"batch plan {field} mismatch")
    current_code = _code_bundle_binding()
    if payload.get("code_bundle") != current_code["files"]:
        raise ValueError("batch plan code bundle manifest mismatch")
    if payload.get("code_bundle_sha256") != current_code["sha256"]:
        raise ValueError("batch plan code bundle hash mismatch")
    source_summary = payload.get("source_canary_summary")
    if not isinstance(source_summary, Mapping):
        raise ValueError("batch plan lacks its source canary summary")
    if payload.get("source_canary_summary_sha256") != _canonical_sha256(source_summary):
        raise ValueError("batch plan source canary summary hash mismatch")
    if source_summary.get("overall_status") != "pass":
        raise ValueError("batch plan source canary summary did not pass")
    plan = _parse_model_plan(model_plan_path)
    expected_revisions = {
        str(row["model_key"]): str(row["revision"])
        for row in plan
    }
    source_models = source_summary.get("models")
    source_models_by_key = {
        str(row.get("model_key") or ""): row
        for row in source_models
        if isinstance(row, Mapping)
    } if isinstance(source_models, list) else {}
    if (
        not isinstance(source_models, list)
        or len(source_models) != len(expected_revisions)
        or set(source_models_by_key) != set(expected_revisions)
        or any(
            str(row.get("revision") or "") != expected_revisions[model_key]
            or not isinstance(row.get("question_token_canary_gate"), Mapping)
            or row["question_token_canary_gate"].get("status") != "pass"
            or row["question_token_canary_gate"].get("contract")
            != QUESTION_TOKEN_CONTRACT
            for model_key, row in source_models_by_key.items()
        )
    ):
        raise ValueError(
            "batch plan lacks exact model/revision-bound passing tokenizer canary gates"
        )
    rows = payload.get("models")
    if not isinstance(rows, list):
        raise ValueError("batch plan lacks model profiles")
    profiles_by_key = {
        str(row.get("model_key") or ""): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(profiles_by_key) != {str(row["model_key"]) for row in plan}:
        raise ValueError("batch plan model-profile set drifted")
    for plan_row in plan:
        profile = profiles_by_key[str(plan_row["model_key"])]
        selected_batch = int(profile.get("microbatch_size") or 0)
        candidate_cap = int(plan_row["full_microbatch_hard_cap"])
        if (
            selected_batch <= 0
            or selected_batch != int(profile.get("effective_microbatch_size") or 0)
            or selected_batch > candidate_cap
            or int(profile.get("tested_candidate_cap") or 0) != candidate_cap
        ):
            raise ValueError(
                f"batch plan exceeds or misstates tested cap for {plan_row['model_key']}"
            )
    schedule = payload.get("schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("batch plan lacks the balanced schedule")
    if payload.get("schedule_sha256") != schedule.get("sha256"):
        raise ValueError("batch plan schedule hash binding mismatch")
    gate = payload.get("utilization_tail_gate")
    if (
        not isinstance(gate, Mapping)
        or gate.get("status") != "pass"
        or float(gate.get("projected_aggregate_busy_fraction") or 0.0)
        != float(schedule.get("projected_aggregate_busy_fraction") or -1.0)
        or float(gate.get("minimum_projected_aggregate_busy_fraction") or 0.0)
        != MIN_PROJECTED_AGGREGATE_BUSY_FRACTION
        or gate.get("fits_two_hour_walltime_with_reserve") is not True
    ):
        raise ValueError("batch plan utilization/tail gate binding mismatch")
    assignments = _validate_balanced_full_schedule(
        schedule,
        model_reports=rows,
        model_plan=plan,
        require_passing=True,
    )
    scheduled = assignments.get(str(assignment_id))
    expected_assignment = {
        "rank": int(rank),
        "model_key": str(model_key),
        "selection_start": int(selection_start),
        "selection_stop": int(selection_stop),
        "wave_index": int(wave_index),
    }
    if scheduled is None or any(
        scheduled.get(field) != value for field, value in expected_assignment.items()
    ):
        raise ValueError("worker assignment does not match the hash-bound schedule")
    match = next(
        (row for row in rows if isinstance(row, dict) and row.get("model_key") == model_key),
        None,
    )
    if not match or int(match.get("microbatch_size") or 0) <= 0:
        raise ValueError(f"batch plan lacks {model_key}")
    candidate = next(
        int(row["full_microbatch_hard_cap"])
        for row in plan
        if row["model_key"] == model_key
    )
    selected = int(match["microbatch_size"])
    if selected > candidate or int(match.get("tested_candidate_cap") or 0) != candidate:
        raise ValueError(f"batch plan exceeds or misstates tested cap for {model_key}")
    if int(scheduled.get("effective_microbatch_size") or 0) != selected:
        raise ValueError("schedule and batch-profile microbatch drifted")
    return selected, payload, dict(scheduled)


def _selected_requests(
    mechanism_dir: Path,
    run_dir: Path,
    prespec_path: Path | None,
    model_plan: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selection = _read_json(run_dir / "selection_manifest.json")
    if selection.get("governing_prespec_sha256") != GOVERNING_PRESPEC_SHA256:
        raise ValueError("selection manifest prespec hash mismatch")
    _, rows_by_arm, audit = load_and_validate_mechanism_subset(
        mechanism_dir, verify_files=False, prespec_path=prespec_path
    )
    selected_ids = list(selection.get("selected_trial_ids") or [])
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selection contains duplicate trial IDs")
    frame_count = int(selection.get("frame_count") or 0)
    if frame_count <= 0 or len(selected_ids) != frame_count * 2:
        raise ValueError("selection frame/trial counts drifted")
    expected_storage = estimate_storage(model_plan, frame_count)
    if selection.get("storage_projection") != expected_storage:
        raise ValueError("selection storage projection drifted from the current model plan")
    if expected_storage["hard_ceiling_status"] != "pass":
        raise ValueError("selection storage projection exceeds the hard ceiling")
    result = {}
    for arm, rows in rows_by_arm.items():
        by_id = {row["item_id"]: row for row in rows}
        if any(trial_id not in by_id for trial_id in selected_ids):
            raise ValueError(f"selection does not map exactly to {arm}")
        result[arm] = [by_id[trial_id] for trial_id in selected_ids]
    return result, {"selection": selection, "mechanism_subset": audit}


def _capture_assignment_binding(
    *,
    mode: str,
    model_row: Mapping[str, Any],
    assignment: Mapping[str, Any],
    selected_trial_ids: Sequence[str],
    batch_plan_sha256: str | None,
    schedule_sha256: str | None,
) -> dict[str, Any]:
    return {
        "kind": "phase2_reanchored_jobB_capture_assignment_v1",
        "mode": mode,
        "model_key": str(model_row["model_key"]),
        "model_id": str(model_row["model_id"]),
        "revision": str(model_row["revision"]),
        "canonical_rank": int(model_row["canonical_rank"]),
        "worker_rank": int(assignment["rank"]),
        "assignment_id": str(assignment["assignment_id"]),
        "wave_index": int(assignment["wave_index"]),
        "selection_start": int(assignment["selection_start"]),
        "selection_stop": int(assignment["selection_stop"]),
        "selected_trial_ids_sha256": _canonical_sha256(selected_trial_ids),
        "schedule_sha256": schedule_sha256,
        "batch_plan_sha256": batch_plan_sha256,
        "governing_prespec_sha256": GOVERNING_PRESPEC_SHA256,
        "output_namespace": str(assignment["output_namespace"]),
    }


def _validate_resume_capture(
    *,
    out_dir: Path,
    capture_root: Path,
    expected_binding: Mapping[str, Any],
    expected_keys: set[tuple[str, str]],
    resume: bool,
) -> set[tuple[str, str]]:
    binding_path = out_dir / "assignment_manifest.json"
    artifact_paths = (
        binding_path,
        out_dir / "capture_summary.json",
        out_dir / "progress.json",
        out_dir / "capture_plan.json",
        capture_root / "capture_index.jsonl",
    )
    existing = [path for path in artifact_paths if path.exists()]
    if existing and not resume:
        raise ValueError("assignment outputs already exist; resume or use a fresh run root")
    if binding_path.is_file():
        observed = _read_json(binding_path)
        if observed != dict(expected_binding):
            raise ValueError("resume assignment binding drifted")
    elif existing:
        raise ValueError("resume artifacts exist without an assignment binding manifest")
    else:
        _write_json(binding_path, dict(expected_binding))
    if not resume:
        return set()
    index_path = capture_root / "capture_index.jsonl"
    if not index_path.is_file():
        return set()
    completed: set[tuple[str, str]] = set()
    tensor_hashes: dict[Path, str] = {}
    tensor_prefixes: set[str] = set()
    for row in _iter_jsonl(index_path):
        if row.get("status") != "pass":
            raise ValueError("resume capture index contains a non-passing row")
        key = (str(row.get("arm") or ""), str(row.get("trial_id") or ""))
        if key not in expected_keys:
            raise ValueError(f"resume capture key is outside its frozen shard: {key}")
        if key in completed:
            raise ValueError(f"duplicate resume capture key: {key}")
        for field in (
            "assignment_id",
            "worker_rank",
            "wave_index",
            "selection_start",
            "selection_stop",
            "schedule_sha256",
            "batch_plan_sha256",
        ):
            if row.get(field) != expected_binding.get(field):
                raise ValueError(f"resume capture row binding drifted at {field}")
        prefix = str(row.get("tensor_prefix") or "")
        if not prefix or prefix in tensor_prefixes:
            raise ValueError("resume tensor prefixes are missing or duplicated")
        tensor_prefixes.add(prefix)
        tensor_rel = str(row.get("tensor_path") or "")
        tensor_path = (capture_root / tensor_rel).resolve()
        resolved_root = capture_root.resolve()
        if (
            not tensor_rel
            or tensor_path == resolved_root
            or resolved_root not in tensor_path.parents
            or not tensor_path.is_file()
        ):
            raise ValueError(f"resume tensor path is missing or unsafe: {tensor_rel}")
        observed_hash = tensor_hashes.get(tensor_path)
        if observed_hash is None:
            observed_hash = _sha256_file(tensor_path)
            tensor_hashes[tensor_path] = observed_hash
        if row.get("tensor_file_sha256") != observed_hash:
            raise ValueError(f"resume tensor hash mismatch: {tensor_path}")
        completed.add(key)
    return completed


def run_model(args: argparse.Namespace) -> int:
    runtime_hashes = {
        name.lower(): _required_sha256_env(name)
        for name in (
            "SIF_SHA256",
            "RELEASE_MANIFEST_SHA256",
            "MECHANISM_MANIFEST_SHA256",
            "PUBLIC_MANIFEST_SHA256",
        )
    }
    run_dir = args.run_dir.expanduser().resolve()
    mechanism_dir = args.mechanism_dir.expanduser().resolve()
    plan = _parse_model_plan(args.model_plan_file)
    rank = int(args.rank)
    if not 0 <= rank < len(plan):
        raise ValueError("rank is outside the frozen four-model plan")
    requests_by_arm, source_audit = _selected_requests(
        mechanism_dir, run_dir, args.prespec_path, plan
    )
    asset_root = Path(source_audit["mechanism_subset"]["asset_root"])
    mode = source_audit["selection"]["mode"]
    if mode == "full":
        if not args.batch_plan_file:
            raise ValueError("full mode requires a canary-frozen batch plan")
        if any(
            value is None
            for value in (
                args.model_key,
                args.assignment_id,
                args.selection_start,
                args.selection_stop,
                args.wave_index,
            )
        ):
            raise ValueError("full mode requires an exact frozen schedule assignment")
        model_rows = {row["model_key"]: row for row in plan}
        if args.model_key not in model_rows:
            raise ValueError("full assignment references an unknown model")
        model_row = model_rows[str(args.model_key)]
        microbatch_size, batch_plan, assignment = _load_frozen_batch_plan(
            args.batch_plan_file,
            str(args.model_key),
            model_plan_path=args.model_plan_file,
            rank=rank,
            assignment_id=str(args.assignment_id),
            selection_start=int(args.selection_start),
            selection_stop=int(args.selection_stop),
            wave_index=int(args.wave_index),
        )
        batch_plan_sha = _sha256_file(args.batch_plan_file.expanduser().resolve())
        schedule_sha = str(batch_plan["schedule_sha256"])
        selection_start = int(assignment["selection_start"])
        selection_stop = int(assignment["selection_stop"])
        out_dir = run_dir / str(assignment["output_namespace"])
    else:
        if args.batch_plan_file or any(
            value is not None
            for value in (
                args.model_key,
                args.assignment_id,
                args.selection_start,
                args.selection_stop,
                args.wave_index,
            )
        ):
            raise ValueError("canary mode may not override its canonical assignment")
        model_row = plan[rank]
        microbatch_size = int(model_row["canary_microbatch_size"])
        assignment = {
            "assignment_id": f"{model_row['model_key']}.canary",
            "rank": rank,
            "wave_index": 0,
            "model_key": model_row["model_key"],
            "canonical_rank": rank,
            "shard_index": 0,
            "shard_count": 1,
            "selection_start": 0,
            "selection_stop": len(next(iter(requests_by_arm.values()))),
            "counterbalance_pair_size": COUNTERBALANCE_PAIR_SIZE,
            "effective_microbatch_size": microbatch_size,
            "output_namespace": model_row["model_key"],
            "fresh_process_required": True,
        }
        batch_plan = {
            "source": "model_plan_canary_candidate",
            "technical_only_selection": True,
        }
        batch_plan_sha = None
        schedule_sha = None
        selection_start = 0
        selection_stop = int(assignment["selection_stop"])
        out_dir = run_dir / model_row["model_key"]
    model_key = str(model_row["model_key"])
    out_dir = out_dir.resolve()
    if run_dir != out_dir and run_dir not in out_dir.parents:
        raise ValueError("assignment output namespace escapes the run directory")
    capture_root = out_dir / "capture"
    capture_root.mkdir(parents=True, exist_ok=True)
    requests_by_arm = {
        arm: rows[selection_start:selection_stop]
        for arm, rows in requests_by_arm.items()
    }
    selected_trial_ids = [row["trial_id"] for row in requests_by_arm[ARMS[0]]]
    if any(
        [row["trial_id"] for row in requests_by_arm[arm]] != selected_trial_ids
        for arm in ARMS
    ):
        raise ValueError("assignment slices do not share the same trial order across arms")
    expected_keys = {
        (arm, row["trial_id"])
        for arm in ARMS
        for row in requests_by_arm[arm]
    }
    assignment_binding = _capture_assignment_binding(
        mode=mode,
        model_row=model_row,
        assignment=assignment,
        selected_trial_ids=selected_trial_ids,
        batch_plan_sha256=batch_plan_sha,
        schedule_sha256=schedule_sha,
    )
    capture_row_binding = {
        "assignment_id": assignment_binding["assignment_id"],
        "worker_rank": assignment_binding["worker_rank"],
        "wave_index": assignment_binding["wave_index"],
        "selection_start": assignment_binding["selection_start"],
        "selection_stop": assignment_binding["selection_stop"],
        "schedule_sha256": assignment_binding["schedule_sha256"],
        "batch_plan_sha256": assignment_binding["batch_plan_sha256"],
    }
    resume_sanitize = None
    resume_error_sanitize = None
    if args.resume:
        resume_sanitize = legacy._sanitize_jsonl(capture_root / "capture_index.jsonl")
        resume_error_sanitize = legacy._sanitize_jsonl(
            capture_root / "capture_errors.jsonl"
        )
    completed = _validate_resume_capture(
        out_dir=out_dir,
        capture_root=capture_root,
        expected_binding=assignment_binding,
        expected_keys=expected_keys,
        resume=bool(args.resume),
    )
    all_assets = [
        _resolve_asset(asset_root, ref)
        for arm in ARMS
        for request in requests_by_arm[arm]
        for ref in request["inputs"]
    ]
    prefetch = _prefetch_assets(all_assets, workers=int(args.prefetch_workers))
    summary: dict[str, Any] = {
        "kind": "phase2_reanchored_jobB_model_capture",
        "model_key": model_key,
        "model_id": model_row["model_id"],
        "revision": model_row["revision"],
        "rank": rank,
        "canonical_rank": int(model_row["canonical_rank"]),
        "world_size": int(args.world_size),
        "process_id": os.getpid(),
        "process_invocation_id": (
            str(getattr(args, "process_invocation_id", ""))
            if getattr(args, "process_invocation_id", "")
            else f"local-pid-{os.getpid()}-{assignment['assignment_id']}"
        ),
        "assignment": assignment,
        "assignment_binding": assignment_binding,
        "assignment_id": assignment_binding["assignment_id"],
        "worker_rank": assignment_binding["worker_rank"],
        "wave_index": assignment_binding["wave_index"],
        "selection_start": assignment_binding["selection_start"],
        "selection_stop": assignment_binding["selection_stop"],
        "schedule_sha256": assignment_binding["schedule_sha256"],
        "dtype": "bf16",
        "persisted_dtype": "float16",
        "runtime_hash_bindings": runtime_hashes,
        "source_audit": source_audit,
        "expected_capture_rows": sum(len(rows) for rows in requests_by_arm.values()),
        "effective_microbatch_size": microbatch_size,
        "batch_plan": {
            "kind": batch_plan.get("kind"),
            "status": batch_plan.get("status"),
            "source": batch_plan.get("source"),
            "schedule_sha256": schedule_sha,
        },
        "batch_plan_sha256": batch_plan_sha,
        "resume_sanitize": resume_sanitize,
        "resume_error_log_sanitize": resume_error_sanitize,
        "resume_rows_validated": len(completed),
        "prefetch": prefetch,
        "governing_prespec_sha256": GOVERNING_PRESPEC_SHA256,
        "started_at": _utc_now(),
        "claim_boundary": CLAIM_BOUNDARY,
        "runtime_versions": gates._runtime_versions(),
        "model_snapshot": gates._snapshot_provenance(
            model_row["model_id"], model_row["revision"], local_files_only=args.local_files_only
        ),
    }
    _write_json(out_dir / "capture_summary.json", summary)
    runtime = gates._load_job0_runtime(args.repo_dir.expanduser().resolve())
    load_started = time.perf_counter()
    try:
        model, processor, device, adapter, tokenizer = runtime._load_model(
            model_id=model_row["model_id"],
            dtype_name="bf16",
            trust_remote_code=True,
            local_files_only=args.local_files_only,
            revision=model_row["revision"],
            force_single_gpu_device_map=True,
        )
        tokenizer = _require_runtime_tokenizer(processor, tokenizer)
    except Exception as exc:
        summary.update(
            {
                "overall_status": "fail",
                "load_elapsed_s": time.perf_counter() - load_started,
                "load_error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(out_dir / "capture_summary.json", summary)
        return 1
    summary["load_elapsed_s"] = time.perf_counter() - load_started
    summary["adapter"] = adapter
    summary["device"] = str(device)
    summary["model_placement"] = gates._model_placement_check(model, dtype_name="bf16")
    summary["config"] = gates._summarize_config(model)
    summary["processor"] = gates._summarize_processor(processor)
    observed_config = summary["config"]
    observed_hidden_size, observed_num_hidden_layers = _observed_language_dimensions(
        observed_config
    )
    observed_vision_hidden_size, observed_vision_source = _observed_vision_hidden_size(
        observed_config, model, model_row
    )
    summary["observed_language_dimensions"] = {
        "hidden_size": observed_hidden_size,
        "num_hidden_layers": observed_num_hidden_layers,
        "source": (
            "text_config"
            if isinstance(observed_config.get("text_config"), Mapping)
            else "model_config"
        ),
    }
    summary["observed_vision_dimensions"] = {
        "hidden_size": observed_vision_hidden_size,
        "source": observed_vision_source,
    }
    _write_json(out_dir / "capture_summary.json", summary)
    if (
        observed_hidden_size != int(model_row["hidden_size"])
        or observed_num_hidden_layers != int(model_row["num_hidden_layers"])
        or observed_vision_hidden_size != int(model_row["vision_hidden_size"])
    ):
        detail = {
            "expected_hidden_size": int(model_row["hidden_size"]),
            "expected_num_hidden_layers": int(model_row["num_hidden_layers"]),
            "observed_hidden_size": observed_hidden_size,
            "observed_num_hidden_layers": observed_num_hidden_layers,
            "expected_vision_hidden_size": int(model_row["vision_hidden_size"]),
            "observed_vision_hidden_size": observed_vision_hidden_size,
            "observed_vision_source": observed_vision_source,
        }
        summary.update(
            {
                "overall_status": "fail",
                "dimension_error": detail,
            }
        )
        _write_json(out_dir / "capture_summary.json", summary)
        raise RuntimeError(
            f"live model dimensions drift from storage/model plan: {detail}"
        )
    if args.post_load_barrier_dir and int(assignment["wave_index"]) != 0:
        raise ValueError("only canonical wave-0 assignments may enter the load barrier")
    if args.post_load_barrier_dir:
        summary["post_load_barrier"] = gates._file_barrier(
            barrier_dir=args.post_load_barrier_dir.expanduser().resolve(),
            rank=rank,
            world_size=int(args.world_size),
            expected_count=4,
            model_key=model_key,
            stage="post_load",
            marker_status="ready",
            timeout_s=float(args.post_load_barrier_timeout_s),
        )
    module_map = gates._module_map(model, max_modules=int(args.max_modules))
    _write_json(out_dir / "module_map.json", module_map)
    family = gates._model_family(getattr(model, "config", None), model_row["model_id"])
    capture_plan = build_capture_plan(module_map["modules"], family, model_row)
    _write_json(out_dir / "capture_plan.json", capture_plan)
    if capture_plan["status"] != "pass":
        raise RuntimeError(f"capture plan unsupported: {capture_plan['missing_required_sites']}")

    chunk_index = legacy._next_chunk_index(capture_root / "chunks")
    tensors_buffer: dict[str, Any] = {}
    rows_buffer: list[dict[str, Any]] = []
    rows_done = rows_failed = rows_skipped = 0
    started = time.perf_counter()
    peak_allocated = 0
    peak_reserved = 0
    for arm in ARMS:
        pending = [
            request for request in requests_by_arm[arm]
            if (arm, request["trial_id"]) not in completed
        ]
        rows_skipped += len(requests_by_arm[arm]) - len(pending)
        for offset in range(0, len(pending), microbatch_size):
            batch = pending[offset : offset + microbatch_size]
            try:
                captured = capture_microbatch(
                    runtime=runtime,
                    model=model,
                    model_id=model_row["model_id"],
                    processor=processor,
                    tokenizer=tokenizer,
                    device=device,
                    requests=batch,
                    battery_dir=asset_root,
                    capture_plan=capture_plan,
                )
            except Exception as exc:
                failures = [
                    {
                            "status": "fail",
                            "arm": arm,
                            "trial_id": request["trial_id"],
                            "item_id": request["item_id"],
                            "request_id": request["request_id"],
                            "public_pair_id": request["public_pair_id"],
                            "counterbalance_pair_index": request[
                                "counterbalance_pair_index"
                            ],
                            "public_order_index": request["public_order_index"],
                            "error": f"{type(exc).__name__}: {exc}",
                            "claim_boundary": CLAIM_BOUNDARY,
                            **capture_row_binding,
                        }
                    for request in batch
                ]
                if args.strict:
                    legacy._append_jsonl(capture_root / "capture_errors.jsonl", failures)
                    rows_failed += len(failures)
                    summary.update(
                        {
                            "overall_status": "fail",
                            "first_capture_error": failures[0]["error"],
                            "rows_done": rows_done,
                            "rows_failed": rows_failed,
                            "rows_skipped": rows_skipped,
                            "ended_at": _utc_now(),
                        }
                    )
                    _write_json(out_dir / "capture_summary.json", summary)
                    _write_json(
                        out_dir / "progress.json",
                        {**summary, "status": "fail"},
                    )
                    raise
                captured = [(metadata, {}) for metadata in failures]
            for metadata, tensors in captured:
                metadata.update(capture_row_binding)
                prefix = (
                    f"row_{_stable_key(model_key, assignment['assignment_id'], arm, metadata['trial_id'])[:20]}"
                )
                metadata["tensor_prefix"] = prefix
                if metadata["status"] == "pass":
                    rows_done += 1
                    tensors_buffer.update(
                        {
                            f"{prefix}__{legacy._safe_tensor_key(name)}": tensor
                            for name, tensor in tensors.items()
                        }
                    )
                    peak = metadata.get("cuda_peak", {})
                    peak_allocated = max(
                        peak_allocated, int(peak.get("max_memory_allocated") or 0)
                    )
                    peak_reserved = max(
                        peak_reserved, int(peak.get("max_memory_reserved") or 0)
                    )
                else:
                    rows_failed += 1
                    legacy._append_jsonl(
                        capture_root / "capture_errors.jsonl", [metadata]
                    )
                    continue
                rows_buffer.append(metadata)
            if len(rows_buffer) >= int(args.chunk_size):
                _flush_chunk(
                    capture_root=capture_root,
                    model_key=model_key,
                    chunk_index=chunk_index,
                    tensors=tensors_buffer,
                    rows=rows_buffer,
                )
                chunk_index += 1
                tensors_buffer = {}
                rows_buffer = []
            elapsed = time.perf_counter() - started
            _write_json(
                out_dir / "progress.json",
                {
                    "status": "running",
                    "model_key": model_key,
                    "assignment_id": assignment["assignment_id"],
                    "wave_index": assignment["wave_index"],
                    "arm": arm,
                    "rows_done": rows_done,
                    "rows_failed": rows_failed,
                    "rows_skipped": rows_skipped,
                    "rows_per_second": rows_done / elapsed if elapsed else None,
                    "effective_microbatch_size": microbatch_size,
                    "peak_gpu_allocated_bytes": peak_allocated,
                    "updated_at": _utc_now(),
                },
            )
    if rows_buffer:
        _flush_chunk(
            capture_root=capture_root,
            model_key=model_key,
            chunk_index=chunk_index,
            tensors=tensors_buffer,
            rows=rows_buffer,
        )
    final_completed = _validate_resume_capture(
        out_dir=out_dir,
        capture_root=capture_root,
        expected_binding=assignment_binding,
        expected_keys=expected_keys,
        resume=True,
    )
    elapsed = time.perf_counter() - started
    expected = summary["expected_capture_rows"]
    summary.update(
        {
            "ended_at": _utc_now(),
            "capture_elapsed_s": elapsed,
            "rows_done": rows_done,
            "rows_failed": rows_failed,
            "rows_skipped": rows_skipped,
            "rows_accounted": rows_done + rows_failed + rows_skipped,
            "rows_complete_total": len(final_completed),
            "rows_per_second": rows_done / elapsed if elapsed else None,
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "effective_microbatch_size": microbatch_size,
            "chunk_count": len(list((capture_root / "chunks").glob("chunk_*.manifest.json"))),
            "fresh_process_contract": {
                "status": "pass",
                "process_id": os.getpid(),
                "process_invocation_id": summary["process_invocation_id"],
                "one_assignment_per_process": True,
                "next_wave_requires_process_exit_and_new_apptainer_exec": True,
            },
        }
    )
    checks = {
        "exact_row_accounting": summary["rows_accounted"] == expected,
        "exact_completed_key_set": final_completed == expected_keys,
        "no_row_failures": rows_failed == 0,
        "capture_plan_supported": capture_plan["status"] == "pass",
        "model_on_cuda_bf16": summary["model_placement"].get("status") == "pass",
        "slot_mapping_fail_closed": rows_failed == 0,
        "full_mode_has_frozen_batch_plan": mode != "full" or bool(batch_plan_sha),
        "assignment_process_isolated": summary["fresh_process_contract"]["status"]
        == "pass",
    }
    summary["pass_fail"] = checks
    summary["overall_status"] = "pass" if all(checks.values()) else "fail"
    _write_json(out_dir / "capture_summary.json", summary)
    _write_json(out_dir / "progress.json", {**summary, "status": summary["overall_status"]})
    return 0 if summary["overall_status"] == "pass" else 1


def _gpu_monitor_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing"}
    values = []
    with path.open(encoding="utf-8") as handle:
        header = next(handle, "").strip().split(",")
        for line in handle:
            fields = [part.strip() for part in line.strip().split(",")]
            if len(fields) != len(header):
                continue
            row = dict(zip(header, fields))
            try:
                values.append(float(row["utilization_gpu_pct"]))
            except (KeyError, ValueError):
                pass
    if not values:
        return {"status": "empty"}
    values.sort()
    return {
        "status": "pass",
        "samples": len(values),
        "mean_pct": sum(values) / len(values),
        "p10_pct": values[max(0, round(0.10 * (len(values) - 1)))],
        "p50_pct": values[round(0.50 * (len(values) - 1))],
        "p90_pct": values[round(0.90 * (len(values) - 1))],
        "max_pct": values[-1],
    }


def _question_token_canary_gate(
    capture_index: Path, *, expected_rows: int
) -> dict[str, Any]:
    issues: list[str] = []
    rows = list(_iter_jsonl(capture_index)) if capture_index.is_file() else []
    if len(rows) != int(expected_rows):
        issues.append("row_count")
    for row_index, row in enumerate(rows):
        audit = row.get("question_token")
        valid = (
            isinstance(audit, Mapping)
            and audit.get("status") == "pass"
            and audit.get("contract") == QUESTION_TOKEN_CONTRACT
            and type(audit.get("index")) is int
            and type(audit.get("input_token_count")) is int
            and 0 <= int(audit["index"]) < int(audit["input_token_count"])
            and type(audit.get("question_token_count")) is int
            and int(audit["question_token_count"]) > 0
            and type(audit.get("tokens_after_question")) is int
            and int(audit["tokens_after_question"]) >= 0
        )
        if not valid:
            issues.append(f"row_{row_index}")
    return {
        "status": "pass" if not issues else "fail",
        "contract": QUESTION_TOKEN_CONTRACT,
        "processor_exercised_with_exact_chat_template": True,
        "row_count": len(rows),
        "expected_row_count": int(expected_rows),
        "issues": issues,
    }


def _aggregate_full_model_shards(
    *,
    run_dir: Path,
    model_row: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    selected_trial_ids: Sequence[str],
    batch_plan_sha256: str,
    schedule_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_key = str(model_row["model_key"])
    model_root = run_dir / model_key
    all_keys: list[tuple[str, str]] = []
    assignment_summaries: list[dict[str, Any]] = []
    capture_plans: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    for assignment in sorted(assignments, key=lambda row: int(row["selection_start"])):
        start = int(assignment["selection_start"])
        stop = int(assignment["selection_stop"])
        shard_trial_ids = [str(value) for value in selected_trial_ids[start:stop]]
        expected_keys = {(arm, trial_id) for arm in ARMS for trial_id in shard_trial_ids}
        out_dir = (run_dir / str(assignment["output_namespace"])).resolve()
        if model_root.resolve() not in out_dir.parents:
            raise ValueError("full shard output escaped its model namespace")
        capture_root = out_dir / "capture"
        expected_binding = _capture_assignment_binding(
            mode="full",
            model_row=model_row,
            assignment=assignment,
            selected_trial_ids=shard_trial_ids,
            batch_plan_sha256=batch_plan_sha256,
            schedule_sha256=schedule_sha256,
        )
        completed = _validate_resume_capture(
            out_dir=out_dir,
            capture_root=capture_root,
            expected_binding=expected_binding,
            expected_keys=expected_keys,
            resume=True,
        )
        if completed != expected_keys:
            raise ValueError(f"full shard is incomplete: {assignment['assignment_id']}")
        summary_path = out_dir / "capture_summary.json"
        plan_path = out_dir / "capture_plan.json"
        if not summary_path.is_file() or not plan_path.is_file():
            raise ValueError(f"full shard summary/plan is missing: {assignment['assignment_id']}")
        summary = _read_json(summary_path)
        if (
            summary.get("overall_status") != "pass"
            or summary.get("assignment_binding") != expected_binding
            or int(summary.get("rows_complete_total") or 0) != len(expected_keys)
            or int(summary.get("rows_failed") or 0) != 0
            or not all((summary.get("pass_fail") or {}).values())
        ):
            raise ValueError(f"full shard summary failed: {assignment['assignment_id']}")
        process_contract = summary.get("fresh_process_contract")
        if not isinstance(process_contract, Mapping) or process_contract.get("status") != "pass":
            raise ValueError(f"full shard lacks process-isolation receipt: {assignment['assignment_id']}")
        rows = list(_iter_jsonl(capture_root / "capture_index.jsonl"))
        observed_keys = [
            (str(row.get("arm") or ""), str(row.get("trial_id") or ""))
            for row in rows
        ]
        if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != expected_keys:
            raise ValueError(f"full shard key union failed: {assignment['assignment_id']}")
        all_keys.extend(observed_keys)
        plan = _read_json(plan_path)
        if plan.get("status") != "pass":
            raise ValueError(f"full shard capture plan failed: {assignment['assignment_id']}")
        capture_plans.append(plan)
        assignment_summaries.append(summary)
        shard_records.append(
            {
                "assignment_id": assignment["assignment_id"],
                "worker_rank": assignment["rank"],
                "wave_index": assignment["wave_index"],
                "selection_start": start,
                "selection_stop": stop,
                "capture_root": os.path.relpath(capture_root, model_root),
                "capture_index": os.path.relpath(
                    capture_root / "capture_index.jsonl", model_root
                ),
                "capture_summary": os.path.relpath(summary_path, model_root),
                "capture_plan": os.path.relpath(plan_path, model_root),
                "row_count": len(observed_keys),
                "process_id": summary["process_id"],
                "process_invocation_id": summary["process_invocation_id"],
            }
        )
    expected_all = {
        (arm, str(trial_id)) for arm in ARMS for trial_id in selected_trial_ids
    }
    if len(all_keys) != len(set(all_keys)):
        raise ValueError(f"cross-shard output collision for {model_key}")
    if set(all_keys) != expected_all:
        raise ValueError(f"cross-shard disjoint union has a gap for {model_key}")
    plan_hashes = {_canonical_sha256(plan) for plan in capture_plans}
    if len(plan_hashes) != 1:
        raise ValueError(f"capture plan changed between shards for {model_key}")
    invocation_ids = [str(row["process_invocation_id"]) for row in assignment_summaries]
    if len(invocation_ids) != len(set(invocation_ids)):
        raise ValueError(f"sequential shard model reload did not use fresh processes for {model_key}")
    _write_json(model_root / "capture_plan.json", capture_plans[0])
    capture_elapsed = sum(float(row.get("capture_elapsed_s") or 0.0) for row in assignment_summaries)
    model_summary = {
        "kind": "phase2_reanchored_jobB_model_shard_aggregate",
        "overall_status": "pass",
        "model_key": model_key,
        "model_id": model_row["model_id"],
        "revision": model_row["revision"],
        "canonical_rank": int(model_row["canonical_rank"]),
        "dtype": "bf16",
        "persisted_dtype": "float16",
        "governing_prespec_sha256": GOVERNING_PRESPEC_SHA256,
        "batch_plan_sha256": batch_plan_sha256,
        "schedule_sha256": schedule_sha256,
        "expected_capture_rows": len(expected_all),
        "rows_done": len(all_keys),
        "rows_failed": 0,
        "rows_complete_total": len(all_keys),
        "rows_accounted": len(all_keys),
        "capture_elapsed_s": capture_elapsed,
        "load_elapsed_s": sum(
            float(row.get("load_elapsed_s") or 0.0) for row in assignment_summaries
        ),
        "rows_per_second": len(all_keys) / capture_elapsed if capture_elapsed > 0 else None,
        "peak_gpu_allocated_bytes": max(
            int(row.get("peak_gpu_allocated_bytes") or 0) for row in assignment_summaries
        ),
        "peak_gpu_reserved_bytes": max(
            int(row.get("peak_gpu_reserved_bytes") or 0) for row in assignment_summaries
        ),
        "effective_microbatch_size": int(
            assignment_summaries[0]["effective_microbatch_size"]
        ),
        "capture_shards": shard_records,
        "assignment_summaries": assignment_summaries,
        "exact_disjoint_union_no_gap": True,
        "one_result_per_model_arm_trial_key": True,
        "shard_output_namespaces_unique": True,
        "sequential_model_unload_reload": {
            "status": "pass",
            "method": "one_assignment_per_fresh_apptainer_python_process",
            "process_invocation_ids": invocation_ids,
            "namespace_local_process_ids": [
                int(row["process_id"]) for row in assignment_summaries
            ],
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(model_root / "aggregate_summary.json", model_summary)
    _write_json(model_root / "capture_summary.json", model_summary)
    return model_summary, assignment_summaries


def aggregate(args: argparse.Namespace) -> int:
    runtime_hashes = {
        name.lower(): _required_sha256_env(name)
        for name in (
            "SIF_SHA256",
            "RELEASE_MANIFEST_SHA256",
            "MECHANISM_MANIFEST_SHA256",
            "PUBLIC_MANIFEST_SHA256",
        )
    }
    run_dir = args.run_dir.expanduser().resolve()
    plan = _parse_model_plan(args.model_plan_file)
    selection = _read_json(run_dir / "selection_manifest.json")
    mode = str(selection.get("mode") or "")
    models = []
    missing = []
    batch_plan = None
    schedule = None
    if mode == "full":
        if not args.batch_plan_file:
            raise ValueError("full aggregation requires the frozen batch plan")
        batch_plan_path = args.batch_plan_file.expanduser().resolve()
        batch_plan = _read_json(batch_plan_path)
        schedule = batch_plan.get("schedule")
        if not isinstance(schedule, Mapping):
            raise ValueError("full aggregation batch plan lacks its schedule")
        assignments = _schedule_assignments(schedule)
        for assignment in assignments:
            _load_frozen_batch_plan(
                batch_plan_path,
                str(assignment["model_key"]),
                model_plan_path=args.model_plan_file,
                rank=int(assignment["rank"]),
                assignment_id=str(assignment["assignment_id"]),
                selection_start=int(assignment["selection_start"]),
                selection_stop=int(assignment["selection_stop"]),
                wave_index=int(assignment["wave_index"]),
            )
        batch_plan_sha = _sha256_file(batch_plan_path)
        selected_ids = [str(value) for value in selection.get("selected_trial_ids") or []]
        for row in plan:
            model_assignments = [
                assignment
                for assignment in assignments
                if assignment["model_key"] == row["model_key"]
            ]
            if not model_assignments:
                missing.append(row["model_key"])
                continue
            summary, _ = _aggregate_full_model_shards(
                run_dir=run_dir,
                model_row=row,
                assignments=model_assignments,
                selected_trial_ids=selected_ids,
                batch_plan_sha256=batch_plan_sha,
                schedule_sha256=str(schedule["sha256"]),
            )
            assigned_ranks = sorted(
                {int(assignment["rank"]) for assignment in model_assignments}
            )
            summary["gpu_utilization_by_rank"] = {
                str(rank): _gpu_monitor_summary(
                    run_dir / f"gpu_monitor_rank_{rank}.csv"
                )
                for rank in assigned_ranks
            }
            _write_json(run_dir / row["model_key"] / "capture_summary.json", summary)
            models.append(summary)
    elif mode == "canary":
        for rank, row in enumerate(plan):
            path = run_dir / row["model_key"] / "capture_summary.json"
            if not path.is_file():
                missing.append(row["model_key"])
                continue
            summary = _read_json(path)
            summary["question_token_canary_gate"] = _question_token_canary_gate(
                run_dir / row["model_key"] / "capture" / "capture_index.jsonl",
                expected_rows=int(summary.get("expected_capture_rows") or -1),
            )
            if summary["question_token_canary_gate"]["status"] != "pass":
                summary["overall_status"] = "fail"
            summary["gpu_utilization"] = _gpu_monitor_summary(
                run_dir / f"gpu_monitor_rank_{rank}.csv"
            )
            _write_json(path, summary)
            models.append(summary)
    else:
        raise ValueError(f"unknown selection mode: {mode}")
    sequential_process_gate = None
    if mode == "full" and schedule is not None:
        shard_by_assignment = {
            str(shard["assignment_id"]): shard
            for model in models
            for shard in model.get("capture_shards", [])
        }
        rank_checks = []
        for rank_row in schedule["ranks"]:
            expected = list(rank_row["assignments"])
            observed = [shard_by_assignment.get(str(row["assignment_id"])) for row in expected]
            invocation_ids = [
                str(row["process_invocation_id"])
                for row in observed
                if isinstance(row, Mapping)
            ]
            waves = [int(row["wave_index"]) for row in observed if isinstance(row, Mapping)]
            status = (
                len(observed) == len(expected)
                and all(isinstance(row, Mapping) for row in observed)
                and len(invocation_ids) == len(set(invocation_ids))
                and waves == list(range(len(expected)))
            )
            if not status:
                raise ValueError(f"rank {rank_row['rank']} did not unload/reload sequentially")
            rank_checks.append(
                {
                    "rank": rank_row["rank"],
                    "assignment_ids": [row["assignment_id"] for row in expected],
                    "process_invocation_ids": invocation_ids,
                    "wave_indices": waves,
                    "status": "pass",
                }
            )
        sequential_process_gate = {
            "status": "pass",
            "method": "ordered_shell_loop_with_one_fresh_apptainer_python_process_per_assignment",
            "ranks": rank_checks,
        }
    payload = {
        "kind": "phase2_reanchored_jobB_capture_summary",
        "created_at": _utc_now(),
        "run_dir": str(run_dir),
        "governing_prespec_sha256": GOVERNING_PRESPEC_SHA256,
        "selection_manifest_sha256": _sha256_file(run_dir / "selection_manifest.json"),
        "storage_projection": selection.get("storage_projection"),
        "model_plan": plan,
        "models": models,
        "missing_models": missing,
        "mode": mode,
        "schedule_sha256": schedule.get("sha256") if schedule else None,
        "sequential_model_unload_reload": sequential_process_gate,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["overall_status"] = (
        "pass"
        if not missing and len(models) == 4
        and all(row.get("overall_status") == "pass" for row in models)
        else "fail"
    )
    _write_json(run_dir / "capture_summary.json", payload)
    if args.emit_batch_plan and mode == "canary" and payload["overall_status"] == "pass":
        if any(
            int(row.get("rows_skipped") or 0) != 0
            or int(row.get("rows_done") or 0) != int(row.get("expected_capture_rows") or -1)
            for row in models
        ):
            raise ValueError("batch-plan canary must be a clean non-resumed throughput run")
        model_profiles = [
            {
                "model_key": row["model_key"],
                "canonical_rank": int(row["canonical_rank"]),
                "microbatch_size": int(row["effective_microbatch_size"]),
                "effective_microbatch_size": int(row["effective_microbatch_size"]),
                "tested_candidate_cap": next(
                    int(plan_row["full_microbatch_hard_cap"])
                    for plan_row in plan
                    if plan_row["model_key"] == row["model_key"]
                ),
                "load_elapsed_s": float(row["load_elapsed_s"]),
                "rows_per_second": float(row["rows_per_second"]),
                "peak_gpu_allocated_bytes": int(row["peak_gpu_allocated_bytes"]),
            }
            for row in models
        ]
        balanced_schedule = build_balanced_full_schedule(model_profiles)
        _validate_balanced_full_schedule(
            balanced_schedule,
            model_reports=model_profiles,
            model_plan=plan,
            require_passing=False,
        )
        utilization_tail_gate = {
            "status": balanced_schedule["threshold_status"],
            "projected_aggregate_busy_fraction": balanced_schedule[
                "projected_aggregate_busy_fraction"
            ],
            "minimum_projected_aggregate_busy_fraction": (
                MIN_PROJECTED_AGGREGATE_BUSY_FRACTION
            ),
            "projected_max_tail_idle_fraction": balanced_schedule[
                "projected_max_tail_idle_fraction"
            ],
            "projected_makespan_s": balanced_schedule["projected_makespan_s"],
            "fits_two_hour_walltime_with_reserve": balanced_schedule[
                "fits_two_hour_walltime_with_reserve"
            ],
            "failure_action": balanced_schedule["failure_action"],
        }
        source_canary_summary = {
            "kind": payload["kind"],
            "overall_status": payload["overall_status"],
            "governing_prespec_sha256": payload["governing_prespec_sha256"],
            "selection_manifest_sha256": payload["selection_manifest_sha256"],
            "models": [
                {
                    "model_key": row["model_key"],
                    "revision": row["revision"],
                    "canonical_rank": row["canonical_rank"],
                    "overall_status": row["overall_status"],
                    "load_elapsed_s": row["load_elapsed_s"],
                    "effective_microbatch_size": row["effective_microbatch_size"],
                    "rows_per_second": row["rows_per_second"],
                    "peak_gpu_allocated_bytes": row["peak_gpu_allocated_bytes"],
                    "gpu_utilization": row["gpu_utilization"],
                    "question_token_canary_gate": row[
                        "question_token_canary_gate"
                    ],
                }
                for row in models
            ],
        }
        code_bundle = _code_bundle_binding()
        batch_plan = {
            "kind": FROZEN_BATCH_PLAN_KIND,
            "status": utilization_tail_gate["status"],
            "created_at": _utc_now(),
            "governing_prespec_sha256": GOVERNING_PRESPEC_SHA256,
            "model_plan_sha256": _sha256_file(args.model_plan_file.expanduser().resolve()),
            "runner_code_sha256": _sha256_file(Path(__file__).resolve()),
            "code_bundle": code_bundle["files"],
            "code_bundle_sha256": code_bundle["sha256"],
            "sif_sha256": runtime_hashes["sif_sha256"],
            "release_manifest_sha256": runtime_hashes["release_manifest_sha256"],
            "mechanism_manifest_sha256": runtime_hashes["mechanism_manifest_sha256"],
            "public_manifest_sha256": runtime_hashes["public_manifest_sha256"],
            "source_canary_summary": source_canary_summary,
            "source_canary_summary_sha256": _canonical_sha256(source_canary_summary),
            "technical_only_selection": True,
            "selection_uses_labels_or_outcomes": False,
            "evaluator_joined": False,
            "schedule": balanced_schedule,
            "schedule_sha256": balanced_schedule["sha256"],
            "utilization_tail_gate": utilization_tail_gate,
            "models": model_profiles,
        }
        _write_json(run_dir / "frozen_batch_plan.json", batch_plan)
        payload["frozen_batch_plan_sha256"] = _sha256_file(run_dir / "frozen_batch_plan.json")
        payload["balanced_full_schedule"] = balanced_schedule
        payload["utilization_tail_gate"] = utilization_tail_gate
        if utilization_tail_gate["status"] != "pass":
            payload["overall_status"] = "fail"
        _write_json(run_dir / "capture_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["overall_status"] == "pass" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    subset = commands.add_parser("prepare-subset")
    subset.add_argument("--mechanism-dir", type=Path, required=True)
    subset.add_argument("--run-dir", type=Path, required=True)
    subset.add_argument("--prespec-path", type=Path)
    subset.add_argument("--model-plan-file", type=Path)
    subset.add_argument("--mode", choices=("canary", "full"), required=True)
    subset.add_argument("--allow-storage-over-ceiling", action="store_true")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    common.add_argument("--run-dir", type=Path, required=True)
    common.add_argument("--model-plan-file", type=Path)
    common.add_argument("--strict", action="store_true")

    run = commands.add_parser("run-model", parents=[common])
    run.add_argument("--mechanism-dir", type=Path, required=True)
    run.add_argument("--prespec-path", type=Path)
    run.add_argument("--batch-plan-file", type=Path)
    run.add_argument("--rank", type=int, required=True)
    run.add_argument("--world-size", type=int, required=True)
    run.add_argument("--process-invocation-id")
    run.add_argument("--model-key")
    run.add_argument("--assignment-id")
    run.add_argument("--selection-start", type=int)
    run.add_argument("--selection-stop", type=int)
    run.add_argument("--wave-index", type=int)
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--chunk-size", type=int, default=8)
    run.add_argument("--prefetch-workers", type=int, default=4)
    run.add_argument("--max-modules", type=int, default=30000)
    run.add_argument("--post-load-barrier-dir", type=Path)
    run.add_argument("--post-load-barrier-timeout-s", type=float, default=1200.0)

    aggregate_parser = commands.add_parser("aggregate", parents=[common])
    aggregate_parser.add_argument("--emit-batch-plan", action="store_true")
    aggregate_parser.add_argument("--batch-plan-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare-subset":
        return prepare_subset(args)
    if args.command == "run-model":
        return run_model(args)
    if args.command == "aggregate":
        return aggregate(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
