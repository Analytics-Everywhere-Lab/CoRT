#!/usr/bin/env python3
"""Run W0 compact activation capture on TamIA.

This is the first real mechanism-study capture job after the instrument gates.
It loads one VLM per GPU worker, runs W0 method-facing arms, and stores compact
selected vectors rather than full hidden-state tensors. Hidden answers and VISOR
geometry are never inserted into model prompts; they are saved only as
evaluator-side metadata for offline probes and analysis.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory import run_phase2_mech_job1_instrument_gates as gates


DEFAULT_W0_DIR = gates.DEFAULT_W0_DIR
DEFAULT_MODEL_PLAN = gates.DEFAULT_MODEL_PLAN
DEFAULT_ARMS = ("visual_prompt", "single_frame", "crop", "sequential")
CONTACT_SHEET_ARMS = frozenset(
    {"visual_prompt", "crop", "plain_contact_sheet", "marked_plain_prompt"}
)
DEFAULT_LAYER_STRIDE = 1
DEFAULT_CHUNK_SIZE = 8
DEFAULT_FLUSH_EVERY_N = 4
DEFAULT_TOPK = 20
DEFAULT_POST_LOAD_BARRIER_TIMEOUT = 900.0
MIN_PROMPT_CHARS = gates.MIN_PROMPT_CHARS


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    new_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(path, existing + new_text)


def _sanitize_jsonl(path: Path) -> dict[str, Any]:
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
    return {"path": str(path), "status": "pass", "dropped_lines": dropped}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe[:120] or "item"


def _parse_csv_list(text: str | None, default: tuple[str, ...]) -> list[str]:
    if not text:
        return list(default)
    values = [item.strip() for item in text.split(",") if item.strip()]
    return values or list(default)


def _parse_model_plan(args: argparse.Namespace) -> list[dict[str, str]]:
    return gates._parse_model_plan(args)


def _select_model_for_rank(args: argparse.Namespace) -> dict[str, str]:
    return gates._select_model_for_rank(args)


def _select_model_for_run(args: argparse.Namespace) -> dict[str, Any]:
    plan = _parse_model_plan(args)
    if getattr(args, "model_key", None):
        requested = str(args.model_key)
        for index, row in enumerate(plan):
            if row["model_key"] == requested:
                return {**row, "model_index": index}
        raise ValueError(f"model_key not found in plan: {requested}")
    if getattr(args, "model_index", None) is not None:
        index = int(args.model_index)
        if index < 0:
            raise ValueError(f"model_index must be non-negative, got {index}")
        if index >= len(plan):
            return {"model_id": "", "model_key": f"idle_model_index_{index:03d}", "revision": "", "model_index": index}
        return {**plan[index], "model_index": index}
    selected = _select_model_for_rank(args)
    rank = int(args.rank)
    return {**selected, "model_index": rank if selected.get("model_id") else None}


def _load_pack(w0_dir: Path, arm: str) -> tuple[dict[str, Any], Path]:
    pack_path = w0_dir / "derived_pack" / f"order_stripped_{arm}_request_pack.json"
    return _read_json(pack_path), pack_path


def _load_hidden_rows(w0_dir: Path, arm: str) -> dict[str, dict[str, Any]]:
    index_path = w0_dir / "derived_pack" / f"order_stripped_{arm}_hidden_index.json"
    payload = _read_json(index_path)
    rows = payload.get("answers", [])
    return {
        str(row.get("request_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("request_id")
    }


def _load_geometry_rows(w0_dir: Path) -> dict[str, dict[str, Any]]:
    path = w0_dir / "geometry_sidecar.json"
    payload = _read_json(path)
    rows = payload.get("requests", {})
    return rows if isinstance(rows, dict) else {}


def _resolve_frame_path(frame_path: str, *, pack_root: Path) -> Path:
    return gates._resolve_frame_path(frame_path, pack_root=pack_root)


def _request_frame_paths(request: dict[str, Any], *, pack_root: Path) -> list[Path]:
    paths: list[Path] = []
    for frame in request.get("frames", []):
        if isinstance(frame, dict) and frame.get("path"):
            paths.append(_resolve_frame_path(str(frame["path"]), pack_root=pack_root))
    return paths


def _contact_sheet_layout(
    *,
    request: dict[str, Any],
    frame_paths: list[Path],
    out_path: Path,
    thumb_width: int = 360,
    thumb_height: int = 220,
    columns: int = 2,
    show_labels: bool = False,
) -> dict[str, Any]:
    """Build the same sheet as the shared helper and return frame transforms."""
    gates.build_contact_sheet_image(
        request=request,
        frame_paths=frame_paths,
        out_path=out_path,
        thumb_width=thumb_width,
        thumb_height=thumb_height,
        columns=columns,
        show_labels=show_labels,
    )
    frames_meta = [
        frame
        for frame in request.get("frames", [])
        if isinstance(frame, dict) and frame.get("path")
    ]
    if len(frames_meta) != len(frame_paths):
        frames_meta = [{} for _ in frame_paths]
    thumb_w = max(96, int(thumb_width))
    thumb_h = max(72, int(thumb_height))
    cols = min(max(1, int(columns)), max(1, len(frame_paths)))
    label_h = 24 if show_labels else 0
    rows = math.ceil(max(1, len(frame_paths)) / cols)
    sheet_width = cols * thumb_w
    sheet_height = rows * (thumb_h + label_h)
    frame_layouts: list[dict[str, Any]] = []
    for index, (frame_meta, frame_path) in enumerate(zip(frames_meta, frame_paths)):
        col = index % cols
        row = index // cols
        with Image.open(frame_path) as image:
            width, height = image.size
            preview = image.convert("RGB")
            preview.thumbnail((thumb_w, thumb_h))
            pasted_w = preview.width
            pasted_h = preview.height
        x = col * thumb_w + (thumb_w - pasted_w) // 2
        y = row * (thumb_h + label_h) + label_h
        frame_layouts.append(
            {
                "input_image_index": 0,
                "source_public_frame_index": int(frame_meta.get("public_frame_index", index))
                if str(frame_meta.get("public_frame_index", index)).isdigit()
                else index,
                "source_frame_path": str(frame_path),
                "source_width": width,
                "source_height": height,
                "sheet_x": x,
                "sheet_y": y,
                "sheet_width": pasted_w,
                "sheet_height": pasted_h,
                "scale_x": pasted_w / width if width else None,
                "scale_y": pasted_h / height if height else None,
            }
        )
    return {
        "mode": "contact_sheet",
        "image_path": str(out_path),
        "image_count": 1,
        "image_sizes": [{"width": sheet_width, "height": sheet_height}],
        "thumb_width": thumb_w,
        "thumb_height": thumb_h,
        "columns": cols,
        "show_labels": show_labels,
        "frame_layouts": frame_layouts,
    }


def _single_or_sequence_layout(
    frame_paths: list[Path],
    *,
    public_frame_indexes: list[int] | None = None,
) -> dict[str, Any]:
    image_sizes: list[dict[str, int]] = []
    frame_layouts: list[dict[str, Any]] = []
    for index, path in enumerate(frame_paths):
        with Image.open(path) as image:
            width, height = image.size
        image_sizes.append({"width": width, "height": height})
        frame_layouts.append(
            {
                "input_image_index": index,
                "source_public_frame_index": (
                    int(public_frame_indexes[index])
                    if public_frame_indexes and index < len(public_frame_indexes)
                    else index
                ),
                "source_frame_path": str(path),
                "source_width": width,
                "source_height": height,
                "sheet_x": 0,
                "sheet_y": 0,
                "sheet_width": width,
                "sheet_height": height,
                "scale_x": 1.0,
                "scale_y": 1.0,
            }
        )
    return {
        "mode": "multi_image" if len(frame_paths) > 1 else "single_image",
        "image_count": len(frame_paths),
        "image_sizes": image_sizes,
        "frame_layouts": frame_layouts,
    }


def _select_requests(
    *,
    requests: list[dict[str, Any]],
    offset: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    start = max(0, int(offset))
    rows = requests[start:]
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def _prepare_capture_sample(
    *,
    w0_dir: Path,
    arm: str,
    request: dict[str, Any],
    pack_path: Path,
    hidden_by_request: dict[str, dict[str, Any]],
    geometry_by_source: dict[str, dict[str, Any]],
    sample_root: Path,
) -> dict[str, Any]:
    request_id = str(request.get("request_id") or "")
    frame_paths = _request_frame_paths(request, pack_root=pack_path.parent)
    if not frame_paths:
        raise RuntimeError(f"{arm}/{request_id} has no frame paths")
    hidden = hidden_by_request.get(request_id, {})
    source_key = hidden.get("source_order_stripped_request_id")
    geometry = geometry_by_source.get(str(source_key), {}) if source_key else {}
    sample_dir = sample_root / _safe_id(arm) / _safe_id(request_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    if arm in CONTACT_SHEET_ARMS:
        image_path = sample_dir / "contact_sheet.jpg"
        layout = _contact_sheet_layout(
            request=request,
            frame_paths=frame_paths,
            out_path=image_path,
            show_labels=False,
        )
        input_image_paths = [image_path]
        presentation_mode = "contact_sheet"
    elif arm == "single_frame":
        input_image_paths = [frame_paths[0]]
        geometry_frames = geometry.get("frames", []) if isinstance(geometry, dict) else []
        public_index = len(geometry_frames) - 1 if geometry_frames else 0
        layout = _single_or_sequence_layout(input_image_paths, public_frame_indexes=[public_index])
        presentation_mode = "single_image"
    elif arm == "sequential":
        input_image_paths = frame_paths
        layout = _single_or_sequence_layout(input_image_paths)
        presentation_mode = "multi_image"
    else:
        raise ValueError(f"unsupported arm: {arm}")

    return {
        "arm": arm,
        "request_id": request_id,
        "query": request.get("query"),
        "prompt": request.get("user_prompt"),
        "method_facing_pack": str(pack_path),
        "method_frame_paths": [
            str(frame.get("path"))
            for frame in request.get("frames", [])
            if isinstance(frame, dict) and frame.get("path")
        ],
        "input_image_paths": [str(path.resolve()) for path in input_image_paths],
        "input_image_count": len(input_image_paths),
        "presentation_mode": presentation_mode,
        "layout": layout,
        "hidden_index": _hidden_metadata_for_output(hidden),
        "geometry": geometry,
        "source_order_stripped_request_id": source_key,
        "claim_boundary": (
            "Model prompt uses only method-facing images and public question text. "
            "Hidden index and geometry are evaluator-side metadata only."
        ),
    }


def _hidden_metadata_for_output(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    keep = (
        "request_id",
        "derived_arm",
        "expected_answer",
        "query_type",
        "target_label",
        "evidence_public_frame_indexes",
        "contact_by_public_frame",
        "visible_by_public_frame",
        "source_order_stripped_request_id",
        "source_clip_request_id",
        "source_query_id",
        "source_scene_id",
        "source_tier1_participant_id",
        "source_tier1_video_id",
        "source_video_id",
        "target_swap_decoy",
        "leakage_boundary",
    )
    return {key: row.get(key) for key in keep if key in row}


def _geometry_boxes_for_sample(sample: dict[str, Any]) -> dict[int, dict[str, list[list[float]]]]:
    """Return per-input-image boxes in input image coordinates."""
    arm = sample["arm"]
    geometry = sample.get("geometry") if isinstance(sample.get("geometry"), dict) else {}
    frames = geometry.get("frames", []) if isinstance(geometry, dict) else []
    if not isinstance(frames, list) or not frames:
        return {}
    by_public: dict[int, dict[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        public = frame.get("public_frame_index")
        if isinstance(public, int):
            by_public[public] = frame
    result: dict[int, dict[str, list[list[float]]]] = {}
    if arm == "crop":
        return result
    for layout in sample.get("layout", {}).get("frame_layouts", []):
        if not isinstance(layout, dict):
            continue
        public_index = layout.get("source_public_frame_index")
        input_index = layout.get("input_image_index")
        if not isinstance(public_index, int) or not isinstance(input_index, int):
            continue
        frame = by_public.get(public_index)
        if not frame:
            continue
        boxes = {"target": [], "hand": [], "background": []}
        target = frame.get("target_geometry", {}).get("target", {})
        for ann in target.get("annotations", []) if isinstance(target, dict) else []:
            box = ann.get("bbox_xyxy") if isinstance(ann, dict) else None
            transformed = _transform_box(box, layout)
            if transformed:
                boxes["target"].append(transformed)
        hand = frame.get("hand_geometry", {})
        hand_annotations = []
        if isinstance(hand, dict):
            hand_annotations = hand.get("contact_hand_annotations") or hand.get("annotations") or []
        for ann in hand_annotations if isinstance(hand_annotations, list) else []:
            box = ann.get("bbox_xyxy") if isinstance(ann, dict) else None
            transformed = _transform_box(box, layout)
            if transformed:
                boxes["hand"].append(transformed)
        current = result.setdefault(input_index, {"target": [], "hand": [], "background": []})
        current["target"].extend(boxes["target"])
        current["hand"].extend(boxes["hand"])
    return result


def _transform_box(box: Any, layout: dict[str, Any]) -> list[float] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
        sx = float(layout.get("scale_x") or 1.0)
        sy = float(layout.get("scale_y") or 1.0)
        ox = float(layout.get("sheet_x") or 0.0)
        oy = float(layout.get("sheet_y") or 0.0)
        return [ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy]
    except Exception:
        return None


def _prepare_inputs_multi(
    *,
    runtime: Any,
    processor: Any,
    prompt: str,
    image_paths: list[Path],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    trace: dict[str, Any] = {"attempts": [], "image_count": len(image_paths)}
    if not image_paths:
        raise RuntimeError("no image paths for processor input")
    image_content_path = [
        {"type": "image", "image": str(path)}
        for path in image_paths
    ]
    image_content_url = [
        {"type": "image", "url": path.resolve().as_uri()}
        for path in image_paths
    ]
    message_attempts: tuple[tuple[str, list[dict[str, Any]]], ...] = (
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
    )
    if hasattr(processor, "apply_chat_template"):
        for name, messages in message_attempts:
            try:
                inputs = runtime._apply_chat_template_direct(
                    processor,
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                if gates._input_get(inputs, "input_ids") is not None:
                    trace["winner"] = name
                    trace["attempts"].append({"name": name, "status": "pass"})
                    return runtime._to_device(inputs, device), trace
                trace["attempts"].append({"name": name, "status": "skip", "reason": "missing_input_ids"})
            except Exception as exc:
                trace["attempts"].append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})

    messages = [{"role": "user", "content": [*image_content_path, {"type": "text", "text": prompt}]}]
    try:
        if hasattr(processor, "apply_chat_template"):
            text = runtime._apply_chat_template_direct(
                processor,
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            trace["winner"] = "processor_text_with_qwen_vl_utils_multi"
        except Exception as vision_exc:
            pil_images = []
            for path in image_paths:
                with Image.open(path) as image:
                    pil_images.append(image.convert("RGB").copy())
            inputs = processor(text=[text], images=pil_images, padding=True, return_tensors="pt")
            trace["winner"] = "processor_text_with_pil_images"
            trace["vision_info_fallback_error"] = f"{type(vision_exc).__name__}: {vision_exc}"
        trace["attempts"].append({"name": trace["winner"], "status": "pass"})
        return runtime._to_device(inputs, device), trace
    except Exception as exc:
        trace["attempts"].append({"name": "processor_fallback_multi", "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
        raise


def _candidate_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    candidates = {
        "true": ["true", " true", "True", " True"],
        "false": ["false", " false", "False", " False"],
        "yes": ["yes", " yes", "YES", " YES"],
        "no": ["no", " no", "NO", " NO"],
        "json_true": ['"answer": true', '"answer":true'],
        "json_false": ['"answer": false', '"answer":false'],
    }
    payload: dict[str, list[int]] = {}
    if tokenizer is None:
        return payload
    for label, texts in candidates.items():
        ids: list[int] = []
        for text in texts:
            try:
                encoded = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(text)
            except Exception:
                encoded = []
            if encoded:
                ids.append(int(encoded[0]))
        payload[label] = sorted(set(ids))
    return payload


def _logit_summary(logits: Any, tokenizer: Any, *, topk: int) -> dict[str, Any]:
    if logits is None or not hasattr(logits, "detach"):
        return {"status": "fail", "reason": "missing_logits"}
    import torch

    vec = logits.detach().float().reshape(-1).cpu()
    finite = bool(torch.isfinite(vec).all().item())
    top_count = min(max(1, int(topk)), int(vec.numel()))
    top_values, top_indices = torch.topk(vec, k=top_count)
    candidate_ids = _candidate_token_ids(tokenizer)
    candidate_logits: dict[str, dict[str, float]] = {}
    for label, ids in candidate_ids.items():
        row: dict[str, float] = {}
        for token_id in ids:
            if 0 <= token_id < vec.numel():
                row[str(token_id)] = float(vec[token_id].item())
        candidate_logits[label] = row
    return {
        "status": "pass" if finite else "fail",
        "finite": finite,
        "shape": list(logits.shape),
        "topk": [
            {"token_id": int(idx), "logit": float(val)}
            for idx, val in zip(top_indices.tolist(), top_values.tolist())
        ],
        "candidate_token_ids": candidate_ids,
        "candidate_logits": candidate_logits,
    }


def _register_vector_hooks(model: Any, names: list[str]) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    module_lookup = dict(model.named_modules())
    records: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    handles: list[Any] = []

    def make_hook(module_name: str) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if module_name in records:
                return
            tensor = gates._first_tensor(output)
            records[module_name] = gates._activation_summary(output)
            vec = _pooled_vector(tensor)
            if vec is not None:
                tensors[f"hook__{_safe_tensor_key(module_name)}__mean"] = vec

        return hook

    for name in names:
        module = module_lookup.get(name)
        if module is None:
            continue
        handles.append(module.register_forward_hook(make_hook(name)))
    return handles, records, tensors


def _pooled_vector(tensor: Any) -> Any | None:
    if tensor is None or not hasattr(tensor, "detach"):
        return None
    import torch

    with torch.no_grad():
        detached = tensor.detach()
        if detached.numel() == 0:
            return None
        if detached.dim() == 0:
            vec = detached.reshape(1)
        elif detached.dim() == 1:
            vec = detached
        else:
            vec = detached.reshape(-1, detached.shape[-1]).mean(dim=0)
        return vec.to(device="cpu", dtype=torch.bfloat16).contiguous()


def _safe_tensor_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "tensor"


def _sequence_masks(
    *,
    inputs: Any,
    model: Any,
    processor: Any,
    model_id: str,
    sample: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    input_ids = gates._input_get(inputs, "input_ids")
    if input_ids is None or not hasattr(input_ids, "detach"):
        return {}, {"status": "fail", "reason": "missing_input_ids"}
    ids = input_ids.detach()
    if ids.dim() == 2:
        ids = ids[0]
    seq_len = int(ids.shape[-1])
    device = ids.device
    image_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    for token_id in gates._image_token_candidates(getattr(model, "config", None), processor):
        image_mask |= ids == int(token_id)
    non_image_mask = ~image_mask
    masks: dict[str, Any] = {
        "last_token": torch.tensor([max(0, seq_len - 1)], dtype=torch.long, device=device),
    }
    non_image_idx = torch.nonzero(non_image_mask, as_tuple=False).reshape(-1)
    if non_image_idx.numel():
        masks["text_tail_mean"] = non_image_idx[-min(8, int(non_image_idx.numel())):]
    image_idx = torch.nonzero(image_mask, as_tuple=False).reshape(-1)
    if image_idx.numel():
        masks["image_all_mean"] = image_idx

    grid = gates._grid_from_inputs(inputs)
    qwen_masks, qwen_meta = _qwen_region_masks(
        image_idx=image_idx,
        grid=grid,
        sample=sample,
        model=model,
        processor=processor,
    )
    masks.update(qwen_masks)
    meta = {
        "status": "pass",
        "seq_len": seq_len,
        "image_token_count": int(image_idx.numel()),
        "non_image_token_count": int(non_image_idx.numel()),
        "visual_grid": grid,
        "region_grouping": qwen_meta,
        "model_family": gates._model_family(getattr(model, "config", None), model_id),
        "mask_token_counts": {
            key: int(value.numel()) if hasattr(value, "numel") else 1
            for key, value in masks.items()
        },
    }
    return masks, meta


def _qwen_region_masks(
    *,
    image_idx: Any,
    grid: dict[str, Any],
    sample: dict[str, Any],
    model: Any,
    processor: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if image_idx is None or not hasattr(image_idx, "numel") or int(image_idx.numel()) == 0:
        return {}, {"status": "skipped", "reason": "no_image_tokens"}
    config = getattr(model, "config", None)
    family = gates._model_family(config, str(sample.get("model_id") or ""))
    if family != "qwen":
        return {}, {"status": "unsupported", "reason": "mask_geometry_only_implemented_for_qwen_grid"}
    if not grid.get("available"):
        return {}, {"status": "unsupported", "reason": "qwen_grid_missing"}
    merge = gates._merge_size(config, processor)
    if not merge:
        return {}, {"status": "unsupported", "reason": "qwen_merge_size_missing"}
    triplets = gates._grid_triplets(grid.get(str(grid.get("grid_key") or ""), []))
    if not triplets:
        return {}, {"status": "unsupported", "reason": "qwen_grid_triplets_missing"}
    image_sizes = sample.get("layout", {}).get("image_sizes", [])
    boxes_by_image = _geometry_boxes_for_sample(sample)
    if not image_sizes or not boxes_by_image:
        return {}, {"status": "unsupported", "reason": "geometry_boxes_unavailable_for_arm"}
    target_positions: list[Any] = []
    hand_positions: list[Any] = []
    background_positions: list[Any] = []
    offset = 0
    unsupported: list[str] = []
    for image_number, triplet in enumerate(triplets):
        t, pre_h, pre_w = triplet
        if t != 1:
            unsupported.append(f"image_{image_number}: temporal_grid_t={t}")
            token_count = max(0, (t * pre_h * pre_w) // (merge * merge))
            offset += token_count
            continue
        post_h = pre_h // merge
        post_w = pre_w // merge
        token_count = post_h * post_w
        current_positions = image_idx[offset: offset + token_count]
        offset += token_count
        if int(current_positions.numel()) != token_count:
            unsupported.append(f"image_{image_number}: token_count_mismatch")
            continue
        if image_number >= len(image_sizes):
            unsupported.append(f"image_{image_number}: image_size_missing")
            continue
        size = image_sizes[image_number]
        width = float(size.get("width") or 0)
        height = float(size.get("height") or 0)
        if width <= 0 or height <= 0:
            unsupported.append(f"image_{image_number}: invalid_image_size")
            continue
        boxes = boxes_by_image.get(image_number, {})
        target_boxes = boxes.get("target", [])
        hand_boxes = boxes.get("hand", [])
        for row in range(post_h):
            for col in range(post_w):
                flat = row * post_w + col
                seq_pos = current_positions[flat]
                cx = (col + 0.5) * width / post_w
                cy = (row + 0.5) * height / post_h
                in_target = _point_in_any_box(cx, cy, target_boxes)
                in_hand = _point_in_any_box(cx, cy, hand_boxes)
                if in_target:
                    target_positions.append(seq_pos)
                if in_hand:
                    hand_positions.append(seq_pos)
                if not in_target and not in_hand:
                    background_positions.append(seq_pos)
    masks: dict[str, Any] = {}
    if target_positions:
        masks["target_mean"] = torch.stack(target_positions).long()
    if hand_positions:
        masks["hand_mean"] = torch.stack(hand_positions).long()
    if background_positions:
        masks["background_mean"] = torch.stack(background_positions).long()
    meta = {
        "status": "pass" if masks else "unsupported",
        "reason": None if masks else "no_region_tokens_selected",
        "unsupported": unsupported,
        "target_token_count": len(target_positions),
        "hand_token_count": len(hand_positions),
        "background_token_count": len(background_positions),
        "grid_triplets": triplets,
        "merge_size": merge,
    }
    return masks, meta


def _point_in_any_box(x: float, y: float, boxes: list[list[float]]) -> bool:
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def _capture_selected_vectors(
    *,
    output: Any,
    masks: dict[str, Any],
    layer_stride: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hidden_states = gates._extract_output_field(output, "hidden_states")
    if not isinstance(hidden_states, (list, tuple)) or not hidden_states:
        return {}, {"status": "fail", "reason": "hidden_states_missing"}
    import torch

    tensors: dict[str, Any] = {}
    layer_indexes = list(range(0, len(hidden_states), max(1, int(layer_stride))))
    if layer_indexes[-1] != len(hidden_states) - 1:
        layer_indexes.append(len(hidden_states) - 1)
    with torch.no_grad():
        for layer_index in layer_indexes:
            layer = hidden_states[layer_index]
            if layer is None or not hasattr(layer, "detach"):
                continue
            seq = layer.detach()
            if seq.dim() == 3:
                seq = seq[0]
            for mask_name, positions in masks.items():
                key = f"layer_{layer_index:03d}__{mask_name}"
                if mask_name == "last_token":
                    pos = int(positions.reshape(-1)[0].item())
                    tensors[key] = seq[pos].to(device="cpu", dtype=torch.bfloat16).contiguous()
                else:
                    if positions is None or int(positions.numel()) == 0:
                        continue
                    tensors[key] = seq[positions].mean(dim=0).to(device="cpu", dtype=torch.bfloat16).contiguous()
    return tensors, {
        "status": "pass" if tensors else "fail",
        "hidden_state_count": len(hidden_states),
        "layer_indexes": layer_indexes,
        "tensor_count": len(tensors),
    }


def _save_tensor_chunk(
    *,
    chunk_path: Path,
    tensors: dict[str, Any],
    tensor_format: str,
) -> str:
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    if tensor_format == "safetensors":
        from safetensors.torch import save_file

        final_path = chunk_path.with_suffix(".safetensors")
        tmp_path = final_path.with_suffix(".safetensors.tmp")
        save_file(tensors, str(tmp_path))
    elif tensor_format == "torch":
        import torch

        final_path = chunk_path.with_suffix(".pt")
        tmp_path = final_path.with_suffix(".pt.tmp")
        torch.save(tensors, tmp_path)
    else:
        raise ValueError(f"unsupported tensor_format: {tensor_format}")
    tmp_path.replace(final_path)
    return str(final_path)


def _next_chunk_index(chunks_dir: Path) -> int:
    max_index = -1
    pattern = re.compile(r"^chunk_(\d+)")
    for path in chunks_dir.glob("chunk_*"):
        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _load_existing_keys(index_path: Path) -> set[tuple[str, str]]:
    if not index_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "pass":
                keys.add((str(row.get("arm")), str(row.get("request_id"))))
    return keys


def _flush_chunk(
    *,
    capture_root: Path,
    model_key: str,
    chunk_index: int,
    tensor_buffer: dict[str, Any],
    metadata_buffer: list[dict[str, Any]],
    tensor_format: str,
) -> list[dict[str, Any]]:
    if not metadata_buffer:
        return []
    chunk_base = capture_root / "chunks" / f"chunk_{chunk_index:05d}"
    tensor_path = (
        _save_tensor_chunk(
            chunk_path=chunk_base,
            tensors=tensor_buffer,
            tensor_format=tensor_format,
        )
        if tensor_buffer
        else None
    )
    rows: list[dict[str, Any]] = []
    for row in metadata_buffer:
        row = dict(row)
        row["tensor_path"] = os.path.relpath(tensor_path, capture_root) if tensor_path else None
        rows.append(row)
    _append_jsonl(capture_root / "capture_index.jsonl", rows)
    _write_json(
        capture_root / "chunks" / f"chunk_{chunk_index:05d}.manifest.json",
        {
            "kind": "phase2_mech_job2_capture_chunk_manifest",
            "model_key": model_key,
            "chunk_index": chunk_index,
            "tensor_path": os.path.relpath(tensor_path, capture_root) if tensor_path else None,
            "row_count": len(rows),
            "tensor_count": len(tensor_buffer),
            "rows": [
                {"arm": row.get("arm"), "request_id": row.get("request_id"), "tensor_prefix": row.get("tensor_prefix")}
                for row in rows
            ],
        },
    )
    return rows


def _capture_one_row(
    *,
    runtime: Any,
    model: Any,
    model_id: str,
    processor: Any,
    tokenizer: Any,
    device: Any,
    sample: dict[str, Any],
    hook_names: list[str],
    hook_role_map: dict[str, str],
    layer_stride: int,
    topk: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    prompt = str(sample.get("prompt") or "")
    if len(prompt.strip()) < MIN_PROMPT_CHARS:
        raise RuntimeError(f"prompt too short: {len(prompt.strip())}")
    image_paths = [Path(path) for path in sample.get("input_image_paths", [])]
    prepare_start = time.perf_counter()
    inputs, prepare_trace = _prepare_inputs_multi(
        runtime=runtime,
        processor=processor,
        prompt=prompt,
        image_paths=image_paths,
        device=device,
    )
    input_prepare_elapsed_s = time.perf_counter() - prepare_start
    grid = gates._grid_from_inputs(inputs)
    visual_token_check = gates._visual_token_check(
        inputs=inputs,
        grid=grid,
        model=model,
        processor=processor,
        model_id=model_id,
    )
    sample_for_masks = dict(sample)
    sample_for_masks["model_id"] = model_id
    masks, mask_meta = _sequence_masks(
        inputs=inputs,
        model=model,
        processor=processor,
        model_id=model_id,
        sample=sample_for_masks,
    )
    handles, hook_records, hook_tensors = _register_vector_hooks(model, hook_names)
    gates._reset_cuda_peak_memory()
    try:
        start = time.perf_counter()
        with torch.inference_mode():
            output = gates._forward_model(
                model,
                inputs,
                extra_kwargs={"output_hidden_states": True},
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_elapsed_s = time.perf_counter() - start
    finally:
        for handle in handles:
            handle.remove()
    logits = gates._last_logits_slice(gates._extract_logits(output))
    selected_tensors, hidden_meta = _capture_selected_vectors(
        output=output,
        masks=masks,
        layer_stride=layer_stride,
    )
    selected_tensors.update(hook_tensors)
    logit_payload = _logit_summary(logits, tokenizer, topk=topk)
    cuda_peak = gates._cuda_peak_memory()
    input_summary = gates._inputs_summary(inputs)
    try:
        del output
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    row_pass = (
        bool(selected_tensors)
        and visual_token_check.get("status") == "pass"
        and hidden_meta.get("status") == "pass"
        and len(hook_records) == len(hook_names)
    )
    meta = {
        "status": "pass" if row_pass else "fail",
        "arm": sample.get("arm"),
        "request_id": sample.get("request_id"),
        "query": sample.get("query"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_char_count": len(prompt),
        "method_facing_pack": sample.get("method_facing_pack"),
        "method_frame_paths": sample.get("method_frame_paths", []),
        "input_image_count": sample.get("input_image_count"),
        "presentation_mode": sample.get("presentation_mode"),
        "input_prepare_elapsed_s": input_prepare_elapsed_s,
        "input_preparation_trace": prepare_trace,
        "input_summary": input_summary,
        "visual_grid": grid,
        "visual_token_check": visual_token_check,
        "mask_meta": mask_meta,
        "hidden_capture": hidden_meta,
        "hook_module_names": hook_names,
        "hook_role_map": hook_role_map,
        "hooked_modules_fired": sorted(hook_records),
        "hooked_modules_missing": sorted(set(hook_names) - set(hook_records)),
        "hook_records": hook_records,
        "forward_elapsed_s": forward_elapsed_s,
        "logits": logit_payload,
        "cuda_peak": cuda_peak,
        "tensor_count": len(selected_tensors),
        "hidden_index": sample.get("hidden_index", {}),
        "source_order_stripped_request_id": sample.get("source_order_stripped_request_id"),
        "claim_boundary": sample.get("claim_boundary"),
    }
    if meta["status"] != "pass":
        meta["reason"] = "missing_hidden_capture_tensors_visual_token_check_or_hook_parity_failed"
    return meta, selected_tensors


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = _utc_now_iso()
    _write_json(path, payload)


def _preflight_payload(
    *,
    w0_dir: Path,
    arms: list[str],
    request_offset: int,
    limit_per_arm: int | None,
) -> dict[str, Any]:
    missing: list[str] = []
    arm_rows: list[dict[str, Any]] = []
    geometry = _load_geometry_rows(w0_dir)
    for arm in arms:
        pack, pack_path = _load_pack(w0_dir, arm)
        hidden = _load_hidden_rows(w0_dir, arm)
        requests = [row for row in pack.get("requests", []) if isinstance(row, dict)]
        selected = _select_requests(requests=requests, offset=request_offset, limit=limit_per_arm)
        frame_count = 0
        for request in selected:
            for path in _request_frame_paths(request, pack_root=pack_path.parent):
                frame_count += 1
                if not path.exists():
                    missing.append(str(path))
        arm_rows.append(
            {
                "arm": arm,
                "pack_path": str(pack_path),
                "request_count_total": len(requests),
                "request_count_selected": len(selected),
                "hidden_index_count": len(hidden),
                "frame_count_selected": frame_count,
                "presentation": "true_multi_image" if arm == "sequential" else ("single_image" if arm == "single_frame" else "contact_sheet"),
            }
        )
    return {
        "kind": "phase2_mech_job2_capture_preflight",
        "w0_dir": str(w0_dir),
        "arms": arm_rows,
        "geometry_request_count": len(geometry),
        "request_offset": request_offset,
        "limit_per_arm": limit_per_arm,
        "missing": missing,
        "status": "pass" if not missing else "fail",
        "claim_boundary": "Preflight checks method-facing frame availability only; hidden metadata is not model-facing.",
    }


def preflight_inputs(args: argparse.Namespace) -> int:
    arms = _parse_csv_list(args.arms, DEFAULT_ARMS)
    limit = int(args.limit_per_arm) if args.limit_per_arm and int(args.limit_per_arm) > 0 else None
    payload = _preflight_payload(
        w0_dir=args.w0_dir.expanduser().resolve(),
        arms=arms,
        request_offset=int(args.request_offset),
        limit_per_arm=limit,
    )
    if args.out:
        _write_json(args.out.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" or not args.strict else 1


def run_model(args: argparse.Namespace) -> int:
    repo_dir = args.repo_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    w0_dir = args.w0_dir.expanduser().resolve()
    selected = _select_model_for_run(args)
    model_key = selected["model_key"]
    model_id = selected["model_id"]
    revision = selected.get("revision", "main")
    out_dir = run_dir / model_key
    capture_root = out_dir / "capture"
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_root.mkdir(parents=True, exist_ok=True)
    if not model_id:
        _write_json(
            out_dir / "capture_summary.json",
            {
                "model_key": model_key,
                "model_index": selected.get("model_index"),
                "status": "idle",
            },
        )
        return 0

    runtime = gates._load_job0_runtime(repo_dir)
    arms = _parse_csv_list(args.arms, DEFAULT_ARMS)
    limit = int(args.limit_per_arm) if args.limit_per_arm and int(args.limit_per_arm) > 0 else None
    plan_count = len(_parse_model_plan(args))
    if getattr(args, "post_load_barrier_expected_count", None):
        expected_count = int(args.post_load_barrier_expected_count)
    else:
        expected_count = min(int(args.world_size), plan_count)
    summary: dict[str, Any] = {
        "kind": "phase2_mech_job2_model_capture_summary",
        "model_key": model_key,
        "model_id": model_id,
        "model_index": selected.get("model_index"),
        "revision": revision,
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "dtype": args.dtype,
        "repo_dir": str(repo_dir),
        "w0_dir": str(w0_dir),
        "arms": arms,
        "request_offset": int(args.request_offset),
        "limit_per_arm": limit,
        "layer_stride": int(args.layer_stride),
        "chunk_size": int(args.chunk_size),
        "tensor_format": args.tensor_format,
        "claim_boundary": (
            "Compact activation capture substrate over W0 method-facing VISOR "
            "derived images. Prompts use only public/method-facing inputs; hidden "
            "labels and geometry are output metadata for offline probes only. "
            "This is not a mechanistic conclusion or intervention result."
        ),
        "started_at": _utc_now_iso(),
        "cuda": gates._cuda_snapshot(runtime),
        "runtime_versions": gates._runtime_versions(),
        "model_snapshot": gates._snapshot_provenance(
            model_id,
            revision,
            local_files_only=bool(args.local_files_only),
        ),
    }
    _write_json(out_dir / "capture_summary.json", summary)

    try:
        load_start = time.perf_counter()
        model, processor, device, adapter, tokenizer = runtime._load_model(
            model_id=model_id,
            dtype_name=args.dtype,
            trust_remote_code=True,
            local_files_only=bool(args.local_files_only),
            revision=revision,
            force_single_gpu_device_map=True,
        )
        summary["load_elapsed_s"] = time.perf_counter() - load_start
        summary["load_status"] = "pass"
        summary["adapter"] = adapter
        summary["device"] = str(device)
        summary["tokenizer_class"] = type(tokenizer).__name__ if tokenizer is not None else None
        summary["config"] = gates._summarize_config(model)
        summary["processor"] = gates._summarize_processor(processor)
        summary["model_placement"] = gates._model_placement_check(model, dtype_name=args.dtype)
        if args.post_load_barrier_dir:
            summary["post_load_barrier"] = gates._file_barrier(
                barrier_dir=args.post_load_barrier_dir.expanduser().resolve(),
                rank=int(args.rank),
                world_size=int(args.world_size),
                expected_count=expected_count,
                model_key=model_key,
                stage="post_load",
                marker_status="ready",
                timeout_s=float(args.post_load_barrier_timeout_s),
                marker_extra={"load_elapsed_s": float(summary["load_elapsed_s"])},
            )
    except Exception as exc:
        summary["load_status"] = "fail"
        summary["load_error"] = f"{type(exc).__name__}: {exc}"
        summary["overall_status"] = "fail"
        if args.post_load_barrier_dir:
            gates._write_barrier_marker(
                barrier_dir=args.post_load_barrier_dir.expanduser().resolve(),
                rank=int(args.rank),
                world_size=int(args.world_size),
                expected_count=expected_count,
                model_key=model_key,
                stage="post_load",
                status="fail",
                extra={"load_error": summary["load_error"]},
            )
        _write_json(out_dir / "capture_summary.json", summary)
        return 1 if args.strict else 0

    module_map = gates._module_map(model, max_modules=int(args.max_modules))
    _write_json(out_dir / "module_map.json", module_map)
    hook_names, hook_role_map = gates._select_hook_modules(module_map["modules"])
    summary["hook_module_names"] = hook_names
    summary["hook_role_map"] = hook_role_map
    summary["module_map_path"] = "module_map.json"

    index_sanitize = _sanitize_jsonl(capture_root / "capture_index.jsonl") if args.resume else None
    existing = _load_existing_keys(capture_root / "capture_index.jsonl") if args.resume else set()
    geometry = _load_geometry_rows(w0_dir)
    total_selected = 0
    rows_done = 0
    rows_skipped = 0
    rows_failed = 0
    chunk_index = _next_chunk_index(capture_root / "chunks")
    tensor_buffer: dict[str, Any] = {}
    metadata_buffer: list[dict[str, Any]] = []
    flushed_rows: list[dict[str, Any]] = []
    progress_path = out_dir / "progress.json"
    first_error: str | None = None
    arm_timing: dict[str, dict[str, float]] = {}
    start = time.perf_counter()
    if index_sanitize:
        summary["capture_index_sanitize"] = index_sanitize

    try:
        for arm in arms:
            pack, pack_path = _load_pack(w0_dir, arm)
            hidden = _load_hidden_rows(w0_dir, arm)
            requests = [row for row in pack.get("requests", []) if isinstance(row, dict)]
            selected_requests = _select_requests(
                requests=requests,
                offset=int(args.request_offset),
                limit=limit,
            )
            total_selected += len(selected_requests)
            _write_progress(
                progress_path,
                {
                    "status": "running",
                    "model_key": model_key,
                    "current_arm": arm,
                    "rows_done": rows_done,
                    "rows_failed": rows_failed,
                    "rows_skipped": rows_skipped,
                    "rows_total_selected_so_far": total_selected,
                },
            )
            for request in selected_requests:
                request_id = str(request.get("request_id") or "")
                key = (arm, request_id)
                if key in existing:
                    rows_skipped += 1
                    continue
                row_start = time.perf_counter()
                row_meta: dict[str, Any]
                row_tensors: dict[str, Any]
                try:
                    sample = _prepare_capture_sample(
                        w0_dir=w0_dir,
                        arm=arm,
                        request=request,
                        pack_path=pack_path,
                        hidden_by_request=hidden,
                        geometry_by_source=geometry,
                        sample_root=out_dir / "input_samples",
                    )
                    row_meta, row_tensors = _capture_one_row(
                        runtime=runtime,
                        model=model,
                        model_id=model_id,
                        processor=processor,
                        tokenizer=tokenizer,
                        device=device,
                        sample=sample,
                        hook_names=hook_names,
                        hook_role_map=hook_role_map,
                        layer_stride=int(args.layer_stride),
                        topk=int(args.topk),
                    )
                except Exception as exc:
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    rows_failed += 1
                    row_meta = {
                        "status": "fail",
                        "arm": arm,
                        "request_id": request_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_s": time.perf_counter() - row_start,
                    }
                    row_tensors = {}
                    if first_error is None:
                        first_error = row_meta["error"]
                else:
                    row_meta["elapsed_s"] = time.perf_counter() - row_start
                    if row_meta.get("status") == "pass":
                        rows_done += 1
                        existing.add(key)
                    else:
                        rows_failed += 1
                        if first_error is None:
                            first_error = str(row_meta.get("reason") or "row_status_fail")
                timing = arm_timing.setdefault(
                    arm,
                    {"rows": 0.0, "failed_rows": 0.0, "elapsed_s": 0.0},
                )
                timing["rows"] += 1.0
                timing["elapsed_s"] += float(row_meta.get("elapsed_s") or 0.0)
                if row_meta.get("status") != "pass":
                    timing["failed_rows"] += 1.0
                row_prefix = f"row_{rows_done + rows_failed + rows_skipped:07d}"
                row_meta["tensor_prefix"] = row_prefix
                prefixed = {
                    f"{row_prefix}__{_safe_tensor_key(name)}": tensor
                    for name, tensor in row_tensors.items()
                }
                if row_meta.get("status") == "pass":
                    tensor_buffer.update(prefixed)
                metadata_buffer.append(row_meta)
                if len(metadata_buffer) >= int(args.chunk_size):
                    flushed_rows.extend(
                        _flush_chunk(
                            capture_root=capture_root,
                            model_key=model_key,
                            chunk_index=chunk_index,
                            tensor_buffer=tensor_buffer,
                            metadata_buffer=metadata_buffer,
                            tensor_format=args.tensor_format,
                        )
                    )
                    chunk_index += 1
                    tensor_buffer = {}
                    metadata_buffer = []
                if (rows_done + rows_failed + rows_skipped) % max(1, int(args.flush_every_n)) == 0:
                    _write_progress(
                        progress_path,
                        {
                            "status": "running",
                            "model_key": model_key,
                            "current_arm": arm,
                            "last_request_id": request_id,
                            "rows_done": rows_done,
                            "rows_failed": rows_failed,
                            "rows_skipped": rows_skipped,
                            "rows_seen": rows_done + rows_failed + rows_skipped,
                            "elapsed_s": time.perf_counter() - start,
                        },
                    )
        if metadata_buffer:
            flushed_rows.extend(
                _flush_chunk(
                    capture_root=capture_root,
                    model_key=model_key,
                    chunk_index=chunk_index,
                    tensor_buffer=tensor_buffer,
                    metadata_buffer=metadata_buffer,
                    tensor_format=args.tensor_format,
                )
            )
    finally:
        _write_progress(
            progress_path,
            {
                "status": "finishing",
                "model_key": model_key,
                "rows_done": rows_done,
                "rows_failed": rows_failed,
                "rows_skipped": rows_skipped,
                "elapsed_s": time.perf_counter() - start,
            },
        )

    summary.update(
        {
            "ended_at": _utc_now_iso(),
            "capture_elapsed_s": time.perf_counter() - start,
            "rows_done": rows_done,
            "rows_failed": rows_failed,
            "rows_skipped": rows_skipped,
            "rows_seen": rows_done + rows_failed + rows_skipped,
            "arm_timing": {
                arm: {
                    "rows": int(timing["rows"]),
                    "failed_rows": int(timing["failed_rows"]),
                    "elapsed_s": timing["elapsed_s"],
                    "mean_row_elapsed_s": (
                        timing["elapsed_s"] / timing["rows"]
                        if timing["rows"]
                        else None
                    ),
                }
                for arm, timing in sorted(arm_timing.items())
            },
            "chunk_count": len(list((capture_root / "chunks").glob("chunk_*.manifest.json"))),
            "capture_index": "capture/capture_index.jsonl",
            "progress_path": "progress.json",
            "first_error": first_error,
        }
    )
    summary["pass_fail"] = {
        "model_loaded": summary.get("load_status") == "pass",
        "cuda_visible_single_gpu": summary.get("cuda", {}).get("status") == "pass",
        "model_on_cuda_bf16": summary.get("model_placement", {}).get("status") == "pass",
        "post_load_barrier_passed": summary.get("post_load_barrier", {}).get("status", "pass") == "pass",
        "module_map_written": bool(module_map.get("modules")),
        "hook_modules_selected": bool(hook_names),
        "rows_captured": rows_done > 0,
        "no_row_failures": rows_failed == 0,
    }
    summary["overall_status"] = "pass" if all(summary["pass_fail"].values()) else "fail"
    _write_json(out_dir / "capture_summary.json", summary)
    _write_progress(progress_path, {**summary, "status": summary["overall_status"]})
    return 0 if summary["overall_status"] == "pass" or not args.strict else 1


def _summarize_gpu_csv(path: Path) -> dict[str, Any]:
    rows = 0
    util: list[float] = []
    mem: list[float] = []
    power: list[float] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            for row in reader:
                rows += 1
                for key, bucket in (
                    ("utilization_gpu_pct", util),
                    ("memory_used_mib", mem),
                    ("power_draw_w", power),
                ):
                    try:
                        value = row.get(key)
                        if value not in (None, "", "N/A", "[N/A]"):
                            bucket.append(float(str(value).strip()))
                    except Exception:
                        pass
    return {
        "path": str(path),
        "rows": rows,
        "median_utilization_gpu_pct": _percentile(util, 0.5),
        "p95_utilization_gpu_pct": _percentile(util, 0.95),
        "max_memory_used_mib": max(mem) if mem else None,
        "median_power_draw_w": _percentile(power, 0.5),
        "status": "pass" if rows and util else "missing_or_empty",
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Mechanism Job2 Battery Capture",
        "",
        f"Status: `{payload['overall_status']}`",
        f"Run dir: `{payload['run_dir']}`",
        "",
        "| Model | Status | Done | Failed | Skipped | Chunks | Capture min |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("models", []):
        elapsed = row.get("capture_elapsed_s")
        elapsed_min = f"{float(elapsed) / 60:.1f}" if isinstance(elapsed, (int, float)) else ""
        lines.append(
            "| {model_key} | {status} | {done} | {failed} | {skipped} | {chunks} | {elapsed} |".format(
                model_key=row.get("model_key"),
                status=row.get("overall_status"),
                done=row.get("rows_done"),
                failed=row.get("rows_failed"),
                skipped=row.get("rows_skipped"),
                chunks=row.get("chunk_count"),
                elapsed=elapsed_min,
            )
        )
    gpu_rows = payload.get("gpu_monitor_summary", [])
    if gpu_rows:
        lines.extend(
            [
                "",
                "## GPU Monitor Summary",
                "",
                "| Rank | Model | Rows | Median util % | P95 util % | Max memory MiB | Median power W |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in gpu_rows:
            gpu = row.get("gpu", {})
            lines.append(
                "| {rank} | {model_key} | {rows} | {median} | {p95} | {mem} | {power} |".format(
                    rank=row.get("rank"),
                    model_key=row.get("model_key"),
                    rows=gpu.get("rows"),
                    median=_fmt_float(gpu.get("median_utilization_gpu_pct")),
                    p95=_fmt_float(gpu.get("p95_utilization_gpu_pct")),
                    mem=_fmt_float(gpu.get("max_memory_used_mib")),
                    power=_fmt_float(gpu.get("median_power_draw_w")),
                )
            )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            payload["claim_boundary"],
            "",
            "This artifact is a compact capture substrate for probes, logit/lens",
            "analysis, and V-information accounting. It is not a behavioral score,",
            "mechanistic conclusion, or causal-intervention result.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt_float(value: Any) -> str:
    return f"{float(value):.1f}" if isinstance(value, (int, float)) else ""


def aggregate(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    plan = _parse_model_plan(args)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for model in plan:
        path = run_dir / model["model_key"] / "capture_summary.json"
        if not path.exists():
            missing.append(model["model_key"])
            continue
        rows.append(_read_json(path))
    gpu_rows = []
    for row in rows:
        rank = row.get("rank")
        if isinstance(rank, int):
            gpu_rows.append(
                {
                    "rank": rank,
                    "model_key": row.get("model_key"),
                    "gpu": _summarize_gpu_csv(run_dir / f"gpu_monitor_rank_{rank}.csv"),
                }
            )
    payload = {
        "kind": "phase2_mech_job2_battery_capture_summary",
        "run_dir": str(run_dir),
        "expected_models": plan,
        "missing_models": missing,
        "models": rows,
        "gpu_monitor_summary": gpu_rows,
        "claim_boundary": (
            "Compact activation capture substrate over W0 method-facing VISOR "
            "derived images. It supports offline probes/lens/V-information analysis; "
            "it is not itself a mechanistic conclusion or causal-intervention result."
        ),
    }
    payload["overall_status"] = (
        "pass"
        if not missing and rows and all(row.get("overall_status") == "pass" for row in rows)
        else "fail"
    )
    _write_json(run_dir / "capture_summary.json", payload)
    (run_dir / "capture_summary.md").write_text(_markdown_summary(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["overall_status"] == "pass" or not args.strict else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    common.add_argument("--w0-dir", type=Path, default=DEFAULT_W0_DIR)
    common.add_argument("--run-dir", type=Path, required=True)
    common.add_argument("--model-plan-file", type=Path)
    common.add_argument("--model-plan-json")
    common.add_argument("--model-plan-env")
    common.add_argument("--strict", action="store_true")

    run = subparsers.add_parser("run-model", parents=[common])
    run.add_argument("--rank", required=True)
    run.add_argument("--world-size", required=True)
    run.add_argument("--model-index", type=int)
    run.add_argument("--model-key")
    run.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    run.add_argument("--request-offset", type=int, default=0)
    run.add_argument("--limit-per-arm", type=int, default=0)
    run.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    run.add_argument("--flush-every-n", type=int, default=DEFAULT_FLUSH_EVERY_N)
    run.add_argument("--layer-stride", type=int, default=DEFAULT_LAYER_STRIDE)
    run.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    run.add_argument("--tensor-format", choices=["safetensors", "torch"], default="safetensors")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-modules", type=int, default=20000)
    run.add_argument("--post-load-barrier-dir", type=Path)
    run.add_argument("--post-load-barrier-timeout-s", type=float, default=DEFAULT_POST_LOAD_BARRIER_TIMEOUT)
    run.add_argument("--post-load-barrier-expected-count", type=int)

    subparsers.add_parser("aggregate", parents=[common])

    preflight = subparsers.add_parser("preflight-inputs")
    preflight.add_argument("--w0-dir", type=Path, default=DEFAULT_W0_DIR)
    preflight.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    preflight.add_argument("--request-offset", type=int, default=0)
    preflight.add_argument("--limit-per-arm", type=int, default=0)
    preflight.add_argument("--out", type=Path)
    preflight.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run-model":
        return run_model(args)
    if args.command == "aggregate":
        return aggregate(args)
    if args.command == "preflight-inputs":
        return preflight_inputs(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
