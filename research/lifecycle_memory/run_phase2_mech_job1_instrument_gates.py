#!/usr/bin/env python3
"""Run mechanism-study instrument gates on TamIA.

This gate is deliberately small. It loads one VLM per GPU worker, prepares two
W0 method-facing image samples, verifies visual-token evidence, hidden-state
shape availability, per-role hook firing, and hook parity, then writes compact
module/input summaries. It does not score model behavior and does not expose
hidden answers to prompts.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory.protocol_v3_3rscan_visual_vlm import build_contact_sheet_image


DEFAULT_W0_DIR = Path("runs/research/lifecycle_memory/phase_2/w0_geometry_sidecar_2026_07_07")
DEFAULT_MODEL_PLAN = [
    {"model_key": "qwen35_4b_hf_bf16", "model_id": "Qwen/Qwen3.5-4B", "revision": "main"},
    {"model_key": "qwen35_27b_hf_bf16", "model_id": "Qwen/Qwen3.5-27B", "revision": "main"},
    {"model_key": "gemma4_12b_hf_bf16", "model_id": "google/gemma-4-12B-it", "revision": "main"},
    {
        "model_key": "gemma4_26b_a4b_hf_bf16",
        "model_id": "google/gemma-4-26B-A4B-it",
        "revision": "main",
    },
]
PASS_MAX_LOGIT_DIFF = 1e-3
MIN_PROMPT_CHARS = 10
DEFAULT_PROBE_WARMUP = 2
DEFAULT_PROBE_REPEAT = 10
DEFAULT_PROBE_MIN_SECONDS = 60.0
DEFAULT_POST_LOAD_BARRIER_TIMEOUT = 900.0
DEFAULT_PROBE_HIDDEN_REPEAT = 3


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe[:120] or "item"


def _load_job0_runtime(repo_dir: Path) -> Any:
    runtime_path = (
        repo_dir
        / "docs/research/lifecycle_memory/phase 2 research/jobs/job0/run_phase1_qwen27b_hf.py"
    )
    if not runtime_path.exists():
        raise FileNotFoundError(f"missing proven HF runtime loader: {runtime_path}")
    spec = importlib.util.spec_from_file_location("phase1_qwen27b_hf_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import runtime loader from {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_model_plan(args: argparse.Namespace) -> list[dict[str, str]]:
    if getattr(args, "model_plan_file", None):
        plan = _read_json(args.model_plan_file.expanduser().resolve())
    elif args.model_plan_json:
        plan = json.loads(args.model_plan_json)
    elif args.model_plan_env and os.environ.get(args.model_plan_env):
        plan = json.loads(os.environ[args.model_plan_env])
    else:
        plan = DEFAULT_MODEL_PLAN
    if not isinstance(plan, list) or not plan:
        raise ValueError("model plan must be a non-empty JSON list")
    normalized: list[dict[str, str]] = []
    for row in plan:
        if not isinstance(row, dict):
            raise ValueError(f"model plan row is not an object: {row!r}")
        model_id = str(row.get("model_id") or "").strip()
        model_key = str(row.get("model_key") or "").strip()
        if not model_id or not model_key:
            raise ValueError(f"model plan row missing model_id/model_key: {row!r}")
        revision = str(row.get("revision") or "main").strip()
        normalized.append({"model_id": model_id, "model_key": model_key, "revision": revision})
    return normalized


def _select_model_for_rank(args: argparse.Namespace) -> dict[str, str]:
    plan = _parse_model_plan(args)
    rank = int(args.rank)
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    if rank >= len(plan):
        return {"model_id": "", "model_key": f"idle_rank_{rank:03d}", "revision": ""}
    return plan[rank]


def _role_for_module(name: str, class_name: str) -> str:
    needle = f"{name}.{class_name}".lower()
    if "audio" in needle:
        return "audio"
    if any(part in needle for part in ("vision", "visual", "vit", "siglip", "clip", "image")):
        return "vision"
    if any(part in needle for part in ("projector", "merger", "multi_modal", "multimodal", "mm_projector")):
        return "projector"
    if any(part in needle for part in ("decoderlayer", "model.layers", ".layers.", ".h.", "block")):
        return "language_block"
    if "attention" in needle or "attn" in needle:
        return "attention"
    if "embed" in needle:
        return "embedding"
    return "other"


def _direct_param_count(module: Any) -> int:
    try:
        return int(sum(param.numel() for param in module.parameters(recurse=False)))
    except Exception:
        return 0


def _module_map(model: Any, *, max_modules: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}
    for index, (name, module) in enumerate(model.named_modules()):
        if index >= max_modules:
            break
        class_name = type(module).__name__
        role = _role_for_module(name, class_name)
        role_counts[role] = role_counts.get(role, 0) + 1
        try:
            child_count = len(list(module.children()))
        except Exception:
            child_count = None
        rows.append(
            {
                "name": name,
                "class_name": class_name,
                "role": role,
                "direct_param_count": _direct_param_count(module),
                "child_count": child_count,
            }
        )
    return {
        "module_count_recorded": len(rows),
        "max_modules": max_modules,
        "role_counts": role_counts,
        "modules": rows,
    }


def _looks_like_container_module(row: dict[str, Any]) -> bool:
    class_name = str(row.get("class_name") or "")
    if class_name in {"ModuleList", "ModuleDict", "ParameterList", "ParameterDict", "Sequential"}:
        return True
    name = str(row.get("name") or "").lower()
    return name.endswith((".layers", ".blocks", ".h"))


def _select_hook_modules(
    module_rows: list[dict[str, Any]],
    *,
    max_per_role: int = 3,
) -> tuple[list[str], dict[str, str]]:
    selected: list[str] = []
    role_map: dict[str, str] = {}
    for role in ("vision", "projector", "language_block"):
        role_rows = [
            row
            for row in module_rows
            if row.get("role") == role
            and str(row.get("name") or "")
            and not _looks_like_container_module(row)
        ]
        if not role_rows:
            continue
        indexes = sorted({0, len(role_rows) // 2, len(role_rows) - 1})
        for index in indexes[:max_per_role]:
            name = str(role_rows[index].get("name") or "")
            if name and name not in role_map:
                selected.append(name)
                role_map[name] = role
    return selected, role_map


def _summarize_config(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {"available": False}
    payload: dict[str, Any] = {"available": True, "class_name": type(config).__name__}
    for key in (
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
        "image_token_id",
        "image_token_index",
        "image_token",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "spatial_merge_size",
        "merge_size",
        "_attn_implementation",
    ):
        if hasattr(config, key):
            value = getattr(config, key)
            if isinstance(value, (str, int, float, bool, list, tuple, type(None))):
                payload[key] = value
    for nested_name in ("vision_config", "text_config"):
        nested = getattr(config, nested_name, None)
        if nested is None:
            continue
        nested_payload: dict[str, Any] = {"class_name": type(nested).__name__}
        for key in (
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "image_size",
            "patch_size",
            "spatial_merge_size",
            "merge_size",
            "num_channels",
            "intermediate_size",
            "image_token_id",
            "image_token_index",
        ):
            if hasattr(nested, key):
                value = getattr(nested, key)
                if isinstance(value, (str, int, float, bool, list, tuple, type(None))):
                    nested_payload[key] = value
        payload[nested_name] = nested_payload
    return payload


def _summarize_processor(processor: Any) -> dict[str, Any]:
    if processor is None:
        return {"available": False}
    payload: dict[str, Any] = {"available": True, "class_name": type(processor).__name__}
    for child_name in ("tokenizer", "image_processor", "video_processor", "feature_extractor"):
        child = getattr(processor, child_name, None)
        if child is not None:
            payload[child_name] = {"class_name": type(child).__name__}
            for key in ("size", "crop_size", "patch_size", "image_size", "min_pixels", "max_pixels"):
                if hasattr(child, key):
                    value = getattr(child, key)
                    if isinstance(value, (str, int, float, bool, dict, list, tuple, type(None))):
                        payload[child_name][key] = value
    for key in (
        "min_pixels",
        "max_pixels",
        "image_seq_length",
        "num_image_tokens",
        "patch_size",
        "image_token_id",
        "image_token_index",
        "spatial_merge_size",
        "merge_size",
    ):
        if hasattr(processor, key):
            value = getattr(processor, key)
            if isinstance(value, (str, int, float, bool, dict, list, tuple, type(None))):
                payload[key] = value
    return payload


def _tensor_payload(value: Any) -> dict[str, Any] | None:
    if not hasattr(value, "shape"):
        return None
    payload = {
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(getattr(value, "dtype", None)),
        "device": str(getattr(value, "device", None)),
    }
    try:
        payload["numel"] = int(value.numel())
    except Exception:
        pass
    return payload


def _input_get(inputs: Any, key: str, default: Any = None) -> Any:
    if isinstance(inputs, dict):
        return inputs.get(key, default)
    getter = getattr(inputs, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                return getter(key)
            except Exception:
                return default
        except Exception:
            return default
    try:
        if key in inputs:
            return inputs[key]
    except Exception:
        pass
    return default


def _input_items(inputs: Any) -> list[tuple[str, Any]]:
    if isinstance(inputs, dict):
        return list(inputs.items())
    items = getattr(inputs, "items", None)
    if callable(items):
        try:
            return [(str(key), value) for key, value in items()]
        except Exception:
            pass
    keys = getattr(inputs, "keys", None)
    if callable(keys):
        rows: list[tuple[str, Any]] = []
        try:
            for key in keys():
                rows.append((str(key), _input_get(inputs, str(key))))
        except Exception:
            return []
        return rows
    return []


def _inputs_summary(inputs: Any) -> dict[str, Any]:
    items = _input_items(inputs)
    if items:
        return {
            key: (_tensor_payload(value) or {"type": type(value).__name__})
            for key, value in items
        }
    return {"type": type(inputs).__name__, "tensor": _tensor_payload(inputs)}


def _first_tensor(value: Any) -> Any | None:
    if hasattr(value, "shape"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _activation_summary(value: Any) -> dict[str, Any]:
    tensor = _first_tensor(value)
    if tensor is None:
        return {"type": type(value).__name__, "tensor_found": False}
    summary = _tensor_payload(tensor) or {}
    try:
        import torch

        with torch.no_grad():
            sample = tensor.detach()
            if sample.numel() > 4096:
                sample = sample.reshape(-1)[:4096]
            sample = sample.float()
            summary.update(
                {
                    "finite": bool(torch.isfinite(sample).all().item()),
                    "mean": float(sample.mean().item()),
                    "std": float(sample.std(unbiased=False).item()),
                    "absmax": float(sample.abs().max().item()),
                }
            )
    except Exception as exc:
        summary["stats_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _extract_logits(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return None


def _extract_output_field(output: Any, field_name: str) -> Any:
    if hasattr(output, field_name):
        return getattr(output, field_name)
    if isinstance(output, dict):
        return output.get(field_name)
    return None


def _forward_model(model: Any, inputs: Any, *, extra_kwargs: dict[str, Any] | None = None) -> Any:
    attempts = (
        {"use_cache": False, "return_dict": True},
        {"return_dict": True},
        {},
    )
    extra_kwargs = extra_kwargs or {}
    last_error: Exception | None = None
    for kwargs in attempts:
        kwargs = {**kwargs, **extra_kwargs}
        try:
            return model(**inputs, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
    assert last_error is not None
    raise last_error


def _last_logits_slice(logits: Any) -> Any:
    if logits is None or not hasattr(logits, "detach"):
        return None
    if len(logits.shape) == 3:
        return logits[:, -1, :].detach().float().cpu()
    if len(logits.shape) == 2:
        return logits[-1:, :].detach().float().cpu()
    return logits.reshape(1, -1).detach().float().cpu()


def _compare_logits(first: Any, second: Any) -> dict[str, Any]:
    if first is None or second is None:
        return {"status": "fail", "reason": "missing_logits"}
    import torch

    if first.shape != second.shape:
        return {
            "status": "fail",
            "reason": "shape_mismatch",
            "first_shape": list(first.shape),
            "second_shape": list(second.shape),
        }
    diff = (first - second).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    return {
        "status": "pass" if max_abs <= PASS_MAX_LOGIT_DIFF else "fail",
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "threshold": PASS_MAX_LOGIT_DIFF,
        "shape": list(first.shape),
        "allclose_1e_3": bool(torch.allclose(first, second, atol=1e-3, rtol=1e-3)),
    }


def _logits_sanity(logits: Any) -> dict[str, Any]:
    if logits is None or not hasattr(logits, "detach"):
        return {"status": "fail", "reason": "missing_logits"}
    import torch

    with torch.no_grad():
        sample = logits.detach().float()
        finite = bool(torch.isfinite(sample).all().item())
        std = float(sample.std(unbiased=False).item()) if sample.numel() else 0.0
        min_value = float(sample.min().item()) if sample.numel() else 0.0
        max_value = float(sample.max().item()) if sample.numel() else 0.0
        argmax = int(sample.reshape(-1).argmax().item()) if sample.numel() else None
    return {
        "status": "pass" if finite and std > 0.0 and max_value != min_value else "fail",
        "finite": finite,
        "std": std,
        "min": min_value,
        "max": max_value,
        "argmax_flat_index": argmax,
    }


def _iter_config_objects(config: Any) -> list[Any]:
    if config is None:
        return []
    objects = [config]
    for name in ("text_config", "vision_config", "vision_config_1", "vision_config_2"):
        child = getattr(config, name, None)
        if child is not None:
            objects.append(child)
    return objects


def _first_int_attr(config: Any, names: tuple[str, ...]) -> int | None:
    for obj in _iter_config_objects(config):
        for name in names:
            value = getattr(obj, name, None)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return int(value)
    return None


def _image_token_candidates(config: Any, processor: Any) -> list[int]:
    candidates: list[int] = []
    attr_names = (
        "image_token_id",
        "image_token_index",
        "image_token",
        "image_placeholder_id",
    )
    for obj in [*_iter_config_objects(config), processor, getattr(processor, "tokenizer", None)]:
        if obj is None:
            continue
        for attr in attr_names:
            value = getattr(obj, attr, None)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                candidates.append(int(value))
            elif isinstance(value, (list, tuple)):
                candidates.extend(int(item) for item in value if isinstance(item, int) and not isinstance(item, bool))
    return sorted({value for value in candidates if value >= 0})


def _merge_size(config: Any, processor: Any) -> int | None:
    value = _first_int_attr(
        config,
        (
            "spatial_merge_size",
            "merge_size",
            "image_merge_size",
            "vision_merge_size",
        ),
    )
    if value:
        return value
    for obj in (processor, getattr(processor, "image_processor", None)):
        if obj is None:
            continue
        for name in ("spatial_merge_size", "merge_size", "image_merge_size"):
            raw = getattr(obj, name, None)
            if isinstance(raw, int) and raw > 0:
                return int(raw)
    return None


def _model_family(config: Any, model_id: str) -> str:
    model_type = str(getattr(config, "model_type", "") or "").lower()
    text = f"{model_type} {model_id}".lower()
    if "qwen" in text:
        return "qwen"
    if "gemma" in text:
        return "gemma"
    return model_type or "unknown"


def _grid_triplets(value: Any) -> list[tuple[int, int, int]]:
    if isinstance(value, list):
        if (
            len(value) >= 3
            and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value[:3])
        ):
            return [(int(value[0]), int(value[1]), int(value[2]))]
        triplets: list[tuple[int, int, int]] = []
        for item in value:
            triplets.extend(_grid_triplets(item))
        return triplets
    return []


def _grid_premerge_token_count(grid: dict[str, Any]) -> int | None:
    key = str(grid.get("grid_key") or "")
    data = grid.get(key)
    triplets = _grid_triplets(data) if key else []
    total = sum(t * h * w for t, h, w in triplets)
    return total if total > 0 else None


def _pixel_tensor_check(inputs: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "fail", "pixel_keys": []}
    if not _input_items(inputs):
        payload["reason"] = "inputs_not_mapping_like"
        return payload
    import torch

    checks = []
    for key in ("pixel_values", "pixel_values_videos"):
        value = _input_get(inputs, key)
        if value is None or not hasattr(value, "detach"):
            continue
        tensor = value.detach()
        item: dict[str, Any] = {
            "key": key,
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": int(tensor.numel()),
        }
        if tensor.numel():
            sample = tensor.reshape(-1)[:4096].float()
            item["finite_sample"] = bool(torch.isfinite(sample).all().item())
            item["absmax_sample"] = float(sample.abs().max().item())
            item["std_sample"] = float(sample.std(unbiased=False).item())
        else:
            item["finite_sample"] = False
            item["absmax_sample"] = 0.0
            item["std_sample"] = 0.0
        checks.append(item)
    payload["pixel_keys"] = [item["key"] for item in checks]
    payload["checks"] = checks
    ok = any(
        item.get("numel", 0) > 0
        and item.get("finite_sample")
        and item.get("absmax_sample", 0.0) > 0.0
        for item in checks
    )
    payload["status"] = "pass" if ok else "fail"
    if not ok:
        payload["reason"] = "no_nonempty_finite_pixel_tensor"
    return payload


def _visual_token_check(
    *,
    inputs: Any,
    grid: dict[str, Any],
    model: Any,
    processor: Any,
    model_id: str,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    family = _model_family(config, model_id)
    candidates = _image_token_candidates(config, processor)
    payload: dict[str, Any] = {
        "model_family": family,
        "image_token_candidates": candidates,
        "status": "fail",
    }
    payload["pixel_tensor_check"] = _pixel_tensor_check(inputs)
    input_ids_value = _input_get(inputs, "input_ids")
    if input_ids_value is None or not hasattr(input_ids_value, "detach"):
        if payload["pixel_tensor_check"].get("status") == "pass":
            payload["status"] = "pass"
            payload["reason"] = "pixel_tensor_present_without_input_ids"
            payload["evidence_level"] = "pixel_tensor_only"
            return payload
        payload["reason"] = "missing_input_ids"
        return payload
    if not candidates:
        if payload["pixel_tensor_check"].get("status") == "pass":
            payload["status"] = "pass"
            payload["reason"] = "pixel_tensor_present_without_image_token_candidate"
            payload["evidence_level"] = "pixel_tensor_only"
            return payload
        payload["reason"] = "no_image_token_candidate_found"
        return payload

    input_ids = input_ids_value.detach()
    counts: dict[str, int] = {}
    total = 0
    for candidate in candidates:
        count = int((input_ids == int(candidate)).sum().item())
        counts[str(candidate)] = count
        total += count
    payload["image_token_counts"] = counts
    payload["image_token_total"] = total

    if total <= 0:
        if payload["pixel_tensor_check"].get("status") == "pass":
            payload["status"] = "pass"
            payload["reason"] = "pixel_tensor_present_without_image_tokens"
            payload["evidence_level"] = "pixel_tensor_only"
            return payload
        payload["reason"] = "no_image_tokens_in_input_ids"
        return payload

    premerge = _grid_premerge_token_count(grid)
    merge = _merge_size(config, processor)
    payload["grid_premerge_tokens"] = premerge
    payload["merge_size"] = merge
    if family == "qwen":
        if not grid.get("available"):
            payload["reason"] = "qwen_grid_metadata_missing"
            return payload
        if premerge is None:
            payload["reason"] = "qwen_premerge_grid_count_missing"
            return payload
        if not merge:
            payload["reason"] = "qwen_merge_size_missing"
            return payload
        expected = premerge // (merge * merge)
        payload["expected_postmerge_image_tokens"] = expected
        if expected != total:
            payload["reason"] = "qwen_image_token_count_mismatch"
            return payload
        if payload["pixel_tensor_check"].get("status") != "pass":
            payload["reason"] = "qwen_pixel_tensor_check_failed"
            return payload
        payload["status"] = "pass"
        payload["reason"] = "qwen_image_tokens_and_grid_present"
        payload["evidence_level"] = "image_tokens_plus_grid"
        return payload

    if payload["pixel_tensor_check"].get("status") != "pass":
        payload["reason"] = "image_tokens_present_but_pixel_tensor_check_failed"
        return payload
    payload["status"] = "pass"
    payload["reason"] = "image_tokens_and_pixels_present"
    payload["evidence_level"] = "image_tokens_plus_pixels"
    return payload


def _expected_hidden_state_count(config: Any) -> int | None:
    layers = _first_int_attr(config, ("num_hidden_layers",))
    return layers + 1 if layers is not None else None


def _tensor_shape_list(value: Any) -> list[int] | None:
    if not hasattr(value, "shape"):
        return None
    return [int(dim) for dim in value.shape]


def _sequence_tensor_summary(values: Any) -> dict[str, Any]:
    if not isinstance(values, (list, tuple)):
        return {"available": False, "type": type(values).__name__}
    payload: dict[str, Any] = {
        "available": True,
        "count": len(values),
        "first_shape": _tensor_shape_list(values[0]) if values else None,
        "last_shape": _tensor_shape_list(values[-1]) if values else None,
        "first_dtype": str(getattr(values[0], "dtype", None)) if values else None,
        "last_dtype": str(getattr(values[-1], "dtype", None)) if values else None,
    }
    return payload


def _cuda_peak_memory() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        torch.cuda.synchronize()
        total_memory = int(torch.cuda.get_device_properties(0).total_memory)
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
        return {
            "available": True,
            "total_memory": total_memory,
            "max_memory_allocated": allocated,
            "max_memory_reserved": reserved,
            "allocated_fraction": allocated / total_memory if total_memory else None,
            "reserved_fraction": reserved / total_memory if total_memory else None,
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _reset_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _timed_forward(model: Any, inputs: Any, *, extra_kwargs: dict[str, Any] | None = None) -> tuple[Any, float]:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = _forward_model(model, inputs, extra_kwargs=extra_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return output, time.perf_counter() - start


def _hidden_state_probe(model: Any, inputs: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    expected = _expected_hidden_state_count(config)
    payload: dict[str, Any] = {"expected_count": expected}
    try:
        import torch

        _reset_cuda_peak_memory()
        output, elapsed_s = _timed_forward(
            model,
            inputs,
            extra_kwargs={"output_hidden_states": True},
        )
        payload["forward_elapsed_s"] = elapsed_s
        hidden_states = _extract_output_field(output, "hidden_states")
        payload.update(_sequence_tensor_summary(hidden_states))
        payload["cuda_peak"] = _cuda_peak_memory()
        count = payload.get("count")
        payload["status"] = (
            "pass"
            if payload.get("available") and count and (expected is None or count == expected)
            else "fail"
        )
        if payload["status"] == "fail":
            payload["reason"] = "hidden_state_count_mismatch_or_missing"
    except Exception as exc:
        payload["status"] = "fail"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["cuda_peak"] = _cuda_peak_memory()
    return payload


def _attention_probe(model: Any, inputs: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        import torch

        _reset_cuda_peak_memory()
        output, elapsed_s = _timed_forward(
            model,
            inputs,
            extra_kwargs={"output_attentions": True},
        )
        payload["forward_elapsed_s"] = elapsed_s
        attentions = _extract_output_field(output, "attentions")
        payload.update(_sequence_tensor_summary(attentions))
        payload["cuda_peak"] = _cuda_peak_memory()
        payload["status"] = "pass" if payload.get("available") and payload.get("count") else "unsupported"
        if payload["status"] != "pass":
            payload["reason"] = "attention_tensors_missing"
    except Exception as exc:
        payload["status"] = "unsupported"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["cuda_peak"] = _cuda_peak_memory()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return payload


def _model_placement_check(model: Any, *, dtype_name: str) -> dict[str, Any]:
    expected_dtype = {
        "bf16": "torch.bfloat16",
        "bfloat16": "torch.bfloat16",
        "fp16": "torch.float16",
        "float16": "torch.float16",
        "fp32": "torch.float32",
        "float32": "torch.float32",
    }.get(dtype_name.lower())
    devices: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    sampled = 0
    for param in model.parameters():
        devices[str(param.device).split(":")[0]] += 1
        dtypes[str(param.dtype)] += 1
        sampled += 1
        if sampled >= 256:
            break
    cuda_only = bool(devices) and set(devices) == {"cuda"}
    dtype_ok = expected_dtype is None or (bool(dtypes) and set(dtypes) == {expected_dtype})
    return {
        "status": "pass" if cuda_only and dtype_ok else "fail",
        "sampled_parameter_count": sampled,
        "device_counts": dict(devices),
        "dtype_counts": dict(dtypes),
        "expected_dtype": expected_dtype,
        "cuda_only": cuda_only,
        "dtype_ok": dtype_ok,
    }


def _runtime_versions() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for module_name in ("torch", "transformers", "huggingface_hub", "accelerate"):
        try:
            module = __import__(module_name)
            payload[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            payload[module_name] = f"unavailable: {type(exc).__name__}: {exc}"
    return payload


def _snapshot_provenance(model_id: str, revision: str, *, local_files_only: bool) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(model_id, revision=revision or None, local_files_only=local_files_only)
        )
        resolved = path.name if path.parent.name == "snapshots" else None
        return {
            "status": "pass",
            "requested_revision": revision,
            "snapshot_path": str(path),
            "resolved_snapshot_revision": resolved,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "requested_revision": revision,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _register_summary_hooks(model: Any, names: list[str]) -> tuple[list[Any], dict[str, Any]]:
    module_lookup = dict(model.named_modules())
    records: dict[str, Any] = {}
    handles: list[Any] = []

    def make_hook(module_name: str) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if module_name not in records:
                records[module_name] = _activation_summary(output)

        return hook

    for name in names:
        module = module_lookup.get(name)
        if module is None:
            continue
        handles.append(module.register_forward_hook(make_hook(name)))
    return handles, records


def _resolve_frame_path(frame_path: str, *, pack_root: Path) -> Path:
    path = Path(frame_path)
    if not path.is_absolute():
        path = pack_root / path
    return path.resolve()


def _load_sample_request(w0_dir: Path, arm: str, sample_index: int) -> tuple[dict[str, Any], Path]:
    pack_path = w0_dir / "derived_pack" / f"order_stripped_{arm}_request_pack.json"
    payload = _read_json(pack_path)
    requests = [row for row in payload.get("requests", []) if isinstance(row, dict)]
    if not requests:
        raise RuntimeError(f"no requests in {pack_path}")
    return requests[sample_index % len(requests)], pack_path


def _prepare_samples(w0_dir: Path, out_dir: Path, sample_index: int) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []

    visual_request, visual_pack = _load_sample_request(w0_dir, "visual_prompt", sample_index)
    visual_pack_root = visual_pack.parent
    visual_paths = [
        _resolve_frame_path(str(frame["path"]), pack_root=visual_pack_root)
        for frame in visual_request.get("frames", [])
        if isinstance(frame, dict) and frame.get("path")
    ]
    contact_sheet = out_dir / f"{_safe_id(str(visual_request.get('request_id', 'visual')))}__visual_prompt_sheet.jpg"
    sheet_builder = {
        "thumb_width": 360,
        "thumb_height": 220,
        "columns": 2,
        "show_labels": False,
    }
    build_contact_sheet_image(
        request=visual_request,
        frame_paths=visual_paths,
        out_path=contact_sheet,
        thumb_width=sheet_builder["thumb_width"],
        thumb_height=sheet_builder["thumb_height"],
        columns=sheet_builder["columns"],
        show_labels=sheet_builder["show_labels"],
    )
    samples.append(
        {
            "sample_key": "visual_prompt_contact_sheet",
            "arm": "visual_prompt",
            "request_id": visual_request.get("request_id"),
            "query": visual_request.get("query"),
            "prompt": visual_request.get("user_prompt"),
            "image_path": str(contact_sheet.resolve()),
            "source_frame_count": len(visual_paths),
            "method_facing_pack": str(visual_pack),
            "sheet_builder": sheet_builder,
        }
    )

    single_request, single_pack = _load_sample_request(w0_dir, "single_frame", sample_index)
    frames = [
        frame for frame in single_request.get("frames", [])
        if isinstance(frame, dict) and frame.get("path")
    ]
    if not frames:
        raise RuntimeError("single_frame sample has no frame path")
    single_path = _resolve_frame_path(str(frames[0]["path"]), pack_root=single_pack.parent)
    samples.append(
        {
            "sample_key": "single_frame",
            "arm": "single_frame",
            "request_id": single_request.get("request_id"),
            "query": single_request.get("query"),
            "prompt": single_request.get("user_prompt"),
            "image_path": str(single_path),
            "source_frame_count": 1,
            "method_facing_pack": str(single_pack),
        }
    )
    _write_json(out_dir / "sample_manifest.json", {"samples": samples})
    return samples


def _preflight_input_frames(w0_dir: Path, sample_index: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm in ("visual_prompt", "single_frame"):
        request, pack_path = _load_sample_request(w0_dir, arm, sample_index)
        frame_paths = [
            _resolve_frame_path(str(frame["path"]), pack_root=pack_path.parent)
            for frame in request.get("frames", [])
            if isinstance(frame, dict) and frame.get("path")
        ]
        if arm == "single_frame":
            frame_paths = frame_paths[:1]
        prompt = str(request.get("user_prompt") or "")
        row = {
            "arm": arm,
            "request_id": request.get("request_id"),
            "pack_path": str(pack_path),
            "frame_count": len(frame_paths),
            "prompt_length": len(prompt),
            "prompt_check": "pass" if len(prompt.strip()) >= MIN_PROMPT_CHARS else "fail",
            "frames": [
                {"path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else None}
                for path in frame_paths
            ],
        }
        for path in frame_paths:
            if not path.exists():
                missing.append(str(path))
        if row["prompt_check"] != "pass":
            missing.append(f"{arm}: prompt too short ({len(prompt.strip())} chars)")
        rows.append(row)
    return {
        "kind": "phase2_mech_job1_input_preflight",
        "w0_dir": str(w0_dir),
        "sample_index": sample_index,
        "status": "pass" if not missing else "fail",
        "missing_or_invalid": missing,
        "arms": rows,
    }


def _prepare_inputs_with_trace(
    *,
    runtime: Any,
    processor: Any,
    prompt: str,
    image_path: Path,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    trace: dict[str, Any] = {"attempts": []}
    if hasattr(processor, "apply_chat_template"):
        message_attempts: tuple[tuple[str, list[dict[str, Any]]], ...] = (
            (
                "chat_image_url_then_text",
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": image_path.resolve().as_uri()},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            ),
            (
                "chat_text_then_image_url",
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "url": image_path.resolve().as_uri()},
                        ],
                    }
                ],
            ),
            (
                "chat_image_path_then_text",
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            ),
            (
                "chat_text_then_image_path",
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": str(image_path)},
                        ],
                    }
                ],
            ),
        )
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
                if _input_get(inputs, "input_ids") is not None:
                    trace["winner"] = name
                    trace["attempts"].append({"name": name, "status": "pass"})
                    return runtime._to_device(inputs, device), trace
                trace["attempts"].append(
                    {"name": name, "status": "skip", "reason": "missing_input_ids"}
                )
            except Exception as exc:
                trace["attempts"].append(
                    {"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}
                )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
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
            trace["winner"] = "processor_text_with_qwen_vl_utils"
        except Exception as vision_exc:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                inputs = processor(text=[text], images=[rgb], padding=True, return_tensors="pt")
            trace["winner"] = "processor_text_with_pil_image"
            trace["vision_info_fallback_error"] = f"{type(vision_exc).__name__}: {vision_exc}"
        trace["attempts"].append({"name": trace["winner"], "status": "pass"})
        return runtime._to_device(inputs, device), trace
    except Exception as exc:
        trace["attempts"].append(
            {"name": "processor_fallback", "status": "fail", "error": f"{type(exc).__name__}: {exc}"}
        )
        raise


def _grid_from_inputs(inputs: Any) -> dict[str, Any]:
    grid: dict[str, Any] = {"available": False}
    if not _input_items(inputs):
        return grid
    for key in ("image_grid_thw", "video_grid_thw"):
        value = _input_get(inputs, key)
        if value is None or not hasattr(value, "detach"):
            continue
        data = value.detach().cpu().tolist()
        grid[key] = data
        triplets = _grid_triplets(data)
        if triplets:
            t, h, w = triplets[0]
            grid.update(
                {
                    "available": True,
                    "grid_key": key,
                    "grid_triplet_count": len(triplets),
                    "t": t,
                    "h": h,
                    "w": w,
                }
            )
    pixel_values = _input_get(inputs, "pixel_values")
    if hasattr(pixel_values, "shape"):
        grid["pixel_values_shape"] = [int(dim) for dim in pixel_values.shape]
    return grid


def _draw_grid_overlay(image_path: Path, out_path: Path, grid: dict[str, Any]) -> dict[str, Any]:
    if not grid.get("available"):
        return {"status": "skipped", "reason": "no_grid_metadata"}
    h = int(grid.get("h") or 0)
    w = int(grid.get("w") or 0)
    if h <= 0 or w <= 0:
        return {"status": "skipped", "reason": "invalid_grid"}
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    width, height = rgb.size
    step_x = max(1, math.ceil(w / 48))
    step_y = max(1, math.ceil(h / 48))
    for gx in range(0, w + 1, step_x):
        x = round(width * gx / w)
        draw.line([(x, 0), (x, height)], fill=(255, 220, 0), width=1)
    for gy in range(0, h + 1, step_y):
        y = round(height * gy / h)
        draw.line([(0, y), (width, y)], fill=(255, 220, 0), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path, quality=90)
    return {
        "status": "written",
        "path": str(out_path),
        "grid_h": h,
        "grid_w": w,
        "draw_step_h": step_y,
        "draw_step_w": step_x,
    }


def _cuda_snapshot(runtime: Any) -> dict[str, Any]:
    try:
        snapshot = runtime._assert_single_visible_gpu()
        snapshot["status"] = "pass"
        return snapshot
    except Exception as exc:
        return {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def _run_sample_gate(
    *,
    runtime: Any,
    model: Any,
    model_id: str,
    processor: Any,
    device: Any,
    sample: dict[str, Any],
    out_dir: Path,
    hook_names: list[str],
    hook_role_map: dict[str, str],
) -> dict[str, Any]:
    sample_key = str(sample["sample_key"])
    sample_dir = out_dir / "samples" / _safe_id(sample_key)
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(str(sample["image_path"]))
    prompt = str(sample.get("prompt") or "")
    row: dict[str, Any] = {
        "sample_key": sample_key,
        "request_id": sample.get("request_id"),
        "arm": sample.get("arm"),
        "query": sample.get("query"),
        "image_path": str(image_path),
        "prompt_length": len(prompt),
        "sheet_builder": sample.get("sheet_builder"),
    }
    start = time.perf_counter()
    try:
        row["prompt_check"] = {
            "status": "pass" if len(prompt.strip()) >= MIN_PROMPT_CHARS else "fail",
            "min_chars": MIN_PROMPT_CHARS,
        }
        if row["prompt_check"]["status"] != "pass":
            raise RuntimeError(
                f"method-facing prompt too short for gate: {len(prompt.strip())} chars"
            )
        prepare_start = time.perf_counter()
        inputs, prepare_trace = _prepare_inputs_with_trace(
            runtime=runtime,
            processor=processor,
            prompt=prompt,
            image_path=image_path,
            device=device,
        )
        row["input_prepare_elapsed_s"] = time.perf_counter() - prepare_start
        row["input_preparation"] = "pass"
        row["input_preparation_trace"] = prepare_trace
        row["input_summary"] = _inputs_summary(inputs)
        grid = _grid_from_inputs(inputs)
        row["visual_grid"] = grid
        row["visual_token_check"] = _visual_token_check(
            inputs=inputs,
            grid=grid,
            model=model,
            processor=processor,
            model_id=model_id,
        )
        row["grid_overlay"] = _draw_grid_overlay(
            image_path=image_path,
            out_path=sample_dir / "processor_grid_overlay.jpg",
            grid=grid,
        )

        _reset_cuda_peak_memory()
        vanilla, vanilla_elapsed_s = _timed_forward(model, inputs)
        vanilla_logits = _last_logits_slice(_extract_logits(vanilla))
        row["vanilla_forward_elapsed_s"] = vanilla_elapsed_s
        row["vanilla_logits"] = _tensor_payload(vanilla_logits) if vanilla_logits is not None else None
        row["vanilla_logits_sanity"] = _logits_sanity(vanilla_logits)
        row["vanilla_cuda_peak"] = _cuda_peak_memory()

        row["hidden_state_probe"] = _hidden_state_probe(model, inputs)

        handles, hook_records = _register_summary_hooks(model, hook_names)
        try:
            _reset_cuda_peak_memory()
            hooked, hooked_elapsed_s = _timed_forward(model, inputs)
        finally:
            for handle in handles:
                handle.remove()
        hooked_logits = _last_logits_slice(_extract_logits(hooked))
        row["hooked_forward_elapsed_s"] = hooked_elapsed_s
        row["hooked_modules_requested"] = hook_names
        row["hook_role_map"] = hook_role_map
        row["hooked_modules_fired"] = sorted(hook_records)
        row["hooked_modules_missing"] = sorted(set(hook_names) - set(hook_records))
        row["hooked_roles_fired"] = sorted(
            {hook_role_map.get(name, "unknown") for name in hook_records}
        )
        row["hook_records"] = hook_records
        row["hook_parity"] = _compare_logits(vanilla_logits, hooked_logits)
        row["hooked_logits_sanity"] = _logits_sanity(hooked_logits)
        row["hooked_cuda_peak"] = _cuda_peak_memory()
        required_roles = {"vision", "language_block"}
        if "projector" in set(hook_role_map.values()):
            required_roles.add("projector")
        fired_roles = set(row["hooked_roles_fired"])
        row["hook_role_coverage"] = {
            "required_roles": sorted(required_roles),
            "fired_roles": sorted(fired_roles),
            "projector_selected": "projector" in set(hook_role_map.values()),
            "status": "pass" if required_roles.issubset(fired_roles) else "fail",
        }
        row["hook_capture_status"] = (
            "pass"
            if hook_records
            and not row["hooked_modules_missing"]
            and row["hook_role_coverage"]["status"] == "pass"
            else "fail"
        )
        if sample_key == "single_frame":
            row["attention_probe"] = _attention_probe(model, inputs)
        else:
            row["attention_probe"] = {
                "status": "skipped",
                "reason": "contact_sheet_attention_probe_omitted_to_bound_memory",
            }
    except Exception as exc:
        row["input_or_forward_error"] = f"{type(exc).__name__}: {exc}"
        row.setdefault("input_preparation", "fail")
        row.setdefault("hook_capture_status", "fail")
        row.setdefault("hook_parity", {"status": "fail", "reason": "exception"})
        row.setdefault("visual_token_check", {"status": "fail", "reason": "exception"})
        row.setdefault("hidden_state_probe", {"status": "fail", "reason": "exception"})
        row.setdefault("vanilla_logits_sanity", {"status": "fail", "reason": "exception"})
        row.setdefault("prompt_check", {"status": "fail", "reason": "exception"})
    row["elapsed_s"] = time.perf_counter() - start
    _write_json(sample_dir / "sample_gate.json", row)
    return row


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    avg = _mean(values)
    assert avg is not None
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _input_sequence_length(inputs: Any) -> int | None:
    value = _input_get(inputs, "input_ids")
    if hasattr(value, "shape"):
        if len(value.shape) >= 2:
            return int(value.shape[-1])
        if len(value.shape) == 1:
            return int(value.shape[0])
    return None


def _ordered_probe_samples(samples: list[dict[str, Any]], sample_key: str) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for sample in samples:
        if sample.get("sample_key") == sample_key:
            ordered.append(sample)
    for sample in samples:
        if sample not in ordered:
            ordered.append(sample)
    return ordered


def _run_one_utilization_probe(
    *,
    runtime: Any,
    model: Any,
    model_id: str,
    processor: Any,
    device: Any,
    sample: dict[str, Any],
    warmup: int,
    repeat: int,
    min_seconds: float,
    hidden_repeat: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_id": model_id,
        "sample_key": sample.get("sample_key"),
        "arm": sample.get("arm"),
        "request_id": sample.get("request_id"),
        "warmup_target": int(warmup),
        "repeat_target": int(repeat),
        "min_seconds_target": float(min_seconds),
        "hidden_repeat_target": int(hidden_repeat),
        "started_at": _utc_now_iso(),
    }
    try:
        import torch

        image_path = Path(str(sample["image_path"]))
        prompt = str(sample.get("prompt") or "")
        if len(prompt.strip()) < MIN_PROMPT_CHARS:
            raise RuntimeError(f"method-facing prompt too short for probe: {len(prompt.strip())} chars")
        prepare_start = time.perf_counter()
        inputs, prepare_trace = _prepare_inputs_with_trace(
            runtime=runtime,
            processor=processor,
            prompt=prompt,
            image_path=image_path,
            device=device,
        )
        payload["input_prepare_elapsed_s"] = time.perf_counter() - prepare_start
        payload["input_preparation_trace"] = prepare_trace
        grid = _grid_from_inputs(inputs)
        payload["visual_grid"] = grid
        visual_token_check = _visual_token_check(
            inputs=inputs,
            grid=grid,
            model=model,
            processor=processor,
            model_id=model_id,
        )
        payload["visual_token_check"] = visual_token_check
        payload["seq_len"] = _input_sequence_length(inputs)

        _reset_cuda_peak_memory()
        warmup_latencies: list[float] = []
        for _ in range(max(0, int(warmup))):
            output, elapsed_s = _timed_forward(model, inputs)
            warmup_latencies.append(elapsed_s)
            del output

        latencies: list[float] = []
        loop_start = time.perf_counter()
        target_repeat = max(1, int(repeat))
        target_seconds = max(0.0, float(min_seconds))
        while len(latencies) < target_repeat or (time.perf_counter() - loop_start) < target_seconds:
            output, elapsed_s = _timed_forward(model, inputs)
            latencies.append(elapsed_s)
            del output
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        loop_elapsed_s = time.perf_counter() - loop_start
        vanilla_memory = _cuda_peak_memory()

        _reset_cuda_peak_memory()
        hidden_latencies: list[float] = []
        for _ in range(max(1, int(hidden_repeat))):
            output, elapsed_s = _timed_forward(
                model,
                inputs,
                extra_kwargs={"output_hidden_states": True},
            )
            hidden_latencies.append(elapsed_s)
            del output
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        hidden_memory = _cuda_peak_memory()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        mean_latency = _mean(latencies)
        hidden_mean = _mean(hidden_latencies)
        payload.update(
            {
                "status": "pass" if visual_token_check.get("status") == "pass" else "fail",
                "warmup_latencies_s": warmup_latencies,
                "forward_latencies_s": latencies,
                "repeat_count": len(latencies),
                "loop_elapsed_s": loop_elapsed_s,
                "forward_mean_s": mean_latency,
                "forward_std_s": _std(latencies),
                "forward_min_s": min(latencies) if latencies else None,
                "forward_max_s": max(latencies) if latencies else None,
                "forwards_per_second": (len(latencies) / loop_elapsed_s) if loop_elapsed_s else None,
                "tokens_per_second_approx": (
                    (float(payload["seq_len"]) * len(latencies) / loop_elapsed_s)
                    if loop_elapsed_s and payload.get("seq_len")
                    else None
                ),
                "cuda_peak": vanilla_memory,
                "hidden_forward_latencies_s": hidden_latencies,
                "hidden_forward_mean_s": hidden_mean,
                "hidden_forward_std_s": _std(hidden_latencies),
                "hidden_cuda_peak": hidden_memory,
                "h100_memory_headroom_fraction": (
                    1.0 - float(hidden_memory.get("reserved_fraction"))
                    if isinstance(hidden_memory.get("reserved_fraction"), (int, float))
                    else (
                        1.0 - float(vanilla_memory.get("reserved_fraction"))
                        if isinstance(vanilla_memory.get("reserved_fraction"), (int, float))
                        else None
                    )
                ),
            }
        )
        if payload["status"] != "pass":
            payload["reason"] = "visual_token_check_failed"
    except Exception as exc:
        payload["status"] = "fail"
        payload["error"] = f"{type(exc).__name__}: {exc}"
    payload["ended_at"] = _utc_now_iso()
    return payload


def _run_utilization_probe(
    *,
    runtime: Any,
    model: Any,
    model_id: str,
    processor: Any,
    device: Any,
    samples: list[dict[str, Any]],
    out_dir: Path,
    sample_key: str,
    warmup: int,
    repeat: int,
    min_seconds: float,
) -> dict[str, Any]:
    probe_dir = out_dir / "utilization_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    if not samples:
        payload = {"status": "fail", "reason": "no_samples_available"}
        _write_json(probe_dir / "utilization_probe.json", payload)
        return payload
    payload: dict[str, Any] = {
        "kind": "phase2_mech_job1_utilization_probe",
        "model_id": model_id,
        "sample_key_requested": sample_key,
        "warmup_target": int(warmup),
        "repeat_target": int(repeat),
        "min_seconds_target": float(min_seconds),
        "min_seconds_applies_per_arm": True,
        "hidden_repeat_target": DEFAULT_PROBE_HIDDEN_REPEAT,
        "started_at": _utc_now_iso(),
    }
    probes = [
        _run_one_utilization_probe(
            runtime=runtime,
            model=model,
            model_id=model_id,
            processor=processor,
            device=device,
            sample=sample,
            warmup=warmup,
            repeat=repeat,
            min_seconds=min_seconds,
            hidden_repeat=DEFAULT_PROBE_HIDDEN_REPEAT,
        )
        for sample in _ordered_probe_samples(samples, sample_key)
    ]
    primary = next((row for row in probes if row.get("sample_key") == sample_key), probes[0])
    payload.update(
        {
            "status": "pass" if all(row.get("status") == "pass" for row in probes) else "fail",
            "probe_count": len(probes),
            "sample_keys": [row.get("sample_key") for row in probes],
            "probes": probes,
            "primary_sample_key": primary.get("sample_key"),
            "forward_mean_s": primary.get("forward_mean_s"),
            "hidden_forward_mean_s": primary.get("hidden_forward_mean_s"),
            "loop_elapsed_s": primary.get("loop_elapsed_s"),
            "repeat_count": primary.get("repeat_count"),
            "ended_at": _utc_now_iso(),
        }
    )
    _write_json(probe_dir / "utilization_probe.json", payload)
    return payload


def _estimate_runtime(summary: dict[str, Any]) -> dict[str, Any]:
    samples = [row for row in summary.get("samples", []) if isinstance(row, dict)]
    sample_count = len(samples)
    if not sample_count:
        try:
            sample_count = int(summary.get("sample_count") or 0)
        except Exception:
            sample_count = 0
    sample_elapsed = sum(float(row.get("elapsed_s") or 0.0) for row in samples)
    input_prepare = sum(float(row.get("input_prepare_elapsed_s") or 0.0) for row in samples)
    forward_elapsed = 0.0
    single_frame_attention_count = 0
    for row in samples:
        forward_elapsed += float(row.get("vanilla_forward_elapsed_s") or 0.0)
        forward_elapsed += float(row.get("hidden_state_probe", {}).get("forward_elapsed_s") or 0.0)
        forward_elapsed += float(row.get("hooked_forward_elapsed_s") or 0.0)
        attn = row.get("attention_probe", {})
        if isinstance(attn, dict):
            forward_elapsed += float(attn.get("forward_elapsed_s") or 0.0)
        if row.get("sample_key") == "single_frame":
            single_frame_attention_count += 1
    if not single_frame_attention_count and "single_frame" in set(summary.get("available_sample_keys", [])):
        single_frame_attention_count = 1
    load_elapsed = float(summary.get("load_elapsed_s") or 0.0)
    observed_rank_elapsed = load_elapsed + sample_elapsed
    probe = summary.get("utilization_probe", {})
    probe_mean = probe.get("forward_mean_s") if isinstance(probe, dict) else None
    hidden_probe_mean = probe.get("hidden_forward_mean_s") if isinstance(probe, dict) else None
    probe_by_sample_key = {}
    if isinstance(probe, dict):
        for row in probe.get("probes", []):
            if isinstance(row, dict) and row.get("sample_key"):
                probe_by_sample_key[str(row["sample_key"])] = row
    if input_prepare <= 0 and isinstance(probe, dict):
        input_prepare = float(probe.get("input_prepare_elapsed_s") or 0.0) * max(1, sample_count)
    probe_based_main_gate_estimate_s = None
    probe_based_capture_forward_estimate_s = None
    if isinstance(probe_mean, (int, float)) and sample_count:
        # Current main gate runs vanilla + hidden + hooked for both samples,
        # plus attention on the single-frame sample.
        expected_forward_count = 3 * sample_count + single_frame_attention_count
        probe_based_main_gate_estimate_s = load_elapsed + input_prepare + expected_forward_count * float(probe_mean)
        capture_forward_total = 0.0
        for row in samples or [{"sample_key": key} for key in summary.get("available_sample_keys", [])]:
            sample_key = str(row.get("sample_key") or "")
            sample_probe = probe_by_sample_key.get(sample_key, {})
            sample_vanilla = sample_probe.get("forward_mean_s", probe_mean)
            sample_hidden = sample_probe.get("hidden_forward_mean_s", hidden_probe_mean or sample_vanilla)
            if isinstance(sample_vanilla, (int, float)) and isinstance(sample_hidden, (int, float)):
                # Vanilla + hooked + hidden-state forward, with attention only
                # for the single-frame diagnostic path.
                capture_forward_total += 2.0 * float(sample_vanilla) + float(sample_hidden)
                if sample_key == "single_frame":
                    capture_forward_total += float(sample_vanilla)
        if capture_forward_total:
            probe_based_capture_forward_estimate_s = load_elapsed + input_prepare + capture_forward_total
    return {
        "load_elapsed_s": load_elapsed,
        "sample_elapsed_s": sample_elapsed,
        "sample_count": sample_count,
        "input_prepare_elapsed_s": input_prepare,
        "forward_elapsed_s": forward_elapsed,
        "observed_rank_elapsed_s": observed_rank_elapsed,
        "probe_based_main_gate_estimate_s": probe_based_main_gate_estimate_s,
        "probe_based_capture_forward_estimate_s": probe_based_capture_forward_estimate_s,
        "single_frame_attention_count": single_frame_attention_count,
        "formula": (
            "rank ~= load + input_prepare + "
            "sum_per_sample(2 * probe_forward_mean + probe_hidden_forward_mean) "
            "+ single_frame_attention_count * probe_forward_mean"
        ),
        "claim_boundary": "Estimate is for this small instrumentation gate, not the full downstream capture job.",
    }


def _write_barrier_marker(
    *,
    barrier_dir: Path,
    rank: int,
    world_size: int,
    expected_count: int,
    model_key: str,
    stage: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    barrier_dir.mkdir(parents=True, exist_ok=True)
    marker = barrier_dir / f"rank_{rank:03d}_{_safe_id(model_key)}.json"
    payload = {
        "rank": rank,
        "world_size": world_size,
        "expected_count": expected_count,
        "model_key": model_key,
        "stage": stage,
        "status": status,
        "ready_at": _utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    _write_json(marker, payload)
    return marker


def _file_barrier(
    *,
    barrier_dir: Path,
    rank: int,
    world_size: int,
    expected_count: int,
    model_key: str,
    stage: str,
    marker_status: str,
    timeout_s: float,
    marker_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _write_barrier_marker(
        barrier_dir=barrier_dir,
        rank=rank,
        world_size=world_size,
        expected_count=expected_count,
        model_key=model_key,
        stage=stage,
        status=marker_status,
        extra=marker_extra,
    )
    start = time.perf_counter()
    seen: list[str] = []
    markers: list[dict[str, Any]] = []
    while True:
        marker_paths = sorted(barrier_dir.glob("rank_*.json"))
        seen = [path.name for path in marker_paths]
        if len(seen) >= expected_count:
            status = "pass"
            break
        if (time.perf_counter() - start) >= timeout_s:
            status = "timeout"
            break
        time.sleep(5.0)
    for path in sorted(barrier_dir.glob("rank_*.json")):
        try:
            loaded = _read_json(path)
            if isinstance(loaded, dict):
                markers.append(loaded)
        except Exception:
            markers.append({"path": str(path), "status": "unreadable"})
    failed_markers = [
        marker for marker in markers
        if str(marker.get("status") or "").lower() not in {"ready", "pass"}
    ]
    elapsed = time.perf_counter() - start
    return {
        "status": status,
        "stage": stage,
        "barrier_dir": str(barrier_dir),
        "rank": rank,
        "world_size": world_size,
        "expected_count": expected_count,
        "ready_count": len(seen),
        "ready_markers": seen,
        "failed_markers": failed_markers,
        "all_markers_ready": not failed_markers,
        "wait_elapsed_s": elapsed,
        "timeout_s": timeout_s,
    }


def run_model(args: argparse.Namespace) -> int:
    repo_dir = args.repo_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    w0_dir = args.w0_dir.expanduser().resolve()
    selected = _select_model_for_rank(args)
    model_key = selected["model_key"]
    model_id = selected["model_id"]
    revision = selected.get("revision", "main")
    out_dir = run_dir / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    if not model_id:
        _write_json(out_dir / "model_gate_summary.json", {"model_key": model_key, "status": "idle"})
        return 0

    runtime = _load_job0_runtime(repo_dir)
    summary: dict[str, Any] = {
        "kind": "phase2_mech_job1_model_gate_summary",
        "model_key": model_key,
        "model_id": model_id,
        "revision": revision,
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "dtype": args.dtype,
        "repo_dir": str(repo_dir),
        "w0_dir": str(w0_dir),
        "claim_boundary": (
            "Instrumentation gate over W0 method-facing VISOR derived images. "
            "This run records module/input/visual-token/hidden-state/hook "
            "diagnostics only; it does not score model behavior or expose hidden "
            "answers to prompts."
        ),
        "cuda": _cuda_snapshot(runtime),
        "runtime_versions": _runtime_versions(),
        "model_snapshot": _snapshot_provenance(
            model_id,
            revision,
            local_files_only=bool(args.local_files_only),
        ),
    }
    _write_json(out_dir / "model_gate_summary.json", summary)

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
        summary["config"] = _summarize_config(model)
        summary["processor"] = _summarize_processor(processor)
        summary["model_placement"] = _model_placement_check(model, dtype_name=args.dtype)
        if args.post_load_barrier_dir:
            plan_count = len(_parse_model_plan(args))
            expected_count = min(int(args.world_size), plan_count)
            summary["post_load_barrier"] = _file_barrier(
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
        plan_count = len(_parse_model_plan(args))
        expected_count = min(int(args.world_size), plan_count)
        marker_extra = {"load_status": "fail", "load_error": summary["load_error"]}
        if args.post_load_barrier_dir:
            _write_barrier_marker(
                barrier_dir=args.post_load_barrier_dir.expanduser().resolve(),
                rank=int(args.rank),
                world_size=int(args.world_size),
                expected_count=expected_count,
                model_key=model_key,
                stage="post_load",
                status="fail",
                extra=marker_extra,
            )
        if args.pre_probe_barrier_dir:
            _write_barrier_marker(
                barrier_dir=args.pre_probe_barrier_dir.expanduser().resolve(),
                rank=int(args.rank),
                world_size=int(args.world_size),
                expected_count=expected_count,
                model_key=model_key,
                stage="pre_probe",
                status="fail",
                extra=marker_extra,
            )
        _write_json(out_dir / "model_gate_summary.json", summary)
        return 1 if args.strict else 0

    module_map = _module_map(model, max_modules=int(args.max_modules))
    _write_json(out_dir / "module_map.json", module_map)
    hook_names, hook_role_map = _select_hook_modules(module_map["modules"])
    summary["hook_module_names"] = hook_names
    summary["hook_role_map"] = hook_role_map
    summary["module_map_path"] = "module_map.json"

    samples = _prepare_samples(w0_dir=w0_dir, out_dir=out_dir / "input_samples", sample_index=int(args.sample_index))
    if args.utilization_probe and args.probe_only:
        probe = _run_utilization_probe(
            runtime=runtime,
            model=model,
            model_id=model_id,
            processor=processor,
            device=device,
            samples=samples,
            out_dir=out_dir,
            sample_key=args.probe_sample_key,
            warmup=int(args.probe_warmup),
            repeat=int(args.probe_repeat),
            min_seconds=float(args.probe_min_seconds),
        )
        summary["samples"] = []
        summary["sample_count"] = len(samples)
        summary["available_sample_keys"] = [str(sample.get("sample_key")) for sample in samples]
        summary["utilization_probe"] = probe
        summary["pass_fail"] = {
            "model_loaded": summary.get("load_status") == "pass",
            "cuda_visible_single_gpu": summary.get("cuda", {}).get("status") == "pass",
            "model_on_cuda_bf16": summary.get("model_placement", {}).get("status") == "pass",
            "module_map_written": bool(module_map.get("modules")),
            "post_load_barrier_passed": summary.get("post_load_barrier", {}).get("status", "pass") == "pass",
            "utilization_probe_passed": probe.get("status") == "pass",
        }
        summary["runtime_estimate"] = _estimate_runtime(summary)
        summary["overall_status"] = "pass" if all(summary["pass_fail"].values()) else "fail"
        _write_json(out_dir / "model_gate_summary.json", summary)
        return 0 if summary["overall_status"] == "pass" or not args.strict else 1

    sample_rows = [
        _run_sample_gate(
            runtime=runtime,
            model=model,
            model_id=model_id,
            processor=processor,
            device=device,
            sample=sample,
            out_dir=out_dir,
            hook_names=hook_names,
            hook_role_map=hook_role_map,
        )
        for sample in samples
    ]
    summary["samples"] = sample_rows
    summary["sample_count"] = len(sample_rows)
    if args.utilization_probe:
        if args.pre_probe_barrier_dir:
            plan_count = len(_parse_model_plan(args))
            expected_count = min(int(args.world_size), plan_count)
            summary["pre_probe_barrier"] = _file_barrier(
                barrier_dir=args.pre_probe_barrier_dir.expanduser().resolve(),
                rank=int(args.rank),
                world_size=int(args.world_size),
                expected_count=expected_count,
                model_key=model_key,
                stage="pre_probe",
                marker_status="ready",
                timeout_s=float(args.post_load_barrier_timeout_s),
                marker_extra={"sample_gate_status": "complete"},
            )
        summary["utilization_probe"] = _run_utilization_probe(
            runtime=runtime,
            model=model,
            model_id=model_id,
            processor=processor,
            device=device,
            samples=samples,
            out_dir=out_dir,
            sample_key=args.probe_sample_key,
            warmup=int(args.probe_warmup),
            repeat=int(args.probe_repeat),
            min_seconds=float(args.probe_min_seconds),
        )
    summary["pass_fail"] = {
        "model_loaded": summary.get("load_status") == "pass",
        "cuda_visible_single_gpu": summary.get("cuda", {}).get("status") == "pass",
        "model_on_cuda_bf16": summary.get("model_placement", {}).get("status") == "pass",
        "post_load_barrier_passed": summary.get("post_load_barrier", {}).get("status", "pass") == "pass",
        "module_map_written": bool(module_map.get("modules")),
        "hook_modules_selected": bool(hook_names),
        "has_samples": bool(sample_rows),
        "all_prompts_valid": all(
            row.get("prompt_check", {}).get("status") == "pass" for row in sample_rows
        ),
        "all_inputs_prepared": all(row.get("input_preparation") == "pass" for row in sample_rows),
        "all_visual_tokens_present": all(
            row.get("visual_token_check", {}).get("status") == "pass" for row in sample_rows
        ),
        "all_hidden_states_available": all(
            row.get("hidden_state_probe", {}).get("status") == "pass" for row in sample_rows
        ),
        "all_logits_sane": all(
            row.get("vanilla_logits_sanity", {}).get("status") == "pass"
            and row.get("hooked_logits_sanity", {}).get("status") == "pass"
            for row in sample_rows
        ),
        "all_hooks_fired": all(row.get("hook_capture_status") == "pass" for row in sample_rows),
        "all_required_hook_roles_fired": all(
            row.get("hook_role_coverage", {}).get("status") == "pass" for row in sample_rows
        ),
        "all_hook_parity": all(row.get("hook_parity", {}).get("status") == "pass" for row in sample_rows),
        "any_visual_grid_metadata": any(row.get("visual_grid", {}).get("available") for row in sample_rows),
    }
    if args.utilization_probe:
        summary["pass_fail"]["utilization_probe_passed"] = (
            summary.get("utilization_probe", {}).get("status") == "pass"
        )
    required = (
        "model_loaded",
        "cuda_visible_single_gpu",
        "model_on_cuda_bf16",
        "post_load_barrier_passed",
        "module_map_written",
        "hook_modules_selected",
        "has_samples",
        "all_prompts_valid",
        "all_inputs_prepared",
        "all_visual_tokens_present",
        "all_hidden_states_available",
        "all_logits_sane",
        "all_hooks_fired",
        "all_required_hook_roles_fired",
        "all_hook_parity",
    )
    if args.utilization_probe:
        required = (*required, "utilization_probe_passed")
    summary["overall_status"] = (
        "pass" if all(summary["pass_fail"].get(key) for key in required) else "fail"
    )
    summary["runtime_estimate"] = _estimate_runtime(summary)
    _write_json(out_dir / "model_gate_summary.json", summary)
    return 0 if summary["overall_status"] == "pass" or not args.strict else 1


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Mechanism Job1 Instrument Gates",
        "",
        f"Status: `{payload['overall_status']}`",
        f"Run dir: `{payload['run_dir']}`",
        "",
        "| Model | Status | CUDA/bf16 | Inputs | Image tokens | Hidden states | Hooks | Parity | Probe | Rank est. min |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for row in payload.get("models", []):
        pf = row.get("pass_fail", {}) if isinstance(row.get("pass_fail"), dict) else {}
        estimate = row.get("runtime_estimate", {}) if isinstance(row.get("runtime_estimate"), dict) else {}
        rank_est = estimate.get("probe_based_main_gate_estimate_s") or estimate.get("observed_rank_elapsed_s")
        rank_est_min = f"{float(rank_est) / 60:.1f}" if isinstance(rank_est, (int, float)) else ""
        probe = row.get("utilization_probe", {}) if isinstance(row.get("utilization_probe"), dict) else {}
        lines.append(
            "| {model_key} | {status} | {cuda} | {inputs} | {tokens} | {hidden} | {hooks} | {parity} | {probe} | {rank_est_min} |".format(
                model_key=row.get("model_key"),
                status=row.get("overall_status", row.get("status")),
                cuda=pf.get("cuda_visible_single_gpu") and pf.get("model_on_cuda_bf16"),
                inputs=pf.get("all_inputs_prepared"),
                tokens=pf.get("all_visual_tokens_present"),
                hidden=pf.get("all_hidden_states_available"),
                hooks=pf.get("all_hooks_fired"),
                parity=pf.get("all_hook_parity"),
                probe=probe.get("status", ""),
                rank_est_min=rank_est_min,
            )
            )
    gpu_rows = payload.get("gpu_monitor_summary", [])
    if gpu_rows:
        lines.extend(
            [
                "",
                "## GPU Monitor Summary",
                "",
                "| Rank | Model | Rows | Median util % | P95 util % | Median power W | Max memory MiB |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in gpu_rows:
            lines.append(
                "| {rank} | {model_key} | {rows} | {median_util} | {p95_util} | {median_power} | {max_mem} |".format(
                    rank=row.get("rank"),
                    model_key=row.get("model_key"),
                    rows=row.get("rows_in_window"),
                    median_util=_fmt_optional_float(row.get("median_utilization_gpu_pct")),
                    p95_util=_fmt_optional_float(row.get("p95_utilization_gpu_pct")),
                    median_power=_fmt_optional_float(row.get("median_power_draw_w")),
                    max_mem=_fmt_optional_float(row.get("max_memory_used_mib")),
                )
            )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            payload["claim_boundary"],
            "",
            "This gate only proves that the selected runtime/model pairs can be loaded",
            "on one H100 GPU each, prepared from W0 method-facing image inputs, shown",
            "to contain visual tokens, probed for hidden-state shape availability,",
            "hooked without logit drift, and summarized compactly. Full activation",
            "capture, probes, and causal interventions remain downstream work.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt_optional_float(value: Any) -> str:
    return f"{float(value):.1f}" if isinstance(value, (int, float)) else ""


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _probe_window(row: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    starts: list[datetime] = []
    ends: list[datetime] = []
    probe = row.get("utilization_probe", {})
    if isinstance(probe, dict):
        for item in probe.get("probes", []):
            if not isinstance(item, dict):
                continue
            start = _parse_iso_datetime(item.get("started_at"))
            end = _parse_iso_datetime(item.get("ended_at"))
            if start is not None:
                starts.append(start)
            if end is not None:
                ends.append(end)
        start = _parse_iso_datetime(probe.get("started_at"))
        end = _parse_iso_datetime(probe.get("ended_at"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    return (min(starts) if starts else None, max(ends) if ends else None)


def _float_field(row: dict[str, str], key: str) -> float | None:
    try:
        value = row.get(key)
        if value is None or str(value).strip() in {"", "[N/A]", "N/A"}:
            return None
        return float(str(value).strip())
    except Exception:
        return None


def _summarize_gpu_monitor_csv(
    *,
    path: Path,
    model_row: dict[str, Any],
) -> dict[str, Any]:
    start, end = _probe_window(model_row)
    rows_total = 0
    rows_in_window = 0
    util_values: list[float] = []
    power_values: list[float] = []
    memory_values: list[float] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            for csv_row in reader:
                rows_total += 1
                timestamp = _parse_iso_datetime(csv_row.get("timestamp"))
                in_window = True
                if start is not None and timestamp is not None and timestamp < start:
                    in_window = False
                if end is not None and timestamp is not None and timestamp > end:
                    in_window = False
                if not in_window:
                    continue
                rows_in_window += 1
                util = _float_field(csv_row, "utilization_gpu_pct")
                power = _float_field(csv_row, "power_draw_w")
                memory = _float_field(csv_row, "memory_used_mib")
                if util is not None:
                    util_values.append(util)
                if power is not None:
                    power_values.append(power)
                if memory is not None:
                    memory_values.append(memory)
    return {
        "rank": model_row.get("rank"),
        "model_key": model_row.get("model_key"),
        "source_path": str(path),
        "window_start": start.isoformat() if start is not None else None,
        "window_end": end.isoformat() if end is not None else None,
        "rows_total": rows_total,
        "rows_in_window": rows_in_window,
        "median_utilization_gpu_pct": _percentile(util_values, 0.50),
        "p95_utilization_gpu_pct": _percentile(util_values, 0.95),
        "median_power_draw_w": _percentile(power_values, 0.50),
        "max_memory_used_mib": max(memory_values) if memory_values else None,
        "status": "pass" if rows_in_window and util_values else "missing_or_empty",
    }


def _gpu_monitor_summary(run_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in rows:
        rank = row.get("rank")
        if not isinstance(rank, int):
            continue
        summaries.append(
            _summarize_gpu_monitor_csv(
                path=run_dir / f"gpu_monitor_rank_{rank}.csv",
                model_row=row,
            )
        )
    return summaries


def aggregate(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    plan = _parse_model_plan(args)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for model in plan:
        path = run_dir / model["model_key"] / "model_gate_summary.json"
        if not path.exists():
            missing.append(model["model_key"])
            continue
        rows.append(_read_json(path))
    payload = {
        "kind": "phase2_mech_job1_instrument_gates_summary",
        "run_dir": str(run_dir),
        "expected_models": plan,
        "missing_models": missing,
        "models": rows,
        "claim_boundary": (
            "Mechanism-study instrumentation gate over W0 method-facing VISOR "
            "derived images. This is a readiness check for hook/input geometry "
            "capture, not a behavioral or mechanistic result."
        ),
    }
    payload["gpu_monitor_summary"] = _gpu_monitor_summary(run_dir, rows)
    payload["overall_status"] = (
        "pass"
        if not missing and rows and all(row.get("overall_status") == "pass" for row in rows)
        else "fail"
    )
    _write_json(run_dir / "gate_summary.json", payload)
    (run_dir / "gate_summary.md").write_text(_markdown_summary(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["overall_status"] == "pass" or not args.strict else 1


def preflight_inputs(args: argparse.Namespace) -> int:
    w0_dir = args.w0_dir.expanduser().resolve()
    payload = _preflight_input_frames(w0_dir=w0_dir, sample_index=int(args.sample_index))
    if args.out:
        out = args.out.expanduser().resolve()
        _write_json(out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" or not args.strict else 1


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
    run.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--sample-index", type=int, default=0)
    run.add_argument("--max-modules", type=int, default=20000)
    run.add_argument("--utilization-probe", action="store_true")
    run.add_argument("--probe-only", action="store_true")
    run.add_argument("--probe-sample-key", default="single_frame")
    run.add_argument("--probe-warmup", type=int, default=DEFAULT_PROBE_WARMUP)
    run.add_argument("--probe-repeat", type=int, default=DEFAULT_PROBE_REPEAT)
    run.add_argument("--probe-min-seconds", type=float, default=DEFAULT_PROBE_MIN_SECONDS)
    run.add_argument("--post-load-barrier-dir", type=Path)
    run.add_argument("--pre-probe-barrier-dir", type=Path)
    run.add_argument("--post-load-barrier-timeout-s", type=float, default=DEFAULT_POST_LOAD_BARRIER_TIMEOUT)

    subparsers.add_parser("aggregate", parents=[common])

    preflight = subparsers.add_parser("preflight-inputs")
    preflight.add_argument("--w0-dir", type=Path, default=DEFAULT_W0_DIR)
    preflight.add_argument("--sample-index", type=int, default=0)
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
