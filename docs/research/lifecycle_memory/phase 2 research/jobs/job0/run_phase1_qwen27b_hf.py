#!/usr/bin/env python3
"""TamIA sharded Hugging Face runner for the Phase 1 EPIC/VISOR paper battery.

The worker side only reads method-facing request packs and RGB frames. Hidden
indices are read only by the merge/score step after predictions exist.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
from typing import Any

from PIL import Image

from research.lifecycle_memory.build_epic_visor_dynamic_smoke import score_visor_predictions
from research.lifecycle_memory.protocol_v3_3rscan_visual_vlm import build_contact_sheet_image
from research.lifecycle_memory.run_epic_visor_dynamic_llamacpp_metal import (
    _blank_image_same_size,
    _llamacpp_yes_no_prompt,
    _parse_visor_answer,
    _prompt_echo_detected,
    _safe_request_id,
    build_gate_report,
)
from research.lifecycle_memory.run_epic_visor_dynamic_vlm_smoke import (
    _read_json,
    _resolve_path,
    _show_frame_labels,
)


CONTROL_ARMS: dict[str, dict[str, Any]] = {
    "real": {
        "pack": "order_stripped_request_pack.json",
        "hidden": "order_stripped_hidden_index.json",
        "blank_image": False,
    },
    "blank": {
        "pack": "order_stripped_request_pack.json",
        "hidden": "order_stripped_hidden_index.json",
        "blank_image": True,
    },
    "shuffled": {
        "pack": "order_stripped_shuffled_request_pack.json",
        "hidden": "order_stripped_shuffled_hidden_index.json",
        "blank_image": False,
    },
    "wrong_window": {
        "pack": "order_stripped_wrong_window_request_pack.json",
        "hidden": "order_stripped_wrong_window_hidden_index.json",
        "blank_image": False,
    },
    "target_swap": {
        "pack": "order_stripped_target_swap_request_pack.json",
        "hidden": "order_stripped_target_swap_hidden_index.json",
        "blank_image": False,
    },
    "target_swap_blank": {
        "pack": "order_stripped_target_swap_request_pack.json",
        "hidden": "order_stripped_target_swap_hidden_index.json",
        "blank_image": True,
    },
}

DEFAULT_ARMS = tuple(CONTROL_ARMS)
DEFAULT_MODEL_KEY = "qwen35_27b_hf_bf16"
DEFAULT_MAX_TOKENS = 8
DEFAULT_STDOUT_CAP = 4096


def _apply_chat_template_direct(processor: Any, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    """Apply Qwen chat templates with thinking disabled when the runtime supports it."""
    attempts = (
        {**kwargs, "enable_thinking": False},
        {**kwargs, "chat_template_kwargs": {"enable_thinking": False}},
        kwargs,
    )
    last_type_error = None
    for attempt in attempts:
        try:
            return processor.apply_chat_template(messages, **attempt)
        except TypeError as exc:
            message = str(exc)
            if (
                "enable_thinking" in message
                or "chat_template_kwargs" in message
                or "unexpected keyword" in message
                or "not a valid argument" in message
            ):
                last_type_error = exc
                continue
            raise
    if last_type_error is not None:
        raise last_type_error
    raise RuntimeError("failed to apply chat template")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _dtype_from_name(torch: Any, name: str) -> Any:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def _cuda_snapshot() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    return {
        "cuda_available": True,
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def _assert_single_visible_gpu() -> dict[str, Any]:
    snapshot = _cuda_snapshot()
    if not snapshot["cuda_available"]:
        raise RuntimeError(
            "Expected exactly one visible CUDA GPU per Slurm worker, but CUDA is "
            "not available. Check Slurm GPU allocation, Apptainer --nv, and "
            f"CUDA_VISIBLE_DEVICES={snapshot.get('cuda_visible_devices')!r}."
        )
    if snapshot["cuda_device_count"] != 1:
        raise RuntimeError(
            "Expected exactly one visible CUDA GPU per Slurm worker, got "
            f"{snapshot['cuda_device_count']} with CUDA_VISIBLE_DEVICES="
            f"{snapshot.get('cuda_visible_devices')!r}"
        )
    return snapshot


def _model_type_for_id(
    *,
    model_id: str,
    trust_remote_code: bool,
    local_files_only: bool,
    revision: str | None = None,
) -> str:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return str(getattr(config, "model_type", "") or "")


def _patch_processor_mixin_optional_attributes() -> None:
    """Load remote processors that pass declared optional attrs to ProcessorMixin."""
    from transformers.processing_utils import ProcessorMixin

    if getattr(ProcessorMixin, "_mani_optional_attr_patch", False):
        return
    original_init = ProcessorMixin.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        optional = set(getattr(self, "optional_attributes", ()) or ())
        saved = {key: kwargs.pop(key) for key in list(kwargs) if key in optional}
        original_init(self, *args, **kwargs)
        for key, value in saved.items():
            setattr(self, key, value)

    ProcessorMixin.__init__ = patched_init  # type: ignore[method-assign]
    ProcessorMixin._mani_optional_attr_patch = True  # type: ignore[attr-defined]


def _load_model(
    *,
    model_id: str,
    dtype_name: str,
    trust_remote_code: bool,
    local_files_only: bool,
    revision: str | None = None,
    force_single_gpu_device_map: bool = False,
) -> tuple[Any, Any, Any, str, Any]:
    import torch
    from transformers import AutoProcessor

    import transformers

    dtype = _dtype_from_name(torch, dtype_name)
    model_type = _model_type_for_id(
        model_id=model_id,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        revision=revision,
    )
    if model_type == "internvl_chat":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            use_fast=False,
            fix_mistral_regex=True,
        )
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "revision": revision,
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
            "use_flash_attn": False,
        }
        try:
            model = AutoModel.from_pretrained(model_id, **load_kwargs)
        except TypeError:
            load_kwargs.pop("use_flash_attn", None)
            model = AutoModel.from_pretrained(model_id, **load_kwargs)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        device = next(model.parameters()).device
        return model, None, device, "internvl_chat", tokenizer

    if model_type == "molmo2":
        _patch_processor_mixin_optional_attributes()

    auto_class_names = (
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    )
    auto_classes = [
        (class_name, getattr(transformers, class_name))
        for class_name in auto_class_names
        if getattr(transformers, class_name, None) is not None
    ]
    if not auto_classes:  # pragma: no cover - environment failure path
        raise RuntimeError(
            "Transformers has no supported multimodal auto-model class. "
            "Install a current Transformers build with Qwen3.5 support."
        )

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    device_map: Any = "auto"
    if force_single_gpu_device_map and torch.cuda.is_available() and torch.cuda.device_count() == 1:
        device_map = {"": 0}
    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
        "device_map": device_map,
        "revision": revision,
    }
    if dtype is not torch.float32:
        load_kwargs["dtype"] = dtype
    load_errors = []
    model = None
    for class_name, AutoVLM in auto_classes:
        candidate_kwargs = dict(load_kwargs)
        try:
            model = AutoVLM.from_pretrained(model_id, **candidate_kwargs)
            break
        except TypeError:
            if "dtype" not in candidate_kwargs:
                raise
            candidate_kwargs["torch_dtype"] = candidate_kwargs.pop("dtype")
            try:
                model = AutoVLM.from_pretrained(model_id, **candidate_kwargs)
                break
            except Exception as exc:  # pragma: no cover - live model mismatch path
                load_errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
        except Exception as exc:  # pragma: no cover - live model mismatch path
            message = str(exc).lower()
            if "out of memory" in message or "cuda error" in message:
                raise
            load_errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
    if model is None:
        raise RuntimeError(
            "No generic Transformers multimodal auto class could load "
            f"{model_id}. Errors: {' | '.join(load_errors)}"
        )
    model.eval()
    device = getattr(model, "device", None) or next(model.parameters()).device
    return model, processor, device, "generic_generate", None


def _to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def _prepare_inputs(*, processor: Any, prompt: str, image_path: Path, device: Any) -> Any:
    """Prepare inputs for Qwen/Gemma/Molmo-style image-text processors."""
    if hasattr(processor, "apply_chat_template"):
        message_attempts = (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_path.resolve().as_uri()},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "url": image_path.resolve().as_uri()},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": str(image_path)},
                    ],
                }
            ],
        )
        for messages in message_attempts:
            try:
                inputs = _apply_chat_template_direct(
                    processor,
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                if isinstance(inputs, dict) and "input_ids" in inputs:
                    return _to_device(inputs, device)
            except Exception:
                continue

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        text = _apply_chat_template_direct(
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
    except Exception:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            inputs = processor(text=[text], images=[rgb], padding=True, return_tensors="pt")

    return _to_device(inputs, device)


def _internvl_dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    import math

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda item: item[0] * item[1])

    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for index in range(blocks):
        box = (
            (index % (target_width // image_size)) * image_size,
            (index // (target_width // image_size)) * image_size,
            ((index % (target_width // image_size)) + 1) * image_size,
            ((index // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _internvl_pixel_values(*, image_path: Path, dtype: Any, max_tiles: int) -> Any:
    import torch
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    with Image.open(image_path) as image:
        tiles = _internvl_dynamic_preprocess(
            image.convert("RGB"),
            image_size=448,
            max_num=max_tiles,
            use_thumbnail=True,
        )
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()
    return pixel_values.to(dtype)


def _decode_new_tokens(*, processor: Any, inputs: Any, output_ids: Any) -> str:
    input_ids = inputs.get("input_ids")
    if input_ids is not None and output_ids.shape[-1] >= input_ids.shape[-1]:
        generated = output_ids[:, input_ids.shape[-1] :]
    else:
        generated = output_ids
    if hasattr(processor, "batch_decode"):
        return str(processor.batch_decode(generated, skip_special_tokens=True)[0]).strip()
    return str(processor.decode(generated[0], skip_special_tokens=True)).strip()


def _run_one(
    *,
    model: Any,
    processor: Any,
    device: Any,
    adapter: str,
    tokenizer: Any,
    dtype_name: str,
    prompt: str,
    image_path: Path,
    max_tokens: int,
    internvl_max_tiles: int,
) -> tuple[str, float, dict[str, Any]]:
    import torch

    if adapter == "internvl_chat":
        dtype = _dtype_from_name(torch, dtype_name)
        pixel_values = _internvl_pixel_values(
            image_path=image_path,
            dtype=dtype,
            max_tiles=internvl_max_tiles,
        )
        generation_config = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
        }
        start = time.perf_counter()
        with torch.inference_mode():
            response = model.chat(
                tokenizer,
                pixel_values,
                f"<image>\n{prompt}",
                generation_config,
            )
        elapsed = time.perf_counter() - start
        metadata = {
            "adapter": adapter,
            "image_tiles": int(pixel_values.shape[0]),
            "input_tokens": None,
            "generated_tokens": None,
        }
        return str(response).strip(), elapsed, metadata

    inputs = _prepare_inputs(processor=processor, prompt=prompt, image_path=image_path, device=device)
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - start
    raw_text = _decode_new_tokens(processor=processor, inputs=inputs, output_ids=output_ids)
    metadata = {
        "adapter": adapter,
        "input_tokens": int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else None,
        "generated_tokens": int(output_ids.shape[-1] - inputs["input_ids"].shape[-1])
        if "input_ids" in inputs and output_ids.shape[-1] >= inputs["input_ids"].shape[-1]
        else None,
    }
    return raw_text, elapsed, metadata


def _iter_tasks(*, pack_dir: Path, arms: list[str]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for arm in arms:
        arm_spec = CONTROL_ARMS[arm]
        pack_path = pack_dir / str(arm_spec["pack"])
        request_pack = _read_json(pack_path)
        requests = [row for row in request_pack.get("requests", []) if isinstance(row, dict)]
        for request_index, request in enumerate(requests):
            tasks.append(
                {
                    "arm": arm,
                    "request_index": request_index,
                    "request": request,
                    "pack_path": pack_path,
                    "blank_image": bool(arm_spec["blank_image"]),
                }
            )
    return tasks


def run_shard(args: argparse.Namespace) -> int:
    pack_dir = args.pack_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    rank = int(args.rank)
    world_size = int(args.world_size)
    if rank < 0 or world_size < 1 or rank >= world_size:
        raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")

    arms = list(args.arms)
    shard_dir = run_dir / "shards" / f"rank_{rank:03d}"
    cuda_info = _assert_single_visible_gpu()
    _write_json(
        shard_dir / "manifest.json",
        {
            "kind": "phase1_qwen27b_hf_shard_manifest",
            "model_id": args.model_id,
            "model_key": args.model_key,
            "dtype": args.dtype,
            "pack_dir": str(pack_dir),
            "run_dir": str(run_dir),
            "rank": rank,
            "world_size": world_size,
            "arms": arms,
            "max_tokens": args.max_tokens,
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
            "hf_home": os.environ.get("HF_HOME"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_info": cuda_info,
            "slurm_localid": os.environ.get("SLURM_LOCALID"),
            "slurm_procid": os.environ.get("SLURM_PROCID"),
        },
    )

    model, processor, device, adapter, tokenizer = _load_model(
        model_id=args.model_id,
        dtype_name=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    manifest_path = shard_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    manifest["adapter"] = adapter
    manifest["device"] = str(device)
    manifest["internvl_max_tiles"] = int(args.internvl_max_tiles)
    _write_json(manifest_path, manifest)

    tasks = _iter_tasks(pack_dir=pack_dir, arms=arms)
    assigned = [task for index, task in enumerate(tasks) if index % world_size == rank]
    if args.max_tasks_per_rank is not None:
        assigned = assigned[: int(args.max_tasks_per_rank)]
    _write_json(
        shard_dir / "assignment.json",
        {
            "rank": rank,
            "world_size": world_size,
            "assigned_task_count": len(assigned),
            "total_task_count": len(tasks),
            "arms": arms,
        },
    )

    error_count = 0
    for completed, task in enumerate(assigned, start=1):
        arm = str(task["arm"])
        request = task["request"]
        assert isinstance(request, dict)
        request_pack_path = Path(task["pack_path"])
        request_pack = _read_json(request_pack_path)
        pack_presentation = (
            request_pack.get("presentation")
            if isinstance(request_pack.get("presentation"), dict)
            else {}
        )
        pack_root = request_pack_path.parent
        model_request_id = str(request.get("request_id", f"request_{task['request_index']:04d}"))
        frame_control = request.get("frame_control") if isinstance(request.get("frame_control"), dict) else {}
        request_id = str(frame_control.get("source_request_id") or model_request_id)
        safe_id = _safe_request_id(model_request_id)
        arm_dir = shard_dir / arm
        contact_sheet_dir = arm_dir / "contact_sheet_samples"
        raw_dir = arm_dir / "raw"
        contact_sheet_path = contact_sheet_dir / f"{safe_id}.jpg"
        frame_paths = [
            _resolve_path(frame.get("path"), project_root=pack_dir, pack_root=pack_root)
            for frame in request.get("frames", [])
            if isinstance(frame, dict) and frame.get("path")
        ]
        build_contact_sheet_image(
            request=request,
            frame_paths=frame_paths,
            out_path=contact_sheet_path,
            thumb_width=360,
            thumb_height=220,
            columns=2,
            show_labels=_show_frame_labels(request=request, pack_presentation=pack_presentation),
        )
        image_path = contact_sheet_path
        if bool(task["blank_image"]):
            blank_path = contact_sheet_dir / f"{safe_id}__blank.jpg"
            image_path = _blank_image_same_size(contact_sheet_path, blank_path)

        prompt = _llamacpp_yes_no_prompt(
            request,
            pack_presentation=pack_presentation,
            blank_image=bool(task["blank_image"]),
        )
        error = None
        raw_text = ""
        latency_s = 0.0
        generation_metadata: dict[str, Any] = {}
        try:
            raw_text, latency_s, generation_metadata = _run_one(
                model=model,
                processor=processor,
                device=device,
                adapter=adapter,
                tokenizer=tokenizer,
                dtype_name=args.dtype,
                prompt=prompt,
                image_path=image_path,
                max_tokens=int(args.max_tokens),
                internvl_max_tiles=int(args.internvl_max_tiles),
            )
            parsed, parse_error = _parse_visor_answer(raw_text)
            if parse_error:
                error = parse_error
        except Exception as exc:  # pragma: no cover - live model failure path
            parsed = {
                "answer": None,
                "abstain": True,
                "abstention_reason": "exception",
                "confidence": 0.0,
            }
            error = f"{type(exc).__name__}: {exc}"
        if error:
            error_count += 1

        raw_text_path = raw_dir / f"{_safe_request_id(request_id)}.txt"
        raw_text_path.parent.mkdir(parents=True, exist_ok=True)
        raw_text_path.write_text(raw_text, encoding="utf-8")
        row = {
            "request_id": request_id,
            "model_request_id": model_request_id,
            "query_id": request.get("query", {}).get("query_id"),
            "answer": parsed.get("answer"),
            "abstain": bool(parsed.get("abstain", False)),
            "abstention_reason": parsed.get("abstention_reason"),
            "confidence": parsed.get("confidence", 0.0),
            "raw_text": raw_text,
            "raw_text_path": str(raw_text_path),
            "stderr_text_path": None,
            "stderr_tail": "",
            "error": error,
            "prompt_echo": _prompt_echo_detected(prompt=prompt, raw_text=raw_text),
            "latency_s": latency_s,
            "returncode": 0 if error is None else 1,
            "command": None,
            "execution_mode": "tamia_hf_sharded",
            "generation_metadata": generation_metadata,
            "contact_sheet_path": str(contact_sheet_path),
            "image_path": str(image_path),
            "rank": rank,
            "world_size": world_size,
            "arm": arm,
        }
        _append_jsonl(arm_dir / "predictions.jsonl", row)
        _write_json(
            shard_dir / "progress.json",
            {
                "rank": rank,
                "world_size": world_size,
                "completed": completed,
                "assigned": len(assigned),
                "current_arm": arm,
                "current_request_id": request_id,
                "error_count": error_count,
            },
        )
    if error_count and not args.allow_generation_errors:
        raise RuntimeError(f"rank {rank} had {error_count} generation/parse errors")
    return 0


def _expected_request_order(*, pack_dir: Path, arm: str) -> list[str]:
    pack_path = pack_dir / str(CONTROL_ARMS[arm]["pack"])
    request_pack = _read_json(pack_path)
    order: list[str] = []
    for index, request in enumerate(request_pack.get("requests", [])):
        if not isinstance(request, dict):
            continue
        model_request_id = str(request.get("request_id", f"request_{index:04d}"))
        frame_control = request.get("frame_control") if isinstance(request.get("frame_control"), dict) else {}
        order.append(str(frame_control.get("source_request_id") or model_request_id))
    return order


def _prediction_payload(
    *,
    model_id: str,
    model_key: str,
    request_pack_path: Path,
    run_dir: Path,
    responses: list[dict[str, Any]],
    blank_image: bool,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "kind": "epic_visor_dynamic_smoke_predictions",
        "backend": "hf_transformers_contact_sheet_sharded",
        "execution_mode": "tamia_hf_sharded",
        "model": model_id,
        "model_key": model_key,
        "request_pack": str(request_pack_path),
        "run_dir": str(run_dir),
        "request_ids": None,
        "blank_image": blank_image,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "response_count": len(responses),
        "claim_boundary": (
            "TamIA Hugging Face sharded VISOR contact-sheet run. Prediction workers "
            "use only contact sheets and public query text; hidden answers are used "
            "only by the merge/score step after responses are written."
        ),
        "responses": responses,
    }


def _score_row(score: dict[str, Any], gate_report: dict[str, Any]) -> dict[str, Any]:
    total = int(score.get("total", 0) or 0)
    correct = int(score.get("correct", 0) or 0)
    return {
        "gate_status": gate_report.get("status"),
        "accuracy": score.get("accuracy"),
        "correct": correct,
        "total": total,
        "parse_failure_count": next(
            (
                int(check.get("failure_count", 0) or 0)
                for check in gate_report.get("checks", [])
                if isinstance(check, dict) and check.get("name") == "all_responses_parsed"
            ),
            None,
        ),
        "stdout_cap_failure_count": next(
            (
                int(check.get("failure_count", 0) or 0)
                for check in gate_report.get("checks", [])
                if isinstance(check, dict) and check.get("name") == "stdout_below_cap"
            ),
            None,
        ),
    }


def _write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# EPIC/VISOR TamIA HF Control Battery",
        "",
        f"Model: `{summary['model_id']}`",
        f"Pack dir: `{summary['pack_dir']}`",
        f"Run dir: `{summary['run_dir']}`",
        "",
        "| Model | Arm | Gate | Correct | Total | Accuracy | Parse failures |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("rows", []):
        acc = row.get("accuracy")
        acc_text = "" if acc is None else f"{float(acc):.3f}"
        lines.append(
            "| {model_key} | {arm} | {gate_status} | {correct} | {total} | {accuracy} | {parse_failure_count} |".format(
                model_key=row.get("model_key"),
                arm=row.get("arm"),
                gate_status=row.get("gate_status"),
                correct=row.get("correct"),
                total=row.get("total"),
                accuracy=acc_text,
                parse_failure_count=row.get("parse_failure_count"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_score(args: argparse.Namespace) -> int:
    pack_dir = args.pack_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    arms = list(args.arms)
    rows: list[dict[str, Any]] = []
    for arm in arms:
        arm_run_dir = run_dir / args.model_key / arm
        expected_order = _expected_request_order(pack_dir=pack_dir, arm=arm)
        by_request_id: dict[str, dict[str, Any]] = {}
        for shard_file in sorted((run_dir / "shards").glob(f"rank_*/{arm}/predictions.jsonl")):
            for row in _read_jsonl(shard_file):
                by_request_id[str(row["request_id"])] = row
        missing = [request_id for request_id in expected_order if request_id not in by_request_id]
        if missing and not args.allow_incomplete:
            raise RuntimeError(f"{arm}: missing {len(missing)} predictions, first={missing[:5]}")
        responses = [by_request_id[request_id] for request_id in expected_order if request_id in by_request_id]
        payload = _prediction_payload(
            model_id=args.model_id,
            model_key=args.model_key,
            request_pack_path=pack_dir / str(CONTROL_ARMS[arm]["pack"]),
            run_dir=arm_run_dir,
            responses=responses,
            blank_image=bool(CONTROL_ARMS[arm]["blank_image"]),
            max_tokens=args.max_tokens,
        )
        _write_json(arm_run_dir / "predictions.json", payload)
        _write_jsonl(arm_run_dir / "predictions.jsonl", responses)
        gate_report = build_gate_report(
            responses=responses,
            stdout_cap=args.stdout_cap,
            allow_one_label=bool(args.allow_one_label),
        )
        gate_report["kind"] = "epic_visor_dynamic_hf_tamia_gate_report"
        _write_json(arm_run_dir / "gate_report.json", gate_report)
        if gate_report.get("status") != "pass" and not args.allow_gate_fail:
            raise RuntimeError(f"{arm}: gate failed; see {arm_run_dir / 'gate_report.json'}")
        score = score_visor_predictions(
            predictions=payload,
            hidden_index=_read_json(pack_dir / str(CONTROL_ARMS[arm]["hidden"])),
        )
        _write_json(arm_run_dir / "score.json", score)
        rows.append(
            {
                "model_key": args.model_key,
                "model": args.model_id,
                "arm": arm,
                "run_dir": str(arm_run_dir),
                "predictions": str(arm_run_dir / "predictions.json"),
                "score": str(arm_run_dir / "score.json"),
                "gate_report": str(arm_run_dir / "gate_report.json"),
                "request_pack": str(pack_dir / str(CONTROL_ARMS[arm]["pack"])),
                "hidden_index": str(pack_dir / str(CONTROL_ARMS[arm]["hidden"])),
                "answer_distribution": dict(Counter(str(row.get("answer")) for row in responses)),
                **_score_row(score=score, gate_report=gate_report),
            }
        )

    summary = {
        "kind": "epic_visor_hf_tamia_control_battery_summary",
        "model_id": args.model_id,
        "model_key": args.model_key,
        "pack_dir": str(pack_dir),
        "run_dir": str(run_dir),
        "arms": arms,
        "rows": rows,
        "claim_boundary": (
            "Full-weight Hugging Face/PyTorch rerun of the Phase 1 EPIC/VISOR paper "
            "battery on TamIA. Rows are model+runtime measurements, not population "
            "claims beyond the frozen audit pack."
        ),
    }
    _write_json(run_dir / "battery_summary.json", summary)
    _write_markdown_summary(run_dir / "battery_summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument("--pack-dir", type=Path, required=True)
    common_parent.add_argument("--run-dir", type=Path, required=True)
    common_parent.add_argument("--model-id", default="Qwen/Qwen3.5-27B")
    common_parent.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    common_parent.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(CONTROL_ARMS))
    common_parent.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)

    run_parser = subparsers.add_parser("run-shard", parents=[common_parent])
    run_parser.add_argument("--rank", required=True)
    run_parser.add_argument("--world-size", required=True)
    run_parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    run_parser.add_argument("--trust-remote-code", action="store_true")
    run_parser.add_argument("--local-files-only", action="store_true")
    run_parser.add_argument("--allow-generation-errors", action="store_true")
    run_parser.add_argument("--max-tasks-per-rank", type=int)
    run_parser.add_argument("--internvl-max-tiles", type=int, default=6)

    merge_parser = subparsers.add_parser("merge-score", parents=[common_parent])
    merge_parser.add_argument("--stdout-cap", type=int, default=DEFAULT_STDOUT_CAP)
    merge_parser.add_argument("--allow-one-label", action="store_true")
    merge_parser.add_argument("--allow-incomplete", action="store_true")
    merge_parser.add_argument("--allow-gate-fail", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run-shard":
        return run_shard(args)
    if args.command == "merge-score":
        return merge_score(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
