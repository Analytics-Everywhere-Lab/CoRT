"""Local-only BF16 temporal forecasting, with fail-closed full-forward patching.

Dependencies at backend construction: torch, accelerate, transformers (with
Gemma4 and Qwen3_5 support), Pillow, and their HF dependencies. Importing this module and
testing its partitions requires only the standard library. Cuts are zero-based
decoder block OUTPUT indexes. Log probabilities are natural logs; margins and
control tolerances are bits. This is a derived task, not the official STA metric.
No evaluator labels, suffix replay, hidden-state stacks, or remote downloads.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import time

from .common import digest
from .protocol import (CURRENT, FORECAST, PROMPTS, DENSE_FIXED_CUTS, DENSE_SEED,
                       is_dense_current, validate_contract, validate_rows_contract)
from .recognition_protocol import (is_recognition, HISTORY_ROLES as RECOGNITION_HISTORY_ROLES,
                                   HISTORY_KEYS as RECOGNITION_HISTORY_KEYS,
                                   FIXED_CUTS as RECOGNITION_FIXED_CUTS)


MODEL_PINS = {
    "gemma31": {
        "revision": "3548789868c5356dbf307c98e6f609007b82b3eb",
        "model_class": "Gemma4ForConditionalGeneration",
        "processor_class": "Gemma4Processor", "image_token_id": 258880,
        "layers": 60,
    },
    "qwen27": {
        "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654",
        "model_class": "Qwen3_5ForConditionalGeneration",
        "processor_class": "Qwen3VLProcessor", "image_token_id": 248056,
        "layers": 64,
    },
}
CANONICAL_MODEL_KEYS = {"gemma31": "gemma4_31b_hf_bf16", "qwen27": "qwen35_27b_hf_bf16"}
QUESTION = PROMPTS[FORECAST]
CONTROL_TOLERANCE_BITS = 0.001
SCENE_IMAGE_SIZE = (896, 672)
CANDIDATE_IMAGE_SIZE = (336, 336)
# Roles describe input slots, not the provenance of their replacement pixels.
_SCENES = {
    "B0": (("H1", "past1"), ("H2", "past2"), ("V", "current")),
    "B1": (("V", "current"),), "B2": (("H1", "past1"),),
    "B3": (("H2", "past2"),),
    "B4": (("H1", "current"), ("H2", "current"), ("V", "current")),
    "B5": (),
    "B6": (("H1", "current"), ("H2", "past2"), ("V", "current")),
    "B7": (("H1", "past1"), ("H2", "current"), ("V", "current")),
    "B8": (("H1", "past2"), ("H2", "past1"), ("V", "current")),
    # B9 replaces history with neutral RGB 128 images of unchanged dimensions.
    "B9": (("H1", "past1"), ("H2", "past2"), ("V", "current")),
}
_DENSE_HISTORY_ROLES = tuple(f"H{i}" for i in range(1, 9))
_DENSE_HISTORY_KEYS = tuple(f"past{i}" for i in range(1, 9))


def _dense_history_shuffle():
    # Hash-ranked rejection sampling freezes a derangement independently of
    # episode metadata, candidate order, and the process-global random state.
    attempt = 0
    while True:
        shuffled = tuple(sorted(_DENSE_HISTORY_KEYS, key=lambda key: digest(
            {"seed": DENSE_SEED, "attempt": attempt, "history": key})))
        if all(a != b for a, b in zip(shuffled, _DENSE_HISTORY_KEYS)):
            return shuffled
        attempt += 1


_DENSE_SCENES = {
    "B0": (*zip(_DENSE_HISTORY_ROLES, _DENSE_HISTORY_KEYS), ("V", "current")),
    "B1": (("V", "current"),),
    "B4": (*((role, "current") for role in _DENSE_HISTORY_ROLES), ("V", "current")),
    "B8": (*zip(_DENSE_HISTORY_ROLES, _dense_history_shuffle()), ("V", "current")),
}

_RECOGNITION_SCENES = {
    "B0": (*zip(RECOGNITION_HISTORY_ROLES, RECOGNITION_HISTORY_KEYS), ("V", "current")),
    "B1": (("V", "current"),),
    "B4": (*((role, "current") for role in RECOGNITION_HISTORY_ROLES), ("V", "current")),
    "B2": (("H1", "past1"),),
}


class GateError(RuntimeError):
    """An identity, boundary, or numerical control failed."""


class UnsupportedError(NotImplementedError):
    """No certified implementation exists for this intervention."""


def _require(ok, message):
    if not ok:
        raise GateError(message)


def _model_alias(model_key):
    reverse = {value: key for key, value in CANONICAL_MODEL_KEYS.items()}
    alias = reverse.get(model_key, model_key)
    _require(alias in MODEL_PINS, "unknown pinned model key")
    return alias


def slot_plan(condition, order, contract=None):
    """Return (role, image key) pairs, without consulting episode metadata."""
    if contract is not None:
        validate_contract(contract)
    scenes = _DENSE_SCENES if is_dense_current(contract) else _SCENES
    if is_recognition(contract):
        scenes = _RECOGNITION_SCENES
    _require(condition in scenes, f"unknown condition: {condition}")
    _require(type(order) is int and order in (0, 1), "order must be 0 or 1")
    return (*scenes[condition], ("CA", f"candidate{order}"),
            ("CB", f"candidate{1 - order}"))


def _history_roles(contract):
    if contract is not None:
        validate_contract(contract)
    if is_recognition(contract):
        return RECOGNITION_HISTORY_ROLES
    return _DENSE_HISTORY_ROLES if is_dense_current(contract) else ("H1", "H2")


def _validate_dense_roles(roles):
    _require(tuple(roles) in ((*_DENSE_HISTORY_ROLES, "V", "CA", "CB"), ("V", "CA", "CB")),
             "dense slots must be H1..H8,V,CA,CB or V,CA,CB in order")


def _validate_recognition_roles(roles):
    _require(tuple(roles) in ((*RECOGNITION_HISTORY_ROLES, "V", "CA", "CB"),
                             ("V", "CA", "CB"), ("H1", "CA", "CB")),
             "recognition slots must be H1..H16,V,CA,CB, V,CA,CB or H1,CA,CB in order")


def _tolist(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def _row(value):
    rows = _tolist(value)
    _require(isinstance(rows, (tuple, list)) and len(rows) == 1
             and isinstance(rows[0], (tuple, list)), "expected batch size one")
    return list(rows[0])


def _runs(positions):
    runs = []
    for p in positions:
        if not runs or p != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(p)
    return runs


def image_slot_positions(inputs, model_key, roles, *, merge_size=None, pooling_kernel=None, contract=None):
    """Dynamic official-processor slot validation; accepts pure fake tensors."""
    model_key = _model_alias(model_key)
    _history_roles(contract)
    if is_dense_current(contract):
        _require(model_key == "gemma31", "dense current supports Gemma31 only")
        _validate_dense_roles(roles)
    elif is_recognition(contract):
        _require(model_key == "gemma31", "recognition supports Gemma31 only")
        _validate_recognition_roles(roles)
    else:
        _require(len(roles) in (2, 3, 5) and len(set(roles)) == len(roles),
                 "expected distinct roles for 2, 3 or 5 slots")
    ids = _row(inputs["input_ids"])
    runs = _runs([i for i, token in enumerate(ids)
                  if token == MODEL_PINS[model_key]["image_token_id"]])
    _require(len(runs) == len(roles), "image-token run count differs from image slots")
    if model_key == "qwen27":
        grids = _tolist(inputs.get("image_grid_thw"))
        _require(isinstance(grids, (list, tuple)) and len(grids) == len(roles),
                 "Qwen image_grid_thw count differs from slots")
        _require(type(merge_size) is int and merge_size > 0, "missing Qwen merge size")
        for run, grid in zip(runs, grids):
            _require(len(grid) == 3 and all(type(n) is int and n > 0 for n in grid),
                     "invalid image_grid_thw")
            t, h, w = grid
            _require(t == 1 and h % merge_size == w % merge_size == 0,
                     "not a still-image grid divisible by merge size")
            _require(len(run) == t * (h // merge_size) * (w // merge_size),
                     "image-token count disagrees with Qwen grid")
        shape = getattr(inputs.get("pixel_values"), "shape", ())
        _require(len(shape) == 2 and shape[0] == sum(t * h * w for t, h, w in grids),
                 "Qwen flattened pixel patch count disagrees with grids")
    else:
        shape = getattr(inputs.get("pixel_values"), "shape", ())
        _require(len(shape) == 3 and shape[0] == len(roles),
                 "Gemma pixel_values first dimension differs from slots")
        coordinates = _tolist(inputs.get("image_position_ids"))
        _require(type(pooling_kernel) is int and pooling_kernel > 0,
                 "missing Gemma pooling kernel")
        _require(isinstance(coordinates, list) and len(coordinates) == len(roles),
                 "Gemma image_position_ids count differs from slots")
        for run, coords in zip(runs, coordinates):
            _require(len(coords) == shape[1] and all(
                len(xy) == 2 and all(type(x) is int for x in xy)
                and (xy == [-1, -1] or min(xy) >= 0) for xy in coords),
                "invalid Gemma patch coordinates/padding")
            valid = [tuple(xy) for xy in coords if xy != [-1, -1]]
            _require(valid and len(set(valid)) == len(valid), "missing/duplicate Gemma patches")
            width, height = max(x for x, _ in valid) + 1, max(y for _, y in valid) + 1
            _require(width % pooling_kernel == height % pooling_kernel == 0
                     and len(valid) == width * height,
                     "Gemma patches do not form complete rectangular pool bins")
            _require(len(run) == len(valid) // pooling_kernel**2,
                     "image-token count disagrees with Gemma pool geometry")
    return dict(zip(roles, runs))


def exact_text_positions(ids, tokenizer, text):
    """Match literal question tokens, permitting only boundary whitespace."""
    for variant in (text, " " + text, "\n" + text):
        tokens = list(tokenizer.encode(variant, add_special_tokens=False))
        hits = [i for i in range(len(ids) - len(tokens) + 1)
                if tokens and ids[i:i + len(tokens)] == tokens]
        _require(len(hits) <= 1, "ambiguous literal question-token span")
        if hits:
            return list(range(hits[0], hits[0] + len(tokens)))
    raise GateError("literal question-token span not found")


def partition_positions(ids, attention_mask, slots, question, prefix, *, contract=None):
    """C/Q/P/R partition K; contract-defined history+V=F_video; F_video+K=A."""
    history_roles = _history_roles(contract)
    if is_dense_current(contract):
        _validate_dense_roles(slots)
    elif is_recognition(contract):
        _validate_recognition_roles(slots)
    _require(len(ids) == len(attention_mask) and bool(ids), "invalid attention mask")
    _require(all(x in (0, 1) for x in attention_mask), "nonbinary attention mask")
    active = {i for i, x in enumerate(attention_mask) if x}
    _require(active and max(active) == len(ids) - 1,
             "readout must be the last active AND final sequence token")
    _require(set(slots) <= {*history_roles, "V", "CA", "CB"}
             and {"CA", "CB"} <= set(slots), "invalid slot roles")
    groups = {key: list(slots.get(key, [])) for key in (*history_roles, "V", "CA", "CB")}
    if is_dense_current(contract):
        _require(all(groups[key] for key in slots), "empty dense image slot")
    elif is_recognition(contract):
        _require(all(groups[key] for key in slots), "empty recognition image slot")
    groups.update(Q=list(question), P=list(prefix))
    _require(groups["Q"] and groups["P"] and groups["P"][-1] == max(active),
             "missing question/prefix or unverified readout")
    _require(groups["P"] == list(range(groups["P"][0], len(ids)))
             and max(groups["Q"]) < min(groups["P"]), "invalid question/prefix order")
    used = set()
    for name, positions in groups.items():
        _require(all(type(p) is int for p in positions)
                 and positions == sorted(set(positions)), f"invalid positions: {name}")
        values = set(positions)
        _require(values <= active and not used & values, f"overlap/inactive positions: {name}")
        used |= values
    groups["C"] = sorted(groups["CA"] + groups["CB"])
    groups["H"] = sorted(p for role in history_roles for p in groups[role])
    groups["F_video"] = sorted(groups["H"] + groups["V"])
    groups["R"] = sorted(active - used)
    groups["K"] = sorted(active - set(groups["F_video"]))
    groups["A"] = sorted(active)
    groups["C+Q"] = sorted(groups["C"] + groups["Q"])
    _require(sorted(groups["C"] + groups["Q"] + groups["P"] + groups["R"])
             == groups["K"], "C/Q/P/R must exhaust K")
    return groups


def select_positions(positions, position_set):
    """Resolve named groups and all 15 nonempty C/Q/P/R coalitions."""
    if position_set in positions:
        selected = positions[position_set]
    else:
        names = position_set.split("+")
        _require(len(set(names)) == len(names) and set(names) <= {"C", "Q", "P", "R"},
                 "unknown or duplicated position_set component")
        selected = sorted({p for name in names for p in positions[name]})
    _require(bool(selected), "empty position_set")
    return selected


def prepare_processor_inputs(processor, messages, roles, model_key, *, merge_size=None, question=QUESTION,
                             contract=None):
    """Native template boundaries and one-token answers, without HF imports."""
    if contract is not None:
        validate_contract(contract)
        question = contract["prompt"]

    def template(generation, tokenize):
        kwargs = dict(add_generation_prompt=generation, tokenize=tokenize,
                      enable_thinking=False)
        if tokenize:
            kwargs.update(return_dict=True, return_tensors="pt")
        return processor.apply_chat_template(messages, **kwargs)

    closed, rendered = template(False, False), template(True, False)
    _require(isinstance(closed, str) and isinstance(rendered, str)
             and rendered.startswith(closed) and len(rendered) > len(closed),
             "native template has no exact assistant-prefix boundary")
    tokenizer = processor.tokenizer
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    closed_text_ids, rendered_ids = encode(closed), encode(rendered)
    _require(rendered_ids[:len(closed_text_ids)] == closed_text_ids,
             "native assistant prefix crosses tokenizer boundary")
    prefix_ids = rendered_ids[len(closed_text_ids):]
    _require(prefix_ids, "empty native assistant prefix")
    answers = {}
    for choice in ("A", "B"):
        complete = encode(rendered + choice)
        _require(complete[:-1] == rendered_ids and len(complete) == len(rendered_ids) + 1,
                 f"answer {choice} is not one exact contextual token")
        answers[choice] = complete[-1]
    _require(answers["A"] != answers["B"], "answer token IDs collide")
    # Compare expanded processor IDs as well, not just unexpanded text tokens.
    closed_ids = _row(template(False, True)["input_ids"])
    inputs = dict(template(True, True))
    ids = _row(inputs["input_ids"])
    _require(ids == closed_ids + prefix_ids, "processor assistant-prefix boundary drift")
    slots = image_slot_positions(
        inputs, model_key, roles, merge_size=merge_size,
        pooling_kernel=getattr(processor.image_processor, "pooling_kernel_size", None),
        contract=contract,
    )
    positions = partition_positions(
        ids, _row(inputs["attention_mask"]), slots,
        exact_text_positions(ids, tokenizer, question),
        list(range(len(closed_ids), len(ids))),
        contract=contract,
    )
    return {"inputs": inputs, "positions": positions, "answer_ids": answers,
            "slot_roles": list(roles), "readout_position": len(ids) - 1}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path):
    if not path:
        return {"path": None, "sha256": None, "status": "unavailable"}
    path = Path(path)
    try:
        return {"path": str(path), "sha256": _sha256(path), "status": "hashed"}
    except OSError:
        return {"path": str(path), "sha256": None, "status": "unreadable"}


class HFBackend:
    """Not thread-safe: hooks are scoped to one serial, batch-one operation."""

    def __init__(self, model_key, model_path, device="cuda:0", attention_backend="eager", *, contract=None):
        self.contract = deepcopy(validate_contract(contract)) if contract is not None else None
        model_key = _model_alias(model_key)
        if is_dense_current(self.contract):
            _require(model_key == "gemma31", "dense current supports Gemma31 only")
            _require(attention_backend == "sdpa", "dense current requires SDPA")
        elif is_recognition(self.contract):
            _require(model_key == "gemma31", "recognition supports Gemma31 only")
            _require(attention_backend == "sdpa", "recognition requires SDPA")
        pin = MODEL_PINS[model_key]
        path = Path(model_path).expanduser().resolve()
        _require(path.is_dir() and path.name == pin["revision"],
                 "model_path must resolve to the pinned local snapshot revision directory")
        config = json.loads((path / "config.json").read_text())
        _require(config.get("architectures") == [pin["model_class"]]
                 and config.get("_commit_hash", pin["revision"]) == pin["revision"],
                 "local model config does not match pinned architecture/revision")
        _require(not config.get("quantization_config"), "quantized snapshots are unsupported")
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch, self.device = torch, torch.device(device)
        self.model_key = CANONICAL_MODEL_KEYS[model_key]
        self.processor = AutoProcessor.from_pretrained(
            str(path), revision=pin["revision"], local_files_only=True, trust_remote_code=False)
        _require(type(self.processor).__name__ == pin["processor_class"], "unexpected processor class")
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(path), revision=pin["revision"], local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={"": str(self.device)}, attn_implementation=attention_backend,
        )
        self.model.eval().requires_grad_(False)
        _require(type(self.model).__name__ == pin["model_class"], "unexpected model class")
        self.layers = self.model.model.language_model.layers
        _require(len(self.layers) == pin["layers"], "decoder layer count drift")
        text_config = self.model.config.text_config
        if model_key == "gemma31":
            _require(not getattr(text_config, "num_kv_shared_layers", 0)
                     and not getattr(text_config, "hidden_size_per_layer_input", 0),
                     "Gemma shared KV/per-layer inputs bypass raw-output-only patching")
            _require(self.processor.image_processor.pooling_kernel_size
                     == self.model.config.vision_config.pooling_kernel_size,
                     "model/processor pooling-kernel disagreement")
        self.merge_size = None
        if model_key == "qwen27":
            self.merge_size = self.model.config.vision_config.spatial_merge_size
            _require(self.processor.image_processor.merge_size == self.merge_size,
                     "model/processor merge-size disagreement")
        self.keep_last = "logits_to_keep" in inspect.signature(self.model.forward).parameters
        self.clear_cache()
        self.provenance = {
            "model_key": self.model_key, "revision": pin["revision"], "snapshot": str(path),
            "config_sha256": _sha256(path / "config.json"),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "cuda": torch.version.cuda,
            "device": str(self.device), "dtype": "bfloat16", "attention_backend": attention_backend,
            "local_files_only": True, "device_map": {"": str(self.device)},
            "logits_to_keep": 1 if self.keep_last else None,
            "sif": _fingerprint(os.environ.get("APPTAINER_CONTAINER")
                                or os.environ.get("SINGULARITY_CONTAINER")),
            "sources": {name: _fingerprint(inspect.getfile(obj)) for name, obj in (
                ("torch", torch), ("transformers", transformers),
                ("model", type(self.model)), ("processor", type(self.processor)))},
        }

    def close(self):
        """Release owned model references; parent may collect/empty CUDA cache."""
        self.clear_cache()
        self.layers, self.model, self.processor = (), None, None

    def clear_cache(self):
        """At most B0/B4 for one immutable staged episode/order and one cut."""
        self._scope = self._cut = None
        self._prepared_cache, self._native_cache = {}, {}
        self._self_checks, self._full_checks = {}, {}

    def _slot_plan(self, episode, order, condition):
        return slot_plan(condition, order, getattr(self, "contract", None))

    def _scope_key(self, episode, order):
        # The parent stages hash-verified immutable files. Stat identity also
        # invalidates reuse when those files are replaced or edited in place.
        files = []
        contract = getattr(self, "contract", None)
        for _, role in slot_plan("B0", 0, contract):
            path = Path(episode["images"][role]["path"])
            _require(path.is_absolute(), "episode image paths must be absolute")
            stat = path.stat()
            files.append((role, str(path.resolve()), stat.st_dev, stat.st_ino,
                          stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        scope = episode.get("episode_id"), order, tuple(files)
        return (*scope, digest(contract)) if contract is not None else scope

    def _activate_cache(self, episode, order, conditions, cut=None):
        contract = getattr(self, "contract", None)
        if contract is not None:
            validate_contract(contract)
        validate_rows_contract([episode], contract)
        _require(type(order) is int and order in (0, 1), "order must be 0 or 1")
        if is_dense_current(contract):
            _require(_model_alias(self.model_key) == "gemma31", "dense current supports Gemma31 only")
            for condition in conditions:
                self._slot_plan(episode, order, condition)
        elif is_recognition(contract):
            _require(_model_alias(self.model_key) == "gemma31", "recognition supports Gemma31 only")
            for condition in conditions:
                self._slot_plan(episode, order, condition)
        if not set(conditions) <= {"B0", "B4"}:
            self.clear_cache()
            return False
        scope = self._scope_key(episode, order)
        if scope != self._scope:
            self.clear_cache()
            self._scope = scope
        if cut is not None and cut != self._cut:
            self._self_checks, self._full_checks = {}, {}
            self._cut = cut
        return True

    def _prepared(self, episode, order, condition, cached):
        if cached and condition in self._prepared_cache:
            return self._prepared_cache[condition]
        prepared = self._prepare(episode, order, condition)
        if cached:
            self._prepared_cache[condition] = prepared
        return prepared

    def _remember_native(self, condition, scores, cached):
        if cached:
            if condition in self._native_cache:
                self._control(self._native_cache[condition], scores, "cached native score drift")
            else:
                self._native_cache[condition] = dict(scores)

    def __enter__(self):
        _require(self.model is not None, "backend is closed")
        return self

    def __exit__(self, *_exc):
        self.close()

    def _prepare(self, episode, order, condition):
        _require(self.processor is not None, "backend is closed")
        contract = getattr(self, "contract", None)
        if contract is not None:
            validate_contract(contract)
        validate_rows_contract([episode], contract)
        question = contract["prompt"] if contract else QUESTION
        from PIL import Image

        plan = self._slot_plan(episode, order, condition)
        images = []
        try:
            for role, key in plan:
                path = Path(episode["images"][key]["path"])
                _require(path.is_absolute(), "episode image paths must be absolute")
                with Image.open(path) as source:
                    expected = CANDIDATE_IMAGE_SIZE if role in ("CA", "CB") else SCENE_IMAGE_SIZE
                    _require(source.size == expected,
                             f"image geometry mismatch for {key}: {source.size} != {expected}; no resizing")
                    image = source.convert("RGB")
                if condition == "B9" and role in ("H1", "H2"):
                    size = image.size
                    image.close()
                    image = Image.new("RGB", size, (128, 128, 128))
                images.append(image)
            messages = [{"role": "user", "content": [
                *[{"type": "image", "image": im} for im in images],
                {"type": "text", "text": question},
            ]}]
            prepared = prepare_processor_inputs(
                self.processor, messages, [role for role, _ in plan], self.model_key,
                merge_size=self.merge_size, question=question, contract=contract,
            )
        finally:
            for image in images:
                image.close()
        prepared["inputs"] = {
            key: value.to(device=self.device, dtype=self.torch.bfloat16)
            if value.is_floating_point() else value.to(self.device)
            for key, value in prepared["inputs"].items()
        }
        return prepared

    def _synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _forward(self, prepared):
        _require(not ({"labels", "past_key_values", "cache_params", "inputs_embeds"}
                      & prepared["inputs"].keys()), "unexpected labels/cache/embeddings in processor inputs")
        kwargs = {"use_cache": False, "return_dict": True,
                  "past_key_values": None,
                  "output_hidden_states": False, "output_attentions": False}
        if self.keep_last:
            kwargs["logits_to_keep"] = 1
        self.model.eval()
        with self.torch.inference_mode():
            self._forward_count = getattr(self, "_forward_count", 0) + 1
            output = self.model(**prepared["inputs"], **kwargs)
            _require(getattr(output, "past_key_values", None) is None, "model returned an unexpected KV/recurrent cache")
            logits = output.logits
            length = prepared["readout_position"] + 1
            _require(logits.ndim == 3 and logits.shape[0] == 1
                     and logits.shape[1] == (1 if self.keep_last else length),
                     "unexpected logits/readout geometry")
            logprobs = self.torch.log_softmax(logits[0, -1].float(), dim=-1)
            a, b = (float(logprobs[prepared["answer_ids"][choice]].item()) for choice in ("A", "B"))
        _require(math.isfinite(a) and math.isfinite(b), "nonfinite A/B scores")
        return {"a_logprob": a, "b_logprob": b, "margin_ab_bits": (a - b) / math.log(2)}

    def _result(self, scores, prepared, started):
        self._synchronize()
        contract = getattr(self, "contract", None)
        binding = {"task": contract["task"], "contract_sha256": digest(contract)} if contract else {}
        return {**scores, **binding, "model_key": self.model_key,
                "positions": {key: list(value) for key, value in prepared["positions"].items()},
                "readout_position": prepared["readout_position"],
                "answer_token_ids": dict(prepared["answer_ids"]),
                "elapsed_s": time.perf_counter() - started, "provenance": deepcopy(self.provenance)}

    def score(self, episode: dict, order: int, condition: str) -> dict:
        self._synchronize()
        started = time.perf_counter()
        cached = self._activate_cache(episode, order, [condition])
        prepared = self._prepared(episode, order, condition, cached)
        scores = self._forward(prepared)
        self._remember_native(condition, scores, cached)
        return self._result(scores, prepared, started)

    def _align(self, donor, recipient):
        _require(all(donor[key] == recipient[key] for key in (
            "positions", "answer_ids", "slot_roles", "readout_position")),
            "donor/recipient partitions or readout differ")
        a, b = donor["inputs"], recipient["inputs"]
        _require(a.keys() == b.keys(), "processor input keys differ")
        for key in a:
            _require(a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
                     and a[key].device == b[key].device, f"processor geometry differs: {key}")
            if key != "pixel_values":
                _require(self.torch.equal(a[key], b[key]), f"processor side input differs: {key}")

    def _hidden(self, output, length):
        hidden = output[0] if isinstance(output, tuple) else output
        _require(self.torch.is_tensor(hidden) and hidden.ndim == 3
                 and hidden.shape[:2] == (1, length), "unexpected raw block-output geometry")
        return hidden

    @contextmanager
    def _capture(self, cut, length):
        state = {}

        def hook(_module, _args, output):
            _require(not state, "capture block executed more than once")
            hidden = self._hidden(output, length)
            _require(bool(self.torch.isfinite(hidden).all()), "nonfinite donor activation")
            state["hidden"] = hidden.detach().clone()

        handle = self.layers[cut].register_forward_hook(hook)
        try:
            yield state
            _require("hidden" in state, "capture block did not execute")
        except BaseException:
            state.clear()
            raise
        finally:
            handle.remove()

    def _patched(self, prepared, cut, source, positions):
        length = prepared["readout_position"] + 1
        _require(positions and positions == sorted(set(positions))
                 and set(positions) <= set(prepared["positions"]["A"]), "invalid patch selection")
        audit = {"hook_calls": 0}

        def hook(_module, _args, output):
            audit["hook_calls"] += 1
            _require(audit["hook_calls"] == 1, "patch block executed more than once")
            hidden = self._hidden(output, length)
            _require(hidden.shape == source.shape and hidden.dtype == source.dtype
                     and hidden.device == source.device, "activation geometry/dtype/device mismatch")
            patched = hidden.clone()
            patched[:, positions, :] = source[:, positions, :]
            selected = set(positions)
            outside = [i for i in range(length) if i not in selected]
            audit["selected_rows_equal"] = bool(self.torch.equal(
                patched[:, positions, :], source[:, positions, :]))
            audit["preserved_rows_equal"] = bool(self.torch.equal(
                patched[:, outside, :], hidden[:, outside, :]))
            _require(audit["selected_rows_equal"] and audit["preserved_rows_equal"],
                     "raw-output replacement failed row identity checks")
            return (patched, *output[1:]) if isinstance(output, tuple) else patched

        handle = self.layers[cut].register_forward_hook(hook)
        try:
            scores = self._forward(prepared)
            _require(audit["hook_calls"] == 1, "patch block did not execute")
            return scores, audit
        finally:
            handle.remove()

    @staticmethod
    def _control(reference, actual, label):
        delta = {key: abs(actual[key] - reference[key]) / (1 if key == "margin_ab_bits" else math.log(2))
                 for key in ("a_logprob", "b_logprob", "margin_ab_bits")}
        _require(all(math.isfinite(x) and x <= CONTROL_TOLERANCE_BITS for x in delta.values()),
                 f"{label} failed 0.001-bit tolerance: {delta}; smaller patches are not certified")
        return {"passed": True, "absolute_error_bits": delta,
                "max_drift_bits": max(delta.values()), "scores": actual}

    def _validate_patch_request(self, donor_condition, recipient_condition, cut,
                                position_set, self_patch):
        contract = getattr(self, "contract", None)
        if contract is not None:
            validate_contract(contract)
        dense = is_dense_current(contract)
        if contract and contract["task"] == CURRENT and not dense:
            raise UnsupportedError("current-interaction companion supports descriptive behavior only")
        _require(type(cut) is int and 0 <= cut < len(self.layers), "cut is a zero-based block index")
        _require(type(self_patch) is bool, "self_patch must be boolean")
        if dense:
            _require(_model_alias(self.model_key) == "gemma31", "dense current supports Gemma31 only")
            _require(cut in DENSE_FIXED_CUTS, "dense current requires fixed cuts 22,34")
            if self_patch:
                _require((donor_condition, recipient_condition, position_set) == ("B0", "B0", "A"),
                         "dense current self patch must be B0-to-B0 A")
            else:
                _require((donor_condition, recipient_condition) in (("B0", "B4"), ("B4", "B0"))
                         and position_set in ("H", "V", "K", "A"),
                         "dense current patches are B0/B4 H,V,K,A only")
        elif is_recognition(contract):
            _require(_model_alias(self.model_key) == "gemma31", "recognition supports Gemma31 only")
            _require(cut in RECOGNITION_FIXED_CUTS, "recognition requires fixed cuts 22,34")
            if self_patch:
                _require((donor_condition, recipient_condition, position_set) == ("B0", "B0", "A"),
                         "recognition self patch must be B0-to-B0 A")
            else:
                _require((donor_condition, recipient_condition) in (("B0", "B4"), ("B4", "B0"))
                         and position_set in ("H", "V", "K", "A"),
                         "recognition patches are B0/B4 H,V,K,A only")

    def patch(self, episode, order, donor_condition, recipient_condition, cut: int,
              position_set: str, self_patch=False) -> dict:
        self._validate_patch_request(donor_condition, recipient_condition, cut,
                                     position_set, self_patch)
        self._synchronize()
        started = time.perf_counter()
        forwards_before = getattr(self, "_forward_count", 0)
        cached = self._activate_cache(episode, order, [donor_condition, recipient_condition], cut)
        recipient = self._prepared(episode, order, recipient_condition, cached)
        donor = self._prepared(episode, order, donor_condition, cached)
        self._align(donor, recipient)
        positions = select_positions(recipient["positions"], position_set)
        length, active = recipient["readout_position"] + 1, recipient["positions"]["A"]
        recipient_self = self._self_checks.get(recipient_condition) if cached else None
        donor_self = self._self_checks.get(donor_condition) if cached else None
        full_control = self._full_checks.get((donor_condition, recipient_condition)) if cached else None
        reused = {"recipient_self": recipient_self is not None and not self_patch,
                  "donor_self": donor_self is not None, "full_A": full_control is not None}
        selected_self = None
        # No activations are cached. Scalar certificates are bound to the same
        # immutable pair, direction and cut; each new donor capture rechecks score drift.
        if recipient_self is None or self_patch:
            with self._capture(cut, length) as captured:
                recipient_native = self._forward(recipient)
            try:
                self._remember_native(recipient_condition, recipient_native, cached)
                same, _ = self._patched(recipient, cut, captured["hidden"], active)
                recipient_self = self._control(recipient_native, same, "recipient self patch")
                if cached:
                    self._self_checks[recipient_condition] = recipient_self
                if self_patch:
                    result, row_audit = self._patched(recipient, cut, captured["hidden"], positions)
                    selected_self = self._control(recipient_native, result, "selected self patch")
            finally:
                captured.clear()
        else:
            recipient_native = dict(self._native_cache[recipient_condition])
        with self._capture(cut, length) as captured:
            donor_native = self._forward(donor)
        try:
            self._remember_native(donor_condition, donor_native, cached)
            if donor_self is None:
                same, _ = self._patched(donor, cut, captured["hidden"], active)
                donor_self = self._control(donor_native, same, "donor self patch")
                if cached:
                    self._self_checks[donor_condition] = donor_self
            else:
                donor_self = self._control(donor_native, donor_self["scores"], "cached donor self")
            if full_control is None:
                full, full_audit = self._patched(recipient, cut, captured["hidden"], active)
                full_control = {**self._control(donor_native, full, "full A transfer"), **full_audit}
                if cached:
                    self._full_checks[donor_condition, recipient_condition] = full_control
            else:
                full_control = {**full_control, **self._control(
                    donor_native, full_control["scores"], "cached full A transfer")}
            if not self_patch:
                result, row_audit = self._patched(recipient, cut, captured["hidden"], positions)
                if positions == active:
                    self._control(donor_native, result, "selected full A transfer")
        finally:
            captured.clear()
        return {**self._result(result, recipient, started),
                "donor_score": donor_native, "recipient_score": recipient_native,
                "donor_margin_ab_bits": donor_native["margin_ab_bits"],
                "recipient_margin_ab_bits": recipient_native["margin_ab_bits"],
                "donor_condition": donor_condition, "recipient_condition": recipient_condition,
                "cut": cut, "position_set": position_set, "self_patch": self_patch,
                "technical": {"alignment_pass": True, "self_patch_pass": True,
                    "full_A_pass": True, "full_forward_reference": True, "suffix_replay": False,
                    "forward_count": self._forward_count - forwards_before,
                    "controls_reused": reused,
                    "self_max_drift_bits": max(recipient_self["max_drift_bits"], donor_self["max_drift_bits"],
                                               selected_self["max_drift_bits"] if selected_self else 0.0),
                    "full_closure_max_drift_bits": full_control["max_drift_bits"],
                    "tolerance_bits": CONTROL_TOLERANCE_BITS, **row_audit},
                "controls": deepcopy({"recipient_self": recipient_self, "donor_self": donor_self,
                             "selected_self": selected_self,
                             "full_A": full_control})}

    def prune(self, episode, order, condition, cut, position_set) -> dict:
        raise UnsupportedError(
            "Temporal pruning is unsupported: existing compression is certified only for "
            "the three-image Gemma SDPA operator, not dynamic temporal slots or Qwen "
            "hybrid side inputs. No zeroing/masking approximation is substituted.")
