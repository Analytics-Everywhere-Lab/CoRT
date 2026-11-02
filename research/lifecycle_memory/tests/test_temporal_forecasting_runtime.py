"""Pure processor/partition tests and optional tiny CPU-only hook checks."""
import os
import json
from itertools import combinations
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from research.lifecycle_memory.temporal_forecasting import runtime as rt


class FakeTensor:
    def __init__(self, values=None, shape=None):
        self.values = values
        self.shape = shape

    def tolist(self):
        return self.values


class FakeTokenizer:
    def __init__(self, image_id):
        self.image_id = image_id

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        result = []
        while text:
            if text.startswith("<image>"):
                result.append(self.image_id)
                text = text[len("<image>"):]
            else:
                result.append(ord(text[0]) + 10)
                text = text[1:]
        return result


class FakeProcessor:
    def __init__(self, model_key, slots):
        self.tokenizer = FakeTokenizer(rt.MODEL_PINS[model_key]["image_token_id"])
        self.image_processor = SimpleNamespace(pooling_kernel_size=3)
        self.model_key, self.slots = model_key, slots
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["enable_thinking"] is False
        text = "<user>" + "<image>\n" * self.slots + rt.QUESTION + "</user>\n"
        if kwargs["add_generation_prompt"]:
            text += "<assistant>\n"
        if not kwargs["tokenize"]:
            return text
        ids, index = [], 0
        for token in self.tokenizer.encode(text):
            if token == self.tokenizer.image_id:
                index += 1
                ids.extend([token] * index)
            else:
                ids.append(token)
        inputs = {"input_ids": FakeTensor([ids]),
                  "attention_mask": FakeTensor([[1] * len(ids)]),
                  "pixel_values": FakeTensor(shape=(self.slots, 4, 3))}
        if self.model_key == "qwen27":
            inputs["image_grid_thw"] = FakeTensor([[1, 2, 2 * (i + 1)] for i in range(self.slots)])
            inputs["pixel_values"].shape = (sum(4 * (i + 1) for i in range(self.slots)), 12)
        else:
            max_patches = self.slots * 9
            inputs["pixel_values"].shape = (self.slots, max_patches, 12)
            inputs["image_position_ids"] = FakeTensor([
                [[x, y] for y in range(3) for x in range(3 * (i + 1))]
                + [[-1, -1]] * (max_patches - 9 * (i + 1)) for i in range(self.slots)])
        return inputs


def prepared_fake(model_key="gemma31", condition="B0", order=0):
    roles = [role for role, _ in rt.slot_plan(condition, order)]
    processor = FakeProcessor(model_key, len(roles))
    prepared = rt.prepare_processor_inputs(processor, [], roles, model_key, merge_size=2)
    return processor, prepared


def test_import_is_lazy_in_fresh_process():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'PIL'}:
            raise AssertionError('heavy dependency imported: ' + fullname)
sys.meta_path.insert(0, Block())
from research.lifecycle_memory.temporal_forecasting import runtime
assert runtime.CONTROL_TOLERANCE_BITS == 0.001
"""
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("condition,sources", [
    ("B0", ["past1", "past2", "current"]), ("B1", ["current"]),
    ("B2", ["past1"]), ("B3", ["past2"]),
    ("B4", ["current", "current", "current"]), ("B5", []),
    ("B6", ["current", "past2", "current"]),
    ("B7", ["past1", "current", "current"]),
    ("B8", ["past2", "past1", "current"]),
    ("B9", ["past1", "past2", "current"]),
])
def test_condition_images_and_counterbalancing(condition, sources):
    assert [key for _, key in rt.slot_plan(condition, 0)] == sources + ["candidate0", "candidate1"]
    assert [key for _, key in rt.slot_plan(condition, 1)] == sources + ["candidate1", "candidate0"]


@pytest.mark.parametrize("model_key", ["gemma31", "qwen27"])
@pytest.mark.parametrize("condition", [f"B{i}" for i in range(10)])
def test_dynamic_native_processor_partitions(model_key, condition):
    processor, prepared = prepared_fake(model_key, condition)
    p = prepared["positions"]
    slots = len(rt.slot_plan(condition, 0))
    assert slots in (2, 3, 5)
    assert len(p["CA"]) == slots - 1 and len(p["CB"]) == slots
    assert p["C"] == sorted(p["CA"] + p["CB"])
    assert p["F_video"] == sorted(p["H1"] + p["H2"] + p["V"])
    assert p["H"] == sorted(p["H1"] + p["H2"])
    assert p["K"] == sorted(p["C"] + p["Q"] + p["P"] + p["R"])
    assert p["A"] == sorted(p["F_video"] + p["K"])
    assert len(set(p["K"])) == len(p["K"])
    assert p["P"][-1] == p["A"][-1] == prepared["readout_position"]
    assert len(p["P"]) == len(processor.tokenizer.encode("<assistant>\n"))
    ids = rt._row(prepared["inputs"]["input_ids"])
    assert [ids[i] for i in p["Q"]] == processor.tokenizer.encode(rt.QUESTION)
    assert prepared["answer_ids"] == {"A": ord("A") + 10, "B": ord("B") + 10}
    assert len(processor.calls) == 4


@pytest.mark.parametrize("condition", ["B0", "B1", "B5"])
@pytest.mark.parametrize("bad", ["run_count", "grid_count", "grid_geometry", "merge_size"])
def test_qwen_bad_geometry_fails_closed(condition, bad):
    _, prepared = prepared_fake("qwen27", condition)
    inputs, merge = prepared["inputs"], 2
    if bad == "run_count":
        inputs["input_ids"].values[0][6] = 1  # Remove the first image run.
    elif bad == "grid_count":
        inputs["image_grid_thw"].values.pop()
    elif bad == "grid_geometry":
        inputs["image_grid_thw"].values[0] = [1, 3, 2]
    else:
        merge = None
    with pytest.raises(rt.GateError):
        rt.image_slot_positions(inputs, "qwen27", prepared["slot_roles"], merge_size=merge)


@pytest.mark.parametrize("condition", ["B0", "B1", "B5"])
def test_gemma_bad_image_batch_fails_closed(condition):
    _, prepared = prepared_fake(condition=condition)
    prepared["inputs"]["pixel_values"].shape = (99, 4, 3)
    with pytest.raises(rt.GateError, match="first dimension"):
        rt.image_slot_positions(prepared["inputs"], "gemma31", prepared["slot_roles"])


@pytest.mark.parametrize("bad", ["missing", "duplicate", "partial", "count", "padding", "kernel"])
def test_gemma_patch_coordinates_must_prove_each_soft_token_count(bad):
    _, prepared = prepared_fake()
    inputs, kernel = prepared["inputs"], 3
    coords = inputs["image_position_ids"].values
    if bad == "missing":
        inputs.pop("image_position_ids")
    elif bad == "duplicate":
        coords[0][1] = coords[0][0]
    elif bad == "partial":
        coords[0][0] = [-1, -1]
    elif bad == "count":
        coords[0] = coords[1]  # Two pool bins for a one-token slot.
    elif bad == "padding":
        coords[0][0] = [-1, 0]
    else:
        kernel = None
    with pytest.raises(rt.GateError):
        rt.image_slot_positions(inputs, "gemma31", prepared["slot_roles"], pooling_kernel=kernel)


def test_qwen_flattened_pixel_count_is_not_the_image_batch_count():
    _, prepared = prepared_fake("qwen27")
    prepared["inputs"]["pixel_values"].shape = (5, 12)
    with pytest.raises(rt.GateError, match="flattened pixel patch count"):
        rt.image_slot_positions(prepared["inputs"], "qwen27", prepared["slot_roles"], merge_size=2)


@pytest.mark.parametrize("bad", ["overlap", "padding", "missing_prefix", "readout", "duplicate"])
def test_partition_rejects_invalid_role_boundaries(bad):
    ids, mask = list(range(12)), [1] * 12
    slots, q, p = {"CA": [1, 2], "CB": [4, 5]}, [7, 8], [10, 11]
    if bad == "overlap":
        q = [5, 6]
    elif bad == "padding":
        mask[2] = 0
    elif bad == "missing_prefix":
        p = []
    elif bad == "readout":
        mask[-1] = 0
    else:
        slots["CA"] = [1, 1]
    with pytest.raises(rt.GateError):
        rt.partition_positions(ids, mask, slots, q, p)


def test_question_matching_is_unique_and_not_a_hand_guessed_offset():
    tokenizer = FakeTokenizer(99)
    q = tokenizer.encode(rt.QUESTION)
    assert rt.exact_text_positions([1, 2] + q + [3], tokenizer, rt.QUESTION) == list(range(2, len(q) + 2))
    with pytest.raises(rt.GateError, match="ambiguous"):
        rt.exact_text_positions(q + q, tokenizer, rt.QUESTION)
    with pytest.raises(rt.GateError, match="not found"):
        rt.exact_text_positions([1, 2], tokenizer, rt.QUESTION)


@pytest.mark.parametrize("bad", ["native_boundary", "expanded_boundary", "multi_token_answer"])
def test_native_template_and_answer_boundaries_fail_closed(bad):
    processor = FakeProcessor("gemma31", 2)
    original_template = processor.apply_chat_template
    original_encode = processor.tokenizer.encode

    def template(messages, **kwargs):
        result = original_template(messages, **kwargs)
        if kwargs["add_generation_prompt"]:
            if bad == "native_boundary" and not kwargs["tokenize"]:
                return "changed" + result
            if bad == "expanded_boundary" and kwargs["tokenize"]:
                result["input_ids"].values[0][0] += 1
        return result

    def encode(text, **kwargs):
        ids = original_encode(text, **kwargs)
        return ids + [999] if bad == "multi_token_answer" and text.endswith("\nA") else ids

    processor.apply_chat_template, processor.tokenizer.encode = template, encode
    with pytest.raises(rt.GateError):
        rt.prepare_processor_inputs(processor, [], ["CA", "CB"], "gemma31")


def test_prune_is_explicitly_unsupported_without_loading_anything():
    backend = rt.HFBackend.__new__(rt.HFBackend)
    with pytest.raises(rt.UnsupportedError, match="No zeroing/masking"):
        backend.prune({}, 0, "B0", 1, "H")


@pytest.mark.parametrize("model_key", ["gemma31", "qwen27"])
@pytest.mark.parametrize("canonical", [False, True])
def test_loader_is_pinned_bf16_local_only_and_device_portable(monkeypatch, model_key, canonical):
    pin = rt.MODEL_PINS[model_key]
    calls = {}
    path = Path("/local/snapshots") / pin["revision"]
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    monkeypatch.setattr(Path, "is_dir", lambda self: self == path)
    monkeypatch.setattr(Path, "read_text", lambda self: json.dumps({"architectures": [pin["model_class"]]}))
    monkeypatch.setattr(rt, "_sha256", lambda path: "test-hash")
    monkeypatch.setattr(rt, "_fingerprint", lambda path: {"path": str(path), "sha256": "test-hash"})
    torch, transformers = ModuleType("torch"), ModuleType("transformers")
    torch.__file__, transformers.__file__ = "/fake/torch.py", "/fake/transformers.py"
    torch.__version__, transformers.__version__ = "test-torch", "test-transformers"
    torch.version, torch.bfloat16 = SimpleNamespace(cuda="test-cuda"), "bf16"
    class Device:
        def __init__(self, device):
            self.type, self.name = device.split(":")[0], device

        def __str__(self):
            return self.name

    torch.device = Device
    processor = type(pin["processor_class"], (), {})()
    processor.image_processor = SimpleNamespace(merge_size=2, pooling_kernel_size=3)

    def model_to(self, device):
        raise AssertionError("do not first materialize the full checkpoint on CPU")

    def model_eval(self):
        calls["eval"] = True
        return self

    def requires_grad(self, flag):
        calls["requires_grad"] = flag
        return self

    def forward(self, logits_to_keep=0):
        raise AssertionError("loader must not run any forwards")

    model = type(pin["model_class"], (), {"to": model_to, "eval": model_eval,
                  "requires_grad_": requires_grad, "forward": forward})()
    model.model = SimpleNamespace(language_model=SimpleNamespace(layers=[None] * pin["layers"]))
    model.config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2, pooling_kernel_size=3),
                                   text_config=SimpleNamespace())

    def load_processor(source, **kwargs):
        calls["processor"] = (source, kwargs)
        return processor

    def load_model(source, **kwargs):
        calls["model"] = (source, kwargs)
        return model

    transformers.AutoProcessor = SimpleNamespace(from_pretrained=load_processor)
    transformers.AutoModelForImageTextToText = SimpleNamespace(from_pretrained=load_model)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    key = rt.CANONICAL_MODEL_KEYS[model_key] if canonical else model_key
    backend = rt.HFBackend(key, str(path), device="cuda:3", attention_backend="sdpa")
    for kind in ("processor", "model"):
        source, kwargs = calls[kind]
        assert source == str(path)
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        assert kwargs["revision"] == pin["revision"]
    assert calls["model"][1]["dtype"] == "bf16"
    assert calls["model"][1]["device_map"] == {"": "cuda:3"}
    assert calls["model"][1]["attn_implementation"] == "sdpa"
    assert calls["eval"] and calls["requires_grad"] is False
    assert backend.keep_last and backend.provenance["revision"] == pin["revision"]
    assert backend.provenance["torch"] == "test-torch" and "sif" in backend.provenance
    assert backend.model_key == backend.provenance["model_key"] == rt.CANONICAL_MODEL_KEYS[model_key]
    with backend as same:
        assert same is backend
    assert backend.model is None and backend.processor is None and not backend.layers
    backend.close()  # Idempotent.
    with pytest.raises(rt.GateError, match="closed"):
        backend.__enter__()


@pytest.mark.parametrize("condition", [f"B{i}" for i in range(10)])
@pytest.mark.parametrize("wrong_size", [False, True])
def test_image_loading_is_local_counterbalanced_and_gray_is_explicit(monkeypatch, condition, wrong_size):
    created, seen = [], {}

    class Image:
        def __init__(self, path=None, rgb=None):
            size = rt.CANDIDATE_IMAGE_SIZE if path and "candidate" in path else rt.SCENE_IMAGE_SIZE
            self.path, self.rgb, self.size, self.closed = path, rgb, ((24, 16) if wrong_size else size), False
            created.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def convert(self, mode):
            assert mode == "RGB"
            return Image(self.path)

        def close(self):
            self.closed = True

    def new(mode, size, rgb):
        assert mode == "RGB" and size == rt.SCENE_IMAGE_SIZE
        return Image(rgb=rgb)

    pil = ModuleType("PIL")
    pil.Image = SimpleNamespace(open=lambda path: Image(str(path)), new=new)
    monkeypatch.setitem(sys.modules, "PIL", pil)

    def prepare(processor, messages, roles, model_key, **kwargs):
        seen["roles"] = roles
        content = messages[0]["content"]
        assert content[-1] == {"type": "text", "text": rt.QUESTION}
        seen["images"] = [x["image"] for x in content[:-1]]
        return {"inputs": {}}

    monkeypatch.setattr(rt, "prepare_processor_inputs", prepare)
    backend = rt.HFBackend.__new__(rt.HFBackend)
    backend.processor, backend.model_key, backend.merge_size = object(), "gemma31", None
    keys = ("past1", "past2", "current", "candidate0", "candidate1")
    episode = {"images": {key: {"path": "/resolved/" + key + ".png"} for key in keys}}
    if wrong_size:
        with pytest.raises(rt.GateError, match="geometry mismatch.*no resizing"):
            backend._prepare(episode, 1, condition)
        assert all(image.closed for image in created) and not seen
        return
    backend._prepare(episode, 1, condition)
    assert all(image.closed for image in created)
    for image, (role, key) in zip(seen["images"], rt.slot_plan(condition, 1)):
        if condition == "B9" and role in ("H1", "H2"):
            assert image.rgb == (128, 128, 128) and image.path is None
        else:
            assert image.path == episode["images"][key]["path"] and image.rgb is None
    assert seen["images"][-2].path.endswith("candidate1.png")
    assert seen["images"][-1].path.endswith("candidate0.png")


@pytest.mark.parametrize("order", [-1, 2, True, "0"])
def test_invalid_order_rejected(order):
    with pytest.raises(rt.GateError):
        rt.slot_plan("B0", order)


def test_all_fifteen_task_role_coalitions_and_canonical_mapping():
    _, prepared = prepared_fake()
    positions = prepared["positions"]
    count = 0
    for size in range(1, 5):
        for names in combinations("CQPR", size):
            selected = rt.select_positions(positions, "+".join(names))
            assert selected == sorted(p for name in names for p in positions[name])
            count += 1
    assert count == 15
    assert rt.select_positions(positions, "C+Q+P+R") == positions["K"]
    assert rt.image_slot_positions(prepared["inputs"], "gemma4_31b_hf_bf16", prepared["slot_roles"], pooling_kernel=3)
    with pytest.raises(rt.GateError):
        rt.select_positions(positions, "C+C")
    with pytest.raises(rt.GateError):
        rt.select_positions(positions, "gold")


def cpu_backend(*, tuple_output=False, bypass=False, keep_last=True):
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def forward(self, hidden):
            result = hidden + hidden.cumsum(dim=1) * 0.05
            return (result, "untouched auxiliary") if tuple_output else result

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(), Block(), Block()])
            self.calls = 0

        def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
            assert not self.training and torch.is_inference_mode_enabled()
            assert kwargs["use_cache"] is False and kwargs["output_hidden_states"] is False
            assert kwargs["past_key_values"] is None
            assert kwargs.get("logits_to_keep") == (1 if keep_last else None)
            self.calls += 1
            hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2).clone() * 0.01
            hidden[:, 0, 0] += pixel_values.item()
            for block in self.layers:
                result = block(hidden)
                if tuple_output:
                    assert result[1] == "untouched auxiliary"
                    hidden = result[0]
                else:
                    hidden = result
            logits = torch.stack((hidden[..., 0], hidden[..., 1], -hidden[..., 0]), dim=-1)
            if bypass:
                logits[..., 0] += pixel_values.item()  # Deliberate unpatched side path.
            return SimpleNamespace(logits=logits[:, -1:] if keep_last else logits)

    backend = rt.HFBackend.__new__(rt.HFBackend)
    backend.torch, backend.device = torch, torch.device("cpu")
    backend.model_key = "gemma4_31b_hf_bf16"
    backend.model, backend.keep_last = Tiny(), keep_last
    backend.layers = backend.model.layers
    backend.provenance = {"test_only": True}
    backend.clear_cache()
    backend._scope_key = lambda episode, order: (id(episode), order)

    def prepare(episode, order, condition):
        assert set(episode) == {"images"}  # No gold labels in method-facing requests.
        return {"inputs": {"input_ids": torch.arange(8).reshape(1, 8),
                           "attention_mask": torch.ones(1, 8, dtype=torch.long),
                           "pixel_values": torch.tensor([1.0 if condition == "B0" else 0.0])},
                "positions": {"A": list(range(8)), "H": [0, 1], "P": [7]},
                "answer_ids": {"A": 0, "B": 1}, "slot_roles": ["H1", "H2", "V", "CA", "CB"],
                "readout_position": 7}

    backend._prepare = prepare
    return backend


@pytest.mark.parametrize("tuple_output", [False, True])
@pytest.mark.parametrize("keep_last", [False, True])
@pytest.mark.parametrize("cut", [0, 1, 2])
def test_cpu_raw_output_patch_and_per_cut_controls(tuple_output, keep_last, cut):
    backend = cpu_backend(tuple_output=tuple_output, keep_last=keep_last)
    episode = {"images": {}}
    score = backend.score(episode, 0, "B0")
    assert score["a_logprob"] < 0 and score["b_logprob"] < 0
    assert score["model_key"] == "gemma4_31b_hf_bf16"
    before = backend.model.calls
    result = backend.patch(episode, 0, "B0", "B4", cut, "H")
    assert backend.model.calls - before == 6  # Native pair, self pair, full A, selected.
    assert result["donor_margin_ab_bits"] == score["margin_ab_bits"]
    assert result["recipient_margin_ab_bits"] == result["recipient_score"]["margin_ab_bits"]
    technical = result["technical"]
    assert technical["selected_rows_equal"] and technical["preserved_rows_equal"]
    assert technical["full_A_pass"] and technical["self_patch_pass"]
    assert technical["self_max_drift_bits"] == technical["full_closure_max_drift_bits"] == 0.0
    assert technical["suffix_replay"] is False and result["elapsed_s"] >= 0
    assert all(not block._forward_hooks for block in backend.layers)
    full = backend.patch(episode, 0, "B0", "B4", cut, "A")
    assert full["margin_ab_bits"] == full["donor_margin_ab_bits"]


def test_cpu_self_patch_is_recipient_identity_and_still_checks_full_transfer():
    backend = cpu_backend()
    result = backend.patch({"images": {}}, 0, "B0", "B4", 1, "H", self_patch=True)
    assert result["margin_ab_bits"] == result["recipient_margin_ab_bits"]
    assert result["controls"]["full_A"]["scores"]["margin_ab_bits"] == result["donor_margin_ab_bits"]
    assert backend.model.calls == 6


def test_cpu_full_closure_failure_blocks_smaller_patch_and_cleans_hooks():
    backend = cpu_backend(bypass=True)
    with pytest.raises(rt.GateError, match="full A transfer failed"):
        backend.patch({"images": {}}, 0, "B0", "B4", 1, "H")
    assert backend.model.calls == 5  # No selected patch after a failed full transfer.
    assert all(not block._forward_hooks for block in backend.layers)


def test_cpu_alignment_rejects_nonvisual_change_before_any_forward():
    backend = cpu_backend()
    original = backend._prepare

    def changed(*args):
        prepared = original(*args)
        if args[-1] == "B4":
            prepared["inputs"]["input_ids"][0, 0] += 1
        return prepared

    backend._prepare = changed
    with pytest.raises(rt.GateError, match="side input differs"):
        backend.patch({"images": {}}, 0, "B0", "B4", 0, "H")
    assert backend.model.calls == 0


def test_cpu_bounded_cache_reduces_repeat_forwards_but_recaptures_donor():
    backend = cpu_backend()
    episode = {"images": {}}
    preparations = []
    original = backend._prepare

    def prepare(*args):
        preparations.append(args[-1])
        return original(*args)

    backend._prepare = prepare
    first = backend.patch(episode, 0, "B0", "B4", 0, "H")
    assert backend.model.calls == 6 and len(preparations) == 2
    assert first["technical"]["forward_count"] == 6
    second = backend.patch(episode, 0, "B0", "B4", 0, "P")
    assert backend.model.calls == 8 and len(preparations) == 2
    assert second["technical"]["forward_count"] == 2
    assert all(second["technical"]["controls_reused"].values())
    assert not any(first["technical"]["controls_reused"].values())
    # Opposite direction needs its own full-A proof, not just a reversed flag.
    reverse = backend.patch(episode, 0, "B4", "B0", 0, "H")
    assert backend.model.calls == 11
    assert reverse["technical"]["controls_reused"] == {
        "recipient_self": True, "donor_self": True, "full_A": False}
    # A new cut discards every certificate but keeps only the two prepared inputs.
    backend.patch(episode, 0, "B0", "B4", 1, "H")
    assert backend.model.calls == 17 and len(preparations) == 2
    assert set(backend._prepared_cache) == {"B0", "B4"}
    assert len(backend._full_checks) == 1 and len(backend._self_checks) == 2
    assert all(not backend.torch.is_tensor(value) for entry in backend._native_cache.values()
               for value in entry.values())
    # Changing order evicts inputs, scores and certificates, even for the same episode.
    backend.patch(episode, 1, "B0", "B4", 1, "H")
    assert backend.model.calls == 23 and len(preparations) == 4
    backend.patch({"images": {}}, 1, "B0", "B4", 1, "H")
    assert backend.model.calls == 29 and len(preparations) == 6
    backend.close()
    assert not backend._prepared_cache and not backend._native_cache and not backend._full_checks


def test_cache_scope_includes_file_identity_without_consulting_labels(monkeypatch):
    backend = rt.HFBackend.__new__(rt.HFBackend)
    keys = ("past1", "past2", "current", "candidate0", "candidate1")
    episode = {"episode_id": "fixture", "images": {key: {"path": "/staged/" + key} for key in keys}}
    stat = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4, st_ctime_ns=5)
    monkeypatch.setattr(Path, "stat", lambda self: stat)
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    key = backend._scope_key(episode, 0)
    assert key != backend._scope_key(episode, 1)
    assert key != backend._scope_key({**episode, "episode_id": "different"}, 0)
    for field in ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
        setattr(stat, field, getattr(stat, field) + 1)
        updated = backend._scope_key(episode, 0)
        assert updated != key
        key = updated


def test_cpu_returned_rows_cannot_mutate_cached_certificates_or_positions():
    backend = cpu_backend()
    episode = {"images": {}}
    result = backend.patch(episode, 0, "B0", "B4", 0, "H")
    result["positions"]["A"].clear()
    result["controls"]["full_A"]["scores"]["margin_ab_bits"] = 999
    result["controls"]["recipient_self"]["passed"] = False
    again = backend.patch(episode, 0, "B0", "B4", 0, "H")
    assert again["margin_ab_bits"] == result["margin_ab_bits"]
    assert again["controls"]["recipient_self"]["passed"]
    assert len(again["positions"]["A"]) == 8


def test_cpu_cached_native_drift_fails_before_selected_patch():
    backend = cpu_backend()
    episode = {"images": {}}
    backend.patch(episode, 0, "B0", "B4", 0, "H")
    backend._native_cache["B0"]["margin_ab_bits"] += 1
    with pytest.raises(rt.GateError, match="cached native score drift"):
        backend.patch(episode, 0, "B0", "B4", 0, "H")
    assert backend.model.calls == 7  # Fresh donor capture still runs; selected patch does not.
    assert all(not block._forward_hooks for block in backend.layers)


def test_cpu_forward_rejects_supplied_or_returned_recurrent_state():
    backend = cpu_backend()
    prepared = backend._prepare({"images": {}}, 0, "B0")
    prepared["inputs"]["past_key_values"] = object()
    with pytest.raises(rt.GateError, match="unexpected labels/cache"):
        backend._forward(prepared)
    assert backend.model.calls == 0
    prepared["inputs"].pop("past_key_values")
    original = backend.model.forward

    def forward(*args, **kwargs):
        output = original(*args, **kwargs)
        output.past_key_values = object()
        return output

    backend.model.forward = forward
    with pytest.raises(rt.GateError, match="unexpected KV/recurrent cache"):
        backend._forward(prepared)


def test_cpu_exception_removes_capture_and_patch_hooks():
    backend = cpu_backend()
    prepared = backend._prepare({"images": {}}, 0, "B0")
    with pytest.raises(RuntimeError, match="intentional"):
        with backend._capture(0, 8):
            backend._forward(prepared)
            raise RuntimeError("intentional")
    assert all(not block._forward_hooks for block in backend.layers)
    with pytest.raises(rt.GateError, match="geometry/dtype/device"):
        backend._patched(prepared, 0, backend.torch.zeros(1, 7, 2), [0])
    assert all(not block._forward_hooks for block in backend.layers)


@pytest.mark.parametrize("delta,passed", [(0.0009, True), (0.0011, False), (float("nan"), False)])
def test_controls_measure_absolute_not_relative_drift(delta, passed):
    native = {"a_logprob": -1.0, "b_logprob": -2.0, "margin_ab_bits": 10.0}
    patched = {**native, "margin_ab_bits": 10.0 + delta}
    if passed:
        assert rt.HFBackend._control(native, patched, "test")["passed"]
    else:
        with pytest.raises(rt.GateError):
            rt.HFBackend._control(native, patched, "test")
