"""Recognition geometry and tiny CPU-only tests; no real model/media access."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from research.lifecycle_memory.temporal_forecasting import protocol as p, runtime as rt
from research.lifecycle_memory.temporal_forecasting import recognition_protocol as r
from research.lifecycle_memory.temporal_forecasting.common import digest
from research.lifecycle_memory.tests.test_temporal_forecasting_runtime import FakeProcessor, cpu_backend
from research.lifecycle_memory.tests.test_temporal_recognition_protocol import make_protocol, public_rows


def contract(task=r.NEW_INTERACTION):
    return p.protocol_contract(make_protocol(task))


@pytest.mark.parametrize("task", r.TASKS)
@pytest.mark.parametrize("order", [0, 1])
def test_exact_slot_sources_and_counterbalancing(task, order):
    spec = contract(task)
    candidates = (("CA", f"candidate{order}"), ("CB", f"candidate{1 - order}"))
    assert rt.slot_plan("B0", order, spec) == (*zip(r.HISTORY_ROLES, r.HISTORY_KEYS), ("V", "current"), *candidates)
    assert rt.slot_plan("B4", order, spec) == (*((role, "current") for role in r.HISTORY_ROLES), ("V", "current"), *candidates)
    assert rt.slot_plan("B1", order, spec) == (("V", "current"), *candidates)
    assert rt.slot_plan("B2", order, spec) == (("H1", "past1"), *candidates)
    for invalid in ("B3", "B5", "B6", "B7", "B8", "B9", "reverse"):
        with pytest.raises(rt.GateError):
            rt.slot_plan(invalid, order, spec)
    for invalid_order in (True, -1, 2, .0):
        with pytest.raises(rt.GateError):
            rt.slot_plan("B0", invalid_order, spec)


def prepared(monkeypatch, task=r.NEW_INTERACTION, condition="B0"):
    spec = contract(task)
    monkeypatch.setattr(rt, "QUESTION", spec["prompt"])
    roles = [role for role, _ in rt.slot_plan(condition, 0, spec)]
    processor = FakeProcessor("gemma31", len(roles))
    return rt.prepare_processor_inputs(processor, [], roles, p.GEMMA, contract=spec)


@pytest.mark.parametrize("task", r.TASKS)
@pytest.mark.parametrize("condition", r.CONDITIONS)
def test_native_processor_exhaustive_partition(monkeypatch, task, condition):
    value = prepared(monkeypatch, task, condition)
    groups = value["positions"]
    assert len(value["slot_roles"]) == (19 if condition in ("B0", "B4") else 3)
    assert groups["H"] == sorted(pos for role in r.HISTORY_ROLES for pos in groups[role])
    assert groups["K"] == sorted(groups["C"] + groups["Q"] + groups["P"] + groups["R"])
    assert groups["A"] == sorted(groups["H"] + groups["V"] + groups["K"])
    assert not set(groups["H"]) & set(groups["V"] + groups["K"])
    assert groups["P"][-1] == value["readout_position"]
    assert bool(groups["H"]) == (condition != "B1")
    assert bool(groups["V"]) == (condition != "B2")
    assert value["answer_ids"]["A"] != value["answer_ids"]["B"]


@pytest.mark.parametrize("bad", ["role_order", "missing", "duplicate", "extra", "qwen", "pixels", "coordinates", "kernel", "legacy"])
def test_image_geometry_fails_closed(monkeypatch, bad):
    value = prepared(monkeypatch)
    roles, inputs, spec, model, kernel = value["slot_roles"], value["inputs"], contract(), p.GEMMA, 3
    if bad == "role_order":
        roles[0], roles[1] = roles[1], roles[0]
    elif bad == "missing":
        roles.pop(15)
    elif bad == "duplicate":
        roles[15] = roles[14]
    elif bad == "extra":
        roles[15] = "H17"
    elif bad == "qwen":
        model = p.QWEN
    elif bad == "pixels":
        inputs["pixel_values"].shape = (11, 171, 12)
    elif bad == "coordinates":
        inputs["image_position_ids"].values[15][1] = inputs["image_position_ids"].values[15][0]
    elif bad == "kernel":
        kernel = None
    else:
        spec = None
    with pytest.raises(rt.GateError):
        rt.image_slot_positions(inputs, model, roles, pooling_kernel=kernel, contract=spec)


@pytest.mark.parametrize("bad", ["overlap", "inactive", "empty", "role", "duplicate", "readout"])
def test_partition_rejects_invalid_boundaries(bad):
    ids, mask = list(range(23)), [1] * 23
    slots = {role: [i] for i, role in enumerate((*r.HISTORY_ROLES, "V", "CA", "CB"))}
    if bad == "overlap":
        slots["H16"] = [19]
    elif bad == "inactive":
        mask[15] = 0
    elif bad == "empty":
        slots["H16"] = []
    elif bad == "role":
        slots["H17"] = slots.pop("H16")
    elif bad == "duplicate":
        slots["H16"] = [15, 15]
    else:
        mask[-1] = 0
    with pytest.raises(rt.GateError):
        rt.partition_positions(ids, mask, slots, [19, 20], [22], contract=contract())


def test_scope_binds_all_nineteen_files_order_task_and_contract(monkeypatch):
    backend = rt.HFBackend.__new__(rt.HFBackend)
    backend.contract = contract()
    episode = {"episode_id": "fixture", "images": {key: {"path": f"/staged/{key}"} for key in r.SLOTS}}
    stats = {key: SimpleNamespace(st_dev=1, st_ino=i, st_size=3, st_mtime_ns=4, st_ctime_ns=5)
             for i, key in enumerate(r.SLOTS)}
    monkeypatch.setattr(Path, "stat", lambda path: stats[path.name])
    monkeypatch.setattr(Path, "resolve", lambda path: path)
    original = backend._scope_key(episode, 0)
    assert [entry[0] for entry in original[2]] == list(r.SLOTS)
    assert original != backend._scope_key(episode, 1)
    for key in r.SLOTS:
        stats[key].st_mtime_ns += 1
        changed = backend._scope_key(episode, 0)
        assert changed != original
        original = changed
    backend.contract = contract(r.INTERACTION_ORDER)
    assert original != backend._scope_key(episode, 0)
    original = backend._scope_key(episode, 0)
    backend.contract["exposure_policy"]["temporal_exposure_sha256"] = "c" * 64
    assert original != backend._scope_key(episode, 0)


@pytest.mark.parametrize("task", r.TASKS)
@pytest.mark.parametrize("condition", r.CONDITIONS)
@pytest.mark.parametrize("order", [0, 1])
def test_prepare_passes_exact_pixels_prompt_and_roles(monkeypatch, task, condition, order):
    backend = rt.HFBackend.__new__(rt.HFBackend)
    backend.contract, backend.model_key = contract(task), p.GEMMA
    backend.processor, backend.merge_size = object(), None
    episode = {**public_rows(make_protocol(task))[0], "images": {key: {"path": f"/staged/{key}"} for key in r.SLOTS}}
    images = []

    class Image:
        def __init__(self, key):
            self.key, self.closed = key, False
            self.size = rt.CANDIDATE_IMAGE_SIZE if key.startswith("candidate") else rt.SCENE_IMAGE_SIZE
            images.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def convert(self, mode):
            assert mode == "RGB"
            return Image(self.key)

        def close(self):
            self.closed = True

    def processor_inputs(processor, messages, roles, model_key, **kwargs):
        plan = rt.slot_plan(condition, order, backend.contract)
        assert processor is backend.processor and model_key == p.GEMMA
        assert roles == [role for role, _ in plan]
        assert [item["image"].key for item in messages[0]["content"][:-1]] == [key for _, key in plan]
        assert messages[0]["content"][-1] == {"type": "text", "text": r.PROMPTS[task]}
        assert kwargs["contract"] == backend.contract and kwargs["question"] == r.PROMPTS[task]
        return {"inputs": {}}

    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=SimpleNamespace(open=lambda path: Image(path.name))))
    monkeypatch.setattr(rt, "prepare_processor_inputs", processor_inputs)
    assert backend._prepare(episode, order, condition) == {"inputs": {}}
    assert len(images) == 2 * (19 if condition in ("B0", "B4") else 3)
    assert all(image.closed for image in images)


def test_loader_rejects_wrong_model_or_attention_before_loading():
    with pytest.raises(rt.GateError, match="Gemma31 only"):
        rt.HFBackend(p.QWEN, "/not-a-model", attention_backend="sdpa", contract=contract())
    with pytest.raises(rt.GateError, match="SDPA"):
        rt.HFBackend(p.GEMMA, "/not-a-model", attention_backend="eager", contract=contract())


@pytest.mark.parametrize("donor,recipient,cut,group,self_patch", [
    ("B0", "B4", 20, "H", False), ("B0", "B4", 22.0, "H", False),
    ("B0", "B4", 22, "C+Q", False), ("B0", "B2", 22, "H", False),
    ("B0", "B1", 22, "H", False), ("B0", "B8", 22, "H", False),
    ("B0", "B0", 22, "A", False), ("B4", "B4", 22, "A", True),
    ("B0", "B0", 22, "H", True), ("B0", "B4", 22, "A", True),
])
def test_noninventory_interventions_rejected(donor, recipient, cut, group, self_patch):
    backend = rt.HFBackend.__new__(rt.HFBackend)
    backend.contract, backend.model_key, backend.layers = contract(), p.GEMMA, [None] * 60
    with pytest.raises(rt.GateError):
        backend.patch({}, 0, donor, recipient, cut, group, self_patch=self_patch)


def recognition_cpu_backend(task, bypass=False):
    backend = cpu_backend(bypass=bypass)
    torch = backend.torch
    backend.model.layers = torch.nn.ModuleList([type(backend.layers[0])() for _ in range(60)])
    backend.layers, backend.contract = backend.model.layers, contract(task)

    def prepare(episode, order, condition):
        roles = [role for role, _ in rt.slot_plan(condition, order, backend.contract)]
        ids, mask = list(range(23)), [1] * 23
        slots = {role: [i] for i, role in enumerate(roles)}
        return {"inputs": {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask]),
                           "pixel_values": torch.tensor([1. if condition == "B0" else 0.])},
                "positions": rt.partition_positions(ids, mask, slots, [19, 20], [22], contract=backend.contract),
                "answer_ids": {"A": 0, "B": 1}, "slot_roles": roles, "readout_position": 22}

    backend._prepare = prepare
    return backend


@pytest.mark.parametrize("task", r.TASKS)
@pytest.mark.parametrize("cut", [22, 34])
@pytest.mark.parametrize("order", [0, 1])
def test_cpu_patches_controls_and_bounded_cache(task, cut, order):
    backend = recognition_cpu_backend(task)
    episode = {**public_rows(make_protocol(task))[0], "images": {}}
    for donor, recipient in (("B0", "B4"), ("B4", "B0")):
        for group in ("H", "V", "K", "A"):
            result = backend.patch(episode, order, donor, recipient, cut, group)
            technical = result["technical"]
            assert technical["self_patch_pass"] and technical["full_A_pass"]
            assert technical["selected_rows_equal"] and technical["preserved_rows_equal"]
            assert technical["self_max_drift_bits"] == technical["full_closure_max_drift_bits"] == 0
            assert technical["full_forward_reference"] and not technical["suffix_replay"]
            assert result["task"] == task and result["contract_sha256"] == digest(backend.contract)
            assert all(not block._forward_hooks for block in backend.layers)
    result = backend.patch(episode, order, "B0", "B0", cut, "A", self_patch=True)
    assert result["margin_ab_bits"] == result["recipient_margin_ab_bits"]
    assert result["controls"]["selected_self"]["passed"]
    assert set(backend._prepared_cache) == {"B0", "B4"}
    assert all(not backend.torch.is_tensor(value) for entry in backend._native_cache.values() for value in entry.values())


def test_cpu_full_closure_failure_blocks_small_patch_and_cleans_hooks():
    backend = recognition_cpu_backend(r.NEW_INTERACTION, bypass=True)
    episode = {**public_rows(make_protocol())[0], "images": {}}
    with pytest.raises(rt.GateError, match="full A transfer failed"):
        backend.patch(episode, 0, "B0", "B4", 22, "H")
    assert backend.model.calls == 5
    assert all(not block._forward_hooks for block in backend.layers)


@pytest.mark.parametrize("task", r.TASKS)
def test_cpu_behavior_baselines_clear_cache_and_bind_task(task):
    backend = recognition_cpu_backend(task)
    episode = {**public_rows(make_protocol(task))[0], "images": {}}
    for order in (0, 1):
        for condition in r.CONDITIONS:
            result = backend.score(episode, order, condition)
            assert result["task"] == task
            assert result["contract_sha256"] == digest(backend.contract)
            if condition in ("B1", "B2"):
                assert backend._prepared_cache == backend._native_cache == {}
            assert all(not block._forward_hooks for block in backend.layers)


def test_wrong_row_contract_rejected_before_forward():
    backend = recognition_cpu_backend(r.NEW_INTERACTION)
    episode = {**public_rows(make_protocol(r.INTERACTION_ORDER))[0], "images": {}}
    with pytest.raises(ValueError, match="contract mismatch"):
        backend.score(episode, 0, "B0")
    assert backend.model.calls == 0
