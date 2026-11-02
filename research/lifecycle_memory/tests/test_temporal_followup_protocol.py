"""Follow-up inventory and gates keep original data and old studies immutable."""

from collections import Counter
from copy import deepcopy
import time

import pytest

from research.lifecycle_memory.temporal_forecasting import protocol as p
from research.lifecycle_memory.temporal_forecasting import followup_protocol as f
from research.lifecycle_memory.temporal_forecasting import execution as e
from research.lifecycle_memory.temporal_forecasting.common import atomic_json, digest
from research.lifecycle_memory.tests.test_temporal_recognition_protocol import make_protocol, public_rows


def design():
    return f.followup_protocol(make_protocol(), **{key: "a" * 64 for key in f.PIN_FIELDS})


def test_unchanged_input_contract_and_complete_new_inventory():
    source, followup = make_protocol(), design()
    rows = public_rows(source)
    assert p.protocol_contract(source) == p.protocol_contract(followup)
    assert p.validate_protocol(followup) == followup
    for phase, n in (("recognition", 8), ("recognition-canary", 4)):
        tasks = p.inventory(rows, followup, phase)
        assert len(tasks) == n * 78
        assert len({t["task_id"] for t in tasks}) == len(tasks)
        assert Counter(t["kind"] for t in tasks) == {"behavior": 10*n, "patch": 64*n, "self": 4*n}
        assert {t["cut"] for t in tasks if "cut" in t} == {34}
        assert {t["position_set"] for t in tasks if t["kind"] == "patch"} == {*f.SUBSETS, "A"}
        assert set(t["task_id"] for t in tasks).isdisjoint(
            t["task_id"] for t in p.inventory(rows, source, phase))
    assert len(p.inventory(rows, source, "recognition-canary")) == 176


@pytest.mark.parametrize("change", [
    {"schema_version": 5.0}, {"study": "temporal-recognition-v1"},
    {"fixed_cuts": [22, 34]}, {"conditions": ["B0", "S1"]}, {"followup": {}},
    {"unknown": True}, {"source_result_set_sha256": "bad"}, {"bootstrap_draws": 100},
])
def test_followup_drift_rejected(change):
    with pytest.raises(ValueError):
        p.validate_protocol({**design(), **change})


def test_second_generation_source_and_dataset_drift_rejected():
    with pytest.raises(ValueError):
        f.followup_protocol(design(), **{key: "a" * 64 for key in f.PIN_FIELDS})
    f.verify_followup_dataset(design(), "a" * 64)
    with pytest.raises(ValueError, match="dataset"):
        f.verify_followup_dataset(design(), "b" * 64)


def test_permutations_are_frozen_distinct_and_endpoint_matched():
    for task in f.TASKS:
        for episode in ("episode-1", "episode-2"):
            permutations = [f.scene_permutation(task, episode, c) for c in ("S1", "S2")]
            assert permutations[0] != permutations[1]
            for condition, perm in zip(("S1", "S2"), permutations):
                assert sorted(perm) == list(range(17)) and perm != list(range(17))
                assert perm[0] == 0 and perm[-1] == 16
                assert perm == f.scene_permutation(task, episode, condition)
            assert f.scene_permutation(task, episode, "RV") == list(reversed(range(17)))
            assert f.scene_permutation(task, episode, "B4") == [16] * 17


def test_worker_requires_matching_sealed_protocol_before_backend(tmp_path):
    followup = design()
    context = {"run_dir": str(tmp_path), "identity": {"recognition_suite": f.SUITE,
        "protocol_sha256": digest(followup), "simulation": True}, "contract": p.protocol_contract(followup),
        "followup_protocol": followup, "assignments": [[]], "episodes": [],
        "deadline_epoch": time.time()+30}
    path = tmp_path / "context.json"
    atomic_json(path, context)
    assert e.run_worker(path, 0, backend_factory=lambda *a, **k: None)["status"] == "pass"
    for mutation in (None, {**followup, "seed": 1}, {**followup, "design_sha256": "b"*64}):
        changed = {**context, "followup_protocol": mutation}
        atomic_json(path, changed)
        with pytest.raises(ValueError):
            e.run_worker(path, 0, backend_factory=lambda *a, **k: None)


def test_canary_count_is_suite_bound():
    identity = {key: "a" * 64 for key in ("release_sha256", "public_sha256", "protocol_sha256",
        "evaluator_sha256", "dataset_manifest_sha256", "runtime_identity", "runtime_spec_sha256")}
    identity.update(simulation=False, model_inventories={p.GEMMA: "a"*64}, recognition_suite=f.SUITE)
    receipt = {"status": "pass", "stage": "recognition-canary", "identity": identity,
        "tasks": 312, "technical_checks_sha256": "a"*64, "task_inventory_sha256": "a"*64,
        "result_set_sha256": "a"*64, "staging_s": 1., "elapsed_s": 10.,
        "task_class_mean_s": {f"{p.GEMMA}:{k}": 1. for k in ("behavior", "patch", "self")}}
    e.canary_admission(receipt, identity, [], 4, 82800)
    with pytest.raises(ValueError, match="technical receipt"):
        e.canary_admission({**receipt, "tasks": 176}, identity, [], 4, 82800)
    wrong = deepcopy(identity)
    wrong.pop("recognition_suite")
    with pytest.raises(ValueError, match="recognition_suite"):
        e.canary_admission(receipt, wrong, [], 4, 82800)
