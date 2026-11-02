"""Recognition contracts never reinterpret legacy tasks or select on outcomes."""

from collections import Counter
from copy import deepcopy

import pytest

from research.lifecycle_memory.temporal_forecasting import protocol as p
from research.lifecycle_memory.temporal_forecasting import recognition_protocol as r
from research.lifecycle_memory.temporal_forecasting.common import digest


POLICY = {"version": 1, "kind": "matched_static_reuse",
          "historical_static_sha256": "a" * 64, "temporal_exposure_sha256": "b" * 64}


def make_protocol(task=r.NEW_INTERACTION, count=8, minimum=4):
    return r.recognition_protocol(task, POLICY, episode_limit=count, minimum_clusters=minimum)


def public_rows(protocol):
    contract = p.protocol_contract(protocol)
    return [{"episode_id": f"episode-{i}", "cluster_id": f"wearer-{i // 2}",
             "task": protocol["task"], "contract_sha256": digest(contract)}
            for i in range(protocol["episode_limit"])]


@pytest.mark.parametrize("task", r.TASKS)
def test_factory_dispatch_and_exact_contract(task):
    protocol = make_protocol(task)
    contract = p.protocol_contract(protocol)
    assert contract == {"version": 3, "task": task, "prompt": r.PROMPTS[task],
                        "exposure_policy": POLICY, "scene_count": 17, "patching": r.PATCHING}
    assert p.validate_contract(contract) is contract
    assert p.validate_protocol(protocol) is protocol
    assert p.recognition_protocol(task, POLICY, episode_limit=8) == protocol
    assert p.is_recognition(protocol) and p.is_recognition(contract)
    assert not p.is_dense_current(protocol) and not p.is_dense_current(contract)
    assert protocol["models"] == {p.GEMMA: p.MODELS[p.GEMMA]}
    assert protocol["seed"] == 20261025
    assert protocol["bootstrap_draws"] == 19999
    assert protocol["fixed_cuts"] == [22, 34]
    assert protocol["conditions"] == ["B0", "B1", "B4", "B2"]
    assert protocol["primary_family"] == "descriptive_only" and protocol["alpha"] is None
    assert "not a future interaction" in contract["prompt"]
    assert contract["prompt"].endswith("Answer with only A or B.")
    protocol["exposure_policy"]["temporal_exposure_sha256"] = "c" * 64
    protocol["models"][p.GEMMA]["layers"] = 1
    assert POLICY["temporal_exposure_sha256"] == "b" * 64
    assert p.MODELS[p.GEMMA]["layers"] == 60


@pytest.mark.parametrize("task", r.TASKS)
@pytest.mark.parametrize("version", [1, 2])
def test_new_tasks_never_enter_old_contracts(task, version):
    contract = p.protocol_contract(p.dense_current_protocol(POLICY) if version == 2
                                   else p.matched_protocol(p.CURRENT, POLICY))
    contract.update(task=task, prompt=r.PROMPTS[task])
    with pytest.raises(ValueError):
        p.validate_contract(contract)
    with pytest.raises(ValueError):
        p.matched_protocol(task, POLICY)


def test_legacy_factories_keep_frozen_hashes_and_dense_conditions():
    assert digest(p.default_protocol()) == "1a54d3216e5d12ba89787def9187a3436a48951aca7994ed51d60ee63c742410"
    for task, checksum in ((p.CURRENT, "b620cd756d3024c6dd865da13092620ae525028c85d8e83c1f73d4dbfcdc8e4f"),
                           (p.FORECAST, "feba574c04191dca924e876bdb0864536175c0983e1f23f06190aa78a789c809")):
        protocol = p.matched_protocol(task, POLICY)
        assert digest(protocol) == checksum
        assert not p.is_recognition(protocol)
    assert p.DENSE_CONDITIONS == ("B0", "B1", "B4", "B8")
    assert digest(p.dense_current_protocol(POLICY)) == "75734d0b7b638174ba7e3636b906ed758522633318721316f0bef6c5eb6c4508"
    assert not p.is_recognition(p.dense_current_protocol(POLICY))
    assert set(p.PROMPTS) == {p.CURRENT, p.FORECAST}


@pytest.mark.parametrize("task", [None, [], p.CURRENT])
def test_factory_rejects_unsupported_tasks(task):
    with pytest.raises(ValueError, match="unsupported recognition task"):
        make_protocol(task)


@pytest.mark.parametrize("change", [
    {"version": 3.0}, {"version": True}, {"task": p.CURRENT}, {"task": []},
    {"prompt": "choose A"}, {"scene_count": 16}, {"scene_count": 17.0},
    {"patching": p.DENSE_PATCHING}, {"unknown": 1},
    {"exposure_policy": {**POLICY, "temporal_exposure_sha256": "bad"}},
    {"exposure_policy": {**POLICY, "kind": "new"}},
])
def test_contract_drift_rejected(change):
    contract = {**p.protocol_contract(make_protocol()), **change}
    with pytest.raises(ValueError):
        p.validate_contract(contract)
    assert not p.is_recognition(contract)


@pytest.mark.parametrize("change", [
    {"schema_version": 4.0}, {"study": p.DENSE_STUDY}, {"models": p.MODELS},
    {"attention_backend": "eager"}, {"dtype": "float32"}, {"seed": 20261023},
    {"seed": 20261025.0}, {"fixed_cuts": [22.0, 34]}, {"fixed_cuts": [22, 35]},
    {"conditions": ["B0", "B1", "B4", "B8"]}, {"cut_selection": "selected"},
    {"episode_limit": 3}, {"episode_limit": 97}, {"episode_limit": 8.0},
    {"minimum_clusters": 3}, {"minimum_clusters": 9}, {"minimum_clusters": True},
    {"max_episodes_per_cluster": 4}, {"canary_limit": 3}, {"bootstrap_draws": 1},
    {"candidate_policy": "preserved static"}, {"primary_models": [p.GEMMA]},
    {"alpha": .05}, {"pruning_enabled": 0}, {"unknown": True},
])
def test_protocol_drift_rejected(change):
    with pytest.raises(ValueError):
        p.validate_protocol({**make_protocol(), **change})


@pytest.mark.parametrize("count", [4, 96])
def test_limit_bounds_are_supported(count):
    assert make_protocol(count=count)["episode_limit"] == count


@pytest.mark.parametrize("draws", [2, 99, 19998, 20000, 19999.0, True, None])
def test_production_bootstrap_draws_are_frozen(draws):
    with pytest.raises(ValueError, match="bootstrap_draws"):
        p.validate_protocol({**make_protocol(), "bootstrap_draws": draws})


@pytest.mark.parametrize("count", [4, 8])
def test_analysis_episode_ids_bind_full_or_canary_scope(count):
    protocol = make_protocol()
    protocol["episode_ids"] = [f"episode-{i}" for i in range(count)]
    assert p.validate_protocol(protocol) is protocol


@pytest.mark.parametrize("ids", [[], ["one"], [f"episode-{i}" for i in range(5)],
                                 [f"episode-{i}" for i in range(9)], ["duplicate"] * 4,
                                 ["a", "b", "c", " "], ("a", "b", "c", "d")])
def test_analysis_episode_ids_reject_arbitrary_subset_or_invalid_identity(ids):
    with pytest.raises(ValueError, match="analysis episode IDs"):
        p.validate_protocol({**make_protocol(), "episode_ids": ids})


@pytest.mark.parametrize("task", r.TASKS)
def test_selection_uses_entire_pool_and_four_distinct_canary_wearers(task):
    protocol = make_protocol(task)
    rows = public_rows(protocol)
    selected = p.selected_episodes(rows, protocol, "recognition")
    assert len(selected) == 8 and {r["episode_id"] for r in selected} == {r["episode_id"] for r in rows}
    assert p.selected_episodes(list(reversed(rows)), protocol, "recognition") == selected
    canary = p.selected_episodes(rows, protocol, "recognition-canary")
    assert len(canary) == len({r["cluster_id"] for r in canary}) == 4
    enriched = [{**row, "correct_candidate": i % 2} for i, row in enumerate(rows)]
    assert [r["episode_id"] for r in p.selected_episodes(enriched, protocol, "recognition-canary")] == [r["episode_id"] for r in canary]


@pytest.mark.parametrize("stage", ["recognition", "recognition-canary"])
@pytest.mark.parametrize("bad", ["missing", "extra", "duplicate", "cluster_count", "cluster_cap", "identity", "task", "hash"])
def test_selection_fails_closed_on_bad_published_pool(stage, bad):
    protocol = make_protocol()
    rows = public_rows(protocol)
    if bad == "missing":
        rows.pop()
    elif bad == "extra":
        rows.append({**rows[-1], "episode_id": "extra"})
    elif bad == "duplicate":
        rows[-1] = rows[0]
    elif bad == "cluster_count":
        rows[-1]["cluster_id"] = rows[-2]["cluster_id"] = rows[0]["cluster_id"]
    elif bad == "cluster_cap":
        rows[2]["cluster_id"] = rows[4]["cluster_id"] = rows[0]["cluster_id"]
    elif bad == "identity":
        rows[0]["cluster_id"] = " "
    elif bad == "task":
        rows[0]["task"] = r.INTERACTION_ORDER
    else:
        rows[0]["contract_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        p.selected_episodes(rows, protocol, stage)


def test_three_per_wearer_allowed_and_minimum_is_enforced():
    protocol = make_protocol(count=12)
    rows = public_rows(protocol)
    for i, row in enumerate(rows):
        row["cluster_id"] = f"wearer-{i // 3}"
    assert len(p.selected_episodes(rows, protocol, "recognition")) == 12
    protocol["minimum_clusters"] = 5
    with pytest.raises(ValueError, match="cluster coverage"):
        p.selected_episodes(rows, protocol, "recognition")


@pytest.mark.parametrize("task", r.TASKS)
@pytest.mark.parametrize("stage,count", [("recognition", 8), ("recognition-canary", 4)])
def test_inventory_exact_44_cells_and_hash_bindings(task, stage, count):
    protocol = make_protocol(task)
    rows = public_rows(protocol)
    tasks = p.inventory(rows, protocol, stage)
    assert len(tasks) == len({t["task_id"] for t in tasks}) == 44 * count
    assert Counter(t["kind"] for t in tasks) == {"behavior": 8 * count, "patch": 32 * count, "self": 4 * count}
    for episode in {t["episode_id"] for t in tasks}:
        for order in (0, 1):
            block = [t for t in tasks if t["episode_id"] == episode and t["order"] == order]
            assert {t["condition"] for t in block if t["kind"] == "behavior"} == set(r.CONDITIONS)
            assert {(t["cut"], t["donor_condition"], t["recipient_condition"], t["position_set"])
                    for t in block if t["kind"] == "patch"} == {
                        (cut, donor, recipient, group) for cut in (22, 34)
                        for donor, recipient in (("B0", "B4"), ("B4", "B0")) for group in ("H", "V", "K", "A")}
            assert {(t["cut"], t["donor_condition"], t["recipient_condition"], t["position_set"])
                    for t in block if t["kind"] == "self"} == {(22, "B0", "B0", "A"), (34, "B0", "B0", "A")}
    for row in tasks:
        assert row["model_key"] == p.GEMMA and row["task"] == task
        assert row["contract_sha256"] == digest(p.protocol_contract(protocol))
        assert row["task_id"] == digest({"protocol": protocol, "task": {k: v for k, v in row.items() if k != "task_id"}})
    for workers in (1, 2, 8):
        assigned = [t["task_id"] for shard in p.assign_tasks(tasks, workers) for t in shard]
        assert len(assigned) == len(set(assigned)) == len(tasks)


def test_cross_task_identity_and_legacy_stage_rejections():
    first, second = (make_protocol(task) for task in r.TASKS)
    a, b = (p.inventory(public_rows(proto), proto, "recognition") for proto in (first, second))
    assert not {r["task_id"] for r in a} & {r["task_id"] for r in b}
    with pytest.raises(ValueError):
        p.inventory(public_rows(first), second, "recognition")
    with pytest.raises(ValueError):
        p.validate_dataset_protocol({"contract": p.protocol_contract(first)}, second)
    for stage in ("current", "pilot", "confirmation"):
        with pytest.raises(ValueError):
            p.inventory(public_rows(first), first, stage)
    with pytest.raises(ValueError, match="fixed cuts"):
        p.inventory(public_rows(first), first, "recognition", freeze={})
    for protocol in (p.default_protocol(), p.matched_protocol(p.CURRENT, POLICY), p.dense_current_protocol(POLICY)):
        with pytest.raises(ValueError):
            p.selected_episodes([], protocol, "recognition")
