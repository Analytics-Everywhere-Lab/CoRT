from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.lifecycle_memory import qwen_role_factorial as q


def config():
    return {"schema_version": 1, "kind": "qwen_role_factorial", "model": q.MODEL,
        "post_block_index": 36, "prompt": q.old.EXPECTED_PROMPT,
        "public_requests": {"path": "public.jsonl", "sha256": q.PUBLIC_SHA256},
        "technical_gates": {"self_patch_max_abs_error_bits": .001,
            "full_sequence_closure_max_abs_error_bits": .001,
            "answer_token_equivalence_tolerance_bits": .001},
        "analysis": {"bootstrap_draws": 200, "seed": 42, "min_donor_gap_bits": 1e-12,
            "min_k_recovery": .1, "min_fraction_denominator": .1},
        "provenance": {"sif_sha256": "a" * 64, "slurm_script_sha256": "b" * 64}}


def cohort():
    public, hidden = [], []
    for frame in range(46):
        for order in range(2):
            row = {"request_id": f"request_{frame:02d}_{order}", "frame_unit_id": f"frame_{frame:02d}",
                   "order_id": f"order_{order}", "counterbalanced_request_id": f"request_{frame:02d}_{1-order}"}
            public.append(row)
            hidden.append({**row, "correct_choice": "AB"[order],
                           "participant_cluster_id": f"cluster_{frame:02d}", "ego4d_participant_id": f"person_{frame:02d}"})
    return public, hidden


def partition():
    return q.old.build_component_partition(sequence_length=926, active_positions=tuple(range(926)),
        image_slot_positions=(tuple(range(48, 636)), tuple(range(638, 738)), tuple(range(740, 840))),
        literal_user_prompt_start=841, literal_user_prompt_count=73, answer_prefix_token_count=3)


def request_rows(request, rank, gap=1., recovery=.8):
    rows = []
    part = partition()
    for corruption in q.old.CORRUPTIONS:
        for direction in ("rescue", "damage"):
            donor, recipient = (gap, 0.) if direction == "rescue" else (0., gap)
            for operator in q.OPERATORS:
                positions, receipt = q.old._operator_positions(operator=operator, model_key=q.MODEL_KEY,
                    layer=36, partition=part, prespec=q.make_prespec(config()))
                rho = 1. if operator == "A" else recovery
                patched = donor + (1-rho) * (recipient-donor)
                rows.append({"kind": "patchaudit_component_closure_outcome", "output_schema_version": 1,
                    "model_key": q.MODEL_KEY, "rank": rank, **{k: request[k] for k in ("request_id", "frame_unit_id", "order_id")},
                    "corruption": corruption, "direction": direction, "start_layer": 36, "operator": operator,
                    "source_arm": "real" if direction == "rescue" else corruption,
                    "recipient_arm": corruption if direction == "rescue" else "real",
                    "real_margin_a_minus_b_bits": gap, "corrupt_margin_a_minus_b_bits": 0.,
                    "recipient_margin_a_minus_b_bits": recipient, "donor_margin_a_minus_b_bits": donor,
                    "patched_margin_a_minus_b_bits": patched, "baseline_donor_gap_abs_bits": gap,
                    "patched_donor_error_abs_bits": abs(patched-donor),
                    "operator_position_count": len(positions), "operator_positions_sha256": q.sha256_json(list(positions)),
                    "operator_receipt": receipt,
                    "partition": {"sequence_length": 926, "active_count": 926, "full_frame_count": 588,
                        "complement_count": 338, "padding_count": 0, "active_sha256": part.active.sha256,
                        "full_frame_sha256": part.full_frame.sha256, "complement_sha256": part.complement.sha256,
                        "component_partition_sha256": q.sha256_json(part.as_dict())},
                    "audit": {"status": "pass", "all_fired_once": True, "all_removed": True, "all_finite": True}})
    return rows


def bundle(request, rank, mode="canary", binding=None):
    binding = binding or {"test": True}
    checks = [{"operator": op, "start_layer": 36, "status": "pass", "margin_error_bits": 0.,
               "audit": {"status": "pass", "all_fired_once": True, "all_removed": True, "all_finite": True}} for op in q.OPERATORS]
    return {"binding_sha256": q.sha256_json(binding), "request_id": request["request_id"], "mode": mode,
        "rank": rank, "rows": request_rows(request, rank),
        "detail": {"partition": partition().as_dict(), "self_checks": checks if mode == "canary" else []},
        "answer_token_equivalence": {"status": "pass", "maximum_absolute_difference_bits": 0.,
            "construction": "one_shared_prefix_forward_vs_exact_appended_suffix", "label_joined": False,
            "choices": {choice: {"one_pass_logprob": -1., "suffix_logprob": -1.,
                "absolute_difference_bits": 0., "answer_token_ids": [token]}
                for choice, token in (("A", 32), ("B", 33))}}}


def write_run(root, mode="canary", binding=None):
    binding = binding or {"test": True}
    public, _ = cohort()
    for rank in range(4):
        selected = q.shard_requests(public, rank, mode=mode)
        directory = root / mode / f"rank_{rank:03d}"
        hashes = {}
        for request in selected:
            path = q.request_path(directory, request["request_id"])
            q.save_request(path, bundle(request, rank, mode, binding))
            hashes[path.name] = q.sha256_file(path)
        q.old.atomic_write_json(directory / "worker_manifest.json", {
            "status": "pass", "rank": rank, "mode": mode, "world_size": 4,
            "binding": binding, "binding_sha256": q.sha256_json(binding), "hidden_index_opened": False,
            "selected_request_ids": [r["request_id"] for r in selected], "request_files": hashes,
            "checks": {"geometry": "pass", "all_active_closure": "pass",
                "self_patch": "pass" if mode == "canary" else "bound_canary_pass",
                "answer_token_equivalence": "pass" if mode == "canary" else "bound_canary_pass"},
            "runtime_fingerprint": {"model_revision": q.REVISION, "processor_revision": q.REVISION,
                "torch": "fixture", "transformers": "fixture"},
            "model_load_elapsed_s": 0., "work_elapsed_s": 0.,
            "requests_completed": len(selected), "outcome_rows": len(selected) * 68})
    return public, binding


def test_frozen_config_and_legacy_default():
    q.validate_config(config())
    cfg = config()
    cfg["technical_gates"]["full_sequence_closure_max_abs_error_bits"] = .01
    with pytest.raises(q.GateError):
        q.validate_config(cfg)
    spec = q.make_prespec(config())
    assert q.old.expected_outcomes_per_request(q.MODEL_KEY, spec) == 68
    del spec["operator_override"]
    assert q.old.operators_for_layer(q.MODEL_KEY, 36, spec) == ("F", "K", "A")


def test_shards_exact_and_canary_both_orders():
    public, _ = cohort()
    shards = [q.shard_requests(public[::-1], rank) for rank in range(4)]
    assert [len(s) // 2 for s in shards] == [12, 12, 11, 11]
    assert sum(len(s)*68 for s in shards) == 6256
    assert sum(len(q.shard_requests(public, r, mode="canary"))*68 for r in range(4)) == 544
    assert len({r["request_id"] for s in shards for r in s}) == 92
    for rank in range(4):
        assert {r["order_id"] for r in q.shard_requests(public, rank, mode="canary")} == {"order_0", "order_1"}
    with pytest.raises(q.GateError):
        q.shard_requests(public[:-1], 0)


def test_geometry_exact_not_count_only():
    q.validate_geometry(partition().as_dict())
    value = partition().as_dict()
    value["candidate_crops"]["positions"] = tuple(range(200))
    with pytest.raises(q.GateError):
        q.validate_geometry(value)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite", "algebra", "layer", "native"])
def test_rows_fail_closed(mutation):
    request = cohort()[0][0]
    rows = request_rows(request, 0)
    q.validate_rows(rows, request, 0, config())
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif mutation == "nonfinite":
        rows[0]["patched_margin_a_minus_b_bits"] = float("nan")
    elif mutation == "algebra":
        rows[0]["patched_donor_error_abs_bits"] += 1
    elif mutation == "layer":
        rows[0]["start_layer"] = 37
    else:
        rows[0]["real_margin_a_minus_b_bits"] += 1
    with pytest.raises(q.GateError):
        q.validate_rows(rows, request, 0, config())


def test_request_atomic_integrity(tmp_path):
    request = cohort()[0][0]
    payload = bundle(request, 0)
    path = q.request_path(tmp_path, request["request_id"])
    q.save_request(path, payload)
    kwargs = dict(request=request, rank=0, binding_sha256=q.sha256_json({"test": True}), config=config(), mode="canary")
    q.read_request(path, **kwargs)
    envelope = q.read_json(path)
    envelope["payload"]["rows"][0]["operator"] = "broken"
    q.old.atomic_write_json(path, envelope)
    with pytest.raises(q.GateError, match="hash"):
        q.read_request(path, **kwargs)


def test_shared_real_margin_across_corruptions():
    request = cohort()[0][0]
    rows = request_rows(request, 0)
    for row in rows:
        if row["corruption"] == "wrong_time":
            for key in ("real_margin_a_minus_b_bits", "corrupt_margin_a_minus_b_bits",
                        "recipient_margin_a_minus_b_bits", "donor_margin_a_minus_b_bits",
                        "patched_margin_a_minus_b_bits"):
                row[key] += 10.
    with pytest.raises(q.GateError, match="real native margin"):
        q.validate_rows(rows, request, 0, config())


@pytest.mark.parametrize("fault", ["underreported_max", "fabricated_error", "missing_choice", "nonfinite", "token"])
def test_equivalence_recomputed_from_choice_logprobs(tmp_path, fault):
    request = cohort()[0][0]
    payload = bundle(request, 0)
    audit = payload["answer_token_equivalence"]
    if fault == "underreported_max":
        audit["choices"]["A"].update(one_pass_logprob=-1., suffix_logprob=-1.-10*np.log(2), absolute_difference_bits=10.)
    elif fault == "fabricated_error":
        audit["choices"]["A"].update(one_pass_logprob=-1., suffix_logprob=-2., absolute_difference_bits=0.)
    elif fault == "missing_choice":
        del audit["choices"]["B"]
    elif fault == "nonfinite":
        audit["choices"]["A"]["suffix_logprob"] = float("inf")
    else:
        audit["choices"]["A"]["answer_token_ids"] = [True]
    if fault == "nonfinite":
        with pytest.raises(q.GateError):
            q.validate_equivalence(audit, .001)
        return
    path = q.request_path(tmp_path, request["request_id"])
    q.save_request(path, payload)
    with pytest.raises(q.GateError):
        q.read_request(path, request=request, rank=0, binding_sha256=q.sha256_json({"test": True}),
                       config=config(), mode="canary")


def test_canary_technical_only_and_missing_rank(tmp_path, monkeypatch):
    public, binding = write_run(tmp_path)
    rows, receipt = q.collect_run(tmp_path, "canary", config(), public, binding)
    assert len(rows) == 544 and receipt["technical_only"]
    assert "analysis" not in receipt and "primary" not in receipt
    monkeypatch.setattr(q, "load_inputs", lambda *a: (config(), public, binding))
    args = SimpleNamespace(config=tmp_path/"config", data_dir=tmp_path, run_dir=tmp_path, mode="canary", evaluator_index=None)
    q.aggregate_command(args)
    summary = (tmp_path/"canary/RESULTS.md").read_text()
    assert "Technical-only" in summary and "Recovery |" not in summary
    assert "signed contrast" not in summary
    args.evaluator_index = tmp_path/"must_not_open"
    with pytest.raises(q.GateError, match="technical-only"):
        q.aggregate_command(args)
    (tmp_path/"canary/rank_003/worker_manifest.json").unlink()
    with pytest.raises(q.analysis.AnalysisError):
        q.collect_run(tmp_path, "canary", config(), public, binding)


def test_label_refusal_and_source_hash_drift(tmp_path):
    with pytest.raises(q.GateError):
        q.reject_labels({"nested": [{"correct_choice": "A"}]})
    path = tmp_path/"file"
    path.write_text("before")
    digest = q.sha256_file(path)
    q.check_hash(path, digest)
    path.write_text("after")
    with pytest.raises(q.GateError):
        q.check_hash(path, digest)
    with pytest.raises(q.GateError):
        q.contained(tmp_path, "../escape")


def test_full_6256_technical_aggregate_then_explicit_evaluator(tmp_path, monkeypatch):
    public, binding = write_run(tmp_path, "canary")
    write_run(tmp_path, "full")
    monkeypatch.setattr(q, "load_inputs", lambda *a: (config(), public, binding))
    args = SimpleNamespace(config=tmp_path/"config", data_dir=tmp_path, run_dir=tmp_path,
                           mode="full", evaluator_index=None)
    q.aggregate_command(args)
    output = q.read_json(tmp_path/"full/aggregate.json")
    assert output["outcome_rows"] == 6256 and output["technical_only"]
    assert "analysis" not in output
    args.evaluator_index = tmp_path/"evaluator.jsonl"
    q.old.atomic_write_jsonl(args.evaluator_index, cohort()[1])
    q.aggregate_command(args)
    output = q.read_json(tmp_path/"full/aggregate.json")
    assert not output["technical_only"]
    assert output["evaluator_index_sha256"] == q.sha256_file(args.evaluator_index)
    assert output["analysis"]["K_recovery"] == pytest.approx(.8)
    summary = (tmp_path/"full/RESULTS.md").read_text()
    assert "reused 46-frame" in summary and "rho(CQ)/rho(K)" in summary
    assert "(phi_C+phi_Q)/rho(K)" in summary and "noncandidate_below_75pct_K" in summary
    extra = tmp_path/"full/rank_004"
    extra.mkdir()
    with pytest.raises(q.GateError, match="rank directories"):
        q.collect_run(tmp_path, "full", config(), public, binding)


def test_readonly_external_canary_and_tampered_check(tmp_path, monkeypatch):
    public, binding = write_run(tmp_path, "canary")
    monkeypatch.setenv("QRF_CANARY_DIR", str(tmp_path/"canary"))
    q.validate_canary(tmp_path/"other_run", config(), public, binding)
    path = tmp_path/"canary/rank_000/worker_manifest.json"
    receipt = q.read_json(path)
    receipt["checks"]["self_patch"] = "fail"
    q.old.atomic_write_json(path, receipt)
    with pytest.raises(q.GateError, match="technical check"):
        q.validate_canary(tmp_path/"other_run", config(), public, binding)


def test_model_config_revision_and_strict_json(tmp_path):
    cfg = deepcopy(config())
    cfg["model"]["processor_revision"] = "wrong"
    with pytest.raises(q.GateError):
        q.validate_config(cfg)
    path = tmp_path/"bad.json"
    for value in ('{"x": NaN}', '{"x":1,"x":2}'):
        path.write_text(value)
        with pytest.raises(q.analysis.AnalysisError):
            q.read_json(path)


def analysis_fixture(recovery=.8, variable=False, zero=False):
    public, hidden = cohort()
    rows = []
    for rank in range(4):
        for request in q.shard_requests(public, rank):
            frame = int(request["frame_unit_id"].split("_")[-1])
            gap = 0. if zero else (1. if frame < 23 else 9.)
            rho = (0. if frame < 23 else .9) if variable else recovery
            rows.extend(request_rows(request, rank, gap=gap, recovery=rho))
    return rows, public, hidden


def test_ratio_cluster_means_and_bonferroni_contrasts():
    rows, public, hidden = analysis_fixture(variable=True)
    result = q.analyze_rows(rows, public, hidden, config())
    assert result["K_recovery"] == pytest.approx(.81)
    assert result["K_recovery"] != pytest.approx(.45)
    assert result["primary"]["noncandidate_below_75pct_K"]["signed_contrast"] == pytest.approx(-.2025)
    assert result["primary"]["prefix_excluded_above_75pct_K"]["signed_contrast"] == pytest.approx(.2025)
    indexes = q.analysis._bootstrap_indexes(clusters=46, draws=200, seed=42)
    base = np.array([1.]*23+[9.]*23)
    error = np.array([1.]*23+[.9]*23)
    boot = 1-error[indexes].mean(1)/base[indexes].mean(1)
    assert result["primary"]["noncandidate_below_75pct_K"]["lower_97_5"] == pytest.approx(np.quantile(-.25*boot, .025))
    assert "pvalue" not in str(result) and "p_value" not in str(result)
    assert sum(result["factorial"]["shapley"].values()) == pytest.approx(.81)
    assert result["factorial"]["CQ_recovery_fraction"] == pytest.approx(1.)
    assert result["factorial"]["CQ_shapley_fraction"] == pytest.approx(.5)
    assert len(result["factorial"]["CQ_recovery_fraction_descriptive_ci95"]) == 2


@pytest.mark.parametrize("recovery,zero", [(.05, False), (.8, True)])
def test_failed_guards_are_descriptive(recovery, zero):
    result = q.analyze_rows(*analysis_fixture(recovery=recovery, zero=zero), config())
    assert result["status"] == "descriptive_guard_failed"
    summary = q.results_markdown({"status": "pass", "mode": "full", "outcome_rows": 6256,
        "technical_only": False, "analysis": result})
    assert "Technical completion: pass" in summary and "descriptive_guard_failed" in summary
    if not zero:
        assert result["factorial"]["CQ_shapley_fraction"] is None
        assert all(v["status"] == "descriptive_guard_failed" for v in result["primary"].values())


def test_old_game_bit_order():
    weights = [1., 2., 3., 4.]
    values = np.array([sum(w for bit, w in enumerate(weights) if mask & (1 << bit)) for mask in range(16)])
    assert list(q.FACTORIAL) == ["".join(c for i,c in enumerate("CQPR") if mask & (1 << i)) for mask in range(1,16)]
    np.testing.assert_allclose(q.analysis._shapley_from_values(values), weights)
    mobius = q.analysis._mobius_from_values(values)
    assert mobius[[1,2,4,8]].tolist() == weights
    assert sum(mobius) == 10.


def test_atomic_rank_progress_has_only_technical_counts(tmp_path):
    for done, status in ((0, "loading_model"), (1, "running"), (2, "done")):
        q.write_progress(tmp_path, mode="canary", rank=3, done=done, expected=2,
                         elapsed=float(done), status=status)
        value = q.read_json(tmp_path/"rankprogress.json")
        assert value["requests_done"] == done and value["requests_expected"] == 2
        assert value["outcomes_done"] == done * 68 and value["outcomes_expected"] == 136
        assert value["status"] == status and "analysis" not in value
        assert "rho" not in str(value) and not list(tmp_path.glob(".*.tmp.*"))


def test_tiny_qwen_real_tensor_factorial_and_hook_restore(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5TextModel"):
        pytest.skip("requires installed Transformers with Qwen3.5; never downloads")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model_config = transformers.Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=1,
        linear_num_value_heads=2, layer_types=["linear_attention", "full_attention"])

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = transformers.Qwen3_5TextModel(model_config)
            self.head = torch.nn.Linear(16, 32, bias=False)

        def forward(self, inputs_embeds, **kwargs):
            output = self.decoder(inputs_embeds=inputs_embeds, use_cache=False)
            return SimpleNamespace(logits=self.head(output.last_hidden_state))

    model = Tiny().eval()
    tiny_partition = q.old.build_component_partition(sequence_length=12, active_positions=tuple(range(12)),
        image_slot_positions=((1,2),(3,4),(5,6)), literal_user_prompt_start=7,
        literal_user_prompt_count=3, answer_prefix_token_count=1)
    inputs = {arm: {"inputs_embeds": torch.randn(1,12,16)} for arm in q.old.ARMS}
    prepared = {arm: SimpleNamespace(choices={choice: SimpleNamespace(inputs=inputs[arm],
        base_length=12, candidate_ids=[index+1]) for index,choice in enumerate(("A","B"))}) for arm in q.old.ARMS}
    monkeypatch.setattr(q.old._answer, "prepare_arm", lambda **kw: prepared[kw["arm"]])
    monkeypatch.setattr(q.old._base, "assert_prepared_alignment", lambda *a: None)
    monkeypatch.setattr(q.old, "_partition_for_prepared", lambda *a: tiny_partition)
    sites = [q.old.TransportSite(site_id=f"block_{i}", module_name=f"decoder.layers.{i}", layer_index=i) for i in range(2)]
    # The tiny adapter maps the production L36 intervention to its first block.
    by_layer = {36: sites[0]}
    with torch.inference_mode():
        before = model(**inputs["blank_full"]).logits.clone()
    for request in cohort()[0][:2]:
        rows, detail = q.old.evaluate_request(runtime=None, model=model, processor=None, tokenizer=None,
            device="cpu", model_key=q.MODEL_KEY, rank=0, request=request, data_dir=tmp_path,
            sites=sites, sites_by_layer=by_layer, prespec=q.make_prespec(config()), canary=True)
        assert len(rows) == 68 and {r["operator"] for r in rows} == set(q.OPERATORS)
        assert len(detail["self_checks"]) == 17
        assert all(r["patched_donor_error_abs_bits"] < .001 for r in rows if r["operator"] == "A")
        assert all(r["audit"]["all_removed"] for r in rows)
    with torch.inference_mode():
        after = model(**inputs["blank_full"]).logits
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert all(not layer._forward_hooks for layer in model.decoder.layers)
