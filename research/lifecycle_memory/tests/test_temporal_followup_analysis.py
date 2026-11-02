"""Synthetic numeric follow-up tests; no models, media, or real receipts."""

from copy import deepcopy
import inspect
import json
import unittest
from unittest.mock import patch

from research.lifecycle_memory.temporal_forecasting.analysis import _bootstrap, _interval, _ratio
from research.lifecycle_memory.temporal_forecasting.common import digest
from research.lifecycle_memory.temporal_forecasting.followup_analysis import (
    COHORTS, K, SEED, _absolute_recovery, _relative_ratio,
    analyze_followup, validate_followup_technical,
)
from research.lifecycle_memory.temporal_forecasting.followup_protocol import (
    CONDITIONS, PIN_FIELDS, SUBSETS, followup_inventory, followup_protocol, scene_permutation,
)
from research.lifecycle_memory.temporal_forecasting.protocol import protocol_contract
from research.lifecycle_memory.temporal_forecasting.recognition_protocol import (
    SCENE_SLOTS, TASKS, recognition_protocol, selected_recognition_episodes,
)
from research.lifecycle_memory.tests.test_temporal_dense_analysis import runtime_evidence, scores


POLICY = {"version": 1, "kind": "matched_static_reuse",
          "historical_static_sha256": "a" * 64, "temporal_exposure_sha256": "b" * 64}


def position_map():
    result = {f"H{i}": [i - 1] for i in range(1, 17)}
    result.update(V=[16], CA=[17], CB=[18], C=[17, 18], Q=[19, 20],
                  R=[21, 22], P=[23, 24], H=list(range(16)), F_video=list(range(17)),
                  K=list(range(17, 25)), A=list(range(25)))
    for subset in SUBSETS:
        result[subset] = sorted(p for name in subset.split("+") for p in result[name])
    return result


def frame_audit(row, condition):
    indices = scene_permutation(row["task"], row["episode_id"], condition)
    source = [digest([row["episode_id"], key]) for key in SCENE_SLOTS]
    scene_keys = [SCENE_SLOTS[i] for i in indices]
    scene_hashes = [source[i] for i in indices]
    candidates = [f"candidate{row['order']}", f"candidate{1 - row['order']}"]
    candidate_hashes = [digest([row["episode_id"], key]) for key in candidates]
    return {"condition": condition, "seed": SEED, "task": row["task"], "episode_id": row["episode_id"],
            "scene_permutation": indices, "source_frame_keys": list(SCENE_SLOTS),
            "source_frame_sha256": source, "scene_frame_keys": scene_keys, "scene_frame_sha256": scene_hashes,
            "candidate_keys": candidates, "candidate_sha256": candidate_hashes,
            "scene_order_sha256": digest({"indices": indices, "keys": scene_keys, "sha256": scene_hashes}),
            "candidate_order_sha256": digest({"keys": candidates, "sha256": candidate_hashes}),
            "endpoints_preserved": scene_hashes[0] == source[0] and scene_hashes[-1] == source[-1],
            "content_multiset_preserved": sorted(scene_hashes) == sorted(source),
            "prompt_chronology_violated": condition in ("S1", "S2", "RV")}


def token_audit():
    positions = position_map()
    ids = list(range(1000, 1025))
    return {"input_ids_sha256": digest(ids), "attention_mask_sha256": digest([1] * 25),
            "cardinalities": {k: len(v) for k, v in positions.items()},
            "position_sha256": {k: digest(v) for k, v in positions.items()},
            "token_ids_sha256": {k: digest([ids[i] for i in v]) for k, v in positions.items()},
            "disjoint_exhaustive": True, "readout_in_P": True, "readout_excluded_from_CQR": True}


def fixture(task=TASKS[0], stage="recognition", native=None, fraction=None):
    clips, wearers = COHORTS[task]
    source = recognition_protocol(task, POLICY, episode_limit=clips, minimum_clusters=wearers)
    protocol = followup_protocol(source, **dict.fromkeys(PIN_FIELDS, "e" * 64))
    contract = protocol_contract(protocol)
    sizes = (3,) * 8 + (2,) + (1,) * 11 if clips == 37 else (3,) * 6 + (2,) + (1,) * 7
    episodes = [{"episode_id": f"e{w:02}-{e}", "cluster_id": f"w{w:02}",
                 "task": task, "contract_sha256": digest(contract)}
                for w, size in enumerate(sizes) for e in range(size)]
    selected = selected_recognition_episodes(episodes, protocol, stage)
    labels = [{"episode_id": e["episode_id"], "correct_candidate": int(e["cluster_id"][1:]) % 2,
               "label_status": "visually_audited", "reviewer": "synthetic-test",
               "review_evidence_sha256": "c" * 64, "review_receipt_sha256": "d" * 64}
              for e in selected]
    gold = {r["episode_id"]: r["correct_candidate"] for r in labels}
    rows, cached_frames = [], {}
    partition = token_audit()

    def frames(item, condition):
        key = (item["episode_id"], item["order"], condition)
        if key not in cached_frames:
            cached_frames[key] = frame_audit(item, condition)
        return deepcopy(cached_frames[key])

    for item in followup_inventory(episodes, protocol, stage):
        sign = 1 if item["order"] == gold[item["episode_id"]] else -1
        values = native(item) if native else {"B0": 3., "B4": -1., "S1": 1., "S2": -1., "RV": -3.}
        values = {key: value * sign for key, value in values.items()}
        row = {**item, "status": "pass", "phase": stage,
               "identity": {"protocol_sha256": digest(protocol), "simulation": False},
               "positions": position_map(), "readout_position": 24, "answer_token_ids": {"A": 100, "B": 101},
               "token_partition": deepcopy(partition)}
        if item["kind"] == "behavior":
            row.update(scores(values[item["condition"]]))
            row["frame_audit"] = frames(item, item["condition"])
        else:
            donor, recipient = (values[item[f"{role}_condition"]] for role in ("donor", "recipient"))
            subset = set(item["position_set"].split("+"))
            recovery = sum({"C": .08, "Q": .16, "P": .04, "R": .02}.get(p, 0.) for p in subset)
            recovery += .12 * ({"C", "Q"} <= subset) + .1 * ({"C", "P"} <= subset)
            recovery *= 1 if item["donor_condition"] == "B0" else .8
            if fraction:
                recovery = fraction(item, recovery)
            if item["position_set"] == "A":
                recovery = 1.
            row.update(scores(recipient + recovery * (donor - recipient)),
                       donor_margin_ab_bits=donor, recipient_margin_ab_bits=recipient,
                       donor_score=scores(donor), recipient_score=scores(recipient))
            row.update(runtime_evidence(donor, recipient, item["kind"] == "self"))
            row["frame_audit"] = frames(item, item["recipient_condition"])
            row["donor_frame_audit"] = frames(item, item["donor_condition"])
            row["donor_token_partition"] = deepcopy(partition)
            names = ("A",) if item["position_set"] == "A" else item["position_set"].split("+")
            selected_positions = sorted({p for name in names for p in row["positions"][name]})
            row["patch_selection"] = {"positions": selected_positions, "cardinality": len(selected_positions),
                                      "positions_sha256": digest(selected_positions),
                                      "token_ids_sha256": digest([1000 + p for p in selected_positions]),
                                      "includes_readout": 24 in selected_positions}
        rows.append(row)
    protocol["episode_ids"] = [e["episode_id"] for e in selected]
    return rows, labels, protocol


def analyze(values, draws=12, stage="recognition"):
    return analyze_followup(*values, stage, draws=draws)


def direction(result, reverse=False):
    return result["factorial"][int(reverse)]


def profile(result, subset=K, reverse=False):
    return next(p for p in direction(result, reverse)["subsets"] if p["position_set"] == subset)


class FollowupAnalysisTests(unittest.TestCase):
    def test_exact_full_cohorts_both_tasks_and_no_input_mutation(self):
        total = 0
        for task in TASKS:
            values = fixture(task=task)
            before = deepcopy(values)
            result = analyze(values, draws=2)
            self.assertEqual(values, before)
            self.assertTrue(result["analysis_complete"], result["technical_controls"]["failures"][:2])
            self.assertTrue(result["scientific_analysis_performed"])
            clips, wearers = COHORTS[task]
            total += clips
            self.assertEqual(result["coverage"]["expected_rows"], clips * 78)
            self.assertEqual(result["coverage"]["observed_clusters"], wearers)
            self.assertEqual(result["coverage"]["counts_by_kind"],
                             {"behavior": clips * 10, "patch": clips * 64, "self": clips * 4})
            self.assertEqual(len(result["patch_profiles"]), 32)
            self.assertEqual(result["family_size"], 0)
            self.assertFalse(result["primary_complete"])
            encoded = json.dumps(result, allow_nan=False)
            for forbidden in ('"p_value"', '"reject"', '"holm_adjusted_p"'):
                self.assertNotIn(forbidden, encoded)
        self.assertEqual(total, 64)

    def test_real_default_draw_count_and_shared_seed(self):
        self.assertEqual(inspect.signature(analyze_followup).parameters["draws"].default, 19999)
        for task in TASKS:
            result = analyze_followup(*fixture(task=task), "recognition")
            self.assertEqual(result["bootstrap"]["draws"], 19999)
            self.assertEqual(result["bootstrap"]["seed"], 20261025)
            self.assertTrue(result["bootstrap"]["joint_resamples"])
            self.assertEqual(profile(result)["recovery"]["bootstrap_draws"], 19999)

    def test_canary_is_label_free_and_requires_four_distinct_wearers(self):
        for task in TASKS:
            rows, _, protocol = fixture(task=task, stage="recognition-canary")
            with patch("research.lifecycle_memory.temporal_forecasting.data._labels",
                       side_effect=AssertionError("labels opened")):
                report = validate_followup_technical(rows, protocol["episode_ids"])
                result = analyze_followup(rows, object(), protocol, "recognition-canary")
            self.assertTrue(report["passed"], report["failures"][:2])
            self.assertEqual(report["coverage"]["expected_rows"], 312)
            self.assertTrue(result["analysis_complete"])
            self.assertFalse(result["labels_opened"])
            self.assertFalse(result["scientific_analysis_performed"])
            self.assertEqual(result["behavior"], [])
            wearers = sorted({r["cluster_id"] for r in rows})
            for row in rows:
                if row["cluster_id"] == wearers[1]:
                    row["cluster_id"] = wearers[0]
            self.assertFalse(validate_followup_technical(rows, protocol["episode_ids"])["passed"])

    def test_subset_recovery_interactions_and_shapley_in_both_directions(self):
        result = analyze(fixture())
        for reverse, scale in ((False, 1.), (True, .8)):
            block = direction(result, reverse)
            self.assertAlmostEqual(profile(result, reverse=reverse)["recovery"]["estimate"], .52 * scale)
            self.assertEqual(profile(result, "A", reverse)["recovery"]["ci95"], [1., 1.])
            for name, effect in (("K-minus-QPR", .3), ("K-minus-CQR", .14)):
                delta = block["paired_differences"][name]["recovery_difference"]
                self.assertAlmostEqual(delta["estimate"], effect * scale)
                self.assertAlmostEqual(delta["ci95"][0], effect * scale)
            self.assertAlmostEqual(block["relative_recoveries"]["QPR/K"]["estimate"], .22 / .52)
            self.assertAlmostEqual(block["relative_recoveries"]["CQR/K"]["estimate"], .38 / .52)
            self.assertAlmostEqual(block["relative_recoveries"]["CQ/K"]["estimate"], .36 / .52)
            interaction = block["CQ_interaction"]
            self.assertAlmostEqual(interaction["recovery"]["estimate"], .12 * scale)
            self.assertAlmostEqual(interaction["raw_gold_margin_bits"]["estimate"],
                                   .48 * scale * (-1 if reverse else 1))
            for player, value in {"C": .19, "Q": .22, "P": .09, "R": .02}.items():
                self.assertAlmostEqual(block["shapley"][player]["value"]["estimate"], value * scale)
                self.assertAlmostEqual(block["shapley"][player]["share"]["estimate"], value / .52)
            self.assertTrue(block["shapley_efficiency"]["verified"])
            self.assertAlmostEqual(block["shapley_efficiency"]["share_sum"], 1.)
            self.assertAlmostEqual(block["shapley_efficiency"]["value_sum"], .52 * scale)
            self.assertNotAlmostEqual(profile(result, "C+Q", reverse)["recovery"]["estimate"],
                                      sum(block["shapley"][p]["value"]["estimate"] for p in "CQ"))

    def test_orders_then_clips_then_wearers(self):
        def native(row):
            episode = int(row["episode_id"].split("-")[1])
            margin = 2 + 2 * episode + row["order"] if row["cluster_id"] == "w00" else 10.
            return {"B0": margin, "B4": 0., "S1": 1., "S2": -1., "RV": -3.}
        result = analyze(fixture(native=native))
        metric = result["behavior"][0]["metrics"]["gold_margin_bits"]
        self.assertEqual(metric["wearer_values"]["w00"], 4.5)
        self.assertEqual(metric["estimate"], (4.5 + 19 * 10) / 20)
        self.assertAlmostEqual(profile(result)["recovery"]["estimate"], .52)

    def test_transformed_agreement_and_main_average_not_flipped_truth(self):
        result = analyze(fixture())
        behavior = {r["condition"]: r for r in result["behavior"]}
        self.assertIn("accuracy", behavior["B0"]["metrics"])
        self.assertIn("accuracy", behavior["B4"]["metrics"])
        for condition in ("S1", "S2", "RV"):
            self.assertNotIn("accuracy", behavior[condition]["metrics"])
            self.assertFalse(behavior[condition]["truth_accuracy"])
            self.assertIn("original_label_loss_nats", behavior[condition]["metrics"])
        self.assertEqual(behavior["RV"]["metrics"]["original_answer_agreement"]["estimate"], 0.)
        contrasts = {r["control"]: r for r in result["order_contrasts"]}
        main = contrasts["mean(S1,S2)"]
        self.assertTrue(main["main_order_diagnostic"])
        self.assertEqual(main["metrics"]["original_answer_agreement_difference"]["estimate"], .5)
        self.assertEqual(main["metrics"]["original_label_margin_difference_bits"]["estimate"], 3.)
        self.assertLess(main["metrics"]["original_label_loss_difference_nats"]["estimate"], 0.)
        self.assertEqual(contrasts["RV"]["metrics"]["original_label_margin_difference_bits"]["estimate"], 6.)

    def test_paired_intervals_use_joint_wearer_draws(self):
        def fraction(row, value):
            if row["position_set"] == K:
                return .3 + int(row["cluster_id"][1:]) / 100
            if row["position_set"] == "Q+P+R":
                return .1 + int(row["cluster_id"][1:]) / 100
            return value
        result = analyze(fixture(fraction=fraction), draws=257)
        block = direction(result)
        paired = block["paired_differences"]["K-minus-QPR"]["recovery_difference"]
        self.assertAlmostEqual(paired["ci95"][0], .2)
        self.assertAlmostEqual(paired["ci95"][1], .2)
        k = profile(result)["T_bits"]["wearer_values"]
        qpr = profile(result, "Q+P+R")["T_bits"]["wearer_values"]
        baseline = profile(result)["D0_bits"]["wearer_values"]
        keys = sorted(k)
        means, _ = _bootstrap([[v[w] for w in keys] for v in (k, qpr, baseline)], 257, SEED)
        expected = _interval((means[:, 0] - means[:, 1]) / means[:, 2])
        for actual, target in zip(paired["ci95"], expected):
            self.assertAlmostEqual(actual, target)

    def test_k_guard_is_in_bits_not_dimensionless_recovery(self):
        native = {f"w{i}": 10. for i in range(4)}
        tiny_k = {w: .02 for w in native}
        values = {w: .01 for w in native}
        ratio = _relative_ratio(values, tiny_k, native, 32)
        self.assertTrue(ratio["stable"])
        self.assertEqual(ratio["estimate"], .5)
        self.assertEqual(_absolute_recovery(tiny_k, native, 32)["estimate"], .002)
        too_small = _relative_ratio(values, dict.fromkeys(native, .005), native, 32)
        self.assertFalse(too_small["stable"])
        self.assertIsNone(too_small["estimate"])
        self.assertIsNone(too_small["ci95"])

    def test_ratio_guards_match_existing_helper_and_require_both(self):
        numerator = {f"w{i}": .01 for i in range(4)}
        native = dict.fromkeys(numerator, 1.)
        for values in ([.01] * 4, [-.1, .2, .2, .2], [.2] * 4, [0., .03, .04, .06]):
            denominator = dict(zip(numerator, values))
            expected = _ratio(numerator, denominator, 199, SEED, .01)
            actual = _relative_ratio(numerator, denominator, native, 199)
            for key in ("stable", "estimate", "ci95"):
                self.assertEqual(actual[key], expected[key])
        fail_native = _relative_ratio(numerator, native, dict.fromkeys(native, .005), 32)
        self.assertFalse(fail_native["native_guard_passed"])
        self.assertTrue(fail_native["k_effect_guard_passed"])
        self.assertIsNone(fail_native["estimate"])
        for bad in ({"extra": 1.}, {**numerator, "w0": float("nan")}):
            with self.assertRaisesRegex(ValueError, "finite paired"):
                _relative_ratio(bad, native, native, 32)
        single = _relative_ratio({"w": .2}, {"w": .4}, {"w": 1.}, 32)
        self.assertFalse(single["stable"])

    def test_second_guard_one_percent_boundary_and_no_nonpositive_draws(self):
        import numpy as np

        numerator = {f"w{i}": .02 for i in range(4)}
        k_effect, native = dict.fromkeys(numerator, .04), dict.fromkeys(numerator, 1.)
        for bad_count, bad_value, stable in ((10, .005, True), (11, .005, False), (1, 0., False), (1, -.01, False)):
            means = np.tile([.02, .04], (1000, 1))
            means[:bad_count, 1] = bad_value
            with patch("research.lifecycle_memory.temporal_forecasting.analysis._bootstrap",
                       return_value=(means, np.zeros_like(means))):
                result = _relative_ratio(numerator, k_effect, native, 1000)
            self.assertEqual(result["stable"], stable)
            self.assertEqual(result["unstable_draw_fraction"], bad_count / 1000)

    def test_recovery_is_ratio_of_wearer_means_not_mean_row_ratios(self):
        def native(row):
            return {"B0": 10. if row["cluster_id"] == "w00" else 1.,
                    "B4": 0., "S1": 0., "S2": 0., "RV": 0.}

        def fraction(row, _value):
            return .8 if row["cluster_id"] == "w00" else .2

        result = analyze(fixture(native=native, fraction=fraction))
        rho = profile(result)["recovery"]
        self.assertAlmostEqual(rho["estimate"], (10 * .8 + 19 * .2) / (10 + 19))
        self.assertNotAlmostEqual(rho["estimate"], (.8 + 19 * .2) / 20)

    def test_full_shapley_grid_with_three_and_four_player_interactions(self):
        def fraction(row, value):
            subset = set(row["position_set"].split("+"))
            return value + .13 * (set("QPR") <= subset) + .24 * (set("CQPR") <= subset)

        result = analyze(fixture(fraction=fraction))
        allocation = direction(result)["shapley"]
        expected = {"C": .19 + .06, "Q": .22 + .13 / 3 + .06,
                    "P": .09 + .13 / 3 + .06, "R": .02 + .13 / 3 + .06}
        for player, value in expected.items():
            self.assertAlmostEqual(allocation[player]["value"]["estimate"], value)
        self.assertAlmostEqual(direction(result)["shapley_efficiency"]["value_sum"], .89)

    def test_small_gaps_negative_values_and_zero_k_are_not_exclusions(self):
        def native(row):
            gap = 0. if row["cluster_id"] == "w00" else 2.
            return {"B0": gap, "B4": 0., "S1": 0., "S2": 0., "RV": 0.}
        result = analyze(fixture(native=native, fraction=lambda r, v: -.2), draws=64)
        rho = profile(result)["recovery"]
        self.assertTrue(result["analysis_complete"])
        self.assertAlmostEqual(rho["estimate"], -.2)
        self.assertEqual(rho["zero_denominator_rows"], 6)
        self.assertEqual(rho["excluded_rows"], 0)
        self.assertEqual(rho["excluded_clusters"], 0)
        self.assertEqual(rho["n_clusters"], 20)
        self.assertFalse(direction(result)["relative_recoveries"]["CQ/K"]["stable"])
        zeros = analyze(fixture(fraction=lambda r, v: 0.))
        self.assertTrue(zeros["analysis_complete"])
        self.assertEqual(profile(zeros)["recovery"]["estimate"], 0.)
        self.assertIsNone(direction(zeros)["shapley"]["C"]["share"]["estimate"])

    def test_missing_extra_duplicate_and_failed_rows_fail_without_complete_case_selection(self):
        rows, labels, protocol = fixture(stage="recognition-canary")
        expected = protocol["episode_ids"]
        variants = (rows[1:], rows + [rows[0]], [],
                    [r for r in rows if r["episode_id"] != expected[0]],
                    [{**rows[0], "task_id": "extra", "condition": "B1"}, *rows[1:]],
                    [{**rows[0], "status": "fail"}, *rows[1:]])
        for bad in variants:
            self.assertFalse(validate_followup_technical(bad, expected)["passed"])
        for bad_expected in (None, [], "not-an-inventory", expected + expected[:1]):
            self.assertFalse(validate_followup_technical(rows, bad_expected)["passed"])
        rows, labels, protocol = fixture()
        result = analyze((rows[1:], labels, protocol))
        self.assertFalse(result["analysis_complete"])
        self.assertEqual(result["factorial"], [])
        self.assertEqual(result["behavior"], [])
        self.assertEqual(result["coverage"]["missing_rows"], 1)

    def test_baseline_equality_stricter_than_control_tolerance(self):
        rows, _, protocol = fixture(stage="recognition-canary")
        row = next(r for r in rows if r["kind"] == "patch" and r["position_set"] == "Q")
        row["donor_margin_ab_bits"] += 1e-7
        report = validate_followup_technical(rows, protocol["episode_ids"])
        self.assertFalse(report["passed"])
        self.assertTrue(any(c["kind"] == "shared_denominator" and not c["passed"] for c in report["checks"]))

    def test_malformed_score_runtime_certificates_simulation_and_positions_rejected(self):
        rows, _, protocol = fixture(stage="recognition-canary")
        patch_index = next(i for i, r in enumerate(rows) if r["kind"] == "patch")
        mutations = [
            lambda r: r.pop("technical"), lambda r: r.pop("controls"), lambda r: r.pop("donor_score"),
            lambda r: r["technical"].update(hook_calls=2),
            lambda r: r["technical"].update(suffix_replay=True),
            lambda r: r["controls"]["full_A"].update(passed=False),
            lambda r: r.update(margin_ab_bits=float("nan")),
            lambda r: r["donor_score"].update(a_logprob=r["donor_score"]["a_logprob"] - .1,
                                             b_logprob=r["donor_score"]["b_logprob"] - .1),
            lambda r: r.update(order=True), lambda r: r.update(cut=22),
            lambda r: r.update(position_set="K"), lambda r: r.update(position_set="Q+C"),
            lambda r: r.pop("identity"), lambda r: r["identity"].update(simulation=True),
            lambda r: r.update(simulation=True), lambda r: r.pop("positions"),
            lambda r: r["positions"].update(C=[17]),
            lambda r: r["positions"].update(R=[21, 22, 24]),
            lambda r: r["positions"].update(P=[23]),
            lambda r: r["positions"].update(Q=[19, 20, 23]),
            lambda r: r.update(readout_position=23),
            lambda r: r.update(answer_token_ids={"A": 100, "B": 100}),
        ]
        for i, change in enumerate(mutations):
            with self.subTest(mutation=i):
                bad = deepcopy(rows)
                change(bad[patch_index])
                self.assertFalse(validate_followup_technical(bad, protocol["episode_ids"])["passed"])

    def test_protocol_phase_contract_and_label_drift_rejected(self):
        values = fixture()
        for stage in ("confirmation", "recognition-canary"):
            with self.assertRaises(ValueError):
                analyze(values, stage=stage)
        for field, value in (("schema_version", 4), ("study", "temporal-recognition-v1"),
                             ("seed", 1), ("bootstrap_seed", 1), ("episode_limit", 36)):
            rows, labels, protocol = deepcopy(values)
            protocol[field] = value
            with self.assertRaises(ValueError):
                analyze((rows, labels, protocol))
        rows, labels, protocol = deepcopy(values)
        rows[0]["contract_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            analyze((rows, labels, protocol))
        rows, labels, protocol = deepcopy(values)
        for row in rows:
            row["identity"]["protocol_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "protocol identity"):
            analyze((rows, labels, protocol))
        for labels in (values[1][1:], values[1] + values[1][:1], [{**values[1][0], "correct_candidate": True}, *values[1][1:]]):
            with self.assertRaises(ValueError):
                analyze((values[0], labels, values[2]))
        for draws in (True, 1, 2.5):
            with self.assertRaises(ValueError):
                analyze(values, draws=draws)

    def test_required_permutation_and_token_provenance_fail_closed(self):
        rows, _, protocol = fixture(stage="recognition-canary")
        patch_index = next(i for i, r in enumerate(rows) if r["kind"] == "patch")
        mutations = [lambda r: r.pop("frame_audit"), lambda r: r.pop("token_partition"),
                     lambda r: r.pop("donor_frame_audit"), lambda r: r.pop("donor_token_partition"),
                     lambda r: r.pop("patch_selection"),
                     lambda r: r["frame_audit"].update(seed=42),
                     lambda r: r["frame_audit"].update(scene_permutation=list(range(17))),
                     lambda r: r["frame_audit"].update(source_frame_sha256=["f" * 64] * 17),
                     lambda r: r["frame_audit"].update(candidate_sha256=["f" * 64] * 2),
                     lambda r: r["frame_audit"].update(candidate_keys=["candidate1", "candidate0"]),
                     lambda r: r["frame_audit"].update(scene_order_sha256="f" * 64),
                     lambda r: r["donor_frame_audit"].update(condition="B4"),
                     lambda r: r["donor_token_partition"].update(input_ids_sha256="f" * 64),
                     lambda r: r["token_partition"]["cardinalities"].update(C=100),
                     lambda r: r["token_partition"]["position_sha256"].update(C="f" * 64),
                     lambda r: r["token_partition"].update(readout_excluded_from_CQR=False),
                     lambda r: r["patch_selection"].update(positions=[17, 18, 24]),
                     lambda r: r["patch_selection"].update(cardinality=3),
                     lambda r: r["patch_selection"].update(positions_sha256="f" * 64),
                     lambda r: r["patch_selection"].update(token_ids_sha256="f" * 64),
                     lambda r: r["patch_selection"].update(includes_readout=True)]
        for i, change in enumerate(mutations):
            with self.subTest(mutation=i):
                bad = deepcopy(rows)
                change(bad[patch_index])
                self.assertFalse(validate_followup_technical(bad, protocol["episode_ids"])["passed"])
        for condition in ("S1", "S2", "RV"):
            bad = deepcopy(rows)
            row = next(r for r in bad if r["kind"] == "behavior" and r["condition"] == condition)
            row["frame_audit"]["prompt_chronology_violated"] = False
            self.assertFalse(validate_followup_technical(bad, protocol["episode_ids"])["passed"])

    def test_every_subset_token_hash_is_bound_to_its_native_group(self):
        for task in TASKS:
            rows, _, protocol = fixture(task=task, stage="recognition-canary")
            self.assertTrue(validate_followup_technical(rows, protocol["episode_ids"])["passed"])
            for subset in SUBSETS:
                with self.subTest(task=task, subset=subset):
                    bad = deepcopy(rows)
                    row = next(r for r in bad if r["kind"] == "patch" and r["position_set"] == subset)
                    self.assertEqual(row["patch_selection"]["token_ids_sha256"],
                                     row["token_partition"]["token_ids_sha256"][subset])
                    row["patch_selection"]["token_ids_sha256"] = "0" * 64
                    checks = validate_followup_technical(bad, protocol["episode_ids"])
                    self.assertFalse(checks["passed"])
                    self.assertTrue(any("patch_selection.named_group_token_hash" in c["errors"]
                                        for c in checks["failures"]))

    def test_missing_coalitions_cannot_bypass_native_hash_binding(self):
        rows, _, protocol = fixture(stage="recognition-canary")
        for subset in ("C+Q+R", "Q+P+R"):
            with self.subTest(subset=subset):
                bad = deepcopy(rows)
                for row in bad:
                    row["positions"].pop(subset)
                    for name in ("token_partition", "donor_token_partition"):
                        if name in row:
                            for mapping in ("cardinalities", "position_sha256", "token_ids_sha256"):
                                row[name][mapping].pop(subset)
                checks = validate_followup_technical(bad, protocol["episode_ids"])
                self.assertFalse(checks["passed"])
                self.assertTrue(any("unexpected_or_missing_position_groups" in c["errors"]
                                    for c in checks["failures"]))

    def test_shuffles_endpoints_content_and_physical_pairing_across_orders(self):
        rows, _, protocol = fixture(stage="recognition-canary")
        episode = protocol["episode_ids"][0]
        by_condition = {(r["condition"], r["order"]): r["frame_audit"] for r in rows
                        if r["episode_id"] == episode and r["kind"] == "behavior"}
        for condition in ("S1", "S2"):
            first, second = by_condition[condition, 0], by_condition[condition, 1]
            self.assertEqual(first["scene_permutation"], second["scene_permutation"])
            self.assertEqual(first["scene_frame_sha256"], second["scene_frame_sha256"])
            self.assertTrue(first["endpoints_preserved"])
            self.assertTrue(first["content_multiset_preserved"])
            self.assertEqual(first["candidate_sha256"], second["candidate_sha256"][::-1])
        self.assertNotEqual(by_condition["S1", 0]["scene_permutation"], by_condition["S2", 0]["scene_permutation"])
        bad = deepcopy(rows)
        row = next(r for r in bad if r["kind"] == "behavior" and r["condition"] == "S1")
        row["frame_audit"]["scene_permutation"] = by_condition["S2", 0]["scene_permutation"]
        self.assertFalse(validate_followup_technical(bad, protocol["episode_ids"])["passed"])


if __name__ == "__main__":
    unittest.main()
