"""Bounded synthetic evaluator tests: no models, media, or private data."""

from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

from research.lifecycle_memory.temporal_forecasting import protocol as protocols
from research.lifecycle_memory.temporal_forecasting.common import digest
from research.lifecycle_memory.temporal_forecasting.dense_analysis import (
    GEMMA, _recovery, analyze_dense_current, validate_dense_technical,
)


def scores(margin, common_shift=0):
    delta = margin * math.log(2)
    a = -(max(-delta, 0) + math.log1p(math.exp(-abs(delta)))) + common_shift
    return {"a_logprob": a, "b_logprob": a - delta, "margin_ab_bits": margin}


def runtime_evidence(donor, recipient, self_patch):
    def certificate(margin):
        return {"passed": True, "absolute_error_bits": dict.fromkeys(
            ("a_logprob", "b_logprob", "margin_ab_bits"), 0.),
            "max_drift_bits": 0., "scores": scores(margin)}

    audit = {"hook_calls": 1, "selected_rows_equal": True, "preserved_rows_equal": True}
    return {"technical": {"alignment_pass": True, "self_patch_pass": True,
            "full_A_pass": True, "full_forward_reference": True, "suffix_replay": False,
            "forward_count": 6, "controls_reused": dict.fromkeys(("recipient_self", "donor_self", "full_A"), False),
            "self_max_drift_bits": 0., "full_closure_max_drift_bits": 0.,
            "tolerance_bits": .001, **audit},
            "controls": {"recipient_self": certificate(recipient), "donor_self": certificate(donor),
                         "full_A": {**certificate(donor), **audit},
                         "selected_self": certificate(recipient) if self_patch else None}}


def fixture(n=4, null=False, phase="current-canary", margins=None):
    protocol = protocols.dense_current_protocol({
        "version": 1, "kind": "matched_static_reuse",
        "historical_static_sha256": "a" * 64, "temporal_exposure_sha256": "b" * 64,
    })
    protocol["bootstrap_seed"] = 19
    contract = protocols.protocol_contract(protocol)
    episodes = [{"episode_id": f"episode-{i:03}", "cluster_id": f"wearer-{i:03}",
                 "split": "confirmation", "task": contract["task"],
                 "contract_sha256": digest(contract)} for i in range(n)]
    labels = [{"episode_id": e["episode_id"], "correct_candidate": i % 2,
               "label_status": "audited", "static_label_preserved": True,
               "static_source_sha256": "c" * 64} for i, e in enumerate(episodes)]
    protocol["episode_ids"] = [e["episode_id"] for e in episodes]
    tasks = protocols.inventory(episodes, protocol, phase)
    rows = []
    for task in tasks:
        i = int(task["episode_id"].rsplit("-", 1)[1])
        sign = 1 if task["order"] == i % 2 else -1
        native = (margins(i, task["order"]) if margins else
                  {"B0": sign * (1 + i / 10),
                   "B1": sign * (1 + i / 10) if null else -sign,
                   "B4": sign * (1 + i / 10) if null else -sign * (1 + i / 10),
                   "B8": sign * (1 + i / 10) if null else -sign * .5})
        row = {**task, "status": "pass"}
        if task["kind"] == "behavior":
            row.update(scores(native[task["condition"]]))
        else:
            donor, recipient = native[task["donor_condition"]], native[task["recipient_condition"]]
            fraction = {"H": .25 + i / 20, "V": .5, "K": -.25, "A": 1}[task["position_set"]]
            if task["donor_condition"] == "B4" and task["position_set"] == "H":
                fraction = .1
            patched = donor if task["kind"] == "self" else recipient + fraction * (donor - recipient)
            row.update(scores(patched), donor_margin_ab_bits=donor, recipient_margin_ab_bits=recipient,
                       donor_score=scores(donor), recipient_score=scores(recipient))
            row.update(runtime_evidence(donor, recipient, task["kind"] == "self"))
        rows.append(row)
    return rows, labels, protocol


def analyze(fixture_values, phase="current-canary", draws=40):
    return analyze_dense_current(*fixture_values, phase, draws=draws)


def profile(result, direction="B0->B4", cut=22, positions="H"):
    return next(p for p in result["patch_profiles"] if
                (p["direction"], p["cut"], p["position_set"]) == (direction, cut, positions))


class DenseAnalysisTests(unittest.TestCase):
    def test_inventory_and_descriptive_only_no_mutation(self):
        inputs = fixture()
        before = deepcopy(inputs)
        result = analyze(inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(result["status"], "descriptive")
        self.assertTrue(result["data_complete"])
        self.assertTrue(result["analysis_complete"])
        self.assertFalse(result["primary_complete"])
        self.assertEqual(result["primary_family"], [])
        self.assertEqual(result["family_size"], 0)
        self.assertEqual(result["coverage"]["expected_rows"], 176)
        self.assertEqual(result["coverage"]["counts_by_kind"], {"behavior": 32, "patch": 128, "self": 16})
        self.assertEqual(len(result["patch_profiles"]), 16)
        self.assertEqual(result["technical_controls"]["coverage"]["expected_rows"], 48)
        self.assertEqual(result["cuts"], {GEMMA: [22, 34]})
        encoded = json.dumps(result, allow_nan=False)
        for field in ('"p_value"', '"reject"', '"holm_adjusted_p"', '"confirmatory": true'):
            self.assertNotIn(field, encoded)

    def test_full_96_wearers_and_incomplete_full_not_small_sample_confirmation(self):
        inputs = fixture(96, phase="current")
        result = analyze(inputs, phase="current", draws=8)
        self.assertTrue(result["analysis_complete"])
        self.assertEqual(result["coverage"]["expected_rows"], 4224)
        self.assertEqual(result["behavior"][0]["metrics"]["accuracy"]["n_clusters"], 96)
        small = analyze(fixture(), phase="current")
        self.assertFalse(small["coverage"]["scope_complete"])
        self.assertFalse(small["data_complete"])
        self.assertEqual(small["behavior"][0]["metrics"], {})

    def test_positive_benefits_and_order_control_metrics(self):
        result = analyze(fixture())
        self.assertEqual(result["behavior"][0]["metrics"]["accuracy"]["estimate"], 1)
        for contrast in result["history_contrasts"]:
            self.assertEqual(contrast["metrics"]["accuracy_improvement"]["estimate"], 1)
            self.assertGreater(contrast["metrics"]["loss_improvement_nats"]["estimate"], 0)
        self.assertEqual(result["history_contrasts"][-1]["comparison"], "chronological_order")
        self.assertTrue(result["bootstrap"]["orders_paired"])

    def test_candidate_order_reversal_invariant_behavior(self):
        inputs = fixture()
        original = analyze(inputs)
        rows, labels, protocol = deepcopy(inputs)
        for row in rows:
            row["order"] = 1 - row["order"]
            row["a_logprob"], row["b_logprob"] = row["b_logprob"], row["a_logprob"]
            row["margin_ab_bits"] *= -1
            for role in ("donor", "recipient"):
                if f"{role}_score" in row:
                    row[f"{role}_margin_ab_bits"] *= -1
                    score = row[f"{role}_score"]
                    score["a_logprob"], score["b_logprob"] = score["b_logprob"], score["a_logprob"]
                    score["margin_ab_bits"] *= -1
            for certificate in row.get("controls", {}).values():
                if certificate is not None:
                    score = certificate["scores"]
                    score["a_logprob"], score["b_logprob"] = score["b_logprob"], score["a_logprob"]
                    score["margin_ab_bits"] *= -1
        reversed_result = analyze((rows, labels, protocol))
        self.assertEqual(original["behavior"], reversed_result["behavior"])
        self.assertEqual(original["history_contrasts"], reversed_result["history_contrasts"])
        self.assertEqual(profile(original)["recovery"], profile(reversed_result)["recovery"])
        for name in ("gold_signed_margin_movement_bits", "gold_signed_donor_discrepancy_bits"):
            self.assertEqual(profile(original)["metrics"][name], profile(reversed_result)["metrics"][name])

    def test_conditional_loss_and_accuracy_are_not_vocabulary_probability(self):
        inputs = fixture()
        expected = analyze(inputs)
        rows, labels, protocol = deepcopy(inputs)
        for row in rows:
            nested = [c["scores"] for c in row.get("controls", {}).values() if c is not None]
            for score in (row, row.get("donor_score", {}), row.get("recipient_score", {}), *nested):
                for key in ("a_logprob", "b_logprob"):
                    if key in score:
                        score[key] -= 100
        actual = analyze((rows, labels, protocol))
        for before, after in zip(expected["behavior"], actual["behavior"]):
            for metric in ("accuracy", "loss_nats"):
                self.assertAlmostEqual(before["metrics"][metric]["estimate"], after["metrics"][metric]["estimate"])

    def test_null_is_valid_and_zero_discrepancy_recovery_not_fabricated(self):
        result = analyze(fixture(null=True))
        self.assertTrue(result["analysis_complete"])
        for contrast in result["history_contrasts"]:
            for metric in contrast["metrics"].values():
                self.assertEqual(metric["estimate"], 0)
                self.assertEqual(metric["ci95"], [0, 0])
                self.assertTrue(metric["valid"])
        for p in result["patch_profiles"]:
            recovery = p["recovery"]
            self.assertIsNone(recovery["estimate"])
            self.assertIsNone(recovery["ci95"])
            self.assertEqual(recovery["zero_denominator_rows"], 8)
            self.assertEqual(recovery["zero_denominator_clusters"], 4)
            self.assertEqual(recovery["excluded_rows"], 0)
            self.assertEqual(recovery["reason"], "small_or_zero_denominator")
        json.dumps(result, allow_nan=False)

    def test_ties_and_single_wearer_have_no_manufactured_ci(self):
        inputs = fixture(1, margins=lambda i, o: dict.fromkeys(("B0", "B1", "B4", "B8"), 0))
        result = analyze(inputs)
        self.assertTrue(result["data_complete"])
        for condition in result["behavior"]:
            self.assertEqual(condition["metrics"]["accuracy"]["estimate"], .5)
            self.assertAlmostEqual(condition["metrics"]["loss_nats"]["estimate"], math.log(2))
            self.assertIsNone(condition["metrics"]["accuracy"]["ci95"])
            self.assertEqual(condition["metrics"]["accuracy"]["n_clusters"], 1)

    def test_raw_signed_cancellation_preserved_and_orders_exposed(self):
        result = analyze(fixture(2, margins=lambda i, o: {
            "B0": (1 if o == 0 else -1) * 2,
            "B1": 0, "B4": 0, "B8": 0}))
        p = profile(result)
        self.assertEqual(p["metrics"]["signed_margin_movement_bits"]["estimate"], 0)
        self.assertEqual(p["metrics"]["signed_margin_movement_bits"]["ci95"], [0, 0])
        self.assertEqual(p["metrics"]["signed_margin_movement_bits"]["n_clusters"], 2)
        self.assertEqual(p["metrics"]["signed_donor_discrepancy_bits"]["estimate"], 0)
        self.assertEqual(p["metrics"]["absolute_donor_discrepancy_bits"]["estimate"], 2)
        self.assertGreater(p["by_order"][0]["metrics"]["signed_margin_movement_bits"]["estimate"], 0)
        self.assertLess(p["by_order"][1]["metrics"]["signed_margin_movement_bits"]["estimate"], 0)
        self.assertAlmostEqual(p["recovery"]["estimate"], .275)

    def test_ratio_of_cluster_means_and_directions_not_pooled(self):
        result = analyze(fixture(2, margins=lambda i, o: {
            "B0": (1 if o == 0 else -1) * (1 if i == 0 else 3),
            "B1": 0, "B4": 0, "B8": 0}))
        forward = profile(result)["recovery"]
        reverse = profile(result, direction="B4->B0")["recovery"]
        self.assertAlmostEqual(forward["estimate"], (.25 * 1 + .3 * 3) / 4)
        self.assertNotAlmostEqual(forward["estimate"], (.25 + .3) / 2)
        self.assertAlmostEqual(reverse["estimate"], .1)
        self.assertAlmostEqual(profile(result, positions="A")["recovery"]["estimate"], 1)
        self.assertLess(profile(result, positions="K")["recovery"]["estimate"], 0)

    def test_gold_margins_do_not_cancel_opposite_AB_signs_across_orders(self):
        result = analyze(fixture(2))
        self.assertAlmostEqual(result["behavior"][0]["metrics"]["gold_margin_bits"]["estimate"], 1.05)
        forward = profile(result)
        self.assertEqual(forward["metrics"]["signed_margin_movement_bits"]["estimate"], 0)
        self.assertEqual(forward["metrics"]["signed_donor_discrepancy_bits"]["estimate"], 0)
        for name, expected in (("gold_signed_margin_movement_bits", .58),
                               ("gold_signed_donor_discrepancy_bits", 2.1)):
            self.assertAlmostEqual(forward["metrics"][name]["estimate"], expected)
            self.assertGreater(forward["metrics"][name]["ci95"][0], 0)
            self.assertTrue(all(v > 0 for v in forward["metrics"][name]["wearer_values"].values()))
            for by_order in forward["by_order"]:
                self.assertAlmostEqual(by_order["metrics"][name]["estimate"], expected)
        reverse = profile(result, direction="B4->B0")
        self.assertAlmostEqual(reverse["metrics"]["gold_signed_margin_movement_bits"]["estimate"], -.21)
        self.assertAlmostEqual(reverse["metrics"]["gold_signed_donor_discrepancy_bits"]["estimate"], -2.1)

    def test_gold_orientation_uses_label_not_donor_direction(self):
        rows, labels, protocol = fixture(2)
        original = analyze((rows, labels, protocol))
        flipped = analyze((rows, [{**label, "correct_candidate": 1 - label["correct_candidate"]}
                                  for label in labels], protocol))
        self.assertAlmostEqual(flipped["behavior"][0]["metrics"]["gold_margin_bits"]["estimate"], -1.05)
        for direction in ("B0->B4", "B4->B0"):
            before, after = profile(original, direction), profile(flipped, direction)
            for name in ("gold_signed_margin_movement_bits", "gold_signed_donor_discrepancy_bits"):
                self.assertAlmostEqual(before["metrics"][name]["estimate"], -after["metrics"][name]["estimate"])
            self.assertEqual(before["recovery"], after["recovery"])
            self.assertEqual(before["metrics"]["signed_margin_movement_bits"], after["metrics"]["signed_margin_movement_bits"])

    def test_recovery_guards_resampled_denominator_without_selecting_clusters(self):
        result = _recovery({"a": 0, "b": 1}, {"a": 0, "b": 2}, 200, 5, .01)
        self.assertEqual(result["estimate"], .5)
        self.assertEqual(result["small_denominator_clusters"], 1)
        self.assertEqual(result["excluded_clusters"], 0)
        self.assertGreater(result["excluded_bootstrap_draws"], 0)
        self.assertEqual(result["valid_bootstrap_draws"] + result["excluded_bootstrap_draws"], 200)
        self.assertIsNone(result["ci95"])
        small = _recovery({"a": .001, "b": .001}, {"a": .002, "b": .002}, 20, 5, .01)
        self.assertIsNone(small["estimate"])
        self.assertEqual(small["small_denominator_clusters"], 2)

    def test_overshoot_is_residual_distance_reduction_not_signed_fraction(self):
        rows, labels, protocol = fixture()
        for row in rows:
            if (row["kind"] == "patch" and row["donor_condition"] == "B0"
                    and row["position_set"] == "H" and row["cut"] == 22):
                donor, recipient = row["donor_margin_ab_bits"], row["recipient_margin_ab_bits"]
                row.update(scores(recipient + 2.5 * (donor - recipient)))
        p = profile(analyze((rows, labels, protocol)))
        self.assertAlmostEqual(p["recovery"]["estimate"], -.5)
        self.assertLess(p["metrics"]["absolute_residual_reduction_bits"]["estimate"], 0)

    def test_negative_history_benefit_is_retained(self):
        result = analyze(fixture(2, margins=lambda i, o: {
            "B0": -(1 if o == i % 2 else -1),
            **{c: (1 if o == i % 2 else -1) for c in ("B1", "B4", "B8")}}))
        self.assertTrue(result["analysis_complete"])
        for contrast in result["history_contrasts"]:
            self.assertEqual(contrast["metrics"]["accuracy_improvement"]["estimate"], -1)
            self.assertLess(contrast["metrics"]["loss_improvement_nats"]["estimate"], 0)

    def test_empty_and_oversized_canary_are_incomplete(self):
        rows, labels, protocol = fixture()
        empty = analyze(([], [], {**protocol, "episode_ids": []}))
        self.assertFalse(empty["data_complete"])
        self.assertTrue(all(not b["metrics"] for b in empty["behavior"]))
        oversized = analyze(fixture(5))
        self.assertFalse(oversized["coverage"]["scope_complete"])
        self.assertTrue(all(not b["metrics"] for b in oversized["behavior"]))

    def test_parent_dispatch_uses_same_interface(self):
        from research.lifecycle_memory.temporal_forecasting.analysis import analyze as dispatch

        inputs = fixture()
        self.assertEqual(dispatch(*inputs, "current-canary", draws=8), analyze(inputs, draws=8))

    def test_missing_and_failed_rows_never_shrink_estimation_cohort(self):
        rows, labels, protocol = fixture()
        absent = next(r for r in rows if r["kind"] == "behavior" and r["condition"] == "B8")
        rows.remove(absent)
        failed = next(r for r in rows if r["kind"] == "patch" and r["position_set"] == "H")
        failed["status"] = "failed"
        result = analyze((rows, labels, protocol))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["coverage"]["missing_rows"], 1)
        self.assertEqual(result["coverage"]["failed_rows"], 1)
        self.assertEqual(result["history_contrasts"][-1]["metrics"], {})
        self.assertEqual(result["behavior"][-1]["metrics"], {})
        self.assertTrue(result["behavior"][0]["metrics"])
        self.assertTrue(all(not p["metrics"] for p in result["patch_profiles"]))

    def test_absent_whole_episode_is_not_silently_omitted(self):
        rows, labels, protocol = fixture()
        rows = [r for r in rows if r["episode_id"] != labels[-1]["episode_id"]]
        result = analyze((rows, labels, protocol))
        self.assertEqual(result["coverage"]["missing_rows"], 44)
        self.assertEqual(result["coverage"]["expected_episodes"], 4)
        self.assertTrue(all(not b["metrics"] for b in result["behavior"]))

    def test_self_and_full_A_controls_check_common_mode_logprob_drift(self):
        for kind in ("self", "patch"):
            with self.subTest(kind=kind):
                rows, labels, protocol = fixture()
                control = next(r for r in rows if r["kind"] == kind and r["position_set"] == "A")
                control["a_logprob"] -= .01
                control["b_logprob"] -= .01
                result = analyze((rows, labels, protocol))
                self.assertTrue(result["data_complete"])
                self.assertFalse(result["analysis_complete"])
                self.assertFalse(result["technical_controls"]["passed"])
                self.assertEqual(result["technical_controls"]["failed_checks"], 1)
                self.assertTrue(all(not p["metrics"] for p in result["patch_profiles"]))

    def test_patch_native_baseline_and_reported_technical_failure_checked(self):
        for change in ({"donor_margin_ab_bits": 99}, {"technical": {"alignment_pass": False}},
                       {"donor_score": None}, {"technical": {"self_max_drift_bits": .002}}):
            with self.subTest(change=change):
                rows, labels, protocol = fixture()
                next(r for r in rows if r["kind"] == "patch").update(change)
                result = analyze((rows, labels, protocol))
                self.assertFalse(result["technical_controls"]["passed"])
                self.assertTrue(all(not p["metrics"] for p in result["patch_profiles"]))

    def test_missing_controls_fail_closed(self):
        for kind in ("self", "patch"):
            with self.subTest(kind=kind):
                rows, labels, protocol = fixture()
                control = next(r for r in rows if r["kind"] == kind and r["position_set"] == "A")
                rows.remove(control)
                result = analyze((rows, labels, protocol))
                self.assertFalse(result["technical_controls"]["passed"])
                self.assertTrue(all(not p["metrics"] for p in result["patch_profiles"]))

    def test_public_technical_gate_is_label_free_and_does_not_compute_statistics(self):
        rows, _, _ = fixture()
        before = deepcopy(rows)
        module = "research.lifecycle_memory.temporal_forecasting.dense_analysis"
        with patch(module + "._behavior_metrics", side_effect=AssertionError("scientific metric")), \
                patch(module + "._bootstrap", side_effect=AssertionError("bootstrap")), \
                patch("research.lifecycle_memory.temporal_forecasting.data._labels", side_effect=AssertionError("labels")):
            report = validate_dense_technical(rows)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["n_checks"], 144)
        self.assertEqual(report["failed_checks"], 0)
        self.assertEqual(report["failure_count"], 0)
        self.assertEqual(report["failures"], [])
        self.assertEqual(rows, before)
        json.dumps(report, allow_nan=False)

    def test_public_technical_gate_catches_independent_native_score_drift(self):
        rows, _, _ = fixture()
        native = next(r for r in rows if r["kind"] == "behavior" and r["condition"] == "B0")
        native["a_logprob"] -= .01
        native["b_logprob"] -= .01
        report = validate_dense_technical(rows)
        self.assertEqual(report["status"], "fail")
        self.assertGreater(report["failed_checks"], 0)
        self.assertTrue(any("donor_score_mismatch" in c["errors"] for c in report["failures"]))

    def test_mandatory_runtime_evidence_is_required_by_both_entry_points(self):
        inputs = fixture(1)
        target = next(r for r in inputs[0] if r["kind"] == "patch")
        for key in target["technical"]:
            rows, labels, protocol = deepcopy(inputs)
            next(r for r in rows if r["kind"] == "patch")["technical"].pop(key)
            with self.subTest(key=key):
                self.assertEqual(validate_dense_technical(rows)["status"], "fail")
                self.assertFalse(analyze((rows, labels, protocol))["technical_controls"]["passed"])
        for field in ("technical", "controls", "donor_score", "recipient_score"):
            rows, _, _ = deepcopy(inputs)
            next(r for r in rows if r["kind"] == "patch").pop(field)
            self.assertEqual(validate_dense_technical(rows)["status"], "fail", field)
        rows, _, _ = deepcopy(inputs)
        next(r for r in rows if r["kind"] == "patch")["technical"] = {}
        self.assertEqual(validate_dense_technical(rows)["status"], "fail")

    def test_nested_control_failure_and_score_drift_are_not_overridden_by_flags(self):
        for name in ("recipient_self", "donor_self", "full_A", "selected_self"):
            rows, labels, protocol = fixture(1)
            row = next(r for r in rows if r["kind"] == ("self" if name == "selected_self" else "patch"))
            row["controls"][name]["passed"] = False
            with self.subTest(control=name):
                report = validate_dense_technical(rows)
                self.assertEqual(report["status"], "fail")
                self.assertEqual(report["failed_checks"], 1)
                self.assertFalse(analyze((rows, labels, protocol))["technical_controls"]["passed"])
        rows, _, _ = fixture(1)
        certificate = next(r for r in rows if r["kind"] == "patch")["controls"]["full_A"]
        certificate["scores"]["a_logprob"] -= .01
        certificate["scores"]["b_logprob"] -= .01
        self.assertEqual(validate_dense_technical(rows)["status"], "fail")

    def test_public_technical_gate_rejects_empty_missing_duplicate_and_invalid_scores(self):
        self.assertEqual(validate_dense_technical([])["status"], "fail")
        rows, _, _ = fixture(1)
        self.assertEqual(validate_dense_technical(rows[:-1])["status"], "fail")
        self.assertEqual(validate_dense_technical(rows + [deepcopy(rows[0])])["status"], "fail")
        for key in ("a_logprob", "b_logprob", "margin_ab_bits"):
            changed = deepcopy(rows)
            changed[0][key] = float("nan")
            report = validate_dense_technical(changed)
            self.assertEqual(report["status"], "fail")
            json.dumps(report, allow_nan=False)

    def test_duplicate_unknown_cells_mixed_clusters_and_missing_label_rejected(self):
        changes = [lambda rows: rows.append(deepcopy(rows[0])),
                   lambda rows: rows[0].update(condition="B2"),
                   lambda rows: rows[0].update(model_key=protocols.QWEN),
                   lambda rows: rows[0].update(order=True),
                   lambda rows: rows[0].update(phase="confirmation"),
                   lambda rows: rows[0].update(contract_sha256="d" * 64),
                   lambda rows: next(r for r in rows if r["kind"] == "patch").update(cut=24),
                   lambda rows: next(r for r in rows if r["kind"] == "patch").update(position_set="Q"),
                   lambda rows: rows[0].update(a_logprob=float("nan")),
                   lambda rows: rows[0].update(margin_ab_bits=100),
                   lambda rows: [r.update(cluster_id="same") for r in rows]]
        for change in changes:
            rows, labels, protocol = fixture()
            change(rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze((rows, labels, protocol))
        rows, labels, protocol = fixture()
        with self.assertRaises(ValueError):
            analyze((rows, labels[:-1], protocol))

    def test_protocol_drift_phase_and_draws_rejected(self):
        for change in ({"schema_version": 2}, {"study": "other"}, {"fixed_cuts": [20, 34]},
                       {"task": protocols.FORECAST}, {"history_offsets_s": [-2, -1, 0]},
                       {"rho_guard_bits": -1}, {"bootstrap_seed": True}):
            rows, labels, protocol = fixture()
            protocol.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze((rows, labels, protocol))
        for draws in (0, 1, True, 2.5):
            with self.subTest(draws=draws), self.assertRaises(ValueError):
                analyze(fixture(), draws=draws)
        with self.assertRaises(ValueError):
            analyze(fixture(), phase="confirmation")

    def test_reproducible_under_row_and_label_order_and_alias(self):
        inputs = fixture()
        expected = analyze(inputs)
        rows, labels, protocol = deepcopy(inputs)
        for row in rows:
            row["model_key"] = "gemma31"
        actual = analyze((list(reversed(rows)), list(reversed(labels)), protocol))
        for key in ("behavior", "history_contrasts", "patch_profiles"):
            self.assertEqual(expected[key], actual[key])


if __name__ == "__main__":
    unittest.main()
