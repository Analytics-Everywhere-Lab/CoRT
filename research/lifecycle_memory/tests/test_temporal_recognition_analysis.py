"""Small, synthetic evaluator fixtures; no models, media, or production receipts."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from research.lifecycle_memory.temporal_forecasting.common import digest
from research.lifecycle_memory.temporal_forecasting.protocol import GEMMA, inventory, protocol_contract, selected_episodes
from research.lifecycle_memory.temporal_forecasting.recognition_protocol import TASKS, recognition_protocol
from research.lifecycle_memory.temporal_forecasting.recognition_analysis import (
    analyze_recognition, validate_recognition_technical,
)
from research.lifecycle_memory.tests.test_temporal_dense_analysis import scores, runtime_evidence


POLICY = {"version": 1, "kind": "matched_static_reuse",
          "historical_static_sha256": "a" * 64, "temporal_exposure_sha256": "b" * 64}


def fixture(sizes=(3, 1, 1, 1), task=TASKS[0], phase="recognition", native=None, minimum_clusters=4):
    protocol = recognition_protocol(task, POLICY, episode_limit=sum(sizes), minimum_clusters=minimum_clusters)
    contract = protocol_contract(protocol)
    episodes = [{"episode_id": f"e{wearer}-{episode}", "cluster_id": f"w{wearer}",
                 "task": task, "contract_sha256": digest(contract)}
                for wearer, size in enumerate(sizes) for episode in range(size)]
    selected = selected_episodes(episodes, protocol, phase)
    tasks = inventory(episodes, protocol, phase)
    labels = [{"episode_id": e["episode_id"], "correct_candidate": 0, "label_status": "visually_audited",
               "reviewer": "synthetic-test", "review_evidence_sha256": "c" * 64,
               "review_receipt_sha256": "d" * 64}
              for e in selected]
    rows = []
    for item in tasks:
        sign = 1 if item["order"] == 0 else -1
        values = (native(item) if native else {"B0": 2. * sign, "B1": -.5 * sign,
                                             "B4": -sign, "B2": -2. * sign})
        row = {**item, "status": "pass", "elapsed_s": .01}
        if item["kind"] == "behavior":
            row.update(scores(values[item["condition"]]))
        else:
            donor, recipient = (values[item[f"{role}_condition"]] for role in ("donor", "recipient"))
            fraction = {"H": .75 if item["donor_condition"] == "B0" else .25, "V": .5, "K": -.25, "A": 1.}[item["position_set"]]
            row.update(scores(recipient + fraction * (donor - recipient)),
                       donor_margin_ab_bits=donor, recipient_margin_ab_bits=recipient,
                       donor_score=scores(donor), recipient_score=scores(recipient))
            row.update(runtime_evidence(donor, recipient, item["kind"] == "self"))
        rows.append(row)
    protocol["episode_ids"] = [e["episode_id"] for e in selected]
    return rows, labels, protocol


def analyze(values, phase="recognition", draws=12):
    return analyze_recognition(*values, phase, draws=draws)


def profile(result, direction="B0->B4", positions="H", cut=22):
    return next(row for row in result["patch_profiles"] if
                (row["direction"], row["position_set"], row["cut"]) == (direction, positions, cut))


class RecognitionAnalysisTests(unittest.TestCase):
    def test_both_tasks_exact_inventory_and_descriptive_only(self):
        for task in TASKS:
            values = fixture(task=task)
            before = deepcopy(values)
            result = analyze(values)
            self.assertEqual(values, before)
            self.assertTrue(result["analysis_complete"])
            self.assertEqual(result["coverage"]["expected_rows"], 6 * 44)
            self.assertEqual(result["coverage"]["observed_clusters"], 4)
            self.assertEqual(result["coverage"]["counts_by_kind"], {"behavior": 48, "patch": 192, "self": 24})
            self.assertEqual(result["cuts"], {GEMMA: [22, 34]})
            self.assertEqual(result["family_size"], 0)
            self.assertFalse(result["primary_complete"])
            self.assertEqual(len(result["patch_profiles"]), 16)
            for forbidden in ('"p_value"', '"reject"', '"holm_adjusted_p"'):
                self.assertNotIn(forbidden, json.dumps(result, allow_nan=False))
            self.assertEqual([r["control"] for r in result["history_contrasts"]], ["B4", "B1", "B2"])
            for row in result["history_contrasts"]:
                self.assertEqual(set(row["metrics"]), {"accuracy_improvement", "loss_improvement_nats", "gold_margin_improvement_bits"})
                self.assertTrue(all(v["estimate"] > 0 for v in row["metrics"].values()))

    def test_orders_then_episodes_then_participants_not_pooled_rows(self):
        def native(row):
            wearer = int(row["cluster_id"][1:])
            episode = int(row["episode_id"].split("-")[1])
            margin = (2 + 2 * episode + row["order"]) if wearer == 0 else 10
            return {"B0": margin * (1 if row["order"] == 0 else -1), "B1": 0., "B4": 0., "B2": 0.}
        result = analyze(fixture(native=native))
        metric = result["behavior"][0]["metrics"]["gold_margin_bits"]
        self.assertAlmostEqual(metric["wearer_values"]["w0"], 4.5)
        self.assertAlmostEqual(metric["estimate"], (4.5 + 30) / 4)
        self.assertEqual(metric["n_clusters"], 4)
        self.assertAlmostEqual(profile(result)["recovery"]["estimate"], .75)

    def test_bidirectional_recovery_and_full_A(self):
        result = analyze(fixture())
        for cut in (22, 34):
            self.assertAlmostEqual(profile(result, cut=cut)["recovery"]["estimate"], .75)
            self.assertAlmostEqual(profile(result, direction="B4->B0", cut=cut)["recovery"]["estimate"], .25)
            self.assertAlmostEqual(profile(result, positions="K", cut=cut)["recovery"]["estimate"], -.25)
            for direction in ("B0->B4", "B4->B0"):
                self.assertAlmostEqual(profile(result, direction, "A", cut)["recovery"]["estimate"], 1.)

    def test_order_reversal_preserves_gold_metrics_and_recovery(self):
        values = fixture()
        before = analyze(values)
        rows, labels, protocol = deepcopy(values)
        for row in rows:
            row["order"] = 1 - row["order"]
            nested = [row]
            for role in ("donor", "recipient"):
                if f"{role}_score" in row:
                    row[f"{role}_margin_ab_bits"] *= -1
                    nested.append(row[f"{role}_score"])
            nested.extend(c["scores"] for c in row.get("controls", {}).values() if c is not None)
            for score in nested:
                score["a_logprob"], score["b_logprob"] = score["b_logprob"], score["a_logprob"]
                score["margin_ab_bits"] *= -1
        after = analyze((rows, labels, protocol))
        self.assertEqual(before["behavior"], after["behavior"])
        self.assertEqual(before["history_contrasts"], after["history_contrasts"])
        self.assertEqual(profile(before)["recovery"], profile(after)["recovery"])

    def test_full_frozen_count_and_maximum_96_episodes(self):
        result = analyze(fixture(sizes=(3,) * 32), draws=2)
        self.assertTrue(result["analysis_complete"])
        self.assertEqual(result["coverage"]["expected_rows"], 4224)
        self.assertEqual(result["coverage"]["observed_clusters"], 32)
        rows, labels, protocol = fixture()
        protocol["episode_limit"] += 1
        protocol.pop("episode_ids")
        result = analyze((rows, labels, protocol))
        self.assertFalse(result["analysis_complete"])
        self.assertTrue(all(not row["metrics"] for row in result["behavior"]))

    def test_production_16_episode_eight_wearer_floor_is_protocol_bound(self):
        values = fixture(sizes=(3, 3, 2, 2, 2, 2, 1, 1), minimum_clusters=8)
        result = analyze(values)
        self.assertTrue(result["analysis_complete"])
        self.assertEqual(result["coverage"]["expected_rows"], 704)
        self.assertEqual(result["coverage"]["observed_clusters"], 8)
        self.assertEqual(result["bootstrap"]["seed"], 20261025)
        self.assertEqual(values[2]["bootstrap_draws"], 19999)
        rows, labels, protocol = deepcopy(values)
        for row in rows:
            if row["cluster_id"] == "w7":
                row["cluster_id"] = "w6"
        result = analyze((rows, labels, protocol))
        self.assertFalse(result["coverage"]["scope_complete"])
        self.assertFalse(result["analysis_complete"])

    def test_small_discrepancy_retains_every_participant(self):
        def native(row):
            value = 0. if row["cluster_id"] == "w0" else 2.
            sign = 1 if row["order"] == 0 else -1
            return {"B0": value * sign, "B1": 0., "B4": 0., "B2": 0.}
        recovery = profile(analyze(fixture(native=native), draws=64))["recovery"]
        self.assertEqual(recovery["n_clusters"], 4)
        self.assertEqual(recovery["excluded_clusters"], 0)
        self.assertEqual(recovery["excluded_rows"], 0)
        self.assertEqual(recovery["zero_denominator_clusters"], 1)
        self.assertEqual(recovery["zero_denominator_rows"], 6)
        self.assertAlmostEqual(recovery["estimate"], .75)
        zero = fixture(native=lambda row: dict.fromkeys(("B0", "B1", "B4", "B2"), 0.))
        result = analyze(zero)
        self.assertTrue(result["analysis_complete"])
        self.assertIsNone(profile(result)["recovery"]["estimate"])
        self.assertEqual(profile(result)["recovery"]["excluded_clusters"], 0)

    def test_missing_whole_episode_and_missing_order_suppress_not_drop(self):
        rows, labels, protocol = fixture()
        missing = [r for r in rows if r["episode_id"] != "e0-0"]
        result = analyze((missing, labels, protocol))
        self.assertFalse(result["analysis_complete"])
        self.assertEqual(result["coverage"]["missing_rows"], 44)
        self.assertFalse(validate_recognition_technical(missing, protocol["episode_ids"])["passed"])
        self.assertTrue(all(not b["metrics"] for b in result["behavior"]))
        missing = [r for r in rows if not (r["kind"] == "behavior" and r["condition"] == "B2" and r["order"] == 1)]
        result = analyze((missing, labels, protocol))
        self.assertFalse(result["analysis_complete"])
        self.assertEqual(result["behavior"][-1]["metrics"], {})

    def test_technical_checks_are_label_free_and_enforce_dense_certificates(self):
        rows, _, protocol = fixture()
        with patch("research.lifecycle_memory.temporal_forecasting.data._labels", side_effect=AssertionError("labels accessed")):
            self.assertTrue(validate_recognition_technical(rows, protocol["episode_ids"])["passed"])
        for field in ("controls", "technical", "donor_score"):
            bad = deepcopy(rows)
            next(row for row in bad if row["kind"] == "patch").pop(field)
            self.assertFalse(validate_recognition_technical(bad)["passed"])
        bad = deepcopy(rows)
        row = next(row for row in bad if row["kind"] == "behavior" and row["condition"] == "B0")
        row["a_logprob"] -= .01
        row["b_logprob"] -= .01
        self.assertFalse(validate_recognition_technical(bad)["passed"])

    def test_failed_control_suppresses_patches_but_keeps_behavior(self):
        rows, labels, protocol = fixture()
        next(row for row in rows if row["kind"] == "patch")["technical"]["hook_calls"] = 2
        result = analyze((rows, labels, protocol))
        self.assertFalse(result["analysis_complete"])
        self.assertTrue(result["behavior"][0]["metrics"])
        self.assertTrue(all(p["recovery"] is None for p in result["patch_profiles"]))

    def test_canary_requires_four_distinct_wearers(self):
        values = fixture(phase="recognition-canary")
        result = analyze(values, phase="recognition-canary")
        self.assertTrue(result["analysis_complete"])
        self.assertEqual(result["coverage"]["expected_rows"], 176)
        rows, labels, protocol = values
        for row in rows:
            if row["cluster_id"] == "w1":
                row["cluster_id"] = "w0"
        self.assertFalse(analyze((rows, labels, protocol), phase="recognition-canary")["analysis_complete"])

    def test_duplicate_unexpected_and_malformed_rows_fail_closed(self):
        rows, labels, protocol = fixture()
        for bad in (rows + [rows[0]], [], [{**rows[0], "order": True}]):
            self.assertFalse(validate_recognition_technical(bad)["passed"])
        for field, value in (("cut", 23), ("position_set", "Q"), ("recipient_condition", "B2")):
            bad = deepcopy(rows)
            next(r for r in bad if r["kind"] == "patch")[field] = value
            self.assertFalse(validate_recognition_technical(bad)["passed"])
        bad = deepcopy(rows)
        for row in bad:
            if row["cluster_id"] == "w1":
                row["cluster_id"] = "w0"
        with self.assertRaisesRegex(ValueError, "three episodes"):
            analyze((bad, labels, protocol))


if __name__ == "__main__":
    unittest.main()
