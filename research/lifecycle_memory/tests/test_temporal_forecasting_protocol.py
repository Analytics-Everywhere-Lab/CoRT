from copy import deepcopy
import unittest

from research.lifecycle_memory.temporal_forecasting.protocol import (
    GEMMA, QWEN, assign_tasks, default_protocol, inventory, selected_episodes, validate_protocol,
)


def public_rows(split, count):
    return [{"episode_id": f"e{i}", "cluster_id": f"c{i}", "split": split} for i in range(count)]


class ProtocolTests(unittest.TestCase):
    def test_full_survey_count_and_zero_based_cuts(self):
        tasks = inventory(public_rows("exploration", 60), default_protocol(), "exploration")
        patches = [t for t in tasks if t["kind"] == "patch"]
        self.assertEqual(len(patches), 59520)
        self.assertEqual(len(tasks), 69360)
        for model, maximum in ((GEMMA, 58), (QWEN, 62)):
            cuts = {t["cut"] for t in patches if t["model_key"] == model}
            self.assertEqual(cuts, set(range(0, maximum + 1, 2)))

    def test_pilot(self):
        tasks = inventory(public_rows("pilot", 100), default_protocol(), "pilot")
        self.assertEqual(len(tasks), 800)
        self.assertEqual({t["kind"] for t in tasks}, {"behavior"})
        self.assertEqual({t["model_key"] for t in tasks}, {GEMMA})

    def test_canary_exercises_all_classes_and_both_models(self):
        tasks = inventory(public_rows("exploration", 60), default_protocol(), "exploration-canary")
        self.assertEqual({t["kind"] for t in tasks}, {"behavior", "patch", "self"})
        self.assertEqual({t["model_key"] for t in tasks}, {GEMMA, QWEN})
        self.assertEqual(len(tasks), 592)

    def test_assignments_independent_of_gpu_count(self):
        tasks = inventory(public_rows("exploration", 2), default_protocol(), "exploration-canary")
        for n in (1, 2, 4, 8):
            parts = assign_tasks(tasks, n)
            self.assertEqual(sorted(t["task_id"] for p in parts for t in p), sorted(t["task_id"] for t in tasks))
            if n >= 2:
                self.assertTrue(all(len({t["model_key"] for t in p}) <= 1 for p in parts))

    def test_confirmation_needs_freeze_and_enough_clusters(self):
        protocol = default_protocol()
        rows = public_rows("confirmation", 46)
        with self.assertRaisesRegex(ValueError, "frozen"):
            inventory(rows, protocol, "confirmation")
        freeze = {"cuts": {GEMMA: {"early_cut": 22, "carrier_cut": 26, "late_cut": 34},
                            QWEN: {"early_cut": 24, "carrier_cut": 28, "late_cut": 36}}}
        tasks = inventory(rows, protocol, "confirmation", freeze)
        self.assertEqual(len(tasks), 16744)
        unions = {t["position_set"] for t in tasks if "position_set" in t and set(t["position_set"].split("+")) <= set("CQPR")}
        self.assertEqual(len(unions), 15)
        with self.assertRaisesRegex(ValueError, "clusters"):
            inventory(rows[:45], protocol, "confirmation", freeze)

    def test_no_pruning_or_mutable_model_pin(self):
        p = default_protocol()
        p["pruning_enabled"] = True
        with self.assertRaisesRegex(ValueError, "pruning"):
            validate_protocol(p)
        p = deepcopy(default_protocol())
        p["models"][GEMMA]["revision"] = "changed"
        with self.assertRaisesRegex(ValueError, "pins"):
            validate_protocol(p)

    def test_selection_independent_of_input_order(self):
        rows = public_rows("pilot", 102)
        p = default_protocol()
        self.assertEqual(selected_episodes(rows, p, "pilot"), selected_episodes(list(reversed(rows)), p, "pilot"))


if __name__ == "__main__":
    unittest.main()
