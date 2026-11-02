"""Independent CPU scientific-oracle, saved-result, privacy and tamper tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import replay_practical as replay
from analysis.export_practical_replay import privacy_check, project_units

DIRECTORY = ROOT / "results/replay_practical"


def read(name):
    return json.loads((DIRECTORY / name).read_text())


def synthetic_rows(data):
    """Entirely invented requests; no private joins or reconstructed identities."""
    public, predictions, timings, labels = [], [], [], {}
    names = replay.inventory(data["session"])[0]
    for i, unit in enumerate(data["units"]):
        for j in range(unit["requests"]):
            rid = f"invented-request-{i:03d}-{j:03d}"
            row = {"request_id": rid, "frame_key": f"invented-frame-{i:03d}-{j//2:03d}",
                   "cluster_key": f"invented-unit-{i:03d}", "source_family": unit["source"],
                   "order_id": f"order_{j % 2}"}
            public.append(row)
            labels[rid] = "AB"[(i + j) % 2]
            for v, name in enumerate(names):
                margin = ((i * 7 + j * 3 + v * 11) % 37 - 18) / 3
                predictions.append({**row, "variant": name, "margin_ab": margin,
                                    "prediction": "A" if margin >= 0 else "B"})
                if j < unit["timing_pairs"] // 5:
                    for repeat in range(5):
                        forward = 20 + i / 3 + j / 7 + v / 2 + repeat / 9
                        timings.append({**row, "variant": name, "repeat_id": repeat,
                            "device_id": 0, "batch_size": 1, "elapsed_ms": forward,
                            "model_forward_ms": forward, "preparation_ms": 2., "plan_lookup_ms": .5,
                            "postprocess_ms": .25, "end_to_end_elapsed_ms": forward + 2.75,
                            "peak_allocated_bytes": 1024 + 64 * i + j,
                            "peak_reserved_bytes": 8192 + 128 * i + j})
    return public, predictions, timings, labels


class PracticalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {s: read(f"{s}.json") for s in replay.SESSIONS}
        cls.actual = {s: replay.replay(d)[0] for s, d in cls.data.items()}

    def test_all_saved_fields_and_manifest(self):
        manifest = read("manifest.json")
        for session, data in self.data.items():
            expected = read(f"{session}_expected.json")
            replay.compare(self.actual[session], expected)
            self.assertEqual(manifest["sessions"][session], {
                "data_sha256": replay.canonical_digest(data), "expected_sha256": replay.canonical_digest(expected)})

    def test_independent_original_scientific_analyzers(self):
        from research.lifecycle_memory import role_aware_policy, role_aware_transfer, decoder_cached_validation

        for session, data in self.data.items():
            with self.subTest(session=session):
                public, predictions, timings, labels = synthetic_rows(data)
                projected = project_units(session, public, predictions, timings, labels)
                actual, _ = replay._replay(projected)
                if session == "621156":
                    config = {"bootstrap": {"draws": 10000, "seed": 20261015},
                              "fresh_data": {"scope": "cross_source_transfer"}}
                    expected = decoder_cached_validation._summaries(
                        public, predictions, timings, {r: {"correct_label": l} for r, l in labels.items()},
                        config, source_equal=True)
                    decoder_cached_validation._descriptive_summary(expected)
                else:
                    module = role_aware_policy if session == "6469516" else role_aware_transfer
                    expected = module.summarize(public, predictions, timings, labels)
                replay.compare(actual, replay.project_summary(expected))

    def test_historical_failures_are_not_successes(self):
        combined = self.actual["6469516"]["decision"]["co_primaries"]
        self.assertFalse(combined["E1"]["upper_left_pass"])
        self.assertFalse(combined["E2"]["upper_left_pass"])
        transfer = self.actual["6478505"]["decision"]
        self.assertTrue(transfer["co_primaries"]["seed17"]["upper_left_pass"])
        self.assertFalse(transfer["co_primaries"]["seed29"]["accuracy_pass"])
        self.assertFalse(transfer["both_seeds_pass"])
        self.assertFalse(self.actual["621156"]["primary_decisions"]["eligible"])

    def test_native_baselines_and_retention_are_session_local(self):
        native = [self.actual[s]["timing"]["native"]["request_ms"] for s in replay.SESSIONS[:2]]
        self.assertNotEqual(*native)
        result = self.actual["6478505"]
        for seed in (17, 29):
            policy, base = f"F133_L34_readout_seed{seed}", f"native_readout_seed{seed}"
            for metric in replay.METRICS:
                estimate = result["metrics"][policy][metric]["estimate"] - result["metrics"][base][metric]["estimate"]
                self.assertAlmostEqual(result["retention_diagnostics"][policy][metric]["estimate"], estimate)

    def test_missing_timing_excluded_and_pair_counts(self):
        for data in self.data.values():
            self.assertEqual(len(data["timed_positions"]), 44)
            self.assertEqual(sum(u["timing_pairs"] == 0 for u in data["units"]), 6)
            self.assertEqual(sum(u["requests"] for u in data["units"]), 894)
            self.assertEqual(sum(u["timing_pairs"] for u in data["units"]), 128 * 5)
            self.assertTrue(all(len(v) == 44 for cell in data["variants"].values() for v in cell["timing"].values()))

    def test_independent_historical_rng_order(self):
        data = self.data["6478505"]
        units, timed = data["units"], data["timed_positions"]
        rng = np.random.default_rng(20261019)
        for s in replay.SOURCES:
            n = sum(u["source"] == s for u in units)
            rng.integers(0, n, size=(10000, n))
        naive_rng = np.random.default_rng(20261019)
        arrays, naive = [], []
        name = "F133_L34_readout_seed29"
        for generator, destination in ((rng, arrays), (naive_rng, naive)):
            for s in replay.SOURCES:
                group = np.array([j for j, i in enumerate(timed) if units[i]["source"] == s])
                indices = group[generator.integers(0, len(group), size=(10000, len(group)))]
                destination.append([np.array(data["variants"][v]["timing"]["end_to_end_elapsed_ms"])[indices].mean(1)
                                    for v in ("native", name)])
        correct = np.mean(arrays, axis=0)
        wrong = np.mean(naive, axis=0)
        ci = np.quantile(correct[0] / correct[1], [.025, .975]).tolist()
        self.assertEqual(ci, self.actual["6478505"]["timing"][name]["speedup_ci95"])
        self.assertNotEqual(ci, np.quantile(wrong[0] / wrong[1], [.025, .975]).tolist())

    def test_schema_and_numeric_tampering_rejected(self):
        changes = [lambda d: d["bootstrap"].update(seed=29),
                   lambda d: d["units"][0].update(request_id="private"),
                   lambda d: d["timed_positions"].append(d["timed_positions"][-1]),
                   lambda d: d["variants"]["native"]["scores"]["accuracy"].pop(),
                   lambda d: d["variants"]["native"]["scores"]["accuracy"].__setitem__(0, float("nan")),
                   lambda d: d["variants"]["native"]["timing"]["end_to_end_elapsed_ms"].__setitem__(0, 0)]
        for change in changes:
            data = deepcopy(self.data["6478505"])
            change(data)
            with self.assertRaises(ValueError):
                replay.validate(data)

    def test_valid_numeric_tamper_does_not_replace_reference(self):
        data = deepcopy(self.data["6478505"])
        data["variants"]["native"]["scores"]["accuracy"][0] -= .001
        actual, _ = replay.replay(data)
        with self.assertRaisesRegex(ValueError, "numeric mismatch"):
            replay.compare(actual, read("6478505_expected.json"))

    def test_checksum_tampering_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for path in DIRECTORY.glob("*.json"):
                (directory / path.name).write_bytes(path.read_bytes())
            data = deepcopy(self.data["6469516"])
            data["variants"]["native"]["scores"]["accuracy"][0] += .001
            (directory / "6469516.json").write_text(json.dumps(data))
            output = directory / "output.json"
            with self.assertRaisesRegex(ValueError, "checksum"):
                replay.run(directory, output)
            self.assertFalse(output.exists())

    def test_raw_duplicate_and_unpaired_timing_rejected(self):
        rows = synthetic_rows(self.data["6478505"])
        public, predictions, timings, labels = rows
        with self.assertRaisesRegex(ValueError, "prediction inventory"):
            project_units("6478505", public, predictions + predictions[:1], timings, labels)
        with self.assertRaisesRegex(ValueError, "unpaired timing"):
            project_units("6478505", public, predictions, timings[:-1], labels)

    def test_privacy_and_no_weights(self):
        for path in DIRECTORY.iterdir():
            self.assertEqual(path.suffix, ".json")
            privacy_check(json.loads(path.read_text()))
        for bad in ({"request_id": "x"}, {"frame_key": "x"}, {"a": "/home/" + "private/data"},
                    {"source_hashes": []}, {"unit_ids": []}):
            with self.assertRaises(ValueError):
                privacy_check(bad)

    def test_cli_determinism_and_aggregate_preservation(self):
        paths = list((ROOT / "results/aggregate").glob("*.json"))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "practical.json"
            subprocess.run([sys.executable, str(ROOT / "analysis/replay_practical.py"), "--output", str(output)],
                           cwd=tmp, check=True, capture_output=True, timeout=90)
            self.assertEqual(output.read_bytes(), (ROOT / "results/rebuilt/practical_replay.json").read_bytes())
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__ == "__main__":
    unittest.main()
