"""CPU/stdlib-only tests; no real inference, labels, network or scheduler calls."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from research.lifecycle_memory import role_aware_killarney_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "out"
        self.out.mkdir()
        self.config = self.out / "frozen_config.json"
        self.config.write_text('{"fixture": true}')
        self.events = []
        self.measurements = 0
        self.drift_at = None

    def factory(self, spec, *args, **kwargs):
        def forward(*unused):
            raise AssertionError("No model forward is allowed in these tests")
        return forward

    def measure(self, engine, row, data_dir, arm, cache, **kwargs):
        self.measurements += 1
        self.events.append("timer_stopped")
        value = 2.000001 if self.measurements == self.drift_at else 2.0
        result = {"logits_ab": [value, 1.0], "prediction": "A"}
        metrics = {"geometry_sha256": "g", "schedule_sha256": arm["id"],
                   "end_to_end_elapsed_ms": 123.0, "model_forward_ms": 100.0}
        return result, metrics

    def fake_runtime(self, stage, mode, rank, *, base):
        self.assertEqual(stage, "evaluate")
        directory = Path(base) / f"out/evaluate/rank_{rank:03d}"
        directory.mkdir(parents=True)
        variants = runtime.runner.rap.variant_specs()
        forwards = {s["id"]: runtime.runner.make_forward(s) for s in variants}
        predictions, timings = [], []

        def call(rid, spec, phase):
            result, metrics = runtime.benchmark.benchmark_measurement(
                None, {"request_id": rid}, None, {"id": spec["compression_id"]}, None,
                phase=phase, forward=forwards[spec["id"]])
            self.assertEqual(metrics["end_to_end_elapsed_ms"], 123.0)
            return {"request_id": rid, "variant": spec["id"], "rank": rank, **metrics, **result}

        for spec in variants:
            for _ in range(2):
                call("order0", spec, "warmup")
        for spec in reversed(variants):
            call("order0", spec, "timed")
        for rid in ("order0", "order1"):
            for spec in reversed(variants):
                predictions.append(call(rid, spec, "timed"))
        repeat_count = 2 if mode == "canary" else 5
        for repeat in range(repeat_count):
            for rid in ("order0", "order1"):
                for spec in variants:
                    row = call(rid, spec, "timed")
                    row.pop("logits_ab")
                    timings.append({**row, "repeat_id": repeat})
        for name, records in (("predictions.jsonl", predictions), ("timings.jsonl", timings)):
            (directory / name).write_text("".join(json.dumps(r) + "\n" for r in records))
        runtime.base.safe.atomic_json(directory / "worker_receipt.json", {
            "status": "pass", "mode": mode, "rank": rank, "config_sha256": runtime.runner.sha256(self.config),
            "predictions": len(predictions), "timings": len(timings)})
        return {"status": "pass"}

    def evaluate(self, mode="canary", rank=0):
        with patch.object(runtime.runner, "make_forward", side_effect=self.factory), \
                patch.object(runtime.benchmark, "benchmark_measurement", side_effect=self.measure), \
                patch.object(runtime.base, "runtime", side_effect=self.fake_runtime):
            original_factory = runtime.runner.make_forward
            original_measurement = runtime.benchmark.benchmark_measurement
            try:
                return runtime.runtime("evaluate", mode, rank, root=self.root)
            finally:
                self.assertIs(runtime.runner.make_forward, original_factory)
                self.assertIs(runtime.benchmark.benchmark_measurement, original_measurement)

    def test_full_and_canary_sidecars_account_for_all_calls(self):
        for rank, mode in enumerate(("canary", "full")):
            self.evaluate(mode, rank)
            directory = self.out / f"evaluate/rank_{rank:03d}"
            evidence = runtime.validate_sidecar(directory, self.config, mode, rank)
            self.assertEqual(evidence["unique_keys"], 52)
            repeats = 2 if mode == "canary" else 5
            self.assertEqual(evidence["calls"], 26 * (3 + 2 + 2 * repeats))
            self.assertEqual(len(evidence["observations"]), 52)

    def test_capture_runs_after_stop_and_preserves_result_metrics(self):
        result = {"logits_ab": [2.0, 1.0]}
        metrics = {"geometry_sha256": "g", "schedule_sha256": "s", "end_to_end_elapsed_ms": 7.0}
        def original(*args, **kwargs):
            self.events.append("stop")
            return result, metrics
        real_digest = runtime.digest
        def checked_digest(value):
            self.assertEqual(self.events[-1], "stop")
            self.events.append("digest")
            return real_digest(value)
        with patch.object(runtime.runner, "make_forward", side_effect=self.factory), \
                patch.object(runtime.benchmark, "benchmark_measurement", side_effect=original) as measured, \
                patch.object(runtime, "digest", side_effect=checked_digest):
            with runtime.LogitCapture().installed():
                forward = runtime.runner.make_forward({"id": "native", "compression_id": "native"})
                observed = runtime.benchmark.benchmark_measurement(None, {"request_id": "r"}, None,
                    {"id": "native"}, None, forward=forward)
            self.assertIs(observed[0], result)
            self.assertIs(observed[1], metrics)
            self.assertEqual(measured.call_count, 1)

    def test_finite_same_answer_drift_fails_and_restores_hooks(self):
        # 78 warmup/probe + 52 prediction calls precede the first benchmark call.
        self.drift_at = 131
        with self.assertRaisesRegex(ValueError, "repeated logits"):
            self.evaluate()
        directory = self.out / "evaluate/rank_000"
        self.assertTrue((directory / "error.json").exists())
        self.assertFalse((directory / runtime.SIDECAR).exists())

    def test_call_counts_unique_keys_digests_and_bindings_reject_tampering(self):
        self.evaluate()
        directory = self.out / "evaluate/rank_000"
        evidence = runtime.runner.read_json(directory / runtime.SIDECAR)
        mutations = [lambda e: e["observations"][0].update(timed_calls=999),
                     lambda e: e["observations"][0].update(logits_sha256="0" * 64),
                     lambda e: e["observations"].append(dict(e["observations"][0])),
                     lambda e: e.update(rank=1), lambda e: e.update(mode="full"),
                     lambda e: e.update(wrapper_sha256="0" * 64),
                     lambda e: e.update(config_sha256="0" * 64),
                     lambda e: e.update(calls=0), lambda e: e.update(unique_keys=0)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(evidence)
                mutate(changed)
                with self.assertRaises(ValueError):
                    runtime.validate_sidecar(directory, self.config, "canary", 0, evidence=changed)
        with (directory / "timings.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "provenance/hash"):
            runtime.validate_sidecar(directory, self.config, "canary", 0)

    def test_aggregate_requires_all_sidecars_before_base_runtime(self):
        for rank in range(4):
            self.evaluate(rank=rank)
        with patch.object(runtime.base, "runtime", return_value={"status": "pass"}) as delegated:
            self.assertEqual(runtime.runtime("aggregate", "canary", root=self.root), {"status": "pass"})
            delegated.assert_called_once_with("aggregate", "canary", 0, base=self.root)
            (self.out / f"evaluate/rank_003/{runtime.SIDECAR}").unlink()
            delegated.reset_mock()
            with self.assertRaises(FileNotFoundError):
                runtime.runtime("aggregate", "canary", root=self.root)
            delegated.assert_not_called()

    def test_sidecar_write_failure_fails_closed(self):
        atomic = runtime.base.safe.atomic_json
        def write(path, value):
            if Path(path).name == runtime.SIDECAR:
                raise OSError("sidecar disk failure")
            return atomic(path, value)
        with patch.object(runtime.base.safe, "atomic_json", side_effect=write):
            with self.assertRaisesRegex(OSError, "sidecar disk"):
                self.evaluate()
        directory = self.out / "evaluate/rank_000"
        self.assertTrue((directory / "error.json").exists())
        self.assertFalse((directory / runtime.SIDECAR).exists())

    def test_exact_cli_and_import_do_not_load_torch(self):
        with patch.object(runtime, "runtime", return_value={"status": "pass"}) as delegated, \
                patch("sys.stdout", new_callable=io.StringIO):
            runtime.main(["runtime", "--stage", "evaluate", "--mode", "full", "--rank", "3"])
            delegated.assert_called_with("evaluate", "full", 3)
            runtime.main(["runtime", "--stage", "aggregate", "--mode", "canary"])
            delegated.assert_called_with("aggregate", "canary", 0)
        command = [sys.executable, "-B", "-S", "-c",
                   "import sys; from research.lifecycle_memory import role_aware_killarney_runtime; "
                   "assert 'torch' not in sys.modules"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
