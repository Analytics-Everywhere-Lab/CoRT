"""Bounded stdlib/fake tests. Optional NumPy exercises the real bootstrap, never a model."""
from contextlib import contextmanager, nullcontext
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.lifecycle_memory import role_aware_transfer as policy
from research.lifecycle_memory import run_role_aware_transfer as run
from research.lifecycle_memory import role_aware_policy as original
from research.lifecycle_memory import run_role_aware_policy as base
from research.lifecycle_memory.decoder_token_matrix import Schedule, ScheduledStage, build_arms

ROOT = Path(__file__).resolve().parents[3]


def public_rows(mode):
    result = []
    for i in range(16 if mode == "canary" else 447):
        source = run.data.SOURCE_FAMILIES[i % 3]
        for order in range(2):
            result.append({"request_id": f"r{i}_{order}", "frame_key": f"f{i}",
                "cluster_key": f"{source}_{i % 15}", "source_family": source,
                "sampling_unit_kind": run.data.SOURCE_KINDS[source], "order_id": f"order_{order}"})
    return result


class Contract:
    def __init__(self, mode):
        self.mode = mode
        self.config = run.read_json(ROOT / policy.REFERENCE_CONFIG)
        self.checks = []

    def rows(self, directory, mode):
        return public_rows(mode)

    def shards(self, rows):
        return run.data.frame_shards(rows)

    def benchmark(self, rows, mode):
        frames = set(run.data._source_balanced_frames(rows)[:16 if mode == "canary" else 64])
        return {r["request_id"] for r in rows if r["frame_key"] in frames}

    def check(self, directory, *, public_only):
        self.checks.append(public_only)
        return {"status": "pass", "manifest_sha256": policy.EVALUATION_SHA,
                "evaluator_files_verified": not public_only}


def fixture_root(root, mode="canary"):
    release = root / "release"
    for relative in (policy.CONFIG, policy.REFERENCE_CONFIG):
        dest = release / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
    dest = release / policy.REFERENCE_ADAPTERS
    dest.mkdir(parents=True)
    for seed in policy.SEEDS:
        name = f"{policy.CASE}.seed{seed}.final.safetensors"
        shutil.copyfile(ROOT / policy.REFERENCE_ADAPTERS / name, dest / name)
    contract = Contract(mode)
    run.freeze(root, mode, contract=contract)
    return contract


class Tensor:
    def __init__(self, values, shape=None, dtype="float32"):
        self.values = list(values)
        self.shape = tuple(shape or (len(self.values),))
        self.dtype = dtype

    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def clone(self): return Tensor(self.values, self.shape, self.dtype)
    def reshape(self, *shape): return Tensor(self.values, (len(self.values),), self.dtype)
    def tolist(self): return list(self.values)
    def view(self, dtype):
        return Tensor(list(struct.pack(f">{len(self.values)}d", *self.values)), dtype=dtype)


class Truth:
    def __init__(self, value): self.value = value
    def all(self): return self.value


TORCH = SimpleNamespace(no_grad=nullcontext, uint8="uint8",
    equal=lambda a, b: a.values == b.values and a.shape == b.shape and a.dtype == b.dtype,
    isfinite=lambda a: Truth(all(math.isfinite(v) for v in a.values)))


def plan_for(arm):
    roles = {"F": list(range(1, 135)), "C": [135], "Q": [136], "P": [139], "R": [0, 137, 138]}
    before = tuple((i,) for i in range(140))
    groups = tuple(g for g in before if g[0] != 134)
    stage = ScheduledStage(34, "prune", before, groups, tuple(((g[0], 1),) for g in groups),
        {"F": {"original": 134, "protected": 0, "eligible_before": 134,
               "eligible_retained": 133, "after": 133, "effective_method": "prune", "per_slot_retained": [133]}})
    return Schedule(140, arm, (0, 139), () if arm["id"] == "native" else (stage,), {"roles": roles})


class FakeRuntime:
    def __init__(self, model, layers, device):
        self.engine, self.active, self.sealed = model, None, False
        model.runtime = self

    def __enter__(self): return self
    def __exit__(self, *args): self.active = None
    def prepare(self, plan): pass
    def seal(self): self.sealed = True
    def report(self): return {"sealed": self.sealed}

    @contextmanager
    def activate(self, plan, adapter=None, index=None):
        if self.active is not None:
            raise RuntimeError("nested runtime")
        self.active = (plan, adapter)
        try:
            yield
        finally:
            self.active = None


class FakeIndex:
    def get(self, plan, target, positions_fn, device): return positions_fn()
    def seal(self): pass
    def report(self): return {"sealed": True}


class Engine:
    def __init__(self, config):
        self.torch, self.model, self.layers, self.device = TORCH, self, [], "cpu_fake"
        self.runtime = None
        self.runtime_receipt = {"model_revision": policy.MODEL["revision"], "torch": policy.MODEL["torch"],
            "transformers": policy.MODEL["transformers"], "bf16": True, "use_cache": False,
            "attention_backend": "sdpa", "batch_size": 1, "device": "fake_cpu",
            "gemma_source_sha256": "a" * 64, "model_provenance": {"fake": True}}

    def prepare(self, row, directory):
        return SimpleNamespace(row=row, pilot_partition=SimpleNamespace(answer_prefix=SimpleNamespace(positions=(139,))))

    def geometry(self, prepared):
        payload = {"fake": 140}
        return run.digest(payload), payload

    def schedule(self, prepared, arm, geometry): return plan_for(arm)

    def logits(self, prepared):
        state = Tensor([2., 1., 0., 0.], (1, 1, 4))
        active = self.runtime.active if self.runtime else None
        if active and active[1] is not None:
            state = active[1](state)
        values = state.values[:2]
        if active and active[0].stages:
            values[0] -= .1
        return Tensor(values)

    def matrix_identity_gate(self, prepared, geometry):
        return {"status": "pass", "keep_all_ab_logits_bit_exact": True}

    def matrix_forward(self, prepared, plan):
        with self.runtime.activate(plan):
            return self.logits(prepared), {}


def load_adapters(engine, config, directory):
    def adapter(seed):
        return lambda t: Tensor([v + seed / 100 for v in t.values], t.shape)
    return ({(policy.CASE, s): adapter(s) for s in policy.SEEDS}, lambda t: t,
            [{k: a[k] for k in ("case_id", "seed", "sha256")} for a in config["adapters"]])


@contextmanager
def legacy(engine, prepared, adapter, case, training=False, schedule=None):
    with engine.runtime.activate(schedule, adapter):
        yield engine.logits(prepared)


def measurement(torch, fn):
    return fn(), {"device_id": "fake_cpu", "batch_size": 1, "synchronized": True,
                  "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}


def evaluate_fake(root, contract, mode, rank):
    from research.lifecycle_memory import decoder_token_matrix_steering as steering
    with patch.object(steering, "correction_forward", legacy):
        return run.evaluate(root, mode, rank, contract=contract, engine_factory=Engine,
            adapter_loader=load_adapters, measure=measurement, synchronize=lambda: None,
            runtime_factory=FakeRuntime, index_factory=FakeIndex)


def replace_json(path, value):
    path.unlink()
    run.write_json(path, value)


def replace_rows(path, values):
    path.unlink()
    path.touch()
    for row in values:
        run.append(path, row)


def reseal(directory):
    # Attack fixtures update container hashes too, exercising semantic checks.
    side = run.read_json(directory / "timed_logits_validation.json")
    side["files_sha256"] = {n: base.sha256(directory / n) for n in ("predictions.jsonl", "timings.jsonl")}
    replace_json(directory / "timed_logits_validation.json", side)
    worker = run.read_json(directory / "worker_receipt.json")
    worker["files_sha256"] = {n: base.sha256(directory / n) for n in run.FILES}
    replace_json(directory / "worker_receipt.json", worker)


class TransferTests(unittest.TestCase):
    def test_config_import_and_originals_unchanged(self):
        config = run.read_json(ROOT / policy.CONFIG)
        self.assertEqual(config, policy.source_config())
        self.assertEqual(len(policy.variant_specs()), 5)
        baseline = copy.deepcopy(original.DECISION), original.variant_specs, base.make_forward
        before = {p: base.sha256(ROOT / p) for p in (policy.REFERENCE_CONFIG,
            "docs/research/lifecycle_memory/phase 2 research/jobs/role_aware_policy/config.json")}
        a, b = policy.estimator(), policy.estimator()
        a.DECISION["bounds"] = 200
        self.assertEqual(b.DECISION["bounds"], 4)
        self.assertEqual(b.seed_mean_groups(), {})
        self.assertEqual(baseline, (original.DECISION, original.variant_specs, base.make_forward))
        self.assertEqual(before, {p: base.sha256(ROOT / p) for p in before})
        command = "import sys; import research.lifecycle_memory.run_role_aware_transfer; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-B", "-c", command], cwd=ROOT, check=True, timeout=10)
        for field, value in (("training_allowed", True), ("deployment_seed", 29), ("cut", 30)):
            bad = copy.deepcopy(config); bad[field] = value
            with self.assertRaises(ValueError): policy.validate_config(bad)
        with self.assertRaises(ValueError): run.runtime("train", "canary")

    def test_freeze_exact_assets_mode_and_no_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = base.code_manifest()
            contract = fixture_root(root)
            frozen_path = root / "out/frozen_config.json"
            frozen = run.read_json(frozen_path)
            self.assertEqual(len(list((root / "out/adapters").iterdir())), 2)
            for a in frozen["adapters"]:
                self.assertEqual(base.sha256(root / "out/adapters" / a["path"]), policy.HASHES[a["seed"]])
            run.validate_frozen(frozen, "canary")
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                run.validate_frozen(frozen, "full")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                run.freeze(root, "canary", contract=contract)
            bad = copy.deepcopy(frozen); bad["freeze"]["inventory"]["benchmark_ids"].pop()
            with self.assertRaisesRegex(ValueError, "selection drift"):
                run.validate_frozen(bad, "canary")
            self.assertEqual(before, base.code_manifest())
            self.assertTrue(all(contract.checks))

    def test_structural_state_and_retention_failure_paths(self):
        arms = {a["id"]: a for a in build_arms()}
        plan = plan_for(arms[policy.COMPRESSION])
        self.assertEqual(run.schedule_evidence(plan)["eligible_scene_retained"], 133)
        self.assertEqual(run.serialized_schedule_evidence(plan.receipt()), run.schedule_evidence(plan))
        bad = copy.deepcopy(plan.receipt()); bad["stages"][0]["groups"].remove([135])
        with self.assertRaisesRegex(ValueError, "non-F"):
            run.serialized_schedule_evidence(bad)
        a, b = run.StateCapture(lambda t: t), run.StateCapture(lambda t: t)
        a(Tensor([1., 2.], (1, 1, 2))); b(Tensor([1., 2.], (1, 1, 2)))
        self.assertTrue(run.state_equality(TORCH, a, b)["exact"])
        b.states[0][1].values[0] += .01
        with self.assertRaisesRegex(ValueError, "state differs"): run.state_equality(TORCH, a, b)
        with self.assertRaisesRegex(ValueError, "once"): a(Tensor([1., 2.], (1, 1, 2)))

    def test_freeze_receipt_provenance_and_offline_environment_restoration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_root(root)
            path = root / "out/frozen_config.json"
            config = run.read_json(path)
            run.validate_freeze_receipt(path, config, "canary")
            receipt_path = root / "out/freeze_receipt.json"
            receipt = run.read_json(receipt_path); receipt["mode"] = "full"
            replace_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "freeze receipt"):
                run.validate_freeze_receipt(path, config, "canary")
            bad = copy.deepcopy(config); bad["freeze"]["code"] = {}
            with self.assertRaisesRegex(ValueError, "code drift"):
                run.validate_frozen(bad, "canary")
            before = dict(os.environ)
            def fail(*args):
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["HF_HOME"], str(root / "model"))
                raise RuntimeError("fake worker failure")
            with patch.object(run, "evaluate", fail):
                with self.assertRaisesRegex(RuntimeError, "fake worker"):
                    run.runtime("evaluate", "canary", root=root)
            self.assertEqual(dict(os.environ), before)

    def test_invalid_worker_is_rejected_before_full_evaluator_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); contract = fixture_root(root, "full")
            out = root / "out/evaluate"
            for rank in range(4): (out / f"rank_{rank:03d}").mkdir(parents=True)
            config = root / "out/frozen_config.json"
            # No label access is needed to reject a corrupt/incomplete worker.
            with self.assertRaises((ValueError, FileNotFoundError)):
                run.aggregate(root, "full", contract=contract, draws=10)
            self.assertTrue(all(contract.checks))
            self.assertFalse((out / "analysis_summary.json").exists())

    def test_canary_end_to_end_all_requests_and_repeats_no_labels(self):
        from research.lifecycle_memory import decoder_token_matrix_steering as steering
        old_forward = steering.correction_forward
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); contract = fixture_root(root)
            for rank in range(4): evaluate_fake(root, contract, "canary", rank)
            result = run.aggregate(root, "canary", contract=contract)
            self.assertEqual(result["prediction_rows"], 32 * 5)
            self.assertEqual(result["timing_rows"], 32 * 2 * 5)
            self.assertFalse(result["labels_accessed"])
            self.assertTrue(all(contract.checks))
            self.assertFalse((root / "eval/evaluator").exists())
            self.assertTrue((root / "out/evaluate/RESULTS.md").is_file())
            with self.assertRaisesRegex(ValueError, "overwrite"):
                run.aggregate(root, "canary", contract=contract)
        self.assertIs(steering.correction_forward, old_forward)

    def test_timed_drift_stops_worker_and_saves_failure(self):
        from research.lifecycle_memory import decoder_cached_benchmark as benchmark
        real = benchmark.benchmark_measurement
        calls = {}
        def corrupt(*args, **kwargs):
            output, metrics = real(*args, **kwargs)
            key = (args[1]["request_id"], kwargs["forward"])
            if kwargs["phase"] == "timed":
                calls[key] = calls.get(key, 0) + 1
                if calls[key] == 2:
                    output["logits_ab"][0] += .01
                    output["margin_ab"] = (output["logits_ab"][0] - output["logits_ab"][1]) / math.log(2)
            return output, metrics
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); contract = fixture_root(root)
            with patch.object(benchmark, "benchmark_measurement", corrupt):
                with self.assertRaisesRegex(ValueError, "timed output drift"):
                    evaluate_fake(root, contract, "canary", 0)
            self.assertTrue((root / "out/evaluate/rank_000/error.json").is_file())

    def test_verifier_rejects_resealed_missing_duplicate_and_changed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); contract = fixture_root(root)
            evaluate_fake(root, contract, "canary", 0)
            source = root / "out/evaluate/rank_000"
            config = root / "out/frozen_config.json"
            def mutation(name, fn):
                d = root / name; shutil.copytree(source, d)
                fn(d); reseal(d)
                with self.assertRaises((ValueError, KeyError)):
                    run.validate_worker(d, config, "canary", 0)
            def timing(d, kind):
                p = d / "timings.jsonl"; rows = run.read_rows(p)
                if kind == "missing": rows.pop()
                if kind == "duplicate": rows.append(rows[0])
                if kind == "changed": rows[0]["logits_sha256"] = "b" * 64
                if kind == "order": rows[0]["variant_order"] = 99
                replace_rows(p, rows)
            for kind in ("missing", "duplicate", "changed", "order"):
                mutation(kind, lambda d, kind=kind: timing(d, kind))
            def sidecar(d):
                p = d / "timed_logits_validation.json"; obj = run.read_json(p)
                obj["observations"].pop(); replace_json(p, obj)
            mutation("sidecar", sidecar)
            def states(d):
                p = d / "cold_start.json"; obj = run.read_json(p)
                obj["gates"][0]["states"]["17"]["exact"] = False; replace_json(p, obj)
            mutation("states", states)
            def runtime_drift(d):
                p = d / "worker_receipt.json"; obj = run.read_json(p)
                obj["runtime"]["model_revision"] = "wrong"; replace_json(p, obj)
            mutation("runtime", runtime_drift)
            with self.assertRaises(ValueError): run.validate_worker(source, config, "canary", 1)
            (source / "timings.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run.validate_worker(source, config, "canary", 0)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional CPU NumPy not installed")
    def test_full_fake_counts_and_real_estimator_individual_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); contract = fixture_root(root, "full")
            for rank in range(4): evaluate_fake(root, contract, "full", rank)
            labels = root / "eval/evaluator/labels.jsonl"
            labels.parent.mkdir(parents=True); labels.touch()
            for r in public_rows("full"):
                run.append(labels, {"request_id": r["request_id"], "correct_label": "A"})
            result = run.aggregate(root, "full", contract=contract, draws=40)
            self.assertEqual(result["prediction_rows"], 894 * 5)
            self.assertEqual(result["timing_rows"], 128 * 5 * 5)
            self.assertEqual(set(result["per_source"]), set(run.data.SOURCE_FAMILIES))
            self.assertTrue(result["labels_accessed"])
            self.assertFalse(contract.checks[-1])
            self.assertEqual(set(result["decision"]["co_primaries"]), {"seed17", "seed29"})
            self.assertFalse(any("seed_mean" in k for k in result["metrics"]))
            self.assertEqual(len(result["retention_diagnostics"]), 2)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional CPU NumPy not installed")
    def test_seed17_can_pass_without_seed29_and_retention_is_not_a_gate(self):
        rows = public_rows("canary"); predictions, timings, labels = [], [], {}
        for i, row in enumerate(rows):
            labels[row["request_id"]] = "A"
            for spec in policy.variant_specs():
                name = spec["id"]
                # Seed17 loses to its unpruned control but still beats native.
                good = name.startswith("native_readout") or (name == "F133_L34_readout_seed17" and i % 4 != 0)
                p = "A" if good else "B"
                predictions.append({**row, "variant": name, "prediction": p, "margin_ab": 1. if good else -1.})
                for repeat in range(2):
                    timings.append({**row, "variant": name, "repeat_id": repeat,
                        "end_to_end_elapsed_ms": 8. if name.startswith("F133") else 10., "model_forward_ms": 7.})
        before = copy.deepcopy(original.DECISION)
        result = policy.summarize(rows, predictions, timings, labels, draws=100)
        decisions = result["decision"]["co_primaries"]
        self.assertTrue(decisions["seed17"]["upper_left_pass"])
        self.assertFalse(decisions["seed29"]["upper_left_pass"])
        self.assertFalse(result["decision"]["both_seeds_pass"])
        self.assertLess(result["retention_diagnostics"]["F133_L34_readout_seed17"]["accuracy"]["estimate"], 0)
        self.assertEqual(original.DECISION, before)


if __name__ == "__main__":
    unittest.main()
