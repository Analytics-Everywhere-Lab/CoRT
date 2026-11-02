"""CPU tests for the role-aware policy experiment; no downloads, GPU, or real model."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import save_file

from research.lifecycle_memory import role_aware_policy as rap
from research.lifecycle_memory import run_role_aware_policy as run
from research.lifecycle_memory.application_pilot_steering import LowRankSteering, SteeringTarget
from research.lifecycle_memory.decoder_token_matrix import Arm, Stage, build_schedule, matrix_context
from research.lifecycle_memory.decoder_token_matrix_steering import composite_context, target_positions
from research.lifecycle_memory.tests.test_decoder_token_matrix import fixture
from research.lifecycle_memory.tests.test_decoder_token_matrix_steering import Model

torch.set_num_threads(1)
REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / rap.JOB_REL / "config.json"
TOY_STAGES = {"native": (), rap.PRIMARY: (Stage(30, targets={"F": 0}, method="prune"),),
              rap.SECONDARY: (Stage(34, fractions={"F": .5, "C": .5}, method="prune"),)}


def toy_side(n):
    return {"position_ids": torch.arange(n)[None],
            "position_embeddings": (torch.arange(n).view(1, n, 1).expand(1, n, 8).float(), torch.zeros(1, n, 8)),
            "cache_position": torch.arange(n),
            "attention_mask": torch.triu(torch.full((1, 1, n, n), -100.), diagonal=1)}


def nonzero_adapter(seed):
    torch.manual_seed(seed)
    adapter = LowRankSteering(8)
    with torch.no_grad():
        adapter.up.weight.normal_(std=.3)
    return adapter


class DesignTests(unittest.TestCase):
    def test_import_is_torch_free(self):
        code = "import sys; import research.lifecycle_memory.role_aware_policy; print('torch' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False")

    def test_frozen_config_and_inventory(self):
        config = rap.validate_config(json.loads(CONFIG.read_text()))
        self.assertFalse(rap.adapters_complete(config))
        ids = [s["id"] for s in rap.variant_specs()]
        self.assertEqual(len(ids), 14)
        for left, right in rap.identity_pairs():
            self.assertIn(left, ids)
            self.assertIn(right, ids)
        groups = rap.seed_mean_groups()
        self.assertIn(rap.DECISION["primary_variant"], groups)
        self.assertEqual(len(groups), 5)
        bad = copy.deepcopy(config)
        bad["decision"]["speedup_floor"] = 0.9
        with self.assertRaises(ValueError):
            rap.validate_config(bad)


class FusedCorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.partition, self.coords = fixture()
        self.n = self.partition.sequence_length
        self.model = Model(count=7)
        self.hidden = torch.randn(1, self.n, 8)
        self.side = toy_side(self.n)

    def plan(self, stages):
        return build_schedule(partition=self.partition, image_position_ids=self.coords,
                              arm=Arm("toy", tuple(stages)), protected_positions=[], pooling_kernel=1)

    def fused(self, plan, adapter, cut=3):
        cache = rap.IndexCache(cut=cut)
        index = cache.get(plan, "CQ", lambda: target_positions(self.partition, "CQ"), "cpu")
        with torch.no_grad(), rap.fused_correction(self.model.layers[cut], adapter, index):
            if plan.stages:
                with matrix_context(self.model.layers, plan):
                    return self.model(self.hidden, self.side)
            return self.model(self.hidden, self.side)

    def legacy(self, plan, adapter, cut=3):
        target = SteeringTarget("toy", cut, "CQ")
        with torch.no_grad(), composite_context(self.model, self.model.layers, plan, target,
                                                 target_positions(self.partition, "CQ"), adapter):
            return self.model(self.hidden, self.side)

    def assert_clean(self):
        for block in self.model.layers:
            self.assertFalse(block._forward_hooks or block._forward_pre_hooks)
            self.assertNotIn("forward", block.__dict__)

    def test_fused_is_bit_exact_with_legacy_composite_path(self):
        designs = [(), (Stage(1, targets={"F": 0}, method="prune"),),
                   (Stage(3, fractions={"F": .5, "C": .5}, method="prune"),),
                   (Stage(1, targets={"F": 4}), Stage(3, fractions={"C": .5}, method="prune"))]
        for stages in designs:
            with self.subTest(stages=stages):
                plan = self.plan(stages)
                adapter = nonzero_adapter(11)
                fused = self.fused(plan, adapter)
                self.assert_clean()
                legacy = self.legacy(plan, copy.deepcopy(adapter))
                self.assert_clean()
                self.assertTrue(torch.equal(fused, legacy))

    def test_zero_adapter_is_identity(self):
        plan = self.plan((Stage(1, targets={"F": 0}, method="prune"),))
        with torch.no_grad(), matrix_context(self.model.layers, plan):
            baseline = self.model(self.hidden, self.side)
        self.assertTrue(torch.equal(self.fused(plan, LowRankSteering(8)), baseline))

    def test_indices_skip_pruned_and_follow_compaction(self):
        plan = self.plan((Stage(1, targets={"F": 0}, method="prune"),))
        original = target_positions(self.partition, "CQ")
        indices = rap.correction_indices(plan, original, cut=3)
        self.assertEqual(len(indices), len(original))
        self.assertEqual(indices[0], original[0] - self.partition.full_frame.count)

    def test_cut_must_run_exactly_once_and_cache_seals(self):
        adapter = LowRankSteering(8)
        with self.assertRaises(RuntimeError):
            with rap.fused_correction(self.model.layers[3], adapter, torch.tensor([0])):
                pass
        self.assert_clean()
        cache = rap.IndexCache(cut=3)
        plan = self.plan(())
        cache.get(plan, "CQ", lambda: target_positions(self.partition, "CQ"), "cpu")
        cache.seal()
        cache.get(plan, "CQ", lambda: self.fail("must not recompute"), "cpu")
        with self.assertRaises(RuntimeError):
            cache.get(self.plan(()), "CQ", lambda: target_positions(self.partition, "CQ"), "cpu")


# --------------------------------------------------------------------------- toy end-to-end

class ToyEngine:
    """36-block toy decoder with the MatrixEngine surface used by the runner."""

    def __init__(self, config):
        torch.manual_seed(3)
        self.torch = torch
        self.device = "cpu"
        self.partition, self.coords = fixture()
        self.n = self.partition.sequence_length
        self.model = Model(count=36)
        self.model.requires_grad_(False)
        self.layers = list(self.model.layers)
        self.model.config.text_config = SimpleNamespace(hidden_size=8)
        self.runtime_receipt = {"toy": True}

    def prepare(self, row, data_dir):
        seed = int(hashlib.sha256(row["request_id"].encode()).hexdigest()[:8], 16)
        hidden = torch.randn(1, self.n, 8, generator=torch.Generator().manual_seed(seed))
        return SimpleNamespace(pilot_partition=self.partition, hidden=hidden)

    def geometry(self, prepared):
        payload = {"n": self.n}
        return checksum(payload), payload

    def schedule(self, prepared, arm, geometry):
        return build_schedule(partition=self.partition, image_position_ids=self.coords,
                              arm=Arm(arm["id"], TOY_STAGES[arm["id"]]), protected_positions=[], pooling_kernel=1)

    def logits(self, prepared):
        out = self.model(prepared.hidden, toy_side(self.n))
        return out[0, -1, :2].float()

    def matrix_forward(self, prepared, schedule):
        with matrix_context(self.layers, schedule) as audit:
            return self.logits(prepared), audit


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ToyContract:
    def __init__(self, rows, labels_dir=None):
        self._rows = rows

    def check(self, data_dir, *, public_only):
        return {"status": "pass", "toy": True}

    def rows(self, data_dir, mode):
        return list(self._rows)

    def shards(self, rows):
        frames = sorted({r["frame_key"] for r in rows})
        return {r["request_id"]: frames.index(r["frame_key"]) % 4 for r in rows}

    def benchmark(self, rows, mode):
        frames = sorted({r["frame_key"] for r in rows})[::2]
        return {r["request_id"] for r in rows if r["frame_key"] in frames}


def toy_rows():
    rows = []
    for i in range(24):
        source = rap.SOURCE_FAMILIES[i % 3]
        for order in ("order_0", "order_1"):
            rows.append({"request_id": f"r{i}_{order}", "frame_key": f"f{i:02d}",
                         "cluster_key": f"{source}_u{i % 6}", "order_id": order, "source_family": source})
    return rows


def fake_measure(_torch, fn):
    with torch.no_grad():
        return fn(), {"elapsed_ms": 1.0, "peak_allocated_bytes": 0}


class EndToEndTests(unittest.TestCase):
    def test_four_rank_evaluation_and_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = json.loads(CONFIG.read_text())
            adapters_dir = tmp / "adapters"
            adapters_dir.mkdir()
            records = []
            for k, case in enumerate((*rap.TRAIN_CASES, rap.REFERENCE_CASE)):
                for seed in rap.SEEDS:
                    adapter = nonzero_adapter(100 * k + seed)
                    path = adapters_dir / f"{case}.seed{seed}.safetensors"
                    save_file({n: v.contiguous() for n, v in adapter.state_dict().items()}, str(path))
                    records.append({"case_id": case, "seed": seed, "path": path.name, "sha256": run.sha256(path),
                                    "cut": 34, "rank": 4, "parameters": 64})
            config["adapters"] = records
            rap.require(rap.adapters_complete(config), "toy adapters incomplete")
            config_path = tmp / "config.json"
            # Toy adapters are hidden=8 (64 parameters); bypass only the production shape check.
            original = rap.validate_config
            def lenient(cfg):
                saved = [dict(a) for a in cfg["adapters"]]
                for a in cfg["adapters"]:
                    a["parameters"] = 43008
                try:
                    return original(cfg)
                finally:
                    cfg["adapters"] = saved
            config_path.write_text(json.dumps(config))
            rows = toy_rows()
            contract = ToyContract(rows)
            arms = [{"id": name, "stages": [s.as_dict() for s in stages]} for name, stages in TOY_STAGES.items()]
            out = tmp / "out"
            rap.validate_config = lenient
            try:
                for rank in range(4):
                    args = SimpleNamespace(config=config_path, mode="full", data_dir=tmp, adapters_dir=adapters_dir,
                                           out_dir=out, rank=rank, world_size=4)
                    receipt = run.evaluate_worker(args, contract=contract, engine_factory=ToyEngine,
                                                  measure=fake_measure, synchronize=lambda: None, arms=arms,
                                                  checksum_fn=checksum, receipt_fn=lambda plan: plan.receipt())
                    self.assertEqual(receipt["status"], "pass")
                gates = json.loads((out / "rank_000/cold_start.json").read_text())["identity_gates"]
                self.assertTrue(gates and all(g["bit_exact"] for g in gates))
                (tmp / "evaluator").mkdir()
                with (tmp / "evaluator/labels.jsonl").open("w") as handle:
                    for row in rows:
                        handle.write(json.dumps({"request_id": row["request_id"],
                                                 "correct_label": "A" if row["order_id"] == "order_0" else "B"}) + "\n")
                result = run.aggregate(SimpleNamespace(config=config_path, mode="full", data_dir=tmp, out_dir=out),
                                       contract=contract)
            finally:
                rap.validate_config = original
            self.assertEqual(result["status"], "pass")
            summary = json.loads((out / "analysis_summary.json").read_text())
            self.assertIn("upper_left_pass", summary["decision"]["primary"])
            self.assertIn(rap.DECISION["primary_variant"], summary["timing"])
            self.assertTrue((out / "RESULTS.md").read_text().startswith("# Role-Aware Policy Results"))
            # Fused and legacy variants must have identical predictions on every request.
            preds = [json.loads(line) for r in range(4)
                     for line in (out / f"rank_{r:03d}/predictions.jsonl").read_text().splitlines()]
            by = {(p["request_id"], p["variant"]): p["logits_ab"] for p in preds}
            for left, right in rap.identity_pairs():
                for row in rows:
                    self.assertEqual(by[(row["request_id"], left)], by[(row["request_id"], right)])


class SummaryDecisionTests(unittest.TestCase):
    def synthetic(self, primary_correct, primary_ms):
        public = toy_rows()
        labels = {r["request_id"]: "A" if r["order_id"] == "order_0" else "B" for r in public}
        predictions, timings = [], []
        for spec in rap.variant_specs():
            for i, row in enumerate(public):
                good = (spec["id"].startswith(f"{rap.PRIMARY}__CQ_fused") and primary_correct) or i % 5
                label = labels[row["request_id"]]
                pred = label if good else ("B" if label == "A" else "A")
                margin = 3.0 if pred == "A" else -3.0
                predictions.append({**row, "variant": spec["id"], "prediction": pred, "margin_ab": margin})
                ms = primary_ms if spec["id"].startswith(f"{rap.PRIMARY}__CQ_fused") else 306.0
                for repeat in range(2):
                    timings.append({**row, "variant": spec["id"], "repeat_id": repeat,
                                    "end_to_end_elapsed_ms": ms + 0.1 * (i % 3), "model_forward_ms": ms - 57})
        return public, predictions, timings, labels

    def test_upper_left_passes_only_when_both_better(self):
        public, p, t, labels = self.synthetic(True, 299.0)
        self.assertTrue(rap.summarize(public, p, t, labels, draws=500)["decision"]["primary"]["upper_left_pass"])
        public, p, t, labels = self.synthetic(True, 310.0)
        self.assertFalse(rap.summarize(public, p, t, labels, draws=500)["decision"]["primary"]["upper_left_pass"])
        public, p, t, labels = self.synthetic(False, 299.0)
        self.assertFalse(rap.summarize(public, p, t, labels, draws=500)["decision"]["primary"]["upper_left_pass"])


class FreezeTests(unittest.TestCase):
    def test_freeze_copies_and_hashes_six_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            job = tmp / rap.JOB_REL
            job.mkdir(parents=True)
            (job / "config.json").write_text(CONFIG.read_text())
            ref_dir = tmp / run.REFERENCE_ADAPTER_DIR
            ref_dir.mkdir(parents=True)
            refs = []
            for seed in rap.SEEDS:
                path = ref_dir / f"{rap.REFERENCE_CASE}.seed{seed}.final.safetensors"
                path.write_bytes(f"reference{seed}".encode())
                refs.append({"case_id": rap.REFERENCE_CASE, "seed": seed, "path": path.name,
                             "sha256": run.sha256(path), "parameters": 43008})
            (tmp / run.CACHED_VALIDATION_CONFIG).write_text(json.dumps({"adapters": refs}))
            train = tmp / "train"
            for rank, (case, seed) in enumerate(run.training_tasks()):
                task = train / f"rank_{rank:03d}/{case}_seed{seed}"
                task.mkdir(parents=True)
                ckpt = task / f"{case}.seed{seed}.final.safetensors"
                ckpt.write_bytes(f"{case}{seed}".encode())
                (task / "training_receipt.json").write_text(json.dumps({
                    "status": "pass", "mode": "full", "steps": 256, "parameters": 43008,
                    "checkpoint": ckpt.name, "checkpoint_sha256": run.sha256(ckpt)}))
            result = run.freeze(job / "config.json", train, repo=tmp)
            self.assertEqual(result["adapters"], 6)
            frozen = rap.validate_config(json.loads((job / "config.json").read_text()))
            self.assertTrue(rap.adapters_complete(frozen))
            with self.assertRaises(ValueError):
                run.freeze(job / "config.json", train, repo=tmp)


if __name__ == "__main__":
    unittest.main()
