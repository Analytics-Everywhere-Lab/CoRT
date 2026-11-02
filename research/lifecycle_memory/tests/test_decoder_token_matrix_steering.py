"""CPU replay/gradient and release-inventory tests; no downloads or GPU model."""
from contextlib import contextmanager
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from research.lifecycle_memory import decoder_token_matrix_steering as steering
from research.lifecycle_memory.application_pilot_steering import (
    LowRankSteering, SteeringTarget, TrainingRow, train_adapter,
)
from research.lifecycle_memory.decoder_token_matrix import (
    Arm, Stage, build_arms, build_schedule, compact_states, matrix_context,
)
from research.lifecycle_memory.tests.test_decoder_token_matrix import fixture


torch.set_num_threads(1)


def config():
    return {"steering": {"case_count": 14, "seeds": [17, 29], "rank": 4, "steps": 256,
            "gradient_accumulation": 4, "learning_rate": .001, "checkpoint_selection": "final",
            "canary_steps": 2, "canary_eval_per_split": 4},
            "compression": {"benchmark_requests": 64, "timing_repeats": 3,
                            "canary_timing_repeats": 2, "warmups": 2},
            "bootstrap": {"draws": 100, "seed": 12345}}


class Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = torch.nn.Linear(width, width)
        self.self_attn = SimpleNamespace(layer_type="sliding_attention")
        self.observed = []
        self.fail = False

    def forward(self, hidden_states, per_layer_input=None, **side):
        if self.fail:
            raise RuntimeError("intentional failure")
        n = hidden_states.shape[1]
        assert side["position_ids"].shape[-1] == n
        assert side["attention_mask"].shape[-2:] == (n, n)
        assert side["position_embeddings"][0].shape[-2] == n
        assert side["cache_position"].shape[-1] == n
        self.observed.append(tuple(side["position_ids"][0].tolist()))
        pos = side["position_embeddings"][0].to(hidden_states.dtype) * .003
        h = torch.tanh(self.linear(hidden_states + pos))
        weights = torch.softmax(side["attention_mask"][0, 0].to(h.dtype), -1)
        return hidden_states + .1 * weights.matmul(h)


class Model(torch.nn.Module):
    def __init__(self, count=7, width=8):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(width) for _ in range(count)])
        self.config = SimpleNamespace(num_hidden_layers=count, hidden_size=width,
                                      vision_config=SimpleNamespace(pooling_kernel_size=1))

    def forward(self, hidden, side):
        for block in self.layers:
            hidden = block(hidden, **side)
        return hidden


class CompositeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.partition, self.coords = fixture()
        self.n = self.partition.sequence_length
        self.model = Model()
        self.hidden = torch.randn(1, self.n, 8)
        self.side = {"position_ids": torch.arange(self.n)[None],
                     "position_embeddings": (torch.arange(self.n).view(1, self.n, 1).expand(1, self.n, 8).float(),
                                             torch.zeros(1, self.n, 8)),
                     "cache_position": torch.arange(self.n),
                     "attention_mask": torch.triu(torch.full((1, 1, self.n, self.n), -100.), diagonal=1)}

    def schedule(self, stages=()):
        return build_schedule(partition=self.partition, image_position_ids=self.coords,
                              arm=Arm("toy", tuple(stages)), protected_positions=[], pooling_kernel=1)

    def baseline(self, schedule):
        with torch.no_grad(), matrix_context(self.model.layers, schedule):
            return self.model(self.hidden, self.side)

    def assert_clean(self):
        for block in self.model.layers:
            self.assertNotIn("forward", block.__dict__)
            self.assertFalse(hasattr(block, "_decoder_token_matrix_active"))
            self.assertFalse(block._forward_hooks or block._forward_pre_hooks)
        self.assertFalse(hasattr(self.model, "_pilot_steering_active"))

    def reference(self, schedule, target, positions, adapter):
        """Independent full-graph reference using original-coordinate side inputs."""
        h = self.hidden
        plans = {stage.cut: stage for stage in schedule.stages}
        selected = sorted({p for p in schedule.remap_positions(positions, target.layer_index - 1) if p is not None})
        for index, block in enumerate(self.model.layers):
            keep = torch.tensor([group[-1] for group in schedule.groups_at(index - 1)])
            side = {"position_ids": self.side["position_ids"].index_select(-1, keep),
                    "position_embeddings": tuple(value.index_select(-2, keep) for value in self.side["position_embeddings"]),
                    "cache_position": self.side["cache_position"].index_select(-1, keep),
                    "attention_mask": self.side["attention_mask"].index_select(-2, keep).index_select(-1, keep)}
            with torch.set_grad_enabled(index > target.layer_index):
                h = block(h, **side)
            if index == target.layer_index:
                h = h.detach()
                chosen = torch.tensor(selected)
                h = h.index_copy(1, chosen, adapter(h.index_select(1, chosen)))
            if index in plans:
                h = compact_states(h, plans[index])
        return h

    def test_zero_identity_backward_and_reference_gradients(self):
        designs = [([], "CQ"), ([Stage(1, targets={"F": 4})], "CQ"),
                   ([Stage(1, targets={"C": 4})], "FQ"),
                   ([Stage(1, targets={"C": 4})], "CQ"),
                   ([Stage(3, targets={"F": 4})], "FQ"),
                   ([Stage(3, fractions={"F": .5, "R": .5})], "CQ"),
                   ([Stage(4, targets={"C": 4})], "CQ"),
                   ([Stage(1, targets={"F": 4}), Stage(4, targets={"F": 2})], "readout")]
        for stages, role in designs:
            for nonzero in (False, True):
                with self.subTest(stages=stages, role=role, nonzero=nonzero):
                    plan = self.schedule(stages)
                    target = SteeringTarget("toy", 3, role)
                    positions = steering.target_positions(self.partition, role)
                    adapter = LowRankSteering(8)
                    if nonzero:
                        with torch.no_grad():
                            adapter.up.weight.normal_(std=.2)
                    reference_adapter = copy.deepcopy(adapter)
                    frozen = [p.clone().detach() for p in self.model.parameters()]
                    modes = [m.training for m in self.model.modules()]
                    flags = [p.requires_grad for p in self.model.parameters()]
                    baseline = self.baseline(plan)
                    with steering.composite_context(self.model, self.model.layers, plan, target,
                                                     positions, adapter, training=True) as audit:
                        actual = self.model(self.hidden, self.side)
                        if not nonzero:
                            self.assertTrue(torch.equal(baseline, actual))
                        actual[0, -1, 0].backward()
                        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.parameters()))
                        self.assertTrue(all(not m.training for m in self.model.modules()))
                        self.assertTrue(all(audit["block_calls"][str(i)] > 1 for i in range(4, 7)))
                    self.assertGreater(adapter.up.weight.grad.norm().item(), 0)
                    self.assertTrue(all(torch.equal(before, after) for before, after in zip(frozen, self.model.parameters())))
                    self.assertEqual(flags, [p.requires_grad for p in self.model.parameters()])
                    self.assertEqual(modes, [m.training for m in self.model.modules()])
                    self.assert_clean()
                    # Frozen independent reference is intentionally not checkpointed.
                    self.model.requires_grad_(False)
                    reference = self.reference(plan, target, positions, reference_adapter)
                    reference[0, -1, 0].backward()
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    for left, right in zip(adapter.parameters(), reference_adapter.parameters()):
                        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                    for parameter, flag in zip(self.model.parameters(), flags):
                        parameter.requires_grad_(flag)

    def test_remapped_cq_indices_and_same_cut_before_compression(self):
        target = SteeringTarget("test", 3, "CQ")
        positions = steering.target_positions(self.partition, "CQ")
        plan = self.schedule([Stage(1, targets={"F": 4}), Stage(3, targets={"C": 4})])
        adapter = LowRankSteering(8)
        with steering.composite_context(self.model, self.model.layers, plan, target, positions, adapter) as audit:
            self.model(self.hidden, self.side)
        expected = tuple(sorted(set(plan.remap_positions(positions, 2))))
        self.assertEqual(audit["selected_positions"], expected)
        self.assertEqual(expected[0], positions[0] - 4)
        self.assertEqual(len(expected), len(positions))
        self.assertNotEqual(expected, tuple(sorted(set(plan.remap_positions(positions, 3)))))
        self.assertEqual(audit["same_cut_order"], "correction_then_compression")

    def test_cleanup_failure_cache_nested_and_bad_side_inputs(self):
        plan = self.schedule([Stage(1, targets={"C": 4})])
        target = SteeringTarget("test", 3, "Q")
        adapter = LowRankSteering(8)
        positions = self.partition.literal_user_prompt.positions
        self.model.layers[5].fail = True
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with steering.composite_context(self.model, self.model.layers, plan, target, positions, adapter, True):
                self.model(self.hidden, self.side)
        self.assert_clean()
        self.model.layers[5].fail = False
        for key, value in (("use_cache", True), ("position_embeddings", None)):
            with self.assertRaises(ValueError):
                with steering.composite_context(self.model, self.model.layers, plan, target, positions, adapter):
                    hidden = self.hidden
                    for index, layer in enumerate(self.model.layers):
                        hidden = layer(hidden, **({**self.side, key: value} if index >= 2 else self.side))
            self.assert_clean()
        with matrix_context(self.model.layers, plan):
            with self.assertRaisesRegex(ValueError, "stack"):
                with steering.composite_context(self.model, self.model.layers, plan, target, positions, adapter):
                    pass
            self.model(self.hidden, self.side)
        self.assert_clean()

    def test_instance_forward_restoration_and_backward_exception(self):
        layer = self.model.layers[0]
        original = layer.forward
        layer.forward = original
        plan = self.schedule([Stage(1, targets={"F": 4})])
        adapter = LowRankSteering(8)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with steering.composite_context(self.model, self.model.layers, plan, SteeringTarget("t", 3, "Q"),
                                             self.partition.literal_user_prompt.positions, adapter, True):
                result = self.model(self.hidden, self.side)
                self.model.layers[5].fail = True
                result[0, -1, 0].backward()
        self.assertIs(layer.__dict__["forward"], original)
        del layer.forward
        self.assert_clean()

    def test_bf16_frozen_base_train_adapter_two_steps(self):
        self.model.to(torch.bfloat16)
        self.hidden = self.hidden.bfloat16()
        plan = self.schedule([Stage(1, targets={"C": 4}), Stage(3, targets={"F": 4})])
        target = SteeringTarget("actual_FQ", 3, "FQ")
        before = [p.detach().clone() for p in self.model.parameters()]

        @contextmanager
        def forward(prepared, adapter, placeholder, training):
            self.assertEqual(placeholder.name, "l34_cq")
            with steering.composite_context(self.model, self.model.layers, plan, target,
                                             steering.target_positions(self.partition, "FQ"), adapter, training):
                yield self.model(self.hidden, self.side)[0, -1, :2].float()

        rows = [TrainingRow("train", {"split": "train"}, 1, "train")]
        with tempfile.TemporaryDirectory() as directory:
            result = train_adapter(self.model, rows, hidden_size=8, target="l34_cq", prepare=lambda row: row,
                                   forward=forward, checkpoint_dir=directory, seed=17,
                                   optimizer_steps=2, canary=True)
            self.assertEqual(result.optimizer_steps, 2)
            self.assertTrue(result.checkpoint.exists())
            self.assertTrue(all(r.nonzero_gradient and r.nonzero_update and r.frozen_base_checked for r in result.receipts))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, self.model.parameters())))
        self.assert_clean()

    def test_worker_real_training_artifacts_three_variants_and_actual_checkpoint(self):
        from research.lifecycle_memory import run_decoder_application_pilots as pilot
        self.model = Model(count=37).to(torch.bfloat16).eval().requires_grad_(False)
        self.hidden = self.hidden.bfloat16()
        plan = self.schedule([Stage(22, targets={"C": 4})])
        case = steering.cases()[-1]
        prepared = SimpleNamespace(pilot_partition=self.partition)
        cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                               max_memory_allocated=lambda: 0, empty_cache=lambda: None)
        cpu = SimpleNamespace(cuda=cuda, no_grad=torch.no_grad, equal=torch.equal,
                              isfinite=torch.isfinite, bfloat16=torch.bfloat16)
        engine = SimpleNamespace(model=self.model, layers=self.model.layers, device="cpu", torch=cpu,
            prepare=lambda row, directory: prepared,
            logits=lambda prepared: self.model(self.hidden, self.side)[0, -1, :2].float())
        from research.lifecycle_memory.application_pilot_data import build_dataset, export_data
        from research.lifecycle_memory.tests.test_application_pilot_data import battery
        data = build_dataset(*battery(), expected_frames=10)
        rows = data.public_rows

        def measure(_, fn):
            start = time.perf_counter()
            with torch.no_grad():
                result = fn()
            return result, {"elapsed_ms": (time.perf_counter() - start) * 1000,
                            "synchronized": True, "batch_size": 1, "device_id": "cpu-fixture"}

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            dataset = out / "dataset"
            export_data(data, dataset)
            args = SimpleNamespace(rank=13, world_size=28, mode="canary", data_dir=out)
            with patch.object(pilot, "read_rows", wraps=pilot.read_rows) as reader, \
                    patch.object(pilot, "measure", side_effect=measure), \
                    patch.object(steering, "_schedule", return_value=plan):
                count, projected, technical = steering.steering_worker(args, config(), engine, rows, dataset, out)
            reader.assert_called_once_with(dataset / "labels/train.jsonl")
            self.assertEqual(count, 8 * 3)
            self.assertGreater(projected, 0)
            self.assertEqual(technical["status"], "pass")
            predictions = pilot.read_rows(out / "predictions.jsonl")
            timing = pilot.read_rows(out / "timings.jsonl")
            self.assertEqual({row["variant"] for row in predictions}, set(steering.VARIANTS))
            self.assertEqual({row["comparison_target"] for row in predictions}, {case["id"]})
            self.assertEqual(len(timing), 8 * 3 * 2)
            self.assertTrue(all("correct_label" not in row for row in predictions))
            taskdir = out / f"{case['id']}_seed17"
            receipt = pilot.read_json(taskdir / "training_receipt.json")
            self.assertEqual(receipt["actual_target"], asdict(SteeringTarget(case["id"], 34, "FQ")))
            self.assertEqual(receipt["legacy_driver_target"]["name"], "l34_cq")
            self.assertEqual(receipt["training_examples_seen"], 8)
            checkpoint = taskdir / receipt["checkpoint"]
            self.assertEqual(pilot.digest(checkpoint), receipt["checkpoint_sha256"])
            self.assertIn(case["id"], checkpoint.name)
            self.assertFalse(list(taskdir.glob("l34_cq*")))
            if checkpoint.suffix == ".safetensors":
                from safetensors import safe_open
                with safe_open(checkpoint, framework="pt") as handle:
                    metadata = handle.metadata()
                    self.assertEqual(metadata["target"], case["id"])
                    self.assertEqual(metadata["positions"], "FQ")
                    self.assertEqual(metadata["layer_index"], "34")
        self.assert_clean()

    def test_pinned_tiny_gemma_compression_backward(self):
        try:
            import transformers
        except ImportError:
            self.skipTest("optional isolated Transformers 5.13.0 runtime unavailable")
        if transformers.__version__ != "5.13.0":
            self.skipTest("requires pinned Transformers 5.13.0")
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig
        model_config = Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=60, num_attention_heads=4, num_key_value_heads=2,
            num_global_key_value_heads=2, head_dim=8, global_head_dim=8,
            hidden_size_per_layer_input=0, num_kv_shared_layers=0, attention_k_eq_v=True,
            sliding_window=8, max_position_embeddings=32, attention_dropout=.3)
        model_config._attn_implementation = "sdpa"
        model = Gemma4ForCausalLM(model_config).eval().bfloat16().requires_grad_(False)
        layers = model.model.layers
        ids = torch.arange(self.n)[None] % 60 + 2
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        frozen = {name: value.detach().clone() for name, value in model.state_dict().items()}
        exports = [layer.self_attn.store_full_length_kv for layer in layers]

        def logits():
            return model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[0, -1, [10, 11]]

        for stages, role in (([Stage(22, targets={"C": 4})], "FQ"),
                             ([Stage(34, targets={"F": 4})], "CQ"),
                             ([Stage(22, targets={"F": 4}), Stage(40, targets={"F": 2})], "CQ")):
            with self.subTest(stages=stages):
                plan = self.schedule(stages)
                adapter = LowRankSteering(32)
                with torch.no_grad(), matrix_context(layers, plan):
                    baseline = logits()
                with steering.composite_context(model, layers, plan, SteeringTarget("tiny", 34, role),
                        steering.target_positions(self.partition, role), adapter, True) as audit:
                    value = logits()
                    self.assertTrue(torch.equal(value, baseline))
                    torch.nn.functional.cross_entropy(value.float()[None], torch.tensor([1])).backward()
                    self.assertEqual(audit["steering_receipt"].checkpointed_blocks, 25)
                    self.assertTrue(all(audit["block_calls"][str(i)] > 1 for i in range(35, 60)))
                    self.assertGreater(adapter.up.weight.grad.norm().item(), 0)
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
                self.assertEqual(exports, [layer.self_attn.store_full_length_kv for layer in layers])
                self.assertTrue(all(torch.equal(value, frozen[name]) for name, value in model.state_dict().items()))
                self.assertTrue(all("forward" not in layer.__dict__ and not layer._forward_hooks
                                    and not layer._forward_pre_hooks for layer in layers))


def public_fixture(frames=2):
    return [{"request_id": f"{split}-{frame}-{order}", "frame_key": f"{split}-{frame}",
             "cluster_key": f"{split}-participant-{frame}", "split_group_key": f"{split}-group-{frame}",
             "order_id": f"order_{order}", "split": split, "arm": "real"}
            for split in ("train", "validation", "test") for frame in range(frames) for order in (0, 1)]


def fake_budget(seed, steps=2):
    return {"seed": seed, "rank": 4, "steps": steps, "gradient_accumulation": 4,
            "learning_rate": .001, "checkpoint_selection": "fixed_final_step", "base_dtype": "bfloat16",
            "data_order_sha256": "f" * 64, "training_pool_size": 6,
            "training_examples_seen": steps * 4, "parameters": 64}


def analysis_fixture():
    rows = public_fixture(3)
    labels = [{**row, "correct_label": "A" if row["order_id"] == "order_0" else "B"} for row in rows]
    by_id = {row["request_id"]: row for row in rows}
    preds, timings = [], []
    for case, seed, variant, request in sorted(steering.expected_prediction_keys(rows, "canary", config())):
        row = by_id[request]
        pred = {**row, "task_id": "steering", "comparison_target": case, "seed": seed,
                "variant": variant, "prediction": "A" if row["order_id"] == "order_0" else "B",
                "margin_ab": 1. if row["order_id"] == "order_0" else -1.,
                "matched_training_budget": fake_budget(seed)}
        preds.append(pred)
        for repeat in (0, 1):
            timings.append({**pred, "repeat_id": repeat, "is_warmup": False, "warmup_complete": True,
                "elapsed_ms": 10., "synchronized": True, "batch_size": 1, "device_id": "cpu-fixture",
                "timing_scope": steering.TIMING_SCOPE})
    return rows, labels, preds, timings


class InventoryAnalysisTests(unittest.TestCase):
    def test_actual_dataset_export_reaches_training_without_public_mutation(self):
        from research.lifecycle_memory.application_pilot_data import build_dataset, export_data, read_json_rows
        from research.lifecycle_memory.tests.test_application_pilot_data import battery
        data = build_dataset(*battery(), expected_frames=10)
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            export_data(data, dataset)
            rows = read_json_rows(dataset / "public/requests.jsonl")
            original = copy.deepcopy(rows)
            values = steering.training_rows(rows, dataset)
        self.assertEqual(rows, original)
        self.assertEqual(len(values), 12)
        self.assertEqual({r.row_id for r in values},
                         {r["request_id"] for r in rows if r["split"] == "train"})
        self.assertTrue(all("correct_label" not in r.row for r in values))

    def test_export_metadata_cannot_smuggle_labels_or_alternative_inputs(self):
        from research.lifecycle_memory.application_pilot_data import build_dataset
        from research.lifecycle_memory.tests.test_application_pilot_data import battery
        rows = build_dataset(*battery(), expected_frames=10).public_rows
        for key in ("correct_label", "answer", "oracle_donor", "unknown_metadata"):
            with self.subTest(key=key):
                bad = copy.deepcopy(rows)
                bad[0][key] = "secret"
                with self.assertRaisesRegex(ValueError, "label-free.*" + key):
                    steering._public_only(bad)
        for update in ({"schema_version": 99}, {"schema_version": True},
                       {"frame_unit_id": "other-frame"},
                       {"counterbalanced_request_id": "missing-request"},
                       {"arms": {"wrong_time": {"inputs": rows[0]["inputs"]}}},
                       {"arms": {"real": {"inputs": rows[0]["inputs"], "answer": "A"}}}):
            with self.subTest(update=update):
                bad = copy.deepcopy(rows)
                bad[0].update(update)
                with self.assertRaises(ValueError):
                    steering._public_only(bad)

    def test_import_safe_and_exact_geometry_cases(self):
        script = "from research.lifecycle_memory.decoder_token_matrix_steering import cases; import sys; assert len(cases()) == 14; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", script], check=True)
        values = steering.cases()
        self.assertEqual(len(values), len({case["id"] for case in values}))
        self.assertEqual([case["target"] for case in values[:5]], ["C", "Q", "CQ", "FQ", "readout"])
        arms = {arm["id"]: arm for arm in build_arms()}
        self.assertEqual(arms[values[-1]["compression_id"]]["stages"], [{"cut": 22, "method": "merge", "targets": {"C": 379}}])
        renamed = [{**arm, "id": "new-" + arm["id"]} for arm in build_arms()]
        with patch("research.lifecycle_memory.decoder_token_matrix.build_arms", return_value=renamed):
            self.assertTrue(all(case["compression_id"].startswith("new-") for case in steering.cases()))

    def test_inventory_and_config(self):
        rows = public_fixture(40)
        self.assertEqual(len(steering.expected_prediction_keys(rows, "canary", config())), 14 * 2 * 3 * 8)
        self.assertEqual(len(steering.expected_timing_keys(rows, "full", config())), 14 * 2 * 3 * 64 * 3)
        steering.validate_config(config())
        bad = config()
        bad["steering"]["steps"] = 71
        with self.assertRaises(ValueError):
            steering.validate_config(bad)

    def test_train_file_only_and_no_eval_labels(self):
        rows = public_fixture()
        training_labels = [{**row, "correct_label": "A"} for row in rows if row["split"] == "train"]
        reads = []

        def read(path):
            reads.append(str(path))
            return training_labels

        with patch("research.lifecycle_memory.run_decoder_application_pilots.read_rows", side_effect=read):
            values = steering.training_rows(rows, Path("/unused/dataset"))
            self.assertEqual(reads, ["/unused/dataset/labels/train.jsonl"])
            self.assertTrue(all(v.split == "train" and "correct_label" not in v.row for v in values))
            with self.assertRaisesRegex(ValueError, "label-free"):
                steering.training_rows([{**rows[0], "oracle_donor": "x"}], Path("/unused"))
            training_labels[0]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "identity"):
                steering.training_rows(rows, Path("/unused"))

    def test_paired_case_specific_report_test_primary_and_validation_bias_only(self):
        rows, labels, preds, timings = analysis_fixture()
        from research.lifecycle_memory import application_pilot_analysis as analysis
        fitted_splits = []
        original_fit = analysis.fit_ab_bias

        def fit(cell):
            fitted_splits.extend(row["split"] for row in cell.values())
            return original_fit(cell)

        with patch.object(analysis, "fit_ab_bias", side_effect=fit):
            report = steering.aggregate_steering(preds, timings, labels, rows, config())
        self.assertEqual(len(report["reports"]), 14)
        self.assertEqual(report["inference"]["bootstrap_draws"], 100)
        self.assertEqual(report["inference"]["bootstrap_seed"], 12345)
        self.assertEqual(set(fitted_splits), {"validation"})
        self.assertTrue(all(r["split_scope"] == "test" and len(r["comparisons"]) == 3 for r in report["reports"]))
        self.assertIn("not fresh confirmation", steering.render_report(report))
        for result in report["reports"]:
            for contrast in result["comparisons"]:
                self.assertEqual(contrast["participant_equal_difference_variant_minus_reference"]["accuracy"]["estimate"], 0.)
                self.assertEqual(set(contrast["per_seed"]), {"17", "29"})
                self.assertTrue(contrast["timing"]["available"])
                self.assertEqual(contrast["timing"]["speedup_reference_over_variant"], 1.)
                loss = contrast["binary_log_loss_nats"]
                self.assertAlmostEqual(loss["reference"]["estimate"], math.log(1.5))
                self.assertEqual(loss["difference_variant_minus_reference"]["estimate"], 0.)
        with self.assertRaisesRegex(ValueError, "prediction inventory"):
            steering.aggregate_steering(preds[:-1], timings, labels, rows, config())
        with self.assertRaisesRegex(ValueError, "timing inventory"):
            steering.aggregate_steering(preds, timings[:-1], labels, rows, config())
        bad = copy.deepcopy(preds)
        bad[0]["matched_training_budget"]["steps"] = 256
        with self.assertRaisesRegex(ValueError, "budget"):
            steering.aggregate_steering(bad, timings, labels, rows, config())
        bad = copy.deepcopy(preds)
        bad[0]["error"] = "failed request with leftover finite margins"
        with self.assertRaisesRegex(ValueError, "error predictions"):
            steering.aggregate_steering(bad, timings, labels, rows, config())

    def test_logloss_equal_participants_and_seeds_not_requests(self):
        import numpy as np

        def cell(margins):
            return {str(i): {"cluster_key": cluster, "margin": margin, "correct_label": "A"}
                    for i, (cluster, margin) in enumerate(margins)}

        left = [cell([("x", 0.), ("x", 0.), ("y", 2.)]),
                cell([("x", 1.), ("y", 3.), ("y", 3.)])]
        right = [cell([("x", 10.), ("x", 10.), ("y", 10.)]),
                 cell([("x", 10.), ("y", 10.), ("y", 10.)])]
        indexes = np.array([[0, 0], [1, 1], [0, 1]])
        result = steering._log_loss_comparison(left, right, indexes)
        loss = lambda margin: math.log1p(2 ** (-margin))
        expected = sum(loss(margin) for margin in (0., 1., 2., 3.)) / 4
        self.assertAlmostEqual(result["reference"]["estimate"], expected)
        self.assertAlmostEqual(result["variant"]["estimate"], loss(10.))
        self.assertAlmostEqual(result["difference_variant_minus_reference"]["estimate"], loss(10.) - expected)
        self.assertLess(result["difference_variant_minus_reference"]["ci95"][1], 0.)


if __name__ == "__main__":
    unittest.main()
