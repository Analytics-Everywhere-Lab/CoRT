"""Offline launch/coordinator contracts with temporary synthetic fixtures only."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from research.lifecycle_memory.temporal_forecasting import cli, execution, hpc, recognition_launch as launch
from research.lifecycle_memory.temporal_forecasting.common import atomic_json, digest, read_json, sha256
from research.lifecycle_memory.temporal_forecasting.protocol import GEMMA, MODELS, assign_tasks, inventory, protocol_contract
from research.lifecycle_memory.temporal_forecasting.recognition_analysis import validate_recognition_technical
from research.lifecycle_memory.temporal_forecasting.recognition_protocol import TASKS
from research.lifecycle_memory.tests.test_temporal_dense_execution import FakeDenseBackend
from research.lifecycle_memory.tests.test_temporal_recognition_analysis import POLICY, fixture, scores


class FakeRecognitionBackend(FakeDenseBackend):
    @staticmethod
    def native(order, condition):
        return scores((1 if order == 0 else -1) * {"B0": 4., "B1": 2., "B4": 1., "B2": 3.}[condition])


class RecognitionLaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.release, self.out = self.root / "release", self.root / "out"
        self.release.mkdir()
        self.sizes = (2,) * 8
        self.rows, self.labels, self.protocol = fixture(sizes=self.sizes)
        self.protocol.pop("episode_ids")
        self.episodes = list({r["episode_id"]: {k: r[k] for k in
            ("episode_id", "cluster_id", "task", "contract_sha256")} for r in self.rows}.values())
        atomic_json(self.release / "protocol.json", self.protocol)
        self.data = self.root / "data"
        self.data.mkdir()
        self.manifest = {"public_sha256": "a" * 64, "evaluator_sha256": "b" * 64,
                         "contract": protocol_contract(self.protocol)}
        atomic_json(self.data / "manifest.json", self.manifest)
        (self.root / "runtime.sif").touch()
        (self.root / "model").mkdir()
        model_inventory = {"schema_version": 1, "model_key": GEMMA, **MODELS[GEMMA],
                           "files": [{"path": "weights.safetensors", "sha256": "c" * 64, "bytes": 1}]}
        model_inventory["inventory_sha256"] = digest(model_inventory)
        atomic_json(self.root / "inventory.json", model_inventory)
        self.dep = {"account": "REPLACE_WITH_ACCOUNT", "data_root": str(self.data),
                    "resources": {"gpus_per_node": 4},
                    "runtime": {"mode": "apptainer", "image": str(self.root / "runtime.sif"),
                                "python": "python", "sha256": "d" * 64,
                                "versions": {"torch": "2.12.1+cu126", "transformers": "5.13.0"}},
                    "models": {GEMMA: {"snapshot": str(self.root / "model"),
                                        "inventory": str(self.root / "inventory.json")}}}
        self.config = {"release_root": str(self.release), "output_root": str(self.out),
                       "profile": "fir", "deployment": self.dep}
        self.config_path = self.root / "config.json"
        atomic_json(self.config_path, self.config)
        self.identity = {"simulation": False, "release_sha256": "e" * 64,
                         "public_sha256": self.manifest["public_sha256"],
                         "evaluator_sha256": self.manifest["evaluator_sha256"],
                         "protocol_sha256": digest(self.protocol), "freeze_sha256": None,
                         "dataset_manifest_sha256": sha256(self.data / "manifest.json"),
                         "runtime_identity": "f" * 64,
                         "runtime_spec_sha256": execution.recognition_runtime_spec(self.dep["runtime"]),
                         "model_inventories": {GEMMA: model_inventory["inventory_sha256"]}}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in ("research.lifecycle_memory.temporal_forecasting.release.verify_release",
                       "research.lifecycle_memory.temporal_forecasting.recognition_launch.verify_release"):
            self.stack.enter_context(patch(target, return_value={"release_sha256": self.identity["release_sha256"]}))
        for target in ("research.lifecycle_memory.temporal_forecasting.execution.verify_loaded_code",
                       "research.lifecycle_memory.temporal_forecasting.recognition_launch.verify_loaded_code"):
            self.stack.enter_context(patch(target, return_value=None))
        for target in ("research.lifecycle_memory.temporal_forecasting.data.validate_dataset",
                       "research.lifecycle_memory.temporal_forecasting.recognition_launch.validate_dataset"):
            self.stack.enter_context(patch(target, return_value={"rows": self.episodes, "manifest": self.manifest}))

    def set_cohort(self, sizes, task=TASKS[0]):
        self.sizes = sizes
        self.rows, self.labels, self.protocol = fixture(sizes=sizes, task=task)
        self.protocol.pop("episode_ids")
        self.episodes[:] = list({r["episode_id"]: {k: r[k] for k in
            ("episode_id", "cluster_id", "task", "contract_sha256")} for r in self.rows}.values())
        self.manifest["contract"] = protocol_contract(self.protocol)
        atomic_json(self.release / "protocol.json", self.protocol)
        atomic_json(self.data / "manifest.json", self.manifest)
        self.identity["protocol_sha256"] = digest(self.protocol)
        self.identity["dataset_manifest_sha256"] = sha256(self.data / "manifest.json")

    def synthetic_receipt(self, simulation=False):
        # Unit-test evidence only. All paths are temporary; nothing is submitted.
        canary_rows, _, _ = fixture(sizes=self.sizes, task=self.episodes[0]["task"], phase="recognition-canary")
        technical = validate_recognition_technical(canary_rows)
        path = self.out / "canary" / "technical_checks.json"
        atomic_json(path, technical)
        tasks = inventory(self.episodes, self.protocol, "recognition-canary")
        receipt = {"status": "pass", "stage": "recognition-canary", "tasks": len(tasks),
                   "identity": {**self.identity, "simulation": simulation}, "elapsed_s": 40., "staging_s": 1.,
                   "task_class_mean_s": {f"{GEMMA}:{kind}": 1. for kind in ("behavior", "patch", "self")},
                   "technical_checks_sha256": sha256(path), "task_inventory_sha256": digest(tasks),
                   "result_set_sha256": "1" * 64}
        atomic_json(path.parent / "receipt.json", receipt)
        return receipt

    def test_metadata_check_and_idempotent_canary_plan(self):
        with patch.object(launch, "submit_plan", side_effect=AssertionError("submission")):
            result = launch.prepare(self.config_path, "check")
            self.assertEqual((result["canary_rows"], result["full_rows"]), (176, 704))
            self.assertFalse(self.out.exists())
            result = launch.prepare(self.config_path, "canary")
            self.assertEqual(result, launch.prepare(self.config_path, "canary"))
        script = (Path(result["plan_dir"]) / "job.sh").read_text()
        self.assertIn("--stage recognition-canary", script)
        self.assertEqual(result["walltime"], "00:30:00")
        self.config["deployment"]["account"] = "changed"
        atomic_json(self.config_path, self.config)
        with self.assertRaisesRegex(ValueError, "deployment changed"):
            launch.prepare(self.config_path, "canary")

    def test_full_requires_canary_and_uses_measured_timing(self):
        with self.assertRaises(FileNotFoundError):
            launch.prepare(self.config_path, "full")
        self.synthetic_receipt()
        with patch.object(launch, "canary_admission", wraps=execution.canary_admission) as admission:
            result = launch.prepare(self.config_path, "full")
        self.assertEqual(result["stage"], "recognition")
        self.assertEqual(result["rows"], 704)
        self.assertEqual(len(admission.call_args.args[2]), 704)
        self.assertEqual(result["walltime"], "01:00:00")
        self.assertEqual(result, launch.prepare(self.config_path, "full"))

    def test_simulation_and_missing_real_identity_never_authorize_full(self):
        self.synthetic_receipt(simulation=True)
        with self.assertRaisesRegex(ValueError, "invalid"):
            launch.prepare(self.config_path, "full")
        for key in ("runtime_identity", "runtime_spec_sha256", "model_inventories", "simulation"):
            receipt = self.synthetic_receipt()
            receipt["identity"].pop(key)
            atomic_json(self.out / "canary/receipt.json", receipt)
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.prepare(self.config_path, "full")

    def compute_fixture(self, stack):
        hardware = [{"name": "H100", "memory_bytes": 80_000_000_000}] * 4
        self.identity["runtime_identity"] = digest({
            "sha256": self.dep["runtime"]["sha256"], "versions": self.dep["runtime"]["versions"],
            "hardware_types": sorted({(h["name"], h["memory_bytes"]) for h in hardware})})
        dep = {**self.dep, "local_scratch": str(self.root), "walltime_seconds": 3600,
               "canary_receipt": str(self.out / "canary/receipt.json")}
        path = self.root / "execute.json"
        atomic_json(path, dep)
        stack.enter_context(patch.object(execution, "require_compute", return_value=hardware))
        stack.enter_context(patch.object(execution.importlib.metadata, "version",
                                        side_effect=lambda p: dep["runtime"]["versions"][p]))
        staged = stack.enter_context(patch.object(execution, "stage_inputs", return_value=(
            self.episodes, {GEMMA: "/unused"}, self.identity["model_inventories"])))
        dispatch = stack.enter_context(patch.object(execution.subprocess, "Popen",
                                                   side_effect=RuntimeError("dispatch reached")))
        stack.enter_context(patch.object(execution, "read_jsonl", side_effect=AssertionError("private labels read")))
        return path, staged, dispatch

    def test_full_floor_at_prepare_preflight_and_execute_for_both_tasks(self):
        for task in TASKS:
            for sizes in ((2,) * 7 + (1,), (3, 3, 2, 2, 2, 2, 2)):
                with self.subTest(task=task, episodes=sum(sizes), wearers=len(sizes)), ExitStack() as stack:
                    self.set_cohort(sizes, task)
                    path, staged, dispatch = self.compute_fixture(stack)
                    self.synthetic_receipt()
                    with self.assertRaisesRegex(ValueError, "16 episodes and 8 wearers"):
                        launch.prepare(self.config_path, "full")
                    with self.assertRaisesRegex(ValueError, "16 episodes and 8 wearers"):
                        cli.preflight_plan(self.release, read_json(path), "recognition")
                    with self.assertRaisesRegex(ValueError, "16 episodes and 8 wearers"):
                        execution.execute(self.release, path, "recognition", self.root / "underfloor")
                    staged.assert_not_called()
                    dispatch.assert_not_called()
                    self.assertFalse((self.out / "full.deployment.json").exists())

    def test_exact_floor_passes_prepare_preflight_and_compute_dispatch(self):
        for task in TASKS:
            with self.subTest(task=task), ExitStack() as stack:
                self.set_cohort((2,) * 8, task)
                path, _, dispatch = self.compute_fixture(stack)
                self.synthetic_receipt()
                self.assertEqual(launch.prepare(self.config_path, "full")["rows"], 704)
                self.assertEqual(cli.preflight_plan(self.release, read_json(path), "recognition")["status"], "pass")
                with self.assertRaisesRegex(RuntimeError, "dispatch reached"):
                    execution.execute(self.release, path, "recognition", self.root / task)
                dispatch.assert_called_once()

    def test_four_episode_canary_remains_valid_at_prepare_and_compute(self):
        for task in TASKS:
            with self.subTest(task=task), ExitStack() as stack:
                self.set_cohort((1,) * 4, task)
                path, _, dispatch = self.compute_fixture(stack)
                self.assertEqual(launch.prepare(self.config_path, "canary")["rows"], 176)
                with self.assertRaisesRegex(RuntimeError, "dispatch reached"):
                    execution.execute(self.release, path, "recognition-canary", self.root / task)
                dispatch.assert_called_once()

    def test_canary_sidecar_is_rechecked_at_execute_and_preflight(self):
        for state in ("absent", "mutated", "nonpass", "contradictory", "malformed"):
            with self.subTest(state=state), ExitStack() as stack:
                path, _, dispatch = self.compute_fixture(stack)
                receipt = self.synthetic_receipt()
                sidecar = self.out / "canary/technical_checks.json"
                if state == "absent":
                    sidecar.unlink()
                else:
                    report = ({"status": "pass", "passed": True, "changed": True} if state == "mutated"
                              else {"status": "fail", "passed": state == "contradictory"})
                    atomic_json(sidecar, [] if state == "malformed" else report)
                    if state != "mutated":
                        receipt["technical_checks_sha256"] = sha256(sidecar)
                        atomic_json(self.out / "canary/receipt.json", receipt)
                with self.assertRaisesRegex(ValueError, "technical report"):
                    cli.preflight_plan(self.release, read_json(path), "recognition")
                run = self.root / state
                with self.assertRaisesRegex(ValueError, "technical report"):
                    execution.execute(self.release, path, "recognition", run)
                dispatch.assert_not_called()
                self.assertFalse((run / "worker_context.json").exists())
                self.assertFalse((run / "receipt.json").exists())

    def test_wrong_code_data_protocol_runtime_model_and_technical_bindings_reject(self):
        for key in ("release_sha256", "public_sha256", "protocol_sha256", "evaluator_sha256",
                    "dataset_manifest_sha256", "runtime_spec_sha256"):
            receipt = self.synthetic_receipt()
            receipt["identity"][key] = "0" * 64
            atomic_json(self.out / "canary/receipt.json", receipt)
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.prepare(self.config_path, "full")
        for key in ("technical_checks_sha256", "task_inventory_sha256", "stage"):
            receipt = self.synthetic_receipt()
            receipt[key] = "0" * 64
            atomic_json(self.out / "canary/receipt.json", receipt)
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.prepare(self.config_path, "full")
        receipt = self.synthetic_receipt()
        receipt["identity"]["model_inventories"][GEMMA] = "0" * 64
        atomic_json(self.out / "canary/receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "model_inventories"):
            launch.prepare(self.config_path, "full")

    def test_budget_missing_classes_invalid_costs_and_strict_23_hour_limit(self):
        tasks = inventory(self.episodes, self.protocol, "recognition")
        for cost in (-1., 0., float("nan"), float("inf")):
            receipt = self.synthetic_receipt()
            receipt["task_class_mean_s"][f"{GEMMA}:patch"] = cost
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                execution.canary_admission(receipt, self.identity, tasks, 4, 82800)
        receipt = self.synthetic_receipt()
        receipt["task_class_mean_s"].pop(f"{GEMMA}:self")
        with self.assertRaisesRegex(ValueError, "every full-run task class"):
            execution.canary_admission(receipt, self.identity, tasks, 4, 82800)
        self.synthetic_receipt()
        with patch.object(launch, "canary_admission", return_value=82500), self.assertRaisesRegex(ValueError, "below.*23-hour"):
            launch.prepare(self.config_path, "full")

    def test_hpc_supports_both_recognition_stages_without_changing_legacy(self):
        for profile in ("fir", "tamia", "killarney", "generic-slurm", "local-cloud"):
            for stage in ("recognition", "recognition-canary"):
                dep = deepcopy(self.dep)
                dep["resources"].update(gpu_type="h100", gpu_memory_gb=80, cpus_per_task=12,
                                        memory_gb=480, node_cpu_limit=48, node_memory_gb=512)
                dep.update(deployment_path=str(self.root / "deployment.json"), walltime_seconds=3600)
                if profile == "local-cloud":
                    dep.pop("account")
                    dep["local_scratch"] = str(self.root / "scratch")
                plan = hpc.build_plan(hpc.load_profile(profile), dep, stage, self.root / "run", self.release, "01:00:00")
                self.assertIn(f"--stage {stage}", plan["script_text"])

    def test_cli_constructs_both_recognition_protocols_and_rejects_legacy_flags(self):
        policy = self.root / "policy.json"
        atomic_json(policy, POLICY)
        for task in TASKS:
            path = self.root / f"{task}.json"
            with redirect_stdout(io.StringIO()):
                cli.main(["init-protocol", "--task", task, "--matched-policy", str(policy),
                          "--episode-limit", "6", "--minimum-clusters", "4", "--out", str(path)])
            self.assertEqual(read_json(path)["episode_limit"], 6)
            self.assertEqual(read_json(path)["schema_version"], 4)
        with self.assertRaisesRegex(ValueError, "recognition-only"):
            cli.main(["init-protocol", "--episode-limit", "6", "--out", str(self.root / "bad.json")])
        args = cli.parser().parse_args(["estimate", "--release", "/r", "--deployment", "/d",
                                       "--stage", "recognition", "--gpus", "4"])
        self.assertEqual(args.stage, "recognition")

    def test_simulated_worker_resume_label_boundary_and_no_real_backend_injection(self):
        tasks = inventory(self.episodes, self.protocol, "recognition-canary")
        context = {"run_dir": str(self.root / "simulated"), "identity": {"simulation": True},
                   "contract": protocol_contract(self.protocol), "assignments": assign_tasks(tasks, 1),
                   "episodes": self.episodes, "model_paths": {GEMMA: "/not-a-model"},
                   "attention_backend": "sdpa", "deadline_epoch": time.time() + 60}
        path = self.root / "context.json"
        atomic_json(path, context)
        FakeRecognitionBackend.calls = 0
        self.assertEqual(execution.run_worker(path, 0, FakeRecognitionBackend)["status"], "pass")
        self.assertEqual(FakeRecognitionBackend.calls, 176)
        execution.run_worker(path, 0, FakeRecognitionBackend)
        self.assertEqual(FakeRecognitionBackend.calls, 176)
        rows = execution.collect(context["run_dir"], tasks, context["identity"])
        self.assertTrue(validate_recognition_technical(rows)["passed"])
        with self.assertRaisesRegex(ValueError, "simulation"):
            execution.canary_admission({"stage": "recognition-canary", "status": "pass", "identity": context["identity"]},
                                      context["identity"], tasks, 1, 82800)
        context["identity"]["simulation"] = False
        atomic_json(path, context)
        with self.assertRaisesRegex(ValueError, "injected.*simulation"):
            execution.run_worker(path, 0, FakeRecognitionBackend)
        context["identity"]["simulation"] = True
        context["episodes"][0]["correct_candidate"] = 0
        atomic_json(path, context)
        with self.assertRaisesRegex(ValueError, "label-free"):
            execution.run_worker(path, 0, FakeRecognitionBackend)

    def test_coordinator_checks_controls_before_passing_receipt_without_labels(self):
        rows, _, _ = fixture(sizes=self.sizes, phase="recognition-canary")
        next(r for r in rows if r["kind"] == "patch")["controls"].pop("full_A")
        dep = {**self.dep, "local_scratch": str(self.root), "walltime_seconds": 1800}
        deployment_path = self.root / "execute.json"
        atomic_json(deployment_path, dep)
        run = self.root / "coordinator"
        with ExitStack() as stack:
            stack.enter_context(patch.object(execution, "require_compute", return_value=[{"name": "H100", "memory_bytes": 80_000_000_000}] * 4))
            stack.enter_context(patch.object(execution.importlib.metadata, "version", side_effect=lambda p: dep["runtime"]["versions"][p]))
            stack.enter_context(patch.object(execution, "stage_inputs", return_value=(self.episodes, {GEMMA: "/unused"}, self.identity["model_inventories"])))
            process = stack.enter_context(patch.object(execution.subprocess, "Popen")).return_value
            process.poll.return_value = 0
            process.returncode = 0
            stack.enter_context(patch.object(execution, "collect", return_value=rows))
            stack.enter_context(patch.object(execution, "read_jsonl", side_effect=AssertionError("private labels read")))
            with self.assertRaisesRegex(ValueError, "recognition technical controls failed"):
                execution.execute(self.release, deployment_path, "recognition-canary", run)
        self.assertFalse((run / "receipt.json").exists())
        self.assertFalse((run / "completion.json").exists())
        self.assertFalse(read_json(run / "technical_checks.json")["passed"])

    def test_real_recognition_worker_records_lifetime_cuda_peaks_on_pass_and_fail(self):
        import torch

        task = next(t for t in inventory(self.episodes, self.protocol, "recognition-canary")
                    if t["kind"] == "behavior")
        for failing in (False, True):
            with self.subTest(failing=failing), ExitStack() as stack:
                run = self.root / f"memory-{failing}"
                context = {"run_dir": str(run), "identity": self.identity,
                           "contract": protocol_contract(self.protocol), "assignments": [[], [task]],
                           "episodes": self.episodes, "model_paths": {GEMMA: "/not-a-model"},
                           "attention_backend": "sdpa", "deadline_epoch": time.time() + 60}
                path = self.root / "memory-context.json"
                atomic_json(path, context)
                stack.enter_context(patch("research.lifecycle_memory.temporal_forecasting.runtime.HFBackend",
                                          FakeRecognitionBackend))
                allocated = stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=123456))
                reserved = stack.enter_context(patch.object(torch.cuda, "max_memory_reserved", return_value=234567))
                properties = stack.enter_context(patch.object(torch.cuda, "get_device_properties"))
                properties.return_value.total_memory = 80_000_000_000
                reset = stack.enter_context(patch.object(torch.cuda, "reset_peak_memory_stats"))
                empty = stack.enter_context(patch.object(torch.cuda, "empty_cache"))
                if failing:
                    stack.enter_context(patch.object(FakeRecognitionBackend, "score", side_effect=RuntimeError("mock OOM")))
                    with self.assertRaisesRegex(RuntimeError, "mock OOM"):
                        execution.run_worker(path, 1)
                else:
                    execution.run_worker(path, 1)
                receipt = read_json(run / "workers/1.json")
                self.assertEqual(receipt["status"], "fail" if failing else "pass")
                self.assertEqual(receipt["peak_memory_allocated_bytes"], 123456)
                self.assertEqual(receipt["peak_memory_reserved_bytes"], 234567)
                self.assertEqual(receipt["gpu_total_memory_bytes"], 80_000_000_000)
                allocated.assert_called_once_with("cuda:1")
                reserved.assert_called_once_with("cuda:1")
                properties.assert_called_once_with("cuda:1")
                reset.assert_not_called()
                empty.assert_not_called()

    def test_memory_probe_skips_simulation_and_legacy_workers(self):
        import torch

        for recognition, simulation in ((True, True), (False, False)):
            with self.subTest(recognition=recognition), ExitStack() as stack:
                context = {"run_dir": str(self.root / f"no-memory-{recognition}"),
                           "identity": {"simulation": simulation}, "assignments": [[]], "episodes": []}
                if recognition:
                    context["contract"] = protocol_contract(self.protocol)
                path = self.root / "no-memory-context.json"
                atomic_json(path, context)
                allocated = stack.enter_context(patch.object(torch.cuda, "max_memory_allocated"))
                reserved = stack.enter_context(patch.object(torch.cuda, "max_memory_reserved"))
                properties = stack.enter_context(patch.object(torch.cuda, "get_device_properties"))
                receipt = execution.run_worker(path, 0)
                self.assertEqual(receipt["status"], "pass")
                self.assertFalse(any(k.startswith("peak_memory") for k in receipt))
                self.assertNotIn("gpu_total_memory_bytes", receipt)
                allocated.assert_not_called()
                reserved.assert_not_called()
                properties.assert_not_called()

    def analysis_fixture(self):
        run = self.root / "analysis-run"
        tasks = inventory(self.episodes, self.protocol, "recognition")
        for row in self.rows:
            result = {**row, "identity": self.identity}
            result["result_sha256"] = digest(result)
            atomic_json(execution.task_result_path(run, row["task_id"]), result)
        rows = execution.collect(run, tasks, self.identity)
        atomic_json(run / "technical_checks.json", validate_recognition_technical(rows))
        atomic_json(run / "worker_context.json", {"identity": self.identity, "stage": "recognition",
                    "task_inventory_sha256": digest(tasks), "freeze": None})
        atomic_json(run / "receipt.json", {"status": "pass", "stage": "recognition", "identity": self.identity,
                    "task_inventory_sha256": digest(tasks), "tasks": len(tasks),
                    "technical_checks_sha256": sha256(run / "technical_checks.json"),
                    "result_set_sha256": digest(sorted(r["result_sha256"] for r in rows))})
        return run

    def test_full_analysis_dispatches_only_recognition_and_frozen_bootstrap(self):
        from research.lifecycle_memory.temporal_forecasting.recognition_analysis import analyze_recognition

        run = self.analysis_fixture()
        def bounded(*args, **kwargs):
            self.assertEqual(kwargs["draws"], 19999)
            return analyze_recognition(*args, draws=12)
        with patch.object(execution, "read_jsonl", return_value=self.labels), \
                patch("research.lifecycle_memory.temporal_forecasting.analysis.analyze", side_effect=AssertionError("legacy analyzer invoked")), \
                patch("research.lifecycle_memory.temporal_forecasting.recognition_analysis.analyze_recognition", side_effect=bounded) as analyzer:
            result = execution.analyze_run(run, self.release, self.data)
        self.assertTrue(result["analysis_complete"])
        self.assertEqual(analyzer.call_count, 1)
        self.assertEqual(read_json(run / "completion.json")["analysis_sha256"], sha256(run / "analysis.json"))

    def test_analysis_rejects_stale_controls_draw_override_and_wrong_stage_before_labels(self):
        run = self.analysis_fixture()
        with patch.object(execution, "read_jsonl", side_effect=AssertionError("labels read before gate")):
            with self.assertRaisesRegex(ValueError, "bootstrap count is frozen"):
                execution.analyze_run(run, self.release, self.data, draws=10)
            atomic_json(run / "technical_checks.json", {"passed": True})
            with self.assertRaisesRegex(ValueError, "technical receipt"):
                execution.analyze_run(run, self.release, self.data)
            receipt = read_json(run / "receipt.json")
            receipt["stage"] = "current"
            atomic_json(run / "receipt.json", receipt)
            with self.assertRaisesRegex(ValueError, "matching full stage"):
                execution.analyze_run(run, self.release, self.data)

    def test_invalid_results_never_produce_analysis_completion(self):
        next(r for r in self.rows if r["kind"] == "patch")["controls"].pop("full_A")
        run = self.analysis_fixture()
        with self.assertRaisesRegex(ValueError, "technical receipt"):
            execution.analyze_run(run, self.release, self.data)
        self.assertFalse((run / "completion.json").exists())

    def test_staging_projects_only_public_runtime_inputs(self):
        from research.lifecycle_memory.temporal_forecasting.recognition_protocol import SLOTS

        row = {**self.episodes[0], "correct_candidate": 0, "recognition_provenance": {"private": True},
               "reviewer": "never-stage", "images": {
                   slot: {"path": f"assets/{slot}.png", "sha256": "a" * 64, "bytes": 1} for slot in SLOTS}}
        manifest = {**self.manifest, "asset_inventory": []}
        with patch("PIL.Image.open") as open_image:
            open_image.return_value.__enter__.return_value.size = (10, 10)
            episodes, _, _ = execution.stage_inputs(self.dep, {"manifest": manifest, "rows": [row]}, [], self.root / "scratch")
        self.assertEqual(set(episodes[0]), {"episode_id", "cluster_id", "task", "contract_sha256", "images"})
        self.assertTrue(all(Path(image["path"]).is_absolute() for image in episodes[0]["images"].values()))
        execution._recognition_worker_inputs({"episodes": episodes}, [])


if __name__ == "__main__":
    unittest.main()
