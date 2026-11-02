"""Single-node, multi-GPU execution with durable per-task checkpoints.

Slurm and rented H100 servers share this coordinator. No scheduler commands or
model imports run during planning. Workers receive public inputs only.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gc
import importlib.metadata
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from .common import atomic_json, checked_relative, digest, read_json, read_jsonl, sha256
from .protocol import (MODELS, assign_tasks, inventory, selected_episodes, validate_protocol,
                       protocol_contract, validate_contract, validate_dataset_protocol, validate_rows_contract,
                       is_dense_current)
from .recognition_protocol import TASKS as RECOGNITION_TASKS, is_recognition
from .followup_protocol import (SUITE as FOLLOWUP_SUITE, is_followup,
                                validate_followup_protocol, verify_followup_dataset)


def recognition_runtime_spec(runtime):
    """Portable runtime metadata; the on-compute identity also binds GPU type."""
    if (not isinstance(runtime, dict) or not _sha256_pin(runtime.get("sha256"))
            or not isinstance(runtime.get("versions"), dict)
            or not {"torch", "transformers"} <= runtime["versions"].keys()
            or any(not isinstance(v, str) or not v for v in runtime["versions"].values())):
        raise ValueError("recognition runtime requires SHA256 and pinned package versions")
    return digest({"sha256": runtime["sha256"], "versions": runtime["versions"]})


def _sha256_pin(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def require_recognition_full_cohort(rows):
    """Production floor over validated public episodes, not expanded task rows."""
    if (len({row["episode_id"] for row in rows}) < 16
            or len({row["cluster_id"] for row in rows}) < 8):
        raise ValueError("recognition full run requires at least 16 episodes and 8 wearers")


def verify_recognition_technical_receipt(receipt_path, receipt):
    """Require the hash-bound passing sidecar at both planning and execution."""
    path = Path(receipt_path).parent / "technical_checks.json"
    message = "recognition canary technical report is missing, stale, or nonpassing"
    try:
        if not path.is_file() or sha256(path) != receipt.get("technical_checks_sha256"):
            raise ValueError(message)
        report = read_json(path)
        if (not isinstance(report, dict) or report.get("passed") is not True
                or report.get("status") != "pass"):
            raise ValueError(message)
    except (OSError, ValueError) as error:
        raise ValueError(message) from error
    return report


def recognition_canary_binding(receipt, deployment, tasks):
    """Login-side runtime/model/technical binding, without reading weight shards."""
    expected_models = {task["model_key"] for task in tasks}
    pins = {}
    for model in expected_models:
        inv = read_json(deployment["models"][model]["inventory"])
        body = {k: v for k, v in inv.items() if k != "inventory_sha256"}
        if (inv.get("inventory_sha256") != digest(body) or inv.get("model_key") != model
                or any(inv.get(k) != v for k, v in MODELS[model].items())
                or not any(item["path"].endswith(".safetensors") for item in inv.get("files", []))):
            raise ValueError("recognition model inventory mismatch")
        pins[model] = inv["inventory_sha256"]
    identity = {**receipt.get("identity", {}), "model_inventories": pins,
                "runtime_spec_sha256": recognition_runtime_spec(deployment["runtime"]), "simulation": False}
    # The actual GPU/runtime hash is independently rederived on compute.
    # Full workload timing is checked with the actual allocation by the launcher
    # and again on compute; this call validates receipt evidence and bindings.
    if receipt.get("stage") != "recognition-canary":
        raise ValueError("recognition requires its matching canary stage")
    canary_admission(receipt, identity, [], 1, 23 * 3600)


def _recognition_worker_inputs(context, tasks):
    """Workers consume only image references and opaque task/episode identities."""
    allowed = {"episode_id", "cluster_id", "split", "task", "contract_sha256", "images"}
    task_fields = allowed - {"images"} | {"model_key", "order", "kind", "condition",
        "donor_condition", "recipient_condition", "cut", "position_set", "task_id"}
    if any(set(row) - allowed for row in context["episodes"]) or any(set(t) - task_fields for t in tasks):
        raise ValueError("recognition worker inputs must be label-free public fields")
    for episode in context["episodes"]:
        for image in episode.get("images", {}).values():
            if not isinstance(image, dict) or set(image) - {"path", "sha256", "bytes"}:
                raise ValueError("recognition image references must be label-free")


def verify_loaded_code(release):
    """A separately verified directory is insufficient: bind the imported code."""
    package = Path(__file__).resolve().parent
    prefix = "research/lifecycle_memory/temporal_forecasting/"
    expected = {item["path"].removeprefix(prefix): item["sha256"]
                for item in release["files"] if item["path"].startswith(prefix)}
    actual = {p.name: sha256(p) for p in package.glob("*.py")}
    if actual != expected:
        raise ValueError("loaded implementation differs from the sealed release; run its CLI")


def verify_freeze(freeze, identity, protocol, public_rows):
    if not isinstance(freeze, dict):
        raise ValueError("missing frozen design")
    checksum = digest({k: v for k, v in freeze.items() if k != "freeze_sha256"})
    if checksum != freeze.get("freeze_sha256") or checksum != identity.get("freeze_sha256"):
        raise ValueError("freeze hash differs from the run identity")
    for key in ("release_sha256", "public_sha256", "protocol_sha256", "evaluator_sha256", "dataset_manifest_sha256"):
        if freeze.get(key) != identity.get(key):
            raise ValueError(f"freeze identity differs: {key}")
    expected = [r["episode_id"] for r in selected_episodes(public_rows, protocol, "confirmation")]
    if freeze.get("episode_ids") != expected or freeze.get("bootstrap_draws") != protocol["bootstrap_draws"]:
        raise ValueError("frozen selection/bootstrap differs from protocol")


def require_compute():
    import torch

    if not torch.cuda.is_available():
        raise ValueError("GPU execution is compute-node only; CUDA is unavailable")
    properties = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    if not properties or any("H100" not in p.name.upper() or "MIG" in p.name.upper()
                             or p.total_memory < 80_000_000_000 for p in properties):
        raise ValueError("requires full H100 GPUs with at least 80 GB")
    return [{"name": p.name, "memory_bytes": p.total_memory} for p in properties]


def model_inventory(model_key, snapshot, out_path):
    """Run once on a compute node, not the login node: reads full weight shards."""
    require_compute()
    snapshot = Path(snapshot).resolve()
    if model_key not in MODELS or snapshot.name != MODELS[model_key]["revision"]:
        raise ValueError("snapshot directory must be the exact pinned revision")
    files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            relative = path.relative_to(snapshot).as_posix()
            files.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
    if not any(f["path"].endswith(".safetensors") for f in files) or not any(f["path"] == "config.json" for f in files):
        raise ValueError("snapshot lacks configuration/weight files")
    result = {"schema_version": 1, "model_key": model_key, **MODELS[model_key], "files": files}
    result["inventory_sha256"] = digest(result)
    if Path(out_path).exists():
        if read_json(out_path) != result:
            raise ValueError("refusing to replace a different model inventory")
    else:
        atomic_json(out_path, result)
    return result


def _copy_verified(source, destination, checksum, size):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("staged asset cannot be a symlink")
    if destination.exists():
        if destination.stat().st_size != size or sha256(destination) != checksum:
            raise ValueError(f"staged asset changed: {destination}")
        return
    temporary = destination.with_name(destination.name + f".copy-{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != size or sha256(temporary) != checksum:
            raise ValueError(f"copy verification failed: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def stage_inputs(deployment, validated, models, cache):
    """One shared copy per node; no evaluator is copied to the worker cache."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    source = Path(deployment["data_root"])
    manifest = validated["manifest"]
    # Fixed canvas geometry makes canary memory/latency representative of the
    # full inventory. Construction, not the runtime, owns resizing/cropping.
    from PIL import Image
    geometry = {}
    for row in validated["rows"]:
        for role, spec in row["images"].items():
            family = "candidate" if role.startswith("candidate") else "scene"
            path = checked_relative(source, spec["path"])
            with Image.open(path) as image:
                size = tuple(image.size)
                image.verify()
            if geometry.setdefault(family, size) != size:
                raise ValueError("render a uniform scene canvas and a uniform crop canvas before submission")
    model_specs, inventories = {}, {}
    data_cache = cache / "public-data"
    total = sum(item["bytes"] for item in manifest["asset_inventory"]
                if not checked_relative(data_cache, item["path"]).exists())
    for model in models:
        spec = deployment["models"][model]
        inv = read_json(spec["inventory"])
        body = {k: v for k, v in inv.items() if k != "inventory_sha256"}
        if inv.get("inventory_sha256") != digest(body) or inv.get("model_key") != model:
            raise ValueError("invalid model inventory")
        if any(inv.get(k) != v for k, v in MODELS[model].items()):
            raise ValueError("model inventory revision/configuration mismatch")
        if not any(item["path"].endswith(".safetensors") for item in inv["files"]):
            raise ValueError("model inventory contains no weights")
        inventories[model] = inv
        target = cache / "models" / model / MODELS[model]["revision"]
        total += sum(item["bytes"] for item in inv["files"]
                     if not checked_relative(target, item["path"]).exists())
    if shutil.disk_usage(cache).free < total * 1.10:
        raise ValueError(f"node-local scratch needs at least {int(total * 1.10)} free bytes")
    for item in manifest["asset_inventory"]:
        _copy_verified(checked_relative(source, item["path"]), checked_relative(data_cache, item["path"]), item["sha256"], item["bytes"])
    for model, inv in inventories.items():
        snapshot = Path(deployment["models"][model]["snapshot"])
        target = cache / "models" / model / MODELS[model]["revision"]
        for item in inv["files"]:
            # HF snapshots use links into their immutable blob cache; only the
            # supplied inventory controls destination paths and accepted bytes.
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe model inventory path")
            _copy_verified(snapshot / relative, checked_relative(target, relative), item["sha256"], item["bytes"])
        model_specs[model] = str(target)
    episodes = []
    for row in validated["rows"]:
        public = ({k: row[k] for k in ("episode_id", "cluster_id", "task", "contract_sha256")}
                  if is_recognition(manifest.get("contract")) else dict(row))
        public["images"] = {role: {**spec, "path": str(checked_relative(data_cache, spec["path"]))}
                            for role, spec in row["images"].items()}
        episodes.append(public)
    return episodes, model_specs, {m: i["inventory_sha256"] for m, i in inventories.items()}


def task_result_path(run_dir, task_id):
    return Path(run_dir) / "results" / task_id[:2] / f"{task_id}.json"


def load_result(run_dir, task, identity):
    path = task_result_path(run_dir, task["task_id"])
    if not path.exists():
        return None
    result = read_json(path)
    checksum = result.pop("result_sha256", None)
    if checksum != digest(result) or result.get("identity") != identity:
        raise ValueError(f"invalid result/provenance: {path}")
    if any(result.get(k) != v for k, v in task.items()) or result.get("status") != "pass":
        raise ValueError(f"result task contract mismatch: {path}")
    result["result_sha256"] = checksum
    return result


def collect(run_dir, tasks, identity):
    expected = {task["task_id"] for task in tasks}
    actual = {p.stem for p in (Path(run_dir) / "results").glob("*/*.json")}
    if expected != actual:
        raise ValueError(f"result inventory mismatch: missing={len(expected-actual)} extra={len(actual-expected)}")
    return [load_result(run_dir, task, identity) for task in tasks]


def run_worker(context_path, rank, backend_factory=None):
    """A stopped worker leaves completed atomic task files and a non-pass receipt."""
    context = read_json(context_path)
    run = Path(context["run_dir"])
    tasks = context["assignments"][rank]
    episode_map = {r["episode_id"]: r for r in context["episodes"]}
    identity = context["identity"]
    contract = context.get("contract")
    followup = context.get("followup_protocol")
    if identity.get("recognition_suite") == FOLLOWUP_SUITE:
        validate_followup_protocol(followup)
        if digest(followup) != identity["protocol_sha256"] or protocol_contract(followup) != contract:
            raise ValueError("follow-up worker protocol differs from its sealed identity")
    elif followup is not None or identity.get("recognition_suite") is not None:
        raise ValueError("unexpected follow-up worker protocol or suite")
    if contract is not None:
        validate_contract(contract)
    if is_recognition(contract):
        _recognition_worker_inputs(context, tasks)
        if backend_factory is not None and identity.get("simulation") is not True:
            raise ValueError("injected recognition backend requires simulation=true")
    validate_rows_contract(tasks, contract)
    validate_rows_contract(context["episodes"], contract)
    record_cuda_peak = (is_recognition(contract) and identity.get("simulation") is False
                        and backend_factory is None)
    started, interrupted = time.monotonic(), False
    old_handlers = {}

    def peak_memory():
        if not record_cuda_peak:
            return {}
        try:
            import torch

            # Read lifetime high-water marks, including model load; never reset.
            return {"peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(f"cuda:{rank}")),
                    "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(f"cuda:{rank}")),
                    "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(f"cuda:{rank}").total_memory)}
        except Exception as error:
            # Preserve the original worker failure if CUDA itself is unavailable.
            return {"peak_memory_error": f"{type(error).__name__}: {error}"}

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, stop)
    backend, current_model = None, None
    completed = 0
    try:
        if backend_factory is None:
            if followup is not None:
                from .followup_runtime import FollowupHFBackend
                backend_factory = FollowupHFBackend
            else:
                from .runtime import HFBackend
                backend_factory = HFBackend
        for task in tasks:
            if interrupted or time.time() > context["deadline_epoch"]:
                raise TimeoutError("checkpointed stop at task boundary")
            if load_result(run, task, identity) is not None:
                completed += 1
                continue
            if current_model != task["model_key"]:
                if backend is not None:
                    del backend
                    gc.collect()
                    if not identity["simulation"]:
                        import torch
                        torch.cuda.empty_cache()
                current_model = task["model_key"]
                backend = backend_factory(current_model, context["model_paths"][current_model],
                                          device=f"cuda:{rank}", attention_backend=context["attention_backend"],
                                          **({"contract": contract} if contract else {}))
            episode = episode_map[task["episode_id"]]
            if task["kind"] == "behavior":
                values = backend.score(episode, task["order"], task["condition"])
            elif task["kind"] == "prune":
                values = backend.prune(episode, task["order"], task["condition"], task["cut"], task["position_set"])
            else:
                values = backend.patch(episode, task["order"], task["donor_condition"], task["recipient_condition"],
                                       task["cut"], task["position_set"], self_patch=task["kind"] == "self")
            if not isinstance(values, dict) or not all(k in values for k in ("a_logprob", "b_logprob", "margin_ab_bits", "elapsed_s")):
                raise ValueError("backend omitted required scores/timing")
            if contract:
                validate_rows_contract([values], contract)
            result = {**values, **task, "status": "pass", "identity": identity}
            if task["kind"] == "self" and abs(result["margin_ab_bits"] - result["recipient_margin_ab_bits"]) > .001:
                raise ValueError("self-patch control failed")
            if task["kind"] == "patch" and task["position_set"] == "A" and abs(result["margin_ab_bits"] - result["donor_margin_ab_bits"]) > .001:
                raise ValueError("full-state closure failed")
            result["result_sha256"] = digest(result)
            atomic_json(task_result_path(run, task["task_id"]), result)
            completed += 1
        receipt = {"status": "pass", "rank": rank, "completed": completed, "expected": len(tasks),
                   "identity": identity, "elapsed_s": time.monotonic() - started, **peak_memory()}
        atomic_json(run / "workers" / f"{rank}.json", receipt)
        return receipt
    except BaseException as error:
        atomic_json(run / "workers" / f"{rank}.json", {"status": "fail", "rank": rank,
                    "completed": completed, "identity": identity, "error": f"{type(error).__name__}: {error}",
                    **peak_memory()})
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def canary_admission(receipt, identity, tasks, workers, seconds):
    """Measured class maxima plus fixed safety margins; no unmeasured class admitted."""
    if receipt.get("status") != "pass" or not receipt.get("stage", "").endswith("-canary"):
        raise ValueError("full run requires a successful canary")
    for key in ("release_sha256", "public_sha256", "protocol_sha256", "freeze_sha256", "evaluator_sha256",
                "dataset_manifest_sha256", "runtime_identity", "model_inventories", "recognition_suite"):
        if receipt.get("identity", {}).get(key) != identity.get(key):
            raise ValueError(f"canary provenance differs: {key}")
    if receipt["identity"].get("simulation") or identity.get("simulation"):
        raise ValueError("simulation cannot authorize GPU production")
    recognition = (receipt.get("stage") == "recognition-canary"
                   or any(t.get("task") in RECOGNITION_TASKS for t in tasks))
    if recognition:
        if receipt.get("stage") != "recognition-canary":
            raise ValueError("recognition requires its matching canary stage")
        for value in (receipt["identity"], identity):
            if value.get("simulation") is not False:
                raise ValueError("recognition requires explicitly non-simulation identity")
            for key in ("release_sha256", "public_sha256", "protocol_sha256", "evaluator_sha256",
                        "dataset_manifest_sha256", "runtime_identity", "runtime_spec_sha256"):
                if not _sha256_pin(value.get(key)):
                    raise ValueError(f"recognition canary missing identity: {key}")
        if receipt["identity"]["runtime_spec_sha256"] != identity["runtime_spec_sha256"]:
            raise ValueError("canary provenance differs: runtime_spec_sha256")
        from .protocol import GEMMA

        if (not isinstance(identity.get("model_inventories"), dict)
                or set(identity["model_inventories"]) != {GEMMA}
                or any(not _sha256_pin(v) for v in identity["model_inventories"].values())):
            raise ValueError("recognition canary missing model inventories")
        suite = identity.get("recognition_suite")
        if suite not in (None, FOLLOWUP_SUITE):
            raise ValueError("unknown recognition suite")
        expected_canary_tasks = 312 if suite == FOLLOWUP_SUITE else 176
        if (type(receipt.get("tasks")) is not int or receipt["tasks"] != expected_canary_tasks
                or any(not _sha256_pin(receipt.get(k)) for k in
                       ("technical_checks_sha256", "task_inventory_sha256", "result_set_sha256"))):
            raise ValueError("recognition canary lacks complete technical receipt")
    costs = receipt.get("task_class_mean_s", receipt.get("task_class_max_s", {}))
    if recognition:
        if (not isinstance(costs, dict)
                or set(costs) != {f"{GEMMA}:{kind}" for kind in ("behavior", "patch", "self")}):
            raise ValueError("canary did not exercise every full-run task class")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 300 <= seconds <= 23 * 3600:
            raise ValueError("invalid recognition walltime budget")
        timing = [*costs.values(), receipt.get("staging_s"), receipt.get("elapsed_s")]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in timing):
            raise ValueError("recognition canary timing must be finite and nonnegative")
        if any(v <= 0 for v in costs.values()) or receipt["elapsed_s"] <= 0:
            raise ValueError("recognition canary timing must be positive")
    estimates = []
    for shard in assign_tasks(tasks, workers):
        count = Counter(f"{t['model_key']}:{t['kind']}" for t in shard)
        if any(k not in costs for k in count):
            raise ValueError("canary did not exercise every full-run task class")
        estimates.append(sum(costs[k] * n for k, n in count.items()))
    estimate = 1.6 * max(estimates, default=0) + receipt.get("staging_s", 0) + 1200
    if estimate > seconds - 180 or (recognition and estimate + 180 >= 23 * 3600):
        raise ValueError(f"measured budget needs about {estimate/3600:.2f} h; request more time (max 23h) or an explicitly revised protocol")
    return estimate


def execute(release_root, deployment_path, stage, run_dir):
    from .data import validate_dataset
    from .release import verify_release

    release_root, run = Path(release_root), Path(run_dir)
    release = verify_release(release_root)
    verify_loaded_code(release)
    dep = read_json(deployment_path)
    protocol = validate_protocol(read_json(release_root / "protocol.json"))
    hardware = require_compute()
    started = time.monotonic()
    validated = validate_dataset(dep["data_root"], verify_assets=True)
    validate_dataset_protocol(validated["manifest"], protocol)
    workers = int(os.environ.get("TF_WORKER_COUNT", dep.get("resources", {}).get("gpus_per_node", len(hardware))))
    if workers > len(hardware) or workers < 1 or int(os.environ.get("TF_NODES", "1")) != 1:
        raise ValueError("single-node worker allocation mismatch")
    for package, expected in dep["runtime"]["versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"runtime version differs: {package}")
    if not {"torch", "transformers"} <= dep["runtime"]["versions"].keys():
        raise ValueError("pin torch and transformers in deployment.runtime.versions")
    runtime_identity = {"runtime": dep["runtime"], "hardware": hardware,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}
    # Paths/device ordinals vary by host; immutable runtime bytes + actual GPU
    # type, not host location, establish compatibility for a portable canary.
    runtime_identity = digest({"sha256": dep["runtime"]["sha256"], "versions": dep["runtime"]["versions"],
                               "hardware_types": sorted({(h["name"], h["memory_bytes"]) for h in hardware})})
    metadata_identity = {"release_sha256": release["release_sha256"],
                         "protocol_sha256": digest(protocol), "public_sha256": validated["manifest"]["public_sha256"],
                         "evaluator_sha256": validated["manifest"]["evaluator_sha256"],
                         "dataset_manifest_sha256": sha256(Path(dep["data_root"]) / "manifest.json")}
    verify_followup_dataset(protocol, metadata_identity["dataset_manifest_sha256"])
    if is_followup(protocol):
        metadata_identity["recognition_suite"] = FOLLOWUP_SUITE
    freeze = read_json(dep["freeze_path"]) if dep.get("freeze_path") else None
    if stage.removesuffix("-canary") == "confirmation":
        verify_freeze(freeze, {**metadata_identity, "freeze_sha256": freeze.get("freeze_sha256") if freeze else None},
                      protocol, validated["rows"])
    tasks = inventory(validated["rows"], protocol, stage, freeze)
    if is_recognition(protocol) and stage == "recognition":
        require_recognition_full_cohort(validated["rows"])
    models = sorted({t["model_key"] for t in tasks})
    scratch = dep.get("local_scratch") or os.environ.get("SLURM_TMPDIR")
    if not scratch or not Path(scratch).is_dir():
        raise ValueError("provide existing node-local scratch, or bind SLURM_TMPDIR")
    cache = Path(scratch) / "temporal-forecasting" / release["release_sha256"] / validated["manifest"]["public_sha256"]
    episodes, model_paths, model_inventories = stage_inputs(dep, validated, models, cache)
    identity = {**metadata_identity, "freeze_sha256": freeze.get("freeze_sha256") if freeze else None,
                "runtime_identity": runtime_identity, "model_inventories": model_inventories, "simulation": False}
    if is_recognition(protocol):
        identity["runtime_spec_sha256"] = recognition_runtime_spec(dep["runtime"])
    run.mkdir(parents=True, exist_ok=True)
    lock = run / ".execute.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    processes = []
    def interrupted(signum, _frame):
        raise InterruptedError(f"coordinator received signal {signum}; retaining checkpoints")
    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        seconds = int(dep.get("walltime_seconds", 1200 if stage.endswith("-canary") else 23 * 3600))
        if not 300 <= seconds <= 23 * 3600:
            raise ValueError("walltime_seconds must be 300..82800")
        if "TF_WALLTIME_SECONDS" in os.environ and seconds != int(os.environ["TF_WALLTIME_SECONDS"]):
            raise ValueError("deployment walltime differs from submitted allocation")
        if stage.removesuffix("-canary") not in ("pilot", "current", "recognition"):
            pilot_path = dep.get("pilot_receipt")
            if not pilot_path or dep.get("pilot_review_approved") is not True:
                raise ValueError("review the completed pilot; provide pilot_receipt and pilot_review_approved=true")
            pilot = read_json(pilot_path)
            if pilot.get("status") != "pass" or pilot.get("stage") != "pilot" or pilot["identity"].get("simulation"):
                raise ValueError("pilot gate requires a complete real pilot")
            if any(pilot["identity"].get(k) != v for k, v in metadata_identity.items()):
                raise ValueError("pilot gate provenance mismatch")
        pending = [task for task in tasks if load_result(run, task, identity) is None]
        if not stage.endswith("-canary"):
            canary_path = dep.get("canary_receipt")
            if not canary_path:
                raise ValueError("full run needs deployment.canary_receipt")
            receipt = read_json(canary_path)
            if receipt.get("stage") != stage + "-canary":
                raise ValueError("canary must target this scientific stage")
            if is_recognition(protocol) and receipt.get("task_inventory_sha256") != digest(
                    inventory(validated["rows"], protocol, "recognition-canary")):
                raise ValueError("recognition canary inventory differs from the frozen design")
            if is_recognition(protocol):
                verify_recognition_technical_receipt(canary_path, receipt)
            if is_followup(protocol):
                from .followup_compare import require_canary_reference

                require_canary_reference(protocol, dep, receipt)
            canary_admission(receipt, identity, pending, workers, seconds)
        context = {"run_dir": str(run), "identity": identity, "stage": stage, "freeze": freeze,
                   "task_inventory_sha256": digest(tasks), "expected_tasks": len(tasks),
                   "assignments": assign_tasks(pending, workers),
                   "episodes": episodes, "model_paths": model_paths,
                   "attention_backend": protocol["attention_backend"],
                   "deadline_epoch": time.time() + max(0, seconds - (time.monotonic() - started) - 180)}
        if protocol_contract(protocol):
            context["contract"] = protocol_contract(protocol)
        if is_followup(protocol):
            context["followup_protocol"] = protocol
        context_path = run / "worker_context.json"
        if context_path.exists():
            old = read_json(context_path)
            if old["identity"] != identity or old["task_inventory_sha256"] != context["task_inventory_sha256"]:
                raise ValueError("resume requires identical inputs and task inventory")
        atomic_json(context_path, context)
        staging_s = time.monotonic() - started
        env = {**os.environ, "PYTHONPATH": str(release_root), "PYTHONDONTWRITEBYTECODE": "1"}
        for rank in range(workers):
            log = (run / f"worker-{rank}.log").open("a")
            try:
                process = subprocess.Popen([sys.executable, "-B", "-m", "research.lifecycle_memory.temporal_forecasting.cli",
                                            "worker", "--context", str(context_path), "--rank", str(rank)],
                                           cwd=release_root, env=env, stdout=log, stderr=subprocess.STDOUT)
            finally:
                log.close()
            processes.append(process)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("a worker failed; see per-worker log/receipt")
            if time.time() > context["deadline_epoch"]:
                raise TimeoutError("allocation safety deadline reached; completed tasks preserved")
            time.sleep(1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("worker failure")
        results = collect(run, tasks, identity)
        if is_followup(protocol):
            from .followup_analysis import validate_followup_technical

            technical = validate_followup_technical(results, {t["episode_id"] for t in tasks})
            atomic_json(run / "technical_checks.json", technical)
            if technical["status"] != "pass":
                raise ValueError("follow-up technical controls failed; see technical_checks.json")
        elif is_recognition(protocol):
            from .recognition_analysis import validate_recognition_technical

            technical = validate_recognition_technical(results, {t["episode_id"] for t in tasks})
            atomic_json(run / "technical_checks.json", technical)
            if technical["status"] != "pass":
                raise ValueError("recognition technical controls failed; see technical_checks.json")
        elif is_dense_current(protocol):
            from .dense_analysis import validate_dense_technical

            technical = validate_dense_technical(results)
            atomic_json(run / "technical_checks.json", technical)
            if technical["status"] != "pass":
                raise ValueError("dense-current technical controls failed; see technical_checks.json")
        costs, elapsed_by_class = {}, {}
        for row in results:
            key = f"{row['model_key']}:{row['kind']}"
            costs[key] = max(costs.get(key, 0), row["elapsed_s"])
            elapsed_by_class.setdefault(key, []).append(row["elapsed_s"])
        receipt = {"status": "pass", "stage": stage, "identity": identity,
                   "task_inventory_sha256": digest(tasks), "tasks": len(tasks), "workers": workers,
                   "elapsed_s": time.monotonic() - started, "staging_s": staging_s,
                   "task_class_max_s": costs,
                   "task_class_mean_s": {k: sum(v)/len(v) for k, v in elapsed_by_class.items()},
                   "completed_at": datetime.now(timezone.utc).isoformat(),
                   "result_set_sha256": digest(sorted(r["result_sha256"] for r in results))}
        if is_dense_current(protocol) or is_recognition(protocol):
            receipt["technical_checks_sha256"] = sha256(run / "technical_checks.json")
        atomic_json(run / "receipt.json", receipt)
        if not stage.endswith("-canary"):
            summary = analyze_run(run, release_root, dep["data_root"])
            atomic_json(run / "completion.json", {"status": "pass", "analysis_status": summary["status"],
                        "analysis_sha256": sha256(run / "analysis.json"), "receipt_sha256": sha256(run / "receipt.json")})
        return receipt
    except BaseException as error:
        atomic_json(run / "failure.json", {"status": "fail", "stage": stage, "error": f"{type(error).__name__}: {error}", "identity": identity})
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        stop_deadline = time.monotonic() + 30
        for process in processes:
            try:
                process.wait(timeout=max(0, stop_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        lock.unlink(missing_ok=True)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def analyze_run(run_dir, release_root, data_root, draws=None):
    from .data import validate_dataset
    from .release import verify_release

    run = Path(run_dir)
    release = verify_release(release_root)
    verify_loaded_code(release)
    context, receipt = read_json(run / "worker_context.json"), read_json(run / "receipt.json")
    if receipt.get("status") != "pass" or receipt["identity"] != context["identity"] or receipt["identity"]["simulation"]:
        raise ValueError("cannot analyze incomplete/simulated scientific run")
    if receipt["stage"].endswith("-canary"):
        raise ValueError("canaries certify technical execution, not scientific outcomes")
    validated = validate_dataset(data_root)
    identity = receipt["identity"]
    if (release["release_sha256"] != identity["release_sha256"]
            or validated["manifest"]["public_sha256"] != identity["public_sha256"]
            or validated["manifest"]["evaluator_sha256"] != identity.get("evaluator_sha256")
            or sha256(Path(data_root) / "manifest.json") != identity.get("dataset_manifest_sha256")):
        raise ValueError("analysis provenance mismatch")
    protocol = read_json(Path(release_root) / "protocol.json")
    validate_dataset_protocol(validated["manifest"], protocol)
    verify_followup_dataset(protocol, identity["dataset_manifest_sha256"])
    if digest(protocol) != identity["protocol_sha256"]:
        raise ValueError("analysis protocol differs from run")
    if is_recognition(protocol):
        if receipt["stage"] != "recognition" or context.get("stage") != "recognition":
            raise ValueError("recognition analysis requires the matching full stage")
        if draws is not None and draws != protocol["bootstrap_draws"]:
            raise ValueError("recognition bootstrap count is frozen; overrides are not allowed")
        technical_path = run / "technical_checks.json"
        if (sha256(technical_path) != receipt.get("technical_checks_sha256")
                or read_json(technical_path).get("passed") is not True):
            raise ValueError("recognition technical receipt is missing or stale")
    if receipt["stage"] == "confirmation":
        verify_freeze(context.get("freeze"), identity, protocol, validated["rows"])
        if draws is not None and draws != protocol["bootstrap_draws"]:
            raise ValueError("confirmation bootstrap count is frozen; overrides are not allowed")
    tasks = inventory(validated["rows"], protocol, receipt["stage"], context.get("freeze"))
    if (digest(tasks) != receipt["task_inventory_sha256"] or len(tasks) != receipt["tasks"]
            or context.get("task_inventory_sha256") != digest(tasks)):
        raise ValueError("analysis task inventory differs from recorded design")
    results = collect(run, tasks, receipt["identity"])
    if receipt["result_set_sha256"] != digest(sorted(r["result_sha256"] for r in results)):
        raise ValueError("results differ from the completed run receipt")
    protocol["episode_ids"] = [r["episode_id"] for r in selected_episodes(validated["rows"], protocol, receipt["stage"])]
    protocol["bootstrap_seed"] = protocol["seed"]
    if context.get("freeze"):
        protocol["cuts"] = context["freeze"]["cuts"]
        protocol["freeze"] = context["freeze"]
    labels = [r for r in read_jsonl(Path(data_root) / "evaluator.jsonl") if r["episode_id"] in protocol["episode_ids"]]
    if is_followup(protocol):
        from .followup_analysis import analyze_followup

        summary = analyze_followup(results, labels, protocol, receipt["stage"], draws=protocol["bootstrap_draws"])
    elif is_recognition(protocol):
        from .recognition_analysis import analyze_recognition

        summary = analyze_recognition(results, labels, protocol, receipt["stage"], draws=protocol["bootstrap_draws"])
    else:
        from .analysis import analyze

        summary = analyze(results, labels, protocol, receipt["stage"], draws=draws or protocol["bootstrap_draws"])
    summary["provenance"] = {"receipt_sha256": sha256(run / "receipt.json"), "identity": receipt["identity"]}
    atomic_json(run / "analysis.json", summary)
    if protocol.get("schema_version") == 3 and summary.get("analysis_complete") is not True:
        raise ValueError("dense-current analysis is incomplete or its technical controls failed; see analysis.json")
    if is_recognition(protocol) and summary.get("analysis_complete") is not True:
        raise ValueError("recognition analysis is incomplete or its technical controls failed; see analysis.json")
    atomic_json(run / "completion.json", {"status": "pass", "analysis_status": summary["status"],
                "analysis_sha256": sha256(run / "analysis.json"), "receipt_sha256": sha256(run / "receipt.json")})
    return summary
