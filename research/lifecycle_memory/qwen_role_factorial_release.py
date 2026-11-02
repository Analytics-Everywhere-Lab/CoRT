"""Offline, job-specific release and TamIA lifecycle for the Qwen L36 factorial."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback

# Direct-file execution works under python -B -S without site packages.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.lifecycle_memory.application_pilot_release import (
    SIF_SHA256, atomic_json, canonical, checked_hash, digest, entry, inventory,
    model_inventory as cached_model_inventory, number, publish, read_json,
    relative_path, require, sha256, tree_files, verify_copyback, verify_inventory,
    _copy_checked, _run_container,
)

JOB_REL = "docs/research/lifecycle_memory/phase 2 research/jobs/qwen_role_factorial"
HELPER = "research/lifecycle_memory/qwen_role_factorial_release.py"
SHARED = "research/lifecycle_memory/application_pilot_release.py"
RUNTIME = "research/lifecycle_memory/qwen_role_factorial.py"
JOB = JOB_REL + "/job_tamia_qwen_role_factorial_h100.sh"
PUBLIC = "docs/research/lifecycle_memory/phase 2 research/jobs/patchaudit_jobE_heldout_component_closure/heldout_public_requests.jsonl"
PUBLIC_SHA = "2b0e31236a0ea531a4a9849b00356e2fabfa153084dd12487511b2523d50635f"
SELECTION = str(Path(PUBLIC).with_name("heldout_selection.json"))
SELECTION_SHA = "73ee030a8369f05ba11f439fa8b83ecfb5e694ff3eecb1d47c032c9807cef2de"
RUNTIME_VERSIONS = {"torch": "2.12.1+cu126", "transformers": "5.13.0"}
REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
MODEL_ID = "Qwen/Qwen3.5-27B"
SNAPSHOT = "hub/models--Qwen--Qwen3.5-27B/snapshots/" + REVISION
MANIFEST = "release_manifest.json"
LIMIT = 32 * 1024 * 1024


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def selected_requests(root):
    path = Path(root) / PUBLIC
    require(sha256(path) == PUBLIC_SHA, "frozen public physical SHA drift")
    rows = jsonl(path)
    require(len(rows) == 92 and len({r["request_id"] for r in rows}) == 92, "expected 92 unique public requests")
    frames = sorted({r["frame_unit_id"] for r in rows})
    require(len(frames) == 46, "expected 46 frames")
    for frame in frames:
        orders = [r["order_id"] for r in rows if r["frame_unit_id"] == frame]
        require(sorted(orders) == ["order_0", "order_1"], "both frame orders required")
    return rows


def expected_binding(root):
    root = Path(root)
    config_path = root / JOB_REL / "config.json"
    cfg = read_json(config_path)
    return {"config_sha256": sha256(config_path), "model": cfg["model"],
            "source_files": cfg["source_files"], "data_files": cfg["data_files"],
            "public_requests_sha256": cfg["public_requests"]["sha256"],
            "model_manifest_sha256": cfg["model_manifest"]["sha256"], "provenance": cfg["provenance"]}


def source_files(repo_root):
    root = Path(repo_root)
    pending = {HELPER, RUNTIME, SHARED,
               "docs/research/lifecycle_memory/phase 2 research/jobs/job0/run_phase1_qwen27b_hf.py"}
    selected = {PUBLIC, SELECTION, JOB, *(JOB_REL + "/" + n for n in ("worker.sh", "submit_tamia.sh", "config.json", "README.md", "RUNBOOK.md"))}
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        path = root / name
        require(path.is_file() and not path.is_symlink(), f"missing/unsafe import dependency: {name}")
        selected.add(name)
        # Only imported stdlib-only primitives from this helper execute here.
        if name == SHARED:
            continue
        modules = set()
        for node in ast.walk(ast.parse(path.read_text(), filename=name)):
            if isinstance(node, ast.Import):
                modules.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join(name.removesuffix(".py").split("/")[:-node.level] + ([base] if base else []))
                modules.add(base)
                modules.update(base + "." + a.name for a in node.names if a.name != "*")
        for module in modules:
            parts = module.split(".")
            for suffix in (".py", "/__init__.py"):
                candidate = "/".join(parts) + suffix
                if (root / candidate).is_file():
                    pending.add(candidate)
            for stop in range(1, len(parts)):
                marker = "/".join(parts[:stop]) + "/__init__.py"
                if (root / marker).is_file():
                    pending.add(marker)
    return sorted(selected)


def bundle(repo_root, out_dir):
    root, out = Path(repo_root).resolve(), Path(out_dir)
    require(not out.exists(), "source bundle must be new")
    names = source_files(root)
    require(sum((root / n).stat().st_size for n in names) < LIMIT, "source bundle exceeds 32MiB")
    for name in names:
        src = root / name
        require(not src.is_symlink() and root in src.resolve().parents, "source escapes repository")
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    atomic_json(out / "source_manifest.json", {"files": [entry(p, out) for p in tree_files(out)]})
    return {"source_sha256": sha256(out / "source_manifest.json"), "files": len(names)}


def data_inventory(source, assets_root):
    selected = selected_requests(source)
    pool_path = Path(assets_root) / "mechanism/public/mechanism_requests.jsonl"
    require(pool_path.stat().st_size < LIMIT, "source metadata too large for bootstrap")
    pool = jsonl(pool_path)
    by_id = {r["request_id"]: r for r in pool}
    require(len(by_id) == len(pool), "duplicate source request")
    result = {}
    for request in selected:
        require(request == by_id.get(request["request_id"]), "selected source request drift")
        for arm in request["arms"].values():
            for asset in arm["inputs"]:
                name = relative_path(asset["path"])
                require(name.startswith("assets/"), "nonasset public input")
                path = Path(assets_root) / name
                require(path.is_file() and Path(assets_root).resolve() in path.resolve().parents, "missing/escaping source asset")
                row = {"path": name, "bytes": path.stat().st_size, "sha256": checked_hash(asset["sha256"])}
                require(name not in result or result[name] == row, "conflicting asset digest")
                result[name] = row
    return {"files": [result[k] for k in sorted(result)], "source_public_physical_sha256": sha256(pool_path)}


def validate_model_inventory(value, model_root):
    rows = inventory(value)
    names = set()
    for row in rows:
        require(row["path"].startswith(SNAPSHOT + "/"), "model inventory revision/root drift")
        path = Path(model_root) / row["path"]
        require(path.is_file() and Path(model_root).resolve() in path.resolve().parents, "missing/escaping cached model file")
        require(path.stat().st_size == row["bytes"], "cached model size drift")
        names.add(row["path"][len(SNAPSHOT) + 1:])
    require({"config.json", "tokenizer_config.json", "preprocessor_config.json", "model.safetensors.index.json"} <= names,
            "missing processor/model metadata")
    index = Path(model_root) / SNAPSHOT / "model.safetensors.index.json"
    require(index.stat().st_size < LIMIT, "oversized model metadata")
    require(set(read_json(index)["weight_map"].values()) <= names, "missing model shards")
    return {"files": rows}


def evaluator_contract(root):
    path = Path(root) / SELECTION
    require(sha256(path) == SELECTION_SHA, "old selection contract drift")
    selection = read_json(path)
    return {"source_physical_sha256": selection["source"]["hidden_sha256"],
            "selected_jsonl_sha256": selection["selected"]["hidden_jsonl_sha256"],
            "request_count": 92, "frame_count": 46, "participant_cluster_count": 46}


def bootstrap(source_root, source_sha, output_root, assets_root, model_root, sif_path, model_inventory=None):
    source, output = Path(source_root).resolve(), Path(output_root).absolute()
    require(sha256(source / "source_manifest.json") == checked_hash(source_sha), "source manifest drift")
    sources = read_json(source / "source_manifest.json")
    verify_inventory(source, sources["files"], excluded=("source_manifest.json",))
    require(not output.exists(), "immutable output exists; use a new release")
    require(Path(sif_path).is_file(), "pre-staged SIF missing")
    cfg = read_json(source / JOB_REL / "config.json")
    require(cfg["model"]["model_id"] == MODEL_ID and cfg["model"]["revision"] == REVISION
            and cfg["model"]["processor_revision"] == REVISION, "model/processor pin drift")
    data = data_inventory(source, assets_root)
    if model_inventory:
        require(Path(model_inventory).stat().st_size < LIMIT, "model inventory too large")
        model = read_json(model_inventory)
    else:
        model = cached_model_inventory(model_root, {"model": {"id": MODEL_ID, "revision": REVISION}})
    model = validate_model_inventory(model, model_root)
    output.mkdir(parents=True)
    release = output / "release"
    release.mkdir()
    for row in sources["files"]:
        dest = release / row["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / row["path"], dest)
    atomic_json(release / "data_inventory.json", data)
    atomic_json(release / "model_inventory.json", model)
    cfg["source_files"] = {r["path"]: r["sha256"] for r in sources["files"] if r["path"].endswith(".py")}
    cfg["data_files"] = {r["path"]: r["sha256"] for r in data["files"]}
    cfg["model_manifest"] = {"path": "model_inventory.json", "sha256": sha256(release / "model_inventory.json")}
    cfg["provenance"] = {"sif_sha256": SIF_SHA256, "slurm_script_sha256": sha256(release / JOB)}
    atomic_json(release / JOB_REL / "config.json", cfg)
    spec = {"schema_version": 1, "assets_root": str(Path(assets_root).resolve()),
            "model_root": str(Path(model_root).resolve()), "sif_path": str(Path(sif_path).resolve()),
            "sif_sha256": SIF_SHA256, "sif_bytes": Path(sif_path).stat().st_size,
            "workloads": {"canary": [136] * 4, "full": [1632, 1632, 1496, 1496]}}
    hidden = Path(assets_root) / "mechanism/hidden/mechanism_evaluator_index.jsonl"
    require(hidden.stat().st_size < LIMIT, "evaluator metadata exceeds bootstrap limit")
    spec["evaluator_contract"] = evaluator_contract(source)
    require(sha256(hidden) == spec["evaluator_contract"]["source_physical_sha256"], "frozen source evaluator drift")
    atomic_json(release / "deployment.json", spec)
    atomic_json(release / MANIFEST, {"schema_version": 1, "files": [entry(p, release) for p in tree_files(release)]})
    expected = sha256(release / MANIFEST)
    verify(release, expected)
    (output / "release.sha256").write_text(expected + "\n")
    atomic_json(output / "bootstrap_receipt.json", {"status": "pass", "submitted": False,
                "source_sha256": source_sha, "release_sha256": expected, "large_hashing": "compute-only"})
    for path in tree_files(release):
        path.chmod(0o444)
    for path in sorted(release.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    release.chmod(0o555)
    return {"release": str(release), "release_sha256": expected, "submitted": False}


def verify(release, expected_sha):
    root = Path(release)
    require(sha256(root / MANIFEST) == checked_hash(expected_sha), "release manifest drift")
    manifest = read_json(root / MANIFEST)
    require(manifest["schema_version"] == 1, "unsupported release")
    for row in inventory(manifest):
        require("__pycache__" not in Path(row["path"]).parts and not row["path"].endswith((".pyc", ".pyo")), "bytecode forbidden")
    verify_inventory(root, manifest["files"], excluded=(MANIFEST,))
    cfg = read_json(root / JOB_REL / "config.json")
    require(cfg["model"]["revision"] == cfg["model"]["processor_revision"] == REVISION, "revision drift")
    require(cfg["model"]["model_id"] == MODEL_ID and cfg["post_block_index"] == 36, "model/site drift")
    require(cfg["provenance"] == {"sif_sha256": SIF_SHA256, "slurm_script_sha256": sha256(root / JOB)}, "provenance drift")
    require(cfg["model_manifest"] == {"path": "model_inventory.json", "sha256": sha256(root / "model_inventory.json")}, "model manifest drift")
    for name, expected in cfg["source_files"].items():
        require(sha256(root / relative_path(name)) == checked_hash(expected), "runtime code drift")
    require(set(cfg["source_files"]) == {r["path"] for r in manifest["files"] if r["path"].endswith(".py")}, "source closure map drift")
    require(cfg["data_files"] == {r["path"]: r["sha256"] for r in inventory(read_json(root / "data_inventory.json"))}, "data map drift")
    selected_requests(root)
    spec = read_json(root / "deployment.json")
    require(spec["evaluator_contract"] == evaluator_contract(root), "evaluator contract drift")
    require(spec["sif_sha256"] == SIF_SHA256 and spec["workloads"] == {"canary": [136] * 4, "full": [1632, 1632, 1496, 1496]}, "deployment drift")
    return spec


def identity(release, expected_sha):
    root = Path(release)
    verify(root, expected_sha)
    return {"release_sha256": expected_sha, "config_sha256": sha256(root / JOB_REL / "config.json"),
            "public_sha256": PUBLIC_SHA, "data_sha256": sha256(root / "data_inventory.json"),
            "model_sha256": sha256(root / "model_inventory.json"), "model_revision": REVISION,
            "processor_revision": REVISION, "script_sha256": sha256(root / JOB), "sif_sha256": SIF_SHA256,
            "evaluator_contract_sha256": digest(read_json(root / "deployment.json")["evaluator_contract"])}


def failure(work, stage_name, rank=None):
    path = Path(work) / "context.json"
    ctx = read_json(path) if path.exists() else {}
    atomic_json(Path(work) / "out" / f"failure_{stage_name}_{rank}.json",
                {"status": "fail", "stage": stage_name, "rank": rank, "identity": ctx.get("identity"),
                 "run_id": ctx.get("run_id"), "traceback": traceback.format_exc(), "time": time.time()})


def stage(release, expected_sha, work_dir, mode, run_id, executing_script, walltime_seconds, canary_dir=None):
    work = Path(work_dir).absolute()
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_TMPDIR"), "staging is compute-only")
    require(Path(os.environ["SLURM_TMPDIR"]).resolve() in work.resolve().parents, "stage outside node-local storage")
    require(not work.exists(), "work directory exists")
    (work / "out").mkdir(parents=True)
    started = time.monotonic()
    try:
        require(mode in ("canary", "full") and re.fullmatch(r"[0-9]+", run_id), "invalid mode/run")
        spec, ident = verify(release, expected_sha), identity(release, expected_sha)
        require(sha256(executing_script) == ident["script_sha256"], "executing Slurm script drift")
        if mode == "canary":
            require(walltime_seconds == 1200, "canary requires 20 minutes")
        else:
            require(canary_dir is not None, "full requires verified canary")
            require(gate(release, expected_sha, canary_dir) <= walltime_seconds <= 10800, "full time outside measured bound")
        ctx = {"schema_version": 1, "identity": ident, "identity_sha256": digest(ident), "spec": spec,
               "binding": expected_binding(release), "mode": mode, "run_id": run_id,
               "walltime_seconds": walltime_seconds, "status": "staging"}
        atomic_json(work / "context.json", ctx)
        shutil.copytree(release, work / "release")
        verify(work / "release", expected_sha)
        for kind, source_root in (("data", spec["assets_root"]), ("model", spec["model_root"])):
            for row in inventory(read_json(Path(release) / f"{kind}_inventory.json")):
                _copy_checked(Path(source_root) / row["path"], work / kind / row["path"], row, source_root)
        _copy_checked(spec["sif_path"], work / "runtime.sif", {"bytes": spec["sif_bytes"], "sha256": SIF_SHA256}, Path(spec["sif_path"]).parent)
        if mode == "full":
            shutil.copytree(canary_dir, work / "canary")
            gate(work / "release", expected_sha, work / "canary")
            ctx["canary_runtime_fingerprint"] = read_json(work / "canary/completion_receipt.json")["runtime_fingerprint"]
            ctx["canary_completion_sha256"] = sha256(work / "canary/completion_receipt.json")
        require(not (work / "data" / "mechanism").exists(), "hidden metadata must not be staged")
        ctx.update(status="running", staging_seconds=time.monotonic() - started)
        atomic_json(work / "context.json", ctx)
        atomic_json(work / "out" / "job_manifest.json", ctx)
        return ctx
    except Exception:
        failure(work, "stage")
        raise


def container_command(work, ctx, operation, rank=None, evaluator=False):
    work = Path(work).resolve()
    require(operation in ("worker", "aggregate"), "unknown container operation")
    if operation == "worker":
        require(type(rank) is int and 0 <= rank < 4, "worker requires a valid rank")
    else:
        require(rank is None, "aggregate must not have a GPU rank")
    require(not evaluator or (operation == "aggregate" and ctx["mode"] == "full"), "hidden bind forbidden")
    # Separate contained /tmp and home for concurrent launches, and never mount
    # payloads back under /tmp where Apptainer creates those temporary trees.
    session = work / "container_sessions" / (f"worker-{rank}" if rank is not None else "aggregate")
    session.mkdir(parents=True, exist_ok=True, mode=0o700)
    inside = Path("/mnt/qrf")
    command = ["apptainer", "exec", "--containall", "--cleanenv", "--no-mount", "bind-paths,hostfs",
               "--workdir", str(session)]
    env = {"PYTHONPATH": str(inside / "release"), "HF_HOME": str(inside / "model"),
           "HF_HUB_CACHE": str(inside / "model/hub"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "WANDB_MODE": "offline", "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "12", "SIF_SHA256": SIF_SHA256,
           "TMPDIR": "/tmp", "TMP": "/tmp", "TEMP": "/tmp",
           "SLURM_SCRIPT_SHA256": ctx["identity"]["script_sha256"]}
    if rank is not None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        require(visible and len(visible.split(",")) == 1 and visible not in ("-1", "NoDevFiles"), "Slurm must bind one GPU")
        env["CUDA_VISIBLE_DEVICES"] = visible
        command.append("--nv")
    for name in ("release", "data", "model", "out"):
        command.extend(["--bind", f"{work / name}:{inside / name}:{'rw' if name == 'out' else 'ro'}"])
    if ctx["mode"] == "full":
        command.extend(["--bind", f"{work / 'canary'}:{inside / 'canary'}:ro"])
        env["QRF_CANARY_DIR"] = str(inside / "canary/canary")
    if evaluator:
        command.extend(["--bind", f"{work / 'evaluator'}:{inside / 'evaluator'}:ro"])
    command.extend(["--pwd", str(inside / "release")])
    for key, value in env.items():
        command.extend(["--env", f"{key}={value}"])
    command.extend([str(work / "runtime.sif"), "python", "-B", "-u", "-m", "research.lifecycle_memory.qwen_role_factorial", operation,
                    "--config", str(inside / "release" / JOB_REL / "config.json"), "--data-dir", str(inside / "data"),
                    "--run-dir", str(inside / "out"), "--mode", ctx["mode"]])
    if rank is not None:
        command.extend(["--rank", str(rank), "--world-size", "4"])
    if evaluator:
        command.extend(["--evaluator-index", str(inside / "evaluator/index.jsonl")])
    return command


def runtime_worker(out, ctx, rank):
    path = Path(out) / ctx["mode"] / f"rank_{rank:03d}" / "worker_manifest.json"
    value = read_json(path)
    require(value.get("status") == "pass" and value.get("mode") == ctx["mode"] and value.get("rank") == rank, "worker failed/mismatched")
    expected = ctx["spec"]["workloads"][ctx["mode"]][rank]
    require(value.get("outcome_rows") == expected and value.get("requests_completed") == expected // 68, "worker count drift")
    require(value.get("world_size") == 4 and value.get("hidden_index_opened") is False, "worker isolation/rank drift")
    for key in ("geometry", "self_patch", "all_active_closure", "answer_token_equivalence"):
        expected_check = "bound_canary_pass" if ctx["mode"] == "full" and key in ("self_patch", "answer_token_equivalence") else "pass"
        require(value.get("checks", {}).get(key) == expected_check, f"worker check failed: {key}")
    fp = value["runtime_fingerprint"]
    require(fp["model_revision"] == fp["processor_revision"] == REVISION
            and all(fp.get(k) == v for k, v in RUNTIME_VERSIONS.items()), "runtime fingerprint drift")
    if ctx["mode"] == "full":
        require(fp == ctx["canary_runtime_fingerprint"], "full/canary runtime drift")
    for key in ("model_load_elapsed_s", "work_elapsed_s"):
        number(value[key], key)
    require(value.get("binding_sha256") == digest(value["binding"]), "worker binding hash drift")
    require(value["binding"] == ctx["binding"], "worker release binding drift")
    requests = path.parent / "requests"
    rows = value.get("request_files", {})
    require(isinstance(rows, dict) and len(rows) == expected // 68, "request inventory count drift")
    require({p.name for p in tree_files(requests)} == set(rows), "request inventory files drift")
    for name, expected_hash in rows.items():
        require(Path(relative_path(name)).name == name and sha256(requests / name) == expected_hash, "request artifact drift")
    return value


def worker(work_dir, rank):
    work = Path(work_dir)
    try:
        require(type(rank) is int and 0 <= rank < 4, "invalid rank")
        ctx = read_json(work / "context.json")
        started = time.monotonic()
        log = work / "out" / f"worker_rank_{rank}.log"
        _run_container(container_command(work, ctx, "worker", rank), log)
        value = runtime_worker(work / "out", ctx, rank)
        result = {"status": "pass", "rank": rank, "mode": ctx["mode"], "identity_sha256": ctx["identity_sha256"],
                  "run_id": ctx["run_id"], "elapsed_seconds": time.monotonic() - started,
                  "runtime_manifest_sha256": sha256(work / "out" / ctx["mode"] / f"rank_{rank:03d}" / "worker_manifest.json"),
                  "log_sha256": sha256(log), "runtime_fingerprint": value["runtime_fingerprint"]}
        result.update(model_load_elapsed_s=value["model_load_elapsed_s"], work_elapsed_s=value["work_elapsed_s"])
        atomic_json(work / "out" / f"worker_rank_{rank}.json", result)
        return result
    except Exception:
        failure(work, "worker", rank)
        raise


def check_workers(out, ctx):
    out, workers, bindings = Path(out), [], []
    for rank in range(4):
        runtime = runtime_worker(out, ctx, rank)
        receipt = read_json(out / f"worker_rank_{rank}.json")
        require(receipt["identity_sha256"] == ctx["identity_sha256"] and receipt["run_id"] == ctx["run_id"]
                and receipt["rank"] == rank and receipt["mode"] == ctx["mode"] and receipt["status"] == "pass", "harness receipt drift")
        require(receipt["runtime_manifest_sha256"] == sha256(out / ctx["mode"] / f"rank_{rank:03d}" / "worker_manifest.json")
                and receipt["log_sha256"] == sha256(out / f"worker_rank_{rank}.log"), "worker artifact drift")
        number(receipt["elapsed_seconds"], "worker elapsed")
        require(runtime["model_load_elapsed_s"] + runtime["work_elapsed_s"] <= receipt["elapsed_seconds"] * 1.05 + 1, "runtime timing exceeds measured elapsed")
        require(receipt["runtime_fingerprint"] == runtime["runtime_fingerprint"], "runtime fingerprint drift")
        require(all(receipt[k] == runtime[k] for k in ("model_load_elapsed_s", "work_elapsed_s")), "runtime timing receipt drift")
        workers.append(receipt)
        bindings.append(runtime["binding"])
    require(all(w["runtime_fingerprint"] == workers[0]["runtime_fingerprint"] for w in workers), "rank runtime disagreement")
    require(all(b == bindings[0] for b in bindings), "rank common binding disagreement")
    return workers


def telemetry(out):
    out = Path(out)
    require((out / "gpu_telemetry.err").read_text().strip() == "", "telemetry errors")
    with (out / "gpu_telemetry.csv").open() as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    require(rows and {r.get("index", "").strip() for r in rows} == {"0", "1", "2", "3"}, "incomplete four-GPU telemetry")
    require(all(sum(r["index"].strip() == str(rank) for r in rows) >= 2 for rank in range(4)), "two telemetry samples per GPU required")
    for row in rows:
        for field, unit, maximum in (("utilization.gpu [%]", "%", 100), ("memory.used [MiB]", "MiB", 100000), ("power.draw [W]", "W", 1500)):
            raw = row.get(field, "").strip()
            require(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:\s*" + unit + r")?", raw), "nonnumeric telemetry sample")
            value = float(raw.removesuffix(unit).strip())
            require(math.isfinite(value) and 0 <= value <= maximum, "telemetry sample out of range")


def project_seconds(ctx, workers, aggregate_seconds):
    number(ctx["staging_seconds"], "staging seconds")
    number(aggregate_seconds, "aggregation seconds")
    # Loading/import startup happen once. Work still includes canary-only self
    # checks when runtime does not separately time them, a conservative bound.
    longest = 0
    for rank, worker in enumerate(workers):
        elapsed = number(worker["elapsed_seconds"], "worker elapsed")
        load = number(worker["model_load_elapsed_s"], "model load seconds")
        work = number(worker["work_elapsed_s"], "work seconds")
        startup = max(0, elapsed - load - work)
        longest = max(longest, startup + load + work * ctx["spec"]["workloads"]["full"][rank] / 136)
    aggregate = aggregate_seconds * 6256 / 544
    return max(3600, math.ceil(((longest + ctx["staging_seconds"] + aggregate) * 1.4 + 300) / 60) * 60)


def check_aggregate(value, ctx):
    require(value.get("status") == "pass" and value.get("mode") == ctx["mode"], "technical aggregate failed")
    require(isinstance(value.get("ranks"), list) and [r.get("rank") for r in value["ranks"]] == [0, 1, 2, 3]
            and value.get("outcome_rows") == sum(ctx["spec"]["workloads"][ctx["mode"]]), "aggregate inventory drift")
    require(value.get("binding") == ctx["binding"] and value.get("binding_sha256") == digest(ctx["binding"]), "aggregate release binding drift")
    if ctx["mode"] == "canary":
        require(value.get("technical_only") is True and not value.get("analysis"), "canary must not produce hypothesis results")


def stage_evaluator(work, ctx):
    source = Path(ctx["spec"]["assets_root"]) / "mechanism/hidden/mechanism_evaluator_index.jsonl"
    require(source.stat().st_size < LIMIT, "evaluator metadata too large")
    contract = ctx["spec"]["evaluator_contract"]
    require(sha256(source) == contract["source_physical_sha256"], "frozen evaluator metadata drift")
    records = jsonl(source)
    by_id = {r["request_id"]: r for r in records}
    require(len(by_id) == len(records), "duplicate evaluator request")
    public = selected_requests(work / "release")
    chosen = [by_id[r["request_id"]] for r in public]
    require(len(chosen) == 92 and len({r["frame_unit_id"] for r in chosen}) == 46
            and len({r["participant_cluster_id"] for r in chosen}) == 46, "evaluator cohort/cluster count drift")
    for visible, hidden in zip(public, chosen):
        require(all(hidden[k] == visible[k] for k in ("request_id", "frame_unit_id", "order_id", "counterbalanced_request_id")), "evaluator counterbalance drift")
    (work / "evaluator").mkdir()
    with (work / "evaluator/index.jsonl").open("xb") as handle:
        for row in chosen:
            handle.write(canonical(row) + b"\n")
    require(sha256(work / "evaluator/index.jsonl") == contract["selected_jsonl_sha256"], "selected evaluator digest drift")
    atomic_json(work / "out/evaluator_receipt.json", {"status": "pass", "rows": len(chosen),
                "sha256": sha256(work / "evaluator/index.jsonl"), "staged_after_technical_gate": True})


def aggregate(work_dir):
    work = Path(work_dir)
    try:
        ctx, out = read_json(work / "context.json"), work / "out"
        workers = check_workers(out, ctx)
        telemetry(out)
        started = time.monotonic()
        _run_container(container_command(work, ctx, "aggregate"), out / "aggregate.log")
        result_path = out / ctx["mode"] / "aggregate.json"
        check_aggregate(read_json(result_path), ctx)
        require(read_json(result_path)["ranks"] == [runtime_worker(out, ctx, r) for r in range(4)], "aggregate worker manifest drift")
        shutil.copyfile(result_path, out / "technical_aggregate.json")
        if ctx["mode"] == "full":
            stage_evaluator(work, ctx)
            _run_container(container_command(work, ctx, "aggregate", evaluator=True), out / "analysis.log")
            check_aggregate(read_json(result_path), ctx)
            require(read_json(result_path).get("technical_only") is False and "analysis" in read_json(result_path), "missing full analysis")
        elapsed = time.monotonic() - started
        result = {"status": "pass", "mode": ctx["mode"], "identity": ctx["identity"], "identity_sha256": ctx["identity_sha256"],
                  "run_id": ctx["run_id"], "aggregate_seconds": elapsed, "runtime_fingerprint": workers[0]["runtime_fingerprint"],
                  "aggregate_sha256": sha256(result_path), "technical_sha256": sha256(out / "technical_aggregate.json")}
        atomic_json(out / "release_profile.json", {"identity": ctx["identity"], "binding_sha256": digest(ctx["binding"]),
                    "mode": ctx["mode"], "runtime_fingerprint": result["runtime_fingerprint"],
                    "aggregate_sha256": result["aggregate_sha256"], "analysis_path": ctx["mode"] + "/aggregate.json" if ctx["mode"] == "full" else None,
                    "claim_scope": "prospective follow-up on reused Job E cohort; not fresh confirmation"})
        if ctx["mode"] == "canary":
            result["full_walltime_seconds"] = project_seconds(ctx, workers, elapsed)
            result["full_budget_status"] = "pass" if result["full_walltime_seconds"] <= 10800 else "over_budget"
        atomic_json(out / "completion_receipt.json", result)
        return result
    except Exception:
        failure(work, "aggregate")
        raise


def finish(work_dir, exit_code, stage_name):
    work, out = Path(work_dir), Path(work_dir) / "out"
    out.mkdir(parents=True, exist_ok=True)
    job = read_json(work / "context.json") if (work / "context.json").exists() else {"schema_version": 1}
    success = exit_code == 0 and stage_name == "complete" and (out / "completion_receipt.json").exists()
    if success:
        try:
            check_workers(out, job)
            telemetry(out)
            require(not (out / "periodic_copyback_errors.log").exists(), "periodic copyback failed")
        except Exception:
            failure(work, "finish")
            success = False
    job.update(status="pass" if success else "fail", exit_code=0 if success else (exit_code or 1), final_stage=stage_name, finished_at=time.time())
    atomic_json(out / "job_manifest.json", job)
    atomic_json(out / "copyback_manifest.json", {"schema_version": 1, "status": job["status"],
                "files": [entry(p, out) for p in tree_files(out, ("copyback_manifest.json", "copyback_verified.json"))]})
    return job


def gate(release, expected_sha, canary_dir):
    ident, out = identity(release, expected_sha), Path(canary_dir)
    require((out / "copyback_verified.json").exists(), "canary copyback not published")
    require(verify_copyback(out)["status"] == "pass", "canary copyback failed")
    ctx = read_json(out / "job_manifest.json")
    require(ctx["status"] == "pass" and ctx["exit_code"] == 0 and ctx["mode"] == "canary", "canary job failed")
    require(ctx["identity"] == ident and ctx["identity_sha256"] == digest(ident), "stale canary identity")
    require(ctx["spec"] == verify(release, expected_sha), "canary deployment drift")
    require(ctx["binding"] == expected_binding(release), "canary runtime binding drift")
    workers = check_workers(out, ctx)
    telemetry(out)
    result = read_json(out / "completion_receipt.json")
    require(result["identity"] == ident and result["status"] == "pass" and result["mode"] == "canary"
            and result["run_id"] == ctx["run_id"], "completion receipt drift")
    for path, key in ((out / "canary/aggregate.json", "aggregate_sha256"), (out / "technical_aggregate.json", "technical_sha256")):
        require(result[key] == sha256(path), "aggregate digest drift")
        check_aggregate(read_json(path), ctx)
        require(read_json(path)["ranks"] == [runtime_worker(out, ctx, r) for r in range(4)], "aggregate worker evidence drift")
    seconds = project_seconds(ctx, workers, result["aggregate_seconds"])
    require(result["full_walltime_seconds"] == seconds and result["full_budget_status"] == "pass", "canary projection drift/over budget")
    require(seconds <= 10800, "measured full projection exceeds three hours")
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fields = {
        "bundle": ("repo-root", "out-dir"),
        "bootstrap": ("source-root", "source-sha", "output-root", "assets-root", "model-root", "sif-path"),
        "verify": ("release", "expected-sha"),
        "stage": ("release", "expected-sha", "work-dir", "mode", "run-id", "executing-script", "walltime-seconds"),
        "worker": ("work-dir", "rank"), "aggregate": ("work-dir",),
        "finish": ("work-dir", "exit-code", "stage"), "publish": ("out-dir",),
        "verify-copyback": ("out-dir",), "gate": ("release", "expected-sha", "canary-dir"),
    }
    for command, names in fields.items():
        child = sub.add_parser(command)
        for name in names:
            child.add_argument("--" + name, required=True, type=int if name in ("rank", "exit-code", "walltime-seconds") else str)
        if command == "bootstrap":
            child.add_argument("--model-inventory")
        if command == "stage":
            child.add_argument("--canary-dir")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "finish":
        args["stage_name"] = args.pop("stage")
    result = globals()[command.replace("-", "_")](**args)
    print(result if command == "gate" else canonical(result).decode())
    return result.get("exit_code", 0) if command == "finish" else 0


if __name__ == "__main__":
    raise SystemExit(main())
