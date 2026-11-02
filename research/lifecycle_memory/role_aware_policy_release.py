"""Deployment wrapper for the unchanged two-policy design; stdlib on login.

The supplied job folder and scientific modules are copied byte-for-byte. All
training, model checks, asset hashing and analysis run in the Slurm allocation.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.lifecycle_memory import decoder_token_matrix_release as safe
from research.lifecycle_memory import decoder_cached_validation_release as cv

JOB_REL = "docs/research/lifecycle_memory/phase 2 research/jobs/role_aware_policy"
CONFIG = JOB_REL + "/config.json"
JOB = JOB_REL + "/job_tamia_role_aware.sh"
HELPER = "research/lifecycle_memory/role_aware_policy_release.py"
RUNTIME = "research/lifecycle_memory/run_role_aware_policy.py"
PRIOR_SHA = "0e61acb3576e4568557821ec9a319e3a4414146c28b6d58cd56067880d1dc75c"
TRAIN_SHA = "a5e21649d4f18fc7af0c6d44b71432e6a6c05b9c73eb3e90603e0a1ce438d904"
TRAIN_INVENTORY_SHA = "b7e579c8e4607738d133d3cbe354f9bd0eb060e973e9148be86febf1a9d5d60d"
MODEL_SHA = "f7423b54084ade93a1255970fb17bbecac4811430cf5fb5670bdbe379163766f"
EVAL_SHA = "e8807ff5f26a65191cb6714866ca1338825c01a60df9d3ea0ade9be87bd7343f"
REFERENCE_CASES = ("native__correct_L34_CQ", "native__correct_L34_readout")


def source_files(root):
    root = Path(root)
    # Established loaders include a file import under the spaced job0 path.
    pending = {HELPER, RUNTIME, "research/lifecycle_memory/role_aware_policy.py"}
    selected = set(safe.source_files(root)) | set(cv.source_files(root))
    selected.update(str(p.relative_to(root)) for p in safe.tree_files(root / JOB_REL))
    selected.update({"scripts/deploy_role_aware_policy_tamia.sh", "scripts/fetch_role_aware_policy_tamia.sh"})
    selected.add("scripts/run_role_aware_combined_tamia.sh")
    guide = root / "docs/research/lifecycle_memory/phase 2 research/jobs/role_aware_combined"
    if guide.exists():
        selected.update(str(p.relative_to(root)) for p in safe.tree_files(guide))
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        modules = set()
        for node in ast.walk(ast.parse((root / name).read_text(), filename=name)):
            if isinstance(node, ast.Import):
                modules.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                modules.add(base)
                modules.update(base + "." + a.name for a in node.names)
        for module in modules:
            parts = module.split(".")
            names = ["/".join(parts) + ".py"]
            names += ["/".join(parts[:i]) + "/__init__.py" for i in range(1, len(parts) + 1)]
            pending.update(p for p in names if (root / p).is_file() and p not in selected)
    return sorted(selected)


def bundle(repo_root, out_dir):
    root, out = Path(repo_root).resolve(), Path(out_dir)
    safe.require(not out.exists(), "bundle destination must be new")
    from research.lifecycle_memory.role_aware_policy import validate_config
    validate_config(safe.read_json(root / CONFIG))
    names = source_files(root)
    safe.require(sum((root / n).stat().st_size for n in names) < cv.LIMIT, "bundle exceeds 32MiB")
    for name in names:
        src, dest = root / name, out / name
        safe.require(src.is_file() and not src.is_symlink(), "unsafe source")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    safe.atomic_json(out / "source_manifest.json", {"schema_version": 1,
                     "files": [safe.entry(p, out) for p in safe.tree_files(out)]})
    return safe.sha256(out / "source_manifest.json")


def verify_source(source_root, source_sha):
    return cv.bounded_verify(source_root, "source_manifest.json", source_sha)


def workloads(data_root, reference_config):
    from research.lifecycle_memory import decoder_cached_validation as data
    result = {}
    for mode in ("canary", "full"):
        rows = data.public_rows(data_root, mode, reference_config)
        shards, timed = data.frame_shards(rows), data.benchmark_ids(rows, mode, reference_config)
        repeats = 2 if mode == "canary" else 5
        result[mode] = [{"predictions": sum(shards[r["request_id"]] == rank for r in rows) * 26,
                         "timings": sum(shards[rid] == rank for rid in timed) * 26 * repeats}
                        for rank in range(4)]
    return result


def prepare(source_root, source_sha, prior_release, prior_sha, train_root, eval_root, eval_sha, out_dir):
    source, prior, out = Path(source_root), Path(prior_release), Path(out_dir)
    verify_source(source, source_sha)
    safe.require(prior_sha == PRIOR_SHA and eval_sha == EVAL_SHA, "prior/data pin drift")
    old, model = cv.prior_metadata(prior, prior_sha)
    safe.require(safe.sha256(prior / old["model"]["inventory"]) == MODEL_SHA, "model inventory drift")
    safe.require(safe.sha256(prior / "data_inventory.json") == TRAIN_INVENTORY_SHA, "training inventory drift")
    safe.require(old["sif"]["sha256"] == safe.SIF_SHA256, "SIF pin drift")
    train = Path(train_root) / "dataset"
    safe.require(train.is_absolute() and safe.sha256(train / "split_manifest.json") == TRAIN_SHA,
                 "training split absent or changed")
    cv.upload_metadata(eval_root, eval_sha)
    safe.require(not out.exists() and not out.is_symlink(), "release destination must be new")
    spec = {"schema_version": 1, "source_sha256": source_sha, "prior_sha256": prior_sha,
            "model": {"root": old["model"]["root"], "inventory": "model_inventory.json"},
            "sif": old["sif"], "train_root": str(train), "assets_root": old["data"]["assets_root"],
            "eval_root": str(Path(eval_root)), "training_manifest_sha256": TRAIN_SHA,
            "evaluation_manifest_sha256": EVAL_SHA, "config_path": CONFIG,
            "workloads": workloads(eval_root, cv.small_json(source / cv.CONFIG))}
    out.parent.mkdir(parents=True, exist_ok=True)
    # Uploaded sources are sealed read-only. Copy bytes into fresh writable
    # directories, then publish only after all metadata and hashes verify.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=out.parent) as temp:
        staged = Path(temp) / "release"
        staged.mkdir()
        for src in safe.tree_files(source):
            dest = staged / src.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        for src, name in ((prior / "data_inventory.json", "training_inventory.json"),
                          (prior / old["model"]["inventory"], "model_inventory.json"),
                          (Path(eval_root) / "data_manifest.json", "evaluation_manifest.json")):
            shutil.copyfile(src, staged / name)
        safe.atomic_json(staged / "role_deployment.json", spec)
        safe.atomic_json(staged / "release_manifest.json", {"schema_version": 1,
                         "files": [safe.entry(p, staged) for p in safe.tree_files(staged)]})
        digest = safe.sha256(staged / "release_manifest.json")
        verify(staged, digest)
        safe.require(not out.exists() and not out.is_symlink(), "release destination must be new")
        staged.rename(out)
    return digest


def verify(release, expected_sha):
    root = Path(release)
    manifest = cv.bounded_verify(root, "release_manifest.json", expected_sha)
    spec = cv.small_json(root / "role_deployment.json")
    safe.require(spec["schema_version"] == 1 and spec["prior_sha256"] == PRIOR_SHA, "release identity drift")
    source = cv.small_json(root / "source_manifest.json")
    safe.require(safe.sha256(root / "source_manifest.json") == spec["source_sha256"], "source pin drift")
    safe.verify_inventory(root, source["files"], exact=False)
    extras = {"source_manifest.json", "role_deployment.json", "training_inventory.json",
              "model_inventory.json", "evaluation_manifest.json"}
    safe.require({r["path"] for r in manifest["files"]} == {r["path"] for r in source["files"]} | extras,
                 "release inventory drift")
    for name, value in (("training_inventory.json", TRAIN_INVENTORY_SHA),
                        ("model_inventory.json", MODEL_SHA), ("evaluation_manifest.json", EVAL_SHA)):
        safe.require(safe.sha256(root / name) == value, f"pin drift: {name}")
    safe.require(spec["sif"]["sha256"] == safe.SIF_SHA256 and spec["training_manifest_sha256"] == TRAIN_SHA
                 and spec["evaluation_manifest_sha256"] == EVAL_SHA, "data/runtime pin drift")
    return spec


def identity(release, sha):
    spec = verify(release, sha)
    return {"release_sha256": sha, "source_sha256": spec["source_sha256"],
            "script_sha256": safe.sha256(Path(release) / JOB), "sif_sha256": safe.SIF_SHA256,
            "training_manifest_sha256": TRAIN_SHA, "evaluation_manifest_sha256": EVAL_SHA,
            "model_inventory_sha256": MODEL_SHA}


def stage(release, expected_sha, work_dir, mode, executing_script, walltime_seconds, canary_dir=None):
    safe.require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_TMPDIR"), "compute allocation required")
    work, release = Path(work_dir).absolute(), Path(release)
    safe.require(Path(os.environ["SLURM_TMPDIR"]).resolve() in work.resolve().parents, "work must be node-local")
    safe.require(mode in ("canary", "full") and not work.exists(), "invalid mode or existing work tree")
    start = time.monotonic()
    spec, ident = verify(release, expected_sha), identity(release, expected_sha)
    expected = 1200 if mode == "canary" else gate(release, expected_sha, canary_dir)
    safe.require(walltime_seconds == expected, "walltime differs from canary gate")
    safe.require(safe.sha256(executing_script) == ident["script_sha256"], "executing script drift")
    (work / "out").mkdir(parents=True)
    ctx = {"mode": mode, "run_id": os.environ["SLURM_JOB_ID"], "identity": ident, "spec": spec,
           "walltime_seconds": walltime_seconds, "status": "staging"}
    safe.atomic_json(work / "context.json", ctx)
    shutil.copytree(release, work / "release")
    verify(work / "release", expected_sha)
    safe.archive_release(work / "release", work / "out")
    for row in safe.inventory(cv.small_json(release / "training_inventory.json")):
        if row.get("source", "package") == "assets":
            root, dest = Path(spec["assets_root"]), work / "train_assets" / row["path"]
        else:
            # Only training labels and the public split are ever mounted here.
            if row["path"] not in ("split_manifest.json", "public/requests.jsonl", "labels/train.jsonl"):
                continue
            root, dest = Path(spec["train_root"]), work / "dataset" / row["path"]
        safe._copy_checked(root / row["path"], dest, row, root)
    safe.require(safe.sha256(work / "dataset/split_manifest.json") == TRAIN_SHA, "staged split drift")
    for row in safe.inventory(cv.small_json(release / "evaluation_manifest.json")):
        private = row["path"].startswith("evaluator/")
        if private and mode == "canary":
            continue
        dest = work / ("private_eval" if private else "eval") / row["path"]
        safe._copy_checked(Path(spec["eval_root"]) / row["path"], dest, row, spec["eval_root"])
    shutil.copyfile(release / "evaluation_manifest.json", work / "eval/data_manifest.json")
    (work / "eval/evaluator").mkdir(exist_ok=True)
    for row in safe.inventory(cv.small_json(release / "model_inventory.json")):
        safe._copy_checked(Path(spec["model"]["root"]) / row["path"], work / "model" / row["path"],
                           row, spec["model"]["root"])
    sif = Path(spec["sif"]["path"])
    safe._copy_checked(sif, work / "runtime.sif", {"bytes": sif.stat().st_size,
                       "sha256": safe.SIF_SHA256}, sif.parent)
    ctx.update(status="running", staging_seconds=time.monotonic() - start)
    safe.atomic_json(work / "context.json", ctx)
    safe.atomic_json(work / "out/job_manifest.json", ctx)
    return ctx


def container_command(work, ctx, operation, rank=None):
    work = Path(work).resolve()
    safe.require(operation in ("train", "freeze", "evaluate", "aggregate"), "invalid stage")
    gpu = operation in ("train", "evaluate")
    safe.require((gpu and type(rank) is int and 0 <= rank < 4) or (not gpu and rank is None), "invalid rank")
    command = ["apptainer", "exec", "--containall", "--cleanenv", "--no-mount", "bind-paths,hostfs,cwd,home"]
    private = work / "apptainer-work" / f"{operation}-{rank}"
    private.mkdir(parents=True, exist_ok=True)
    command += ["--workdir", str(private)]
    names = ["release", "out"]
    if operation == "train":
        names += ["dataset", "train_assets", "model"]
    if operation in ("evaluate", "aggregate"):
        names += ["eval"]
    if operation == "evaluate":
        names += ["model"]
    for name in names:
        command += ["--bind", f"{work / name}:/mnt/rap/{name}:{'rw' if name == 'out' else 'ro'}"]
    if operation == "aggregate" and ctx["mode"] == "full":
        command += ["--bind", f"{work / 'private_eval/evaluator'}:/mnt/rap/eval/evaluator:ro"]
    env = {"PYTHONPATH": "/mnt/rap/release", "HF_HOME": "/mnt/rap/model", "HF_HUB_CACHE": "/mnt/rap/model/hub",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
           "OMP_NUM_THREADS": "12", "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
           "SLURM_PROCID": str(rank) if gpu else "0"}
    if gpu:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        safe.require(visible and len(visible.split(",")) == 1 and visible not in ("-1", "NoDevFiles"), "one Slurm GPU required")
        env["CUDA_VISIBLE_DEVICES"] = visible
        command += ["--nv"]
    for key, value in env.items():
        command += ["--env", f"{key}={value}"]
    command += ["--pwd", "/mnt/rap/release", str(work / "runtime.sif"), "python", "-u", "-m",
                "research.lifecycle_memory.role_aware_policy_release", "runtime", "--stage", operation,
                "--mode", ctx["mode"]]
    if gpu:
        command += ["--rank", str(rank)]
    return command


def runtime(stage_name, mode, rank=0, base=Path("/mnt/rap")):
    """Call public Python APIs; the legacy freeze CLI is intentionally unused."""
    from research.lifecycle_memory import run_role_aware_policy as runner
    out, release = base / "out", base / "release"
    source_config, frozen = release / CONFIG, out / "frozen_config.json"
    args = SimpleNamespace(mode=mode, rank=rank, world_size=4, config=source_config,
                           dataset_dir=base / "dataset", data_dir=base / "train_assets",
                           out_dir=out / "train", adapters_dir=out / "adapters")
    if stage_name == "train":
        return runner.train_worker(args)
    if stage_name == "freeze":
        return runner.freeze(source_config, out / "train", mode=mode, output_config=frozen,
                             adapters_dir=out / "adapters")
    args.config, args.data_dir, args.out_dir = frozen, base / "eval", out / "evaluate"
    if stage_name == "evaluate":
        return runner.evaluate_worker(args)
    return runner.aggregate(args)


def run_stage(work_dir, stage_name, rank=None):
    work = Path(work_dir)
    ctx = safe.read_json(work / "context.json")
    start = time.monotonic()
    try:
        command = container_command(work, ctx, stage_name, rank)
        log = work / "out" / f"container-{stage_name}-{rank}.log"
        safe._run_container(command, log)
        result = {"status": "pass", "stage": stage_name, "rank": rank, "mode": ctx["mode"],
                  "identity": ctx["identity"], "elapsed_s": time.monotonic() - start}
        safe.atomic_json(work / "out" / f"harness-{stage_name}-{rank}.json", result)
        if stage_name == "aggregate":
            return finalize_evidence(work)
        return result
    except Exception:
        safe.failure(work, stage_name, rank)
        raise


def projected_walltime(ctx, training, workers, aggregate_s):
    """Conservative per-rank scaling; retain staging/loading once, add 25%+5min."""
    train_seconds = max(sum(r["load_s"] + r["train_s"] * 256 / r["steps"]
                            for r in training if r["worker_rank"] == rank) for rank in range(4))
    # Use exact per-rank counts; the balanced-by-source timing subset can be
    # uneven across the frame-hashed GPU shards.
    eval_seconds = max(w["setup_s"] + max(0., w["work_s"] - w["setup_s"]) *
                       max(ctx["spec"]["workloads"]["full"][w["rank"]]["predictions"] / w["predictions"],
                           ctx["spec"]["workloads"]["full"][w["rank"]]["timings"] / w["timings"])
                       for w in workers)
    projected = ctx["staging_seconds"] + train_seconds + eval_seconds + aggregate_s
    return {"train_s": train_seconds, "evaluate_s": eval_seconds, "projected_s": projected,
            "recommended_walltime_s": max(3600, math.ceil((1.25 * projected + 300) / 300) * 300)}


def finalize_evidence(work):
    work = Path(work)
    out = work / "out"
    ctx = safe.read_json(work / "context.json")
    for step, ranks in (("train", range(4)), ("evaluate", range(4)), ("freeze", [None]), ("aggregate", [None])):
        for rank in ranks:
            r = safe.read_json(out / f"harness-{step}-{rank}.json")
            safe.require(r["status"] == "pass" and r["identity"] == ctx["identity"] and r["mode"] == ctx["mode"],
                         "stage receipt mismatch")
    safe.require(not list(out.rglob("error.json")) and not list(out.glob("failure_*.json")), "failure artifacts found")
    frozen = safe.read_json(out / "frozen_config.json")
    safe.require(frozen["freeze"]["mode"] == ctx["mode"] and len(frozen["adapters"]) == 10, "adapter freeze mismatch")
    training = [safe.read_json(p) for p in sorted((out / "train").glob("rank_*/*/training_receipt.json"))]
    safe.require(len(training) == 6 and all(r["mode"] == ctx["mode"] and r["status"] == "pass" for r in training),
                 "six trained adapters required")
    workers = []
    for rank in range(4):
        d = out / "evaluate" / f"rank_{rank:03d}"
        r = safe.read_json(d / "worker_receipt.json")
        safe.require(r["status"] == "pass" and r["rank"] == rank and r["mode"] == ctx["mode"], "worker mismatch")
        runtime_info = r["runtime"]
        model = safe.read_json(work / "release" / CONFIG)["model"]
        safe.require(runtime_info is not None and runtime_info["model_revision"] == model["revision"]
                     and runtime_info["torch"] == model["torch"] and runtime_info["transformers"] == model["transformers"]
                     and "H100" in runtime_info["device"], "runtime/model/GPU mismatch")
        workers.append({**r, "setup_s": safe.read_json(d / "cold_start.json")["setup_s"]})
    summary = safe.read_json(out / "evaluate/analysis_summary.json")
    safe.require(summary["status"] == "pass" and summary["mode"] == ctx["mode"], "aggregate missing/failed")
    expected_p = (32 if ctx["mode"] == "canary" else 894) * 26
    expected_t = (32 * 2 if ctx["mode"] == "canary" else 128 * 5) * 26
    safe.require(summary["prediction_rows"] == expected_p and summary["timing_rows"] == expected_t,
                 "aggregate inventory mismatch")
    budget = projected_walltime(ctx, training, workers,
                                safe.read_json(out / "harness-aggregate-None.json")["elapsed_s"])
    result = {"status": "pass", "mode": ctx["mode"], "run_id": ctx["run_id"], "identity": ctx["identity"],
              "training_tasks": 6, "prediction_rows": expected_p, "timing_rows": expected_t,
              "budget": budget, "full_launch_allowed": budget["recommended_walltime_s"] < 10800,
              "evaluation_scope": "reused descriptive development; not fresh confirmation"}
    safe.atomic_json(out / "technical_summary.json", result)
    shutil.copyfile(out / "evaluate/analysis_summary.json", out / "analysis_summary.json")
    if ctx["mode"] == "full":
        shutil.copyfile(out / "evaluate/RESULTS.md", out / "RESULTS.md")
    return result


def finish(work_dir, exit_code, stage_name):
    work, out = Path(work_dir), Path(work_dir) / "out"
    out.mkdir(parents=True, exist_ok=True)
    ctx = safe.read_json(work / "context.json") if (work / "context.json").exists() else {}
    technical = safe.read_json(out / "technical_summary.json") if (out / "technical_summary.json").exists() else {}
    status = "pass" if exit_code == 0 and stage_name == "complete" and technical.get("status") == "pass" else "fail"
    job = {"status": status, "exit_code": exit_code, "stage": stage_name,
           "identity": ctx.get("identity"), "mode": ctx.get("mode"), "run_id": ctx.get("run_id"),
           "scientific_success": "see analysis_summary.json; technical pass is not a scientific win"}
    safe.atomic_json(out / "job_receipt.json", job)
    safe.atomic_json(out / "copyback_manifest.json", {"schema_version": 1, "status": status,
                     "files": [safe.entry(p, out) for p in safe.tree_files(out, ("copyback_manifest.json",))]})
    return job


def gate(release, expected_sha, canary_dir):
    safe.require(canary_dir is not None, "full requires a canary")
    out = Path(canary_dir)
    # Login uses bounded receipts only; large output integrity was sealed by publish.
    pub = cv.small_json(out / "copyback_verified.json")
    safe.require(pub["status"] == "pass" and pub["manifest_sha256"] == safe.sha256(out / "copyback_manifest.json"),
                 "canary copyback not verified")
    manifest = cv.small_json(out / "copyback_manifest.json")
    names = {r["path"]: r for r in safe.inventory(manifest)}
    for name in ("technical_summary.json", "job_receipt.json"):
        safe.require(name in names and safe.sha256(out / name) == names[name]["sha256"], "canary receipt hash drift")
    t, job = cv.small_json(out / "technical_summary.json"), cv.small_json(out / "job_receipt.json")
    ident = identity(release, expected_sha)
    safe.require(job["status"] == "pass" and job["exit_code"] == 0 and job["stage"] == "complete"
                 and job["mode"] == t["mode"] == "canary" and job["identity"] == t["identity"] == ident,
                 "canary status or identity mismatch")
    safe.require(job["run_id"] == t["run_id"] == out.name, "canary ID mismatch")
    safe.require(t["training_tasks"] == 6 and t["prediction_rows"] == 832 and t["timing_rows"] == 1664,
                 "incomplete canary")
    seconds = t["budget"]["recommended_walltime_s"]
    safe.require(t["full_launch_allowed"] and type(seconds) is int and 3600 <= seconds < 10800
                 and seconds % 300 == 0, "full budget too large; inspect measurement before submission")
    return seconds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    fields = {
        "bundle": ["repo-root", "out-dir"], "verify-source": ["source-root", "source-sha"],
        "prepare": ["source-root", "source-sha", "prior-release", "prior-sha", "train-root", "eval-root", "eval-sha", "out-dir"],
        "verify": ["release", "expected-sha"], "gate": ["release", "expected-sha", "canary-dir"],
        "stage": ["release", "expected-sha", "work-dir", "mode", "executing-script", "walltime-seconds"],
        "worker": ["work-dir", "stage", "rank"], "freeze": ["work-dir"], "aggregate": ["work-dir"],
        "finish": ["work-dir", "exit-code", "stage-name"], "publish": ["out-dir"],
        "verify-copyback": ["out-dir"], "runtime": ["stage", "mode"]}
    for cmd, names in fields.items():
        parser = sub.add_parser(cmd)
        for name in names:
            parser.add_argument("--" + name, required=True, type=int if name in ("rank", "exit-code", "walltime-seconds") else str)
        if cmd == "stage":
            parser.add_argument("--canary-dir")
        if cmd == "runtime":
            parser.add_argument("--rank", type=int, default=0)
    args = vars(p.parse_args(argv))
    cmd = args.pop("command")
    if cmd == "worker":
        result = run_stage(args["work_dir"], args["stage"], args["rank"])
    elif cmd in ("freeze", "aggregate"):
        result = run_stage(args["work_dir"], cmd)
    elif cmd == "runtime":
        result = runtime(args["stage"], args["mode"], args["rank"])
    else:
        fun = {"verify-copyback": safe.verify_copyback, "publish": safe.publish,
               "verify-source": verify_source}.get(cmd, globals().get(cmd))
        try:
            result = fun(**args)
        except Exception:
            if cmd == "stage" and Path(args["work_dir"]).exists():
                safe.failure(args["work_dir"], "stage")
            raise
    print(result if isinstance(result, (str, int)) else json.dumps(result, sort_keys=True))
    if cmd == "finish" and args["exit_code"] == 0 and result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
