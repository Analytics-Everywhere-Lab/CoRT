"""Eval-only Killarney release. Login commands read bounded metadata only."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.lifecycle_memory import role_aware_killarney_release as previous
from research.lifecycle_memory import role_aware_policy_release as policy
from research.lifecycle_memory import role_aware_transfer as science

safe, cv = policy.safe, policy.cv
JOB_REL = "docs/research/lifecycle_memory/phase 2 research/jobs/role_aware_transfer"
CONFIG = JOB_REL + "/config.json"
JOB = JOB_REL + "/job_killarney.sh"
HELPER = "research/lifecycle_memory/role_aware_transfer_release.py"
RUNTIME_MODULE = "research.lifecycle_memory.run_role_aware_transfer"
NEW_FILES = {HELPER, CONFIG, JOB, JOB_REL + "/worker.sh", JOB_REL + "/submit.sh", JOB_REL + "/README.md",
             "research/lifecycle_memory/role_aware_transfer.py",
             "research/lifecycle_memory/run_role_aware_transfer.py",
             "research/lifecycle_memory/tests/test_role_aware_transfer_release.py",
             "research/lifecycle_memory/tests/test_role_aware_transfer.py",
             "research/lifecycle_memory/tests/test_role_aware_transfer_deploy.py",
             "scripts/run_role_aware_transfer_killarney.py",
             "docs/research/lifecycle_memory/role_aware_followup_protocol_2026_10_20.md"}
MODULES = ["StdEnv/2023", "python/3.12.4", "apptainer/1.4.5", "slurm/killarney/25.05.9"]
COUNTS = {"canary": (160, 320), "full": (4470, 3200)}
EXTRAS = {"source_manifest.json", "role_deployment.json", "model_inventory.json", "evaluation_manifest.json",
          "platform.json", "previous_deployment.json", "previous_release_manifest.json"}


def private_module(module, name):
    spec = importlib.util.spec_from_file_location(name, module.__file__)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def validate_platform(value):
    profile = previous.validate_platform(value)
    safe.require(profile["modules"] == MODULES and profile["memory_gb"] == 480,
                 "exact verified modules and 480G required")
    return profile


def config_check(root):
    root = Path(root)
    config = science.validate_config(cv.small_json(root / CONFIG))
    safe.require(config == science.source_config(), "source config must be unfrozen")
    safe.require(safe.sha256(root / science.REFERENCE_CONFIG) == science.REFERENCE_SHA,
                 "reference config pin drift")
    reference = cv.small_json(root / science.REFERENCE_CONFIG)
    safe.require(reference["source_run"]["id"] == "620395" and reference["model"] == config["model"],
                 "reference model/run drift")
    # These tiny adapters belong to the bounded source closure, not bulk inputs.
    for seed, digest in science.HASHES.items():
        name = f"{science.CASE}.seed{seed}.final.safetensors"
        safe.require(safe.sha256(root / science.REFERENCE_ADAPTERS / name) == digest,
                     "reference adapter pin drift")
    return config


def source_files(root):
    return sorted(set(previous.source_files(root)) | NEW_FILES)


def bundle(repo_root, out_dir):
    root, out = Path(repo_root).resolve(), Path(out_dir)
    previous.require_new(out)
    config_check(root)
    names = source_files(root)
    for name in names:
        path = root / name
        safe.require(path.is_file() and not path.is_symlink() and root in path.resolve().parents,
                     f"missing/unsafe source: {name}")
    safe.require(sum((root / name).stat().st_size for name in names) <= cv.LIMIT, "bundle exceeds 32MiB")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".transfer-bundle-", dir=out.parent) as temp:
        staged = Path(temp) / "source"
        previous.bundle(root, staged)
        for name in names:
            dest = staged / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, dest)
        manifest(staged, "source_manifest.json")
        digest = safe.sha256(staged / "source_manifest.json")
        verify_source(staged, digest)
        previous.require_new(out)
        staged.rename(out)
    return digest


def manifest(root, name):
    safe.atomic_json(Path(root) / name, {"schema_version": 1,
        "files": [safe.entry(p, root) for p in safe.tree_files(root, (name,))]})


def verify_source(source_root, source_sha):
    result = previous.verify_source(source_root, source_sha)
    safe.require(NEW_FILES <= {r["path"] for r in result["files"]}, "incomplete transfer source")
    config_check(source_root)
    return result


def transfer_workloads(old):
    result = {}
    for mode, totals in COUNTS.items():
        rows = old[mode]
        safe.require(isinstance(rows, list) and len(rows) == 4, "four workload shards required")
        result[mode] = []
        for row in rows:
            new = {}
            for key in ("predictions", "timings"):
                n = row[key]
                safe.require(type(n) is int and n > 0 and n % 26 == 0, "invalid prior workload")
                new[key] = n // 26 * 5
            result[mode].append(new)
        safe.require(tuple(sum(r[k] for r in result[mode]) for k in ("predictions", "timings")) == totals,
                     "prior data/workload totals drift")
    return result


def prepare(source_root, source_sha, previous_release, previous_sha, out_dir):
    source, parent, out = Path(source_root), Path(previous_release), Path(out_dir)
    previous.require_new(out)
    verify_source(source, source_sha)
    old = previous.verify(parent, previous_sha)
    profile = validate_platform(old["platform"])
    spec = {"schema_version": 1, "experiment_id": science.EXPERIMENT_ID, "training_allowed": False,
            "source_sha256": source_sha, "config_path": CONFIG, "config_sha256": safe.sha256(source / CONFIG),
            "previous_release": previous.absolute_path(parent), "previous_sha256": previous_sha,
            "previous_deployment_sha256": safe.sha256(parent / "role_deployment.json"),
            "model": copy.deepcopy(old["model"]), "sif": copy.deepcopy(old["sif"]),
            "eval_root": old["eval_root"], "evaluation_manifest_sha256": policy.EVAL_SHA,
            "platform": profile, "workloads": transfer_workloads(old["workloads"]),
            "reference_config_sha256": science.REFERENCE_SHA,
            "adapter_sha256": {str(k): v for k, v in science.HASHES.items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".transfer-prepare-", dir=out.parent) as temp:
        staged = Path(temp) / "release"
        for src in safe.tree_files(source):
            dest = staged / src.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        for name, dest in (("model_inventory.json", "model_inventory.json"),
                           ("evaluation_manifest.json", "evaluation_manifest.json"),
                           ("release_manifest.json", "previous_release_manifest.json"),
                           ("role_deployment.json", "previous_deployment.json")):
            shutil.copyfile(parent / name, staged / dest)
        safe.atomic_json(staged / "platform.json", profile)
        safe.atomic_json(staged / "role_deployment.json", spec)
        manifest(staged, "release_manifest.json")
        digest = safe.sha256(staged / "release_manifest.json")
        verify(staged, digest)
        previous.require_new(out)
        staged.rename(out)
    return digest


def verify(release, expected_sha):
    root = Path(release)
    inventory = cv.bounded_verify(root, "release_manifest.json", expected_sha)
    spec = cv.small_json(root / "role_deployment.json")
    safe.require(spec["schema_version"] == 1 and spec["experiment_id"] == science.EXPERIMENT_ID
                 and spec["training_allowed"] is False, "eval-only release identity drift")
    source = cv.small_json(root / "source_manifest.json")
    safe.require(safe.sha256(root / "source_manifest.json") == spec["source_sha256"], "source pin drift")
    safe.verify_inventory(root, source["files"], exact=False)
    names = {r["path"] for r in source["files"]}
    safe.require(set(source_files(root)) <= names, "incomplete transfer source")
    safe.require({r["path"] for r in inventory["files"]} == names | EXTRAS, "release inventory drift")
    config_check(root)
    safe.require(spec["config_path"] == CONFIG and spec["config_sha256"] == safe.sha256(root / CONFIG)
                 and spec["reference_config_sha256"] == science.REFERENCE_SHA
                 and spec["adapter_sha256"] == {str(k): v for k, v in science.HASHES.items()}, "config/adapter pin drift")
    parent_manifest = cv.small_json(root / "previous_release_manifest.json")
    safe.require(safe.sha256(root / "previous_release_manifest.json") == safe.checked_hash(spec["previous_sha256"]),
                 "parent release pin drift")
    parents = {r["path"]: r for r in safe.inventory(parent_manifest)}
    old = cv.small_json(root / "previous_deployment.json")
    safe.require(safe.sha256(root / "previous_deployment.json") == spec["previous_deployment_sha256"]
                 == parents["role_deployment.json"]["sha256"], "parent deployment pin drift")
    for name, digest in (("model_inventory.json", policy.MODEL_SHA), ("evaluation_manifest.json", policy.EVAL_SHA)):
        safe.require(safe.sha256(root / name) == digest == parents[name]["sha256"], f"pin drift: {name}")
    safe.require(spec["model"] == old["model"] and spec["model"]["inventory"] == "model_inventory.json"
                 and spec["sif"] == old["sif"] and spec["sif"]["sha256"] == safe.SIF_SHA256
                 and spec["eval_root"] == old["eval_root"] and spec["evaluation_manifest_sha256"] == policy.EVAL_SHA,
                 "immutable input path/pin drift")
    profile = validate_platform(cv.small_json(root / "platform.json"))
    safe.require(profile == spec["platform"] == old["platform"], "platform drift")
    safe.require(spec["workloads"] == transfer_workloads(old["workloads"]), "workload drift")
    for path in (spec["model"]["root"], spec["sif"]["path"], spec["eval_root"], spec["previous_release"]):
        previous.absolute_path(path)
    return spec


def identity(release, sha):
    spec = verify(release, sha)
    return {"release_sha256": sha, "source_sha256": spec["source_sha256"],
            "config_sha256": spec["config_sha256"], "script_sha256": safe.sha256(Path(release) / JOB),
            "sif_sha256": safe.SIF_SHA256, "evaluation_manifest_sha256": policy.EVAL_SHA,
            "model_inventory_sha256": policy.MODEL_SHA, "previous_sha256": spec["previous_sha256"],
            "reference_config_sha256": science.REFERENCE_SHA, "adapter_sha256": spec["adapter_sha256"],
            "cluster": "killarney", "platform_sha256": safe.digest(spec["platform"])}


def allocation():
    safe.require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_TMPDIR"), "compute allocation required")
    safe.require(os.environ.get("SLURM_CLUSTER_NAME", "").startswith("killarney")
                 and os.environ.get("SLURM_JOB_NUM_NODES") == "1", "single-node Killarney allocation required")


def disk_requirement(release, spec, mode):
    root = Path(release)
    total = 2 * sum(r["bytes"] for r in safe.inventory(cv.small_json(root / "release_manifest.json")))
    total += 2 * (root / "release_manifest.json").stat().st_size
    for name in ("evaluation_manifest.json", "model_inventory.json"):
        total += sum(r["bytes"] for r in safe.inventory(cv.small_json(root / name))
                     if not (name == "evaluation_manifest.json" and mode == "canary" and r["path"].startswith("evaluator/")))
    return total + Path(spec["sif"]["path"]).stat().st_size + spec["platform"]["disk_margin_bytes"]


def stage(release, expected_sha, work_dir, mode, executing_script, walltime_seconds, canary_dir=None):
    allocation()
    work, release = Path(work_dir).absolute(), Path(release)
    previous.require_new(work)
    safe.require(mode in COUNTS, "invalid mode")
    local = Path(os.environ["SLURM_TMPDIR"]).resolve()
    safe.require(local in work.resolve().parents, "work must be node-local")
    start = time.monotonic()
    spec, ident = verify(release, expected_sha), identity(release, expected_sha)
    safe.require(os.environ.get("RAP_PLATFORM_SHA256") == safe.digest(spec["platform"]), "allocation platform drift")
    expected = 1200 if mode == "canary" else gate(release, expected_sha, canary_dir)
    safe.require(type(walltime_seconds) is int and walltime_seconds == expected, "walltime differs from canary gate")
    safe.require(safe.sha256(executing_script) == ident["script_sha256"], "executing script drift")
    safe.require(shutil.disk_usage(local).free >= disk_requirement(release, spec, mode), "insufficient node-local disk")
    (work / "out").mkdir(parents=True)
    ctx = {"mode": mode, "run_id": os.environ["SLURM_JOB_ID"], "identity": ident, "spec": spec,
           "walltime_seconds": walltime_seconds, "status": "staging"}
    safe.atomic_json(work / "context.json", ctx)
    shutil.copytree(release, work / "release")
    verify(work / "release", expected_sha)
    safe.archive_release(work / "release", work / "out")
    for row in safe.inventory(cv.small_json(release / "evaluation_manifest.json")):
        private = row["path"].startswith("evaluator/")
        if private and mode == "canary":
            continue
        dest = work / ("private_eval" if private else "eval") / row["path"]
        safe._copy_checked(Path(spec["eval_root"]) / row["path"], dest, row, spec["eval_root"])
    shutil.copyfile(release / "evaluation_manifest.json", work / "eval/data_manifest.json")
    (work / "eval/evaluator").mkdir(exist_ok=True)
    for row in safe.inventory(cv.small_json(release / "model_inventory.json")):
        safe._copy_checked(Path(spec["model"]["root"]) / row["path"], work / "model" / row["path"], row, spec["model"]["root"])
    sif = Path(spec["sif"]["path"])
    safe._copy_checked(sif, work / "runtime.sif", {"bytes": sif.stat().st_size, "sha256": safe.SIF_SHA256}, sif.parent)
    ctx.update(status="running", staging_seconds=time.monotonic() - start)
    safe.atomic_json(work / "context.json", ctx)
    safe.atomic_json(work / "out/job_manifest.json", ctx)
    return ctx


def container_command(work, ctx, operation, rank=None):
    allocation()
    safe.require(operation in ("freeze", "evaluate", "aggregate"), "eval-only stage required")
    profile = validate_platform(ctx["spec"]["platform"])
    safe.require(ctx["identity"]["cluster"] == "killarney" and
                 ctx["identity"]["platform_sha256"] == safe.digest(profile), "runtime platform drift")
    command = policy.container_command(work, ctx, operation, rank)
    if operation == "freeze":
        command[2:2] = ["--bind", f"{Path(work).resolve() / 'eval'}:/mnt/rap/eval:ro"]
    command[command.index("OMP_NUM_THREADS=12")] = "OMP_NUM_THREADS=6"
    command[command.index("research.lifecycle_memory.role_aware_policy_release")] = "research.lifecycle_memory.role_aware_transfer_release"
    command[command.index("--pwd") + 1] = "/mnt/rap/out"
    return command


def runtime(stage_name, mode, rank=0, root=Path("/mnt/rap")):
    safe.require(stage_name in ("freeze", "evaluate", "aggregate") and mode in COUNTS, "invalid runtime stage/mode")
    from research.lifecycle_memory import run_role_aware_transfer as runner
    if stage_name == "aggregate":
        config = Path(root) / "out/frozen_config.json"
        for worker_rank in range(4):
            runner.validate_worker(Path(root) / "out/evaluate" / f"rank_{worker_rank:03d}", config, mode, worker_rank)
    return runner.runtime(stage_name, mode, rank=rank, root=Path(root))


def positive(value, name, zero=False):
    safe.require(type(value) in (int, float) and math.isfinite(value) and (value >= 0 if zero else value > 0),
                 f"invalid measured {name}")
    return value


def projected_walltime(ctx, workers, freeze_s, aggregate_s):
    safe.require(len(workers) == 4 and {w["rank"] for w in workers} == set(range(4)), "four measured workers required")
    evaluations = []
    for worker in workers:
        setup = positive(worker["setup_s"], "setup_s", zero=True)
        elapsed = positive(worker["work_s"], "work_s")
        safe.require(elapsed > setup, "work must exceed setup")
        full = ctx["spec"]["workloads"]["full"][worker["rank"]]
        scale = max(full[k] / positive(worker[k], k) for k in ("predictions", "timings"))
        # setup_s includes per-request reference/gate/prediction forwards, not
        # just model loading. Scale the entire lifetime conservatively.
        evaluations.append(elapsed * scale)
    fixed = positive(ctx["staging_seconds"], "staging_seconds", zero=True)
    fixed += positive(freeze_s, "freeze_s", zero=True) + positive(aggregate_s, "aggregate_s", zero=True)
    projected = fixed + max(evaluations)
    return {"fixed_s": fixed, "evaluate_s": max(evaluations), "projected_s": projected,
            "recommended_walltime_s": max(3600, math.ceil((1.25 * projected + 300) / 300) * 300)}


def finalize_evidence(work):
    from research.lifecycle_memory import run_role_aware_transfer as runner
    work, out = Path(work), Path(work) / "out"
    ctx = safe.read_json(work / "context.json")
    safe.require(ctx["mode"] in COUNTS, "invalid mode")
    for step, ranks in (("freeze", [None]), ("evaluate", range(4)), ("aggregate", [None])):
        for rank in ranks:
            receipt = safe.read_json(out / f"harness-{step}-{rank}.json")
            safe.require(receipt["status"] == "pass" and receipt["stage"] == step and receipt["rank"] == rank
                         and receipt["identity"] == ctx["identity"] and receipt["mode"] == ctx["mode"], "stage receipt mismatch")
            positive(receipt["elapsed_s"], "stage elapsed")
    safe.require(not list(out.rglob("error.json")) and not list(out.glob("failure_*.json"))
                 and not (out / "train").exists() and not list(out.glob("harness-train-*")), "failure/training artifacts found")
    frozen = science.validate_config(safe.read_json(out / "frozen_config.json"))
    runner.validate_frozen(frozen, ctx["mode"])
    source = config_check(work / "release")
    safe.require(frozen["status"] == "adapters_frozen" and len(frozen["adapters"]) == 2
                 and frozen["freeze"]["mode"] == ctx["mode"], "adapter freeze mismatch")
    freeze = safe.read_json(out / "freeze_receipt.json")
    safe.require(freeze["status"] == "pass" and freeze["mode"] == ctx["mode"]
                 and freeze["experiment_id"] == science.EXPERIMENT_ID and freeze["training_allowed"] is False
                 and freeze["config_sha256"] == safe.sha256(out / "frozen_config.json")
                 and freeze["adapters"] == frozen["adapters"] and freeze["code"] == frozen["freeze"]["code"],
                 "freeze receipt mismatch")
    safe.require({k: v for k, v in frozen.items() if k not in ("adapters", "freeze", "status")} ==
                 {k: v for k, v in source.items() if k not in ("adapters", "freeze", "status")}, "frozen design drift")
    safe.require({p.name for p in safe.tree_files(out / "adapters")} == {a["path"] for a in frozen["adapters"]},
                 "exactly two adapter files required")
    for adapter in frozen["adapters"]:
        safe.require(safe.sha256(out / "adapters" / adapter["path"]) == science.HASHES[adapter["seed"]], "adapter bytes drift")
    dirs = {p.name for p in (out / "evaluate").glob("rank_*")}
    safe.require(dirs == {f"rank_{rank:03d}" for rank in range(4)}, "rank directory inventory drift")
    workers = []
    for rank in range(4):
        directory = out / "evaluate" / f"rank_{rank:03d}"
        runner.validate_worker(directory, out / "frozen_config.json", ctx["mode"], rank)
        r = safe.read_json(directory / "worker_receipt.json")
        safe.require(r["status"] == "pass" and r["rank"] == rank and r["mode"] == ctx["mode"]
                     and r["world_size"] == 4 and r["labels_accessed"] is False
                     and r["config_sha256"] == safe.sha256(out / "frozen_config.json"), "worker provenance mismatch")
        info, model = r["runtime"], source["model"]
        safe.require(info and info["model_revision"] == model["revision"] and info["torch"] == model["torch"]
                     and info["transformers"] == model["transformers"] and "H100" in info["device"], "runtime/model/GPU mismatch")
        for key in ("predictions", "timings"):
            safe.require(type(r[key]) is int and r[key] == ctx["spec"]["workloads"][ctx["mode"]][rank][key], "worker counts drift")
        setup = positive(safe.read_json(directory / "cold_start.json")["setup_s"], "setup_s", zero=True)
        elapsed = positive(r["work_s"], "work_s")
        safe.require(elapsed > setup, "work must exceed setup")
        harness = safe.read_json(out / f"harness-evaluate-{rank}.json")["elapsed_s"]
        # Include the complete process lifetime in the conservative scaling.
        workers.append({**r, "setup_s": setup, "work_s": max(elapsed, harness)})
    summary = safe.read_json(out / "evaluate/analysis_summary.json")
    p, t = COUNTS[ctx["mode"]]
    safe.require(summary["status"] == "pass" and summary["mode"] == ctx["mode"] and
                 summary["prediction_rows"] == p and summary["timing_rows"] == t
                 and summary["config_sha256"] == safe.sha256(out / "frozen_config.json")
                 and summary["labels_accessed"] is (ctx["mode"] == "full"), "aggregate inventory mismatch")
    budget = projected_walltime(ctx, workers, safe.read_json(out / "harness-freeze-None.json")["elapsed_s"],
                                safe.read_json(out / "harness-aggregate-None.json")["elapsed_s"])
    result = {"status": "pass", "mode": ctx["mode"], "run_id": ctx["run_id"], "identity": ctx["identity"],
              "frozen_adapters": 2, "prediction_rows": p, "timing_rows": t, "budget": budget,
              "full_launch_allowed": budget["recommended_walltime_s"] < 10800,
              "evaluation_scope": "reused descriptive development; not fresh confirmation"}
    shutil.copyfile(out / "evaluate/analysis_summary.json", out / "analysis_summary.json")
    shutil.copyfile(out / "evaluate/RESULTS.md", out / "RESULTS.md")
    safe.atomic_json(out / "technical_summary.json", result)
    return result


def gate(release, expected_sha, canary_dir):
    safe.require(canary_dir is not None, "full requires a canary")
    out = Path(canary_dir)
    pub = cv.small_json(out / "copyback_verified.json")
    sealed = cv.small_json(out / "copyback_manifest.json")
    safe.require(pub["status"] == "pass" and pub["manifest_sha256"] == safe.sha256(out / "copyback_manifest.json"),
                 "canary copyback not verified")
    safe.require(sealed["status"] == "pass", "failed copyback")
    names = {r["path"]: r for r in safe.inventory(sealed)}
    for name in ("technical_summary.json", "job_receipt.json"):
        cv.small_json(out / name)
        safe.require(name in names and safe.sha256(out / name) == names[name]["sha256"], "canary receipt hash drift")
    t, job = cv.small_json(out / "technical_summary.json"), cv.small_json(out / "job_receipt.json")
    ident = identity(release, expected_sha)
    safe.require(job["status"] == t["status"] == "pass" and job["exit_code"] == 0 and job["stage"] == "complete"
                 and job["mode"] == t["mode"] == "canary" and job["identity"] == t["identity"] == ident,
                 "canary status or identity mismatch")
    safe.require(job["run_id"] == t["run_id"] == out.name and t["frozen_adapters"] == 2
                 and t["prediction_rows"] == 160 and t["timing_rows"] == 320, "incomplete canary")
    seconds = t["budget"]["recommended_walltime_s"]
    safe.require(t["full_launch_allowed"] is True and type(seconds) is int and 3600 <= seconds < 10800
                 and seconds % 300 == 0, "full budget too large; inspect measurement before submission")
    return seconds


# Hooks are private: old launchers and imported scientific modules are untouched.
_engine = private_module(policy, "_role_transfer_policy_engine")
_engine.container_command, _engine.finalize_evidence = container_command, finalize_evidence
_submit = private_module(previous, "_role_transfer_submission_engine")
_submit.verify, _submit.identity, _submit.gate = verify, identity, gate
_submit.validate_platform = validate_platform
_submit.JOB, _submit.HELPER, _submit.__file__ = JOB, HELPER, __file__
submission = _submit.submission


def run_stage(work_dir, stage_name, rank=None):
    allocation()
    safe.require(stage_name in ("freeze", "evaluate", "aggregate"), "eval-only stage required")
    return _engine.run_stage(work_dir, stage_name, rank)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fields = {"bundle": ["repo-root", "out-dir"], "verify-source": ["source-root", "source-sha"],
        "prepare": ["source-root", "source-sha", "previous-release", "previous-sha", "out-dir"],
        "verify": ["release", "expected-sha"], "platform": ["release", "expected-sha"],
        "gate": ["release", "expected-sha", "canary-dir"],
        "stage": ["release", "expected-sha", "work-dir", "mode", "executing-script", "walltime-seconds"],
        "worker": ["work-dir", "stage", "rank"], "freeze": ["work-dir"], "aggregate": ["work-dir"],
        "runtime": ["stage", "mode"], "finish": ["work-dir", "exit-code", "stage-name"],
        "publish": ["out-dir"], "verify-copyback": ["out-dir"]}
    for name, args in fields.items():
        command = sub.add_parser(name)
        for arg in args:
            command.add_argument("--" + arg, required=True, type=int if arg in ("rank", "exit-code", "walltime-seconds") else str)
        if name == "stage":
            command.add_argument("--canary-dir")
        if name == "runtime":
            command.add_argument("--rank", type=int, default=0)
    command = sub.add_parser("submission")
    command.add_argument("action", choices=("plan", "submit"))
    command.add_argument("mode", choices=("canary", "full"))
    command.add_argument("canary_id", nargs="?")
    command.add_argument("--root")
    args = vars(parser.parse_args(argv))
    cmd = args.pop("command")
    if cmd == "platform":
        profile = verify(**args)["platform"]
        safe.require(os.environ.get("RAP_PLATFORM_SHA256") == safe.digest(profile), "allocation platform drift")
        print(" ".join([*profile["modules"], str(profile["cpus_per_task"]), str(profile["memory_gb"])]))
        return
    if cmd == "worker":
        safe.require(args["stage"] == "evaluate", "worker is evaluate-only")
        result = run_stage(args["work_dir"], args["stage"], args["rank"])
    elif cmd in ("freeze", "aggregate"):
        result = run_stage(args["work_dir"], cmd)
    elif cmd == "runtime":
        result = runtime(args["stage"], args["mode"], args["rank"])
    else:
        function = {"verify-source": verify_source, "finish": policy.finish,
                    "publish": safe.publish, "verify-copyback": safe.verify_copyback}.get(cmd, globals().get(cmd))
        try:
            result = function(**args)
        except Exception:
            if cmd == "stage" and Path(args["work_dir"]).exists():
                safe.failure(args["work_dir"], "stage")
            raise
    print(result if isinstance(result, (str, int)) else json.dumps(result, sort_keys=True))
    if cmd == "finish" and args["exit_code"] == 0 and result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
