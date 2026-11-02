"""Killarney platform adapter for the byte-identical role-aware experiment.

Login operations inspect bounded metadata only. Bulk integrity checks and all
scientific work remain in the original helper, inside a compute allocation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import socket
import subprocess
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.lifecycle_memory import role_aware_policy_release as base

safe, cv = base.safe, base.cv
JOB_REL = "docs/research/lifecycle_memory/phase 2 research/jobs/role_aware_killarney"
JOB = JOB_REL + "/job_killarney.sh"
HELPER = "research/lifecycle_memory/role_aware_killarney_release.py"
RUNTIME_MODULE = "research.lifecycle_memory.role_aware_killarney_runtime"
NEW_FILES = {HELPER, JOB, JOB_REL + "/worker.sh", JOB_REL + "/submit.sh",
             "scripts/run_role_aware_killarney.sh",
             "skills/killarney-slurm/scripts/discover.py",
             "research/lifecycle_memory/role_aware_killarney_transfer.py",
             "research/lifecycle_memory/role_aware_killarney_runtime.py",
             "research/lifecycle_memory/tests/test_role_aware_killarney_runtime.py",
             "research/lifecycle_memory/tests/test_role_aware_killarney.py"}


def validate_platform(value):
    required = {"cluster", "account", "modules", "preflight_confirmed"}
    optional = {"schema_version", "partition", "qos", "cpus_per_task", "memory_gb", "disk_margin_bytes", "gpu_type", "gpus"}
    safe.require(isinstance(value, dict) and required <= value.keys()
                 and value.keys() <= required | optional, "invalid platform fields")
    p = {"schema_version": 1, "gpu_type": "h100", "gpus": 4,
         "cpus_per_task": 6, "memory_gb": 480, "disk_margin_bytes": 20 * 1024**3,
         "partition": None, "qos": None, **value}
    safe.require(type(p["schema_version"]) is int and p["schema_version"] == 1
                 and p["cluster"] == "killarney", "Killarney platform required")
    safe.require(p["gpu_type"] == "h100" and type(p["gpus"]) is int and p["gpus"] == 4,
                 "exactly four H100 GPUs required")
    safe.require(type(p["preflight_confirmed"]) is bool, "preflight_confirmed must be boolean")
    for name in ("partition", "qos"):
        if p[name] == "":
            p[name] = None
    for name in ("account", "partition", "qos"):
        v = p[name]
        safe.require((name != "account" and v is None) or
                     (isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", v)),
                     f"invalid platform {name}")
    safe.require(isinstance(p["modules"], list) and len(p["modules"]) == 4, "four versioned modules required, including site Slurm")
    for name, value in zip(("StdEnv", "python", "apptainer", "slurm/killarney"), p["modules"]):
        safe.require(isinstance(value, str) and re.fullmatch(name + r"/[0-9][A-Za-z0-9_.-]{0,63}", value),
                     f"explicit module version required: {name}")
    for name, low, high in (("cpus_per_task", 6, 6), ("memory_gb", 1, 2048),
                            ("disk_margin_bytes", 1024**3, 800 * 1024**3)):
        safe.require(type(p[name]) is int and low <= p[name] <= high, f"invalid platform {name}")
    return p


def absolute_path(value):
    value = str(value)
    safe.require(Path(value).is_absolute() and ".." not in Path(value).parts and value != "/"
                 and not any(c in value for c in "\n\r\x00:,\\"), "unsafe absolute platform path")
    return value


def require_new(path):
    safe.require(not Path(path).exists() and not Path(path).is_symlink(), "destination must be new")


def source_files(root):
    root = Path(root)
    return sorted(set(base.source_files(root)) | NEW_FILES |
                  {str(p.relative_to(root)) for p in safe.tree_files(root / JOB_REL)})


def bundle(repo_root, out_dir):
    root, out = Path(repo_root).resolve(), Path(out_dir)
    require_new(out)
    names = source_files(root)
    for name in names:
        p = root / name
        safe.require(p.is_file() and not p.is_symlink() and root in p.resolve().parents,
                     f"missing/unsafe source: {name}")
    safe.require(sum((root / n).stat().st_size for n in names) <= cv.LIMIT, "bundle exceeds 32MiB")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".killarney-bundle-", dir=out.parent) as temp:
        staged = Path(temp) / "source"
        # Original config validation and the entire original closure are retained.
        base.bundle(root, staged)
        for name in names:
            dest = staged / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, dest)
        safe.atomic_json(staged / "source_manifest.json", {"schema_version": 1,
                         "files": [safe.entry(p, staged) for p in safe.tree_files(staged, ("source_manifest.json",))]})
        digest = safe.sha256(staged / "source_manifest.json")
        verify_source(staged, digest)
        require_new(out)
        staged.rename(out)
    return digest


def verify_source(source_root, source_sha):
    manifest = base.verify_source(source_root, source_sha)
    names = {r["path"] for r in manifest["files"]}
    safe.require(NEW_FILES <= names and set(base.source_files(source_root)) <= names,
                 "incomplete Killarney source bundle")
    return manifest


def prepare(source_root, source_sha, prior_release, prior_sha, train_root, eval_root, eval_sha,
            out_dir, model_root, assets_root, sif_path, platform_json):
    out = Path(out_dir)
    require_new(out)
    profile = validate_platform(cv.small_json(platform_json))
    model_root, assets_root, sif_path, train_root, eval_root = map(
        absolute_path, (model_root, assets_root, sif_path, train_root, eval_root))
    verify_source(source_root, source_sha)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".killarney-prepare-", dir=out.parent) as temp:
        staged = Path(temp) / "release"
        base.prepare(source_root, source_sha, prior_release, prior_sha, train_root, eval_root, eval_sha, staged)
        spec = cv.small_json(staged / "role_deployment.json")
        spec["model"]["root"], spec["assets_root"], spec["sif"]["path"] = model_root, assets_root, sif_path
        spec["platform"] = profile
        safe.atomic_json(staged / "role_deployment.json", spec)
        safe.atomic_json(staged / "platform.json", profile)
        safe.atomic_json(staged / "release_manifest.json", {"schema_version": 1,
                         "files": [safe.entry(p, staged) for p in safe.tree_files(staged, ("release_manifest.json",))]})
        digest = safe.sha256(staged / "release_manifest.json")
        verify(staged, digest)
        require_new(out)
        staged.rename(out)
    return digest


def verify(release, expected_sha):
    root = Path(release)
    manifest = cv.bounded_verify(root, "release_manifest.json", expected_sha)
    spec = cv.small_json(root / "role_deployment.json")
    safe.require(spec["schema_version"] == 1 and spec["prior_sha256"] == base.PRIOR_SHA, "release identity drift")
    source = cv.small_json(root / "source_manifest.json")
    safe.require(safe.sha256(root / "source_manifest.json") == spec["source_sha256"], "source pin drift")
    safe.verify_inventory(root, source["files"], exact=False)
    names = {r["path"] for r in source["files"]}
    safe.require(NEW_FILES <= names and set(base.source_files(root)) <= names, "incomplete Killarney source bundle")
    extras = {"source_manifest.json", "role_deployment.json", "training_inventory.json",
              "model_inventory.json", "evaluation_manifest.json", "platform.json"}
    safe.require({r["path"] for r in manifest["files"]} == names | extras, "release inventory drift")
    for name, digest in (("training_inventory.json", base.TRAIN_INVENTORY_SHA),
                         ("model_inventory.json", base.MODEL_SHA), ("evaluation_manifest.json", base.EVAL_SHA)):
        safe.require(safe.sha256(root / name) == digest, f"pin drift: {name}")
    safe.require(spec["sif"]["sha256"] == safe.SIF_SHA256
                 and spec["training_manifest_sha256"] == base.TRAIN_SHA
                 and spec["evaluation_manifest_sha256"] == base.EVAL_SHA, "data/runtime pin drift")
    safe.require(spec["config_path"] == base.CONFIG and spec["model"]["inventory"] == "model_inventory.json",
                 "science config/inventory drift")
    profile = validate_platform(cv.small_json(root / "platform.json"))
    safe.require(spec["platform"] == profile == cv.small_json(root / "platform.json"), "platform drift")
    for path in (spec["model"]["root"], spec["assets_root"], spec["sif"]["path"], spec["train_root"], spec["eval_root"]):
        absolute_path(path)
    return spec


def identity(release, sha):
    spec = verify(release, sha)
    return {"release_sha256": sha, "source_sha256": spec["source_sha256"],
            "script_sha256": safe.sha256(Path(release) / JOB), "sif_sha256": safe.SIF_SHA256,
            "training_manifest_sha256": base.TRAIN_SHA, "evaluation_manifest_sha256": base.EVAL_SHA,
            "model_inventory_sha256": base.MODEL_SHA, "prior_sha256": base.PRIOR_SHA,
            "cluster": "killarney", "platform_sha256": safe.digest(spec["platform"])}


def gate(release, expected_sha, canary_dir):
    return _engine.gate(release, expected_sha, canary_dir)


def disk_requirement(release, spec, mode):
    root = Path(release)
    total = 2 * sum(r["bytes"] for r in safe.inventory(cv.small_json(root / "release_manifest.json")))
    total += 2 * (root / "release_manifest.json").stat().st_size
    for name in ("training_inventory.json", "evaluation_manifest.json", "model_inventory.json"):
        for row in safe.inventory(cv.small_json(root / name)):
            if name == "training_inventory.json" and row.get("source", "package") != "assets" and row["path"] not in (
                    "split_manifest.json", "public/requests.jsonl", "labels/train.jsonl"):
                continue
            if name == "evaluation_manifest.json" and mode == "canary" and row["path"].startswith("evaluator/"):
                continue
            total += row["bytes"]
    total += Path(spec["sif"]["path"]).stat().st_size
    return total + spec["platform"]["disk_margin_bytes"]


def stage(release, expected_sha, work_dir, mode, executing_script, walltime_seconds, canary_dir=None):
    safe.require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_TMPDIR"), "compute allocation required")
    safe.require(os.environ.get("SLURM_CLUSTER_NAME", "").lower().startswith("killarney"), "Killarney allocation required")
    safe.require(os.environ.get("SLURM_JOB_NUM_NODES") == "1", "single-node allocation required")
    spec = verify(release, expected_sha)
    safe.require(os.environ.get("RAP_PLATFORM_SHA256") == safe.digest(spec["platform"]), "allocation platform drift")
    safe.require(mode in ("canary", "full"), "invalid mode")
    safe.require(safe.sha256(executing_script) == safe.sha256(Path(release) / JOB), "executing script drift")
    local = Path(os.environ["SLURM_TMPDIR"]).resolve()
    safe.require(local in Path(work_dir).resolve().parents, "work must be node-local")
    safe.require(shutil.disk_usage(local).free >= disk_requirement(release, spec, mode),
                 "insufficient node-local disk: estimated copies plus margin")
    return _engine.stage(release, expected_sha, work_dir, mode, executing_script, walltime_seconds, canary_dir)


def container_command(work, ctx, operation, rank=None):
    profile = validate_platform(ctx["spec"]["platform"])
    safe.require(ctx["identity"].get("cluster") == "killarney"
                 and ctx["identity"].get("platform_sha256") == safe.digest(profile), "runtime platform drift")
    command = base.container_command(work, ctx, operation, rank)
    safe.require(command.count("OMP_NUM_THREADS=12") == 1, "upstream OMP contract changed")
    command[command.index("OMP_NUM_THREADS=12")] = "OMP_NUM_THREADS=6"
    if operation in ("evaluate", "aggregate"):
        module = "research.lifecycle_memory.role_aware_policy_release"
        safe.require(command.count(module) == 1, "upstream runtime module changed")
        command[command.index(module)] = RUNTIME_MODULE
    return command


# Private instance: upstream functions resolve these platform hooks without
# monkey-patching the imported TamIA module or duplicating scientific logic.
_module_spec = importlib.util.spec_from_file_location("_role_aware_killarney_engine", base.__file__)
_engine = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_engine)
_engine.verify, _engine.identity, _engine.container_command = verify, identity, container_command


def preflight(profile):
    safe.require(os.environ.get("KILLARNEY_PREFLIGHT_CONFIRMED") == "1", "KILLARNEY_PREFLIGHT_CONFIRMED=1 required")
    safe.require(profile["preflight_confirmed"] is True, "frozen platform preflight confirmation required")
    known_host = socket.gethostname().split(".")[0].lower().startswith("killarney")
    if not known_host:
        result = subprocess.run(["scontrol", "show", "config"], capture_output=True, text=True, timeout=10)
        safe.require(result.returncode == 0 and len(result.stdout) <= 256 * 1024
                     and re.search(r"(?mi)^\s*ClusterName\s*=\s*killarney[A-Za-z0-9_.-]*\s*$", result.stdout),
                     "Killarney scheduler identity required")
    cluster = os.environ.get("SLURM_CLUSTER_NAME")
    safe.require(cluster is None or cluster.lower().startswith("killarney"), "conflicting scheduler cluster")
    if os.environ.get("KILLARNEY_MANUAL_CONFIG_VALIDATED") == "1":
        safe.require(os.environ.get("KILLARNEY_MANUAL_CONFIG_SHA256") == safe.digest(profile),
                     "manual validation must attest exact frozen platform hash")
        return "operator-manual"
    result = subprocess.run(["sacctmgr", "-nP", "show", "assoc", "where", f"user={pwd.getpwuid(os.geteuid()).pw_name}",
                             "format=Cluster,Account,Partition,QOS"], capture_output=True, text=True, timeout=10)
    safe.require(result.returncode == 0 and len(result.stdout) <= 65536, "association query failed; manual validation required")
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 4:
            cluster, account, partition, qos = fields[:4]
            if (cluster.lower().startswith("killarney") and account == profile["account"]
                    and (not partition or partition == profile["partition"])
                    and (profile["qos"] is None or profile["qos"] in qos.split(","))):
                return "association"
    raise ValueError("no matching Killarney association; operator manual validation required")


def submission(action, mode, canary_id=None, root=None):
    safe.require(action in ("plan", "submit") and mode in ("canary", "full"), "invalid submission action/mode")
    root = Path(absolute_path(root or os.environ.get("RAP_ROOT", "")))
    source_sha = safe.checked_hash((root / "source.sha256").read_text().strip())
    sha = safe.checked_hash((root / "release.sha256").read_text().strip())
    release = root / "releases" / source_sha
    spec = verify(release, sha)
    safe.require(spec["source_sha256"] == source_sha, "source pointer drift")
    # Require the actual entrypoint/helper to be the release's verified bytes.
    safe.require(safe.sha256(__file__) == safe.sha256(release / HELPER), "submission helper drift")
    profile = spec["platform"]
    safe.require((root / "account").read_text().strip() == profile["account"], "account pointer drift")
    safe.require(validate_platform(cv.small_json(root / "platform.json")) == profile,
                 "deployment root platform differs from frozen release; use a fresh root")
    canary = None
    if mode == "full":
        safe.require(isinstance(canary_id, str) and re.fullmatch(r"[0-9]+", canary_id), "full requires CANARY_ID")
        canary = root / "runs/canary" / canary_id
        recorded = cv.small_json(root / "submissions/canary/receipt.json")
        safe.require(recorded["jobid"] == canary_id and recorded["release_sha256"] == sha, "unrecorded canary")
        seconds = gate(release, sha, canary)
    else:
        safe.require(canary_id is None, "canary ID only valid for full")
        seconds = 1200
    walltime = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    command = ["sbatch", "--parsable", f"--account={profile['account']}", "--nodes=1",
               "--gpus-per-node=h100:4", "--ntasks=4", "--ntasks-per-node=4",
               f"--cpus-per-task={profile['cpus_per_task']}", f"--mem={profile['memory_gb']}G",
               f"--time={walltime}", "--export=ALL", "--no-requeue", "--signal=B:USR1@300",
               f"--output={root}/logs/slurm-%j.out", f"--error={root}/logs/slurm-%j.err"]
    for name in ("partition", "qos"):
        if profile[name] is not None:
            command.append(f"--{name}={profile[name]}")
    command.append(str(release / JOB))
    ident = {"release_sha256": sha, "mode": mode, "canary_id": canary_id,
             "platform_sha256": safe.digest(profile), "walltime_seconds": seconds, "command": command}
    if action == "plan":
        return {"action": "plan", "submitted": False, **ident}
    validation = preflight(profile)
    for directory in (root / "submissions", root / "logs"):
        safe.require(not directory.is_symlink(), "unsafe submission directory")
        directory.mkdir(exist_ok=True)
    lock = root / "submissions" / mode
    try:
        lock.mkdir()
    except FileExistsError:
        safe.require(not lock.is_symlink(), "unsafe submission lock")
        safe.require((lock / "receipt.json").is_file(), "ambiguous submission; inspect permanent lock before manual recovery")
        receipt = cv.small_json(lock / "receipt.json")
        safe.require(receipt["identity"] == ident and re.fullmatch(r"[0-9]+", receipt["jobid"]), "submission identity drift")
        return {"already_submitted": True, **receipt}
    safe.atomic_json(lock / "intent.json", {"identity": ident, "validation": validation, "status": "submitting"})
    # Drop inherited SBATCH_* overrides; only the frozen profile sets resources.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    env.pop("RAP_CANARY_DIR", None)
    env.update(RAP_ROOT=str(root), RAP_RELEASE=str(release), RAP_RELEASE_SHA256=sha, RAP_MODE=mode,
               RAP_WALLTIME_SECONDS=str(seconds), RAP_PLATFORM_SHA256=safe.digest(profile))
    if canary is not None:
        env["RAP_CANARY_DIR"] = str(canary)
    with (lock / "sbatch.stdout").open("w") as stdout, (lock / "sbatch.stderr").open("w") as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=60)
    response = (lock / "sbatch.stdout").read_text().strip()
    safe.require(result.returncode == 0 and re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", response),
                 "failed or ambiguous sbatch; permanent lock retained, no automatic retry")
    receipt = {"identity": ident, "jobid": response.split(";")[0], "release_sha256": sha, "status": "submitted"}
    safe.atomic_json(lock / "receipt.json", receipt, exclusive=True)
    with (lock / "jobid.tmp").open("x") as handle:
        handle.write(receipt["jobid"] + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    (lock / "jobid.tmp").replace(lock / "jobid")
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    fields = {
        "bundle": ["repo-root", "out-dir"], "verify-source": ["source-root", "source-sha"],
        "prepare": ["source-root", "source-sha", "prior-release", "prior-sha", "train-root", "eval-root", "eval-sha",
                    "out-dir", "model-root", "assets-root", "sif-path", "platform-json"],
        "verify": ["release", "expected-sha"], "platform": ["release", "expected-sha"],
        "gate": ["release", "expected-sha", "canary-dir"],
        "stage": ["release", "expected-sha", "work-dir", "mode", "executing-script", "walltime-seconds"],
        "worker": ["work-dir", "stage", "rank"], "freeze": ["work-dir"], "aggregate": ["work-dir"],
        "finish": ["work-dir", "exit-code", "stage-name"], "publish": ["out-dir"], "verify-copyback": ["out-dir"]}
    for cmd, names in fields.items():
        parser = sub.add_parser(cmd)
        for name in names:
            parser.add_argument("--" + name, required=True, type=int if name in ("rank", "exit-code", "walltime-seconds") else str)
        if cmd == "stage":
            parser.add_argument("--canary-dir")
    parser = sub.add_parser("submission")
    parser.add_argument("action", choices=("plan", "submit"))
    parser.add_argument("mode", choices=("canary", "full"))
    parser.add_argument("canary_id", nargs="?")
    parser.add_argument("--root")
    args = vars(p.parse_args(argv))
    cmd = args.pop("command")
    if cmd == "platform":
        profile = verify(**args)["platform"]
        safe.require(os.environ.get("RAP_PLATFORM_SHA256") == safe.digest(profile), "allocation platform drift")
        print(" ".join([*profile["modules"], str(profile["cpus_per_task"]), str(profile["memory_gb"])]))
        return
    if cmd == "worker":
        result = _engine.run_stage(args["work_dir"], args["stage"], args["rank"])
    elif cmd in ("freeze", "aggregate"):
        result = _engine.run_stage(args["work_dir"], cmd)
    else:
        fun = {"verify-source": verify_source, "verify-copyback": safe.verify_copyback,
               "publish": safe.publish, "finish": base.finish}.get(cmd, globals().get(cmd))
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
