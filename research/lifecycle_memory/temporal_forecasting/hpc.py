"""Portable launch plans; staging, runtime verification and workers belong to CLI.

Deployment contract: ``deployment_path`` is the absolute JSON path passed to the
coordinator. ``runtime`` supplies mode (native/venv/apptainer), python, and sha256
of that interpreter (native/venv) or image (apptainer). Apptainer additionally
requires image and may supply binds [{source, target, mode: ro|rw}]. The parent
must verify the full pinned environment, release and deployment, not just these
launcher-level file hashes. runtime.versions must pin torch and transformers
(the coordinator verifies installed versions). Optional resources override
profile defaults. deployment.walltime_seconds must equal requested HH:MM:SS;
only pilot/exploration/confirmation, optionally suffixed -canary, are accepted.
The full deployment dictionary is pinned with canonical JSON SHA-256, checked
using only Python's standard library (-S) before GPU preflight and execution.
TF_WALLTIME_SECONDS exposes the baked allocation budget to the coordinator;
Apptainer explicitly overrides TF_WORKER_COUNT, TF_GPUS_PER_NODE and TF_NODES
with the plan's allocation rather than trusting image environment defaults.

Apptainer uses identical host/container paths for release, deployment, run_dir,
and deployment.data_root (required absolute path). deployment.models is a
nonempty mapping of canonical protocol model keys to {snapshot, inventory},
both absolute paths mounted read-only, as are optional freeze_path,
pilot_receipt and canary_receipt files, plus the pilot receipt's sibling
analysis.json, the recognition canary's sibling technical_checks.json, and the
optional followup_reference_check sidecar for
metadata provenance checks. No separate evaluator key is used: coordinator analysis
reads it from data_root; worker public-input isolation belongs to execution.py.
Snapshot links into an external HF blob cache require an explicit read-only
runtime.binds entry for that cache; the adapter does not discover link targets.
Additional runtime.binds have absolute
source/target paths; target defaults to source, mode defaults to ro. Remapped
targets must agree with the parent's deployment paths. Sources must already be
readable on the compute node; no implicit download or mount discovery occurs.
deployment.local_scratch explicitly selects writable staging storage, otherwise
SLURM_TMPDIR must be set at execution. The effective path is exported as both
TF_LOCAL_SCRATCH and SLURM_TMPDIR and mounted read-write at the same container
path. local-cloud must supply local_scratch. Input roots must not overlap writable
run/scratch mounts. No input/evaluator contents are read by this adapter.

Multinode is deliberately rejected until the parent's staging/worker/collection
contract is implemented and tested. Profiles are dated proposals, not live site
eligibility checks. No code here downloads, stages, or contacts remote hosts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess

from .protocol import MODELS, STAGES


PROFILE_DIR = (Path(__file__).resolve().parents[3] / "docs/research/lifecycle_memory"
               / "phase 2 research/jobs/temporal_forecasting/profiles")
CLI = "research.lifecycle_memory.temporal_forecasting.cli"
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value, name):
    _require(isinstance(value, str) and bool(value), f"{name} must be nonempty text")
    _require(not any(ord(c) < 32 or ord(c) == 127 for c in value),
             f"{name} contains control characters")
    return value


def _token(value, name):
    _require(_TOKEN.fullmatch(_text(value, name)), f"invalid {name}")
    return value


def _integer(value, name, maximum=65536):
    _require(type(value) is int and 0 < value <= maximum, f"invalid {name}")
    return value


def _path(value, name):
    _require(isinstance(value, (str, os.PathLike)), f"{name} must be a path")
    value = _text(os.fspath(value), name)
    _require(Path(value).is_absolute() and Path(value) != Path("/") and ".." not in Path(value).parts,
             f"{name} must be an absolute path without traversal")
    return value


def _clean(value):
    if isinstance(value, str):
        _text(value, "configuration value")
    elif isinstance(value, dict):
        for key, item in value.items():
            _text(key, "configuration key")
            _clean(item)
    elif isinstance(value, list):
        for item in value:
            _clean(item)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def load_profile(path_or_name) -> dict:
    """Load a named bundled profile or an explicit JSON path, without side effects."""
    candidate = Path(path_or_name)
    if len(candidate.parts) == 1 and not candidate.exists():
        candidate = PROFILE_DIR / (candidate.stem + ".json")
    with candidate.open(encoding="utf-8") as handle:
        profile = json.load(handle)
    _require(isinstance(profile, dict), "profile must be an object")
    _clean(profile)
    _require(profile.get("name") in {"tamia", "killarney", "fir", "generic-slurm", "local-cloud"},
             "unknown profile name")
    expected = "command" if profile["name"] == "local-cloud" else "slurm"
    _require(profile.get("scheduler") == expected, "profile scheduler mismatch")
    return profile


def _runtime(runtime, release, deployment_path, run_dir, input_roots, walltime_seconds, nodes, gpus):
    _require(isinstance(runtime, dict), "deployment.runtime is required")
    versions = runtime.get("versions")
    _require(isinstance(versions, dict) and {"torch", "transformers"} <= versions.keys(),
             "runtime.versions must pin torch and transformers")
    for package, version in versions.items():
        _token(package, "runtime package")
        _require(isinstance(version, str) and re.fullmatch(r"[0-9][A-Za-z0-9_.+!-]*", version),
                 "runtime versions must be exact nonempty version strings")
    mode = runtime.get("mode")
    _require(mode in {"native", "venv", "apptainer"}, "unsupported runtime mode")
    sha = runtime.get("sha256", "")
    _require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha),
             "runtime.sha256 must pin the supplied interpreter/image")
    prefix = []
    if mode == "apptainer":
        artifact = _path(runtime.get("image", ""), "runtime.image")
        python = _text(runtime.get("python", "python"), "runtime.python")
        _require(python == "python" or Path(python).is_absolute(), "invalid container python")
        prefix = [_text(runtime.get("apptainer", "apptainer"), "apptainer"), "exec", "--nv"]
        _require(prefix[0] == "apptainer" or Path(prefix[0]).is_absolute(), "invalid apptainer executable")
        binds = [{"source": release, "target": release, "mode": "ro"},
                 {"source": deployment_path, "target": deployment_path, "mode": "ro"},
                 {"source": run_dir, "target": run_dir, "mode": "rw"}]
        binds += [{"source": path, "target": path, "mode": "ro"} for path in input_roots]
        extra = runtime.get("binds", [])
        _require(isinstance(extra, list), "runtime.binds must be a list")
        for bind in binds + extra:
            source = _path(bind["source"], "bind source")
            target = _path(bind.get("target", source), "bind target")
            access = bind.get("mode", "ro")
            _require(access in {"ro", "rw"} and not any(c in source + target for c in ":,"),
                     "invalid bind")
            if access == "rw":
                for protected in [release, deployment_path, *input_roots]:
                    for writable in (source, target):
                        _require(not (Path(writable).is_relative_to(Path(protected)) or
                                      Path(protected).is_relative_to(Path(writable))),
                                 "writable bind overlaps a read-only input")
            prefix += ["--bind", f"{source}:{target}:{access}"]
        prefix += ["--env", f"PYTHONPATH={release}", "--env", "PYTHONDONTWRITEBYTECODE=1",
                   "--env", "HF_HUB_OFFLINE=1", "--env", "HF_DATASETS_OFFLINE=1",
                   "--env", "TRANSFORMERS_OFFLINE=1", "--env", "WANDB_MODE=offline",
                   "--env", f"TF_WALLTIME_SECONDS={walltime_seconds}",
                   "--env", f"TF_WORKER_COUNT={nodes * gpus}",
                   "--env", f"TF_GPUS_PER_NODE={gpus}", "--env", f"TF_NODES={nodes}", artifact]
    else:
        python = _path(runtime.get("python", ""), "runtime.python")
        artifact = python
    # sha256sum consumes stdin: no filename interpretation or checksum-line injection.
    check = f'test "$(sha256sum < {shlex.quote(artifact)} | cut -d " " -f 1)" = {sha}'
    return prefix + [python, "-B"], check


def build_plan(profile, deployment: dict, stage: str, run_dir: Path,
               release_root: Path, walltime: str, nodes: int = 1,
               gpus_per_node: int | None = None) -> dict:
    """Render an inert plan. Runtime and data paths may exist only on the target."""
    if not isinstance(profile, dict):
        profile = load_profile(profile)
    profile = json.loads(_canonical(profile))
    _require(isinstance(deployment, dict), "deployment must be an object")
    _clean(profile)
    _clean(deployment)
    name = profile.get("name")
    _require(name in {"tamia", "killarney", "fir", "generic-slurm", "local-cloud"}, "unknown profile")
    scheduler = "command" if name == "local-cloud" else "slurm"
    _require(profile.get("scheduler") == scheduler, "profile scheduler mismatch")
    _require(_integer(nodes, "nodes") == 1,
             "multinode is not implemented: requires tested per-node staging and collection")
    _require(isinstance(walltime, str) and re.fullmatch(r"\d{2}:\d{2}:\d{2}", walltime),
             "walltime must be HH:MM:SS")
    hours, minutes, seconds = map(int, walltime.split(":"))
    walltime_seconds = hours * 3600 + minutes * 60 + seconds
    _require(minutes < 60 and seconds < 60 and 300 <= walltime_seconds <= 23 * 3600,
             "walltime must be 00:05:00..23:00:00 (execution contract)")
    _require(type(deployment.get("walltime_seconds")) is int
             and deployment["walltime_seconds"] == walltime_seconds,
             "deployment.walltime_seconds must match requested walltime")
    stage = _token(stage, "stage")
    stages = (*STAGES, "current", "recognition")
    _require(stage in {*stages, *(name + "-canary" for name in stages)}, "invalid execution stage")
    if name == "fir" and not stage.endswith("-canary"):
        _require(walltime_seconds >= 3600, "Fir non-canary jobs require at least 01:00:00")
    release = _path(release_root, "release_root")
    run = _path(run_dir, "run_dir")
    dep = _path(deployment.get("deployment_path", ""), "deployment_path")
    _require(not Path(run).is_relative_to(Path(release)), "run_dir must be outside release")
    input_roots = [_path(deployment.get("data_root"), "data_root")]
    models = deployment.get("models")
    _require(isinstance(models, dict) and bool(models), "deployment.models must be a nonempty mapping")
    for model, spec in models.items():
        _require(model in MODELS and isinstance(spec, dict), "invalid canonical model specification")
        input_roots += [_path(spec.get(field), f"models.{model}.{field}") for field in ("snapshot", "inventory")]
    for field in ("freeze_path", "pilot_receipt", "canary_receipt", "followup_reference_check"):
        value = deployment.get(field)
        if value is not None:
            input_roots.append(_path(value, field))
            if field == "pilot_receipt":
                input_roots.append(str(Path(value).parent / "analysis.json"))
            elif field == "canary_receipt" and stage == "recognition":
                input_roots.append(str(Path(value).parent / "technical_checks.json"))
    input_roots = list(dict.fromkeys(input_roots))
    scratch = deployment.get("local_scratch")
    if scratch is not None:
        scratch = _path(scratch, "local_scratch")
    _require(scheduler != "command" or scratch is not None,
             "local-cloud requires deployment.local_scratch")
    for source in [release, dep, *input_roots]:
        for writable in [run] + ([scratch] if scratch else []):
            _require(not (Path(writable).is_relative_to(Path(source)) or Path(source).is_relative_to(Path(writable))),
                     "writable run/scratch path overlaps a read-only input")
    defaults = profile.get("resources", {})
    overrides = deployment.get("resources", {})
    _require(isinstance(defaults, dict) and isinstance(overrides, dict), "resources must be objects")
    allowed = {"gpus_per_node", "cpus_per_task", "memory_gb", "gpu_type", "gpu_memory_gb",
               "node_cpu_limit", "node_memory_gb"}
    _require(not (set(defaults) | set(overrides)) - allowed, "unknown resource field")
    resources = {**defaults, **overrides}
    if gpus_per_node is not None:
        resources["gpus_per_node"] = gpus_per_node
    gpus = _integer(resources.get("gpus_per_node"), "gpus_per_node", 64)
    if name == "tamia":
        _require(gpus == 4, "TamIA requires all four H100 GPUs")
    if name == "fir":
        _require(1 <= gpus <= 4, "Fir supports one to four full H100 GPUs")
    if name == "killarney":
        _require(gpus in {4, 8}, "Killarney supports four/eight H100 workers")
        if "memory_gb" not in overrides:
            resources["memory_gb"] = 120 * gpus
    _require(resources.get("gpu_type") == "h100", "only H100 is supported; other GPUs unsupported")
    _require(_integer(resources.get("gpu_memory_gb"), "gpu_memory_gb", 1024) >= 80,
             "at least 80 GB GPU memory required")
    cpus = _integer(resources.get("cpus_per_task"), "cpus_per_task", 1024)
    memory = _integer(resources.get("memory_gb"), "memory_gb", 1048576)
    cpu_limit = _integer(resources.get("node_cpu_limit"), "node_cpu_limit")
    mem_limit = _integer(resources.get("node_memory_gb"), "node_memory_gb", 1048576)
    if name in {"tamia", "killarney"}:
        _require(cpu_limit <= 48, "H100 node CPU limit exceeds dated profile capacity")
        _require(mem_limit <= (512 if name == "tamia" else 2048), "node memory limit exceeds profile")
    if name == "fir":
        _require(cpu_limit <= 48, "Fir node CPU limit exceeds 48 CPUs")
        _require(mem_limit <= 1125, "Fir node memory limit exceeds 1125G")
    _require(cpus * gpus <= cpu_limit, "CPU request exceeds per-node capacity")
    _require(memory <= mem_limit and memory >= cpus * gpus, "host memory request outside sane bounds")
    args = []
    if scheduler == "slurm":
        account = _token(deployment.get("account"), "account")
        gpu_flag = "--gpus" if name == "tamia" and nodes == 1 else "--gpus-per-node"
        args = [f"--account={account}", f"--nodes={nodes}", f"{gpu_flag}=h100:{gpus}",
                f"--ntasks={nodes * gpus}", f"--ntasks-per-node={gpus}",
                f"--cpus-per-task={cpus}", f"--mem={memory}G", f"--time={walltime}",
                "--export=ALL", "--no-requeue"]
        for field in ("partition", "qos", "job_name"):
            value = deployment.get(field)
            if value is not None:
                value = _token(value, field)
                _require(not (name == "tamia" and field == "partition" and value != "default"),
                         "TamIA concrete partitions are forbidden")
                args.append(f"--{field.replace('_', '-')}={value}")
    else:
        _require(not any(deployment.get(k) is not None for k in ("account", "partition", "qos")),
                 "local-cloud does not accept Slurm configuration")
    deployment_sha256 = _digest(deployment)
    python, check = _runtime(deployment.get("runtime"), release, dep, run, input_roots, walltime_seconds, nodes, gpus)
    command = python + ["-m", CLI, "execute", "--release", release, "--deployment", dep,
                        "--stage", stage, "--run-dir", run]
    # No torch, package metadata, site hooks or deployment contents in diagnostics.
    deployment_check = (
        "import hashlib,json,sys\n"
        "def pairs(items):\n"
        " result={}\n"
        " for key,value in items:\n"
        "  if key in result: raise SystemExit('duplicate deployment JSON key')\n"
        "  result[key]=value\n"
        " return result\n"
        "with open(sys.argv[1],encoding='utf-8') as handle:\n"
        " data=json.load(handle,object_pairs_hook=pairs)\n"
        "try:\n"
        " canonical=json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)\n"
        "except ValueError: raise SystemExit('invalid deployment JSON value')\n"
        "if hashlib.sha256(canonical.encode()).hexdigest()!=sys.argv[2]:\n"
        " raise SystemExit('deployment JSON differs from the launch plan')\n"
    )
    verify_deployment = shlex.join(python + ["-S", "-c", deployment_check, dep, deployment_sha256])
    # Query inside the actual runtime. Never remap CUDA_VISIBLE_DEVICES or infer
    # GPU IDs from local rank; workers must keep the scheduler's binding intact.
    gpu_check = ("import torch\n"
                 f"n={gpus}\n"
                 "if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')\n"
                 "if torch.cuda.device_count() < n: raise RuntimeError('insufficient visible GPUs')\n"
                 "p=[torch.cuda.get_device_properties(i) for i in range(n)]\n"
                 "if not all('H100' in x.name.upper() and 'MIG' not in x.name.upper() "
                 f"and x.total_memory >= {resources['gpu_memory_gb'] * 1000000000} for x in p):\n"
                 " raise RuntimeError('requires full H100 with configured memory, at least 80GB')\n")
    lines = ["#!/bin/bash"] + [f"#SBATCH {arg}" for arg in args]
    lines += ["set -Eeuo pipefail", "umask 077", f"export PYTHONPATH={shlex.quote(release)}",
              "export PYTHONDONTWRITEBYTECODE=1", "export HF_HUB_OFFLINE=1",
              "export HF_DATASETS_OFFLINE=1", "export TRANSFORMERS_OFFLINE=1",
              "export WANDB_MODE=offline", "export TOKENIZERS_PARALLELISM=false",
              f"export OMP_NUM_THREADS={cpus}", f"export TF_WORKER_COUNT={nodes * gpus}",
              f"export TF_GPUS_PER_NODE={gpus}", f"export TF_NODES={nodes}",
              f"export TF_WALLTIME_SECONDS={walltime_seconds}"]
    lines += ([f"export TF_LOCAL_SCRATCH={shlex.quote(scratch)}"] if scratch else
              [': "${SLURM_TMPDIR:?set SLURM_TMPDIR or deployment.local_scratch}"',
               'export TF_LOCAL_SCRATCH="$SLURM_TMPDIR"'])
    # For environment-supplied scratch, validate shell-safe bind syntax on-node.
    lines += ['TF_LOCAL_SCRATCH="${TF_LOCAL_SCRATCH%/}"',
              '[[ "$TF_LOCAL_SCRATCH" == /* && "$TF_LOCAL_SCRATCH" != / && "$TF_LOCAL_SCRATCH" != *//* ]]',
              '[[ "$TF_LOCAL_SCRATCH" != *[[:cntrl:]]* && "$TF_LOCAL_SCRATCH" != *:* && "$TF_LOCAL_SCRATCH" != *,* ]]',
              '[[ "/${TF_LOCAL_SCRATCH#/}/" != */../* && "/${TF_LOCAL_SCRATCH#/}/" != */./* ]]',
              'export SLURM_TMPDIR="$TF_LOCAL_SCRATCH"']
    for source in [release, dep, *input_roots]:
        lines += [f"test -r {shlex.quote(source)}",
                  f'[[ "$TF_LOCAL_SCRATCH" != {shlex.quote(source)} && "$TF_LOCAL_SCRATCH" != {shlex.quote(source + "/")}* ]]',
                  f'[[ {shlex.quote(source)} != "$TF_LOCAL_SCRATCH"/* ]]']
    lines += ['mkdir -p -- "$TF_LOCAL_SCRATCH"', 'test -w "$TF_LOCAL_SCRATCH"']
    if deployment["runtime"]["mode"] == "apptainer":
        lines += ['export APPTAINER_BINDPATH="${TF_LOCAL_SCRATCH}:${TF_LOCAL_SCRATCH}:rw"',
                  'export APPTAINERENV_SLURM_TMPDIR="$TF_LOCAL_SCRATCH"',
                  'export APPTAINERENV_TF_LOCAL_SCRATCH="$TF_LOCAL_SCRATCH"',
                  'if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then',
                  '  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"',
                  'else', '  unset APPTAINERENV_CUDA_VISIBLE_DEVICES', 'fi']
        for bind in deployment["runtime"].get("binds", []):
            lines += [f"test -r {shlex.quote(bind['source'])}"]
            if bind.get("mode", "ro") == "rw":
                lines += [f"test -w {shlex.quote(bind['source'])}"]
    lines += [f"mkdir -p -- {shlex.quote(run)}", f"cd -- {shlex.quote(release)}", check,
              verify_deployment, shlex.join(python + ["-c", gpu_check]),
              verify_deployment, "exec " + shlex.join(command)]
    plan = {"schema_version": 1, "profile": profile, "scheduler": scheduler,
            "sbatch_args": args, "script_text": "\n".join(lines) + "\n", "command": command,
            "worker_count": nodes * gpus, "nodes": nodes, "gpus_per_node": gpus,
            "resources": resources, "stage": stage, "release_root": release,
            "deployment_path": dep, "run_dir": run, "walltime": walltime,
            "deployment_sha256": deployment_sha256,
            "deployment_digest_format": "canonical-json-sha256-v1",
            "walltime_seconds": walltime_seconds,
            "local_scratch": scratch, "input_roots": input_roots,
            "runtime": deployment["runtime"], "multinode_supported": False}
    plan["plan_sha256"] = _digest(plan)
    return json.loads(_canonical(plan))


def _no_links(path):
    path = Path(os.path.abspath(path))
    for item in (path, *path.parents):
        _require(not item.is_symlink(), f"symlink refused: {item}")
    return path


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive(path, text, mode=0o600):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_dir(path.parent)


def _validate_plan(plan):
    _require(isinstance(plan, dict) and plan.get("schema_version") == 1, "invalid plan schema")
    _require(plan.get("plan_sha256") == _digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
             "plan identity mismatch")
    _require(plan.get("scheduler") in {"slurm", "command"}, "invalid scheduler")
    _require(isinstance(plan.get("sbatch_args"), list), "invalid sbatch args")
    for arg in plan["sbatch_args"]:
        _text(arg, "sbatch argument")
    _require(isinstance(plan.get("script_text"), str), "invalid script")


def write_plan(plan, out_dir):
    """Exclusively create a private directory; plan.json is the completion marker.

    Existing paths (including symlinks) are never overwritten. On interruption,
    incomplete directories remain for inspection and cannot be submitted.
    """
    plan = json.loads(_canonical(plan))
    _validate_plan(plan)
    out = _no_links(out_dir)
    out.mkdir(mode=0o700)
    _sync_dir(out.parent)
    _exclusive(out / "job.sh", plan["script_text"], 0o700)
    _exclusive(out / "plan.json", _canonical(plan) + "\n")
    return {"plan_dir": str(out), "plan_sha256": plan["plan_sha256"]}


def _read(path):
    _no_links(path)
    _require(stat.S_ISREG(path.stat().st_mode), f"not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def submit_plan(plan_dir, confirm=False) -> dict:
    """Only literal True authorizes sbatch; ambiguous intent permanently blocks retry.

    Receipts are idempotent for the same plan directory and identity. An operator
    must reconcile a pending intent with Slurm before any manual recovery. This
    function never automatically retries or executes local-cloud commands.
    """
    root = _no_links(plan_dir)
    plan = _read(root / "plan.json")
    _validate_plan(plan)
    script = _no_links(root / "job.sh")
    _require(script.is_file() and script.read_text(encoding="utf-8") == plan["script_text"],
             "job.sh differs from recorded plan")
    identity = plan["plan_sha256"]
    receipt_path = _no_links(root / "receipt.json")
    if receipt_path.exists():
        receipt = _read(receipt_path)
        _require(receipt.get("plan_sha256") == identity and receipt.get("status") == "submitted"
                 and re.fullmatch(r"[1-9][0-9]*", str(receipt.get("jobid", ""))),
                 "receipt identity/status mismatch")
        return {**receipt, "already_submitted": True}
    intent = _no_links(root / "intent.json")
    _require(not intent.exists(), "pending submission intent: ambiguous acceptance; manual reconciliation required")
    command = (["sbatch", "--parsable", *plan["sbatch_args"],
                f"--output={root}/slurm-%j.out", f"--error={root}/slurm-%j.err", str(script)]
               if plan["scheduler"] == "slurm" else ["bash", str(script)])
    if confirm is not True:
        return {"submitted": False, "status": "planned", "command": command, "plan_sha256": identity}
    _require(plan["scheduler"] == "slurm", "local-cloud is command-only; execute job.sh explicitly")
    # O_EXCL is the interprocess submit lock. Keep even incomplete intents after
    # errors/timeouts: sbatch may have accepted the job before the caller failed.
    _exclusive(intent, _canonical({"status": "pending", "plan_sha256": identity, "command": command}) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=env, cwd=root)
    response = result.stdout.strip()
    _require(result.returncode == 0 and re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?", response),
             "failed or ambiguous sbatch; intent retained, no automatic retry")
    fields = response.split(";", 1)
    receipt = {"status": "submitted", "submitted": True, "plan_sha256": identity,
               "jobid": fields[0], "cluster": fields[1] if len(fields) == 2 else None}
    _exclusive(receipt_path, _canonical(receipt) + "\n")
    return receipt
