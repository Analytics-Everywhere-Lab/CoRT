"""Small offline releases for fresh-data cached-plan validation.

Integration contracts (owned by the runtime/data producer): config.model uses
the matrix Gemma pin; config.adapters pins paths relative to job-local
artifacts/adapters. Data manifests inventory the entire package, including
freshness.json, public/requests.jsonl, evaluator/labels.jsonl and assets/*.
Freshness is produced upstream; this module never manufactures a pass.

Runtime writes rank_NNN/worker_receipt.json with status, mode, rank,
world_size and full_projected_worker_s, plus aggregate_receipt.json with
status, mode, labels_opened and full_walltime_seconds after aggregation.
These receipts attest validation, not merely successful process exit.

Bootstrap workflow: upload source/data and submit the canonical canary script.
Its first real CPU srun step verifies data and prepares the immutable release;
the same batch script then stages and runs the normal four-GPU canary. Login
preflight checks bounded metadata only. Full submission needs that canary.

Legacy post-selection workflow: upload a source bundle without preparing a release;
inside an already authorized compute allocation, run submit_tamia.sh verify-data
SOURCE_SHA with explicit DATA_ROOT and DATA_RECEIPT. This is a one-task srun
verification step, not model execution or a new allocation. Then run prepare on
the uploaded source, using the receipt, prior release and pinned prior digest.
Only a separate explicit submit can launch a canary or gated full job. Source
upload is allowed while selection is pending; verification/submission is not.
"""
from __future__ import annotations

import argparse
import ast
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.lifecycle_memory import decoder_token_matrix_release as safe

JOB_REL = "docs/research/lifecycle_memory/phase 2 research/jobs/decoder_cached_validation"
HELPER = "research/lifecycle_memory/decoder_cached_validation_release.py"
RUNTIME = "research/lifecycle_memory/run_decoder_cached_validation.py"
CONFIG = JOB_REL + "/config.json"
JOB = JOB_REL + "/job_tamia_decoder_cached_validation_h100.sh"
ADAPTERS = JOB_REL + "/artifacts/adapters"
REGISTRY = JOB_REL + "/artifacts/exposure_registry.json"
MANIFEST = "release_manifest.json"
LIMIT = 32 * 1024 * 1024
MODEL = {"id": "google/gemma-4-31B-it", "revision": "3548789868c5356dbf307c98e6f609007b82b3eb"}


def small_json(path):
    safe.require(Path(path).stat().st_size <= LIMIT, f"oversized metadata: {path}")
    return safe.read_json(path)


def config_check(root):
    from research.lifecycle_memory.decoder_cached_validation import validate_config
    config = small_json(Path(root) / CONFIG)
    validate_config(config)
    safe.require(all(config["model"].get(k) == v for k, v in MODEL.items()), "Gemma31 pin drift")
    small_json(Path(root) / REGISTRY)
    safe.require(safe.sha256(Path(root) / REGISTRY) == safe.checked_hash(config["exposure_registry_sha256"]),
                 "frozen exposure registry hash drift")
    rows = [{"path": a["path"], "sha256": a["sha256"], "bytes": (Path(root) / ADAPTERS / a["path"]).stat().st_size}
            for a in config["adapters"]]
    safe.inventory({"files": rows})
    return rows


def source_files(root):
    """Static local import closure without importing model/runtime modules."""
    root = Path(root)
    pending = {HELPER, RUNTIME}
    selected = {CONFIG, JOB, REGISTRY, JOB_REL + "/worker.sh", JOB_REL + "/submit_tamia.sh",
                JOB_REL + "/launch_tamia.sh"}
    selected.update(ADAPTERS + "/" + r["path"] for r in config_check(root))
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        tree = ast.parse((root / name).read_text(), filename=name)
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join(name[:-3].split("/")[:-node.level] + ([base] if base else []))
                modules.add(base)
                modules.update(base + "." + a.name for a in node.names if a.name != "*")
            # Runtime loaders sometimes use explicit file imports. The old
            # release helper's unrelated CLI paths are not runtime dependencies.
            elif (name != safe.HELPER and isinstance(node, ast.Constant)
                  and isinstance(node.value, str) and node.value.startswith(("research/", "docs/"))
                  and node.value.endswith(".py") and (root / node.value).is_file()):
                pending.add(node.value)
        for module in modules:
            parts = module.split(".")
            candidates = ["/".join(parts) + ".py"]
            candidates += ["/".join(parts[:i]) + "/__init__.py" for i in range(1, len(parts) + 1)]
            pending.update(p for p in candidates if (root / p).is_file() and p not in selected)
    return sorted(selected)


def bounded_verify(root, manifest_name, expected_sha):
    root = Path(root)
    value = small_json(root / manifest_name)
    safe.require(safe.sha256(root / manifest_name) == safe.checked_hash(expected_sha), "manifest digest drift")
    rows = safe.inventory(value)
    safe.require(sum(r["bytes"] for r in rows) <= LIMIT, "release exceeds 32MiB")
    safe.verify_inventory(root, rows, excluded=(manifest_name,))
    return value


def bundle(repo_root, out_dir):
    root, out = Path(repo_root).resolve(), Path(out_dir)
    safe.require(not out.exists(), "bundle destination must be new")
    names = source_files(root)
    total = 0
    for name in names:
        path = root / name
        safe.require(path.is_file() and not path.is_symlink() and root in path.resolve().parents, "unsafe source")
        total += path.stat().st_size
        safe.require(total <= LIMIT, "source/adapters exceed 32MiB")
    safe.verify_inventory(root / ADAPTERS, config_check(root), exact=False)
    out.mkdir(parents=True)
    for name in names:
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dest)
    safe.atomic_json(out / "source_manifest.json", {"schema_version": 1, "files": [safe.entry(p, out) for p in safe.tree_files(out)]})
    return safe.sha256(out / "source_manifest.json")


def data_metadata(data_root):
    root = Path(data_root)
    safe.require(root.is_absolute() and root.is_dir() and not root.is_symlink(), "explicit absolute DATA_ROOT required")
    rows = safe.inventory(small_json(root / "data_manifest.json"))
    names = {r["path"] for r in rows}
    safe.require(sum(r["bytes"] for r in rows if not r["path"].startswith("assets/")) <= LIMIT,
                 "non-asset metadata too large for login preflight")
    safe.require({"public/requests.jsonl", "evaluator/labels.jsonl", "freshness.json", "exposure_registry.json"} <= names, "fresh data package incomplete")
    safe.require(any(n.startswith("assets/") for n in names), "asset inventory required")
    for row in rows:
        name = row["path"]
        safe.require(row.get("source", "package") == "package", "assets must be package-relative")
        safe.require(name in {"public/requests.jsonl", "evaluator/labels.jsonl", "freshness.json", "exposure_registry.json"}
                     or name.startswith("assets/"), "unexpected data inventory path")
        path = root / name
        safe.require(path.is_file() and root.resolve() in path.resolve().parents and not path.is_symlink(), "unsafe data path")
        safe.require(path.stat().st_size == row["bytes"], "data size drift")
    freshness = small_json(root / "freshness.json")
    safe.require(freshness.get("status") == "pass", "freshness must pass upstream participant/video-disjoint validation")
    row = next(r for r in rows if r["path"] == "freshness.json")
    safe.require(safe.sha256(root / "freshness.json") == row["sha256"], "freshness hash drift")
    return rows, {"data_manifest_sha256": safe.sha256(root / "data_manifest.json"),
                  "freshness_sha256": row["sha256"]}


def upload_metadata(data_root, data_sha):
    """No image, label, model, or SIF reads; compute certifies their bytes."""
    rows, identity = data_metadata(data_root)
    safe.require(identity["data_manifest_sha256"] == safe.checked_hash(data_sha), "data manifest drift")
    return identity


def prior_metadata(prior_release, prior_sha):
    prior = Path(prior_release)
    bounded_verify(prior, MANIFEST, prior_sha)
    old = small_json(prior / "deployment.json")
    inventory_name = safe.relative_path(old["model"]["inventory"])
    model = small_json(prior / inventory_name)
    model_rows = safe.inventory(model)
    prefix = "hub/models--google--gemma-4-31B-it/snapshots/" + MODEL["revision"] + "/"
    safe.require(all(r["path"].startswith(prefix) for r in model_rows), "prior HF inventory is not the pinned Gemma snapshot")
    safe.checked_hash(old["sif"]["sha256"])
    safe.require(Path(old["sif"]["path"]).is_absolute() and Path(old["model"]["root"]).is_absolute(), "remote paths must be absolute")
    return old, model


def bootstrap_preflight(source_root, source_sha, prior_release, prior_sha, data_root, data_sha):
    bounded_verify(source_root, "source_manifest.json", source_sha)
    config_check(source_root)
    config = small_json(Path(source_root) / CONFIG)
    safe.require(config["fresh_data"].get("status") == "frozen", "data design still awaits selection/freeze")
    prior_metadata(prior_release, prior_sha)
    identity = upload_metadata(data_root, data_sha)
    rows, _ = data_metadata(data_root)
    registry = next(r for r in rows if r["path"] == "exposure_registry.json")
    safe.require(registry["sha256"] == config["exposure_registry_sha256"], "data exposure registry pin drift")
    return identity


def bootstrap(source_root, source_sha, prior_release, prior_sha, data_root, data_sha, out_dir, executing_script):
    """One real compute step; never synthesize Slurm identity for verification."""
    safe.require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
                 and Path(os.environ.get("SLURM_TMPDIR", "/nonexistent")).is_dir(),
                 "bootstrap requires a compute allocation step with SLURM_TMPDIR")
    bootstrap_preflight(source_root, source_sha, prior_release, prior_sha, data_root, data_sha)
    safe.require(safe.sha256(executing_script) == safe.sha256(Path(source_root) / JOB), "executing script drift")
    out = Path(out_dir)
    safe.require(not out.exists() and not out.is_symlink(), "release destination must be new")
    temporary = out.with_name(out.name + ".preparing-" + os.environ["SLURM_JOB_ID"])
    receipt = Path(os.environ["SLURM_TMPDIR"]) / "cached-data-verification.json"
    verify_data(data_root, Path(source_root) / CONFIG, receipt)
    result = prepare(source_root, source_sha, prior_release, prior_sha, data_root, receipt, temporary)
    # Recheck the requested identity immediately before publishing the release.
    upload_metadata(data_root, data_sha)
    temporary.rename(out)
    for path in [*safe.tree_files(out), *sorted((p for p in out.rglob("*") if p.is_dir()), reverse=True), out]:
        path.chmod(path.stat().st_mode & ~0o222)
    return result


def verify_data(data_root, config, out_dir):
    """Run in an existing compute allocation; output is outside the package."""
    safe.require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
                 and Path(os.environ.get("SLURM_TMPDIR", "/nonexistent")).is_dir(),
                 "full data hashing requires a compute allocation step with SLURM_TMPDIR")
    dest = Path(out_dir).absolute()
    safe.require(dest.resolve() != Path(data_root).resolve() and Path(data_root).resolve() not in dest.resolve().parents,
                 "receipt must be outside data package")
    safe.require(not dest.exists(), "data receipt must be new")
    frozen = small_json(config)
    from research.lifecycle_memory.decoder_cached_validation import validate_config
    validate_config(frozen)
    safe.require(frozen["fresh_data"].get("status") == "frozen", "data design still awaits selection/freeze")
    rows, identity = data_metadata(data_root)
    safe.verify_inventory(data_root, rows, excluded=("data_manifest.json",))
    from research.lifecycle_memory.decoder_cached_validation import check_dataset
    check_dataset(data_root, frozen, verify_assets=False)
    safe.require(data_metadata(data_root) == (rows, identity) and small_json(config) == frozen,
                 "data/config changed during verification")
    safe.atomic_json(dest, {**identity, "status": "pass", "hashes_verified": True,
                           "config_sha256": safe.digest(frozen),
                           "job_id": os.environ["SLURM_JOB_ID"],
                           "step_id": os.environ["SLURM_STEP_ID"]}, exclusive=True)


def checked_data(data_root, receipt, config, *, metadata_only=False):
    rows, identity = data_metadata(data_root)
    value = small_json(receipt)
    safe.require(value.get("status") == "pass" and value.get("hashes_verified") is True
                 and value.get("config_sha256") == safe.digest(config)
                 and all(value.get(k) == v for k, v in identity.items()), "missing/stale data hash verification")
    safe.require(sum(r["bytes"] for r in rows if not r["path"].startswith("assets/")) <= LIMIT,
                 "non-asset metadata too large for login preflight")
    if not metadata_only:
        from research.lifecycle_memory.decoder_cached_validation import check_dataset
        check_dataset(data_root, config, verify_assets=False)
    return rows, identity


def prepare(source_root, source_sha, prior_release, prior_sha, data_root, data_receipt, out_dir):
    source, prior, out = Path(source_root), Path(prior_release), Path(out_dir)
    bounded_verify(source, "source_manifest.json", source_sha)
    old, model = prior_metadata(prior, prior_sha)
    rows, identity = checked_data(data_root, data_receipt, small_json(source / CONFIG))
    safe.require(not out.exists(), "release destination must be new")
    shutil.copytree(source, out, copy_function=shutil.copyfile)
    # Uploaded sources are read-only. Only the unpublished copy is writable
    # while replacing its source manifest with release metadata.
    for directory in [out, *(p for p in out.rglob("*") if p.is_dir())]:
        directory.chmod(directory.stat().st_mode | 0o700)
    (out / "source_manifest.json").unlink()
    safe.atomic_json(out / "model_inventory.json", model)
    safe.atomic_json(out / "data_inventory.json", {"files": rows})
    safe.atomic_json(out / "data_verification.json", small_json(data_receipt))
    safe.atomic_json(out / "deployment.json", {"schema_version": 1, "kind": "decoder_cached_validation",
        "data_root": str(Path(data_root).resolve()), "data_identity": identity,
        "prior_release_sha256": prior_sha, "source_sha256": source_sha,
        "model": {"root": old["model"]["root"]}, "sif": old["sif"]})
    safe.atomic_json(out / MANIFEST, {"schema_version": 1, "files": [safe.entry(p, out) for p in safe.tree_files(out)]})
    result = safe.sha256(out / MANIFEST)
    verify(out, result)
    return result


def verify(release, expected_sha):
    root = Path(release)
    bounded_verify(root, MANIFEST, expected_sha)
    spec = small_json(root / "deployment.json")
    safe.require(spec.get("kind") == "decoder_cached_validation" and spec.get("schema_version") == 1, "wrong release kind")
    safe.verify_inventory(root / ADAPTERS, config_check(root))
    safe.require(set(source_files(root)) <= {str(p.relative_to(root)) for p in safe.tree_files(root)}, "source closure missing")
    return spec


def preflight(release, expected_sha, data_root):
    spec = verify(release, expected_sha)
    safe.require(str(Path(data_root).resolve()) == spec["data_root"], "DATA_ROOT drift")
    rows, identity = checked_data(data_root, Path(release) / "data_verification.json",
                                  small_json(Path(release) / CONFIG), metadata_only=True)
    safe.require(identity == spec["data_identity"] and rows == small_json(Path(release) / "data_inventory.json")["files"], "data identity drift")
    return spec


def gate(release, expected_sha, data_root, canary_dir):
    preflight(release, expected_sha, data_root)
    out = Path(canary_dir)
    safe.require((out / "copyback_verified.json").is_file(), "canary copyback not published")
    rows = safe.inventory(small_json(out / "copyback_manifest.json"))
    safe.require(sum(r["bytes"] for r in rows) <= LIMIT, "canary evidence exceeds bounded login verification; review on compute")
    safe.require(safe.verify_copyback(out)["status"] == "pass", "canary failed")
    result = small_json(out / "job_receipt.json")
    safe.require(result.get("status") == "pass" and result.get("mode") == "canary"
                 and result.get("release_sha256") == expected_sha, "same-release passing canary required")
    seconds = safe.number(result["full_walltime_seconds"], "full walltime", integer=True)
    safe.require(3600 <= seconds <= 10800, "measured full budget outside 1-3h; split workload, do not cap")
    return seconds


def stage(release, expected_sha, data_root, work_dir, mode, executing_script, walltime_seconds, canary_dir=None):
    safe.require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_TMPDIR"), "compute allocation required")
    work, root = Path(work_dir).absolute(), Path(release)
    safe.require(Path(os.environ["SLURM_TMPDIR"]).resolve() in work.resolve().parents, "work must be node-local")
    safe.require(mode in ("canary", "full"), "invalid mode")
    started = time.monotonic()
    spec = preflight(root, expected_sha, data_root)
    expected_time = 1200 if mode == "canary" else gate(root, expected_sha, data_root, canary_dir)
    safe.require(walltime_seconds == expected_time, "submission budget drift")
    safe.require(safe.sha256(executing_script) == safe.sha256(root / JOB), "executing script drift")
    work.mkdir(parents=True, exist_ok=False)
    (work / "out").mkdir()
    shutil.copytree(root, work / "release")
    verify(work / "release", expected_sha)
    for row in safe.inventory(small_json(root / "data_inventory.json")):
        name = row["path"]
        dest = work / "checked-data" / name
        safe._copy_checked(Path(data_root) / name, dest, row, data_root)
    shutil.copyfile(Path(data_root) / "data_manifest.json", work / "checked-data/data_manifest.json")
    safe.require(safe.sha256(work / "checked-data/data_manifest.json") == spec["data_identity"]["data_manifest_sha256"], "staged data manifest drift")
    from research.lifecycle_memory.decoder_cached_validation import check_dataset
    check_dataset(work / "checked-data", small_json(work / "release" / CONFIG), verify_assets=False)
    (work / "evaluator").mkdir()
    (work / "checked-data/evaluator").rename(work / "evaluator/evaluator")
    (work / "checked-data").rename(work / "data")
    for row in safe.inventory(small_json(root / "model_inventory.json")):
        safe._copy_checked(Path(spec["model"]["root"]) / row["path"], work / "model" / row["path"], row, spec["model"]["root"])
    sif = Path(spec["sif"]["path"])
    safe._copy_checked(sif, work / "runtime.sif", {"bytes": sif.stat().st_size, "sha256": spec["sif"]["sha256"]}, sif.parent)
    # Only provenance, never labels/data/model weights, is copied back.
    with tarfile.open(work / "out/source_release.tar.gz", "w:gz") as archive:
        for path in safe.tree_files(work / "release"):
            archive.add(path, arcname=str(path.relative_to(work / "release")), recursive=False)
    ctx = {"release_sha256": expected_sha, "mode": mode, "run_id": os.environ["SLURM_JOB_ID"],
           "script_sha256": safe.sha256(executing_script), "sif_sha256": spec["sif"]["sha256"],
           "model_inventory_sha256": safe.sha256(root / "model_inventory.json"),
           "config_sha256": safe.sha256(work / "release" / CONFIG),
           "config_digest": safe.digest(small_json(work / "release" / CONFIG)),
           "staging_seconds": time.monotonic() - started, "data_identity": spec["data_identity"]}
    safe.atomic_json(work / "context.json", ctx)
    return ctx


def container_command(work, ctx, operation, rank=None):
    work = Path(work).resolve()
    safe.require(operation in ("worker", "aggregate"), "invalid operation")
    safe.require((operation == "worker" and type(rank) is int and 0 <= rank < 4)
                 or (operation == "aggregate" and rank is None), "invalid rank")
    command = ["apptainer", "exec", "--containall", "--cleanenv", "--no-mount", "bind-paths,hostfs,cwd,home"]
    private = work / "private" / (str(rank) if rank is not None else "aggregate")
    private.mkdir(parents=True, exist_ok=True)
    command += ["--workdir", str(private)]
    for name in ("release", "data", "model", "out"):
        command += ["--bind", f"{work / name}:/mnt/cv/{name}:{'rw' if name == 'out' else 'ro'}"]
    if operation == "aggregate" and ctx["mode"] == "full":
        command += ["--bind", f"{work / 'evaluator' / 'evaluator'}:/mnt/cv/data/evaluator:ro"]
    env = {"PYTHONPATH": "/mnt/cv/release", "HF_HOME": "/mnt/cv/model", "HF_HUB_CACHE": "/mnt/cv/model/hub",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "12", "TOKENIZERS_PARALLELISM": "false",
           "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
           "SLURM_PROCID": str(rank) if rank is not None else "0"}
    if rank is not None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        safe.require(visible and len(visible.split(",")) == 1 and visible not in ("-1", "NoDevFiles"), "Slurm must bind one GPU")
        command += ["--nv"]
        env["CUDA_VISIBLE_DEVICES"] = visible
    for key, value in env.items():
        command += ["--env", f"{key}={value}"]
    command += ["--pwd", "/mnt/cv/release", str(work / "runtime.sif"), "python", "-u", "-m",
                "research.lifecycle_memory.run_decoder_cached_validation", operation, "--mode", ctx["mode"],
                "--data-dir", "/mnt/cv/data", "--config", "/mnt/cv/release/" + CONFIG, "--out-dir", "/mnt/cv/out"]
    if rank is not None:
        command += ["--assets-dir", "/mnt/cv/data/assets", "--adapters-dir", "/mnt/cv/release/" + ADAPTERS,
                    "--rank", str(rank), "--world-size", "4"]
    return command


def runtime_receipt(path, ctx, rank=None):
    value = small_json(path)
    safe.require(value.get("status") == "pass" and value.get("mode") == ctx["mode"], "runtime validation did not pass")
    if rank is not None:
        safe.require(value.get("rank") == rank and value.get("world_size") == 4, "runtime rank drift")
        safe.require(value.get("config_sha256") == ctx["config_sha256"], "runtime config drift")
        safe.number(value["full_projected_worker_s"], "projected full worker seconds")
        for key in ("load_s", "setup_s", "work_s"):
            safe.number(value[key], key)
        hashes = value["files_sha256"]
        safe.require(set(hashes) == {"predictions.jsonl", "timings.jsonl", "cold_start.json", "cache_receipt.json", "runtime_receipt.json"}, "runtime evidence incomplete")
        for name, expected in hashes.items():
            safe.require(safe.sha256(Path(path).parent / name) == safe.checked_hash(expected), "runtime output hash drift")
    else:
        safe.require(value.get("config_sha256") == ctx["config_digest"], "aggregate config drift")
        safe.require(value.get("labels_opened") is (ctx["mode"] == "full"), "label isolation receipt mismatch")
        safe.number(value["full_walltime_seconds"], "runtime full walltime", integer=True)
        safe.require(value.get("full_launch_allowed") is (value["full_walltime_seconds"] <= 10800),
                     "aggregate budget flag mismatch")
    return value


def worker_evidence(out, ctx):
    workers = []
    for rank in range(4):
        harness = small_json(Path(out) / f"harness_rank_{rank}.json")
        path = Path(out) / f"rank_{rank:03d}/worker_receipt.json"
        safe.require(all(harness.get(key) == ctx[key] for key in
                         ("release_sha256", "mode", "run_id", "config_sha256", "data_identity"))
                     and harness.get("status") == "pass"
                     and harness["runtime_sha256"] == safe.sha256(path), "worker evidence drift")
        runtime = runtime_receipt(path, ctx, rank)
        runtime["harness_elapsed_seconds"] = safe.number(harness["elapsed_seconds"], "measured worker seconds")
        safe.require(runtime["setup_s"] + runtime["work_s"] <= runtime["harness_elapsed_seconds"] * 1.05 + 1,
                     "runtime timing exceeds measured elapsed")
        workers.append(runtime)
    return workers


def aggregate_evidence(out, ctx):
    value = runtime_receipt(Path(out) / "aggregate_receipt.json", ctx)
    summary = runtime_receipt(Path(out) / "analysis_summary.json", ctx)
    for key in ("status", "mode", "config_sha256", "labels_opened", "full_walltime_seconds", "full_launch_allowed"):
        safe.require(value[key] == summary[key], "aggregate summary/receipt drift")
    return value


def run(work_dir, rank=None):
    work = Path(work_dir)
    ctx, out = small_json(work / "context.json"), work / "out"
    operation = "worker" if rank is not None else "aggregate"
    receipt = out / (f"rank_{rank:03d}/worker_receipt.json" if rank is not None else "aggregate_receipt.json")
    safe.require(not receipt.exists(), "stale runtime receipt")
    try:
        workers = []
        if rank is None:
            safe.require(not (out / "analysis_summary.json").exists(), "stale aggregate summary")
            workers = worker_evidence(out, ctx)
        started = time.monotonic()
        safe._run_container(container_command(work, ctx, operation, rank), out / (f"runtime_rank_{rank}.log" if rank is not None else "aggregate.log"))
        elapsed = time.monotonic() - started
        value = runtime_receipt(receipt, ctx, rank) if rank is not None else aggregate_evidence(out, ctx)
        result = {**ctx, "status": "pass", "elapsed_seconds": elapsed, "runtime_sha256": safe.sha256(receipt)}
        if rank is not None:
            safe.require(value["setup_s"] + value["work_s"] <= elapsed * 1.05 + 1, "runtime timing exceeds measured elapsed")
            safe.atomic_json(out / f"harness_rank_{rank}.json", result, exclusive=True)
        else:
            estimate = (max(max(w["full_projected_worker_s"], w["harness_elapsed_seconds"]) for w in workers) * 1.4
                        + ctx["staging_seconds"] + elapsed + 600)
            result["full_walltime_seconds"] = max(3600, value["full_walltime_seconds"], math.ceil(estimate / 60) * 60)
            result["full_launch_allowed"] = result["full_walltime_seconds"] <= 10800
            safe.atomic_json(out / "job_receipt.json", result, exclusive=True)
        return result
    except BaseException:
        safe.atomic_json(out / f"failure_{operation}_{rank}.json", {**ctx, "status": "fail", "traceback": traceback.format_exc()})
        raise


def finish(work_dir, exit_code, stage_name):
    out = Path(work_dir) / "out"
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "job_receipt.json"
    success = (exit_code == 0 and stage_name == "complete" and receipt.is_file()
               and small_json(receipt).get("status") == "pass")
    if success:
        try:
            ctx = small_json(Path(work_dir) / "context.json")
            worker_evidence(out, ctx)
            aggregate_evidence(out, ctx)
            job = small_json(receipt)
            safe.require(all(job.get(key) == ctx[key] for key in ("release_sha256", "mode", "run_id", "data_identity"))
                         and job["runtime_sha256"] == safe.sha256(out / "aggregate_receipt.json"),
                         "completion identity drift")
        except Exception:
            success = False
            safe.atomic_json(out / "failure_finish.json", {"status": "fail", "traceback": traceback.format_exc()})
    if receipt.exists() and not success:
        value = small_json(receipt)
        value["status"] = "fail"
        safe.atomic_json(receipt, value)
    safe.atomic_json(out / "shell_exit.json", {"exit_code": 0 if success else (exit_code or 1), "stage": stage_name, "status": "pass" if success else "fail"})
    safe.atomic_json(out / "copyback_manifest.json", {"schema_version": 1, "status": "pass" if success else "fail",
        "files": [safe.entry(p, out) for p in safe.tree_files(out, ("copyback_manifest.json", "copyback_verified.json"))]})
    return success


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    fields = {
        "bundle": ("repo-root", "out-dir"),
        "verify-source": ("source-root", "source-sha"),
        "verify-data": ("data-root", "config", "out-dir"),
        "upload-metadata": ("data-root", "data-sha"),
        "bootstrap-preflight": ("source-root", "source-sha", "prior-release", "prior-sha", "data-root", "data-sha"),
        "bootstrap": ("source-root", "source-sha", "prior-release", "prior-sha", "data-root", "data-sha", "out-dir", "executing-script"),
        "prepare": ("source-root", "source-sha", "prior-release", "prior-sha", "data-root", "data-receipt", "out-dir"),
        "verify": ("release", "expected-sha"),
        "preflight": ("release", "expected-sha", "data-root"),
        "gate": ("release", "expected-sha", "data-root", "canary-dir"),
        "stage": ("release", "expected-sha", "data-root", "work-dir", "mode", "executing-script"),
        "worker": ("work-dir",), "aggregate": ("work-dir",), "finish": ("work-dir", "stage-name"),
        "publish": ("out-dir",), "verify-copyback": ("out-dir",)}
    for command, names in fields.items():
        p = subs.add_parser(command)
        for name in names:
            p.add_argument("--" + name, required=True)
        if command == "stage":
            p.add_argument("--walltime-seconds", type=int, required=True)
            p.add_argument("--canary-dir")
        if command == "worker":
            p.add_argument("--rank", type=int, required=True)
        if command == "finish":
            p.add_argument("--exit-code", type=int, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "verify-source":
        result = bounded_verify(args["source_root"], "source_manifest.json", args["source_sha"])
    elif command in ("publish", "verify-copyback"):
        result = getattr(safe, command.replace("-", "_"))(**args)
    elif command in ("worker", "aggregate"):
        result = run(**args)
    else:
        result = globals()[command.replace("-", "_")](**args)
    if command == "finish" and args["exit_code"] == 0:
        safe.require(result, "successful shell exit without passing aggregate receipt")
    print(result if isinstance(result, (str, int)) else safe.canonical(result).decode())


if __name__ == "__main__":
    main()
