"""Offline release, staging, receipts and canary gate for decoder token matrix.

This module deliberately has no model/data downloads or torch imports. Runtime
receipts are a contract, not an inference from a subprocess's successful exit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

JOB_REL = "docs/research/lifecycle_memory/phase 2 research/jobs/decoder_token_matrix"
HELPER = "research/lifecycle_memory/decoder_token_matrix_release.py"
RUNTIME = "research/lifecycle_memory/run_decoder_token_matrix.py"
JOB = JOB_REL + "/job_tamia_decoder_token_matrix_h100.sh"
MANIFEST = "release_manifest.json"
EXPERIMENTS = ("compression", "steering")
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_RELEASE_BYTES = 32 * 1024 * 1024
METADATA_HASH_LIMIT = 32 * 1024 * 1024
SIF_SHA256 = "6ee3e881a0fd3feaa6a5b19a5e0f3c104cf2250c3468f8d6254a04164adfb905"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def bad_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs,
                      parse_constant=bad_constant)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(name, path)
        else:
            os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def relative_path(value):
    require(isinstance(value, str) and value, "empty relative path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in value
            and str(path) == value and value != ".", f"unsafe relative path: {value}")
    return value


def checked_hash(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "invalid sha256")
    return value


def number(value, name, *, positive=True, integer=False):
    require(type(value) in ((int,) if integer else (int, float)), f"invalid {name}")
    require(math.isfinite(value) and (value > 0 if positive else value >= 0), f"invalid {name}")
    return value


def entry(path, root):
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def tree_files(root, excluded=()):
    result = []
    for path in sorted(Path(root).rglob("*")):
        require(not path.is_symlink(), f"symlink in sealed tree: {path}")
        if path.is_file() and str(path.relative_to(root)) not in excluded:
            result.append(path)
    return result


def inventory(value):
    rows = value["files"]
    require(isinstance(rows, list) and rows, "empty inventory")
    seen = set()
    for row in rows:
        name = relative_path(row["path"])
        require(name not in seen, f"duplicate inventory path: {name}")
        seen.add(name)
        checked_hash(row["sha256"])
        number(row["bytes"], "bytes", positive=False, integer=True)
        require(row.get("source", "package") in ("package", "assets"), "invalid inventory source")
    return rows


def verify_inventory(root, rows, *, exact=True, excluded=()):
    root = Path(root)
    inventory({"files": rows})
    if exact:
        actual = {str(p.relative_to(root)) for p in tree_files(root, excluded)}
        require(actual == {r["path"] for r in rows}, "inventory tree mismatch")
    for row in rows:
        path = root / row["path"]
        require(path.is_file() and not path.is_symlink(), f"missing/unsafe file: {path}")
        require(path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
                f"file hash/size drift: {path}")


def validate_workloads(spec):
    require(set(spec["workloads"]) == set(EXPERIMENTS), "both frozen experiment workloads required")
    for experiment in EXPERIMENTS:
        plan = spec["workloads"][experiment]
        require(set(plan) == {"canary", "full"}, "canary and full workload required")
        for mode in ("canary", "full"):
            require(isinstance(plan[mode], list) and len(plan[mode]) == 4, "exactly four rank workloads required")
            for counts in plan[mode]:
                require(isinstance(counts, dict) and counts, "every GPU must have work")
                for key, value in counts.items():
                    require(isinstance(key, str) and key, "invalid work category")
                    number(value, "work count", integer=True)
        for rank in range(4):
            require(set(plan["canary"][rank]) == set(plan["full"][rank]), "unmeasured full work category")


def validate_spec(spec):
    require(spec["schema_version"] == 1, "unsupported deployment schema")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", spec["release_id"]), "unsafe release ID")
    files = [relative_path(p) for p in spec["files"]]
    require(len(set(files)) == len(files), "duplicate release file")
    required = {HELPER, RUNTIME, JOB, JOB_REL + "/worker.sh", JOB_REL + "/submit_tamia.sh",
                relative_path(spec["config_path"])}
    for key in ("data", "model"):
        if key == "model":
            required.add(relative_path(spec[key]["inventory"]))
        require(Path(spec[key]["root"]).is_absolute(), f"{key} root must be absolute")
    require(Path(spec["data"]["assets_root"]).is_absolute(), "assets root must be absolute")
    require(required <= set(files), f"missing release dependencies: {sorted(required - set(files))}")
    require(spec["data"]["inventory"] == "data_inventory.json", "data inventory is generated")
    require(not {MANIFEST, "deployment.json", "data_inventory.json", "dataset_split_manifest.json"} & set(files), "reserved release filenames")
    require(Path(spec["sif"]["path"]).is_absolute(), "SIF path must be absolute")
    checked_hash(spec["sif"]["sha256"])
    validate_workloads(spec)


def compression_arm_count(repo_root):
    """Ask the selected source tree, never an already-imported ambient module."""
    root = Path(repo_root).resolve()
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from research.lifecycle_memory.decoder_token_matrix import build_arms; "
            "print(len(build_arms()))")
    result = subprocess.run([sys.executable, "-B", "-S", "-c", code, str(root)],
                            cwd=root, capture_output=True, text=True, timeout=30,
                            env={**os.environ, "PYTHONPATH": str(root),
                                 "PYTHONDONTWRITEBYTECODE": "1"})
    require(result.returncode == 0, f"frozen build_arms() failed: {result.stderr[-4000:]}")
    require(result.stdout.strip() == "71", "compression requires exactly 71 frozen arms")
    return 71


def workload_plan(config, split, repo_root=None):
    """Frame-hash ordering matches the pilot: select first 16, then rank::4."""
    arm_count = compression_arm_count(repo_root or Path(__file__).resolve().parents[2])
    require(config["compression"]["canary_frames"] == 16, "canary requires 16 frames / 32 requests")
    require(config["compression"]["expected_arm_count"] == arm_count, "configured arm count drift")
    require(config["compression"]["benchmark_requests"] == 64
            and config["compression"]["timing_repeats"] == 3, "frozen benchmark budget drift")
    require((config["model"]["id"], config["model"]["revision"]) == (
        "google/gemma-4-31B-it", "3548789868c5356dbf307c98e6f609007b82b3eb"), "Gemma31 pin drift")
    rows = split["assignments"]
    require(len(rows) == split["counts"]["rows"] == 1600, "expected exactly 1600 public requests")
    require(len({r["request_id"] for r in rows}) == len(rows), "duplicate split request")
    require(len({r["frame_key"] for r in rows}) == split["counts"]["frames"] == 800, "expected 800 frames")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["frame_key"], []).append(row)
    require(all(len(group) == 2 and len({r["split"] for r in group}) == 1
                for group in grouped.values()), "each frame needs two requests in one split")
    for key in ("train", "validation", "test"):
        require(sum(r["split"] == key for r in rows) == split["counts_by_split"][key]["rows"], "split count drift")
    order = lambda value: hashlib.sha256(f"application-pilot-v1:{value}".encode()).hexdigest()
    frames = sorted({r["frame_key"] for r in rows}, key=order)
    steering = config["steering"]
    require(type(steering["case_count"]) is int and steering["case_count"] == 14,
            "steering requires 14 frozen cases")
    require(steering["seeds"] == [17, 29], "steering requires frozen seeds 17 and 29")
    require(steering["canary_eval_per_split"] == 4, "steering canary requires four requests per split")
    tasks = [(case, seed) for seed in steering["seeds"] for case in range(steering["case_count"])]
    result = {exp: {} for exp in EXPERIMENTS}
    for mode in ("canary", "full"):
        selected = frames[:config["compression"]["canary_frames"]] if mode == "canary" else frames
        result["compression"][mode] = [{"prediction_rows": sum(r["frame_key"] in set(selected[rank::4]) for r in rows)
                                        * arm_count} for rank in range(4)]
        eval_rows = [r for r in rows if r["split"] in ("validation", "test")]
        if mode == "canary":
            selected_frames = {frame for split_name in ("validation", "test")
                for frame in sorted({r["frame_key"] for r in eval_rows if r["split"] == split_name}, key=order)[:2]}
            eval_rows = [r for r in eval_rows if r["frame_key"] in selected_frames]
            require(len(eval_rows) == 8, "incomplete steering canary evaluation rows")
        result["steering"][mode] = [{"prediction_rows": len(tasks[rank::4]) * len(eval_rows) * 3}
                                    for rank in range(4)]
    return result


def data_inventory(dataset_dir, assets_dir):
    dataset, assets = Path(dataset_dir), Path(assets_dir)
    require((dataset / "split_manifest.json").stat().st_size <= METADATA_HASH_LIMIT, "split metadata too large")
    split = read_json(dataset / "split_manifest.json")
    result = [entry(dataset / "split_manifest.json", dataset)]
    metadata_bytes = (dataset / "split_manifest.json").stat().st_size
    for name, metadata in split["exports"].items():
        relative_path(name)
        metadata_bytes += (dataset / name).stat().st_size
        require(metadata_bytes <= METADATA_HASH_LIMIT, "dataset metadata exceeds login-safe 32MiB limit")
        row = entry(dataset / name, dataset)
        require(row["sha256"] == metadata["sha256"], f"dataset export drift: {name}")
        result.append(row)
    seen = {}
    with (dataset / "public/requests.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            for asset in row["inputs"]:
                name = relative_path(asset["path"])
                require(not name.startswith(("hidden/", "evaluator/", "labels/", "public/")), "label path in asset inventory")
                checked_hash(asset["sha256"])
                require(name not in seen or seen[name]["sha256"] == asset["sha256"], "conflicting asset hashes")
                source = assets / name
                require(source.is_file() and assets.resolve() in source.resolve().parents, f"missing/unsafe local asset: {name}")
                seen[name] = {"path": name, "sha256": asset["sha256"], "bytes": source.stat().st_size, "source": "assets"}
    result.extend(seen[name] for name in sorted(seen))
    inventory({"files": result})
    return {"files": result}, split


def source_files(repo_root):
    """Scoped import closure, including the legacy loader's explicit file import."""
    root = Path(repo_root)
    pending = {HELPER, RUNTIME, "research/lifecycle_memory/decoder_token_matrix.py",
               "research/lifecycle_memory/decoder_token_matrix_steering.py",
               "research/lifecycle_memory/decoder_token_matrix_analysis.py",
               "research/lifecycle_memory/application_pilot_data.py",
               "docs/research/lifecycle_memory/phase 2 research/jobs/job0/run_phase1_qwen27b_hf.py"}
    selected = {JOB, JOB_REL + "/worker.sh", JOB_REL + "/submit_tamia.sh", JOB_REL + "/config.json"}
    selected.update(JOB_REL + "/" + name for name in ("README.md", "RUNBOOK.md", "VERIFICATION.md"))
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        path = root / name
        require(path.is_file(), f"missing runtime dependency: {name}")
        selected.add(name)
        tree = ast.parse(path.read_text(), filename=name)
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    package = name.removesuffix(".py").split("/")[:-node.level]
                    base = ".".join(package + ([base] if base else []))
                modules.add(base)
                modules.update(base + "." + alias.name for alias in node.names if alias.name != "*")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if value.startswith(("research/", "docs/")) and value.endswith(".py") and (root / value).is_file():
                    pending.add(value)
        for module in modules:
            parts = module.split(".")
            for suffix in (".py", "/__init__.py"):
                candidate = "/".join(parts) + suffix
                if (root / candidate).is_file() and candidate not in selected:
                    pending.add(candidate)
            for stop in range(1, len(parts)):
                marker = "/".join(parts[:stop]) + "/__init__.py"
                if (root / marker).is_file() and marker not in selected:
                    pending.add(marker)
    return sorted(selected)


def bundle(repo_root, out_dir):
    root, out = Path(repo_root).resolve(), Path(out_dir)
    require(not out.exists(), "source bundle must be new")
    names = source_files(root)
    total = 0
    for name in names:
        path = root / name
        require(not path.is_symlink() and root in path.resolve().parents, "unsafe source dependency")
        require(path.stat().st_size <= MAX_FILE_BYTES, f"large source dependency: {name}")
        total += path.stat().st_size
    require(total <= MAX_RELEASE_BYTES, "source closure exceeds 32MiB")
    out.mkdir(parents=True)
    for name in names:
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dest)
    manifest = {"schema_version": 1, "files": [entry(p, out) for p in tree_files(out)]}
    atomic_json(out / "source_manifest.json", manifest)
    return {"source_sha256": sha256(out / "source_manifest.json"), "files": len(names), "bytes": total}


def model_inventory(model_root, config):
    """Use content-addressed HF blob names for weights; never hash big files here."""
    root = Path(model_root).resolve()
    model = config["model"]
    require(re.fullmatch(r"[0-9a-f]{40}", model["revision"]), "model revision must be pinned commit")
    relative = Path("hub") / ("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"]
    snapshot = root / relative
    require(snapshot.is_dir() and root in snapshot.resolve().parents, "pinned HF snapshot missing; pre-stage it offline")
    rows, metadata_bytes = [], 0
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            require(not path.is_symlink(), "symlinked model directories not supported")
            continue
        resolved = path.resolve()
        require(path.is_file() and root in resolved.parents, f"model symlink escapes HF_HOME: {path}")
        size = path.stat().st_size
        if resolved.parent.name == "blobs" and re.fullmatch(r"[a-f0-9]{64}", resolved.name):
            expected = resolved.name
        else:
            metadata_bytes += size
            require(size <= METADATA_HASH_LIMIT and metadata_bytes <= METADATA_HASH_LIMIT,
                    f"unknown large/non-content-addressed model file: {path}; supply a preverified HF SHA256 blob offline")
            expected = sha256(path)
        rows.append({"path": str(path.relative_to(root)), "bytes": size, "sha256": expected})
    require(rows, "empty pinned model snapshot")
    index_path = snapshot / "model.safetensors.index.json"
    require(index_path.is_file(), "pinned sharded model index is missing")
    require(index_path.stat().st_size <= METADATA_HASH_LIMIT, "model index too large")
    index = read_json(index_path)
    names = {str(Path(row["path"]).relative_to(relative)) for row in rows}
    require(set(index["weight_map"].values()) <= names, "missing pinned model weight shards")
    require({"config.json", "tokenizer_config.json"} <= names, "missing pinned model metadata")
    return {"files": rows}


def verify_source_data(assets_root, config):
    pins = config["data"]["source_hashes"]
    require(set(pins) == {"public/real_request_pack.jsonl", "hidden/evaluator_index.jsonl"},
            "both frozen source metadata hashes required")
    for name, expected in pins.items():
        path = Path(assets_root) / name
        require(path.is_file() and path.stat().st_size <= METADATA_HASH_LIMIT,
                f"missing/oversized source metadata: {name}")
        require(sha256(path) == checked_hash(expected), f"source metadata hash drift: {name}")


def validate_dataset_sources(config, split):
    pins = config["data"]["source_hashes"]
    require(set(pins) == {"public/real_request_pack.jsonl", "hidden/evaluator_index.jsonl"},
            "both frozen source metadata hashes required")
    inputs = split.get("input_hashes", {})
    for name, expected in pins.items():
        require(isinstance(inputs.get(name), dict)
                and inputs[name].get("sha256") == checked_hash(expected),
                f"dataset source metadata hash drift: {name}")


def bootstrap(source_root, source_sha, output_root, assets_root, model_root, sif_path, sif_sha=SIF_SHA256):
    """Login-safe metadata preparation; all large byte verification is deferred."""
    source, output = Path(source_root).resolve(), Path(output_root).absolute()
    require(sha256(source / "source_manifest.json") == checked_hash(source_sha), "uploaded source manifest drift")
    sources = read_json(source / "source_manifest.json")
    verify_inventory(source, sources["files"], excluded=("source_manifest.json",))
    require(not output.exists(), "bootstrap root exists; use a new version, never overwrite a release")
    require(Path(sif_path).is_file(), "pre-staged SIF missing; no remote download/build allowed")
    config = read_json(source / JOB_REL / "config.json")
    verify_source_data(assets_root, config)
    output.mkdir(parents=True)
    sys.path.insert(0, str(source))
    from research.lifecycle_memory.application_pilot_data import prepare_data, export_data
    dataset = output / "dataset"
    prepared = prepare_data(assets_root, expected_frames=800)
    for name, expected in config["data"]["source_hashes"].items():
        require(prepared.manifest["input_hashes"][name]["sha256"] == expected, "source changed during preparation")
    export_data(prepared, dataset)
    model_rows = model_inventory(model_root, config)
    # Keep the digest-keyed upload exact so interrupted uploads can be resumed.
    metadata = Path(tempfile.mkdtemp(prefix=".metadata-", dir=output))
    shutil.copytree(source, metadata, dirs_exist_ok=True)
    model_rel = JOB_REL + "/model_inventory.json"
    atomic_json(metadata / model_rel, model_rows)
    spec = {"schema_version": 1, "release_id": output.name,
            "files": sorted([row["path"] for row in sources["files"]] + [model_rel]),
            "config_path": JOB_REL + "/config.json",
            "data": {"root": str(dataset), "assets_root": str(Path(assets_root).resolve()), "inventory": "data_inventory.json"},
            "model": {"root": str(Path(model_root).resolve()), "inventory": model_rel},
            "sif": {"path": str(Path(sif_path).resolve()), "sha256": checked_hash(sif_sha)}}
    spec_path = output / "bootstrap_spec.json"
    atomic_json(spec_path, spec)
    try:
        result = prepare(metadata, spec_path, output / "release", dataset, assets_root)
    finally:
        shutil.rmtree(metadata)
    atomic_json(output / "bootstrap_receipt.json", {**result, "source_sha256": source_sha,
                "large_payload_hashing": "deferred to compute", "status": "pass"})
    (output / "release.sha256").write_text(result["release_sha256"] + "\n")
    for path in tree_files(output / "release"):
        path.chmod(0o444)
    for path in sorted((output / "release").rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    (output / "release").chmod(0o555)
    return {**result, "dataset": str(dataset), "status": "pass", "submitted": False}


def prepare(repo_root, spec_path, out_dir, dataset_dir=None, assets_dir=None):
    root, out = Path(repo_root).resolve(), Path(out_dir).absolute()
    require(not out.exists(), "immutable release already exists")
    spec = read_json(spec_path)
    generated_data, split = data_inventory(dataset_dir or spec["data"]["root"], assets_dir or spec["data"]["assets_root"])
    config = read_json(root / relative_path(spec["config_path"]))
    validate_dataset_sources(config, split)
    spec["data"]["inventory"] = "data_inventory.json"
    spec["workloads"] = workload_plan(config, split, root)
    validate_spec(spec)
    require(set(source_files(root)) <= set(spec["files"]), "incomplete runtime/doc/config source closure")
    total = 0
    for name in spec["files"]:
        path = root / name
        require(path.is_file() and root in path.resolve().parents and
                not any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)),
                f"missing or symlink release source: {name}")
        require(path.suffix in {".py", ".sh", ".json", ".md", ".txt"}, f"forbidden release asset: {name}")
        require(path.stat().st_size <= MAX_FILE_BYTES, f"large release file: {name}")
        total += path.stat().st_size
    require(total + len(canonical(spec)) <= MAX_RELEASE_BYTES, "release exceeds size limit")
    for key in ("data", "model"):
        rows = inventory(generated_data if key == "data" else read_json(root / spec[key]["inventory"]))
        if key == "data":
            required_data = {"split_manifest.json", "public/requests.jsonl", "public/train.jsonl",
                             "public/validation.jsonl", "public/test.jsonl", "labels/train.jsonl",
                             "labels/validation.jsonl", "evaluator/test_labels.jsonl", "evaluator/all_labels.jsonl"}
            require(required_data == {r["path"] for r in rows if r.get("source", "package") == "package"},
                    "unexpected or incomplete prepared dataset inventory")
            for row in rows:
                if row.get("source", "package") == "assets":
                    require(not row["path"].startswith(("hidden/", "evaluator/", "labels/", "public/")),
                            "assets inventory contains metadata/labels")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dtm-release-", dir=out.parent))
    try:
        for name in spec["files"]:
            dest = staging / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, dest)
        atomic_json(staging / "deployment.json", spec)
        atomic_json(staging / "data_inventory.json", generated_data)
        atomic_json(staging / "dataset_split_manifest.json", split)
        rows = [entry(p, staging) for p in tree_files(staging)]
        require(all(row["bytes"] <= MAX_FILE_BYTES for row in rows)
                and sum(row["bytes"] for row in rows) <= MAX_RELEASE_BYTES, "generated release exceeds size limits")
        atomic_json(staging / MANIFEST, {"schema_version": 1, "release_id": spec["release_id"], "files": rows})
        verify(staging, sha256(staging / MANIFEST))
        # rename refuses replacing a populated directory; callers must use new IDs.
        require(not out.exists(), "release destination appeared during preparation")
        staging.rename(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"release_sha256": sha256(out / MANIFEST), "release": str(out)}


def verify(release, expected_sha):
    root = Path(release)
    require(sha256(root / MANIFEST) == checked_hash(expected_sha), "release manifest hash drift")
    manifest = read_json(root / MANIFEST)
    require(manifest["schema_version"] == 1, "unsupported release schema")
    verify_inventory(root, manifest["files"], excluded=(MANIFEST,))
    spec = read_json(root / "deployment.json")
    validate_spec(spec)
    require(manifest["release_id"] == spec["release_id"], "release ID drift")
    require({r["path"] for r in manifest["files"]} == set(spec["files"]) | {"deployment.json", "data_inventory.json", "dataset_split_manifest.json"},
            "release selection drift")
    require(set(source_files(root)) <= set(spec["files"]), "incomplete runtime/doc/config source closure")
    config, split = read_json(root / spec["config_path"]), read_json(root / "dataset_split_manifest.json")
    validate_dataset_sources(config, split)
    require(spec["workloads"] == workload_plan(config, split, root),
            "frozen workload counts drift")
    return spec


def identity(release, release_sha):
    root = Path(release)
    spec = verify(root, release_sha)
    data_rows = inventory(read_json(root / spec["data"]["inventory"]))
    split = [row for row in data_rows if row["path"] == "split_manifest.json"]
    require(len(split) == 1, "data inventory must pin split_manifest.json")
    return {"release_sha256": release_sha, "config_sha256": sha256(root / spec["config_path"]),
            "split_manifest_sha256": split[0]["sha256"],
            "sif_sha256": spec["sif"]["sha256"], "script_sha256": sha256(root / JOB),
            "data_inventory_sha256": sha256(root / spec["data"]["inventory"]),
            "model_inventory_sha256": sha256(root / spec["model"]["inventory"])}


def _copy_checked(source, dest, row, source_root):
    source = Path(source)
    require(source.is_file() and Path(source_root).resolve() in source.resolve().parents,
            f"source escapes root or missing: {source}")
    # Only the staged copy runs, so hashing it (not also the Lustre source)
    # catches drift with one fewer full read of every large payload.
    require(source.stat().st_size == row["bytes"], f"source size drift: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    require(dest.stat().st_size == row["bytes"] and sha256(dest) == row["sha256"],
            f"staged hash drift: {dest}")


def _context(work):
    return read_json(Path(work) / "context.json")


def failure(work, stage_name, rank=None):
    ctx_path = Path(work) / "context.json"
    ctx = read_json(ctx_path) if ctx_path.exists() else {}
    payload = {"schema_version": 1, "status": "fail", "stage": stage_name, "rank": rank,
               "identity": ctx.get("identity"), "run_id": ctx.get("run_id"),
               "traceback": traceback.format_exc(), "time": time.time()}
    name = f"failure_rank_{rank}.json" if rank is not None else f"failure_{stage_name}.json"
    atomic_json(Path(work) / "out" / name, payload)


def archive_release(release, out):
    """Preserve only the bounded code/config/metadata release, never payloads."""
    root = Path(release)
    paths = tree_files(root)
    require(all(p.stat().st_size <= MAX_FILE_BYTES for p in paths)
            and sum(p.stat().st_size for p in paths) <= MAX_RELEASE_BYTES,
            "source provenance exceeds release size bounds")
    destination = Path(out) / "provenance/source_release.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), "source provenance already exists")
    with tarfile.open(destination, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)


def stage(release, expected_sha, work_dir, experiment, mode, run_id, executing_script,
          walltime_seconds, canary_dir=None):
    work = Path(work_dir).absolute()
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_TMPDIR"), "staging is compute-only")
    require(Path(os.environ["SLURM_TMPDIR"]).resolve() in work.resolve().parents,
            "work directory must be below SLURM_TMPDIR")
    require(not work.exists(), "work directory must be new")
    (work / "out").mkdir(parents=True)
    started = time.monotonic()
    try:
        require(experiment in EXPERIMENTS and mode in ("canary", "full"), "invalid experiment/mode")
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", run_id), "unsafe run ID")
        spec = verify(release, expected_sha)
        ident = identity(release, expected_sha)
        require(sha256(executing_script) == ident["script_sha256"], "executing Slurm script drift")
        number(walltime_seconds, "walltime", integer=True)
        require(300 <= walltime_seconds <= 14400, "walltime outside bounds")
        if mode == "canary":
            require(walltime_seconds == 1200, "canary walltime must be 1200 seconds")
        if mode == "full":
            require(canary_dir is not None, "full requires passing canary")
            seconds = gate(release, expected_sha, experiment, canary_dir)
            require(walltime_seconds >= seconds, "full walltime below measured bound")
        ctx = {"schema_version": 1, "identity": ident, "identity_sha256": digest(ident),
               "experiment": experiment, "mode": mode, "run_id": run_id, "spec": spec,
               "walltime_seconds": walltime_seconds, "status": "staging"}
        atomic_json(work / "context.json", ctx)
        shutil.copytree(release, work / "release")
        verify(work / "release", expected_sha)
        archive_release(work / "release", work / "out")
        for key in ("data", "model"):
            base = spec[key]
            for row in inventory(read_json(Path(release) / base["inventory"])):
                source_root = base["root"] if row.get("source", "package") == "package" else base["assets_root"]
                rel = row["path"]
                # Test labels are never visible to prediction/training workers.
                if key == "data" and (rel.startswith("evaluator/") or rel == "labels/validation.jsonl"
                                      or (rel == "labels/train.jsonl" and experiment == "compression")):
                    dest = work / "evaluator" / rel
                elif key == "data" and row.get("source", "package") == "package":
                    dest = work / "dataset" / rel
                else:
                    dest = work / key / rel
                _copy_checked(Path(source_root) / rel, dest, row, source_root)
        (work / "dataset" / "evaluator").mkdir(parents=True, exist_ok=True)
        require(not any((work / "dataset" / "evaluator").iterdir()), "worker evaluator directory not empty")
        require(not (work / "data" / "hidden").exists(), "source hidden data must not be staged")
        require(read_json(work / "dataset/split_manifest.json") ==
                read_json(work / "release/dataset_split_manifest.json"), "staged split manifest drift")
        sif = Path(spec["sif"]["path"])
        _copy_checked(sif, work / "runtime.sif", {"bytes": sif.stat().st_size,
                      "sha256": spec["sif"]["sha256"]}, sif.parent)
        ctx.update(status="running", staging_seconds=time.monotonic() - started)
        atomic_json(work / "context.json", ctx)
        atomic_json(work / "out" / "job_manifest.json", ctx)
        return ctx
    except Exception:
        failure(work, "stage")
        raise


def _receipt_base(ctx):
    return {key: ctx[key] for key in ("schema_version", "experiment", "mode", "run_id", "identity_sha256")}


def _counts_sum(counts):
    totals = {}
    for row in counts:
        for key, value in row.items():
            totals[key] = totals.get(key, 0) + value
    return totals


def check_runtime(value, ctx, rank=None):
    for key in ("experiment", "mode"):
        require(value.get(key) == ctx[key], f"runtime receipt {key} drift")
    for key in ("config_sha256", "split_manifest_sha256"):
        require(value.get(key) == ctx["identity"][key], f"runtime receipt {key} drift")
    status_ok = value.get("status") == "pass" or (
        rank is None and ctx["mode"] == "canary" and value.get("status") == "needs_review"
        and value.get("runtime_budget_status") == "needs_review")
    require(status_ok and value.get("technical_gate", {}).get("status") == "pass"
            and value.get("oom", False) is False and value.get("runtime", {}).get("oom", False) is False,
            "runtime failed technical/OOM gate")
    plan = ctx["spec"]["workloads"][ctx["experiment"]][ctx["mode"]]
    if rank is not None:
        expected = plan[rank]
        require(type(value.get("rank")) is int and value["rank"] == rank and value.get("world_size") == 4,
                "runtime rank/world-size drift")
        number(value.get("prediction_rows"), "completed prediction rows", integer=True)
        require(value.get("prediction_rows") == expected["prediction_rows"],
                "runtime prediction count drift")
        for key in ("load_s", "work_s", "projected_full_work_s"):
            number(value.get(key), key)
    else:
        number(value.get("recommended_walltime_s"), "profile walltime")
        if ctx["mode"] == "canary":
            # Over-budget is a full-run decision, not a technical canary failure.
            require(value.get("runtime_budget_status") in ("pass", "needs_review"),
                    "invalid runtime budget status")
        require(value.get("prediction_inventory", {}).get("status") == "pass"
                and value["prediction_inventory"].get("cells") == sum(r["prediction_rows"] for r in plan),
                "aggregate prediction inventory incomplete")
    return value


def container_command(work, ctx, operation, rank=None):
    work = Path(work).resolve()
    require(operation in ("worker", "aggregate"), "invalid container operation")
    require((operation == "worker" and type(rank) is int and 0 <= rank < 4)
            or (operation == "aggregate" and rank is None), "invalid container rank")
    namespace = Path("/mnt/dtm")
    repo, data, model, out = (namespace / name for name in ("release", "data", "model", "out"))
    dataset = namespace / "dataset"
    command = ["apptainer", "exec", "--containall", "--cleanenv", "--no-mount", "bind-paths,hostfs,cwd,home"]
    private = work / "apptainer-work" / (f"rank-{rank}" if rank is not None else "aggregate")
    private.mkdir(parents=True, exist_ok=True)
    command.extend(["--workdir", str(private)])
    env = {"PYTHONPATH": str(repo), "HF_HOME": str(model), "HF_HUB_CACHE": str(model / "hub"),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline",
           "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "12", "PYTHONUNBUFFERED": "1",
           "DECODER_TOKEN_MATRIX_IDENTITY_SHA256": ctx["identity_sha256"],
           "DECODER_TOKEN_MATRIX_RUN_ID": ctx["run_id"]}
    env["SLURM_JOB_ID"] = os.environ.get("SLURM_JOB_ID", "")
    env["SLURM_PROCID"] = os.environ.get("SLURM_PROCID", "0")
    if rank is not None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        require(visible and len(visible.split(",")) == 1 and visible not in ("-1", "NoDevFiles"),
                "Slurm must bind one GPU per worker")
        env["CUDA_VISIBLE_DEVICES"] = visible
        command.append("--nv")
    for name in ("release", "data", "dataset", "model", "out"):
        mode = "rw" if name == "out" else "ro"
        command.extend(["--bind", f"{work / name}:{namespace / name}:{mode}"])
    if operation == "aggregate":
        require((work / "evaluator").is_dir(), "missing evaluator-only data")
        command.extend(["--bind", f"{work / 'evaluator'}:{namespace / 'evaluator'}:ro",
                        "--bind", f"{work / 'evaluator' / 'evaluator'}:{dataset / 'evaluator'}:ro"])
    command.extend(["--pwd", str(repo)])
    for key, value in env.items():
        command.extend(["--env", f"{key}={value}"])
    command.extend([str(work / "runtime.sif"), "python", "-u", "-m",
                    "research.lifecycle_memory.run_decoder_token_matrix", operation,
                    "--experiment", ctx["experiment"], "--mode", ctx["mode"],
                    "--data-dir", str(data), "--dataset-dir", str(dataset), "--out-dir", str(out),
                    "--config", str(repo / ctx["spec"]["config_path"])])
    if rank is not None:
        command.extend(["--rank", str(rank), "--world-size", "4"])
    return command


def container_environment():
    """Keep only harness-owned cache/temp options; CLI defines all other options."""
    allowed = {"APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"}
    prefixes = ("APPTAINER_", "SINGULARITY_", "APPTAINERENV_", "SINGULARITYENV_")
    return {key: value for key, value in os.environ.items()
            if key in allowed or not key.startswith(prefixes)}


def _run_container(command, log):
    """Forward batch termination to the container group before sealing outputs."""
    with Path(log).open("xb") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=container_environment())
        old = {}

        def terminate(signum, frame):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise InterruptedError(f"container terminated by signal {signum}")

        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                old[sig] = signal.signal(sig, terminate)
            status = process.wait()
            require(status == 0, f"container exited {status}; first exception is in {log}")
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def worker(work_dir, rank):
    work = Path(work_dir)
    try:
        require(type(rank) is int and 0 <= rank < 4, "invalid rank")
        ctx = _context(work)
        out = work / "out"
        rank_dir = out / f"rank_{rank:03d}"
        require(not rank_dir.exists(), "stale runtime rank directory")
        started = time.monotonic()
        log = out / f"worker_rank_{rank}.log"
        _run_container(container_command(work, ctx, "worker", rank), log)
        receipt_path = rank_dir / "worker_receipt.json"
        runtime = check_runtime(read_json(receipt_path), ctx, rank)
        result = {**runtime, **_receipt_base(ctx), "elapsed_seconds": time.monotonic() - started,
                  "runtime_receipt_sha256": sha256(receipt_path), "log_sha256": sha256(log)}
        atomic_json(out / f"worker_rank_{rank}.json", result)
        return result
    except Exception:
        failure(work, "worker", rank)
        raise


def check_workers(out, ctx):
    workers = []
    for rank in range(4):
        harness = check_runtime(read_json(out / f"worker_rank_{rank}.json"), ctx, rank)
        for key, expected in _receipt_base(ctx).items():
            require(harness.get(key) == expected, f"harness {key} drift")
        runtime_path = out / f"rank_{rank:03d}" / "worker_receipt.json"
        runtime = check_runtime(read_json(runtime_path), ctx, rank)
        predictions = out / f"rank_{rank:03d}" / "predictions.jsonl"
        require(runtime["predictions_sha256"] == sha256(predictions), "prediction hash drift")
        with predictions.open() as handle:
            require(sum(bool(line.strip()) for line in handle) == runtime["prediction_rows"], "prediction rows drift")
        require(harness["runtime_receipt_sha256"] == sha256(runtime_path), "runtime receipt hash drift")
        require(harness["log_sha256"] == sha256(out / f"worker_rank_{rank}.log"), "worker log hash drift")
        for key in ("runtime", "load_s", "work_s", "projected_full_work_s", "prediction_rows"):
            require(harness[key] == runtime[key], f"worker {key} drift")
        number(harness["elapsed_seconds"], "worker elapsed")
        require(runtime["work_s"] + runtime["load_s"] <= harness["elapsed_seconds"] * 1.05 + 1,
                "runtime timing exceeds measured worker elapsed")
        workers.append(harness)
    return workers


def project_seconds(ctx, workers, aggregate_seconds):
    number(ctx["staging_seconds"], "staging seconds")
    number(aggregate_seconds, "aggregation seconds")
    # Runtime projects the sum of each rank's task work, including train/eval and
    # benchmark categories. Use the slowest rank, never average node throughput.
    projected = max(number(w["projected_full_work_s"], "full work seconds") for w in workers)
    load = max(number(w["load_s"], "load seconds") for w in workers)
    estimate = projected * 1.4 + load + ctx["staging_seconds"] + aggregate_seconds + 600
    return max(3600, math.ceil(estimate / 60) * 60)


def budget_status(seconds):
    return "pass" if seconds <= 14400 else "needs_review"


def aggregate(work_dir):
    work = Path(work_dir)
    try:
        ctx, out = _context(work), work / "out"
        workers = check_workers(out, ctx)
        require(not (out / "profile.json").exists(), "stale aggregate receipt")
        started = time.monotonic()
        _run_container(container_command(work, ctx, "aggregate"), out / "aggregate.log")
        elapsed = time.monotonic() - started
        runtime = check_runtime(read_json(out / "profile.json"), ctx)
        if ctx["experiment"] == "compression":
            require((out / "arm_inventory.json").is_file(), "missing runtime arm_inventory.json snapshot")
        result = {**runtime, **_receipt_base(ctx), "elapsed_seconds": elapsed,
                  "runtime_receipt_sha256": sha256(out / "profile.json"),
                  "log_sha256": sha256(out / "aggregate.log")}
        atomic_json(out / "aggregate_receipt.json", result)
        receipt = {**_receipt_base(ctx), "status": "pass", "identity": ctx["identity"]}
        if ctx["mode"] == "canary":
            receipt["full_walltime_seconds"] = max(project_seconds(ctx, workers, elapsed),
                math.ceil(runtime["recommended_walltime_s"] / 60) * 60)
            # Keep technically passing evidence; the full gate refuses >4h.
            receipt["full_budget_status"] = budget_status(receipt["full_walltime_seconds"])
        receipt["evidence"] = [entry(p, out) for p in tree_files(out)]
        name = "canary_receipt.json" if ctx["mode"] == "canary" else "completion_receipt.json"
        atomic_json(out / name, receipt)
        return receipt
    except Exception:
        failure(work, "aggregate")
        raise


def finish(work_dir, exit_code, stage_name):
    work, out = Path(work_dir), Path(work_dir) / "out"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "job_manifest.json"
    job = read_json(path) if path.exists() else {"schema_version": 1}
    name = "canary_receipt.json" if job.get("mode") == "canary" else "completion_receipt.json"
    success = exit_code == 0 and (out / name).is_file() and (out / "aggregate_receipt.json").is_file()
    job.update(status="pass" if success else "fail", exit_code=exit_code if exit_code else (0 if success else 1),
               final_stage=stage_name, finished_at=time.time())
    atomic_json(path, job)
    if not success:
        for receipt in ("canary_receipt.json", "completion_receipt.json"):
            (out / receipt).unlink(missing_ok=True)
    elif (out / name).exists():
        receipt = read_json(out / name)
        # Job status is final only now, after all child processes have stopped.
        receipt["evidence"] = [entry(p, out) for p in tree_files(out, (name, "copyback_manifest.json"))]
        atomic_json(out / name, receipt)
    rows = [entry(p, out) for p in tree_files(out, ("copyback_manifest.json",))]
    atomic_json(out / "copyback_manifest.json", {"schema_version": 1, "status": job["status"], "files": rows})
    return job


def verify_copyback(out_dir):
    out = Path(out_dir)
    value = read_json(out / "copyback_manifest.json")
    require(value["schema_version"] == 1 and value["status"] in ("pass", "fail"), "invalid copyback manifest")
    verify_inventory(out, value["files"], excluded=("copyback_manifest.json", "copyback_verified.json"))
    published = out / "copyback_verified.json"
    if published.exists():
        receipt = read_json(published)
        require(receipt["manifest_sha256"] == sha256(out / "copyback_manifest.json")
                and receipt["status"] == value["status"], "stale copyback publication")
    return value


def publish(out_dir):
    out = Path(out_dir)
    require(not (out / "copyback_verified.json").exists(), "copyback already published")
    result = verify_copyback(out)
    receipt = {"schema_version": 1, "status": result["status"],
               "manifest_sha256": sha256(out / "copyback_manifest.json")}
    atomic_json(out / "copyback_verified.json", receipt)
    return receipt


def _fetch_paths(out_dir):
    requested = Path(out_dir).absolute()
    require(not requested.is_symlink(), "fetch destination must not be a symlink")
    out = requested.resolve()
    require(out.name and (not out.exists() or out.is_dir()), "invalid fetch destination")
    receipt = out.with_name(out.name + ".fetch.json")
    diagnostics = out.with_name(out.name + ".diagnostics")
    require(not receipt.is_symlink() and not diagnostics.is_symlink(), "unsafe fetch sidecar path")
    return out, receipt


def fetch_begin(out_dir, remote, source, identity_file):
    """Bind resumable payloads to an authenticated endpoint and immutable run."""
    require(re.fullmatch(r"[A-Za-z0-9._-]+@[A-Za-z0-9.-]+", remote), "invalid fetch remote")
    path = PurePosixPath(source)
    require(path.is_absolute() and str(path) == source and ".." not in path.parts
            and len(path.parts) >= 5, "invalid fetch source")
    require(Path(identity_file).stat().st_size <= MAX_FILE_BYTES, "oversized fetch identity")
    run = read_json(identity_file)
    require(run.get("schema_version") == 1 and run.get("experiment") in EXPERIMENTS
            and run.get("mode") in ("canary", "full"), "invalid remote run identity")
    checked_hash(run["release_sha256"])
    require(isinstance(run.get("run_id"), str) and re.fullmatch(r"[0-9]+", run["run_id"]),
            "invalid remote run ID")
    require(tuple(path.parts[-3:]) == (run["experiment"], run["mode"], run["run_id"]),
            "remote run identity does not match requested path")
    out, receipt = _fetch_paths(out_dir)
    binding = {"schema_version": 1, "remote": remote, "source": source,
               "destination": str(out), "run_identity": run}
    if not receipt.exists():
        require(not out.exists(), "existing fetch destination has no identity receipt")
        try:
            atomic_json(receipt, {**binding, "status": "transferring"}, exclusive=True)
        except FileExistsError:
            pass  # Another fetch must prove the same binding before it can resume.
    current = read_json(receipt)
    require(all(current.get(key) == value for key, value in binding.items()),
            "fetch identity mismatch; use a different destination")
    require(current.get("status") in ("transferring", "partial", "verified"), "invalid local fetch receipt")
    if current["status"] == "verified":
        verify_copyback(out)
        require(current["manifest_sha256"] == sha256(out / "copyback_manifest.json"),
                "verified fetch changed locally")
        return {"status": "verified"}
    out.mkdir(parents=True, exist_ok=True)
    return {"status": "transferring"}


def fetch_finish(out_dir, partial=False):
    out, receipt = _fetch_paths(out_dir)
    current = read_json(receipt)
    require(current.get("destination") == str(out), "fetch destination identity drift")
    require(current.get("status") in ("transferring", "partial", "verified"), "invalid local fetch receipt")
    if not partial or current["status"] == "verified":
        require((out / "copyback_verified.json").is_file(), "remote copyback not published")
        result = verify_copyback(out)
        job = read_json(out / "job_manifest.json")
        run = current["run_identity"]
        if job.get("identity") is not None:
            require(all(job.get(key) == run[key] for key in ("run_id", "experiment", "mode"))
                    and job["identity"]["release_sha256"] == run["release_sha256"],
                    "fetched job identity mismatch")
        else:
            require(job.get("status") == result["status"] == "fail", "successful fetch lacks job identity")
        manifest_sha = sha256(out / "copyback_manifest.json")
        require(current["status"] != "verified" or current["manifest_sha256"] == manifest_sha,
                "verified fetch changed locally")
        current.update(status="verified", manifest_sha256=manifest_sha, job_status=result["status"])
    else:
        current["status"] = "partial"
    atomic_json(receipt, current)
    return {"status": current["status"], "receipt": str(receipt),
            "diagnostics": str(out.with_name(out.name + ".diagnostics"))}


def gate(release, expected_sha, experiment, canary_dir):
    require(experiment in EXPERIMENTS, "invalid experiment")
    ident = identity(release, expected_sha)
    spec = read_json(Path(release) / "deployment.json")
    out = Path(canary_dir)
    require((out / "copyback_verified.json").is_file(), "canary copyback not published")
    require(verify_copyback(out)["status"] == "pass", "failed copyback receipt")
    job = read_json(out / "job_manifest.json")
    require(job["status"] == "pass" and job["exit_code"] == 0, "canary job failed")
    require(job["identity"] == ident and job["identity_sha256"] == digest(ident), "stale canary identity")
    require(job["experiment"] == experiment and job["mode"] == "canary", "wrong experiment/mode receipt")
    require(job["spec"] == spec, "canary workload/spec drift")
    workers = check_workers(out, job)
    aggregate_value = check_runtime(read_json(out / "aggregate_receipt.json"), job)
    runtime = check_runtime(read_json(out / "profile.json"), job)
    require(aggregate_value["runtime_receipt_sha256"] == sha256(out / "profile.json"), "aggregate hash drift")
    require(runtime["workers"] == [read_json(out / f"rank_{rank:03d}" / "worker_receipt.json") for rank in range(4)],
            "profile worker identity drift")
    require(runtime["analysis_summary_sha256"] == sha256(out / "analysis_summary.json"), "analysis summary hash drift")
    require(aggregate_value["log_sha256"] == sha256(out / "aggregate.log"), "aggregate log drift")
    receipt = read_json(out / "canary_receipt.json")
    require(receipt["status"] == "pass" and receipt["identity"] == ident, "invalid canary receipt")
    for key, value in _receipt_base(job).items():
        require(receipt[key] == value, f"canary {key} drift")
    verify_inventory(out, receipt["evidence"], excluded=("canary_receipt.json", "copyback_manifest.json", "copyback_verified.json"))
    seconds = max(project_seconds(job, workers, aggregate_value["elapsed_seconds"]),
                  math.ceil(runtime["recommended_walltime_s"] / 60) * 60)
    require(receipt["full_walltime_seconds"] == seconds
            and receipt.get("full_budget_status") == budget_status(seconds), "runtime projection drift")
    if experiment == "compression":
        require((out / "arm_inventory.json").is_file(), "missing runtime arm_inventory.json snapshot")
    require(seconds <= 14400, f"canary passed technically, but projected full walltime {seconds}s "
            "exceeds the 4h hard stop; needs review (split the work or prepare a new release)")
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("bundle")
    p.add_argument("--repo-root", required=True)
    p.add_argument("--out-dir", required=True)
    p = subs.add_parser("bootstrap")
    for key in ("source-root", "source-sha", "output-root", "assets-root", "model-root", "sif-path"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--sif-sha", default=SIF_SHA256)
    p = subs.add_parser("prepare")
    for key in ("repo-root", "out-dir"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--spec", dest="spec_path", required=True)
    p.add_argument("--dataset-dir")
    p.add_argument("--assets-dir")
    for command in ("verify", "gate", "stage"):
        p = subs.add_parser(command)
        p.add_argument("--release", required=True)
        p.add_argument("--expected-sha", required=True)
        if command != "verify":
            p.add_argument("--experiment", choices=EXPERIMENTS, required=True)
            p.add_argument("--canary-dir", required=command == "gate")
        if command == "stage":
            for key in ("work-dir", "run-id", "executing-script"):
                p.add_argument("--" + key, required=True)
            p.add_argument("--mode", choices=("canary", "full"), required=True)
            p.add_argument("--walltime-seconds", type=int, required=True)
    for command in ("worker", "aggregate", "finish"):
        p = subs.add_parser(command)
        p.add_argument("--work-dir", required=True)
        if command == "worker":
            p.add_argument("--rank", type=int, required=True)
        if command == "finish":
            p.add_argument("--exit-code", type=int, required=True)
            p.add_argument("--stage", dest="stage_name", required=True)
    for command in ("verify-copyback", "publish"):
        p = subs.add_parser(command)
        p.add_argument("--out-dir", required=True)
    p = subs.add_parser("fetch-begin")
    for key in ("out-dir", "remote", "source", "identity-file"):
        p.add_argument("--" + key, required=True)
    p = subs.add_parser("fetch-finish")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--partial", action="store_true")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        result = globals()[command.replace("-", "_")](**args)
        print(result if command == "gate" else result["status"] if command == "fetch-begin"
              else json.dumps(result, sort_keys=True))
        if command == "finish" and result["status"] != "pass":
            return 1
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
