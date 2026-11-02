"""Small, offline, portable temporal-suite code releases (no datasets or models).

prepare_release(repo_root, protocol_path, out_dir) returns the release manifest.
verify_release(root) validates physical bytes and returns that same manifest.
Neither imports runtime backends nor executes any bundled Python source.

The allowlist is all direct temporal_forecasting/*.py (including untracked new
modules), Git-indexed direct lifecycle_memory/*.py, the two parent package
initializers, and direct JSON profiles in PROFILE_DIR. Working-tree bytes are
sealed, not HEAD bytes; unrelated untracked helpers and nested tests are omitted.
Missing research/__init__.py is represented by an empty release-only initializer
because the source tree may use a namespace package. No source tree is modified.

protocol.json is validated by protocol.validate_protocol and copied byte-for-byte.
release.json contains schema_version=1, sorted files [{path, sha256, bytes}],
protocol_sha256, total_bytes (payload only), and release_sha256. The release hash
is common.digest(manifest without release_sha256), independent of paths outside
the release, timestamps, Git HEAD, and file modes. Payload plus manifest is at
most 30,000,000 bytes. Metadata is bounded and paths must be canonical relative
POSIX paths. Only protocol/profile JSON and the specified Python files are copied.

Publication is a verified same-filesystem directory rename under an exclusive
sibling lock. Existing releases are read-only verified, compared to the requested
snapshot, and never repaired or overwritten, even if incomplete. A trusted parent
directory is required against noncooperating rename races. A stale lock after a
process crash requires operator inspection. No wall-clock durability guarantee
or cryptographic authenticity is implied by the integrity hash.

Verification rejects all unlisted files and symlinks, including extra .py/.json,
but ignores real __pycache__ directories. Releases are logically immutable; file
modes are not changed, so read-only shared releases and local copies both work.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile

from .common import atomic_json, canonical, checked_relative, digest, read_json, sha256
from .protocol import validate_protocol


PACKAGE_DIR = "research/lifecycle_memory/temporal_forecasting"
HELPER_DIR = "research/lifecycle_memory"
PROFILE_DIR = "docs/research/lifecycle_memory/phase 2 research/jobs/temporal_forecasting/profiles"
MAX_RELEASE_BYTES = 30_000_000
MAX_METADATA_BYTES = 1_000_000
_INITIALIZERS = {"research/__init__.py", "research/lifecycle_memory/__init__.py"}
_REQUIRED = _INITIALIZERS | {
    f"{PACKAGE_DIR}/__init__.py", f"{PACKAGE_DIR}/common.py",
    f"{PACKAGE_DIR}/protocol.py", f"{PACKAGE_DIR}/release.py", "protocol.json",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _relative(value):
    _require(isinstance(value, str) and bool(value), "invalid relative path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts and "\\" not in value
             and ":" not in value and str(path) == value and value != "."
             and not any(ord(char) < 32 or ord(char) == 127 for char in value),
             "unsafe/noncanonical relative path")
    return path


def _allowed(value):
    path = _relative(value)
    return (value == "protocol.json" or value in _INITIALIZERS
            or (str(path.parent) in {PACKAGE_DIR, HELPER_DIR} and path.suffix == ".py")
            or (str(path.parent) == PROFILE_DIR and path.suffix == ".json"))


def _file(root, relative):
    _relative(relative)
    path = checked_relative(root, relative)
    _require(stat.S_ISREG(path.lstat().st_mode), f"not a regular file: {relative}")
    _require(path.stat().st_size <= MAX_RELEASE_BYTES, "file exceeds release size limit")
    return path


def _json_file(path):
    _require(stat.S_ISREG(path.lstat().st_mode), "metadata must be a regular nonsymlink file")
    _require(path.stat().st_size <= MAX_METADATA_BYTES, "metadata too large")
    value = read_json(path)
    _require(isinstance(value, dict), "metadata must be a JSON object")
    canonical(value)  # Also rejects nonfinite numbers decoded from overflowing exponents.
    return value


def _source_paths(repo):
    # Git wildcards can include subdirectories, so filter parent paths explicitly.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--", f"{HELPER_DIR}/*.py"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, env=env,
    )
    selected = {name for name in result.stdout.decode("utf-8").split("\0")
                if name and PurePosixPath(name).parent == PurePosixPath(HELPER_DIR)
                and PurePosixPath(name).suffix == ".py"}
    for directory, suffix in ((PACKAGE_DIR, ".py"), (PROFILE_DIR, ".json")):
        source = checked_relative(repo, directory)
        _require(source.is_dir(), f"missing source directory: {directory}")
        found = [item for item in source.iterdir() if item.suffix == suffix]
        _require(bool(found), f"empty source directory: {directory}")
        selected.update(f"{directory}/{item.name}" for item in found)
    selected.update(_INITIALIZERS)
    _require(_REQUIRED - {"protocol.json"} <= selected, "missing required package modules")
    for name in selected:
        _require(_allowed(name), "source is outside the code/profile allowlist")
    return sorted(selected)


def _snapshot(repo, protocol_path):
    _json_protocol = _json_file(protocol_path)
    validate_protocol(_json_protocol)
    protocol_raw = protocol_path.read_bytes()
    # Read JSON again from a consistent snapshot if the caller rewrites the file.
    _require(sha256(protocol_path) == hashlib.sha256(protocol_raw).hexdigest(),
             "protocol changed while reading")
    payload = {"protocol.json": protocol_raw}
    total = len(protocol_raw)
    paths = _source_paths(repo)
    for relative in paths:
        source = checked_relative(repo, relative)
        if relative == "research/__init__.py" and not os.path.lexists(source):
            raw = b""
        else:
            source = _file(repo, relative)
            _require(total + source.stat().st_size <= MAX_RELEASE_BYTES,
                     "release exceeds 30 MB size limit")
            if source.suffix == ".json":
                _json_file(source)
            raw = source.read_bytes()
        total += len(raw)
        _require(total <= MAX_RELEASE_BYTES, "release exceeds 30 MB size limit")
        payload[relative] = raw
    files = [{"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
             for name, raw in sorted(payload.items())]
    manifest = {"schema_version": 1, "files": files, "total_bytes": total,
                "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest()}
    manifest["release_sha256"] = digest(manifest)
    _require(len(canonical(manifest).encode("utf-8")) + 1 <= MAX_METADATA_BYTES,
             "release manifest too large")
    _require(total + len(canonical(manifest).encode("utf-8")) + 1 <= MAX_RELEASE_BYTES,
             "release including manifest exceeds 30 MB size limit")
    return payload, manifest


def _same_existing(out, expected):
    actual = verify_release(out)
    _require(actual["release_sha256"] == expected["release_sha256"],
             "existing release differs from requested source/protocol snapshot")
    return actual


def prepare_release(repo_root: Path, protocol_path: Path, out_dir: Path) -> dict:
    """Publish a new verified code release, or verify an identical existing one."""
    repo = Path(repo_root).resolve(strict=True)
    protocol_path = Path(protocol_path)
    out = Path(out_dir)
    _require(out.name not in {"", ".", ".."}, "invalid output directory")
    payload, manifest = _snapshot(repo, protocol_path)
    if os.path.lexists(out):
        return _same_existing(out, manifest)
    parent = out.parent.resolve(strict=True)
    out = parent / out.name
    lock = parent / ("." + out.name + ".release.lock")
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    stage = None
    try:
        if os.path.lexists(out):
            return _same_existing(out, manifest)
        stage = Path(tempfile.mkdtemp(prefix="." + out.name + ".release-", dir=parent))
        for relative, raw in payload.items():
            target = checked_relative(stage, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(raw)
        atomic_json(stage / "release.json", manifest)
        verify_release(stage)
        if os.path.lexists(out):
            return _same_existing(out, manifest)
        os.rename(stage, out)
        stage = None
        return manifest
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.unlink()


def _physical_files(root):
    files = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            _require(not (Path(directory) / name).is_symlink(), "symlink in release")
        _require(".git" not in dirs + names, ".git is forbidden in a code release")
        dirs[:] = [name for name in dirs if name != "__pycache__"]
        for name in names:
            path = Path(directory) / name
            _require(stat.S_ISREG(path.lstat().st_mode), "nonregular release file")
            files.add(path.relative_to(root).as_posix())
    return files


def verify_release(root: Path) -> dict:
    """Verify metadata, protocol and exact physical source inventory; no imports."""
    root = Path(root)
    _require(not root.is_symlink() and root.is_dir(), "invalid release root")
    root = root.resolve(strict=True)
    manifest_path = _file(root, "release.json")
    manifest = _json_file(manifest_path)
    _require(set(manifest) == {"schema_version", "files", "total_bytes",
                               "protocol_sha256", "release_sha256"}, "invalid release schema")
    _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
             "unsupported release schema")
    for key in ("release_sha256", "protocol_sha256"):
        _require(isinstance(manifest[key], str) and _SHA.fullmatch(manifest[key]),
                 f"invalid {key}")
    _require(manifest["release_sha256"] == digest({key: value for key, value in manifest.items()
                                                 if key != "release_sha256"}),
             "release manifest hash mismatch")
    rows = manifest["files"]
    _require(isinstance(rows, list) and bool(rows), "empty/invalid code inventory")
    paths = []
    total = 0
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {"path", "sha256", "bytes"},
                 "invalid inventory entry")
        _require(_allowed(row["path"]), "inventory includes non-code/non-profile material")
        _require(type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_RELEASE_BYTES,
                 "invalid inventory bytes")
        _require(isinstance(row["sha256"], str) and _SHA.fullmatch(row["sha256"]),
                 "invalid inventory sha256")
        paths.append(row["path"])
        total += row["bytes"]
    _require(paths == sorted(set(paths)), "inventory must be sorted with unique paths")
    _require(_REQUIRED <= set(paths), "required release files missing")
    _require(any(PurePosixPath(path).parent == PurePosixPath(PROFILE_DIR) for path in paths),
             "profile inventory missing")
    _require(type(manifest["total_bytes"]) is int and manifest["total_bytes"] == total,
             "total bytes mismatch")
    _require(total + manifest_path.stat().st_size <= MAX_RELEASE_BYTES,
             "release exceeds 30 MB size limit")
    _require(_physical_files(root) == set(paths) | {"release.json"},
             "extra or missing release files")
    for row in rows:
        path = _file(root, row["path"])
        _require(path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
                 f"release file hash/size drift: {row['path']}")
        if path.suffix == ".json":
            _json_file(path)
    protocol_path = _file(root, "protocol.json")
    _require(sha256(protocol_path) == manifest["protocol_sha256"], "protocol hash mismatch")
    validate_protocol(_json_file(protocol_path))
    return manifest
