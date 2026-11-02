#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable


MANIFEST_NAME = "release_manifest.json"
MANIFEST_KIND = "phase2_reanchored_exact_release_v1"
EXCLUDED_PARTS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _allowed(path: Path) -> bool:
    return (
        not any(part in EXCLUDED_PARTS for part in path.parts)
        and path.suffix not in EXCLUDED_SUFFIXES
        and path.name not in {".DS_Store"}
        and not path.name.startswith("._")
    )


def _walk_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if _allowed(root):
            yield root
        return
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"release source contains a symlink: {path}")
        if path.is_file() and _allowed(path):
            yield path


def selected_files(repo_root: Path) -> list[Path]:
    repo_root = repo_root.expanduser().resolve()
    phase2 = repo_root / "docs/research/lifecycle_memory/phase 2 research"
    optional_roots = {
        phase2 / "jobs/reanchored_jobAR_answer_token",
        phase2 / "jobs/reanchored_jobCR_answer_token_causal_patch",
        phase2 / "jobs/patchaudit_jobD_transport_decomposition",
        phase2 / "jobs/patchaudit_jobE_heldout_component_closure",
    }
    roots = [
        repo_root / "research/lifecycle_memory",
        phase2 / "preregistration",
        phase2 / "jobs/job0",
        phase2 / "jobs/reanchored_jobA_behavior",
        phase2 / "jobs/reanchored_jobAR_answer_token",
        phase2 / "jobs/reanchored_jobB_layerwise",
        phase2 / "jobs/reanchored_jobC_causal_patch",
        phase2 / "jobs/reanchored_jobCR_answer_token_causal_patch",
        phase2 / "jobs/patchaudit_jobD_transport_decomposition",
        phase2 / "jobs/patchaudit_jobE_heldout_component_closure",
        phase2 / "jobs/reanchored_release",
        phase2 / "job3_stage_a_rights_and_provenance_ledger_2026_07_14.md",
    ]
    selected: dict[str, Path] = {}
    for root in roots:
        if root in optional_roots and not root.exists():
            continue
        for path in _walk_files(root):
            relative = str(path.relative_to(repo_root))
            if relative in selected and selected[relative] != path:
                raise ValueError(f"duplicate release path: {relative}")
            selected[relative] = path
    if not selected:
        raise ValueError("release selection is empty")
    return [selected[key] for key in sorted(selected)]


def _entry(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_release(repo_root: Path, output_root: Path, release_id: str) -> dict[str, object]:
    repo_root = repo_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"release output already exists: {output_root}")
    files = selected_files(repo_root)
    output_root.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, object]] = []
    try:
        for source in files:
            relative = source.relative_to(repo_root)
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            entries.append(_entry(target, output_root))
        content_sha256 = canonical_sha256(entries)
        manifest = {
            "kind": MANIFEST_KIND,
            "status": "pass",
            "release_id": release_id,
            "file_count": len(entries),
            "total_bytes": sum(int(row["bytes"]) for row in entries),
            "content_sha256": content_sha256,
            "entries": entries,
        }
        manifest_path = output_root / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_release(output_root, manifest_path)
        return {
            **{key: value for key, value in manifest.items() if key != "entries"},
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def verify_release(root: Path, manifest_path: Path | None = None) -> dict[str, object]:
    root = root.expanduser().resolve()
    manifest_path = (manifest_path or root / MANIFEST_NAME).expanduser().resolve()
    if manifest_path != root / MANIFEST_NAME:
        raise ValueError("release manifest must be at the release root")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != MANIFEST_KIND or manifest.get("status") != "pass":
        raise ValueError("invalid release manifest kind or status")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("release manifest entries are missing")
    for row in entries:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or type(row.get("bytes")) is not int
            or int(row["bytes"]) < 0
            or not isinstance(row.get("sha256"), str)
            or len(str(row["sha256"])) != 64
        ):
            raise ValueError("release manifest contains an invalid entry")
    expected = {str(row.get("path")): row for row in entries if isinstance(row, dict)}
    if len(expected) != len(entries) or any(
        not path or Path(path).is_absolute() or ".." in Path(path).parts
        for path in expected
    ):
        raise ValueError("release manifest has unsafe or duplicate paths")
    observed_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"release tree contains a symlink: {path}")
        if path.is_file() and path != manifest_path:
            observed_paths.add(str(path.relative_to(root)))
    if observed_paths != set(expected):
        missing = sorted(set(expected) - observed_paths)
        extra = sorted(observed_paths - set(expected))
        raise ValueError(f"release tree mismatch: missing={missing[:8]} extra={extra[:8]}")
    observed_entries = []
    for relative in sorted(expected):
        path = root / relative
        row = _entry(path, root)
        if row != expected[relative]:
            raise ValueError(f"release file drift: {relative}")
        observed_entries.append(row)
    if manifest.get("file_count") != len(entries):
        raise ValueError("release file count drift")
    if manifest.get("total_bytes") != sum(int(row["bytes"]) for row in entries):
        raise ValueError("release byte count drift")
    if manifest.get("content_sha256") != canonical_sha256(observed_entries):
        raise ValueError("release content hash drift")
    return {
        "kind": "phase2_reanchored_exact_release_verification",
        "status": "pass",
        "release_id": manifest.get("release_id"),
        "root": str(root),
        "file_count": len(entries),
        "total_bytes": manifest.get("total_bytes"),
        "content_sha256": manifest.get("content_sha256"),
        "manifest_sha256": sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify an exact Phase 2 release tree.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--release-id", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        result = build_release(args.repo_root, args.output_root, args.release_id)
    else:
        result = verify_release(args.root, args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
