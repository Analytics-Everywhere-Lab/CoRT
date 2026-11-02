#!/usr/bin/env python3
"""Inventory a public release; this does not create an experimental freeze."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "__pycache__", ".pytest_cache", "outputs", "build", "dist"}


def release_paths(root=ROOT):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIP or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.name in {"RELEASE_MANIFEST.json", ".DS_Store"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not permitted: {relative}")
        if path.is_file():
            yield path


def main() -> None:
    files = []
    for path in release_paths():
        relative = path.relative_to(ROOT)
        content = path.read_bytes()
        files.append({"path": relative.as_posix(), "bytes": len(content),
                      "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"schema_version": 1, "release": "1.2.0",
                "description": "Public artifact inventory, not an experimental preregistration.",
                "files": files}
    (ROOT / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Inventoried {len(files)} files, {sum(f['bytes'] for f in files):,} bytes.")


if __name__ == "__main__":
    main()
