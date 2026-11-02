#!/usr/bin/env python3
"""Verify that every published file matches the release inventory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from build_manifest import release_paths


def verify(root):
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text())
    failures = []
    seen = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"Invalid path: {relative}")
            continue
        if relative in seen:
            failures.append(f"Duplicate path: {relative}")
            continue
        seen.add(relative)
        path = root / relative
        if not path.is_file() or path.is_symlink() or root.resolve() not in path.resolve().parents:
            failures.append(f"Missing or nonregular file: {relative}")
            continue
        content = path.read_bytes()
        if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            failures.append(f"Checksum mismatch: {relative}")
    try:
        actual = {path.relative_to(root) for path in release_paths(root)}
        failures.extend(f"Unlisted release file: {path}" for path in sorted(actual - seen))
    except ValueError as error:
        failures.append(str(error))
    return failures


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = verify(root)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Verified {len(list(release_paths(root)))} release files, with no unlisted files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
