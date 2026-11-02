from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


APPLEDOUBLE_MAGIC_VERSION = bytes.fromhex("0005160700020000")


def _validate_sidecar(
    path: Path,
    *,
    relative: str,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, object]:
    payload = path.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_size:
        raise ValueError(
            f"refusing nonmatching sidecar size: {relative} "
            f"expected={expected_size} observed={len(payload)}"
        )
    if payload[: len(APPLEDOUBLE_MAGIC_VERSION)] != APPLEDOUBLE_MAGIC_VERSION:
        raise ValueError(f"refusing file without AppleDouble magic/version: {relative}")
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"refusing nonmatching sidecar SHA-256: {relative} "
            f"expected={expected_sha256} observed={observed_sha256}"
        )
    return {"path": relative, "bytes": len(payload), "sha256": observed_sha256}


def quarantine_sidecars(
    *,
    root: Path,
    quarantine: Path,
    expected_count: int,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    quarantine = quarantine.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is missing: {root}")
    if quarantine == root or root in quarantine.parents:
        raise ValueError("quarantine must be outside the frozen dataset tree")

    paths = sorted(path for path in root.rglob("._*") if path.is_file())
    validation_path = quarantine / "validation_receipt.json"
    final_path = quarantine / "quarantine_receipt.json"
    if not paths and not quarantine.exists() and expected_count == 0:
        return {
            "kind": "appledouble_quarantine_receipt",
            "status": "already_clean",
            "root": str(root),
            "quarantine": str(quarantine),
            "quarantined_count": 0,
        }
    if not paths and not quarantine.exists():
        raise ValueError(
            "expected AppleDouble sidecars are absent and no validated quarantine exists: "
            f"expected={expected_count}"
        )
    if quarantine.exists():
        if not validation_path.is_file():
            raise ValueError("existing quarantine lacks its pre-move validation receipt")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validated = validation.get("files")
        expected_binding = {
            "root": str(root),
            "quarantine": str(quarantine),
            "expected_count": expected_count,
            "expected_size": expected_size,
            "expected_sha256": expected_sha256,
            "appledouble_magic_version_hex": APPLEDOUBLE_MAGIC_VERSION.hex(),
        }
        if not isinstance(validated, list) or any(
            validation.get(key) != value for key, value in expected_binding.items()
        ):
            raise ValueError("existing quarantine validation receipt drifted")
    else:
        if len(paths) != expected_count:
            raise ValueError(
                f"AppleDouble candidate count drifted: expected={expected_count} "
                f"observed={len(paths)}"
            )
        validated = [
            _validate_sidecar(
                path,
                relative=str(path.relative_to(root)),
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
            for path in paths
        ]
        validation = {
            "kind": "appledouble_pre_move_validation",
            "status": "pass",
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "root": str(root),
            "quarantine": str(quarantine),
            "expected_count": expected_count,
            "expected_size": expected_size,
            "expected_sha256": expected_sha256,
            "appledouble_magic_version_hex": APPLEDOUBLE_MAGIC_VERSION.hex(),
            "files": validated,
        }
        quarantine.mkdir(parents=True, exist_ok=False)
        validation_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if len(validated) != expected_count or any(
        not isinstance(row, dict) for row in validated
    ):
        raise ValueError("quarantine validation receipt has the wrong file count")
    planned: set[str] = set()
    for row in validated:
        relative = str(row.get("path") or "")
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in planned
        ):
            raise ValueError(f"unsafe or duplicate quarantine path: {relative}")
        planned.add(relative)
        source = root / relative_path
        target = quarantine / relative_path
        if source.exists() == target.exists():
            raise ValueError(
                f"quarantine recovery requires exactly one source copy: {relative}"
            )
        observed = _validate_sidecar(
            source if source.exists() else target,
            relative=relative,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        if observed != row:
            raise ValueError(f"quarantine validation row drifted: {relative}")

    unplanned = sorted(
        str(path.relative_to(root))
        for path in root.rglob("._*")
        if path.is_file() and str(path.relative_to(root)) not in planned
    )
    if unplanned:
        raise ValueError(f"unplanned AppleDouble candidates remain: {unplanned[:8]}")

    for row in validated:
        source = root / str(row["path"])
        target = quarantine / str(row["path"])
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)

    revalidated = []
    for row in validated:
        relative = str(row["path"])
        target = quarantine / relative
        if not target.is_file() or (root / relative).exists():
            raise ValueError(f"quarantine move did not leave exactly one target copy: {relative}")
        observed = _validate_sidecar(
            target,
            relative=relative,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        if observed != row:
            raise ValueError(f"post-move quarantine validation drifted: {relative}")
        revalidated.append(observed)

    receipt = {
        "kind": "appledouble_quarantine_receipt",
        "status": "pass",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "root": str(root),
        "quarantine": str(quarantine),
        "quarantined_count": len(validated),
        "expected_size": expected_size,
        "expected_sha256": expected_sha256,
        "appledouble_magic_version_hex": APPLEDOUBLE_MAGIC_VERSION.hex(),
        "files": revalidated,
    }
    final_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and quarantine a known set of AppleDouble sidecars."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    receipt = quarantine_sidecars(
        root=args.root,
        quarantine=args.quarantine,
        expected_count=args.expected_count,
        expected_size=args.expected_size,
        expected_sha256=args.expected_sha256,
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
