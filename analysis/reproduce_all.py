"""Recompute the released numerical analyses and bind their verification reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = "manifests/replay_verification.json"
PACKAGES = {
    "static_supplement": {
        "script": "analysis/replay_static_supplement.py",
        "report": "results/rebuilt/static_supplement_replay.json",
        "data": "results/replay_static_supplement",
        "guide": "docs/static_supplement_replay.md",
    },
    "temporal": {
        "script": "analysis/replay_temporal.py",
        "report": "results/rebuilt/temporal_replay.json",
        "data": "results/replay_temporal",
        "guide": "docs/temporal_replay.md",
    },
    "practical": {
        "script": "analysis/replay_practical.py",
        "report": "results/rebuilt/practical_replay.json",
        "data": "results/replay_practical",
        "guide": "docs/practical_replay.md",
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(root=ROOT):
    paths = set((root / "analysis").glob("*.py"))
    paths.update((root / "research").rglob("*.py"))
    for directory in ("aggregate", "replay", "replay_static_supplement", "replay_temporal", "replay_practical"):
        paths.update(p for p in (root / "results" / directory).rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)
            if "__pycache__" not in p.parts}


def verified_receipt(root=ROOT):
    path = root / RECEIPT
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "pass" or receipt.get("input_sha256") != inputs(root):
        return None
    if set(receipt.get("packages", {})) != set(PACKAGES):
        return None
    for name, spec in PACKAGES.items():
        record = receipt["packages"][name]
        report_path = root / spec["report"]
        if (record.get("status") != "pass" or record.get("report") != spec["report"]
                or not report_path.is_file() or record.get("sha256") != digest(report_path)):
            return None
    return receipt


def run_replay(spec, runner, root=ROOT):
    path = root / spec["report"]
    path.unlink(missing_ok=True)
    runner(spec["script"])
    report = json.loads(path.read_text())
    if report.get("status") != "pass":
        raise ValueError(f"Replay did not pass: {spec['script']}")
    return dict(spec, sha256=digest(path), status="pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true", help="Also render the data-only figures")
    args = parser.parse_args()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"

    def run(script):
        subprocess.run([sys.executable, "-B", script], cwd=ROOT, env=env, check=True, timeout=900)

    (ROOT / RECEIPT).unlink(missing_ok=True)
    run("analysis/rebuild_public_results.py")
    before = inputs()
    reports = {}
    for name, spec in PACKAGES.items():
        reports[name] = run_replay(spec, run)
    if before != inputs():
        raise ValueError("Replay changed an input or its implementation")
    receipt = {"schema_version": 1, "status": "pass", "input_sha256": before,
               "packages": reports, "core_static_reference_values": 252,
               "scope": "CPU replay of released statistics; no new inference or dataset reconstruction"}
    (ROOT / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    run("analysis/rebuild_public_results.py")
    if args.figures:
        run("analysis/render_public_figures.py")
    if verified_receipt() is None:
        raise ValueError("Final replay binding did not verify")
    print("All four numerical replay groups passed. See manifests/replay_verification.json.")


if __name__ == "__main__":
    main()
