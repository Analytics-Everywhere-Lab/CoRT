"""Bounded two-stage recognition launcher; planning never loads a model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys

from .common import atomic_json, canonical, digest, read_json, sha256
from .cli import preflight_plan
from .data import validate_dataset
from .execution import canary_admission, require_recognition_full_cohort, verify_loaded_code
from .hpc import build_plan, load_profile, submit_plan, write_plan
from .protocol import GEMMA, inventory, validate_protocol
from .recognition_protocol import is_recognition
from .followup_protocol import is_followup
from .release import verify_release


def prepare(config_path, mode):
    """Create immutable plan/deployment files; submission requires explicit intent."""
    if mode not in ("check", "canary", "full"):
        raise ValueError("mode must be check, canary, or full")
    config = read_json(config_path)
    release, root = Path(config["release_root"]).resolve(), Path(config["output_root"]).resolve()
    if root.is_relative_to(release) or release.is_relative_to(root):
        raise ValueError("output_root and release_root must not overlap")
    sealed = verify_release(release)
    verify_loaded_code(sealed)
    protocol = validate_protocol(read_json(release / "protocol.json"))
    if not is_recognition(protocol):
        raise ValueError("this launcher requires schema 4 or 5 / recognition contract version 3")
    profile = load_profile(config["profile"])
    base = deepcopy(config["deployment"])
    if set(base.get("models", {})) != {GEMMA}:
        raise ValueError("recognition deployment requires exactly Gemma31")
    for spec in base["models"].values():
        for key in ("snapshot", "inventory"):
            if not Path(spec[key]).exists():
                raise ValueError(f"stage the missing model {key}: {spec[key]}")
    runtime = base["runtime"]
    artifact = runtime["image"] if runtime["mode"] == "apptainer" else runtime["python"]
    if not Path(artifact).is_file():
        raise ValueError(f"stage the missing runtime: {artifact}")
    data = validate_dataset(base["data_root"])
    canary_tasks = inventory(data["rows"], protocol, "recognition-canary")
    full_tasks = inventory(data["rows"], protocol, "recognition")
    if mode == "full":
        require_recognition_full_cohort(data["rows"])
    if mode == "check":
        dep = {**base, "deployment_path": str(root / "canary.deployment.json"), "walltime_seconds": 1800}
        gates = preflight_plan(release, dep, "recognition-canary")
        plan = build_plan(profile, dep, "recognition-canary", root / "canary", release, "00:30:00")
        return {"status": "metadata_ready", "canary_rows": len(canary_tasks), "full_rows": len(full_tasks),
                "gpus": plan["worker_count"], "gates": gates,
                "scope": "No GPU execution, runtime compatibility, or scientific outcome is certified."}
    stage = "recognition-canary" if mode == "canary" else "recognition"
    dep = deepcopy(base)
    job_family = "temporal-followup" if is_followup(protocol) else "temporal-recognition"
    dep.update(deployment_path=str(root / f"{mode}.deployment.json"),
               job_name=f"{job_family}-{mode}", walltime_seconds=1800)
    workers = dep.get("resources", {}).get("gpus_per_node", profile["resources"]["gpus_per_node"])
    seconds = 1800
    if mode == "full":
        receipt_path = root / "canary" / "receipt.json"
        dep["canary_receipt"] = str(receipt_path)
        if is_followup(protocol):
            reference_path = root / "canary-reference-comparison.json"
            if not reference_path.is_file():
                raise ValueError("follow-up needs the passing historical canary comparison")
            dep["followup_reference_check"] = str(reference_path)
            dep["followup_reference_check_sha256"] = sha256(reference_path)
        # Bind code, data, protocol, runtime metadata, model inventories, and
        # technical certificates before accepting any measured timing.
        preflight_plan(release, dep, stage)
        receipt = read_json(receipt_path)
        estimate = canary_admission(receipt, receipt["identity"], full_tasks, workers, 23 * 3600)
        seconds = max(3600, math.ceil((estimate + 180) / 300) * 300)
        if seconds >= 23 * 3600:
            raise ValueError("measured full runtime must remain below the 23-hour project limit")
        dep["walltime_seconds"] = seconds
    preflight_plan(release, dep, stage)
    walltime = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    plan = build_plan(profile, dep, stage, root / mode, release, walltime)
    root.mkdir(parents=True, exist_ok=True)
    path = Path(dep["deployment_path"])
    if path.exists():
        if read_json(path) != dep:
            raise ValueError("deployment changed; use a fresh output_root rather than overwriting a run")
    else:
        atomic_json(path, dep)
    plan_dir = root / f"{mode}-plan"
    if plan_dir.exists():
        if read_json(plan_dir / "plan.json") != plan:
            raise ValueError("existing launch plan differs; use a fresh output_root")
    else:
        write_plan(plan, plan_dir)
    return {"status": "planned", "stage": stage, "walltime": walltime,
            "plan_dir": str(plan_dir), "deployment_sha256": digest(dep),
            "rows": len(canary_tasks) if mode == "canary" else len(full_tasks)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "canary", "full"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "check" and args.submit:
        parser.error("check never submits")
    result = prepare(args.config, args.mode)
    if args.submit:
        result["submission"] = submit_plan(Path(result["plan_dir"]), confirm=True)
    print(canonical(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(2)
