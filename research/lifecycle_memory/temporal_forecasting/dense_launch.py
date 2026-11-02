"""Two-stage launcher for the fixed dense-current study; no model work on login."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys

from .common import atomic_json, digest, read_json
from .cli import preflight_plan
from .execution import canary_admission, verify_loaded_code
from .hpc import build_plan, load_profile, submit_plan, write_plan
from .protocol import GEMMA, inventory, validate_protocol
from .release import verify_release
from .data import validate_dataset


def prepare(config_path, mode):
    """Write immutable allocation files; submission is a separate explicit step."""
    if mode not in ("check", "canary", "full"):
        raise ValueError("mode must be check, canary, or full")
    config = read_json(config_path)
    release = Path(config["release_root"]).resolve()
    root = Path(config["output_root"]).resolve()
    if root.is_relative_to(release) or release.is_relative_to(root):
        raise ValueError("output_root and release_root must not overlap")
    sealed = verify_release(release)
    verify_loaded_code(sealed)
    protocol = validate_protocol(read_json(release / "protocol.json"))
    if protocol.get("schema_version") != 3:
        raise ValueError("this launcher accepts only the dense-current protocol")
    profile = load_profile(config["profile"])
    base = deepcopy(config["deployment"])
    if set(base.get("models", {})) != {GEMMA}:
        raise ValueError("dense-current deployment requires exactly Gemma31")
    for spec in base["models"].values():
        for key in ("snapshot", "inventory"):
            if not Path(spec[key]).exists():
                raise ValueError(f"stage the missing model {key}: {spec[key]}")
    runtime = base["runtime"]
    artifact = runtime["image"] if runtime["mode"] == "apptainer" else runtime["python"]
    if not Path(artifact).is_file():
        raise ValueError(f"stage the missing runtime: {artifact}")
    data = validate_dataset(base["data_root"])
    canary_tasks = inventory(data["rows"], protocol, "current-canary")
    full_error, full_count = None, None
    try:
        full_count = len(inventory(data["rows"], protocol, "current"))
    except ValueError as error:
        full_error = str(error)
    if mode == "check":
        # Also check the actual data/protocol binding and scheduler geometry.
        dep = {**base, "deployment_path": str(root / "canary.deployment.json"),
               "walltime_seconds": 1800}
        gates = preflight_plan(release, dep, "current-canary")
        plan = build_plan(profile, dep, "current-canary", root / "canary", release, "00:30:00")
        return {"status": "metadata_ready", "canary_rows": len(canary_tasks),
                "full_rows": full_count, "full_blocker": full_error,
                "gpus": plan["worker_count"], "gates": gates,
                "scope": "No GPU execution, runtime compatibility, or scientific outcome is certified."}
    if mode == "full" and full_error:
        raise ValueError(full_error)
    stage = "current-canary" if mode == "canary" else "current"
    dep = deepcopy(base)
    dep.update(deployment_path=str(root / f"{mode}.deployment.json"),
               job_name=f"temporal-current-{mode}", walltime_seconds=1800)
    workers = dep.get("resources", {}).get("gpus_per_node", profile["resources"]["gpus_per_node"])
    seconds = 1800
    if mode == "full":
        receipt_path = root / "canary" / "receipt.json"
        dep["canary_receipt"] = str(receipt_path)
        receipt = read_json(receipt_path)
        # Metadata identity is checked below; compute rechecks hardware and bytes.
        estimate = canary_admission(receipt, receipt["identity"],
                                    inventory(data["rows"], protocol, "current"), workers, 23 * 3600)
        seconds = max(3600, math.ceil((estimate + 180) / 300) * 300)
        if seconds > 23 * 3600:
            raise ValueError("measured full runtime exceeds the 23-hour project limit")
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
            "rows": len(canary_tasks) if mode == "canary" else full_count}


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
    from .common import canonical
    print(canonical(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(2)
