"""Entry point: plan/prepare locally; execute models only on allocated GPUs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from .common import atomic_json, digest, read_json, sha256
from .protocol import (CURRENT, FORECAST, default_protocol, matched_protocol, inventory,
                       selected_episodes, validate_dataset_protocol)
from .recognition_protocol import TASKS as RECOGNITION_TASKS, is_recognition


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-protocol")
    init.add_argument("--out", type=Path, required=True)
    init.add_argument("--task", choices=(CURRENT, FORECAST, *RECOGNITION_TASKS))
    init.add_argument("--matched-policy", type=Path)
    init.add_argument("--dense-current", action="store_true",
                      help="nine-frame current-task extension with fixed cuts")
    init.add_argument("--episode-limit", type=int)
    init.add_argument("--minimum-clusters", type=int)
    data = commands.add_parser("prepare-data")
    data.add_argument("--observations", type=Path, required=True)
    data.add_argument("--labels", type=Path, required=True)
    data.add_argument("--exposure", type=Path)
    data.add_argument("--protocol", type=Path)
    data.add_argument("--historical-static", type=Path)
    data.add_argument("--out", type=Path, required=True)
    validate = commands.add_parser("verify-data")
    validate.add_argument("--data", type=Path, required=True)
    validate.add_argument("--hash-assets", action="store_true")
    release = commands.add_parser("release")
    release.add_argument("--repo", type=Path, default=Path.cwd())
    release.add_argument("--protocol", type=Path, required=True)
    release.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify-release")
    verify.add_argument("--release", type=Path, required=True)
    inv = commands.add_parser("inventory-model")
    inv.add_argument("--model", required=True)
    inv.add_argument("--snapshot", type=Path, required=True)
    inv.add_argument("--out", type=Path, required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--profile", required=True)
    plan.add_argument("--deployment", type=Path, required=True)
    plan.add_argument("--release", type=Path, required=True)
    plan.add_argument("--stage", required=True)
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--walltime", required=True)
    plan.add_argument("--gpus", type=int)
    plan.add_argument("--out", type=Path, required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--confirm", action="store_true")
    estimate = commands.add_parser("estimate")
    estimate.add_argument("--release", type=Path, required=True)
    estimate.add_argument("--deployment", type=Path, required=True)
    estimate.add_argument("--stage", choices=("pilot", "exploration", "confirmation", "current", "recognition"), required=True)
    estimate.add_argument("--gpus", type=int, required=True)
    estimate.add_argument("--run-dir", type=Path)
    execute = commands.add_parser("execute")
    for flag in ("release", "deployment", "run-dir"):
        execute.add_argument(f"--{flag}", type=Path, required=True)
    execute.add_argument("--stage", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--context", type=Path, required=True)
    worker.add_argument("--rank", type=int, required=True)
    analyze = commands.add_parser("analyze")
    for flag in ("run-dir", "release", "data"):
        analyze.add_argument(f"--{flag}", type=Path, required=True)
    analyze.add_argument("--draws", type=int)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--exploration-run", type=Path, required=True)
    freeze.add_argument("--release", type=Path, required=True)
    freeze.add_argument("--data", type=Path, required=True)
    freeze.add_argument("--confirm-review", action="store_true")
    freeze.add_argument("--rho-guard-bits", type=float, default=.01)
    freeze.add_argument("--out", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    return p


def freeze_design(exploration_run, release_root, data_root, out, confirm=False, rho_guard_bits=.01):
    from .data import validate_dataset
    from .release import verify_release
    from .execution import verify_loaded_code
    import math

    if not confirm:
        raise ValueError("review pilot/exploration and candidate audit, then pass --confirm-review")
    if not math.isfinite(rho_guard_bits) or rho_guard_bits < 0:
        raise ValueError("rho stability guard must be finite and nonnegative")
    if Path(out).exists():
        raise ValueError("freeze exists; do not overwrite a frozen design")
    release = verify_release(release_root)
    verify_loaded_code(release)
    protocol = read_json(Path(release_root) / "protocol.json")
    dataset = validate_dataset(data_root)
    validate_dataset_protocol(dataset["manifest"], protocol)
    receipt = read_json(Path(exploration_run) / "receipt.json")
    summary = read_json(Path(exploration_run) / "analysis.json")
    if protocol.get("task") == CURRENT and summary.get("data_complete") is not True:
        raise ValueError("current companion freeze needs complete descriptive exploration")
    if receipt.get("status") != "pass" or receipt.get("stage") != "exploration" or receipt["identity"].get("simulation"):
        raise ValueError("need complete real exploration, not a canary/mock")
    metadata_identity = {"release_sha256": release["release_sha256"], "public_sha256": dataset["manifest"]["public_sha256"],
                         "evaluator_sha256": dataset["manifest"]["evaluator_sha256"],
                         "dataset_manifest_sha256": sha256(Path(data_root) / "manifest.json"), "protocol_sha256": digest(protocol)}
    if any(receipt["identity"].get(k) != v for k, v in metadata_identity.items()):
        raise ValueError("freeze provenance mismatch")
    if summary.get("provenance", {}).get("receipt_sha256") != sha256(Path(exploration_run) / "receipt.json"):
        raise ValueError("analysis is not bound to the exploration receipt")
    completion = read_json(Path(exploration_run) / "completion.json")
    if completion.get("analysis_sha256") != sha256(Path(exploration_run) / "analysis.json") or completion.get("receipt_sha256") != sha256(Path(exploration_run) / "receipt.json"):
        raise ValueError("exploration completion/analysis hashes differ")
    selected = selected_episodes(dataset["rows"], protocol, "confirmation")
    result = {"schema_version": 1, **metadata_identity,
              "cuts": summary["cuts"], "episode_ids": [r["episode_id"] for r in selected],
              "primary_family": protocol["primary_family"], "alpha": protocol["alpha"],
              "bootstrap_draws": protocol["bootstrap_draws"], "rho_guard_bits": rho_guard_bits,
              "exploration_receipt_sha256": sha256(Path(exploration_run) / "receipt.json"),
              "exploration_analysis_sha256": sha256(Path(exploration_run) / "analysis.json"),
              "created_at": datetime.now(timezone.utc).isoformat(), "review_acknowledged": True,
              "scope": "timestamped local freeze; no external registration or proof of prior noninspection"}
    inventory(dataset["rows"], protocol, "confirmation", result)
    result["freeze_sha256"] = digest(result)
    atomic_json(out, result)
    return result


def preflight_plan(release_root, deployment, stage):
    """Cheap login-side gates: no model loads and no recursive weight hashing."""
    from .data import validate_dataset
    from .release import verify_release
    from .execution import (require_recognition_full_cohort, verify_loaded_code,
                            verify_recognition_technical_receipt)
    from .followup_protocol import SUITE, is_followup, verify_followup_dataset

    release = verify_release(release_root)
    verify_loaded_code(release)
    protocol = read_json(Path(release_root) / "protocol.json")
    dataset = validate_dataset(deployment["data_root"])
    validate_dataset_protocol(dataset["manifest"], protocol)
    freeze = read_json(deployment["freeze_path"]) if deployment.get("freeze_path") else None
    tasks = inventory(dataset["rows"], protocol, stage, freeze)
    if is_recognition(protocol) and stage == "recognition":
        require_recognition_full_cohort(dataset["rows"])
    required = {t["model_key"] for t in tasks}
    if not required <= deployment.get("models", {}).keys():
        raise ValueError(f"deployment is missing required models: {sorted(required-set(deployment.get('models', {})))}")
    identity = {"release_sha256": release["release_sha256"],
                "public_sha256": dataset["manifest"]["public_sha256"], "protocol_sha256": digest(protocol),
                "evaluator_sha256": dataset["manifest"]["evaluator_sha256"],
                "dataset_manifest_sha256": sha256(Path(deployment["data_root"]) / "manifest.json")}
    verify_followup_dataset(protocol, identity["dataset_manifest_sha256"])
    if is_followup(protocol):
        identity["recognition_suite"] = SUITE
    base = stage.removesuffix("-canary")
    if base not in ("pilot", "current", "recognition"):
        if deployment.get("pilot_review_approved") is not True or not deployment.get("pilot_receipt"):
            raise ValueError("complete and review the pilot before planning exploration or confirmation")
        pilot = read_json(deployment["pilot_receipt"])
        if pilot.get("status") != "pass" or pilot.get("stage") != "pilot" or pilot.get("identity", {}).get("simulation"):
            raise ValueError("pilot receipt is not a complete real pilot")
        if any(pilot["identity"].get(k) != v for k, v in identity.items()):
            raise ValueError("pilot receipt does not match release/data/protocol")
        pilot_summary = read_json(Path(deployment["pilot_receipt"]).parent / "analysis.json")
        if pilot_summary.get("provenance", {}).get("receipt_sha256") != sha256(deployment["pilot_receipt"]):
            raise ValueError("pilot analysis is absent or stale")
    if base == "confirmation":
        if freeze.get("freeze_sha256") != digest({k: v for k, v in freeze.items() if k != "freeze_sha256"}):
            raise ValueError("freeze hash mismatch")
        if any(freeze.get(k) != v for k, v in identity.items()):
            raise ValueError("freeze does not match release/data/protocol")
        if freeze.get("episode_ids") != [r["episode_id"] for r in selected_episodes(dataset["rows"], protocol, "confirmation")]:
            raise ValueError("confirmation selection differs from freeze")
    if not stage.endswith("-canary"):
        if not deployment.get("canary_receipt"):
            raise ValueError("full run needs the completed matching-stage canary receipt")
        canary = read_json(deployment["canary_receipt"])
        if canary.get("status") != "pass" or canary.get("stage") != stage + "-canary" or canary.get("identity", {}).get("simulation"):
            raise ValueError("canary status/stage invalid")
        if any(canary["identity"].get(k) != v for k, v in identity.items()):
            raise ValueError("canary does not match release/data/protocol")
        if is_recognition(protocol):
            from .execution import recognition_canary_binding

            recognition_canary_binding(canary, deployment, tasks)
            if canary.get("task_inventory_sha256") != digest(inventory(dataset["rows"], protocol, "recognition-canary")):
                raise ValueError("recognition canary inventory differs from the frozen design")
            verify_recognition_technical_receipt(deployment["canary_receipt"], canary)
        if is_followup(protocol):
            from .followup_compare import require_canary_reference

            require_canary_reference(protocol, deployment, canary)
    return {"status": "pass", "tasks": len(tasks), "identity": identity}


def main(argv=None):
    args = parser().parse_args(argv)
    command = args.command
    if command == "init-protocol":
        if args.out.exists():
            raise ValueError("refusing to overwrite existing protocol")
        if (args.task is None) != (args.matched_policy is None):
            raise ValueError("explicit task and matched-policy must be supplied together")
        recognition = args.task in RECOGNITION_TASKS
        if not recognition and (args.episode_limit is not None or args.minimum_clusters is not None):
            raise ValueError("episode-limit/minimum-clusters are recognition-only options")
        if args.dense_current:
            if args.task != CURRENT or args.matched_policy is None:
                raise ValueError("dense-current requires --task current_interaction and --matched-policy")
            from .protocol import dense_current_protocol
            result = dense_current_protocol(read_json(args.matched_policy))
        elif recognition:
            from .recognition_protocol import recognition_protocol

            result = recognition_protocol(args.task, read_json(args.matched_policy),
                episode_limit=96 if args.episode_limit is None else args.episode_limit,
                minimum_clusters=4 if args.minimum_clusters is None else args.minimum_clusters)
        else:
            result = (matched_protocol(args.task, read_json(args.matched_policy))
                      if args.task else default_protocol())
        atomic_json(args.out, result)
    elif command == "prepare-data":
        from .data import prepare_dataset
        result = prepare_dataset(args.observations, args.labels, args.out, args.exposure,
                                 protocol=read_json(args.protocol) if args.protocol else None,
                                 historical_static_path=args.historical_static)
    elif command == "verify-data":
        from .data import validate_dataset
        result = validate_dataset(args.data, verify_assets=args.hash_assets)["manifest"]
    elif command == "release":
        from .release import prepare_release
        result = prepare_release(args.repo, args.protocol, args.out)
    elif command == "verify-release":
        from .release import verify_release
        result = verify_release(args.release)
    elif command == "inventory-model":
        from .execution import model_inventory
        result = model_inventory(args.model, args.snapshot, args.out)
    elif command == "plan":
        from .hpc import build_plan, load_profile, write_plan
        from .release import verify_release
        dep = read_json(args.deployment)
        if dep.get("deployment_path") != str(args.deployment.resolve()):
            raise ValueError("deployment_path must identify this exact deployment JSON")
        preflight_plan(args.release, dep, args.stage)
        result = build_plan(load_profile(args.profile), dep, args.stage, args.run_dir.resolve(),
                            args.release.resolve(), args.walltime, gpus_per_node=args.gpus)
        write_plan(result, args.out)
    elif command == "submit":
        from .hpc import submit_plan
        result = submit_plan(args.plan, confirm=args.confirm)
    elif command == "estimate":
        import math
        from .data import validate_dataset
        from .execution import canary_admission, load_result
        dep = read_json(args.deployment)
        preflight_plan(args.release, dep, args.stage)
        dataset = validate_dataset(dep["data_root"])
        protocol = read_json(args.release / "protocol.json")
        freeze = read_json(dep["freeze_path"]) if dep.get("freeze_path") else None
        tasks = inventory(dataset["rows"], protocol, args.stage, freeze)
        receipt = read_json(dep["canary_receipt"])
        pending = [t for t in tasks if args.run_dir is None or load_result(args.run_dir, t, receipt["identity"]) is None]
        seconds = canary_admission(receipt, receipt["identity"], pending, args.gpus, 23 * 3600)
        rounded = math.ceil((seconds + 180) / 300) * 300
        result = {"status": "estimate_only", "total_tasks": len(tasks), "unfinished_tasks": len(pending),
                  "estimated_seconds_including_reserves": seconds, "walltime_seconds": rounded,
                  "walltime": f"{rounded//3600:02d}:{(rounded%3600)//60:02d}:00",
                  "condition": "valid only for canary-compatible hardware/runtime; recomputed and checked on compute"}
    elif command == "execute":
        from .execution import execute
        result = execute(args.release, args.deployment, args.stage, args.run_dir)
    elif command == "worker":
        from .execution import run_worker
        result = run_worker(args.context, args.rank)
    elif command == "analyze":
        from .execution import analyze_run
        result = analyze_run(args.run_dir, args.release, args.data, draws=args.draws)
    elif command == "freeze":
        result = freeze_design(args.exploration_run, args.release, args.data, args.out, args.confirm_review, args.rho_guard_bits)
    else:
        context_path = args.run_dir / "worker_context.json"
        if context_path.exists():
            context = read_json(context_path)
            expected = context.get("expected_tasks", sum(map(len, context["assignments"])))
            done = len(list((args.run_dir / "results").glob("*/*.json")))
        else:
            expected, done = None, 0
        result = {"rows_written": done, "expected": expected,
                  "receipt": read_json(args.run_dir / "receipt.json") if (args.run_dir / "receipt.json").exists() else None,
                  "failure": read_json(args.run_dir / "failure.json") if (args.run_dir / "failure.json").exists() else None}
    from .common import canonical
    print(canonical(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(2)
