"""Killarney-only, post-timer logit validation around the frozen runtime.

No model calls or evaluator-label reads are added. The release wrapper routes
only evaluate/aggregate here; training, freezing and estimators stay unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import math
from pathlib import Path
import traceback

from research.lifecycle_memory import decoder_cached_benchmark as benchmark
from research.lifecycle_memory import role_aware_policy_release as base
from research.lifecycle_memory import run_role_aware_policy as runner

SIDECAR = "timed_logits_validation.json"
SCHEMA = 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(result):
    values = result.get("logits_ab", [])
    require(len(values) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in values),
            "timed logit validation requires two finite logits")
    return runner.logits_digest(result)


class LogitCapture:
    def __init__(self):
        self.forwards = {}
        self.variants = set()
        self.observations = {}

    @contextmanager
    def installed(self):
        original_factory = runner.make_forward
        original_measurement = benchmark.benchmark_measurement

        def make_forward(spec, *args, **kwargs):
            forward = original_factory(spec, *args, **kwargs)
            require(forward not in self.forwards and spec["id"] not in self.variants,
                    "duplicate forward callable/variant")
            self.forwards[forward] = (spec["id"], spec["compression_id"])
            self.variants.add(spec["id"])
            return forward

        def measurement(engine, row, data_dir, arm, cache, **kwargs):
            result, metrics = original_measurement(engine, row, data_dir, arm, cache, **kwargs)
            # The original has synchronized, read out logits, and stopped its timer.
            forward = kwargs.get("forward")
            require(forward in self.forwards, "unregistered measured forward callable")
            variant, compression = self.forwards[forward]
            require(arm["id"] == compression, "measured compression/variant mismatch")
            phase = kwargs.get("phase", "timed")
            require(phase in ("warmup", "timed"), "unknown measurement phase")
            key = (row["request_id"], variant)
            identity = {"request_id": key[0], "variant": variant,
                        "geometry_sha256": metrics["geometry_sha256"],
                        "schedule_sha256": metrics["schedule_sha256"],
                        "logits_sha256": digest(result)}
            observed = self.observations.get(key)
            if observed is None:
                observed = {**identity, "warmup_calls": 0, "timed_calls": 0, "calls": 0}
                self.observations[key] = observed
            require(all(observed[k] == v for k, v in identity.items()),
                    f"repeated logits/geometry mismatch: {key}; "
                    f"expected={observed['logits_sha256']} actual={identity['logits_sha256']}")
            observed[phase + "_calls"] += 1
            observed["calls"] += 1
            return result, metrics

        runner.make_forward = make_forward
        benchmark.benchmark_measurement = measurement
        try:
            yield self
        finally:
            runner.make_forward = original_factory
            benchmark.benchmark_measurement = original_measurement


def _bindings(directory, config_path, mode, rank):
    return {"schema_version": SCHEMA, "status": "pass", "mode": mode, "rank": rank,
            "wrapper_sha256": runner.sha256(__file__), "config_sha256": runner.sha256(config_path),
            "files_sha256": {name: runner.sha256(directory / name) for name in
                             ("predictions.jsonl", "timings.jsonl", "worker_receipt.json")}}


def validate_sidecar(directory, config_path, mode, rank, *, evidence=None):
    """Check hashes and exact observed-call accounting without evaluator labels."""
    directory = Path(directory)
    require(mode in ("canary", "full") and type(rank) is int and 0 <= rank < runner.WORLD_SIZE,
            "invalid validation mode/rank")
    require(not (directory / "error.json").exists(), "worker error artifact present")
    evidence = runner.read_json(directory / SIDECAR) if evidence is None else evidence
    require(all(evidence.get(k) == v for k, v in _bindings(directory, config_path, mode, rank).items()),
            "logit sidecar provenance/hash mismatch")
    predictions = rows(directory / "predictions.jsonl")
    timings = rows(directory / "timings.jsonl")
    receipt = runner.read_json(directory / "worker_receipt.json")
    require(all(receipt.get(k) == v for k, v in {"status": "pass", "mode": mode, "rank": rank,
                "config_sha256": evidence["config_sha256"], "predictions": len(predictions),
                "timings": len(timings)}.items()), "worker receipt/count mismatch")
    expected_variants = {s["id"] for s in runner.rap.variant_specs()}
    by_key = {(p["request_id"], p["variant"]): p for p in predictions}
    request_ids = {key[0] for key in by_key}
    require(predictions and len(by_key) == len(predictions)
            and set(by_key) == {(rid, v) for rid in request_ids for v in expected_variants},
            "prediction key inventory mismatch")
    observed = evidence.get("observations", [])
    captured = {(r["request_id"], r["variant"]): r for r in observed}
    require(len(captured) == len(observed) and set(captured) == set(by_key),
            "sidecar unique-key inventory mismatch")
    require(evidence.get("unique_keys") == len(captured), "sidecar unique-key count mismatch")
    timing_keys = [(t["request_id"], t["variant"], t["repeat_id"]) for t in timings]
    require(timing_keys and len(timing_keys) == len(set(timing_keys)), "duplicate/empty timing keys")
    repeats = runner.rap.MEASUREMENT["canary_timing_repeats" if mode == "canary" else "timing_repeats"]
    timed_ids = {key[0] for key in timing_keys}
    require(all(type(k) is int for _, _, k in timing_keys) and set(timing_keys) ==
            {(rid, v, k) for rid in timed_ids for v in expected_variants for k in range(repeats)},
            "timing repeat/variant inventory mismatch")
    timing_counts = Counter((r, v) for r, v, _ in timing_keys)
    for row in [*predictions, *timings]:
        key = (row["request_id"], row["variant"])
        require(key in captured and row.get("rank") == rank, "observation identity/rank mismatch")
        require(all(row.get(k) == captured[key][k] for k in ("geometry_sha256", "schedule_sha256")),
                "observation geometry/schedule mismatch")
    representatives = defaultdict(dict)
    for key, entry in captured.items():
        require(entry["logits_sha256"] == digest(by_key[key]), "saved prediction/logit digest mismatch")
        warm, timed, calls = (entry.get(k) for k in ("warmup_calls", "timed_calls", "calls"))
        require(all(type(v) is int for v in (warm, timed, calls)) and
                warm in (0, runner.rap.MEASUREMENT["warmups"]), "invalid warmup/call counts")
        # A geometry representative also has one sealed-cache probe before prediction.
        require(timed == 1 + timing_counts[key] + int(warm > 0) and calls == warm + timed,
                "measured call count does not match predictions/timings/probes")
        if warm:
            group = representatives[entry["geometry_sha256"]]
            require(key[1] not in group, "duplicate geometry warmup representative")
            group[key[1]] = key[0]
    require(set(representatives) == {r["geometry_sha256"] for r in captured.values()}
            and all(set(group) == expected_variants and len(set(group.values())) == 1
                    for group in representatives.values()), "incomplete geometry warmup coverage")
    require(evidence.get("calls") == sum(r["calls"] for r in observed), "sidecar total call count mismatch")
    return evidence


def runtime(stage_name, mode, rank=0, *, root=Path("/mnt/rap")):
    require(stage_name in ("evaluate", "aggregate") and mode in ("canary", "full"), "invalid runtime stage/mode")
    require(type(rank) is int and 0 <= rank < runner.WORLD_SIZE, "invalid runtime rank")
    root = Path(root)
    evaluation = root / "out/evaluate"
    config_path = root / "out/frozen_config.json"
    if stage_name == "aggregate":
        for worker in range(runner.WORLD_SIZE):
            validate_sidecar(evaluation / f"rank_{worker:03d}", config_path, mode, worker)
        return base.runtime(stage_name, mode, rank, base=root)
    directory = evaluation / f"rank_{rank:03d}"
    require(not (directory / SIDECAR).exists(), "refusing to replace logit validation evidence")
    try:
        capture = LogitCapture()
        with capture.installed():
            result = base.runtime(stage_name, mode, rank, base=root)
        evidence = {**_bindings(directory, config_path, mode, rank),
                    "unique_keys": len(capture.observations),
                    "calls": sum(r["calls"] for r in capture.observations.values()),
                    "observations": [capture.observations[k] for k in sorted(capture.observations)]}
        validate_sidecar(directory, config_path, mode, rank, evidence=evidence)
        base.safe.atomic_json(directory / SIDECAR, evidence)
        return result
    except Exception as exc:
        if not (directory / "error.json").exists():
            base.safe.atomic_json(directory / "error.json", {"status": "fail", "error": str(exc),
                                  "traceback": traceback.format_exc(), "stage": "timed_logit_validation"})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("runtime")
    command.add_argument("--stage", choices=("evaluate", "aggregate"), required=True)
    command.add_argument("--mode", choices=("canary", "full"), required=True)
    command.add_argument("--rank", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(runtime(args.stage, args.mode, args.rank), sort_keys=True))


if __name__ == "__main__":
    main()
