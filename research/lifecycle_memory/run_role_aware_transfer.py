"""Transfer core: freeze/evaluate/aggregate; platform owns staging and canary promotion.

runtime(stage_name, mode, rank=0, root=Path('/mnt/rap')) is the platform API.
validate_worker(directory, config_path, mode, rank) requires no evaluator mount.
No training, network, release tooling, or eager torch imports.

Integration: release/ contains this source config and the original cached_validation
config/assets; model/ is the offline HF cache and eval/ the existing public data.
Freeze writes out/{adapters,frozen_config.json,freeze_receipt.json}. Four evaluate
ranks write out/evaluate/rank_NNN; aggregate alone may mount evaluator labels and
writes out/evaluate/{analysis_summary.json,RESULTS.md}. Copyback verification needs
the frozen config, adjacent freeze receipt, worker artifacts and same-release code.
The platform must enforce same-release canary/copyback approval before full work.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback

from research.lifecycle_memory import role_aware_transfer as policy
from research.lifecycle_memory import run_role_aware_policy as base
from research.lifecycle_memory import decoder_cached_validation as data

require = policy.require
WORLD_SIZE = 4
FILES = ("predictions.jsonl", "timings.jsonl", "cold_start.json", "cache_receipt.json",
         "timed_logits_validation.json")
GATES = ("original_native_bit_exact", "legacy_bit_exact", "zero_bit_exact", "state_bit_exact")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append(path, row):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def code_manifest():
    return base.code_manifest()


def mode_rank(mode, rank):
    require(mode in ("canary", "full") and type(rank) is int and 0 <= rank < WORLD_SIZE,
            "invalid mode/rank")


def contract_for(release):
    path = Path(release) / policy.REFERENCE_CONFIG
    require(base.sha256(path) == policy.REFERENCE_SHA, "reference config hash drift")
    config = data.validate_config(read_json(path))
    require(config["model"] == policy.MODEL and config["source_run"]["id"] == "620395"
            and config["fresh_data"]["data_manifest_sha256"] == policy.EVALUATION_SHA,
            "reference model/data provenance drift")
    return base.DataContract(config)


def inventory(contract, data_dir, mode):
    rows = contract.rows(data_dir, mode)
    return {"public": [{k: r[k] for k in policy.IDENTITY} for r in rows],
            "shards": contract.shards(rows), "benchmark_ids": sorted(contract.benchmark(rows, mode))}


def validate_inventory(inv, mode):
    public = inv["public"]
    require(len(public) == (32 if mode == "canary" else 894), "request count drift")
    ids = {r["request_id"] for r in public}
    require(len(ids) == len(public) and all(set(r) == set(policy.IDENTITY) for r in public),
            "public identity inventory drift")
    frames, clusters = {}, {}
    for r in public:
        require(r["source_family"] in data.SOURCE_KINDS
                and r["sampling_unit_kind"] == data.SOURCE_KINDS[r["source_family"]], "source kind drift")
        require(clusters.setdefault(r["cluster_key"], r["source_family"]) == r["source_family"],
                "cluster crosses sources")
        key = (r["cluster_key"], r["source_family"], r["sampling_unit_kind"])
        frame = frames.setdefault(r["frame_key"], {"key": key, "orders": []})
        require(frame["key"] == key, "frame identity drift")
        frame["orders"].append(r["order_id"])
    require(all(sorted(f["orders"]) == ["order_0", "order_1"] for f in frames.values()), "unpaired orders")
    require(inv["shards"] == data.frame_shards(public), "shard selection drift")
    chosen = set(data._source_balanced_frames(public)[:16 if mode == "canary" else 64])
    expected = sorted(r["request_id"] for r in public if r["frame_key"] in chosen)
    require(inv["benchmark_ids"] == expected and len(expected) == (32 if mode == "canary" else 128),
            "benchmark selection drift")


def validate_frozen(config, mode):
    mode_rank(mode, 0)
    policy.validate_config(config)
    require(config["status"] == "adapters_frozen", "config not frozen")
    frozen = config.get("freeze", {})
    require(frozen.get("mode") == mode, "freeze mode mismatch; no canary promotion")
    require(frozen.get("code") == code_manifest(), "frozen code drift")
    require(frozen.get("source_design_sha256") == digest(policy.source_config())
            and frozen.get("source_config_sha256") == base.sha256(base.repo_root() / policy.CONFIG)
            and frozen.get("adapters") == config["adapters"], "freeze binding drift")
    require(frozen.get("reference_config_sha256") == policy.REFERENCE_SHA
            and frozen.get("evaluation_manifest_sha256") == policy.EVALUATION_SHA, "freeze provenance drift")
    require(frozen.get("dataset", {}).get("status") == "pass"
            and frozen["dataset"].get("manifest_sha256") == policy.EVALUATION_SHA
            and frozen["dataset"].get("evaluator_files_verified") is False, "dataset provenance drift")
    validate_inventory(frozen["inventory"], mode)
    return frozen


def validate_freeze_receipt(config_path, config, mode):
    receipt = read_json(Path(config_path).parent / "freeze_receipt.json")
    expected = {"status": "pass", "mode": mode, "experiment_id": policy.EXPERIMENT_ID,
                "config_sha256": base.sha256(config_path), "adapters": config["adapters"],
                "code": config["freeze"]["code"], "training_allowed": False}
    require(receipt == expected, "freeze receipt mismatch")


def freeze(root, mode, *, contract=None):
    mode_rank(mode, 0)
    root = Path(root)
    release, out = root / "release", root / "out"
    source = policy.validate_config(read_json(release / policy.CONFIG))
    require(source == policy.source_config(), "freeze requires pristine source config")
    require(not (out / "frozen_config.json").exists() and not (out / "freeze_receipt.json").exists()
            and not (out / "adapters").exists(), "refusing to overwrite freeze/adapters")
    contract = contract or contract_for(release)
    receipt = contract.check(root / "eval", public_only=True)
    inv = inventory(contract, root / "eval", mode)
    validate_inventory(inv, mode)
    references = [a for a in contract.config["adapters"] if a["case_id"] == policy.CASE]
    references = sorted(references, key=lambda a: a["seed"])
    config = {**source, "status": "adapters_frozen", "adapters": references}
    policy.validate_config(config)
    for record in references:
        path = release / policy.REFERENCE_ADAPTERS / record["path"]
        require(path.is_file() and not path.is_symlink() and base.sha256(path) == record["sha256"],
                "reference adapter hash/path mismatch")
    config["freeze"] = {"mode": mode, "code": code_manifest(), "adapters": references,
        "source_design_sha256": digest(source), "source_config_sha256": base.sha256(release / policy.CONFIG),
        "reference_config_sha256": policy.REFERENCE_SHA,
        "evaluation_manifest_sha256": policy.EVALUATION_SHA, "dataset": receipt, "inventory": inv}
    validate_frozen(config, mode)
    (out / "adapters").mkdir(parents=True)
    for record in references:
        dest = out / "adapters" / record["path"]
        shutil.copyfile(release / policy.REFERENCE_ADAPTERS / record["path"], dest)
        require(base.sha256(dest) == record["sha256"], "copied adapter hash mismatch")
    write_json(out / "frozen_config.json", config)
    result = {"status": "pass", "mode": mode, "experiment_id": policy.EXPERIMENT_ID,
              "config_sha256": base.sha256(out / "frozen_config.json"), "adapters": references,
              "code": config["freeze"]["code"], "training_allowed": False}
    write_json(out / "freeze_receipt.json", result)
    return result


def variant_order(request_id, phase, repeat=0):
    values = [s["id"] for s in policy.variant_specs()]
    random.Random(digest([policy.EXPERIMENT_ID, request_id, phase, repeat])).shuffle(values)
    return values


def logits_digest(result):
    values = result.get("logits_ab", [])
    require(len(values) == 2 and all(type(v) in (float, int) and math.isfinite(v) for v in values),
            "invalid A/B logits")
    require(result.get("prediction") == ("A" if values[0] >= values[1] else "B")
            and result.get("margin_ab") == (values[0] - values[1]) / math.log(2), "invalid answer readout")
    return base.logits_digest(result)


def schedule_evidence(plan):
    require(len(plan.stages) == 1 and plan.stages[0].cut == 34, "only same-cut L34 pruning allowed")
    stage = plan.stages[0]
    roles = plan.geometry["roles"]
    require(stage.method == "prune" and set(stage.role_counts) == {"F"}
            and all(len(g) == 1 for g in stage.groups), "F-only prune required")
    keep, scene = set(stage.keep), set(roles["F"])
    require(tuple(stage.keep) == tuple(sorted(keep)) and keep <= set(range(plan.original_length))
            and tuple(tuple(g) for g in stage.input_groups) == tuple((i,) for i in range(plan.original_length)),
            "retained coordinates/input groups drift")
    require(set(plan.protected_positions) <= keep and
            set(range(plan.original_length)) - scene <= keep, "protected/non-F positions removed")
    require(len((keep & scene) - set(plan.protected_positions)) == 133, "eligible F retention drift")
    return {"original_length": plan.original_length, "retained_positions": list(stage.keep),
            "protected_positions": list(plan.protected_positions), "cut": 34,
            "role_counts": {k: len(keep & set(v)) for k, v in roles.items()},
            "eligible_scene_retained": 133, "correction_before_prune": True}


class StateCapture:
    """Untimed adapter-boundary observation, before the runtime's prune gather."""
    def __init__(self, adapter):
        self.adapter, self.states = adapter, []

    def __call__(self, hidden):
        require(not self.states and tuple(hidden.shape[:2]) == (1, 1), "readout must be corrected once")
        before = hidden.detach().clone()
        result = self.adapter(hidden)
        self.states.append((before, result.detach().clone()))
        return result


def state_equality(torch, left, right):
    require(len(left.states) == len(right.states) == 1, "missing correction state")
    hashes = []
    for a, b in zip(left.states[0], right.states[0]):
        require(a.shape == b.shape and a.dtype == b.dtype and torch.isfinite(a).all()
                and torch.equal(a, b), "same-seed correction state differs")
        raw = bytes(a.contiguous().view(torch.uint8).cpu().reshape(-1).tolist())
        require(raw == bytes(b.contiguous().view(torch.uint8).cpu().reshape(-1).tolist()),
                "same-seed correction state bytes differ")
        hashes.append(digest({"shape": list(a.shape), "dtype": str(a.dtype),
                              "bytes_sha256": hashlib.sha256(raw).hexdigest()}))
    return {"pre_sha256": hashes[0], "post_sha256": hashes[1], "exact": True}


def technical_gate(engine, prepared, plans, runtime, indexes, adapters, zero, original_native):
    from research.lifecycle_memory.decoder_token_matrix_steering import correction_forward, target_positions
    require(not plans["native"].stages, "native plan must not prune")
    positions = target_positions(prepared.pilot_partition, "readout")
    evidence = schedule_evidence(plans[policy.COMPRESSION])
    require(all(base.rap.correction_indices(p, positions) == positions for p in plans.values()),
            "pre-correction remapping changed")
    results, states = {}, {}
    with engine.torch.no_grad():
        native = base.readout(engine.logits(prepared), prepared, plans["native"])
        require(logits_digest(native) == original_native, "original/session native differs")
        results["native"] = logits_digest(native)
        for compression, plan in plans.items():
            with runtime.activate(plan):
                optimized_plain = engine.logits(prepared)
            plain = engine.matrix_forward(prepared, plan)[0]
            index = indexes.get(plan, "readout", lambda: positions, engine.device)
            with runtime.activate(plan, zero, index):
                zero_logits = engine.logits(prepared)
            require(engine.torch.equal(plain, optimized_plain) and engine.torch.equal(plain, zero_logits),
                    "zero/compression-only identity differs")
            for seed in policy.SEEDS:
                adapter = adapters[(policy.CASE, seed)]
                capture = StateCapture(adapter)
                with runtime.activate(plan, capture, index):
                    output = engine.logits(prepared)
                case = {"id": policy.CASE, "compression_id": compression, "target": "readout"}
                with correction_forward(engine, prepared, adapter, case, schedule=plan) as legacy:
                    require(engine.torch.equal(output, legacy), "optimized/legacy logits differ")
                name = f"{'native' if compression == 'native' else 'F133_L34'}_readout_seed{seed}"
                results[name] = logits_digest(base.readout(output, prepared, plan))
                states[(compression, seed)] = capture
        equal = {str(seed): state_equality(engine.torch, states[("native", seed)],
                                         states[(policy.COMPRESSION, seed)]) for seed in policy.SEEDS}
    return {**{k: True for k in GATES}, "states": equal, "schedule": evidence, "logits": results}


def evaluate(root, mode, rank, *, contract=None, engine_factory=None, adapter_loader=None,
             measure=None, synchronize=None, gate=technical_gate, cache_factory=None,
             runtime_factory=None, index_factory=None, clock=time.perf_counter):
    from research.lifecycle_memory.decoder_cached_benchmark import CachedScheduleCache, benchmark_measurement
    from research.lifecycle_memory.decoder_token_matrix import build_arms
    mode_rank(mode, rank)
    root = Path(root)
    config_path = root / "out/frozen_config.json"
    directory = root / f"out/evaluate/rank_{rank:03d}"
    require(not directory.exists(), "refusing to mix worker outputs")
    directory.mkdir(parents=True)
    try:
        config = read_json(config_path)
        frozen = validate_frozen(config, mode)
        validate_freeze_receipt(config_path, config, mode)
        contract = contract or contract_for(root / "release")
        dataset = contract.check(root / "eval", public_only=True)
        require(dataset == frozen["dataset"] and inventory(contract, root / "eval", mode) == frozen["inventory"],
                "worker dataset/inventory drift")
        rows = contract.rows(root / "eval", mode)
        selected = [r for r in rows if frozen["inventory"]["shards"][r["request_id"]] == rank]
        benchmark = set(frozen["inventory"]["benchmark_ids"])
        specs = {s["id"]: s for s in policy.variant_specs()}
        arms = {a["id"]: a for a in build_arms() if a["id"] in ("native", policy.COMPRESSION)}
        started = clock()
        adapter_dir = root / "out/adapters"
        require({p.name for p in adapter_dir.iterdir()} == {a["path"] for a in config["adapters"]}
                and all(not p.is_symlink() and p.is_file() for p in adapter_dir.iterdir()),
                "adapter directory inventory drift")
        engine = (engine_factory or base._matrix_engine)(config)
        adapters, zero, adapter_receipts = (adapter_loader or base.load_adapters)(engine, config, root / "out/adapters")
        sync = synchronize or engine.torch.cuda.synchronize
        runtime = (runtime_factory or base.rap.OptimizedDecoder)(engine.model, engine.layers, engine.device)
        indexes = (index_factory or base.rap.IndexCache)()
        cache = (cache_factory or CachedScheduleCache)(list(arms.values()), max_geometries=64, clock=clock)
        forwards = {v: base.make_forward(s, adapters, zero, indexes, runtime) for v, s in specs.items()}
        measure = measure or base._io().measure
        originals, representatives, geometry = {}, {}, {}
        with engine.torch.no_grad():
            for row in selected:
                prepared = engine.prepare(row, root / "eval")
                key, payload = engine.geometry(prepared)
                originals[row["request_id"]] = logits_digest(base.readout(engine.logits(prepared), prepared, None))
                geometry[row["request_id"]] = key
                if key not in representatives:
                    identity = engine.matrix_identity_gate(prepared, payload)
                    require(identity.get("status") == "pass" and identity.get("keep_all_ab_logits_bit_exact") is True,
                            "native keep-all gate failed")
                    representatives[key] = row
        def request(row, name, phase):
            return benchmark_measurement(engine, row, root / "eval", arms[specs[name]["compression_id"]], cache,
                phase=phase, forward=forwards[name], postprocess=base.readout, measure=measure,
                clock=clock, synchronize=sync)
        gates, cold, timings_validation = [], [], []
        for filename in ("predictions.jsonl", "timings.jsonl"):
            (directory / filename).touch(exist_ok=False)
        with cache, runtime, engine.torch.no_grad():
            for key, row in representatives.items():
                for name in variant_order(row["request_id"], "warmup"):
                    first = None
                    for repeat in range(policy.MEASUREMENT["warmups"]):
                        output, metrics = request(row, name, "warmup")
                        value = logits_digest(output)
                        require(first is None or first == value, "warmup output drift")
                        first = value
                        cold.append({"request_id": row["request_id"], "variant": name, "repeat_id": repeat,
                                     "logits_sha256": value, "geometry_sha256": key,
                                     "end_to_end_elapsed_ms": metrics["end_to_end_elapsed_ms"]})
            cache.seal()
            indexes.seal()
            runtime.seal()
            predictions = {}
            # Every request's structural/reference gates complete BEFORE any benchmark repeat.
            for row in selected:
                prepared = engine.prepare(row, root / "eval")
                plans = {name: cache.get(engine, prepared, arm).plan for name, arm in arms.items()}
                evidence = gate(engine, prepared, plans, runtime, indexes, adapters, zero, originals[row["request_id"]])
                gates.append({"request_id": row["request_id"], "geometry_sha256": geometry[row["request_id"]], **evidence})
                for name in variant_order(row["request_id"], "prediction"):
                    output, metrics = request(row, name, "timed")
                    require(logits_digest(output) == evidence["logits"][name], "prediction/reference mismatch")
                    record = observation(row, specs[name], rank, metrics)
                    predictions[(row["request_id"], name)] = {**record, **output}
                    append(directory / "predictions.jsonl", {**record, **output})
            del prepared, plans
            setup_s = clock() - started
            repeats = policy.MEASUREMENT["canary_timing_repeats" if mode == "canary" else "timing_repeats"]
            for row in selected:
                if row["request_id"] not in benchmark:
                    continue
                for repeat in range(repeats):
                    for order, name in enumerate(variant_order(row["request_id"], "benchmark", repeat)):
                        output, metrics = request(row, name, "timed")
                        # The synchronized end-to-end timer has stopped before digest/validation work.
                        value = logits_digest(output)
                        require(value == logits_digest(predictions[(row["request_id"], name)]), "timed output drift")
                        record = {**observation(row, specs[name], rank, metrics), "repeat_id": repeat,
                                  "variant_order": order, "logits_sha256": value}
                        append(directory / "timings.jsonl", record)
                        timings_validation.append({k: record[k] for k in ("request_id", "variant", "repeat_id",
                            "rank", "geometry_sha256", "schedule_sha256", "logits_sha256")})
            cache.verify_integrity()
            write_json(directory / "cache_receipt.json", cache.report())
            write_json(directory / "cold_start.json", {"status": "pass", "setup_s": setup_s,
                "gates": gates, "warmups": cold, "optimized": runtime.report(), "index_cache": indexes.report(),
                "adapters": adapter_receipts})
        common = {"status": "pass", "mode": mode, "rank": rank, "experiment_id": policy.EXPERIMENT_ID,
                  "config_sha256": base.sha256(config_path), "code": frozen["code"]}
        write_json(directory / "timed_logits_validation.json", {**common, "outside_timer": True,
            "observations": timings_validation, "files_sha256": {n: base.sha256(directory / n)
                for n in ("predictions.jsonl", "timings.jsonl")}})
        receipt = {**common, "world_size": WORLD_SIZE, "labels_accessed": False, "dataset": dataset,
            "predictions": len(predictions), "timings": len(timings_validation), "work_s": clock() - started,
            "runtime": engine.runtime_receipt, "files_sha256": {n: base.sha256(directory / n) for n in FILES}}
        write_json(directory / "worker_receipt.json", receipt)
        validate_worker(directory, config_path, mode, rank)
        return receipt
    except Exception as exc:
        if not (directory / "error.json").exists():
            write_json(directory / "error.json", {"status": "fail", "error": str(exc), "traceback": traceback.format_exc()})
        raise


def observation(row, spec, rank, metrics):
    require(metrics["plan_cache_hit"] is True, "timed cache miss")
    return {**{k: row[k] for k in policy.IDENTITY}, "rank": rank, "variant": spec["id"],
        **{k: spec.get(k) for k in ("compression_id", "path", "case_id", "seed", "target")},
        **{k: metrics[k] for k in ("geometry_sha256", "schedule_sha256", "plan_cache_hit", "preparation_ms",
            "plan_lookup_ms", "model_forward_ms", "postprocess_ms", "end_to_end_elapsed_ms")},
        **{k: metrics["forward_metrics"].get(k) for k in
            ("device_id", "batch_size", "synchronized", "peak_allocated_bytes", "peak_reserved_bytes")}}


def keyed(rows, fields, expected, description):
    keys = [tuple(r[k] for k in fields) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, description + " inventory mismatch")
    return dict(zip(keys, rows))


def validate_worker(directory, config_path, mode, rank):
    """Portable, label-free post-fetch verifier; uses frozen public selection."""
    mode_rank(mode, rank)
    directory = Path(directory)
    config = read_json(config_path)
    frozen = validate_frozen(config, mode)
    validate_freeze_receipt(config_path, config, mode)
    require(not list(directory.rglob("error.json")), "worker error artifact present")
    receipt = read_json(directory / "worker_receipt.json")
    common = {"status": "pass", "mode": mode, "rank": rank, "experiment_id": policy.EXPERIMENT_ID,
              "config_sha256": base.sha256(config_path), "code": frozen["code"]}
    require(all(receipt.get(k) == v for k, v in {**common, "world_size": WORLD_SIZE,
        "labels_accessed": False, "dataset": frozen["dataset"]}.items()), "worker provenance mismatch")
    require(receipt.get("files_sha256") == {n: base.sha256(directory / n) for n in FILES}, "worker file hash mismatch")
    info = receipt.get("runtime", {})
    require(all(info.get(k) == v for k, v in {"model_revision": policy.MODEL["revision"],
        "torch": policy.MODEL["torch"], "transformers": policy.MODEL["transformers"],
        "bf16": True, "use_cache": False, "attention_backend": "sdpa", "batch_size": 1}.items())
        and isinstance(info.get("device"), str) and info["device"]
        and base.rap.is_digest(info.get("gemma_source_sha256")) and info.get("model_provenance"),
        "runtime/model provenance mismatch")
    inv = frozen["inventory"]
    public = {r["request_id"]: r for r in inv["public"] if inv["shards"][r["request_id"]] == rank}
    specs = {s["id"]: s for s in policy.variant_specs()}
    repeats = policy.MEASUREMENT["canary_timing_repeats" if mode == "canary" else "timing_repeats"]
    pkeys = {(r, v) for r in public for v in specs}
    tkeys = {(r, v, k) for r in set(public) & set(inv["benchmark_ids"]) for v in specs for k in range(repeats)}
    predictions, timings = read_rows(directory / "predictions.jsonl"), read_rows(directory / "timings.jsonl")
    pp = keyed(predictions, ("request_id", "variant"), pkeys, "prediction")
    tt = keyed(timings, ("request_id", "variant", "repeat_id"), tkeys, "timing")
    require(receipt.get("predictions") == len(pkeys) and receipt.get("timings") == len(tkeys), "worker counts mismatch")
    cold = read_json(directory / "cold_start.json")
    require(type(cold.get("setup_s")) in (int, float) and math.isfinite(cold["setup_s"])
            and cold["setup_s"] >= 0 and type(receipt.get("work_s")) in (int, float)
            and math.isfinite(receipt["work_s"]) and receipt["work_s"] >= cold["setup_s"], "setup/work timing drift")
    require(cold.get("status") == "pass" and cold["optimized"]["sealed"] is True
            and cold["index_cache"]["sealed"] is True, "cold/cache setup invalid")
    require(cold.get("adapters") == [{k: a[k] for k in ("case_id", "seed", "sha256")} for a in config["adapters"]],
            "loaded adapter provenance drift")
    gates = keyed(cold["gates"], ("request_id",), {(r,) for r in public}, "gate")
    cache = read_json(directory / "cache_receipt.json")
    geometries = {g["geometry_sha256"] for g in gates.values()}
    require(cache["state"] == "sealed" and cache["geometry_count"] == len(geometries)
            and set(cache["geometries"]) == geometries, "cache geometry inventory drift")
    plans = keyed(cache["plans"], ("geometry_sha256", "arm_id"),
                  {(g, c) for g in geometries for c in ("native", policy.COMPRESSION)}, "plan")
    for g in gates.values():
        require(all(g.get(k) is True for k in GATES) and set(g["states"]) == {"17", "29"}, "technical gate failed")
        require(all(s.get("exact") is True and all(base.rap.is_digest(s.get(k)) for k in ("pre_sha256", "post_sha256"))
                    for s in g["states"].values()), "correction-state evidence invalid")
        require(set(g["logits"]) == set(specs), "gate logits inventory drift")
        plan = plans[(g["geometry_sha256"], policy.COMPRESSION)]["receipt"]["schedule"]
        require(g["schedule"] == serialized_schedule_evidence(plan), "retention evidence drift")
    for record in cache["plans"]:
        from research.lifecycle_memory.run_decoder_token_matrix import checksum
        from research.lifecycle_memory.decoder_token_matrix import build_arms
        require(checksum(record["receipt"]) == record["schedule_sha256"], "schedule receipt hash drift")
        saved = record["receipt"]["schedule"]
        arm = next(a for a in build_arms() if a["id"] == record["arm_id"])
        require(saved["arm"] == arm and saved["sha256"] ==
                checksum({k: v for k, v in saved.items() if k != "sha256"}), "saved schedule identity drift")
        require(checksum(cache["geometries"][record["geometry_sha256"]]) == record["geometry_sha256"],
                "geometry payload hash drift")
        if record["arm_id"] == "native":
            require(record["receipt"]["schedule"]["stages"] == [], "native schedule prunes")
    for row in predictions + timings:
        rid, name = row["request_id"], row["variant"]
        spec, g = specs[name], gates[(rid,)]
        require(all(row.get(k) == v for k, v in public[rid].items()) and row.get("rank") == rank,
                "row identity/rank drift")
        require(all(row.get(k) == spec.get(k) for k in ("compression_id", "path", "case_id", "seed", "target")),
                "row variant provenance drift")
        require(row.get("geometry_sha256") == g["geometry_sha256"] and row.get("schedule_sha256") ==
                plans[(g["geometry_sha256"], spec["compression_id"])]["schedule_sha256"]
                and row.get("plan_cache_hit") is True, "row plan provenance drift")
        components = [row[k] for k in ("preparation_ms", "plan_lookup_ms", "model_forward_ms", "postprocess_ms")]
        total = row["end_to_end_elapsed_ms"]
        require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in [*components, total])
                and total > 0 and row["model_forward_ms"] > 0
                and math.isclose(sum(components), total, rel_tol=1e-7, abs_tol=1e-5), "invalid timing components")
        require(row.get("batch_size") == 1 and row.get("synchronized") is True
                and isinstance(row.get("device_id"), str) and row["device_id"], "invalid timing runtime")
        require(all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                    for k in ("peak_allocated_bytes", "peak_reserved_bytes")), "invalid memory telemetry")
    require(len({r["device_id"] for r in predictions + timings}) == 1, "worker device changed")
    for key, row in pp.items():
        require(logits_digest(row) == gates[(key[0],)]["logits"][key[1]], "prediction/gate logits mismatch")
    for (rid, name, repeat), row in tt.items():
        require(type(repeat) is int and row["variant_order"] == variant_order(rid, "benchmark", repeat).index(name),
                "repeat/order drift")
        require(row["logits_sha256"] == logits_digest(pp[(rid, name)]), "timed/prediction logits mismatch")
    sidecar = read_json(directory / "timed_logits_validation.json")
    require(all(sidecar.get(k) == v for k, v in common.items()) and sidecar.get("outside_timer") is True
            and sidecar.get("files_sha256") == {n: base.sha256(directory / n) for n in
                ("predictions.jsonl", "timings.jsonl")}, "timed validation provenance mismatch")
    evidence = keyed(sidecar["observations"], ("request_id", "variant", "repeat_id"), tkeys, "timed validation")
    for key, row in evidence.items():
        require(set(row) == {"request_id", "variant", "repeat_id", "rank", "geometry_sha256", "schedule_sha256", "logits_sha256"}
                and all(tt[key].get(k) == v for k, v in row.items()), "timed validation row mismatch")
    warm = cold["warmups"]
    representatives = {}
    for row in warm:
        rid, name = row["request_id"], row["variant"]
        require((rid, name) in pp and row["geometry_sha256"] == pp[(rid, name)]["geometry_sha256"]
                and row["logits_sha256"] == logits_digest(pp[(rid, name)]), "warm/cached logits drift")
        require(type(row.get("repeat_id")) is int and type(row.get("end_to_end_elapsed_ms")) in (int, float)
                and math.isfinite(row["end_to_end_elapsed_ms"]) and row["end_to_end_elapsed_ms"] > 0,
                "invalid warmup timing/repeat")
        require(representatives.setdefault(row["geometry_sha256"], rid) == rid, "warmup representative drift")
    keyed(warm, ("request_id", "variant", "repeat_id"), {(r, v, k) for r in representatives.values()
          for v in specs for k in range(policy.MEASUREMENT["warmups"])}, "warmup")
    require(set(representatives) == geometries, "missing geometry warmups")
    return receipt


def serialized_schedule_evidence(plan):
    """Recheck retained coordinates from the saved schedule without torch/data."""
    from types import SimpleNamespace
    require(len(plan["stages"]) == 1, "unexpected pruning stages")
    stage = plan["stages"][0]
    wrapped = SimpleNamespace(**stage)
    wrapped.keep = tuple(g[-1] for g in stage["groups"])
    return schedule_evidence(SimpleNamespace(original_length=plan["original_length"],
        stages=[wrapped], geometry=plan["geometry"], protected_positions=plan["protected_positions"]))


def aggregate(root, mode, *, contract=None, draws=10000):
    mode_rank(mode, 0)
    root = Path(root)
    out, config_path = root / "out/evaluate", root / "out/frozen_config.json"
    require(not (out / "analysis_summary.json").exists() and not (out / "RESULTS.md").exists(),
            "refusing to overwrite aggregate")
    require({p.name for p in out.glob("rank_*")} == {f"rank_{r:03d}" for r in range(WORLD_SIZE)},
            "rank inventory mismatch")
    config = read_json(config_path)
    frozen = validate_frozen(config, mode)
    predictions, timings = [], []
    for rank in range(WORLD_SIZE):
        directory = out / f"rank_{rank:03d}"
        validate_worker(directory, config_path, mode, rank)
        predictions.extend(read_rows(directory / "predictions.jsonl"))
        timings.extend(read_rows(directory / "timings.jsonl"))
    contract = contract or contract_for(root / "release")
    # Public provenance is checked before the only evaluator access in this core.
    require(inventory(contract, root / "eval", mode) == frozen["inventory"], "aggregate public inventory drift")
    checked = contract.check(root / "eval", public_only=mode == "canary")
    require({**checked, "evaluator_files_verified": False} == frozen["dataset"], "aggregate dataset drift")
    result = {"status": "pass", "mode": mode, "experiment_id": policy.EXPERIMENT_ID,
              "config_sha256": base.sha256(config_path), "prediction_rows": len(predictions),
              "timing_rows": len(timings), "scope": policy.SCOPE, "labels_accessed": mode == "full"}
    if mode == "full":
        label_rows = read_rows(root / "eval/evaluator/labels.jsonl")
        labels = {r["request_id"]: r["correct_label"] for r in label_rows}
        public = frozen["inventory"]["public"]
        require(len(labels) == len(label_rows) and set(labels) == {r["request_id"] for r in public},
                "label inventory mismatch")
        result.update(policy.summarize(public, predictions, timings, labels, draws=draws))
        result["per_source"] = {}
        result["per_source_retention"] = {}
        for source in data.SOURCE_FAMILIES:
            subset = [r for r in public if r["source_family"] == source]
            ids = {r["request_id"] for r in subset}
            effects = policy.summarize(subset, [r for r in predictions if r["request_id"] in ids],
                [r for r in timings if r["request_id"] in ids], labels, draws=draws, source_equal=False)
            result["per_source"][source] = effects["accuracy_differences_vs_native"]
            result["per_source_retention"][source] = effects["retention_diagnostics"]
        result["inference"] = {"draws": draws, "seed": policy.BOOTSTRAP["seed"], "alpha": .0125}
    write_json(out / "analysis_summary.json", result)
    with (out / "RESULTS.md").open("x") as handle:
        handle.write(policy.render_markdown(result))
    return result


@contextmanager
def offline_model_environment(root):
    env = {"HF_HOME": str(root / "model"), "HF_HUB_CACHE": str(root / "model/hub"),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, value in saved.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value


def runtime(stage_name, mode, rank=0, root=Path("/mnt/rap")):
    require(stage_name in ("freeze", "evaluate", "aggregate"), "unknown stage; training is forbidden")
    mode_rank(mode, rank)
    require(stage_name == "evaluate" or rank == 0, "non-worker stage requires rank zero")
    root = Path(root)
    if stage_name == "freeze":
        return freeze(root, mode)
    if stage_name == "aggregate":
        return aggregate(root, mode)
    with offline_model_environment(root):
        return evaluate(root, mode, rank)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("runtime", "validate-worker"))
    parser.add_argument("--stage", choices=("freeze", "evaluate", "aggregate"))
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path("/mnt/rap"))
    args = parser.parse_args(argv)
    if args.command == "runtime":
        require(args.stage is not None, "--stage required")
        result = runtime(args.stage, args.mode, args.rank, args.root)
    else:
        result = validate_worker(args.root / f"out/evaluate/rank_{args.rank:03d}",
                                 args.root / "out/frozen_config.json", args.mode, args.rank)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
