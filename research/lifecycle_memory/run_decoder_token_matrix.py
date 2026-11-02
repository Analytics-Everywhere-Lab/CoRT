"""Offline decoder token matrix and optional learned correction with exact receipts.

The older application-pilot runtime remains unchanged. Compression workers never
open labels; optional steering reads only training labels. Evaluation outcomes
are joined only during aggregation. No KV cache, generation, or scientific gates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import random
import resource
from pathlib import Path
import sys
import time
import traceback

from research.lifecycle_memory import run_decoder_application_pilots as pilot
from research.lifecycle_memory.application_pilot_compression import estimated_decoder_flops

Engine = pilot.Engine
measure = pilot.measure
prediction = pilot.prediction
atomic_json = pilot.atomic_json
append_row = pilot.append_row
read_json = pilot.read_json
read_rows = pilot.read_rows
digest = pilot.digest
selected_rows = pilot.selected_rows
stable_order = pilot.stable_order

ARM_COUNT = 71
WORLD_SIZE = 4
FULL_REQUESTS = 1600
CANARY_FRAMES = 16
BENCHMARK_REQUESTS = 64
BLOCKS = 60
SEED = 0
ROLE_FIELDS = {"F": "full_frame", "C": "candidate_crops", "Q": "literal_user_prompt",
               "P": "answer_prefix", "R": "structural_remainder"}


def operator():
    from research.lifecycle_memory import decoder_token_matrix
    return decoder_token_matrix


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_model_config(config):
    model = config["model"]
    if (model["key"], model["id"], model["revision"]) != (
            pilot.MODEL_KEY, "google/gemma-4-31B-it", pilot.REVISION):
        raise ValueError("model revision changed")
    if config["measurement"]["batch_size"] != 1:
        raise ValueError("batch size must be one")


def validate_config(config):
    """Validate only the new compression contract, never the old pilot validator."""
    validate_model_config(config)
    arms = operator().build_arms()
    ids = [arm["id"] for arm in arms]
    if len(ids) != ARM_COUNT or len(set(ids)) != ARM_COUNT or ids.count("native") != 1:
        raise ValueError("unexpected frozen arm inventory")
    settings = config["compression"]
    declared = settings.get("arms", settings.get("variants"))
    if declared is not None and declared != arms and declared != ids:
        raise ValueError("configuration differs from frozen arm inventory")
    if declared is None and (settings.get("arm_inventory_version"), settings.get("expected_arm_count")) != (1, ARM_COUNT):
        raise ValueError("missing frozen arm inventory version/count")
    for key, expected in {"canary_frames": CANARY_FRAMES,
                          "benchmark_requests": BENCHMARK_REQUESTS,
                          "timing_repeats": 3, "canary_timing_repeats": 2,
                          "warmups": 2}.items():
        if settings.get(key) != expected:
            raise ValueError(f"frozen compression budget changed: {key}")
    if settings.get("seeds", [SEED]) != [SEED]:
        raise ValueError("compression uses only seed zero")
    if settings.get("full_requests", FULL_REQUESTS) != FULL_REQUESTS:
        raise ValueError("full request count changed")
    return arms


def validate_public_rows(rows, *, expected_requests=FULL_REQUESTS):
    if len(rows) != expected_requests or len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("unexpected or duplicate public request inventory")
    frames = defaultdict(list)
    for row in rows:
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("invalid public split")
        frames[row["frame_key"]].append(row)
    for pair in frames.values():
        if (len(pair) != 2 or {r["order_id"] for r in pair} != {"order_0", "order_1"}
                or len({(r["cluster_key"], r["split"]) for r in pair}) != 1):
            raise ValueError("frame must retain both candidate orders and one cluster/split")


def request_layout(rows, mode, world_size=WORLD_SIZE):
    if mode not in ("canary", "full") or world_size != WORLD_SIZE:
        raise ValueError("matrix uses canary/full and four frame shards")
    selected = selected_rows(rows, mode, CANARY_FRAMES)
    frames = sorted({r["frame_key"] for r in selected}, key=stable_order)
    ranks = {frame: index % world_size for index, frame in enumerate(frames)}
    selected = sorted(selected, key=lambda r: (stable_order(r["frame_key"]), r["order_id"]))
    return selected, {r["request_id"]: ranks[r["frame_key"]] for r in selected}


def benchmark_ids(rows, mode):
    selected, _ = request_layout(rows, mode)
    if mode == "canary":
        return {r["request_id"] for r in selected}
    return {r["request_id"] for r in sorted(selected, key=lambda r: stable_order(r["request_id"]))
            [:BENCHMARK_REQUESTS]}


def arm_order(arms, request_index, *, phase="prediction", repeat=0):
    """Randomized cyclic crossover, reversed on alternating complete cycles."""
    ids = [arm["id"] for arm in arms]
    n = len(ids)
    cycle, offset = divmod(request_index, n)
    random.Random(stable_order(f"matrix-v1:{phase}:{repeat}:{cycle // 2}")).shuffle(ids)
    if cycle % 2:
        ids.reverse()
    return ids[offset:] + ids[:offset]


def _lengths(value):
    if isinstance(value, dict):
        if set(value) == set(range(BLOCKS)):
            value = [value[i] for i in range(BLOCKS)]
        elif set(value) == {str(i) for i in range(BLOCKS)}:
            value = [value[str(i)] for i in range(BLOCKS)]
        else:
            raise ValueError("sequence audit must contain exactly all 60 blocks")
    result = list(value)
    if len(result) != BLOCKS or any(type(n) is not int or n <= 0 for n in result):
        raise ValueError("sequence audit must contain 60 positive integer lengths")
    return result


def expected_lengths(schedule):
    return schedule_inventory(schedule)["sequence_lengths"]


def schedule_inventory(schedule):
    """Derive token counts from actual retained groups, never requested targets."""
    lengths = [len(schedule.groups_at(i - 1)) for i in range(BLOCKS)]
    roles = schedule.geometry["roles"]
    stages = []
    for stage in schedule.stages:
        keep = set(stage.keep)
        if not set(schedule.protected_positions) <= keep:
            raise ValueError("schedule removed protected positions")
        stages.append({"cut": stage.cut, "length": len(stage.groups),
            "role_counts": {role: len(keep & set(positions)) for role, positions in roles.items()},
            "selected_role_counts": stage.role_counts})
    return {"sequence_lengths": _lengths(lengths), "stages": stages,
            "final_length": len(schedule.groups_at(BLOCKS - 1))}


def plan_receipt(schedule):
    return {"schedule": schedule.receipt(), **schedule_inventory(schedule)}


def protection_receipt(engine, prepared):
    p = prepared.pilot_partition
    inputs = prepared.choices["A"].inputs
    ids = inputs["input_ids"].detach().cpu().reshape(-1).tolist()
    if len(ids) != p.sequence_length:
        raise ValueError("input IDs and partition length disagree")
    special_ids = set(engine.tokenizer.all_special_ids)
    for config in (engine.model.config, getattr(engine.model.config, "text_config", None)):
        if config is not None:
            for name, value in vars(config).items():
                if name.endswith("token_id") and isinstance(value, int):
                    special_ids.add(value)
                elif name.endswith("token_ids") and isinstance(value, (list, tuple)):
                    special_ids.update(v for v in value if isinstance(v, int))
    # The shared production wrapper distinguishes processor-proved image soft
    # tokens from structural specials. Never exempt an arbitrary special merely
    # because it lies in a visual slot, or a payload ID outside those slots.
    evidence = operator().protection_receipt(prepared,
        tokenizer=engine.tokenizer, special_token_ids=special_ids,
        image_token_id=getattr(engine.model.config, "image_token_id", None),
        protected_positions=p.padding.positions)
    visual = set(p.full_frame.positions) | set(p.candidate_crops.positions)
    if set(evidence["image_payload_positions"]) != visual:
        raise ValueError("official visual partition contains unexpected non-payload IDs")
    protected = set(evidence["protected_positions"])
    eligible = {role: len(set(getattr(p, field).positions) - protected)
                for role, field in ROLE_FIELDS.items()}
    if eligible["P"] or eligible["F"] != 266 or eligible["C"] != 512:
        raise ValueError("unexpected protected visual/prefix geometry: special/image IDs overlap compression roles")
    return {"protected_positions": sorted(protected), "eligible_counts": eligible,
            "role_counts": {role: getattr(p, field).count for role, field in ROLE_FIELDS.items()},
            "special_token_ids": evidence["special_token_ids"], "protection_evidence": evidence,
            "scope": "P/BOS/image boundaries/specials protected; only verified image payload ID inside official F/C slots exempt"}


class MatrixEngine(Engine):
    def geometry(self, prepared):
        protection = protection_receipt(self, prepared)
        payload = {"partition": asdict(prepared.pilot_partition), **protection,
                   "image_position_ids": prepared.choices["A"].inputs["image_position_ids"].detach().cpu().tolist(),
                   "pooling_kernel": self.model.config.vision_config.pooling_kernel_size}
        return checksum(payload), payload

    def schedule(self, prepared, arm, geometry):
        return operator().build_schedule(partition=prepared.pilot_partition,
            image_position_ids=prepared.choices["A"].inputs["image_position_ids"], arm=arm,
            protected_positions=geometry["protected_positions"],
            pooling_kernel=geometry["pooling_kernel"])

    def matrix_forward(self, prepared, schedule):
        with operator().matrix_context(self.layers, schedule) as audit:
            logits = self.logits(prepared)
        actual = _lengths(audit["sequence_lengths"])
        if actual != expected_lengths(schedule):
            raise RuntimeError("observed sequence lengths differ from the actual schedule")
        return logits, audit

    def matrix_identity_gate(self, prepared, geometry):
        keep_all = {"id": "keep_all_identity", "family": "technical",
                    "stages": [{"cut": 34, "targets": {"F": 266, "C": 512,
                                "Q": geometry["eligible_counts"]["Q"],
                                "R": geometry["eligible_counts"]["R"]}, "method": "merge"}]}
        schedule = self.schedule(prepared, keep_all, geometry)
        with self.torch.no_grad():
            native = self.logits(prepared)
            same, audit = self.matrix_forward(prepared, schedule)
            restored = self.logits(prepared)
        if (not self.torch.isfinite(native).all() or not self.torch.equal(native, same)
                or not self.torch.equal(native, restored)):
            raise RuntimeError("native/keep-all/restored logits are not finite and bit-exact")
        if _lengths(audit["sequence_lengths"]) != [prepared.pilot_partition.sequence_length] * BLOCKS:
            raise RuntimeError("keep-all changed the sequence")
        return {"status": "pass", "keep_all_ab_logits_bit_exact": True,
                "zero_changed_bit_exact": True, "native_restored_bit_exact": True,
                "partition": asdict(prepared.pilot_partition), "protection": geometry,
                "scope": "A/B next-token scoring only; no KV cache or generation"}


class ScheduleCache:
    """Cache by all geometry inputs, checking stored plan integrity before reuse."""
    def __init__(self, arms):
        self.arms = arms
        self.entries = {}

    def get(self, engine, prepared):
        started = time.perf_counter()
        key, geometry = engine.geometry(prepared)
        if key != checksum(geometry):
            raise ValueError("geometry checksum is invalid")
        hit = key in self.entries
        costs = {}
        if not hit:
            plans, receipts = {}, {}
            for arm in self.arms:
                before = time.perf_counter()
                plan = engine.schedule(prepared, arm, geometry)
                receipt = plan_receipt(plan)
                expected_lengths(plan)
                costs[arm["id"]] = 1000 * (time.perf_counter() - before)
                plans[arm["id"]] = plan
                receipts[arm["id"]] = receipt
            self.entries[key] = (plans, receipts, checksum(receipts), geometry)
        plans, receipts, receipt_hash, original = self.entries[key]
        if geometry != original or checksum({name: plan_receipt(p) for name, p in plans.items()}) != receipt_hash:
            raise RuntimeError("cached schedule checksum changed")
        return {"key": key, "geometry": geometry, "plans": plans, "receipts": receipts,
                "receipt_sha256": receipt_hash, "cache_hit": hit,
                "plan_ms": {a["id"]: costs.get(a["id"], 0.0) for a in self.arms},
                "lookup_ms": 1000 * (time.perf_counter() - started) - sum(costs.values())}


def _validate_metrics(row):
    for field in ("elapsed_ms", "cuda_elapsed_ms", "model_forward_ms", "plan_construction_ms",
                  "plan_lookup_ms", "preparation_ms", "request_ms_estimate"):
        value = row[field]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (field in ("elapsed_ms", "model_forward_ms") and value == 0)):
            raise ValueError(f"invalid timing: {field}")
    if not math.isclose(row["elapsed_ms"], row["model_forward_ms"], abs_tol=1e-6):
        raise ValueError("forward-only elapsed timing mismatch")
    inclusive = row["elapsed_ms"] + row["plan_construction_ms"] + row["plan_lookup_ms"]
    if not math.isclose(row["request_ms_estimate"], row["preparation_ms"] + inclusive, abs_tol=1e-6):
        raise ValueError("preparation-inclusive timing mismatch")
    _lengths(row["sequence_lengths"])
    if row["batch_size"] != 1 or row["synchronized"] is not True:
        raise ValueError("invalid measurement contract")


def validate_prediction_inventory(predictions, public_rows, experiment, mode, config):
    if experiment != "compression":
        raise ValueError("compression only")
    arms = validate_config(config)
    selected, ranks = request_layout(public_rows, mode)
    public = {r["request_id"]: r for r in selected}
    if len({r["request_id"] for r in public_rows}) != len(public_rows):
        raise ValueError("duplicate public request")
    expected = {(a["id"], SEED, rid, rank) for rid, rank in ranks.items() for a in arms}
    seen = set()
    for row in predictions:
        if type(row["seed"]) is not int or type(row["rank"]) is not int:
            raise ValueError("invalid prediction seed/rank")
        key = (row["variant"], row["seed"], row["request_id"], row["rank"])
        if key in seen:
            raise ValueError(f"duplicate prediction cell: {key}")
        seen.add(key)
        source = public.get(row["request_id"])
        if source is None or any(row[k] != source[k] for k in ("frame_key", "cluster_key", "split", "order_id")):
            raise ValueError("prediction/public identity mismatch")
        if row.get("error") or row["task_id"] != experiment or row["prediction"] not in ("A", "B"):
            raise ValueError("invalid prediction task or answer")
        if not math.isfinite(row["margin_ab"]):
            raise ValueError("invalid margin")
        if row["prediction"] != ("A" if row["margin_ab"] >= 0 else "B"):
            raise ValueError("prediction disagrees with margin")
        _validate_metrics(row)
    if seen != expected:
        raise ValueError(f"incomplete prediction inventory: missing={len(expected-seen)} extra={len(seen-expected)}")
    return {"status": "pass", "cells": len(seen), "expected_cells_sha256": checksum(sorted(expected)),
            "arms": len(arms), "requests": len(selected), "seeds": [SEED], "ranks": list(range(WORLD_SIZE))}


def validate_timing_inventory(timings, public_rows, mode, config):
    arms = validate_config(config)
    _, ranks = request_layout(public_rows, mode)
    benchmark = benchmark_ids(public_rows, mode)
    repeats = config["compression"]["canary_timing_repeats" if mode == "canary" else "timing_repeats"]
    expected = {(a["id"], SEED, rid, ranks[rid], repeat)
                for rid in benchmark for a in arms for repeat in range(repeats)}
    seen = set()
    for row in timings:
        if any(type(row[field]) is not int for field in ("seed", "rank", "repeat_id")):
            raise ValueError("invalid timing seed/rank/repeat")
        key = (row["variant"], row["seed"], row["request_id"], row["rank"], row["repeat_id"])
        if key in seen:
            raise ValueError(f"duplicate timing cell: {key}")
        seen.add(key)
        if (row.get("error") or row["task_id"] != "compression" or row["is_warmup"] is not False
                or row["warmup_complete"] is not True):
            raise ValueError("invalid benchmark timing contract")
        _validate_metrics(row)
        if (not math.isfinite(row["end_to_end_elapsed_ms"])
                or row["end_to_end_elapsed_ms"] < row["elapsed_ms"]
                or row["independent_preparation_and_plan"] is not True):
            raise ValueError("benchmark must independently measure preparation, plan and forward")
    if seen != expected:
        raise ValueError(f"incomplete timing inventory: missing={len(expected-seen)} extra={len(seen-expected)}")
    return {"status": "pass", "cells": len(seen), "benchmark_requests": len(benchmark),
            "repeats": repeats, "expected_cells_sha256": checksum(sorted(expected))}


def projected_compression_seconds(prediction_s, preparation_s, selected_count,
                                  full_requests, benchmark_requests, repeats, *,
                                  benchmark_s, benchmark_cells, arm_count,
                                  setup_s, warmup_s, plan_build_s, lookup_s, bookkeeping_s=0.0):
    """All arms, actual shard benchmark cells, fixed warmups and plan overhead."""
    if selected_count <= 0 or (benchmark_requests > 0 and benchmark_cells <= 0):
        raise ValueError("projection requires prediction and benchmark measurements")
    prediction = (prediction_s + preparation_s + lookup_s + bookkeeping_s) * full_requests / selected_count
    benchmark = benchmark_s / benchmark_cells * benchmark_requests * repeats * arm_count if benchmark_cells else 0.0
    return setup_s + warmup_s + plan_build_s + prediction + benchmark


def walltime_profile(estimate):
    if not math.isfinite(estimate) or estimate <= 0:
        raise ValueError("invalid runtime projection")
    requested = max(3600, math.ceil((1.4 * estimate + 600) / 300) * 300)
    return {"unbuffered_projected_worker_s": estimate,
            "recommended_walltime_s": requested, "walltime_floor_s": 3600,
            "walltime_cap_s": 14400, "runtime_budget_status": "pass" if requested <= 14400 else "needs_review",
            "projection_policy": "1.4 * projected worker seconds + 600; round up to 300s; floor 1h; no truncation above 4h",
            "full_launch_allowed": requested <= 14400}


def _measurement(engine, prepared, plan, *, prep_ms=0.0, build_ms=0.0, lookup_ms=0.0):
    (logits, audit), metrics = measure(engine.torch, lambda: engine.matrix_forward(prepared, plan))
    lengths = _lengths(audit["sequence_lengths"])
    if lengths != expected_lengths(plan):
        raise RuntimeError("unexpected observed token inventory")
    forward_ms = metrics["elapsed_ms"]
    metrics.update(model_forward_ms=forward_ms, preparation_ms=prep_ms,
                   plan_construction_ms=build_ms, plan_lookup_ms=lookup_ms,
                   elapsed_ms=forward_ms,
                   request_ms_estimate=prep_ms + forward_ms + build_ms + lookup_ms,
                   sequence_lengths=lengths,
                   decoder_dense_matmul_flops_estimate=estimated_decoder_flops(engine.model.config.text_config, lengths),
                   timing_scope="synchronized model forward including actual compaction; excludes preparation and host planning",
                   cuda_timing_scope="CUDA events for model forward only; excludes host planning and preparation",
                   request_estimate_scope="shared preparation + actual attributed plan overhead + synchronized forward; not independent end-to-end timing")
    _validate_metrics(metrics)
    return logits, metrics


def benchmark_measurement(engine, row, data_dir, arm):
    """A true, independent prepare + build + forward repetition for one arm."""
    engine.torch.cuda.synchronize()
    started = time.perf_counter()
    before = time.perf_counter()
    prepared = engine.prepare(row, data_dir)
    engine.torch.cuda.synchronize()
    prep_ms = 1000 * (time.perf_counter() - before)
    before = time.perf_counter()
    key, geometry = engine.geometry(prepared)
    if key != checksum(geometry):
        raise ValueError("benchmark geometry checksum is invalid")
    plan = engine.schedule(prepared, arm, geometry)
    receipt = plan_receipt(plan)
    plan_ms = 1000 * (time.perf_counter() - before)
    _, metrics = _measurement(engine, prepared, plan, prep_ms=prep_ms, build_ms=plan_ms)
    engine.torch.cuda.synchronize()
    metrics.update(end_to_end_elapsed_ms=1000 * (time.perf_counter() - started),
        independent_preparation_and_plan=True, plan_cache_hit=False,
        end_to_end_timing_scope="independent synchronized preparation + fresh schedule construction + model forward including compaction; no cached prepared input or schedule",
        geometry_sha256=key, schedule_sha256=checksum(receipt))
    return metrics


def compression_worker(args, config, engine, rows, out):
    arms = validate_config(config)
    arms_by_id = {a["id"]: a for a in arms}
    selected_all, ranks = request_layout(rows, args.mode, args.world_size)
    selected = [(i, r) for i, r in enumerate(selected_all) if ranks[r["request_id"]] == args.rank]
    if not selected:
        raise RuntimeError("empty matrix worker")
    out.mkdir(parents=True, exist_ok=True)
    for filename in ("predictions.jsonl", "timings.jsonl", "plan_receipts.jsonl", "warmups.jsonl"):
        (out / filename).touch(exist_ok=False)
    setup_start = time.perf_counter()
    first = engine.prepare(selected[0][1], args.data_dir)
    _, geometry = engine.geometry(first)
    technical = engine.matrix_identity_gate(first, geometry)
    engine.torch.cuda.synchronize()
    setup_s = time.perf_counter() - setup_start
    atomic_json(out / "technical_gate.json", technical)
    del first
    cache = ScheduleCache(arms)
    warmed, geometry_receipts = set(), {}
    totals = {k: 0.0 for k in ("prediction_s", "preparation_s", "benchmark_s", "warmup_s", "plan_build_s", "lookup_s")}
    done = timing_count = 0
    benchmark = benchmark_ids(rows, args.mode)
    repeats = config["compression"]["canary_timing_repeats" if args.mode == "canary" else "timing_repeats"]
    started = time.perf_counter()
    for index, row in selected:
        before = time.perf_counter()
        prepared = engine.prepare(row, args.data_dir)
        engine.torch.cuda.synchronize()
        prep_ms = 1000 * (time.perf_counter() - before)
        totals["preparation_s"] += prep_ms / 1000
        entry = cache.get(engine, prepared)
        key, plans = entry["key"], entry["plans"]
        totals["plan_build_s"] += sum(entry["plan_ms"].values()) / 1000
        totals["lookup_s"] += entry["lookup_ms"] / 1000
        geometry_receipts[key] = {"geometry": entry["geometry"], "arms": entry["receipts"],
                                  "arms_sha256": entry["receipt_sha256"]}
        atomic_json(out / "compression_plans.json", geometry_receipts)
        append_row(out / "plan_receipts.jsonl", {"request_id": row["request_id"], "rank": args.rank,
            "seed": SEED, "geometry_sha256": key, "arms_sha256": entry["receipt_sha256"],
            "cache_hit": entry["cache_hit"], "plan_construction_ms": entry["plan_ms"],
            "plan_lookup_ms": entry["lookup_ms"], "preparation_ms": prep_ms})
        if not warmed:
            before = time.perf_counter()
            for repeat in range(config["compression"]["warmups"]):
                for order_index, variant in enumerate(arm_order(arms, 0, phase="warmup", repeat=repeat)):
                    with engine.torch.no_grad():
                        _, audit = engine.matrix_forward(prepared, plans[variant])
                    if _lengths(audit["sequence_lengths"]) != expected_lengths(plans[variant]):
                        raise RuntimeError("warmup sequence audit failed")
                    append_row(out / "warmups.jsonl", {"variant": variant, "rank": args.rank,
                        "seed": SEED, "repeat_id": repeat, "order_index": order_index,
                        "request_id": row["request_id"], "geometry_sha256": key})
            engine.torch.cuda.synchronize()
            totals["warmup_s"] += time.perf_counter() - before
            warmed.add(key)
        for order_index, variant in enumerate(arm_order(arms, index)):
            logits, metrics = _measurement(engine, prepared, plans[variant], prep_ms=prep_ms,
                build_ms=entry["plan_ms"][variant], lookup_ms=entry["lookup_ms"] / len(arms))
            totals["prediction_s"] += metrics["model_forward_ms"] / 1000
            metrics.update(geometry_sha256=key, schedule_sha256=checksum(entry["receipts"][variant]),
                           plan_cache_hit=entry["cache_hit"], order_index=order_index)
            append_row(out / "predictions.jsonl", prediction(row, logits, experiment="compression",
                variant=variant, seed=SEED, rank=args.rank, metrics=metrics))
            done += 1
        del prepared
        if row["request_id"] in benchmark:
            for repeat in range(repeats):
                # Separate randomized whole-arm passes avoid arm-local repeat blocks.
                for order_index, variant in enumerate(arm_order(arms, index, phase="benchmark", repeat=repeat)):
                    metrics = benchmark_measurement(engine, row, args.data_dir, arms_by_id[variant])
                    if metrics["geometry_sha256"] != key or metrics["schedule_sha256"] != checksum(entry["receipts"][variant]):
                        raise RuntimeError("independent benchmark geometry or plan changed")
                    totals["benchmark_s"] += metrics["end_to_end_elapsed_ms"] / 1000
                    append_row(out / "timings.jsonl", {"task_id": "compression", "seed": SEED,
                        "request_id": row["request_id"], "variant": variant, "rank": args.rank,
                        "repeat_id": repeat, "order_index": order_index, "is_warmup": False,
                        "warmup_complete": True, **metrics})
                    timing_count += 1
        atomic_json(out / "progress.json", {"status": "running", "rows_done": done,
            "rows_expected": len(selected) * len(arms), "timing_rows": timing_count,
            "requests_done": done // len(arms), "elapsed_s": time.perf_counter() - started})
    elapsed_s = time.perf_counter() - started
    bookkeeping_s = max(0.0, elapsed_s - sum(totals.values()))
    full, full_ranks = request_layout(rows, "full")
    full_ids = {r["request_id"] for r in full if full_ranks[r["request_id"]] == args.rank}
    projection = projected_compression_seconds(totals["prediction_s"], totals["preparation_s"],
        len(selected), len(full_ids), len(full_ids & benchmark_ids(rows, "full")), 3,
        benchmark_s=totals["benchmark_s"], benchmark_cells=timing_count, arm_count=len(arms),
        setup_s=setup_s, warmup_s=totals["warmup_s"], plan_build_s=totals["plan_build_s"],
        lookup_s=totals["lookup_s"], bookkeeping_s=bookkeeping_s)
    atomic_json(out / "cost_projection.json", {**totals, "setup_s": setup_s,
        "bookkeeping_s": bookkeeping_s, "observed_loop_s": elapsed_s,
        "selected_requests": len(selected), "full_shard_requests": len(full_ids),
        "full_shard_benchmark_requests": len(full_ids & benchmark_ids(rows, "full")),
        "arm_count": len(arms), "warmups_per_arm": 2, "full_timing_repeats": 3,
        "benchmark_cells_measured": timing_count, "unique_geometries": len(cache.entries),
        "projected_full_work_s": projection,
        "benchmark_projection_scope": "independent measured end-to-end preparation + fresh planning + forward, all arms and repetitions",
        "bookkeeping_projection_policy": "unattributed loop wall time including receipts/progress scaled by shard request count; conservative for canary benchmark logging",
        "plan_projection_policy": "observed unique geometry builds once; per-request geometry/checksum lookup scaled to full shard"})
    return done, projection, technical


RECEIPT_FILES = ("predictions.jsonl", "timings.jsonl", "plan_receipts.jsonl", "warmups.jsonl",
                 "compression_plans.json", "technical_gate.json", "cost_projection.json")


def worker(args):
    config = read_json(args.config)
    if args.experiment == "compression":
        validate_config(config)
    else:
        from research.lifecycle_memory.decoder_token_matrix_steering import validate_config as validate_steering
        validate_model_config(config)
        validate_steering(config)
    dataset = Path(args.dataset_dir or args.out_dir / "dataset")
    manifest = read_json(dataset / "split_manifest.json")
    public_path = dataset / "public/requests.jsonl"
    if digest(public_path) != manifest["exports"]["public/requests.jsonl"]["sha256"]:
        raise RuntimeError("dataset public identity mismatch")
    if args.experiment == "steering" and digest(dataset / "labels/train.jsonl") != manifest["exports"]["labels/train.jsonl"]["sha256"]:
        raise RuntimeError("training label identity mismatch")
    rows = read_rows(public_path)
    validate_public_rows(rows)
    if args.experiment == "steering":
        from research.lifecycle_memory.decoder_token_matrix_steering import _public_only
        _public_only(rows)
    out = args.out_dir / f"rank_{args.rank:03d}"
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in (*RECEIPT_FILES, "worker_receipt.json", "failure.json")):
        raise RuntimeError("refusing to mix matrix outputs with a previous invocation")
    started = time.perf_counter()
    engine = MatrixEngine(config)
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    if args.experiment == "compression":
        count, projection, technical = compression_worker(args, config, engine, rows, out)
        receipt_files = RECEIPT_FILES
    else:
        from research.lifecycle_memory.decoder_token_matrix_steering import steering_worker
        count, projection, technical = steering_worker(args, config, engine, rows, dataset, out)
        receipt_files = tuple(sorted(str(path.relative_to(out)) for path in out.rglob("*")
                                     if path.is_file() and path != out / "progress.json"))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    files = {name: digest(out / name) for name in receipt_files}
    atomic_json(out / "worker_receipt.json", {"status": "pass", "experiment": args.experiment,
        "mode": args.mode, "rank": args.rank, "world_size": args.world_size,
        "prediction_rows": count, "timing_rows": len(read_rows(out / "timings.jsonl")),
        "config_sha256": digest(args.config), "split_manifest_sha256": digest(dataset / "split_manifest.json"),
        "load_s": load_s, "work_s": time.perf_counter() - started, "projected_full_work_s": projection,
        "max_host_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "process_cpu_seconds": usage.ru_utime + usage.ru_stime, "technical_gate": {"status": technical["status"]},
        "runtime": engine.runtime_receipt, "files_sha256": files,
        "predictions_sha256": files["predictions.jsonl"], "timings_sha256": files["timings.jsonl"]})
    atomic_json(out / "progress.json", {"status": "complete", "prediction_rows": count})


def validate_plan_receipts(rankdir, rank, selected, predictions, timings, arms):
    plans = read_json(rankdir / "compression_plans.json")
    expected_ids = {r["request_id"] for r in selected}
    rows = read_rows(rankdir / "plan_receipts.jsonl")
    if len(rows) != len(expected_ids) or {r["request_id"] for r in rows} != expected_ids:
        raise ValueError("incomplete or duplicate request plan receipts")
    by_request = {r["request_id"]: r for r in rows}
    if set(plans) != {r["geometry_sha256"] for r in rows}:
        raise ValueError("plan geometry inventory mismatch")
    for key, bundle in plans.items():
        if checksum(bundle["geometry"]) != key or checksum(bundle["arms"]) != bundle["arms_sha256"]:
            raise ValueError("plan checksum mismatch")
        if set(bundle["arms"]) != {a["id"] for a in arms}:
            raise ValueError("plan arm inventory mismatch")
    for receipt in rows:
        bundle = plans[receipt["geometry_sha256"]]
        if receipt["rank"] != rank or receipt["seed"] != SEED or receipt["arms_sha256"] != bundle["arms_sha256"]:
            raise ValueError("plan request identity mismatch")
        if set(receipt["plan_construction_ms"]) != {a["id"] for a in arms}:
            raise ValueError("plan construction inventory mismatch")
        if (not isinstance(receipt["cache_hit"], bool)
                or any(not math.isfinite(ms) or ms < 0 for ms in receipt["plan_construction_ms"].values())
                or not math.isfinite(receipt["plan_lookup_ms"]) or receipt["plan_lookup_ms"] < 0):
            raise ValueError("invalid plan cost receipt")
    for row in [*predictions, *timings]:
        receipt = by_request[row["request_id"]]
        if row["geometry_sha256"] != receipt["geometry_sha256"]:
            raise ValueError("request geometry mismatch")
        plan = plans[row["geometry_sha256"]]["arms"][row["variant"]]
        if checksum(plan) != row["schedule_sha256"] or _lengths(plan["sequence_lengths"]) != row["sequence_lengths"]:
            raise ValueError("prediction/timing schedule mismatch")
    warmups = read_rows(rankdir / "warmups.jsonl")
    expected = {(a["id"], SEED, rank, repeat) for a in arms for repeat in range(2)}
    actual = [(r["variant"], r["seed"], r["rank"], r["repeat_id"]) for r in warmups]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("warmup inventory mismatch")
    first_id = selected[0]["request_id"]
    if any(r["request_id"] != first_id or r["geometry_sha256"] != by_request[first_id]["geometry_sha256"] for r in warmups):
        raise ValueError("warmup request/geometry mismatch")


def efficiency_summary(predictions):
    # Q/R special-token membership may vary. Reuse the established summary per
    # exact geometry instead of claiming a single invariant token schedule.
    groups = defaultdict(list)
    for row in predictions:
        groups[row["geometry_sha256"]].append(row)
    return {"by_geometry": {key: pilot.efficiency_summary(rows) for key, rows in sorted(groups.items())},
            "geometry_count": len(groups), "timing_scope": "elapsed_ms/model_forward_ms exclude preparation and host planning; repeated end_to_end_elapsed_ms includes independent preparation, fresh planning and forward"}


def _worker_outputs(args, dataset, rank, required_files, *, exact_files=True):
    rankdir = args.out_dir / f"rank_{rank:03d}"
    if (rankdir / "failure.json").exists():
        raise RuntimeError(f"worker {rank} has a failure receipt")
    receipt = read_json(rankdir / "worker_receipt.json")
    expected = {"status": "pass", "experiment": args.experiment, "mode": args.mode,
        "rank": rank, "world_size": WORLD_SIZE, "config_sha256": digest(args.config),
        "split_manifest_sha256": digest(dataset / "split_manifest.json")}
    if any(receipt.get(k) != v for k, v in expected.items()) or receipt["technical_gate"]["status"] != "pass":
        raise RuntimeError(f"worker {rank} receipt contract failed")
    files = set(receipt["files_sha256"])
    if not set(required_files) <= files or (exact_files and files != set(required_files)):
        raise RuntimeError("worker receipt file inventory mismatch")
    for name, sha in receipt["files_sha256"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or digest(rankdir / path) != sha:
            raise RuntimeError(f"worker output changed after receipt: {name}")
    if read_json(rankdir / "technical_gate.json")["status"] != "pass":
        raise RuntimeError("worker technical gate failed")
    predictions = read_rows(rankdir / "predictions.jsonl")
    timings = read_rows(rankdir / "timings.jsonl")
    if len(predictions) != receipt["prediction_rows"] or len(timings) != receipt["timing_rows"]:
        raise RuntimeError("worker count mismatch")
    if any(r["rank"] != rank for r in [*predictions, *timings]):
        raise RuntimeError("worker file contains another rank")
    for field in ("load_s", "work_s", "projected_full_work_s"):
        if not math.isfinite(receipt[field]) or receipt[field] < 0:
            raise RuntimeError(f"invalid worker cost: {field}")
    return receipt, predictions, timings


def aggregate_steering_run(args):
    from research.lifecycle_memory import decoder_token_matrix_steering as steering
    config = read_json(args.config)
    validate_model_config(config)
    steering.validate_config(config)
    dataset = Path(args.dataset_dir or args.out_dir / "dataset")
    manifest = read_json(dataset / "split_manifest.json")
    for name in ("public/requests.jsonl", "labels/train.jsonl", "evaluator/all_labels.jsonl"):
        if digest(dataset / name) != manifest["exports"][name]["sha256"]:
            raise RuntimeError(f"dataset identity mismatch: {name}")
    public = read_rows(dataset / "public/requests.jsonl")
    validate_public_rows(public)
    selected = steering.evaluation_rows(public, args.mode, config)
    sources = {r["request_id"]: r for r in selected}
    tasks = [(case["id"], seed) for seed in steering.SEEDS for case in steering.cases()]
    task_ranks = {task: index % WORLD_SIZE for index, task in enumerate(tasks)}
    expected_p = {(*key, task_ranks[key[:2]]) for key in steering.expected_prediction_keys(public, args.mode, config)}
    expected_t = {(*key, task_ranks[key[:2]]) for key in steering.expected_timing_keys(public, args.mode, config)}
    predictions, timings, receipts = [], [], []
    for rank in range(WORLD_SIZE):
        required = ["predictions.jsonl", "timings.jsonl", "technical_gate.json"]
        for case, seed in tasks[rank::WORLD_SIZE]:
            required.extend(f"{case}_seed{seed}/{name}" for name in ("training_receipt.json", "zero_adapter_gate.json"))
        receipt, rank_predictions, rank_timings = _worker_outputs(args, dataset, rank, required, exact_files=False)
        for case, seed in tasks[rank::WORLD_SIZE]:
            taskdir = args.out_dir / f"rank_{rank:03d}" / f"{case}_seed{seed}"
            training = read_json(taskdir / "training_receipt.json")
            checkpoint = Path(training["checkpoint"])
            if (training["status"] != "pass" or checkpoint.name != str(checkpoint)
                    or digest(taskdir / checkpoint) != training["checkpoint_sha256"]):
                raise RuntimeError("steering training/checkpoint receipt failed")
            zero = read_json(taskdir / "zero_adapter_gate.json")
            if zero["status"] != "pass" or zero["ab_logits_bit_exact"] is not True:
                raise RuntimeError("zero steering correction failed identity")
        predictions.extend(rank_predictions)
        timings.extend(rank_timings)
        receipts.append(receipt)
    def key(row):
        return (row["comparison_target"], row["seed"], row["variant"], row["request_id"])
    actual_p = [(*key(row), row["rank"]) for row in predictions]
    actual_t = [(*key(row), row["repeat_id"], row["rank"]) for row in timings]
    if len(actual_p) != len(set(actual_p)) or set(actual_p) != expected_p:
        raise ValueError("incomplete or duplicate steering prediction inventory/rank assignment")
    if len(actual_t) != len(set(actual_t)) or set(actual_t) != expected_t:
        raise ValueError("incomplete or duplicate steering timing inventory/rank assignment")
    for row in predictions:
        source = sources[row["request_id"]]
        if any(row[k] != source[k] for k in ("frame_key", "cluster_key", "split", "order_id")):
            raise ValueError("steering/public identity mismatch")
    summary = steering.aggregate_steering(predictions, timings, read_rows(dataset / "evaluator/all_labels.jsonl"),
                                          public, {**config, "mode": args.mode})
    atomic_json(args.out_dir / "analysis_summary.json", summary)
    (args.out_dir / "RESULTS.md").write_text(steering.render_report(summary))
    atomic_json(args.out_dir / "efficiency_summary.json", {"by_case": {
        case["id"]: pilot.efficiency_summary([r for r in predictions if r["comparison_target"] == case["id"]])
        for case in steering.cases()}})
    profile = walltime_profile(max(r["load_s"] + r["projected_full_work_s"] for r in receipts))
    atomic_json(args.out_dir / "profile.json", {"status": profile["runtime_budget_status"],
        "experiment": "steering", "mode": args.mode, "config_sha256": digest(args.config),
        "split_manifest_sha256": digest(dataset / "split_manifest.json"), **profile,
        "workers": receipts, "technical_gate": {"status": "pass"},
        "prediction_inventory": {"status": "pass", "cells": len(actual_p), "expected_cells_sha256": checksum(sorted(expected_p))},
        "timing_inventory": {"status": "pass", "cells": len(actual_t), "expected_cells_sha256": checksum(sorted(expected_t))},
        "quality_gate": "technical only; negative scientific results retained",
        "analysis_summary_sha256": digest(args.out_dir / "analysis_summary.json")})


def aggregate(args):
    if args.experiment == "steering":
        return aggregate_steering_run(args)
    from research.lifecycle_memory.decoder_token_matrix_analysis import analyze_run, render_report
    config = read_json(args.config)
    arms = validate_config(config)
    out = args.out_dir
    dataset = Path(args.dataset_dir or out / "dataset")
    manifest = read_json(dataset / "split_manifest.json")
    for name in ("public/requests.jsonl", "evaluator/all_labels.jsonl"):
        if digest(dataset / name) != manifest["exports"][name]["sha256"]:
            raise RuntimeError(f"dataset identity mismatch: {name}")
    public = read_rows(dataset / "public/requests.jsonl")
    validate_public_rows(public)
    selected, ranks = request_layout(public, args.mode)
    predictions, timings, receipts = [], [], []
    for rank in range(args.world_size):
        rankdir = out / f"rank_{rank:03d}"
        receipt, rank_predictions, rank_timings = _worker_outputs(args, dataset, rank, RECEIPT_FILES)
        validate_plan_receipts(rankdir, rank, [r for r in selected if ranks[r["request_id"]] == rank],
                               rank_predictions, rank_timings, arms)
        predictions.extend(rank_predictions)
        timings.extend(rank_timings)
        receipts.append(receipt)
    inventory = validate_prediction_inventory(predictions, public, "compression", args.mode, config)
    timing_inventory = validate_timing_inventory(timings, public, args.mode, config)
    summary = analyze_run(predictions, timings, read_rows(dataset / "evaluator/all_labels.jsonl"), selected, config)
    atomic_json(out / "arm_inventory.json", {"schema_version": 1, "arm_count": len(arms),
        "arms": arms, "arms_sha256": checksum(arms), "config_sha256": digest(args.config)})
    atomic_json(out / "analysis_summary.json", summary)
    atomic_json(out / "efficiency_summary.json", efficiency_summary(predictions))
    (out / "RESULTS.md").write_text(render_report(summary))
    estimate = max(r["load_s"] + r["projected_full_work_s"] for r in receipts)
    profile = walltime_profile(estimate)
    atomic_json(out / "profile.json", {"status": profile["runtime_budget_status"], "experiment": "compression",
        "mode": args.mode, "config_sha256": digest(args.config),
        "split_manifest_sha256": digest(dataset / "split_manifest.json"), **profile,
        "workers": receipts, "prediction_inventory": inventory, "timing_inventory": timing_inventory,
        "technical_gate": {"status": "pass"}, "quality_gate": "technical only; negative scientific results retained",
        "arm_inventory_sha256": digest(out / "arm_inventory.json"),
        "analysis_summary_sha256": digest(out / "analysis_summary.json")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "aggregate"))
    parser.add_argument("--experiment", choices=("compression", "steering"), required=True)
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE)
    args = parser.parse_args(argv)
    if args.world_size != WORLD_SIZE or args.rank not in range(WORLD_SIZE):
        parser.error("the frozen matrix uses exactly four workers")
    try:
        (worker if args.command == "worker" else aggregate)(args)
    except BaseException as error:
        destination = args.out_dir / (f"rank_{args.rank:03d}" if args.command == "worker" else "aggregate")
        atomic_json(destination / "failure.json", {"status": "fail", "stage": args.command,
            "rank": args.rank, "error": str(error), "traceback": traceback.format_exc()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
