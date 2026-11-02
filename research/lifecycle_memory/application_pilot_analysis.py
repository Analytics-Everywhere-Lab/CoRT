"""Paired participant analysis for standalone compression and steering pilots.

Prediction JSONL schema: task_id ('compression'/'steering'), variant, seed (int),
request_id, frame_key, cluster_key, split, order_id ('order_0'/'order_1'),
correct_label ('A'/'B', joined by evaluator), prediction ('A'/'B' or null),
optional error, margin_ab (A-minus-B in bits; margin alias accepted), elapsed_ms,
peak_allocated_bytes, peak_reserved_bytes. Null/invalid/error predictions are
incorrect, never filtered. One row per task/variant/seed/request, not per repeat,
except native steering rows also key on comparison_target for same-device pairing.
Optional metrics may be null; incomplete metrics are reported unavailable.

Prediction elapsed_ms is descriptive single-forward latency only. Separate
timing_rows (CLI --timing) contain task_id, variant, seed, request_id, repeat_id
(nonnegative int), elapsed_ms, timing_scope, device_id, batch_size, is_warmup,
warmup_complete, synchronized and optional peak memory fields. Measured elapsed
is full request/batch duration, not divided by batch size. At least two paired
measured repeats per request are required. Timed subsets are permitted and their
coverage is explicit. Scope examples: model_forward, remaining_blocks_including_merge_expand, or
end_to_end_including_preprocessing. Separate scopes require separate analyses.
Comparability checks are metadata checks, NOT verification of synchronization,
warmup, physical shortening, instrumentation, or identical hardware execution.

API: join_labels(predictions, labels); analyze_predictions(rows, expected_rows=...)
expects the FULL public inventory for compression and each included steering
split. Steering validation/test are separate; training results are rejected.
Seeds are averaged within each participant before cluster resampling; they are
not additional independent participants. CIs are unadjusted descriptive 95%
percentile intervals, not significance or equivalence decisions.

CLI::

    python -m research.lifecycle_memory.application_pilot_analysis \
      --predictions predictions.jsonl --labels dataset/evaluator/all_labels.jsonl \
      --expected-rows dataset/public/requests.jsonl --output analysis.json

For full-run gates pass --require-variant TASK:VARIANT and --require-seed TASK:N
for every planned variant/seed. Without these, entire missing variants/seeds
cannot be detected. There is no built-in accuracy-loss tolerance.
The compression runner requires native plus the four same-count F/C x L22/L34
merge cells for the paired role-by-depth contrast. Standalone callers can set
require_role_by_depth=True (CLI --require-role-by-depth) to reject an inventory
that entirely omits the role/depth controls; legacy native/F34-only reports do
not implicitly request a factorial analysis. Seeing any other factorial control
also requires the complete factorial. Missing arms/requests/seeds fail closed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from research.lifecycle_memory.application_pilot_data import read_json_rows
from research.lifecycle_memory.application_pilot_compression import ROLE_DEPTH_WEIGHTS

IDENTITY_FIELDS = ("frame_key", "cluster_key", "split", "order_id")
MEMORY_METRICS = ("peak_allocated_bytes", "peak_reserved_bytes", "incremental_peak_allocated_bytes",
                  "starting_allocated_bytes", "starting_reserved_bytes")
OPTIONAL_METRICS = ("margin", "elapsed_ms", *MEMORY_METRICS)
BOOTSTRAP_METRICS = ("accuracy", "both_orders_accuracy", "correct_order_flip_rate", "correct_margin")
# Validation is never used for training, tuning or checkpoint selection, so it
# is pooled with test as the declared primary steering report.
POOLED_SCOPE = "validation+test"
LN2 = math.log(2)


def join_labels(predictions: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Evaluator-only join; reject duplicate labels and conflicting identities."""
    index = {}
    for label in labels:
        key = label.get("request_id")
        if not isinstance(key, str) or not key or key in index or label.get("correct_label") not in {"A", "B"}:
            raise ValueError("invalid or duplicate label request_id/correct_label")
        index[key] = label
    result = []
    for source in predictions:
        if source.get("request_id") not in index:
            raise ValueError("prediction has no evaluator label")
        row = dict(source)
        label = index[row["request_id"]]
        for field in (*IDENTITY_FIELDS, "correct_label", "split_group_key"):
            if field in label:
                if field in row and row[field] != label[field]:
                    raise ValueError(f"prediction/label mismatch: {field}")
                row[field] = label[field]
        result.append(row)
    return result


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    if field != "margin" and value < 0:
        raise ValueError(f"{field} must be nonnegative")
    if field == "elapsed_ms" and value == 0:
        raise ValueError("elapsed_ms must be positive, not a missing/error sentinel")
    return float(value)


def _correct(row: Mapping[str, Any]) -> float:
    return float(not row.get("error") and row.get("prediction") == row["correct_label"])


def _valid(row: Mapping[str, Any]) -> bool:
    return not row.get("error") and row.get("prediction") in ("A", "B")


def _scope(row: Mapping[str, Any]) -> str:
    return "all" if row["task_id"] == "compression" else row["split"]


def _target_key(row, baseline):
    target = row.get("comparison_target", "")
    if target is None:
        target = ""
    if not isinstance(target, str):
        raise ValueError("comparison_target must be a string")
    if row["task_id"] != "steering":
        if target:
            raise ValueError("comparison_target is only supported for steering")
        return ""
    if row["variant"] != baseline:
        if target and target != row["variant"]:
            raise ValueError("steering intervention comparison_target must match variant")
        return ""
    return target


def _validate(rows: Sequence[Mapping[str, Any]], expected_rows: Sequence[Mapping[str, Any]] | None, baseline):
    if not rows:
        raise ValueError("no predictions")
    cells = defaultdict(dict)
    identities, frame_identities, participant_splits = {}, {}, {}
    for source in rows:
        row = dict(source)
        if row.get("margin_ab") is not None:
            _number(row["margin_ab"], "margin")
            if row.get("margin") is not None and row["margin"] != row["margin_ab"]:
                raise ValueError("conflicting margin/margin_ab")
            row["margin"] = row["margin_ab"]
        for field in ("task_id", "variant", "request_id", *IDENTITY_FIELDS):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"missing nonempty {field}")
        if row["task_id"] not in {"compression", "steering"} or type(row.get("seed")) is not int:
            raise ValueError("invalid task_id or seed")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("invalid split")
        if row["task_id"] == "steering" and row["split"] == "train":
            raise ValueError("steering analysis accepts only validation/test, never train")
        if row["order_id"] not in {"order_0", "order_1"} or row.get("correct_label") not in {"A", "B"}:
            raise ValueError("invalid order_id or correct_label")
        if "prediction" not in row:
            raise ValueError("prediction key is required; use null for failed requests")
        for field in OPTIONAL_METRICS:
            if row.get(field) is not None:
                _number(row[field], field)
        identity = tuple(row[field] for field in (*IDENTITY_FIELDS, "correct_label"))
        if identities.setdefault(row["request_id"], identity) != identity:
            raise ValueError("request identity/label mismatch across cells")
        frame_identity = (row["cluster_key"], row["split"])
        if frame_identities.setdefault(row["frame_key"], frame_identity) != frame_identity:
            raise ValueError("frame identity mismatch")
        if participant_splits.setdefault(row["cluster_key"], row["split"]) != row["split"]:
            raise ValueError("participant crosses dataset splits")
        key = (row["task_id"], _scope(row), row["variant"], row["seed"], _target_key(row, baseline))
        if row["request_id"] in cells[key]:
            raise ValueError("duplicate prediction row; timing repeats must be aggregated first")
        cells[key][row["request_id"]] = row
    expected = {}
    if expected_rows is not None:
        for row in expected_rows:
            request = row.get("request_id")
            if not isinstance(request, str) or not request or request in expected:
                raise ValueError("invalid/duplicate expected request_id")
            if any(field not in row for field in IDENTITY_FIELDS) or row.get("arm", "real") != "real":
                raise ValueError("expected inventory needs real-arm identities")
            expected[request] = row
    inventories = {}
    for key, cell in cells.items():
        task, scope, variant, seed, target = key
        if expected_rows is not None:
            required = {r for r, row in expected.items() if task == "compression" or row["split"] == scope}
            if set(cell) != required:
                raise ValueError(f"incomplete inventory for {key}: missing={len(required-set(cell))}, extra={len(set(cell)-required)}")
            for request, row in cell.items():
                if any(row[field] != expected[request][field] for field in IDENTITY_FIELDS):
                    raise ValueError("prediction/expected identity mismatch")
        if inventories.setdefault((task, scope), set(cell)) != set(cell):
            raise ValueError("paired cells have different request inventories")
        pairs = defaultdict(list)
        for row in cell.values():
            pairs[row["frame_key"]].append(row)
        for pair in pairs.values():
            if len(pair) != 2 or {r["order_id"] for r in pair} != {"order_0", "order_1"}:
                raise ValueError("every frame must have exactly both orders")
            if {r["correct_label"] for r in pair} != {"A", "B"}:
                raise ValueError("correct labels must flip across orders")
    return cells


def _cluster_metrics(cell: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    frames = defaultdict(list)
    for row in cell.values():
        frames[row["frame_key"]].append(row)
    clusters = defaultdict(list)
    for pair in frames.values():
        margins = [(1 if r["correct_label"] == "A" else -1) * r["margin"]
                   for r in pair if r.get("margin") is not None and not r.get("error")]
        clusters[pair[0]["cluster_key"]].append({
            "accuracy": mean(_correct(r) for r in pair),
            "both_orders_accuracy": float(all(_correct(r) for r in pair)),
            "correct_order_flip_rate": float(all(_valid(r) for r in pair) and pair[0]["prediction"] != pair[1]["prediction"]),
            "correct_margin": mean(margins) if len(margins) == 2 else None,
        })
    return {cluster: {metric: mean(frame[metric] for frame in values) if all(
        frame[metric] is not None for frame in values) else None for metric in BOOTSTRAP_METRICS}
        for cluster, values in clusters.items()}


def _estimate(values: list[float | None], indexes: np.ndarray | None) -> dict[str, Any]:
    if any(value is None for value in values):
        return {"estimate": None, "ci95": None, "reason": "metric missing or errored on at least one required row"}
    array = np.asarray(values, dtype=float)
    result = {"estimate": float(array.mean()), "ci95": None}
    if indexes is None:
        result["reason"] = "fewer than two participant clusters; uncertainty unavailable"
    else:
        result["ci95"] = [float(x) for x in np.quantile(array[indexes].mean(axis=1), [0.025, 0.975])]
    return result


def _average_cluster_metrics(cells: Sequence[Mapping[str, Mapping[str, Any]]]):
    metrics = [_cluster_metrics(cell) for cell in cells]
    clusters = sorted(metrics[0])
    return {cluster: {metric: mean(m[cluster][metric] for m in metrics) if all(
        m[cluster][metric] is not None for m in metrics) else None for metric in BOOTSTRAP_METRICS}
        for cluster in clusters}


def _summary(cells, indexes):
    clusters = _average_cluster_metrics(cells)
    all_rows = [r for cell in cells for r in cell.values()]
    return {
        "request_count_per_seed": len(cells[0]), "frame_count": len(cells[0]) // 2,
        "participant_count": len(clusters), "seed_count": len(cells),
        "trial_accuracy": mean(_correct(r) for r in all_rows),
        "error_count": sum(bool(r.get("error")) for r in all_rows),
        "invalid_prediction_count": sum(not _valid(r) for r in all_rows),
        "single_forward_mean_elapsed_ms": mean(r["elapsed_ms"] for r in all_rows) if all(
            r.get("elapsed_ms") is not None and not r.get("error") for r in all_rows) else None,
        "participant_equal": {metric: _estimate([clusters[c][metric] for c in sorted(clusters)], indexes)
                              for metric in BOOTSTRAP_METRICS},
    }


def _timing_cells(timing_rows, cells, baseline):
    """Associate separate benchmark rows with predictions, without outcome selection."""
    lookup = {(key[0], key[2], key[3], request, key[4]): (key, row)
              for key, cell in cells.items() for request, row in cell.items()}
    result = defaultdict(dict)
    for source in timing_rows or []:
        row = dict(source)
        identity = (row.get("task_id"), row.get("variant"), row.get("seed"), row.get("request_id"), _target_key(row, baseline))
        if identity not in lookup:
            raise ValueError("timing row has no corresponding prediction")
        key, prediction = lookup[identity]
        for field in IDENTITY_FIELDS:
            if field in row and row[field] != prediction[field]:
                raise ValueError("timing/prediction identity mismatch")
            row[field] = prediction[field]
        if type(row.get("is_warmup")) is not bool:
            raise ValueError("timing is_warmup must be explicit boolean")
        if row["is_warmup"]:
            continue
        repeat = row.get("repeat_id")
        if type(repeat) is not int or repeat < 0:
            raise ValueError("measured timing repeat_id must be a nonnegative integer")
        for field in OPTIONAL_METRICS[1:]:
            if row.get(field) is not None:
                _number(row[field], field)
        pair_key = (row["request_id"], repeat)
        if pair_key in result[key]:
            raise ValueError("duplicate measured timing repeat")
        result[key][pair_key] = row
    return result


def _timing(left_cells, right_cells):
    unavailable = lambda reason: {"available": False, "reason": reason}
    if not left_cells or any(not cell for cell in left_cells + right_cells):
        return unavailable("separate repeated benchmark timings not supplied for every paired seed/variant")
    for left, right in zip(left_cells, right_cells):
        if set(left) != set(right):
            return unavailable("unpaired measured timing request/repeat inventories")
        counts = defaultdict(int)
        for request, repeat in left:
            counts[request] += 1
        if min(counts.values()) < 2:
            return unavailable("at least two measured repeats per timed request are required")
    if any(set(cell) != set(left_cells[0]) for cell in left_cells):
        return unavailable("timed cohorts/repeats differ between seeds")
    pairs = [(left[r], right[r]) for left, right in zip(left_cells, right_cells) for r in sorted(left)]
    if any(a.get("error") or b.get("error") for a, b in pairs):
        return unavailable("error rows present; no complete successful timing cohort")
    scopes, batches, devices = set(), set(), set()
    for a, b in pairs:
        if any(row.get("warmup_complete") is not True or row.get("synchronized") is not True for row in (a, b)):
            return unavailable("measured timings require warmup_complete=true and synchronized=true")
        if a.get("elapsed_ms") is None or b.get("elapsed_ms") is None:
            return unavailable("missing elapsed_ms; timings are not zero-filled or filtered")
        for field in ("timing_scope", "device_id"):
            if not isinstance(a.get(field), str) or not a[field] or a[field] != b.get(field):
                return unavailable(f"missing or mismatched {field}")
        batch = a.get("batch_size")
        if type(batch) is not int or batch < 1 or batch != b.get("batch_size"):
            return unavailable("missing or mismatched batch_size")
        scopes.add(a["timing_scope"])
        batches.add(batch)
        devices.add(a["device_id"])
    if len(scopes) != 1 or len(batches) != 1:
        return unavailable("mixed timing scopes/batch sizes; analyze each separately")
    left_ms, right_ms = mean(a["elapsed_ms"] for a, b in pairs), mean(b["elapsed_ms"] for a, b in pairs)
    result = {"available": True, "timing_scope": next(iter(scopes)), "device_ids": sorted(devices),
              "batch_size": next(iter(batches)), "reference_mean_elapsed_ms": left_ms,
              "variant_mean_elapsed_ms": right_ms, "speedup_reference_over_variant": left_ms / right_ms,
              "latency_change_ms": right_ms - left_ms, "timed_request_seed_repeat_pairs": len(pairs),
              "timed_unique_requests": len({a["request_id"] for a, b in pairs}),
              "timed_participant_count": len({a["cluster_key"] for a, b in pairs}),
              "implied_requests_per_second": {"reference": 1000 * next(iter(batches)) / left_ms,
                                               "variant": 1000 * next(iter(batches)) / right_ms},
              "throughput_scope": "implied at recorded batch size, not measured sustained service throughput",
              "protocol_verification": "scope/device/batch metadata matched; synchronization/warmup/overhead inclusion not independently verified"}
    for field in MEMORY_METRICS:
        if all(a.get(field) is not None and b.get(field) is not None for a, b in pairs):
            result[field] = {"reference_mean": mean(a[field] for a, b in pairs),
                             "variant_mean": mean(b[field] for a, b in pairs),
                             "paired_mean_change": mean(b[field]-a[field] for a, b in pairs),
                             "reference_max": max(a[field] for a, b in pairs),
                             "variant_max": max(b[field] for a, b in pairs)}
        else:
            result[field] = {"available": False, "reason": "incomplete memory observations"}
    result["memory_scope"] = "total peaks and any runtime-reported incremental/starting allocations; starting allocation is not a weight-only measurement; reserved pool may persist across conditions"
    return result


def _single_forward_overhead(left_cells, right_cells):
    """Describe paired prediction latency; never promote it to warmed speedup."""
    pairs = [(left[r], right[r]) for left, right in zip(left_cells, right_cells) for r in sorted(left)]
    scopes, batches = set(), set()
    for a, b in pairs:
        if any(r.get("elapsed_ms") is None or r.get("error") for r in (a, b)):
            return {"available": False, "reason": "incomplete single-forward timings"}
        for field in ("device_id", "timing_scope"):
            if not isinstance(a.get(field), str) or not a[field] or a[field] != b.get(field):
                return {"available": False, "reason": f"missing or mismatched {field}"}
        if type(a.get("batch_size")) is not int or a["batch_size"] < 1 or a["batch_size"] != b.get("batch_size"):
            return {"available": False, "reason": "missing or mismatched batch_size"}
        scopes.add(a["timing_scope"])
        batches.add(a["batch_size"])
    if len(scopes) != 1 or len(batches) != 1:
        return {"available": False, "reason": "mixed timing scopes/batches"}
    return {"available": True, "timing_scope": next(iter(scopes)), "batch_size": next(iter(batches)),
            "reference_mean_elapsed_ms": mean(a["elapsed_ms"] for a, b in pairs),
            "variant_mean_elapsed_ms": mean(b["elapsed_ms"] for a, b in pairs),
            "paired_mean_overhead_ms": mean(b["elapsed_ms"] - a["elapsed_ms"] for a, b in pairs),
            "request_seed_pairs": len(pairs),
            "interpretation": "descriptive single-forward overhead only; not a warmed repeated efficiency benchmark"}


def _comparison(left_cells, right_cells, indexes, left_timing=None, right_timing=None):
    left = _average_cluster_metrics(left_cells)
    right = _average_cluster_metrics(right_cells)
    clusters = sorted(left)
    differences = {}
    for metric in BOOTSTRAP_METRICS:
        values = [None if left[c][metric] is None or right[c][metric] is None else right[c][metric]-left[c][metric]
                  for c in clusters]
        differences[metric] = _estimate(values, indexes)
    per_cluster = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(int)
    margin_changes, margins_complete = defaultdict(list), True
    for a, b in zip(left_cells, right_cells):
        for request in sorted(a):
            x, y = a[request], b[request]
            if any(r.get("margin") is None or r.get("error") for r in (x, y)):
                margins_complete = False
            else:
                margin_changes[x["cluster_key"]].append(abs(y["margin"] - x["margin"]))
            cx, cy = _correct(x), _correct(y)
            values = {"wrong_to_right": float(not cx and cy), "right_to_wrong": float(cx and not cy),
                      "answer_agreement": float(_valid(x) and _valid(y) and x["prediction"] == y["prediction"])}
            for metric, value in values.items():
                per_cluster[x["cluster_key"]][metric].append(value)
                counts[metric] += int(value)
            counts["reference_wrong"] += int(not cx)
            counts["reference_right"] += int(cx)
            counts["request_seed_pairs"] += 1
    transitions = {metric: _estimate([mean(per_cluster[c][metric]) for c in clusters], indexes)
                   for metric in ("wrong_to_right", "right_to_wrong", "answer_agreement")}
    fidelity = [mean(margin_changes[c]) if margins_complete else None for c in clusters]
    return {"participant_equal_difference_variant_minus_reference": differences,
            "participant_equal_transitions": transitions,
            "margin_fidelity": {"mean_absolute_margin_change_bits": _estimate(fidelity, indexes),
                                "scope": "participant-equal mean over paired request-seed rows of |variant - reference| A-minus-B margin in bits; unsigned, so opposite shifts do not cancel"},
            "transition_counts": dict(counts),
            "transition_denominator": "all paired request-seed rows; unconditional rates (not baseline-error subsets)",
            "single_forward_overhead": _single_forward_overhead(left_cells, right_cells),
            "timing": _timing(left_timing or [], right_timing or [])}


def _role_by_depth(arm_cells, native, arm_timings, native_timing, indexes, bootstrap_draws, bootstrap_seed):
    """Paired 2x2 contrast, never a subtraction of independently bootstrapped CIs."""
    missing = sorted(set(ROLE_DEPTH_WEIGHTS) - set(arm_cells))
    if missing:
        raise ValueError(f"incomplete F/C x L22/L34 factorial: missing variants {missing}")
    metrics = {arm: _average_cluster_metrics(arm_cells[arm]) for arm in ROLE_DEPTH_WEIGHTS}
    clusters = sorted(next(iter(metrics.values())))
    differences = {}
    for metric in BOOTSTRAP_METRICS:
        values = [sum(weight * metrics[arm][c][metric] for arm, weight in ROLE_DEPTH_WEIGHTS.items())
                  if all(metrics[arm][c][metric] is not None for arm in ROLE_DEPTH_WEIGHTS) else None
                  for c in clusters]
        differences[metric] = _estimate(values, indexes)

    # Absolute drift is relative to the SAME native request/seed, before taking
    # the signed factorial contrast. Opposite margin perturbations cannot cancel.
    drift = defaultdict(list)
    complete = True
    for i, reference in enumerate(native):
        for request in sorted(reference):
            base = reference[request]
            variants = {arm: arm_cells[arm][i][request] for arm in ROLE_DEPTH_WEIGHTS}
            if any(r.get("margin") is None or r.get("error") for r in [base, *variants.values()]):
                complete = False
                continue
            drift[base["cluster_key"]].append(sum(weight * abs(variants[arm]["margin"] - base["margin"])
                                                   for arm, weight in ROLE_DEPTH_WEIGHTS.items()))
    differences["mean_absolute_margin_change_bits"] = _estimate(
        [mean(drift[c]) if complete else None for c in clusters], indexes)

    result = {"available": True, "formula": "(F34 - C34) - (F22 - C22)",
              "variant_weights": dict(ROLE_DEPTH_WEIGHTS), "participant_count": len(clusters),
              "request_count_per_seed": len(native[0]), "seed_count": len(native),
              "participant_equal_difference_of_differences": differences,
              "sign": {
                  "accuracy": "positive: F-minus-C accuracy advantage is larger at L34 than L22; fraction, multiply by 100 for percentage points",
                  "both_orders_accuracy": "positive: F-minus-C both-orders accuracy advantage is larger at L34",
                  "correct_order_flip_rate": "positive: F-minus-C valid answer-flip rate is larger at L34; flipping alone need not be correct",
                  "correct_margin": "positive: F-minus-C correct-label margin advantage is larger at L34; bits",
                  "mean_absolute_margin_change_bits": "negative: F-minus-C absolute margin drift against native is smaller at L34 than L22, favoring F's relative fidelity"},
              "inference": "equal participants; average paired requests and seeds within participant, then resample whole participants jointly across all four arms; unadjusted descriptive percentile 95%, no confirmatory test"}

    # Validate all five timing cells against the same native cell, including
    # request/repeat inventories, devices, scope, seeds and warmup metadata.
    timing_checks = {arm: _timing(native_timing, arm_timings.get(arm, [])) for arm in ROLE_DEPTH_WEIGHTS}
    failed = {arm: check["reason"] for arm, check in timing_checks.items() if not check["available"]}
    if failed:
        result["timing"] = {"available": False, "reason": "incomplete or incomparable paired factorial benchmark", "by_variant": failed}
        return result
    values = defaultdict(lambda: defaultdict(list))
    for i, reference in enumerate(native_timing):
        requests = defaultdict(list)
        for request, repeat in sorted(reference):
            requests[request].append(repeat)
        for request, repeats in requests.items():
            cluster = reference[(request, repeats[0])]["cluster_key"]
            latencies, speeds = [], []
            for repeat in repeats:
                key = (request, repeat)
                base_ms = reference[key]["elapsed_ms"]
                arm_ms = {arm: arm_timings[arm][i][key]["elapsed_ms"] for arm in ROLE_DEPTH_WEIGHTS}
                latencies.append(sum(weight * arm_ms[arm] for arm, weight in ROLE_DEPTH_WEIGHTS.items()))
                speeds.append(sum(weight * (math.log(base_ms) - math.log(arm_ms[arm]))
                                  for arm, weight in ROLE_DEPTH_WEIGHTS.items()))
            values[cluster]["latency_ms"].append(mean(latencies))
            values[cluster]["log_speedup_native_over_variant"].append(mean(speeds))
    timed_clusters = sorted(values)
    n = len(timed_clusters)
    timed_indexes = np.random.default_rng(bootstrap_seed).integers(0, n, size=(bootstrap_draws, n)) if n > 1 else None
    check = next(iter(timing_checks.values()))
    result["timing"] = {"available": True,
        "timing_scope": check["timing_scope"], "batch_size": check["batch_size"],
        "device_ids": check["device_ids"], "timed_unique_requests": check["timed_unique_requests"],
        "timed_participant_count": n, "timed_request_seed_repeat_pairs": check["timed_request_seed_repeat_pairs"],
        "participant_equal_difference_of_differences": {
            metric: _estimate([mean(values[c][metric]) for c in timed_clusters], timed_indexes)
            for metric in ("latency_ms", "log_speedup_native_over_variant")},
        "sign": {"latency_ms": "negative: F-minus-C latency is smaller at L34 than L22",
                 "log_speedup_native_over_variant": "positive: F's relative speed advantage over C is larger at L34; natural log of (C34/F34)/(C22/F22) latency ratios"},
        "inference": "average paired repeats within request, requests/seeds within participant, then bootstrap timed participants equally; descriptive 95% percentile intervals; subset coverage above, no full-cohort extrapolation",
        "protocol_verification": check["protocol_verification"]}
    return result


def _add_pooled_cells(cells, timings):
    """Pool steering validation and test per variant/seed/target, all or nothing."""
    keys = {split: {(k[0], k[2], k[3], k[4]) for k in cells if k[0] == "steering" and k[1] == split}
            for split in ("validation", "test")}
    if not keys["validation"] or keys["validation"] != keys["test"]:
        return
    for task, variant, seed, target in keys["validation"]:
        parts = [(task, split, variant, seed, target) for split in ("validation", "test")]
        pooled = (task, POOLED_SCOPE, variant, seed, target)
        cells[pooled] = {request: row for key in parts for request, row in cells[key].items()}
        timings[pooled] = {pair: row for key in parts for pair, row in timings.get(key, {}).items()}


def _report_role(task, scope):
    if task == "compression":
        return "primary"
    return "primary_pooled_heldout" if scope == POOLED_SCOPE else "secondary_per_split"


def _sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


def fit_ab_bias(cell):
    """One offset b in bits minimizing log-loss of p(A) = sigmoid(ln2 * (margin - b)).

    The loss is strictly convex in b whenever both labels occur, so bisection on
    its monotone derivative finds the unique minimum. Returns None if any margin
    is missing/errored or only one label occurs.
    """
    rows = list(cell.values())
    if not rows or any(r.get("margin") is None or r.get("error") for r in rows):
        return None
    a = [r["margin"] for r in rows if r["correct_label"] == "A"]
    b = [r["margin"] for r in rows if r["correct_label"] == "B"]
    if not a or not b:
        return None

    def slope(bias):
        return sum(_sigmoid(LN2 * (bias - m)) for m in a) - sum(_sigmoid(LN2 * (m - bias)) for m in b)
    lo, hi = min(a + b) - 64, max(a + b) + 64
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if slope(mid) > 0 else (mid, hi)
    return (lo + hi) / 2


def apply_ab_bias(cell, bias):
    """Shift margins by the fitted offset; same A-if-margin>=0 rule as the runner."""
    result = {}
    for request, source in cell.items():
        row = dict(source, margin=source["margin"] - bias, elapsed_ms=None)
        row["prediction"] = "A" if row["margin"] >= 0 else "B"
        result[request] = row
    return result


def _ab_bias_baseline(cells, task, scope, seeds, targets, baseline, target_specific, indexes):
    unavailable = lambda reason: {"available": False, "reason": reason}
    if scope not in ("test", POOLED_SCOPE):
        return unavailable("validation-only report; the bias is fit on validation for the test report")
    result = {"available": True, "parameters": 1, "fit": "native margins, log-loss, offset in bits",
              "fit_policy": "validation->test" if scope == "test" else "cross-fit validation->test and test->validation",
              "by_target": {}}
    for target in targets:
        key = lambda split, seed: (task, split, baseline, seed, target if target_specific else "")
        biased, fitted = [], {}
        for seed in seeds:
            if key("validation", seed) not in cells or key("test", seed) not in cells:
                return unavailable("validation and test native cells are both required")
            validation, test = cells[key("validation", seed)], cells[key("test", seed)]
            parts = [(test, fit_ab_bias(validation))]
            fitted[str(seed)] = {"validation_to_test": parts[0][1]}
            if scope == POOLED_SCOPE:
                parts.append((validation, fit_ab_bias(test)))
                fitted[str(seed)]["test_to_validation"] = parts[1][1]
            if any(bias is None for _, bias in parts):
                return unavailable("missing/errored native margins or one-sided labels")
            biased.append({request: row for cell, bias in parts for request, row in apply_ab_bias(cell, bias).items()})
        native = [cells[key(scope, seed)] for seed in seeds]
        steered = [cells[(task, scope, target, seed, "")] for seed in seeds]
        result["by_target"][target] = {"fitted_bias_bits": fitted,
            "bias_minus_native": _comparison(native, biased, indexes),
            "target_minus_bias": _comparison(biased, steered, indexes)}
    return result


def analyze_predictions(
    rows: Sequence[Mapping[str, Any]], *, baseline: str = "native",
    bootstrap_draws: int | None = None, bootstrap_seed: int | None = None,
    expected_rows: Sequence[Mapping[str, Any]] | None = None,
    comparisons: Sequence[tuple[str, str]] | None = None,
    accuracy_loss_tolerance: float | None = None,
    required_variants: Mapping[str, Sequence[str]] | None = None,
    required_seeds: Mapping[str, Sequence[int]] | None = None,
    required_splits: Mapping[str, Sequence[str]] | None = None,
    timing_rows: Sequence[Mapping[str, Any]] | None = None,
    require_role_by_depth: bool = False,
    draws: int | None = None, seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate complete paired cells; optional tolerance is a prechosen fraction.

    ``comparisons`` adds (reference, variant) contrasts. By default steering also
    compares every pair of non-native targets. This is a descriptive contrast,
    not proof that capacity/optimizer/training budgets were actually matched.
    """
    if bootstrap_draws is not None and draws is not None and bootstrap_draws != draws:
        raise ValueError("conflicting draws/bootstrap_draws")
    if bootstrap_seed is not None and seed is not None and bootstrap_seed != seed:
        raise ValueError("conflicting seed/bootstrap_seed")
    bootstrap_draws = bootstrap_draws if bootstrap_draws is not None else (draws if draws is not None else 2000)
    bootstrap_seed = bootstrap_seed if bootstrap_seed is not None else (seed if seed is not None else 20261014)
    if type(bootstrap_draws) is not int or bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be an integer >= 100")
    if type(bootstrap_seed) is not int:
        raise ValueError("bootstrap_seed must be an integer")
    if accuracy_loss_tolerance is not None:
        if isinstance(accuracy_loss_tolerance, bool) or not math.isfinite(accuracy_loss_tolerance) or not 0 <= accuracy_loss_tolerance <= 1:
            raise ValueError("accuracy_loss_tolerance must be a fraction in [0, 1]")
    cells = _validate(rows, expected_rows, baseline)
    timings = _timing_cells(timing_rows, cells, baseline)
    split_tasks = {(key[0], key[1]) for key in cells}
    for task in set(required_variants or {}) | set(required_seeds or {}) | set(required_splits or {}):
        if task not in {key[0] for key in cells}:
            raise ValueError(f"required task missing: {task}")
    for task, splits in (required_splits or {}).items():
        if {scope for task_id, scope in split_tasks if task_id == task} != set(splits):
            raise ValueError("split inventory differs from required splits")
    _add_pooled_cells(cells, timings)
    # The pooled held-out report is listed first because it is primary.
    tasks = sorted({(key[0], key[1]) for key in cells}, key=lambda t: (t[0], t[1] != POOLED_SCOPE, t[1]))
    reports = []
    for task, scope in tasks:
        variants = sorted({key[2] for key in cells if key[:2] == (task, scope)})
        if baseline not in variants:
            raise ValueError(f"baseline missing for {task}/{scope}")
        if required_variants and task in required_variants and set(variants) != set(required_variants[task]):
            raise ValueError("variant inventory differs from required variants")
        seeds_by_variant = {v: sorted({key[3] for key in cells if key[:3] == (task, scope, v)}) for v in variants}
        seeds = seeds_by_variant[baseline]
        if any(values != seeds for values in seeds_by_variant.values()):
            raise ValueError("variants have different seed sets; baseline must be paired per seed")
        if required_seeds and task in required_seeds and set(seeds) != set(required_seeds[task]):
            raise ValueError("seed inventory differs from required seeds")
        native_targets = sorted({key[4] for key in cells if key[:3] == (task, scope, baseline)})
        target_specific = native_targets != [""]
        if target_specific and set(native_targets) != set(variants) - {baseline}:
            raise ValueError("target-specific native baselines must cover exactly every steering target")

        def selected_keys(variant, target=""):
            selected_target = target if variant == baseline and target_specific else ""
            keys = [(task, scope, variant, s, selected_target) for s in seeds]
            if any(key not in cells for key in keys):
                raise ValueError("target-specific baseline is missing a paired seed")
            return keys

        def selected_cells(variant, target=""):
            return [cells[key] for key in selected_keys(variant, target)]

        def selected_timing(variant, target=""):
            return [timings.get(key, {}) for key in selected_keys(variant, target)]

        n = len({r["cluster_key"] for r in selected_cells(baseline, native_targets[0])[0].values()})
        indexes = np.random.default_rng(bootstrap_seed).integers(0, n, size=(bootstrap_draws, n)) if n > 1 else None

        def summary(variant, target=""):
            selected = selected_cells(variant, target)
            return {"pooled_seeds": _summary(selected, indexes),
                    "per_seed": {str(s): _summary([cell], indexes) for s, cell in zip(seeds, selected)}}

        summaries = {v: summary(v) for v in variants if v != baseline}
        summaries[baseline] = {"by_comparison_target": {target: summary(baseline, target) for target in native_targets}} if target_specific else summary(baseline)
        pairs = [(baseline, v) for v in variants if v != baseline]
        additional = list(comparisons) if comparisons is not None else (
            list(itertools.combinations([v for v in variants if v != baseline], 2)) if task == "steering" else [])
        for pair in additional:
            if len(pair) != 2 or pair[0] == pair[1] or any(v not in variants for v in pair):
                raise ValueError(f"invalid comparison for {task}/{scope}: {pair}")
            if tuple(pair) not in pairs:
                pairs.append(tuple(pair))
        contrasts = []
        for reference, variant in pairs:
            left_cells = selected_cells(reference, variant)
            right_cells = selected_cells(variant, reference)
            left_times = selected_timing(reference, variant)
            right_times = selected_timing(variant, reference)
            contrast = _comparison(left_cells, right_cells, indexes, left_times, right_times)
            contrast.update(reference=reference, variant=variant,
                            comparison_kind="native_reference" if reference == baseline else "matched_target_descriptive")
            if target_specific and reference == baseline:
                contrast["comparison_target"] = variant
            contrast["per_seed"] = {str(s): _comparison([left], [right], indexes, [lt], [rt])
                                     for s, left, right, lt, rt in zip(seeds, left_cells, right_cells, left_times, right_times)}
            if task == "compression":
                delta = contrast["participant_equal_difference_variant_minus_reference"]["accuracy"]
                timing = contrast["timing"]
                contrast["quality_cost_tradeoff"] = {
                    "accuracy_change": delta, "preselected_accuracy_loss_tolerance": accuracy_loss_tolerance,
                    "point_estimate_within_tolerance": None if accuracy_loss_tolerance is None else delta["estimate"] >= -accuracy_loss_tolerance,
                    "paired_ci_lower_bound_above_negative_tolerance": None if accuracy_loss_tolerance is None or delta["ci95"] is None else delta["ci95"][0] > -accuracy_loss_tolerance,
                    "speedup": timing.get("speedup_reference_over_variant"),
                    "interpretation": "descriptive quality/cost tradeoff; a non-inferiority claim additionally requires an outcome-blind prespecified tolerance/decision rule and multiplicity policy; no equivalence claim"}
            else:
                contrast["matching_scope"] = "request/seed/participant matched; module capacity, steps, optimizer and selection budgets require runtime provenance"
            contrasts.append(contrast)
        report = {"task_id": task, "split_scope": scope, "seeds": seeds,
                  "report_role": _report_role(task, scope),
                  "variants": summaries, "comparisons": contrasts}
        if task == "steering":
            report["ab_bias_baseline"] = _ab_bias_baseline(
                cells, task, scope, seeds, [v for v in variants if v != baseline],
                baseline, target_specific, indexes)
        elif require_role_by_depth or (set(variants) & (set(ROLE_DEPTH_WEIGHTS) - {"merge_F_L34"})):
            if baseline != "native":
                raise ValueError("role-by-depth fidelity and speed require the native baseline")
            factorial = {v: selected_cells(v) for v in ROLE_DEPTH_WEIGHTS if v in variants}
            factorial_timing = {v: selected_timing(v) for v in factorial}
            report["role_by_depth"] = _role_by_depth(factorial, selected_cells(baseline),
                factorial_timing, selected_timing(baseline), indexes, bootstrap_draws, bootstrap_seed)
            if report["role_by_depth"]["available"]:
                report["role_by_depth"]["per_seed"] = {
                    str(s): _role_by_depth({v: [values[i]] for v, values in factorial.items()},
                        [selected_cells(baseline)[i]], {v: [values[i]] for v, values in factorial_timing.items()},
                        [selected_timing(baseline)[i]], indexes, bootstrap_draws, bootstrap_seed)
                    for i, s in enumerate(seeds)}
        reports.append(report)
    return {"schema_version": 1, "reports": reports,
            "coverage": {"expected_inventory_verified": expected_rows is not None,
                         "required_variants": dict(required_variants or {}), "required_seeds": dict(required_seeds or {}),
                         "required_splits": dict(required_splits or {}),
                         "warning": "entire missing variants/seeds/splits are detectable only against a supplied run plan; without expected_rows entire missing frames are undetectable"},
            "inference": {"unit": "participant cluster_key", "weighting": "equal participants; equal frames within participant; both orders per frame",
                          "seed_handling": "average within participant, then resample participants with all seeds/orders paired",
                          "bootstrap_draws": bootstrap_draws, "bootstrap_seed": bootstrap_seed,
                          "interval": "unadjusted descriptive percentile 95%; no p-values or equivalence claims",
                          "correct_order_flip_rate": "valid A/B prediction changes across counterbalanced orders; includes both-right and both-wrong pairs",
                          "correct_margin": "A-minus-B margin_ab in bits signed by evaluator label",
                          "pooled_steering": f"{POOLED_SCOPE} is the declared primary steering report: validation is not used for training, tuning or checkpoint selection, and participants are disjoint across splits; per-split reports are secondary",
                          "ab_bias_baseline": "one offset in bits fit to frozen native margins by log-loss; test report fits on validation, pooled report cross-fits validation<->test; never in-sample"}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--expected-rows", type=Path)
    parser.add_argument("--timing", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="native")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261014)
    parser.add_argument("--accuracy-loss-tolerance", type=float)
    parser.add_argument("--require-role-by-depth", action="store_true",
                        help="require native and all four F/C x L22/L34 merge cells for compression")
    parser.add_argument("--require-variant", action="append", default=[], metavar="TASK:VARIANT")
    parser.add_argument("--require-seed", action="append", default=[], metavar="TASK:SEED")
    parser.add_argument("--require-split", action="append", default=[], metavar="TASK:SPLIT")
    args = parser.parse_args(argv)
    rows = read_json_rows(args.predictions)
    if args.labels:
        rows = join_labels(rows, read_json_rows(args.labels))
    variants, seeds, splits = defaultdict(list), defaultdict(list), defaultdict(list)
    for value in args.require_variant:
        task, variant = value.split(":", 1)
        variants[task].append(variant)
    for value in args.require_seed:
        task, seed = value.split(":", 1)
        seeds[task].append(int(seed))
    for value in args.require_split:
        task, split = value.split(":", 1)
        splits[task].append(split)
    result = analyze_predictions(rows, baseline=args.baseline, bootstrap_draws=args.bootstrap_draws,
                                 bootstrap_seed=args.bootstrap_seed,
                                 expected_rows=read_json_rows(args.expected_rows) if args.expected_rows else None,
                                 accuracy_loss_tolerance=args.accuracy_loss_tolerance,
                                 require_role_by_depth=args.require_role_by_depth,
                                 required_variants=variants, required_seeds=seeds, required_splits=splits,
                                 timing_rows=read_json_rows(args.timing) if args.timing else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "reports": len(result["reports"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
