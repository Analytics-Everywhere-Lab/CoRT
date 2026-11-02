"""Evaluator-only descriptive analysis of the complete compression matrix.

No model execution. Bias fits and configuration selection exclude the reported
outer fold. Prior outcome inspection remains part of the study's history.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from statistics import mean

import numpy as np

from research.lifecycle_memory.application_pilot_analysis import analyze_predictions, join_labels


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def cluster_folds(cluster_keys, count=5, seed=20261014):
    """Balance cluster counts; never split a camera wearer or its answer orders."""
    keys = sorted(set(cluster_keys), key=lambda k: hashlib.sha256(
        f"token-matrix-fold:{seed}:{k}".encode()).hexdigest())
    _require(type(count) is int and 2 <= count <= len(keys), "insufficient clusters for folds")
    return {key: i % count for i, key in enumerate(keys)}


def fit_bias(rows):
    """Fit b in sigmoid(ln(2)*(margin_ab-b)), giving each cluster equal weight."""
    _require(bool(rows), "empty bias training fold")
    _require({r["correct_label"] for r in rows} == {"A", "B"}, "bias requires both labels")
    sizes = defaultdict(int)
    for r in rows:
        _require(not r.get("error"), "failed prediction in bias fit")
        sizes[r["cluster_key"]] += 1
    margin = np.asarray([r["margin_ab"] for r in rows], dtype=float)
    _require(bool(np.isfinite(margin).all()), "nonfinite bias margin")
    labels = np.asarray([r["correct_label"] == "A" for r in rows], dtype=float)
    weights = np.asarray([1 / sizes[r["cluster_key"]] for r in rows])
    lo, hi = float(margin.min()) - 64, float(margin.max()) + 64
    for _ in range(64):
        mid = (lo + hi) / 2
        z = np.clip((margin - mid) * math.log(2), -700, 700)
        derivative = float(np.dot(weights, 1 / (1 + np.exp(-z)) - labels))
        if derivative > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _cluster_metrics(rows, biases=None):
    groups = defaultdict(list)
    for r in rows:
        groups[r["cluster_key"]].append(r)
    result = {}
    for cluster, values in sorted(groups.items()):
        correct, losses, frames = [], [], defaultdict(list)
        for r in values:
            _require(not r.get("error") and math.isfinite(r["margin_ab"]),
                     "failed/nonfinite prediction in corrected metrics")
            bias = 0.0 if biases is None else biases[r["request_id"]]
            margin = r["margin_ab"] - bias
            pred = "A" if margin >= 0 else "B"
            hit = float(pred == r["correct_label"])
            signed = margin * (1 if r["correct_label"] == "A" else -1) * math.log(2)
            correct.append(hit)
            losses.append(max(0, -signed) + math.log1p(math.exp(-abs(signed))))
            frames[r["frame_key"]].append(hit)
        _require(all(len(pair) == 2 for pair in frames.values()), "missing counterbalanced order")
        result[cluster] = {"accuracy": mean(correct), "binary_log_loss_nats": mean(losses),
                           "both_orders_accuracy": mean(float(all(pair)) for pair in frames.values())}
    return result


def _interval(values, indexes):
    a = np.asarray(values, dtype=float)
    _require(len(a) == indexes.shape[1] and bool(np.isfinite(a).all()), "invalid bootstrap values")
    boot = a[indexes].mean(axis=1)
    return {"estimate": float(a.mean()), "ci95": [float(v) for v in np.quantile(boot, [.025, .975])]}


def _metric_report(metrics, reference, indexes):
    keys = sorted(metrics)
    _require(keys == sorted(reference), "unpaired participant metrics")
    return {field: {
        "absolute": _interval([metrics[k][field] for k in keys], indexes),
        "difference_vs_reference": _interval([metrics[k][field] - reference[k][field] for k in keys], indexes)
    } for field in ("accuracy", "both_orders_accuracy", "binary_log_loss_nats")}


def _speedups(timings, allowed_requests):
    cells = defaultdict(dict)
    for r in timings:
        if r["request_id"] not in allowed_requests:
            continue
        _require(not r.get("error") and r.get("is_warmup") is False
                 and r.get("warmup_complete") is True and r.get("synchronized") is True
                 and r.get("batch_size") == 1, "invalid benchmark measurement")
        _require(type(r["repeat_id"]) is int and r["repeat_id"] >= 0, "invalid benchmark repeat")
        _require(all(isinstance(r.get(field), (int, float)) and not isinstance(r[field], bool)
                     and math.isfinite(r[field]) and r[field] > 0
                     for field in ("elapsed_ms", "end_to_end_elapsed_ms")), "invalid benchmark duration")
        _require(r["end_to_end_elapsed_ms"] >= r["elapsed_ms"], "end-to-end duration excludes forward")
        _require(all(isinstance(r.get(field), str) and r[field]
                     for field in ("device_id", "timing_scope", "end_to_end_timing_scope")),
                 "missing benchmark provenance")
        key = (r["request_id"], r["repeat_id"])
        _require(key not in cells[r["variant"]], "duplicate timing cell")
        cells[r["variant"]][key] = r
    if not cells.get("native"):
        _require(not cells, "missing native benchmark")
        return {}
    repeats = defaultdict(set)
    for request, repeat in cells["native"]:
        repeats[request].add(repeat)
    _require(len({tuple(sorted(v)) for v in repeats.values()}) == 1
             and all(v in ({0, 1}, {0, 1, 2}) for v in repeats.values()),
             "incomplete benchmark repeats")
    result = {}
    for variant, rows in cells.items():
        _require(set(rows) == set(cells["native"]), "incomplete paired benchmark inventory")
        paired = sorted(rows)
        field = "end_to_end_elapsed_ms"
        _require(all(rows[k][name] == cells["native"][k][name] for k in paired
                     for name in ("device_id", "batch_size", "timing_scope", "end_to_end_timing_scope")),
                 "unmatched benchmark provenance")
        numer = mean(cells["native"][k][field] for k in paired)
        denom = mean(rows[k][field] for k in paired)
        _require(math.isfinite(numer) and math.isfinite(denom) and min(numer, denom) > 0,
                 "invalid end-to-end latency")
        result[variant] = numer / denom
    return result


def bias_and_selection(rows, timings, config, indexes):
    """Outer-fold reporting with inner-fold configuration selection.

    The same per-arm forward predictions can be reused because their operators
    are frozen and label-free. No evaluation labels select a configuration or b.
    """
    settings = config["bias_correction"]
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["variant"]].append(r)
    native = by_arm["native"]
    folds = cluster_folds([r["cluster_key"] for r in native], settings["folds"], settings["seed"])
    per_arm_bias = {v: {} for v in by_arm}
    receipts, selected_rows = [], {"fast": [], "accurate": []}
    selected_bias = {"fast": {}, "accurate": {}}
    for fold in range(settings["folds"]):
        train_clusters = {k for k, f in folds.items() if f != fold}
        test_clusters = {k for k, f in folds.items() if f == fold}
        inner_folds = cluster_folds(train_clusters, min(4, len(train_clusters)), settings["seed"] + fold + 1)
        inner_val = {k for k, f in inner_folds.items() if f == 0}
        inner_train = train_clusters - inner_val
        _require(inner_train and inner_val and not (train_clusters & test_clusters), "empty/leaking nested split")
        estimates, fitted = {}, {}
        train_requests = {r["request_id"] for r in native if r["cluster_key"] in train_clusters}
        speeds = _speedups(timings, train_requests)
        for variant, values in by_arm.items():
            fit_rows = [r for r in values if r["cluster_key"] in train_clusters]
            outer_bias = fit_bias(fit_rows)
            fitted[variant] = outer_bias
            for r in values:
                if r["cluster_key"] in test_clusters:
                    per_arm_bias[variant][r["request_id"]] = outer_bias
            inner_bias = fit_bias([r for r in values if r["cluster_key"] in inner_train])
            val_rows = [r for r in values if r["cluster_key"] in inner_val]
            metrics = _cluster_metrics(val_rows, {r["request_id"]: inner_bias for r in val_rows})
            estimates[variant] = mean(m["accuracy"] for m in metrics.values())
        tolerance = settings["selection_accuracy_loss_tolerance_pp"] / 100
        minimum_speed = settings["selection_minimum_speedup"]
        eligible = [v for v in by_arm if v != "native" and speeds.get(v, 0) >= minimum_speed
                    and estimates[v] >= estimates["native"] - tolerance]
        choices = {
            "fast": min(eligible, key=lambda v: (-speeds[v], -estimates[v], v)) if eligible else "native",
            "accurate": min(eligible, key=lambda v: (-estimates[v], -speeds[v], v)) if eligible else "native"
        }
        for policy, variant in choices.items():
            test_rows = [r for r in by_arm[variant] if r["cluster_key"] in test_clusters]
            selected_rows[policy].extend(test_rows)
            selected_bias[policy].update({r["request_id"]: fitted[variant] for r in test_rows})
        receipts.append({"fold": fold, "train_clusters": sorted(train_clusters),
                         "test_clusters": sorted(test_clusters), "inner_train_clusters": sorted(inner_train),
                         "inner_validation_clusters": sorted(inner_val), "selected": choices,
                         "bias_bits_by_arm": fitted, "eligible_count": len(eligible),
                         "selection_timing_scope": "outer-training requests only; measured end-to-end latency"})
    raw_reference = _cluster_metrics(native)
    calibrated_reference = _cluster_metrics(native, per_arm_bias["native"])
    arm_reports = {}
    for variant, values in by_arm.items():
        _require(len(per_arm_bias[variant]) == len(values), "incomplete cross-fitted predictions")
        corrected = _cluster_metrics(values, per_arm_bias[variant])
        arm_reports[variant] = {
            "vs_native": _metric_report(corrected, raw_reference, indexes),
            "vs_native_with_bias": _metric_report(corrected, calibrated_reference, indexes),
            "correction_only": _metric_report(corrected, _cluster_metrics(values), indexes)}
    policy_reports = {}
    for policy, values in selected_rows.items():
        metrics = _cluster_metrics(values, selected_bias[policy])
        policy_reports[policy] = {
            "vs_native": _metric_report(metrics, raw_reference, indexes),
            "vs_native_with_bias": _metric_report(metrics, calibrated_reference, indexes)}
    return {"status": "pass", "folds": settings["folds"], "fold_by_cluster": folds,
            "fit_policy": "One participant-equal binary-log-loss A/B offset per predictor, fitted without the outer evaluation fold.",
            "selection_policy": "Inner validation accuracy and outer-training timing select within each outer fold; then refit offset on outer-training rows.",
            "interval_scope": "Descriptive cluster intervals conditional on the fitted fold procedures; not a bootstrap of retraining or fresh confirmation.",
            "latency_scope": "Compression timings exclude the scalar postprocessing operation; no measured speedup claim for bias-adjusted predictions.",
            "receipts": receipts, "arms": arm_reports, "selected_policies": policy_reports}


def analyze_run(predictions, timings, labels, public_rows, config):
    from research.lifecycle_memory.decoder_token_matrix import build_arms
    arms = build_arms()
    names = [a["id"] for a in arms]
    _require(len(names) == config["compression"]["expected_arm_count"], "arm count drift")
    joined = join_labels(predictions, labels)
    _require(all(not r.get("error") and r.get("prediction") in {"A", "B"}
                 and isinstance(r.get("margin_ab"), (int, float))
                 and math.isfinite(r["margin_ab"])
                 and r["prediction"] == ("A" if r["margin_ab"] >= 0 else "B")
                 for r in joined), "failed/invalid prediction in complete matrix")
    expected = {(v, r["request_id"]) for v in names for r in public_rows}
    seen = {(r["variant"], r["request_id"]) for r in joined}
    _require(len(seen) == len(joined) and seen == expected, "incomplete/duplicate matrix cells")
    if timings:
        _require({r["variant"] for r in timings} == set(names), "incomplete benchmark arm inventory")
    summary = analyze_predictions(joined, expected_rows=public_rows, timing_rows=timings,
        bootstrap_draws=config["bootstrap"]["draws"], bootstrap_seed=config["bootstrap"]["seed"],
        required_variants={"compression": names}, required_seeds={"compression": [0]},
        required_splits={"compression": ["all"]}, comparisons=[],
        accuracy_loss_tolerance=config["compression"]["accuracy_loss_tolerance_pp"] / 100)
    clusters = sorted({r["cluster_key"] for r in joined})
    indexes = np.random.default_rng(config["bootstrap"]["seed"]).integers(
        0, len(clusters), size=(config["bootstrap"]["draws"], len(clusters)))
    summary["claim_scope"] = config["claim_scope"]
    summary["arm_inventory"] = arms
    summary["bias_correction"] = bias_and_selection(joined, timings, config, indexes)
    requests = {r["request_id"] for r in public_rows}
    summary["end_to_end_speedups"] = _speedups(timings, requests)
    summary["inference"].pop("pooled_steering", None)
    summary["inference"]["ab_bias_baseline"] = summary["bias_correction"]["fit_policy"]
    summary["counts"] = {"rows": len(joined), "requests": len(public_rows),
                         "frames": len({r["frame_key"] for r in joined}),
                         "clusters": len(clusters), "arms": len(arms)}
    by_arm = defaultdict(list)
    for r in joined:
        by_arm[r["variant"]].append(r)
    raw = {v: _cluster_metrics(rows) for v, rows in by_arm.items()}
    summary["binary_log_loss"] = {v: _metric_report(m, raw["native"], indexes)["binary_log_loss_nats"]
                                   for v, m in raw.items()}
    return summary


def render_report(summary):
    c = summary["counts"]
    lines = ["# Decoder Token Matrix", "", summary["claim_scope"], "",
             f"Complete: {c['arms']} conditions, {c['frames']} frames, {c['clusters']} participant clusters, {c['rows']} prediction rows.",
             "", "All intervals are unadjusted descriptive 95% cluster-bootstrap intervals. Ranking many configurations is exploratory.",
             "", "## Compression", "",
             "| Configuration | Accuracy change (pp) | 95% interval | Forward speedup | End-to-end speedup |",
             "|---|---:|---:|---:|---:|"]
    fmt = lambda x: "unavailable" if x is None else f"{x:.3f}x"
    for r in summary["reports"]:
        for contrast in r["comparisons"]:
            v = contrast["variant"]
            a = contrast["participant_equal_difference_variant_minus_reference"]["accuracy"]
            ci = a["ci95"]
            interval = "unavailable" if ci is None else f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]"
            lines.append(f"| {v} | {100*a['estimate']:.2f} | {interval} | {fmt(contrast['timing'].get('speedup_reference_over_variant'))} | {fmt(summary['end_to_end_speedups'].get(v))} |")
    lines += ["", "## Accuracy Correction", "",
              "Five participant folds fit a scalar A/B offset without evaluation-fold labels. The JSON contains each arm's corrected accuracy, both-orders accuracy and binary log-loss, compared with both native and bias-corrected native.",
              "Configuration selection also occurs inside each outer fold. The policies below are evaluated on their excluded fold, not the data that picked the settings.",
              "", "| Nested selection policy + bias | Accuracy change vs native (pp) | Descriptive 95% interval |",
              "|---|---:|---:|"]
    for policy, values in summary["bias_correction"]["selected_policies"].items():
        a = values["vs_native"]["accuracy"]["difference_vs_reference"]
        lines.append(f"| {policy} | {100*a['estimate']:.2f} | [{100*a['ci95'][0]:.2f}, {100*a['ci95'][1]:.2f}] |")
    lines += ["", "A/B calibration is a scoring correction, not evidence of improved visual grounding. No learned adapter is included in this compression report.",
              "Speed and memory measurements concern batch-one answer scoring, not autoregressive generation or KV-cache savings.",
              "See analysis_summary.json for fold memberships, selection receipts, log-loss, transitions and all intervals; efficiency_summary.json records token counts and estimated decoder matrix operations."]
    return "\n".join(lines) + "\n"
