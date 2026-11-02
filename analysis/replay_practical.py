"""Offline replay of anonymous practical unit statistics; no model imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCES = ("thu_read", "escape", "youtube")
SESSIONS = ("6469516", "6478505", "621156")
COMPONENTS = ("preparation_ms", "plan_lookup_ms", "model_forward_ms", "postprocess_ms")
METRICS = ("accuracy", "log_loss_nats")
MEMORY = ("peak_allocated_bytes", "peak_reserved_bytes")
PRIMARY = "F0_L30_prune"
SECONDARY = "Fhalf_Chalf_L34_prune"
EARLY_CASES = ("native__correct_L34_FQ", "native__correct_L34_CQ",
               "native__correct_L34_readout", "C379_L22_merge__correct_L34_FQ",
               "F67_L34_merge__correct_L34_readout")
FIELDS = ("counts", "metrics", "accuracy", "accuracy_differences_vs_native",
          "accuracy_differences", "timing", "overhead", "decision", "primary_decisions",
          "diagnostic_decisions", "retention_diagnostics", "per_source", "per_source_retention")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric_projection(value):
    """Drop prose/provenance, retaining numerical leaves and fixed scientific names."""
    if isinstance(value, dict):
        return {k: clean for k, v in value.items()
                if (clean := numeric_projection(v)) is not None}
    if isinstance(value, list):
        return [numeric_projection(v) for v in value]
    if isinstance(value, (int, float)):
        require(math.isfinite(value), "nonfinite result")
        return value
    return None


def project_summary(summary):
    result = {k: numeric_projection(summary[k]) for k in FIELDS if k in summary}
    if "per_source" in summary and "accuracy" in summary:
        result["per_source"] = {s: project_summary(part) for s, part in summary["per_source"].items()}
    return result


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def compare(actual, expected, path="result"):
    """Check every saved field, including failures; never refit or replace CIs."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{path}: fields differ")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{path}: length differs")
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{path}[{index}]")
    elif isinstance(expected, bool):
        require(type(actual) is bool and actual == expected, f"{path}: decision differs")
    elif isinstance(expected, (float, int)):
        require(not isinstance(actual, bool) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12),
                f"{path}: numeric mismatch ({actual!r} != {expected!r})")
    else:
        require(actual == expected, f"{path}: value differs")


def inventory(session):
    """Frozen variant names, separate from any private configuration."""
    if session == "6469516":
        from research.lifecycle_memory import role_aware_policy as policy
        specs = policy.variant_specs()
        return [s["id"] for s in specs], policy.seed_mean_groups(), policy.DECISION["co_primaries"]
    if session == "6478505":
        names = ["native"] + [f"{p}_readout_seed{s}" for p in ("native", "F133_L34") for s in (17, 29)]
        return names, {}, {f"seed{s}": f"F133_L34_readout_seed{s}" for s in (17, 29)}
    require(session == "621156", "unknown session")
    names = ["native", SECONDARY, "C384_L22_merge__F67_L34_merge", "F0_L22_prune",
             PRIMARY, "F0_L34_prune", "C379_L22_merge", "F67_L34_merge"]
    names += [f"{c}_seed{s}" for c in EARLY_CASES for s in (17, 29)]
    return names, {c + "_seed_mean": [f"{c}_seed{s}" for s in (17, 29)] for c in EARLY_CASES}, {}


def validate(data):
    require(set(data) == {"schema_version", "session", "bootstrap", "units", "timed_positions",
                          "variants", "frozen_bias", "counts", "source_status"}, "unknown data fields")
    session = data["session"]
    require(session in SESSIONS and data["schema_version"] == 1, "unknown schema/session")
    early = session == "621156"
    require(data["source_status"] == "pass", "source analysis did not pass")
    require(data["bootstrap"] == {"draws": 10000, "seed": 20261015 if early else 20261019,
                                  "alpha": .05 / (3 if early else 4)}, "bootstrap contract drift")
    units, timed = data["units"], data["timed_positions"]
    require(len(units) == 50 and len(timed) == 44, "unit coverage drift")
    require(all(type(i) is int for i in timed) and timed == sorted(set(timed))
            and min(timed) >= 0 and max(timed) < len(units), "invalid timed membership/order")
    for unit in units:
        require(set(unit) == {"source", "requests", "timing_pairs"}, "unknown unit fields")
        require(unit["source"] in SOURCES and type(unit["requests"]) is int
                and unit["requests"] > 0 and unit["requests"] % 2 == 0, "invalid unit/pair counts")
        require(type(unit["timing_pairs"]) is int and unit["timing_pairs"] >= 0
                and unit["timing_pairs"] % 10 == 0, "invalid timing pair counts")
    require([i for i, u in enumerate(units) if u["timing_pairs"]] == timed, "timed membership drift")
    require(sum(u["requests"] for u in units) == 894
            and sum(u["timing_pairs"] for u in units) == 640, "request coverage drift")
    require(all(any(units[i]["source"] == s for i in timed) for s in SOURCES), "missing source")
    names, _, _ = inventory(session)
    require(set(data["variants"]) == set(names), "variant inventory drift")
    require(data["counts"] == {"frames": 447, "requests": 894, "clusters": 50,
            "timed_clusters": 44, "prediction_rows": 894 * len(names),
            "timing_rows": 640 * len(names)}, "counts drift")

    def vector(values, n, nonnegative=True):
        require(isinstance(values, list) and len(values) == n, "vector length drift")
        require(all(type(v) in (float, int) and math.isfinite(v)
                    and (not nonnegative or v >= 0) for v in values), "invalid vector values")

    def scores(vals):
        require(set(vals) == set(METRICS), "metric fields drift")
        for metric in METRICS:
            vector(vals[metric], len(units))
        require(all(v <= 1 for v in vals["accuracy"]), "accuracy out of bounds")

    time_fields = {"end_to_end_elapsed_ms", *COMPONENTS, *MEMORY}
    if early:
        time_fields.add("elapsed_ms")
    for vals in data["variants"].values():
        require(set(vals) == {"scores", "timing", "memory_max"}, "variant fields drift")
        scores(vals["scores"])
        require(set(vals["timing"]) == time_fields and set(vals["memory_max"]) == set(MEMORY),
                "timing fields drift")
        for values in (*vals["timing"].values(), *vals["memory_max"].values()):
            vector(values, len(timed))
        require(all(v > 0 for v in vals["timing"]["end_to_end_elapsed_ms"]), "nonpositive time")
        require(np.allclose(sum(np.array(vals["timing"][f]) for f in COMPONENTS),
                            vals["timing"]["end_to_end_elapsed_ms"], rtol=1e-6, atol=1e-6),
                "components do not sum to total")
    expected_bias = {"native", PRIMARY, SECONDARY} if session == "6469516" else set()
    require(set(data["frozen_bias"]) == expected_bias, "bias inventory drift")
    for vals in data["frozen_bias"].values():
        scores(vals)


def _replay(data, source=None):
    session = data["session"]
    early = session == "621156"
    names, seed_groups, primaries = inventory(session)
    units = data["units"]
    positions = [i for i, u in enumerate(units) if source is None or u["source"] == source]
    timed = [i for i in data["timed_positions"] if i in positions]
    timing_offsets = [data["timed_positions"].index(i) for i in timed]
    rng = np.random.default_rng(data["bootstrap"]["seed"])

    def resampler(keys):
        groups = ([np.array([i for i, k in enumerate(keys) if units[k]["source"] == s])
                   for s in SOURCES] if source is None else [np.arange(len(keys))])
        indices = [g[rng.integers(0, len(g), size=(data["bootstrap"]["draws"], len(g)))] for g in groups]

        def stats(array):
            return (float(np.mean([array[g].mean() for g in groups])),
                    np.mean([array[i].mean(axis=1) for i in indices], axis=0))
        return stats

    acc_stats = resampler(positions)

    def interval(array, stats=acc_stats):
        point, samples = stats(array)
        return {"estimate": point, "ci95": np.quantile(samples, [.025, .975]).tolist(),
                "family_lower_bound": float(np.quantile(samples, data["bootstrap"]["alpha"]))}

    vectors = {name: {m: np.array(data["variants"][name]["scores"][m])[positions] for m in METRICS}
               for name in names}
    for name, members in seed_groups.items():
        vectors[name] = {m: np.mean([vectors[v][m] for v in members], axis=0) for m in METRICS}
    for name, vals in data["frozen_bias"].items():
        vectors[name + "_frozen_bias"] = {m: np.array(vals[m])[positions] for m in METRICS}
    metrics = {name: {m: interval(v) for m, v in vals.items()} for name, vals in vectors.items()}
    differences = {name: interval(vals["accuracy"] - vectors["native"]["accuracy"])
                   for name, vals in vectors.items() if name != "native"}
    if session == "6469516":
        for name in primaries.values():
            differences[name + "_minus_native_frozen_bias"] = interval(
                vectors[name]["accuracy"] - vectors["native_frozen_bias"]["accuracy"])
    # The original RNG first draws accuracy indices, then timing indices.
    time_stats = resampler(timed)

    def tv(name, field, section="timing"):
        return np.array(data["variants"][name][section][field])[timing_offsets]

    totals = {name: tv(name, "end_to_end_elapsed_ms") for name in names}
    if not early:
        for name, members in seed_groups.items():
            totals[name] = np.mean([totals[v] for v in members], axis=0)
    timing, components = {}, {}
    for name, total in totals.items():
        a, aa = time_stats(totals["native"])
        b, bb = time_stats(total)
        ratio = aa / bb
        common = {"speedup": a / b, "speedup_ci95": np.quantile(ratio, [.025, .975]).tolist()}
        lower = float(np.quantile(ratio, data["bootstrap"]["alpha"]))
        if early:
            timing[name] = {**common, "native_mean_ms": a, "cached_request_mean_ms": b,
                "family_lower_bound": lower, "forward_mean_ms": time_stats(tv(name, "elapsed_ms"))[0],
                "peak_allocated_bytes": float(max(tv(name, "peak_allocated_bytes", "memory_max"))),
                "forward_ms": interval(tv(name, "elapsed_ms"), time_stats),
                "unit_peak_allocated_bytes": interval(tv(name, "peak_allocated_bytes", "memory_max"), time_stats),
                "native_ms": interval(totals["native"], time_stats), "cached_request_ms": interval(total, time_stats),
                "cluster_equal_components_ms": {f: time_stats(tv(name, f))[0] for f in COMPONENTS}}
        else:
            timing[name] = {**common, "native_ms": a, "request_ms": b,
                "speedup_family_lower_bound": lower, "latency_change_percent": 100 * (b / a - 1)}
            if name in names:
                timing[name]["model_forward_ms"] = time_stats(tv(name, "model_forward_ms"))[0]
                timing[name]["memory_bytes"] = {f: {
                    "source_unit_equal_mean": time_stats(tv(name, f))[0],
                    "max": float(max(tv(name, f, "memory_max")))} for f in MEMORY}
        if name in names:
            components[name] = {f: time_stats(tv(name, f))[0] for f in COMPONENTS}
    counts = {"frames": sum(units[i]["requests"] for i in positions) // 2,
              "clusters": len(positions), "timed_clusters": len(timed)}
    if early:
        diagnostic = {"compression_accuracy_pass": differences[SECONDARY]["family_lower_bound"] > -.01,
                      "compression_speed_pass": timing[SECONDARY]["family_lower_bound"] > 1.01,
                      "steering_accuracy_pass": differences["native__correct_L34_FQ_seed_mean"]["family_lower_bound"] > 0,
                      "confirmatory": False}
        diagnostic["compression_joint_pass"] = diagnostic["compression_accuracy_pass"] and diagnostic["compression_speed_pass"]
        result = {"counts": counts, "accuracy": metrics, "accuracy_differences": differences, "timing": timing,
                  "diagnostic_decisions": diagnostic, "primary_decisions": {"confirmatory": False, "eligible": False}}
    else:
        counts["requests"] = counts["frames"] * 2
        decisions = {}
        for experiment, name in primaries.items():
            a, b = differences[name]["family_lower_bound"], timing[name]["speedup_family_lower_bound"]
            decisions[experiment] = {"accuracy_lower_bound": a, "speedup_lower_bound": b,
                                    "accuracy_pass": a > 0, "speed_pass": b > 1, "upper_left_pass": a > 0 and b > 1}
        overhead = {}
        if session == "6469516":
            for name in names:
                if name.endswith("_fused_zero"):
                    overhead[name] = timing[name]["model_forward_ms"] - timing[name.split("__")[0]]["model_forward_ms"]
        result = {"counts": counts, "metrics": metrics, "accuracy_differences_vs_native": differences,
                  "timing": timing, "overhead": {"zero_minus_matching_compression_forward_ms": overhead},
                  "decision": {"co_primaries": decisions, "winner_selection": False}}
        if session == "6478505":
            result["decision"].update(deployment_seed=17, both_seeds_pass=all(d["upper_left_pass"] for d in decisions.values()))
            result["retention_diagnostics"] = {}
            for seed in (17, 29):
                name, baseline = f"F133_L34_readout_seed{seed}", f"native_readout_seed{seed}"
                result["retention_diagnostics"][name] = {
                    m: {k: v for k, v in interval(vectors[name][m] - vectors[baseline][m]).items()
                        if k != "family_lower_bound"} for m in METRICS}
    return result, components


def replay(data):
    validate(data)
    result, components = _replay(data)
    result["per_source"] = {}
    if data["session"] == "6478505":
        result["per_source_retention"] = {}
    for source in SOURCES:
        part, _ = _replay(data, source)
        if data["session"] == "621156":
            result["per_source"][source] = part
        else:
            result["per_source"][source] = part["accuracy_differences_vs_native"]
            if data["session"] == "6478505":
                result["per_source_retention"][source] = part["retention_diagnostics"]
    return result, components


def run(directory, output):
    manifest = json.loads((directory / "manifest.json").read_text())
    require(set(manifest) == {"schema_version", "sessions"} and manifest["schema_version"] == 1,
            "manifest schema drift")
    require(set(manifest["sessions"]) == set(SESSIONS), "session inventory drift")
    report = {"schema_version": 1, "status": "pass", "sessions": {}}
    for session in SESSIONS:
        data = json.loads((directory / f"{session}.json").read_text())
        expected = json.loads((directory / f"{session}_expected.json").read_text())
        require(manifest["sessions"][session] == {"data_sha256": canonical_digest(data),
                "expected_sha256": canonical_digest(expected)}, "public artifact checksum mismatch")
        require(data["session"] == session, "session binding mismatch")
        result, components = replay(data)
        compare(result, expected)
        def count_leaves(value):
            if isinstance(value, dict):
                return sum(count_leaves(v) for v in value.values())
            if isinstance(value, list):
                return sum(count_leaves(v) for v in value)
            return int(isinstance(value, (int, float)) and not isinstance(value, bool))

        report["sessions"][session] = {"status": "pass", "verification": "all_selected_saved_numeric_fields_match",
            "coverage": {"saved_sections": sorted(expected), "numeric_reference_values": count_leaves(expected),
                "prediction_variants": len(data["variants"]), "confidence_intervals": "historical_rng_and_quantiles_matched",
                "raw_request_rescoring": False, "timing_missing_units": 6,
                "excluded": ["raw_predictions", "raw_timing_repeats", "order_specific_diagnostics",
                             "frozen_bias_fitting", "training_receipts", "adapter_weights", "cold_setup_costs"]},
            "counts": data["counts"], "replayed": result, "additional_component_means_ms": components,
            "decision_variant_map": inventory(session)[2],
            "retention_reference_map": ({f"F133_L34_readout_seed{s}": f"native_readout_seed{s}"
                                         for s in (17, 29)} if session == "6478505" else {}),
            "source_status": data["source_status"],
            "primary_status": "not_applicable" if session == "621156" else "descriptive_joint_gates"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "results/replay_practical")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rebuilt/practical_replay.json")
    args = parser.parse_args()
    report = run(args.data_dir, args.output)
    print(json.dumps({s: r["verification"] for s, r in report["sessions"].items()}))


if __name__ == "__main__":
    main()
