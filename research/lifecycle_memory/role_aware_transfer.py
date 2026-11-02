"""Frozen native correction followed by F-only pruning; no model imports."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

from research.lifecycle_memory import role_aware_policy as original

require = original.require
EXPERIMENT_ID = "role-aware-transfer-2026-10-20-v1"
JOB_REL = "docs/research/lifecycle_memory/phase 2 research/jobs/role_aware_transfer"
CONFIG = JOB_REL + "/config.json"
REFERENCE_CONFIG = "docs/research/lifecycle_memory/phase 2 research/jobs/decoder_cached_validation/config.json"
REFERENCE_ADAPTERS = str(Path(REFERENCE_CONFIG).parent / "artifacts/adapters")
REFERENCE_SHA = "272267e90364808ca9efd487c8691824ca3421e90e8ef4145cb3afa3dca53b29"
EVALUATION_SHA = "e8807ff5f26a65191cb6714866ca1338825c01a60df9d3ea0ade9be87bd7343f"
CASE = "native__correct_L34_readout"
COMPRESSION = "F133_L34_prune"
SEEDS = (17, 29)
HASHES = {17: "11ece5ca5ded81021ae05af569db2b95b825a12cbd7389a67287896bce73c39b",
          29: "124b3698defbd884212c9a7ca53d30b959d2ae8a186af5d6a0950a01d2899db3"}
MODEL = {"id": "google/gemma-4-31B-it", "key": "gemma4_31b_hf_bf16",
         "revision": "3548789868c5356dbf307c98e6f609007b82b3eb",
         "torch": "2.12.1+cu126", "transformers": "5.13.0"}
MEASUREMENT = copy.deepcopy(original.MEASUREMENT)
BOOTSTRAP = copy.deepcopy(original.BOOTSTRAP)
DECISION = {"family_alpha": .05, "bounds": 4, "accuracy_floor": 0, "speedup_floor": 1,
            "co_primaries": {f"seed{s}": f"F133_L34_readout_seed{s}" for s in SEEDS}}
SCOPE = ("Prospective development on REUSED 447-frame data (job 621156), NOT fresh confirmation. "
         "Individual frozen adapters, source-equal unit weighting, warm batch-one next-answer timing. "
         "No training, seed selection, ensemble, generation, KV-cache or generalization claim.")
IDENTITY = ("request_id", "frame_key", "cluster_key", "order_id", "source_family", "sampling_unit_kind")


def variant_specs():
    specs = [{"id": "native", "compression_id": "native", "path": "native"}]
    for compression, name in (("native", "native"), (COMPRESSION, "F133_L34")):
        for seed in SEEDS:
            specs.append({"id": f"{name}_readout_seed{seed}", "compression_id": compression,
                          "path": "fused", "case_id": CASE, "seed": seed, "target": "readout"})
    return specs


def source_config():
    return copy.deepcopy({"schema_version": 1, "experiment_id": EXPERIMENT_ID,
        "status": "awaiting_freeze", "training_allowed": False, "cut": 34,
        "compression_id": COMPRESSION, "seeds": list(SEEDS), "deployment_seed": 17,
        "model": MODEL, "measurement": MEASUREMENT, "bootstrap": BOOTSTRAP,
        "decision": DECISION, "variants": variant_specs(), "adapters": [],
        "reference_config_sha256": REFERENCE_SHA, "evaluation_manifest_sha256": EVALUATION_SHA,
        "scope": SCOPE})


def validate_config(config):
    expected = source_config()
    require(set(config) in (set(expected), set(expected) | {"freeze"}), "config fields drift")
    require(all(config.get(k) == v for k, v in expected.items() if k not in ("status", "adapters")),
            "transfer design drift")
    require(config.get("status") in ("awaiting_freeze", "adapters_frozen"), "invalid freeze status")
    records = config.get("adapters")
    if config["status"] == "awaiting_freeze":
        require(records == [] and "freeze" not in config, "unfrozen config contains bindings")
    else:
        require(isinstance(records, list) and len(records) == 2 and
                [a.get("seed") for a in records] == list(SEEDS), "two ordered adapters required")
        for a in records:
            require(all(a.get(k) == v for k, v in {
                "case_id": CASE, "target": "readout", "cut": 34, "rank": 4, "parameters": 43008,
                "sha256": HASHES[a["seed"]], "path": f"{CASE}.seed{a['seed']}.final.safetensors"}.items()),
                "frozen adapter identity drift")
            require(original.is_digest(a.get("source_training_receipt_sha256")), "missing training provenance")
    return config


def estimator():
    """Private module globals: the original estimator/module is never patched.

    The old estimator's optional bias controls are neutralized and removed from
    the public result. No seed means are constructed. Core bootstrap math stays
    exactly in the original implementation.
    """
    spec = importlib.util.spec_from_file_location("_transfer_estimator", original.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.variant_specs = variant_specs
    module.seed_mean_groups = lambda: {}
    module.PRIMARY = module.SECONDARY = "native"
    module.SAVED_BIAS_BITS = 0
    module.DECISION = copy.deepcopy(DECISION)
    return module


def summarize(public, predictions, timings, labels, *, draws=10000, source_equal=True):
    module = estimator()
    options = {"draws": draws, "seed": BOOTSTRAP["seed"], "source_equal": source_equal}
    result = module.summarize(public, predictions, timings, labels, **options)
    for field in ("metrics", "accuracy_differences_vs_native"):
        result[field] = {k: v for k, v in result[field].items() if "frozen_bias" not in k}
    result["scope"] = SCOPE
    result["decision"]["method"] = (
        "Four one-sided percentile bootstrap bounds at alpha=0.0125 across two individual "
        "pruned adapters and accuracy/speed; source-equal unit weighting and source-stratified "
        "paired cluster bootstrap; both orders and all timing repeat pairs retained within units.")
    result["decision"]["deployment_seed"] = 17
    result["decision"]["both_seeds_pass"] = all(
        d["upper_left_pass"] for d in result["decision"]["co_primaries"].values())
    result["retention_diagnostics"] = {}
    # Rebase the existing estimator on each matched adapter, also using its
    # accuracy-difference machinery for log loss. These intervals are descriptive.
    vectors = module._cluster_vectors
    for seed in SEEDS:
        baseline, policy = f"native_readout_seed{seed}", f"F133_L34_readout_seed{seed}"
        pp = [p for p in predictions if p["variant"] != "native"] + [
            {**p, "variant": "native"} for p in predictions if p["variant"] == baseline]
        tt = [t for t in timings if t["variant"] != "native"] + [
            {**t, "variant": "native"} for t in timings if t["variant"] == baseline]
        diagnostic = {}
        for metric in ("accuracy", "log_loss_nats"):
            def metric_vectors(*args, metric=metric):
                return {k: {**v, "accuracy": v[metric]} for k, v in vectors(*args).items()}
            module._cluster_vectors = metric_vectors
            interval = module.summarize(public, pp, tt, labels, **options)["accuracy_differences_vs_native"][policy]
            diagnostic[metric] = {k: interval[k] for k in ("estimate", "ci95")}
        module._cluster_vectors = vectors
        result["retention_diagnostics"][policy] = {"baseline": baseline, **diagnostic,
                                                   "scope": "descriptive; not an equivalence/outcome gate"}
    return result


def render_markdown(summary):
    lines = ["# Role-Aware Transfer Results", "", SCOPE, ""]
    if summary["mode"] == "canary":
        return "\n".join(lines + ["Technical canary only; evaluator labels were not opened.", ""])
    lines += ["| Variant | Accuracy | Request ms |", "|---|---:|---:|"]
    for spec in variant_specs():
        name = spec["id"]
        lines.append(f"| {name} | {100*summary['metrics'][name]['accuracy']['estimate']:.2f}% | "
                     f"{summary['timing'][name]['request_ms']:.2f} |")
    for seed, d in summary["decision"]["co_primaries"].items():
        lines += ["", f"{seed}: **{'PASS' if d['upper_left_pass'] else 'FAIL'}** "
                  f"(accuracy LB {100*d['accuracy_lower_bound']:+.2f} pp; "
                  f"speedup LB {d['speedup_lower_bound']:.4f}x)."]
    lines += ["", summary["decision"]["method"], "", "Seed17 is the predetermined deployment candidate.",
              "Matched retention and per-source effects are descriptive diagnostics in analysis_summary.json.",
              "Original run 6469516 and its failed E1/E2 decisions remain unchanged.", ""]
    return "\n".join(lines)
