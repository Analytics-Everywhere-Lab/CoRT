"""Two accuracy/latency hypotheses on REUSED descriptive development data.

E1 corrects readout at L34 before half-F/half-C pruning at that same cut. E2
deletes F after L30 and corrects CQ at L34. Deletion/readout is a comparator.
``fused`` is a historical path label, not a single-kernel claim. All optimized
paths require bit-exact agreement with their legacy counterparts before timing.
Importing this module never imports torch.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

EXPERIMENT_ID = "role-aware-policy-2026-10-19-v2"
JOB_REL = "docs/research/lifecycle_memory/phase 2 research/jobs/role_aware_policy"
CUT = 34
PRIMARY = "F0_L30_prune"              # delete all scene tokens after block 30
SECONDARY = "Fhalf_Chalf_L34_prune"   # keep half scene + half candidates after L34
TRAIN_CASES = (f"{SECONDARY}__correct_L34_readout", f"{PRIMARY}__correct_L34_CQ",
               f"{PRIMARY}__correct_L34_readout")
REFERENCE_CASES = ("native__correct_L34_readout", "native__correct_L34_CQ")
SEEDS = (17, 29)
SAVED_BIAS_BITS = 5.806289363588375   # frozen answer-bias offset (bias_baseline.json)
SOURCE_FAMILIES = ("thu_read", "escape", "youtube")
TRAINING = {"rank": 4, "steps": 256, "gradient_accumulation": 4, "learning_rate": .001,
            "checkpoint_selection": "fixed_final_step", "canary_steps": 2}
MEASUREMENT = {"batch_size": 1, "canary_frames": 16, "benchmark_requests": 128,
               "timing_repeats": 5, "canary_timing_repeats": 2, "warmups": 2}
BOOTSTRAP = {"draws": 10000, "seed": 20261019}
DECISION = {"family_alpha": .05, "bounds": 4, "accuracy_floor": 0.0, "speedup_floor": 1.0,
            "co_primaries": {"E1": f"{SECONDARY}__readout_fused_seed_mean",
                             "E2": f"{PRIMARY}__CQ_fused_seed_mean"}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def case_parts(case_id):
    compression, _, target = case_id.partition("__correct_L34_")
    require(compression in ("native", PRIMARY, SECONDARY) and target in ("CQ", "readout"),
            f"invalid correction case: {case_id}")
    return compression, target


def variant_specs():
    """26 frozen variants; adapters are referenced by (case, seed)."""
    specs = [{"id": "native", "compression_id": "native", "path": "native"},
             {"id": PRIMARY, "compression_id": PRIMARY, "path": "matrix"},
             {"id": SECONDARY, "compression_id": SECONDARY, "path": "matrix"}]
    for case in TRAIN_CASES:
        compression, target = case_parts(case)
        specs.append({"id": f"{compression}__{target}_fused_zero", "compression_id": compression,
                      "target": target, "path": "fused_zero"})
    for seed in SEEDS:
        for case in (*TRAIN_CASES, *REFERENCE_CASES):
            compression, target = case_parts(case)
            for path in ("legacy", "fused"):
                specs.append({"id": f"{compression}__{target}_{path}_seed{seed}",
                              "compression_id": compression, "target": target,
                              "path": path, "case_id": case, "seed": seed})
    require(len(specs) == 26 and len({s["id"] for s in specs}) == 26, "variant inventory drift")
    return specs


def seed_mean_groups():
    """Seed-mean report name -> the two per-seed variant ids it averages."""
    groups = {}
    for spec in variant_specs():
        if "seed" in spec:
            groups.setdefault(spec["id"].rsplit("_seed", 1)[0] + "_seed_mean", []).append(spec["id"])
    require(all(len(v) == len(SEEDS) for v in groups.values()), "seed group drift")
    return groups


def identity_pairs():
    """Pairs whose A/B logits must be bit-exact before timing starts."""
    pairs = []
    for spec in variant_specs():
        if spec["path"] == "fused_zero":
            pairs.append((spec["id"], spec["compression_id"]))
        elif spec["path"] == "fused":
            pairs.append((spec["id"], spec["id"].replace("_fused_", "_legacy_")))
    return pairs


def validate_config(config):
    require(config.get("schema_version") == 2 and config.get("experiment_id") == EXPERIMENT_ID,
            "experiment identity drift")
    require(config.get("cut") == CUT, "correction site drift")
    require(config.get("primary_compression") == PRIMARY and
            config.get("secondary_compression") == SECONDARY, "compression drift")
    require(config.get("training_cases") == list(TRAIN_CASES) and
            config.get("reference_cases") == list(REFERENCE_CASES) and config.get("seeds") == list(SEEDS),
            "case/seed drift")
    require(config.get("training") == TRAINING, "training budget drift")
    require(config.get("measurement") == MEASUREMENT, "timing budget drift")
    require(config.get("bootstrap") == BOOTSTRAP, "bootstrap drift")
    require(config.get("decision") == DECISION, "decision contract drift")
    require(config.get("saved_bias_bits") == SAVED_BIAS_BITS, "frozen bias offset drift")
    require(config.get("variants") == variant_specs(),
            "variant inventory drift")
    seen = set()
    for adapter in config.get("adapters", []):
        key = (adapter["case_id"], adapter["seed"])
        require(key not in seen, "duplicate adapter")
        seen.add(key)
        require(adapter["case_id"] in (*TRAIN_CASES, *REFERENCE_CASES) and adapter["seed"] in SEEDS,
                "unknown adapter")
        require(adapter["parameters"] == 43008 and adapter["rank"] == 4 and adapter["cut"] == CUT,
                "adapter shape drift")
        require(is_digest(adapter["sha256"]), "adapter hash missing")
        require(adapter.get("target") == case_parts(adapter["case_id"])[1], "adapter target drift")
        require(isinstance(adapter.get("path"), str) and adapter["path"] not in ("", ".", "..")
                and "/" not in adapter["path"] and "\\" not in adapter["path"], "unsafe adapter path")
    return config


def adapters_complete(config):
    keys = {(a["case_id"], a["seed"]) for a in config.get("adapters", [])}
    return (len(config.get("adapters", [])) == len(keys)
            and keys == {(case, seed) for case in (*TRAIN_CASES, *REFERENCE_CASES) for seed in SEEDS})


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


# --------------------------------------------------------------------------- fused correction

def correction_indices(schedule, original_positions, cut=CUT):
    """Positions the correction touches at ``cut``, in post-compression coordinates.

    Mirrors composite_context: stages STRICTLY BEFORE the cut remap the original
    positions; pruned members disappear and merged members alias.
    """
    mapped = schedule.remap_positions(tuple(original_positions), cut - 1)
    positions = tuple(sorted({p for p in mapped if p is not None}))
    require(positions, "correction target has no surviving positions")
    return positions


class IndexCache:
    """Device index tensors keyed by the identity of an immutable cached plan.

    Built during warmup only; a timed miss is an error, so timed requests never
    construct tensors or remap positions.
    """

    def __init__(self, cut=CUT):
        self._entries = {}
        self.sealed = False
        self.cut = cut

    def get(self, schedule, target, positions_fn, device):
        """``positions_fn()`` supplies original positions; it runs only on a miss."""
        import torch
        key = (id(schedule), target, str(torch.device(device)))
        entry = self._entries.get(key)
        if entry is None:
            if self.sealed:
                raise RuntimeError("index cache miss after sealing")
            positions = correction_indices(schedule, positions_fn(), self.cut)
            entry = (schedule, positions, torch.tensor(positions, dtype=torch.long, device=device))
            self._entries[key] = entry
        if entry[0] is not schedule:
            raise RuntimeError("plan identity changed under the index cache")
        return entry[2]

    def seal(self):
        require(self._entries, "empty index cache cannot be sealed")
        self.sealed = True

    def report(self):
        return {"entries": len(self._entries), "sealed": self.sealed,
                "corrected_positions": sorted({len(e[1]) for e in self._entries.values()})}


@contextmanager
def fused_correction(cut_layer, adapter, index):
    """Apply ``adapter`` to the selected rows of the cut block's output, once.

    Register before entering any compression context: forward hooks run in
    registration order, so a same-cut compression compacts the corrected output
    (correction, then compression), exactly as the legacy path does.
    """
    import torch
    state = {"applications": 0}

    def hook(module, args, output):
        if state["applications"]:
            raise RuntimeError("correction cut executed more than once")
        state["applications"] += 1
        hidden = output if torch.is_tensor(output) else output[0]
        corrected = adapter(hidden.index_select(1, index))
        steered = hidden.index_copy(1, index, corrected)
        if torch.is_tensor(output):
            return steered
        return (steered, *output[1:]) if isinstance(output, tuple) else [steered, *output[1:]]

    handle = cut_layer.register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()
    if state["applications"] != 1:
        raise RuntimeError("correction cut must execute exactly once")


class OptimizedDecoder:
    """Single-threaded inference wrappers installed once, not on every request.

    Native and legacy requests pass through inactive wrappers. Unused Gemma KV
    exports are disabled for the whole session (and restored on exit); native
    logits must pass the original-vs-session equivalence gate. No kernels are
    fused here. Device indexes are owned by immutable schedule identities.
    """

    def __init__(self, model, layers, device, cut=CUT):
        import torch
        from research.lifecycle_memory.application_pilot_steering import _validate_architecture
        self.model, self.layers = model, tuple(layers)
        self.device, self.cut = torch.device(device), cut
        _validate_architecture(model, self.layers)
        require(0 <= cut < len(self.layers) and len({id(x) for x in self.layers}) == len(self.layers),
                "invalid decoder/cut")
        require(all(not m.training for m in model.modules()) and
                all(not p.requires_grad and p.grad is None for p in model.parameters()),
                "base must already be frozen and eval")
        members = {id(m) for m in model.modules()}
        require(all(id(layer) in members for layer in self.layers), "foreign decoder layer")
        require(all(not layer._forward_hooks and not layer._forward_pre_hooks
                    and not hasattr(layer, "_decoder_token_matrix_active") for layer in self.layers),
                "external decoder hooks are unsupported")
        self.entries, self.active, self.sealed = {}, None, False
        self.originals = []
        self.exports = [(m, m.store_full_length_kv) for layer in self.layers
                        for m in layer.modules() if hasattr(m, "store_full_length_kv")]

    def prepare(self, schedule):
        import torch
        entry = self.entries.get(id(schedule))
        if entry is not None:
            require(entry[0] is schedule, "plan identity changed")
            return entry
        if self.sealed:
            raise RuntimeError("device plan cache miss after sealing")
        require(not schedule.stages or schedule.stages[-1].cut < len(self.layers), "cut outside decoder")
        stages = {stage.cut: stage for stage in schedule.stages}
        tensors = {}

        def index(values):
            values = tuple(values)
            if values not in tensors:
                tensors[values] = torch.tensor(values, dtype=torch.long, device=self.device)
            return tensors[values]

        blocks = []
        for i in range(len(self.layers)):
            groups = schedule.groups_at(i - 1)
            active = tuple(g[-1] for g in groups)
            side = index(active) if len(active) != schedule.original_length else None
            prune = None
            stage = stages.get(i)
            if stage is not None:
                require(all(len(r) == 1 for r in stage.reductions), "optimized path supports pruning only")
                if stage.groups != stage.input_groups:
                    source = {g[-1]: j for j, g in enumerate(stage.input_groups)}
                    prune = index(source[p] for p in stage.keep)
            blocks.append((len(active), side, prune))
        entry = (schedule, tuple(blocks), len(tensors))
        self.entries[id(schedule)] = entry
        return entry

    def seal(self):
        require(bool(self.entries), "cannot seal empty device plans")
        self.sealed = True

    def report(self):
        return {"plans": len(self.entries), "device_index_tensors": sum(e[2] for e in self.entries.values()),
                "sealed": self.sealed, "base_frozen_eval": True, "unused_kv_exports_disabled": True,
                "per_request_model_walk": False, "single_kernel_claim": False}

    @contextmanager
    def activate(self, schedule, adapter=None, index=None):
        import torch
        require(self.originals and self.active is None, "decoder session inactive or nested")
        entry = self.entries.get(id(schedule))
        if entry is None:
            raise RuntimeError("device plan cache miss; prepare before activation")
        require(entry[0] is schedule, "plan identity changed")
        require((adapter is None) == (index is None), "adapter/index mismatch")
        if index is not None:
            require(index.device == self.device and index.dtype == torch.long, "correction index device/dtype")
        self.active = [entry, adapter, index, 0]
        try:
            with torch.no_grad():
                yield
            require(self.active[3] == len(self.layers), "incomplete optimized decoder forward")
        finally:
            self.active = None

    def _wrap(self, number, original):
        import torch

        def forward(*args, **kwargs):
            state = self.active
            if state is None:
                return original(*args, **kwargs)
            require(number == state[3], "one ordered decoder forward required")
            state[3] += 1
            schedule = state[0][0]
            length, keep, prune = state[0][1][number]
            require(len(args) <= 2 and (len(args) < 2 or args[1] is None), "unsupported positional inputs")
            require(not kwargs.get("use_cache", False) and not kwargs.get("return_shared_kv_states", False)
                    and all(kwargs.get(k) is None for k in ("past_key_value", "past_key_values", "per_layer_input")),
                    "KV/generation/per-layer inputs unsupported")
            hidden = args[0] if args else kwargs["hidden_states"]
            require(torch.is_tensor(hidden) and hidden.ndim == 3 and hidden.shape[:2] == (1, length)
                    and hidden.device == self.device, "unexpected hidden geometry/device")
            if keep is not None:
                kwargs = dict(kwargs)

                def restrict(value, axis):
                    require(torch.is_tensor(value) and value.ndim >= abs(axis)
                            and value.shape[axis] == schedule.original_length and value.device == self.device,
                            "side input must retain original coordinates and device")
                    return value.index_select(axis, keep)

                kwargs["position_ids"] = restrict(kwargs.get("position_ids"), -1)
                rope = kwargs.get("position_embeddings")
                require(isinstance(rope, (tuple, list)) and len(rope) == 2, "missing original RoPE pair")
                kwargs["position_embeddings"] = tuple(restrict(v, -2) for v in rope)
                if kwargs.get("cache_position") is not None:
                    kwargs["cache_position"] = restrict(kwargs["cache_position"], -1)
                mask = kwargs.get("attention_mask")
                if mask is None:
                    require(getattr(getattr(self.layers[number], "self_attn", None), "layer_type", None)
                            == "full_attention", "missing explicit sliding mask")
                else:
                    require(torch.is_tensor(mask) and mask.ndim == 4, "unexpected mask")
                    kwargs["attention_mask"] = restrict(restrict(mask, -2), -1)
            output = original(*args, **kwargs)
            require(torch.is_tensor(output) and output.shape == hidden.shape, "unexpected decoder output")
            if number == self.cut and state[1] is not None:
                output = output.index_copy(1, state[2], state[1](output.index_select(1, state[2])))
            return output if prune is None else output.index_select(1, prune)
        return forward

    def __enter__(self):
        require(not self.originals, "decoder session already installed")
        self.originals = [(layer, "forward" in layer.__dict__, layer.__dict__.get("forward"), layer.forward)
                          for layer in self.layers]
        for module, _ in self.exports:
            module.store_full_length_kv = False
        for i, (layer, _, _, original) in enumerate(self.originals):
            layer.forward = self._wrap(i, original)
        return self

    def __exit__(self, *_):
        for layer, existed, value, _ in self.originals:
            if existed:
                layer.forward = value
            else:
                layer.__dict__.pop("forward", None)
        for module, value in self.exports:
            module.store_full_length_kv = value
        self.originals = []
        self.active = None


# --------------------------------------------------------------------------- analysis

def _cluster_vectors(predictions, labels, clusters):
    """Per-variant unit means; both candidate orders stay in their unit."""
    from collections import defaultdict
    import numpy as np
    cells = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        label = labels[row["request_id"]]
        margin = row["margin_ab"]
        correct = row["prediction"] == label
        signed = margin * math.log(2) * (1 if label == "A" else -1)
        loss = max(0., -signed) + math.log1p(math.exp(-abs(signed)))
        cells[row["variant"]][row["cluster_key"]].append((float(correct), loss))
    out = {}
    for variant, by_cluster in cells.items():
        require(set(by_cluster) == set(clusters), f"cluster coverage drift: {variant}")
        out[variant] = {
            "accuracy": np.array([np.mean([x[0] for x in by_cluster[c]]) for c in clusters]),
            "log_loss_nats": np.array([np.mean([x[1] for x in by_cluster[c]]) for c in clusters])}
    return out


def _bias_vectors(predictions, labels, clusters, variant, bias_bits):
    """Frozen answer-bias offset applied to an existing variant's margins."""
    shifted = [{**row, "variant": f"{variant}_frozen_bias",
                "margin_ab": row["margin_ab"] - bias_bits,
                "prediction": "A" if row["margin_ab"] - bias_bits >= 0 else "B"}
               for row in predictions if row["variant"] == variant]
    return _cluster_vectors(shifted, labels, clusters)[f"{variant}_frozen_bias"]


def summarize(public, predictions, timings, labels, *, draws=BOOTSTRAP["draws"],
              seed=BOOTSTRAP["seed"], source_equal=True):
    """Source-equal pooled estimates with participant/recording-cluster bootstrap.

    ``labels`` maps request_id -> "A"/"B". Accuracy, log-loss, paired accuracy
    differences against native, total warm request speedups, and the frozen
    joint decision ("faster and more accurate than native") are reported.
    """
    from collections import defaultdict
    import numpy as np

    specs = {s["id"]: s for s in variant_specs()}
    validate_observations(public, predictions, timings, labels)
    clusters = sorted({r["cluster_key"] for r in public})
    sources = {r["cluster_key"]: r.get("source_family") for r in public}
    rng = np.random.default_rng(seed)

    def resampler(keys):
        groups = ([np.array([i for i, k in enumerate(keys) if sources[k] == s]) for s in SOURCE_FAMILIES]
                  if source_equal else [np.arange(len(keys))])
        require(all(len(g) for g in groups), "missing source coverage")
        index = [g[rng.integers(0, len(g), size=(draws, len(g)))] for g in groups]

        def stats(array):
            return (float(np.mean([array[g].mean() for g in groups])),
                    np.mean([array[i].mean(axis=1) for i in index], axis=0))
        return stats

    acc_stats = resampler(clusters)
    alpha = DECISION["family_alpha"] / DECISION["bounds"]

    def interval(array, stats):
        estimate, samples = stats(array)
        return {"estimate": estimate, "ci95": np.quantile(samples, [.025, .975]).tolist(),
                "family_lower_bound": float(np.quantile(samples, alpha))}

    vectors = _cluster_vectors(predictions, labels, clusters)
    require(set(vectors) == set(specs), "prediction variant inventory drift")
    for name, members in seed_mean_groups().items():
        vectors[name] = {m: np.mean([vectors[v][m] for v in members], axis=0)
                         for m in ("accuracy", "log_loss_nats")}
    for variant in ("native", PRIMARY, SECONDARY):
        vectors[f"{variant}_frozen_bias"] = _bias_vectors(predictions, labels, clusters, variant,
                                                          SAVED_BIAS_BITS)
    metrics = {name: {m: interval(v, acc_stats) for m, v in vals.items()} for name, vals in vectors.items()}
    differences = {name: interval(vals["accuracy"] - vectors["native"]["accuracy"], acc_stats)
                   for name, vals in vectors.items() if name != "native"}
    for variant in DECISION["co_primaries"].values():
        differences[f"{variant}_minus_native_frozen_bias"] = interval(
            vectors[variant]["accuracy"] - vectors["native_frozen_bias"]["accuracy"], acc_stats)

    timed_clusters = sorted({r["cluster_key"] for r in timings})
    time_stats = resampler(timed_clusters)
    cells = defaultdict(dict)
    for row in timings:
        cells[row["variant"]][(row["request_id"], row["repeat_id"])] = row
    require(set(cells) == set(specs), "timing variant inventory drift")

    def cluster_means(variant, field):
        grouped = defaultdict(list)
        for row in cells[variant].values():
            grouped[row["cluster_key"]].append(row[field])
        return np.array([np.mean(grouped[c]) for c in timed_clusters])

    native_total = cluster_means("native", "end_to_end_elapsed_ms")
    timing = {}
    total_by_variant = {v: cluster_means(v, "end_to_end_elapsed_ms") for v in cells}
    for name, members in seed_mean_groups().items():
        total_by_variant[name] = np.mean([total_by_variant[v] for v in members], axis=0)
    for name, total in total_by_variant.items():
        a_mean, a_samples = time_stats(native_total)
        b_mean, b_samples = time_stats(total)
        ratio = a_samples / b_samples
        timing[name] = {"native_ms": a_mean, "request_ms": b_mean, "speedup": a_mean / b_mean,
                        "speedup_ci95": np.quantile(ratio, [.025, .975]).tolist(),
                        "speedup_family_lower_bound": float(np.quantile(ratio, alpha)),
                        "latency_change_percent": 100 * (b_mean / a_mean - 1)}
        if name in cells:
            timing[name]["model_forward_ms"] = time_stats(cluster_means(name, "model_forward_ms"))[0]
            if all("peak_allocated_bytes" in r and "peak_reserved_bytes" in r for r in cells[name].values()):
                timing[name]["memory_bytes"] = {
                    field: {"source_unit_equal_mean": time_stats(cluster_means(name, field))[0],
                            "max": max(r[field] for r in cells[name].values())}
                    for field in ("peak_allocated_bytes", "peak_reserved_bytes")}
    overhead = {"zero_minus_matching_compression_forward_ms": {
        s["id"]: timing[s["id"]]["model_forward_ms"] - timing[s["compression_id"]]["model_forward_ms"]
        for s in variant_specs() if s["path"] == "fused_zero"}}

    def joint(name):
        return {"accuracy_lower_bound": differences[name]["family_lower_bound"],
                "speedup_lower_bound": timing[name]["speedup_family_lower_bound"],
                "accuracy_pass": differences[name]["family_lower_bound"] > DECISION["accuracy_floor"],
                "speed_pass": timing[name]["speedup_family_lower_bound"] > DECISION["speedup_floor"]}

    decisions = {}
    for experiment, variant in DECISION["co_primaries"].items():
        result = joint(variant)
        decisions[experiment] = {"variant": variant, **result,
                                "upper_left_pass": result["accuracy_pass"] and result["speed_pass"]}
    return {"counts": {"frames": len({r["frame_key"] for r in public}), "requests": len(public),
                       "clusters": len(clusters), "timed_clusters": len(timed_clusters)},
            "metrics": metrics, "accuracy_differences_vs_native": differences, "timing": timing,
            "overhead": overhead,
            "decision": {"co_primaries": decisions, "winner_selection": False,
                         "method": f"Four one-sided percentile bootstrap bounds at alpha={alpha:g} "
                                   "(Bonferroni across BOTH experiments and accuracy/speed); source-equal "
                                   "unit weighting and source-stratified paired cluster bootstrap; both "
                                   "orders, seeds and all timing repeat pairs retained within units."},
            "scope": ("REUSED 447-frame descriptive development evaluation (job 621156), NOT fresh confirmation. "
                      "Conditional on frozen adapters and batch-one next-answer timing; "
                      "no participant-generalization, generation, or KV-cache claim.")}


def render_markdown(summary):
    """Short human-readable result table."""
    t, d = summary["timing"], summary["accuracy_differences_vs_native"]
    rows = [s["id"] for s in variant_specs() if "seed" not in s] + list(seed_mean_groups())
    lines = ["# Role-Aware Policy Results", "",
             "| Variant | Accuracy | Change vs native (pp) [95% CI] | Request ms | Latency change |",
             "|---|---:|---:|---:|---:|"]
    for name in rows:
        acc = summary["metrics"][name]["accuracy"]["estimate"] * 100
        if name == "native":
            change = "reference"
        else:
            diff = d[name]
            change = f"{diff['estimate']*100:+.2f} [{diff['ci95'][0]*100:+.2f}, {diff['ci95'][1]*100:+.2f}]"
        lines.append(f"| {name} | {acc:.2f}% | {change} | {t[name]['request_ms']:.2f} | "
                     f"{t[name]['latency_change_percent']:+.2f}% |")
    for experiment, p in summary["decision"]["co_primaries"].items():
        lines += ["", f"{experiment}: **{'PASS' if p['upper_left_pass'] else 'FAIL'}** "
                  f"(accuracy lower bound {p['accuracy_lower_bound']*100:+.2f} pp; "
                  f"speedup lower bound {p['speedup_lower_bound']:.4f}x)."]
    lines += ["", summary["decision"]["method"], "", summary["scope"]]
    return "\n".join(lines) + "\n"


def validate_observations(public, predictions, timings, labels):
    """Fail closed before deriving estimates; never collapse duplicate rows."""
    by_id = {r["request_id"]: r for r in public}
    require(len(by_id) == len(public) and public, "duplicate/empty public inventory")
    require(set(by_id) <= set(labels) and all(labels[r] in ("A", "B") for r in by_id), "invalid labels")
    specs = {s["id"] for s in variant_specs()}
    expected = {(rid, v) for rid in by_id for v in specs}
    seen = [(r["request_id"], r["variant"]) for r in predictions]
    require(len(set(seen)) == len(seen) and set(seen) == expected, "prediction inventory mismatch")
    frames, units = {}, {}
    for row in public:
        source = row.get("source_family")
        require(source in SOURCE_FAMILIES, "unknown source")
        unit = row["cluster_key"]
        require(units.setdefault(unit, source) == source, "cluster crosses sources")
        identity = (unit, source)
        frame = frames.setdefault(row["frame_key"], {"identity": identity, "orders": []})
        require(frame["identity"] == identity, "frame identity mismatch")
        frame["orders"].append(row["order_id"])
    require(all(sorted(f["orders"]) == ["order_0", "order_1"] for f in frames.values()), "unpaired frames")
    for row in [*predictions, *timings]:
        require(row.get("status", "pass") == "pass" and not row.get("error"), "failed observation")
        source = by_id.get(row["request_id"])
        require(source is not None and all(row.get(k) == source.get(k) for k in
                ("frame_key", "cluster_key", "source_family", "order_id")), "observation identity drift")
    for row in predictions:
        require(math.isfinite(row["margin_ab"]) and row["prediction"] ==
                ("A" if row["margin_ab"] >= 0 else "B"), "invalid prediction")
    keys = [(r["request_id"], r["variant"], r["repeat_id"]) for r in timings]
    require(keys and len(keys) == len(set(keys)), "duplicate/empty timings")
    reference = {(r, k) for r, v, k in keys if v == "native"}
    require(set(keys) == {(r, v, k) for r, k in reference for v in specs}, "unpaired timing inventory")
    repeats = {k for _, k in reference}
    ids = {r for r, _ in reference}
    require(all(type(k) is int and k >= 0 for k in repeats) and repeats == set(range(len(repeats)))
            and reference == {(r, k) for r in ids for k in repeats}, "timing repeat schedule drift")
    for row in timings:
        require(all(math.isfinite(row[k]) and row[k] > 0 for k in
                    ("end_to_end_elapsed_ms", "model_forward_ms")), "nonfinite/nonpositive timing")
