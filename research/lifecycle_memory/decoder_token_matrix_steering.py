"""Optional, outcome-blind compression plus learned correction matrix.

Importing this module or calling cases/inventory helpers never imports torch.
Training uses only existing public train inputs and labels/train.jsonl. These
800 frames were previously examined: test is primary, not fresh confirmation.
All 14 cases are fixed before this run, not selected from the 71-arm outcomes.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

SEEDS = (17, 29)
VARIANTS = ("native", "compression_only", "adapted")
TIMING_SCOPE = ("synchronized model forward including matrix compression and optional correction; "
                "excludes preparation and schedule construction; adapter resident for all controls")
PUBLIC_FIELDS = {"request_id", "item_id", "frame_key", "cluster_key", "split_group_key",
                 "split", "order_id", "arm", "system_prompt", "user_prompt", "inputs",
                 "schema_version", "frame_unit_id", "arms", "counterbalanced_request_id"}


def cases():
    """Resolve the fixed 14 cases from family plus exact stage geometry."""
    from research.lifecycle_memory.decoder_token_matrix import build_arms
    arms = build_arms()

    def find(family, cut=None, targets=None, fractions=None):
        def matches(arm):
            families = {arm["family"], *arm.get("group_memberships", [])}
            if family not in families:
                return False
            if cut is None:
                return not arm["stages"]
            return (len(arm["stages"]) == 1
                    and arm["stages"][0]["cut"] == cut
                    and arm["stages"][0]["method"] == "merge"
                    and arm["stages"][0].get("targets", {}) == (targets or {})
                    and arm["stages"][0].get("fractions", {}) == (fractions or {}))
        found = [arm for arm in arms if matches(arm)]
        if len(found) != 1:
            raise ValueError(f"ambiguous/missing frozen steering geometry: {family}/{cut}/{targets}/{fractions}")
        return found[0]["id"]

    designs = [
        (find("native"), ("C", "Q", "CQ", "FQ", "readout")),
        (find("F_only", 34, {"F": 133}), ("C", "Q", "CQ", "readout")),
        (find("F_only", 34, {"F": 67}), ("CQ", "readout")),
        (find("late_subset", 34, fractions={"F": .5, "C": .5}), ("Q",)),
        (find("late_subset", 34, fractions={"F": .5, "R": .5}), ("CQ",)),
        (find("matched_total_C", 22, {"C": 379}), ("FQ",)),
    ]
    return [{"id": f"{compression}__correct_L34_{target}",
             "compression_id": compression, "target": target}
            for compression, targets in designs for target in targets]


def validate_config(config):
    settings = config["steering"]
    for key, expected in {"case_count": 14, "seeds": [17, 29], "rank": 4,
                          "steps": 256, "gradient_accumulation": 4,
                          "learning_rate": .001, "checkpoint_selection": "final",
                          "canary_steps": 2, "canary_eval_per_split": 4}.items():
        if settings.get(key) != expected:
            raise ValueError(f"frozen steering setting changed: {key}")
    for key, expected in {"benchmark_requests": 64, "timing_repeats": 3,
                          "canary_timing_repeats": 2, "warmups": 2}.items():
        if config["compression"].get(key) != expected:
            raise ValueError(f"frozen timing setting changed: {key}")


def evaluation_rows(public_rows, mode, config):
    from research.lifecycle_memory.run_decoder_application_pilots import steering_eval_rows
    if mode not in ("canary", "full"):
        raise ValueError("mode must be canary or full")
    return steering_eval_rows(public_rows, mode, config["steering"]["canary_eval_per_split"])


def expected_prediction_keys(public_rows, mode, config):
    """Keys are (comparison_target, seed, variant, request_id)."""
    selected = evaluation_rows(public_rows, mode, config)
    return {(case["id"], seed, variant, row["request_id"])
            for case in cases() for seed in SEEDS for variant in VARIANTS for row in selected}


def benchmark_ids(public_rows, mode, config):
    from research.lifecycle_memory.run_decoder_application_pilots import stable_order
    selected = evaluation_rows(public_rows, mode, config)
    if mode == "full":
        selected = sorted(selected, key=lambda r: stable_order(r["request_id"]))[:64]
    return {r["request_id"] for r in selected}


def expected_timing_keys(public_rows, mode, config):
    """Prediction keys with a final repeat_id, excluding warmups."""
    ids = benchmark_ids(public_rows, mode, config)
    repeats = 2 if mode == "canary" else 3
    return {(*key, repeat) for key in expected_prediction_keys(public_rows, mode, config)
            if key[-1] in ids for repeat in range(repeats)}


def target_positions(partition, target):
    if target == "readout":
        return (partition.answer_prefix.positions[-1],)
    roles = {"C": partition.candidate_crops.positions,
             "Q": partition.literal_user_prompt.positions, "F": partition.full_frame.positions}
    if target not in ("C", "Q", "CQ", "FQ"):
        raise ValueError("unknown correction target")
    return tuple(sorted({p for role in target for p in roles[role]}))


@contextmanager
def composite_context(model, layers, schedule, target, original_positions, adapter, training=False):
    """Replay-safe matrix schedule plus the original frozen-base steering driver.

    Each captured forward contains its own stateless side-input restriction and
    compression, so checkpoint replay cannot bypass compression hooks. Original
    positions map through stages STRICTLY BEFORE the correction cut. Same-cut
    compression is an outer wrapper, after the correction, outside checkpoint.
    Keep this context alive through backward. No generation or external hooks.
    """
    import torch
    from research.lifecycle_memory.application_pilot_steering import steering_context
    from research.lifecycle_memory.decoder_token_matrix import compact_states
    layers = tuple(layers)
    if not layers or not 0 <= target.layer_index < len(layers):
        raise ValueError("correction cut outside decoder")
    if schedule.stages and schedule.stages[-1].cut >= len(layers):
        raise ValueError("compression cut outside decoder")
    marker = "_decoder_token_matrix_active"
    if any(hasattr(layer, marker) or layer._forward_hooks or layer._forward_pre_hooks for layer in layers):
        raise ValueError("cannot stack matrix contexts or external block hooks")
    mapped = schedule.remap_positions(original_positions, target.layer_index - 1)
    positions = tuple(sorted({p for p in mapped if p is not None}))
    if not positions:
        raise ValueError("correction target has no surviving positions")
    plans = {stage.cut: stage for stage in schedule.stages}
    originals = [(layer, "forward" in layer.__dict__, layer.__dict__.get("forward"), layer.forward)
                 for layer in layers]
    audit = {"selected_positions": positions, "original_positions": tuple(original_positions),
             "pruned_target_members": sum(p is None for p in mapped),
             "same_cut_order": "correction_then_compression", "sequence_lengths": {},
             "schedule_sha256": schedule.receipt()["sha256"], "block_calls": {}}

    def wrap(index, original):
        groups = schedule.groups_at(index - 1)
        active = tuple(group[-1] for group in groups)

        def forward(*args, **kwargs):
            if len(args) > 2 or (len(args) == 2 and args[1] is not None):
                raise ValueError("only hidden_states and optional None may be positional")
            if (any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value", "per_layer_input"))
                    or kwargs.get("use_cache", False)):
                raise ValueError("KV cache/generation/per-layer inputs unsupported")
            hidden = args[0] if args else kwargs["hidden_states"]
            if hidden.ndim != 3 or hidden.shape[:2] != (1, len(active)):
                raise ValueError(f"unexpected input geometry at block {index}")
            audit["sequence_lengths"][str(index)] = len(active)
            audit["block_calls"][str(index)] = audit["block_calls"].get(str(index), 0) + 1
            if len(active) != schedule.original_length:
                kwargs = dict(kwargs)
                keep = torch.tensor(active, dtype=torch.long, device=hidden.device)

                def restrict(value, axis):
                    if (not torch.is_tensor(value) or value.ndim < abs(axis)
                            or value.shape[axis] != schedule.original_length):
                        raise ValueError("side input must have original sequence coordinates")
                    return value.index_select(axis, keep.to(value.device))

                kwargs["position_ids"] = restrict(kwargs.get("position_ids"), -1)
                rope = kwargs.get("position_embeddings")
                if not isinstance(rope, (tuple, list)) or len(rope) != 2:
                    raise ValueError("missing original RoPE pair")
                kwargs["position_embeddings"] = tuple(restrict(value, -2) for value in rope)
                if kwargs.get("cache_position") is not None:
                    kwargs["cache_position"] = restrict(kwargs["cache_position"], -1)
                mask = kwargs.get("attention_mask")
                if mask is None:
                    if getattr(getattr(layers[index], "self_attn", None), "layer_type", None) != "full_attention":
                        raise ValueError("missing explicit multimodal sliding mask")
                elif not torch.is_tensor(mask) or mask.ndim != 4:
                    raise ValueError("unexpected native attention-mask shape")
                else:
                    kwargs["attention_mask"] = restrict(restrict(mask, -2), -1)
            output = original(*args, **kwargs)
            if not torch.is_tensor(output) or output.shape != hidden.shape:
                raise ValueError("unexpected raw decoder output geometry")
            if index in plans and index != target.layer_index:
                output = compact_states(output, plans[index])
            return output
        return forward

    try:
        for index, (layer, _, _, original) in enumerate(originals):
            setattr(layer, marker, True)
            layer.forward = wrap(index, original)
        with steering_context(model, layers, target.layer_index, positions, adapter, training) as receipt:
            cut_layer = layers[target.layer_index]
            corrected_forward = cut_layer.forward
            if target.layer_index in plans:
                stage = plans[target.layer_index]

                def corrected_then_compressed(*args, **kwargs):
                    return compact_states(corrected_forward(*args, **kwargs), stage)

                cut_layer.forward = corrected_then_compressed
            try:
                audit["steering_receipt"] = receipt
                yield audit
                if len(audit["sequence_lengths"]) != len(layers):
                    raise ValueError("incomplete decoder forward")
            finally:
                cut_layer.forward = corrected_forward
    finally:
        for layer, existed, value, _ in originals:
            if existed:
                layer.forward = value
            else:
                layer.__dict__.pop("forward", None)
            layer.__dict__.pop(marker, None)


def _schedule(engine, prepared, compression_id):
    from research.lifecycle_memory.decoder_token_matrix import build_arms, make_schedule
    arm = next(arm for arm in build_arms() if arm["id"] == compression_id)
    return make_schedule(prepared, arm, engine.model.config.vision_config.pooling_kernel_size,
                         tokenizer=engine.tokenizer)


def _target(case):
    from research.lifecycle_memory.application_pilot_steering import SteeringTarget
    return SteeringTarget(case["id"], 34, case["target"])


@contextmanager
def correction_forward(engine, prepared, adapter, case, training=False, schedule=None):
    schedule = schedule if schedule is not None else _schedule(engine, prepared, case["compression_id"])
    with composite_context(engine.model, engine.layers, schedule, _target(case),
                           target_positions(prepared.pilot_partition, case["target"]), adapter, training):
        yield engine.logits(prepared)


def _public_only(rows):
    from research.lifecycle_memory.application_pilot_data import SCHEMA_VERSION
    by_id = {r["request_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate public request")
    for row in rows:
        unexpected = sorted(set(row) - PUBLIC_FIELDS)
        if unexpected or row.get("arm", "real") != "real":
            raise ValueError("method inputs must be real, label-free public schema rows; "
                             f"unexpected fields={unexpected}; arm={row.get('arm', 'real')}")
        # The dataset exporter adds these label-free aliases for existing engines.
        # Accept only their exact real-input meaning, never arbitrary nested data.
        if "schema_version" in row and (type(row["schema_version"]) is not int
                                         or row["schema_version"] != SCHEMA_VERSION):
            raise ValueError("unsupported public schema_version")
        if "frame_unit_id" in row and row["frame_unit_id"] != row["frame_key"]:
            raise ValueError("public frame alias mismatch")
        if "arms" in row and row["arms"] != {"real": {"inputs": row["inputs"]}}:
            raise ValueError("public arms must alias real inputs only")
        if "counterbalanced_request_id" in row:
            other = by_id.get(row["counterbalanced_request_id"])
            if (other is None or other.get("counterbalanced_request_id") != row["request_id"]
                    or {row["order_id"], other["order_id"]} != {"order_0", "order_1"}
                    or any(other.get(key) != row.get(key) for key in
                           ("frame_key", "cluster_key", "split_group_key", "split"))):
                raise ValueError("public counterbalance alias mismatch")


def training_rows(rows, dataset):
    """Read exactly the permitted label file and verify train identity/split."""
    from research.lifecycle_memory.application_pilot_steering import TrainingRow
    from research.lifecycle_memory.run_decoder_application_pilots import read_rows
    _public_only(rows)
    public = {r["request_id"]: r for r in rows if r["split"] == "train"}
    labels = read_rows(Path(dataset) / "labels/train.jsonl")
    if len(labels) != len(public) or {r["request_id"] for r in labels} != set(public):
        raise ValueError("train label/public inventories differ")
    by_id = {}
    for label in labels:
        row = public[label["request_id"]]
        if label.get("correct_label") not in ("A", "B") or any(
                label.get(key) != row.get(key) for key in
                ("split", "frame_key", "cluster_key", "order_id", "split_group_key")):
            raise ValueError("train label/public identity mismatch")
        by_id[label["request_id"]] = label["correct_label"]
    if not public:
        raise ValueError("empty training pool")
    return [TrainingRow(key, row, int(by_id[key] == "B"), "train") for key, row in public.items()]


def _actual_checkpoint(result, case):
    """Replace the new legacy-dispatch artifact with truthfully named metadata."""
    destination = result.checkpoint.with_name(f"{case['id']}.seed{result.seed}.final{result.checkpoint.suffix}")
    if destination.exists():
        raise FileExistsError("refusing to replace a correction checkpoint")
    if result.checkpoint.suffix == ".safetensors":
        from safetensors.torch import save_file
        state = {name: value.detach().cpu().contiguous() for name, value in result.adapter.state_dict().items()}
        save_file(state, str(destination), metadata={
            "target": case["id"], "positions": case["target"], "layer_index": "34",
            "compression_id": case["compression_id"], "legacy_driver_target": result.target.name,
            "selection": "fixed_final_step", "steps": str(result.optimizer_steps),
            "seed": str(result.seed), "rank": "4", "canary": str(result.canary)})
        result.checkpoint.unlink()
    else:
        # Direct tensor state has no embedded target; its sibling receipt does.
        result.checkpoint.rename(destination)
    return destination


def scope_metadata():
    return {"selection": "all 14 cases declared outcome-blind before this run; no best-of-71 selection",
            "primary_split": "test", "validation_use": "A/B bias calibration only, no checkpoint/case selection",
            "prior_exposure": "800 frames previously examined; not fresh confirmation",
            "training": "existing public split=train and labels/train.jsonl only; no folds or donor/oracle inputs",
            "base": "frozen BF16", "rank": 4, "seeds": list(SEEDS), "steps": 256,
            "gradient_accumulation": 4, "learning_rate": .001, "checkpoint_selection": "fixed_final_step",
            "same_cut_order": "correction_then_compression", "timing_scope": TIMING_SCOPE,
            "timing_inventory": "64 outcome-blind validation/test requests per case/seed, 3 warm repeats; test report uses test subset",
            "control_budget": "controls carry the paired adapter training budget, but do not apply the adapter"}


def _log_loss_comparison(left_cells, right_cells, indexes):
    """Average orders/frames within each seed/participant, then seeds equally.

    margins are A-minus-B in bits; binary cross entropy is reported in nats.
    Resample whole participants with all seeds paired, not individual requests.
    """
    from research.lifecycle_memory import application_pilot_analysis as analysis

    def participant_losses(cell):
        groups = {}
        for row in cell.values():
            if row.get("error"):
                raise ValueError("error predictions cannot supply binary log loss")
            margin = row.get("margin")
            if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin):
                raise ValueError("binary log loss requires finite A/B margins")
            z = margin * math.log(2) * (1 if row["correct_label"] == "A" else -1)
            loss = max(0., -z) + math.log1p(math.exp(-abs(z)))
            groups.setdefault(row["cluster_key"], []).append(loss)
        return {cluster: sum(values) / len(values) for cluster, values in groups.items()}

    left = [participant_losses(cell) for cell in left_cells]
    right = [participant_losses(cell) for cell in right_cells]
    clusters = sorted(left[0])
    if len(left) != len(right) or any(set(values) != set(clusters) for values in left + right):
        raise ValueError("binary log loss requires paired seed/participant inventories")
    reference = [sum(values[cluster] for values in left) / len(left) for cluster in clusters]
    variant = [sum(values[cluster] for values in right) / len(right) for cluster in clusters]
    return {"reference": analysis._estimate(reference, indexes),
            "variant": analysis._estimate(variant, indexes),
            "difference_variant_minus_reference": analysis._estimate([b - a for a, b in zip(reference, variant)], indexes),
            "unit": "nats; lower is better; equal participants, equal seeds within participant"}


def _paired_comparison(left, right, indexes, left_timing=None, right_timing=None):
    from research.lifecycle_memory import application_pilot_analysis as analysis
    return {**analysis._comparison(left, right, indexes, left_timing, right_timing),
            "binary_log_loss_nats": _log_loss_comparison(left, right, indexes)}


def steering_worker(args, config, engine, rows, dataset, out):
    """Old runner contract; use the old Engine, never its old steering targets."""
    from research.lifecycle_memory import run_decoder_application_pilots as pilot
    from research.lifecycle_memory.application_pilot_steering import LowRankSteering, train_adapter
    from research.lifecycle_memory.decoder_token_matrix import matrix_context
    validate_config(config)
    if type(args.world_size) is not int or args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("invalid worker shard")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    training = training_rows(rows, dataset)
    evaluations = evaluation_rows(rows, args.mode, config)
    full_evaluations = evaluation_rows(rows, "full", config)
    if not evaluations:
        raise ValueError("empty evaluation pool")
    benchmark = benchmark_ids(rows, args.mode, config)
    full_benchmark = benchmark_ids(rows, "full", config)
    settings = config["steering"]
    text_config = getattr(engine.model.config, "text_config", engine.model.config)
    hidden_size = text_config.hidden_size
    if any(p.is_floating_point() and p.dtype != engine.torch.bfloat16 for p in engine.model.parameters()):
        raise ValueError("correction release requires BF16 base weights")
    tasks = [(case, seed) for seed in SEEDS for case in cases()][args.rank::args.world_size]
    technical = {"status": "pass", "scope": scope_metadata(), "cases": [],
                 "runtime_gpu_canary_required": True}
    done, projected = 0, 0.
    for case, seed in tasks:
        task_start = time.perf_counter()
        taskdir = out / f"{case['id']}_seed{seed}"
        taskdir.mkdir()
        sample = engine.prepare(training[0].row, args.data_dir)
        plan = _schedule(engine, sample, case["compression_id"])
        zero = LowRankSteering(hidden_size, 4).to(engine.device)
        with engine.torch.no_grad():
            with matrix_context(engine.layers, plan):
                baseline = engine.logits(sample)
            with correction_forward(engine, sample, zero, case, schedule=plan) as identity:
                if not engine.torch.equal(baseline, identity) or not engine.torch.isfinite(identity).all():
                    raise RuntimeError("zero correction differs from compression-only logits")
        pilot.atomic_json(taskdir / "zero_adapter_gate.json", {"status": "pass", **case,
            "baseline": "compression_only", "ab_logits_bit_exact": True,
            "schedule": plan.receipt()})
        del sample, plan, zero, baseline, identity
        train_start = time.perf_counter()
        engine.torch.cuda.reset_peak_memory_stats()
        first_step = {}

        def progress(receipt):
            if receipt.step == 1:
                first_step["s"] = time.perf_counter() - train_start
            pilot.atomic_json(taskdir / "progress.json", {"status": "training", **asdict(receipt)})

        def forward(prepared, adapter, driver_target, training):
            # The legacy driver validates a fixed name. It does not choose the
            # actual layer/positions: this callback always uses the real case.
            return correction_forward(engine, prepared, adapter, case, training)

        result = train_adapter(engine.model, training, hidden_size=hidden_size, target="l34_cq",
            prepare=lambda row: engine.prepare(row, args.data_dir), forward=forward,
            checkpoint_dir=taskdir, rank=4, seed=seed, device=engine.device,
            optimizer_steps=2 if args.mode == "canary" else 256,
            early_report=progress, canary=args.mode == "canary")
        engine.torch.cuda.synchronize()
        train_s = time.perf_counter() - train_start
        checkpoint = _actual_checkpoint(result, case)
        projected_train = pilot.projected_training_seconds(train_s, first_step["s"], result.optimizer_steps, 256)
        budget = {"seed": seed, "rank": 4, "parameters": result.parameter_count,
                  "training_pool_size": len(training), "training_examples_seen": len(result.data_order),
                  "steps": result.optimizer_steps, "gradient_accumulation": 4, "learning_rate": .001,
                  "checkpoint_selection": "fixed_final_step", "base_dtype": "bfloat16",
                  "data_order_sha256": hashlib.sha256(json.dumps(result.data_order).encode()).hexdigest()}
        pilot.atomic_json(taskdir / "training_receipt.json", {"status": "pass", **case, **budget,
            "actual_target": asdict(_target(case)), "legacy_driver_target": asdict(result.target),
            "legacy_target_is_dispatch_placeholder": True, "scope": scope_metadata(),
            "train_s": train_s, "first_step_s": first_step["s"], "projected_full_train_s": projected_train,
            "peak_allocated_bytes": engine.torch.cuda.max_memory_allocated(),
            "final_update_norm": result.final_update_norm, "checkpoint": checkpoint.name,
            "checkpoint_sha256": pilot.digest(checkpoint),
            "step_receipts": [asdict(receipt) for receipt in result.receipts]})
        technical["cases"].append({**case, "seed": seed, "zero_adapter_identity": True,
            "backward_nonzero": all(r.nonzero_gradient for r in result.receipts),
            "frozen_base_checked": all(r.frozen_base_checked for r in result.receipts)})
        prediction_s, benchmark_s, benchmark_count, warmup_s = 0., 0., 0, 0.
        for row_index, row in enumerate(evaluations):
            prepare_start = time.perf_counter()
            prepared = engine.prepare(row, args.data_dir)
            plan = _schedule(engine, prepared, case["compression_id"])
            engine.torch.cuda.synchronize()
            prediction_s += time.perf_counter() - prepare_start

            def compressed():
                with matrix_context(engine.layers, plan):
                    return engine.logits(prepared)

            def adapted():
                with correction_forward(engine, prepared, result.adapter, case, schedule=plan) as value:
                    return value

            variants = [("native", lambda: engine.logits(prepared)),
                        ("compression_only", compressed), ("adapted", adapted)]
            if row_index == 0:
                warm_start = time.perf_counter()
                for _, fn in variants:
                    for _ in range(2):
                        with engine.torch.no_grad():
                            fn()
                engine.torch.cuda.synchronize()
                warmup_s = time.perf_counter() - warm_start
            random.Random(pilot.stable_order(f"{case['id']}:{seed}:{row['request_id']}")).shuffle(variants)
            for variant, fn in variants:
                logits, metrics = pilot.measure(engine.torch, fn)
                metrics["timing_scope"] = TIMING_SCOPE
                prediction_s += metrics["elapsed_ms"] / 1000
                common = {"comparison_target": case["id"], "compression_id": case["compression_id"],
                          "correction_target": case["target"], "matched_training_budget": budget,
                          "trained_adapter_applied": variant == "adapted"}
                pilot.append_row(out / "predictions.jsonl", pilot.prediction(row, logits,
                    experiment="steering", variant=variant, seed=seed, rank=args.rank,
                    metrics={**metrics, **common}))
                done += 1
            if row["request_id"] in benchmark:
                for repeat in range(2 if args.mode == "canary" else 3):
                    random.Random(pilot.stable_order(f"timing:{case['id']}:{seed}:{row['request_id']}:{repeat}")).shuffle(variants)
                    for variant, fn in variants:
                        _, timing = pilot.measure(engine.torch, fn)
                        timing["timing_scope"] = TIMING_SCOPE
                        benchmark_s += timing["elapsed_ms"] / 1000
                        benchmark_count += 1
                        pilot.append_row(out / "timings.jsonl", {"task_id": "steering", "variant": variant,
                            "seed": seed, "request_id": row["request_id"], "comparison_target": case["id"],
                            "matched_training_budget": budget, "rank": args.rank, "repeat_id": repeat,
                            "is_warmup": False, "warmup_complete": True, **timing})
            del prepared, plan
        overhead_s = max(0., time.perf_counter() - task_start - train_s - prediction_s - benchmark_s)
        projected += (projected_train + overhead_s + prediction_s * len(full_evaluations) / len(evaluations)
                      + benchmark_s / max(1, benchmark_count) * len(full_benchmark) * 3 * 3)
        pilot.atomic_json(taskdir / "progress.json", {"status": "complete", "train_s": train_s,
            "warmup_s": warmup_s, "unscaled_overhead_s": overhead_s,
            "eval_requests": len(evaluations), "projected_full_train_s": projected_train})
        del result
        engine.torch.cuda.empty_cache()
    technical["status"] = "pass" if all(c["backward_nonzero"] and c["frozen_base_checked"] for c in technical["cases"]) else "fail"
    pilot.atomic_json(out / "technical_gate.json", technical)
    if technical["status"] != "pass":
        raise RuntimeError("steering backward/frozen-base gate failed")
    return done, projected, technical


def aggregate_steering(predictions, timings, labels, public_rows, config):
    """Per-case paired metrics; test primary, validation -> test bias only.

    Reuse the pilot's participant/seed-paired primitives, not its public entry
    point: that entry point pools validation+test and cross-fits on test labels.
    Removing comparison_target is safe only AFTER isolating one fixed case.
    """
    import numpy as np
    from research.lifecycle_memory import application_pilot_analysis as analysis
    validate_config(config)
    key = lambda row: (row["comparison_target"], row["seed"], row["variant"], row["request_id"])
    actual = [key(row) for row in predictions]
    if len(actual) != len(set(actual)) or any(row["task_id"] != "steering" for row in predictions):
        raise ValueError("duplicate/invalid steering predictions")
    if any(row.get("error") for row in predictions):
        raise ValueError("error predictions cannot be scored as valid finite-margin successes")
    modes = [mode for mode in ("full", "canary") if set(actual) == expected_prediction_keys(public_rows, mode, config)]
    if not modes:
        raise ValueError("incomplete frozen steering prediction inventory")
    mode = config.get("mode", modes[0])
    if mode not in modes:
        raise ValueError("steering inventory differs from declared mode")
    timing_keys = [(*key(row), row["repeat_id"]) for row in timings]
    if (len(timing_keys) != len(set(timing_keys))
            or set(timing_keys) != expected_timing_keys(public_rows, mode, config)
            or any(row.get("task_id") != "steering" or row.get("is_warmup") is not False
                   or row.get("warmup_complete") is not True for row in timings)):
        raise ValueError("incomplete/invalid paired steering timing inventory")
    selected = evaluation_rows(public_rows, mode, config)
    joined = analysis.join_labels(predictions, labels)
    draws = config.get("bootstrap", {}).get("draws", 2000)
    bootstrap_seed = config.get("bootstrap", {}).get("seed", 20261014)
    if type(draws) is not int or draws < 100:
        raise ValueError("bootstrap_draws must be >= 100")
    if type(bootstrap_seed) is not int:
        raise ValueError("bootstrap seed must be an integer")
    reports = []
    for case in cases():
        def isolate(values):
            return [{k: v for k, v in row.items() if k != "comparison_target"}
                    for row in values if row["comparison_target"] == case["id"]]
        case_rows = isolate(joined)
        cells = analysis._validate(case_rows, selected, "native")
        times = analysis._timing_cells(isolate(timings), cells, "native")
        get = lambda source, variant, split="test": [source.get(("steering", split, variant, seed, ""), {}) for seed in SEEDS]
        n = len({row["cluster_key"] for row in get(cells, "native")[0].values()})
        indexes = np.random.default_rng(bootstrap_seed).integers(0, n, size=(draws, n)) if n > 1 else None
        budgets = {}
        for seed in SEEDS:
            values = [row.get("matched_training_budget") for row in case_rows if row["seed"] == seed]
            if not values or any(not isinstance(v, dict) or v != values[0] for v in values):
                raise ValueError("matched controls require identical seed/data/training budget receipts")
            budget = values[0]
            expected = {"seed": seed, "rank": 4, "steps": 2 if mode == "canary" else 256,
                        "gradient_accumulation": 4, "learning_rate": .001,
                        "checkpoint_selection": "fixed_final_step", "base_dtype": "bfloat16"}
            expected.update(training_pool_size=sum(row["split"] == "train" for row in public_rows),
                            training_examples_seen=expected["steps"] * 4)
            order_hash = budget.get("data_order_sha256", "")
            if (any(budget.get(k) != v for k, v in expected.items())
                    or not isinstance(order_hash, str) or len(order_hash) != 64
                    or any(char not in "0123456789abcdef" for char in order_hash)
                    or type(budget.get("parameters")) is not int or budget["parameters"] <= 0):
                raise ValueError("invalid matched training budget")
            budgets[str(seed)] = budget
        if reports and budgets != reports[0]["matched_training_budget"]:
            raise ValueError("cases must share initialization seed and training data/order/budget")
        if any(row.get("matched_training_budget") != budgets[str(row["seed"])]
               for row in isolate(timings)):
            raise ValueError("timing matched training budget differs from predictions")
        comparisons = []
        for reference, variant in (("native", "compression_only"), ("native", "adapted"), ("compression_only", "adapted")):
            left, right = get(cells, reference), get(cells, variant)
            lt, rt = get(times, reference), get(times, variant)
            comparisons.append({"reference": reference, "variant": variant,
                **_paired_comparison(left, right, indexes, lt, rt),
                "per_seed": {str(seed): _paired_comparison([a], [b], indexes, [at], [bt])
                             for seed, a, b, at, bt in zip(SEEDS, left, right, lt, rt)}})
        calibration = {}
        for reference in ("native", "compression_only"):
            fitted = [analysis.fit_ab_bias(cell) for cell in get(cells, reference, "validation")]
            if any(bias is None for bias in fitted):
                calibration[reference] = {"available": False, "reason": "missing valid validation margins"}
                continue
            biased = [analysis.apply_ab_bias(cell, bias) for cell, bias in zip(get(cells, reference), fitted)]
            calibration[reference] = {"available": True, "fit_policy": "validation->test",
                "fitted_bias_bits": dict(zip(map(str, SEEDS), fitted)),
                "bias_minus_reference": _paired_comparison(get(cells, reference), biased, indexes),
                "adapted_minus_bias": _paired_comparison(biased, get(cells, "adapted"), indexes),
                "per_seed": {str(seed): {
                    "bias_minus_reference": _paired_comparison([raw], [calibrated], indexes),
                    "adapted_minus_bias": _paired_comparison([calibrated], [adapted], indexes)}
                    for seed, raw, calibrated, adapted in zip(SEEDS, get(cells, reference), biased, get(cells, "adapted"))}}
        reports.append({**case, "task_id": "steering", "comparison_target": case["id"],
            "split_scope": "test", "report_role": "primary", "matched_training_budget": budgets,
            "variants": {variant: {"pooled_seeds": analysis._summary(get(cells, variant), indexes),
                                   "per_seed": {str(seed): analysis._summary([cell], indexes)
                                                for seed, cell in zip(SEEDS, get(cells, variant))}}
                         for variant in VARIANTS},
            "comparisons": comparisons, "ab_bias_baseline": calibration})
    return {"schema_version": 1, "scope": scope_metadata(), "mode": mode, "cases": cases(),
            "reports": reports, "coverage": {"prediction_rows": len(predictions), "timing_rows": len(timings),
            "expected_inventory_verified": True}, "inference": {
                "unit": "participant cluster_key; equal participants and paired seeds/orders",
                "bootstrap_draws": draws, "bootstrap_seed": bootstrap_seed,
                "interval": "unadjusted descriptive percentile 95%; no equivalence or fresh-confirmation claim"}}


def render_report(report):
    """Small standalone Markdown report, with no pooled validation/test claim."""
    lines = ["# Optional Learned Correction", "", "Test primary; validation used only for A/B bias calibration.",
             "800 frames previously examined; not fresh confirmation. All 14 cases retained.", "",
             "Timing is synchronized model-forward only; excludes preparation and schedule construction.",
             "No application end-to-end speedup claim. Binary A/B log loss is in nats (lower is better).", "",
             "| Case | Reference | Variant | Paired accuracy delta | Paired log-loss delta |",
             "| --- | --- | --- | --- | --- |"]
    for result in report["reports"]:
        for contrast in result["comparisons"]:
            delta = contrast["participant_equal_difference_variant_minus_reference"]["accuracy"]["estimate"]
            loss = contrast["binary_log_loss_nats"]["difference_variant_minus_reference"]["estimate"]
            lines.append(f"| {result['id']} | {contrast['reference']} | {contrast['variant']} | {delta} | {loss} |")
    return "\n".join(lines) + "\n"
