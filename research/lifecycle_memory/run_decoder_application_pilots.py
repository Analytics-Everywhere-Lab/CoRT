"""Bounded, offline Gemma31 compression and steering application pilots.

All model execution is compute-node-only. Evaluation workers never read test
labels; the aggregate process joins them after prediction files are closed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
from statistics import mean
import time
import traceback

from research.lifecycle_memory.application_pilot_compression import COMPRESSION_ARMS

MODEL_KEY = "gemma4_31b_hf_bf16"
REVISION = "3548789868c5356dbf307c98e6f609007b82b3eb"
VARIANTS = tuple(COMPRESSION_ARMS)
TARGETS = ("l34_cq", "l22_cq", "l34_readout", "l34_f")


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def append_row(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def stable_order(value):
    return hashlib.sha256(f"application-pilot-v1:{value}".encode()).hexdigest()


def selected_rows(rows, mode, frames):
    if mode == "full":
        return rows
    selected = set(sorted({r["frame_key"] for r in rows}, key=stable_order)[:frames])
    return [r for r in rows if r["frame_key"] in selected]


def steering_eval_rows(rows, mode, limit):
    values = [r for r in rows if r["split"] in ("validation", "test")]
    if mode == "full":
        return values
    if limit < 2 or limit % 2:
        raise ValueError("canary evaluation must retain both candidate orders")
    return [r for split in ("validation", "test")
            for r in selected_rows([r for r in values if r["split"] == split], "canary", limit//2)]


def validate_config(config):
    if config["model"]["revision"] != REVISION or config["model"]["key"] != MODEL_KEY or config["model"]["id"] != "google/gemma-4-31B-it":
        raise ValueError("model revision changed")
    if config["compression"]["variants"] != list(VARIANTS):
        raise ValueError("unexpected compression arm inventory")
    if config["steering"]["targets"] != list(TARGETS) or config["steering"]["seeds"] != [17, 29]:
        raise ValueError("unexpected matched steering design")
    if config["compression"]["remove_count"] != 133 or config["measurement"]["batch_size"] != 1:
        raise ValueError("unsupported compression ratio/batch geometry")
    if config["steering"]["steps"] != 256 or config["steering"]["rank"] != 4:
        raise ValueError("steering budgets must remain matched and frozen")
    if config["steering"]["learning_rate"] != .001 or config["steering"]["gradient_accumulation"] != 4:
        raise ValueError("steering optimizer settings differ from the implementation")
    if config["steering"]["checkpoint_selection"] != "final":
        raise ValueError("no validation or test checkpoint search is allowed")
    if config["compression"]["canary_frames"] != 16 or config["steering"]["canary_steps"] != 2:
        raise ValueError("canary budget changed")


def validate_prediction_inventory(predictions, public_rows, experiment, mode, config):
    """Completeness means every frozen cell, not just a plausible line count."""
    public = {r["request_id"]:r for r in public_rows}
    if len(public) != len(public_rows):
        raise ValueError("duplicate public request")
    expected = set()
    if experiment == "compression":
        selected = selected_rows(public_rows,mode,config["compression"]["canary_frames"])
        expected = {(v,v,0,r["request_id"]) for r in selected for v in VARIANTS}
    else:
        selected = steering_eval_rows(public_rows,mode,config["steering"]["canary_eval_per_split"])
        expected = {(target,v,seed,r["request_id"]) for r in selected
                    for target in TARGETS for seed in config["steering"]["seeds"] for v in (target,"native")}
    seen = set()
    for row in predictions:
        key = (row.get("comparison_target",row["variant"]),row["variant"],row["seed"],row["request_id"])
        if key in seen:
            raise ValueError(f"duplicate prediction cell: {key}")
        seen.add(key)
        source = public.get(row["request_id"])
        if source is None or any(row[k] != source[k] for k in ("frame_key","cluster_key","split","order_id")):
            raise ValueError("prediction/public identity mismatch")
        if row["task_id"] != experiment or row["prediction"] not in ("A","B"):
            raise ValueError("invalid prediction task or answer")
        if not math.isfinite(row["margin_ab"]) or not math.isfinite(row["elapsed_ms"]) or row["elapsed_ms"] <= 0:
            raise ValueError("invalid score/timing")
    if seen != expected:
        raise ValueError(f"incomplete prediction inventory: missing={len(expected-seen)} extra={len(seen-expected)}")
    return {"status":"pass", "cells":len(seen)}


def efficiency_summary(predictions):
    result = {}
    for variant in sorted({r["variant"] for r in predictions}):
        rows = [r for r in predictions if r["variant"] == variant]
        summary = {"requests":len(rows)}
        for field in ("peak_allocated_bytes","peak_reserved_bytes","incremental_peak_allocated_bytes",
                      "starting_allocated_bytes","request_ms_estimate","plan_construction_ms",
                      "decoder_dense_matmul_flops_estimate"):
            if all(field in r for r in rows):
                summary[field] = {"mean":mean(r[field] for r in rows),"max":max(r[field] for r in rows)}
        if all("sequence_lengths" in r for r in rows):
            lengths = {tuple(r["sequence_lengths"]) for r in rows}
            if len(lengths) != 1:
                raise ValueError("a fixed operator changed its token-count schedule")
            summary["sequence_lengths_by_block"] = list(next(iter(lengths)))
        result[variant] = summary
    return {"variants":result,
        "compute_scope":"estimated dense decoder matmul FLOPs; excludes vision, head, normalization, softmax and merging overhead; not measured hardware operations",
        "memory_scope":"single-forward allocated/reserved total peak and incremental allocated peak; frozen weights and full pre-compression activations remain present; no KV cache or autoregressive decoding",
        "latency_scope":"use analysis repeated-timing comparisons for speed claims; request_ms_estimate additionally includes preparation and plan construction"}


class Engine:
    def __init__(self, config):
        import torch
        import transformers
        from research.lifecycle_memory import run_phase2_reanchored_causal_patch as base
        from research.lifecycle_memory import run_phase2_reanchored_answer_token_causal_patch as answer
        from research.lifecycle_memory.application_pilot_compression import architecture_gate
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("one Slurm-bound CUDA device is required; no login/laptop model execution")
        if transformers.__version__ != config["model"]["transformers"] or torch.__version__ != config["model"]["torch"]:
            raise RuntimeError(f"runtime drift: torch={torch.__version__} transformers={transformers.__version__}")
        torch.set_num_threads(8)
        self.torch = torch
        self.answer = answer
        self.runtime, self.model, self.processor, self.tokenizer, self.device, self.provenance = base._load_anchor_model(
            model_key=MODEL_KEY, dtype="bfloat16", local_files_only=True)
        self.model.eval().requires_grad_(False)
        self.model.set_attn_implementation("sdpa")
        self.layers = architecture_gate(self.model)
        if any(p.device.type != "cuda" for p in self.model.parameters()):
            raise RuntimeError("CPU/disk offload would invalidate the efficiency comparison")
        self.runtime_receipt = {"torch":torch.__version__, "transformers":transformers.__version__,
            "model_revision":REVISION, "model_provenance":self.provenance,
            "gemma_source_sha256":digest(__import__(type(self.model).__module__, fromlist=["x"]).__file__),
            "device":torch.cuda.get_device_name(), "bf16":True, "use_cache":False,
            "attention_backend":"sdpa", "batch_size":1}

    def prepare(self, row, data_dir):
        from research.lifecycle_memory.run_patchaudit_heldout_component_closure import _partition_for_prepared
        prepared = self.answer.prepare_arm(runtime=self.runtime, processor=self.processor,
            tokenizer=self.tokenizer, device=self.device, model=self.model, model_key=MODEL_KEY,
            row=row, arm="real", data_dir=Path(data_dir), candidate_completions={"A":"A","B":"B"})
        partition = _partition_for_prepared(prepared)
        counts = [partition.sequence_length, partition.full_frame.count,
                  partition.candidate_crops.count, partition.literal_user_prompt.count,
                  partition.answer_prefix.count, partition.structural_remainder.count]
        if counts != [921,266,512,75,3,65] or partition.padding.count:
            raise RuntimeError(f"frozen prompt/processor geometry changed: {counts}")
        if partition.answer_prefix.positions[-1] != partition.sequence_length - 1:
            raise RuntimeError("answer readout is not the final active position")
        prepared.pilot_partition = partition
        for choice in ("A", "B"):
            if len(prepared.choices[choice].candidate_ids) != 1:
                raise RuntimeError("answers must each be one token")
        return prepared

    def logits(self, prepared):
        output = self.model(**prepared.choices["A"].inputs, use_cache=False, logits_to_keep=1, return_dict=True)
        ids = [prepared.choices[c].candidate_ids[0] for c in ("A","B")]
        return output.logits[0, -1, ids].float()

    def compression_plan(self, prepared, variant, remove_count=133):
        from research.lifecycle_memory.application_pilot_compression import make_plan
        group, _, _, method = COMPRESSION_ARMS[variant]
        p = prepared.pilot_partition
        return make_plan(sequence_length=p.sequence_length,
            slots=[s.positions for s in p.image_slots],
            image_position_ids=prepared.choices["A"].inputs["image_position_ids"],
            pooling_kernel=self.model.config.vision_config.pooling_kernel_size,
            group=group, remove_count=remove_count, method=method)

    def compression_forward(self, prepared, variant, plans):
        from research.lifecycle_memory.application_pilot_compression import compression_context
        if variant == "native":
            return self.logits(prepared), {"sequence_lengths":{str(i):921 for i in range(60)}}
        _, cut, expand, _ = COMPRESSION_ARMS[variant]
        with compression_context(self.layers, plans[variant], cut=cut, expand_before=expand) as audit:
            value = self.logits(prepared)
        return value, audit

    @contextmanager
    def steering_forward(self, prepared, adapter, target, training):
        from research.lifecycle_memory.application_pilot_steering import steering_context
        p = prepared.pilot_partition
        positions = {"C+Q":tuple(sorted(p.candidate_crops.positions + p.literal_user_prompt.positions)),
                     "readout":(p.answer_prefix.positions[-1],), "F":p.full_frame.positions}[target.positions]
        with steering_context(self.model, self.layers, target.layer_index, positions, adapter, training=training):
            yield self.logits(prepared)

    def identity_gate(self, prepared):
        from research.lifecycle_memory.application_pilot_compression import compression_context
        with self.torch.no_grad():
            native = self.logits(prepared)
            plan = self.compression_plan(prepared, "merge_F_L34", remove_count=0)
            with compression_context(self.layers, plan, cut=34):
                same = self.logits(prepared)
        if not self.torch.equal(native, same) or not self.torch.isfinite(native).all():
            raise RuntimeError("keep-all compact adapter does not match native A/B logits bit-exactly")
        return {"status":"pass", "keep_all_ab_logits_bit_exact":True,
                "partition":asdict(prepared.pilot_partition), "serialization":prepared.serialization,
                "scope":"A/B next-token scoring, real input, no cache or free generation"}


def measure(torch, fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall = time.perf_counter()
    with torch.no_grad():
        result = fn()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall) * 1000
    return result, {"elapsed_ms":wall_ms, "cuda_elapsed_ms":start.elapsed_time(end),
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
        "incremental_peak_allocated_bytes":max(0, torch.cuda.max_memory_allocated()-before_allocated),
        "starting_allocated_bytes":before_allocated, "starting_reserved_bytes":before_reserved,
        "timing_scope":"synchronized model forward including compression/expansion or steering; excludes preparation",
        "device_id":f"{os.environ.get('SLURM_JOB_ID')}:{os.environ.get('SLURM_PROCID')}:{torch.cuda.get_device_name()}",
        "batch_size":1,"synchronized":True,
        "reserved_memory_scope":"persistent allocator pool; not reset across conditions"}


def prediction(row, logits, *, experiment, variant, seed, rank, metrics):
    values = logits.detach().cpu().tolist()
    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise RuntimeError("non-finite or invalid answer scores")
    return {**{k:row[k] for k in ("request_id","frame_key","cluster_key","split","order_id")},
        "task_id":experiment, "variant":variant, "seed":seed, "rank":rank,
        "prediction":"A" if values[0] >= values[1] else "B",
        "margin_ab":(values[0]-values[1])/math.log(2), "logits_ab":values, **metrics}


def projected_training_seconds(train_s, first_step_s, steps, full_steps):
    """Count first-step warmup once; scale only steady steps to the full budget."""
    if not 0 < first_step_s <= train_s or steps < 1:
        raise ValueError("invalid training timing")
    if steps == 1:
        return train_s * full_steps
    return first_step_s + (train_s-first_step_s) / (steps-1) * (full_steps-1)


def projected_compression_seconds(measured_s, preparation_s, selected_count,
                                  full_requests, benchmark_requests, repeats, fixed_s=0.0):
    """Measured forward time sums every frozen arm; benchmark counts are per shard."""
    return fixed_s + ((measured_s + preparation_s) * full_requests
            + measured_s * benchmark_requests * repeats) / selected_count


def compression_worker(args, config, engine, rows, out):
    from research.lifecycle_memory.application_pilot_compression import estimated_decoder_flops
    settings = config["compression"]
    all_frames = sorted({r["frame_key"] for r in rows}, key=stable_order)
    selected = selected_rows(rows, args.mode, settings["canary_frames"])
    shard_frames = set(sorted({r["frame_key"] for r in selected}, key=stable_order)[args.rank::args.world_size])
    selected = [r for r in selected if r["frame_key"] in shard_frames]
    benchmark = set(r["request_id"] for r in sorted(rows, key=lambda r:stable_order(r["request_id"]))[:settings["benchmark_requests"]])
    if not selected:
        raise RuntimeError("empty compression worker")
    done = 0
    setup_start = time.perf_counter()
    first = engine.prepare(selected[0], args.data_dir)
    technical = engine.identity_gate(first)
    engine.torch.cuda.synchronize()
    fixed_s = time.perf_counter() - setup_start
    atomic_json(out / "technical_gate.json", technical)
    del first
    started = time.perf_counter()
    measured_s = 0
    preparation_s = 0
    for row_index, row in enumerate(selected):
        prep_start = time.perf_counter()
        prepared = engine.prepare(row, args.data_dir)
        engine.torch.cuda.synchronize()
        prep_ms = 1000*(time.perf_counter()-prep_start)
        preparation_s += prep_ms/1000
        plans, plan_ms = {}, {"native":0.0}
        for variant in VARIANTS:
            if variant != "native":
                plan_start = time.perf_counter()
                plans[variant] = engine.compression_plan(prepared,variant)
                plan_ms[variant] = (time.perf_counter()-plan_start)*1000
        preparation_s += sum(plan_ms.values())/1000
        if row_index == 0:
            atomic_json(out / "compression_plans.json", {v:p.receipt() for v,p in plans.items()})
            warmup_start = time.perf_counter()
            for v in VARIANTS:
                for _ in range(settings["warmups"]):
                    with engine.torch.no_grad():
                        engine.compression_forward(prepared,v,plans)
            engine.torch.cuda.synchronize()
            fixed_s += time.perf_counter() - warmup_start
        variants = list(VARIANTS)
        random.Random(stable_order(row["request_id"])).shuffle(variants)
        for variant in variants:
            (logits, audit), metrics = measure(engine.torch, lambda:engine.compression_forward(prepared,variant,plans))
            metrics["preparation_ms"] = prep_ms
            metrics["plan_construction_ms"] = plan_ms[variant]
            metrics["request_ms_estimate"] = prep_ms + plan_ms[variant] + metrics["elapsed_ms"]
            metrics["request_estimate_scope"] = "shared measured image/prompt preparation + variant plan construction + synchronized forward; not independent end-to-end repetitions"
            lengths = [audit["sequence_lengths"][str(i)] for i in range(60)]
            metrics["decoder_dense_matmul_flops_estimate"] = estimated_decoder_flops(engine.model.config.text_config,lengths)
            metrics["sequence_lengths"] = lengths
            append_row(out / "predictions.jsonl", prediction(row,logits,experiment="compression",variant=variant,seed=0,rank=args.rank,metrics=metrics))
            measured_s += metrics["elapsed_ms"] / 1000
            done += 1
            if args.mode == "canary" or row["request_id"] in benchmark:
                for repeat in range(settings.get("canary_timing_repeats",2) if args.mode == "canary" else settings["timing_repeats"]):
                    (_, _), timing = measure(engine.torch,lambda:engine.compression_forward(prepared,variant,plans))
                    append_row(out / "timings.jsonl", {"task_id":"compression", "seed":0,
                        "request_id":row["request_id"], "variant":variant,
                        "rank":args.rank, "repeat_id":repeat, "is_warmup":False,"warmup_complete":True, **timing})
        atomic_json(out / "progress.json", {"status":"running", "rows_done":done,
            "rows_expected":len(selected)*len(VARIANTS), "elapsed_s":time.perf_counter()-started})
        del prepared, plans
    # Extra repetitions are for a fixed 64-request benchmark in full mode.
    full_shard_frames = set(all_frames[args.rank::args.world_size])
    full_shard = [r for r in rows if r["frame_key"] in full_shard_frames]
    benchmark_count = sum(r["request_id"] in benchmark for r in full_shard)
    projected = projected_compression_seconds(measured_s, preparation_s, len(selected),
        len(full_shard), benchmark_count, settings["timing_repeats"], fixed_s)
    return done, projected, technical


def steering_worker(args, config, engine, rows, dataset, out):
    from research.lifecycle_memory.application_pilot_steering import train_adapter, TrainingRow, LowRankSteering, TARGETS as TARGET_MAP
    settings = config["steering"]
    labels = {r["request_id"]:r["correct_label"] for r in read_rows(dataset / "labels/train.jsonl")}
    training = [TrainingRow(r["request_id"], r, int(labels[r["request_id"]] == "B"), "train") for r in rows if r["split"] == "train"]
    evaluations = steering_eval_rows(rows,args.mode,settings["canary_eval_per_split"])
    all_evaluations = steering_eval_rows(rows,"full",settings["canary_eval_per_split"])
    full_benchmark = {r["request_id"] for r in sorted(all_evaluations,key=lambda r:stable_order(r["request_id"]))[:config["compression"]["benchmark_requests"]]}
    technical = engine.identity_gate(engine.prepare(training[0].row,args.data_dir))
    atomic_json(out / "technical_gate.json", technical)
    tasks = [(target,seed) for seed in settings["seeds"] for target in settings["targets"]]
    done, projected = 0, 0.0
    for target, seed in tasks[args.rank::args.world_size]:
        taskdir = out / f"{target}_seed{seed}"
        taskdir.mkdir()
        zero = LowRankSteering(5376,settings["rank"]).to(engine.device)
        sample = engine.prepare(training[0].row,args.data_dir)
        with engine.torch.no_grad():
            native = engine.logits(sample)
            with engine.steering_forward(sample,zero,TARGET_MAP[target],False) as zero_logits:
                if not engine.torch.equal(native,zero_logits):
                    raise RuntimeError("zero steering adapter changes native answer logits")
        atomic_json(taskdir/"zero_adapter_gate.json", {"status":"pass","ab_logits_bit_exact":True})
        del zero, sample, native, zero_logits
        train_start = time.perf_counter()
        engine.torch.cuda.reset_peak_memory_stats()
        first_step = {}
        def progress(receipt):
            # Receipts hold synchronized loss/gradient floats, so this marks a
            # completed first step: one-time CUDA/allocator warmup stays there.
            if receipt.step == 1:
                first_step["s"] = time.perf_counter()-train_start
            atomic_json(taskdir / "progress.json", {"status":"training", **asdict(receipt)})
        result = train_adapter(engine.model, training, hidden_size=5376, target=target,
            prepare=lambda r:engine.prepare(r,args.data_dir), forward=engine.steering_forward,
            checkpoint_dir=taskdir, rank=settings["rank"], seed=seed,
            optimizer_steps=settings["canary_steps"] if args.mode == "canary" else settings["steps"],
            device=engine.device, early_report=progress, canary=args.mode == "canary")
        engine.torch.cuda.synchronize()
        train_s = time.perf_counter()-train_start
        train_peak = engine.torch.cuda.max_memory_allocated()
        projected_train_s = projected_training_seconds(train_s, first_step["s"],
                                                       result.optimizer_steps, settings["steps"])
        atomic_json(taskdir / "training_receipt.json", {"status":"pass", "target":target,"seed":seed,
            "steps":result.optimizer_steps,"accumulation_steps":result.accumulation_steps,
            "parameters":result.parameter_count,"train_s":train_s,"first_step_s":first_step["s"],
            "projected_full_train_s":projected_train_s,"peak_allocated_bytes":train_peak,
            "final_update_norm":result.final_update_norm,"checkpoint":str(result.checkpoint.name),
            "checkpoint_sha256":digest(result.checkpoint), "data_order_sha256":hashlib.sha256(json.dumps(result.data_order).encode()).hexdigest(),
            "step_receipts":[asdict(r) for r in result.receipts], "checkpoint_selection":"fixed final step; validation not used for tuning"})
        eval_start = time.perf_counter()
        eval_prediction_s, benchmark_s, benchmark_count = 0.0, 0.0, 0
        for row_index, row in enumerate(evaluations):
            prep_start = time.perf_counter()
            prepared = engine.prepare(row,args.data_dir)
            engine.torch.cuda.synchronize()
            eval_prediction_s += time.perf_counter()-prep_start
            def steered():
                with engine.steering_forward(prepared,result.adapter,result.target,False) as value:
                    return value
            variants = [(target,steered),("native",lambda:engine.logits(prepared))]
            if row_index == 0:
                for _, fn in variants:
                    for _ in range(config["compression"]["warmups"]):
                        with engine.torch.no_grad():
                            fn()
                engine.torch.cuda.synchronize()
            # Counterbalance target/native order, with each pair on the same GPU.
            random.Random(stable_order(f"{seed}:{row['request_id']}")).shuffle(variants)
            for variant,fn in variants:
                logits, metrics = measure(engine.torch,fn)
                eval_prediction_s += metrics["elapsed_ms"]/1000
                append_row(out / "predictions.jsonl", prediction(row,logits,experiment="steering",variant=variant,
                    seed=seed,rank=args.rank,metrics={**metrics,"comparison_target":target}))
                done += 1
                if args.mode == "canary" or row["request_id"] in full_benchmark:
                    repeats = config["compression"].get("canary_timing_repeats",2) if args.mode == "canary" else config["compression"]["timing_repeats"]
                    for repeat in range(repeats):
                        _, timing = measure(engine.torch,fn)
                        benchmark_s += timing["elapsed_ms"]/1000
                        benchmark_count += 1
                        append_row(out/"timings.jsonl", {"task_id":"steering", "variant":variant,
                            "seed":seed,"request_id":row["request_id"],"comparison_target":target,
                            "rank":args.rank,"repeat_id":repeat,"is_warmup":False,
                            "warmup_complete":True,**timing})
            del prepared
        eval_s = time.perf_counter()-eval_start
        full_eval = sum(r["split"] in ("validation","test") for r in rows)
        projected += (projected_train_s
                      + eval_prediction_s * full_eval / len(evaluations)
                      + benchmark_s/max(1,benchmark_count) * len(full_benchmark)*2*config["compression"]["timing_repeats"])
        atomic_json(taskdir / "progress.json", {"status":"complete", "eval_requests":len(evaluations),"train_s":train_s,"eval_s":eval_s})
        del result
        engine.torch.cuda.empty_cache()
    return done, projected, technical


def worker(args):
    config = read_json(args.config)
    validate_config(config)
    dataset = Path(args.dataset_dir or Path(args.out_dir)/"dataset")
    manifest = read_json(dataset / "split_manifest.json")
    for name in ("public/requests.jsonl", "labels/train.jsonl"):
        if digest(dataset/name) != manifest["exports"][name]["sha256"]:
            raise RuntimeError(f"dataset identity mismatch: {name}")
    rows = read_rows(dataset / "public/requests.jsonl")
    out = Path(args.out_dir) / f"rank_{args.rank:03d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out/"predictions.jsonl").exists():
        raise RuntimeError("refusing to mix predictions with a previous invocation")
    started = time.perf_counter()
    engine = Engine(config)
    load_s = time.perf_counter()-started
    work_start = time.perf_counter()
    if args.experiment == "compression":
        count, projection, technical = compression_worker(args,config,engine,rows,out)
    else:
        count, projection, technical = steering_worker(args,config,engine,rows,dataset,out)
    receipt = {"status":"pass", "experiment":args.experiment, "mode":args.mode,
        "rank":args.rank, "world_size":args.world_size, "prediction_rows":count,
        "config_sha256":digest(args.config), "split_manifest_sha256":digest(dataset/"split_manifest.json"),
        "load_s":load_s,"work_s":time.perf_counter()-work_start,"projected_full_work_s":projection,
        "max_host_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "process_cpu_seconds":resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        "technical_gate":{"status":technical["status"]}, "runtime":engine.runtime_receipt,
        "predictions_sha256":digest(out/"predictions.jsonl")}
    atomic_json(out/"worker_receipt.json", receipt)
    atomic_json(out/"progress.json", {"status":"complete","prediction_rows":count})


def aggregate(args):
    from research.lifecycle_memory.application_pilot_analysis import analyze_predictions, join_labels
    config = read_json(args.config)
    validate_config(config)
    out = Path(args.out_dir)
    dataset = Path(args.dataset_dir or out/"dataset")
    receipts, predictions = [], []
    for rank in range(args.world_size):
        rankdir = out / f"rank_{rank:03d}"
        receipt = read_json(rankdir/"worker_receipt.json")
        expected = {"status":"pass", "experiment":args.experiment,"mode":args.mode,"rank":rank,
            "world_size":args.world_size,"config_sha256":digest(args.config),
            "split_manifest_sha256":digest(dataset/"split_manifest.json")}
        if any(receipt.get(k) != v for k,v in expected.items()) or receipt["technical_gate"]["status"] != "pass":
            raise RuntimeError(f"worker {rank} receipt contract failed")
        if digest(rankdir/"predictions.jsonl") != receipt["predictions_sha256"]:
            raise RuntimeError("worker predictions changed after receipt")
        rankrows = read_rows(rankdir/"predictions.jsonl")
        if len(rankrows) != receipt["prediction_rows"]:
            raise RuntimeError("prediction count mismatch")
        predictions.extend(rankrows)
        receipts.append(receipt)
    inventory = validate_prediction_inventory(predictions,read_rows(dataset/"public/requests.jsonl"),
                                              args.experiment,args.mode,config)
    # The final table is a new application analysis, not a preregistered claim.
    labels = read_rows(dataset/"evaluator/all_labels.jsonl")
    if digest(dataset/"evaluator/all_labels.jsonl") != read_json(dataset/"split_manifest.json")["exports"]["evaluator/all_labels.jsonl"]["sha256"]:
        raise RuntimeError("evaluator label file changed")
    predictions = join_labels(predictions,labels)
    timings = []
    for rank in range(args.world_size):
        path = out/f"rank_{rank:03d}"/"timings.jsonl"
        if path.exists():
            timings.extend(read_rows(path))
    expected_rows = read_rows(dataset/"public/requests.jsonl")
    expected_rows = (selected_rows(expected_rows,args.mode,config["compression"]["canary_frames"])
                     if args.experiment == "compression" else
                     steering_eval_rows(expected_rows,args.mode,config["steering"]["canary_eval_per_split"]))
    summary = analyze_predictions(predictions,
        bootstrap_draws=config["bootstrap"]["draws"], bootstrap_seed=config["bootstrap"]["seed"],
        expected_rows=expected_rows, timing_rows=timings,
        require_role_by_depth=args.experiment == "compression",
        required_variants={args.experiment:list(VARIANTS) if args.experiment=="compression" else ["native",*TARGETS]},
        required_seeds={args.experiment:[0] if args.experiment=="compression" else config["steering"]["seeds"]},
        required_splits={args.experiment:["all"] if args.experiment=="compression" else ["validation","test"]},
        accuracy_loss_tolerance=config["compression"]["accuracy_loss_tolerance_pp"]/100 if args.experiment == "compression" else None)
    atomic_json(out/"analysis_summary.json", summary)
    atomic_json(out/"efficiency_summary.json", efficiency_summary(predictions))
    lines = [f"# {args.experiment.title()} Pilot ({args.mode})", "",
             "All planned prediction cells passed the completeness check. Intervals are descriptive, not new confirmation tests.", ""]
    def interval(estimate, scale, digits):
        ci = estimate.get("ci95")
        return f"[{scale*ci[0]:.{digits}f}, {scale*ci[1]:.{digits}f}]" if ci else "unavailable"

    def table_row(label, comparison):
        acc = comparison["participant_equal_difference_variant_minus_reference"]["accuracy"]
        fidelity = comparison["margin_fidelity"]["mean_absolute_margin_change_bits"]
        speed = comparison["timing"].get("speedup_reference_over_variant")
        fidelity_text = (f"{fidelity['estimate']:.3f} {interval(fidelity, 1, 3)}"
                         if fidelity["estimate"] is not None else "unavailable")
        speed_text = f"{speed:.3f}x" if speed is not None else "unavailable"
        acc_text = f"{100*acc['estimate']:.2f}" if acc["estimate"] is not None else "unavailable"
        return f"| {label} | {acc_text} | {interval(acc, 100, 2)} | {fidelity_text} | {speed_text} |"

    roles = {"primary": "primary", "primary_pooled_heldout": "primary, pooled held-out",
             "secondary_per_split": "secondary, per split"}
    for report in summary["reports"]:
        lines += [f"## {report['split_scope']} ({roles[report['report_role']]})", "",
                  "| Comparison | Accuracy difference (percentage points) | Descriptive 95% interval | Mean abs margin change, bits [95% interval] | Model-forward speedup |",
                  "|---|---:|---:|---:|---:|"]
        for comparison in report["comparisons"]:
            lines.append(table_row(f"{comparison['variant']} vs {comparison['reference']}", comparison))
        factorial = report.get("role_by_depth")
        if factorial and factorial["available"]:
            lines += ["", "### Paired Role-by-Depth Contrast", "",
                      f"{factorial['formula']}, using only the four 133-row merge arms (no expansion/pruning).",
                      "Positive accuracy/correct-margin values favor F more at L34; negative absolute margin-drift values favor F's relative fidelity.",
                      "Participant-equal paired descriptive bootstrap, not a confirmatory interaction test.", "",
                      "| Metric | Difference of differences | Descriptive 95% interval |",
                      "|---|---:|---:|"]
            for metric, scale in (("accuracy", 100), ("correct_margin", 1), ("mean_absolute_margin_change_bits", 1)):
                value = factorial["participant_equal_difference_of_differences"][metric]
                point = f"{scale*value['estimate']:.3f}" if value["estimate"] is not None else "unavailable"
                label = "accuracy (pp)" if metric == "accuracy" else f"{metric} (bits)"
                lines.append(f"| {label} | {point} | {interval(value, scale, 3)} |")
            timing = factorial["timing"]
            if timing["available"]:
                for metric, value in timing["participant_equal_difference_of_differences"].items():
                    lines.append(f"| {metric} | {value['estimate']:.3f} | {interval(value, 1, 3)} |")
                lines += ["", f"Timing covers {timing['timed_unique_requests']} requests / {timing['timed_participant_count']} participants.",
                          "Negative latency contrast and positive log-speedup contrast favor F's relative speed at L34.",
                          "Log speedup uses natural log(native/arm) per paired timing; its contrast is log((C34/F34)/(C22/F22))."]
            else:
                lines += ["", f"Paired role-by-depth timing unavailable: {timing['reason']}."]
        elif factorial:
            lines += ["", f"Role-by-depth contrast unavailable: {factorial['reason']}."]
        bias = report.get("ab_bias_baseline")
        if bias and bias["available"]:
            lines += ["", f"One-parameter A/B bias baseline on frozen native margins ({bias['fit_policy']}):", "",
                      "| Comparison | Accuracy difference (percentage points) | Descriptive 95% interval | Mean abs margin change, bits [95% interval] | Model-forward speedup |",
                      "|---|---:|---:|---:|---:|"]
            first = next(iter(bias["by_target"].values()))
            lines.append(table_row("native + A/B bias vs native", first["bias_minus_native"]))
            for target, value in bias["by_target"].items():
                lines.append(table_row(f"{target} vs native + A/B bias", value["target_minus_bias"]))
        elif bias:
            lines += ["", f"A/B bias baseline unavailable: {bias['reason']}."]
        lines.append("")
    lines += ["See analysis_summary.json for per-seed results and paired wrong-to-right/right-to-wrong changes,",
              "and efficiency_summary.json for memory and compute estimates. For steering, validation+test is",
              "the declared primary report because validation was never used for training, tuning or selection.",
              "A gain over native alone is task adaptation; target-selection evidence requires the matched controls."]
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n")
    estimate = max(r["load_s"]+r["projected_full_work_s"] for r in receipts)
    seconds = max(3600, math.ceil((1.4*estimate+600)/300)*300)
    atomic_json(out/"profile.json", {"status":"pass", "experiment":args.experiment,"mode":args.mode,
        "config_sha256":digest(args.config), "split_manifest_sha256":digest(dataset/"split_manifest.json"),
        "recommended_walltime_s":seconds, "runtime_budget_status":"pass" if seconds <= 10800 else "needs_review",
        "unbuffered_projected_worker_s":estimate,"workers":receipts,
        "prediction_inventory":inventory,
        "technical_gate":{"status":"pass"},
        "quality_gate":"technical only; negative scientific results retained",
        "analysis_summary_sha256":digest(out/"analysis_summary.json")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker","aggregate"))
    parser.add_argument("--experiment", choices=("compression","steering"), required=True)
    parser.add_argument("--mode", choices=("canary","full"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args(argv)
    if args.world_size != 4 or args.rank not in range(4):
        parser.error("the frozen pilot uses exactly four GPU workers")
    try:
        (worker if args.command == "worker" else aggregate)(args)
    except BaseException as error:
        destination = args.out_dir / (f"rank_{args.rank:03d}" if args.command == "worker" else "aggregate")
        atomic_json(destination/"failure.json", {"status":"fail", "stage":args.command,
            "rank":args.rank,"error":str(error),"traceback":traceback.format_exc()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
