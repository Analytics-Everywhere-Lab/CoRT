"""Role-aware policy experiment: train, freeze, evaluate, aggregate.

Commands (``python -m research.lifecycle_memory.run_role_aware_policy ...``):

  train      six (case, seed) tasks, once each, strided over four GPU workers.
  freeze     CPU-only: verify receipts, write a separate mode-bound config and
             ten adapters (six trained, four frozen native job-620395 controls).
  evaluate   one frame shard per rank on the cross-source set; never opens labels.
  aggregate  join evaluator labels, compute the prespecified summary and report.

Model execution is guarded by the existing Engine (one Slurm-bound CUDA device).
All torch/transformers imports are lazy; CPU tests inject fakes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import struct
import time
import traceback

from research.lifecycle_memory import role_aware_policy as rap

WORLD_SIZE = 4
CACHED_VALIDATION_CONFIG = "docs/research/lifecycle_memory/phase 2 research/jobs/decoder_cached_validation/config.json"
REFERENCE_ADAPTER_DIR = "docs/research/lifecycle_memory/phase 2 research/jobs/decoder_cached_validation/artifacts/adapters"
TRAIN_MANIFEST = "a5e21649d4f18fc7af0c6d44b71432e6a6c05b9c73eb3e90603e0a1ce438d904"


def _io():
    from research.lifecycle_memory import run_decoder_application_pilots as pilot
    return pilot


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def repo_root():
    return Path(__file__).resolve().parents[2]


def code_manifest(repo=None):
    root = Path(repo or repo_root())
    return {str(p.relative_to(root)): sha256(p)
            for p in sorted((root / "research/lifecycle_memory").glob("*.py"))}


def design_digest(config):
    source = {k: v for k, v in config.items() if k not in ("adapters", "freeze", "status")}
    return hashlib.sha256(json.dumps(source, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_frozen(config, mode, *, repo=None):
    rap.require(mode in ("canary", "full") and rap.adapters_complete(config), "incomplete frozen config")
    binding = config.get("freeze", {})
    rap.require(binding.get("mode") == mode, "adapter train/evaluation mode mismatch; no canary promotion")
    rap.require(binding.get("design_sha256") == design_digest(config), "frozen design drift")
    rap.require(binding.get("code") == code_manifest(repo), "frozen code drift")
    root = Path(repo or repo_root())
    rap.require(binding.get("reference_config_sha256") == sha256(root / CACHED_VALIDATION_CONFIG),
                "frozen reference/data contract drift")
    reference = read_json(root / CACHED_VALIDATION_CONFIG)
    rap.require(config["model"] == reference["model"] and binding.get("evaluation_manifest_sha256") ==
                reference["fresh_data"]["data_manifest_sha256"], "model/evaluation data drift")
    rap.require(binding.get("training_manifest_sha256") == TRAIN_MANIFEST, "training manifest drift")
    rap.require(binding.get("adapters") == config["adapters"], "frozen adapter inventory drift")
    for adapter in config["adapters"]:
        if adapter["case_id"] in rap.REFERENCE_CASES:
            matches = [r for r in reference["adapters"] if (r["case_id"], r["seed"]) ==
                       (adapter["case_id"], adapter["seed"])]
            rap.require(len(matches) == 1 and all(adapter.get(k) == v for k, v in matches[0].items()),
                        "reference adapter provenance drift")
        else:
            r = adapter.get("training_receipt", {})
            fields = {"id": adapter["case_id"], "seed": adapter["seed"], "target": adapter["target"],
                      "mode": mode, "status": "pass", "checkpoint_sha256": adapter["sha256"],
                      "steps": rap.TRAINING["canary_steps" if mode == "canary" else "steps"],
                      "code": binding["code"], "design_sha256": binding["design_sha256"],
                      "source_config_sha256": binding["source_config_sha256"],
                      "split_manifest_sha256": TRAIN_MANIFEST, "training_pool_size": 1010,
                      "training_frames": 505, "training_clusters": 88, "gradient_accumulation": 4,
                      "rank": 4, "parameters": 43008, "checkpoint_selection": "fixed_final_step"}
            rap.require(adapter.get("training_mode") == mode and
                        rap.is_digest(adapter.get("training_receipt_sha256")) and
                        all(r.get(k) == v for k, v in fields.items()), "trained adapter provenance drift")
    return binding


# --------------------------------------------------------------------------- training

def training_tasks(rank=None):
    tasks = [(case, seed) for seed in rap.SEEDS for case in rap.TRAIN_CASES]
    rap.require(len(tasks) == 6 and len(set(tasks)) == 6, "training inventory drift")
    return tasks if rank is None else tasks[rank::WORLD_SIZE]


def train_worker(args, *, engine_factory=None):
    """Load once per worker; process each assigned case/seed exactly once."""
    rap.require(args.world_size == WORLD_SIZE and 0 <= args.rank < WORLD_SIZE, "four training workers required")
    resident = []

    def engine(config):
        if not resident:
            resident.append((engine_factory or _matrix_engine)(config))
        return resident[0]

    receipts = [_train_one(args, task, engine_factory=engine) for task in training_tasks(args.rank)]
    result = {"status": "pass", "mode": args.mode, "rank": args.rank, "tasks": receipts}
    _io().atomic_json(Path(args.out_dir) / f"rank_{args.rank:03d}" / "training_worker_receipt.json", result)
    return result


def _train_one(args, task, *, engine_factory):
    from research.lifecycle_memory.application_pilot_steering import LowRankSteering, train_adapter
    from research.lifecycle_memory.decoder_token_matrix import matrix_context
    from research.lifecycle_memory.decoder_token_matrix_steering import (
        _actual_checkpoint, _public_only, _schedule, correction_forward, training_rows,
    )
    pilot = _io()
    config = rap.validate_config(read_json(args.config))
    rap.require(args.mode in ("canary", "full") and args.world_size == WORLD_SIZE
                and 0 <= args.rank < WORLD_SIZE, "train requires four ranks and canary/full")
    dataset = Path(args.dataset_dir)
    rap.require(sha256(dataset / "split_manifest.json") == TRAIN_MANIFEST, "not the fixed 620395 training split")
    manifest = read_json(dataset / "split_manifest.json")
    for name in ("public/requests.jsonl", "labels/train.jsonl"):
        rap.require(pilot.digest(dataset / name) == manifest["exports"][name]["sha256"],
                    f"dataset identity mismatch: {name}")
    rows = pilot.read_rows(dataset / "public/requests.jsonl")
    _public_only(rows)
    training = training_rows(rows, dataset)
    rap.require(len(training) == 1010 and len({r.row["frame_key"] for r in training}) == 505
                and len({r.row["cluster_key"] for r in training}) == 88, "training pool inventory drift")
    case_id, seed = task
    compression, target = rap.case_parts(case_id)
    case = {"id": case_id, "compression_id": compression, "target": target}
    out = Path(args.out_dir) / f"rank_{args.rank:03d}" / f"{case_id}_seed{seed}"
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    engine = engine_factory(config)
    load_s = time.perf_counter() - started
    hidden = getattr(engine.model.config, "text_config", engine.model.config).hidden_size
    sample = engine.prepare(training[0].row, args.data_dir)
    plan = _schedule(engine, sample, compression)
    zero = LowRankSteering(hidden, rap.TRAINING["rank"]).to(engine.device)
    with engine.torch.no_grad():
        with matrix_context(engine.layers, plan):
            baseline = engine.logits(sample)
        with correction_forward(engine, sample, zero, case, schedule=plan) as identity:
            rap.require(bool(engine.torch.equal(baseline, identity)), "zero correction differs from compression-only")
    pilot.atomic_json(out / "zero_adapter_gate.json", {"status": "pass", **case, "ab_logits_bit_exact": True})
    del sample, plan, zero, baseline, identity
    steps = rap.TRAINING["canary_steps"] if args.mode == "canary" else rap.TRAINING["steps"]
    train_start = time.perf_counter()
    result = train_adapter(
        engine.model, training, hidden_size=hidden, target="l34_readout" if target == "readout" else "l34_cq",
        prepare=lambda row: engine.prepare(row, args.data_dir),
        forward=lambda prepared, adapter, _target, is_training: correction_forward(
            engine, prepared, adapter, case, is_training),
        checkpoint_dir=out, rank=rap.TRAINING["rank"], seed=seed, device=engine.device,
        optimizer_steps=steps, canary=args.mode == "canary",
        early_report=lambda receipt: pilot.atomic_json(out / "progress.json",
                                                       {"status": "training", **asdict(receipt)}))
    engine.torch.cuda.synchronize()
    train_s = time.perf_counter() - train_start
    checkpoint = _actual_checkpoint(result, case)
    receipt = {"status": "pass", "experiment_id": rap.EXPERIMENT_ID, "mode": args.mode, **case,
               "seed": seed, "cut": rap.CUT, "rank": rap.TRAINING["rank"],
               "parameters": result.parameter_count, "steps": result.optimizer_steps,
               "checkpoint_selection": "fixed_final_step", "training_pool_size": len(training),
               "data_order_sha256": hashlib.sha256(json.dumps(result.data_order).encode()).hexdigest(),
               "checkpoint": checkpoint.name, "checkpoint_sha256": pilot.digest(checkpoint),
               "load_s": load_s, "train_s": train_s, "final_update_norm": result.final_update_norm,
               "finite_gradients": all(r.finite_gradients for r in result.receipts),
               "backward_nonzero": any(r.nonzero_gradient for r in result.receipts),
               "nonzero_update": any(r.nonzero_update for r in result.receipts),
               "frozen_base_checked": all(r.frozen_base_checked for r in result.receipts),
               "training_frames": 505, "training_clusters": 88,
               "gradient_accumulation": 4, "learning_rate": .001, "worker_rank": args.rank,
               "design_sha256": design_digest(config), "code": code_manifest(),
               "source_config_sha256": sha256(args.config),
               "split_manifest_sha256": pilot.digest(dataset / "split_manifest.json")}
    rap.require(all(receipt[k] for k in ("finite_gradients", "backward_nonzero", "nonzero_update", "frozen_base_checked"))
                and math.isfinite(result.final_update_norm) and result.final_update_norm > 0, "training gate failed")
    pilot.atomic_json(out / "training_receipt.json", receipt)
    pilot.atomic_json(out / "progress.json", {"status": "complete"})
    return receipt


def _matrix_engine(config):
    from research.lifecycle_memory.run_decoder_token_matrix import MatrixEngine
    return MatrixEngine(config)


# --------------------------------------------------------------------------- freeze

def training_technical(config_path, train_run, mode, *, repo=None):
    """Read-only receipt validation, usable without torch/model imports."""
    repo = Path(repo or repo_root())
    config = rap.validate_config(read_json(config_path))
    rap.require(mode in ("canary", "full"), "invalid training mode")
    steps = rap.TRAINING["canary_steps" if mode == "canary" else "steps"]
    paths = list(Path(train_run).glob("rank_*/*/training_receipt.json"))
    expected = {Path(train_run) / f"rank_{i % WORLD_SIZE:03d}/{case}_seed{seed}/training_receipt.json": (case, seed)
                for i, (case, seed) in enumerate(training_tasks())}
    rap.require(len(paths) == len(expected) and set(paths) == set(expected), "training receipt inventory mismatch")
    rap.require(not list(Path(train_run).rglob("error.json")), "training error artifact present")
    checked = []
    codes = code_manifest(repo)
    for path, (case_id, seed) in expected.items():
        receipt = read_json(path)
        compression, target = rap.case_parts(case_id)
        rank = training_tasks().index((case_id, seed)) % WORLD_SIZE
        fields = {"status": "pass", "mode": mode, "experiment_id": rap.EXPERIMENT_ID,
                  "id": case_id, "compression_id": compression, "target": target, "seed": seed,
                  "cut": 34, "rank": 4, "worker_rank": rank, "parameters": 43008,
                  "steps": steps, "checkpoint_selection": "fixed_final_step", "training_pool_size": 1010,
                  "training_frames": 505, "training_clusters": 88, "gradient_accumulation": 4,
                  "learning_rate": .001, "split_manifest_sha256": TRAIN_MANIFEST,
                  "design_sha256": design_digest(config), "source_config_sha256": sha256(config_path), "code": codes}
        rap.require(all(receipt.get(k) == v for k, v in fields.items()), "training receipt provenance mismatch")
        rap.require(all(receipt.get(k) is True for k in
                    ("finite_gradients", "backward_nonzero", "nonzero_update", "frozen_base_checked"))
                    and math.isfinite(receipt["final_update_norm"]) and receipt["final_update_norm"] > 0,
                    "training technical gate failed")
        rap.require(rap.is_digest(receipt.get("data_order_sha256")), "missing training order hash")
        name = receipt["checkpoint"]
        rap.require(Path(name).name == name and name.endswith(".safetensors"), "invalid checkpoint path/format")
        source = path.parent / name
        rap.require(sha256(source) == receipt["checkpoint_sha256"], "adapter hash mismatch")
        gate = read_json(path.parent / "zero_adapter_gate.json")
        rap.require(gate.get("status") == "pass" and gate.get("ab_logits_bit_exact") is True
                    and gate.get("id") == case_id, "training zero gate mismatch")
        checked.append((path, receipt, source))
    for seed in rap.SEEDS:
        rap.require(len({r["data_order_sha256"] for _, r, _ in checked if r["seed"] == seed}) == 1,
                    "case training orders differ for matched seed")
    return checked


def freeze(config_path, train_run, *, mode="full", output_config=None, adapters_dir=None, repo=None):
    """Write a new mode-bound snapshot; never modify source config or old outputs."""
    repo = Path(repo or repo_root())
    config_path = Path(config_path)
    config = rap.validate_config(read_json(config_path))
    rap.require(not config.get("adapters") and not config.get("freeze"), "source config must be unfrozen")
    rap.require(output_config is not None and adapters_dir is not None, "freeze needs output-config and adapters-dir")
    output_config, dest = Path(output_config), Path(adapters_dir)
    rap.require(output_config.resolve() != config_path.resolve() and not output_config.exists(),
                "refusing to replace source or frozen config")
    rap.require(not dest.exists() or not any(dest.iterdir()), "adapter destination must be empty")
    checked = training_technical(config_path, train_run, mode, repo=repo)
    adapters, copies = [], []
    for path, receipt, source in checked:
        adapters.append({"case_id": receipt["id"], "seed": receipt["seed"], "target": receipt["target"],
                         "path": source.name, "sha256": sha256(source), "cut": rap.CUT, "rank": 4,
                         "parameters": receipt["parameters"], "training_mode": mode,
                         "training_receipt_sha256": sha256(path), "training_receipt": receipt})
        copies.append(source)
    reference = read_json(repo / CACHED_VALIDATION_CONFIG)
    rap.require(reference.get("source_run", {}).get("id") == "620395", "incorrect reference training run")
    refs = [r for r in reference["adapters"] if r["case_id"] in rap.REFERENCE_CASES]
    rap.require(len(refs) == 4 and {(r["case_id"], r["seed"]) for r in refs} ==
                {(c, s) for c in rap.REFERENCE_CASES for s in rap.SEEDS}, "reference inventory mismatch")
    for record in refs:
        source = repo / REFERENCE_ADAPTER_DIR / record["path"]
        rap.require(Path(record["path"]).name == record["path"], "unsafe reference path")
        rap.require(sha256(source) == record["sha256"], "reference adapter hash mismatch")
        adapters.append({**record, "source": "job 620395 (frozen)", "training_mode": "full_reference"})
        copies.append(source)
    config["adapters"] = adapters
    config["status"] = "adapters_frozen"
    config["freeze"] = {"mode": mode, "design_sha256": design_digest(config), "code": code_manifest(repo),
                        "source_config_sha256": sha256(config_path), "training_manifest_sha256": TRAIN_MANIFEST,
                        "reference_config_sha256": sha256(repo / CACHED_VALIDATION_CONFIG), "adapters": adapters,
                        "evaluation_manifest_sha256": reference["fresh_data"]["data_manifest_sha256"]}
    rap.validate_config(config)
    validate_frozen(config, mode, repo=repo)
    dest.mkdir(parents=True, exist_ok=True)
    for source in copies:
        shutil.copyfile(source, dest / source.name)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("x") as handle:
        json.dump(config, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {"status": "pass", "mode": mode, "adapters": len(adapters), "output_config": str(output_config),
            "config_sha256": sha256(output_config)}


# --------------------------------------------------------------------------- evaluation

def load_adapters(engine, config, adapters_dir):
    """Verify hashes/shapes, load FP32 adapters on the device, plus a zero adapter."""
    from safetensors.torch import load_file
    from research.lifecycle_memory.application_pilot_steering import LowRankSteering
    hidden = getattr(engine.model.config, "text_config", engine.model.config).hidden_size
    resident, receipts = {}, []
    for record in config["adapters"]:
        path = Path(adapters_dir) / record["path"]
        rap.require(sha256(path) == record["sha256"], f"adapter checksum mismatch: {record['path']}")
        state = load_file(str(path), device="cpu")
        rap.require(set(state) == {"down.weight", "up.weight"}
                    and list(state["down.weight"].shape) == [4, hidden]
                    and list(state["up.weight"].shape) == [hidden, 4]
                    and all(v.dtype == engine.torch.float32 and engine.torch.isfinite(v).all()
                            for v in state.values()), "adapter shape/dtype/finite check failed")
        module = LowRankSteering(hidden, rank=4)
        module.load_state_dict(state, strict=True)
        module.eval().requires_grad_(False).to(device=engine.device)
        resident[(record["case_id"], record["seed"])] = module
        receipts.append({"case_id": record["case_id"], "seed": record["seed"], "sha256": record["sha256"]})
    zero = LowRankSteering(hidden, rank=4).eval().requires_grad_(False).to(device=engine.device)
    return resident, zero, receipts


def make_forward(spec, adapters, zero, index_cache, runtime):
    """Return forward(engine, prepared, plan) for one frozen variant."""
    from research.lifecycle_memory.decoder_token_matrix_steering import correction_forward, target_positions
    path = spec["path"]
    if path == "native":
        return lambda engine, prepared, plan: engine.logits(prepared)
    if path == "legacy":
        adapter = adapters[(spec["case_id"], spec["seed"])]
        case = {"id": spec["case_id"], "compression_id": spec["compression_id"], "target": spec["target"]}

        def legacy(engine, prepared, plan):
            with correction_forward(engine, prepared, adapter, case, training=False, schedule=plan) as logits:
                return logits
        return legacy
    adapter = None if path == "matrix" else zero if path == "fused_zero" else adapters[(spec["case_id"], spec["seed"])]

    def fused(engine, prepared, plan):
        runtime.prepare(plan)
        index = None if adapter is None else index_cache.get(plan, spec["target"],
                                lambda: target_positions(prepared.pilot_partition, spec["target"]), engine.device)
        with runtime.activate(plan, adapter, index):
            return engine.logits(prepared)
    return fused


def readout(logits, prepared, plan):
    values = logits.detach().cpu().reshape(-1).tolist()
    rap.require(len(values) == 2 and all(math.isfinite(v) for v in values), "expected two finite A/B logits")
    return {"prediction": "A" if values[0] >= values[1] else "B", "logits_ab": values,
            "margin_ab": (values[0] - values[1]) / math.log(2)}


def logits_digest(result):
    return hashlib.sha256(struct.pack(">dd", *result["logits_ab"])).hexdigest()


def variant_order(request_id, *, phase, repeat=0):
    ids = sorted(spec["id"] for spec in rap.variant_specs())
    seed = hashlib.sha256(f"{rap.EXPERIMENT_ID}:{phase}:{repeat}:{request_id}".encode()).digest()
    random.Random(int.from_bytes(seed, "big")).shuffle(ids)
    return ids


@dataclass
class DataContract:
    """Thin adapter over the frozen cross-source data contract (job 621156)."""
    config: dict

    def module(self):
        from research.lifecycle_memory import decoder_cached_validation
        return decoder_cached_validation

    def check(self, data_dir, *, public_only):
        return self.module().check_dataset(data_dir, self.config, verify_assets=public_only,
                                           public_only=public_only)

    def rows(self, data_dir, mode):
        return self.module().public_rows(data_dir, mode, self.config)

    def shards(self, rows):
        return self.module().frame_shards(rows)

    def benchmark(self, rows, mode):
        return set(self.module().benchmark_ids(rows, mode, self.config))


def _contract():
    return DataContract(read_json(repo_root() / CACHED_VALIDATION_CONFIG))


def evaluate_worker(args, *, contract=None, engine_factory=None, measure=None, synchronize=None,
                    arms=None, checksum_fn=None, receipt_fn=None, clock=time.perf_counter):
    from research.lifecycle_memory.decoder_cached_benchmark import CachedScheduleCache, benchmark_measurement
    pilot = _io()
    out = Path(args.out_dir) / f"rank_{args.rank:03d}"
    out.mkdir(parents=True, exist_ok=True)
    rap.require(not any(out.iterdir()), "refusing to mix worker outputs with an earlier invocation")
    try:
        config = rap.validate_config(read_json(args.config))
        validate_frozen(config, args.mode)
        rap.require(args.mode in ("canary", "full") and args.world_size == WORLD_SIZE
                    and 0 <= args.rank < WORLD_SIZE, "evaluate requires four ranks and canary/full")
        contract = contract or _contract()
        dataset_receipt = contract.check(args.data_dir, public_only=True)
        rows = contract.rows(args.data_dir, args.mode)
        rap.require(len({r["request_id"] for r in rows}) == len(rows), "duplicate public requests")
        shards = contract.shards(rows)
        bench = contract.benchmark(rows, args.mode)
        selected = [r for r in rows if shards[r["request_id"]] == args.rank]
        rap.require(selected, "empty shard")
        specs = rap.variant_specs()
        by_id = {s["id"]: s for s in specs}
        started = clock()
        engine = (engine_factory or _matrix_engine)(config)
        adapters, zero, adapter_receipts = load_adapters(engine, config, args.adapters_dir)
        sync = synchronize or (lambda: engine.torch.cuda.synchronize())
        measure = measure or pilot.measure
        if arms is None:
            from research.lifecycle_memory.decoder_token_matrix import build_arms
            arms = build_arms()
        needed = {s["compression_id"] for s in specs}
        arm_by_id = {a["id"]: a for a in arms if a["id"] in needed}
        rap.require(set(arm_by_id) == needed, "unknown compression arm")
        index_cache = rap.IndexCache()
        runtime = rap.OptimizedDecoder(engine.model, engine.layers, engine.device)
        forwards = {s["id"]: make_forward(s, adapters, zero, index_cache, runtime) for s in specs}
        repeats = rap.MEASUREMENT["canary_timing_repeats" if args.mode == "canary" else "timing_repeats"]
        kwargs = {"max_geometries": 64, "clock": clock}
        if checksum_fn is not None:
            kwargs.update(checksum_fn=checksum_fn, receipt_fn=receipt_fn)
        cache = CachedScheduleCache(list(arm_by_id.values()), **kwargs)

        def request(row, spec, phase):
            return benchmark_measurement(engine, row, args.data_dir, arm_by_id[spec["compression_id"]], cache,
                                         phase=phase, forward=forwards[spec["id"]], postprocess=readout,
                                         measure=measure, clock=clock, synchronize=sync)

        gates, cold, keys, representatives, original_native = [], {}, {}, {}, {}
        # All public requests (32 in canary, 894 full), not one geometry proxy.
        # Retain only logits digests; all of this is outside benchmark timing.
        with engine.torch.no_grad():
            for row in selected:
                prepared = engine.prepare(row, args.data_dir)
                key, geometry = engine.geometry(prepared)
                original_native[row["request_id"]] = logits_digest(readout(engine.logits(prepared), prepared, None))
                if key not in representatives:
                    gate = engine.matrix_identity_gate(prepared, geometry)
                    rap.require(gate.get("status") == "pass" and gate.get("keep_all_ab_logits_bit_exact") is True,
                                "keep-all identity gate failed")
                    representatives[key] = row
                del prepared
                keys[row["request_id"]] = key
        with cache, runtime:
            for key, row in representatives.items():
                digests = {}
                for variant in variant_order(row["request_id"], phase="warmup"):
                    first = None
                    for repeat in range(rap.MEASUREMENT["warmups"]):
                        result, metrics = request(row, by_id[variant], "warmup")
                        digest = logits_digest(result)
                        first = first or digest
                        rap.require(digest == first, f"warmup logits unstable: {variant}")
                    digests[variant] = first
                    cold[(key, variant)] = first
                for left, right in rap.identity_pairs():
                    rap.require(digests[left] == digests[right], f"bit-exact identity gate failed: {left} vs {right}")
            cache.seal()
            index_cache.seal()
            runtime.seal()
            for (key, variant), digest in cold.items():
                result, _ = request(representatives[key], by_id[variant], "timed")
                rap.require(logits_digest(result) == digest, "sealed cached logits differ from warmup")
            setup_s = clock() - started
            (out / "predictions.jsonl").touch()
            (out / "timings.jsonl").touch()
            counts = Counter()
            # Complete EVERY request's full-logit gates before timing any arm.
            for row in selected:
                digests = {}
                for phase, reps in (("prediction", 1),):
                    for repeat in range(reps):
                        for variant in variant_order(row["request_id"], phase=phase, repeat=repeat):
                            spec = by_id[variant]
                            result, metrics = request(row, spec, "timed")
                            rap.require(metrics["plan_cache_hit"] and metrics["geometry_sha256"] == keys[row["request_id"]],
                                        "timed request missed the plan cache")
                            record = {**{k: row[k] for k in ("request_id", "frame_key", "cluster_key", "order_id")},
                                      **{k: row[k] for k in ("source_family", "sampling_unit_kind") if k in row},
                                      "variant": variant, "compression_id": spec["compression_id"],
                                      "path": spec["path"], "seed": spec.get("seed", 0), "rank": args.rank,
                                      "case_id": spec.get("case_id"), "target": spec.get("target"),
                                      "geometry_sha256": metrics["geometry_sha256"],
                                      "schedule_sha256": metrics["schedule_sha256"], "plan_cache_hit": metrics["plan_cache_hit"],
                                      **{k: metrics[k] for k in ("preparation_ms", "plan_lookup_ms",
                                         "model_forward_ms", "postprocess_ms", "end_to_end_elapsed_ms")},
                                      "peak_allocated_bytes": metrics["forward_metrics"].get("peak_allocated_bytes")}
                            if phase == "prediction":
                                pilot.append_row(out / "predictions.jsonl", {**record, **result})
                                digests[variant] = logits_digest(result)
                                counts["predictions"] += 1
                rap.require(digests["native"] == original_native[row["request_id"]], "original/session native logits differ")
                for left, right in rap.identity_pairs():
                    rap.require(digests[left] == digests[right], f"bit-exact request gate failed: {left} vs {right}")
                # Compression-only optimized vs existing matrix path as well.
                prepared = engine.prepare(row, args.data_dir)
                with engine.torch.no_grad():
                    for compression in (rap.PRIMARY, rap.SECONDARY):
                        plan = cache.get(engine, prepared, arm_by_id[compression]).plan
                        baseline = readout(engine.matrix_forward(prepared, plan)[0], prepared, plan)
                        rap.require(logits_digest(baseline) == digests[compression], "optimized pruning differs from matrix")
                gates.append({"geometry_sha256": keys[row["request_id"]], "request_id": row["request_id"],
                              "identity_pairs": [list(p) for p in rap.identity_pairs()], "bit_exact": True,
                              "original_native_bit_exact": True, "compression_bit_exact": True})
            pilot.atomic_json(out / "cold_start.json", {"status": "pass", "identity_gates": gates,
                              "setup_s": setup_s, "index_cache": index_cache.report(), "optimized": runtime.report(),
                              "equivalence_scope": "all requests; complete 32-request canary before timing",
                              "adapters": adapter_receipts})
            # Timing repeats use one shared request/repeat block across all variants.
            for row in selected:
                if row["request_id"] not in bench:
                    continue
                for repeat in range(repeats):
                    for order, variant in enumerate(variant_order(row["request_id"], phase="benchmark", repeat=repeat)):
                        spec = by_id[variant]
                        _, metrics = request(row, spec, "timed")
                        record = {**{k: row[k] for k in ("request_id", "frame_key", "cluster_key", "order_id", "source_family")},
                                  "sampling_unit_kind": row.get("sampling_unit_kind"), "variant": variant,
                                  "compression_id": spec["compression_id"], "path": spec["path"],
                                  "case_id": spec.get("case_id"), "target": spec.get("target"), "seed": spec.get("seed", 0),
                                  "rank": args.rank, "repeat_id": repeat, "variant_order": order,
                                  **{k: metrics[k] for k in ("preparation_ms", "plan_lookup_ms", "model_forward_ms",
                                     "postprocess_ms", "end_to_end_elapsed_ms", "plan_cache_hit", "geometry_sha256", "schedule_sha256")},
                                  "peak_allocated_bytes": metrics["forward_metrics"].get("peak_allocated_bytes"),
                                  "peak_reserved_bytes": metrics["forward_metrics"].get("peak_reserved_bytes"),
                                  **{k: metrics["forward_metrics"].get(k) for k in
                                     ("device_id", "batch_size", "synchronized")}}
                        pilot.append_row(out / "timings.jsonl", record)
                        counts["timings"] += 1
                pilot.atomic_json(out / "progress.json", {"status": "running", **counts})
            cache.verify_integrity()
            pilot.atomic_json(out / "cache_receipt.json", cache.report())
        expected_t = sum(1 for r in selected if r["request_id"] in bench) * repeats * len(specs)
        rap.require(counts["predictions"] == len(selected) * len(specs) and counts["timings"] == expected_t,
                    "worker output inventory mismatch")
        receipt = {"status": "pass", "mode": args.mode, "rank": args.rank, **counts,
                   "work_s": clock() - started, "dataset": dataset_receipt, "labels_accessed": False,
                   "config_sha256": sha256(args.config), "code": code_manifest(), "world_size": WORLD_SIZE,
                   "runtime": getattr(engine, "runtime_receipt", None),
                   "files_sha256": {n: sha256(out / n) for n in
                       ("predictions.jsonl", "timings.jsonl", "cold_start.json", "cache_receipt.json")}}
        pilot.atomic_json(out / "worker_receipt.json", receipt)
        return receipt
    except Exception as exc:
        pilot.atomic_json(out / "error.json", {"status": "fail", "error": str(exc),
                                                "traceback": traceback.format_exc()})
        raise


# --------------------------------------------------------------------------- aggregate

def aggregate(args, *, contract=None):
    pilot = _io()
    config = rap.validate_config(read_json(args.config))
    validate_frozen(config, args.mode)
    contract = contract or _contract()
    data_receipt = contract.check(args.data_dir, public_only=args.mode == "canary")
    rows = contract.rows(args.data_dir, args.mode)
    by_id = {r["request_id"]: r for r in rows}
    rap.require(len(by_id) == len(rows), "duplicate public requests")
    shards = contract.shards(rows)
    bench = contract.benchmark(rows, args.mode)
    repeats = rap.MEASUREMENT["canary_timing_repeats" if args.mode == "canary" else "timing_repeats"]
    specs = [s["id"] for s in rap.variant_specs()]
    predictions, timings = [], []
    root = Path(args.out_dir)
    rap.require({p.name for p in root.glob("rank_*")} == {f"rank_{r:03d}" for r in range(WORLD_SIZE)},
                "worker directory inventory mismatch")
    rap.require(not list(root.rglob("error.json")), "worker error artifact present")
    rap.require(not (root / "analysis_summary.json").exists() and not (root / "RESULTS.md").exists(),
                "refusing to overwrite derived saved outputs")
    for rank in range(WORLD_SIZE):
        rankdir = Path(args.out_dir) / f"rank_{rank:03d}"
        receipt = read_json(rankdir / "worker_receipt.json")
        rap.require(all(receipt.get(k) == v for k, v in {"status": "pass", "mode": args.mode, "rank": rank,
                    "world_size": WORLD_SIZE, "config_sha256": sha256(args.config), "code": code_manifest(),
                    "labels_accessed": False}.items()), f"rank {rank} receipt provenance drift")
        expected_data = dict(data_receipt)
        if "evaluator_files_verified" in expected_data:
            expected_data["evaluator_files_verified"] = False
        rap.require(receipt["dataset"] == expected_data, "worker data provenance mismatch")
        files = ("predictions.jsonl", "timings.jsonl", "cold_start.json", "cache_receipt.json")
        rap.require(receipt.get("files_sha256") == {n: sha256(rankdir / n) for n in files}, "worker file hash mismatch")
        part_p, part_t = pilot.read_rows(rankdir / files[0]), pilot.read_rows(rankdir / files[1])
        rap.require(receipt["predictions"] == len(part_p) and receipt["timings"] == len(part_t), "receipt count mismatch")
        cold = read_json(rankdir / "cold_start.json")
        expected_adapters = [{k: a[k] for k in ("case_id", "seed", "sha256")} for a in config["adapters"]]
        rap.require(cold.get("status") == "pass" and cold.get("adapters") == expected_adapters
                    and cold["index_cache"]["sealed"] is True and cold["optimized"]["sealed"] is True,
                    "cold-start adapter/cache provenance mismatch")
        gates = cold["identity_gates"]
        local_ids = {rid for rid in by_id if shards[rid] == rank}
        rap.require(len(gates) == len(local_ids) and {g["request_id"] for g in gates} == local_ids
                    and all(g.get("bit_exact") is True and g.get("original_native_bit_exact") is True
                            and g.get("compression_bit_exact") is True and
                            g.get("identity_pairs") == [list(p) for p in rap.identity_pairs()] for g in gates),
                    "incomplete equivalence gates")
        geometry_by_id = {g["request_id"]: g["geometry_sha256"] for g in gates}
        cache = read_json(rankdir / "cache_receipt.json")
        plans = {(p["geometry_sha256"], p["arm_id"]): p for p in cache["plans"]}
        rap.require(len(plans) == len(cache["plans"]) and cache["state"] == "sealed"
                    and set(plans) == {(g, c) for g in set(geometry_by_id.values())
                                      for c in ("native", rap.PRIMARY, rap.SECONDARY)}, "schedule inventory drift")
        spec_by_id = {s["id"]: s for s in rap.variant_specs()}
        for r in [*part_p, *part_t]:
            rap.require(r.get("status", "pass") == "pass" and not r.get("error"), "failed observation")
            spec = spec_by_id.get(r["variant"])
            source = by_id.get(r["request_id"])
            rap.require(spec is not None and source is not None and r["rank"] == rank == shards[r["request_id"]],
                        "row rank/inventory drift")
            rap.require(all(r.get(k) == source.get(k) for k in
                        ("frame_key", "cluster_key", "order_id", "source_family", "sampling_unit_kind")), "row identity drift")
            rap.require(all(r.get(k) == spec.get(k) for k in ("compression_id", "path", "case_id", "target"))
                        and r.get("seed") == spec.get("seed", 0), "variant/seed/adapter drift")
            plan = plans[(geometry_by_id[r["request_id"]], spec["compression_id"])]
            rap.require(r.get("plan_cache_hit") is True and r["geometry_sha256"] == plan["geometry_sha256"]
                        and r["schedule_sha256"] == plan["schedule_sha256"], "row schedule drift")
            components = [r[k] for k in ("preparation_ms", "plan_lookup_ms", "model_forward_ms", "postprocess_ms")]
            total = r["end_to_end_elapsed_ms"]
            rap.require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in [*components, total])
                        and total > 0 and math.isclose(sum(components), total, abs_tol=1e-5, rel_tol=1e-7),
                        "timing components/total mismatch")
        for r in part_t:
            expected_order = variant_order(r["request_id"], phase="benchmark", repeat=r["repeat_id"])
            rap.require(r["variant_order"] == expected_order.index(r["variant"]), "timing order drift")
            rap.require(r.get("batch_size") == 1 and r.get("synchronized") is True
                        and isinstance(r.get("device_id"), str) and r["device_id"], "timing device/batch provenance missing")
            rap.require(all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) and r[k] >= 0
                            for k in ("peak_allocated_bytes", "peak_reserved_bytes")), "invalid memory telemetry")
        rap.require(len({r["device_id"] for r in part_t}) <= 1, "worker device changed")
        for r in part_p:
            logits = r.get("logits_ab", [])
            rap.require(len(logits) == 2 and all(math.isfinite(x) for x in logits)
                        and r["margin_ab"] == (logits[0] - logits[1]) / math.log(2)
                        and r["prediction"] == ("A" if logits[0] >= logits[1] else "B"), "invalid A/B logits")
        predictions += part_p
        timings += part_t
    expected_p = {(rid, v) for rid in by_id for v in specs}
    seen_p = [(r["request_id"], r["variant"]) for r in predictions]
    rap.require(len(seen_p) == len(set(seen_p)) and set(seen_p) == expected_p, "prediction inventory mismatch")
    rap.require(all(r["rank"] == shards[r["request_id"]] for r in predictions), "rank drift")
    expected_t = {(rid, v, k) for rid in bench for v in specs for k in range(repeats)}
    seen_t = [(r["request_id"], r["variant"], r["repeat_id"]) for r in timings]
    rap.require(len(seen_t) == len(set(seen_t)) and set(seen_t) == expected_t, "timing inventory mismatch")
    lookup = {(r["request_id"], r["variant"]): logits_digest(r) for r in predictions}
    for rid in by_id:
        for left, right in rap.identity_pairs():
            rap.require(lookup[(rid, left)] == lookup[(rid, right)], "saved logits violate equivalence gate")
    rap.validate_observations(rows, predictions, timings, {rid: "A" for rid in by_id})
    result = {"status": "pass", "mode": args.mode, "experiment_id": rap.EXPERIMENT_ID,
              "prediction_rows": len(predictions), "timing_rows": len(timings),
              "config_sha256": sha256(args.config)}
    if args.mode == "canary":
        result["scope"] = "technical canary only; labels not opened"
        pilot.atomic_json(Path(args.out_dir) / "analysis_summary.json", result)
        return result
    label_rows = pilot.read_rows(Path(args.data_dir) / "evaluator/labels.jsonl")
    labels = {r["request_id"]: r["correct_label"] for r in label_rows}
    rap.require(len(label_rows) == len(labels) and set(labels) == set(by_id), "label inventory mismatch")
    summary = rap.summarize(rows, predictions, timings, labels)
    summary["per_source"] = {}
    for source in rap.SOURCE_FAMILIES:
        ids = {r["request_id"] for r in rows if r.get("source_family") == source}
        part = [r for r in rows if r["request_id"] in ids]
        summary["per_source"][source] = rap.summarize(
            part, [r for r in predictions if r["request_id"] in ids],
            [r for r in timings if r["request_id"] in ids], labels, source_equal=False)["accuracy_differences_vs_native"]
    pilot.atomic_json(Path(args.out_dir) / "analysis_summary.json", {**result, **summary})
    (Path(args.out_dir) / "RESULTS.md").write_text(rap.render_markdown(summary))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("train", "freeze", "evaluate", "aggregate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("canary", "full"), default="full")
    parser.add_argument("--data-dir", type=Path, help="image assets (train) or cross-source data root (evaluate)")
    parser.add_argument("--dataset-dir", type=Path, help="token-matrix dataset with labels/train.jsonl (train)")
    parser.add_argument("--adapters-dir", type=Path)
    parser.add_argument("--train-run", type=Path, help="completed training run directory (freeze)")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE)
    args = parser.parse_args(argv)
    if args.command == "train":
        result = train_worker(args)
    elif args.command == "freeze":
        result = freeze(args.config, args.train_run)
    elif args.command == "evaluate":
        result = evaluate_worker(args)
    else:
        result = aggregate(args)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
