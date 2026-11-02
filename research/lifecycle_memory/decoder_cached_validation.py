"""Frozen shortlist and fresh-data contracts for cached-plan validation.

No model loading at import. Public workers never need evaluator labels. Source
selection is frozen from completed exploratory runs; new outcomes cannot select
adapters, sites, seeds, or compression budgets.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import mean

from research.lifecycle_memory.decoder_token_matrix_release import (
    atomic_json, canonical, checked_hash, digest, entry, inventory, read_json,
    relative_path, require, sha256,
)

JOB_REL = "docs/research/lifecycle_memory/phase 2 research/jobs/decoder_cached_validation"
COMPRESSIONS = (
    "native", "Fhalf_Chalf_L34_prune", "C384_L22_merge__F67_L34_merge",
    "F0_L22_prune", "F0_L30_prune", "F0_L34_prune", "C379_L22_merge",
    "F67_L34_merge",
)
CASES = (
    "native__correct_L34_FQ", "native__correct_L34_CQ",
    "native__correct_L34_readout", "C379_L22_merge__correct_L34_FQ",
    "F67_L34_merge__correct_L34_readout",
)
SEEDS = (17, 29)
CROSS_SOURCE = "cross_source_transfer"
SOURCE_FAMILIES = ("thu_read", "escape", "youtube")
SOURCE_KINDS = {"thu_read": "participant", "escape": "recording", "youtube": "recording"}
MODEL = {"key": "gemma4_31b_hf_bf16", "id": "google/gemma-4-31B-it",
         "revision": "3548789868c5356dbf307c98e6f609007b82b3eb",
         "torch": "2.12.1+cu126", "transformers": "5.13.0"}
# Exact source-battery wording; freeze the task as well as the token geometry.
PROMPTS = {
    "system_prompt": (
        "Answer the first-versus-second-order target question using only the provided "
        "RGB inputs. Return exactly one JSON object with keys: answer, abstain, "
        "abstention_reason, confidence. The answer value must be A or B."),
    "user_prompt": (
        "Input 1 is a full frame. Inputs 2 and 3 are unmarked target-only reference "
        "crops A and B. Which referenced target is the first-order interacting object, "
        "meaning the object directly manipulated by a visible hand rather than a "
        "second-order object interacting through it? Return exactly one JSON object "
        "with answer A or B plus the frozen abstention and confidence keys."),
}


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def order(value):
    return hashlib.sha256(("cached-validation-v1:" + str(value)).encode()).hexdigest()


def _cross_source(config):
    return config["fresh_data"]["scope"] == CROSS_SOURCE


def _count_contract(counts):
    require(isinstance(counts, dict) and set(counts) == {"frames", "requests", "clusters"}
            and all(type(v) is int and v > 0 for v in counts.values()), "invalid expected counts")
    require(counts["requests"] == 2 * counts["frames"] and counts["clusters"] <= counts["frames"],
            "invalid expected counts")


def validate_config(config):
    require(config["schema_version"] == 1 and config["model"] == MODEL, "model/config drift")
    require(config["compression_ids"] == list(COMPRESSIONS), "shortlist changed")
    require(config["correction_cases"] == list(CASES) and config["seeds"] == list(SEEDS), "corrections changed")
    require(config["measurement"] == {"batch_size": 1, "canary_frames": 16,
            "benchmark_requests": 128, "timing_repeats": 5,
            "canary_timing_repeats": 2, "warmups": 2}, "timing budget drift")
    require(config["bootstrap"] == {"draws": 10000, "seed": 20261015}, "bootstrap drift")
    require(config["decision"] == {"family_alpha": .05, "bounds": 3,
            "compression_accuracy_floor": -.01, "compression_speedup_floor": 1.01,
            "steering_accuracy_floor": 0.0}, "decision contract drift")
    fresh = config["fresh_data"]
    require(fresh["scope"] in ("participant_and_video_disjoint", "video_disjoint", CROSS_SOURCE), "invalid freshness scope")
    require(type(fresh["minimum_frames"]) is int and fresh["minimum_frames"] >= 32, "fresh data budget too small")
    if _cross_source(config):
        require(fresh.get("source_families") == list(SOURCE_FAMILIES), "cross-source families changed")
        for key, floor in (("minimum_units_per_source", 4), ("minimum_frames_per_source", 8)):
            require(type(fresh.get(key)) is int and fresh[key] >= floor, "cross-source budget too small")
        if "minimum_clusters" in fresh:
            require(type(fresh["minimum_clusters"]) is int and fresh["minimum_clusters"] >= 12,
                    "cross-source cluster budget too small")
        if "expected_counts" in fresh:
            _count_contract(fresh["expected_counts"])
        if "expected_source_counts" in fresh:
            require(set(fresh["expected_source_counts"]) == set(SOURCE_FAMILIES), "expected source coverage")
            for counts in fresh["expected_source_counts"].values():
                _count_contract(counts)
        if "data_manifest_sha256" in fresh:
            checked_hash(fresh["data_manifest_sha256"])
    else:
        require(fresh["minimum_clusters"] >= 10, "fresh data budget too small")
    require(config["training_allowed"] is False, "this is evaluation only")
    if "exposure_registry_sha256" in config:
        checked_hash(config["exposure_registry_sha256"])
    expected = {(case, seed) for case in CASES for seed in SEEDS}
    observed = set()
    for a in config["adapters"]:
        require((a["case_id"], a["seed"]) not in observed, "duplicate adapter")
        observed.add((a["case_id"], a["seed"]))
        relative_path(a["path"]); checked_hash(a["sha256"])
        require(a["parameters"] == 43008 and a["rank"] == 4 and a["cut"] == 34,
                "adapter shape drift")
        require(a["target"] == a["case_id"].split("__correct_L34_")[1], "adapter target drift")
    require(observed == expected, "adapter inventory mismatch")
    return config


def variant_specs(config):
    validate_config(config)
    variants = [{"id": name, "compression_id": name} for name in COMPRESSIONS]
    for a in config["adapters"]:
        variants.append({"id": f"{a['case_id']}_seed{a['seed']}",
                         "compression_id": a["case_id"].split("__correct_L34_")[0], "adapter": a})
    require(len(variants) == 18 and len({v["id"] for v in variants}) == 18, "variant drift")
    return variants


def freeze_config(repo, steering_run):
    """Copy only ten tiny frozen adapters, never a base model or previous inputs."""
    repo, run = Path(repo), Path(steering_run)
    from research.lifecycle_memory.decoder_token_matrix_release import verify_copyback
    verify_copyback(run)
    out = repo / JOB_REL
    path = out / "config.json"
    require(not path.exists(), "refusing to overwrite an existing experiment freeze")
    adapters = []
    for case in CASES:
        for seed in SEEDS:
            matches = list(run.glob(f"rank_*/{case}_seed{seed}/training_receipt.json"))
            require(len(matches) == 1, "missing/ambiguous trained adapter")
            receipt = read_json(matches[0])
            require(receipt["status"] == "pass" and receipt["steps"] == 256
                    and receipt["checkpoint_selection"] == "fixed_final_step", "not a final full adapter")
            source = matches[0].parent / receipt["checkpoint"]
            require(sha256(source) == receipt["checkpoint_sha256"], "adapter hash mismatch")
            name = source.name
            dest = out / "artifacts/adapters" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            require(not dest.exists(), "adapter destination already exists")
            shutil.copyfile(source, dest)
            adapters.append({"path": name, "sha256": sha256(dest), "case_id": case,
                "target": receipt["target"], "seed": seed, "cut": 34, "rank": 4,
                "parameters": receipt["parameters"], "source_training_receipt_sha256": sha256(matches[0])})
    config = {"schema_version": 1, "experiment_id": "decoder-cached-validation-2026-10-15-v1",
        "model": MODEL, "compression_ids": list(COMPRESSIONS), "correction_cases": list(CASES),
        "seeds": list(SEEDS), "adapters": adapters, "training_allowed": False,
        "source_run": {"id": "620395", "analysis_sha256": sha256(run / "analysis_summary.json")},
        "fresh_data": {"scope": "participant_and_video_disjoint", "minimum_frames": 400,
                       "minimum_clusters": 50, "status": "awaiting_data_selection"},
        "measurement": {"batch_size": 1, "canary_frames": 16, "benchmark_requests": 128,
                        "timing_repeats": 5, "canary_timing_repeats": 2, "warmups": 2},
        "bootstrap": {"draws": 10000, "seed": 20261015},
        "decision": {"family_alpha": .05, "bounds": 3, "compression_accuracy_floor": -.01,
                     "compression_speedup_floor": 1.01, "steering_accuracy_floor": 0.0},
        "scope": "Prospective evaluation of an exploratory shortlist, conditional on frozen trained adapters. No retraining, new-data calibration, or selection. No general generation/KV-cache claim."}
    validate_config(config)
    atomic_json(path, config)
    return {"config": str(path), "sha256": sha256(path), "adapters": len(adapters),
            "status": "source_frozen_data_pending"}


def freeze_exposure(repo, audit_path):
    """Bind metadata-only exclusions after rechecking the audit's local inputs."""
    repo, audit_path = Path(repo).resolve(), Path(audit_path)
    audit = read_json(audit_path)
    require(audit["exposure_index_count"] == len(audit["exposure_indices"]) == 17, "audit source count drift")
    for source in audit["inputs"]:
        path = Path(source["path"])
        require(repo in path.resolve().parents and path.stat().st_size == source["bytes"]
                and sha256(path) == checked_hash(source["sha256"]), "audit input changed")
    u = audit["exposure_union"]
    registry = {"schema_version": 1, "audit_sha256": sha256(audit_path),
        "sources": [{"path": str(Path(s["path"]).relative_to(repo)), "sha256": s["sha256"]}
                    for s in audit["exposure_indices"]],
        "participants": sorted(set(u["ego4d_participant_ids"]) | set(u["namespaced_participant_or_subject_keys"])),
        "videos": u["video_ids"], "frame_ids": u["frame_ids"],
        "asset_sha256": sorted(set(u["source_rgb_sha256"]) | set(u["declared_nonblank_asset_sha256"])),
        "candidate_ids": u["candidate_ids"], "component_fingerprints": u["selected_component_fingerprints"],
        "scope": "Seventeen audited exposure/construction indices; declared prior asset hashes. New-data builder must additionally verify raw bytes, provenance and near-duplicates."}
    root = repo / JOB_REL
    config = read_json(root / "config.json")
    require("exposure_registry_sha256" not in config, "exposure registry already frozen")
    dest = root / "artifacts/exposure_registry.json"
    require(not dest.exists(), "refusing to replace exposure registry")
    atomic_json(dest, registry)
    config["exposure_registry_sha256"] = sha256(dest)
    validate_config(config)
    atomic_json(root / "config.json", config)
    return {"status": "exposure_frozen_data_pending", "sources": 17,
            "registry_sha256": sha256(dest), "bytes": dest.stat().st_size}


def _validate_rows(public, *, cross_source=False):
    from research.lifecycle_memory.decoder_token_matrix_steering import _public_only
    metadata = {"source_family", "sampling_unit_kind"} if cross_source else set()
    _public_only([{k: v for k, v in r.items() if k not in metadata} for r in public])
    from research.lifecycle_memory.run_decoder_token_matrix import validate_public_rows
    validate_public_rows(public, expected_requests=len(public))
    require(all(r["split"] == "test" for r in public), "new data must be test-only")
    pairs = defaultdict(list)
    cluster_sources = {}
    for row in public:
        if cross_source:
            source = row.get("source_family")
            require(isinstance(source, str) and source in SOURCE_KINDS, "invalid public source family")
            require(row.get("sampling_unit_kind") == SOURCE_KINDS[source], "public sampling unit kind mismatch")
            require(row.get("split_group_key") == row["cluster_key"], "public cluster alias mismatch")
            require(cluster_sources.setdefault(row["cluster_key"], source) == source,
                    "cluster source mismatch")
        require(row["arm"] == "real", "only real inputs")
        require(row.get("arms") == {"real": {"inputs": row["inputs"]}}, "missing real-input engine alias")
        require(all(row[k] == value for k, value in PROMPTS.items()), "source task prompt changed")
        require([a["kind"] for a in row["inputs"]] ==
                ["full_frame", "reference_crop_a", "reference_crop_b"], "asset slot drift")
        for asset in row["inputs"]:
            name = relative_path(asset["path"]); checked_hash(asset["sha256"])
            require(name.startswith("assets/"), "assets must use the label-free assets directory")
        pairs[row["frame_key"]].append(row)
    for pair in pairs.values():
        a, b = sorted(pair, key=lambda r: r["order_id"])
        require(a["inputs"][0] == b["inputs"][0] and all(
            a["inputs"][i][field] == b["inputs"][j][field]
            for i,j in ((1,2),(2,1)) for field in ("path","sha256","width","height")),
            "orders must be exact crop swaps")
        require(all(a[k] == b[k] for k in ("system_prompt", "user_prompt", "cluster_key")), "paired prompt drift")
        if cross_source:
            require(all(a[k] == b[k] for k in metadata), "paired source metadata mismatch")
    return pairs


def check_dataset(data_dir, config, verify_assets=True, *, public_only=False):
    """Recompute overlap; workers verify public files without opening labels.

    The manifest retains the sealed evaluator digest, but that directory is not
    mounted for workers. Full package verification happens before isolation.
    """
    validate_config(config)
    require(config["fresh_data"].get("status") == "frozen", "data design still awaits selection/freeze")
    root = Path(data_dir)
    manifest = read_json(root / "data_manifest.json")
    cross_source = _cross_source(config)
    if cross_source and "data_manifest_sha256" in config["fresh_data"]:
        require(sha256(root / "data_manifest.json") == config["fresh_data"]["data_manifest_sha256"],
                "frozen data manifest digest drift")
    entries = inventory(manifest)
    by_path = {e["path"]: e for e in entries}
    required = {"public/requests.jsonl", "evaluator/labels.jsonl", "freshness.json", "exposure_registry.json"}
    require(required <= set(by_path), "incomplete data manifest")
    for name, record in by_path.items():
        require(name.startswith(("assets/", "public/", "evaluator/")) or name in
                ("freshness.json", "exposure_registry.json"), "unexpected data file")
        if public_only and name.startswith("evaluator/"):
            continue
        file = root / name
        require(file.is_file() and not file.is_symlink() and root.resolve() in file.resolve().parents,
                f"missing/unsafe data file {name}")
        require(file.stat().st_size == record["bytes"], f"data size drift: {name}")
        if verify_assets or not name.startswith("assets/"):
            require(sha256(file) == record["sha256"], f"data digest drift: {name}")
    public = rows(root / "public/requests.jsonl")
    pairs = _validate_rows(public, cross_source=cross_source)
    fresh = read_json(root / "freshness.json")
    registry = read_json(root / "exposure_registry.json")
    require(fresh["exposure_registry_sha256"] == sha256(root / "exposure_registry.json"), "exposure registry drift")
    require(sha256(root / "exposure_registry.json") == checked_hash(config.get("exposure_registry_sha256")),
            "exposure registry differs from frozen audited history")
    require(registry["sources"] and all(checked_hash(s["sha256"]) and s["path"] for s in registry["sources"]),
            "exposure sources missing")
    require(fresh["scope"] == config["fresh_data"]["scope"] and fresh["status"] == "pass", "freshness contract mismatch")
    identities = fresh["identities"]
    require(len(identities) == len(pairs) and {i["frame_key"] for i in identities} == set(pairs), "fresh identity coverage")
    field_map = {"participant_id": "participants", "video_id": "videos", "source_frame_id": "frame_ids"}
    overlaps = {}
    for field, namespace in field_map.items():
        require(all(field in i and ((cross_source and field == "participant_id" and i[field] is None)
                    or (isinstance(i[field], str) and i[field].strip())) for i in identities),
                f"missing identity {field}")
        overlaps[namespace] = sorted({i[field] for i in identities if i[field] is not None} & set(registry[namespace]))
    require(not overlaps["videos"] and not overlaps["frame_ids"], "previously exposed video/frame")
    if fresh["scope"] == "participant_and_video_disjoint" or cross_source:
        require(not overlaps["participants"], "previously exposed participant")
    source_hashes = {checked_hash(i["source_image_sha256"]) for i in identities}
    require(len(source_hashes) == len(identities) and
            len({i["source_frame_id"] for i in identities}) == len(identities), "duplicate source frame/image")
    video_participants = {}
    for identity in identities:
        require(video_participants.setdefault(identity["video_id"], identity["participant_id"]) == identity["participant_id"],
                "one video assigned inconsistent participant identities")
    asset_hashes = {a["sha256"] for r in public for a in r["inputs"]}
    require(not (source_hashes | asset_hashes) & set(registry["asset_sha256"]), "previously exposed image bytes")
    owners = {}
    unit_sources, video_sources = {}, {}
    for identity in identities:
        r = pairs[identity["frame_key"]][0]
        if cross_source:
            source, kind, unit = (identity.get(k) for k in ("source_family", "sampling_unit_kind", "sampling_unit_id"))
            require(isinstance(source, str) and source in SOURCE_KINDS and kind == SOURCE_KINDS[source],
                    "source sampling unit mismatch")
            require(isinstance(unit, str) and unit.strip(), "missing sampling unit identity")
            require((kind == "participant" and identity["participant_id"] is not None
                     and unit == identity["participant_id"])
                    or (kind == "recording" and identity["participant_id"] is None and unit == identity["video_id"]),
                    "sampling unit/participant identity mismatch")
            require(r["source_family"] == source and r["sampling_unit_kind"] == kind, "public/identity source mismatch")
            require(unit_sources.setdefault((kind, unit), source) == source
                    and video_sources.setdefault(identity["video_id"], (source, kind, unit)) == (source, kind, unit),
                    "inconsistent source for video/sampling unit")
            expected = kind + "_" + digest(unit)[:24]
        else:
            expected = "participant_" + digest(identity["participant_id"])[:24]
        require(r["cluster_key"] == expected, "sampling participant/cluster alias mismatch")
        for a in r["inputs"]:
            require(a["path"] in by_path and by_path[a["path"]]["sha256"] == a["sha256"], "asset inventory mismatch")
            require(owners.setdefault(a["sha256"], r["frame_key"]) == r["frame_key"], "image reused across new frames")
    counts = {"frames": len(pairs), "requests": len(public), "clusters": len({r["cluster_key"] for r in public})}
    require(counts["frames"] >= config["fresh_data"]["minimum_frames"] and
            counts["clusters"] >= config["fresh_data"].get("minimum_clusters", 0), "new data below frozen budget")
    require(manifest["counts"] == counts, "data count drift")
    extra = {}
    if cross_source:
        source_counts = _source_counts(public)
        require(manifest.get("source_counts") == source_counts, "source count drift")
        for count in source_counts.values():
            require(count["frames"] >= config["fresh_data"]["minimum_frames_per_source"]
                    and count["clusters"] >= config["fresh_data"]["minimum_units_per_source"],
                    "source below frozen budget")
        for key, observed in (("expected_counts", counts), ("expected_source_counts", source_counts)):
            require(key not in config["fresh_data"] or config["fresh_data"][key] == observed,
                    "frozen expected counts drift")
        extra = {"source_counts": source_counts, "sampling_unit_kinds": SOURCE_KINDS}
    return {"status": "pass", "scope": fresh["scope"], "counts": counts, **extra,
            "manifest_sha256": sha256(root / "data_manifest.json"), "overlaps": overlaps,
            "exposure_registry_sha256": fresh["exposure_registry_sha256"],
            "evaluator_files_verified": not public_only}


def _source_counts(public):
    return {source: {"frames": len({r["frame_key"] for r in public if r["source_family"] == source}),
                     "requests": sum(r["source_family"] == source for r in public),
                     "clusters": len({r["cluster_key"] for r in public if r["source_family"] == source})}
            for source in SOURCE_FAMILIES}


def _source_balanced_frames(public):
    """Round-robin sources, then units within source, without opening labels."""
    queues = {}
    for source in SOURCE_FAMILIES:
        grouped = defaultdict(set)
        for row in public:
            if row["source_family"] == source:
                grouped[row["cluster_key"]].add(row["frame_key"])
        require(grouped, "missing source coverage")
        units = {unit: sorted(frames, key=order) for unit, frames in grouped.items()}
        queue = []
        while any(units.values()):
            for unit in sorted(units, key=order):
                if units[unit]:
                    queue.append(units[unit].pop(0))
        queues[source] = queue
    frames = []
    while any(queues.values()):
        for source in SOURCE_FAMILIES:
            if queues[source]:
                frames.append(queues[source].pop(0))
    return frames


def public_rows(data_dir, mode, config):
    require(mode in ("canary", "full"), "invalid mode")
    public = rows(Path(data_dir) / "public/requests.jsonl")
    cross_source = _cross_source(config)
    _validate_rows(public, cross_source=cross_source)
    frames = sorted({r["frame_key"] for r in public}, key=order)
    if cross_source:
        frames = _source_balanced_frames(public)
    selected = set(frames[:config["measurement"]["canary_frames"]] if mode == "canary" else frames)
    return sorted((r for r in public if r["frame_key"] in selected), key=lambda r: (order(r["frame_key"]), r["order_id"]))


def frame_shards(public):
    ranks = {f: i % 4 for i, f in enumerate(sorted({r["frame_key"] for r in public}, key=order))}
    return {r["request_id"]: ranks[r["frame_key"]] for r in public}


def benchmark_ids(public, mode, config):
    """Cluster-round-robin sample, retaining both orders; independent of labels."""
    require(mode in ("canary", "full"), "invalid mode")
    if _cross_source(config):
        _validate_rows(public, cross_source=True)
        frames = _source_balanced_frames(public)
        cap = len(frames) if mode == "canary" else config["measurement"]["benchmark_requests"] // 2
        require(cap >= len(SOURCE_FAMILIES), "timing subset lacks source coverage")
        chosen = set(frames[:cap])
        return {r["request_id"] for r in public if r["frame_key"] in chosen}
    grouped = defaultdict(set)
    for r in public:
        grouped[r["cluster_key"]].add(r["frame_key"])
    queues = {c: sorted(v, key=order) for c, v in grouped.items()}
    frames = []
    while any(queues.values()):
        for c in sorted(queues, key=order):
            if queues[c]:
                frames.append(queues[c].pop(0))
    cap = len(frames) if mode == "canary" else config["measurement"]["benchmark_requests"] // 2
    chosen = set(frames[:cap])
    return {r["request_id"] for r in public if r["frame_key"] in chosen}


def aggregate_results(out_dir, data_dir, config, mode):
    """Inventory first; canaries never open labels or report accuracy."""
    out = Path(out_dir)
    if _cross_source(config):
        # Source attribution is a join against the sealed public dataset, not
        # unverified worker metadata. Canaries must not touch the label mount.
        check_dataset(data_dir, config, verify_assets=False, public_only=mode == "canary")
    public = public_rows(data_dir, mode, config)
    by_id = {r["request_id"]: r for r in public}
    variants = variant_specs(config)
    ranks = frame_shards(public)
    benchmark = benchmark_ids(public, mode, config)
    repeats = config["measurement"]["canary_timing_repeats" if mode == "canary" else "timing_repeats"]
    predictions, timings, workers = [], [], []
    for rank in range(4):
        directory = out / f"rank_{rank:03d}"
        worker = read_json(directory / "worker_receipt.json")
        require(worker["status"] == "pass" and worker["mode"] == mode, "worker failed/mode drift")
        workers.append(worker)
        predictions.extend(rows(directory / "predictions.jsonl"))
        timings.extend(rows(directory / "timings.jsonl"))
    def verify(values, timed):
        expected = {(v["id"], r["request_id"], rep) for v in variants for r in public
                    if not timed or r["request_id"] in benchmark for rep in (range(repeats) if timed else [-1])}
        seen = set()
        for r in values:
            rid = r["request_id"]
            require(rid in by_id and r["rank"] == ranks[rid], "request/rank drift")
            key = (r["variant"], rid, r["repeat_id"] if timed else -1)
            require(key in expected and key not in seen, "duplicate/extra outcome")
            seen.add(key)
            require(not r.get("error"), "error outcome")
            require(all(r[k] == by_id[rid][k] for k in ("frame_key", "cluster_key", "split", "order_id")), "identity drift")
            if _cross_source(config):
                require(all(k not in r or r[k] == by_id[rid][k]
                            for k in ("source_family", "sampling_unit_kind")), "outcome source metadata drift")
            if timed:
                require(r["plan_cache_hit"] is True and r["plan_construction_ms"] == 0,
                        "timed cache miss/build")
                require(r["warmup_complete"] is True and r["is_warmup"] is False,
                        "cold measurement mixed with warm timing")
                for field in ("elapsed_ms", "end_to_end_elapsed_ms"):
                    require(math.isfinite(r[field]) and r[field] > 0, "invalid time")
                require(r["end_to_end_elapsed_ms"] >= r["elapsed_ms"] and r["synchronized"] is True,
                        "invalid timing scope")
            else:
                require(math.isfinite(r["margin_ab"]) and r["prediction"] == ("A" if r["margin_ab"] >= 0 else "B"), "invalid answer")
        require(seen == expected, "incomplete outcome inventory")
    verify(predictions, False); verify(timings, True)
    projected = max(w["full_projected_worker_s"] for w in workers)
    require(all(math.isfinite(w["full_projected_worker_s"]) and w["full_projected_worker_s"] > 0
                for w in workers), "invalid projected runtime")
    walltime = max(3600, math.ceil((1.4 * projected + 600) / 300) * 300)
    result = {"status": "pass", "mode": mode, "prediction_rows": len(predictions), "timing_rows": len(timings),
              "config_sha256": digest(config), "full_walltime_seconds": walltime,
              "full_launch_allowed": walltime <= 10800, "labels_opened": mode == "full"}
    if _cross_source(config):
        result.update(scope=CROSS_SOURCE, confirmatory=False, population_generalization=False,
                      source_counts=_source_counts(public))
    if mode == "canary":
        atomic_json(out / "analysis_summary.json", {**result, "scope": "technical-only; no new-data outcomes inspected"})
        return result
    label_rows = rows(Path(data_dir) / "evaluator/labels.jsonl")
    labels = {r["request_id"]: r for r in label_rows}
    require(len(labels) == len(label_rows) and set(labels) == set(by_id), "label inventory mismatch")
    for rid, label in labels.items():
        require(label["correct_label"] in ("A", "B") and all(label[k] == by_id[rid][k]
                for k in ("frame_key", "cluster_key", "split", "order_id")), "label identity mismatch")
    for frame in {r["frame_key"] for r in public}:
        require({labels[r["request_id"]]["correct_label"] for r in public if r["frame_key"] == frame} == {"A", "B"}, "labels fail counterbalance")
    summary = _summaries(public, predictions, timings, labels, config, source_equal=_cross_source(config))
    if _cross_source(config):
        _descriptive_summary(summary)
        summary["pooling"] = "equal source means; equal sampling units within each source"
        summary["resampling"] = {
            "method": "independent within-source cluster bootstrap, then equal-source pooling",
            "sampling_unit_kinds": SOURCE_KINDS,
            "participant_generalization": False,
        }
        summary["per_source"] = {}
        for source in SOURCE_FAMILIES:
            selected = [r for r in public if r["source_family"] == source]
            ids = {r["request_id"] for r in selected}
            part = _summaries(selected, [r for r in predictions if r["request_id"] in ids],
                              [r for r in timings if r["request_id"] in ids], labels, config)
            _descriptive_summary(part)
            part["resampling"] = {"sampling_unit_kind": SOURCE_KINDS[source],
                                  "method": "equal-unit paired cluster bootstrap within source"}
            summary["per_source"][source] = part
    atomic_json(out / "analysis_summary.json", {**result, **summary})
    return result


def _descriptive_summary(summary):
    diagnostic = summary.pop("primary_decisions")
    diagnostic.update(confirmatory=False, interpretation="diagnostic/descriptive only",
                      method="Same frozen numerical bounds; descriptive bootstrap diagnostics, not population generalization.")
    summary["diagnostic_decisions"] = diagnostic
    summary["primary_decisions"] = {"confirmatory": False, "eligible": False, "status": "not_applicable",
        "reason": "Cross-source transfer is descriptive, not confirmatory population generalization."}
    summary["interval_scope"] = (
        "Descriptive only, conditional on frozen adapters and sampled sources/units. "
        "THU resamples known participants; escape and youtube resample recordings, not people. "
        "Seeds and paired orders are averaged within units, not independent observations.")
    summary["memory_scope"] += (
        " unit_peak_allocated_bytes summarizes per-unit maxima with descriptive cluster bootstrap intervals; "
        "peak_allocated_bytes is the observed overall maximum, not a bootstrapped maximum.")


def _summaries(public, predictions, timings, labels, config, *, source_equal=False):
    """Pair within units; optionally give each source equal pooled weight."""
    import numpy as np

    cells = defaultdict(dict)
    for r in predictions:
        cells[r["variant"]][r["request_id"]] = r
    clusters = sorted({r["cluster_key"] for r in public})
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    cluster_sources = {r["cluster_key"]: r.get("source_family") for r in public}
    def resampler(keys):
        groups = ([np.array([i for i, key in enumerate(keys) if cluster_sources[key] == source])
                   for source in SOURCE_FAMILIES] if source_equal else [np.arange(len(keys))])
        require(all(len(g) for g in groups), "missing source coverage in bootstrap")
        draws = [g[rng.integers(0, len(g), size=(config["bootstrap"]["draws"], len(g)))] for g in groups]
        def stats(array):
            return (float(np.mean([array[g].mean() for g in groups])),
                    np.mean([array[d].mean(axis=1) for d in draws], axis=0))
        return stats
    accuracy_stats = resampler(clusters)
    vectors = {}
    for name, cell in cells.items():
        per_cluster = defaultdict(list)
        for rid, r in cell.items():
            correct = r["prediction"] == labels[rid]["correct_label"]
            signed = r["margin_ab"] * math.log(2) * (1 if labels[rid]["correct_label"] == "A" else -1)
            per_cluster[r["cluster_key"]].append((float(correct), max(0., -signed) + math.log1p(math.exp(-abs(signed)))))
        vectors[name] = {"accuracy": np.array([mean(x[0] for x in per_cluster[c]) for c in clusters]),
                         "log_loss_nats": np.array([mean(x[1] for x in per_cluster[c]) for c in clusters])}
    for case in CASES:
        vectors[case + "_seed_mean"] = {m: np.mean([vectors[f"{case}_seed{s}"][m] for s in SEEDS], axis=0)
                                           for m in ("accuracy", "log_loss_nats")}
    def interval(array, stats=accuracy_stats):
        estimate, samples = stats(array)
        return {"estimate": estimate, "ci95": np.quantile(samples, [.025, .975]).tolist(),
                "family_lower_bound": float(np.quantile(samples, .05 / 3))}
    accuracy = {name: {m: interval(v) for m, v in metrics.items()} for name, metrics in vectors.items()}
    differences = {name: interval(v["accuracy"] - vectors["native"]["accuracy"])
                   for name, v in vectors.items() if name != "native"}
    time_cells = defaultdict(dict)
    for r in timings:
        time_cells[r["variant"]][(r["request_id"], r["repeat_id"])] = r
    timed_clusters = sorted({r["cluster_key"] for r in timings})
    time_stats = resampler(timed_clusters)
    timing = {}
    for name, cell in time_cells.items():
        left, right = defaultdict(list), defaultdict(list)
        for key, r in cell.items():
            base = time_cells["native"][key]
            require(base["device_id"] == r["device_id"] and base["batch_size"] == r["batch_size"] == 1, "unpaired hardware timing")
            left[r["cluster_key"]].append(base["end_to_end_elapsed_ms"])
            right[r["cluster_key"]].append(r["end_to_end_elapsed_ms"])
        a = np.array([mean(left[c]) for c in timed_clusters]); b = np.array([mean(right[c]) for c in timed_clusters])
        a_mean, a_samples = time_stats(a)
        b_mean, b_samples = time_stats(b)
        samples = a_samples / b_samples
        timing[name] = {"native_mean_ms": a_mean, "cached_request_mean_ms": b_mean,
            "speedup": a_mean/b_mean, "speedup_ci95": np.quantile(samples, [.025,.975]).tolist(),
            "family_lower_bound": float(np.quantile(samples, .05/3)),
            "forward_mean_ms": mean(r["elapsed_ms"] for r in cell.values()),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in cell.values())}
        for field, label, reducer in (("elapsed_ms", "forward_ms", mean),
                                      ("peak_allocated_bytes", "unit_peak_allocated_bytes", max)):
            grouped = defaultdict(list)
            for r in cell.values():
                require(isinstance(r[field], (int, float)) and math.isfinite(r[field]) and r[field] >= 0,
                        "invalid timing/memory value")
                grouped[r["cluster_key"]].append(r[field])
            timing[name][label] = interval(np.array([reducer(grouped[c]) for c in timed_clusters]), time_stats)
        if _cross_source(config):
            timing[name]["forward_mean_ms"] = timing[name]["forward_ms"]["estimate"]
        timing[name]["native_ms"] = interval(a, time_stats)
        timing[name]["cached_request_ms"] = interval(b, time_stats)
        component_means = {}
        for field in ("preparation_ms", "plan_lookup_ms", "model_forward_ms", "postprocess_ms"):
            require(all(field in r and math.isfinite(r[field]) and r[field] >= 0 for r in cell.values()),
                    f"missing/invalid timing component: {field}")
            grouped = defaultdict(list)
            for r in cell.values():
                grouped[r["cluster_key"]].append(r[field])
            component_means[field] = time_stats(np.array([mean(grouped[c]) for c in timed_clusters]))[0]
        timing[name]["cluster_equal_components_ms"] = component_means
        require(math.isclose(sum(component_means.values()), b_mean, rel_tol=1e-6, abs_tol=1e-6),
                "request timing components do not sum to total")
    comp = "Fhalf_Chalf_L34_prune"
    fq = "native__correct_L34_FQ_seed_mean"
    decision = {"method": "Three frozen one-sided percentile bootstrap bounds; Bonferroni alpha=.05/3. Conditional on fixed adapters, sampled participants and benchmark hardware.",
        "compression_accuracy_pass": differences[comp]["family_lower_bound"] > -.01,
        "compression_speed_pass": timing[comp]["family_lower_bound"] > 1.01,
        "steering_accuracy_pass": differences[fq]["family_lower_bound"] > 0.0}
    decision["compression_joint_pass"] = decision["compression_accuracy_pass"] and decision["compression_speed_pass"]
    return {"counts": {"frames": len(public)//2, "clusters": len(clusters),
        "timed_clusters": len(timed_clusters)}, "accuracy": accuracy, "accuracy_differences": differences,
        "timing": timing, "primary_decisions": decision,
        "interval_scope": "Other 95% intervals descriptive; no selection among secondary results. Seeds averaged within participant, not independent observations.",
        "memory_scope": "Maximum live GPU allocation within measured forwards; includes the same resident base model and all ten adapters for every variant. Not host RAM, model weight reduction, or generation KV-cache memory.",
        "timing_scope": "Warm plan cache, fresh input preparation and lookup every request, correction included. Cold setup reported separately; no generation/KV-cache inference."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--repo", type=Path, required=True)
    freeze.add_argument("--steering-run", type=Path, required=True)
    exposure = sub.add_parser("freeze-exposure")
    exposure.add_argument("--repo", type=Path, required=True)
    exposure.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_config(args.repo, args.steering_run) if args.command == "freeze" else freeze_exposure(args.repo, args.audit)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
