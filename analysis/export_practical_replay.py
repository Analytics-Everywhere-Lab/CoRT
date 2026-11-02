"""Curator-only, allowlisted upstream-to-public practical numerical projection.

Run from the release root with ``python -m analysis.export_practical_replay
--upstream /path/to/research``. Upstream is read-only; no model/network activity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

import numpy as np

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.export_aggregates import PRIVATE, IDENTITY_KEY
from analysis.replay_practical import (
    COMPONENTS, MEMORY, METRICS, PRIMARY, ROOT, SECONDARY, SESSIONS,
    canonical_digest, compare, inventory, project_summary, replay, require,
)

RUNS = {
    "6469516": "runs/killarney/role-aware-combined/full/6469516",
    "6478505": "runs/killarney/role-aware-transfer/full/6478505",
    "621156": "runs/tamia/decoder-cross-source-validation/full/621156",
}
DATASET = "data/external/decoder_cross_source_validation_2026_10_15_v1"


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def privacy_check(value):
    """Reject forbidden text/keys, in addition to the projection's allowlists."""
    if isinstance(value, dict):
        for key, item in value.items():
            require(not PRIVATE.search(key) and not IDENTITY_KEY.search(key)
                    and key not in {"request_id", "frame_key", "cluster_key", "path", "source_hashes",
                                    "source_files", "credentials", "token", "api_key"}, "private output field")
            privacy_check(item)
    elif isinstance(value, list):
        for item in value:
            privacy_check(item)
    elif isinstance(value, str):
        require(not PRIVATE.search(value), "private output string")


def project_units(session, public, predictions, timings, labels):
    """Reduce validated paired raw rows without exporting any original identity."""
    from research.lifecycle_memory import role_aware_transfer

    names, _, _ = inventory(session)
    # An isolated original validator supplies the complete pairing/identity checks.
    validator = role_aware_transfer.estimator()
    validator.variant_specs = lambda: [{"id": name} for name in names]
    validator.validate_observations(public, predictions, timings, labels)
    require(len(public) == 894 and len(predictions) == 894 * len(names)
            and len(timings) == 640 * len(names), "raw row counts drift")
    require({r["repeat_id"] for r in timings} == set(range(5)), "timing repeats drift")
    native = {(r["request_id"], r["repeat_id"]): r for r in timings if r["variant"] == "native"}
    for row in timings:
        base = native[(row["request_id"], row["repeat_id"])]
        require(row["device_id"] == base["device_id"] and row["batch_size"] == base["batch_size"] == 1,
                "unpaired hardware timing")
    timed_requests = {r["request_id"] for r in native.values()}
    frames = defaultdict(set)
    for row in public:
        frames[row["frame_key"]].add(row["request_id"])
    require(all(not (ids & timed_requests) or ids <= timed_requests for ids in frames.values()),
            "timed candidate orders are not paired")
    clusters = sorted({r["cluster_key"] for r in public})
    timed_clusters = sorted({r["cluster_key"] for r in timings})
    by_unit = {c: [r for r in public if r["cluster_key"] == c] for c in clusters}
    time_counts = {c: sum(r["cluster_key"] == c for r in native.values()) for c in clusters}
    units = [{"source": by_unit[c][0]["source_family"], "requests": len(by_unit[c]),
              "timing_pairs": time_counts[c]} for c in clusters]
    # The earlier analyzer used statistics.mean; final analyzers used np.mean.
    reducer = mean if session == "621156" else lambda x: float(np.mean(x))
    pcells, tcells = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for row in predictions:
        pcells[row["variant"]][row["cluster_key"]].append(row)
    for row in timings:
        tcells[row["variant"]][row["cluster_key"]].append(row)

    def score(name, bias=None):
        output = {m: [] for m in METRICS}
        for cluster in clusters:
            accuracy, losses = [], []
            for row in pcells[name][cluster]:
                margin = row["margin_ab"] - (bias or 0)
                prediction = row["prediction"] if bias is None else ("A" if margin >= 0 else "B")
                label = labels[row["request_id"]]
                signed = margin * math.log(2) * (1 if label == "A" else -1)
                accuracy.append(float(prediction == label))
                losses.append(max(0., -signed) + math.log1p(math.exp(-abs(signed))))
            output["accuracy"].append(reducer(accuracy))
            output["log_loss_nats"].append(reducer(losses))
        return output

    fields = ["end_to_end_elapsed_ms", *COMPONENTS, *MEMORY]
    if session == "621156":
        fields.append("elapsed_ms")
    variants = {}
    for name in names:
        variants[name] = {"scores": score(name), "timing": {}, "memory_max": {}}
        for field in fields:
            require(all(type(r[field]) in (float, int) and math.isfinite(r[field]) and r[field] >= 0
                        for c in timed_clusters for r in tcells[name][c]), "invalid timing component")
            variants[name]["timing"][field] = [reducer([r[field] for r in tcells[name][c]]) for c in timed_clusters]
        for field in MEMORY:
            variants[name]["memory_max"][field] = [max(r[field] for r in tcells[name][c]) for c in timed_clusters]
    result = {"schema_version": 1, "session": session, "source_status": "pass",
        "bootstrap": {"draws": 10000, "seed": 20261015 if session == "621156" else 20261019,
                      "alpha": .05 / (3 if session == "621156" else 4)},
        "units": units, "timed_positions": [clusters.index(c) for c in timed_clusters],
        "variants": variants, "frozen_bias": {},
        "counts": {"frames": len(frames), "requests": len(public), "clusters": len(clusters),
                   "timed_clusters": len(timed_clusters), "prediction_rows": len(predictions),
                   "timing_rows": len(timings)}}
    if session == "6469516":
        result["frozen_bias"] = {name: score(name, 5.806289363588375) for name in ("native", PRIMARY, SECONDARY)}
    privacy_check(result)
    return result


def load_session(upstream, session):
    root = upstream / RUNS[session]
    rows_root = root if session == "621156" else root / "evaluate"
    predictions, timings = [], []
    require(len(list(rows_root.glob("rank_*"))) == 4, "shard inventory drift")
    for rank in range(4):
        shard = rows_root / f"rank_{rank:03d}"
        receipt = read_json(shard / "worker_receipt.json")
        require(receipt["status"] == "pass", "worker receipt failed")
        for filename, destination in (("predictions.jsonl", predictions), ("timings.jsonl", timings)):
            content = (shard / filename).read_bytes()
            require(hashlib.sha256(content).hexdigest() == receipt["files_sha256"][filename],
                    "worker receipt hash mismatch")
            destination.extend(json.loads(line) for line in content.splitlines() if line.strip())
    public = read_rows(upstream / DATASET / "public/requests.jsonl")
    label_rows = read_rows(upstream / DATASET / "evaluator/labels.jsonl")
    labels = {r["request_id"]: r["correct_label"] for r in label_rows}
    require(len(label_rows) == len(labels) and set(labels) == {r["request_id"] for r in public},
            "label inventory drift")
    summary = read_json(root / "analysis_summary.json")
    require(summary["status"] == "pass", "source summary failed")
    return project_units(session, public, predictions, timings, labels), project_summary(summary)


def export(upstream, destination):
    artifacts = {}
    manifest = {"schema_version": 1, "sessions": {}}
    # Verify every session before writing anything. Expected intervals come only
    # from saved scientific summaries, never from this replay implementation.
    for session in SESSIONS:
        data, expected = load_session(upstream, session)
        actual, _ = replay(data)
        compare(actual, expected)
        privacy_check(expected)
        artifacts[f"{session}.json"] = data
        artifacts[f"{session}_expected.json"] = expected
        manifest["sessions"][session] = {"data_sha256": canonical_digest(data),
                                         "expected_sha256": canonical_digest(expected)}
    artifacts["manifest.json"] = manifest
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        (destination / name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/replay_practical")
    args = parser.parse_args()
    export(args.upstream.resolve(), args.output_dir)
    print("Verified and exported three anonymous practical sessions.")


if __name__ == "__main__":
    main()
