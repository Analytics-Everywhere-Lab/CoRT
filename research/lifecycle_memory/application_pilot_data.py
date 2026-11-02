"""Prepare real-only, label-separated application pilot data without loading images.

API: ``prepare_data(battery_root, seed=20261014, expected_frames=800)`` returns
PreparedPilotData; ``export_data(data, output_dir)`` writes an immutable package.
``build_dataset(public_rows, hidden_rows, ...)`` supports small in-memory fixtures.

CLI::

    python -m research.lifecycle_memory.application_pilot_data \
        --battery-root BATTERY --output-dir OUTPUT

Public schema v1: request_id, item_id, frame_key, cluster_key (opaque camera
wearer), split_group_key (connected leakage component), split, order_id, arm,
system_prompt, user_prompt, inputs. Input paths remain relative to BATTERY.
Labels contain request_id, frame_key, cluster_key, split_group_key, split,
order_id, correct_label. Never pass labels to inference workers. Training may
mount labels/train.jsonl, selection may mount labels/validation.jsonl; neither
may mount evaluator/. Compression uses ALL public rows, steering fits on train,
selects on validation, and evaluates ALL test rows, including baseline errors.

No wrong-time/blank export is supported. Declared image hashes are used for
duplicate grouping but image bytes are NOT scanned or verified by this module.
Export directories are logical separation, not an OS permission/security boundary.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
DEFAULT_SEED = 20261014
SPLITS = ("train", "validation", "test")
RATIOS = (0.6, 0.2, 0.2)
INPUT_KINDS = ("full_frame", "reference_crop_a", "reference_crop_b")


def read_json_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL or the battery's JSON requests/answers envelope."""
    path = Path(path)
    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        values = json.loads(path.read_text())
        if isinstance(values, dict):
            keys = [key for key in ("requests", "answers") if key in values]
            if len(keys) != 1:
                raise ValueError("JSON envelope must have exactly one requests/answers key")
            values = values[keys[0]]
    if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
        raise ValueError(f"expected object rows: {path}")
    return values


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"missing nonempty text field: {keys}")


def _asset(asset: Mapping[str, Any]) -> dict[str, Any]:
    path = _text(asset, "path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "\\" in path:
        raise ValueError("asset path must be relative to battery root without traversal")
    sha = _text(asset, "sha256").lower()
    if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("asset sha256 must be a hex digest")
    if any(type(asset.get(key)) is not int or asset[key] <= 0 for key in ("width", "height")):
        raise ValueError("asset dimensions must be positive integers")
    return {"kind": asset["kind"], "path": str(pure), "sha256": sha,
            "width": asset["width"], "height": asset["height"]}


class _Union:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def root(self, key: str) -> str:
        self.parent.setdefault(key, key)
        while key != self.parent[key]:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, left: str, right: str) -> None:
        a, b = sorted((self.root(left), self.root(right)))
        self.parent[b] = a


@dataclass
class PreparedPilotData:
    public_rows: list[dict[str, Any]]
    labels: list[dict[str, Any]]
    manifest: dict[str, Any]

    def rows_for(self, task_id: str, split: str | None = None) -> list[dict[str, Any]]:
        """Compression defaults to all rows; steering defaults to held-out test."""
        if task_id not in {"compression", "steering"}:
            raise ValueError("task_id must be compression or steering")
        if task_id == "compression" and split is not None:
            raise ValueError("compression evaluation must use all splits")
        if task_id == "steering":
            split = split or "test"
            if split not in SPLITS:
                raise ValueError("unknown split")
        return [dict(row) for row in self.public_rows if split is None or row["split"] == split]


def build_dataset(
    public_rows: Sequence[Mapping[str, Any]],
    hidden_rows: Sequence[Mapping[str, Any]],
    *, seed: int = DEFAULT_SEED, expected_frames: int | None = 800,
    input_hashes: Mapping[str, Any] | None = None, max_split_deviation: float = 0.10,
) -> PreparedPilotData:
    """Join behavioral-request schemas and split connected identities, never outcomes.

    Whole connected components are allocated toward 60/20/20 PARTICIPANT counts.
    Components combine participant/cluster, source video/frame/image, candidate,
    selected-component fingerprint, and real/crop asset path/hash identities.
    Donors and the globally shared blank image are intentionally absent because
    they are not used. Fewer than three independent components is a hard failure.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not 0 <= max_split_deviation <= 0.2:
        raise ValueError("max_split_deviation must be between 0 and 0.2")
    hidden: dict[str, Mapping[str, Any]] = {}
    for row in hidden_rows:
        item = _text(row, "item_id")
        if item in hidden:
            raise ValueError(f"duplicate hidden item_id: {item}")
        hidden[item] = row
    union = _Union()
    owners: dict[tuple[str, str], str] = {}
    normalized, labels = [], []
    seen_requests, seen_items = set(), set()
    frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path_hashes: dict[str, str] = {}
    participant_clusters: dict[str, str] = {}
    cluster_participants: dict[str, str] = {}
    for source in public_rows:
        if source.get("arm") != "real":
            raise ValueError("only real-arm rows may enter application pilots")
        item, request = _text(source, "item_id"), _text(source, "request_id")
        if item in seen_items or request in seen_requests:
            raise ValueError("duplicate public item/request ID")
        seen_items.add(item)
        seen_requests.add(request)
        if item not in hidden:
            raise ValueError(f"missing hidden item: {item}")
        h = hidden[item]
        if h.get("request_ids", {}).get("real") != request:
            raise ValueError(f"public/hidden request identity mismatch: {item}")
        if h.get("scoring_eligible") is False:
            raise ValueError("ineligible source row; do not silently filter rows")
        label = h.get("answer", h.get("correct_choice"))
        order = h.get("counterbalance_order_index")
        if label not in {"A", "B"} or type(order) is not int or order not in (0, 1):
            raise ValueError("invalid answer or counterbalance order")
        participant = _text(h, "participant_id", "ego4d_participant_id")
        cluster = _text(h, "cluster_id", "participant_cluster_id")
        if participant_clusters.setdefault(participant, cluster) != cluster:
            raise ValueError("participant has inconsistent cluster identity")
        if cluster_participants.setdefault(cluster, participant) != participant:
            raise ValueError("cluster represents multiple camera wearers")
        cluster_key = "participant_" + _digest(participant)[:24]
        frame = _text(h, "frame_level_unit_id", "frame_unit_id")
        union.root(cluster_key)
        assets = source.get("inputs")
        if not isinstance(assets, list) or len(assets) != 3 or tuple(
            asset.get("kind") for asset in assets
        ) != INPUT_KINDS:
            raise ValueError("inputs must be full frame, crop A, crop B in that order")
        assets = [_asset(asset) for asset in assets]
        # Only observed real-input identities participate, never unused donors.
        identities = [("participant", participant), ("cluster", cluster), ("frame_unit", frame)]
        for field, namespace in (("source_video_id", "video"), ("frame_id", "frame"),
                                 ("source_image_relpath", "source_image")):
            if h.get(field):
                identities.append((namespace, str(h[field])))
        _text(h, "source_video_id")
        _text(h, "frame_id")
        for field in ("first_order_candidate_id", "second_order_candidate_id",
                      "crop_a_candidate_id", "crop_b_candidate_id"):
            if h.get(field):
                identities.append(("candidate", str(h[field])))
        for field in ("crop_a_selected_component_fingerprint", "crop_b_selected_component_fingerprint"):
            if h.get(field):
                identities.append(("component", str(h[field])))
        for index, asset in enumerate(assets):
            identities.extend((("asset_hash", asset["sha256"]), ("asset_path", asset["path"])))
            if path_hashes.setdefault(asset["path"], asset["sha256"]) != asset["sha256"]:
                raise ValueError("same asset path has conflicting hashes")
            hidden_asset = h.get("assets", {}).get(("real_full", "crop_a", "crop_b")[index])
            if hidden_asset and any(hidden_asset.get(key) != asset[key] for key in ("path", "sha256", "width", "height")):
                raise ValueError("public/hidden asset mismatch")
        for identity in identities:
            owner = owners.setdefault(identity, cluster_key)
            union.join(cluster_key, owner)
        row = {"schema_version": SCHEMA_VERSION, "request_id": request, "item_id": item,
               "frame_key": frame, "frame_unit_id": frame,
               "cluster_key": cluster_key, "order_id": f"order_{order}",
               "arm": "real", "system_prompt": _text(source, "system_prompt"),
               "user_prompt": _text(source, "user_prompt"), "inputs": assets,
               "arms": {"real": {"inputs": assets}}}
        normalized.append(row)
        labels.append({key: row[key] for key in ("request_id", "frame_key", "cluster_key", "order_id")}
                      | {"correct_label": label})
        frames[frame].append(row)
    if seen_items != set(hidden):
        raise ValueError("public and hidden item inventories differ")
    if not frames or (expected_frames is not None and len(frames) != expected_frames):
        raise ValueError(f"expected {expected_frames} frames, observed {len(frames)}")
    by_request = {row["request_id"]: row for row in labels}
    for frame, pair in frames.items():
        if len(pair) != 2 or {row["order_id"] for row in pair} != {"order_0", "order_1"}:
            raise ValueError(f"frame does not contain both orders: {frame}")
        if len({row["cluster_key"] for row in pair}) != 1:
            raise ValueError("frame pair has inconsistent participant")
        if {by_request[row["request_id"]]["correct_label"] for row in pair} != {"A", "B"}:
            raise ValueError("counterbalanced correct labels must flip")
        left, right = pair
        left["counterbalanced_request_id"] = right["request_id"]
        right["counterbalanced_request_id"] = left["request_id"]
        if any(left[key] != right[key] for key in ("system_prompt", "user_prompt")):
            raise ValueError("prompts differ within counterbalance pair")
        a, b = left["inputs"], right["inputs"]
        if a[0] != b[0] or any(a[i][key] != b[j][key] for i, j in ((1, 2), (2, 1))
                               for key in ("path", "sha256", "width", "height")):
            raise ValueError("counterbalance pair is not an exact crop swap")
    groups: dict[str, list[str]] = defaultdict(list)
    for cluster_key in sorted(union.parent):
        groups[union.root(cluster_key)].append(cluster_key)
    if len(groups) < 3:
        raise ValueError("need at least three independent identity components for three splits")
    components = {"group_" + _digest(members)[:24]: members for members in groups.values()}
    # Largest-first allocation minimizes deviations caused by linked participants.
    ranked = sorted(components, key=lambda key: (-len(components[key]), _digest([seed, key]), key))
    total = len(union.parent)
    targets = [ratio * total for ratio in RATIOS]
    counts = [0, 0, 0]
    assignment, group_for = {}, {}
    for index, group in enumerate(ranked):
        empty = [i for i, count in enumerate(counts) if count == 0]
        choices = empty if len(ranked) - index == len(empty) else list(range(3))
        weight = len(components[group])
        chosen = min(choices, key=lambda i: (sum((counts[j] + (weight if j == i else 0) - targets[j]) ** 2
                                                for j in range(3)), i))
        counts[chosen] += weight
        for cluster_key in components[group]:
            assignment[cluster_key] = SPLITS[chosen]
            group_for[cluster_key] = group
    deviation_limit = max(max_split_deviation, 1 / total)
    if any(abs(count / total - ratio) > deviation_limit + 1e-12 for count, ratio in zip(counts, RATIOS)):
        raise ValueError("identity overlap prevents a close 60/20/20 split; refusing imbalanced partitions")
    for row in normalized + labels:
        row["split"] = assignment[row["cluster_key"]]
        row["split_group_key"] = group_for[row["cluster_key"]]
    normalized.sort(key=lambda row: (row["frame_key"], row["order_id"], row["request_id"]))
    labels.sort(key=lambda row: (row["frame_key"], row["order_id"], row["request_id"]))
    counts_by_split = {}
    for split in SPLITS:
        rows = [row for row in normalized if row["split"] == split]
        counts_by_split[split] = {"rows": len(rows), "frames": len({r["frame_key"] for r in rows}),
                                  "participants": len({r["cluster_key"] for r in rows}),
                                  "identity_components": len({r["split_group_key"] for r in rows})}
    manifest = {
        "schema_version": SCHEMA_VERSION, "seed": seed, "arm": "real",
        "source_cohort": "reused 800-frame reanchored application pool; not untouched confirmation",
        "input_hashes": dict(input_hashes or {"public_rows_sha256": _digest(list(public_rows)),
                                             "hidden_rows_sha256": _digest(list(hidden_rows))}),
        "input_counts": {"public_rows": len(public_rows), "hidden_rows": len(hidden_rows)},
        "counts": {"rows": len(normalized), "frames": len(frames), "participants": total,
                   "identity_components": len(groups), "unique_asset_paths": len(path_hashes),
                   "unique_asset_hashes": len(set(path_hashes.values()))},
        "counts_by_split": counts_by_split,
        "split_policy": {"target_participant_ratios": dict(zip(SPLITS, RATIOS)),
                         "algorithm": "largest_component_first_squared_count_deficit_seeded_hash_ties",
                         "maximum_absolute_ratio_deviation": deviation_limit,
                         "outcome_blind": True, "wrong_time_and_blank_excluded": True},
        "asset_audit": {"byte_hashes_verified": False,
                        "scope": "declared real/crop hashes and identities only; no image scan",
                        "inventory_sha256": _digest(path_hashes)},
        "evaluation": {"compression": "all rows, both orders", "steering": "all test rows, both orders",
                       "baseline_error_filtering": False, "bootstrap_unit": "cluster_key (participant)"},
        "assignments": [{key: row[key] for key in ("request_id", "frame_key", "cluster_key",
                                                   "split_group_key", "split", "order_id")}
                        for row in normalized],
    }
    return PreparedPilotData(normalized, labels, manifest)


def prepare_data(battery_root: str | Path, *, seed: int = DEFAULT_SEED,
                 expected_frames: int | None = 800, max_split_deviation: float = 0.10) -> PreparedPilotData:
    """Read only two metadata row files plus optional manifest hashes; no asset scan."""
    root = Path(battery_root)
    public = root / "public/real_request_pack.jsonl"
    hidden = root / "hidden/evaluator_index.jsonl"
    hashes = {str(path.relative_to(root)): {"sha256": _file_digest(path), "bytes": path.stat().st_size}
              for path in (public, hidden)}
    for name in ("manifest.json", "hash_manifest.json"):
        path = root / name
        if path.exists():
            hashes[name] = {"sha256": _file_digest(path), "bytes": path.stat().st_size}
    return build_dataset(read_json_rows(public), read_json_rows(hidden), seed=seed,
                         expected_frames=expected_frames, input_hashes=hashes,
                         max_split_deviation=max_split_deviation)


def export_data(data: PreparedPilotData, output_dir: str | Path) -> dict[str, Any]:
    """Write label-separated files to a NEW directory; refuse overwrite/resplitting."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    inventory: dict[str, Any] = {}

    def write(name: str, rows: list[dict[str, Any]]) -> None:
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
        inventory[name] = {"sha256": _file_digest(path), "rows": len(rows)}

    write("public/requests.jsonl", data.public_rows)
    write("evaluator/all_labels.jsonl", data.labels)
    for split in SPLITS:
        write(f"public/{split}.jsonl", [r for r in data.public_rows if r["split"] == split])
        label_file = "evaluator/test_labels.jsonl" if split == "test" else f"labels/{split}.jsonl"
        write(label_file, [r for r in data.labels if r["split"] == split])
    manifest = dict(data.manifest, exports=inventory)
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-frames", type=int, default=800)
    parser.add_argument("--max-split-deviation", type=float, default=0.10)
    args = parser.parse_args(argv)
    data = prepare_data(args.battery_root, seed=args.seed, expected_frames=args.expected_frames,
                        max_split_deviation=args.max_split_deviation)
    manifest = export_data(data, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "counts": manifest["counts"],
                      "counts_by_split": manifest["counts_by_split"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
