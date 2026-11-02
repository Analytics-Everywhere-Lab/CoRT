"""Verify adapter provenance and optionally export the original checkpoint bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "runs/killarney/role-aware-combined/full/6469516/adapters",
    "runs/killarney/role-aware-transfer/full/6478505/adapters",
    "runs/research/lifecycle_memory/release_bundles/role_aware_combined_2026_10_19_v2/"
    "docs/research/lifecycle_memory/phase 2 research/jobs/decoder_cached_validation/artifacts/adapters",
)
NAME = re.compile(r"([A-Za-z0-9_]+)__correct_L34_(CQ|FQ|readout)\.seed(17|29)\.final\.safetensors")


def inspect(path):
    if path.is_symlink() or path.stat().st_size > 200000:
        raise ValueError("Unexpected checkpoint link or size")
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("Truncated safetensors file")
    size = struct.unpack("<Q", raw[:8])[0]
    if not 2 <= size <= 16384 or 8 + size > len(raw):
        raise ValueError("Invalid safetensors header length")
    header = json.loads(raw[8:8 + size])
    match = NAME.fullmatch(path.name)
    if match is None or set(header) != {"__metadata__", "down.weight", "up.weight"}:
        raise ValueError("Unexpected checkpoint name or tensor inventory")
    tensors = {}
    for name, shape, offsets in (
        ("down.weight", [4, 5376], [0, 86016]),
        ("up.weight", [5376, 4], [86016, 172032]),
    ):
        tensor = header[name]
        if tensor != {"dtype": "F32", "shape": shape, "data_offsets": offsets}:
            raise ValueError("Unexpected adapter tensor geometry")
        tensors[name] = {"dtype": "F32", "shape": shape}
    payload = raw[8 + size:]
    if len(payload) != 172032 or not all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", payload)):
        raise ValueError("Invalid or nonfinite adapter parameters")
    meta = header["__metadata__"]
    compression, positions, seed = match.groups()
    driver_target = "l34_readout" if positions == "readout" and compression in {
        "F0_L30_prune", "Fhalf_Chalf_L34_prune"} else "l34_cq"
    expected = {"steps": "256", "layer_index": "34", "positions": positions,
                "seed": seed, "rank": "4", "selection": "fixed_final_step",
                "compression_id": compression, "canary": "False",
                "target": f"{compression}__correct_L34_{positions}",
                "legacy_driver_target": driver_target}
    if meta != expected:
        raise ValueError("Adapter metadata disagrees with the fixed configuration")
    return {"filename": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "compression": compression, "positions": positions,
            "seed": int(seed), "cut": 34, "rank": 4, "training_steps": 256,
            "selection": "fixed_final_step", "parameters": 43008, "tensors": tensors}


def export(upstream, output, *, include_weights=False, reference=None):
    reference = reference or ROOT / "adapters/manifest.json"
    recorded = json.loads(reference.read_text())["checkpoints"]
    by_digest, by_name, sources = {}, {}, {}
    for relative in SOURCES:
        paths = sorted((upstream / relative).glob("*.safetensors"))
        if not paths:
            raise ValueError("Missing original adapter collection")
        for path in paths:
            entry = inspect(path)
            name, digest = entry["filename"], entry["sha256"]
            if name in by_name and by_name[name] != digest:
                raise ValueError("Conflicting checkpoints share a filename")
            by_name[name] = digest
            by_digest[digest] = entry
            sources[name] = path
    entries = sorted(by_digest.values(), key=lambda entry: entry["filename"])
    if len(entries) != 16 or sum(entry["bytes"] for entry in entries) != 2759024:
        raise ValueError("Checkpoint inventory differs from the recorded study")
    if entries != recorded:
        raise ValueError("Checkpoints differ from the published provenance inventory")
    weights = output.parent / "weights"
    if output.is_symlink() or any(p.is_symlink() for p in output.parents) or weights.is_symlink():
        raise ValueError("Export destination must not contain symlinks")
    if weights.exists():
        if not weights.is_dir() or not include_weights:
            raise ValueError("Existing weights require --include-weights")
        if any(p.name not in sources or p.is_symlink() or not p.is_file() for p in weights.iterdir()):
            raise ValueError("Unexpected existing adapter export")
        for path in weights.iterdir():
            if inspect(path) != by_digest[by_name[path.name]]:
                raise ValueError("Existing exported checkpoint differs from its source")
    result = {"schema_version": 1, "kind": "adapter_provenance_inventory",
              "weights_distributed": include_weights, "checkpoint_count": len(entries),
              "total_checkpoint_bytes": sum(entry["bytes"] for entry in entries),
              "base_model": "google/gemma-4-31B-it",
              "base_revision": "3548789868c5356dbf307c98e6f609007b82b3eb",
              "training_frames": 505, "training_clusters": 88,
              "training_requests": 1010, "checkpoints": entries}
    output.parent.mkdir(parents=True, exist_ok=True)
    if include_weights:
        weights.mkdir(exist_ok=True)
        for entry in entries:
            target = weights / entry["filename"]
            if not target.exists():
                shutil.copyfile(sources[entry["filename"]], target)
            if inspect(target) != entry:
                raise ValueError("Exported checkpoint failed byte verification")
        result["weights_directory"] = "weights"
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "adapters/manifest.json")
    parser.add_argument("--include-weights", action="store_true",
                        help="Copy hash-matched original checkpoints beside the inventory")
    parser.add_argument("--reference", type=Path, default=ROOT / "adapters/manifest.json",
                        help="Existing provenance inventory to match before writing")
    args = parser.parse_args()
    result = export(args.upstream.resolve(), args.output, include_weights=args.include_weights,
                    reference=args.reference)
    mode = "original weights and metadata" if args.include_weights else "metadata only"
    print(f"Verified {result['checkpoint_count']} distinct checkpoints; exported {mode}.")
