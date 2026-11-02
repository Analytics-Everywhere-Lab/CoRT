"""Offline STA metadata capacity, not audited episodes or dataset readiness.

Run with python -B -m research.lifecycle_memory.temporal_forecasting_census --out NEW_DIR.
Only stdlib is used. Inputs are capped, hashed before counting, and rechecked on
parse; at most one annotations split is decoded at a time. No media is opened.
Fresh means unexposed in the supplied registry, not independently verified fresh.
Every qualifying row is indexed; no event, history, image, label or allocation is
chosen. main_uid groups and wearers are deduplicated, including across splits;
distinct main_uid groups are NOT verified independent action onsets.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

MIB = 1024 * 1024
MAX_FILE = 128 * MIB
MAX_ROWS = 250_000
BASE = Path("data/external/temporal_forecasting")
STAGES = ("raw", "mapped", "unknown_wearer", "excluded", "fresh",
          "secondary_screen", "fresh_secondary_screen")
REASONS = ("unknown_video", "null_wearer", "exposed_participant", "exposed_video",
           "not_single_object", "invalid_ttc", "invalid_box", "invalid_clock",
           "insufficient_nominal_history")


def require(condition, code):
    if not condition:
        raise ValueError(code)


def identifier(value):
    require(isinstance(value, str) and bool(value) and value == value.strip(), "InvalidID")
    return value


def participant(value):
    if value is None:
        return None
    if type(value) in (int, float):
        require(math.isfinite(value) and value >= 0 and value == int(value)
                and (type(value) is int or value <= 2**53), "InvalidParticipant")
        return str(int(value))
    require(isinstance(value, str), "InvalidParticipant")
    value = value.strip().removeprefix("ego4d_participant:").strip()
    if value.lower() in ("", "null", "none"):
        return None
    require(re.fullmatch(r"[0-9]+(?:\.0+)?", value), "InvalidParticipant")
    return str(int(value.split(".")[0]))


def finite_float(text):
    value = float(text)
    require(math.isfinite(value), "NonfiniteField")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "DuplicateJSONKey")
        result[key] = value
    return result


def read_input(path, limit, *, decode=False, expected=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit, "InvalidInputSize")
        digest, size, chunks = hashlib.sha256(), 0, []
        while chunk := stream.read(min(MIB, limit + 1 - size)):
            size += len(chunk)
            require(size <= limit, "InputTooLarge")
            digest.update(chunk)
            if decode:
                chunks.append(chunk)
        record = {"bytes": size, "sha256": digest.hexdigest()}
        require(size == info.st_size, "InputChanged")
        require(expected is None or record == expected, "InputChanged")
    value = None
    if decode:
        try:
            value = json.loads(b"".join(chunks), object_pairs_hook=unique_object,
                               parse_float=finite_float, parse_constant=finite_float)
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("InvalidJSON") from exc
        require(isinstance(value, dict), "InvalidSchema")
    return value, record


def identities(payload, registry):
    mapping = {}
    require(isinstance(payload.get("videos"), list) and payload["videos"], "MissingMetadata")
    for row in payload["videos"]:
        require(isinstance(row, dict) and "fb_participant_id" in row, "MissingMetadata")
        video = identifier(row.get("video_uid"))
        require(video not in mapping, "DuplicateVideoUID")
        mapping[video] = participant(row["fb_participant_id"])
    require(type(registry.get("version")) is int and registry["version"] == 1,
            "InvalidRegistry")
    for field in ("participants", "videos", "asset_sha256"):
        values = registry.get(field, [] if field == "asset_sha256" else None)
        require(isinstance(values, list), "InvalidRegistry")
        require(len({identifier(v) for v in values}) == len(values), "DuplicateRegistryID")
        if field == "asset_sha256":
            require(all(re.fullmatch(r"[0-9a-f]{64}", v) for v in values), "InvalidRegistry")
    people = {participant(v) for v in registry["participants"]}
    require(None not in people, "InvalidRegistry")
    videos = {v.removeprefix("ego4d:") for v in registry["videos"]}
    # An exposed recording also excludes its mapped wearer's other recordings.
    people.update(mapping[v] for v in videos if mapping.get(v) is not None)
    return mapping, people, videos


class Counts:
    def __init__(self):
        self.stages = {s: [0, set(), set(), set()] for s in STAGES}
        self.reasons = Counter(dict.fromkeys(REASONS, 0))
        self.identities, self.action_mismatches, self.action_missing = {}, set(), set()

    def add(self, stages, reasons, row, person):
        group_id = row["main_uid"]
        signature = tuple(row.get(k) for k in ("video_uid", "clip_uid", "action_start_frame", "action_end_frame"))
        previous = self.identities.setdefault(group_id, signature)
        require(previous[:2] == signature[:2], "ConflictingGroupVideoOrClip")
        if None in signature[2:]:
            self.action_missing.add(group_id)
        if previous[2:] != signature[2:]:
            self.action_mismatches.add(group_id)
        for stage in stages:
            group = self.stages[stage]
            group[0] += 1
            group[1].add(row["main_uid"])
            group[2].add(row["video_uid"])
            if person is not None:
                group[3].add(person)
        self.reasons.update(reasons)

    def public(self):
        return {"stages": {s: dict(zip(("rows", "unique_main_uid_groups", "videos", "mapped_wearers"),
                                     [v[0], *map(len, v[1:])]))
                           for s, v in self.stages.items()},
                "overlapping_attrition_rows": dict(self.reasons),
                "group_identity_checks": {"video_consistent": True, "clip_consistent": True,
                    "action_frame_identity_consistent": not (self.action_mismatches or self.action_missing),
                    "groups_with_inconsistent_action_frame_identity": len(self.action_mismatches),
                    "groups_missing_action_frame_identity": len(self.action_missing),
                    "independent_onsets_verified": False}}


def screen(row, meta):
    objects, fps = row["objects"], meta["fps"]
    reasons = []
    if len(objects) != 1:
        reasons.append("not_single_object")
    else:
        obj = objects[0]
        ttc, box = obj.get("time_to_contact"), obj.get("box")
        if type(ttc) not in (int, float) or ttc <= 0:
            reasons.append("invalid_ttc")
        if not (isinstance(box, list) and len(box) == 4
                and all(type(v) in (int, float) for v in box)
                and 0 <= box[0] < box[2] <= meta["frame_width"]
                and 0 <= box[1] < box[3] <= meta["frame_height"]):
            reasons.append("invalid_box")
    if fps <= 0:
        reasons.append("invalid_clock")
    elif not all(math.isfinite(row[k] / fps) for k in ("frame", "clip_frame")):
        reasons.append("invalid_clock")
    elif min(row["frame"], row["clip_frame"]) / fps < 2:
        reasons.append("insufficient_nominal_history")
    return reasons


def count_split(path, split, pin, mapping, people, videos, seen, combined, index):
    payload, _ = read_input(path, MAX_FILE, decode=True, expected=pin)
    info, rows = payload.get("info"), payload.get("annotations")
    require(isinstance(info, dict) and info.get("version") == "2.0"
            and info.get("include_annotations") is True, "InvalidAnnotationInfo")
    require(info.get("split") == split, "SplitMismatch")
    metadata = info.get("video_metadata")
    require(isinstance(metadata, dict) and metadata, "MissingMetadata")
    require(isinstance(rows, list) and 0 < len(rows) <= MAX_ROWS, "InvalidAnnotations")
    for meta in metadata.values():
        require(isinstance(meta, dict) and type(meta.get("fps")) in (int, float)
                and all(type(meta.get(k)) is int and meta[k] > 0
                        for k in ("frame_width", "frame_height")), "MissingMetadata")
    counts = Counts()
    for row in rows:
        require(isinstance(row, dict), "InvalidAnnotation")
        for key in ("uid", "main_uid", "video_uid", "clip_uid"):
            identifier(row.get(key))
        require(row["uid"] not in seen, "DuplicateUID")
        seen.add(row["uid"])
        video = row["video_uid"]
        require(video in metadata, "MissingMetadata")
        require(all(type(row.get(k)) is int for k in ("frame", "clip_frame")), "InvalidFrame")
        for key, value in row.items():
            if key.endswith(("_frame", "_sec", "_seconds")):
                require(type(value) is int if key.endswith("_frame")
                        else type(value) in (int, float), "InvalidTiming")
        require(isinstance(row.get("objects"), list)
                and all(isinstance(o, dict) for o in row["objects"]), "InvalidObjects")
        person = mapping.get(video)
        reasons = screen(row, metadata[video])
        stages = ["raw", "mapped" if person is not None else "unknown_wearer"]
        if person is None:
            reasons.append("null_wearer" if video in mapping else "unknown_video")
        exposed = person in people or video in videos
        fresh = person is not None and not exposed
        if exposed:
            stages.append("excluded")
        if fresh:
            stages.append("fresh")
        screened = not any(r in reasons for r in REASONS[4:])
        if screened:
            stages.append("secondary_screen")
        if person in people:
            reasons.append("exposed_participant")
        if video in videos:
            reasons.append("exposed_video")
        if fresh and screened:
            stages.append("fresh_secondary_screen")
            candidate = {k: row[k] for k in ("uid", "main_uid", "video_uid", "clip_uid", "frame", "clip_frame")}
            candidate.update(split=split, participant_id=person, fps=metadata[video]["fps"],
                             current_seconds_nominal=row["frame"] / metadata[video]["fps"],
                             clip_seconds_nominal=row["clip_frame"] / metadata[video]["fps"])
            index.write(json.dumps(candidate, sort_keys=True, allow_nan=False) + "\n")
        for target in (counts, combined):
            target.add(stages, reasons, row, person)
    return counts.public()


def census(source_dir, receipt_path, metadata_path, registry_path, out):
    """Write a new output directory; return public aggregates, never source IDs."""
    paths = {s: (Path(source_dir) / f"fho_sta_{s}.json", MAX_FILE) for s in ("train", "val")}
    paths.update(receipt=(receipt_path, MIB), metadata=(metadata_path, 96 * MIB),
                 registry=(registry_path, 8 * MIB))
    pins = {role: read_input(p, limit)[1] for role, (p, limit) in paths.items()}
    def load(role):
        return read_input(*paths[role], decode=True, expected=pins[role])[0]
    receipt = load("receipt")
    require(receipt.get("status") == "source_metadata_validated"
            and isinstance(receipt.get("files"), dict), "InvalidReceipt")
    require(set(receipt["files"]) == {f"fho_sta_{s}.json" for s in ("train", "val")}, "InvalidReceipt")
    for split in ("train", "val"):
        record = receipt["files"][f"fho_sta_{split}.json"]
        require(isinstance(record, dict) and type(record.get("size")) is int
                and record["size"] == pins[split]["bytes"]
                and record.get("sha256") == pins[split]["sha256"], "ReceiptMismatch")
    mapping, people, videos = identities(load("metadata"), load("registry"))
    out = Path(out).absolute()
    require(not any(p.is_symlink() for p in (out, *out.parents)), "UnsafeOutput")
    out.mkdir(mode=0o755, exist_ok=False)
    hidden = out / "hidden"
    candidate_path, summary_path = hidden / "candidateindex.jsonl", out / "summary.json"
    try:
        hidden.mkdir(mode=0o700)
        combined, seen, splits = Counts(), set(), {}
        with os.fdopen(os.open(candidate_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as index:
            for split in ("train", "val"):
                splits[split] = count_split(paths[split][0], split, pins[split], mapping,
                                           people, videos, seen, combined, index)
        counts = combined.public()
        capacity = counts["stages"]["fresh_secondary_screen"]["mapped_wearers"]
        summary = {"status": "metadata_capacity_only", "sources": pins, "splits": splits,
                   "combined": counts, "capacity_count": capacity,
                   "gate46": {"pass": capacity >= 46, "scope": "metadata_capacity", "threshold": 46},
                   "dataset_ready": False, "audited": False, "image_qa": False,
                   "timestamp_verified": False, "nominal_history_seconds": 2,
                   "allocated": False}
        with summary_path.open("x") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return summary
    except BaseException:
        candidate_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        hidden.rmdir()
        out.rmdir()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=BASE / "source/v2")
    parser.add_argument("--receipt", type=Path, default=BASE / "preparation/annotation_acquisition.json")
    parser.add_argument("--metadata", type=Path, default=Path("data/external/egohos/licensed_metadata/v2/ego4d.json"))
    parser.add_argument("--registry", type=Path, default=BASE / "preparation/hidden/exposure_registry.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = census(args.source_dir, args.receipt, args.metadata, args.registry, args.out)
    except (ValueError, OSError, OverflowError, RecursionError):
        print(json.dumps({"status": "error", "error_code": "CensusValidationFailed"}))
        return 1
    print(json.dumps({k: result[k] for k in ("status", "capacity_count", "gate46", "dataset_ready")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
