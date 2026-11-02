"""Bounded offline source-identity overlap; never episode or dataset validation.

No media, network, labels, exposure registry, selection, or model execution.
All frame equality, offsets, and action containment are conditional on unverified
EgoHOS frame numbering in the STA video clock. Shared video or participant identity
does not establish event correspondence.
Run with python -B -m research.lifecycle_memory.temporal_forecasting_static_overlap.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import stat

from research.lifecycle_memory import temporal_forecasting_census as c

BASE = c.BASE
STATIC_BASE = Path("runs/research/lifecycle_memory/phase_2")
DEFAULT_800 = STATIC_BASE / "reanchored_egohos_battery_v3_2026_07_21/hidden/evaluator_index.jsonl"
DEFAULT_320 = STATIC_BASE / "reanchored_answer_token_battery_v3_2026_07_22/hidden/evaluator_index.jsonl"
DEFAULT_OUT = BASE / "preparation/static_overlap_2026_10_23"
DEFAULT_EXPLORATION = DEFAULT_320.parent.parent / "mechanism/hidden/mechanism_evaluator_index.jsonl"
DEFAULT_CONFIRMATION_INDEX = DEFAULT_800.parent.parent / "mechanism/hidden/mechanism_evaluator_index.jsonl"
DEFAULT_CONFIRMATION_SELECTION = Path("docs/research/lifecycle_memory/phase 2 research/jobs/patchaudit_jobE_heldout_component_closure/heldout_selection.json")
STATIC_LIMIT = 16 * c.MIB
STATIC_ROWS = 10_000
THRESHOLDS = (1, 2, 5, 10)
CONDITION = "Conditional on unverified EgoHOS frame numbering in the STA video clock; not episode identity."


def static_frames(path, pin, parents=None):
    """Deduplicate exactly two trials per source; ignore all donor/answer fields."""
    frames, counts, units = {}, Counter(), {}
    digest, size, trial_count = hashlib.sha256(), 0, 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        c.require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= STATIC_LIMIT, "InvalidInputSize")
        while raw := stream.readline(c.MIB + 1):
            size += len(raw)
            c.require(len(raw) <= c.MIB and size <= STATIC_LIMIT, "InputTooLarge")
            digest.update(raw)
            trial_count += 1
            c.require(trial_count <= STATIC_ROWS, "TooManyStaticRows")
            try:
                row = json.loads(raw, object_pairs_hook=c.unique_object,
                                 parse_float=c.finite_float, parse_constant=c.finite_float)
            except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                raise ValueError("InvalidStaticJSON") from exc
            c.require(isinstance(row, dict), "InvalidStaticRow")
            mechanism_id = None
            if parents is not None:
                parent = parents.get(c.identifier(row.get("frame_id")))
                c.require(parent is not None, "MechanismSourceMissing")
                mechanism_id = c.identifier(row.get("frame_unit_id"))
                c.require(c.identifier(row.get("source_frame_level_unit_id")) == parent.get("frame_level_unit_id"),
                          "MechanismSourceUnitMismatch")
                c.require("ego4d_participant_id" in row and "participant_cluster_id" in row,
                          "MissingMechanismPerson")
                row = {**row, "participant_id": row["ego4d_participant_id"],
                       "cluster_id": row["participant_cluster_id"],
                       "source_image_relpath": parent["source_image_relpath"],
                       "frame_level_unit_id": parent["frame_level_unit_id"]}
            c.require({"frame_id", "source_video_id", "source_frame_unit", "participant_id",
                       "cluster_id", "source_image_relpath"} <= row.keys(), "MissingStaticIdentity")
            frame = c.identifier(row["frame_id"])
            video = c.identifier(row["source_video_id"]).removeprefix("ego4d:")
            c.identifier(video)
            unit = row["source_frame_unit"]
            c.require(type(unit) is int and 0 <= unit <= 2**53, "InvalidStaticFrame")
            person = c.participant(row["participant_id"])
            cluster = row["cluster_id"]
            c.require(cluster is None or (isinstance(cluster, str)
                      and cluster.startswith("ego4d_participant:")), "InvalidStaticNamespace")
            c.require(c.participant(cluster) == person, "StaticPersonMismatch")
            image = c.identifier(row["source_image_relpath"])
            value = dict(frame_id=frame, source_video_id=video, source_frame_unit=unit,
                         participant_id=person, source_image_relpath=image)
            if "frame_level_unit_id" in row:
                value["frame_level_unit_id"] = c.identifier(row["frame_level_unit_id"])
            if parents is not None:
                c.require(value == parents[frame], "MechanismSourceIdentityMismatch")
                value["mechanism_frame_unit_id"] = mechanism_id
            c.require(frames.setdefault(frame, value) == value, "UnstableStaticIdentity")
            c.require(units.setdefault((video, unit), frame) == frame, "AliasedStaticFrame")
            counts[frame] += 1
        c.require({"bytes": size, "sha256": digest.hexdigest()} == pin, "InputChanged")
    c.require(bool(frames) and set(counts.values()) == {2}, "ExpectedTwoTrialsPerFrame")
    return list(frames.values())


def sta_index(paths, pins, mapping):
    """Decode one split at a time; retain only identity, timing, and screen flags."""
    by_video, seen, groups, clocks = defaultdict(list), set(), {}, {}
    inventory = Counter(dict.fromkeys(("rows", "unknown_video_rows", "null_wearer_rows"), 0))
    for split in ("train", "val"):
        payload = c.read_input(*paths[split], decode=True, expected=pins[split])[0]
        info, rows = payload.get("info"), payload.get("annotations")
        c.require(isinstance(info, dict) and info.get("version") == "2.0"
                  and info.get("split") == split and info.get("include_annotations") is True,
                  "InvalidAnnotationInfo")
        metadata = info.get("video_metadata")
        c.require(isinstance(metadata, dict) and metadata, "MissingMetadata")
        c.require(isinstance(rows, list) and 0 < len(rows) <= c.MAX_ROWS, "InvalidAnnotations")
        for row in rows:
            c.require(isinstance(row, dict), "InvalidAnnotation")
            uid, group, video, clip = (c.identifier(row.get(k)) for k in
                                      ("uid", "main_uid", "video_uid", "clip_uid"))
            c.require(uid not in seen, "DuplicateUID")
            seen.add(uid)
            c.require(groups.setdefault(group, (video, clip)) == (video, clip), "ConflictingGroupIdentity")
            meta = metadata.get(video)
            c.require(isinstance(meta, dict) and type(meta.get("fps")) in (int, float)
                      and math.isfinite(meta["fps"]) and meta["fps"] >= 0
                      and all(type(meta.get(k)) is int and meta[k] > 0
                              for k in ("frame_width", "frame_height")), "InvalidVideoMetadata")
            fps = meta["fps"]
            c.require(clocks.setdefault(video, fps) == fps, "ConflictingVideoClock")
            c.require(all(type(row.get(k)) is int and 0 <= row[k] <= 2**53
                          for k in ("frame", "clip_frame")), "InvalidSTAFrame")
            start, end = row.get("action_start_frame"), row.get("action_end_frame")
            c.require(all(v is None or (type(v) is int and 0 <= v <= 2**53)
                          for v in (start, end)), "InvalidActionFrame")
            c.require(start is None or end is None or start <= end, "InvalidActionInterval")
            c.require(isinstance(row.get("objects"), list)
                      and all(isinstance(o, dict) for o in row["objects"]), "InvalidObjects")
            by_video[video].append((row["frame"], group, start, end, not c.screen(row, meta)))
            inventory["rows"] += 1
            inventory["unknown_video_rows"] += video not in mapping
            inventory["null_wearer_rows"] += video in mapping and mapping[video] is None
        del payload, rows
    inventory.update(unique_main_uid_groups=len(groups), videos=len(by_video),
                     official_wearers=len({mapping[v] for v in by_video if mapping.get(v) is not None}))
    return by_video, clocks, dict(inventory)


def aggregate(frames, mapping, by_video, clocks):
    joins = Counter(dict.fromkeys(("video_found", "video_missing", "official_wearer_null",
                                   "static_wearer_unknown", "wearer_identity_verified"), 0))
    people_videos = defaultdict(set)
    for video in by_video:
        if mapping.get(video) is not None:
            people_videos[mapping[video]].add(video)
    shared = {r["source_video_id"] for r in frames} & by_video.keys()
    rows = [r for video in shared for r in by_video[video]]
    proximity = {"condition": CONDITION, "timestamp_verified": False, "exact_raw_frame_index_frames": 0,
                 "nominal_offset_eligible_frames": 0, "same_video_invalid_clock_frames": 0,
                 "nearest_sta_row_within_seconds_frames": {str(t): 0 for t in THRESHOLDS},
                 "within_any_action_start_end_frames_inclusive": 0,
                 "action_interval_eligible_frames": 0,
                 "sta_rows_missing_action_interval_on_shared_videos": sum(r[2] is None or r[3] is None for r in rows)}
    indexes = {}
    for video in shared:
        intervals = sorted((r[2], r[3]) for r in by_video[video] if r[2] is not None and r[3] is not None)
        ends, maximum = [], -1
        for _, end in intervals:
            maximum = max(maximum, end)
            ends.append(maximum)
        indexes[video] = (sorted({r[0] for r in by_video[video]}), [r[0] for r in intervals], ends)
    linkage = []
    for row in frames:
        video, unit = row["source_video_id"], row["source_frame_unit"]
        person = mapping.get(video)
        joins["video_found" if video in mapping else "video_missing"] += 1
        joins["official_wearer_null"] += video in mapping and person is None
        joins["static_wearer_unknown"] += row["participant_id"] is None
        if person is not None and row["participant_id"] is not None:
            c.require(person == row["participant_id"], "OfficialWearerMismatch")
            joins["wearer_identity_verified"] += 1
        same = video in by_video
        elsewhere = person is not None and bool(people_videos[person] - {video})
        link = {**row, "official_participant_id": person, "same_video_has_sta": same,
                "official_wearer_has_sta_elsewhere": elsewhere, "exact_raw_frame_index": False,
                "nearest_sta_offset_seconds_nominal": None, "within_action_interval": False,
                "temporal_condition": CONDITION}
        if same:
            times, starts, ends = indexes[video]
            pos = bisect_left(times, unit)
            exact = pos < len(times) and times[pos] == unit
            link["exact_raw_frame_index"] = exact
            proximity["exact_raw_frame_index_frames"] += exact
            fps = clocks[video]
            if fps > 0:
                delta = min(abs(times[i] - unit) for i in (pos - 1, pos) if 0 <= i < len(times)) / fps
                c.require(math.isfinite(delta), "InvalidNominalOffset")
                link["nearest_sta_offset_seconds_nominal"] = delta
                proximity["nominal_offset_eligible_frames"] += 1
                for threshold in THRESHOLDS:
                    proximity["nearest_sta_row_within_seconds_frames"][str(threshold)] += delta <= threshold
            else:
                proximity["same_video_invalid_clock_frames"] += 1
            proximity["action_interval_eligible_frames"] += bool(starts)
            index = bisect_right(starts, unit) - 1
            link["within_action_interval"] = index >= 0 and ends[index] >= unit
            proximity["within_any_action_start_end_frames_inclusive"] += link["within_action_interval"]
        linkage.append(link)
    def sources(predicate):
        selected = [r for r in linkage if predicate(r)]
        return {"frames": len(selected), "videos": len({r["source_video_id"] for r in selected}),
                "official_wearers": len({r["official_participant_id"] for r in selected
                                         if r["official_participant_id"] is not None})}
    summary = {"trials": len(frames) * 2, "sources": sources(lambda r: True),
               "static_recorded_wearers": len({r["participant_id"] for r in frames if r["participant_id"] is not None}),
               "metadata_identity_join_frames": dict(joins),
               "same_video_has_sta": sources(lambda r: r["same_video_has_sta"]),
               "wearer_has_sta_elsewhere": sources(lambda r: r["official_wearer_has_sta_elsewhere"]),
               "wearer_has_sta_elsewhere_without_same_video": sources(lambda r: r["official_wearer_has_sta_elsewhere"] and not r["same_video_has_sta"]),
               "shared_video_sta": {"rows": len(rows), "unique_main_uid_groups": len({r[1] for r in rows}),
                                    "screened_rows": sum(r[4] for r in rows),
                                    "screened_unique_main_uid_groups": len({r[1] for r in rows if r[4]})},
               "conditional_temporal_proximity": proximity}
    return summary, linkage


def audit(original800, separate320, source_dir, receipt_path, metadata_path, out, *,
          exploration=None, confirmation_index=None, confirmation_selection=None):
    out = Path(out).absolute()
    c.require(not any(p.is_symlink() for p in (out, *out.parents)), "UnsafeOutput")
    c.require(not out.exists(), "OutputExists")
    paths = {s: (Path(source_dir) / f"fho_sta_{s}.json", c.MAX_FILE) for s in ("train", "val")}
    paths.update(original800=(original800, STATIC_LIMIT), separate320=(separate320, STATIC_LIMIT),
                 receipt=(receipt_path, c.MIB), metadata=(metadata_path, 96 * c.MIB))
    c.require((confirmation_index is None) == (confirmation_selection is None), "IncompleteConfirmationInputs")
    for role, path in (("exploration_index", exploration), ("confirmation_index", confirmation_index),
                       ("confirmation_selection", confirmation_selection)):
        if path is not None:
            paths[role] = (path, STATIC_LIMIT)
    pins = {role: c.read_input(*spec)[1] for role, spec in paths.items()}
    receipt = c.read_input(*paths["receipt"], decode=True, expected=pins["receipt"])[0]
    c.require(receipt.get("status") == "source_metadata_validated"
              and isinstance(receipt.get("files"), dict)
              and set(receipt["files"]) == {"fho_sta_train.json", "fho_sta_val.json"}, "InvalidReceipt")
    for split in ("train", "val"):
        record = receipt["files"][f"fho_sta_{split}.json"]
        c.require(isinstance(record, dict) and type(record.get("size")) is int
                  and record["size"] == pins[split]["bytes"]
                  and record.get("sha256") == pins[split]["sha256"], "ReceiptMismatch")
    metadata = c.read_input(*paths["metadata"], decode=True, expected=pins["metadata"])[0]
    mapping, _, _ = c.identities(metadata, {"version": 1, "participants": [], "videos": []})
    del metadata
    static = {role: static_frames(paths[role][0], pins[role]) for role in ("original800", "separate320")}
    if exploration is not None:
        static["exploration80"] = static_frames(exploration, pins["exploration_index"],
                                                {r["frame_id"]: r for r in static["separate320"]})
    if confirmation_index is not None:
        mechanism = static_frames(confirmation_index, pins["confirmation_index"],
                                  {r["frame_id"]: r for r in static["original800"]})
        selection = c.read_input(*paths["confirmation_selection"], decode=True,
                                 expected=pins["confirmation_selection"])[0].get("selected")
        c.require(isinstance(selection, dict) and isinstance(selection.get("frame_unit_ids"), list),
                  "InvalidConfirmationSelection")
        ids = [c.identifier(v) for v in selection["frame_unit_ids"]]
        c.require(ids and len(ids) == len(set(ids)), "InvalidConfirmationSelection")
        by_unit = {r["mechanism_frame_unit_id"]: r for r in mechanism}
        c.require(len(by_unit) == len(mechanism) and set(ids) <= by_unit.keys(), "ConfirmationSourceMissing")
        static["confirmation46"] = [by_unit[v] for v in ids]
    by_video, clocks, inventory = sta_index(paths, pins, mapping)
    datasets, links = {}, {}
    for role, frames in static.items():
        datasets[role], links[role] = aggregate(frames, mapping, by_video, clocks)
    summary = {"status": "metadata_feasibility_only", "dataset_ready": False,
               "timeline_verified": False, "situations_verified": False,
               "prior_static_use_excludes_people": False, "input_pins": pins,
               "sta_inventory": inventory, "datasets": datasets,
               "semantics": {"temporal": CONDITION, "thresholds": "Cumulative absolute offsets, including exact equality, using STA video fps.",
                             "wearer_elsewhere": "Officially mapped wearer has STA in at least one different video, possibly also this video.",
                             "action_intervals": "Any STA row's inclusive action_start_frame/action_end_frame on the source video; not necessarily the nearest row.",
                             "screen": "Census single-object, positive TTC, valid box/clock, >=2s nominal frame and clip history; no exposure exclusions or selection.",
                             "identity": "Same wearer or video does not establish the same situation; main_uid groups are not verified independent onsets."}}
    # All validation completes before creating any output; never overwrite a run.
    out.mkdir(mode=0o755)
    hidden = out / "hidden"
    created = []
    try:
        hidden.mkdir(mode=0o700)
        for role, records in links.items():
            path = hidden / f"{role}_linkage.jsonl"
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
                created.append(path)
                for record in records:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        path = out / "summary.json"
        with path.open("x") as stream:
            created.append(path)
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except BaseException:
        for path in created:
            path.unlink()
        if hidden.exists():
            hidden.rmdir()
        out.rmdir()
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original800", type=Path, default=DEFAULT_800)
    parser.add_argument("--separate320", type=Path, default=DEFAULT_320)
    parser.add_argument("--source-dir", type=Path, default=BASE / "source/v2")
    parser.add_argument("--receipt", type=Path, default=BASE / "preparation/annotation_acquisition.json")
    parser.add_argument("--metadata", type=Path, default=Path("data/external/egohos/licensed_metadata/v2/ego4d.json"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exploration", type=Path, default=DEFAULT_EXPLORATION)
    parser.add_argument("--confirmation-index", type=Path, default=DEFAULT_CONFIRMATION_INDEX)
    parser.add_argument("--confirmation-selection", type=Path, default=DEFAULT_CONFIRMATION_SELECTION)
    args = parser.parse_args(argv)
    try:
        result = audit(args.original800, args.separate320, args.source_dir, args.receipt, args.metadata, args.out,
                       exploration=args.exploration, confirmation_index=args.confirmation_index,
                       confirmation_selection=args.confirmation_selection)
    except (ValueError, OSError, OverflowError, RecursionError):
        print(json.dumps({"status": "error", "error_code": "StaticOverlapValidationFailed"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
