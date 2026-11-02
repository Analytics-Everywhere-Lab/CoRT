"""Bounded offline dense CURRENT preparation, never automatic scientific approval.

Run ``python -m research.lifecycle_memory.temporal_dense_current_data --help``.
Inputs are the v4 plan directory and one or more exact-URI local media maps.
The full selection screens metadata for exported two-second windows, then
freezes 96 distinct allocated wearers before opening labels or checking the
cache. The technical canary selects at most four cached wearers.
Both preserve the original temporal_cohort; neither reassigns confirmation rows.

Nine decoded scenes are private QA assets. Only the eight past scenes enter the
task: CURRENT and candidate0/1 are byte-for-byte copies of the original static
index real_full/crop_a/crop_b, including the original answer and crop order.
No label is inferred from the decoded current. Scene PTS are integer timestamps
in the probed rational time base, selected at or before each target.

Without explicit review proofs the output contains only QA, attrition, and
readiness, not audited observations or a release. A proof JSON has exactly
``{"version":1,"items":[{"proposal_id":...,"evidence_binding_sha256":...,
"reviewer":...,"evidence_sha256":...,"gates":{...}}]}``. The binding comes from
each item's metadata.json and pins the source/queue, all presented image bytes,
PTS provenance, and the original private static record. Every REVIEW_GATES
entry must be true. Receipt validation checks consistency, not the accuracy of
the review. Review the exact output before supplying its receipt.

Publication additionally requires a schema-3 protocol and its pinned temporal
and historical exposure registries. Full current publication requires all 96
frozen rows. A canary release is technical-only; full is descriptive-only.
Successful publication authorizes the declared descriptive scope, never confirmation.
All outputs must be new private directories outside Git or ignored by Git.
There is no downloading, scheduler submission, GPU work, or forecast support.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from research.lifecycle_memory import temporal_matched_media as media
from research.lifecycle_memory.temporal_forecasting import data


SCENES = data.DENSE_SCENE_SLOTS
OFFSETS = data.DENSE_HISTORY_OFFSETS_S
REVIEW_GATES = ("source_alignment", "timeline_verified", "orientation_verified",
                "history_integrity_visibility", "current_task_candidates")
ROOT = Path(__file__).resolve().parents[2]
HEX = re.compile(r"[0-9a-f]{64}\Z")


def _hash(value):
    return media.digest(data._canonical(value).encode())


def _json(path, limit=media.MAX_MAP):
    with media.open_input(path, limit) as stream:
        return media.strict_json(stream.read(limit + 1))


def _rows(path):
    rows = []
    with media.open_input(path, media.MAX_QUEUE) as stream:
        for _ in range(media.MAX_ROWS + 1):
            raw = stream.readline(media.MAX_LINE + 1)
            if not raw:
                break
            media.require(len(raw) <= media.MAX_LINE and len(rows) < media.MAX_ROWS, "QueueLimit")
            row = media.strict_json(raw)
            media.require(isinstance(row, dict), "InvalidQueueRow")
            rows.append(row)
    media.require(bool(rows), "EmptyQueue")
    return rows


def _identity(value):
    media.require(isinstance(value, str) and bool(value.strip()), "InvalidIdentity")
    return value


def _hex(value):
    media.require(isinstance(value, str) and HEX.fullmatch(value), "InvalidDigest")
    return value


def eligibility_blocker(row):
    """Availability screen uses source metadata only, never cache or outcomes."""
    history = row.get("history")
    if not isinstance(history, dict) or history.get("source_kind") != "exported_clip":
        return "NotExportedClip"
    if row.get("metadata_blockers"):
        return "ExistingMetadataBlocker"
    keys = ("scene_seconds_nominal", "history_start_seconds_nominal", "history_seconds",
            "fps_nominal", "parent_video_start_seconds_nominal")
    if any(type(history.get(k)) not in (int, float) or not math.isfinite(history[k]) for k in keys):
        return "InvalidHistoryClock"
    current, start, duration, fps, offset = (history[k] for k in keys)
    if not (2 <= current <= 86400 and 0 <= start <= current - 2 + 1e-8
            and 2 <= duration <= 30 and 1 <= fps <= 240 and 0 <= offset <= 86400 - current
            and abs(current - start - duration) <= 1 / fps + 1e-8):
        return "InvalidTwoSecondWindow"
    uri = history.get("s3_uri")
    if not isinstance(uri, str) or not re.fullmatch(r"s3://[^/@\s]+/[^?\s#]+", uri):
        return "ExportedURIUnavailable"
    return None


def eligibility_summary(rows):
    reasons = Counter(eligibility_blocker(row) for row in rows)
    eligible = [row for row in rows if eligibility_blocker(row) is None]
    return {"input_rows": len(rows), "eligible_rows": len(eligible),
            "eligible_wearers": len({r["participant_id"] for r in eligible}),
            "excluded_rows": len(rows) - len(eligible),
            "excluded_by_reason": {key: count for key, count in reasons.items() if key is not None},
            "screen": "metadata_exported_two_second_window_before_hash_selection_not_cache_conditioned"}


def select_current_rows(current_rows, allocation, *, stage="current", media_uris=None):
    """Outcome-blind selection; full selection never depends on cached media."""
    media.require(stage in ("current", "current-canary"), "CurrentStageOnly")
    media.require(isinstance(allocation, dict) and isinstance(allocation.get("participants"), list),
                  "InvalidAllocation")
    assigned = {}
    for person in allocation["participants"]:
        wearer = _identity(person.get("participant_id"))
        cohort = person.get("temporal_cohort")
        split = person.get("temporal_split")
        media.require(wearer not in assigned and cohort in data.SPLITS
                      and split == ("confirmation" if cohort == "confirmation" else "development"),
                      "InvalidAllocation")
        assigned[wearer] = (cohort, split)
    grouped, seen, sources, proposals = {}, set(), set(), set()
    for row in current_rows:
        media.require(row.get("task") == "current_interaction", "CurrentQueueOnly")
        wearer = _identity(row.get("participant_id"))
        key = _hex(row.get("queue_key"))
        source = _hex(row.get("source_key"))
        proposal = _hex(row.get("scene_proposal_id"))
        media.require(key not in seen and source not in sources and proposal not in proposals,
                      "DuplicateCurrentQueueIdentity")
        seen.add(key)
        sources.add(source)
        proposals.add(proposal)
        media.require(assigned.get(wearer) == (row.get("temporal_cohort"), row.get("temporal_split")),
                      "AllocationMismatch")
        _identity(row.get("source_video_id"))
        media.require(type(row.get("source_frame_unit")) is int and row["source_frame_unit"] >= 0,
                      "InvalidStaticFrame")
        if eligibility_blocker(row) is not None:
            continue
        # The canary is explicitly cache-conditioned, but never label-conditioned.
        if stage == "current-canary" and (row.get("history") or {}).get("s3_uri") not in (media_uris or set()):
            continue
        grouped.setdefault(wearer, []).append(row)
    representatives = [min(rows, key=lambda r: _hash(["dense-current-source-v1", r["queue_key"]]))
                       for rows in grouped.values()]
    representatives.sort(key=lambda r: _hash(["dense-current-wearer-v1", r["participant_id"]]))
    selected = representatives[:96 if stage == "current" else 4]
    if stage == "current":
        media.require(len(selected) == 96, "Full96InsufficientWearers")
        media.require({r["temporal_cohort"] for r in selected} == set(data.SPLITS),
                      "Full96MissingAllocatedSplit")
    return selected


def _output(path):
    path = media.safe_path(path)
    media.require(path.parent.is_dir() and not os.path.lexists(path), "OutputExistsOrParentMissing")
    if path.is_relative_to(ROOT):
        result = subprocess.run(["git", "check-ignore", "--quiet", "--", str(path / "readiness.json")],
                                cwd=ROOT, timeout=5, check=False, capture_output=True)
        media.require(result.returncode == 0, "OutputMustBeIgnored")
    else:
        # An unrelated tracked checkout is not an acceptable private output tree.
        media.require(not any((p / ".git").exists() for p in path.parents), "OutputInsideOtherRepository")
    return path


def _maps(paths):
    mapping, pins = {}, []
    for path in paths:
        values, pin = media.load_map(path)
        pins.append(pin)
        for uri, local in values.items():
            media.require(uri not in mapping or mapping[uri] == local, "ConflictingMediaMaps")
            mapping[uri] = local
    return mapping, pins


def _join(row, proposal):
    media.require(proposal is not None and proposal["task"] == "current_interaction"
                  and proposal["source_video_id"] == row["source_video_id"]
                  and proposal["temporal_cohort"] == row["temporal_cohort"]
                  and proposal["temporal_split"] == row["temporal_split"], "ProposalIdentityMismatch")
    history = media.window(proposal)
    original = row.get("history") or {}
    for key in ("s3_uri", "scene_seconds_nominal", "parent_video_start_seconds_nominal", "fps_nominal"):
        media.require(original.get(key) == history[key], "ProposalClockMismatch")
    media.require(proposal["request"]["parent_scene_frame_nominal"] == row["source_frame_unit"],
                  "OriginalCurrentAnchorMismatch")
    media.require(math.isclose(history["scene_seconds_nominal"] + history["parent_video_start_seconds_nominal"],
                              row["source_frame_unit"] / history["fps_nominal"], rel_tol=0, abs_tol=1e-7),
                  "OriginalCurrentAnchorMismatch")
    return history


def _static_records(paths, selected):
    wanted = {ref["frame_id"] for row in selected for ref in row.get("static_references", [])}
    indexed = {}
    for path in paths:
        path = media.safe_path(path)
        for record in _rows(path):
            if record.get("frame_id") in wanted:
                indexed.setdefault(record["frame_id"], []).append((path, record))
    return indexed


def _copy_asset(path, folder, name, expected=None):
    path = media.safe_path(path)
    media.require(path.suffix.lower() in data.EXTENSIONS, "UnsupportedStaticAsset")
    with media.open_input(path, 64 * media.MIB) as stream:
        raw = stream.read()
    pin = media.digest(raw)
    media.require(expected is None or pin == expected, "StaticAssetHashMismatch")
    if folder is None:
        return {"path": str(path), "sha256": pin}
    target = folder / (name + path.suffix.lower())
    with target.open("xb") as stream:
        os.chmod(stream.fileno(), 0o600)
        stream.write(raw)
    return {"path": str(target), "sha256": pin}


def preserve_static(row, indexed, folder, static_image_root):
    """Join an original trial by source identity, never by its correct answer."""
    references = row.get("static_references")
    media.require(isinstance(references, list) and references, "StaticReferenceUnavailable")
    choices = []
    for ref in references:
        for path, trial in indexed.get(ref["frame_id"], []):
            media.require(trial.get("source_video_id", "").removeprefix("ego4d:") == row["source_video_id"]
                          and trial.get("source_frame_unit") == row["source_frame_unit"]
                          and trial.get("source_image_relpath") == ref["source_image_relpath"],
                          "StaticIdentityMismatch")
            person = trial.get("participant_id")
            media.require(person is None or str(person).removeprefix("ego4d_participant:") == row["participant_id"],
                          "StaticWearerMismatch")
            order = trial.get("counterbalance_order_index")
            media.require(type(order) is int and order in (0, 1), "StaticOrderUnavailable")
            choices.append((ref["frame_id"], order, str(path), path, trial))
    media.require(bool(choices), "StaticRecordUnavailable")
    _, _, _, path, trial = min(choices, key=lambda item: item[:3])
    # Preserve the actual static task input, not a new rendering of its raw JPEG.
    assets = trial.get("assets", {})
    images = {}
    for slot, key in (("current", "real_full"), ("candidate0", "crop_a"), ("candidate1", "crop_b")):
        asset = assets.get(key)
        media.require(isinstance(asset, dict), "StaticAssetUnavailable")
        _hex(asset.get("sha256"))
        source = data._asset(path.parent.parent, asset.get("path"))
        images[slot] = _copy_asset(source, folder, slot + "_static", asset["sha256"])
    # Retain the raw-still pin for human source-alignment review, not as task input.
    raw_path = data._asset(media.safe_path(static_image_root), trial["source_image_relpath"])
    with media.open_input(raw_path, 64 * media.MIB) as stream:
        raw_still_sha = media.digest(stream.read())
    answer = trial.get("answer")
    media.require(answer in ("A", "B"), "StaticLabelUnavailable")
    candidate_ids = [trial.get("crop_a_candidate_id"), trial.get("crop_b_candidate_id")]
    media.require(all(isinstance(v, str) and v for v in candidate_ids)
                  and candidate_ids[0] != candidate_ids[1], "StaticCandidatesUnavailable")
    media.require(candidate_ids[0 if answer == "A" else 1] == trial.get("first_order_candidate_id"),
                  "StaticLabelIdentityMismatch")
    fixed = {"sha256": images["current"]["sha256"],
             "candidate_sha256": [images["candidate0"]["sha256"], images["candidate1"]["sha256"]],
             "static_record_sha256": _hash(trial), "source_frame_unit": row["source_frame_unit"]}
    return {"images": images, "fixed_current": fixed, "correct_candidate": 0 if answer == "A" else 1,
            "candidate_pair_id": _hash(candidate_ids), "event_id": trial["frame_id"],
            "raw_static_sha256": raw_still_sha, "static_index_sha256": data._file_sha(path)}


def choose_pts(ticks, time_base, start_pts, targets):
    """Rational comparison avoids rounded ffprobe *_time cutoff errors."""
    media.require(len(targets) == len(SCENES), "NineTargetsRequired")
    time_base = Fraction(time_base)
    media.require(0 < time_base <= 1 and type(start_pts) is int, "InvalidSourceClock")
    media.require(all(type(tick) is int for tick in ticks), "InvalidSourcePTS")
    result, previous = {}, None
    for slot, target in zip(SCENES, targets):
        target = Fraction(str(target))
        eligible = [tick for tick in ticks if 0 <= target - (tick - start_pts) * time_base <= Fraction(1, 4)]
        media.require(bool(eligible), "PrecedingFrameUnavailable")
        pts = max(eligible)
        media.require(previous is None or pts > previous, "RepeatedOrUnorderedPTS")
        result[slot] = {"target_seconds": float(target), "pts": pts}
        previous = pts
    media.require(set(result) == set(SCENES), "NineTargetsRequired")
    return result


def decode_dense(source, folder, history, deadline, ffmpeg, ffprobe):
    """Decode exactly the selected integer PTS; no preview timestamp guessing."""
    restrict = ["-protocol_whitelist", "file", "-format_whitelist", "mov,matroska,webm,avi",
                "-max_alloc", str(64 * media.MIB), "-max_pixels", str(media.MAX_PIXELS)]
    probe = [ffprobe, "-v", "error", *restrict, "-threads", "1", "-select_streams", "v:0"]
    info = media.run_tool([*probe, "-show_entries", "stream=width,height,time_base,start_pts",
                           "-of", "json", str(source)], deadline, capture=True)
    streams = info.get("streams", [])
    media.require(len(streams) == 1, "MissingVideoStream")
    stream = streams[0]
    width, height = stream.get("width"), stream.get("height")
    media.require(type(width) is int and type(height) is int and 0 < width <= 4096 and 0 < height <= 4096
                  and width * height <= media.MAX_PIXELS, "FrameSizeLimit")
    tb = stream.get("time_base")
    media.require(isinstance(tb, str) and re.fullmatch(r"[1-9][0-9]{0,11}/[1-9][0-9]{0,11}", tb),
                  "InvalidSourceClock")
    time_base, start = Fraction(tb), stream.get("start_pts")
    media.require(type(start) is int and 0 < time_base <= 1, "InvalidSourceClock")
    current = Fraction(str(history["scene_seconds_nominal"]))
    targets = [current + Fraction(str(delta)) for delta in OFFSETS]
    # ffprobe intervals are on the source timestamp clock, unlike relative -ss.
    start_seconds = start * time_base
    interval_start = max(start_seconds, targets[0] + start_seconds - Fraction(1, 2))
    interval_end = current + start_seconds + Fraction(1, 4)
    frames = media.run_tool([*probe, "-read_intervals", f"{float(interval_start):.9f}%{float(interval_end):.9f}",
                            "-show_entries", "frame=pts", "-of", "json", str(source)], deadline, capture=True)
    ticks = [frame["pts"] for frame in frames.get("frames", []) if "pts" in frame]
    chosen = choose_pts(ticks, time_base, start, targets)
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", *restrict,
            "-threads", "1", "-hwaccel", "none", "-noautorotate", "-filter_threads", "1"]
    images = {}
    for slot, frame in chosen.items():
        target = folder / f"{slot}_decoded.png"
        seek = max(0, frame["target_seconds"] - .5)
        media.run_tool([*base, "-copyts", "-ss", f"{seek:.9f}", "-t", "0.751", "-i", str(source),
                        "-an", "-sn", "-dn", "-map", "0:v:0", "-map_metadata", "-1",
                        "-vf", f"settb=expr={tb},select='eq(pts,{frame['pts']})',{media.CANVAS}",
                        "-vsync", "0", "-frames:v", "1", "-c:v", "png", "-threads", "1", str(target)], deadline)
        with media.open_input(target, 64 * media.MIB) as image:
            pin = media.digest(image.read())
        os.chmod(target, 0o600)
        images[slot] = {"path": str(target), "sha256": pin}
    return {"frames": chosen, "images": images, "source_time_base": tb, "source_start_pts": start,
            "orientation_policy": "stored_pixels_no_autorotate_pending_review",
            "selection_policy": "exact_integer_pts_latest_not_after_target"}


def render_review(out):
    """Create four-column private sheets with eight pasts and both current views."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    out = media.safe_path(out)
    report = _json(out / "readiness.json", media.MAX_QUEUE)
    font = ImageFont.load_default(size=16)
    paths = []
    for item in report["items"]:
        if not item.get("images") or not item.get("decoded"):
            continue
        frames = item["decoded"]["frames"]
        cells = [(slot + f" target={frames[slot]['target_seconds']:.6f} PTS={frames[slot]['pts']}",
                  out / item["images"][slot]["path"]) for slot in SCENES[:-1]]
        cells += [(label, out / item["images"][slot]["path"]) for slot, label in (
            ("current", "CURRENT: original static input"), ("candidate0", "Candidate A: original crop"),
            ("candidate1", "Candidate B: original crop"))]
        cells.append(("Decoded current: alignment QA only", Path(item["decoded"]["images"]["current"]["path"])))
        canvas = Image.new("RGB", (1440, 930), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), f"{item['proposal_id'][:16]} / {item['split']} / no gate approval implied", fill="black", font=font)
        for i, (label, path) in enumerate(cells):
            x, y = (i % 4) * 360, 30 + (i // 4) * 300
            draw.text((x + 5, y), label, fill="black", font=font)
            with Image.open(media.safe_path(path)) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (350, 268))
                canvas.paste(thumb, (x + (360 - thumb.width) // 2, y + 25 + (268 - thumb.height) // 2))
        target = out / item["proposal_id"] / "dense_review.png"
        with target.open("xb") as stream:
            os.chmod(stream.fileno(), 0o600)
            canvas.save(stream, format="PNG")
        paths.append(str(target))
    return paths


def inventory(plan_dir, media_maps, out, *, static_indexes=None, static_image_root=None, byte_receipts=()):
    """Freeze the full 96 and audit cheap local metadata/assets without decoding."""
    from research.lifecycle_memory.temporal_forecasting_static_overlap import DEFAULT_800, DEFAULT_320
    from research.lifecycle_memory.temporal_matched_preparation import STATIC_IMAGE_ROOT
    plan_dir, out = media.safe_path(plan_dir), _output(out)
    rows = _rows(plan_dir / "hidden/current_history.jsonl")
    allocation = _json(plan_dir / "hidden/matched_allocation.json")
    selected = select_current_rows(rows, allocation, stage="current")
    mapping, _ = _maps(media_maps)
    proposals, proposal_pin = media.load_queue(plan_dir / "hidden/scene_proposals.jsonl", media.MAX_ROWS)
    by_id = {r["proposal_id"]: r for r, _, _ in proposals}
    sizes = {}
    # Existing download plans contain signed URLs. Never retain those fields.
    for path in byte_receipts:
        receipt = _json(path)
        for obj in receipt.get("objects", []):
            uri, size = obj.get("s3_uri"), obj.get("bytes")
            media.require(type(size) is int and size > 0 and isinstance(uri, str), "InvalidSizeReceipt")
            media.require(uri not in sizes or sizes[uri] == size, "ConflictingSizeReceipts")
            sizes[uri] = size
    indexed = _static_records([Path(p).absolute() for p in (static_indexes or (DEFAULT_800, DEFAULT_320))], selected)
    static_image_root = Path(static_image_root or STATIC_IMAGE_ROOT).absolute()
    items, sources = [], {}
    for row in selected:
        item = {"proposal_id": row["scene_proposal_id"], "participant_id": row["participant_id"],
                "split": row["temporal_cohort"], "metadata_blockers": [], "static_blockers": []}
        items.append(item)
        try:
            history = _join(row, by_id.get(row["scene_proposal_id"]))
            media.require(not row.get("metadata_blockers"), "ExistingMetadataBlocker")
            media.require(history["source_kind"] in ("exported_clip", "full_recording")
                          and isinstance(history["s3_uri"], str), "SourceMediaUnavailable")
            uri = history["s3_uri"]
            cached = uri in mapping and Path(mapping[uri]).is_file()
            size = Path(mapping[uri]).stat().st_size if cached else sizes.get(uri)
            clip = sources.setdefault(uri, {"s3_uri": uri, "kind": history["source_kind"],
                                            "bytes": size, "cached": cached, "proposal_ids": []})
            clip["proposal_ids"].append(row["scene_proposal_id"])
            item.update(s3_uri=uri, cached=cached, history_start_seconds=history["scene_seconds_nominal"] - 2,
                        current_seconds=history["scene_seconds_nominal"], source_kind=history["source_kind"])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            code = str(exc) if type(exc) is ValueError and re.fullmatch(r"[A-Za-z0-9]+", str(exc)) else "MetadataUnavailable"
            item["metadata_blockers"].append(code)
        item["alternative_exported_proposal_ids"] = sorted(r["scene_proposal_id"] for r in rows
            if r["participant_id"] == row["participant_id"] and not r.get("metadata_blockers")
            and (r.get("history") or {}).get("source_kind") == "exported_clip")
        try:
            static = preserve_static(row, indexed, None, static_image_root)
            item["static_record_sha256"] = static["fixed_current"]["static_record_sha256"]
        except (ValueError, OSError, KeyError, TypeError) as exc:
            code = str(exc) if type(exc) is ValueError and re.fullmatch(r"[A-Za-z0-9]+", str(exc)) else "StaticAssetsUnavailable"
            item["static_blockers"].append(code)
    source_list = [sources[key] for key in sorted(sources)]
    clip_list = [source for source in source_list if source["kind"] == "exported_clip"]
    needed = [clip for clip in clip_list if not clip["cached"]]
    needed_media = [source for source in source_list if not source["cached"]]
    report = {"version": 1, "stage": "current", "selection_sha256": _hash(selected), "rows": selected,
              "preselection_attrition": eligibility_summary(rows),
              "proposal_pin": proposal_pin, "items": items, "clips": clip_list, "sources": source_list,
              "counts": {"wearers": len(selected), "by_split": dict(Counter(r["temporal_cohort"] for r in selected)),
                         "usable_time_windows": sum(not i["metadata_blockers"] for i in items),
                         "usable_exported_windows": sum(i.get("source_kind") == "exported_clip" for i in items),
                         "selected_full_recordings": sum(i.get("source_kind") == "full_recording" for i in items),
                         "wearers_with_exported_alternative": sum(bool(i["alternative_exported_proposal_ids"]) for i in items),
                         "static_tasks_verified": sum(not i["static_blockers"] for i in items),
                         "cached_clips": sum(c["cached"] for c in clip_list), "needed_clips": len(needed),
                         "known_needed_bytes": sum(c["bytes"] or 0 for c in needed),
                         "unknown_needed_sizes": sum(c["bytes"] is None for c in needed),
                         "needed_media_objects": len(needed_media),
                         "unknown_media_sizes": sum(c["bytes"] is None for c in needed_media)},
              "metadata_attrition": dict(Counter(b for i in items for b in i["metadata_blockers"])),
              "static_attrition": dict(Counter(b for i in items for b in i["static_blockers"])),
              "metadata_only_not_source_alignment_proof": True}
    out.mkdir(mode=0o700)
    media.write_json(out / "full96_inventory.json", report)
    _write_rows(out / "needed_clips.jsonl", needed)
    _write_rows(out / "needed_media.jsonl", needed_media)
    return report


def write_config(plan_dir, out, historical_registry, review_indexes):
    """Pin historical reuse and conservatively register every recorded preview.

    Temporal exposure is wearer/video-wide, including previews never accepted
    as data. This is an inventory of supplied local receipts, not a claim that
    unrecorded reviews did not happen. It deliberately never edits old registries.
    """
    from research.lifecycle_memory.temporal_forecasting.protocol import dense_current_protocol
    plan_dir, out = media.safe_path(plan_dir), _output(out)
    rows = _rows(plan_dir / "hidden/current_history.jsonl")
    video_wearers = {}
    for row in rows:
        video, wearer = row["source_video_id"], row["participant_id"]
        media.require(video_wearers.setdefault(video, wearer) == wearer, "AmbiguousVideoWearer")
    proposals, _ = media.load_queue(plan_dir / "hidden/scene_proposals.jsonl", media.MAX_ROWS)
    proposal_videos = {r["proposal_id"]: r["source_video_id"] for r, _, _ in proposals}
    exposed, receipt_pins = set(), []
    media.require(bool(review_indexes), "TemporalExposureEvidenceRequired")
    for path in review_indexes:
        receipt = _json(path, media.MAX_QUEUE)
        media.require(isinstance(receipt.get("items"), list), "InvalidReviewIndex")
        receipt_pins.append({"path": str(path), "sha256": data._file_sha(Path(path))})
        for item in receipt["items"]:
            if not any(item.get(key) for key in ("raw_source", "assets", "decoded")):
                continue
            video = item.get("source_video_id") or proposal_videos.get(item.get("proposal_id"))
            media.require(video in video_wearers, "UnmappedTemporalExposure")
            exposed.add(video)
    media.require(bool(exposed), "TemporalExposureEvidenceRequired")
    with media.open_input(historical_registry, media.MAX_MAP) as stream:
        historical = stream.read()
    data._exposure(media.strict_json(historical))
    temporal = {"version": 1, "participants": sorted({video_wearers[video] for video in exposed}),
                "videos": sorted(exposed)}
    out.mkdir(mode=0o700)
    history_path, temporal_path = out / "historical_static_registry.json", out / "temporal_exposure_registry.json"
    with history_path.open("xb") as stream:
        os.chmod(stream.fileno(), 0o600)
        stream.write(historical)
    media.write_json(temporal_path, temporal)
    policy = {"version": 1, "kind": "matched_static_reuse", "historical_static_sha256": media.digest(historical),
              "temporal_exposure_sha256": data._file_sha(temporal_path)}
    media.write_json(out / "protocol.json", dense_current_protocol(policy))
    report = {"version": 1, "policy": policy, "historical_registry_copied_unchanged": True,
              "historical_static_freshness_claimed": False, "temporal_receipt_pins": receipt_pins,
              "temporal_counts": {"participants": len(temporal["participants"]), "videos": len(temporal["videos"])},
              "scope": "all supplied recorded local previews; caller must reconcile any unrecorded temporal exposure",
              "protocol": str(out / "protocol.json"), "historical_static": str(history_path), "exposure": str(temporal_path)}
    media.write_json(out / "config_provenance.json", report)
    return report


def _proofs(path):
    if path is None:
        return {}
    receipt = _json(path)
    data._keys(receipt, {"version", "items"})
    media.require(type(receipt["version"]) is int and receipt["version"] == 1
                  and isinstance(receipt["items"], list), "InvalidProofReceipt")
    indexed = {}
    for item in receipt["items"]:
        data._keys(item, {"proposal_id", "evidence_binding_sha256", "reviewer", "evidence_sha256", "gates"})
        key = _hex(item["proposal_id"])
        media.require(key not in indexed, "DuplicateProof")
        _hex(item["evidence_binding_sha256"])
        _hex(item["evidence_sha256"])
        _identity(item["reviewer"])
        data._keys(item["gates"], REVIEW_GATES)
        media.require(all(type(v) is bool for v in item["gates"].values()), "InvalidProofGates")
        indexed[key] = item
    return indexed


def _observation(row, static, decoded, pin, history, out):
    provenance = {"version": 1, "source_media_sha256": pin["sha256"],
                  "source_time_base": decoded["source_time_base"], "source_start_pts": decoded["source_start_pts"],
                  "parent_video_start_seconds": history["parent_video_start_seconds_nominal"],
                  "frames": decoded["frames"], "fixed_current": static["fixed_current"]}
    offset = provenance["parent_video_start_seconds"]
    tb = Fraction(provenance["source_time_base"])
    times = {slot: float((provenance["frames"][slot]["pts"] - provenance["source_start_pts"]) * tb) + offset
             for slot in SCENES[:-1]}
    times.update(dict.fromkeys(("current", "candidate0", "candidate1"), history["scene_seconds_nominal"] + offset))
    images = {**{slot: decoded["images"][slot] for slot in SCENES[:-1]}, **static["images"]}
    images = {slot: {**value, "path": str(Path(value["path"]).relative_to(out))} for slot, value in images.items()}
    observation = {"episode_id": row["queue_key"], "cluster_id": row["participant_id"],
                   "video_id": row["source_video_id"], "split": row["temporal_cohort"],
                   "candidate_pair_id": static["candidate_pair_id"], "event_id": static["event_id"],
                   "task": "current_interaction", "static_pair_preserved": True,
                   "annotation_source": "audited_static_current_interaction", "observation_times_s": times,
                   "timestamps_verified": True, "candidates_visible_at_cutoff": True,
                   "images": images, "dense_current_provenance": provenance}
    label = {"episode_id": row["queue_key"], "correct_candidate": static["correct_candidate"],
             "label_status": "audited", "static_label_preserved": True,
             "static_source_sha256": static["fixed_current"]["static_record_sha256"]}
    # Paths and mutable output locations are excluded so proof survives a fresh build.
    binding = _hash({"queue": row, "provenance": provenance,
                     "image_sha256": {slot: image["sha256"] for slot, image in images.items()},
                     "decoded_current_sha256": decoded["images"]["current"]["sha256"],
                     "raw_static_sha256": static["raw_static_sha256"],
                     "static_index_sha256": static["static_index_sha256"]})
    return observation, label, binding


def _write_rows(path, rows):
    with path.open("xb") as stream:
        os.chmod(stream.fileno(), 0o600)
        stream.write(data._lines(rows))


def build(plan_dir, media_maps, out, *, stage="current-canary", timeout_seconds=120,
          static_indexes=None, static_image_root=None, proof_path=None, protocol_path=None,
          exposure_path=None, historical_static_path=None):
    """Return readiness; missing assets/proofs are recorded, never auto-approved."""
    media.require(type(timeout_seconds) is int and 1 <= timeout_seconds <= 3600, "InvalidTimeout")
    deadline = time.monotonic() + timeout_seconds
    from research.lifecycle_memory.temporal_forecasting_static_overlap import DEFAULT_800, DEFAULT_320
    from research.lifecycle_memory.temporal_matched_preparation import STATIC_IMAGE_ROOT
    plan_dir, out = media.safe_path(plan_dir), _output(out)
    indexes = [Path(p).absolute() for p in (static_indexes or (DEFAULT_800, DEFAULT_320))]
    static_image_root = Path(static_image_root or STATIC_IMAGE_ROOT).absolute()
    current_path, allocation_path = plan_dir / "hidden/current_history.jsonl", plan_dir / "hidden/matched_allocation.json"
    current_rows, allocation = _rows(current_path), _json(allocation_path)
    mapping, map_pins = _maps(media_maps)
    selected = select_current_rows(current_rows, allocation, stage=stage, media_uris=set(mapping))
    proposals, proposal_pin = media.load_queue(plan_dir / "hidden/scene_proposals.jsonl", media.MAX_ROWS)
    proposal_index = {r["proposal_id"]: r for r, _, _ in proposals}
    proofs = _proofs(proof_path)
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    out.mkdir(mode=0o700)
    report = {"version": 1, "stage": stage, "status": "blocked", "task": "current_interaction",
              "history_offsets_s": list(OFFSETS), "dataset_ready": False, "accepted": False,
              "sealed": False, "execution_authorized": False, "scientific_gates": dict.fromkeys(REVIEW_GATES, False),
              "full96_ready": False, "technical_canary_ready": False, "network_used": False, "gpu_used": False,
              "selection_policy": "hash_ranked_one_per_allocated_wearer" + ("_cached_only" if stage == "current-canary" else ""),
              "preselection_attrition": eligibility_summary(current_rows),
              "selection_sha256": _hash(selected), "current_queue_sha256": data._file_sha(current_path),
              "allocation_sha256": data._file_sha(allocation_path), "proposal_pin": proposal_pin,
              "media_map_pins": map_pins, "items": []}
    media.write_json(out / "selection.json", {"stage": stage, "selection_sha256": report["selection_sha256"],
                                               "rows": selected})
    observations, labels = [], []
    try:
        indexed = _static_records(indexes, selected)
        static_error = None
    except (ValueError, OSError):
        indexed, static_error = {}, "StaticInventoryUnavailable"
    for row in selected:
        key = row["scene_proposal_id"]
        item = {"proposal_id": key, "queue_key": row["queue_key"], "split": row["temporal_cohort"],
                "status": "blocked", "blockers": []}
        report["items"].append(item)
        folder = out / key
        try:
            media.require(time.monotonic() < deadline, "WorkerTimeBudget")
            history = _join(row, proposal_index.get(key))
            uri = history["s3_uri"]
            media.require(uri in mapping and Path(mapping[uri]).is_file(), "CachedMediaUnavailable")
            media.require(ffmpeg is not None and ffprobe is not None, "DecoderUnavailable")
            source = media.safe_path(mapping[uri])
            folder.mkdir(mode=0o700)
            with media.open_input(source, media.MAX_MEDIA) as stream:
                pin = media.media_pin(stream, deadline)
                before = os.fstat(stream.fileno())
                decoded = decode_dense(source, folder, history, deadline, ffmpeg, ffprobe)
                after = source.stat()
                media.require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                              (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "MediaChanged")
            item["decoded"] = {**decoded, "source_media_sha256": pin["sha256"]}
            media.require(static_error is None, static_error or "StaticInventoryUnavailable")
            static = preserve_static(row, indexed, folder, static_image_root)
            observation, label, binding = _observation(row, static, decoded, pin, history, out)
            # Locally verify every mechanical invariant before asking a reviewer.
            data._dense_provenance(observation)
            item.update(evidence_binding_sha256=binding, fixed_current=static["fixed_current"],
                        images=observation["images"], dense_current_provenance=observation["dense_current_provenance"])
            proof = proofs.get(key)
            media.require(proof is not None, "StaticCurrentReviewProofUnavailable")
            media.require(proof["evidence_binding_sha256"] == binding, "ReviewProofBindingMismatch")
            media.require(all(proof["gates"].values()), "ScientificReviewGateUnapproved")
            observations.append(observation)
            labels.append(label)
            item.update(status="verified_input", proof_sha256=_hash(proof))
        except (ValueError, OSError, KeyError, TypeError, ZeroDivisionError) as exc:
            code = str(exc) if type(exc) is ValueError and re.fullmatch(r"[A-Za-z0-9]+", str(exc)) else "PreparationFailed"
            item["blockers"].append(code)
            # Keep completed decode evidence for review, discard partial decode work.
            if folder.exists() and "decoded" not in item:
                shutil.rmtree(folder)
        if folder.exists():
            media.write_json(folder / "metadata.json", item)
    if observations:
        _write_rows(out / "observations.jsonl", observations)
        _write_rows(out / "labels.jsonl", labels)
    complete = len(observations) == len(selected)
    technical = bool(observations) and complete and (stage == "current-canary" or len(observations) == 96)
    report["inputs_verified"] = technical
    report["publication_blockers"] = []
    if not technical:
        report["publication_blockers"].append("Full96Incomplete" if stage == "current" else "CanaryIncomplete")
    if not all((protocol_path, exposure_path, historical_static_path)):
        report["publication_blockers"].append("PinnedProtocolAndRegistriesRequired")
    if technical and not report["publication_blockers"]:
        try:
            from research.lifecycle_memory.temporal_forecasting.protocol import protocol_contract, validate_protocol
            protocol = validate_protocol(_json(protocol_path))
            media.require(protocol["schema_version"] == 3
                          and data.slots_for_contract(protocol_contract(protocol)) == data.DENSE_SLOTS,
                          "DenseCurrentProtocolRequired")
            data.prepare_dataset(out / "observations.jsonl", out / "labels.jsonl", out / "dataset",
                                 exposure_path, protocol=protocol, historical_static_path=historical_static_path)
            report.update(status="technical_canary_only" if stage == "current-canary" else "full96_descriptive_ready",
                          dataset_published=True, dataset_ready=True, accepted=True, execution_authorized=True,
                          full96_ready=stage == "current", technical_canary_ready=stage == "current-canary",
                          scientific_gates=dict.fromkeys(REVIEW_GATES, True), confirmatory_authorized=False,
                          authorized_scope="technical_canary" if stage == "current-canary" else "full96_descriptive")
        except (ValueError, OSError, KeyError):
            report["publication_blockers"].append("DatasetValidationFailed")
    elif technical:
        report["status"] = "verified_inputs_only"
    report["counts"] = {"selected": len(selected), "verified": len(observations),
                        "blocked": len(selected) - len(observations),
                        "selected_by_split": dict(Counter(r["temporal_cohort"] for r in selected)),
                        "verified_by_split": dict(Counter(r["split"] for r in observations))}
    report["attrition"] = dict(Counter(code for item in report["items"] for code in item["blockers"]))
    media.write_json(out / "readiness.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--media-map", dest="media_maps", required=True, action="append", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stage", choices=("current", "current-canary"), default="current-canary")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--static-index", dest="static_indexes", action="append", type=Path)
    parser.add_argument("--static-image-root", type=Path)
    parser.add_argument("--proof", dest="proof_path", type=Path)
    parser.add_argument("--protocol", dest="protocol_path", type=Path)
    parser.add_argument("--exposure", dest="exposure_path", type=Path)
    parser.add_argument("--historical-static", dest="historical_static_path", type=Path)
    parser.add_argument("--inventory-only", action="store_true", help="Freeze full96 and write clip/static feasibility, without decoding.")
    parser.add_argument("--config-only", action="store_true", help="Copy historical registry and derive conservative temporal registry from supplied review indexes.")
    parser.add_argument("--review-index", dest="review_indexes", action="append", default=[], type=Path)
    parser.add_argument("--byte-receipt", dest="byte_receipts", action="append", default=[], type=Path)
    parser.add_argument("--render-review", action="store_true", help="Render contact sheets after preparation (requires Pillow).")
    try:
        args = vars(parser.parse_args(argv))
        inventory_only, render = args.pop("inventory_only"), args.pop("render_review")
        config_only, review_indexes = args.pop("config_only"), args.pop("review_indexes")
        media.require(not (config_only and inventory_only), "OnePreparationModeRequired")
        byte_receipts = args.pop("byte_receipts")
        if config_only:
            report = write_config(args["plan_dir"], args["out"], args["historical_static_path"], review_indexes)
            print(json.dumps(report))
            return 0
        if inventory_only:
            report = inventory(**{key: args[key] for key in ("plan_dir", "media_maps", "out", "static_indexes", "static_image_root")},
                               byte_receipts=byte_receipts)
            print(json.dumps({"selection_sha256": report["selection_sha256"], "counts": report["counts"],
                              "metadata_attrition": report["metadata_attrition"], "static_attrition": report["static_attrition"]}))
            return 0
        report = build(**args)
        if render:
            render_review(args["out"])
    except (ValueError, OSError, KeyError, TypeError):
        print(json.dumps({"status": "blocked", "error_code": "DensePreparationInputInvalid", "dataset_ready": False}))
        return 1
    print(json.dumps({key: report[key] for key in ("status", "counts", "attrition", "full96_ready", "technical_canary_ready")}))
    return 0 if report.get("dataset_published") else 2


if __name__ == "__main__":
    raise SystemExit(main())
