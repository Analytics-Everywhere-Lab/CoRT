"""Private, bounded matched-static preparation queues, never a sealed dataset.

Offline stdlib only: no media reads, network, model calls, labels in outputs,
registry edits, or inferred human approvals. Clocks remain provisional.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

from research.lifecycle_memory import temporal_forecasting_census as c
from research.lifecycle_memory import temporal_forecasting_static_overlap as s


ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = ROOT / c.BASE / "preparation"
DEFAULT_OUT = PRIVATE_ROOT / "matched_static_plan_2026_10_23_v2"
STATIC_IMAGE_ROOT = ROOT / "data/external/egohos/raw"
SPLITS = ("development", "confirmation", "unassigned", "conflict", "unknown")
ALLOCATION_SEED = 20261023
ALLOCATION_DOMAIN = "temporal-matched-development-v1"
GATES = {
    "protocol_amendment": [],
    "split_and_yield_review": ["protocol_amendment"],
    "source_alignment": ["protocol_amendment"],
    "history_integrity_visibility": ["source_alignment"],
    "current_task_candidates": ["history_integrity_visibility"],
    "forecast_cutoff_and_onset": ["source_alignment"],
    "future_blind_candidate_freeze": ["forecast_cutoff_and_onset", "history_integrity_visibility"],
    "same_event_object_audit": ["future_blind_candidate_freeze"],
    "forecast_label_semantics": ["same_event_object_audit"],
    "independence_and_exposure_review": ["split_and_yield_review", "forecast_label_semantics"],
    "fir_cpu_canary": ["current_task_candidates", "independence_and_exposure_review"],
    "final_seal_authorization": ["fir_cpu_canary"],
}
CURRENT_GATES = ("protocol_amendment", "split_and_yield_review", "source_alignment",
                 "history_integrity_visibility", "current_task_candidates")
FORECAST_GATES = ("protocol_amendment", "split_and_yield_review", "source_alignment",
                  "history_integrity_visibility", "forecast_cutoff_and_onset",
                  "future_blind_candidate_freeze", "same_event_object_audit",
                  "forecast_label_semantics", "independence_and_exposure_review")


def stable_key(*values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def validate_output(out):
    """Production outputs must live in the ignored private preparation tree."""
    out = Path(os.path.abspath(out))
    c.require(not any(p.is_symlink() for p in (out, *out.parents)), "UnsafeOutput")
    c.require(out.is_relative_to(PRIVATE_ROOT) and out != PRIVATE_ROOT, "OutputOutsidePrivateTree")
    c.require(not out.exists(), "OutputExists")
    c.require(out.parent.is_dir(), "OutputParentMissing")
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(out / "hidden/current_history.jsonl")],
        cwd=ROOT, capture_output=True, timeout=5, check=False,
    )
    c.require(result.returncode == 0, "OutputNotIgnored")
    return out


def load_inputs(original800, separate320, source_dir, receipt, metadata,
                exploration, confirmation_index, confirmation_selection, clip_manifest=None):
    paths = {split: (Path(source_dir) / f"fho_sta_{split}.json", c.MAX_FILE)
             for split in ("train", "val")}
    paths.update(original800=(original800, s.STATIC_LIMIT), separate320=(separate320, s.STATIC_LIMIT),
                 receipt=(receipt, c.MIB), metadata=(metadata, 96 * c.MIB),
                 exploration_index=(exploration, s.STATIC_LIMIT),
                 confirmation_index=(confirmation_index, s.STATIC_LIMIT),
                 confirmation_selection=(confirmation_selection, s.STATIC_LIMIT))
    if clip_manifest is not None:
        paths["clip_manifest"] = (clip_manifest, 96 * c.MIB)
    pins = {role: c.read_input(*spec)[1] for role, spec in paths.items()}

    def load(role):
        return c.read_input(*paths[role], decode=True, expected=pins[role])[0]

    acquired = load("receipt")
    c.require(acquired.get("status") == "source_metadata_validated"
              and isinstance(acquired.get("files"), dict)
              and set(acquired["files"]) == {"fho_sta_train.json", "fho_sta_val.json"}, "InvalidReceipt")
    for split in ("train", "val"):
        record = acquired["files"][f"fho_sta_{split}.json"]
        c.require(isinstance(record, dict) and type(record.get("size")) is int
                  and record["size"] == pins[split]["bytes"]
                  and record.get("sha256") == pins[split]["sha256"], "ReceiptMismatch")
    meta = load("metadata")
    mapping, _, _ = c.identities(meta, {"version": 1, "participants": [], "videos": []})
    cohorts = {name: s.static_frames(paths[name][0], pins[name]) for name in ("original800", "separate320")}
    cohorts["exploration80"] = s.static_frames(exploration, pins["exploration_index"],
                                               {r["frame_id"]: r for r in cohorts["separate320"]})
    mechanism = s.static_frames(confirmation_index, pins["confirmation_index"],
                                {r["frame_id"]: r for r in cohorts["original800"]})
    selected = load("confirmation_selection").get("selected")
    c.require(isinstance(selected, dict) and isinstance(selected.get("frame_unit_ids"), list),
              "InvalidConfirmationSelection")
    ids = [c.identifier(v) for v in selected["frame_unit_ids"]]
    by_unit = {r["mechanism_frame_unit_id"]: r for r in mechanism}
    c.require(ids and len(set(ids)) == len(ids) and len(by_unit) == len(mechanism)
              and set(ids) <= by_unit.keys(), "InvalidConfirmationSelection")
    cohorts["confirmation46"] = [by_unit[v] for v in ids]
    sta, clocks, _ = s.sta_index(paths, pins, mapping)
    # The official metadata's clips table is itself the default interval manifest.
    manifest = (read_clip_csv(clip_manifest, pins["clip_manifest"])
                if clip_manifest is not None and Path(clip_manifest).suffix.lower() == ".csv"
                else load("clip_manifest") if clip_manifest is not None else meta)
    return cohorts, meta, manifest, mapping, sta, clocks, pins


def read_clip_csv(path, pin):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(96 * c.MIB + 1)
    c.require({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} == pin, "InputChanged")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    fields = reader.fieldnames or []
    c.require(len(fields) == len(set(fields)) and {"file_uid", "parent_video_uid", "parent_frame_start",
              "parent_frame_end", "s3_path", "parent_start_sec", "parent_end_sec"} <= set(fields), "InvalidClipCSVColumns")
    locations = [name for name in ("canonical_s3_location", "s3_path") if name in fields]
    c.require(bool(locations), "MissingClipCSVLocation")
    clips = []
    for row in reader:
        c.require(len(clips) < c.MAX_ROWS and None not in row and all(v is not None for v in row.values()),
                  "InvalidClipCSVRow")
        uris = {media_uri(row[key]) for key in locations} - {None}
        c.require(len(uris) <= 1, "ConflictingClipCSVLocation")
        clips.append(dict(clip_uid=c.identifier(row["file_uid"]), video_uid=c.identifier(row["parent_video_uid"]),
            video_start_frame=int(row["parent_frame_start"]), video_end_frame=int(row["parent_frame_end"]),
            video_start_sec=c.finite_float(row["parent_start_sec"]),
            video_end_sec=c.finite_float(row["parent_end_sec"]),
            s3_path=next(iter(uris)) if uris else None, identity_namespace="exported_file_uid"))
    return {"clips": clips}


def metadata_indexes(meta, manifest):
    videos, clips, seen = {}, defaultdict(list), set()
    for row in meta["videos"]:
        video = c.identifier(row.get("video_uid"))
        info = row.get("video_metadata")
        c.require(isinstance(info, dict), "MissingVideoClock")
        fps, count = info.get("fps"), info.get("num_frames")
        c.require(type(fps) in (int, float) and math.isfinite(fps) and fps > 0
                  and type(count) is int and 0 < count <= 2**53, "InvalidVideoClock")
        videos[video] = (fps, count, media_uri(row.get("s3_path")))
    c.require(isinstance(manifest.get("clips"), list), "MissingClipManifest")
    for row in manifest["clips"]:
        c.require(isinstance(row, dict), "InvalidClip")
        uid, video = c.identifier(row.get("clip_uid")), c.identifier(row.get("video_uid"))
        start, end = row.get("video_start_frame"), row.get("video_end_frame")
        c.require(uid not in seen, "DuplicateClip")
        seen.add(uid)
        c.require(type(start) is int and type(end) is int
                  and -(2**53) <= start <= end <= 2**53, "InvalidClipInterval")
        # Some official clips start at -1. Do not silently resolve index origins.
        if video not in videos or start < 0:
            continue
        offset = row.get("video_start_sec", start / videos[video][0])
        end_seconds = row.get("video_end_sec", end / videos[video][0])
        c.require(all(type(v) in (int, float) and math.isfinite(v) for v in (offset, end_seconds))
                  and 0 <= offset <= end_seconds,
                  "InvalidClipTime")
        namespace = row.get("identity_namespace", "official_metadata_clip_uid")
        c.require(namespace in ("exported_file_uid", "official_metadata_clip_uid"), "InvalidClipNamespace")
        clips[video].append((start, end, uid, media_uri(row.get("s3_path")), offset, namespace, end_seconds))
    for intervals in clips.values():
        intervals.sort(key=lambda r: (r[1] - r[0], r[0], r[2]))
    return videos, clips


def media_uri(value):
    if value in (None, ""):
        return None
    c.require(isinstance(value, str), "InvalidMediaURI")
    uri = urlsplit(value)
    c.require(uri.scheme == "s3" and uri.netloc and uri.path not in ("", "/")
              and not uri.query and not uri.fragment and "@" not in uri.netloc
              and not any(ch.isspace() for ch in value), "InvalidMediaURI")
    return value


def source_inventory(cohorts, mapping):
    """Join physical frames across batteries without conflating static task labels."""
    sources, memberships = {}, defaultdict(set)
    for role, split in (("exploration80", "development"), ("confirmation46", "confirmation")):
        for row in cohorts[role]:
            person = mapping.get(row["source_video_id"])
            if person is not None:
                memberships[person].add(split)
    for role, rows in sorted(cohorts.items()):
        for row in rows:
            video, frame = row["source_video_id"], row["source_frame_unit"]
            person = mapping.get(video)
            c.require(person is None or row["participant_id"] is None or person == row["participant_id"],
                      "OfficialWearerMismatch")
            assignments = memberships[person]
            split = ("unknown" if person is None else "conflict" if len(assignments) > 1
                     else next(iter(assignments)) if assignments else "unassigned")
            key = stable_key("source", video, frame)
            value = sources.setdefault(key, dict(source_key=key, source_video_id=video,
                source_frame_unit=frame, participant_id=person, temporal_split=split,
                static_references=[]))
            value["static_references"].append({"cohort": role, "frame_id": row["frame_id"],
                                               "source_image_relpath": row["source_image_relpath"]})
    return sources


def history_window(video, cutoff, videos, clips, seconds):
    if video not in videos:
        return None, ["unknown_source_video"]
    fps, count, full_uri = videos[video]
    start = cutoff - math.ceil(seconds * fps)
    blockers = []
    if not 0 <= cutoff < count:
        blockers.append("cutoff_outside_video")
    if start < 0:
        blockers.append("insufficient_parent_history")
    covers = [r for r in clips[video] if r[0] <= start and cutoff <= r[1]] if not blockers else []
    # Exported frame indices can differ from supplied seconds by one frame.
    # Preserve both clocks, but never decode a negative/out-of-clip request.
    usable = [r for r in covers if r[3] is not None
              and abs(r[4] * fps - r[0]) <= 1 + 1e-6
              and abs(r[6] * fps - r[1]) <= 1 + 1e-6
              and r[4] <= start / fps and cutoff / fps <= r[6]]
    clip = usable[0] if usable else None
    uri = clip[3] if clip else full_uri
    if uri is None:
        blockers.append("missing_media_uri")
    offset = clip[4] if clip else 0.0
    return dict(start_frame_nominal=start, cutoff_frame_nominal=cutoff, fps_nominal=fps,
                history_seconds=seconds, timeline_verified=False,
                source_kind="exported_clip" if clip else "full_recording",
                s3_uri=uri, parent_video_start_seconds_nominal=offset,
                parent_video_end_seconds_nominal=(count - 1) / fps if clip is None else clip[6],
                parent_frame_minus_seconds_nominal=0 if clip is None else clip[0] - offset * fps,
                history_start_seconds_nominal=start / fps - offset,
                scene_seconds_nominal=cutoff / fps - offset,
                covering_clip_without_usable_uri=bool(covers) and not bool(usable),
                clip_uid=None if clip is None else clip[2],
                clip_identity_namespace=None if clip is None else clip[5],
                media_parent_start_frame_nominal=0 if clip is None else clip[0],
                media_parent_end_frame_nominal=count - 1 if clip is None else clip[1],
                clip_start_frame_nominal=None if clip is None else round(start - offset * fps),
                clip_cutoff_frame_nominal=None if clip is None else round(cutoff - offset * fps),
                manifest_coverage_only=True, media_availability_verified=False), blockers


def split_blockers(split):
    return [] if split in ("development", "confirmation") else [f"split_{split}"]


def counts(rows):
    return {"items": len(rows), "videos": len({r["source_video_id"] for r in rows}),
            "wearers": len({r["participant_id"] for r in rows if r["participant_id"] is not None})}


def matched_allocation(cohorts, mapping, sources, pins):
    """Freeze the explicitly authorized 147-person matched, not fresh, cohort."""
    def people(role):
        return {mapping.get(r["source_video_id"]) for r in cohorts[role]}
    development, old_confirmation = people("exploration80"), people("confirmation46")
    all_people = {r["participant_id"] for r in sources.values()}
    c.require(None not in all_people and None not in development | old_confirmation
              and len(development) == 80 and len(old_confirmation) == 46
              and len(all_people) == 147 and not development & old_confirmation
              and development | old_confirmation <= all_people, "MatchedAllocationCohortMismatch")
    rank = lambda person: (stable_key(ALLOCATION_DOMAIN, ALLOCATION_SEED, person), person)
    pilot = set(sorted(development, key=rank)[:30])
    records = [dict(participant_id=person,
        temporal_split="development" if person in development else "confirmation",
        temporal_cohort="pilot" if person in pilot else "exploration" if person in development else "confirmation",
        prior_static_group="exploration80" if person in development else "confirmation46" if person in old_confirmation else "remaining21",
        allocation_hash=rank(person)[0]) for person in sorted(all_people)]
    return dict(schema_version=1, seed=ALLOCATION_SEED, domain=ALLOCATION_DOMAIN,
                hash_encoding="SHA256(JSON compact array [domain, integer seed, normalized wearer string]); lexical hash then wearer tie-break",
                participants=records, input_pins=pins, accepted=False, globally_fresh=False)


def wearer_first_order(rows):
    """One unblocked candidate per eligible wearer before the remaining queue."""
    buckets = defaultdict(list)
    for row in rows:
        if row["participant_id"] is not None:
            buckets[row["participant_id"]].append(row)
    first = []
    for person in sorted(buckets, key=lambda p: (stable_key("temporal-matched-queue-v1", ALLOCATION_SEED, p), p)):
        candidates = [r for r in buckets[person] if not r["metadata_blockers"]]
        if candidates:
            first.append(candidates[0])
    selected = {r["queue_key"] for r in first}
    return first + [r for r in rows if r["queue_key"] not in selected]


def make_plan(cohorts, meta, manifest, mapping, sta, clocks, pins, *,
              history_seconds=2.0, max_current=1200, max_forecast=2000, max_events_per_video=32,
              allocate_matched=False, allocation_out=None):
    c.require(type(history_seconds) in (int, float) and math.isfinite(history_seconds)
              and 2 <= history_seconds <= 30, "InvalidHistorySeconds")
    c.require(all(type(v) is int and 1 <= v <= 20000
                  for v in (max_current, max_forecast, max_events_per_video)), "InvalidQueueCap")
    videos, clips = metadata_indexes(meta, manifest)
    sources = source_inventory(cohorts, mapping)
    allocation = None
    if allocate_matched:
        allocation = matched_allocation(cohorts, mapping, sources, pins)
        assigned = {r["participant_id"]: r for r in allocation["participants"]}
        for source in sources.values():
            person = assigned[source["participant_id"]]
            source.update(temporal_split=person["temporal_split"], temporal_cohort=person["temporal_cohort"])
        if allocation_out is not None:
            allocation_out.update(allocation)
    current, by_video = [], defaultdict(list)
    for key, source in sorted(sources.items()):
        video = source["source_video_id"]
        by_video[video].append(source)
        window, blockers = history_window(video, source["source_frame_unit"], videos, clips, history_seconds)
        current.append({**source, "queue_key": stable_key("current", key), "task": "current_interaction",
                        "history": window, "metadata_blockers": blockers + split_blockers(source["temporal_split"]),
                        "candidate_policy": "Retain original task and candidate identities only after source audit; no label export.",
                        "accepted": False})
    groups = defaultdict(list)
    for video in sorted(by_video):
        for frame, group, start, end, screened in sta.get(video, []):
            groups[(video, group)].append((frame, start, end, screened))
    forecast, rejected = [], Counter()
    for (video, group), observations in sorted(groups.items()):
        intervals = {(r[1], r[2]) for r in observations}
        if len(intervals) != 1 or any(None in interval for interval in intervals):
            rejected["missing_or_inconsistent_action_interval"] += 1
            continue
        start, end = next(iter(intervals))
        if video not in videos or not math.isclose(clocks.get(video, 0), videos[video][0], rel_tol=0, abs_tol=1e-6):
            rejected["video_sta_clock_mismatch"] += 1
            continue
        # Earliest screened observation is fixed before any object match is audited.
        # Action intervals describe the annotated action, not a contact/onset clock.
        # Positive STA TTC is a metadata screen, never a strict-contact certificate.
        eligible = sorted({r[0] for r in observations if r[3]
                           and r[0] >= math.ceil(history_seconds * videos[video][0])})
        if not eligible:
            rejected["no_screened_sta_observation_with_history"] += 1
            continue
        cutoff = eligible[0]
        window, blockers = history_window(video, cutoff, videos, clips, history_seconds)
        if not 0 <= start <= end < videos[video][1]:
            blockers.append("action_interval_outside_video")
        links = sorted(r["source_key"] for r in by_video[video])
        same_event = sorted(r["source_key"] for r in by_video[video] if start <= r["source_frame_unit"] <= end)
        source = by_video[video][0]
        split = source["temporal_split"]
        forecast.append(dict(queue_key=stable_key("forecast", video, group), task="forecast_next_annotated_interaction",
            source_video_id=video, participant_id=mapping.get(video), temporal_split=split,
            main_uid=group, observation_frame_nominal=cutoff, action_start_frame_nominal=start,
            action_end_frame_nominal=end, history=window, same_video_source_keys=links,
            nominal_same_event_source_keys=same_event, same_event_verified=False,
            relation="nominal_same_event_candidate" if same_event else "same_video_only",
            strict_direct_hand_contact=None, independent_onset_verified=False,
            cutoff_before_contact_verified=False,
            candidate_generation_status="not_generated", candidate_freeze_sha256=None,
            candidate_policy="Generate only from cutoff/history; freeze before hidden label join; never use later static crops or replace pairs after mismatch.",
            metadata_screen_only=True, metadata_blockers=blockers + split_blockers(split), accepted=False))
        if allocate_matched:
            forecast[-1]["temporal_cohort"] = source["temporal_cohort"]
    forecast.sort(key=lambda r: (not bool(r["nominal_same_event_source_keys"]), r["queue_key"]))
    if allocate_matched:
        forecast = wearer_first_order(forecast)
    limited, per_video = [], Counter()
    for row in forecast:
        if per_video[row["source_video_id"]] < max_events_per_video:
            limited.append(row)
            per_video[row["source_video_id"]] += 1
    selected_current, selected_forecast = current[:max_current], limited[:max_forecast]
    reviews = []
    for row in selected_current + selected_forecast:
        gates = CURRENT_GATES if row["task"] == "current_interaction" else FORECAST_GATES
        reviews.append(dict(queue_key=row["queue_key"], task=row["task"], temporal_split=row["temporal_split"],
                            status="blocked_metadata" if row["metadata_blockers"] else "pending_human_review",
                            metadata_blockers=row["metadata_blockers"],
                            gates={gate: {"status": "pending", "reviewer": None, "evidence": None} for gate in gates}))
        if allocate_matched:
            reviews[-1]["temporal_cohort"] = row["temporal_cohort"]
    cohort_yield = {}
    for role, rows in sorted(cohorts.items()):
        keys = {stable_key("source", r["source_video_id"], r["source_frame_unit"]) for r in rows}
        matches = [r for r in current if r["source_key"] in keys]
        same_video = [r for r in forecast if keys.intersection(r["same_video_source_keys"])]
        same_event = [r for r in forecast if keys.intersection(r["nominal_same_event_source_keys"])]
        kept = [r for r in selected_forecast if keys.intersection(r["nominal_same_event_source_keys"])]
        available = {r["participant_id"] for r in same_event if r["participant_id"] is not None and not r["metadata_blockers"]}
        expected = {r["participant_id"] for r in matches if r["participant_id"] is not None}
        cohort_yield[role] = dict(current_sources=counts(matches),
            current_metadata_unblocked=counts([r for r in matches if not r["metadata_blockers"]]),
            same_video_forecast_groups=counts(same_video), nominal_same_event_groups=counts(same_event),
            queued_nominal_same_event_groups=counts(kept),
            wearers_without_unblocked_nominal_same_event=len(expected - available),
            accepted_current=0, accepted_forecast=0, accepted_wearer_gap=len(expected))
    summary = dict(schema_version=1, status="private_preparation_only", dataset_ready=False, sealed=False,
        input_pins=pins, timeline_verified=False, situations_verified=False, candidates_future_blind_verified=False,
        clip_manifest_unknown_video_rows=sum(r["video_uid"] not in videos for r in manifest["clips"]),
        clip_manifest_negative_start_rows=sum(r["video_start_frame"] < 0 for r in manifest["clips"]),
        prior_static_use_excludes_people=False, histories_seconds=history_seconds,
        limits=dict(max_current=max_current, max_forecast=max_forecast, max_events_per_video=max_events_per_video),
        current=dict(inventory=counts(current), queued=counts(selected_current), cap_omitted=len(current)-len(selected_current),
            overlapping_metadata_blockers=dict(Counter(b for r in current for b in r["metadata_blockers"]))),
        forecast=dict(same_video_main_uid_groups_seen=len(groups), eligible_groups=counts(forecast),
            queued=counts(selected_forecast), rejected_groups=dict(sorted(rejected.items())),
            per_video_cap_omitted=len(forecast)-len(limited), total_cap_omitted=len(limited)-len(selected_forecast),
            same_event_verified=0, accepted=0),
        cohort_yield=cohort_yield,
        by_temporal_split={split: dict(current=counts([r for r in current if r["temporal_split"] == split]),
            forecast=counts([r for r in forecast if r["temporal_split"] == split]),
            queued_current=counts([r for r in selected_current if r["temporal_split"] == split]),
            queued_forecast=counts([r for r in selected_forecast if r["temporal_split"] == split])) for split in SPLITS},
        review_queue=dict(items=len(reviews), statuses=dict(Counter(r["status"] for r in reviews))),
        required_gates={name: dict(status="pending", depends_on=deps) for name, deps in GATES.items()},
        execution=dict(target="Fir", submitted=False, local_threads=1, local_timeout_seconds=120,
                       gpu_requested=False, media_downloaded=False, media_hashed=False,
                       next_step="Human protocol/split approval, then separately authorized Fir CPU source preparation and canary; no submission by this planner."),
        semantics=dict(timing=s.CONDITION,
            groups="main_uid deduplicates repeated observations, not independently verified onsets.",
            labels="STA next annotated interaction is not a strict direct-hand-contact label.",
            action_intervals="Action start/end do not define the interaction onset; use screened STA observations as proposed cutoffs, then audit onset separately.",
            privacy="Every queue is reviewer-private, label-conditioned, and not method-facing input.",
            split="Preserve prior static wearer assignments; conflicts and unassigned wearers require human resolution.",
            candidates="Object candidates do not exist yet. Freeze cutoff-only candidates before hidden future-label matching; reject mismatch under a fixed rule."))
    if allocate_matched:
        raw = (json.dumps(allocation, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        eligible_people = {r["participant_id"] for r in forecast if not r["metadata_blockers"]}
        queued_people = {r["participant_id"] for r in selected_forecast if not r["metadata_blockers"]}
        summary["matched_allocation"] = dict(enabled=True, seed=ALLOCATION_SEED, domain=ALLOCATION_DOMAIN,
            manifest="hidden/matched_allocation.json", sha256=hashlib.sha256(raw).hexdigest(),
            policy="All old exploration wearers remain development (30 pilot/50 exploration); old46 plus remaining21 form NEW confirmation67, not the old46 cohort.",
            globally_fresh=False, confirmation_minimum_after_qa=46, accepted_confirmation_wearers=0,
            confirmation_qa_gate_passed=False,
            wearer_counts=dict(Counter(r["temporal_cohort"] for r in allocation["participants"])),
            queue_policy="First round: one metadata-unblocked event per eligible wearer, deterministic wearer hash order; then fill existing event order, retaining per-video and global caps.",
            eligible_forecast_wearers=len(eligible_people), queued_eligible_forecast_wearers=len(queued_people),
            cap_omitted_eligible_forecast_wearers=len(eligible_people - queued_people))
        summary["by_temporal_cohort"] = {group: dict(
            current=counts([r for r in current if r["temporal_cohort"] == group]),
            forecast=counts([r for r in forecast if r["temporal_cohort"] == group]),
            queued_forecast=counts([r for r in selected_forecast if r["temporal_cohort"] == group]))
            for group in ("pilot", "exploration", "confirmation")}
        summary["semantics"]["split"] = summary["matched_allocation"]["policy"]
    return summary, selected_current, selected_forecast, reviews


def scene_proposals(current, forecast):
    """Whitelist CPU extraction inputs; never propagate private future linkage."""
    scenes = {}
    for row in current + forecast:
        current_task = row["task"] == "current_interaction"
        task = "current_interaction" if current_task else "forecast_observation"
        history = row["history"]
        cutoff = row["source_frame_unit"] if current_task else row["observation_frame_nominal"]
        key = stable_key("scene", task, row["source_video_id"], cutoff)
        row["scene_proposal_id"] = key
        media = None if history is None else dict(
            s3_uri=history["s3_uri"], kind=history["source_kind"],
            identity_namespace=history["clip_identity_namespace"],
            clip_uid=history["clip_uid"] if history["clip_identity_namespace"] == "official_metadata_clip_uid" else None,
            file_uid=history["clip_uid"] if history["clip_identity_namespace"] == "exported_file_uid" else None,
            parent_frame_start=history["media_parent_start_frame_nominal"],
            parent_frame_end=history["media_parent_end_frame_nominal"],
            parent_frame_minus_seconds_nominal=history["parent_frame_minus_seconds_nominal"],
            parent_video_end_seconds_nominal=history["parent_video_end_seconds_nominal"],
            parent_video_start_seconds_nominal=history["parent_video_start_seconds_nominal"])
        request = None if history is None else dict(
            history_start_seconds_nominal=history["history_start_seconds_nominal"],
            scene_seconds_nominal=history["scene_seconds_nominal"],
            fps_nominal=history["fps_nominal"], history_seconds=history["history_seconds"],
            past1_seconds_nominal=history["scene_seconds_nominal"] - 1.0,
            past1_frame_nominal=round((history["scene_seconds_nominal"] - 1) * history["fps_nominal"]),
            current_frame_nominal=round(history["scene_seconds_nominal"] * history["fps_nominal"]),
            history_start_frame_nominal=round(history["history_start_seconds_nominal"] * history["fps_nominal"]),
            parent_history_start_frame_nominal=history["start_frame_nominal"],
            parent_scene_frame_nominal=history["cutoff_frame_nominal"],
            decode_end_seconds_nominal=history["scene_seconds_nominal"])
        stills = sorted({str(STATIC_IMAGE_ROOT / r["source_image_relpath"]) for r in row["static_references"]}) if current_task else []
        proposal = dict(proposal_id=key, task=task, temporal_split=row["temporal_split"],
                        source_video_id=row["source_video_id"],
                        media=media, request=request, static_still_paths=stills,
                        timeline_verified=False, ready_for_media_prep=not bool(row["metadata_blockers"]),
                        execution_authorized=False, blockers=row["metadata_blockers"])
        if "temporal_cohort" in row:
            proposal["temporal_cohort"] = row["temporal_cohort"]
        c.require(key not in scenes or scenes[key] == proposal, "ConflictingSceneProposal")
        scenes[key] = proposal
    return [scenes[key] for key in sorted(scenes)]


def write_plan(out, summary, current, forecast, reviews, allocation=None):
    scenes = scene_proposals(current, forecast)
    summary["scene_proposals"] = {"items": len(scenes), "metadata_ready": sum(r["ready_for_media_prep"] for r in scenes),
                                  "execution_authorized": False, "contains_future_labels": False}
    out.mkdir(mode=0o700)
    hidden = out / "hidden"
    hidden.mkdir(mode=0o700)
    created = []
    try:
        for name, rows in (("scene_proposals", scenes), ("current_history", current),
                           ("forecast_events", forecast), ("review_queue", reviews)):
            path = hidden / f"{name}.jsonl"
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
                created.append(path)
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        if allocation:
            path = hidden / "matched_allocation.json"
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
                created.append(path)
                json.dump(allocation, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
        path = out / "summary.json"
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            created.append(path)
            json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    except BaseException:
        for path in created:
            path.unlink()
        hidden.rmdir()
        out.rmdir()
        raise


def build(original800, separate320, source_dir, receipt, metadata, out, *, exploration,
          confirmation_index, confirmation_selection, clip_manifest=None, **limits):
    out = validate_output(out)
    inputs = load_inputs(original800, separate320, source_dir, receipt, metadata, exploration,
                         confirmation_index, confirmation_selection, clip_manifest)
    allocation = {}
    summary, current, forecast, reviews = make_plan(*inputs, allocation_out=allocation, **limits)
    write_plan(out, summary, current, forecast, reviews, allocation=allocation)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original800", type=Path, default=s.DEFAULT_800)
    parser.add_argument("--separate320", type=Path, default=s.DEFAULT_320)
    parser.add_argument("--source-dir", type=Path, default=c.BASE / "source/v2")
    parser.add_argument("--receipt", type=Path, default=c.BASE / "preparation/annotation_acquisition.json")
    parser.add_argument("--metadata", type=Path, default=Path("data/external/egohos/licensed_metadata/v2/ego4d.json"))
    parser.add_argument("--clip-manifest", type=Path, help="Official exported clips CSV or JSON clips table; defaults to metadata.clips.")
    parser.add_argument("--exploration", type=Path, default=s.DEFAULT_EXPLORATION)
    parser.add_argument("--confirmation-index", type=Path, default=s.DEFAULT_CONFIRMATION_INDEX)
    parser.add_argument("--confirmation-selection", type=Path, default=s.DEFAULT_CONFIRMATION_SELECTION)
    parser.add_argument("--history-seconds", type=float, default=2.0)
    parser.add_argument("--max-current", type=int, default=1200)
    parser.add_argument("--max-forecast", type=int, default=2000)
    parser.add_argument("--max-events-per-video", type=int, default=32)
    parser.add_argument("--allocate-matched", action="store_true", help="Opt in to frozen matched 30 pilot/50 exploration/67 new confirmation wearer allocation.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = vars(parser.parse_args(argv))
    try:
        summary = build(**args)
    except (ValueError, OSError, OverflowError, RecursionError, csv.Error, subprocess.SubprocessError):
        print(json.dumps({"status": "error", "error_code": "MatchedPreparationValidationFailed"}))
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
