"""Private CPU previews of label-blind scene proposals, never ready datasets.

The path map is a JSON object mapping exact S3 URIs to absolute local regular
files. Queue, map and output paths must also be absolute and symlink-free.
No credentials, downloading, submission, candidates or labels are used.
Frames retain stored orientation; humans must verify orientation and alignment.
Only scene_proposals.jsonl rows are supported, not reviewer-private queues.
Suite slots are chronological: past1=current-2s, past2=current-1s, current.
The planner's request.past1_seconds_nominal names the newer past frame, so it
maps explicitly to worker past2 (including filenames), never worker past1.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit


MIB = 1024 * 1024
MAX_QUEUE = 64 * MIB
MAX_MAP = 8 * MIB
MAX_MEDIA = 4 * 1024 * MIB
MAX_LINE = 64 * 1024
MAX_ROWS = 20000
MAX_PIXELS = 12_000_000
CANVAS = "scale=w='trunc(min(896,672*dar)/2)*2':h='trunc(min(672,896/dar)/2)*2',setsar=1,pad=896:672:(ow-iw)/2:(oh-ih)/2"
GATES = {name: False for name in (
    "dataset_ready", "accepted", "sealed", "source_alignment", "timeline_verified",
    "orientation_verified", "history_integrity_visibility", "current_task_candidates",
    "forecast_cutoff_and_onset", "future_blind_candidate_freeze", "same_event_object_audit",
    "forecast_label_semantics", "candidates_future_blind_verified", "same_event_verified",
    "independent_onset_verified", "protocol_amendment", "split_and_yield_review",
    "independence_and_exposure_review", "fir_cpu_canary", "final_seal_authorization",
    "execution_authorized", "cutoff_before_contact_verified")}
PROPOSAL_FIELDS = {"proposal_id", "task", "temporal_split", "temporal_cohort", "source_video_id", "media", "request",
                   "static_still_paths", "timeline_verified", "ready_for_media_prep",
                   "execution_authorized", "blockers"}
MEDIA_FIELDS = {"s3_uri", "kind", "identity_namespace", "clip_uid", "file_uid", "parent_frame_start",
                "parent_frame_end", "parent_video_start_seconds_nominal", "parent_video_end_seconds_nominal",
                "parent_frame_minus_seconds_nominal"}
REQUEST_FIELDS = {"history_start_seconds_nominal", "scene_seconds_nominal", "fps_nominal", "history_seconds",
                  "past1_seconds_nominal", "past1_frame_nominal", "current_frame_nominal",
                  "history_start_frame_nominal", "parent_history_start_frame_nominal",
                  "parent_scene_frame_nominal", "decode_end_seconds_nominal"}


def require(ok, code):
    if not ok:
        raise ValueError(code)


def safe_path(value):
    require(isinstance(value, (str, os.PathLike)), "InvalidLocalPath")
    value = os.fspath(value)
    require(value.startswith("/") and not any(ord(c) < 32 for c in value)
            and ".." not in Path(value).parts, "InvalidLocalPath")
    path = Path(value)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "SymlinkRefused")
    return path


def open_input(path, limit):
    path = safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    stream = os.fdopen(fd, "rb")
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
        stream.close()
        raise ValueError("InputSizeOrType")
    return stream


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "DuplicateJSONKey")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("NonfiniteJSON")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
        stream.write(raw)


def load_queue(path, max_items):
    require(Path(path).name == "scene_proposals.jsonl", "SceneProposalsOnly")
    selected, seen, sha, size = [], set(), hashlib.sha256(), 0
    with open_input(path, MAX_QUEUE) as stream:
        for line_number in range(1, MAX_ROWS + 2):
            raw = stream.readline(MAX_LINE + 1)
            if not raw:
                break
            size += len(raw)
            require(size <= MAX_QUEUE and len(raw) <= MAX_LINE and line_number <= MAX_ROWS, "QueueLimit")
            sha.update(raw)
            row = strict_json(raw)
            require(isinstance(row, dict) and row.get("task") in {"current_interaction", "forecast_observation"}
                    and not set(row) - PROPOSAL_FIELDS, "NotLabelBlindSceneProposal")
            if "temporal_cohort" in row:
                require(isinstance(row["temporal_cohort"], str)
                        and row["temporal_cohort"] in {"pilot", "exploration", "confirmation"},
                        "InvalidTemporalCohort")
            for field, allowed in (("media", MEDIA_FIELDS), ("request", REQUEST_FIELDS)):
                value = row.get(field)
                require(value is None or isinstance(value, dict) and not set(value) - allowed,
                        "NotLabelBlindSceneProposal")
            require(row.get("task") != "forecast_observation" or row.get("static_still_paths") == [],
                    "ForecastStaticStillsForbidden")
            key = row.get("proposal_id")
            require(isinstance(key, str) and re.fullmatch(r"[0-9a-f]{64}", key)
                    and key not in seen, "InvalidQueueIdentity")
            seen.add(key)
            if len(selected) < max_items:
                selected.append((row, line_number, digest(raw)))
    require(bool(selected), "EmptyQueue")
    return selected, {"sha256": sha.hexdigest(), "bytes": size, "rows": len(seen)}


def load_map(path):
    with open_input(path, MAX_MAP) as stream:
        raw = stream.read(MAX_MAP + 1)
    require(len(raw) <= MAX_MAP, "MapLimit")
    mapping = strict_json(raw)
    require(isinstance(mapping, dict) and len(mapping) <= MAX_ROWS, "InvalidPathMap")
    for uri, path in mapping.items():
        parsed = urlsplit(uri)
        require(parsed.scheme == "s3" and parsed.netloc and parsed.path not in ("", "/")
                and not parsed.query and not parsed.fragment and "@" not in parsed.netloc
                and not any(c.isspace() for c in uri), "InvalidMediaURI")
        safe_path(path)
    return mapping, {"sha256": digest(raw), "bytes": len(raw)}


def window(row):
    request, media = row.get("request"), row.get("media")
    require(isinstance(request, dict) and isinstance(media, dict), "MissingHistory")
    history = {**request, "s3_uri": media.get("s3_uri"),
               "parent_video_start_seconds_nominal": media.get("parent_video_start_seconds_nominal"),
               "source_kind": media.get("kind")}
    require(history.get("s3_uri") is None or isinstance(history["s3_uri"], str), "InvalidMediaURI")
    keys = ("history_start_seconds_nominal", "scene_seconds_nominal",
            "parent_video_start_seconds_nominal", "history_seconds", "fps_nominal")
    require(all(type(history.get(k)) in (int, float) and abs(history[k]) <= 86400 and math.isfinite(history[k])
                for k in keys), "InvalidHistoryClock")
    start, current, offset, seconds, fps = (history[k] for k in keys)
    require(type(request.get("decode_end_seconds_nominal")) in (int, float)
            and request["decode_end_seconds_nominal"] == current, "InvalidDecodeCutoff")
    require(type(request.get("past1_seconds_nominal")) in (int, float)
            and abs(request["past1_seconds_nominal"] - (current - 1)) <= 1e-6, "InvalidPlannerPastTime")
    require(0 <= start <= current - 2 + 1e-6 and current >= 2
            and current + offset <= 86400 and offset >= 0
            and 2 <= seconds <= 30 and 1 <= fps <= 240
            and abs(current - start - seconds) <= 1 / fps + 1e-6, "InvalidHistoryWindow")
    return {key: history[key] for key in (*keys, "past1_seconds_nominal", "s3_uri", "source_kind") if key in history}


def child_limits():
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * MIB, 64 * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    # macOS does not reliably support RLIMIT_AS; dimensions and max_alloc also cap decoding.
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * MIB, 2 * 1024 * MIB))


def run_tool(argv, deadline, *, capture=False):
    remaining = deadline - time.monotonic()
    require(remaining > 0, "WorkerTimeBudget")
    # Deliberately do not forward credentials, proxy settings, or decoder stderr.
    env = {"PATH": os.defpath, "HOME": "/nonexistent", "LANG": "C", "OMP_NUM_THREADS": "1"}
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(argv, stdin=subprocess.DEVNULL,
                stdout=output if capture else subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, timeout=min(20, remaining), check=False, preexec_fn=child_limits)
        except subprocess.TimeoutExpired:
            raise ValueError("DecodeTimeout") from None
        except OSError:
            raise ValueError("DecoderUnavailable") from None
        require(result.returncode == 0, "DecodeFailed")
        if capture:
            output.seek(0)
            raw = output.read(MAX_LINE + 1)
            require(len(raw) <= MAX_LINE, "ProbeOutputLimit")
            return strict_json(raw)


def media_pin(stream, deadline):
    sha, size = hashlib.sha256(), 0
    before = os.fstat(stream.fileno())
    while True:
        require(time.monotonic() < deadline, "WorkerTimeBudget")
        chunk = stream.read(MIB)
        if not chunk:
            break
        size += len(chunk)
        require(size <= MAX_MEDIA, "MediaSizeLimit")
        sha.update(chunk)
    after = os.fstat(stream.fileno())
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            and size == after.st_size, "MediaChanged")
    return {"sha256": sha.hexdigest(), "bytes": size, "mtime_ns": after.st_mtime_ns}


def decode(media, folder, history, deadline, ffmpeg, ffprobe):
    restrict = ["-protocol_whitelist", "file", "-format_whitelist", "mov,matroska,webm,avi",
                "-max_alloc", str(64 * MIB), "-max_pixels", str(MAX_PIXELS)]
    info = run_tool([ffprobe, "-v", "error", *restrict, "-threads", "1", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,sample_aspect_ratio:stream_tags=rotate:stream_side_data_list",
        "-of", "json", str(media)], deadline, capture=True)
    streams = info.get("streams", [])
    require(len(streams) == 1, "MissingVideoStream")
    width, height = streams[0].get("width"), streams[0].get("height")
    require(type(width) is int and type(height) is int and 0 < width <= 4096
            and 0 < height <= 4096 and width * height <= MAX_PIXELS, "FrameSizeLimit")
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", *restrict,
            "-threads", "1", "-hwaccel", "none", "-noautorotate",
            "-filter_threads", "1", "-filter_complex_threads", "1"]
    frames = []
    current = history["scene_seconds_nominal"]
    for name, seconds in (("past1", current - 2), ("past2", history["past1_seconds_nominal"]), ("current", current)):
        raw, canvas = folder / f"{name}_source.png", folder / f"{name}_scene.png"
        # Seeking exactly to a between-frame target discards its preceding frame.
        # Stream a short window and update fresh output files through its last
        # non-future frame. This avoids buffering a window of high-resolution frames.
        seek = max(0, seconds - .25)
        duration = seconds - seek + .001
        run_tool([*base, "-copyts", "-start_at_zero", "-ss", f"{seek:.9f}",
            "-t", f"{duration:.9f}", "-i", str(media), "-filter_complex",
            f"[0:v:0]select='lte(t,{seconds:.9f}+0.000000001)',split=2[raw][preview];[preview]{CANVAS}[scene]",
            "-vsync", "0", "-map", "[raw]", "-c:v", "png", "-threads", "1", "-update", "1", str(raw),
            "-map", "[scene]", "-c:v", "png", "-threads", "1", "-update", "1", str(canvas)], deadline)
        require(raw.is_file() and canvas.is_file(), "FrameUnavailable")
        frames.append({"role": name, "source_image": raw.name, "scene_image": canvas.name,
                       "media_seconds_nominal": seconds,
                       "parent_seconds_nominal": seconds + history["parent_video_start_seconds_nominal"]})
    duration = current - history["history_start_seconds_nominal"]
    clip = folder / "history_preview.mp4"
    run_tool([*base, "-ss", f"{history['history_start_seconds_nominal']:.9f}", "-i", str(media),
        "-t", f"{duration:.9f}", "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
        "-vf", CANVAS, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-threads", "1", str(clip)], deadline)
    require(clip.is_file() and clip.stat().st_size > 0, "HistoryUnavailable")
    return {"frames": frames, "history_preview": clip.name, "history_seconds": history["history_seconds"],
            "decoded_interval_seconds_nominal": duration, "source_stream": streams[0],
            "canvas": {"width": 896, "height": 672, "mode": "aspect_preserving_letterbox"},
            "orientation_policy": "stored_pixels_no_autorotate_pending_human_review",
            "frame_role_policy": "past1=current-2s; past2=request.past1_seconds_nominal=current-1s; current=cutoff",
            "frame_selection_policy": "latest_decoded_frame_not_after_nominal_target_within_0.25s_lookback",
            "timing_policy": "nominal_targets_with_preceding_frame_selection_not_verified_source_alignment"}


def build(queue, media_map, out, *, max_items=8, timeout_seconds=120):
    require(type(max_items) is int and 1 <= max_items <= 2000, "InvalidMaxItems")
    require(type(timeout_seconds) is int and 1 <= timeout_seconds <= 3600, "InvalidTimeout")
    deadline = time.monotonic() + timeout_seconds
    selected, queue_pin = load_queue(queue, max_items)
    mapping, map_pin = load_map(media_map)
    out = safe_path(out)
    require(out.parent.is_dir() and not out.exists(), "OutputExistsOrParentMissing")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    require(ffmpeg is not None and ffprobe is not None, "DecoderUnavailable")
    out.mkdir(mode=0o700)
    report = {"schema_version": 1, "status": "private_media_review_only", **GATES,
              "queue_pin": queue_pin, "media_map_pin": map_pin, "selection": "first_rows_in_queue_order",
              "max_items": max_items, "gpu_used": False, "network_used": False, "items": []}
    for row, line, row_hash in selected:
        key = row["proposal_id"]
        item = {"proposal_id": key, "task": row["task"],
                "source_video_id": row.get("source_video_id"), "temporal_split": row.get("temporal_split"),
                "queue_line": line, "queue_row_sha256": row_hash, **GATES}
        if "temporal_cohort" in row:
            item["temporal_cohort"] = row["temporal_cohort"]
        report["items"].append(item)
        folder = out / key
        created = False
        try:
            require(time.monotonic() < deadline, "WorkerTimeBudget")
            history = window(row)
            uri = history.get("s3_uri")
            if uri not in mapping:
                item.update(status="skipped_missing_media")
                continue
            media = safe_path(mapping[uri])
            try:
                stream = open_input(media, MAX_MEDIA)
            except FileNotFoundError:
                item.update(status="skipped_missing_media")
                continue
            with stream:
                pin = media_pin(stream, deadline)
                before = os.fstat(stream.fileno())
                folder.mkdir(mode=0o700)
                created = True
                assets = decode(media, folder, history, deadline, ffmpeg, ffprobe)
                after = media.stat()
                require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "MediaChanged")
            item.update(status="preview_pending_review", history=history,
                        source_media=row["media"], source_request=row["request"],
                        raw_source={"s3_uri": uri, "local_path": str(media), **pin}, assets=assets)
            write_json(folder / "metadata.json", item)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            # Only fixed codes cross the private report boundary, never decoder/path errors.
            code = str(error) if type(error) is ValueError and re.fullmatch(r"[A-Za-z]+", str(error)) else "MediaProcessingFailed"
            item.update(status="skipped_error", error_code=code)
            item.pop("assets", None)
            if created:
                shutil.rmtree(folder)
    report["counts"] = {status: sum(item["status"] == status for item in report["items"])
                        for status in ("preview_pending_review", "skipped_missing_media", "skipped_error")}
    write_json(out / "review_index.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True, help="Absolute path to scene_proposals.jsonl only.")
    parser.add_argument("--media-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    try:
        result = build(**vars(parser.parse_args(argv)))
    except (ValueError, OSError, RecursionError):
        print('{"status":"error","error_code":"MediaPreviewValidationFailed"}')
        return 1
    print(json.dumps({"status": result["status"], "items": len(result["items"]), "dataset_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
