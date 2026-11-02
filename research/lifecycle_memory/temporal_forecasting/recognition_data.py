"""Schema-4 recognition publication with byte-bound, private visual-review proofs.

Only contract version 3 is accepted. Public records contain opaque identities,
split, task/contract, images, observation_times_s, timestamps_verified and a
review_binding_sha256 of that public presentation (not of the hidden answer).
Candidate timestamps are presentation timestamps, equal to current; candidate
origin times, boxes, and raw-frame pins live only in private provenance.

Input observations require the five identity fields from data._ID_FIELDS plus
split/task/contract_sha256/images/observation_times_s/timestamps_verified and
recognition_provenance. Images have exactly {path, sha256} for all 19 slots.
Provenance version 1 contains source_media_sha256, source_time_base ("n/d"),
source_start_pts, parent_video_start_seconds, frames (17 {target_seconds, pts}
records), candidate_origins (two {source_time_seconds, bbox_xyxy,
annotation_dimensions: [width, height], raw_frame_sha256} records), and
review_evidence/review_receipt: {path, sha256}. Targets are export-local seconds; observation
and candidate-origin times are parent-video seconds. Every PTS must precede its
target by at most .1 s. The 17 uniform targets span at most 8 seconds, with
spacing at most .5 seconds and grid tolerance 1e-7. No contact labels are inferred.

Labels contain episode_id/correct_candidate/label_status/reviewer/
review_evidence_sha256/review_receipt_sha256, with status "visually_audited".
Evidence is the original temporal_event_preview.py evidence.json. A manual
receipt has exactly proposal_id, binding_sha256, reviewer, tasks; each task
entry has status (pass/reject/uncertain), correct_candidate (0/1/null), reason.
Only pass for the selected task may publish. The binding hashes canonical JSON
of the original evidence excluding binding_sha256. Paths are relative to the
observation manifest directory. No replacement review proof is fabricated.

Review evidence is supplied by the parent visual QA process. This module verifies
its consistency and byte bindings, not the truth/authenticity of a human audit.
The original evidence bytes and detailed provenance are retained inside the 0600
evaluator, so validation works after relocation without preparation files.
No media decoding, network access, inferred labels, future TTC or static labels.
"""

from fractions import Fraction
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

from . import data as d
from .common import digest
from .recognition_protocol import (INTERACTION_ORDER, NEW_INTERACTION, SCENE_SLOTS, SLOTS, recognition_contract,
                                   validate_recognition_contract, validate_recognition_protocol)


MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_PROOF_BYTES = 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_ROWS = 96
IDENTIFIERS = frozenset(d._ID_FIELDS)
PRESENTATION = frozenset({"split", "task", "contract_sha256", "images",
                          "observation_times_s", "timestamps_verified"})
LABEL_FIELDS = frozenset({"episode_id", "correct_candidate", "label_status",
                         "reviewer", "review_evidence_sha256", "review_receipt_sha256"})
PRIVATE_FIELDS = frozenset({"source_identifiers", "recognition_provenance", "review_evidence_json", "review_receipt_json"})
PROVENANCE_FIELDS = frozenset({"version", "source_media_sha256", "source_time_base",
    "source_start_pts", "parent_video_start_seconds", "frames", "candidate_origins"})
REVIEW_TASKS = {INTERACTION_ORDER: "interaction_order", NEW_INTERACTION: "new_interaction"}


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _read(path, limit=MAX_INPUT_BYTES):
    path = Path(path)
    d._regular(path)
    d._require(0 < path.stat().st_size <= limit, "recognition input size limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    d._require(len(raw) <= limit, "recognition input size limit")
    return raw


def _review_path(base, value):
    d._identifier(value)
    path = Path(value)
    d._require(".." not in path.parts, "review evidence traversal")
    path = path if path.is_absolute() else base / path
    try:
        relative = path.relative_to(base)
    except ValueError as exc:
        raise ValueError("review evidence escapes observation directory") from exc
    cursor = base
    for part in relative.parts:
        cursor /= part
        d._require(not cursor.is_symlink(), "symlink review evidence")
    d._regular(path)
    return path


def _presentation_binding(row):
    """A worker-verifiable content binding, with no answer-dependent inputs."""
    return digest({key: value for key, value in row.items() if key != "review_binding_sha256"})


def _provenance(row):
    value = row["recognition_provenance"]
    d._keys(value, PROVENANCE_FIELDS | {"review_evidence", "review_receipt"})
    d._require(type(value["version"]) is int and value["version"] == 1, "invalid recognition provenance version")
    d._digest(value["source_media_sha256"])
    tb = value["source_time_base"]
    d._require(isinstance(tb, str) and re.fullmatch(r"[1-9][0-9]{0,11}/[1-9][0-9]{0,11}", tb),
               "invalid recognition time base")
    tb = Fraction(tb)
    d._require(0 < tb <= 1, "invalid recognition time base")
    start_pts, offset = value["source_start_pts"], value["parent_video_start_seconds"]
    d._require(type(start_pts) is int and _finite(offset) and offset >= 0, "invalid source PTS/offset")
    d._keys(value["frames"], SCENE_SLOTS)
    previous_pts = previous_target = None
    targets = []
    for slot in SCENE_SLOTS:
        frame = value["frames"][slot]
        d._keys(frame, {"target_seconds", "pts"})
        target, pts = frame["target_seconds"], frame["pts"]
        d._require(_finite(target) and target >= 0 and type(pts) is int, "invalid recognition frame timestamp")
        actual = (pts - start_pts) * tb
        d._require(actual >= 0 and 0 <= Fraction(str(target)) - actual <= Fraction(1, 10),
                   "recognition PTS must precede target within .1 seconds")
        d._require(previous_pts is None or (pts > previous_pts and target > previous_target),
                   "recognition PTS/targets must be strictly ordered")
        previous_pts, previous_target = pts, target
        targets.append(target)
        d._require(math.isclose(row["observation_times_s"][slot], float(actual) + offset,
                                rel_tol=0, abs_tol=1e-8), "recognition observation PTS mismatch")
    span = targets[-1] - targets[0]
    d._require(span <= 8 + 1e-7
               and all(b - a <= .5 + 1e-7 for a, b in zip(targets, targets[1:])),
               "recognition target span/spacing exceeds bounds")
    d._require(all(math.isclose(target, targets[0] + span * index / 16, rel_tol=0, abs_tol=1e-7)
                   for index, target in enumerate(targets)), "recognition targets must form a uniform 17-point grid")
    d._keys(value["candidate_origins"], {"candidate0", "candidate1"})
    for origin in value["candidate_origins"].values():
        d._keys(origin, {"source_time_seconds", "bbox_xyxy", "annotation_dimensions", "raw_frame_sha256"})
        d._require(_finite(origin["source_time_seconds"]) and origin["source_time_seconds"] >= offset,
                   "invalid candidate source time")
        d._digest(origin["raw_frame_sha256"])
        dimensions = origin["annotation_dimensions"]
        d._require(isinstance(dimensions, list) and len(dimensions) == 2
                   and all(type(v) is int and v > 0 for v in dimensions), "invalid annotation dimensions")
        box = origin["bbox_xyxy"]
        d._require(isinstance(box, list) and len(box) == 4 and all(_finite(v) for v in box)
                   and 0 <= box[0] < box[2] <= dimensions[0]
                   and 0 <= box[1] < box[3] <= dimensions[1], "invalid candidate origin box")
    for key in ("review_evidence", "review_receipt"):
        pin = value[key]
        d._keys(pin, {"path", "sha256"})
        d._identifier(pin["path"])
        d._digest(pin["sha256"])


def observations(rows, public=False, contract=None):
    """Shared strict validator for preparation, release and worker public rows."""
    validate_recognition_contract(contract)
    d._require(isinstance(rows, list) and 0 < len(rows) <= MAX_ROWS, "recognition row limit")
    episodes, assignments = set(), {key: {} for key in IDENTIFIERS - {"episode_id"}}
    fields = IDENTIFIERS | PRESENTATION | {"review_binding_sha256" if public else "recognition_provenance"}
    for row in rows:
        d._keys(row, fields)
        d._require(row["task"] == contract["task"] and row["contract_sha256"] == digest(contract),
                   "recognition task/contract mismatch")
        for key in IDENTIFIERS:
            value = row[key]
            d._identifier(value)
            if public:
                d._require(bool(d._OPAQUE.fullmatch(value)), "recognition public IDs must be opaque")
            if key in ("cluster_id", "video_id"):
                d._require(value.strip().lower() not in {"unknown", "none", "null", "n/a", "na", "unverified", "?"},
                           "unknown recognition wearer/video")
        d._require(row["episode_id"] not in episodes, "duplicate recognition episode")
        episodes.add(row["episode_id"])
        d._require(row["split"] in d.SPLITS, "invalid recognition split")
        for key, assigned in assignments.items():
            d._require(assigned.setdefault(row[key], row["split"]) == row["split"], "cross-split identity overlap")
        d._require(row["timestamps_verified"] is True, "recognition timestamps must be verified")
        times = row["observation_times_s"]
        d._keys(times, SLOTS)
        d._require(all(_finite(t) and t >= 0 for t in times.values()), "invalid recognition observation times")
        d._require(all(times[a] < times[b] for a, b in zip(SCENE_SLOTS, SCENE_SLOTS[1:])),
                   "recognition scene timestamps must be strictly ordered")
        d._require(times["candidate0"] == times["candidate1"] == times["current"],
                   "candidate presentation clocks must equal current; origins are private")
        d._keys(row["images"], SLOTS)
        for image in row["images"].values():
            d._keys(image, {"path", "sha256", "bytes"} if public else {"path", "sha256"})
            d._identifier(image["path"])
            d._digest(image["sha256"])
            if public:
                d._require(type(image["bytes"]) is int and image["bytes"] > 0, "invalid image bytes")
                suffix = Path(image["path"]).suffix
                d._require(suffix in d.EXTENSIONS and image["path"] == f"assets/{image['sha256']}{suffix}",
                           "nonopaque recognition asset path")
        if public:
            d._digest(row["review_binding_sha256"])
            d._require(row["review_binding_sha256"] == _presentation_binding(row), "recognition review binding mismatch")
        else:
            _provenance(row)


def labels(rows, episodes, contract):
    validate_recognition_contract(contract)
    d._require(isinstance(rows, list) and len(rows) <= MAX_ROWS, "recognition label limit")
    result = {}
    for row in rows:
        d._keys(row, LABEL_FIELDS, PRIVATE_FIELDS)
        extras = set(row) & PRIVATE_FIELDS
        d._require(not extras or extras == PRIVATE_FIELDS, "incomplete private recognition metadata")
        d._identifier(row["episode_id"])
        d._identifier(row["reviewer"])
        d._digest(row["review_evidence_sha256"])
        d._digest(row["review_receipt_sha256"])
        d._require(type(row["correct_candidate"]) is int and row["correct_candidate"] in (0, 1),
                   "invalid recognition correct_candidate")
        d._require(row["label_status"] == "visually_audited", "recognition requires visually audited labels")
        d._require(row["episode_id"] not in result, "duplicate recognition label")
        result[row["episode_id"]] = row
    d._require(result.keys() == set(episodes), "missing or extra recognition labels")
    return result


def _review(evidence_raw, receipt_raw, observation, label, images):
    value = observation["recognition_provenance"]
    for key, raw in (("review_evidence", evidence_raw), ("review_receipt", receipt_raw)):
        d._require(len(raw) <= MAX_PROOF_BYTES, "review evidence size limit")
        d._require(d._sha(raw) == value[key]["sha256"] == label[key + "_sha256"], "review evidence/receipt hash mismatch")
    proof, receipt = d._json(evidence_raw), d._json(receipt_raw)
    d._keys(proof, {"version", "status", "proposal_id", "source_media_sha256", "source_time_base",
                    "source_start_pts", "parent_video_start_seconds", "orientation", "tasks", "candidates", "binding_sha256"})
    d._require(type(proof["version"]) is int and proof["version"] == 1 and proof["status"] == "unreviewed",
               "invalid original preview evidence")
    d._require(proof["orientation"] == "stored_pixels_no_autorotate", "unexpected review orientation")
    d._digest(proof["proposal_id"])
    d._require(proof["binding_sha256"] == digest({k: v for k, v in proof.items() if k != "binding_sha256"}),
               "preview evidence binding mismatch")
    for key in ("source_media_sha256", "source_time_base", "source_start_pts", "parent_video_start_seconds"):
        d._require(d._canonical(proof[key]) == d._canonical(value[key]), "review source provenance mismatch")
    d._keys(receipt, {"proposal_id", "binding_sha256", "reviewer", "tasks"})
    d._require(receipt["proposal_id"] == proof["proposal_id"]
               and receipt["binding_sha256"] == proof["binding_sha256"], "manual receipt binding mismatch")
    d._require(receipt["reviewer"] == label["reviewer"], "reviewer mismatch")
    task = REVIEW_TASKS[observation["task"]]
    for container in (proof["tasks"], receipt["tasks"]):
        d._require(isinstance(container, dict) and task in container
                   and container.keys() <= set(REVIEW_TASKS.values()), "missing or unknown reviewed task")
    for entry in receipt["tasks"].values():
        d._keys(entry, {"status", "correct_candidate", "reason"})
        d._require(entry["status"] in ("pass", "reject", "uncertain") and isinstance(entry["reason"], str),
                   "invalid manual task review")
        d._require(entry["correct_candidate"] is None or
                   (type(entry["correct_candidate"]) is int and entry["correct_candidate"] in (0, 1)), "invalid manual review answer")
    selected = receipt["tasks"][task]
    d._require(selected["status"] == "pass" and selected["correct_candidate"] == label["correct_candidate"],
               "selected recognition task must pass review with matching answer")
    frames = proof["tasks"][task]
    d._require(isinstance(frames, list) and len(frames) == len(SCENE_SLOTS), "review needs 17 scene frames")
    for slot, frame in zip(SCENE_SLOTS, frames):
        d._keys(frame, {"slot", "path", "sha256", "target_seconds", "pts", "parent_seconds"})
        d._require(frame["slot"] == slot and frame["sha256"] == images[slot]["sha256"], "review scene image binding mismatch")
        d._require(d._canonical({key: frame[key] for key in ("target_seconds", "pts")}) == d._canonical(value["frames"][slot])
                   and d._canonical(frame["parent_seconds"]) == d._canonical(observation["observation_times_s"][slot]),
                   "review frame timestamps mismatch")
    d._require(isinstance(proof["candidates"], list) and len(proof["candidates"]) == 2, "review needs two candidate origins")
    for slot, candidate in zip(("candidate0", "candidate1"), proof["candidates"]):
        d._keys(candidate, {"path", "sha256", "pts", "source_time_seconds", "annotation_time_seconds",
                            "bbox_xyxy", "annotation_dimensions", "raw_frame_sha256"})
        origin = value["candidate_origins"][slot]
        d._require(candidate["sha256"] == images[slot]["sha256"], "review candidate image binding mismatch")
        d._require(d._canonical({key: candidate[key] for key in origin}) == d._canonical(origin), "review candidate origin mismatch")
        d._require(type(candidate["pts"]) is int and _finite(candidate["annotation_time_seconds"]), "invalid candidate PTS/annotation time")
        actual = float((candidate["pts"] - value["source_start_pts"]) * Fraction(value["source_time_base"])) + value["parent_video_start_seconds"]
        d._require(math.isclose(actual, origin["source_time_seconds"], rel_tol=0, abs_tol=1e-8)
                   and -1e-8 <= candidate["annotation_time_seconds"] - actual <= .1 + 1e-8, "candidate origin PTS mismatch")
    return evidence_raw.decode("utf-8"), receipt_raw.decode("utf-8"), proof["proposal_id"]


def _overlap(rows, registry):
    sets = {key: set(values) for key, values in registry.items()}
    matches, by_split = {key: set() for key in d._EXPOSURE_FIELDS}, dict.fromkeys(d.SPLITS, 0)
    for row in rows:
        overlapping = False
        for key, domain in d._ID_FIELDS.items():
            if row[key] in sets[domain]:
                matches[domain].add(row[key])
                overlapping = True
        for image in row["images"].values():
            if image["sha256"] in sets["asset_sha256"]:
                matches["asset_sha256"].add(image["sha256"])
                overlapping = True
        by_split[row["split"]] += int(overlapping)
    return {"policy": "recognition_descriptive_reuse", "fresh_confirmation_claimed": False,
            "rows_with_overlap": sum(by_split.values()), "rows_with_overlap_by_original_split": by_split,
            "unique_matches_by_domain": {key: len(value) for key, value in matches.items()}}


def _attrition(rows):
    return {"input_observations": len(rows), "input_labels": len(rows), "accepted": len(rows),
            "rejected": 0, "rejected_by_reason": {}, "label_status_counts": {"visually_audited": len(rows)},
            "scope": "preparation_only_not_independent_visual_audit"}


def prepare_recognition_dataset(observations_path, labels_path, out_dir, exposure_path=None,
                                protocol=None, historical_static_path=None):
    """Atomically publish a new private release; return its version-4 manifest."""
    validate_recognition_protocol(protocol)
    contract = recognition_contract(protocol)
    observations_path, labels_path, out_dir = map(Path, (observations_path, labels_path, out_dir))
    d._require(not os.path.lexists(out_dir), "output already exists")
    d._require(exposure_path is not None and historical_static_path is not None,
               "recognition requires historical-static and temporal exposure registries")
    observations_raw, labels_raw = _read(observations_path), _read(labels_path)
    supplied_rows = d._jsonl(observations_raw)
    observations(supplied_rows, contract=contract)
    supplied_labels = d._jsonl(labels_raw)
    for row in supplied_labels:
        d._keys(row, LABEL_FIELDS)
    indexed = labels(supplied_labels, [row["episode_id"] for row in supplied_rows], contract)
    exposure_raw, history_raw = _read(exposure_path), _read(historical_static_path)
    exposure, history = d._exposure(d._json(exposure_raw)), d._exposure(d._json(history_raw))
    policy = contract["exposure_policy"]
    d._require(d._sha(exposure_raw) == policy["temporal_exposure_sha256"], "temporal raw registry hash mismatch")
    d._require(d._sha(history_raw) == policy["historical_static_sha256"], "historical raw registry hash mismatch")
    base = observations_path.parent.resolve()
    identities = {domain: {} for domain in d._ID_FIELDS.values()}

    def opaque(domain, value):
        mapping = identities[domain]
        if value not in mapping:
            mapping[value] = uuid.uuid4().hex
        return mapping[value]

    parent = out_dir.parent.resolve(strict=True)
    out_dir = parent / out_dir.name
    lock = parent / ("." + out_dir.name + ".prepare.lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    stage = None
    try:
        d._require(not os.path.lexists(out_dir), "output already exists")
        stage = Path(tempfile.mkdtemp(prefix="." + out_dir.name + ".prepare-", dir=parent))
        (stage / "assets").mkdir()
        public, private, reviewed_tasks = [], [], set()
        for supplied in supplied_rows:
            row = {key: opaque(domain, supplied[key]) for key, domain in d._ID_FIELDS.items()}
            row.update({key: supplied[key] for key in PRESENTATION - {"images"}})
            row["images"] = {}
            for slot in SLOTS:
                image = supplied["images"][slot]
                source = d._asset(base, image["path"])
                d._require(source.stat().st_size <= MAX_IMAGE_BYTES, "recognition image size limit")
                pin = d._file_sha(source)
                d._require(pin == image["sha256"], "source asset hash mismatch")
                relative = f"assets/{pin}{source.suffix.lower()}"
                target = stage / relative
                if not target.exists():
                    shutil.copyfile(source, target)
                d._require(d._file_sha(target) == pin, "copied asset hash mismatch")
                row["images"][slot] = {"path": relative, "sha256": pin, "bytes": target.stat().st_size}
            label = indexed[supplied["episode_id"]]
            evidence = supplied["recognition_provenance"]["review_evidence"]
            proof_raw = _read(_review_path(base, evidence["path"]), MAX_PROOF_BYTES)
            receipt = supplied["recognition_provenance"]["review_receipt"]
            receipt_raw = _read(_review_path(base, receipt["path"]), MAX_PROOF_BYTES)
            proof_text, receipt_text, proposal_id = _review(proof_raw, receipt_raw, supplied, label, row["images"])
            review_key = (proposal_id, supplied["task"])
            d._require(review_key not in reviewed_tasks, "duplicate reviewed proposal/task")
            reviewed_tasks.add(review_key)
            row["review_binding_sha256"] = _presentation_binding(row)
            public.append(row)
            private.append(dict(label, episode_id=row["episode_id"],
                source_identifiers={key: supplied[key] for key in IDENTIFIERS},
                recognition_provenance={k: v for k, v in supplied["recognition_provenance"].items() if k in PROVENANCE_FIELDS},
                review_evidence_json=proof_text, review_receipt_json=receipt_text))
        registries = []
        for registry in (exposure, history):
            registries.append({key: sorted(values if key == "asset_sha256" else [opaque(key, v) for v in values])
                               for key, values in registry.items()})
        exposure, history = registries
        public_raw, private_raw = d._lines(public), d._lines(private)
        d._require(max(len(public_raw), len(private_raw)) <= MAX_INPUT_BYTES, "recognition release metadata size limit")
        (stage / "public.jsonl").write_bytes(public_raw)
        with (stage / "evaluator.jsonl").open("xb") as stream:
            os.chmod(stream.fileno(), 0o600)
            stream.write(private_raw)
        counts, splits = d._summary(public)
        manifest = {"version": 4, "contract": contract, "public_sha256": d._sha(public_raw),
            "evaluator_sha256": d._sha(private_raw), "counts": counts, "splits": splits,
            "asset_inventory": d._inventory(public),
            "provenance": {"observations_sha256": d._sha(observations_raw), "labels_sha256": d._sha(labels_raw),
                "exposure_sha256": d._sha(exposure_raw), "exposure": exposure,
                "historical_static_sha256": d._sha(history_raw), "historical_static_exposure": history,
                "annotation_source": "visually_audited_recognition", "attrition": _attrition(public),
                "temporal_exposure_overlap": _overlap(public, exposure)}}
        (stage / "manifest.json").write_text(d._canonical(manifest) + "\n", encoding="utf-8")
        validate_recognition_dataset(stage, verify_assets=True)
        d._require(not os.path.lexists(out_dir), "output appeared during preparation")
        os.rename(stage, out_dir)
        stage = None
        return manifest
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.unlink()


def validate_recognition_dataset(root, split=None, verify_assets=False):
    """Validate the complete release and private review binding; return public rows only."""
    d._require(split is None or split in d.SPLITS, "invalid split")
    root = Path(root)
    d._require(not root.is_symlink() and root.is_dir(), "invalid release directory")
    root = root.resolve()
    d._require({item.name for item in root.iterdir()} == {"manifest.json", "public.jsonl", "evaluator.jsonl", "assets"},
               "unexpected or missing release entries")
    manifest = d._json(_read(root / "manifest.json"))
    d._keys(manifest, {"version", "contract", "public_sha256", "evaluator_sha256", "counts", "splits", "asset_inventory", "provenance"})
    d._require(type(manifest["version"]) is int and manifest["version"] == 4, "recognition requires manifest version 4")
    contract = validate_recognition_contract(manifest["contract"])
    public_raw, private_raw = _read(root / "public.jsonl"), _read(root / "evaluator.jsonl")
    for key, raw in (("public_sha256", public_raw), ("evaluator_sha256", private_raw)):
        d._digest(manifest[key])
        d._require(manifest[key] == d._sha(raw), "release metadata hash mismatch")
    rows = d._jsonl(public_raw)
    observations(rows, public=True, contract=contract)
    indexed = labels(d._jsonl(private_raw), [row["episode_id"] for row in rows], contract)
    reconstructed, reviewed_tasks = [], set()
    identity_maps = {key: {} for key in IDENTIFIERS}
    reverse_identity_maps = {key: {} for key in IDENTIFIERS}
    for row in rows:
        label = indexed[row["episode_id"]]
        d._require(PRIVATE_FIELDS <= label.keys(), "missing private review proof")
        identifiers = label["source_identifiers"]
        d._keys(identifiers, IDENTIFIERS)
        for key, value in identifiers.items():
            d._identifier(value)
            d._require(identity_maps[key].setdefault(value, row[key]) == row[key], "private source identity mapping mismatch")
            d._require(reverse_identity_maps[key].setdefault(row[key], value) == value, "private source identity mapping is not one-to-one")
        d._keys(label["recognition_provenance"], PROVENANCE_FIELDS)
        supplied = {**identifiers, **{key: row[key] for key in PRESENTATION - {"images"}},
            "images": {slot: {key: image[key] for key in ("path", "sha256")} for slot, image in row["images"].items()},
            "recognition_provenance": {**label["recognition_provenance"],
                "review_evidence": {"path": "retained-private-evidence.json", "sha256": label["review_evidence_sha256"]},
                "review_receipt": {"path": "retained-private-receipt.json", "sha256": label["review_receipt_sha256"]}}}
        reconstructed.append(supplied)
        d._require(isinstance(label["review_evidence_json"], str) and isinstance(label["review_receipt_json"], str),
                   "missing review evidence/receipt bytes")
        _, _, proposal_id = _review(label["review_evidence_json"].encode("utf-8"),
                                    label["review_receipt_json"].encode("utf-8"), supplied, label, row["images"])
        review_key = (proposal_id, row["task"])
        d._require(review_key not in reviewed_tasks, "duplicate reviewed proposal/task")
        reviewed_tasks.add(review_key)
    observations(reconstructed, contract=contract)
    counts, splits = d._summary(rows)
    d._require(d._canonical(counts) == d._canonical(manifest["counts"]), "counts mismatch")
    d._require(d._canonical(splits) == d._canonical(manifest["splits"]), "split hashes/counts mismatch")
    inventory = d._inventory(rows)
    d._require(inventory == manifest["asset_inventory"], "asset inventory mismatch")
    assets = root / "assets"
    d._require(not assets.is_symlink() and assets.is_dir(), "invalid assets directory")
    d._require({item.name for item in assets.iterdir()} == {Path(item["path"]).name for item in inventory}, "physical inventory mismatch")
    for item in inventory:
        path = d._asset(root, item["path"])
        d._require(path.stat().st_size == item["bytes"] <= MAX_IMAGE_BYTES, "asset size mismatch")
        # Review is of presented bytes, so schema 4 never skips physical hashes.
        d._require(d._file_sha(path) == item["sha256"], "physical asset hash mismatch")
    provenance = manifest["provenance"]
    d._keys(provenance, {"observations_sha256", "labels_sha256", "exposure_sha256", "exposure",
        "historical_static_sha256", "historical_static_exposure", "annotation_source", "attrition", "temporal_exposure_overlap"})
    for key in ("observations_sha256", "labels_sha256", "exposure_sha256", "historical_static_sha256"):
        d._digest(provenance[key])
    d._require(provenance["annotation_source"] == "visually_audited_recognition", "unsupported recognition annotation source")
    policy = contract["exposure_policy"]
    d._require(provenance["exposure_sha256"] == policy["temporal_exposure_sha256"]
               and provenance["historical_static_sha256"] == policy["historical_static_sha256"], "raw registry pin mismatch")
    for key in ("exposure", "historical_static_exposure"):
        registry = provenance[key]
        d._keys(registry, d._EXPOSURE_FIELDS)
        d._exposure({"version": 1, **registry})
        for domain, values in registry.items():
            if domain != "asset_sha256":
                d._require(all(d._OPAQUE.fullmatch(v) for v in values), "exposure identities must be opaque")
    d._require(d._canonical(provenance["temporal_exposure_overlap"]) == d._canonical(_overlap(rows, provenance["exposure"])),
               "recognition exposure disclosure mismatch")
    d._require(d._canonical(provenance["attrition"]) == d._canonical(_attrition(rows)), "recognition attrition mismatch")
    return {"manifest": manifest, "rows": [row for row in rows if split is None or row["split"] == split]}
