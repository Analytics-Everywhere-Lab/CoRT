"""Portable, fail-closed preparation of supplied Ego4D STA observations/labels.

Input JSONL (one object per line; unknown fields are errors)::

    {"episode_id":"e1", "cluster_id":"wearer1", "video_id":"v1",
     "split":"pilot", "candidate_pair_id":"pair1",
     "pair_frozen_before_label_review":true, "annotation_source":"ego4d_sta",
     "observation_times_s":{"past1":0,"past2":1,"current":2,
                            "candidate0":2,"candidate1":2},
     "timestamps_verified":true, "candidates_visible_at_cutoff":true,
     "images":{"past1":{"path":"frames/a.png"},
               "past2":{"path":"frames/b.png"},
               "current":{"path":"frames/c.png"},
               "candidate0":{"path":"crops/a.png"},
               "candidate1":{"path":"crops/b.png"}}}
    {"episode_id":"e1", "correct_candidate":0, "time_to_contact_s":0.5,
     "label_status":"official"}

The first object belongs in observations.jsonl, the second in labels.jsonl.
Both accept optional string event_id/source. Image objects accept optional
sha256. Paths are relative to the observation manifest directory (absolute
paths within that directory also work); traversal, symlinks and nonregular
assets are forbidden. Supported suffixes: .png, .jpg, .jpeg, .webp. Bytes are
copied unchanged, not decoded or sanitized; visual/temporal/crop audits remain
the data owner's responsibility. No annotations, crops or labels are inferred.

observation_times_s is required and public: all five values are finite,
nonnegative seconds on the same source-video clock, past1 < past2 < current,
and both candidate timestamps equal current exactly, matching the requirement
that both crops come from the current cutoff image. They are observation timestamps,
never contact times or time-to-contact labels. Required timestamps_verified=true
and candidates_visible_at_cutoff=true are explicit caller attestations, NOT
automatically proven audits or substitutes for reviewing the source footage.
The builder verifies declared temporal consistency, not the pixels' actual
capture times or crop provenance. No separate audit receipt is inferred.
Fixed canvas geometry for scenes/crops is verified during execution, not here;
preparation does not resize images.

An exposure registry is mandatory for confirmation and has an explicit schema::

    {"version":1, "participants":["previous-wearer"], "videos":["old-video"],
     "episodes":[], "events":[], "candidate_pairs":[], "asset_sha256":[]}

participants/videos are required arrays of unique source IDs; the remaining
arrays are optional. This is NOT the historical exposure-registry schema.
cluster_id must be a known wearer identity, not a guessed video surrogate.
Registry completeness and real-world identity reconciliation are caller audits.
The schema-3 dense CURRENT contract is descriptive matched reuse: split names
retain original allocation provenance, not fresh confirmation claims. Its pinned
temporal registry remains mandatory, but overlap is disclosed in validated
manifest counts instead of rejected. Schema-1/2 confirmation rules are unchanged.

Output is a new directory containing public.jsonl, private evaluator.jsonl,
manifest.json (version 1), and deduplicated assets/<sha256>.<ext>. All public
identifiers are randomized opaque IDs, and public images contain exactly path,
sha256, bytes. Source/event prose is never public. evaluator.jsonl is mode 0600;
the release is private by default. Do NOT stage this whole directory to model
workers: stage only public.jsonl and its assets. Same-account permissions are
not an adversarial security boundary.

prepare_dataset returns the manifest. validate_dataset returns only manifest
and public rows, optionally filtered by split AFTER validating the entire
release. Metadata, evaluator hash/schema, inventory and file sizes are always
checked; verify_assets=True also hashes every physical image. Split hashes use
canonical JSONL of public rows in release order, including trailing newlines.
Hashes provide integrity, not authenticity against coordinated rewriting.

All accepted rows are retained: schema failures abort instead of silently
filtering. Manifest attrition describes this preparation step only, not upstream
annotation yield or a scientific audit certificate. Official STA interaction
labels are accepted without inventing a stricter direct-hand-contact taxonomy
(Oct 23 design feedback). No synthetic fallback or STA-schema guessing exists.

Publication uses a same-filesystem temporary directory and atomic rename under
an exclusive sibling publication lock. Existing outputs, including empty or
read-only directories and broken symlinks, are never reused. Cooperating callers
cannot overwrite one another; the parent directory must be trusted against
noncooperating concurrent replacement. Failed builds remove their own staging
directory and lock, leaving source inputs untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import uuid
from fractions import Fraction

from .protocol import (CURRENT, protocol_contract, validate_contract, validate_protocol,
                       validate_rows_contract)
from .common import digest


SLOTS = ("past1", "past2", "current", "candidate0", "candidate1")
DENSE_HISTORY_OFFSETS_S = (-2, -1.75, -1.5, -1.25, -1, -.75, -.5, -.25, 0)
DENSE_SCENE_SLOTS = tuple(f"past{i}" for i in range(1, 9)) + ("current",)
DENSE_SLOTS = DENSE_SCENE_SLOTS + ("candidate0", "candidate1")
SPLITS = ("pilot", "exploration", "confirmation")
EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_OBS_FIELDS = frozenset({
    "episode_id", "cluster_id", "video_id", "split", "images",
    "pair_frozen_before_label_review", "annotation_source", "candidate_pair_id",
    "observation_times_s", "timestamps_verified", "candidates_visible_at_cutoff",
})
_LABEL_FIELDS = frozenset({
    "episode_id", "correct_candidate", "time_to_contact_s", "label_status",
})
_OPTIONAL = frozenset({"event_id", "source"})
_EXPOSURE_FIELDS = (
    "participants", "videos", "episodes", "events", "candidate_pairs", "asset_sha256",
)
_ID_FIELDS = {
    "episode_id": "episodes", "cluster_id": "participants", "video_id": "videos",
    "candidate_pair_id": "candidate_pairs", "event_id": "events",
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_OPAQUE = re.compile(r"[0-9a-f]{32}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, required, optional=()):
    _require(isinstance(value, dict), "expected a JSON object")
    _require(set(required) <= value.keys(), "missing required fields")
    _require(value.keys() <= set(required) | set(optional), "unexpected fields")


def _identifier(value):
    _require(isinstance(value, str) and bool(value.strip()), "invalid identifier/string")


def _digest(value):
    _require(isinstance(value, str) and _HEX.fullmatch(value), "invalid sha256")


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError("nonfinite JSON number")


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_bad_constant)


def _regular(path):
    _require(stat.S_ISREG(path.lstat().st_mode), "expected regular nonsymlink file")


def _read(path):
    _regular(path)
    return path.read_bytes()


def _jsonl(raw):
    rows = []
    for line in raw.decode("utf-8").splitlines():
        _require(bool(line.strip()), "blank JSONL row")
        row = _json(line)
        _require(isinstance(row, dict), "JSONL row must be an object")
        rows.append(row)
    return rows


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _lines(rows):
    return "".join(_canonical(row) + "\n" for row in rows).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(base, value):
    _identifier(value)
    path = Path(value)
    _require(".." not in path.parts, "asset traversal is forbidden")
    path = path if path.is_absolute() else base / path
    try:
        relative = path.relative_to(base)
    except ValueError as exc:
        raise ValueError("asset escapes manifest directory") from exc
    cursor = base
    for part in relative.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), "symlink assets are forbidden")
    _regular(path)
    _require(path.suffix.lower() in EXTENSIONS, "unsupported image extension")
    _require(path.stat().st_size > 0, "empty image asset")
    return path


def slots_for_contract(contract):
    """Only the version-2 fixed-current contract opts into eleven image slots."""
    if contract is None:
        return SLOTS
    validate_contract(contract)
    if contract["version"] == 3:
        from .recognition_protocol import SLOTS as RECOGNITION_SLOTS

        return RECOGNITION_SLOTS
    if contract["version"] == 1:
        return SLOTS
    _require(contract["version"] == 2 and contract["task"] == CURRENT
             and contract.get("history_offsets_s") == list(DENSE_HISTORY_OFFSETS_S)
             and contract.get("patching") == "fixed_current_history_v1",
             "unsupported dense current contract")
    return DENSE_SLOTS


def _dense_provenance(row):
    """Check measured PTS and immutable static pins, not scientific audit truth."""
    value = row["dense_current_provenance"]
    _keys(value, {"version", "source_media_sha256", "source_time_base", "source_start_pts",
                  "parent_video_start_seconds", "frames", "fixed_current"})
    _require(type(value["version"]) is int and value["version"] == 1,
             "invalid dense provenance version")
    _digest(value["source_media_sha256"])
    _require(isinstance(value["source_time_base"], str)
             and re.fullmatch(r"[1-9][0-9]{0,11}/[1-9][0-9]{0,11}", value["source_time_base"]),
             "invalid dense time base")
    time_base = Fraction(value["source_time_base"])
    _require(0 < time_base <= 1, "invalid dense time base")
    _require(type(value["source_start_pts"]) is int, "invalid dense start PTS")
    offset = value["parent_video_start_seconds"]
    _require(type(offset) in (int, float) and math.isfinite(offset) and offset >= 0,
             "invalid dense parent offset")
    _keys(value["frames"], DENSE_SCENE_SLOTS)
    times = row["observation_times_s"]
    previous = None
    for slot, delta in zip(DENSE_SCENE_SLOTS, DENSE_HISTORY_OFFSETS_S):
        frame = value["frames"][slot]
        _keys(frame, {"target_seconds", "pts"})
        target, pts = frame["target_seconds"], frame["pts"]
        _require(type(target) in (int, float) and math.isfinite(target) and target >= 0
                 and type(pts) is int, "invalid dense frame timestamp")
        _require(math.isclose(target + offset, times["current"] + delta,
                              rel_tol=0, abs_tol=1e-8), "dense target grid mismatch")
        actual = (pts - value["source_start_pts"]) * time_base
        target_q = Fraction(str(target))
        _require(0 <= target_q - actual <= Fraction(1, 4),
                 "dense frame PTS must precede target within lookback")
        _require(previous is None or pts > previous, "dense PTS must be strictly ordered")
        previous = pts
        if slot != "current":
            _require(math.isclose(times[slot], float(actual) + offset,
                                  rel_tol=0, abs_tol=1e-8), "dense observation PTS mismatch")
    fixed = value["fixed_current"]
    _keys(fixed, {"sha256", "candidate_sha256", "static_record_sha256", "source_frame_unit"})
    _digest(fixed["sha256"])
    _digest(fixed["static_record_sha256"])
    _require(type(fixed["source_frame_unit"]) is int and fixed["source_frame_unit"] >= 0,
             "invalid original static frame")
    _require(isinstance(fixed["candidate_sha256"], list) and len(fixed["candidate_sha256"]) == 2,
             "invalid original candidate pins")
    for slot, pin in zip(("current", "candidate0", "candidate1"),
                         [fixed["sha256"], *fixed["candidate_sha256"]]):
        _digest(pin)
        _require(row["images"][slot].get("sha256") == pin, "fixed static asset pin mismatch")


def _observations(rows, public=False, contract=None):
    if contract is not None and type(contract.get("version")) is int and contract["version"] == 3:
        from .recognition_data import observations

        return observations(rows, public=public, contract=contract)
    _require(bool(rows), "empty dataset")
    slots = slots_for_contract(contract)
    dense = slots == DENSE_SLOTS
    current = contract is not None and contract["task"] == CURRENT
    fields = set(_OBS_FIELDS)
    if current:
        fields.remove("pair_frozen_before_label_review")
        fields.add("static_pair_preserved")
    if contract:
        fields.update(("task", "event_id"))
        if public:
            fields.add("contract_sha256")
    if dense:
        fields.add("dense_current_provenance")
    episodes = set()
    assignments = {field: {} for field in _ID_FIELDS if field != "episode_id"}
    for row in rows:
        _keys(row, fields, {"event_id"} if public else _OPTIONAL)
        if contract:
            _require(row["task"] == contract["task"], "observation task mismatch")
            if public:
                validate_rows_contract([row], contract)
        for field in _ID_FIELDS:
            if field not in row:
                continue
            value = row[field]
            _identifier(value)
            if public:
                _require(bool(_OPAQUE.fullmatch(value)), "public IDs must be opaque")
            if field in {"cluster_id", "video_id"}:
                _require(value.strip().lower() not in {
                    "unknown", "none", "null", "n/a", "na", "unverified", "?",
                }, "unknown wearer/video identity")
        if "source" in row:
            _identifier(row["source"])
        _require(row["episode_id"] not in episodes, "duplicate episode")
        episodes.add(row["episode_id"])
        _require(row["split"] in SPLITS, "invalid split")
        for field, assigned in assignments.items():
            if field in row:
                previous = assigned.setdefault(row[field], row["split"])
                _require(previous == row["split"], "cross-split identity overlap")
        if current:
            _require(row["static_pair_preserved"] is True, "static candidate pair must be preserved")
        else:
            _require(row["pair_frozen_before_label_review"] is True, "candidate pair not frozen")
        _require(row["annotation_source"] == (
            "audited_static_current_interaction" if current else "ego4d_sta"), "unsupported annotation source")
        _require(row["timestamps_verified"] is True, "timestamps must be verified by caller")
        _require(row["candidates_visible_at_cutoff"] is True,
                 "candidate visibility must be verified by caller")
        times = row["observation_times_s"]
        _keys(times, slots)
        for value in times.values():
            _require(type(value) in (int, float) and value >= 0,
                     "observation timestamps must be nonnegative numbers")
            if isinstance(value, float):
                _require(math.isfinite(value), "observation timestamps must be finite")
        scene_slots = slots[:-2]
        _require(all(times[a] < times[b] for a, b in zip(scene_slots, scene_slots[1:])),
                 "past timestamps must be strictly ordered before current")
        _require(times["candidate0"] == times["current"]
                 and times["candidate1"] == times["current"],
                 "candidate timestamps must equal the current observation cutoff")
        _keys(row["images"], slots)
        for image in row["images"].values():
            _keys(image, {"path", "sha256", "bytes"} if public else {"path"},
                  () if public else {"sha256"})
            _identifier(image["path"])
            if "sha256" in image:
                _digest(image["sha256"])
            if public:
                _require(type(image["bytes"]) is int and image["bytes"] > 0,
                         "invalid asset bytes")
                suffix = Path(image["path"]).suffix
                _require(suffix in EXTENSIONS and image["path"] ==
                         f"assets/{image['sha256']}{suffix}", "nonopaque asset path")
        if dense:
            _dense_provenance(row)


def _labels(rows, episodes, contract=None):
    if contract is not None and type(contract.get("version")) is int and contract["version"] == 3:
        from .recognition_data import labels

        return labels(rows, episodes, contract)
    indexed = {}
    current = contract is not None and contract["task"] == CURRENT
    fields = (_LABEL_FIELDS - {"time_to_contact_s"}) | {"static_label_preserved", "static_source_sha256"} if current else _LABEL_FIELDS
    for row in rows:
        _keys(row, fields, _OPTIONAL)
        _identifier(row["episode_id"])
        for field in _OPTIONAL & row.keys():
            _identifier(row[field])
        _require(row["episode_id"] not in indexed, "duplicate label episode")
        _require(type(row["correct_candidate"]) is int and
                 row["correct_candidate"] in (0, 1), "invalid correct_candidate")
        if current:
            _require(row["static_label_preserved"] is True, "static label must be preserved")
            _digest(row["static_source_sha256"])
            _require(row["label_status"] == "audited", "current labels require a static-task audit")
        else:
            ttc = row["time_to_contact_s"]
            _require(type(ttc) in (float, int) and ttc > 0, "nonpositive/invalid TTC")
            if isinstance(ttc, float):
                _require(math.isfinite(ttc), "nonfinite TTC")
        _require(row["label_status"] in ("official", "audited"), "invalid label_status")
        indexed[row["episode_id"]] = row
    _require(indexed.keys() == set(episodes), "missing or extra labels")
    return indexed


def _exposure(value):
    _keys(value, {"version", "participants", "videos"},
          set(_EXPOSURE_FIELDS) - {"participants", "videos"})
    _require(type(value["version"]) is int and value["version"] == 1,
             "unsupported exposure registry version")
    result = {}
    for field in _EXPOSURE_FIELDS:
        items = value.get(field, [])
        _require(isinstance(items, list), "exposure sets must be JSON arrays")
        for item in items:
            _digest(item) if field == "asset_sha256" else _identifier(item)
        _require(len(items) == len(set(items)), "duplicate exposure entry")
        result[field] = items
    return result


def _check_exposure(rows, exposure, contract=None):
    sets = {key: set(value) for key, value in (exposure or {}).items()}
    if slots_for_contract(contract) == DENSE_SLOTS:
        _require(exposure is not None, "dense descriptive reuse requires an exposure registry")
        # This contract only executes current/current-canary, not confirmation.
        matches = {key: set() for key in _EXPOSURE_FIELDS}
        by_split = dict.fromkeys(SPLITS, 0)
        for row in rows:
            overlapping = False
            for field, domain in _ID_FIELDS.items():
                value = row.get(field)
                if value in sets[domain]:
                    matches[domain].add(value)
                    overlapping = True
            for image in row["images"].values():
                if image["sha256"] in sets["asset_sha256"]:
                    matches["asset_sha256"].add(image["sha256"])
                    overlapping = True
            by_split[row["split"]] += int(overlapping)
        return {"policy": "dense_current_descriptive_reuse", "fresh_confirmation_claimed": False,
                "rows_with_overlap": sum(by_split.values()),
                "rows_with_overlap_by_original_split": by_split,
                "unique_matches_by_domain": {key: len(values) for key, values in matches.items()}}
    for row in rows:
        if row["split"] != "confirmation":
            continue
        _require(exposure is not None, "confirmation requires an exposure registry")
        for field, domain in _ID_FIELDS.items():
            _require(row.get(field) not in sets[domain], "confirmation identity was exposed")
        for image in row["images"].values():
            _require(image["sha256"] not in sets["asset_sha256"],
                     "confirmation asset was exposed")


def _summary(rows):
    counts = {"episodes": len(rows),
              "participants": len({row["cluster_id"] for row in rows}),
              "videos": len({row["video_id"] for row in rows})}
    splits = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        splits[split] = {
            "episodes": len(selected),
            "participants": len({row["cluster_id"] for row in selected}),
            "videos": len({row["video_id"] for row in selected}),
            "public_sha256": _sha(_lines(selected)),
        }
    return counts, splits


def _inventory(rows):
    inventory = {}
    for row in rows:
        for image in row["images"].values():
            previous = inventory.setdefault(image["path"], image)
            _require(previous == image, "inconsistent asset inventory")
    return [inventory[key] for key in sorted(inventory)]


def _attrition(rows, labels):
    return {"input_observations": len(rows), "input_labels": len(labels),
            "accepted": len(rows), "rejected": 0, "rejected_by_reason": {},
            "label_status_counts": {
                status: sum(row["label_status"] == status for row in labels)
                for status in ("official", "audited")},
            "scope": "preparation_only_not_upstream_annotation_audit"}


def prepare_dataset(
    observations_path: Path, labels_path: Path, out_dir: Path,
    exposure_path: Path | None = None,
    *, protocol: dict | None = None, historical_static_path: Path | None = None,
) -> dict:
    """Validate supplied observations/labels, copy assets, publish a NEW release."""
    if protocol is not None and type(protocol.get("schema_version")) is int and protocol["schema_version"] == 4:
        from .recognition_data import prepare_recognition_dataset

        return prepare_recognition_dataset(observations_path, labels_path, out_dir, exposure_path,
                                           protocol=protocol, historical_static_path=historical_static_path)
    observations_path, labels_path, out_dir = map(Path, (observations_path, labels_path, out_dir))
    contract = protocol_contract(validate_protocol(protocol)) if protocol is not None else None
    slots = slots_for_contract(contract)
    dense = slots == DENSE_SLOTS
    _require(contract is not None or historical_static_path is None,
             "historical reuse requires a versioned matched policy")
    current = contract is not None and contract["task"] == CURRENT
    _require(not os.path.lexists(out_dir), "output already exists")
    observations_raw, labels_raw = _read(observations_path), _read(labels_path)
    observations = _jsonl(observations_raw)
    _observations(observations, contract=contract)
    labels = _labels(_jsonl(labels_raw), [row["episode_id"] for row in observations], contract)
    if dense:
        for row in observations:
            _require(labels[row["episode_id"]]["static_source_sha256"] ==
                     row["dense_current_provenance"]["fixed_current"]["static_record_sha256"],
                     "static label provenance mismatch")
    exposure_raw = _read(Path(exposure_path)) if exposure_path is not None else None
    exposure = _exposure(_json(exposure_raw)) if exposure_raw is not None else None
    history, history_raw = None, None
    if contract:
        _require(historical_static_path is not None and exposure_raw is not None,
                 "matched policy requires historical-static and temporal exposure registries")
        history_raw = _read(Path(historical_static_path))
        history = _exposure(_json(history_raw))
        policy = contract["exposure_policy"]
        _require(_sha(history_raw) == policy["historical_static_sha256"], "historical raw registry hash mismatch")
        _require(_sha(exposure_raw) == policy["temporal_exposure_sha256"], "temporal raw registry hash mismatch")
    _require(exposure is not None or not any(row["split"] == "confirmation"
                                           for row in observations),
             "confirmation requires an exposure registry")
    base = observations_path.parent.resolve()
    # Map identities independently of labels and source spelling; no reversible names.
    identities = {domain: {} for domain in _ID_FIELDS.values()}

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
        _require(not os.path.lexists(out_dir), "output already exists")
        stage = Path(tempfile.mkdtemp(prefix="." + out_dir.name + ".prepare-", dir=parent))
        (stage / "assets").mkdir()
        public, private = [], []
        for observation in observations:
            row = {key: opaque(domain, observation[key])
                   for key, domain in _ID_FIELDS.items() if key in observation}
            row.update({key: observation[key] for key in (
                "split", "static_pair_preserved" if current else "pair_frozen_before_label_review", "annotation_source",
                "observation_times_s", "timestamps_verified", "candidates_visible_at_cutoff")})
            if contract:
                row.update(task=contract["task"], contract_sha256=digest(contract))
            if dense:
                row["dense_current_provenance"] = observation["dense_current_provenance"]
            row["images"] = {}
            for slot in slots:
                supplied = observation["images"][slot]
                source = _asset(base, supplied["path"])
                asset_digest = _file_sha(source)
                _require(supplied.get("sha256", asset_digest) == asset_digest, "source asset hash mismatch")
                relative = f"assets/{asset_digest}{source.suffix.lower()}"
                target = stage / relative
                if not target.exists():
                    shutil.copyfile(source, target)
                _require(_file_sha(target) == asset_digest, "copied asset hash mismatch")
                row["images"][slot] = {
                    "path": relative, "sha256": asset_digest, "bytes": target.stat().st_size,
                }
            public.append(row)
            label = dict(labels[observation["episode_id"]])
            if "event_id" in observation and "event_id" in label:
                _require(observation["event_id"] == label["event_id"], "event identity mismatch")
            label["episode_id"] = row["episode_id"]
            if "event_id" in label:
                label["event_id"] = opaque("events", label["event_id"])
            private.append(label)
        opaque_exposure = None
        if exposure is not None:
            opaque_exposure = {
                key: sorted(values if key == "asset_sha256" else
                            [opaque(key, value) for value in values])
                for key, values in exposure.items()
            }
        opaque_history = None
        if history is not None:
            opaque_history = {
                key: sorted(values if key == "asset_sha256" else [opaque(key, value) for value in values])
                for key, values in history.items()
            }
        exposure_overlap = _check_exposure(public, opaque_exposure, contract)
        public_raw, private_raw = _lines(public), _lines(private)
        (stage / "public.jsonl").write_bytes(public_raw)
        # Create private metadata with restrictive permissions from the first write.
        with (stage / "evaluator.jsonl").open("xb") as stream:
            os.chmod(stream.fileno(), 0o600)
            stream.write(private_raw)
        counts, splits = _summary(public)
        manifest = {
            "version": 1, "public_sha256": _sha(public_raw),
            "evaluator_sha256": _sha(private_raw), "asset_inventory": _inventory(public),
            "counts": counts, "splits": splits,
            "provenance": {
                "observations_sha256": _sha(observations_raw),
                "labels_sha256": _sha(labels_raw),
                "exposure_sha256": _sha(exposure_raw) if exposure_raw is not None else None,
                "exposure": opaque_exposure, "annotation_source": (
                    "audited_static_current_interaction" if current else "ego4d_sta"),
                "attrition": _attrition(public, private),
            },
        }
        if contract:
            manifest.update(version=3 if dense else 2, contract=contract)
            manifest["provenance"].update(historical_static_sha256=_sha(history_raw),
                                          historical_static_exposure=opaque_history)
        if dense:
            manifest["provenance"]["temporal_exposure_overlap"] = exposure_overlap
        (stage / "manifest.json").write_text(_canonical(manifest) + "\n", encoding="utf-8")
        validate_dataset(stage, verify_assets=True)
        _require(not os.path.lexists(out_dir), "output appeared during preparation")
        os.rename(stage, out_dir)
        stage = None
        return manifest
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.unlink()


def validate_dataset(root, split=None, verify_assets=False) -> dict:
    """Validate the entire release, returning no hidden rows, even with no split."""
    _require(split is None or split in SPLITS, "invalid split")
    root = Path(root)
    _require(not root.is_symlink() and root.is_dir(), "invalid release directory")
    root = root.resolve()
    dispatch_manifest = _json(_read(root / "manifest.json"))
    if type(dispatch_manifest.get("version")) is int and dispatch_manifest["version"] == 4:
        from .recognition_data import validate_recognition_dataset

        return validate_recognition_dataset(root, split=split, verify_assets=verify_assets)
    _require({item.name for item in root.iterdir()} == {
        "manifest.json", "public.jsonl", "evaluator.jsonl", "assets",
    }, "unexpected or missing release entries")
    manifest = _json(_read(root / "manifest.json"))
    _require(type(manifest.get("version")) is int and manifest["version"] in (1, 2, 3),
             "unsupported manifest version")
    matched = manifest["version"] in (2, 3)
    _keys(manifest, {"version", "public_sha256", "evaluator_sha256", "asset_inventory",
                     "counts", "splits", "provenance"} | ({"contract"} if matched else set()))
    contract = validate_contract(manifest["contract"]) if matched else None
    _require(contract is None or contract["version"] != 3,
             "recognition contract requires manifest version 4")
    dense = slots_for_contract(contract) == DENSE_SLOTS
    _require(dense == (manifest["version"] == 3), "dense manifest/contract version mismatch")
    current = contract is not None and contract["task"] == CURRENT
    public_raw, private_raw = _read(root / "public.jsonl"), _read(root / "evaluator.jsonl")
    for field, raw in (("public_sha256", public_raw), ("evaluator_sha256", private_raw)):
        _digest(manifest[field])
        _require(_sha(raw) == manifest[field], "release metadata hash mismatch")
    rows = _jsonl(public_raw)
    _observations(rows, public=True, contract=contract)
    labels = _labels(_jsonl(private_raw), [row["episode_id"] for row in rows], contract)
    for row in rows:
        label = labels[row["episode_id"]]
        if dense:
            _require(label["static_source_sha256"] ==
                     row["dense_current_provenance"]["fixed_current"]["static_record_sha256"],
                     "static label provenance mismatch")
        if "event_id" in row and "event_id" in label:
            _require(row["event_id"] == label["event_id"], "event identity mismatch")
    counts, splits = _summary(rows)
    _require(_canonical(manifest["counts"]) == _canonical(counts), "counts mismatch")
    _require(_canonical(manifest["splits"]) == _canonical(splits), "split hashes/counts mismatch")
    inventory = _inventory(rows)
    _require(manifest["asset_inventory"] == inventory, "asset inventory mismatch")
    assets = root / "assets"
    _require(not assets.is_symlink() and assets.is_dir(), "invalid assets directory")
    _require({item.name for item in assets.iterdir()} ==
             {Path(item["path"]).name for item in inventory}, "physical inventory mismatch")
    for item in inventory:
        path = _asset(root, item["path"])
        _require(path.stat().st_size == item["bytes"], "asset size mismatch")
        if verify_assets:
            _require(_file_sha(path) == item["sha256"], "physical asset hash mismatch")
    provenance = manifest["provenance"]
    _keys(provenance, {"observations_sha256", "labels_sha256", "exposure_sha256",
                       "exposure", "annotation_source", "attrition"} |
          ({"historical_static_sha256", "historical_static_exposure"} if matched else set()) |
          ({"temporal_exposure_overlap"} if dense else set()))
    for field in ("observations_sha256", "labels_sha256"):
        _digest(provenance[field])
    _require(provenance["annotation_source"] == (
        "audited_static_current_interaction" if current else "ego4d_sta"), "unsupported annotation source")
    if matched:
        policy = contract["exposure_policy"]
        _require(provenance["historical_static_sha256"] == policy["historical_static_sha256"],
                 "historical raw registry hash mismatch")
        _require(provenance["exposure_sha256"] == policy["temporal_exposure_sha256"]
                 and provenance["exposure"] is not None, "temporal exposure policy mismatch")
        history = provenance["historical_static_exposure"]
        _keys(history, _EXPOSURE_FIELDS)
        _exposure({"version": 1, **history})
        for key, values in history.items():
            if key != "asset_sha256":
                _require(all(_OPAQUE.fullmatch(value) for value in values),
                         "historical identities must be opaque")
    exposure = provenance["exposure"]
    if exposure is None:
        _require(provenance["exposure_sha256"] is None, "inconsistent exposure provenance")
    else:
        _digest(provenance["exposure_sha256"])
        _keys(exposure, _EXPOSURE_FIELDS)
        _exposure({"version": 1, **exposure})
        for key, values in exposure.items():
            if key != "asset_sha256":
                _require(all(_OPAQUE.fullmatch(value) for value in values),
                         "exposure identities must be opaque")
    exposure_overlap = _check_exposure(rows, exposure, contract)
    if dense:
        _require(_canonical(provenance["temporal_exposure_overlap"]) == _canonical(exposure_overlap),
                 "temporal exposure overlap disclosure mismatch")
    _require(_canonical(provenance["attrition"]) ==
             _canonical(_attrition(rows, list(labels.values()))), "attrition counts mismatch")
    return {"manifest": manifest,
            "rows": [row for row in rows if split is None or row["split"] == split]}
