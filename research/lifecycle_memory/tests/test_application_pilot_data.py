import copy
import hashlib
import json
import random

import pytest

from research.lifecycle_memory.application_pilot_data import (
    build_dataset, export_data, prepare_data, read_json_rows,
)


def battery(n=10):
    public, hidden = [], []
    for frame in range(n):
        def asset(name, kind):
            path = f"assets/{frame}_{name}.png"
            return dict(kind=kind, path=path, sha256=hashlib.sha256(path.encode()).hexdigest(),
                        width=20, height=20)
        full = asset("full", "full_frame")
        crops = [asset("a", "reference_crop_a"), asset("b", "reference_crop_b")]
        for order in (0, 1):
            item, request = f"item_{frame}_{order}", f"request_{frame}_{order}"
            inputs = [copy.deepcopy(full)] + [dict(crops[order], kind="reference_crop_a"),
                                              dict(crops[1-order], kind="reference_crop_b")]
            public.append(dict(item_id=item, request_id=request, arm="real", inputs=inputs,
                               system_prompt="system", user_prompt="question", unknown_secret="not exported"))
            hidden.append(dict(item_id=item, request_ids={"real": request},
                               frame_level_unit_id=f"frame_{frame}", participant_id=f"p{frame}",
                               cluster_id=f"cluster_{frame}", counterbalance_order_index=order,
                               answer="AB"[order], source_video_id=f"video_{frame}", frame_id=f"source_{frame}",
                               crop_a_candidate_id=f"candidate_{frame}_{order}",
                               crop_b_candidate_id=f"candidate_{frame}_{1-order}",
                               donor_id="shared_unused_donor", donor_frame_id="shared_unused_frame",
                               assets={"wrong_time_full": dict(full, sha256="0" * 64)}))
    return public, hidden


def test_deterministic_participant_split_and_api():
    public, hidden = battery()
    first = build_dataset(public, hidden, expected_frames=10)
    random.Random(9).shuffle(public)
    random.Random(2).shuffle(hidden)
    second = build_dataset(public, hidden, expected_frames=10)
    assert first.public_rows == second.public_rows
    assert first.labels == second.labels
    assert [first.manifest["counts_by_split"][s]["participants"] for s in ("train", "validation", "test")] == [6, 2, 2]
    assert len(first.rows_for("compression")) == 20
    assert len(first.rows_for("steering")) == 4
    assert len(first.rows_for("steering", "validation")) == 4
    with pytest.raises(ValueError, match="all splits"):
        first.rows_for("compression", "test")
    for row in first.public_rows:
        assert row["arms"]["real"]["inputs"] == row["inputs"]
        assert row["frame_unit_id"] == row["frame_key"]
        other = next(r for r in first.public_rows if r["request_id"] == row["counterbalanced_request_id"])
        assert other["split"] == row["split"]


@pytest.mark.parametrize("identity", ["hash", "path", "video", "candidate", "component", "frame"])
def test_duplicate_identity_components_stay_together(identity):
    public, hidden = battery()
    if identity in {"hash", "path"}:
        for row in public[2:4]:
            row["inputs"][0]["sha256"] = public[0]["inputs"][0]["sha256"]
            if identity == "path":
                row["inputs"][0]["path"] = public[0]["inputs"][0]["path"]
    else:
        field = {"video": "source_video_id", "candidate": "first_order_candidate_id",
                 "component": "crop_a_selected_component_fingerprint", "frame": "frame_id"}[identity]
        for row in hidden[:4]:
            row[field] = "duplicate"
    data = build_dataset(public, hidden, expected_frames=10)
    rows = {r["request_id"]: r for r in data.public_rows}
    assert rows["request_0_0"]["split"] == rows["request_1_0"]["split"]
    assert rows["request_0_0"]["split_group_key"] == rows["request_1_0"]["split_group_key"]
    assert rows["request_0_0"]["cluster_key"] != rows["request_1_0"]["cluster_key"]


def test_extreme_overlap_fails_closed():
    public, hidden = battery()
    for row in hidden[:16]:
        row["source_video_id"] = "shared_video"
    with pytest.raises(ValueError, match="overlap"):
        build_dataset(public, hidden, expected_frames=10)
    for row in hidden:
        row["source_video_id"] = "shared_video"
    with pytest.raises(ValueError, match="three independent"):
        build_dataset(public, hidden, expected_frames=10)


def test_outcome_blind_and_no_donor_grouping():
    public, hidden = battery()
    first = build_dataset(public, hidden, expected_frames=10)
    for row in hidden:
        row["answer"] = "B" if row["answer"] == "A" else "A"
    second = build_dataset(public, hidden, expected_frames=10)
    assert first.public_rows == second.public_rows
    assert first.manifest["counts"]["identity_components"] == 10


@pytest.mark.parametrize("problem", ["arm", "missing", "label", "order", "swap", "join", "path", "hash", "duplicate"])
def test_invalid_inputs_fail(problem):
    public, hidden = battery()
    if problem == "arm":
        public[0]["arm"] = "wrong_time"
    elif problem == "missing":
        public.pop()
    elif problem == "label":
        hidden[0]["answer"] = "B"
    elif problem == "order":
        hidden[0]["counterbalance_order_index"] = 1
    elif problem == "swap":
        public[0]["inputs"][1]["path"] = "assets/another.png"
    elif problem == "join":
        hidden[0]["request_ids"]["real"] = "bad"
    elif problem == "path":
        public[0]["inputs"][0]["path"] = "../secret.png"
    elif problem == "hash":
        public[0]["inputs"][0]["sha256"] = "bad"
    else:
        public.append(public[0])
    with pytest.raises(ValueError):
        build_dataset(public, hidden, expected_frames=10)


def test_export_label_boundary_manifest_and_file_hashes(tmp_path):
    public, hidden = battery()
    root = tmp_path / "battery"
    (root / "public").mkdir(parents=True)
    (root / "hidden").mkdir()
    (root / "public/real_request_pack.jsonl").write_text("\n".join(map(json.dumps, public)))
    (root / "hidden/evaluator_index.jsonl").write_text("\n".join(map(json.dumps, hidden)))
    data = prepare_data(root, expected_frames=10)
    out = tmp_path / "dataset"
    manifest = export_data(data, out)
    assert manifest["input_counts"] == {"public_rows": 20, "hidden_rows": 20}
    public_text = (out / "public/requests.jsonl").read_text()
    for secret in ("correct_label", "unknown_secret", "participant_id", "donor", "target_role"):
        assert secret not in public_text
    for name, receipt in manifest["exports"].items():
        assert receipt["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert len(read_json_rows(out / name)) == receipt["rows"]
    assert {r["split"] for r in read_json_rows(out / "labels/train.jsonl")} == {"train"}
    assert {r["split"] for r in read_json_rows(out / "labels/validation.jsonl")} == {"validation"}
    assert {r["split"] for r in read_json_rows(out / "evaluator/test_labels.jsonl")} == {"test"}
    assert not manifest["asset_audit"]["byte_hashes_verified"]
    with pytest.raises(FileExistsError):
        export_data(data, out)


def test_json_envelopes(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"requests": [{"a": 1}]}))
    assert read_json_rows(path) == [{"a": 1}]
    path.write_text(json.dumps({"answers": [{"a": 1}]}))
    assert read_json_rows(path) == [{"a": 1}]
