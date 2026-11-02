from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import pytest

from research.lifecycle_memory import analyze_phase2_reanchored_behavior as analysis
from research.lifecycle_memory import quarantine_appledouble_sidecars as appledouble
from research.lifecycle_memory import run_phase2_reanchored_behavior as runtime


REPO_ROOT = Path(__file__).resolve().parents[3]
JOB_DIR = REPO_ROOT / "docs/research/lifecycle_memory/phase 2 research/jobs/reanchored_jobA_behavior"
PREREG = REPO_ROOT / "docs/research/lifecycle_memory/phase 2 research/preregistration/reanchored_evidence_failure_localization_v1_2026_07_21.json"
RIGHTS = REPO_ROOT / "docs/research/lifecycle_memory/phase 2 research/job3_stage_a_rights_and_provenance_ledger_2026_07_14.md"


@pytest.fixture(autouse=True)
def _bind_logic_tests_to_current_preregistration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind logic tests to the current preregistration for the isolated unit-test scope."""
    monkeypatch.setattr(
        runtime, "EXPECTED_PREREGISTRATION_SHA256", runtime.sha256_file(PREREG)
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _image(path: Path, color: tuple[int, int, int], size: tuple[int, int]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_hash_manifest(data: Path) -> None:
    entries = []
    for path in sorted(data.rglob("*")):
        if not path.is_file() or path.name == "hash_manifest.json":
            continue
        entries.append(
            {
                "path": str(path.relative_to(data)),
                "sha256": runtime.sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    _write_json(
        data / "hash_manifest.json",
        {
            "kind": "phase2_reanchored_egohos_job_a_2afc_hash_manifest",
            "algorithm": "sha256",
            "entry_count": len(entries),
            "entries": entries,
        },
    )


def _tiny_battery(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    data = tmp_path / "data"
    run = tmp_path / "run"
    prompt = json.loads(PREREG.read_text(encoding="utf-8"))["generation"]["prompt_text"]
    system = "Choose A or B and return the frozen JSON schema."
    requests_by_arm: dict[str, list[dict]] = {arm: [] for arm in runtime.ARMS}
    for frame_index in range(2):
        frame_id = f"frame_{frame_index}"
        crop_x = f"assets/crops/{frame_id}_x.png"
        crop_y = f"assets/crops/{frame_id}_y.png"
        _image(data / crop_x, (10 + frame_index, 20, 30), (4, 4))
        _image(data / crop_y, (40, 50 + frame_index, 60), (4, 4))
        full_by_arm = {
            "real": f"assets/real/{frame_id}.png",
            "blank_full": f"assets/blank/{frame_id}.png",
            "wrong_time": f"assets/wrong/{frame_id}.png",
        }
        for arm_index, (arm, full_path) in enumerate(full_by_arm.items()):
            _image(data / full_path, (80 + arm_index, frame_index, 5), (8, 8))
            for order_index, crop_paths in enumerate(((crop_x, crop_y), (crop_y, crop_x))):
                item_id = f"trial_{frame_index}_{order_index}"
                requests_by_arm[arm].append(
                    {
                        "arm": arm,
                        "item_id": item_id,
                        "request_id": f"request_{arm}_{frame_index}_{order_index}",
                        "benchmark_split": "phase2_reanchored_egohos_job_a_2afc",
                        "system_prompt": system,
                        "user_prompt": prompt,
                        "inputs": [
                            {"path": full_path, "kind": "full_frame", "width": 8, "height": 8},
                            {"path": crop_paths[0], "kind": "reference_crop_a", "width": 4, "height": 4},
                            {"path": crop_paths[1], "kind": "reference_crop_b", "width": 4, "height": 4},
                        ],
                    }
                )
    _write_json(
        data / "manifest.json",
        {
            "kind": "phase2_reanchored_egohos_job_a_2afc_manifest",
            "status": "production_construction_complete",
            "production_eligible": True,
            "governing_prespec_sha256": runtime.EXPECTED_PREREGISTRATION_SHA256,
            "selection_only": False,
            "limited_nonproduction_mode": False,
            "frame_level_unit_count": 2,
            "counterbalanced_trials_per_frame": 2,
            "trial_count": 4,
            "request_count_all_arms": 12,
            "participant_cluster_count": 2,
            "answer_counts_hidden": {"A": 2, "B": 2},
            "class_allocation": {
                "first_order": {"3": 1, "4": 1, "5": 0},
                "second_order": {"6": 1, "7": 1, "8": 0},
            },
            "rendering_contract": {
                "full_frame": {"width": 8, "height": 8, "mode": "RGB"},
                "reference_crop": {"width": 4, "height": 4, "mode": "RGB"},
            },
            "outputs": {
                "public_request_packs": {
                    arm: f"public/{arm}_request_pack.json" for arm in runtime.ARMS
                },
                "hidden_evaluator_index": "hidden/evaluator_index.json",
                "selection_audit": "audits/selection_audit.json",
                "hash_manifest": "hash_manifest.json",
            },
            "analysis_unit_contract": {
                "primary_evidence_estimand": (
                    "intersection_union_test_requiring_positive_participant_equal_real_minus_"
                    "blank_full_and_real_minus_wrong_time_components"
                )
            },
            "ontology_contract": {
                "primary_task": "annotation_native_first_vs_second_order_interacting_object_2afc",
                "single_target_binary_direct_contact_primary": False,
            },
        },
    )
    _write_json(data / "audits/selection_audit.json", {"component_provenance": "fixture"})
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    manifest["selection_audit_sha256"] = runtime.sha256_file(
        data / "audits/selection_audit.json"
    )
    _write_json(data / "manifest.json", manifest)
    _write_json(
        data / "hidden/evaluator_index.json",
        {
            "kind": "phase2_reanchored_egohos_job_a_2afc_hidden_evaluator_index",
            "frame_level_unit_count": 2,
            "trial_count": 4,
            "answers": [],
        },
    )
    for arm, rows in requests_by_arm.items():
        _write_json(
            data / f"public/{arm}_request_pack.json",
            {
                "kind": "phase2_reanchored_egohos_job_a_2afc_request_pack",
                "arm": arm,
                "request_count": 4,
                "frame_level_unit_count": 2,
                "counterbalanced_trials_per_frame": 2,
                "requests": rows,
            },
        )
    _write_hash_manifest(data)
    contract = tmp_path / "contract.json"
    _write_json(
        contract,
        {
            "kind": "phase2_reanchored_job_a_operational_contract",
            "version": 1,
            "governing_preregistration": {
                "path": str(PREREG.relative_to(REPO_ROOT)),
                "sha256": runtime.EXPECTED_PREREGISTRATION_SHA256,
            },
            "governing_rights_ledger": {
                "path": str(RIGHTS.relative_to(REPO_ROOT)),
                "sha256": runtime.sha256_file(RIGHTS),
            },
            "battery": {
                "manifest_relative_path": "manifest.json",
                "hash_manifest_relative_path": "hash_manifest.json",
                "hidden_evaluator_relative_path": "hidden/evaluator_index.json",
                "pack_relative_paths": {
                    arm: f"public/{arm}_request_pack.json"
                    for arm in runtime.ARMS
                },
                "required_frame_count": 2,
                "required_trial_count": 4,
                "required_participant_count": 2,
                "required_trials_per_frame": 2,
                "required_inputs_per_request": 3,
            },
            "canary": {
                "frame_count": 1,
                "label_visibility": "forbidden",
                "utilization_tail_gate": {
                    "telemetry_interval_seconds": 2.0,
                    "minimum_common_allocation_mean_pct": 20.0,
                    "common_allocation_mean_role": "descriptive_advisory_union_window_absent_rank_buckets_zero_filled",
                    "minimum_worker_inference_mean_pct": 20.0,
                    "minimum_worker_inference_p90_pct": 50.0,
                    "minimum_worker_inference_sample_count": 5,
                    "minimum_projected_rank_occupancy_fraction": 0.55,
                    "maximum_projected_rank_tail_idle_fraction": 0.70,
                },
            },
        },
    )
    return data, run, contract, JOB_DIR / "model_plan.json"


def test_strict_choice_and_exact_suffix_scoring_cpu() -> None:
    torch = pytest.importorskip("torch")
    assert runtime.strict_parse_choice(
        '{"answer":"A","abstain":false,"abstention_reason":null,"confidence":1.0}'
    )["answer"] == "A"
    assert not runtime.strict_parse_choice("A")["valid"]
    logits = torch.zeros((1, 3, 8), dtype=torch.float32)
    logits[0, 1, 3] = 8.0
    scored = runtime.score_candidate_suffix_from_logits(
        base_input_ids=[1, 2], full_input_ids=[1, 2, 3], logits=logits
    )
    assert scored["status"] == "pass"
    assert scored["token_count"] == 1
    assert scored["sum_logprob"] < 0.0
    assert scored["sum_logprob"] > -0.01


def test_preflight_is_label_free_and_hash_bound(tmp_path: Path) -> None:
    data, run, contract, model_plan = _tiny_battery(tmp_path)
    report = runtime.preflight(
        data_dir=data,
        run_dir=run,
        model_plan_path=model_plan,
        contract_path=contract,
        mode="canary",
        limit_frames=1,
    )
    assert report["status"] == "pass"
    assert report["frame_count"] == 1
    assert report["trial_count"] == 2
    selection = json.loads((run / "selection_manifest.json").read_text(encoding="utf-8"))
    assert selection["claim_boundary"].startswith("Evaluator-unjoined")
    assert not runtime.contains_forbidden_public_key(selection)
    assert "frame_unit_id" not in json.dumps(selection)
    assert [row["counterbalance_pair_index"] for row in selection["selection"]] == [0, 0]

    real_asset = data / selection["selection"][0]["public_identity_by_arm"]["real"]["ordered_inputs"][0]["path"]
    real_asset.write_bytes(real_asset.read_bytes() + b"drift")
    pack = json.loads(
        (data / "public/real_request_pack.json").read_text(encoding="utf-8")
    )
    observed = runtime.request_identity(
        pack["requests"][0], data_dir=data, image_cache={}
    )
    expected = selection["selection"][0]["public_identity_by_arm"]["real"]
    assert observed != expected


def test_public_label_firewall_rejects_class_field(tmp_path: Path) -> None:
    data, run, contract, model_plan = _tiny_battery(tmp_path)
    path = data / "public/real_request_pack.json"
    pack = json.loads(path.read_text(encoding="utf-8"))
    pack["requests"][0]["class_id"] = 3
    _write_json(path, pack)
    _write_hash_manifest(data)
    with pytest.raises(ValueError, match="hidden-label firewall"):
        runtime.preflight(
            data_dir=data,
            run_dir=run,
            model_plan_path=model_plan,
            contract_path=contract,
            mode="canary",
            limit_frames=1,
        )


def test_hash_manifest_rejects_byte_drift(tmp_path: Path) -> None:
    data, run, contract, model_plan = _tiny_battery(tmp_path)
    path = data / "public/real_request_pack.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash-manifest (byte count|SHA-256) drift"):
        runtime.preflight(
            data_dir=data,
            run_dir=run,
            model_plan_path=model_plan,
            contract_path=contract,
            mode="canary",
            limit_frames=1,
        )


def test_hash_manifest_rejects_appledouble_metadata_with_diagnostics(
    tmp_path: Path,
) -> None:
    data, run, contract, model_plan = _tiny_battery(tmp_path)
    (data / "public/._real_request_pack.json").write_bytes(b"appledouble")
    with pytest.raises(
        ValueError,
        match=r"extra_count=1.*public/\._real_request_pack\.json",
    ):
        runtime.preflight(
            data_dir=data,
            run_dir=run,
            model_plan_path=model_plan,
            contract_path=contract,
            mode="canary",
            limit_frames=1,
        )


def test_appledouble_quarantine_validates_before_moving(tmp_path: Path) -> None:
    root = tmp_path / "battery"
    quarantine = tmp_path / "quarantine"
    payload = appledouble.APPLEDOUBLE_MAGIC_VERSION + b"x" * 155
    expected_sha = hashlib.sha256(payload).hexdigest()
    paths = [root / "._manifest.json", root / "assets/._frame.png"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    receipt = appledouble.quarantine_sidecars(
        root=root,
        quarantine=quarantine,
        expected_count=2,
        expected_size=163,
        expected_sha256=expected_sha,
    )
    assert receipt["status"] == "pass"
    assert receipt["quarantined_count"] == 2
    assert not any(path.exists() for path in paths)
    assert (quarantine / "._manifest.json").read_bytes() == payload
    assert (quarantine / "assets/._frame.png").read_bytes() == payload
    assert (quarantine / "quarantine_receipt.json").is_file()


def test_appledouble_quarantine_fails_closed_before_any_move(tmp_path: Path) -> None:
    root = tmp_path / "battery"
    quarantine = tmp_path / "quarantine"
    good = appledouble.APPLEDOUBLE_MAGIC_VERSION + b"x" * 155
    bad = b"not-appledouble" + b"x" * (163 - len("not-appledouble"))
    first = root / "._first"
    second = root / "nested/._second"
    first.parent.mkdir(parents=True, exist_ok=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(good)
    second.write_bytes(bad)

    with pytest.raises(ValueError, match="AppleDouble magic/version"):
        appledouble.quarantine_sidecars(
            root=root,
            quarantine=quarantine,
            expected_count=2,
            expected_size=163,
            expected_sha256=hashlib.sha256(good).hexdigest(),
        )
    assert first.read_bytes() == good
    assert second.read_bytes() == bad
    assert not quarantine.exists()


def test_appledouble_quarantine_resumes_after_interrupted_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "battery"
    quarantine = tmp_path / "quarantine"
    payload = appledouble.APPLEDOUBLE_MAGIC_VERSION + b"x" * 155
    expected_sha = hashlib.sha256(payload).hexdigest()
    paths = [root / "._first", root / "nested/._second"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    original_rename = Path.rename
    calls = 0

    def flaky_rename(path: Path, target: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interrupted move")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    with pytest.raises(OSError, match="interrupted move"):
        appledouble.quarantine_sidecars(
            root=root,
            quarantine=quarantine,
            expected_count=2,
            expected_size=163,
            expected_sha256=expected_sha,
        )
    monkeypatch.setattr(Path, "rename", original_rename)

    receipt = appledouble.quarantine_sidecars(
        root=root,
        quarantine=quarantine,
        expected_count=2,
        expected_size=163,
        expected_sha256=expected_sha,
    )
    assert receipt["status"] == "pass"
    assert receipt["quarantined_count"] == 2
    assert not any(path.exists() for path in paths)
    assert (quarantine / "._first").read_bytes() == payload
    assert (quarantine / "nested/._second").read_bytes() == payload


def test_prior_preregistration_hash_is_rejected(tmp_path: Path) -> None:
    _, _, contract, _ = _tiny_battery(tmp_path)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["governing_preregistration"]["sha256"] = "0" * 64
    _write_json(contract, payload)
    with pytest.raises(ValueError, match="stale preregistration hash"):
        runtime._load_contract(contract)


def test_public_adjacent_pair_must_be_exact_crop_swap(tmp_path: Path) -> None:
    data, run, contract, model_plan = _tiny_battery(tmp_path)
    for arm in runtime.ARMS:
        path = data / f"public/{arm}_request_pack.json"
        pack = json.loads(path.read_text(encoding="utf-8"))
        pack["requests"][1]["inputs"][1:] = pack["requests"][0]["inputs"][1:]
        _write_json(path, pack)
    _write_hash_manifest(data)
    with pytest.raises(ValueError, match="crop order is not an exact swap"):
        runtime.preflight(
            data_dir=data,
            run_dir=run,
            model_plan_path=model_plan,
            contract_path=contract,
            mode="canary",
            limit_frames=1,
        )


def test_hidden_builder_schema_joins_only_after_selection(tmp_path: Path) -> None:
    hidden = tmp_path / "hidden.json"
    rows = []
    for order_index, answer in enumerate(("A", "B")):
        rows.append(
            {
                "item_id": f"item-{order_index}",
                "frame_level_unit_id": "frame-hidden",
                "counterbalance_order_index": order_index,
                "answer": answer,
                "participant_id": "participant-hidden",
                "first_order_class": 5,
                "second_order_class": 6,
                "request_ids": {arm: f"request-{arm}-{order_index}" for arm in runtime.ARMS},
            }
        )
    _write_json(hidden, {"answers": rows})
    selection = {
        "frame_count": 1,
        "selection": [
            {"item_id": "item-0", "counterbalance_pair_index": 0},
            {"item_id": "item-1", "counterbalance_pair_index": 0},
        ],
    }
    normalized = analysis._normalize_hidden(hidden, selection)
    assert set(normalized) == {"item-0", "item-1"}
    assert normalized["item-0"]["frame_unit_id"] == "frame-hidden"
    assert normalized["item-1"]["second_order_class"] == 6


def test_valid_abstention_is_not_counted_as_target_selection() -> None:
    candidate_rows = {
        "A": {"sum_logprob": -1.0},
        "B": {"sum_logprob": -2.0},
    }
    rows = {
        ("item", "real"): {
            "request_id": "request-real",
            "teacher_forced": {"candidate_rows": candidate_rows},
            "greedy": {
                "strict_choice": {
                    "valid": True,
                    "answer": "A",
                    "abstain": True,
                    "abstention_reason": "insufficient evidence",
                }
            },
        }
    }
    hidden = {
        "item": {
            "frame_unit_id": "frame",
            "participant_id": "participant",
            "correct_choice": "A",
            "first_order_class": 3,
            "second_order_class": 6,
            "request_ids": {"real": "request-real"},
        }
    }
    joined = analysis._joined_trials(rows=rows, hidden=hidden)
    assert joined[0]["abstain"] is True
    assert joined[0]["predicted_choice"] is None
    assert joined[0]["chosen_annotation_identity"] is None
    assert joined[0]["correct"] is False


def test_resume_row_hash_validation() -> None:
    common = {
        "model_key": "m",
        "model_id": "id",
        "revision": "rev",
        "dtype": "bf16",
        "worker_index": 0,
        "selection_manifest_sha256": "s",
        "preregistration_sha256": runtime.EXPECTED_PREREGISTRATION_SHA256,
        "code_snapshot_sha256": "c",
    }
    candidate = {
        "status": "pass",
        "visual_identity_status": "pass",
        "sum_logprob": -1.0,
    }
    row = {
        "kind": "phase2_reanchored_job_a_worker_row",
        "status": "pass",
        **common,
        "item_id": "i",
        "counterbalance_pair_index": 0,
        "arm": "real",
        "teacher_forced": {"candidate_rows": {"A": candidate, "B": candidate}},
        "greedy": {"status": "pass", "strict_choice": {"valid": True, "answer": "A"}},
    }
    row["row_sha256"] = runtime.row_hash(row)
    assert runtime.validate_completed_row(row, expected_common=common) == []
    row["revision"] = "drift"
    assert "revision" in runtime.validate_completed_row(row, expected_common=common)
    assert "row_hash" in runtime.validate_completed_row(row, expected_common=common)


def test_shared_bootstrap_intersection_and_holm() -> None:
    blank = {"p1": [1.0, 0.0], "p2": [0.5]}
    wrong = {"p1": [0.5, 0.5], "p2": [0.0]}
    result = analysis.shared_cluster_bootstrap_intersection(
        blank, wrong, draws=500, seed=7
    )
    assert result["real_minus_blank_full"] == pytest.approx(0.5)
    assert result["real_minus_wrong_time"] == pytest.approx(0.25)
    assert result["estimate"] == pytest.approx(0.25)
    assert result["joint_draw_rule"].startswith("both smooth components")
    assert set(result["component_one_sided_lower_95"]) == {
        "real_minus_blank_full",
        "real_minus_wrong_time",
    }
    assert "ci95" not in result
    adjusted = analysis.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.2})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.08)
    assert adjusted["c"] == pytest.approx(0.2)


def test_centered_cluster_bootstrap_is_one_sided_and_cluster_equal() -> None:
    strong = {f"p{index}": [0.75, 0.75] for index in range(24)}
    null = {f"p{index}": [0.25, 0.25] for index in range(24)}
    assert analysis.centered_cluster_bootstrap_p_value(
        strong, null_value=0.25, draws=2000, seed=3
    ) < 0.01
    assert analysis.centered_cluster_bootstrap_p_value(
        null, null_value=0.25, draws=2000, seed=3
    ) > 0.25

    unequal = {"large": [1.0] * 100, "small": [0.0]}
    summary = analysis.cluster_bootstrap_ci(unequal, draws=500, seed=5)
    assert summary["estimate"] == pytest.approx(0.5)


def test_near_tied_intersection_uses_component_bounds_not_minimum_ci() -> None:
    blank = {f"p{index}": [0.10 + (index % 2) * 0.01] for index in range(30)}
    wrong = {f"p{index}": [0.10 - (index % 2) * 0.01] for index in range(30)}
    result = analysis.shared_cluster_bootstrap_intersection(
        blank, wrong, draws=2000, seed=11
    )
    assert result["estimate"] == pytest.approx(0.095)
    assert result["intersection_union_lower_gate"] == min(
        result["component_one_sided_lower_95"].values()
    )
    assert "non-smooth minimum" in result["inference_rule"]


def test_high_real_accuracy_does_not_hide_failed_pair_correctness() -> None:
    frames = []
    joined = []
    for index in range(40):
        participant = f"p{index:02d}"
        frame = f"f{index:02d}"
        pair_correct = index < 12
        real_accuracy = 1.0 if pair_correct else 0.5
        for arm in runtime.ARMS:
            arm_accuracy = real_accuracy if arm == "real" else 0.0
            frames.append(
                {
                    "frame_unit_id": frame,
                    "participant_id": participant,
                    "arm": arm,
                    "first_order_class": 3,
                    "second_order_class": 7,
                    "mean_trial_accuracy": arm_accuracy,
                    "both_orders_correct": pair_correct if arm == "real" else False,
                    "order_consistent_identity": pair_correct if arm == "real" else False,
                    "raw_A_B_prediction_flip": pair_correct if arm == "real" else False,
                }
            )
            for order, choice in enumerate(runtime.CHOICES):
                correct = (
                    pair_correct
                    or (arm == "real" and order == 0)
                )
                joined.append(
                    {
                        "frame_unit_id": frame,
                        "participant_id": participant,
                        "arm": arm,
                        "correct_choice": choice,
                        "correct": correct,
                        "strict_parse_valid": True,
                        "abstain": False,
                        "correct_choice_margin_bits": 1.0 if correct else -1.0,
                        "chosen_annotation_identity": (
                            "first_order_interacting_object"
                            if correct
                            else "second_order_interacting_object"
                        ),
                    }
                )
    metrics = analysis._model_metrics(
        model_key="synthetic",
        joined=joined,
        frames=frames,
        bootstrap_draws=2000,
        bootstrap_seed=19,
        bootstrap_test_draws=2000,
    )
    assert metrics["real_capability"]["status"] == "pass"
    pair = metrics["primary_counterbalanced_pair_correctness"]
    assert pair["estimate"] == pytest.approx(0.30)
    assert pair["status"] == "fail"
    assert pair["paired_chance_reference"] == pytest.approx(0.25)
    assert "does not prove scene-target binding" in pair["claim_boundary"]
    assert "primary_target_binding" not in metrics


def test_frame_collapse_uses_counterbalanced_pair_as_one_unit() -> None:
    rows = []
    for arm in runtime.ARMS:
        rows.extend(
            [
                {
                    "item_id": "a",
                    "frame_unit_id": "f",
                    "participant_id": "p",
                    "arm": arm,
                    "first_order_class": 3,
                    "second_order_class": 6,
                    "correct_choice": "A",
                    "predicted_choice": "A",
                    "chosen_annotation_identity": "first_order_interacting_object",
                    "strict_parse_valid": True,
                    "abstain": False,
                    "correct": True,
                    "correct_choice_margin_bits": 2.0,
                },
                {
                    "item_id": "b",
                    "frame_unit_id": "f",
                    "participant_id": "p",
                    "arm": arm,
                    "first_order_class": 3,
                    "second_order_class": 6,
                    "correct_choice": "B",
                    "predicted_choice": "B",
                    "chosen_annotation_identity": "first_order_interacting_object",
                    "strict_parse_valid": True,
                    "abstain": False,
                    "correct": True,
                    "correct_choice_margin_bits": 1.0,
                },
            ]
        )
    collapsed = analysis._collapse_frames(rows)
    assert len(collapsed) == 3
    assert all(row["mean_trial_accuracy"] == 1.0 for row in collapsed)
    assert all(row["both_orders_correct"] for row in collapsed)
    assert all(row["order_consistent_identity"] for row in collapsed)
    assert all(row["raw_A_B_prediction_flip"] for row in collapsed)
    strata = analysis._descriptive_class_strata(collapsed)
    assert strata["design_balance"] == "structurally_imbalanced_not_a_balanced_3x3_design"
    assert strata["first_order_class"][0]["stratum"] == "3"
    assert strata["first_order_class"][0]["frame_n"] == 1
    assert strata["class_pair"][0]["stratum"] == "3x6"
    assert strata["class_pair"][0]["trial_n_per_arm"] == 2
    sensitivity = strata["common_support_standardized_sensitivity"]
    assert sensitivity["observed_class_pair_strata"] == ["3x6"]
    assert sensitivity["real_accuracy"] == pytest.approx(1.0)


def test_worker_validation_does_not_open_sealed_evaluator(tmp_path: Path) -> None:
    data, run, contract, model_plan = _tiny_battery(tmp_path)
    runtime.preflight(
        data_dir=data,
        run_dir=run,
        model_plan_path=model_plan,
        contract_path=contract,
        mode="canary",
        limit_frames=1,
    )
    (data / "hidden/evaluator_index.json").unlink()
    selection, _, _ = runtime._load_selection(
        data_dir=data,
        selection_path=run / "selection_manifest.json",
        model_plan_path=model_plan,
        contract_path=contract,
    )
    assert selection["hidden_evaluator_sha256"]
    with pytest.raises(FileNotFoundError):
        runtime._load_selection(
            data_dir=data,
            selection_path=run / "selection_manifest.json",
            model_plan_path=model_plan,
            contract_path=contract,
            verify_hidden_evaluator=True,
        )


def _schedule_reports(model_keys: list[str]) -> list[dict[str, object]]:
    rates = (12.0, 1.4, 6.0, 0.8)
    batches = (8, 2, 4, 2)
    return [
        {
            "model_key": key,
            "worker_index": index,
            "load_time_s": 20.0 + index,
            "decisions_per_second": rates[index],
            "effective_batch_size": batches[index],
            "strategy": "batched",
        }
        for index, key in enumerate(model_keys)
    ]


def test_balanced_schedule_has_no_overlap_and_full_coverage() -> None:
    reports = _schedule_reports(["m0", "m1", "m2", "m3"])
    schedule = runtime.build_balanced_full_schedule(
        reports, selection_count=1600, arm_count=3
    )
    assert schedule["sha256"] == runtime.canonical_sha256(
        {key: value for key, value in schedule.items() if key != "sha256"}
    )
    assignments = [
        assignment
        for rank in schedule["ranks"]
        for assignment in rank["assignments"]
    ]
    assert len({row["assignment_id"] for row in assignments}) == len(assignments)
    assert any(len(rank["assignments"]) == 2 for rank in schedule["ranks"])
    for model_key in ("m0", "m1", "m2", "m3"):
        intervals = sorted(
            (row["selection_start"], row["selection_stop"])
            for row in assignments
            if row["model_key"] == model_key
        )
        cursor = 0
        for start, stop in intervals:
            assert start == cursor
            assert stop > start
            assert start % 2 == 0 and stop % 2 == 0
            cursor = stop
        assert cursor == 1600
    static_totals = [
        float(row["load_time_s"]) + 1600 * 3 / float(row["decisions_per_second"])
        for row in reports
    ]
    static_busy = sum(static_totals) / (4 * max(static_totals))
    assert schedule["projected_aggregate_rank_occupancy_fraction"] > static_busy
    assert schedule["projected_makespan_s"] < max(static_totals)


def test_exact_shard_aggregation_rejects_duplicate_or_missing_keys() -> None:
    selection = [{"item_id": f"item-{index}"} for index in range(4)]
    assignments = [
        {"assignment_id": "s0", "selection_start": 0, "selection_stop": 2},
        {"assignment_id": "s1", "selection_start": 2, "selection_stop": 4},
    ]
    rows = {
        assignment["assignment_id"]: [
            {"item_id": selection[index]["item_id"], "arm": arm}
            for index in range(assignment["selection_start"], assignment["selection_stop"])
            for arm in runtime.ARMS
        ]
        for assignment in assignments
    }
    assert runtime.validate_exact_shard_key_union(
        selection_rows=selection,
        assignments=assignments,
        rows_by_assignment=rows,
    ) == []
    rows["s1"][0] = dict(rows["s0"][0])
    issues = runtime.validate_exact_shard_key_union(
        selection_rows=selection,
        assignments=assignments,
        rows_by_assignment=rows,
    )
    assert "cross_shard_duplicate" in issues
    assert "cross_shard_union" in issues


def test_common_window_mean_is_advisory_when_inference_and_projected_gates_pass() -> None:
    thresholds = {
        "telemetry_interval_seconds": 2.0,
        "minimum_common_allocation_mean_pct": 25.0,
        "common_allocation_mean_role": "descriptive_advisory_union_window_absent_rank_buckets_zero_filled",
        "minimum_worker_inference_mean_pct": 25.0,
        "minimum_worker_inference_p90_pct": 60.0,
        "minimum_worker_inference_sample_count": 5,
        "minimum_projected_rank_occupancy_fraction": 0.70,
        "maximum_projected_rank_tail_idle_fraction": 0.45,
    }
    gate = runtime.evaluate_utilization_tail_gate(
        common_allocation_mean_pct=23.682011411121827,
        minimum_worker_inference_mean_pct=70.0,
        minimum_worker_inference_p90_pct=60.1,
        minimum_worker_inference_sample_count=12,
        projected_rank_occupancy_fraction=0.98217222084804,
        projected_rank_tail_idle_fraction=0.0351050790193045,
        recommended_walltime_s=10_200,
        configured_walltime_s=10_800,
        thresholds=thresholds,
    )
    assert gate["status"] == "pass"
    assert gate["issues"] == []
    assert gate["advisories"] == ["common_allocation_mean_gpu_utilization"]
    failing = runtime.evaluate_utilization_tail_gate(
        common_allocation_mean_pct=80.0,
        minimum_worker_inference_mean_pct=70.0,
        minimum_worker_inference_p90_pct=59.9,
        minimum_worker_inference_sample_count=12,
        projected_rank_occupancy_fraction=0.90,
        projected_rank_tail_idle_fraction=0.10,
        recommended_walltime_s=10_200,
        configured_walltime_s=10_800,
        thresholds=thresholds,
    )
    assert failing["status"] == "fail"
    assert failing["issues"] == ["worker_inference_p90_gpu_utilization"]


def test_utilization_gate_blocks_each_measured_or_projected_failure() -> None:
    thresholds = {
        "telemetry_interval_seconds": 2.0,
        "minimum_common_allocation_mean_pct": 25.0,
        "common_allocation_mean_role": "descriptive_advisory_union_window_absent_rank_buckets_zero_filled",
        "minimum_worker_inference_mean_pct": 25.0,
        "minimum_worker_inference_p90_pct": 60.0,
        "minimum_worker_inference_sample_count": 5,
        "minimum_projected_rank_occupancy_fraction": 0.70,
        "maximum_projected_rank_tail_idle_fraction": 0.45,
    }
    gate = runtime.evaluate_utilization_tail_gate(
        common_allocation_mean_pct=10.0,
        minimum_worker_inference_mean_pct=24.9,
        minimum_worker_inference_p90_pct=59.9,
        minimum_worker_inference_sample_count=4,
        projected_rank_occupancy_fraction=0.69,
        projected_rank_tail_idle_fraction=0.46,
        recommended_walltime_s=10_801,
        configured_walltime_s=10_800,
        thresholds=thresholds,
    )
    assert gate["status"] == "fail"
    assert gate["advisories"] == ["common_allocation_mean_gpu_utilization"]
    assert gate["issues"] == [
        "worker_inference_mean_gpu_utilization",
        "worker_inference_p90_gpu_utilization",
        "worker_inference_sample_count",
        "projected_rank_occupancy_fraction",
        "projected_rank_tail_idle_fraction",
        "three_hour_walltime",
    ]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0, 101.0, True])
def test_utilization_gate_rejects_nonfinite_or_out_of_range_metrics(
    bad_value: float,
) -> None:
    thresholds = {
        "telemetry_interval_seconds": 2.0,
        "minimum_common_allocation_mean_pct": 25.0,
        "common_allocation_mean_role": "descriptive_advisory_union_window_absent_rank_buckets_zero_filled",
        "minimum_worker_inference_mean_pct": 25.0,
        "minimum_worker_inference_p90_pct": 60.0,
        "minimum_worker_inference_sample_count": 5,
        "minimum_projected_rank_occupancy_fraction": 0.70,
        "maximum_projected_rank_tail_idle_fraction": 0.45,
    }
    with pytest.raises(ValueError):
        runtime.evaluate_utilization_tail_gate(
            common_allocation_mean_pct=bad_value,
            minimum_worker_inference_mean_pct=80.0,
            minimum_worker_inference_p90_pct=95.0,
            minimum_worker_inference_sample_count=10,
            projected_rank_occupancy_fraction=0.9,
            projected_rank_tail_idle_fraction=0.1,
            recommended_walltime_s=10_200,
            configured_walltime_s=10_800,
            thresholds=thresholds,
        )


def test_utilization_gate_rejects_common_allocation_role_drift() -> None:
    thresholds = {
        "telemetry_interval_seconds": 2.0,
        "minimum_common_allocation_mean_pct": 25.0,
        "common_allocation_mean_role": "blocking",
        "minimum_worker_inference_mean_pct": 25.0,
        "minimum_worker_inference_p90_pct": 60.0,
        "minimum_worker_inference_sample_count": 5,
        "minimum_projected_rank_occupancy_fraction": 0.70,
        "maximum_projected_rank_tail_idle_fraction": 0.45,
    }
    with pytest.raises(ValueError, match="must remain descriptive and advisory"):
        runtime.evaluate_utilization_tail_gate(
            common_allocation_mean_pct=80.0,
            minimum_worker_inference_mean_pct=80.0,
            minimum_worker_inference_p90_pct=95.0,
            minimum_worker_inference_sample_count=10,
            projected_rank_occupancy_fraction=0.9,
            projected_rank_tail_idle_fraction=0.1,
            recommended_walltime_s=10_200,
            configured_walltime_s=10_800,
            thresholds=thresholds,
        )


def test_gpu_telemetry_window_and_common_zero_fill_are_explicit(tmp_path: Path) -> None:
    header = "timestamp,index,name,utilization_gpu_pct\n"
    paths = []
    for rank in range(4):
        path = tmp_path / f"rank-{rank}.csv"
        rows = [f"2026-07-21T00:00:00+00:00,{rank},H100,100"]
        if rank == 0:
            rows.append(f"2026-07-21T00:00:02+00:00,{rank},H100,100")
        path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
        paths.append(path)
    window = runtime._gpu_utilization_summary(
        paths[0],
        start_at="2026-07-21T00:00:02+00:00",
        end_at="2026-07-21T00:00:02+00:00",
    )
    assert window["status"] == "pass"
    assert window["sample_count"] == 1
    common = runtime._common_allocation_gpu_utilization_summary(
        paths, interval_seconds=2.0
    )
    assert common["status"] == "pass"
    assert common["rank_bucket_count"] == 8
    assert common["zero_filled_rank_bucket_count"] == 3
    assert common["mean_pct"] == pytest.approx(62.5)

    paths[1].write_text(
        header + "2026-07-21T00:00:00+00:00,1,H100,nan\n",
        encoding="utf-8",
    )
    assert runtime._gpu_utilization_summary(paths[1])["status"] == "fail"
    with pytest.raises(ValueError):
        runtime.canonical_sha256({"bad": float("nan")})


def test_full_profile_is_bound_to_exact_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, contract, model_plan_path = _tiny_battery(tmp_path)
    model_plan = json.loads(model_plan_path.read_text(encoding="utf-8"))
    reports = _schedule_reports([row["model_key"] for row in model_plan])
    schedule = runtime.build_balanced_full_schedule(reports, selection_count=4, arm_count=3)
    profile_models = [
        {
            "model_key": row["model_key"],
            "effective_batch_size": row["effective_batch_size"],
            "strategy": row["strategy"],
            "load_time_s": row["load_time_s"],
            "decisions_per_second": row["decisions_per_second"],
            "gpu_inference_window_utilization_distribution": {
                "status": "pass",
                "sample_count": 10,
                "mean_pct": 80.0,
                "p90_pct": 95.0,
            },
        }
        for row in reports
    ]
    thresholds = json.loads(contract.read_text(encoding="utf-8"))["canary"][
        "utilization_tail_gate"
    ]
    bindings = {
        "sif_sha256": "a" * 64,
        "release_manifest_sha256": "b" * 64,
        "data_manifest_sha256": "c" * 64,
        "hash_manifest_sha256": "d" * 64,
    }
    for field, environment in (
        ("sif_sha256", "SIF_SHA256"),
        ("release_manifest_sha256", "RELEASE_MANIFEST_SHA256"),
        ("data_manifest_sha256", "DATA_MANIFEST_SHA256"),
        ("hash_manifest_sha256", "HASH_MANIFEST_SHA256"),
    ):
        monkeypatch.setenv(environment, bindings[field])
    profile = {
        "kind": "phase2_reanchored_job_a_batch_profile",
        "status": "pass",
        "source_mode": "canary",
        "preregistration_sha256": runtime.EXPECTED_PREREGISTRATION_SHA256,
        "model_plan_sha256": runtime.sha256_file(model_plan_path),
        "contract_sha256": runtime.sha256_file(contract),
        "code_snapshot_sha256": runtime._code_snapshot_sha256(contract, model_plan_path),
        **bindings,
        "canary_measurements_sha256": runtime.canonical_sha256(profile_models),
        "fresh_canary_provenance": {
            "status": "pass",
            "required_rows_present_at_start": 0,
            "models": {
                row["model_key"]: {
                    "fresh_canary": True,
                    "rows_present_at_start": [0],
                }
                for row in model_plan
            },
        },
        "models": profile_models,
        "schedule": schedule,
        "common_allocation_gpu_utilization": {
            "status": "pass",
            "mean_pct": 10.0,
        },
        "full_runtime_estimate": {
            "balanced_schedule_projection_s": schedule["projected_makespan_s"],
            "recommended_walltime_s_rounded_to_5_minutes": int(
                math.ceil((schedule["projected_makespan_s"] * 1.20 + 600.0) / 300.0)
                * 300
            ),
            "configured_full_walltime_s": 10800,
            "fits_configured_full_walltime": True,
        },
    }
    profile["utilization_tail_gate"] = runtime.evaluate_utilization_tail_gate(
        common_allocation_mean_pct=10.0,
        minimum_worker_inference_mean_pct=80.0,
        minimum_worker_inference_p90_pct=95.0,
        minimum_worker_inference_sample_count=10,
        projected_rank_occupancy_fraction=schedule[
            "projected_aggregate_rank_occupancy_fraction"
        ],
        projected_rank_tail_idle_fraction=schedule[
            "projected_max_rank_tail_idle_fraction"
        ],
        recommended_walltime_s=profile["full_runtime_estimate"][
            "recommended_walltime_s_rounded_to_5_minutes"
        ],
        configured_walltime_s=10800,
        thresholds=thresholds,
    )
    path = tmp_path / "profile.json"
    _write_json(path, profile)
    loaded = runtime._load_batch_profile(
        path,
        model_plan_path=model_plan_path,
        contract_path=contract,
        model_plan=model_plan,
    )
    assert loaded["schedule"]["sha256"] == schedule["sha256"]

    measurement_drift = json.loads(json.dumps(profile))
    measurement_drift["models"][0]["gpu_inference_window_utilization_distribution"][
        "mean_pct"
    ] = 5.0
    measurement_drift["canary_measurements_sha256"] = runtime.canonical_sha256(
        measurement_drift["models"]
    )
    _write_json(path, measurement_drift)
    with pytest.raises(ValueError, match="gate recomputation drift"):
        runtime._load_batch_profile(
            path,
            model_plan_path=model_plan_path,
            contract_path=contract,
            model_plan=model_plan,
        )

    profile["schedule"]["ranks"][0]["assignments"][0]["selection_stop"] = 2
    profile["schedule"]["sha256"] = runtime.canonical_sha256(
        {key: value for key, value in profile["schedule"].items() if key != "sha256"}
    )
    _write_json(path, profile)
    with pytest.raises(ValueError, match="bound canary measurements"):
        runtime._load_batch_profile(
            path,
            model_plan_path=model_plan_path,
            contract_path=contract,
            model_plan=model_plan,
        )


def test_launcher_enforces_worker_hidden_firewall_and_clean_binds() -> None:
    launcher = (JOB_DIR / "run_tamia_reanchored_jobA.sh").read_text(encoding="utf-8")
    worker = (JOB_DIR / "tamia_jobA_worker.sh").read_text(encoding="utf-8")
    technical_gate = launcher.index("=== deterministic technical verification ===")
    hidden_stage = launcher.index("=== stage sealed evaluator only after technical completeness passes ===")
    assert hidden_stage > technical_gate
    assert "rsync -a --delete \"$SOURCE_DATA\"/ \"$LOCAL_DATA\"/" not in launcher
    assert "--export=PATH=" in launcher
    assert "srun --export=ALL" not in launcher
    assert "--gpus-per-task=h100:1 --gpu-bind=single:1" in launcher
    assert '--env SLURM_JOB_ID="$SLURM_JOB_ID"' in launcher
    assert '--env SLURM_JOB_NAME="$SLURM_JOB_NAME"' in launcher
    assert '--env SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST"' in launcher
    assert "--cleanenv --containall" in worker
    assert '-B "$LOCAL_DATA":"$LOCAL_DATA":ro' in worker
    assert "SOURCE_DATA" not in worker
    assert 'export CUDA_VISIBLE_DEVICES="$SLURM_LOCALID"' not in worker
    assert ': "${CUDA_VISIBLE_DEVICES:?Slurm did not bind a GPU to this worker}"' in worker


def test_full_preflight_container_receives_profile_provenance_hashes() -> None:
    launcher = (JOB_DIR / "run_tamia_reanchored_jobA.sh").read_text(encoding="utf-8")
    start = launcher.index(
        'echo "=== coordinator-only sealed-evaluator hash and public battery preflight ==="'
    )
    stop = launcher.index(
        'echo "=== stage only public manifests, packs, and selected assets ==="'
    )
    preflight = launcher[start:stop]
    assert "apptainer exec --cleanenv --containall" in preflight
    for name in (
        "SIF_SHA256",
        "RELEASE_MANIFEST_SHA256",
        "DATA_MANIFEST_SHA256",
        "HASH_MANIFEST_SHA256",
    ):
        assert f'--env {name}="${name}"' in preflight


def test_full_submit_uses_canary_derived_walltime() -> None:
    submit = (JOB_DIR / "submit_full.sh").read_text(encoding="utf-8")
    assert 'profile["kind"] == "phase2_reanchored_job_a_batch_profile"' in submit
    assert 'profile["status"] == "pass"' in submit
    assert 'profile["fresh_canary_provenance"]' in submit
    assert 'recommended_walltime_s_rounded_to_5_minutes' in submit
    assert 'seconds = max(3600, seconds)' in submit
    assert 'sbatch "${hold_args[@]}" --time="$WALLTIME"' in submit
    assert "--export=ALL" not in submit


def test_job_a_batch_wrappers_resolve_package_outside_slurm_spool() -> None:
    for name in (
        "job_tamia_reanchored_jobA_canary_h100.sh",
        "job_tamia_reanchored_jobA_full_h100.sh",
    ):
        wrapper = (JOB_DIR / name).read_text(encoding="utf-8")
        assert "JOB_A_PACKAGE_DIR" in wrapper
        assert "SLURM_SUBMIT_DIR" in wrapper
        assert 'dirname "${BASH_SOURCE[0]}"' not in wrapper
        assert 'test -f "$script_dir/run_tamia_reanchored_jobA.sh"' in wrapper
    assert 'JOB_A_PACKAGE_DIR=$script_dir' in (
        JOB_DIR / "submit_canary.sh"
    ).read_text(encoding="utf-8")
    assert 'JOB_A_PACKAGE_DIR=$script_dir' in (
        JOB_DIR / "submit_full.sh"
    ).read_text(encoding="utf-8")
