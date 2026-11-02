from __future__ import annotations

import hashlib
from pathlib import Path

from research.lifecycle_memory import analyze_phase2_reanchored_answer_token_confirmation as analysis
from research.lifecycle_memory import run_phase2_reanchored_behavior as runtime


REPO_ROOT = Path(__file__).resolve().parents[3]
JOB_DIR = REPO_ROOT / "docs/research/lifecycle_memory/phase 2 research/jobs/reanchored_jobAR_answer_token"
PREREG = REPO_ROOT / "docs/research/lifecycle_memory/phase 2 research/preregistration/reanchored_choice_likelihood_confirmation_v1_2026_07_22.json"


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        mapping = {
            '{"answer":"': [10, 11],
            "A": [20],
            "B": [21],
            '{"answer":"A': [10, 11, 20],
            '{"answer":"B': [10, 11, 21],
        }
        return mapping[text]


def _row(a_score: float, b_score: float) -> dict:
    return {
        "request_id": "request-real",
        "teacher_forced": {
            "candidate_rows": {
                "A": {"sum_logprob": a_score},
                "B": {"sum_logprob": b_score},
            }
        },
        "greedy": {
            "strict_choice": {
                "valid": False,
                "answer": None,
                "abstain": False,
            }
        },
    }


def test_answer_token_spec_requires_single_exact_boundary_tokens() -> None:
    prefix, prefix_ids, answers = runtime._answer_token_spec(
        _Tokenizer(),
        {
            "candidate_shared_prefix": '{"answer":"',
            "candidate_completions": {"A": "A", "B": "B"},
        },
    )
    assert prefix == '{"answer":"'
    assert prefix_ids == [10, 11]
    assert answers == {"A": [20], "B": [21]}


def test_joined_trials_uses_token_choice_not_invalid_greedy_json() -> None:
    joined = analysis._joined_trials(
        rows={("item-1", "real"): _row(-0.1, -2.0)},
        hidden={
            "item-1": {
                "request_ids": {"real": "request-real"},
                "frame_unit_id": "frame-1",
                "participant_id": "participant-1",
                "first_order_class": 3,
                "second_order_class": 7,
                "correct_choice": "A",
            }
        },
    )
    assert joined[0]["predicted_choice"] == "A"
    assert joined[0]["correct"] is True
    assert joined[0]["strict_greedy_parse_valid"] is False
    assert joined[0]["strict_greedy_correct"] is False


def test_answer_token_tie_is_invalid_and_incorrect() -> None:
    joined = analysis._joined_trials(
        rows={("item-1", "real"): _row(-1.0, -1.0)},
        hidden={
            "item-1": {
                "request_ids": {"real": "request-real"},
                "frame_unit_id": "frame-1",
                "participant_id": "participant-1",
                "first_order_class": 3,
                "second_order_class": 7,
                "correct_choice": "A",
            }
        },
    )
    assert joined[0]["predicted_choice"] is None
    assert joined[0]["answer_token_tie"] is True
    assert joined[0]["correct"] is False


def test_two_anchor_schedule_covers_each_cohort_once() -> None:
    prereg = runtime.read_json(PREREG)
    plan = runtime._load_model_plan(JOB_DIR / "model_plan.json", prereg)
    reports = [
        {
            "model_key": row["model_key"],
            "analysis_model_key": row["analysis_model_key"],
            "worker_index": row["worker_index"],
            "load_time_s": 60.0,
            "decisions_per_second": 2.0,
            "effective_batch_size": 2,
            "strategy": "batched",
        }
        for row in plan
    ]
    schedule = runtime.build_balanced_full_schedule(
        reports, selection_count=640, arm_count=3
    )
    assert schedule["construction"] == (
        "two_anchor_models_two_fixed_counterbalance_aligned_half_shards_each"
    )
    assignments = [
        assignment
        for rank in schedule["ranks"]
        for assignment in rank["assignments"]
    ]
    for model_key in ("qwen35_27b_hf_bf16", "gemma4_31b_hf_bf16"):
        intervals = sorted(
            (row["selection_start"], row["selection_stop"])
            for row in assignments
            if row["analysis_model_key"] == model_key
        )
        assert intervals == [(0, 320), (320, 640)]


def test_replicated_anchor_union_accepts_exact_halves_and_rejects_overlap() -> None:
    selection = [{"item_id": f"item-{index}"} for index in range(4)]
    instances = [{"model_key": "anchor-rank0"}, {"model_key": "anchor-rank1"}]
    assignments = {
        "anchor-rank0": [{"selection_start": 0, "selection_stop": 2}],
        "anchor-rank1": [{"selection_start": 2, "selection_stop": 4}],
    }
    keys = {
        model_key: [
            (selection[index]["item_id"], arm)
            for assignment in rows
            for index in range(
                assignment["selection_start"], assignment["selection_stop"]
            )
            for arm in runtime.ARMS
        ]
        for model_key, rows in assignments.items()
    }
    exact = runtime.audit_replicated_anchor_union(
        selection_rows=selection,
        instances=instances,
        instance_keys=keys,
        instance_assignments=assignments,
    )
    assert exact["status"] == "pass"
    assert exact["observed_row_count"] == exact["expected_row_count"] == 12

    assignments["anchor-rank1"] = [{"selection_start": 1, "selection_stop": 4}]
    keys["anchor-rank1"] = [
        (selection[index]["item_id"], arm)
        for index in range(1, 4)
        for arm in runtime.ARMS
    ]
    overlap = runtime.audit_replicated_anchor_union(
        selection_rows=selection,
        instances=instances,
        instance_keys=keys,
        instance_assignments=assignments,
    )
    assert overlap["status"] == "fail"
    assert "cross_instance_duplicate" in overlap["issues"]
    assert "cross_instance_interval_coverage" in overlap["issues"]


def test_single_instance_full_union_remains_legacy_compatible() -> None:
    selection = [{"item_id": f"item-{index}"} for index in range(4)]
    keys = [
        (row["item_id"], arm)
        for row in selection
        for arm in runtime.ARMS
    ]
    audit = runtime.audit_replicated_anchor_union(
        selection_rows=selection,
        instances=[{"model_key": "legacy-anchor"}],
        instance_keys={"legacy-anchor": keys},
        instance_assignments={
            "legacy-anchor": [{"selection_start": 0, "selection_stop": 4}]
        },
    )
    assert audit["status"] == "not_applicable_single_instance"
    assert audit["issues"] == []


def test_answer_token_row_contract_fails_closed_before_label_join() -> None:
    prefix = '{"answer":"'
    common = {
        "model_key": "m",
        "model_id": "id",
        "revision": "rev",
        "dtype": "bf16",
        "worker_index": 0,
        "selection_manifest_sha256": "selection",
        "preregistration_sha256": runtime.EXPECTED_PREREGISTRATION_SHA256,
        "code_snapshot_sha256": "code",
        "candidate_A_text": "A",
        "candidate_B_text": "B",
        "candidate_shared_prefix": prefix,
    }

    def candidate(choice: str, token_id: int) -> dict:
        return {
            "status": "pass",
            "visual_identity_status": "pass",
            "candidate_construction": "frozen_shared_prefix_then_exact_answer_token_append",
            "candidate_text_sha256": hashlib.sha256(choice.encode()).hexdigest(),
            "shared_prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
            "shared_prefix_token_ids": [10, 11],
            "scored_token_ids": [token_id],
            "combined_prefix_and_answer_token_ids": [10, 11, token_id],
            "token_boundary_status": "pass",
            "token_count": 1,
            "sum_logprob": -1.0,
        }

    row = {
        "kind": "phase2_reanchored_job_a_worker_row",
        "status": "pass",
        **common,
        "item_id": "item",
        "counterbalance_pair_index": 0,
        "arm": "real",
        "teacher_forced": {
            "candidate_rows": {"A": candidate("A", 20), "B": candidate("B", 21)}
        },
        "greedy": {"status": "pass", "strict_choice": {"valid": True, "answer": "A"}},
    }
    row["row_sha256"] = runtime.row_hash(row)
    assert runtime.validate_completed_row(row, expected_common=common) == []

    row["teacher_forced"]["candidate_rows"]["B"]["scored_token_ids"] = [20]
    row["teacher_forced"]["candidate_rows"]["B"][
        "combined_prefix_and_answer_token_ids"
    ] = [10, 11, 20]
    row["row_sha256"] = runtime.row_hash(row)
    assert "candidate_answer_token_collision" in runtime.validate_completed_row(
        row, expected_common=common
    )
