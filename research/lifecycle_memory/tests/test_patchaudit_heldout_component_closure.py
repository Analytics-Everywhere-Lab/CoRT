from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from research.lifecycle_memory import (
    patchaudit_component_partition as partitioning,
)
from research.lifecycle_memory import (
    run_patchaudit_heldout_component_closure as runner,
)
from research.lifecycle_memory import (
    run_phase2_reanchored_answer_token_causal_patch as answer_runner,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = (
    REPO_ROOT
    / "docs/research/lifecycle_memory/phase 2 research/jobs/"
    "patchaudit_jobE_heldout_component_closure/patch_contract.json"
)


def _partition() -> partitioning.ComponentPartition:
    return partitioning.build_component_partition(
        sequence_length=22,
        active_positions=tuple(range(20)),
        image_slot_positions=(
            (0, 1),
            (2, 3, 4, 5),
            (6, 7, 8, 9),
        ),
        literal_user_prompt_start=10,
        literal_user_prompt_count=4,
        answer_prefix_token_count=2,
    )


def test_partition_uses_prepared_arm_question_and_serialization(
    monkeypatch,
) -> None:
    slot_map = SimpleNamespace(
        sequence_length=20,
        positions=lambda indexes: {
            (0,): (0, 1),
            (1,): (2, 3, 4, 5),
            (2,): (6, 7, 8, 9),
        }[tuple(indexes)],
    )
    shared_inputs = object()
    choices = {
        choice: answer_runner.PreparedChoice(
            choice=choice,
            inputs=shared_inputs,
            base_length=20,
            full_ids=list(range(21)),
            candidate_ids=[1 if choice == "A" else 2],
            slot_map=slot_map,
            construction="one_shared_prefix_forward_next_token_score",
            policy_trace_sha256="a" * 64,
        )
        for choice in ("A", "B")
    }
    prepared = answer_runner.PreparedArm(
        arm="real",
        base_inputs=shared_inputs,
        base_messages=[],
        base_ids=list(range(20)),
        question_position=13,
        question_token_audit={
            "status": "pass",
            "index": 13,
            "question_token_count": 4,
        },
        slot_map=slot_map,
        choices=choices,
        serialization={
            "answer_token_contract": {
                "shared_prefix_token_ids": [101, 102],
            }
        },
        asset_receipts=[],
    )
    monkeypatch.setattr(
        runner,
        "active_positions_from_inputs",
        lambda inputs, sequence_length: tuple(range(sequence_length)),
    )

    partition = runner._partition_for_prepared(prepared)

    assert partition.literal_user_prompt.positions == (10, 11, 12, 13)
    assert partition.answer_prefix.positions == (18, 19)
    assert partition.candidate_crops.positions == tuple(range(2, 10))


def test_frozen_contract_has_exact_reviewed_inventory() -> None:
    contract, prespec, hashes = runner.verify_contract(
        repo_dir=REPO_ROOT,
        contract_path=CONTRACT,
    )
    assert contract["completion"]["expected_full_outcome_rows"] == 13_984
    assert contract["canary"]["expected_outcome_rows"] == 408
    assert runner.expected_outcomes_per_request(
        "qwen35_27b_hf_bf16", prespec
    ) == 24
    assert runner.expected_outcomes_per_request(
        "gemma4_31b_hf_bf16", prespec
    ) == 128
    assert len(hashes["contract_sha256"]) == 64
    assert len(hashes["prespec_sha256"]) == 64


def test_layer_specific_operator_plan_is_not_cartesian_overexpanded() -> None:
    _, prespec, _ = runner.verify_contract(
        repo_dir=REPO_ROOT,
        contract_path=CONTRACT,
    )
    assert runner.operators_for_layer(
        "gemma4_31b_hf_bf16", 22, prespec
    ) == ("F", "K", "A")
    late = runner.operators_for_layer(
        "gemma4_31b_hf_bf16", 34, prespec
    )
    assert len(late) == 29
    assert late[:2] == ("F", "C")
    assert "CQPR" in late
    assert late[-1] == "A"
    assert runner.operators_for_layer(
        "qwen35_27b_hf_bf16", 24, prespec
    ) == ("F", "K", "A")


def test_operator_positions_cover_full_complement_and_active() -> None:
    _, prespec, _ = runner.verify_contract(
        repo_dir=REPO_ROOT,
        contract_path=CONTRACT,
    )
    partition = _partition()
    full, _ = runner._operator_positions(
        operator="F",
        model_key="gemma4_31b_hf_bf16",
        layer=22,
        partition=partition,
        prespec=prespec,
    )
    complement, _ = runner._operator_positions(
        operator="K",
        model_key="gemma4_31b_hf_bf16",
        layer=22,
        partition=partition,
        prespec=prespec,
    )
    active, _ = runner._operator_positions(
        operator="A",
        model_key="gemma4_31b_hf_bf16",
        layer=22,
        partition=partition,
        prespec=prespec,
    )
    assert set(full).isdisjoint(complement)
    assert tuple(sorted(full + complement)) == active


def test_gemma_late_factorial_and_benchmarks_are_exact() -> None:
    _, prespec, _ = runner.verify_contract(
        repo_dir=REPO_ROOT,
        contract_path=CONTRACT,
    )
    partition = _partition()
    factorial_hashes = set()
    for operator in runner.GEMMA_FACTORIAL_OPERATORS:
        positions, receipt = runner._operator_positions(
            operator=operator,
            model_key="gemma4_31b_hf_bf16",
            layer=34,
            partition=partition,
            prespec=prespec,
        )
        factorial_hashes.add(runner.sha256_json(list(positions)))
        assert receipt["type"] == "factorial_component_union"
    assert len(factorial_hashes) == 15

    by_group: dict[str, set[str]] = {}
    for operator in runner.GEMMA_CARDINALITY_BENCHMARK_OPERATORS:
        positions, receipt = runner._operator_positions(
            operator=operator,
            model_key="gemma4_31b_hf_bf16",
            layer=34,
            partition=partition,
            prespec=prespec,
        )
        group = str(receipt["target_group"])
        by_group.setdefault(group, set()).add(
            runner.sha256_json(list(positions))
        )
        assert set(positions) <= set(partition.candidate_crops.positions)
        assert len(positions) == partition.component(group).count
    assert all(len(values) == 4 for values in by_group.values())


def test_expected_identity_inventory_uses_per_layer_operators() -> None:
    _, prespec, _ = runner.verify_contract(
        repo_dir=REPO_ROOT,
        contract_path=CONTRACT,
    )
    identities = runner.expected_outcome_identities(
        prespec=prespec,
        model_key="gemma4_31b_hf_bf16",
        request_ids=["r1"],
    )
    assert len(identities) == 128
    assert (
        "gemma4_31b_hf_bf16",
        "r1",
        "blank_full",
        "rescue",
        22,
        "CQPR",
    ) not in identities
    assert (
        "gemma4_31b_hf_bf16",
        "r1",
        "blank_full",
        "rescue",
        34,
        "CQPR",
    ) in identities
