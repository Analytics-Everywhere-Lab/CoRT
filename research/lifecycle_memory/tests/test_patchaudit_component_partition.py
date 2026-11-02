from __future__ import annotations

import hashlib
import json

import pytest

from research.lifecycle_memory import patchaudit_component_partition as partitioning


def _partition(
    *,
    active_positions: tuple[int, ...] = tuple(range(20)),
    image_slot_positions: tuple[tuple[int, ...], ...] = (
        (0, 1),
        (2, 3, 4, 5),
        (6, 7, 8, 9),
    ),
    prompt_start: int = 10,
    prompt_count: int = 4,
    prefix_count: int = 2,
) -> partitioning.ComponentPartition:
    return partitioning.build_component_partition(
        sequence_length=22,
        active_positions=active_positions,
        image_slot_positions=image_slot_positions,
        literal_user_prompt_start=prompt_start,
        literal_user_prompt_count=prompt_count,
        answer_prefix_token_count=prefix_count,
    )


def _hash_positions(positions: tuple[int, ...]) -> str:
    raw = json.dumps(list(positions), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def test_partition_is_disjoint_exhaustive_and_hash_bound() -> None:
    partition = _partition()

    assert partition.full_frame.positions == (0, 1)
    assert partition.candidate_crops.positions == tuple(range(2, 10))
    assert partition.literal_user_prompt.positions == (10, 11, 12, 13)
    assert partition.answer_prefix.positions == (18, 19)
    assert partition.structural_remainder.positions == (14, 15, 16, 17)
    assert partition.complement.positions == tuple(range(2, 20))
    assert partition.padding.positions == (20, 21)
    assert partition.answer_prefix_start == 18

    components = partition.components()
    for left_index, left in enumerate(components):
        for right in components[left_index + 1 :]:
            assert set(left.positions).isdisjoint(right.positions)
    assert {
        position for group in components for position in group.positions
    } == set(partition.active_positions)

    assert partition.active_count == 20
    assert partition.complement_count == 18
    assert partition.active_sha256 == _hash_positions(tuple(range(20)))
    assert partition.complement_sha256 == _hash_positions(tuple(range(2, 20)))
    assert partition.image_slots[0].sha256 == partition.full_frame.sha256
    assert partition.as_dict()["candidate_crops"]["count"] == 8


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"image_slot_positions": ((0, 1), (1, 2), (3, 4))},
            "overlap",
        ),
        (
            {"image_slot_positions": ((0, 1), (2, 3), (4, 22))},
            "out-of-range",
        ),
        (
            {"image_slot_positions": ((), (2, 3), (4, 5))},
            "full-frame",
        ),
        (
            {"image_slot_positions": ((0, 1), (), (4, 5))},
            "candidate-crop",
        ),
        (
            {"image_slot_positions": ((0, 1), (2, 3), ())},
            "candidate-crop",
        ),
        (
            {"prompt_start": 8},
            "overlap",
        ),
    ],
)
def test_partition_rejects_invalid_image_groups_and_overlap(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(partitioning.ComponentPartitionError, match=message):
        _partition(**overrides)  # type: ignore[arg-type]


def test_partition_rejects_noncontiguous_prompt_boundary() -> None:
    active = tuple(position for position in range(20) if position != 12)
    with pytest.raises(
        partitioning.ComponentPartitionError,
        match="prompt boundary is not contiguous",
    ):
        _partition(active_positions=active)


def test_partition_rejects_noncontiguous_prefix_boundary() -> None:
    active = tuple(range(19)) + (20,)
    with pytest.raises(
        partitioning.ComponentPartitionError,
        match="prefix boundary is not contiguous",
    ):
        _partition(active_positions=active)


def test_partition_rejects_prefix_before_prompt_end() -> None:
    with pytest.raises(
        partitioning.ComponentPartitionError,
        match="prefix begins before",
    ):
        _partition(prefix_count=7)


def test_partition_rejects_nonexhaustive_component_inventory() -> None:
    active = partitioning.PositionGroup.from_positions("active", (0, 1, 2))
    incomplete = (
        partitioning.PositionGroup.from_positions("full_frame", (0,)),
        partitioning.PositionGroup.from_positions("candidate_crops", (1,)),
    )
    with pytest.raises(
        partitioning.ComponentPartitionError,
        match="not exhaustive",
    ):
        partitioning._validate_exhaustive_partition(
            active=active, components=incomplete
        )


def test_full_factorial_contains_each_nonempty_union_once() -> None:
    design = partitioning.factorial_component_unions(_partition())

    assert design.component_order == partitioning.FACTORIAL_GROUP_NAMES
    assert design.structurally_absent_groups == ()
    assert design.requested_combination_count == 15
    assert design.nonempty_requested_combination_count == 15
    assert design.empty_requested_group_sets == ()
    assert design.duplicate_requested_combination_count == 0
    assert len(design.cells) == 15
    assert len({cell.positions for cell in design.cells}) == 15
    assert all(cell.count > 0 for cell in design.cells)
    assert all(len(cell.requested_group_sets) == 1 for cell in design.cells)


def test_factorial_deduplicates_absent_group_and_records_it() -> None:
    partition = partitioning.build_component_partition(
        sequence_length=16,
        active_positions=tuple(range(16)),
        image_slot_positions=((0, 1), (2, 3, 4, 5), (6, 7, 8, 9)),
        literal_user_prompt_start=10,
        literal_user_prompt_count=4,
        answer_prefix_token_count=2,
    )
    assert partition.structural_remainder.structurally_absent

    first = partitioning.factorial_component_unions(partition)
    second = partitioning.factorial_component_unions(partition)

    assert first == second
    assert first.structurally_absent_groups == ("structural_remainder",)
    assert first.requested_combination_count == 15
    assert first.nonempty_requested_combination_count == 14
    assert first.empty_requested_group_sets == (("structural_remainder",),)
    assert first.duplicate_requested_combination_count == 7
    assert len(first.cells) == 7
    assert all(len(cell.requested_group_sets) == 2 for cell in first.cells)
    assert all(
        any("structural_remainder" in alias for alias in cell.requested_group_sets)
        for cell in first.cells
    )


def test_candidate_crop_control_is_deterministic_and_same_cardinality() -> None:
    partition = _partition()
    first = partitioning.sample_candidate_crop_control(
        partition,
        namespace="patchaudit-jobE-v1",
        model="qwen35_27b_hf_bf16",
        group="literal_user_prompt",
    )
    second = partitioning.sample_candidate_crop_control(
        partition,
        namespace="patchaudit-jobE-v1",
        model="qwen35_27b_hf_bf16",
        group="literal_user_prompt",
    )

    assert first == second
    assert first.count == partition.literal_user_prompt.count
    assert set(first.positions) <= set(partition.candidate_crops.positions)
    assert first.candidate_crop_pool_sha256 == partition.candidate_crops.sha256
    assert not first.structurally_absent


def test_all_nonvisual_controls_are_named_and_size_matched() -> None:
    partition = _partition()
    controls = partitioning.nonvisual_candidate_crop_controls(
        partition,
        namespace="patchaudit-jobE-v1",
        model="gemma4_31b_hf_bf16",
    )

    assert tuple(controls) == partitioning.NONVISUAL_GROUP_NAMES
    for name, control in controls.items():
        assert control.count == partition.component(name).count
        assert set(control.positions) <= set(partition.candidate_crops.positions)


def test_candidate_crop_control_rejects_oversized_request() -> None:
    with pytest.raises(
        partitioning.ComponentPartitionError,
        match="exceeds candidate-crop pool",
    ):
        partitioning.deterministic_candidate_crop_sample(
            candidate_crop_positions=(2, 3, 4),
            sample_count=4,
            namespace="patchaudit-jobE-v1",
            model="qwen35_27b_hf_bf16",
            group="literal_user_prompt",
        )


def test_candidate_crop_control_records_structurally_absent_target() -> None:
    partition = partitioning.build_component_partition(
        sequence_length=16,
        active_positions=tuple(range(16)),
        image_slot_positions=((0, 1), (2, 3, 4, 5), (6, 7, 8, 9)),
        literal_user_prompt_start=10,
        literal_user_prompt_count=4,
        answer_prefix_token_count=2,
    )
    control = partitioning.sample_candidate_crop_control(
        partition,
        namespace="patchaudit-jobE-v1",
        model="qwen35_27b_hf_bf16",
        group="structural_remainder",
    )
    assert control.structurally_absent
    assert control.count == 0
    assert control.positions == ()
