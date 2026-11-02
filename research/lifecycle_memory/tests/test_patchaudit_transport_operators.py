from __future__ import annotations

import pytest
import torch

from research.lifecycle_memory.patchaudit_transport_operators import (
    PatchInstruction,
    TransportGateError,
    TransportSite,
    active_positions_from_inputs,
    assert_complete_capture,
    capture_transport_states,
    patch_transport_outputs,
    position_partition,
    recipient_positions_for_operator,
    selected_from_capture,
    validate_patch_audits,
)


class AddBlock(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.value


class ToyModel(torch.nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(AddBlock(value) for value in values)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def sites(count: int) -> list[TransportSite]:
    return [
        TransportSite(
            site_id=f"post_block_{index}",
            module_name=f"layers.{index}",
            layer_index=index,
        )
        for index in range(count)
    ]


def test_position_partition_exactly_covers_active_positions() -> None:
    partition = position_partition(
        sequence_length=8,
        active_positions=(0, 1, 2, 3, 4, 5),
        full_frame_positions=(1, 2),
    )
    assert partition.complement_positions == (0, 3, 4, 5)
    assert partition.padding_positions == (6, 7)
    assert set(partition.full_frame_positions).isdisjoint(
        partition.complement_positions
    )
    assert set(partition.full_frame_positions) | set(
        partition.complement_positions
    ) == set(partition.active_positions)


def test_partition_rejects_full_frame_outside_active_set() -> None:
    with pytest.raises(TransportGateError, match="subset"):
        position_partition(
            sequence_length=5,
            active_positions=(0, 1, 2),
            full_frame_positions=(2, 3),
        )


def test_active_positions_from_attention_mask() -> None:
    observed = active_positions_from_inputs(
        {"attention_mask": torch.tensor([[1, 1, 1, 0, 0]])},
        sequence_length=5,
    )
    assert observed == (0, 1, 2)


def test_persistent_clamp_uses_native_layer_specific_donor_states() -> None:
    model = ToyModel([1.0, 2.0, 3.0, 4.0])
    inventory = sites(4)
    partition = position_partition(
        sequence_length=6,
        active_positions=tuple(range(6)),
        full_frame_positions=(1, 2),
    )
    donor = torch.zeros((1, 6, 3))
    with capture_transport_states(
        model=model,
        sites=inventory,
        partition=partition,
        detailed_site_ids=(),
    ) as captures:
        donor_output = model(donor)
    assert_complete_capture(captures, inventory, detailed_site_ids=())
    instructions = [
        PatchInstruction(
            site=site,
            source_selected=selected_from_capture(
                captures[site.site_id],
                operator="persistent_full_frame",
                partition=partition,
            ),
            recipient_positions=partition.full_frame_positions,
            operator="persistent_full_frame",
        )
        for site in inventory[1:]
    ]
    recipient = torch.full((1, 6, 3), 10.0)
    with patch_transport_outputs(
        model=model,
        instructions=instructions,
        sequence_length=6,
    ) as audits:
        patched = model(recipient)
    validate_patch_audits(
        audits,
        expected_site_ids=[site.site_id for site in inventory[1:]],
        allow_noop=False,
    )
    assert torch.equal(patched[:, 1:3], donor_output[:, 1:3])
    assert torch.equal(patched[:, 0], torch.full((1, 3), 20.0))
    assert all(not layer._forward_hooks for layer in model.layers)


def test_complement_patch_preserves_full_frame_positions() -> None:
    model = ToyModel([1.0])
    site = sites(1)[0]
    partition = position_partition(
        sequence_length=5,
        active_positions=(0, 1, 2, 3, 4),
        full_frame_positions=(1, 2),
    )
    donor = torch.zeros((1, 5, 2), dtype=torch.bfloat16)
    with capture_transport_states(
        model=model,
        sites=[site],
        partition=partition,
        detailed_site_ids=[site.site_id],
    ) as captures:
        donor_output = model(donor)
    source = selected_from_capture(
        captures[site.site_id],
        operator="complement_once",
        partition=partition,
    )
    instruction = PatchInstruction(
        site=site,
        source_selected=source,
        recipient_positions=recipient_positions_for_operator(
            "complement_once", partition
        ),
        operator="complement_once",
    )
    recipient = torch.full((1, 5, 2), 10.0, dtype=torch.bfloat16)
    with patch_transport_outputs(
        model=model,
        instructions=[instruction],
        sequence_length=5,
    ) as audits:
        patched = model(recipient)
    validate_patch_audits(
        audits, expected_site_ids=[site.site_id], allow_noop=False
    )
    assert torch.equal(patched[:, 1:3], torch.full((1, 2, 2), 11.0))
    assert torch.equal(patched[:, (0, 3, 4)], donor_output[:, (0, 3, 4)])


def test_full_sequence_patch_is_exact_donor_contract_control() -> None:
    model = ToyModel([1.0, 2.0])
    site = sites(2)[0]
    partition = position_partition(
        sequence_length=4,
        active_positions=(0, 1, 2, 3),
        full_frame_positions=(1,),
    )
    donor = torch.zeros((1, 4, 2))
    with capture_transport_states(
        model=model,
        sites=[site],
        partition=partition,
        detailed_site_ids=[site.site_id],
    ) as captures:
        donor_output = model(donor)
    instruction = PatchInstruction(
        site=site,
        source_selected=selected_from_capture(
            captures[site.site_id],
            operator="full_sequence_once",
            partition=partition,
        ),
        recipient_positions=partition.active_positions,
        operator="full_sequence_once",
    )
    with patch_transport_outputs(
        model=model,
        instructions=[instruction],
        sequence_length=4,
    ) as audits:
        patched = model(torch.full((1, 4, 2), 10.0))
    validate_patch_audits(
        audits, expected_site_ids=[site.site_id], allow_noop=False
    )
    assert torch.equal(patched, donor_output)


def test_final_layer_clamp_equals_single_patch() -> None:
    model = ToyModel([1.0, 2.0])
    final_site = sites(2)[1]
    partition = position_partition(
        sequence_length=4,
        active_positions=(0, 1, 2, 3),
        full_frame_positions=(1, 2),
    )
    donor = torch.zeros((1, 4, 2))
    with capture_transport_states(
        model=model,
        sites=[final_site],
        partition=partition,
        detailed_site_ids=(),
    ) as captures:
        model(donor)
    source = captures[final_site.site_id].full_frame
    outputs = []
    for operator in ("single_full_frame", "persistent_full_frame"):
        instruction = PatchInstruction(
            site=final_site,
            source_selected=source,
            recipient_positions=partition.full_frame_positions,
            operator=operator,
        )
        with patch_transport_outputs(
            model=model,
            instructions=[instruction],
            sequence_length=4,
        ) as audits:
            outputs.append(model(torch.full((1, 4, 2), 10.0)))
        validate_patch_audits(
            audits, expected_site_ids=[final_site.site_id], allow_noop=False
        )
    assert torch.equal(outputs[0], outputs[1])


def test_hook_cleanup_after_shape_failure() -> None:
    model = ToyModel([1.0])
    site = sites(1)[0]
    instruction = PatchInstruction(
        site=site,
        source_selected=torch.zeros((1, 99, 2)),
        recipient_positions=(1,),
        operator="single_full_frame",
    )
    audits = None
    with pytest.raises(TransportGateError, match="shape mismatch"):
        with patch_transport_outputs(
            model=model,
            instructions=[instruction],
            sequence_length=4,
        ) as active_audits:
            audits = active_audits
            model(torch.zeros((1, 4, 2)))
    assert audits is not None
    assert audits[0].hook_removed is True
    assert not model.layers[0]._forward_hooks
