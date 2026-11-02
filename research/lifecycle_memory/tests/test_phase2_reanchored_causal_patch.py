from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory import analyze_phase2_reanchored_causal_patch as analysis
from research.lifecycle_memory import run_phase2_reanchored_causal_patch as runner


def slot_map(sequence_length: int = 8) -> runner.ImageSlotMap:
    slots = tuple(
        runner.SlotBoundary(
            slot_index=index,
            slot_name=runner.IMAGE_SLOT_NAMES[index],
            decoder_positions=positions,
            decoder_start=positions[0],
            decoder_stop=positions[-1] + 1,
            decoder_count=len(positions),
            premerge_count=len(positions),
        )
        for index, positions in enumerate(((1, 2), (3, 4), (5, 6)))
    )
    return runner.ImageSlotMap(
        image_token_id=99,
        sequence_length=sequence_length,
        slots=slots,  # type: ignore[arg-type]
        grid_triplets=((1, 2, 2),) * 3,
        processor_signature_sha256="a" * 64,
    )


class TupleBlock(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value + 1.0,)


def visual_site() -> runner.PatchSite:
    return runner.PatchSite(
        site_id="decoder_visual_token_residual.layer_000",
        family="decoder_visual_token_residual",
        module_name="unused",
        tensor_selector="decoder_slots",
        layer_index=0,
    )


def question_site() -> runner.PatchSite:
    return runner.PatchSite(
        site_id="decoder_final_question_token_residual.layer_000",
        family="decoder_final_question_token_residual",
        module_name="unused",
        tensor_selector="question_token",
        layer_index=0,
    )


def prefusion_site(family: str = "vision_tower_final") -> runner.PatchSite:
    assert family in runner.PREFUSION_TRANSFER_FAMILIES
    return runner.PatchSite(
        site_id=family,
        family=family,
        module_name="unused",
        tensor_selector="stage_slots",
    )


def passing_identity_audit(
    site: runner.PatchSite,
    choices: tuple[str, ...] = runner.CHOICES,
) -> dict:
    tensors = {
        choice: torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 4)
        for choice in choices
    }
    return runner.content_aligned_identity_audit(
        site=site,
        source_by_choice=tensors,
        recipient_by_choice={choice: value.clone() for choice, value in tensors.items()},
    )


def test_patch_changes_only_full_frame_slot_and_removes_hook() -> None:
    block = TupleBlock()
    value = torch.zeros(1, 8, 4)
    baseline = block(value)[0]
    source = torch.full((1, 2, 4), 7.0)
    with runner.patch_module_output(
        module=block,
        site=visual_site(),
        source_selected=source,
        recipient_slot_map=slot_map(),
        recipient_slot_indexes=(0,),
        question_position=7,
        mode="matched",
        seed=3,
    ) as audit:
        patched = block(value)[0]
    assert torch.equal(patched[:, 1:3], source)
    outside = torch.tensor([0, 3, 4, 5, 6, 7])
    assert torch.equal(patched.index_select(1, outside), baseline.index_select(1, outside))
    assert audit.max_abs_change_outside == 0.0
    assert audit.hook_removed
    assert len(block._forward_hooks) == 0


def test_self_patch_is_exact_identity() -> None:
    block = TupleBlock()
    value = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4)
    baseline = block(value)[0]
    source = baseline[:, 1:3].clone()
    with runner.patch_module_output(
        module=block,
        site=visual_site(),
        source_selected=source,
        recipient_slot_map=slot_map(),
        recipient_slot_indexes=(0,),
        question_position=7,
        mode="self",
        seed=4,
    ) as audit:
        patched = block(value)[0]
    assert torch.equal(patched, baseline)
    assert audit.source_delta_norm == 0.0
    assert audit.applied_delta_norm == 0.0
    assert audit.hook_removed


def test_norm_matched_random_is_exact_and_outside_unchanged() -> None:
    block = TupleBlock()
    value = torch.zeros(1, 8, 4)
    baseline = block(value)[0]
    source = torch.full((1, 2, 4), 5.0)
    with runner.patch_module_output(
        module=block,
        site=visual_site(),
        source_selected=source,
        recipient_slot_map=slot_map(),
        recipient_slot_indexes=(0,),
        question_position=7,
        mode="norm_matched_random",
        seed=99,
    ) as audit:
        patched = block(value)[0]
    assert audit.relative_norm_error is not None
    assert audit.relative_norm_error <= runner.NORM_TOLERANCE_REL
    assert audit.max_abs_change_outside == 0.0
    assert not torch.equal(patched[:, 1:3], source)
    outside = torch.tensor([0, 3, 4, 5, 6, 7])
    assert torch.equal(patched.index_select(1, outside), baseline.index_select(1, outside))


def test_norm_matched_random_casts_float32_donor_math_back_to_bf16() -> None:
    block = TupleBlock().to(dtype=torch.bfloat16)
    value = torch.zeros(1, 8, 4, dtype=torch.bfloat16)
    source = torch.full((1, 2, 4), 5.0, dtype=torch.float32)
    with runner.patch_module_output(
        module=block,
        site=visual_site(),
        source_selected=source,
        recipient_slot_map=slot_map(),
        recipient_slot_indexes=(0,),
        question_position=7,
        mode="norm_matched_random",
        seed=99,
    ) as audit:
        patched = block(value)[0]
    assert patched.dtype == torch.bfloat16
    assert audit.finite is True
    assert audit.max_abs_change_outside == 0.0
    assert audit.norm_tolerance_rel == runner.BF16_NORM_TOLERANCE_REL
    assert audit.relative_norm_error is not None
    assert audit.relative_norm_error <= audit.norm_tolerance_rel


def test_norm_matched_random_compensates_for_bf16_quantization() -> None:
    generator = torch.Generator().manual_seed(1)
    recipient = (torch.randn(1024, generator=generator) * 20.0).to(torch.bfloat16)
    source = (
        recipient.float() + torch.randn(1024, generator=generator) * 0.2
    ).to(torch.bfloat16)
    delta = source.float() - recipient.float()
    target_norm = float(delta.norm().item())

    old_generator = torch.Generator().manual_seed(1)
    old_noise = torch.randn(delta.shape, generator=old_generator, dtype=torch.float32)
    old_replacement = recipient + (
        old_noise * (delta.norm() / old_noise.norm())
    ).to(torch.bfloat16)
    old_error = abs(
        float((old_replacement.float() - recipient.float()).norm().item()) - target_norm
    ) / target_norm
    assert old_error > 0.005

    replacement = runner.deterministic_norm_matched_replacement(recipient, delta, 1)
    repeated = runner.deterministic_norm_matched_replacement(recipient, delta, 1)
    realized_error = abs(
        float((replacement.float() - recipient.float()).norm().item()) - target_norm
    ) / target_norm
    assert torch.equal(replacement, repeated)
    assert realized_error <= runner.BF16_NORM_TOLERANCE_REL
    assert runner.BF16_NORM_TOLERANCE_REL == pytest.approx(2.0**-7)


def test_norm_matched_random_never_returns_an_out_of_tolerance_bf16_control() -> None:
    nonzero_cases = 0
    explicitly_unsupported = 0
    for case in range(120):
        generator = torch.Generator().manual_seed(10_000 + case)
        size = (32, 128, 512, 1024)[case % 4]
        recipient_scale = 10.0 ** (-3 + (case % 13) * 0.5)
        delta_scale = 10.0 ** (-6 + (case % 17) * 0.5)
        recipient = (
            torch.randn(size, generator=generator) * recipient_scale
        ).to(torch.bfloat16)
        source = (
            recipient.float()
            + torch.randn(size, generator=generator) * delta_scale
        ).to(torch.bfloat16)
        delta = source.float() - recipient.float()
        target_norm = float(delta.norm().item())
        if target_norm == 0.0:
            continue
        nonzero_cases += 1
        try:
            replacement = runner.deterministic_norm_matched_replacement(
                recipient, delta, case
            )
        except runner.DegenerateIntervention:
            explicitly_unsupported += 1
            continue
        realized_norm = float(
            (replacement.float() - recipient.float()).norm().item()
        )
        relative_error = abs(realized_norm - target_norm) / target_norm
        assert relative_error <= runner.BF16_NORM_TOLERANCE_REL
    assert nonzero_cases >= 90
    assert explicitly_unsupported > 0


def test_norm_matched_random_rejects_zero_donor_delta() -> None:
    torch = pytest.importorskip("torch")
    block = TupleBlock()
    value = torch.zeros(1, 8, 4)
    source = torch.ones(1, 2, 4)
    with pytest.raises(runner.UnsupportedSite, match="zero donor delta"):
        with runner.patch_module_output(
            module=block,
            site=visual_site(),
            source_selected=source,
            recipient_slot_map=slot_map(),
            recipient_slot_indexes=(0,),
            question_position=7,
            mode="norm_matched_random",
            seed=99,
        ):
            block(value)


def test_shape_mismatch_fails_and_hook_is_removed() -> None:
    block = TupleBlock()
    value = torch.zeros(1, 8, 4)
    with pytest.raises(runner.GateError, match="shape mismatch"):
        with runner.patch_module_output(
            module=block,
            site=visual_site(),
            source_selected=torch.zeros(1, 3, 4),
            recipient_slot_map=slot_map(),
            recipient_slot_indexes=(0,),
            question_position=7,
            mode="matched",
            seed=1,
        ):
            block(value)
    assert len(block._forward_hooks) == 0


def test_counterbalanced_mapping_never_touches_full_slot() -> None:
    block = TupleBlock()
    value = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4)
    baseline = block(value)[0]
    donor_crop_a_then_b = torch.cat(
        [torch.full((1, 2, 4), 41.0), torch.full((1, 2, 4), 53.0)], dim=1
    )
    with runner.patch_module_output(
        module=block,
        site=visual_site(),
        source_selected=donor_crop_a_then_b,
        recipient_slot_map=slot_map(),
        recipient_slot_indexes=(2, 1),
        question_position=7,
        mode="counterbalanced_order",
        seed=1,
    ) as audit:
        patched = block(value)[0]
    assert torch.equal(patched[:, 5:7], donor_crop_a_then_b[:, :2])
    assert torch.equal(patched[:, 3:5], donor_crop_a_then_b[:, 2:])
    assert torch.equal(patched[:, 1:3], baseline[:, 1:3])
    assert audit.max_abs_change_outside == 0.0


def test_applicable_decoder_crop_transfer_exact_noop_remains_fatal() -> None:
    block = TupleBlock()
    value = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4)
    baseline = block(value)[0]
    identity_source = torch.cat([baseline[:, 5:7], baseline[:, 3:5]], dim=1)
    with pytest.raises(runner.DegenerateIntervention, match="exact no-op"):
        with runner.patch_module_output(
            module=block,
            site=visual_site(),
            source_selected=identity_source,
            recipient_slot_map=slot_map(),
            recipient_slot_indexes=(2, 1),
            question_position=7,
            mode="counterbalanced_order",
            seed=1,
        ):
            block(value)
    assert len(block._forward_hooks) == 0


def test_qwen_slot_map_uses_three_token_runs_and_grid_counts() -> None:
    class Config:
        spatial_merge_size = 2

    class Model:
        config = Config()

    inputs = {
        "input_ids": torch.tensor([[1, 99, 99, 99, 99, 2, 99, 99, 99, 99, 3, 99, 99, 99, 99, 4]]),
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4], [1, 4, 4]]),
        "pixel_values": torch.zeros(48, 3),
    }
    adapter = {"image_token_id": 99, "family": "qwen35"}
    observed = runner.build_image_slot_map(
        inputs=inputs, model=Model(), processor=object(), adapter=adapter
    )
    assert [slot.decoder_count for slot in observed.slots] == [4, 4, 4]
    assert [slot.premerge_count for slot in observed.slots] == [16, 16, 16]
    assert observed.positions((1, 2)) == tuple(range(6, 10)) + tuple(range(11, 15))


def test_slot_map_fails_when_boundaries_are_not_three_runs() -> None:
    class Model:
        config = object()

    inputs = {
        "input_ids": torch.tensor([[1, 99, 99, 2, 99, 99, 3]]),
        "pixel_values": torch.zeros(3, 2, 4),
    }
    with pytest.raises(runner.GateError, match="exactly three"):
        runner.build_image_slot_map(
            inputs=inputs,
            model=Model(),
            processor=object(),
            adapter={"image_token_id": 99, "family": "gemma4"},
        )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(path: Path, color: tuple[int, int, int], size: tuple[int, int]) -> dict[str, object]:
    Image.new("RGB", size, color).save(path)
    return {"path": str(path), "sha256": _sha(path), "width": size[0], "height": size[1]}


def test_dataset_contract_is_counterbalanced_and_public_label_free(tmp_path: Path) -> None:
    full = _asset(tmp_path / "full.png", (1, 2, 3), (12, 8))
    blank = _asset(tmp_path / "blank.png", (0, 0, 0), (12, 8))
    wrong = _asset(tmp_path / "wrong.png", (9, 8, 7), (12, 8))
    crop_1 = _asset(tmp_path / "crop1.png", (10, 20, 30), (6, 6))
    crop_2 = _asset(tmp_path / "crop2.png", (30, 20, 10), (6, 6))

    def row(request_id: str, order: str, partner: str, crops: list[dict[str, object]]) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_id": request_id,
            "frame_unit_id": "frame-1",
            "order_id": order,
            "counterbalanced_request_id": partner,
            "system_prompt": "",
            "user_prompt": "Which crop is first-order?",
            "arms": {
                "real": {"inputs": [full, *crops]},
                "blank_full": {"inputs": [blank, *crops]},
                "wrong_time": {"inputs": [wrong, *crops]},
            },
        }

    public = [
        row("r0", "order_0", "r1", [crop_1, crop_2]),
        row("r1", "order_1", "r0", [crop_2, crop_1]),
    ]
    hidden = [
        {
            "schema_version": 1,
            "request_id": "r0",
            "frame_unit_id": "frame-1",
            "order_id": "order_0",
            "counterbalanced_request_id": "r1",
            "correct_choice": "A",
            "ego4d_participant_id": "p1",
            "target_component_rule": runner.TARGET_COMPONENT_RULE,
            "target_component_fingerprints": {"A": "a" * 64, "B": "b" * 64},
        },
        {
            "schema_version": 1,
            "request_id": "r1",
            "frame_unit_id": "frame-1",
            "order_id": "order_1",
            "counterbalanced_request_id": "r0",
            "correct_choice": "B",
            "ego4d_participant_id": "p1",
            "target_component_rule": runner.TARGET_COMPONENT_RULE,
            "target_component_fingerprints": {"A": "b" * 64, "B": "a" * 64},
        },
    ]
    receipt = runner.validate_dataset(
        public_rows=public,
        hidden_rows=hidden,
        data_dir=tmp_path,
        expected_frames=1,
    )
    assert receipt["trial_count"] == 2
    assert receipt["frame_unit_count"] == 1
    leaked = dict(public[0])
    leaked["class_id"] = 3
    with pytest.raises(runner.GateError, match="leaks evaluator"):
        runner.validate_dataset(
            public_rows=[leaked, public[1]],
            hidden_rows=hidden,
            data_dir=tmp_path,
            expected_frames=1,
        )


def test_resume_drops_incomplete_and_duplicate_rows(tmp_path: Path) -> None:
    results = tmp_path / "patch_outcomes.jsonl"
    complete = tmp_path / "completed_requests.jsonl"
    binding_sha256 = "a" * 64
    base = {key: None for key in runner.OUTCOME_ROW_FIELDS}
    base.update({
        "schema_version": runner.OUTPUT_SCHEMA_VERSION,
        "phase": "coarse",
        "model_key": "qwen35_27b_hf_bf16",
        "rank": 0,
        "corruption": "blank_full",
        "site_id": "s",
        "intervention": "matched_clean_source_rescue",
        "source_request_id": "source",
    })
    rows = [
        {**base, "request_id": "r1"},
        {**base, "request_id": "r1"},
        {**base, "request_id": "r2"},
    ]
    results.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    complete.write_text(
        json.dumps(
            {
                "request_id": "r1",
                "result_row_count": 1,
                "result_batch_sha256": runner.sha256_json([rows[0]]),
                "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
                "resume_binding_sha256": binding_sha256,
                "model_key": "qwen35_27b_hf_bf16",
                "rank": 0,
                "phase": "coarse",
                "completed_at": "2026-07-22T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = runner.sanitize_resume_files(
        results,
        complete,
        expected_resume_binding_sha256=binding_sha256,
        expected_model_key="qwen35_27b_hf_bf16",
        expected_rank=0,
        expected_phase="coarse",
        expected_request_ids={"r1", "r2"},
    )
    assert receipt["completed_request_ids"] == ["r1"]
    assert receipt["kept_rows"] == 1
    assert receipt["dropped_duplicate_rows"] == 1
    assert receipt["dropped_incomplete_or_invalid_rows"] == 1


def test_resumed_rows_preserve_one_row_activation_peak_evidence() -> None:
    assert runner.combined_source_activation_peak([], resumed_kept_rows=400) == 1
    assert (
        runner.combined_source_activation_peak(
            [{"peak_simultaneous_source_activation_rows": 1}],
            resumed_kept_rows=0,
        )
        == 1
    )
    assert runner.combined_source_activation_peak([], resumed_kept_rows=0) == 0
    with pytest.raises(runner.GateError, match="cannot be negative"):
        runner.combined_source_activation_peak([], resumed_kept_rows=-1)


def test_resume_rejects_stale_completion_binding_before_reuse(tmp_path: Path) -> None:
    results = tmp_path / "patch_outcomes.jsonl"
    complete = tmp_path / "completed_requests.jsonl"
    row = {key: None for key in runner.OUTCOME_ROW_FIELDS}
    row.update(
        {
            "schema_version": runner.OUTPUT_SCHEMA_VERSION,
            "phase": "coarse",
            "model_key": "qwen35_27b_hf_bf16",
            "rank": 0,
            "request_id": "r1",
            "corruption": "blank_full",
            "site_id": "s",
            "intervention": "matched_clean_source_rescue",
            "source_request_id": "source",
        }
    )
    results.write_text(json.dumps(row) + "\n", encoding="utf-8")
    complete.write_text(
        json.dumps(
            {
                "request_id": "r1",
                "result_row_count": 1,
                "result_batch_sha256": runner.sha256_json([row]),
                "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
                "resume_binding_sha256": "b" * 64,
                "model_key": "qwen35_27b_hf_bf16",
                "rank": 0,
                "phase": "coarse",
                "completed_at": "2026-07-22T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(runner.GateError, match="release/schema binding"):
        runner.sanitize_resume_files(
            results,
            complete,
            expected_resume_binding_sha256="a" * 64,
            expected_model_key="qwen35_27b_hf_bf16",
            expected_rank=0,
            expected_phase="coarse",
            expected_request_ids={"r1"},
        )


def test_worker_outcome_is_label_free_and_unrelated_selection_keeps_both_orders() -> None:
    rows = []
    for frame in range(3):
        ids = [f"f{frame}-o0", f"f{frame}-o1"]
        for order in range(2):
            rows.append(
                {
                    "request_id": ids[order],
                    "frame_unit_id": f"f{frame}",
                    "order_id": f"order_{order}",
                    "counterbalanced_request_id": ids[1 - order],
                }
            )
    donors = runner.select_unrelated_donor_pairs(rows)
    for request_id, pair in donors.items():
        assert {row["order_id"] for row in rows if row["request_id"] in pair} == {
            "order_0",
            "order_1",
        }
        recipient_frame = next(
            row["frame_unit_id"] for row in rows if row["request_id"] == request_id
        )
        donor_frames = {
            row["frame_unit_id"] for row in rows if row["request_id"] in pair
        }
        assert donor_frames != {recipient_frame}

    request = {
        "request_id": "r0",
        "frame_unit_id": "f0",
        "order_id": "order_0",
        "counterbalanced_request_id": "r1",
        "system_prompt": "",
        "user_prompt": "prompt",
        "arms": {},
    }
    outcome = runner._outcome_row(
        phase="coarse",
        model_key="qwen35_27b_hf_bf16",
        rank=0,
        request=request,
        corruption="blank_full",
        site=visual_site(),
        patch_direction="clean_to_corrupt_rescue",
        intervention="matched_clean_source_rescue",
        source_request_id="r0",
        source_arm="real",
        recipient_arm="blank_full",
        recipient_a_minus_b_margin=-1.0,
        real_a_minus_b_margin=2.0,
        patched_a_minus_b_margin=1.5,
        audits=[],
        status="pass",
    )
    assert outcome["label_joined"] is False
    assert "correct_choice" not in outcome
    assert "ego4d_participant_id" not in outcome
    assert "absolute_recovery_bits" not in outcome


def test_analyzer_accepts_only_audited_sample_degenerate_unsupported_row() -> None:
    request = {
        "request_id": "r0",
        "frame_unit_id": "f0",
        "order_id": "order_0",
        "counterbalanced_request_id": "r1",
        "system_prompt": "",
        "user_prompt": "prompt",
        "arms": {},
    }
    audit = runner.HookAudit(
        site_id=visual_site().site_id,
        mode="norm_matched_random",
        status="fail",
        hook_count_before=0,
        hook_count_after=0,
        hook_removed=True,
        error="DegenerateIntervention: exactly zero donor delta",
    )
    outcome = runner._outcome_row(
        phase="coarse",
        model_key="qwen35_27b_hf_bf16",
        rank=0,
        request=request,
        corruption="blank_full",
        site=visual_site(),
        patch_direction="clean_to_corrupt_rescue",
        intervention="rescue_norm_matched_random",
        source_request_id="r0",
        source_arm="real",
        recipient_arm="blank_full",
        recipient_a_minus_b_margin=-1.0,
        real_a_minus_b_margin=2.0,
        patched_a_minus_b_margin=None,
        audits=[{"candidate_choice": "A", **runner.asdict(audit)}],
        status="unsupported",
        reason=(
            "degenerate_intervention:DegenerateIntervention:"
            "norm-matched random control is undefined for an exactly zero donor delta"
        ),
    )
    support_gate = analysis.representable_intervention_gate([outcome])
    assert support_gate["status"] == "fail"
    assert support_gate["unsupported_count"] == 1
    assert analysis._allowed_unsupported_row(outcome)
    outcome["reason"] = "unexpected"
    assert not analysis._allowed_unsupported_row(outcome)


@pytest.mark.parametrize(
    ("site", "reason"),
    [
        (
            question_site(),
            runner.QUESTION_TOKEN_TRANSFER_NOT_APPLICABLE_REASON,
        ),
        (
            prefusion_site("vision_tower_final"),
            runner.PREFUSION_TRANSFER_NOT_APPLICABLE_REASON,
        ),
        (
            prefusion_site("projector_or_merger_output"),
            runner.PREFUSION_TRANSFER_NOT_APPLICABLE_REASON,
        ),
    ],
)
def test_structural_crop_transfer_is_not_applicable_not_unsupported(
    site: runner.PatchSite,
    reason: str,
) -> None:
    request = {
        "request_id": "r0",
        "frame_unit_id": "f0",
        "order_id": "order_0",
        "counterbalanced_request_id": "r1",
        "system_prompt": "",
        "user_prompt": "prompt",
        "arms": {},
    }
    outcome = runner._outcome_row(
        phase="coarse",
        model_key="qwen35_27b_hf_bf16",
        rank=0,
        request=request,
        corruption="counterbalanced_order",
        site=site,
        patch_direction="counterbalanced_crop_transfer",
        intervention="counterbalanced_content_aligned_crop_transfer",
        source_request_id="r1",
        source_arm="real",
        recipient_arm="real",
        recipient_a_minus_b_margin=1.0,
        real_a_minus_b_margin=1.0,
        patched_a_minus_b_margin=None,
        audits=[],
        status="not_applicable",
        reason=reason,
        slot_mapping=runner.structural_transfer_slot_mapping(
            site,
            identity_audit=(
                passing_identity_audit(site)
                if site.family in runner.PREFUSION_TRANSFER_FAMILIES
                else None
            ),
        ),
    )
    assert runner.is_structural_not_applicable_row(outcome)
    assert analysis._allowed_not_applicable_row(outcome)
    assert not analysis._allowed_unsupported_row(outcome)
    support_gate = analysis.representable_intervention_gate([outcome])
    assert support_gate["status"] == "pass"
    assert support_gate["unsupported_count"] == 0
    assert support_gate["not_applicable_count"] == 1
    assert support_gate["invalid_not_applicable_count"] == 0
    runtime_gate = runner.result_representability_gate([outcome] * 864)
    assert runtime_gate["status"] == "pass"
    assert runtime_gate["unsupported_count"] == 0
    assert runtime_gate["not_applicable_count"] == 864

    mislabeled = dict(outcome)
    mislabeled["status"] = "pass"
    assert runner.transfer_status_contract_violation(mislabeled) is not None

    outcome["reason"] = "unexpected"
    assert not runner.is_structural_not_applicable_row(outcome)
    assert runner.result_representability_gate([outcome])["status"] == "fail"
    support_gate = analysis.representable_intervention_gate([outcome])
    assert support_gate["status"] == "fail"
    assert support_gate["invalid_not_applicable_count"] == 1


def test_crop_transfer_applicability_is_exact_and_fails_closed_on_drift() -> None:
    assert runner.transfer_site_applicability(visual_site()) == (True, None)
    assert runner.transfer_site_applicability(question_site()) == (
        False,
        runner.QUESTION_TOKEN_TRANSFER_NOT_APPLICABLE_REASON,
    )
    for family in runner.PREFUSION_TRANSFER_FAMILIES:
        assert runner.transfer_site_applicability(prefusion_site(family)) == (
            False,
            runner.PREFUSION_TRANSFER_NOT_APPLICABLE_REASON,
        )
    drifted = runner.PatchSite(
        site_id="vision_tower_final",
        family="vision_tower_final",
        module_name="unused",
        tensor_selector="decoder_slots",
    )
    with pytest.raises(runner.GateError, match="applicability is undefined"):
        runner.transfer_site_applicability(drifted)


def test_prefusion_identity_audit_fails_on_any_activation_difference() -> None:
    site = prefusion_site()
    source = {choice: torch.zeros(1, 2, 4) for choice in runner.CHOICES}
    recipient = {choice: value.clone() for choice, value in source.items()}
    recipient["B"][0, 0, 0] = 1.0
    with pytest.raises(runner.GateError, match="not exact identity"):
        runner.content_aligned_identity_audit(
            site=site,
            source_by_choice=source,
            recipient_by_choice=recipient,
        )


def test_outcome_schema_is_closed_against_arbitrary_extra_fields() -> None:
    request = {
        "request_id": "r0",
        "frame_unit_id": "f0",
        "order_id": "order_0",
        "counterbalanced_request_id": "r1",
        "system_prompt": "",
        "user_prompt": "prompt",
        "arms": {},
    }
    outcome = runner._outcome_row(
        phase="coarse",
        model_key="qwen35_27b_hf_bf16",
        rank=0,
        request=request,
        corruption="real",
        site=visual_site(),
        patch_direction="identity_control",
        intervention="clean_to_clean_self_patch",
        source_request_id="r0",
        source_arm="real",
        recipient_arm="real",
        recipient_a_minus_b_margin=1.0,
        real_a_minus_b_margin=1.0,
        patched_a_minus_b_margin=1.0,
        audits=[],
        status="pass",
    )
    assert analysis.raw_outcome_schema_violations(outcome) == {
        "missing": [],
        "unexpected": [],
    }
    outcome["secret_label"] = "leak"
    assert analysis.raw_outcome_schema_violations(outcome) == {
        "missing": [],
        "unexpected": ["secret_label"],
    }


def test_confirmatory_endpoint_fails_closed_on_one_unsupported_row() -> None:
    interventions = (
        "matched_clean_source_rescue",
        "rescue_unrelated_same_label_donor",
        "rescue_norm_matched_random",
        "matched_corrupt_source_damage",
        "damage_unrelated_same_label_donor",
        "damage_norm_matched_random",
    )
    rows = []
    for intervention in interventions:
        for index in range(runner.MECHANISM_TRIAL_COUNT):
            rows.append(
                {
                    "model_key": "qwen35_27b_hf_bf16",
                    "corruption": "blank_full",
                    "stage_family": "decoder_block",
                    "site_id": "decoder_block.layer_000",
                    "layer_index": 0,
                    "intervention": intervention,
                    "status": (
                        "unsupported"
                        if intervention == "rescue_norm_matched_random" and index == 0
                        else "pass"
                    ),
                }
            )
    with pytest.raises(analysis.AnalysisError, match="exactly 192 trials"):
        analysis._component_cluster_maps(
            rows,
            model="qwen35_27b_hf_bf16",
            corruption="blank_full",
            family="decoder_block",
            layer=0,
        )


def test_analyzer_keeps_only_same_label_unrelated_donor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public_rows = [
        {
            "request_id": "recipient",
            "frame_unit_id": "f0",
            "order_id": "order_0",
            "counterbalanced_request_id": "recipient-pair",
            "system_prompt": "",
            "user_prompt": "prompt",
            "arms": {},
        },
        {
            "request_id": "donor-a",
            "frame_unit_id": "f1",
            "order_id": "order_0",
            "counterbalanced_request_id": "donor-b",
            "system_prompt": "",
            "user_prompt": "prompt",
            "arms": {},
        },
        {
            "request_id": "donor-b",
            "frame_unit_id": "f1",
            "order_id": "order_1",
            "counterbalanced_request_id": "donor-a",
            "system_prompt": "",
            "user_prompt": "prompt",
            "arms": {},
        },
    ]
    hidden_rows = [
        {"request_id": "recipient", "correct_choice": "B", "ego4d_participant_id": "p0"},
        {"request_id": "donor-a", "correct_choice": "A", "ego4d_participant_id": "p1"},
        {"request_id": "donor-b", "correct_choice": "B", "ego4d_participant_id": "p1"},
    ]
    hidden_path = tmp_path / "hidden.jsonl"
    hidden_path.write_text(
        "".join(json.dumps(row) + "\n" for row in hidden_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "validate_mechanism_manifest",
        lambda **_kwargs: {"hidden_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        runner,
        "validate_dataset",
        lambda **_kwargs: {"status": "pass"},
    )
    raw_rows = [
        {
            "phase": "coarse",
            "model_key": "qwen35_27b_hf_bf16",
            "request_id": "recipient",
            "frame_unit_id": "f0",
            "intervention": "matched_clean_source_rescue",
            "patch_direction": "clean_to_corrupt_rescue",
            "source_request_id": "recipient",
            "public_request_sha256": runner.sha256_json(
                runner._public_identity(public_rows[0])
            ),
            "corruption": "blank_full",
            "recipient_baseline_a_minus_b_margin_bits": 1.0,
            "real_baseline_a_minus_b_margin_bits": -2.0,
            "patched_a_minus_b_margin_bits": -1.0,
        }
    ]
    for donor_id in ("donor-a", "donor-b"):
        raw_rows.append(
            {
                "request_id": "recipient",
                "frame_unit_id": "f0",
                    "phase": "coarse",
                    "model_key": "qwen35_27b_hf_bf16",
                    "corruption": "blank_full",
                    "intervention": "rescue_unrelated_donor_candidate",
                    "patch_direction": "clean_to_corrupt_rescue",
                "source_request_id": donor_id,
                "public_request_sha256": runner.sha256_json(
                    runner._public_identity(public_rows[0])
                ),
                "recipient_baseline_a_minus_b_margin_bits": 1.0,
                "real_baseline_a_minus_b_margin_bits": -2.0,
                "patched_a_minus_b_margin_bits": -1.0,
            }
        )
    joined, receipt = analysis.join_hidden_after_technical_gate(
        rows=raw_rows,
        public_rows=public_rows,
        data_dir=tmp_path,
        mechanism_manifest=tmp_path / "manifest.json",
        public_manifest=tmp_path / "public.jsonl",
        hidden_index=hidden_path,
        expected_user_prompt="prompt",
    )
    unrelated_joined = [
        row for row in joined if row["intervention"] == "rescue_unrelated_same_label_donor"
    ]
    assert len(unrelated_joined) == 1
    assert unrelated_joined[0]["source_request_id"] == "donor-b"
    assert unrelated_joined[0]["intervention_effect_bits"] == pytest.approx(2.0)
    assert receipt["same_label_donor_rows_kept"] == 1
    assert receipt["opposite_label_donor_rows_dropped"] == 1


def test_coarse_layer_fraction_mapping_is_frozen() -> None:
    assert runner.coarse_layer_indexes(64) == [0, 8, 16, 24, 32, 39, 47, 55, 63]
    assert runner.coarse_layer_indexes(60) == [0, 7, 15, 22, 30, 37, 44, 52, 59]


def test_refinement_plan_keeps_decoder_families_separate(tmp_path: Path) -> None:
    plan = {
        "governing_prespec_sha256": runner.PRESPEC_SHA256,
        "models": {
            "qwen35_27b_hf_bf16": {
                "decoder_visual_token_residual": {"refinement_layers": [7, 9]},
                "decoder_final_question_token_residual": {"refinement_layers": [22]},
            }
        },
    }
    path = tmp_path / "refinement.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert runner.refinement_layers_for_model(path, "qwen35_27b_hf_bf16") == {
        "decoder_visual_token_residual": [7, 9],
        "decoder_final_question_token_residual": [22],
    }


def test_mechanism_manifest_contract_and_hash_are_exact(tmp_path: Path) -> None:
    public = tmp_path / runner.MECHANISM_PUBLIC_REL
    public.parent.mkdir(parents=True)
    public.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / runner.MECHANISM_MANIFEST_REL
    manifest = {
        "kind": runner.MECHANISM_MANIFEST_KIND,
        "schema_version": 1,
        "status": "complete_production",
        "production_eligible": True,
        "governing_prespec_sha256": runner.PRESPEC_SHA256,
        "selection_namespace": runner.MECHANISM_SELECTION_NAMESPACE,
        "selection_rule": runner.MECHANISM_SELECTION_RULE,
        "selection_uses_model_outputs": False,
        "selection_uses_hidden_answers_or_roles": False,
        "source_battery_frame_count": 800,
        "frame_count": 96,
        "trial_count": 192,
        "participant_cluster_count": 96,
        "counterbalanced_trials_per_frame": 2,
        "arms": list(runner.ARMS),
        "images_per_arm": 3,
        "pair_audit": {
            "status": "pass",
            "full_frame_identity_check_count": 288,
            "crop_ab_exact_permutation_check_count": 288,
            "cross_arm_crop_identity_check_count": 384,
        },
        "fixed_geometry": {
            "full_frame": {"width": 896, "height": 672},
            "reference_crop": {"width": 336, "height": 336},
        },
        "public_schema": {"evaluator_fields_present": False},
        "hidden_schema": {
            "join_key": "request_id",
            "correct_choice": ["A", "B"],
            "participant_field": "ego4d_participant_id",
        },
        "consumer_contract": {
            "job_c": {
                "public_manifest_argument": str(runner.MECHANISM_PUBLIC_REL),
                "hidden_index_argument": str(runner.MECHANISM_HIDDEN_REL),
                "data_dir_argument": ".",
            }
        },
        "files": {
            "public_requests": {
                "path": str(runner.MECHANISM_PUBLIC_REL),
                "sha256": runner.sha256_file(public),
                "rows": 192,
            },
        },
    }
    hidden_inside = tmp_path / runner.MECHANISM_HIDDEN_REL
    hidden_inside.parent.mkdir(parents=True)
    hidden_inside.write_text("{}\n", encoding="utf-8")
    manifest["files"]["hidden_evaluator_index"] = {
        "path": str(runner.MECHANISM_HIDDEN_REL),
        "sha256": runner.sha256_file(hidden_inside),
        "rows": 192,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = runner.validate_mechanism_manifest(
        data_dir=tmp_path,
        manifest_path=manifest_path,
        public_path=public,
    )
    assert receipt["status"] == "pass"
    assert receipt["subset_selection_already_frozen"] is True
    hidden_external = tmp_path.with_name(f"{tmp_path.name}_coordinator_hidden.jsonl")
    hidden_external.write_bytes(hidden_inside.read_bytes())
    external_receipt = runner.validate_mechanism_manifest(
        data_dir=tmp_path,
        manifest_path=manifest_path,
        public_path=public,
        hidden_path=hidden_external,
        require_hidden_outside_data_dir=True,
    )
    assert external_receipt["hidden_path"] == str(hidden_external.resolve())
    with pytest.raises(runner.GateError, match="outside the worker-visible"):
        runner.validate_mechanism_manifest(
            data_dir=tmp_path,
            manifest_path=manifest_path,
            public_path=public,
            hidden_path=hidden_inside,
            require_hidden_outside_data_dir=True,
        )
    public.write_text('{"drift":true}\n', encoding="utf-8")
    with pytest.raises(runner.GateError, match="SHA-256"):
        runner.validate_mechanism_manifest(
            data_dir=tmp_path,
            manifest_path=manifest_path,
            public_path=public,
        )


def test_launcher_uses_fresh_run_root_local_code_and_periodic_copyback() -> None:
    launcher = (
        REPO_ROOT
        / "docs/research/lifecycle_memory/phase 2 research/jobs/"
        / "reanchored_jobC_causal_patch/job_tamia_reanchored_jobC_causal_patch_h100.sh"
    ).read_text(encoding="utf-8")
    assert 'RUN_ROOT="$SCRATCH/tamia_runs/reanchored-jobC-causal-patch/$SLURM_JOB_ID"' in launcher
    assert 'RUN_ROOT="${RESUME_DIR:-' not in launcher
    assert 'LOCAL_REPO="$SLURM_TMPDIR/repo"' in launcher
    assert 'CODE_SNAPSHOT_SHA256=' in launcher
    assert '--env PYTHONPATH="$LOCAL_REPO"' in launcher
    assert '--env PYTHONPATH="$REPO_DIR"' not in launcher
    assert '--repo-dir "$LOCAL_REPO"' in launcher
    assert '"resume_source": "${RESUME_DIR:-}"' in launcher
    assert '"$RESUME_DIR"/ "$LOCAL_RUN"/' in launcher
    assert 'FLUSH_INTERVAL_S="${FLUSH_INTERVAL_S:-180}"' in launcher
    assert "stage only mechanism public rows and referenced images" in launcher
    assert "module load python/3.12" in launcher
    assert 'rsync -a --copy-links --partial' not in launcher
    assert '-B /scratch' not in launcher
    assert "--containall --cleanenv" in launcher
    assert 'copy_back 124' in launcher
    assert 'wait "$SRUN_PID"' in launcher
    assert "CANARY_PROFILE_PATH" in launcher
    assert "JOB_A_ANALYSIS_SUMMARY" in launcher
    assert "CACHE_REVISION_OK" in launcher
    assert "--coarse-only" in launcher
    assert "validate-public-gate" in launcher
    assert '--gpus-per-task=h100:1 --gpu-bind=single:1' in launcher
    assert 'export CUDA_VISIBLE_DEVICES="${SLURM_LOCALID}"' not in launcher
    assert 'LOCAL_EVALUATOR="$SLURM_TMPDIR/jobC_evaluator"' in launcher
    assert 'LOCAL_HIDDEN="$LOCAL_EVALUATOR/mechanism_evaluator_index.jsonl"' in launcher
    assert '-B "$LOCAL_EVALUATOR:$LOCAL_EVALUATOR:ro"' in launcher
    worker_block = launcher.split("launch_workers()", 1)[1].split(
        "launch_workers coarse", 1
    )[0]
    worker_export = worker_block.split("bash -c", 1)[0]
    assert "--export=ALL" not in worker_export
    assert "DATA_DIR" not in worker_export
    assert "SHARED_HIDDEN" not in worker_export
    assert 'LOCAL_HIDDEN' not in worker_export
    assert 'LOCAL_EVALUATOR' not in worker_export
    assert '-B "$LOCAL_EVALUATOR' not in worker_block
    assert "Forbidden source/evaluator variable reached Job C worker" in worker_block
    assert 'worker_log="$LOCAL_RUN/worker_rank_${SLURM_PROCID}_${phase}.log"' in worker_block
    assert 'pipeline_status=("${PIPESTATUS[@]}")' in worker_block
    assert 'tee_status="${pipeline_status[1]}"' in worker_block
    assert 'worker_status="$tee_status"' in worker_block
    assert 'python -u "${args[@]}" 2>&1 | tee "$worker_log"' in worker_block
    assert launcher.index("validate-public-gate") < launcher.index(
        'rsync -a --copy-links "$SHARED_HIDDEN"'
    )


def test_full_submit_rejects_unbound_canary_and_job_a_results() -> None:
    submit = (
        REPO_ROOT
        / "docs/research/lifecycle_memory/phase 2 research/jobs/"
        / "reanchored_jobC_causal_patch/submit_tamia.sh"
    ).read_text(encoding="utf-8")
    assert 'p["canary_aggregate_sha256"] == sha(aggregate_path)' in submit
    assert 'p["mechanism_manifest_sha256"] == sha(mechanism_path)' in submit
    assert 'p["model_plan_sha256"] == sha(model_plan_path)' in submit
    assert 'job_a_manifest["analysis_summary_sha256"] == sha(job_a_summary_path)' in submit
    assert 'job_a_summary["status"] == "pass" and job_a_summary["frame_count"] == 800' in submit
    assert 'assert any(status == "pass" for status in promotion)' in submit
    assert 'sbatch "${hold_args[@]}" "${comment_args[@]}" "${name_args[@]}" --time="$WALLTIME"' in submit
    assert "--export=ALL" not in submit

    resume_gate = submit.index("RESUME_PRE_SBATCH_BINDING_PASS")
    sbatch_call = submit.index('SUBMIT_OUTPUT=$(sbatch')
    assert resume_gate < sbatch_call
    assert "resume completion receipt differs from current release/schema" in submit
    assert 'for pattern in ("completed_requests.jsonl", "patch_outcomes.jsonl")' in submit

    runner_source = (
        REPO_ROOT
        / "research/lifecycle_memory/run_phase2_reanchored_causal_patch.py"
    ).read_text(encoding="utf-8")
    worker_start = runner_source.index("def worker_command(")
    resume_failure = runner_source.index(
        "resume worker manifest differs from current immutable inputs before ",
        worker_start,
    )
    model_load = runner_source.index("_load_anchor_model(", worker_start)
    assert resume_failure < model_load
    sanitize_call = runner_source.index("resume = sanitize_resume_files(", worker_start)
    assert sanitize_call < model_load
    assert '"resume_binding_sha256": resume_binding_sha256' in runner_source


def test_submitter_executes_resume_binding_validator(tmp_path: Path) -> None:
    submit_path = (
        REPO_ROOT
        / "docs/research/lifecycle_memory/phase 2 research/jobs/"
        / "reanchored_jobC_causal_patch/submit_tamia.sh"
    )
    submit = submit_path.read_text(encoding="utf-8")
    marker = (
        '"$MODEL_PLAN_SHA256" "$EXPECTED_SIF_SHA256" '
        '"$RELEASE_MANIFEST_SHA256" <<\'PY\'\n'
    )
    embedded = submit.split(marker, 1)[1].split("\nPY\nfi", 1)[0]

    hashes = {
        "mechanism_manifest_sha256": "1" * 64,
        "public_manifest_sha256": "2" * 64,
        "model_plan_sha256": "3" * 64,
        "sif_sha256": "4" * 64,
        "release_manifest_sha256": "5" * 64,
    }
    immutable = {
        "governing_prespec_sha256": runner.PRESPEC_SHA256,
        "code_snapshot_sha256": "6" * 64,
        **hashes,
        "question_token_contract": runner.QUESTION_TOKEN_CONTRACT,
        "refinement_plan_sha256": None,
        "phase": "coarse",
    }
    binding = {
        "kind": "phase2_reanchored_causal_patch_resume_binding",
        "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
        "model_key": "qwen35_27b_hf_bf16",
        "rank": 0,
        "world_size": 4,
        "shard_index": 0,
        "shard_count": 2,
        "immutable_hashes": immutable,
    }
    binding_sha256 = runner.sha256_json(binding)
    phase_dir = tmp_path / "rank_000" / "coarse"
    phase_dir.mkdir(parents=True)
    (phase_dir / "worker_manifest.json").write_text(
        json.dumps(
            {
                "kind": "phase2_reanchored_causal_patch_worker_manifest",
                "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
                "rank": 0,
                "world_size": 4,
                "model_key": "qwen35_27b_hf_bf16",
                "immutable_hashes": {
                    **immutable,
                    "architecture_receipt_sha256": "7" * 64,
                },
                "resume_binding": binding,
                "resume_binding_sha256": binding_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = {
        "request_id": "r1",
        "result_row_count": 1,
        "result_batch_sha256": "8" * 64,
        "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
        "resume_binding_sha256": binding_sha256,
        "model_key": "qwen35_27b_hf_bf16",
        "rank": 0,
        "phase": "coarse",
        "completed_at": "2026-07-22T00:00:00+00:00",
    }
    complete_path = phase_dir / "completed_requests.jsonl"
    complete_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    argv = [
        sys.executable,
        "-c",
        embedded,
        str(tmp_path),
        hashes["mechanism_manifest_sha256"],
        hashes["public_manifest_sha256"],
        hashes["model_plan_sha256"],
        hashes["sif_sha256"],
        hashes["release_manifest_sha256"],
    ]
    valid = subprocess.run(argv, check=False, capture_output=True, text=True)
    assert valid.returncode == 0, valid.stderr
    assert "RESUME_PRE_SBATCH_BINDING_PASS manifests=1" in valid.stdout

    receipt["resume_binding_sha256"] = "9" * 64
    complete_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    stale = subprocess.run(argv, check=False, capture_output=True, text=True)
    assert stale.returncode != 0
    assert "resume completion receipt differs" in stale.stderr


def test_analyzer_site_contracts_are_exact_and_family_specific(tmp_path: Path) -> None:
    coarse_contract = analysis.coarse_site_contract()
    for model in analysis.MODELS:
        assert len(coarse_contract[model]) == 20
        assert "vision_tower_final" in coarse_contract[model]
        assert "projector_or_merger_output" in coarse_contract[model]

    models: dict[str, object] = {}
    for model in analysis.MODELS:
        layer_count = int(runner.MODEL_ADAPTERS[model]["expected_decoder_layers"])
        coarse = runner.coarse_layer_indexes(layer_count)
        peak = coarse[4]
        refinement = sorted(
            set(range(peak - 2, peak + 3)).difference(coarse)
        )
        models[model] = {
            family: {
                "coarse_layers": coarse,
                "selected_peak_layer": peak,
                "refinement_layers": refinement,
                "inferential_scope": "descriptive_only",
            }
            for family in (
                "decoder_visual_token_residual",
                "decoder_final_question_token_residual",
            )
        }
    path = tmp_path / "refinement_plan.json"
    path.write_text(
        json.dumps(
            {
                "kind": "phase2_reanchored_causal_patch_refinement_plan",
                "status": "pass",
                "governing_prespec_sha256": runner.PRESPEC_SHA256,
                "models": models,
            }
        ),
        encoding="utf-8",
    )
    refinement_contract, receipt = analysis.refinement_site_contract(path)
    assert receipt["status"] == "pass"
    for model in analysis.MODELS:
        families = {
            metadata["stage_family"]
            for metadata in refinement_contract[model].values()
        }
        assert families == {
            "decoder_visual_token_residual",
            "decoder_final_question_token_residual",
        }


def test_raw_public_alignment_is_label_free_and_hash_bound() -> None:
    public_rows: list[dict[str, object]] = []
    for frame_index in range(runner.MECHANISM_FRAME_COUNT):
        request_ids = [f"f{frame_index:03d}-o0", f"f{frame_index:03d}-o1"]
        for order_index in range(2):
            public_rows.append(
                {
                    "request_id": request_ids[order_index],
                    "frame_unit_id": f"f{frame_index:03d}",
                    "order_id": f"order_{order_index}",
                    "counterbalanced_request_id": request_ids[1 - order_index],
                    "system_prompt": "",
                    "user_prompt": "frozen prompt",
                    "arms": {},
                }
            )

    raw_rows: list[dict[str, object]] = []
    for model_index, model in enumerate(analysis.MODELS):
        for shard_index in range(2):
            for public in runner.shard_public_rows(
                public_rows,  # type: ignore[arg-type]
                shard_index=shard_index,
                shard_count=2,
            ):
                raw_rows.append(
                    {
                        "phase": "coarse",
                        "model_key": model,
                        "rank": model_index * 2 + shard_index,
                        "request_id": public["request_id"],
                        "frame_unit_id": public["frame_unit_id"],
                        "order_id": public["order_id"],
                        "public_request_sha256": runner.sha256_json(
                            runner._public_identity(public)
                        ),
                        "corruption": "blank_full",
                        "site_id": "vision_tower_final",
                        "stage_family": "vision_tower_final",
                        "intervention": "matched_clean_source_rescue",
                        "patch_direction": "clean_to_corrupt_rescue",
                        "source_request_id": public["request_id"],
                        "source_arm": "real",
                        "recipient_arm": "blank_full",
                        "slot_mapping": {
                            "source_slots": [0],
                            "recipient_slots": [0],
                            "crop_slots_unchanged": True,
                        },
                    }
                )
    receipt = analysis.validate_raw_public_alignment(
        raw_rows,  # type: ignore[arg-type]
        public_rows,  # type: ignore[arg-type]
    )
    assert receipt["status"] == "pass"
    assert receipt["hidden_index_opened"] is False
    raw_rows[0]["public_request_sha256"] = "0" * 64
    with pytest.raises(analysis.AnalysisError, match="public request hash mismatch"):
        analysis.validate_raw_public_alignment(
            raw_rows,  # type: ignore[arg-type]
            public_rows,  # type: ignore[arg-type]
        )


def test_frame_cluster_values_collapse_orders_then_frames() -> None:
    rows = [
        {"request_id": "a0", "frame_unit_id": "f1", "ego4d_participant_id": "p1", "absolute_recovery_bits": 1.0},
        {"request_id": "a1", "frame_unit_id": "f1", "ego4d_participant_id": "p1", "absolute_recovery_bits": 3.0},
        {"request_id": "b0", "frame_unit_id": "f2", "ego4d_participant_id": "p1", "absolute_recovery_bits": 5.0},
        {"request_id": "b1", "frame_unit_id": "f2", "ego4d_participant_id": "p1", "absolute_recovery_bits": 7.0},
        {"request_id": "c0", "frame_unit_id": "f3", "ego4d_participant_id": "p2", "absolute_recovery_bits": 9.0},
        {"request_id": "c1", "frame_unit_id": "f3", "ego4d_participant_id": "p2", "absolute_recovery_bits": 11.0},
    ]
    values = analysis.frame_cluster_values(rows, value_field="absolute_recovery_bits")
    assert values == {"p1": 4.0, "p2": 10.0}
    ci = analysis.bootstrap_ci(values, draws=100, seed=7)
    assert ci["estimate"] == 7.0
    assert ci["cluster_count"] == 2


def test_holm_adjustment_is_monotone() -> None:
    adjusted = analysis.holm_adjust({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.9})
    assert adjusted["a"] == pytest.approx(0.004)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["d"] == pytest.approx(0.9)


def test_decoder_conjunction_uses_all_six_components_at_one_selected_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    high = {"p0": 1.0, "p1": 1.0}
    low = {"p0": 0.1, "p1": 0.1}
    rescue_components = {
        "rescue_positive",
        "rescue_specificity_vs_unrelated",
        "rescue_specificity_vs_random",
    }

    def fake_component_maps(*_args, layer: int, **_kwargs):
        components = {
            component: (
                high
                if (layer == 0) == (component in rescue_components)
                else low
            )
            for component in analysis.CONFIRMATORY_COMPONENTS
        }
        counts = {component: runner.MECHANISM_TRIAL_COUNT for component in (
            "rescue",
            "rescue_unrelated",
            "rescue_random",
            "damage",
            "damage_unrelated",
            "damage_random",
        )}
        return components, counts, []

    def fake_intervals(layer_clusters, **_kwargs):
        return {
            layer: {
                "estimate": sum(values.values()) / len(values),
                "simultaneous_ci95": [0.01, 2.0],
                "cluster_count": len(values),
                "draws": 100,
            }
            for layer, values in layer_clusters.items()
        }

    monkeypatch.setattr(analysis, "_family_layers", lambda *_args: [0, 1])
    monkeypatch.setattr(analysis, "_component_cluster_maps", fake_component_maps)
    monkeypatch.setattr(analysis, "simultaneous_layer_intervals", fake_intervals)
    monkeypatch.setattr(
        analysis,
        "centered_cluster_bootstrap_pvalue",
        lambda values, **_kwargs: 0.001 if max(values.values()) > 0.5 else 0.4,
    )

    result = analysis._stage_analysis(
        [],
        model="qwen35_27b_hf_bf16",
        corruption="blank_full",
        family="decoder_visual_token_residual",
        bootstrap_draws=100,
        randomization_draws=100,
        seed=7,
    )

    assert result["selected_coarse_layer"] == 0
    assert result["raw_component_pvalues"]["rescue_positive"] == pytest.approx(0.001)
    assert result["raw_component_pvalues"]["damage_positive"] == pytest.approx(0.4)
    assert result["selected_site_unadjusted_conjunctive_pvalue"] == pytest.approx(0.4)
    assert result["layer_selection_candidate_count"] == 2
    assert result["layer_selection_adjustment"] == "bonferroni_over_frozen_coarse_sites"
    assert result["raw_conjunctive_pvalue"] == pytest.approx(0.8)


def test_reverse_damage_direction_is_clean_margin_minus_patched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = {
        "request_id": "r0",
        "frame_unit_id": "f0",
        "order_id": "order_0",
        "counterbalanced_request_id": "r1",
        "system_prompt": "",
        "user_prompt": "prompt",
        "arms": {},
    }
    hidden_path = tmp_path / "hidden.jsonl"
    hidden_path.write_text(
        json.dumps(
            {
                "request_id": "r0",
                "correct_choice": "A",
                "ego4d_participant_id": "p0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner, "validate_mechanism_manifest", lambda **_kwargs: {"hidden_sha256": "c" * 64}
    )
    monkeypatch.setattr(runner, "validate_dataset", lambda **_kwargs: {"status": "pass"})
    common = {
        "phase": "coarse",
        "model_key": "qwen35_27b_hf_bf16",
        "request_id": "r0",
        "frame_unit_id": "f0",
        "corruption": "blank_full",
        "source_request_id": "r0",
        "public_request_sha256": runner.sha256_json(runner._public_identity(public)),
        "real_baseline_a_minus_b_margin_bits": 2.0,
    }
    raw = [
        {
            **common,
            "intervention": "matched_clean_source_rescue",
            "patch_direction": "clean_to_corrupt_rescue",
            "recipient_baseline_a_minus_b_margin_bits": -1.0,
            "patched_a_minus_b_margin_bits": 0.0,
        },
        {
            **common,
            "intervention": "matched_corrupt_source_damage",
            "patch_direction": "corrupt_to_clean_damage",
            "recipient_baseline_a_minus_b_margin_bits": 2.0,
            "patched_a_minus_b_margin_bits": 0.5,
        },
    ]
    joined, _ = analysis.join_hidden_after_technical_gate(
        rows=raw,
        public_rows=[public],
        data_dir=tmp_path,
        mechanism_manifest=tmp_path / "manifest.json",
        public_manifest=tmp_path / "public.jsonl",
        hidden_index=hidden_path,
        expected_user_prompt="prompt",
    )
    damage = next(row for row in joined if row["patch_direction"] == "corrupt_to_clean_damage")
    assert damage["clean_minus_corrupt_deficit_bits"] == pytest.approx(3.0)
    assert damage["intervention_effect_bits"] == pytest.approx(1.5)


def test_corrupt_self_patch_identity_uses_corrupt_recipient_baseline() -> None:
    row = {
        "real_baseline_a_minus_b_margin_bits": 2.0,
        "recipient_baseline_a_minus_b_margin_bits": -1.0,
        "patched_a_minus_b_margin_bits": -1.0,
    }
    assert analysis.self_patch_identity_error_bits(row) == 0.0


def test_job_a_gate_is_hash_bound_and_model_specific(tmp_path: Path) -> None:
    summary = {
        "kind": analysis.JOB_A_SUMMARY_KIND,
        "status": "pass",
        "preregistration_sha256": runner.PRESPEC_SHA256,
        "frame_count": 800,
        "models": [
            {
                "model_key": "qwen35_27b_hf_bf16",
                "promotion_status": "pass",
                "real_capability": {"status": "pass"},
                "primary_counterbalanced_pair_correctness": {"status": "pass"},
                "primary_holm_adjusted_p": 0.01,
            },
            {
                "model_key": "gemma4_31b_hf_bf16",
                "promotion_status": "fail",
                "real_capability": {"status": "pass"},
                "primary_counterbalanced_pair_correctness": {"status": "fail"},
                "primary_holm_adjusted_p": 0.2,
            },
        ],
    }
    summary_path = tmp_path / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = tmp_path / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": analysis.JOB_A_MANIFEST_KIND,
                "status": "pass",
                "analysis_summary_sha256": runner.sha256_file(summary_path),
            }
        ),
        encoding="utf-8",
    )
    gate = analysis.load_job_a_promotion_gate(
        summary_path=summary_path,
        manifest_path=manifest_path,
        expected_summary_sha256=runner.sha256_file(summary_path),
    )
    assert gate["status"] == "pass"
    assert gate["models"]["qwen35_27b_hf_bf16"]["promotion_status"] == "pass"
    assert gate["models"]["qwen35_27b_hf_bf16"][
        "primary_counterbalanced_pair_correctness_status"
    ] == "pass"
    assert gate["models"]["gemma4_31b_hf_bf16"]["promotion_status"] == "fail"
    mismatched = analysis.load_job_a_promotion_gate(
        summary_path=summary_path,
        manifest_path=manifest_path,
        expected_summary_sha256="0" * 64,
    )
    assert mismatched["status"] == "fail"
    assert analysis.load_job_a_promotion_gate(
        summary_path=None, manifest_path=None
    )["status"] == "absent"


def test_baseline_deficit_gate_uses_96_participant_clusters() -> None:
    rows = []
    for index in range(96):
        for order in range(2):
            rows.append(
                {
                    "model_key": "qwen35_27b_hf_bf16",
                    "corruption": "blank_full",
                    "stage_family": "vision_tower_final",
                    "intervention": "matched_clean_source_rescue",
                    "status": "pass",
                    "request_id": f"r{index}-{order}",
                    "frame_unit_id": f"f{index}",
                    "ego4d_participant_id": f"p{index}",
                    "clean_minus_corrupt_deficit_bits": 1.0,
                }
            )
    gate = analysis.baseline_deficit_gate(
        rows,
        model="qwen35_27b_hf_bf16",
        corruption="blank_full",
        draws=200,
        seed=7,
    )
    assert gate["status"] == "pass"
    assert gate["cluster_count"] == 96
    assert gate["one_sided_lower_95"] > 0.0


def test_layer_crosswalk_names_lm_entry_and_post_block() -> None:
    crosswalk = runner.layer_crosswalk("qwen35_27b_hf_bf16")
    assert crosswalk["lm_entry"]["job_b_capture_state_index"] == 0
    assert crosswalk["lm_entry"]["job_c_patch_site"] is None
    assert crosswalk["post_blocks"][0]["state_name"] == "post_block_0"
    assert crosswalk["post_blocks"][0]["job_b_capture_state_index"] == 1
    assert runner.site_state_name(visual_site()) == "post_block_0"
    visual = {
        "stage_family": "decoder_visual_token_residual",
        "selected_coarse_layer": 8,
    }
    question = {
        "stage_family": "decoder_final_question_token_residual",
        "selected_coarse_layer": 8,
    }
    assert analysis.causal_stage_depth(visual) == analysis.causal_stage_depth(question)


def test_four_gpu_topology_assigns_two_shards_per_model() -> None:
    assignments = [runner.model_and_shard_for_rank(rank, 4) for rank in range(4)]
    assert assignments == [
        ("qwen35_27b_hf_bf16", 0, 2),
        ("qwen35_27b_hf_bf16", 1, 2),
        ("gemma4_31b_hf_bf16", 0, 2),
        ("gemma4_31b_hf_bf16", 1, 2),
    ]


def _write_passing_job_c_telemetry(path: Path) -> None:
    lines = [
        "timestamp,index,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w,temperature_gpu"
    ]
    for sample in range(runner.MIN_TELEMETRY_SAMPLES_PER_GPU):
        for gpu in range(4):
            lines.append(
                f"2026-07-21T00:00:{sample:02d}Z,{gpu},{70 + gpu},"
                f"{1000 + gpu},80000,400,60"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _job_c_architecture_receipt(model_key: str, *, coarse: bool) -> dict:
    adapter = runner.MODEL_ADAPTERS[model_key]
    layers = runner.coarse_layer_indexes(int(adapter["expected_decoder_layers"]))
    return {
        "model_key": model_key,
        "adapter": adapter,
        "decoder_layer_count": int(adapter["expected_decoder_layers"]),
        "selected_decoder_layers_by_family": (
            {
                "decoder_visual_token_residual": layers,
                "decoder_final_question_token_residual": layers,
            }
            if coarse
            else {
                "decoder_visual_token_residual": [],
                "decoder_final_question_token_residual": [],
            }
        ),
        "sites": (
            [{"site_id": site_id} for site_id in runner.expected_coarse_site_ids(model_key)]
            if coarse
            else []
        ),
    }


def _structural_canary_gate(model_key: str, choice: str) -> dict:
    records = []
    for site_id in runner.expected_structural_transfer_site_ids(model_key):
        if site_id in runner.PREFUSION_TRANSFER_FAMILIES:
            site = prefusion_site(site_id)
            audit = passing_identity_audit(site, choices=(choice,))
            reason = runner.PREFUSION_TRANSFER_NOT_APPLICABLE_REASON
            mapping = runner.structural_transfer_slot_mapping(
                site, identity_audit=audit
            )
        else:
            layer = int(site_id.rsplit("_", 1)[1])
            site = runner.PatchSite(
                site_id=site_id,
                family="decoder_final_question_token_residual",
                module_name="unused",
                tensor_selector="question_token",
                layer_index=layer,
            )
            reason = runner.QUESTION_TOKEN_TRANSFER_NOT_APPLICABLE_REASON
            mapping = runner.structural_transfer_slot_mapping(site)
        records.append(
            {
                "site_id": site.site_id,
                "stage_family": site.family,
                "reason": reason,
                "slot_mapping": mapping,
            }
        )
    return {
        "status": "pass",
        "not_applicable_count": len(records),
        "site_ids": [row["site_id"] for row in records],
        "intervention": "counterbalanced_content_aligned_crop_transfer",
        "sites": records,
        "reason_counts": dict(
            sorted(
                {
                    runner.PREFUSION_TRANSFER_NOT_APPLICABLE_REASON: 2,
                    runner.QUESTION_TOKEN_TRANSFER_NOT_APPLICABLE_REASON: (
                        len(records) - 2
                    ),
                }.items()
            )
        ),
    }


def _passing_canary_summary(rank: int) -> dict:
    model_key, _, _ = runner.model_and_shard_for_rank(rank, 4)
    site_ids = runner.expected_coarse_site_ids(model_key)
    transfer_site_count = len(
        [site_id for site_id in site_ids if site_id.startswith("decoder_visual_token")]
    )
    structural_not_applicable_site_count = len(site_ids) - transfer_site_count
    per_case_forwards = 8 + 15 * len(site_ids) + transfer_site_count
    cases = []
    for choice in runner.CHOICES:
        cases.append(
            {
                "status": "pass",
                "candidate_choice": choice,
                "forward_count": per_case_forwards,
                "expected_canary_forward_count": per_case_forwards,
                "question_token_audit": {
                    "status": "pass",
                    "contract": runner.QUESTION_TOKEN_CONTRACT,
                },
                "representable_intervention_gate": {
                    "status": "pass",
                    "unsupported_count": 0,
                    "unsupported_events": [],
                    "not_applicable_count": structural_not_applicable_site_count,
                    "invalid_not_applicable_count": 0,
                    "structural_not_applicable_gate": _structural_canary_gate(
                        model_key, choice
                    ),
                },
            }
        )
    return {
        "kind": "phase2_reanchored_causal_patch_evaluator_unjoined_canary",
        "status": "pass",
        "rank": rank,
        "model_key": model_key,
        "phase": "coarse",
        "governing_prespec_sha256": runner.PRESPEC_SHA256,
        "fixed_microbatch_size": 1,
        "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
        "candidate_choices_exercised": list(runner.CHOICES),
        "canary_case_count": len(runner.CHOICES),
        "canary_cases": cases,
        "label_joined": False,
        "outcome_metrics_emitted": False,
        "bidirectional_interventions_exercised": True,
        "architecture": _job_c_architecture_receipt(model_key, coarse=True),
        "code_snapshot_sha256": "a" * 64,
        "mechanism_manifest_sha256": "b" * 64,
        "public_manifest_sha256": "c" * 64,
        "model_plan_sha256": "d" * 64,
        "sif_path": "/scratch/container.sif",
        "sif_sha256": "e" * 64,
        "release_manifest_sha256": "f" * 64,
        "question_token_audit": {
            "status": "pass",
            "contract": runner.QUESTION_TOKEN_CONTRACT,
        },
        "representable_intervention_gate": {
            "status": "pass",
            "unsupported_count": 0,
            "unsupported_events": [],
            "not_applicable_count": (
                structural_not_applicable_site_count * len(runner.CHOICES)
            ),
            "invalid_not_applicable_count": 0,
        },
        "projected_full_inference_hours_per_rank_upper": 0.5,
        "load_elapsed_s": 60.0,
        "expected_canary_forward_count": per_case_forwards * len(runner.CHOICES),
        "forward_count": per_case_forwards * len(runner.CHOICES),
        "full_trials_per_rank": runner.MECHANISM_TRIAL_COUNT // 2,
        "peak_simultaneous_source_activation_rows": 1,
        "peak_cuda_memory_bytes": 1,
    }


def test_technical_canary_exercises_both_candidate_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices_seen: list[str] = []

    def fake_case(**kwargs: object) -> dict:
        choice = str(kwargs["choice"])
        choices_seen.append(choice)
        return {
            "status": "pass",
            "candidate_choice": choice,
            "request_id_sha256": "a" * 64,
            "representable_intervention_gate": {
                "status": "pass",
                "unsupported_count": 0,
                "unsupported_events": [],
                "not_applicable_count": 1,
                "invalid_not_applicable_count": 0,
                "structural_not_applicable_gate": {"status": "pass"},
            },
            "question_token_audit": {
                "status": "pass",
                "contract": runner.QUESTION_TOKEN_CONTRACT,
            },
            "slot_audit": {"status": "pass"},
            "hook_audits": [],
            "forward_count": 10,
            "expected_canary_forward_count": 10,
            "observed_forwards_per_s": 5.0,
            "expected_coarse_forwards_per_trial": 20,
            "expected_refine_upper_forwards_per_trial": 30,
            "full_trials_per_rank": runner.MECHANISM_TRIAL_COUNT // 2,
            "projected_full_inference_seconds_per_rank_upper": 60.0,
            "projected_full_inference_hours_per_rank_upper": 1.0 / 60.0,
            "projection_formula": {"version": 4},
            "peak_simultaneous_source_activation_rows": 1,
            "elapsed_s": 2.0,
        }

    monkeypatch.setattr(runner, "_run_technical_canary_case", fake_case)
    receipt = runner.run_technical_canary(
        runtime=object(),
        model=object(),
        processor=object(),
        tokenizer=object(),
        device=object(),
        model_key="qwen35_27b_hf_bf16",
        rank=0,
        request={},
        public_map={},
        sites=[],
        data_dir=Path("/unused"),
        candidate_completions={"A": "left", "B": "right"},
    )
    assert choices_seen == list(runner.CHOICES)
    assert receipt["status"] == "pass"
    assert receipt["candidate_choices_exercised"] == list(runner.CHOICES)
    assert receipt["canary_case_count"] == 2
    assert receipt["forward_count"] == 20


def test_telemetry_gate_rejects_sparse_single_gpu_evidence(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "timestamp,index,utilization_gpu_pct,memory_used_mib\n"
        "2026-07-21T00:00:00Z,0,100,1000\n",
        encoding="utf-8",
    )
    summary = runner.telemetry_summary(telemetry)
    assert summary["status"] == "fail"
    assert "gpu_inventory" in summary["issues"]
    assert "no_all_gpu_concurrent_residency_sample" in summary["issues"]


def _passing_worker_summary(
    rank: int,
    phase: str,
    *,
    refinement_sha: str | None = None,
    sif_sha: str | None = None,
) -> dict:
    model_key, _, _ = runner.model_and_shard_for_rank(rank, 4)
    expected_trials = runner.MECHANISM_TRIAL_COUNT // 2
    return {
        "kind": "phase2_reanchored_causal_patch_worker_summary",
        "status": "pass",
        "rank": rank,
        "model_key": model_key,
        "phase": phase,
        "governing_prespec_sha256": runner.PRESPEC_SHA256,
        "fixed_microbatch_size": 1,
        "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
        "architecture": _job_c_architecture_receipt(
            model_key, coarse=phase == "coarse"
        ),
        "trials_assigned": expected_trials,
        "trials_completed": expected_trials,
        "errors": 0,
        "one_trial_activation_cache": True,
        "peak_simultaneous_source_activation_rows": 1,
        "source_activation_tensors_persisted": False,
        "code_snapshot_sha256": "a" * 64,
        "mechanism_manifest_sha256": "b" * 64,
        "public_manifest_sha256": "c" * 64,
        "model_plan_sha256": "d" * 64,
        "sif_path": "/scratch/container.sif",
        "sif_sha256": sif_sha or "e" * 64,
        "release_manifest_sha256": "f" * 64,
        "refinement_plan_sha256": refinement_sha if phase == "refine" else None,
        "question_token_gate": {
            "status": "pass",
            "contract": runner.QUESTION_TOKEN_CONTRACT,
        },
        "representable_intervention_gate": {
            "status": "pass",
            "unsupported_count": 0,
            "not_applicable_count": 0,
            "invalid_not_applicable_count": 0,
        },
    }


def test_canary_aggregate_emits_hash_bound_runtime_profile(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.csv"
    _write_passing_job_c_telemetry(telemetry)
    for rank in range(4):
        base = tmp_path / f"rank_{rank:03d}" / "coarse"
        base.mkdir(parents=True)
        (base / "canary_summary.json").write_text(
            json.dumps(_passing_canary_summary(rank)),
            encoding="utf-8",
        )
    result = runner.aggregate_command(
        SimpleNamespace(
            run_dir=tmp_path,
            repo_dir=REPO_ROOT,
            canary=True,
            coarse_only=False,
            telemetry_csv=telemetry,
        )
    )
    assert result == 0
    profile_path = tmp_path / "runtime_profile.json"
    profile = json.loads(profile_path.read_text())
    assert profile["status"] == "pass"
    assert profile["headroom_fraction"] == pytest.approx(0.25)
    assert profile["code_snapshot_sha256"] == "a" * 64
    assert profile["sif_sha256"] == "e" * 64
    assert profile["release_manifest_sha256"] == "f" * 64
    assert profile["canary_aggregate_sha256"] == runner.sha256_file(
        tmp_path / "aggregate_summary.json"
    )
    aggregate = json.loads((tmp_path / "aggregate_summary.json").read_text())
    assert aggregate["unsupported_intervention_count"] == 0
    assert aggregate["structural_not_applicable_intervention_count"] == 88
    assert aggregate["worker_contract_violations"] == []

    failed_path = tmp_path / "rank_003" / "coarse" / "canary_summary.json"
    failed = json.loads(failed_path.read_text())
    failed["status"] = "fail"
    failed["representable_intervention_gate"] = {
        "status": "fail",
        "unsupported_count": 1,
        "not_applicable_count": 0,
        "invalid_not_applicable_count": 0,
    }
    failed_path.write_text(json.dumps(failed), encoding="utf-8")
    assert runner.aggregate_command(
        SimpleNamespace(
            run_dir=tmp_path,
            repo_dir=REPO_ROOT,
            canary=True,
            coarse_only=False,
            telemetry_csv=telemetry,
        )
    ) == 2
    aggregate = json.loads((tmp_path / "aggregate_summary.json").read_text())
    assert aggregate["unsupported_intervention_count"] == 1
    assert "worker_representable_intervention_gate:3:coarse" in aggregate["failures"]


def test_canary_aggregate_rejects_worker_identity_and_site_drift(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.csv"
    _write_passing_job_c_telemetry(telemetry)
    for rank in range(4):
        base = tmp_path / f"rank_{rank:03d}" / "coarse"
        base.mkdir(parents=True)
        payload = _passing_canary_summary(rank)
        if rank == 2:
            payload["model_key"] = "qwen35_27b_hf_bf16"
            payload["architecture"]["sites"] = payload["architecture"]["sites"][:-1]
        (base / "canary_summary.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    assert runner.aggregate_command(
        SimpleNamespace(
            run_dir=tmp_path,
            repo_dir=REPO_ROOT,
            canary=True,
            coarse_only=False,
            telemetry_csv=telemetry,
        )
    ) == 2
    aggregate = json.loads((tmp_path / "aggregate_summary.json").read_text())
    assert "worker_receipt_contract:2:coarse" in aggregate["failures"]
    violations = aggregate["worker_contract_violations"]
    assert violations == [
        {
            "rank": 2,
            "phase": "coarse",
            "violations": [
                "model_key",
                "architecture.coarse_site_inventory",
            ],
        }
    ]


def test_canary_worker_contract_rejects_count_preserving_structural_site_swap() -> None:
    payload = _passing_canary_summary(0)
    gate = payload["canary_cases"][0]["representable_intervention_gate"][
        "structural_not_applicable_gate"
    ]
    gate["sites"][0], gate["sites"][1] = gate["sites"][1], gate["sites"][0]
    violations = runner.worker_receipt_contract_violations(
        payload,
        rank=0,
        phase="coarse",
        canary=True,
    )
    assert "canary_case_0" in violations
    assert "canary_case_0.structural.site_record_ids" in violations


def test_coarse_aggregate_rejects_cross_worker_runtime_binding_drift(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.csv"
    _write_passing_job_c_telemetry(telemetry)
    for rank in range(4):
        base = tmp_path / f"rank_{rank:03d}" / "coarse"
        base.mkdir(parents=True)
        (base / "worker_summary.json").write_text(
            json.dumps(
                _passing_worker_summary(
                    rank,
                    "coarse",
                    sif_sha=("e" if rank < 3 else "9") * 64,
                )
            ),
            encoding="utf-8",
        )
    result = runner.aggregate_command(
        SimpleNamespace(
            run_dir=tmp_path,
            repo_dir=REPO_ROOT,
            canary=False,
            coarse_only=True,
            telemetry_csv=telemetry,
        )
    )
    assert result == 2
    aggregate = json.loads((tmp_path / "coarse_aggregate_summary.json").read_text())
    assert "worker_sif_sha256_binding_mismatch" in aggregate["failures"]


def test_full_aggregate_binds_exact_refinement_plan_across_refine_workers(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.csv"
    _write_passing_job_c_telemetry(telemetry)
    refinement_sha = "9" * 64
    for rank in range(4):
        for phase in ("coarse", "refine"):
            base = tmp_path / f"rank_{rank:03d}" / phase
            base.mkdir(parents=True)
            (base / "worker_summary.json").write_text(
                json.dumps(
                    _passing_worker_summary(
                        rank,
                        phase,
                        refinement_sha=refinement_sha,
                    )
                ),
                encoding="utf-8",
            )
    args = SimpleNamespace(
        run_dir=tmp_path,
        repo_dir=REPO_ROOT,
        canary=False,
        coarse_only=False,
        telemetry_csv=telemetry,
    )
    assert runner.aggregate_command(args) == 0
    aggregate = json.loads((tmp_path / "aggregate_summary.json").read_text())
    assert aggregate["immutable_input_bindings"]["refinement_plan_sha256"] == refinement_sha

    drifted = tmp_path / "rank_003" / "refine" / "worker_summary.json"
    payload = json.loads(drifted.read_text())
    payload["refinement_plan_sha256"] = "8" * 64
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.aggregate_command(args) == 2
    aggregate = json.loads((tmp_path / "aggregate_summary.json").read_text())
    assert "worker_refinement_plan_sha256_binding_mismatch" in aggregate["failures"]

    for rank in range(4):
        path = tmp_path / f"rank_{rank:03d}" / "refine" / "worker_summary.json"
        payload = json.loads(path.read_text())
        payload["refinement_plan_sha256"] = "not-a-sha256"
        path.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.aggregate_command(args) == 2
    aggregate = json.loads((tmp_path / "aggregate_summary.json").read_text())
    assert "worker_refinement_plan_sha256_binding_mismatch" in aggregate["failures"]


def test_job_c_worker_hash_bindings_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_PLAN_SHA256", raising=False)
    with pytest.raises(runner.GateError, match="MODEL_PLAN_SHA256"):
        runner.required_sha256_env("MODEL_PLAN_SHA256")
    monkeypatch.setenv("MODEL_PLAN_SHA256", "g" * 64)
    with pytest.raises(runner.GateError, match="MODEL_PLAN_SHA256"):
        runner.required_sha256_env("MODEL_PLAN_SHA256")
    monkeypatch.setenv("MODEL_PLAN_SHA256", "f" * 64)
    assert runner.required_sha256_env("MODEL_PLAN_SHA256") == "f" * 64


def test_global_holm_family_covers_all_16_configurations() -> None:
    raw = {f"config-{index}": 0.001 * (index + 1) for index in range(16)}
    adjusted = analysis.holm_adjust(raw)
    assert len(adjusted) == 16
    assert set(adjusted) == set(raw)


def test_governing_prespec_hash_is_current() -> None:
    receipt = runner.verify_governing_prespec(REPO_ROOT)
    assert receipt["sha256"] == "bd0f021fd510d873f388075a3d4c0e0d138da07fc6da8215cc0ff16b1d09279f"
    assert set(receipt["payload"]["candidate_completions"]) == {"A", "B"}
