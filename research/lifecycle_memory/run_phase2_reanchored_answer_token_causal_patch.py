#!/usr/bin/env python3
"""Answer-token causal-patch adapter for the validated re-anchored patch engine.

The legacy engine remains the implementation source for architecture discovery,
slot-exact patching, controls, resume state, telemetry, and aggregation. This
module binds that engine to the separately frozen answer-token confirmation and
replaces two candidate-continuation forwards with one shared-prefix forward.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory import run_phase2_reanchored_causal_patch as _engine
from research.lifecycle_memory.run_phase2_reanchored_causal_patch import *  # noqa: F403


PRESPEC_REL = Path(
    "docs/research/lifecycle_memory/phase 2 research/preregistration/"
    "reanchored_answer_token_causal_intervention_v1_2026_07_23.json"
)
PRESPEC_SHA256 = "21dfd51486c95e370dac2e5c9cc297dd897a136ad7772286467525ee06047703"
ANSWER_TOKEN_SOURCE_PRESPEC_SHA256 = (
    "920f441efc5faf4de0070a61cbf64132de7b1ebe2828bea9781e3b8cfa9715ba"
)
ANSWER_SHARED_PREFIX = '{"answer":"'
FROZEN_USER_PROMPT = (
    "Input 1 is a full frame. Inputs 2 and 3 are unmarked target-only reference "
    "crops A and B. Which referenced target is the first-order interacting "
    "object, meaning the object directly manipulated by a visible hand rather "
    "than a second-order object interacting through it? Return exactly one JSON "
    "object with answer A or B plus the frozen abstention and confidence keys."
)
ANSWER_TOKEN_EQUIVALENCE_TOLERANCE_BITS = 1e-3
MECHANISM_MANIFEST_SHA256 = (
    "c219a8f6746382edbb3e6a304fe85e750755e719874cf5775a2a6ef48a3fca88"
)
MECHANISM_PUBLIC_SHA256 = (
    "a6b9d166597efae8a26a65e2c654367da99fddef7ccaab6698a1646930083ced"
)
MECHANISM_HIDDEN_SHA256 = (
    "7a764586ac27a05fbe0d02151a632e91674d4b6c5721f4adba6bdae79a9725f1"
)
SOURCE_BATTERY_FRAME_COUNT = 320
MECHANISM_FRAME_COUNT = 80
MECHANISM_TRIAL_COUNT = 160
MECHANISM_PARTICIPANT_COUNT = 80
DEFAULT_SEED = 20260723


def verify_governing_prespec(repo_dir: Path) -> dict[str, Any]:
    path = (repo_dir / PRESPEC_REL).resolve()
    if not path.is_file():
        raise GateError(f"governing Job C-R prespec is missing: {path}")  # noqa: F405
    observed = sha256_file(path)  # noqa: F405
    if observed != PRESPEC_SHA256:
        raise GateError(  # noqa: F405
            "governing Job C-R prespec hash mismatch: "
            f"expected {PRESPEC_SHA256}, observed {observed}"
        )
    payload = read_json(path)  # noqa: F405
    if (
        payload.get("kind")
        != "phase2_reanchored_answer_token_causal_intervention_prespec"
        or payload.get("status")
        != (
            "prospectively_frozen_after_answer_token_confirmation_and_before_any_"
            "job_c_r_activation_patching_outcome"
        )
    ):
        raise GateError("governing Job C-R prespec kind or status drifted")  # noqa: F405
    if payload.get("candidate_completions") != {"A": "A", "B": "B"}:
        raise GateError("Job C-R answer-token candidates drifted")  # noqa: F405
    if payload.get("candidate_shared_prefix") != ANSWER_SHARED_PREFIX:
        raise GateError("Job C-R shared answer prefix drifted")  # noqa: F405
    if payload.get("generation", {}).get("prompt_text") != FROZEN_USER_PROMPT:
        raise GateError("Job C-R frozen user prompt drifted")  # noqa: F405
    eligibility = payload.get("behavioral_eligibility", {})
    if (
        eligibility.get("analysis_summary_sha256")
        != "8ebee5db19e8d123789cfb73bd6d6f5a7cd55a0f9473af8d01dfd6e3ef23a9c2"
        or eligibility.get("analysis_manifest_sha256")
        != "cb0d5b8d6f1871758b5c90469b1c8198e812ce170d970f61ac971c643c26a482"
        or eligibility.get("source_preregistration_sha256")
        != ANSWER_TOKEN_SOURCE_PRESPEC_SHA256
        or eligibility.get("required_frame_count") != SOURCE_BATTERY_FRAME_COUNT
    ):
        raise GateError("Job C-R behavioral eligibility binding drifted")  # noqa: F405
    expected_models = [
        {
            "model_id": MODEL_ADAPTERS[model_key]["model_id"],  # noqa: F405
            "model_key": model_key,
            "revision": MODEL_ADAPTERS[model_key]["revision"],  # noqa: F405
        }
        for model_key in (
            "qwen35_27b_hf_bf16",
            "gemma4_31b_hf_bf16",
        )
    ]
    if payload.get("models") != expected_models:
        raise GateError("Job C-R frozen model inventory drifted")  # noqa: F405
    job_c = payload.get("job_c", {})
    if job_c.get("models") != [row["model_key"] for row in expected_models]:
        raise GateError("Job C-R primary model list drifted")  # noqa: F405
    if job_c.get("corruptions") != list(CORRUPT_ARMS):  # noqa: F405
        raise GateError("Job C-R corruption inventory drifted")  # noqa: F405
    if job_c.get("primary_patch_families") != [
        "vision_tower_final",
        "projector_or_merger_output",
        "decoder_visual_token_residual",
    ]:
        raise GateError("Job C-R primary patch family drifted")  # noqa: F405
    fractions = payload.get("job_c", {}).get("coarse_decoder_layer_fractions")
    if fractions != [index / 8 for index in range(9)]:
        raise GateError(f"Job C-R coarse layer fractions drifted: {fractions!r}")  # noqa: F405
    mechanism = payload.get("data", {}).get("mechanism_subset", {})
    expected_mechanism = {
        "frame_count": MECHANISM_FRAME_COUNT,
        "trial_count": MECHANISM_TRIAL_COUNT,
        "participant_cluster_count": MECHANISM_PARTICIPANT_COUNT,
        "manifest_sha256": MECHANISM_MANIFEST_SHA256,
        "public_requests_sha256": MECHANISM_PUBLIC_SHA256,
        "hidden_evaluator_sha256": MECHANISM_HIDDEN_SHA256,
    }
    for key, expected in expected_mechanism.items():
        if mechanism.get(key) != expected:
            raise GateError(f"Job C-R mechanism {key} drifted")  # noqa: F405
    return {"path": str(path), "sha256": observed, "payload": payload}


def _answer_token_spec(tokenizer: Any) -> tuple[list[int], dict[str, list[int]]]:
    prefix_ids = _engine.job2c._encode_candidate_text(tokenizer, ANSWER_SHARED_PREFIX)
    if not prefix_ids:
        raise GateError("Job C-R shared answer prefix encoded to no tokens")  # noqa: F405
    answer_ids: dict[str, list[int]] = {}
    for choice in CHOICES:  # noqa: F405
        ids = _engine.job2c._encode_candidate_text(tokenizer, choice)
        if len(ids) != 1:
            raise GateError(  # noqa: F405
                f"Job C-R answer {choice} must encode as one token, got {len(ids)}"
            )
        combined = _engine.job2c._encode_candidate_text(
            tokenizer, ANSWER_SHARED_PREFIX + choice
        )
        if combined != prefix_ids + ids:
            raise GateError(  # noqa: F405
                f"Job C-R answer {choice} crosses the shared-prefix token boundary"
            )
        answer_ids[choice] = [int(ids[0])]
    if answer_ids["A"] == answer_ids["B"]:
        raise GateError("Job C-R A and B token IDs collide")  # noqa: F405
    return [int(value) for value in prefix_ids], answer_ids


def prepare_arm(
    *,
    runtime: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    model: Any,
    model_key: str,
    row: Mapping[str, Any],
    arm: str,
    data_dir: Path,
    candidate_completions: Mapping[str, str],
) -> PreparedArm:  # noqa: F405
    if dict(candidate_completions) != {"A": "A", "B": "B"}:
        raise GateError("Job C-R received non-answer-token candidates")  # noqa: F405
    assets = arm_assets(row, arm, data_dir)  # noqa: F405
    image_paths = [Path(asset["path"]) for asset in assets]
    system_prompt = str(row.get("system_prompt") or "")
    user_prompt = str(row.get("user_prompt") or "")
    if not user_prompt:
        raise GateError("public Job C-R row has no user_prompt")  # noqa: F405
    base_inputs, serialization = _engine.job3._prepare_universal_base(
        runtime=runtime,
        processor=processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=image_paths,
        device=device,
    )
    base_messages = _engine.job3._universal_user_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=image_paths,
    )
    unprefixed_ids = _engine._input_ids(base_inputs)
    if not unprefixed_ids:
        raise GateError("Job C-R base prompt has no tokens")  # noqa: F405
    question_token_audit = _engine.resolve_final_question_token(
        unprefixed_ids, tokenizer, user_prompt
    )
    if question_token_audit.get("status") != "pass":
        raise GateError(  # noqa: F405
            "Job C-R literal question-token boundary failed: "
            f"{question_token_audit.get('reason')}"
        )
    prefix_ids, answer_ids = _answer_token_spec(tokenizer)
    scoring_inputs, prefix_trace = (
        _engine.job2c._append_exact_candidate_continuation(base_inputs, prefix_ids)
    )
    prefix_visual_audit = _engine.job2c._visual_identity_audit(
        base_inputs, scoring_inputs
    )
    if prefix_visual_audit.get("status") != "pass":
        raise GateError("Job C-R shared prefix changed visual tensors")  # noqa: F405
    scoring_ids = _engine._input_ids(scoring_inputs)
    if scoring_ids != unprefixed_ids + prefix_ids:
        raise GateError("Job C-R shared prefix did not append at the exact boundary")  # noqa: F405
    adapter = MODEL_ADAPTERS[model_key]  # noqa: F405
    slot_map = build_image_slot_map(  # noqa: F405
        inputs=scoring_inputs,
        model=model,
        processor=processor,
        adapter=adapter,
    )
    choices: dict[str, PreparedChoice] = {}  # noqa: F405
    for choice in CHOICES:  # noqa: F405
        candidate_ids = answer_ids[choice]
        choices[choice] = PreparedChoice(  # noqa: F405
            choice=choice,
            inputs=scoring_inputs,
            base_length=len(scoring_ids),
            full_ids=[*scoring_ids, *candidate_ids],
            candidate_ids=candidate_ids,
            slot_map=slot_map,
            construction="one_shared_prefix_forward_next_token_score",
            policy_trace_sha256=sha256_json(  # noqa: F405
                {
                    "shared_prefix": ANSWER_SHARED_PREFIX,
                    "shared_prefix_ids": prefix_ids,
                    "answer_token_ids": answer_ids,
                    "prefix_append_trace": prefix_trace,
                    "visual_identity": prefix_visual_audit,
                }
            ),
        )
    serialization = {
        **dict(serialization),
        "answer_token_contract": {
            "status": "pass",
            "construction": "one_shared_prefix_forward_next_token_score",
            "shared_prefix_sha256": hashlib.sha256(
                ANSWER_SHARED_PREFIX.encode("utf-8")
            ).hexdigest(),
            "shared_prefix_token_ids": prefix_ids,
            "answer_token_ids": answer_ids,
            "prefix_visual_identity_status": "pass",
        },
    }
    return PreparedArm(  # noqa: F405
        arm=arm,
        base_inputs=scoring_inputs,
        base_messages=base_messages,
        base_ids=scoring_ids,
        question_position=int(question_token_audit["index"]),
        question_token_audit=question_token_audit,
        slot_map=slot_map,
        choices=choices,
        serialization=serialization,
        asset_receipts=[
            {
                "slot_index": index,
                "slot_name": IMAGE_SLOT_NAMES[index],  # noqa: F405
                "path": str(asset["path"]),
                "sha256": asset["sha256"],
                "width": asset["width"],
                "height": asset["height"],
            }
            for index, asset in enumerate(assets)
        ],
    )


def score_suffix(output: Any, prepared: PreparedChoice) -> float:  # noqa: F405
    import torch

    if len(prepared.candidate_ids) != 1:
        raise GateError("Job C-R candidate is not exactly one answer token")  # noqa: F405
    logits = _engine.job2c._extract_logits(output)
    if logits is None or getattr(logits, "ndim", 0) != 3:
        raise GateError("Job C-R output has no rank-3 logits")  # noqa: F405
    if int(logits.shape[0]) != 1 or int(logits.shape[1]) != prepared.base_length:
        raise GateError(  # noqa: F405
            "Job C-R shared-prefix logit shape drifted: "
            f"shape={list(logits.shape)} base_length={prepared.base_length}"
        )
    token_id = int(prepared.candidate_ids[0])
    if token_id < 0 or token_id >= int(logits.shape[-1]):
        raise GateError("Job C-R answer token ID is outside the vocabulary")  # noqa: F405
    value = float(
        torch.log_softmax(logits[0, prepared.base_length - 1].float(), dim=-1)[
            token_id
        ].item()
    )
    if not math.isfinite(value):
        raise GateError("Job C-R answer-token score is nonfinite")  # noqa: F405
    return value


def _assert_shared_choice_inputs(prepared: PreparedArm) -> PreparedChoice:  # noqa: F405
    first = prepared.choices["A"]
    second = prepared.choices["B"]
    if first.inputs is not second.inputs:
        if _engine._input_ids(first.inputs) != _engine._input_ids(second.inputs):
            raise GateError("Job C-R A/B shared-prefix inputs differ")  # noqa: F405
    if first.base_length != second.base_length:
        raise GateError("Job C-R A/B shared-prefix lengths differ")  # noqa: F405
    return first


def forward_capture_scores(
    *,
    model: Any,
    prepared: PreparedArm,  # noqa: F405
    sites: Sequence[PatchSite],  # noqa: F405
    slot_indexes_by_site: Mapping[str, Sequence[int]],
) -> tuple[dict[str, float], dict[str, dict[str, Any]], int]:
    import torch

    shared = _assert_shared_choice_inputs(prepared)
    with capture_module_outputs(  # noqa: F405
        model=model,
        sites=sites,
        slot_map=shared.slot_map,
        slot_indexes_by_site=slot_indexes_by_site,
        question_position=prepared.question_position,
    ) as captured:
        with torch.inference_mode():
            output = model(**shared.inputs, use_cache=False, return_dict=True)
    missing = [site.site_id for site in sites if site.site_id not in captured]
    if missing:
        raise GateError(f"Job C-R capture hooks did not fire: {missing}")  # noqa: F405
    scores = {
        choice: score_suffix(output, prepared.choices[choice])
        for choice in CHOICES  # noqa: F405
    }
    captures = {choice: captured for choice in CHOICES}  # noqa: F405
    del output
    return scores, captures, 1


def forward_plain_scores(
    model: Any, prepared: PreparedArm  # noqa: F405
) -> tuple[dict[str, float], int]:
    import torch

    shared = _assert_shared_choice_inputs(prepared)
    with torch.inference_mode():
        output = model(**shared.inputs, use_cache=False, return_dict=True)
    scores = {
        choice: score_suffix(output, prepared.choices[choice])
        for choice in CHOICES  # noqa: F405
    }
    del output
    return scores, 1


def forward_patched_scores(
    *,
    model: Any,
    prepared: PreparedArm,  # noqa: F405
    site: PatchSite,  # noqa: F405
    source_by_choice: Mapping[str, Any],
    recipient_slot_indexes: Sequence[int],
    mode: str,
    seed: int,
) -> tuple[dict[str, float] | None, list[dict[str, Any]], int, str | None]:
    import torch

    shared = _assert_shared_choice_inputs(prepared)
    source_a = source_by_choice["A"]
    source_b = source_by_choice["B"]
    if (
        list(source_a.shape) != list(source_b.shape)
        or source_a.dtype != source_b.dtype
        or not torch.equal(source_a, source_b)
    ):
        raise GateError("Job C-R A/B source activations are not shared-prefix identical")  # noqa: F405
    module = module_by_path(model, site.module_name)  # noqa: F405
    audit: HookAudit | None = None  # noqa: F405
    try:
        with patch_module_output(  # noqa: F405
            module=module,
            site=site,
            source_selected=source_a,
            recipient_slot_map=shared.slot_map,
            recipient_slot_indexes=recipient_slot_indexes,
            question_position=prepared.question_position,
            mode=mode,
            seed=seed,
        ) as audit:
            with torch.inference_mode():
                output = model(**shared.inputs, use_cache=False, return_dict=True)
    except DegenerateIntervention as exc:  # noqa: F405
        audits = (
            [{"candidate_choice": "shared_A_B", **asdict(audit)}]
            if audit is not None
            else []
        )
        return (
            None,
            audits,
            0,
            f"degenerate_intervention:{type(exc).__name__}:{exc}",
        )
    except GateError as exc:  # noqa: F405
        raise GateError(  # noqa: F405
            f"Job C-R patch failure site={site.site_id} mode={mode}: {exc}"
        ) from exc
    assert audit is not None
    if audit.status != "pass" or not audit.hook_removed or not audit.finite:
        raise GateError(f"Job C-R patch audit failed at {site.site_id}: {asdict(audit)}")  # noqa: F405
    scores = {
        choice: score_suffix(output, prepared.choices[choice])
        for choice in CHOICES  # noqa: F405
    }
    audits = [
        {
            "candidate_choice": choice,
            "shared_prefix_forward": True,
            **asdict(audit),
        }
        for choice in CHOICES  # noqa: F405
    ]
    del output
    return scores, audits, 1, None


_engine_run_technical_canary = _engine.run_technical_canary


def _answer_token_equivalence_audit(
    *,
    runtime: Any,
    model: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    model_key: str,
    request: dict[str, Any],
    data_dir: Path,
    candidate_completions: Mapping[str, str],
) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    prepared = prepare_arm(
        runtime=runtime,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        model=model,
        model_key=model_key,
        row=request,
        arm="real",
        data_dir=data_dir,
        candidate_completions=candidate_completions,
    )
    one_pass, one_forward = forward_plain_scores(model, prepared)
    rows: dict[str, Any] = {}
    max_abs_bits = 0.0
    for choice in CHOICES:  # noqa: F405
        candidate = prepared.choices[choice]
        full_inputs, append_trace = (
            _engine.job2c._append_exact_candidate_continuation(
                prepared.base_inputs, candidate.candidate_ids
            )
        )
        visual_audit = _engine.job2c._visual_identity_audit(
            prepared.base_inputs, full_inputs
        )
        if visual_audit.get("status") != "pass":
            raise GateError("Job C-R equivalence candidate changed visual tensors")  # noqa: F405
        with torch.inference_mode():
            output = model(**full_inputs, use_cache=False, return_dict=True)
        scored = _engine.job2c.score_candidate_suffix_from_logits(
            base_input_ids=prepared.base_ids,
            full_input_ids=_engine._input_ids(full_inputs),
            logits=_engine.job2c._extract_logits(output),
        )
        if scored.get("status") != "pass":
            raise GateError(f"Job C-R suffix equivalence failed for {choice}: {scored}")  # noqa: F405
        suffix_value = float(scored["sum_logprob"])
        delta_bits = abs(one_pass[choice] - suffix_value) / math.log(2.0)
        max_abs_bits = max(max_abs_bits, delta_bits)
        rows[choice] = {
            "one_pass_logprob": one_pass[choice],
            "suffix_logprob": suffix_value,
            "absolute_difference_bits": delta_bits,
            "answer_token_ids": candidate.candidate_ids,
            "append_trace_sha256": sha256_json(append_trace),  # noqa: F405
        }
        del output, full_inputs
    return {
        "status": (
            "pass"
            if max_abs_bits <= ANSWER_TOKEN_EQUIVALENCE_TOLERANCE_BITS
            else "fail"
        ),
        "construction": "one_shared_prefix_forward_vs_exact_appended_suffix",
        "tolerance_bits": ANSWER_TOKEN_EQUIVALENCE_TOLERANCE_BITS,
        "maximum_absolute_difference_bits": max_abs_bits,
        "choices": rows,
        "forward_count": one_forward + len(CHOICES),  # noqa: F405
        "elapsed_s": time.perf_counter() - started,
        "label_joined": False,
    }


def run_technical_canary(**kwargs: Any) -> dict[str, Any]:
    receipt = _engine_run_technical_canary(**kwargs)
    equivalence = _answer_token_equivalence_audit(
        runtime=kwargs["runtime"],
        model=kwargs["model"],
        processor=kwargs["processor"],
        tokenizer=kwargs["tokenizer"],
        device=kwargs["device"],
        model_key=kwargs["model_key"],
        request=kwargs["request"],
        data_dir=kwargs["data_dir"],
        candidate_completions=kwargs["candidate_completions"],
    )
    receipt["answer_token_equivalence_gate"] = equivalence
    if equivalence["status"] != "pass":
        receipt["status"] = "fail"
        receipt["bidirectional_interventions_exercised"] = False
    site_count = len(kwargs["sites"])
    transfer_site_count = sum(
        int(_engine.transfer_site_applicability(site)[0])
        for site in kwargs["sites"]
    )
    coarse_forwards = 11 + 19 * site_count + transfer_site_count
    refine_upper_forwards = 10 + 19 * 8 + 4
    observed_rate = float(receipt.get("observed_forwards_per_s") or 0.0)
    projected_seconds = (
        (MECHANISM_TRIAL_COUNT // 2)
        * (coarse_forwards + refine_upper_forwards)
        / observed_rate
        if observed_rate > 0
        else None
    )
    receipt["expected_coarse_forwards_per_trial"] = coarse_forwards
    receipt["expected_refine_upper_forwards_per_trial"] = refine_upper_forwards
    receipt["full_trials_per_rank"] = MECHANISM_TRIAL_COUNT // 2
    receipt["projected_full_inference_seconds_per_rank_upper"] = projected_seconds
    receipt["projected_full_inference_hours_per_rank_upper"] = (
        projected_seconds / 3600.0 if projected_seconds is not None else None
    )
    receipt["projection_formula"] = {
        "version": 5,
        "coarse": "11 + 19*coarse_site_count + transfer_site_count",
        "refinement_upper": "10 + 19*8 + 4",
        "single_shared_prefix_forward_for_both_answer_tokens": True,
        "canary_candidate_paths_exercised": list(CHOICES),  # noqa: F405
    }
    return receipt


def _configure_engine() -> None:
    values = {
        "PRESPEC_REL": PRESPEC_REL,
        "PRESPEC_SHA256": PRESPEC_SHA256,
        "MECHANISM_GOVERNING_PRESPEC_SHA256": (
            ANSWER_TOKEN_SOURCE_PRESPEC_SHA256
        ),
        "EXPECTED_MECHANISM_MANIFEST_SHA256": MECHANISM_MANIFEST_SHA256,
        "EXPECTED_MECHANISM_PUBLIC_SHA256": MECHANISM_PUBLIC_SHA256,
        "EXPECTED_MECHANISM_HIDDEN_SHA256": MECHANISM_HIDDEN_SHA256,
        "SOURCE_BATTERY_FRAME_COUNT": SOURCE_BATTERY_FRAME_COUNT,
        "MECHANISM_FRAME_COUNT": MECHANISM_FRAME_COUNT,
        "MECHANISM_TRIAL_COUNT": MECHANISM_TRIAL_COUNT,
        "MECHANISM_PARTICIPANT_COUNT": MECHANISM_PARTICIPANT_COUNT,
        "DEFAULT_SEED": DEFAULT_SEED,
        "PROJECTION_FORMULA_VERSION": 5,
        "REQUIRED_CANARY_ANSWER_TOKEN_EQUIVALENCE": True,
        "CANARY_ANSWER_TOKEN_EQUIVALENCE_TOLERANCE_BITS": (
            ANSWER_TOKEN_EQUIVALENCE_TOLERANCE_BITS
        ),
    }
    for name, value in values.items():
        setattr(_engine, name, value)
    _engine.verify_governing_prespec = verify_governing_prespec
    _engine.prepare_arm = prepare_arm
    _engine.score_suffix = score_suffix
    _engine.forward_capture_scores = forward_capture_scores
    _engine.forward_plain_scores = forward_plain_scores
    _engine.forward_patched_scores = forward_patched_scores
    _engine.run_technical_canary = run_technical_canary


_configure_engine()


def main() -> int:
    _configure_engine()
    return _engine.main()


if __name__ == "__main__":
    raise SystemExit(main())
