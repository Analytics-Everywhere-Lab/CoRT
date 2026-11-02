#!/usr/bin/env python3
"""Analyze bidirectional causal-patch rescue/damage sweeps and descriptive refinements.

All statistics first average the two counterbalanced crop orders within frame,
then frames within participant, and finally give participant clusters equal
weight.  Confirmatory inference uses only frozen coarse sites.  The automatic
plus/minus-two refinement is descriptive because peak selection is not replayed
inside every inferential resample.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory import run_phase2_reanchored_causal_patch as runner


DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_RANDOMIZATION_DRAWS = 100_000
DEFAULT_SEED = 20260721
ALPHA = 0.05
STAGE_FAMILIES = (
    "vision_tower_final",
    "projector_or_merger_output",
    "decoder_visual_token_residual",
    "decoder_final_question_token_residual",
)
PRIMARY_STAGE_FAMILIES = STAGE_FAMILIES
MODELS = ("qwen35_27b_hf_bf16", "gemma4_31b_hf_bf16")
CORRUPTIONS = ("blank_full", "wrong_time")
CONFIRMATORY_COMPONENTS = (
    "rescue_positive",
    "damage_positive",
    "rescue_specificity_vs_unrelated",
    "rescue_specificity_vs_random",
    "damage_specificity_vs_unrelated",
    "damage_specificity_vs_random",
)
JOB_A_SUMMARY_KIND = "phase2_reanchored_job_a_full_analysis"
JOB_A_MANIFEST_KIND = "phase2_reanchored_job_a_analysis_manifest"
BEHAVIORAL_ELIGIBILITY_FRAME_COUNT = 800
BEHAVIORAL_ELIGIBILITY_PRESPEC_SHA256: str | None = None
BEHAVIORAL_ELIGIBILITY_RESULT_KEY = "job_a_800_frame_promotion_gate"
BEHAVIORAL_ELIGIBILITY_CALL_GATE_KEY = "job_a_800_frame_model_promotion_gate"
REPORT_EARLIEST_CAUSAL_SITE = True
ANALYSIS_KIND = "phase2_reanchored_causal_patch_analysis"
ANALYSIS_CLAIM_BOUNDARY = (
    "A positive bidirectional call identifies a rescue/sufficiency and matched damage "
    "bottleneck under this frozen intervention. Even the earliest passing site is not the "
    "historical origin of failure: downstream clean states can bypass upstream defects. "
    "A null is no detected effect under the intervention, not absence of information."
)


class AnalysisError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise AnalysisError(f"non-object row {path}:{line_no}")
            rows.append(value)
    return rows


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise AnalysisError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean(values: Sequence[float]) -> float:
    if not values:
        raise AnalysisError("mean requires values")
    return sum(float(value) for value in values) / len(values)


def self_patch_identity_error_bits(row: Mapping[str, Any]) -> float:
    """Measure an identity patch against its actual recipient, clean or corrupt."""
    return abs(
        float(row["patched_a_minus_b_margin_bits"])
        - float(row["recipient_baseline_a_minus_b_margin_bits"])
    )


def causal_stage_depth(row: Mapping[str, Any]) -> int:
    """Order computation stages without ordering token subsets within one block."""
    family = str(row["stage_family"])
    if family == "vision_tower_final":
        return 0
    if family == "projector_or_merger_output":
        return 1
    layer = row.get("selected_coarse_layer")
    if family in {
        "decoder_visual_token_residual",
        "decoder_final_question_token_residual",
    } and isinstance(layer, int):
        return 2 + layer
    raise AnalysisError(f"cannot order causal stage {family}:{layer}")


def load_phase_rows(run_dir: Path, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in range(4):
        path = run_dir / f"rank_{rank:03d}" / phase / "patch_outcomes.jsonl"
        if not path.exists():
            raise AnalysisError(f"missing worker outcomes: {path}")
        rows.extend(read_jsonl(path))
    return rows


def coarse_site_contract() -> dict[str, dict[str, dict[str, Any]]]:
    contract: dict[str, dict[str, dict[str, Any]]] = {}
    for model in MODELS:
        layer_count = int(runner.MODEL_ADAPTERS[model]["expected_decoder_layers"])
        sites: dict[str, dict[str, Any]] = {
            "vision_tower_final": {
                "stage_family": "vision_tower_final",
                "layer_index": None,
            },
            "projector_or_merger_output": {
                "stage_family": "projector_or_merger_output",
                "layer_index": None,
            },
        }
        for layer in runner.coarse_layer_indexes(layer_count):
            sites[f"decoder_visual_token_residual.layer_{layer:03d}"] = {
                "stage_family": "decoder_visual_token_residual",
                "layer_index": layer,
            }
            sites[f"decoder_final_question_token_residual.layer_{layer:03d}"] = {
                "stage_family": "decoder_final_question_token_residual",
                "layer_index": layer,
            }
        contract[model] = sites
    return contract


def refinement_site_contract(
    path: Path,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    if not path.is_file():
        raise AnalysisError(f"refinement plan is missing: {path}")
    payload = read_json(path)
    if payload.get("kind") != "phase2_reanchored_causal_patch_refinement_plan":
        raise AnalysisError("refinement plan kind mismatch")
    if payload.get("status") != "pass":
        raise AnalysisError("refinement plan is not pass")
    if payload.get("governing_prespec_sha256") != runner.PRESPEC_SHA256:
        raise AnalysisError("refinement plan prespec hash mismatch")
    models = payload.get("models")
    if not isinstance(models, Mapping) or set(models) != set(MODELS):
        raise AnalysisError("refinement plan does not contain exactly both anchor models")

    contract: dict[str, dict[str, dict[str, Any]]] = {}
    receipt_models: dict[str, Any] = {}
    for model in MODELS:
        model_payload = models.get(model)
        if not isinstance(model_payload, Mapping):
            raise AnalysisError(f"invalid refinement payload for {model}")
        layer_count = int(runner.MODEL_ADAPTERS[model]["expected_decoder_layers"])
        coarse = runner.coarse_layer_indexes(layer_count)
        model_sites: dict[str, dict[str, Any]] = {}
        receipt_families: dict[str, Any] = {}
        for family in (
            "decoder_visual_token_residual",
            "decoder_final_question_token_residual",
        ):
            family_payload = model_payload.get(family)
            if not isinstance(family_payload, Mapping):
                raise AnalysisError(f"refinement plan lacks {model}:{family}")
            if family_payload.get("coarse_layers") != coarse:
                raise AnalysisError(f"refinement coarse grid drift for {model}:{family}")
            peak = family_payload.get("selected_peak_layer")
            if not isinstance(peak, int) or peak not in coarse:
                raise AnalysisError(f"invalid selected peak for {model}:{family}")
            raw_layers = family_payload.get("refinement_layers")
            if not isinstance(raw_layers, list) or any(
                not isinstance(layer, int) for layer in raw_layers
            ):
                raise AnalysisError(f"invalid refinement layers for {model}:{family}")
            expected_layers = sorted(
                set(
                    range(
                        max(0, peak - 2),
                        min(layer_count - 1, peak + 2) + 1,
                    )
                ).difference(coarse)
            )
            if raw_layers != expected_layers or not expected_layers:
                raise AnalysisError(
                    f"refinement rule mismatch for {model}:{family}: "
                    f"{raw_layers!r} != {expected_layers!r}"
                )
            if family_payload.get("inferential_scope") != "descriptive_only":
                raise AnalysisError(f"refinement scope drift for {model}:{family}")
            for layer in expected_layers:
                site_id = f"{family}.layer_{layer:03d}"
                model_sites[site_id] = {
                    "stage_family": family,
                    "layer_index": layer,
                }
            receipt_families[family] = {
                "selected_peak_layer": peak,
                "refinement_layers": expected_layers,
            }
        contract[model] = model_sites
        receipt_models[model] = receipt_families
    return contract, {
        "status": "pass",
        "path": str(path),
        "sha256": runner.sha256_file(path),
        "models": receipt_models,
        "inferential_scope": "descriptive_only",
    }


def _allowed_unsupported_row(row: Mapping[str, Any]) -> bool:
    intervention = str(row.get("intervention"))
    reason = str(row.get("reason") or "")
    audits = row.get("patch_audits")
    sample_degenerate = (
        reason.startswith("degenerate_intervention:DegenerateIntervention:")
        and intervention
        not in {"clean_to_clean_self_patch", "corrupt_to_corrupt_self_patch"}
        and row.get("patched_a_minus_b_margin_bits") is None
        and isinstance(audits, list)
        and 1 <= len(audits) <= len(runner.CHOICES)
        and all(isinstance(audit, Mapping) for audit in audits)
        and all(audit.get("status") == "pass" for audit in audits[:-1])
        and isinstance(audits[-1], Mapping)
        and audits[-1].get("status") == "fail"
        and audits[-1].get("hook_removed") is True
        and str(audits[-1].get("error") or "").startswith("DegenerateIntervention:")
    )
    return bool(sample_degenerate)


def _allowed_not_applicable_row(row: Mapping[str, Any]) -> bool:
    return runner.is_structural_not_applicable_row(row)


def raw_outcome_schema_violations(row: Mapping[str, Any]) -> dict[str, list[str]]:
    """Require the exact public worker schema; arbitrary extra fields are forbidden."""
    observed = set(row)
    return {
        "missing": sorted(runner.OUTCOME_ROW_FIELDS - observed),
        "unexpected": sorted(observed - runner.OUTCOME_ROW_FIELDS),
    }


def validate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    phase: str,
    expected_sites_by_model: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise AnalysisError(f"no {phase} rows")
    if phase == "coarse":
        expected_sites = coarse_site_contract()
        if expected_sites_by_model is not None and expected_sites_by_model != expected_sites:
            raise AnalysisError("caller-supplied coarse site contract drifted")
    elif phase == "refine":
        if expected_sites_by_model is None:
            raise AnalysisError("refinement row validation requires the frozen refinement plan")
        expected_sites = {
            str(model): {str(site): dict(metadata) for site, metadata in sites.items()}
            for model, sites in expected_sites_by_model.items()
        }
    else:
        raise AnalysisError(f"unknown phase: {phase}")
    if set(expected_sites) != set(MODELS):
        raise AnalysisError("site contract does not contain exactly both anchor models")
    identities: set[tuple[str, ...]] = set()
    duplicates: list[tuple[str, ...]] = []
    failures: list[str] = []
    for row in rows:
        schema_violations = raw_outcome_schema_violations(row)
        if schema_violations["missing"] or schema_violations["unexpected"]:
            failures.append(
                "raw_outcome_schema:"
                f"missing={schema_violations['missing']}:"
                f"unexpected={schema_violations['unexpected']}"
            )
        if row.get("schema_version") != runner.OUTPUT_SCHEMA_VERSION:
            failures.append("schema_version")
        if row.get("phase") != phase:
            failures.append("phase")
        identity = runner._result_identity(row)
        if identity in identities:
            duplicates.append(identity)
        identities.add(identity)
        status = str(row.get("status"))
        if status not in set(runner.RESULT_STATUSES):
            failures.append(f"status:{identity}:{status}")
        model = str(row.get("model_key"))
        site_id = str(row.get("site_id"))
        site_contract = expected_sites.get(model, {}).get(site_id)
        if site_contract is None:
            failures.append(f"unexpected_site:{model}:{site_id}")
        else:
            if row.get("stage_family") != site_contract["stage_family"]:
                failures.append(f"stage_family:{identity}")
            if row.get("layer_index") != site_contract["layer_index"]:
                failures.append(f"layer_index:{identity}")
        expected_scope = "coarse_primary" if phase == "coarse" else "refinement_descriptive"
        if row.get("inferential_scope") != expected_scope:
            failures.append(f"inferential_scope:{identity}")
        intervention = str(row.get("intervention"))
        corruption = str(row.get("corruption"))
        allowed_corruptions = {
            "matched_clean_source_rescue": set(CORRUPTIONS),
            "rescue_norm_matched_random": set(CORRUPTIONS),
            "rescue_unrelated_donor_candidate": set(CORRUPTIONS),
            "matched_corrupt_source_damage": set(CORRUPTIONS),
            "damage_norm_matched_random": set(CORRUPTIONS),
            "damage_unrelated_donor_candidate": set(CORRUPTIONS),
            "clean_to_clean_self_patch": {"real"},
            "corrupt_to_corrupt_self_patch": set(CORRUPTIONS),
            "counterbalanced_content_aligned_crop_transfer": {"counterbalanced_order"},
        }
        if intervention not in allowed_corruptions or corruption not in allowed_corruptions.get(
            intervention, set()
        ):
            failures.append(f"endpoint_pair:{identity}")
        expected_directions = {
            "matched_clean_source_rescue": "clean_to_corrupt_rescue",
            "rescue_norm_matched_random": "clean_to_corrupt_rescue",
            "rescue_unrelated_donor_candidate": "clean_to_corrupt_rescue",
            "matched_corrupt_source_damage": "corrupt_to_clean_damage",
            "damage_norm_matched_random": "corrupt_to_clean_damage",
            "damage_unrelated_donor_candidate": "corrupt_to_clean_damage",
            "clean_to_clean_self_patch": "identity_control",
            "corrupt_to_corrupt_self_patch": "identity_control",
            "counterbalanced_content_aligned_crop_transfer": "counterbalanced_crop_transfer",
        }
        if row.get("patch_direction") != expected_directions.get(intervention):
            failures.append(f"patch_direction:{identity}")
        expected_state = (
            f"post_block_{row.get('layer_index')}"
            if row.get("layer_index") is not None
            else str(row.get("stage_family"))
        )
        if row.get("state_name") != expected_state:
            failures.append(f"state_name:{identity}")
        if row.get("layer_index") is not None and row.get("layer_crosswalk_rule") != (
            "Job B direct-hook capture state i+1 corresponds to Job C raw post_block_i"
        ):
            failures.append(f"layer_crosswalk:{identity}")
        if status == "unsupported":
            if not _allowed_unsupported_row(row):
                failures.append(f"unsupported_not_predeclared:{identity}")
        if status == "not_applicable":
            if not _allowed_not_applicable_row(row):
                failures.append(f"not_applicable_not_predeclared:{identity}")
        transfer_status_violation = runner.transfer_status_contract_violation(row)
        if transfer_status_violation is not None:
            failures.append(
                f"crop_transfer_status_contract:{identity}:{transfer_status_violation}"
            )
        if row.get("label_joined") is not False:
            failures.append(f"worker_label_firewall:{identity}")
        for forbidden in ("correct_choice", "ego4d_participant_id", "absolute_recovery_bits"):
            if forbidden in row:
                failures.append(f"worker_hidden_field:{forbidden}:{identity}")
        for field in (
            "recipient_baseline_a_minus_b_margin_bits",
            "real_baseline_a_minus_b_margin_bits",
        ):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failures.append(f"nonfinite:{field}:{identity}")
        patched_value = row.get("patched_a_minus_b_margin_bits")
        if status == "pass" and (
            not isinstance(patched_value, (int, float))
            or not math.isfinite(float(patched_value))
        ):
            failures.append(f"nonfinite:patched_a_minus_b_margin_bits:{identity}")
        audits = row.get("patch_audits")
        if status == "pass":
            if not isinstance(audits, list) or len(audits) != len(runner.CHOICES):
                failures.append(f"patch_audits:{identity}")
            else:
                for audit in audits:
                    if not isinstance(audit, Mapping):
                        failures.append(f"patch_audit_type:{identity}")
                        continue
                    if audit.get("status") != "pass":
                        failures.append(f"patch_audit_status:{identity}")
                    if audit.get("hook_removed") is not True:
                        failures.append(f"hook_not_removed:{identity}")
                    if audit.get("finite") is not True:
                        failures.append(f"patch_nonfinite:{identity}")
                    outside = audit.get("max_abs_change_outside")
                    if not isinstance(outside, (int, float)) or float(outside) != 0.0:
                        failures.append(f"unpatched_position_changed:{identity}")
            if intervention in {
                "clean_to_clean_self_patch",
                "corrupt_to_corrupt_self_patch",
            } and isinstance(
                patched_value, (int, float)
            ):
                recipient_value = row.get("recipient_baseline_a_minus_b_margin_bits")
                if (
                    isinstance(recipient_value, (int, float))
                    and self_patch_identity_error_bits(row)
                    > runner.SELF_MARGIN_TOLERANCE_BITS
                ):
                    failures.append(f"self_patch_not_identity:{identity}")
        if not isinstance(row.get("public_request_sha256"), str) or len(
            str(row.get("public_request_sha256"))
        ) != 64:
            failures.append(f"public_request_hash:{identity}")
    if duplicates:
        failures.append(f"duplicate_count:{len(duplicates)}")
    if failures:
        raise AnalysisError(f"row validation failed: {failures[:20]}")
    trial_ids = {str(row["request_id"]) for row in rows}
    frame_ids = {str(row["frame_unit_id"]) for row in rows}
    models = {str(row["model_key"]) for row in rows}
    if models != set(MODELS):
        raise AnalysisError(f"expected both anchor models, found {sorted(models)}")
    by_frame_model: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_frame_model[(str(row["model_key"]), str(row["frame_unit_id"]))].add(str(row["order_id"]))
    bad_pairs = [key for key, orders in by_frame_model.items() if orders != {"order_0", "order_1"}]
    if bad_pairs:
        raise AnalysisError(f"counterbalanced order completeness failed for {bad_pairs[:10]}")
    for model in MODELS:
        model_rows = [row for row in rows if row.get("model_key") == model]
        model_trials = {str(row["request_id"]) for row in model_rows}
        model_frames = {str(row["frame_unit_id"]) for row in model_rows}
        if len(model_trials) != runner.MECHANISM_TRIAL_COUNT:
            raise AnalysisError(f"{model} has {len(model_trials)} trials, expected 192")
        if len(model_frames) != runner.MECHANISM_FRAME_COUNT:
            raise AnalysisError(f"{model} has {len(model_frames)} frames, expected 96")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["model_key"]),
                str(row["corruption"]),
                str(row["site_id"]),
                str(row["intervention"]),
            )
        ].append(row)
    expected_by_intervention = {
        "matched_clean_source_rescue": runner.MECHANISM_TRIAL_COUNT,
        "rescue_norm_matched_random": runner.MECHANISM_TRIAL_COUNT,
        "rescue_unrelated_donor_candidate": runner.MECHANISM_TRIAL_COUNT * 2,
        "matched_corrupt_source_damage": runner.MECHANISM_TRIAL_COUNT,
        "damage_norm_matched_random": runner.MECHANISM_TRIAL_COUNT,
        "damage_unrelated_donor_candidate": runner.MECHANISM_TRIAL_COUNT * 2,
        "clean_to_clean_self_patch": runner.MECHANISM_TRIAL_COUNT,
        "corrupt_to_corrupt_self_patch": runner.MECHANISM_TRIAL_COUNT,
        "counterbalanced_content_aligned_crop_transfer": runner.MECHANISM_TRIAL_COUNT,
    }
    for key, selected in grouped.items():
        intervention = key[-1]
        expected = expected_by_intervention.get(intervention)
        if expected is None or len(selected) != expected:
            raise AnalysisError(
                f"raw endpoint completeness failed for {key}: {len(selected)} != {expected}"
            )
        counts: dict[str, int] = defaultdict(int)
        for row in selected:
            counts[str(row["request_id"])] += 1
        per_request = 2 if intervention in {
            "rescue_unrelated_donor_candidate",
            "damage_unrelated_donor_candidate",
        } else 1
        if len(counts) != runner.MECHANISM_TRIAL_COUNT or set(counts.values()) != {
            per_request
        }:
            raise AnalysisError(f"raw endpoint request multiplicity failed for {key}")
    for model in MODELS:
        site_ids = {
            str(row["site_id"]) for row in rows if row.get("model_key") == model
        }
        expected_site_ids = set(expected_sites[model])
        if site_ids != expected_site_ids:
            raise AnalysisError(
                f"{phase} site completeness failed for {model}: "
                f"missing={sorted(expected_site_ids - site_ids)}, "
                f"unexpected={sorted(site_ids - expected_site_ids)}"
            )
        for site_id in site_ids:
            required = {
                (model, corruption, site_id, intervention)
                for corruption in CORRUPTIONS
                for intervention in (
                    "matched_clean_source_rescue",
                    "rescue_norm_matched_random",
                    "rescue_unrelated_donor_candidate",
                    "matched_corrupt_source_damage",
                    "damage_norm_matched_random",
                    "damage_unrelated_donor_candidate",
                    "corrupt_to_corrupt_self_patch",
                )
            }
            required.update(
                {
                    (model, "real", site_id, "clean_to_clean_self_patch"),
                    (
                        model,
                        "counterbalanced_order",
                        site_id,
                        "counterbalanced_content_aligned_crop_transfer",
                    ),
                }
            )
            missing = required.difference(grouped)
            if missing:
                raise AnalysisError(
                    f"raw required endpoint groups missing for {model}:{site_id}: "
                    f"{sorted(missing)}"
                )
    return {
        "status": "pass",
        "row_count": len(rows),
        "trial_count": len(trial_ids),
        "frame_unit_count": len(frame_ids),
        "models": sorted(models),
        "label_firewall": "pass",
        "endpoint_group_count": len(grouped),
        "expected_site_count_by_model": {
            model: len(expected_sites[model]) for model in MODELS
        },
    }


def representable_intervention_gate(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build an explicit completeness gate for representable interventions."""
    unsupported = [row for row in rows if row.get("status") == "unsupported"]
    not_applicable = [row for row in rows if row.get("status") == "not_applicable"]
    invalid_not_applicable = [
        row for row in not_applicable if not _allowed_not_applicable_row(row)
    ]
    transfer_status_violations = [
        reason
        for row in rows
        for reason in [runner.transfer_status_contract_violation(row)]
        if reason is not None
    ]
    unsupported_grouped: dict[str, int] = defaultdict(int)
    for row in unsupported:
        key = "|".join(
            str(row.get(field) or "")
            for field in ("phase", "model_key", "intervention", "site_id")
        )
        unsupported_grouped[key] += 1
    not_applicable_grouped: dict[str, int] = defaultdict(int)
    for row in not_applicable:
        key = "|".join(
            str(row.get(field) or "")
            for field in ("phase", "model_key", "intervention", "site_id")
        )
        not_applicable_grouped[key] += 1
    return {
        "kind": "phase2_reanchored_causal_patch_representable_intervention_gate",
        "status": (
            "pass"
            if not unsupported
            and not invalid_not_applicable
            and not transfer_status_violations
            else "fail"
        ),
        "unsupported_count": len(unsupported),
        "unsupported_counts_by_phase_model_intervention_site": dict(
            sorted(unsupported_grouped.items())
        ),
        "not_applicable_count": len(not_applicable),
        "invalid_not_applicable_count": len(invalid_not_applicable),
        "transfer_status_violation_count": len(transfer_status_violations),
        "transfer_status_violations": transfer_status_violations[:20],
        "not_applicable_counts_by_phase_model_intervention_site": dict(
            sorted(not_applicable_grouped.items())
        ),
        "rule": (
            "content-aligned crop transfer is structurally not applicable only where fresh "
            "exact activation equality proves an identity, and at question-token sites "
            "without a crop slice; those rows remain audited non-effect records, while all "
            "applicable unsupported rows are preserved and cannot be dropped, imputed, or "
            "converted to zero effects"
        ),
    }


def validate_public_inputs_after_technical_gate(
    *,
    repo_dir: Path,
    data_dir: Path,
    mechanism_manifest: Path,
    public_manifest: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prespec = runner.verify_governing_prespec(repo_dir)
    mechanism = runner.validate_mechanism_manifest(
        data_dir=data_dir,
        manifest_path=mechanism_manifest,
        public_path=public_manifest,
    )
    public_rows = runner.read_jsonl(public_manifest)
    dataset = runner.validate_dataset(
        public_rows=public_rows,
        hidden_rows=None,
        data_dir=data_dir,
        expected_frames=runner.MECHANISM_FRAME_COUNT,
        expected_user_prompt=str(prespec["payload"]["generation"]["prompt_text"]),
    )
    return public_rows, {"mechanism": mechanism, "public_dataset": dataset}


def validate_raw_public_alignment(
    rows: Sequence[dict[str, Any]],
    public_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate all method-facing joins before the evaluator index is opened."""
    public_map: dict[str, dict[str, Any]] = {}
    for public in public_rows:
        request_id = str(public.get("request_id") or "")
        if not request_id or request_id in public_map:
            raise AnalysisError(f"public request id is missing or duplicated: {request_id!r}")
        public_map[request_id] = public
    public_ids = set(public_map)
    if len(public_ids) != runner.MECHANISM_TRIAL_COUNT:
        raise AnalysisError(
            f"public manifest has {len(public_ids)} requests, expected "
            f"{runner.MECHANISM_TRIAL_COUNT}"
        )

    expected_rank: dict[tuple[str, str], int] = {}
    for model_index, model in enumerate(MODELS):
        rank_offset = model_index * 2
        for shard_index in range(2):
            selected = runner.shard_public_rows(
                public_rows,
                shard_index=shard_index,
                shard_count=2,
            )
            for public in selected:
                key = (model, str(public["request_id"]))
                if key in expected_rank:
                    raise AnalysisError(f"public request assigned to multiple shards: {key}")
                expected_rank[key] = rank_offset + shard_index

    donor_pairs = runner.select_unrelated_donor_pairs(public_rows)
    observed_unrelated: dict[tuple[str, str, str, str, str, str], set[str]] = defaultdict(set)
    model_request_ids: dict[str, set[str]] = defaultdict(set)
    checked_hashes: set[str] = set()
    for raw in rows:
        request_id = str(raw["request_id"])
        model = str(raw["model_key"])
        public = public_map.get(request_id)
        if public is None:
            raise AnalysisError(f"raw outcome references absent public request {request_id}")
        model_request_ids[model].add(request_id)
        expected_hash = runner.sha256_json(runner._public_identity(public))
        if raw.get("public_request_sha256") != expected_hash:
            raise AnalysisError(f"public request hash mismatch for raw outcome {request_id}")
        checked_hashes.add(expected_hash)
        if str(raw.get("frame_unit_id")) != str(public["frame_unit_id"]):
            raise AnalysisError(f"frame id mismatch for raw outcome {request_id}")
        if str(raw.get("order_id")) != str(public["order_id"]):
            raise AnalysisError(f"crop order mismatch for raw outcome {request_id}")
        rank = raw.get("rank")
        if not isinstance(rank, int) or rank != expected_rank.get((model, request_id)):
            raise AnalysisError(
                f"static shard mismatch for {model}:{request_id}: "
                f"observed={rank}, expected={expected_rank.get((model, request_id))}"
            )

        intervention = str(raw["intervention"])
        source_id = str(raw.get("source_request_id") or "")
        if intervention in {
            "matched_clean_source_rescue",
            "rescue_norm_matched_random",
            "matched_corrupt_source_damage",
            "damage_norm_matched_random",
            "clean_to_clean_self_patch",
            "corrupt_to_corrupt_self_patch",
        }:
            if source_id != request_id:
                raise AnalysisError(f"self/matched source mismatch for {request_id}")
        elif intervention == "counterbalanced_content_aligned_crop_transfer":
            if source_id != str(public["counterbalanced_request_id"]):
                raise AnalysisError(f"counterbalanced source mismatch for {request_id}")
        elif intervention in {
            "rescue_unrelated_donor_candidate",
            "damage_unrelated_donor_candidate",
        }:
            source = public_map.get(source_id)
            if source is None or str(source["frame_unit_id"]) == str(public["frame_unit_id"]):
                raise AnalysisError(f"unrelated source is absent or same-frame for {request_id}")
            observed_unrelated[
                (
                    str(raw["phase"]),
                    model,
                    str(raw["corruption"]),
                    str(raw["site_id"]),
                    request_id,
                    str(raw["intervention"]),
                )
            ].add(source_id)
        else:
            raise AnalysisError(f"unknown intervention in raw/public audit: {intervention}")

        direction = str(raw.get("patch_direction") or "")
        source_arm = str(raw.get("source_arm") or "")
        recipient_arm = str(raw.get("recipient_arm") or "")
        corruption = str(raw.get("corruption") or "")
        if direction == "clean_to_corrupt_rescue" and (
            source_arm != "real" or recipient_arm != corruption
        ):
            raise AnalysisError(f"rescue source/recipient arm drift for {request_id}")
        if direction == "corrupt_to_clean_damage" and (
            source_arm != corruption or recipient_arm != "real"
        ):
            raise AnalysisError(f"damage source/recipient arm drift for {request_id}")
        if intervention == "clean_to_clean_self_patch" and (
            source_arm != "real" or recipient_arm != "real"
        ):
            raise AnalysisError(f"clean self-patch arm drift for {request_id}")
        if intervention == "corrupt_to_corrupt_self_patch" and (
            source_arm != corruption or recipient_arm != corruption
        ):
            raise AnalysisError(f"corrupt self-patch arm drift for {request_id}")

        family = str(raw["stage_family"])
        mapping = raw.get("slot_mapping")
        if not isinstance(mapping, Mapping):
            raise AnalysisError(f"slot mapping is absent for {request_id}:{raw['site_id']}")
        if intervention == "counterbalanced_content_aligned_crop_transfer":
            if mapping.get("full_frame_slot_unchanged") is not True:
                raise AnalysisError(f"crop-order transfer changed or failed to audit slot 0: {request_id}")
            if raw.get("status") == "pass" and (
                mapping.get("source_slots") != [1, 2]
                or mapping.get("recipient_slots") != [2, 1]
            ):
                raise AnalysisError(f"counterbalanced crop permutation drift for {request_id}")
        else:
            expected_slots = [] if family == "decoder_final_question_token_residual" else [0]
            if mapping.get("source_slots") != expected_slots:
                raise AnalysisError(f"source slot mapping drift for {request_id}:{raw['site_id']}")
            if mapping.get("recipient_slots") != expected_slots:
                raise AnalysisError(f"recipient slot mapping drift for {request_id}:{raw['site_id']}")
            if mapping.get("crop_slots_unchanged") is not True:
                raise AnalysisError(f"crop slots were not proven unchanged for {request_id}")

    if set(model_request_ids) != set(MODELS):
        raise AnalysisError("raw/public audit does not contain exactly both anchor models")
    for model in MODELS:
        if model_request_ids[model] != public_ids:
            raise AnalysisError(
                f"raw/public request completeness failed for {model}: "
                f"missing={len(public_ids - model_request_ids[model])}, "
                f"unexpected={len(model_request_ids[model] - public_ids)}"
            )
    for key, observed in observed_unrelated.items():
        request_id = key[4]
        expected = set(donor_pairs[request_id])
        if observed != expected:
            raise AnalysisError(
                f"deterministic unrelated donor pair drift for {key}: {observed} != {expected}"
            )
    return {
        "status": "pass",
        "raw_row_count": len(rows),
        "public_request_count": len(public_ids),
        "verified_public_request_hash_count": len(checked_hashes),
        "static_shards": "pass",
        "source_request_mappings": "pass",
        "slot_mapping_audits": "pass",
        "hidden_index_opened": False,
    }


def join_hidden_after_technical_gate(
    *,
    rows: Sequence[dict[str, Any]],
    public_rows: Sequence[dict[str, Any]],
    data_dir: Path,
    mechanism_manifest: Path,
    public_manifest: Path,
    hidden_index: Path,
    expected_user_prompt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open labels only after aggregate/raw/public technical validation has passed."""
    mechanism = runner.validate_mechanism_manifest(
        data_dir=data_dir,
        manifest_path=mechanism_manifest,
        public_path=public_manifest,
        hidden_path=hidden_index,
        require_hidden_outside_data_dir=True,
    )
    hidden_rows = runner.read_jsonl(hidden_index)
    hidden_dataset = runner.validate_dataset(
        public_rows=public_rows,
        hidden_rows=hidden_rows,
        data_dir=data_dir,
        expected_frames=runner.MECHANISM_FRAME_COUNT,
        expected_user_prompt=expected_user_prompt,
    )
    hidden_map = {str(row["request_id"]): row for row in hidden_rows}
    public_map = {str(row["request_id"]): row for row in public_rows}
    corrupt_baselines_raw: dict[tuple[str, str, str, str], float] = {}
    for raw in rows:
        if raw.get("intervention") != "matched_clean_source_rescue":
            continue
        key = (
            str(raw.get("phase")),
            str(raw.get("model_key")),
            str(raw.get("request_id")),
            str(raw.get("corruption")),
        )
        value = float(raw["recipient_baseline_a_minus_b_margin_bits"])
        previous = corrupt_baselines_raw.setdefault(key, value)
        if abs(previous - value) > 1e-12:
            raise AnalysisError(f"corrupt baseline drift across sites for {key}")
    joined: list[dict[str, Any]] = []
    dropped_opposite_label = 0
    kept_same_label = 0
    for raw in rows:
        request_id = str(raw["request_id"])
        public = public_map.get(request_id)
        hidden = hidden_map.get(request_id)
        if public is None or hidden is None:
            raise AnalysisError(f"raw outcome cannot join request {request_id}")
        if raw.get("public_request_sha256") != runner.sha256_json(
            runner._public_identity(public)
        ):
            raise AnalysisError(f"public request hash mismatch for outcome {request_id}")
        intervention = str(raw["intervention"])
        source_id = str(raw.get("source_request_id") or "")
        if intervention in {
            "rescue_unrelated_donor_candidate",
            "damage_unrelated_donor_candidate",
        }:
            source_hidden = hidden_map.get(source_id)
            source_public = public_map.get(source_id)
            if source_hidden is None or source_public is None:
                raise AnalysisError(f"unrelated donor cannot join {source_id}")
            if str(source_public["frame_unit_id"]) == str(public["frame_unit_id"]):
                raise AnalysisError("unrelated donor reuses the recipient frame")
            if str(source_hidden["ego4d_participant_id"]) == str(
                hidden["ego4d_participant_id"]
            ):
                raise AnalysisError("unrelated donor reuses the recipient participant")
            if str(source_hidden["correct_choice"]) != str(hidden["correct_choice"]):
                dropped_opposite_label += 1
                continue
            intervention = (
                "rescue_unrelated_same_label_donor"
                if intervention == "rescue_unrelated_donor_candidate"
                else "damage_unrelated_same_label_donor"
            )
            kept_same_label += 1
        correct_choice = str(hidden["correct_choice"])
        sign = 1.0 if correct_choice == "A" else -1.0
        recipient = sign * float(raw["recipient_baseline_a_minus_b_margin_bits"])
        real = sign * float(raw["real_baseline_a_minus_b_margin_bits"])
        baseline_key = (
            str(raw.get("phase")),
            str(raw.get("model_key")),
            request_id,
            str(raw.get("corruption")),
        )
        corrupt_raw = corrupt_baselines_raw.get(baseline_key)
        corrupt = sign * corrupt_raw if corrupt_raw is not None else None
        patched_raw = raw.get("patched_a_minus_b_margin_bits")
        patched = sign * float(patched_raw) if patched_raw is not None else None
        direction = str(raw.get("patch_direction") or "")
        if patched is None:
            effect = None
        elif direction == "clean_to_corrupt_rescue":
            if corrupt is None:
                raise AnalysisError(f"rescue row lacks corrupt baseline for {baseline_key}")
            effect = patched - corrupt
        elif direction == "corrupt_to_clean_damage":
            if corrupt is None:
                raise AnalysisError(f"damage row lacks corrupt baseline for {baseline_key}")
            effect = real - patched
        else:
            effect = patched - recipient
        denominator = real - corrupt if corrupt is not None else None
        normalized = (
            effect / denominator
            if effect is not None and denominator is not None and abs(denominator) > 1e-12
            else None
        )
        joined.append(
            {
                **raw,
                "intervention": intervention,
                "label_joined": True,
                "correct_choice": correct_choice,
                "ego4d_participant_id": str(hidden["ego4d_participant_id"]),
                "recipient_baseline_margin_bits": recipient,
                "real_baseline_margin_bits": real,
                "clean_baseline_margin_bits": real,
                "corrupt_baseline_margin_bits": corrupt,
                "clean_minus_corrupt_deficit_bits": denominator,
                "patched_margin_bits": patched,
                "intervention_effect_bits": effect,
                "absolute_recovery_bits": effect,
                "recovery_denominator_bits": denominator,
                "normalized_recovery": normalized,
            }
        )
    expected_candidates = sum(
        1
        for row in rows
        if row.get("intervention")
        in {"rescue_unrelated_donor_candidate", "damage_unrelated_donor_candidate"}
    )
    if expected_candidates and kept_same_label * 2 != expected_candidates:
        raise AnalysisError("same-label unrelated donor join did not retain exactly one of each pair")
    if dropped_opposite_label != kept_same_label:
        raise AnalysisError("unrelated donor label counterbalance join is asymmetric")
    return joined, {
        "status": "pass",
        "hidden_opened_after_technical_gates": True,
        "hidden_sha256": mechanism["hidden_sha256"],
        "hidden_dataset": hidden_dataset,
        "same_label_donor_rows_kept": kept_same_label,
        "opposite_label_donor_rows_dropped": dropped_opposite_label,
        "joined_row_count": len(joined),
    }


def _endpoint_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int | None]:
    layer = row.get("layer_index")
    return (
        str(row["model_key"]),
        str(row["corruption"]),
        str(row["stage_family"]),
        str(row["site_id"]),
        int(layer) if layer is not None else None,
    )


def frame_cluster_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str = "intervention_effect_bits",
) -> dict[str, float]:
    """Collapse order -> frame -> participant, preserving the statistical unit."""
    by_cluster_frame: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AnalysisError(f"invalid {value_field} for {runner._result_identity(row)}")
        key = (str(row["ego4d_participant_id"]), str(row["frame_unit_id"]))
        by_cluster_frame[key].append(float(value))
    bad_order_counts = [key for key, values in by_cluster_frame.items() if len(values) != 2]
    if bad_order_counts:
        raise AnalysisError(f"endpoint lacks exactly two crop orders: {bad_order_counts[:10]}")
    frames_by_cluster: dict[str, list[float]] = defaultdict(list)
    for (cluster, _frame), values in by_cluster_frame.items():
        frames_by_cluster[cluster].append(mean(values))
    return {cluster: mean(values) for cluster, values in frames_by_cluster.items()}


def paired_cluster_values(
    matched: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    control_map = {str(row["request_id"]): row for row in control}
    differences: list[dict[str, Any]] = []
    for row in matched:
        request_id = str(row["request_id"])
        other = control_map.get(request_id)
        if other is None:
            raise AnalysisError(f"control row missing for {request_id}")
        if str(other["frame_unit_id"]) != str(row["frame_unit_id"]):
            raise AnalysisError("paired control frame mismatch")
        value = float(row["absolute_recovery_bits"]) - float(other["absolute_recovery_bits"])
        differences.append({**row, "paired_difference": value})
    return frame_cluster_values(differences, value_field="paired_difference")


def bootstrap_ci(
    cluster_values: Mapping[str, float], *, draws: int, seed: int
) -> dict[str, Any]:
    keys = sorted(cluster_values)
    if len(keys) < 2:
        raise AnalysisError("cluster bootstrap requires at least two clusters")
    estimate = mean([cluster_values[key] for key in keys])
    values = np.asarray([cluster_values[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    remaining = draws
    while remaining:
        batch = min(4096, remaining)
        indexes = rng.integers(0, len(keys), size=(batch, len(keys)))
        boot.extend(values[indexes].mean(axis=1).tolist())
        remaining -= batch
    return {
        "estimate": estimate,
        "ci95": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "one_sided_lower_95": percentile(boot, 0.05),
        "cluster_count": len(keys),
        "draws": draws,
        "seed": seed,
    }


def centered_cluster_bootstrap_pvalue(
    cluster_values: Mapping[str, float], *, draws: int, seed: int
) -> float:
    """One-sided participant-cluster bootstrap under a zero-mean null."""
    keys = sorted(cluster_values)
    if len(keys) < 2:
        raise AnalysisError("centered cluster bootstrap requires at least two clusters")
    values = np.asarray([float(cluster_values[key]) for key in keys], dtype=np.float64)
    observed = float(values.mean())
    centered = values - observed
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = draws
    while remaining:
        batch = min(4096, remaining)
        indexes = rng.integers(0, len(keys), size=(batch, len(keys)))
        statistics = centered[indexes].mean(axis=1)
        exceed += int(np.count_nonzero(statistics >= observed - 1e-15))
        remaining -= batch
    return (exceed + 1) / (draws + 1)


def sign_flip_pvalue(
    cluster_values: Mapping[str, float], *, draws: int, seed: int
) -> float:
    values = [float(cluster_values[key]) for key in sorted(cluster_values)]
    observed = mean(values)
    if len(values) <= 20:
        total = 1 << len(values)
        exceed = 0
        for mask in range(total):
            statistic = mean(
                [value if (mask >> index) & 1 else -value for index, value in enumerate(values)]
            )
            exceed += int(statistic >= observed - 1e-15)
        return exceed / total
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = draws
    while remaining:
        batch = min(8192, remaining)
        signs = rng.integers(0, 2, size=(batch, len(values)), dtype=np.int8) * 2 - 1
        statistics = signs @ array / len(values)
        exceed += int(np.count_nonzero(statistics >= observed - 1e-15))
        remaining -= batch
    return (exceed + 1) / (draws + 1)


def simultaneous_layer_intervals(
    layer_clusters: Mapping[int, Mapping[str, float]], *, draws: int, seed: int
) -> dict[int, dict[str, Any]]:
    layers = sorted(layer_clusters)
    if not layers:
        return {}
    clusters = sorted(set.intersection(*(set(layer_clusters[layer]) for layer in layers)))
    if len(clusters) < 2:
        raise AnalysisError("simultaneous layer inference requires >=2 common clusters")
    estimates = {
        layer: mean([layer_clusters[layer][cluster] for cluster in clusters]) for layer in layers
    }
    matrix = np.asarray(
        [[layer_clusters[layer][cluster] for cluster in clusters] for layer in layers],
        dtype=np.float64,
    )
    estimate_array = matrix.mean(axis=1)
    rng = np.random.default_rng(seed)
    deviations: list[float] = []
    remaining = draws
    while remaining:
        batch = min(1024, remaining)
        indexes = rng.integers(0, len(clusters), size=(batch, len(clusters)))
        boot_means = matrix[:, indexes].mean(axis=2).T
        deviations.extend(np.max(np.abs(boot_means - estimate_array), axis=1).tolist())
        remaining -= batch
    critical = percentile(deviations, 0.95)
    return {
        layer: {
            "estimate": estimates[layer],
            "simultaneous_ci95": [estimates[layer] - critical, estimates[layer] + critical],
            "max_stat_critical_value": critical,
            "cluster_count": len(clusters),
            "draws": draws,
        }
        for layer in layers
    }


def max_layer_sign_flip_pvalue(
    layer_clusters: Mapping[int, Mapping[str, float]], *, draws: int, seed: int
) -> tuple[float, int, float]:
    layers = sorted(layer_clusters)
    clusters = sorted(set.intersection(*(set(layer_clusters[layer]) for layer in layers)))
    estimates = {
        layer: mean([layer_clusters[layer][cluster] for cluster in clusters]) for layer in layers
    }
    selected_layer = min(layers, key=lambda layer: (-estimates[layer], layer))
    observed = estimates[selected_layer]
    matrix = np.asarray(
        [[layer_clusters[layer][cluster] for cluster in clusters] for layer in layers],
        dtype=np.float64,
    ).T
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = draws
    while remaining:
        batch = min(4096, remaining)
        signs = rng.integers(0, 2, size=(batch, len(clusters)), dtype=np.int8) * 2 - 1
        maxima = np.max(signs @ matrix / len(clusters), axis=1)
        exceed += int(np.count_nonzero(maxima >= observed - 1e-15))
        remaining -= batch
    return (exceed + 1) / (draws + 1), selected_layer, observed


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * float(value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def rows_for_endpoint(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    corruption: str,
    family: str,
    intervention: str,
    layer: int | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("model_key") == model
        and row.get("corruption") == corruption
        and row.get("stage_family") == family
        and row.get("intervention") == intervention
        and row.get("status") == "pass"
    ]
    if layer is not None:
        selected = [row for row in selected if row.get("layer_index") == layer]
    return selected


def _family_layers(rows: Sequence[dict[str, Any]], model: str, family: str) -> list[int]:
    return sorted(
        {
            int(row["layer_index"])
            for row in rows
            if row.get("model_key") == model
            and row.get("stage_family") == family
            and row.get("layer_index") is not None
        }
    )


def _component_cluster_maps(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    corruption: str,
    family: str,
    layer: int | None,
) -> tuple[dict[str, dict[str, float]], dict[str, int], list[str]]:
    endpoints = {
        "rescue": "matched_clean_source_rescue",
        "rescue_unrelated": "rescue_unrelated_same_label_donor",
        "rescue_random": "rescue_norm_matched_random",
        "damage": "matched_corrupt_source_damage",
        "damage_unrelated": "damage_unrelated_same_label_donor",
        "damage_random": "damage_norm_matched_random",
    }
    selected = {
        name: rows_for_endpoint(
            rows,
            model=model,
            corruption=corruption,
            family=family,
            intervention=intervention,
            layer=layer,
        )
        for name, intervention in endpoints.items()
    }
    counts = {name: len(values) for name, values in selected.items()}
    if set(counts.values()) != {runner.MECHANISM_TRIAL_COUNT}:
        raise AnalysisError(
            "bidirectional endpoint must contain exactly "
            f"{runner.MECHANISM_TRIAL_COUNT} trials for "
            f"{model}:{corruption}:{family}:{layer}, got {counts}"
        )
    components = {
        "rescue_positive": frame_cluster_values(selected["rescue"]),
        "damage_positive": frame_cluster_values(selected["damage"]),
        "rescue_specificity_vs_unrelated": paired_cluster_values(
            selected["rescue"], selected["rescue_unrelated"]
        ),
        "rescue_specificity_vs_random": paired_cluster_values(
            selected["rescue"], selected["rescue_random"]
        ),
        "damage_specificity_vs_unrelated": paired_cluster_values(
            selected["damage"], selected["damage_unrelated"]
        ),
        "damage_specificity_vs_random": paired_cluster_values(
            selected["damage"], selected["damage_random"]
        ),
    }
    self_rows = [
        *rows_for_endpoint(
            rows,
            model=model,
            corruption="real",
            family=family,
            intervention="clean_to_clean_self_patch",
            layer=layer,
        ),
        *rows_for_endpoint(
            rows,
            model=model,
            corruption=corruption,
            family=family,
            intervention="corrupt_to_corrupt_self_patch",
            layer=layer,
        ),
    ]
    if len(self_rows) != 2 * runner.MECHANISM_TRIAL_COUNT:
        raise AnalysisError(f"bidirectional self controls are incomplete for {model}:{corruption}:{family}:{layer}")
    failures = [
        str(row.get("request_id"))
        for row in self_rows
        if row.get("status") != "pass"
        or abs(float(row.get("intervention_effect_bits") or 0.0))
        > runner.SELF_MARGIN_TOLERANCE_BITS
    ]
    return components, counts, failures


def _stage_analysis(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    corruption: str,
    family: str,
    bootstrap_draws: int,
    randomization_draws: int,
    seed: int,
) -> dict[str, Any]:
    fixed_stage = family in {"vision_tower_final", "projector_or_merger_output"}
    layers_for_analysis: list[int | None] = (
        [None] if fixed_stage else _family_layers(rows, model, family)
    )
    component_layers: dict[str, dict[int | None, dict[str, float]]] = {
        component: {} for component in CONFIRMATORY_COMPONENTS
    }
    endpoint_counts: dict[str, dict[str, int]] = {}
    self_failures: list[str] = []
    for layer in layers_for_analysis:
        components, counts, failures = _component_cluster_maps(
            rows,
            model=model,
            corruption=corruption,
            family=family,
            layer=layer,
        )
        for component, values in components.items():
            component_layers[component][layer] = values
        endpoint_counts[str(layer)] = counts
        self_failures.extend(failures)

    layer_scores = {
        layer: min(
            mean(list(component_layers[component][layer].values()))
            for component in CONFIRMATORY_COMPONENTS
        )
        for layer in layers_for_analysis
    }
    selected_layer = min(
        layers_for_analysis,
        key=lambda layer: (-layer_scores[layer], -1 if layer is None else layer),
    )
    component_results: dict[str, Any] = {}
    raw_component_pvalues: dict[str, float] = {}
    curves: dict[str, Any] = {}
    for index, component in enumerate(CONFIRMATORY_COMPONENTS):
        by_layer = component_layers[component]
        component_seed = seed + index * 17
        if fixed_stage:
            summary = bootstrap_ci(
                by_layer[None], draws=bootstrap_draws, seed=component_seed
            )
            pvalue = centered_cluster_bootstrap_pvalue(
                by_layer[None], draws=randomization_draws, seed=component_seed + 1
            )
            summary["selection_adjusted"] = False
            summary["pvalue_site_scope"] = "only_fixed_site"
        else:
            integer_layers = {
                int(layer): values for layer, values in by_layer.items() if layer is not None
            }
            simultaneous = simultaneous_layer_intervals(
                integer_layers, draws=bootstrap_draws, seed=component_seed
            )
            selected = simultaneous[int(selected_layer)]
            summary = {
                "estimate": selected["estimate"],
                "ci95": selected["simultaneous_ci95"],
                "one_sided_lower_95": selected["simultaneous_ci95"][0],
                "cluster_count": selected["cluster_count"],
                "draws": selected["draws"],
                "selection_adjusted": True,
            }
            pvalue = centered_cluster_bootstrap_pvalue(
                integer_layers[int(selected_layer)],
                draws=randomization_draws,
                seed=component_seed + 1,
            )
            summary["pvalue_site_scope"] = "jointly_selected_same_site"
            curves[component] = {str(layer): value for layer, value in simultaneous.items()}
        summary["raw_one_sided_pvalue"] = pvalue
        component_results[component] = summary
        raw_component_pvalues[component] = pvalue

    selected_site_conjunctive_pvalue = max(raw_component_pvalues.values())
    layer_selection_candidate_count = len(layers_for_analysis)
    selection_adjusted_conjunctive_pvalue = min(
        1.0,
        selected_site_conjunctive_pvalue * layer_selection_candidate_count,
    )
    return {
        "model_key": model,
        "corruption": corruption,
        "stage_family": family,
        "selected_coarse_layer": selected_layer,
        "selected_state_name": (
            family if selected_layer is None else f"post_block_{selected_layer}"
        ),
        "layer_crosswalk_rule": (
            None
            if selected_layer is None
            else "Job B direct-hook capture state i+1 corresponds to Job C raw post_block_i"
        ),
        "bidirectional_components": component_results,
        "raw_component_pvalues": raw_component_pvalues,
        "selected_site_unadjusted_conjunctive_pvalue": (
            selected_site_conjunctive_pvalue
        ),
        "layer_selection_candidate_count": layer_selection_candidate_count,
        "layer_selection_adjustment": (
            "none_single_fixed_site"
            if fixed_stage
            else "bonferroni_over_frozen_coarse_sites"
        ),
        "raw_conjunctive_pvalue": selection_adjusted_conjunctive_pvalue,
        "conjunctive_rule": (
            "at the one jointly selected site, take the maximum of six fixed-site "
            "one-sided rescue, damage, unrelated-specificity, and random-specificity "
            "component p-values; for decoder families multiply that same-site "
            "intersection-union p-value by the number of frozen coarse sites before "
            f"the global Holm-{len(MODELS) * len(CORRUPTIONS) * len(PRIMARY_STAGE_FAMILIES)} correction"
        ),
        "bidirectional_intersection_score": layer_scores[selected_layer],
        "self_patch_failure_count": len(self_failures),
        "self_patch_failures": sorted(set(self_failures))[:20],
        "endpoint_trial_counts": endpoint_counts,
        "coarse_layer_curves": curves,
    }


def select_refinement_command(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    repo_dir = args.repo_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    mechanism_manifest = args.mechanism_manifest.expanduser().resolve()
    public_manifest = args.public_manifest.expanduser().resolve()
    hidden_index = args.hidden_index.expanduser().resolve()
    prespec = runner.verify_governing_prespec(repo_dir)
    coarse_gate_path = run_dir / "coarse_aggregate_summary.json"
    if not coarse_gate_path.exists():
        raise AnalysisError("coarse technical aggregate gate is missing")
    coarse_gate = read_json(coarse_gate_path)
    if coarse_gate.get("status") != "pass" or coarse_gate.get("mode") != "coarse_gate":
        raise AnalysisError("coarse technical aggregate gate is not pass")
    raw_rows = load_phase_rows(run_dir, "coarse")
    audit = validate_rows(raw_rows, phase="coarse")
    public_rows, public_receipt = validate_public_inputs_after_technical_gate(
        repo_dir=repo_dir,
        data_dir=data_dir,
        mechanism_manifest=mechanism_manifest,
        public_manifest=public_manifest,
    )
    raw_public_receipt = validate_raw_public_alignment(raw_rows, public_rows)
    rows, hidden_receipt = join_hidden_after_technical_gate(
        rows=raw_rows,
        public_rows=public_rows,
        data_dir=data_dir,
        mechanism_manifest=mechanism_manifest,
        public_manifest=public_manifest,
        hidden_index=hidden_index,
        expected_user_prompt=str(prespec["payload"]["generation"]["prompt_text"]),
    )
    models_payload: dict[str, Any] = {}
    for model in MODELS:
        model_payload: dict[str, Any] = {}
        layer_count = int(runner.MODEL_ADAPTERS[model]["expected_decoder_layers"])
        coarse = runner.coarse_layer_indexes(layer_count)
        for family in ("decoder_visual_token_residual", "decoder_final_question_token_residual"):
            scores_by_layer: dict[int, dict[str, float]] = defaultdict(dict)
            for corruption in CORRUPTIONS:
                for layer in coarse:
                    for direction, intervention in (
                        ("rescue", "matched_clean_source_rescue"),
                        ("damage", "matched_corrupt_source_damage"),
                    ):
                        selected = rows_for_endpoint(
                            rows,
                            model=model,
                            corruption=corruption,
                            family=family,
                            intervention=intervention,
                            layer=layer,
                        )
                        values = frame_cluster_values(selected)
                        scores_by_layer[layer][f"{corruption}:{direction}"] = mean(
                            list(values.values())
                        )
            intersection_scores = {
                layer: min(values.values())
                for layer, values in scores_by_layer.items()
            }
            peak = min(coarse, key=lambda layer: (-intersection_scores[layer], layer))
            candidates = {
                value
                for value in range(max(0, peak - 2), min(layer_count - 1, peak + 2) + 1)
                if value not in coarse
            }
            model_payload[family] = {
                "coarse_layers": coarse,
                "intersection_score_by_layer": {
                    str(layer): intersection_scores[layer] for layer in coarse
                },
                "selected_peak_layer": peak,
                "refinement_layers": sorted(candidates),
                "selection_rule": (
                    "maximize min(cluster-equal mean rescue and damage across blank_full and wrong_time); "
                    "tie to lower layer; add untested +/-2 indexes"
                ),
                "inferential_scope": "descriptive_only",
            }
        models_payload[model] = model_payload
    payload = {
        "kind": "phase2_reanchored_causal_patch_refinement_plan",
        "status": "pass",
        "governing_prespec_sha256": runner.PRESPEC_SHA256,
        "coarse_audit": audit,
        "coarse_technical_gate": coarse_gate,
        "public_input_gate": public_receipt,
        "raw_public_alignment_gate": raw_public_receipt,
        "hidden_join_gate": hidden_receipt,
        "models": models_payload,
        "claim_boundary": (
            "Refinement is descriptive because the peak-selection operation is not replayed "
            "inside every inferential resample. Confirmatory calls use coarse sites only."
        ),
    }
    out = args.out.expanduser().resolve()
    atomic_write_json(out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def validate_public_gate_command(args: argparse.Namespace) -> int:
    """Prove coarse row/public integrity before the hidden index is staged."""
    run_dir = args.run_dir.expanduser().resolve()
    repo_dir = args.repo_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    mechanism_manifest = args.mechanism_manifest.expanduser().resolve()
    public_manifest = args.public_manifest.expanduser().resolve()
    runner.verify_governing_prespec(repo_dir)
    coarse_gate_path = run_dir / "coarse_aggregate_summary.json"
    if not coarse_gate_path.is_file():
        raise AnalysisError("coarse technical aggregate gate is missing")
    coarse_gate = read_json(coarse_gate_path)
    if coarse_gate.get("status") != "pass" or coarse_gate.get("mode") != "coarse_gate":
        raise AnalysisError("coarse technical aggregate gate is not pass")
    raw_rows = load_phase_rows(run_dir, "coarse")
    row_gate = validate_rows(raw_rows, phase="coarse")
    support_gate = representable_intervention_gate(raw_rows)
    atomic_write_json(
        run_dir / "coarse_representable_intervention_gate.json", support_gate
    )
    if support_gate["status"] != "pass":
        raise AnalysisError(
            "coarse representable-intervention gate failed before hidden-label access: "
            f"unsupported_count={support_gate['unsupported_count']} "
            f"invalid_not_applicable_count={support_gate['invalid_not_applicable_count']}"
        )
    public_rows, public_gate = validate_public_inputs_after_technical_gate(
        repo_dir=repo_dir,
        data_dir=data_dir,
        mechanism_manifest=mechanism_manifest,
        public_manifest=public_manifest,
    )
    alignment_gate = validate_raw_public_alignment(raw_rows, public_rows)
    payload = {
        "kind": "phase2_reanchored_causal_patch_public_technical_gate",
        "status": "pass",
        "governing_prespec_sha256": runner.PRESPEC_SHA256,
        "coarse_aggregate_sha256": runner.sha256_file(coarse_gate_path),
        "coarse_row_gate": row_gate,
        "representable_intervention_gate": support_gate,
        "public_input_gate": public_gate,
        "raw_public_alignment_gate": alignment_gate,
        "hidden_index_staged": False,
        "hidden_index_opened": False,
    }
    out = args.out.expanduser().resolve()
    atomic_write_json(out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def load_job_a_promotion_gate(
    *,
    summary_path: Path | None,
    manifest_path: Path | None,
    expected_summary_sha256: str | None = None,
) -> dict[str, Any]:
    """Load the frozen behavioral eligibility result without crashing on absence."""
    if summary_path is None or manifest_path is None:
        return {
            "status": "absent",
            "reason": "Job A summary and manifest are both required for confirmatory calls",
            "models": {model: {"promotion_status": "unavailable"} for model in MODELS},
        }
    summary_path = summary_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if not summary_path.is_file() or not manifest_path.is_file():
        return {
            "status": "absent",
            "reason": "Job A summary or manifest path does not exist",
            "summary_path": str(summary_path),
            "manifest_path": str(manifest_path),
            "models": {model: {"promotion_status": "unavailable"} for model in MODELS},
        }
    summary_sha = runner.sha256_file(summary_path)
    try:
        summary = read_json(summary_path)
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "fail",
            "reason": f"Job A result is unreadable: {type(exc).__name__}: {exc}",
            "models": {model: {"promotion_status": "unavailable"} for model in MODELS},
        }
    reasons: list[str] = []
    if manifest.get("kind") != JOB_A_MANIFEST_KIND or manifest.get("status") != "pass":
        reasons.append("analysis_manifest_not_pass")
    if manifest.get("analysis_summary_sha256") != summary_sha:
        reasons.append("analysis_summary_manifest_hash_mismatch")
    if expected_summary_sha256 and expected_summary_sha256.lower() != summary_sha:
        reasons.append("analysis_summary_expected_hash_mismatch")
    if summary.get("kind") != JOB_A_SUMMARY_KIND or summary.get("status") != "pass":
        reasons.append("analysis_summary_not_pass")
    expected_eligibility_prespec = (
        BEHAVIORAL_ELIGIBILITY_PRESPEC_SHA256 or runner.PRESPEC_SHA256
    )
    if summary.get("preregistration_sha256") != expected_eligibility_prespec:
        reasons.append("analysis_summary_prespec_hash_mismatch")
    if int(summary.get("frame_count") or 0) != BEHAVIORAL_ELIGIBILITY_FRAME_COUNT:
        reasons.append(
            "analysis_summary_not_"
            f"{BEHAVIORAL_ELIGIBILITY_FRAME_COUNT}_frames"
        )
    model_rows = {
        str(row.get("model_key")): row
        for row in summary.get("models", [])
        if isinstance(row, Mapping)
    }
    models: dict[str, Any] = {}
    for model in MODELS:
        row = model_rows.get(model)
        if row is None:
            reasons.append(f"missing_model:{model}")
            models[model] = {"promotion_status": "unavailable"}
        else:
            models[model] = {
                "promotion_status": str(row.get("promotion_status") or "fail"),
                "real_capability_status": str(
                    row.get("real_capability", {}).get("status") or "fail"
                ),
                "primary_counterbalanced_pair_correctness_status": str(
                    row.get("primary_counterbalanced_pair_correctness", {}).get("status")
                    or "fail"
                ),
                "primary_holm_adjusted_p": row.get("primary_holm_adjusted_p"),
            }
    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "summary_path": str(summary_path),
        "summary_sha256": summary_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": runner.sha256_file(manifest_path),
        "governing_prespec_sha256": summary.get("preregistration_sha256"),
        "frame_count": summary.get("frame_count"),
        "models": models,
    }


def baseline_deficit_gate(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    corruption: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    selected = rows_for_endpoint(
        rows,
        model=model,
        corruption=corruption,
        family="vision_tower_final",
        intervention="matched_clean_source_rescue",
    )
    if len(selected) != runner.MECHANISM_TRIAL_COUNT:
        raise AnalysisError(
            "baseline deficit requires "
            f"{runner.MECHANISM_TRIAL_COUNT} rows for {model}:{corruption}"
        )
    values = frame_cluster_values(
        selected, value_field="clean_minus_corrupt_deficit_bits"
    )
    summary = bootstrap_ci(values, draws=draws, seed=seed)
    summary.update(
        {
            "model_key": model,
            "corruption": corruption,
            "outcome": "clean_minus_corrupt_correct_margin_bits",
            "status": "pass" if summary["one_sided_lower_95"] > 0.0 else "fail",
            "gate": "participant_cluster_bootstrap_one_sided_lower_95_above_zero",
        }
    )
    return summary


def analyze_command(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    repo_dir = args.repo_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    mechanism_manifest = args.mechanism_manifest.expanduser().resolve()
    public_manifest = args.public_manifest.expanduser().resolve()
    hidden_index = args.hidden_index.expanduser().resolve()
    prespec = runner.verify_governing_prespec(repo_dir)
    aggregate_path = run_dir / "aggregate_summary.json"
    if not aggregate_path.exists() or read_json(aggregate_path).get("status") != "pass":
        raise AnalysisError("fail-closed aggregate summary is missing or not pass")
    aggregate = read_json(aggregate_path)
    if aggregate.get("mode") != "full":
        raise AnalysisError("final aggregate is not a full-mode technical gate")
    coarse_raw = load_phase_rows(run_dir, "coarse")
    coarse_audit = validate_rows(coarse_raw, phase="coarse")
    refinement_plan_path = run_dir / "refinement_plan.json"
    refine_contract, refinement_plan_receipt = refinement_site_contract(
        refinement_plan_path
    )
    refine_raw = load_phase_rows(run_dir, "refine")
    refine_audit = validate_rows(
        refine_raw,
        phase="refine",
        expected_sites_by_model=refine_contract,
    )
    support_gate = representable_intervention_gate([*coarse_raw, *refine_raw])
    atomic_write_json(run_dir / "representable_intervention_gate.json", support_gate)
    if support_gate["status"] != "pass":
        raise AnalysisError(
            "representable-intervention gate failed before hidden-label access: "
            f"unsupported_count={support_gate['unsupported_count']} "
            f"invalid_not_applicable_count={support_gate['invalid_not_applicable_count']}"
        )
    public_rows, public_receipt = validate_public_inputs_after_technical_gate(
        repo_dir=repo_dir,
        data_dir=data_dir,
        mechanism_manifest=mechanism_manifest,
        public_manifest=public_manifest,
    )
    raw_public_receipt = validate_raw_public_alignment(
        [*coarse_raw, *refine_raw],
        public_rows,
    )
    joined, hidden_receipt = join_hidden_after_technical_gate(
        rows=[*coarse_raw, *refine_raw],
        public_rows=public_rows,
        data_dir=data_dir,
        mechanism_manifest=mechanism_manifest,
        public_manifest=public_manifest,
        hidden_index=hidden_index,
        expected_user_prompt=str(prespec["payload"]["generation"]["prompt_text"]),
    )
    coarse = [row for row in joined if row.get("phase") == "coarse"]
    refine = [row for row in joined if row.get("phase") == "refine"]
    job_a_gate = load_job_a_promotion_gate(
        summary_path=args.job_a_analysis_summary,
        manifest_path=args.job_a_analysis_manifest,
        expected_summary_sha256=args.job_a_analysis_summary_sha256,
    )
    baseline_gates: dict[str, dict[str, Any]] = {}
    for model_index, model in enumerate(MODELS):
        for corruption_index, corruption in enumerate(CORRUPTIONS):
            key = f"{model}|{corruption}"
            baseline_gates[key] = baseline_deficit_gate(
                coarse,
                model=model,
                corruption=corruption,
                draws=args.bootstrap_draws,
                seed=args.seed + 80_000 + model_index * 1000 + corruption_index * 100,
            )
    stages: list[dict[str, Any]] = []
    global_raw_pvalues: dict[str, float] = {}
    for model_index, model in enumerate(MODELS):
        for corruption_index, corruption in enumerate(CORRUPTIONS):
            for family_index, family in enumerate(STAGE_FAMILIES):
                result = _stage_analysis(
                    coarse,
                    model=model,
                    corruption=corruption,
                    family=family,
                    bootstrap_draws=args.bootstrap_draws,
                    randomization_draws=args.randomization_draws,
                    seed=args.seed + model_index * 10_000 + corruption_index * 1_000 + family_index * 100,
                )
                stages.append(result)
                if family in PRIMARY_STAGE_FAMILIES:
                    global_raw_pvalues[
                        f"{model}|{corruption}|{family}"
                    ] = float(result["raw_conjunctive_pvalue"])
    expected_global_family_size = (
        len(MODELS) * len(CORRUPTIONS) * len(PRIMARY_STAGE_FAMILIES)
    )
    if len(global_raw_pvalues) != expected_global_family_size:
        raise AnalysisError(
            "global confirmatory family must contain "
            f"{expected_global_family_size} configurations, "
            f"got {len(global_raw_pvalues)}"
        )
    global_adjusted = holm_adjust(global_raw_pvalues)
    for result in stages:
        model = str(result["model_key"])
        corruption = str(result["corruption"])
        family = str(result["stage_family"])
        key = f"{model}|{corruption}|{family}"
        primary_family = family in PRIMARY_STAGE_FAMILIES
        result["confirmatory_family_member"] = primary_family
        result["global_holm_adjusted_pvalue"] = (
            global_adjusted[key] if primary_family else None
        )
        component_lowers = {
            component: float(summary["one_sided_lower_95"])
            for component, summary in result["bidirectional_components"].items()
        }
        bidirectional_components_pass = all(
            value > 0.0 for value in component_lowers.values()
        )
        self_pass = int(result["self_patch_failure_count"]) == 0
        corrected = primary_family and global_adjusted[key] < ALPHA
        baseline_gate = baseline_gates[f"{model}|{corruption}"]
        baseline_pass = baseline_gate["status"] == "pass"
        model_job_a = job_a_gate.get("models", {}).get(model, {})
        job_a_pass = (
            job_a_gate.get("status") == "pass"
            and model_job_a.get("promotion_status") == "pass"
        )
        causal_call = all(
            [
                bidirectional_components_pass,
                self_pass,
                corrected,
                baseline_pass,
                job_a_pass,
                primary_family,
            ]
        )
        result["bidirectional_causal_bottleneck_call"] = causal_call
        result["call_gates"] = {
            "all_six_component_one_sided_lowers_above_zero": bidirectional_components_pass,
            "component_one_sided_lowers": component_lowers,
            "global_holm_adjusted_p_below_0.05": corrected,
            "complete_global_family_size": expected_global_family_size,
            "confirmatory_stage_family": primary_family,
            "clean_minus_corrupt_baseline_deficit_gate": baseline_gate,
            BEHAVIORAL_ELIGIBILITY_CALL_GATE_KEY: model_job_a,
            "job_a_gate_status": job_a_gate.get("status"),
            "self_patch_invariance": self_pass,
        }
        result["conclusion"] = (
            "bidirectional_causal_bottleneck_under_frozen_intervention"
            if causal_call
            else (
                "downstream_diagnostic_only"
                if not primary_family
                else "no_confirmatory_bidirectional_bottleneck_call"
            )
        )
    descriptive_refinement: dict[str, Any] = {"status": "not_run"}
    if refine:
        descriptive_refinement = {
            "status": "descriptive_only",
            "row_count": len(refine),
            "endpoints": [],
        }
        for model in MODELS:
            for corruption in CORRUPTIONS:
                for family in (
                    "decoder_visual_token_residual",
                    "decoder_final_question_token_residual",
                ):
                    for layer in _family_layers(refine, model, family):
                        for direction, intervention in (
                            ("rescue", "matched_clean_source_rescue"),
                            ("damage", "matched_corrupt_source_damage"),
                        ):
                            selected = rows_for_endpoint(
                                refine,
                                model=model,
                                corruption=corruption,
                                family=family,
                                intervention=intervention,
                                layer=layer,
                            )
                            values = frame_cluster_values(selected)
                            descriptive_refinement["endpoints"].append(
                                {
                                    "model_key": model,
                                    "corruption": corruption,
                                    "stage_family": family,
                                    "layer_index": layer,
                                    "state_name": f"post_block_{layer}",
                                    "direction": direction,
                                    **bootstrap_ci(
                                        values,
                                        draws=args.bootstrap_draws,
                                        seed=args.seed + layer + (0 if direction == "rescue" else 10_000),
                                    ),
                                    "inferential_scope": "descriptive_only",
                                }
                            )
    target_rows = [
        row
        for row in coarse
        if row.get("intervention") == "counterbalanced_content_aligned_crop_transfer"
        and row.get("status") == "pass"
    ]
    crop_order_diagnostics: list[dict[str, Any]] = []
    target_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        target_groups[(str(row["model_key"]), str(row["site_id"]))].append(row)
    for (model, site), selected in sorted(target_groups.items()):
        values = frame_cluster_values(selected)
        crop_order_diagnostics.append(
            {
                "model_key": model,
                "site_id": site,
                **bootstrap_ci(values, draws=args.bootstrap_draws, seed=args.seed),
                "interpretation": (
                    "Change under content-aligned counterbalanced crop transfer; full-frame slot "
                    "was unchanged. This is a descriptive crop-transfer diagnostic, not proof of scene binding."
                ),
            }
        )
    structural_transfer_groups: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in coarse:
        if (
            row.get("intervention")
            == "counterbalanced_content_aligned_crop_transfer"
            and row.get("status") == "not_applicable"
        ):
            structural_transfer_groups[
                (
                    str(row["model_key"]),
                    str(row["stage_family"]),
                    str(row["site_id"]),
                    str(row["reason"]),
                )
            ] += 1
    structural_crop_transfer_absences = [
        {
            "model_key": model,
            "stage_family": family,
            "site_id": site_id,
            "reason": reason,
            "row_count": count,
            "interpretation": (
                "Structurally not applicable under the frozen content-aligned transfer; "
                "this is not an estimated zero effect."
            ),
        }
        for (model, family, site_id, reason), count in sorted(
            structural_transfer_groups.items()
        )
    ]
    calls = [row for row in stages if row["bidirectional_causal_bottleneck_call"]]
    earliest: list[dict[str, Any]] = []
    if REPORT_EARLIEST_CAUSAL_SITE:
        for model in MODELS:
            for corruption in CORRUPTIONS:
                candidates = [
                    row
                    for row in calls
                    if row["model_key"] == model and row["corruption"] == corruption
                ]
                if not candidates:
                    continue

                earliest_depth = min(causal_stage_depth(row) for row in candidates)
                tied = sorted(
                    (
                        row
                        for row in candidates
                        if causal_stage_depth(row) == earliest_depth
                    ),
                    key=lambda row: str(row["stage_family"]),
                )
                for selected in tied:
                    earliest.append(
                        {
                            "model_key": model,
                            "corruption": corruption,
                            "stage_family": selected["stage_family"],
                            "selected_coarse_layer": selected["selected_coarse_layer"],
                            "selected_state_name": selected["selected_state_name"],
                            "computation_depth": earliest_depth,
                            "tied_earliest_site_count": len(tied),
                            "claim": "earliest_bidirectional_causal_bottleneck_under_frozen_intervention",
                        }
                    )
    payload = {
        "kind": ANALYSIS_KIND,
        "status": "pass",
        "governing_prespec_sha256": runner.PRESPEC_SHA256,
        "full_technical_gate": aggregate,
        "public_input_gate": public_receipt,
        "raw_public_alignment_gate": raw_public_receipt,
        "hidden_join_gate": hidden_receipt,
        BEHAVIORAL_ELIGIBILITY_RESULT_KEY: job_a_gate,
        "mechanism_subset_baseline_deficit_gates": baseline_gates,
        "coarse_audit": coarse_audit,
        "refinement_audit": refine_audit,
        "representable_intervention_gate": support_gate,
        "refinement_plan_gate": refinement_plan_receipt,
        "confirmatory_coarse_stage_results": stages,
        "descriptive_refinement": descriptive_refinement,
        "counterbalanced_crop_order_diagnostics": crop_order_diagnostics,
        "counterbalanced_crop_transfer_structural_absences": (
            structural_crop_transfer_absences
        ),
        "bidirectional_causal_bottleneck_count": len(calls),
        "bidirectional_causal_bottlenecks": [
            {
                "model_key": row["model_key"],
                "corruption": row["corruption"],
                "stage_family": row["stage_family"],
                "selected_coarse_layer": row["selected_coarse_layer"],
            }
            for row in calls
        ],
        "earliest_bidirectional_causal_bottleneck_under_frozen_intervention": earliest,
        "global_confirmatory_holm_family": {
            "size": expected_global_family_size,
            "stage_families": list(PRIMARY_STAGE_FAMILIES),
            "raw_conjunctive_pvalues": global_raw_pvalues,
            "holm_adjusted_pvalues": global_adjusted,
            "unit": "model x corruption x coarse stage family",
        },
        "layer_semantics": {
            "lm_entry": "Job B direct-hook capture state 0; not represented as a Job C post-block patch",
            "post_block_rule": "Job B direct-hook capture state i+1 corresponds to Job C raw post_block_i",
            "models": {model: runner.layer_crosswalk(model) for model in MODELS},
        },
        "claim_boundary": ANALYSIS_CLAIM_BOUNDARY,
        "bootstrap_draws": args.bootstrap_draws,
        "randomization_draws": args.randomization_draws,
        "seed": args.seed,
    }
    out = args.out.expanduser().resolve()
    atomic_write_json(out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_public_inputs(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--data-dir", type=Path, required=True)
        subparser.add_argument("--mechanism-manifest", type=Path, required=True)
        subparser.add_argument("--public-manifest", type=Path, required=True)

    def add_frozen_inputs(subparser: argparse.ArgumentParser) -> None:
        add_public_inputs(subparser)
        subparser.add_argument("--hidden-index", type=Path, required=True)

    public_gate = subparsers.add_parser("validate-public-gate")
    public_gate.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    public_gate.add_argument("--run-dir", type=Path, required=True)
    public_gate.add_argument("--out", type=Path, required=True)
    add_public_inputs(public_gate)
    public_gate.set_defaults(func=validate_public_gate_command)

    refine = subparsers.add_parser("select-refinement")
    refine.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    refine.add_argument("--run-dir", type=Path, required=True)
    refine.add_argument("--out", type=Path, required=True)
    add_frozen_inputs(refine)
    refine.set_defaults(func=select_refinement_command)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    analyze.add_argument("--run-dir", type=Path, required=True)
    analyze.add_argument("--out", type=Path, required=True)
    add_frozen_inputs(analyze)
    analyze.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    analyze.add_argument("--randomization-draws", type=int, default=DEFAULT_RANDOMIZATION_DRAWS)
    analyze.add_argument("--job-a-analysis-summary", type=Path)
    analyze.add_argument("--job-a-analysis-manifest", type=Path)
    analyze.add_argument("--job-a-analysis-summary-sha256")
    analyze.add_argument("--seed", type=int, default=DEFAULT_SEED)
    analyze.set_defaults(func=analyze_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except (AnalysisError, runner.GateError, runner.UnsupportedSite) as exc:
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
