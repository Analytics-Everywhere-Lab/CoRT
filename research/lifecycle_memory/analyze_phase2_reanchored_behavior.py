#!/usr/bin/env python3
"""Hidden-label full analysis for the re-anchored EgoHOS 2AFC battery."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory import run_phase2_reanchored_behavior as runtime


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(float(value) for value in values) / len(values)


def quantile(values: Sequence[float], proportion: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * proportion
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_equal_mean(values_by_cluster: dict[str, Sequence[float]]) -> float:
    if not values_by_cluster:
        raise ValueError("cluster statistic requires at least one cluster")
    return mean([mean(values) for values in values_by_cluster.values()])


def cluster_bootstrap_ci(
    values_by_cluster: dict[str, Sequence[float]],
    *,
    draws: int = 10000,
    seed: int = 20260721,
) -> dict[str, Any]:
    clusters = sorted(values_by_cluster)
    if not clusters or draws <= 0:
        raise ValueError("invalid cluster bootstrap request")
    cluster_means = {cluster: mean(values_by_cluster[cluster]) for cluster in clusters}
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [rng.choice(clusters) for _ in clusters]
        samples.append(mean([cluster_means[cluster] for cluster in selected]))
    return {
        "estimate": mean(list(cluster_means.values())),
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "one_sided_lower_95": quantile(samples, 0.05),
        "draws": draws,
        "seed": seed,
        "cluster_count": len(clusters),
    }


def shared_cluster_bootstrap_intersection(
    blank_differences_by_cluster: dict[str, Sequence[float]],
    wrong_differences_by_cluster: dict[str, Sequence[float]],
    *,
    draws: int = 10000,
    seed: int = 20260721,
) -> dict[str, Any]:
    """Estimate both smooth components on identical participant resamples.

    The minimum is reported as an effect-size summary, but confirmatory
    inference uses the two component lower bounds as an intersection-union
    test. A percentile interval for the non-smooth minimum is deliberately not
    used for promotion.
    """
    clusters = sorted(set(blank_differences_by_cluster) & set(wrong_differences_by_cluster))
    if not clusters or set(clusters) != set(blank_differences_by_cluster) or set(clusters) != set(
        wrong_differences_by_cluster
    ):
        raise ValueError("intersection bootstrap requires identical nonempty cluster sets")
    blank_means = {key: mean(blank_differences_by_cluster[key]) for key in clusters}
    wrong_means = {key: mean(wrong_differences_by_cluster[key]) for key in clusters}
    blank_point = mean(list(blank_means.values()))
    wrong_point = mean(list(wrong_means.values()))
    rng = random.Random(seed)
    blank_draws: list[float] = []
    wrong_draws: list[float] = []
    for _ in range(draws):
        selected = [rng.choice(clusters) for _ in clusters]
        blank_value = mean([blank_means[key] for key in selected])
        wrong_value = mean([wrong_means[key] for key in selected])
        blank_draws.append(blank_value)
        wrong_draws.append(wrong_value)
    blank_lower = quantile(blank_draws, 0.05)
    wrong_lower = quantile(wrong_draws, 0.05)
    return {
        "estimate": min(blank_point, wrong_point),
        "real_minus_blank_full": blank_point,
        "real_minus_wrong_time": wrong_point,
        "component_ci95": {
            "real_minus_blank_full": [quantile(blank_draws, 0.025), quantile(blank_draws, 0.975)],
            "real_minus_wrong_time": [quantile(wrong_draws, 0.025), quantile(wrong_draws, 0.975)],
        },
        "component_one_sided_lower_95": {
            "real_minus_blank_full": blank_lower,
            "real_minus_wrong_time": wrong_lower,
        },
        "intersection_union_lower_gate": min(blank_lower, wrong_lower),
        "draws": draws,
        "seed": seed,
        "cluster_count": len(clusters),
        "joint_draw_rule": "both smooth components recomputed on every shared participant-cluster draw",
        "inference_rule": (
            "intersection-union: both one-sided component lower bounds must exceed zero; "
            "no direct percentile interval for the non-smooth minimum is used"
        ),
    }


def centered_cluster_bootstrap_p_value(
    values_by_cluster: dict[str, Sequence[float]],
    *,
    null_value: float = 0.0,
    draws: int = 10000,
    seed: int = 20260721,
) -> float:
    """One-sided participant-cluster bootstrap test of mean > null_value."""
    clusters = sorted(values_by_cluster)
    if not clusters or draws <= 0:
        raise ValueError("invalid centered cluster bootstrap request")
    values = [mean(values_by_cluster[cluster]) for cluster in clusters]
    observed_effect = mean(values) - float(null_value)
    centered = [value - observed_effect for value in values]
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        statistic = mean([rng.choice(centered) for _ in centered]) - float(null_value)
        extreme += statistic >= observed_effect - 1e-15
    return (extreme + 1) / (draws + 1)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda row: (row[1], row[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * float(value)))
        adjusted[key] = running
    return adjusted


def _hidden_rows(path: Path) -> list[dict[str, Any]]:
    payload = runtime.read_json(path)
    if isinstance(payload, dict):
        rows = payload.get("answers") or payload.get("rows") or payload.get("trials")
    else:
        rows = payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("hidden evaluator index must contain a row list")
    return rows


def _required_text(row: dict[str, Any], names: Sequence[str], label: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"hidden evaluator row lacks {label}")


def _normalize_hidden(path: Path, selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _hidden_rows(path)
    normalized: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = _required_text(row, ("item_id", "trial_id"), "item ID")
        if item_id in normalized:
            raise ValueError(f"duplicate hidden item ID: {item_id}")
        answer = row.get("answer", row.get("correct_choice", row.get("expected_answer")))
        if answer not in runtime.CHOICES:
            raise ValueError(f"hidden answer is not A/B for {item_id}")
        participant = _required_text(
            row,
            ("ego4d_participant_id", "participant_id", "source_participant_id"),
            "Ego4D participant ID",
        )
        frame_unit = _required_text(
            row,
            (
                "frame_level_unit_id",
                "frame_unit_id",
                "counterbalance_group_id",
                "frame_pair_id",
            ),
            "frame unit ID",
        )
        order_index = row.get("counterbalance_order_index")
        if type(order_index) is not int or order_index not in {0, 1}:
            raise ValueError(f"hidden counterbalance order is not 0/1: {item_id}")
        first_order_class = row.get("first_order_class")
        second_order_class = row.get("second_order_class")
        if first_order_class not in {3, 4, 5} or second_order_class not in {6, 7, 8}:
            raise ValueError(f"hidden ontology class pair is invalid: {item_id}")
        if row.get("scoring_eligible") is False:
            raise ValueError(f"hidden row is not scoring eligible: {item_id}")
        normalized[item_id] = {
            "item_id": item_id,
            "correct_choice": str(answer),
            "participant_id": participant,
            "frame_unit_id": frame_unit,
            "counterbalance_order_index": int(order_index),
            "first_order_class": int(first_order_class),
            "second_order_class": int(second_order_class),
            "request_ids": row.get("request_ids") if isinstance(row.get("request_ids"), dict) else None,
        }
    selected_ids = [str(row["item_id"]) for row in selection["selection"]]
    if set(normalized) != set(selected_ids):
        raise ValueError("hidden evaluator and frozen selection item sets differ")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized.values():
        groups[row["frame_unit_id"]].append(row)
    if len(groups) != int(selection["frame_count"]):
        raise ValueError("hidden frame-unit count drift")
    for frame_unit, group in groups.items():
        if len(group) != 2:
            raise ValueError(f"hidden counterbalance group is not a pair: {frame_unit}")
        if {row["correct_choice"] for row in group} != set(runtime.CHOICES):
            raise ValueError(f"hidden counterbalanced answers do not flip A/B: {frame_unit}")
        if len({row["participant_id"] for row in group}) != 1:
            raise ValueError(f"participant drift inside frame pair: {frame_unit}")
        if {row["counterbalance_order_index"] for row in group} != {0, 1}:
            raise ValueError(f"hidden counterbalance orders do not form 0/1: {frame_unit}")
        if len(
            {
                (row["first_order_class"], row["second_order_class"])
                for row in group
            }
        ) != 1:
            raise ValueError(f"ontology class drift inside frame pair: {frame_unit}")
    public_pairs: dict[int, list[str]] = defaultdict(list)
    for selection_row in selection["selection"]:
        public_pairs[int(selection_row["counterbalance_pair_index"])].append(
            str(selection_row["item_id"])
        )
    if len(public_pairs) != int(selection["frame_count"]):
        raise ValueError("derived public pair count drift")
    for pair_index, item_ids in public_pairs.items():
        if len(item_ids) != 2:
            raise ValueError(f"derived public pair is not a pair: {pair_index}")
        if len({normalized[item_id]["frame_unit_id"] for item_id in item_ids}) != 1:
            raise ValueError(f"derived public pair crosses hidden frame units: {pair_index}")
    return normalized


def _load_model_rows(
    *,
    run_dir: Path,
    model_row: dict[str, Any],
    selection: dict[str, Any],
    contract_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    rows_path = run_dir / "models" / model_row["model_key"] / "rows.jsonl"
    contract, prereg = runtime._load_contract(contract_path)
    base_expected = {
        "model_key": model_row["model_key"],
        "model_id": model_row["model_id"],
        "revision": model_row["revision"],
        "dtype": "bf16",
        "selection_manifest_sha256": runtime.sha256_file(run_dir / "selection_manifest.json"),
        "preregistration_sha256": runtime._contract_preregistration_sha256(contract),
        "code_snapshot_sha256": selection["code_snapshot_sha256"],
        "candidate_A_text": prereg["candidate_completions"]["A"],
        "candidate_B_text": prereg["candidate_completions"]["B"],
        "candidate_shared_prefix": prereg.get("candidate_shared_prefix"),
    }
    assignment_map = {
        str(assignment["assignment_id"]): assignment
        for rank_row in selection.get("batch_profile", {}).get("schedule", {}).get("ranks", [])
        for assignment in rank_row.get("assignments", [])
        if assignment.get("model_key") == model_row["model_key"]
    }
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in runtime.iter_jsonl(rows_path):
        assignment = assignment_map.get(str(row.get("assignment_id") or ""))
        if assignment is None:
            raise ValueError(f"unbound worker shard for {model_row['model_key']}")
        expected_common = {
            **base_expected,
            "worker_index": int(assignment["rank"]),
            "assignment_id": str(assignment["assignment_id"]),
            "selection_start": int(assignment["selection_start"]),
            "selection_stop": int(assignment["selection_stop"]),
        }
        issues = runtime.validate_completed_row(row, expected_common=expected_common)
        if issues:
            raise ValueError(f"invalid worker row for {model_row['model_key']}: {issues}")
        key = (str(row["item_id"]), str(row["arm"]))
        if key in rows:
            raise ValueError(f"duplicate worker row: {model_row['model_key']}:{key}")
        rows[key] = row
    expected = {
        (str(selection_row["item_id"]), arm)
        for selection_row in selection["selection"]
        for arm in runtime.ARMS
    }
    if set(rows) != expected:
        raise ValueError(f"worker row union mismatch for {model_row['model_key']}")
    return rows


def _joined_trials(
    *, rows: dict[tuple[str, str], dict[str, Any]], hidden: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for (item_id, arm), row in sorted(rows.items()):
        label = hidden[item_id]
        request_ids = label.get("request_ids")
        if request_ids is not None and request_ids.get(arm) not in {None, row["request_id"]}:
            raise ValueError(f"hidden/public request join mismatch: {item_id}/{arm}")
        parsed = row["greedy"]["strict_choice"]
        abstain = bool(parsed.get("abstain")) if parsed.get("valid") else False
        predicted = (
            parsed.get("answer") if parsed.get("valid") and not abstain else None
        )
        correct = bool(parsed.get("valid") and not abstain and predicted == label["correct_choice"])
        candidates = row["teacher_forced"]["candidate_rows"]
        raw_margin = (float(candidates["A"]["sum_logprob"]) - float(candidates["B"]["sum_logprob"])) / math.log(2)
        correct_margin = raw_margin if label["correct_choice"] == "A" else -raw_margin
        if not math.isfinite(correct_margin):
            raise ValueError("nonfinite correct-choice margin")
        chosen_identity = None
        if predicted in runtime.CHOICES:
            chosen_identity = (
                "first_order_interacting_object"
                if predicted == label["correct_choice"]
                else "second_order_interacting_object"
            )
        joined.append(
            {
                "item_id": item_id,
                "frame_unit_id": label["frame_unit_id"],
                "participant_id": label["participant_id"],
                "arm": arm,
                "first_order_class": label["first_order_class"],
                "second_order_class": label["second_order_class"],
                "correct_choice": label["correct_choice"],
                "predicted_choice": predicted,
                "chosen_annotation_identity": chosen_identity,
                "strict_parse_valid": bool(parsed.get("valid")),
                "abstain": abstain,
                "correct": correct,
                "raw_A_minus_B_margin_bits": raw_margin,
                "correct_choice_margin_bits": correct_margin,
            }
        )
    return joined


def _collapse_frames(joined: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        groups[(str(row["frame_unit_id"]), str(row["arm"]))].append(row)
    collapsed: list[dict[str, Any]] = []
    for (frame_unit, arm), rows in sorted(groups.items()):
        if len(rows) != 2 or {row["correct_choice"] for row in rows} != set(runtime.CHOICES):
            raise ValueError(f"invalid frame/arm counterbalance pair: {frame_unit}/{arm}")
        participants = {str(row["participant_id"]) for row in rows}
        if len(participants) != 1:
            raise ValueError("participant drift during frame collapse")
        class_pairs = {
            (int(row["first_order_class"]), int(row["second_order_class"]))
            for row in rows
        }
        if len(class_pairs) != 1:
            raise ValueError("ontology class drift during frame collapse")
        first_order_class, second_order_class = next(iter(class_pairs))
        predicted = [row["predicted_choice"] for row in rows]
        identities = [row["chosen_annotation_identity"] for row in rows]
        collapsed.append(
            {
                "frame_unit_id": frame_unit,
                "participant_id": next(iter(participants)),
                "arm": arm,
                "first_order_class": first_order_class,
                "second_order_class": second_order_class,
                "mean_trial_accuracy": mean([float(row["correct"]) for row in rows]),
                "both_orders_correct": all(bool(row["correct"]) for row in rows),
                "order_consistent_identity": identities[0] is not None and identities[0] == identities[1],
                "raw_A_B_prediction_flip": None not in predicted and predicted[0] != predicted[1],
                "mean_correct_choice_margin_bits": mean(
                    [float(row["correct_choice_margin_bits"]) for row in rows]
                ),
                "parse_failure_count": sum(not bool(row["strict_parse_valid"]) for row in rows),
            }
        )
    return collapsed


def _descriptive_class_strata(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    frame_map = {
        (str(row["frame_unit_id"]), str(row["arm"])): row for row in frames
    }
    frame_ids = sorted({str(row["frame_unit_id"]) for row in frames})
    complete: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        rows = {arm: frame_map[(frame_id, arm)] for arm in runtime.ARMS}
        class_pairs = {
            (int(row["first_order_class"]), int(row["second_order_class"]))
            for row in rows.values()
        }
        if len(class_pairs) != 1:
            raise ValueError("ontology class drift across evidence arms")
        first_class, second_class = next(iter(class_pairs))
        real = rows["real"]
        complete.append(
            {
                "participant_id": str(real["participant_id"]),
                "first_order_class": first_class,
                "second_order_class": second_class,
                "real_accuracy": float(real["mean_trial_accuracy"]),
                "real_minus_blank_full": float(real["mean_trial_accuracy"])
                - float(rows["blank_full"]["mean_trial_accuracy"]),
                "real_minus_wrong_time": float(real["mean_trial_accuracy"])
                - float(rows["wrong_time"]["mean_trial_accuracy"]),
                "real_pair_correct": float(real["both_orders_correct"]),
            }
        )

    def summarize(key_fn: Any) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in complete:
            grouped[str(key_fn(row))].append(row)
        summaries = []
        for stratum, rows in sorted(grouped.items()):
            summaries.append(
                {
                    "stratum": stratum,
                    "frame_n": len(rows),
                    "trial_n_per_arm": 2 * len(rows),
                    "participant_n": len({row["participant_id"] for row in rows}),
                    "real_accuracy": mean([row["real_accuracy"] for row in rows]),
                    "real_pair_correct_rate": mean(
                        [row["real_pair_correct"] for row in rows]
                    ),
                    "real_minus_blank_full": mean(
                        [row["real_minus_blank_full"] for row in rows]
                    ),
                    "real_minus_wrong_time": mean(
                        [row["real_minus_wrong_time"] for row in rows]
                    ),
                    "intersection_evidence_gap": min(
                        mean([row["real_minus_blank_full"] for row in rows]),
                        mean([row["real_minus_wrong_time"] for row in rows]),
                    ),
                }
            )
        return summaries

    by_pair = summarize(
        lambda row: f"{row['first_order_class']}x{row['second_order_class']}"
    )
    pair_participant_equal: dict[str, dict[str, float]] = {}
    for pair in sorted(
        {
            (row["first_order_class"], row["second_order_class"])
            for row in complete
        }
    ):
        pair_rows = [
            row
            for row in complete
            if (row["first_order_class"], row["second_order_class"]) == pair
        ]
        by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pair_rows:
            by_participant[row["participant_id"]].append(row)
        pair_key = f"{pair[0]}x{pair[1]}"
        pair_participant_equal[pair_key] = {
            metric: mean(
                [
                    mean([row[metric] for row in participant_rows])
                    for participant_rows in by_participant.values()
                ]
            )
            for metric in (
                "real_accuracy",
                "real_minus_blank_full",
                "real_minus_wrong_time",
            )
        }
    common_pairs = sorted(pair_participant_equal)
    standardized_blank = mean(
        [pair_participant_equal[pair]["real_minus_blank_full"] for pair in common_pairs]
    )
    standardized_wrong = mean(
        [pair_participant_equal[pair]["real_minus_wrong_time"] for pair in common_pairs]
    )
    return {
        "design_balance": "structurally_imbalanced_not_a_balanced_3x3_design",
        "first_order_class": summarize(lambda row: row["first_order_class"]),
        "second_order_class": summarize(lambda row: row["second_order_class"]),
        "class_pair": by_pair,
        "common_support_standardized_sensitivity": {
            "status": "descriptive",
            "observed_class_pair_strata": common_pairs,
            "stratum_count": len(common_pairs),
            "frame_n": len(complete),
            "weighting": (
                "equal weight across observed class-pair strata after participant-equal "
                "aggregation within each stratum"
            ),
            "real_accuracy": mean(
                [pair_participant_equal[pair]["real_accuracy"] for pair in common_pairs]
            ),
            "real_minus_blank_full": standardized_blank,
            "real_minus_wrong_time": standardized_wrong,
            "intersection_evidence_gap": min(standardized_blank, standardized_wrong),
            "inferential_status": "sensitivity_only_no_p_value_or_ci",
        },
    }


def _model_metrics(
    *,
    model_key: str,
    joined: Sequence[dict[str, Any]],
    frames: Sequence[dict[str, Any]],
    bootstrap_draws: int,
    bootstrap_seed: int,
    bootstrap_test_draws: int,
) -> dict[str, Any]:
    frame_map = {
        (str(row["frame_unit_id"]), str(row["arm"])): row for row in frames
    }
    frame_ids = sorted({str(row["frame_unit_id"]) for row in frames})
    if any((frame_id, arm) not in frame_map for frame_id in frame_ids for arm in runtime.ARMS):
        raise ValueError("frame-arm matrix is incomplete")
    real_by_cluster: dict[str, list[float]] = defaultdict(list)
    blank_diff_by_cluster: dict[str, list[float]] = defaultdict(list)
    wrong_diff_by_cluster: dict[str, list[float]] = defaultdict(list)
    pair_correct_by_cluster: dict[str, list[float]] = defaultdict(list)
    for frame_id in frame_ids:
        real = frame_map[(frame_id, "real")]
        blank = frame_map[(frame_id, "blank_full")]
        wrong = frame_map[(frame_id, "wrong_time")]
        participant = str(real["participant_id"])
        if {blank["participant_id"], wrong["participant_id"]} != {participant}:
            raise ValueError("participant drift across evidence arms")
        real_by_cluster[participant].append(float(real["mean_trial_accuracy"]))
        blank_diff_by_cluster[participant].append(
            float(real["mean_trial_accuracy"]) - float(blank["mean_trial_accuracy"])
        )
        wrong_diff_by_cluster[participant].append(
            float(real["mean_trial_accuracy"]) - float(wrong["mean_trial_accuracy"])
        )
        pair_correct_by_cluster[participant].append(float(real["both_orders_correct"]))
    capability = cluster_bootstrap_ci(
        real_by_cluster, draws=bootstrap_draws, seed=bootstrap_seed
    )
    intersection = shared_cluster_bootstrap_intersection(
        blank_diff_by_cluster,
        wrong_diff_by_cluster,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    pair_correctness = cluster_bootstrap_ci(
        pair_correct_by_cluster, draws=bootstrap_draws, seed=bootstrap_seed + 7
    )
    p_blank = centered_cluster_bootstrap_p_value(
        blank_diff_by_cluster,
        null_value=0.0,
        draws=bootstrap_test_draws,
        seed=bootstrap_seed + 11,
    )
    p_wrong = centered_cluster_bootstrap_p_value(
        wrong_diff_by_cluster,
        null_value=0.0,
        draws=bootstrap_test_draws,
        seed=bootstrap_seed + 29,
    )
    p_pair_correctness = centered_cluster_bootstrap_p_value(
        pair_correct_by_cluster,
        null_value=0.25,
        draws=bootstrap_test_draws,
        seed=bootstrap_seed + 43,
    )
    gate = bool(capability["estimate"] >= 0.65 and capability["ci95"][0] > 0.50)
    arm_rows: dict[str, Any] = {}
    for arm in runtime.ARMS:
        arm_trials = [row for row in joined if row["arm"] == arm]
        arm_frames = [row for row in frames if row["arm"] == arm]
        by_position = {
            choice: [row for row in arm_trials if row["correct_choice"] == choice]
            for choice in runtime.CHOICES
        }
        arm_rows[arm] = {
            "trial_count": len(arm_trials),
            "eligible_trial_count": len(arm_trials),
            "ineligible_trial_count": 0,
            "frame_count": len(arm_frames),
            "strict_greedy_trial_accuracy": mean([float(row["correct"]) for row in arm_trials]),
            "frame_collapsed_mean_trial_accuracy": mean(
                [float(row["mean_trial_accuracy"]) for row in arm_frames]
            ),
            "both_counterbalanced_orders_correct_rate": mean(
                [float(row["both_orders_correct"]) for row in arm_frames]
            ),
            "order_consistency_rate": mean(
                [float(row["order_consistent_identity"]) for row in arm_frames]
            ),
            "raw_A_B_prediction_flip_rate": mean(
                [float(row["raw_A_B_prediction_flip"]) for row in arm_frames]
            ),
            "parse_failure_count": sum(not bool(row["strict_parse_valid"]) for row in arm_trials),
            "parse_failure_rate": mean(
                [float(not bool(row["strict_parse_valid"])) for row in arm_trials]
            ),
            "abstention_count": sum(bool(row["abstain"]) for row in arm_trials),
            "mean_correct_choice_margin_bits": mean(
                [float(row["correct_choice_margin_bits"]) for row in arm_trials]
            ),
            "accuracy_by_correct_answer_position": {
                choice: mean([float(row["correct"]) for row in by_position[choice]])
                for choice in runtime.CHOICES
            },
            "selection_rate_by_annotation_identity": {
                "first_order_interacting_object": mean(
                    [
                        float(row["chosen_annotation_identity"] == "first_order_interacting_object")
                        for row in arm_trials
                    ]
                ),
                "second_order_interacting_object": mean(
                    [
                        float(row["chosen_annotation_identity"] == "second_order_interacting_object")
                        for row in arm_trials
                    ]
                ),
                "unparsed_or_abstained": mean(
                    [float(row["chosen_annotation_identity"] is None) for row in arm_trials]
                ),
            },
            "counterbalanced_pair_correctness_on_identical_frames": {
                "pair_correct_rate": mean(
                    [float(row["both_orders_correct"]) for row in arm_frames]
                ),
                "order_consistency_rate": mean(
                    [float(row["order_consistent_identity"]) for row in arm_frames]
                ),
            },
        }
    return {
        "model_key": model_key,
        "status": "pass",
        "frame_count": len(frame_ids),
        "participant_cluster_count": len(real_by_cluster),
        "arms": arm_rows,
        "real_capability": {
            **capability,
            "minimum_accuracy": 0.65,
            "minimum_ci_lower": 0.50,
            "strict_ci_operator": ">",
            "status": "pass" if gate else "fail",
        },
        "primary_intersection_evidence_effect": intersection,
        "primary_counterbalanced_pair_correctness": {
            **pair_correctness,
            "outcome": "real_arm_both_counterbalanced_orders_correct_rate",
            "paired_chance_reference": 0.25,
            "shortcut_defeated": "fixed_A_or_B_position_preference",
            "claim_boundary": (
                "This gate defeats a fixed A/B position shortcut but does not prove "
                "scene-target binding; crop appearance may reveal ontology role."
            ),
            "status": "pass" if pair_correctness["one_sided_lower_95"] > 0.25 else "fail",
        },
        "descriptive_ontology_strata": _descriptive_class_strata(frames),
        "confirmatory_centered_cluster_bootstrap": {
            "real_minus_blank_full_p": p_blank,
            "real_minus_wrong_time_p": p_wrong,
            "counterbalanced_pair_correctness_above_paired_chance_p": p_pair_correctness,
            "model_conjunctive_p": max(p_blank, p_wrong, p_pair_correctness),
            "rule": (
                "maximum one-sided component p-value for the conjunction of both "
                "evidence contrasts and real-arm counterbalanced pair correctness"
            ),
            "draws": bootstrap_test_draws,
            "resampling_unit": "participant cluster",
            "null_construction": "participant means centered at each component null",
        },
    }


def _anchor_contrast(
    model_frames: dict[str, Sequence[dict[str, Any]]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    qwen = "qwen35_27b_hf_bf16"
    gemma = "gemma4_31b_hf_bf16"
    if qwen not in model_frames or gemma not in model_frames:
        return {"status": "unsupported", "reason": "anchor model missing"}

    def component_clusters(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
        frame_map = {(row["frame_unit_id"], row["arm"]): row for row in rows}
        frame_ids = sorted({row["frame_unit_id"] for row in rows})
        blank: dict[str, list[float]] = defaultdict(list)
        wrong: dict[str, list[float]] = defaultdict(list)
        for frame_id in frame_ids:
            real = frame_map[(frame_id, "real")]
            participant = real["participant_id"]
            blank[participant].append(
                real["mean_trial_accuracy"] - frame_map[(frame_id, "blank_full")]["mean_trial_accuracy"]
            )
            wrong[participant].append(
                real["mean_trial_accuracy"] - frame_map[(frame_id, "wrong_time")]["mean_trial_accuracy"]
            )
        return (
            {key: mean(values) for key, values in blank.items()},
            {key: mean(values) for key, values in wrong.items()},
        )

    q_blank, q_wrong = component_clusters(model_frames[qwen])
    g_blank, g_wrong = component_clusters(model_frames[gemma])
    clusters = sorted(set(q_blank) & set(q_wrong) & set(g_blank) & set(g_wrong))
    if not clusters:
        raise ValueError("anchor contrast has no shared participant clusters")
    point = min(mean([q_blank[key] for key in clusters]), mean([q_wrong[key] for key in clusters])) - min(
        mean([g_blank[key] for key in clusters]), mean([g_wrong[key] for key in clusters])
    )
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws):
        selected = [rng.choice(clusters) for _ in clusters]
        q_effect = min(
            mean([q_blank[key] for key in selected]), mean([q_wrong[key] for key in selected])
        )
        g_effect = min(
            mean([g_blank[key] for key in selected]), mean([g_wrong[key] for key in selected])
        )
        samples.append(q_effect - g_effect)
    return {
        "status": "pass",
        "estimand": "Qwen3.5-27B minus Gemma4-31B intersection evidence effect",
        "estimate": point,
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "draws": draws,
        "seed": seed,
        "shared_participant_cluster_count": len(clusters),
    }


def analyze(
    *,
    data_dir: Path,
    run_dir: Path,
    model_plan_path: Path,
    contract_path: Path,
    bootstrap_draws: int | None = None,
    bootstrap_test_draws: int = 10000,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    selection_path = run_dir / "selection_manifest.json"
    technical_path = run_dir / "technical_verify.json"
    if not technical_path.is_file() or runtime.read_json(technical_path).get("status") != "pass":
        raise ValueError("full analysis requires a passing technical verification")
    selection, prereg, model_plan = runtime._load_selection(
        data_dir=data_dir,
        selection_path=selection_path,
        model_plan_path=model_plan_path.expanduser().resolve(),
        contract_path=contract_path.expanduser().resolve(),
    )
    if selection.get("mode") != "full":
        raise ValueError("hidden-label analysis is forbidden outside full mode")
    contract, _ = runtime._load_contract(contract_path)
    hidden_path = data_dir / str(contract["battery"]["hidden_evaluator_relative_path"])
    if runtime.sha256_file(hidden_path) != selection["hidden_evaluator_sha256"]:
        raise ValueError("hidden evaluator hash changed after technical verification")
    hidden = _normalize_hidden(hidden_path, selection)
    draws = int(bootstrap_draws or prereg["behavioral_analysis"]["bootstrap"]["draws"])
    seed = int(prereg["behavioral_analysis"]["bootstrap"]["seed"])

    model_metrics: list[dict[str, Any]] = []
    joined_by_model: dict[str, list[dict[str, Any]]] = {}
    frames_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_row in model_plan:
        rows = _load_model_rows(
            run_dir=run_dir,
            model_row=model_row,
            selection=selection,
            contract_path=contract_path,
        )
        joined = _joined_trials(rows=rows, hidden=hidden)
        frames = _collapse_frames(joined)
        joined_by_model[model_row["model_key"]] = joined
        frames_by_model[model_row["model_key"]] = frames
        model_metrics.append(
            _model_metrics(
                model_key=model_row["model_key"],
                joined=joined,
                frames=frames,
                bootstrap_draws=draws,
                bootstrap_seed=seed,
                bootstrap_test_draws=bootstrap_test_draws,
            )
        )

    raw_primary_p = {
        row["model_key"]: float(
            row["confirmatory_centered_cluster_bootstrap"]["model_conjunctive_p"]
        )
        for row in model_metrics
    }
    wrong_time_raw_p = {
        row["model_key"]: float(
            row["confirmatory_centered_cluster_bootstrap"]["real_minus_wrong_time_p"]
        )
        for row in model_metrics
    }
    adjusted = holm_adjust(raw_primary_p)
    wrong_time_adjusted = holm_adjust(wrong_time_raw_p)
    for row in model_metrics:
        key = row["model_key"]
        row["primary_holm_adjusted_p"] = adjusted[key]
        row["secondary_wrong_time_holm_adjusted_p"] = wrong_time_adjusted[key]
        row["promotion_status"] = (
            "pass"
            if row["real_capability"]["status"] == "pass"
            and row["primary_intersection_evidence_effect"][
                "intersection_union_lower_gate"
            ]
            > 0.0
            and row["primary_counterbalanced_pair_correctness"]["status"] == "pass"
            and adjusted[key] < 0.05
            else "fail"
        )

    joined_dir = run_dir / "analysis"
    joined_dir.mkdir(parents=True, exist_ok=True)
    for model_key, rows in joined_by_model.items():
        runtime.atomic_write_text(
            joined_dir / f"{model_key}_joined_trials.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )
        runtime.atomic_write_text(
            joined_dir / f"{model_key}_frame_collapsed.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in frames_by_model[model_key]),
        )
    summary = {
        "kind": "phase2_reanchored_job_a_full_analysis",
        "status": "pass",
        "preregistration_sha256": runtime._contract_preregistration_sha256(contract),
        "primary_estimand": prereg["behavioral_analysis"]["primary_estimand"],
        "statistical_unit": prereg["data"]["statistical_unit"],
        "frame_count": selection["frame_count"],
        "trial_count": selection["trial_count"],
        "models": model_metrics,
        "anchor_contrast": {
            **_anchor_contrast(frames_by_model, draws=draws, seed=seed + 101),
            "inferential_status": "prespecified_secondary_descriptive_not_in_primary_family",
        },
        "holm_family": raw_primary_p,
        "holm_adjusted": adjusted,
        "secondary_wrong_time_holm_family": wrong_time_raw_p,
        "secondary_wrong_time_holm_adjusted": wrong_time_adjusted,
        "binary_direct_contact_analysis": "omitted",
        "ontology_class_distribution": (
            "structurally imbalanced at n=800; the 3x3 first-order by second-order "
            "class distribution is not treated as balanced"
        ),
        "ontology_stratification_status": (
            "descriptive exact-n summaries plus common-support standardized sensitivity; "
            "primary participant-equal inference is unchanged"
        ),
        "ontology_boundary": "Classes 6-8 are analyzed only as annotation-native second-order interacting objects, never as definitive no-direct-contact negatives.",
        "burned_binary_reference_mixed": False,
    }
    runtime.write_json(joined_dir / "analysis_summary.json", summary)
    _write_markdown(joined_dir / "analysis_summary.md", summary)
    manifest = {
        "kind": "phase2_reanchored_job_a_analysis_manifest",
        "status": "pass",
        "analysis_summary_sha256": runtime.sha256_file(joined_dir / "analysis_summary.json"),
        "technical_verify_sha256": runtime.sha256_file(technical_path),
        "selection_manifest_sha256": runtime.sha256_file(selection_path),
    }
    runtime.write_json(joined_dir / "analysis_manifest.json", manifest)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Re-Anchored Job A 2AFC Analysis",
        "",
        f"Status: **{summary['status']}**",
        "",
        "Primary inference uses 800 frame-level units. Each frame's two counterbalanced crop orders are collapsed before participant-cluster inference.",
        "",
        "| Model | Real accuracy | Real-blank | Real-wrong | IUT lower | Pair correct | Pair lower | Holm p | Capability | Promote |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary["models"]:
        effect = row["primary_intersection_evidence_effect"]
        lines.append(
            "| {model} | {real:.3f} | {blank:.3f} | {wrong:.3f} | {lower:.3f} | {pair:.3f} | {pair_lower:.3f} | {p:.4g} | {gate} | {promote} |".format(
                model=row["model_key"],
                real=row["real_capability"]["estimate"],
                blank=effect["real_minus_blank_full"],
                wrong=effect["real_minus_wrong_time"],
                lower=effect["intersection_union_lower_gate"],
                pair=row["primary_counterbalanced_pair_correctness"]["estimate"],
                pair_lower=row["primary_counterbalanced_pair_correctness"]["one_sided_lower_95"],
                p=row["primary_holm_adjusted_p"],
                gate=row["real_capability"]["status"],
                promote=row["promotion_status"],
            )
        )
    lines.extend(
        [
            "",
            "Classes 6-8 are used only as annotation-native second-order interacting objects. No binary no-direct-contact claim is made.",
            "",
            "Counterbalanced pair correctness is a confirmatory gate against paired chance 0.25. It defeats a fixed A/B position shortcut, but it does not by itself prove scene-target binding because crop appearance may reveal ontology role.",
            "",
            "The 3x3 ontology-class distribution is structurally imbalanced, not balanced. Exact-n first-order-class, second-order-class, and class-pair summaries are descriptive; the common-support standardized estimate is a sensitivity analysis and does not replace the participant-equal primary estimand.",
            "",
        ]
    )
    runtime.atomic_write_text(path, "\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-plan", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int)
    parser.add_argument("--bootstrap-test-draws", type=int, default=10000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = analyze(
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        model_plan_path=args.model_plan,
        contract_path=args.contract,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_test_draws=args.bootstrap_test_draws,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
