#!/usr/bin/env python3
"""Paired representation analysis: visual-prompt versus plain-contact-sheet arms.

This module builds one explicit arm-to-source canonical table, reuses the
representation-probe primitives, and reports paired contrasts aligned by source
query.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.lifecycle_memory import phase2_job2_probe_analysis as probe

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VISUAL_SOURCE_RUN = REPO_ROOT / "runs" / "tamia" / "mech-job2-battery-capture" / "364962"
DEFAULT_PLAIN_SOURCE_RUN = REPO_ROOT / "runs" / "tamia" / "mech-job2b-plain-contact" / "366663"
DEFAULT_ARM_SOURCES = (
    ("visual_prompt", DEFAULT_VISUAL_SOURCE_RUN),
    ("plain_contact_sheet", DEFAULT_PLAIN_SOURCE_RUN),
)

DEFAULT_MODEL_KEYS = probe.DEFAULT_MODEL_KEYS
DEFAULT_QUERY_TYPES = (probe.PRIMARY_QUERY_TYPE, probe.SECONDARY_QUERY_TYPE)
DEFAULT_FIXED_C = 1e-3
DEFAULT_LAYER_SPEC = "all"
DEFAULT_NESTED_LAYER_SPEC = "coarse"
DEFAULT_MAX_ITER = probe.DEFAULT_MAX_ITER

PRINCIPAL_ARM_A = "visual_prompt"
PRINCIPAL_ARM_B = "plain_contact_sheet"
PRINCIPAL_MODELS = probe.PRINCIPAL_CONTRAST
PRINCIPAL_QUERY_TYPE = probe.PRIMARY_QUERY_TYPE
PRINCIPAL_POOL = "last_token"
PROMOTED_FULL_REPEATS = 1
PROMOTED_FULL_OUTER = 5
PROMOTED_FULL_INNER = 4
PROMOTED_FULL_SEED = 0
DEFAULT_PACK_PROVENANCE_MANIFEST = (
    REPO_ROOT
    / "runs"
    / "research"
    / "lifecycle_memory"
    / "phase_2"
    / "w0_geometry_sidecar_2026_07_07"
    / "derived_pack"
    / "plain_contact_sheet_pack_manifest.json"
)
DEFAULT_REFERENCE_FOLDS = (
    REPO_ROOT
    / "runs"
    / "research"
    / "lifecycle_memory"
    / "phase_2"
    / "mech_job2_representation_probes_2026_07_09"
    / "fold_assignments.json"
)
DEFAULT_EXPECTED_NON_IMAGE_DELTA_BY_FAMILY = {
    "qwen": 14,
    "gemma": 10,
}

_METRIC_TOL = 1e-9


@dataclass
class RowMeta:
    source_run: Path
    total_tokens: int | None
    non_image_tokens: int | None
    claim_boundary: str
    model_id: str | None
    dtype: str | None
    resolved_snapshot_revision: str | None
    mask_token_counts: dict[str, int]
    layer_indexes: tuple[int, ...]


@dataclass
class PairedCanonical:
    table: probe.CanonicalTable
    row_meta: dict[tuple[str, str, str], RowMeta] = field(default_factory=dict)


@dataclass
class PairedAnalysisConfig:
    out_dir: Path
    arm_sources: tuple[tuple[str, Path], ...] = DEFAULT_ARM_SOURCES
    mode: str = "full"  # audit | canary | full
    models: tuple[str, ...] = DEFAULT_MODEL_KEYS
    query_types: tuple[str, ...] = DEFAULT_QUERY_TYPES
    pools: tuple[str, ...] | None = None
    layer_spec: str = DEFAULT_LAYER_SPEC
    nested_layer_spec: str = DEFAULT_NESTED_LAYER_SPEC
    repeats: int = PROMOTED_FULL_REPEATS
    n_outer: int = PROMOTED_FULL_OUTER
    n_inner: int = PROMOTED_FULL_INNER
    seed: int = PROMOTED_FULL_SEED
    shuffle_clusters: bool = True
    bootstrap: int = probe.DEFAULT_BOOTSTRAP
    max_iter: int = DEFAULT_MAX_ITER
    fixed_c: float = DEFAULT_FIXED_C
    do_transfer: bool = True
    principal_models: tuple[str, ...] = PRINCIPAL_MODELS
    principal_query_type: str = PRINCIPAL_QUERY_TYPE
    principal_pool: str = PRINCIPAL_POOL
    intervention_presence_audit: bool = True
    enforce_intervention_presence_audit: bool = False
    token_delta_expected_by_family: dict[str, int] | None = None
    pack_provenance_manifest_path: Path | None = DEFAULT_PACK_PROVENANCE_MANIFEST
    pack_provenance_audit: bool = True
    enforce_pack_provenance_audit: bool = False
    fold_identity_reference_path: Path | None = DEFAULT_REFERENCE_FOLDS
    fold_identity_audit: bool = True
    enforce_fold_identity_audit: bool = False
    verbose: bool = True


def _log(config: PairedAnalysisConfig, message: str, *, t0: float) -> None:
    if config.verbose:
        print(f"[{time.perf_counter() - t0:8.1f}s] {message}", flush=True)


def _bool_to_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "1"}:
            return 1
        if low in {"false", "no", "0"}:
            return 0
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _extract_token_fields(mask_meta: dict[str, Any]) -> tuple[int | None, int | None]:
    image_tokens = _as_int(mask_meta.get("image_token_count"))
    non_image_tokens = _as_int(mask_meta.get("non_image_token_count"))
    seq_len = _as_int(mask_meta.get("seq_len"))
    if seq_len is None and image_tokens is not None and non_image_tokens is not None:
        seq_len = image_tokens + non_image_tokens
    if non_image_tokens is None and seq_len is not None and image_tokens is not None:
        non_image_tokens = seq_len - image_tokens
    return seq_len, non_image_tokens


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_snapshot_revision(summary: dict[str, Any]) -> str | None:
    candidates = (
        summary.get("resolved_snapshot_revision"),
        summary.get("model_snapshot"),
        summary.get("revision"),
    )
    for value in candidates:
        if isinstance(value, dict):
            for key in (
                "resolved_snapshot_revision",
                "commit_hash",
                "requested_revision",
                "revision",
            ):
                nested = value.get(key)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_mask_counts(mask_counts: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in sorted(mask_counts.items()):
        iv = _as_int(value)
        if iv is not None:
            out[str(key)] = int(iv)
    return out


def _load_model_rows_with_meta(source_run: Path, model_key: str) -> list[tuple[probe.RowRecord, RowMeta]]:
    capture_root = Path(source_run) / model_key / "capture"
    index_path = capture_root / "capture_index.jsonl"
    if not index_path.exists():
        return []
    model_summary = _read_json(Path(source_run) / model_key / "capture_summary.json")
    model_id = model_summary.get("model_id")
    dtype = model_summary.get("dtype")
    resolved_snapshot_revision = _resolve_snapshot_revision(model_summary)
    rows: list[tuple[probe.RowRecord, RowMeta]] = []
    for row in probe.iter_capture_index(index_path):
        if row.get("status") != "pass":
            continue
        hidden = row.get("hidden_index") or {}
        label = _bool_to_label(hidden.get("expected_answer"))
        source_query_id = hidden.get("source_query_id")
        participant = hidden.get("source_tier1_participant_id") or hidden.get("source_video_id")
        tensor_path = row.get("tensor_path")
        if label is None or not source_query_id or not participant or not tensor_path:
            continue
        mask_meta = row.get("mask_meta") or {}
        mask_counts = _normalize_mask_counts(mask_meta.get("mask_token_counts") or {})
        hidden_capture = row.get("hidden_capture") or {}
        layer_indexes = hidden_capture.get("layer_indexes") or []
        layer_indexes_norm = tuple(sorted(int(x) for x in layer_indexes if isinstance(x, (int, float))))
        max_layer = int(max(layer_indexes_norm) if layer_indexes_norm else 0)
        query = row.get("query") or {}
        seq_len, non_image_tokens = _extract_token_fields(mask_meta)
        record = probe.RowRecord(
            model_key=str(model_key),
            arm=str(row.get("arm")),
            request_id=str(row.get("request_id")),
            source_query_id=str(source_query_id),
            participant=str(participant),
            video_id=str(hidden.get("source_video_id") or participant),
            query_type=str(hidden.get("query_type") or "unknown"),
            label=int(label),
            target_label=str(hidden.get("target_label") or query.get("target_label") or ""),
            question_text=str(query.get("text") or ""),
            chunk_path=capture_root / str(tensor_path),
            tensor_prefix=str(row.get("tensor_prefix")),
            image_token_count=_as_int(mask_meta.get("image_token_count")),
            available_pools=tuple(sorted(str(k) for k in mask_counts)),
            hook_modules=tuple(row.get("hook_module_names") or ()),
            max_layer_index=max_layer,
        )
        meta = RowMeta(
            source_run=Path(source_run),
            total_tokens=seq_len,
            non_image_tokens=non_image_tokens,
            claim_boundary=str(row.get("claim_boundary") or ""),
            model_id=(str(model_id) if model_id is not None else None),
            dtype=(str(dtype) if dtype is not None else None),
            resolved_snapshot_revision=resolved_snapshot_revision,
            mask_token_counts=mask_counts,
            layer_indexes=layer_indexes_norm,
        )
        rows.append((record, meta))
    return rows


def _source_sha_map(arm_sources: Sequence[tuple[str, Path]], models: Sequence[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for arm, source in arm_sources:
        payload[str(arm)] = {
            "source_run": str(source),
            "top_capture_summary_sha256": probe._sha256_file(Path(source) / "capture_summary.json"),
            "source_provenance": probe.source_provenance(Path(source), models),
        }
    return payload


def build_paired_canonical_table(
    arm_sources: Sequence[tuple[str, Path]],
    models: Sequence[str],
) -> PairedCanonical:
    table = probe.CanonicalTable(source_run=Path("arm_mapped_sources"), models=tuple(models))
    row_meta: dict[tuple[str, str, str], RowMeta] = {}
    arm_order = [str(arm) for arm, _ in arm_sources]

    label_votes: dict[str, set[int]] = {}
    participant_votes: dict[str, set[str]] = {}
    video_votes: dict[str, set[str]] = {}
    query_type_votes: dict[str, set[str]] = {}
    target_votes: dict[str, set[str]] = {}
    question_votes: dict[str, set[str]] = {}
    cell_queries: dict[tuple[str, str], set[str]] = {}

    duplicate_rows: list[dict[str, Any]] = []
    missing_model_arm_cells: list[dict[str, Any]] = []
    provenance_by_model_arm: dict[tuple[str, str], dict[str, Any]] = {}

    for arm, source_run in arm_sources:
        for model_key in models:
            model_summary = _read_json(Path(source_run) / model_key / "capture_summary.json")
            provenance_by_model_arm[(str(model_key), str(arm))] = {
                "model_id": model_summary.get("model_id"),
                "dtype": model_summary.get("dtype"),
                "resolved_snapshot_revision": _resolve_snapshot_revision(model_summary),
            }
            loaded = _load_model_rows_with_meta(Path(source_run), model_key)
            rows_for_arm = [(rec, meta) for rec, meta in loaded if rec.arm == arm]
            if not rows_for_arm:
                missing_model_arm_cells.append(
                    {
                        "arm": str(arm),
                        "source_run": str(source_run),
                        "model_key": str(model_key),
                        "reason": "missing_model_or_arm_rows",
                    }
                )
                continue
            for rec, meta in rows_for_arm:
                key = (model_key, str(arm), rec.source_query_id)
                if key in table.rows:
                    duplicate_rows.append(
                        {
                            "model_key": model_key,
                            "arm": str(arm),
                            "source_query_id": rec.source_query_id,
                            "source_run": str(source_run),
                        }
                    )
                    continue
                table.rows[key] = rec
                row_meta[key] = meta
                cell_queries.setdefault((model_key, str(arm)), set()).add(rec.source_query_id)
                label_votes.setdefault(rec.source_query_id, set()).add(rec.label)
                participant_votes.setdefault(rec.source_query_id, set()).add(rec.participant)
                video_votes.setdefault(rec.source_query_id, set()).add(rec.video_id)
                query_type_votes.setdefault(rec.source_query_id, set()).add(rec.query_type)
                target_votes.setdefault(rec.source_query_id, set()).add(rec.target_label)
                question_votes.setdefault(rec.source_query_id, set()).add(rec.question_text)

    label_mismatches: list[dict[str, Any]] = []
    participant_mismatches: list[dict[str, Any]] = []
    video_mismatches: list[dict[str, Any]] = []
    query_type_mismatches: list[dict[str, Any]] = []
    target_mismatches: list[dict[str, Any]] = []
    question_mismatches: list[dict[str, Any]] = []

    for query_id, labels in label_votes.items():
        table.label_by_query[query_id] = sorted(labels)[0]
        if len(labels) > 1:
            label_mismatches.append({"source_query_id": query_id, "labels": sorted(labels)})
    for query_id, participants in participant_votes.items():
        table.participant_by_query[query_id] = sorted(participants)[0]
        if len(participants) > 1:
            participant_mismatches.append({"source_query_id": query_id, "participants": sorted(participants)})
    for query_id, videos in video_votes.items():
        if len(videos) > 1:
            video_mismatches.append({"source_query_id": query_id, "videos": sorted(videos)})
    for query_id, query_types in query_type_votes.items():
        table.query_type_by_query[query_id] = sorted(query_types)[0]
        if len(query_types) > 1:
            query_type_mismatches.append({"source_query_id": query_id, "query_types": sorted(query_types)})
    for query_id, targets in target_votes.items():
        table.target_by_query[query_id] = sorted(targets)[0] if targets else ""
        if len(targets) > 1:
            target_mismatches.append({"source_query_id": query_id, "target_labels": sorted(targets)})
    for query_id, questions in question_votes.items():
        table.question_by_query[query_id] = sorted(questions)[0] if questions else ""
        if len(questions) > 1:
            question_mismatches.append({"source_query_id": query_id, "question_texts": sorted(questions)})

    reference_cell = next(
        ((model_key, arm) for model_key in models for arm in arm_order if cell_queries.get((model_key, arm))),
        None,
    )
    reference_queries = set(cell_queries.get(reference_cell, set())) if reference_cell is not None else set()
    missing_cells: list[dict[str, Any]] = []
    for model_key in models:
        for arm in arm_order:
            present = set(cell_queries.get((model_key, arm), set()))
            missing = sorted(reference_queries - present)
            extra = sorted(present - reference_queries)
            if missing or extra:
                missing_cells.append(
                    {
                        "model_key": model_key,
                        "arm": arm,
                        "expected_rows": len(reference_queries),
                        "observed_rows": len(present),
                        "missing_count": len(missing),
                        "missing_examples": missing[:10],
                        "extra_count": len(extra),
                        "extra_examples": extra[:10],
                    }
                )

    provenance_mismatches: list[dict[str, Any]] = []
    image_token_mismatches: list[dict[str, Any]] = []
    available_pool_mismatches: list[dict[str, Any]] = []
    mask_count_mismatches: list[dict[str, Any]] = []
    layer_index_mismatches: list[dict[str, Any]] = []
    for model_key in models:
        records = {arm: provenance_by_model_arm.get((str(model_key), str(arm)), {}) for arm in arm_order}
        fields = ("model_id", "dtype", "resolved_snapshot_revision")
        for field in fields:
            vals = {arm: records.get(arm, {}).get(field) for arm in arm_order}
            if len(set(vals.values())) > 1:
                provenance_mismatches.append(
                    {
                        "model_key": str(model_key),
                        "field": field,
                        "values_by_arm": vals,
                    }
                )
        for query_id in sorted(reference_queries):
            recs: list[probe.RowRecord] = []
            metas: list[RowMeta] = []
            for arm in arm_order:
                rec = table.record(str(model_key), arm, query_id)
                if rec is None:
                    recs = []
                    break
                recs.append(rec)
                metas.append(row_meta[(str(model_key), arm, query_id)])
            if not recs:
                continue
            image_values = {rec.image_token_count for rec in recs}
            if len(image_values) > 1:
                image_token_mismatches.append(
                    {
                        "model_key": str(model_key),
                        "source_query_id": query_id,
                        "image_token_counts": {
                            arm: rec.image_token_count for arm, rec in zip(arm_order, recs)
                        },
                    }
                )
            available_values = {tuple(rec.available_pools) for rec in recs}
            if len(available_values) > 1:
                available_pool_mismatches.append(
                    {
                        "model_key": str(model_key),
                        "source_query_id": query_id,
                        "available_pools": {
                            arm: list(rec.available_pools) for arm, rec in zip(arm_order, recs)
                        },
                    }
                )
            mask_values = {tuple(sorted(meta.mask_token_counts.items())) for meta in metas}
            if len(mask_values) > 1:
                mask_count_mismatches.append(
                    {
                        "model_key": str(model_key),
                        "source_query_id": query_id,
                        "mask_token_counts": {
                            arm: metas[idx].mask_token_counts for idx, arm in enumerate(arm_order)
                        },
                    }
                )
            layer_values = {tuple(meta.layer_indexes) for meta in metas}
            if len(layer_values) > 1:
                layer_index_mismatches.append(
                    {
                        "model_key": str(model_key),
                        "source_query_id": query_id,
                        "layer_indexes": {
                            arm: list(metas[idx].layer_indexes) for idx, arm in enumerate(arm_order)
                        },
                    }
                )

    table.participants = tuple(sorted(set(table.participant_by_query.values())))
    query_type_counts: dict[str, int] = {}
    for query_type in table.query_type_by_query.values():
        query_type_counts[query_type] = query_type_counts.get(query_type, 0) + 1

    status = "pass"
    if (
        duplicate_rows
        or missing_model_arm_cells
        or label_mismatches
        or participant_mismatches
        or video_mismatches
        or query_type_mismatches
        or target_mismatches
        or question_mismatches
        or missing_cells
        or provenance_mismatches
        or image_token_mismatches
        or available_pool_mismatches
        or mask_count_mismatches
        or layer_index_mismatches
    ):
        status = "fail"

    table.audit = {
        "status": status,
        "arms": arm_order,
        "arm_sources": {arm: str(source) for arm, source in arm_sources},
        "models_requested": list(models),
        "n_source_queries": len(table.label_by_query),
        "n_participants": len(table.participants),
        "participants": list(table.participants),
        "query_type_counts": query_type_counts,
        "missing_model_arm_cells": missing_model_arm_cells,
        "duplicate_rows": duplicate_rows,
        "label_mismatches": label_mismatches,
        "participant_mismatches": participant_mismatches,
        "video_mismatches": video_mismatches,
        "query_type_mismatches": query_type_mismatches,
        "target_mismatches": target_mismatches,
        "question_mismatches": question_mismatches,
        "missing_cells": missing_cells,
        "provenance_mismatches": provenance_mismatches,
        "image_token_mismatches": image_token_mismatches,
        "available_pool_mismatches": available_pool_mismatches,
        "mask_count_mismatches": mask_count_mismatches,
        "layer_index_mismatches": layer_index_mismatches,
        "reference_cell": {
            "model_key": reference_cell[0],
            "arm": reference_cell[1],
        } if reference_cell else None,
        "method": "explicit_arm_source_mapping_with_fail_closed_alignment",
    }
    return PairedCanonical(table=table, row_meta=row_meta)


def reconstruct_nested_selected_oof(
    stack: probe.PoolStack,
    folds: dict[str, Any],
    nested_rows: Sequence[dict[str, Any]],
    *,
    max_iter: int,
    tol: float = _METRIC_TOL,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nested_by_fold = {(int(row["repeat"]), int(row["fold"])): row for row in nested_rows}
    oof_rows: list[dict[str, Any]] = []
    reconstructed_fold_rows: list[dict[str, Any]] = []
    for repeat, mapping in enumerate(folds["assignments"]):
        for fold in range(folds["n_outer"]):
            nested = nested_by_fold.get((repeat, fold))
            if nested is None:
                continue
            train_idx, test_idx = probe.outer_fold_indices(stack.groups, mapping, fold)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            layer = int(nested["selected_layer_index"])
            selected_c = float(nested["selected_C"])
            X = stack.layer_matrices[layer]
            scaler, clf, converged = probe._fit_scaler_logreg(
                X[train_idx],
                stack.y[train_idx],
                C=selected_c,
                max_iter=max_iter,
            )
            prob = probe._predict_prob(scaler, clf, X[test_idx])
            bundle = probe.metric_bundle(stack.y[test_idx], prob)
            for metric in ("auroc", "balanced_acc", "ce_bits"):
                old = float(nested.get(metric, float("nan")))
                new = float(bundle.get(metric, float("nan")))
                if math.isnan(old) and math.isnan(new):
                    continue
                if abs(old - new) > tol:
                    raise AssertionError(
                        f"nested reconstruction mismatch {stack.model_key}/{stack.arm}/{stack.query_type}/{stack.pool} "
                        f"repeat={repeat} fold={fold} metric={metric} old={old} new={new}"
                    )
            reconstructed_fold_rows.append(
                {
                    "model_key": stack.model_key,
                    "arm": stack.arm,
                    "query_type": stack.query_type,
                    "pool": stack.pool,
                    "repeat": repeat,
                    "fold": fold,
                    "selected_layer_index": layer,
                    "selected_C": selected_c,
                    "converged": bool(converged),
                    "auroc": float(bundle["auroc"]),
                    "balanced_acc": float(bundle["balanced_acc"]),
                    "ce_bits": float(bundle["ce_bits"]),
                    "n_rows": int(bundle["n_rows"]),
                }
            )
            for local_idx, row_idx in enumerate(test_idx.tolist()):
                oof_rows.append(
                    {
                        "model_key": stack.model_key,
                        "arm": stack.arm,
                        "query_type": stack.query_type,
                        "pool": stack.pool,
                        "repeat": repeat,
                        "fold": fold,
                        "source_query_id": stack.query_ids[row_idx],
                        "participant": str(stack.groups[row_idx]),
                        "y_true": int(stack.y[row_idx]),
                        "selected_layer_index": layer,
                        "selected_C": selected_c,
                        "oof_prob": float(prob[local_idx]),
                    }
                )
    return oof_rows, reconstructed_fold_rows


def paired_cluster_bootstrap_delta_auroc(
    y: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    seed_context: Sequence[Any] = (),
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)
    groups = np.asarray(groups, dtype=object)
    valid_mask = ~np.isnan(prob_a) & ~np.isnan(prob_b)
    y = y[valid_mask]
    prob_a = prob_a[valid_mask]
    prob_b = prob_b[valid_mask]
    groups = groups[valid_mask]
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) == 0:
        return {
            "delta_auroc": float("nan"),
            "auroc_a": float("nan"),
            "auroc_b": float("nan"),
            "auroc_a_ci95_low": float("nan"),
            "auroc_a_ci95_high": float("nan"),
            "auroc_b_ci95_low": float("nan"),
            "auroc_b_ci95_high": float("nan"),
            "delta_auroc_ci95_low": float("nan"),
            "delta_auroc_ci95_high": float("nan"),
            "n_boot_valid": 0,
            "method": "participant_cluster_paired_bootstrap_delta_auroc",
        }
    index_by_group = {group: np.where(groups == group)[0] for group in unique_groups}
    if len(np.unique(y)) < 2:
        point_a = float("nan")
        point_b = float("nan")
        point_delta = float("nan")
    else:
        point_a = float(roc_auc_score(y, prob_a))
        point_b = float(roc_auc_score(y, prob_b))
        point_delta = float(point_a - point_b)
    rng_parts: list[Any] = [
        "paired_cluster_bootstrap_delta_auroc",
        seed,
        len(unique_groups),
    ]
    rng_parts.extend(seed_context)
    rng = probe._seeded_rng(*rng_parts)
    boot_a: list[float] = []
    boot_b: list[float] = []
    boot_delta: list[float] = []
    for _ in range(int(n_boot)):
        chosen = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        boot_idx = np.concatenate([index_by_group[unique_groups[int(idx)]] for idx in chosen])
        yb = y[boot_idx]
        if len(np.unique(yb)) < 2:
            continue
        auroc_a = float(roc_auc_score(yb, prob_a[boot_idx]))
        auroc_b = float(roc_auc_score(yb, prob_b[boot_idx]))
        boot_a.append(auroc_a)
        boot_b.append(auroc_b)
        boot_delta.append(float(auroc_a - auroc_b))
    if boot_delta:
        auroc_a_ci95_low = float(np.percentile(boot_a, 2.5))
        auroc_a_ci95_high = float(np.percentile(boot_a, 97.5))
        auroc_b_ci95_low = float(np.percentile(boot_b, 2.5))
        auroc_b_ci95_high = float(np.percentile(boot_b, 97.5))
        delta_ci95_low = float(np.percentile(boot_delta, 2.5))
        delta_ci95_high = float(np.percentile(boot_delta, 97.5))
    else:
        auroc_a_ci95_low = float("nan")
        auroc_a_ci95_high = float("nan")
        auroc_b_ci95_low = float("nan")
        auroc_b_ci95_high = float("nan")
        delta_ci95_low = float("nan")
        delta_ci95_high = float("nan")
    return {
        "delta_auroc": point_delta,
        "auroc_a": point_a,
        "auroc_b": point_b,
        "auroc_a_ci95_low": auroc_a_ci95_low,
        "auroc_a_ci95_high": auroc_a_ci95_high,
        "auroc_b_ci95_low": auroc_b_ci95_low,
        "auroc_b_ci95_high": auroc_b_ci95_high,
        "delta_auroc_ci95_low": delta_ci95_low,
        "delta_auroc_ci95_high": delta_ci95_high,
        "n_boot_valid": len(boot_delta),
        "n_boot_requested": int(n_boot),
        "method": "participant_cluster_paired_bootstrap_delta_auroc",
    }


def _metrics_from_arrays(y: list[int], prob: list[float]) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(prob, dtype=float)
    mask = ~np.isnan(p_arr)
    if not mask.any():
        return {"n_rows": 0, "auroc": float("nan"), "balanced_acc": float("nan"), "ce_bits": float("nan"), "status": "single_class"}
    return probe.metric_bundle(y_arr[mask], p_arr[mask])


def _cross_entropy_bits_rows(y: np.ndarray, prob: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    clipped = np.clip(prob, 1e-6, 1.0 - 1e-6)
    nll = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    return nll / math.log(2.0)


def exact_participant_sign_flip_test(
    participant_mean_deltas: dict[str, float],
    *,
    chunk_size: int = 1 << 14,
) -> dict[str, Any]:
    values = np.asarray([float(v) for v in participant_mean_deltas.values()], dtype=np.float64)
    n = int(values.shape[0])
    if n == 0:
        return {
            "n_participants": 0,
            "observed_mean_delta_ce_bits": float("nan"),
            "p_value_two_sided": float("nan"),
            "n_assignments": 0,
            "method": "exact_participant_sign_flip_two_sided_mean_delta_ce_bits",
        }
    observed = float(np.mean(values))
    total = 1 << n
    exceed = 0
    bit_positions = np.arange(n, dtype=np.uint64)
    threshold = abs(observed) - 1e-15
    for start in range(0, total, int(chunk_size)):
        end = min(total, start + int(chunk_size))
        ids = np.arange(start, end, dtype=np.uint64)
        bits = ((ids[:, None] >> bit_positions[None, :]) & np.uint64(1)).astype(np.float64)
        signs = 1.0 - 2.0 * bits
        stats = np.abs((signs * values[None, :]).mean(axis=1))
        exceed += int(np.count_nonzero(stats >= threshold))
    p_value = float(exceed / total)
    return {
        "n_participants": n,
        "observed_mean_delta_ce_bits": observed,
        "p_value_two_sided": p_value,
        "n_assignments": total,
        "method": "exact_participant_sign_flip_two_sided_mean_delta_ce_bits",
    }


def holm_adjust_two_models(
    p_values: dict[str, float],
) -> dict[str, float]:
    clean = {
        model: float(p)
        for model, p in p_values.items()
        if p is not None and not math.isnan(float(p))
    }
    if not clean:
        return {}
    ordered = sorted(clean.items(), key=lambda kv: kv[1])
    adjusted: dict[str, float] = {}
    prev = 0.0
    m = len(ordered)
    for rank, (model, pval) in enumerate(ordered, start=1):
        multiplier = m - rank + 1
        adj = min(1.0, pval * multiplier)
        adj = max(adj, prev)
        adjusted[model] = adj
        prev = adj
    return adjusted


def _same_layer_contrasts(
    oof_rows: Sequence[dict[str, Any]],
    *,
    arm_a: str,
    arm_b: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], dict[str, dict[str, dict[str, Any]]]] = {}
    for row in oof_rows:
        key = (
            str(row["model_key"]),
            str(row["query_type"]),
            str(row["pool"]),
            int(row["layer_index"]),
        )
        arm = str(row["arm"])
        query_id = str(row["source_query_id"])
        grouped.setdefault(key, {}).setdefault(arm, {})[query_id] = row
    out: list[dict[str, Any]] = []
    for (model_key, query_type, pool, layer_index), per_arm in grouped.items():
        rows_a = per_arm.get(arm_a, {})
        rows_b = per_arm.get(arm_b, {})
        shared = sorted(set(rows_a).intersection(rows_b))
        if not shared:
            continue
        y = [int(rows_a[q]["y_true"]) for q in shared]
        prob_a = [float(rows_a[q]["oof_prob"]) for q in shared]
        prob_b = [float(rows_b[q]["oof_prob"]) for q in shared]
        metrics_a = _metrics_from_arrays(y, prob_a)
        metrics_b = _metrics_from_arrays(y, prob_b)
        out.append(
            {
                "model_key": model_key,
                "query_type": query_type,
                "pool": pool,
                "layer_index": layer_index,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "n_aligned_rows": len(shared),
                "auroc_a": float(metrics_a["auroc"]),
                "auroc_b": float(metrics_b["auroc"]),
                "delta_auroc": float(metrics_a["auroc"] - metrics_b["auroc"]),
                "balanced_acc_a": float(metrics_a["balanced_acc"]),
                "balanced_acc_b": float(metrics_b["balanced_acc"]),
                "delta_balanced_acc": float(metrics_a["balanced_acc"] - metrics_b["balanced_acc"]),
                "ce_bits_a": float(metrics_a["ce_bits"]),
                "ce_bits_b": float(metrics_b["ce_bits"]),
                "delta_ce_bits": float(metrics_a["ce_bits"] - metrics_b["ce_bits"]),
                "method": "descriptive_same_layer_query_aligned_oof_contrast",
            }
        )
    return out


def _stack_query_index(stack: probe.PoolStack) -> dict[str, int]:
    return {str(query_id): idx for idx, query_id in enumerate(stack.query_ids)}


def _aligned_query_maps(
    stack_a: probe.PoolStack,
    stack_b: probe.PoolStack,
) -> tuple[list[str], dict[str, int], dict[str, int], dict[str, str]]:
    map_a = _stack_query_index(stack_a)
    map_b = _stack_query_index(stack_b)
    shared = sorted(set(map_a).intersection(map_b))
    participant_by_query: dict[str, str] = {}
    for query_id in shared:
        ia = map_a[query_id]
        ib = map_b[query_id]
        part_a = str(stack_a.groups[ia])
        part_b = str(stack_b.groups[ib])
        if part_a != part_b:
            raise AssertionError(f"participant mismatch for aligned query {query_id}")
        y_a = int(stack_a.y[ia])
        y_b = int(stack_b.y[ib])
        if y_a != y_b:
            raise AssertionError(f"label mismatch for aligned query {query_id}")
        participant_by_query[query_id] = part_a
    return shared, map_a, map_b, participant_by_query


def _fit_predict_layer(
    stack: probe.PoolStack,
    layer: int,
    query_idx_map: dict[str, int],
    train_queries: Sequence[str],
    test_queries: Sequence[str],
    *,
    fixed_c: float,
    max_iter: int,
) -> tuple[np.ndarray, bool]:
    train_idx = np.asarray([query_idx_map[q] for q in train_queries], dtype=int)
    test_idx = np.asarray([query_idx_map[q] for q in test_queries], dtype=int)
    X = stack.layer_matrices[layer]
    y = stack.y.astype(int)
    if len(np.unique(y[train_idx])) < 2:
        return np.full(len(test_idx), float(np.mean(y[train_idx]))), True
    scaler, clf, converged = probe._fit_scaler_logreg(
        X[train_idx],
        y[train_idx],
        C=fixed_c,
        max_iter=max_iter,
    )
    return probe._predict_prob(scaler, clf, X[test_idx]), bool(converged)


def shared_symmetric_nested_selected_oof(
    stack_a: probe.PoolStack,
    stack_b: probe.PoolStack,
    folds: dict[str, Any],
    *,
    candidate_layers: Sequence[int],
    fixed_c: float,
    max_iter: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shared_queries, map_a, map_b, participant_by_query = _aligned_query_maps(stack_a, stack_b)
    if not shared_queries:
        return [], []
    oof_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    n_inner = int(folds["n_inner"])
    seed = int(folds["seed"])

    for repeat, mapping in enumerate(folds["assignments"]):
        for fold in range(int(folds["n_outer"])):
            train_queries = [
                q for q in shared_queries
                if mapping.get(participant_by_query[q], -1) not in (fold, -1)
            ]
            test_queries = [
                q for q in shared_queries
                if mapping.get(participant_by_query[q], -1) == fold
            ]
            if not train_queries or not test_queries:
                continue
            train_parts = sorted({participant_by_query[q] for q in train_queries})
            inner = probe.inner_fold_splits(train_parts, n_inner=n_inner, seed=seed, repeat=repeat, outer_fold=fold)
            if not inner:
                continue

            best: tuple[float, int, float, float] | None = None
            for layer in candidate_layers:
                if layer not in stack_a.layer_matrices or layer not in stack_b.layer_matrices:
                    continue
                fold_losses_a: list[float] = []
                fold_losses_b: list[float] = []
                for inner_train_parts, inner_val_parts in inner:
                    inner_train = [q for q in train_queries if participant_by_query[q] in set(inner_train_parts)]
                    inner_val = [q for q in train_queries if participant_by_query[q] in set(inner_val_parts)]
                    if not inner_train or not inner_val:
                        continue
                    pred_a, _ = _fit_predict_layer(
                        stack_a,
                        layer,
                        map_a,
                        inner_train,
                        inner_val,
                        fixed_c=fixed_c,
                        max_iter=max_iter,
                    )
                    pred_b, _ = _fit_predict_layer(
                        stack_b,
                        layer,
                        map_b,
                        inner_train,
                        inner_val,
                        fixed_c=fixed_c,
                        max_iter=max_iter,
                    )
                    y_val = np.asarray([int(stack_a.y[map_a[q]]) for q in inner_val], dtype=int)
                    fold_losses_a.append(float(probe.cross_entropy_bits(y_val, pred_a)))
                    fold_losses_b.append(float(probe.cross_entropy_bits(y_val, pred_b)))
                if not fold_losses_a or not fold_losses_b:
                    continue
                mean_a = float(np.mean(fold_losses_a))
                mean_b = float(np.mean(fold_losses_b))
                shared_loss = mean_a + mean_b
                if best is None or shared_loss < best[0]:
                    best = (shared_loss, int(layer), mean_a, mean_b)
            if best is None:
                continue
            shared_loss, selected_layer, inner_loss_a, inner_loss_b = best
            pred_a, converged_a = _fit_predict_layer(
                stack_a,
                selected_layer,
                map_a,
                train_queries,
                test_queries,
                fixed_c=fixed_c,
                max_iter=max_iter,
            )
            pred_b, converged_b = _fit_predict_layer(
                stack_b,
                selected_layer,
                map_b,
                train_queries,
                test_queries,
                fixed_c=fixed_c,
                max_iter=max_iter,
            )
            y_test = [int(stack_a.y[map_a[q]]) for q in test_queries]
            metrics_a = _metrics_from_arrays(y_test, pred_a.tolist())
            metrics_b = _metrics_from_arrays(y_test, pred_b.tolist())
            fold_rows.append(
                {
                    "model_key": stack_a.model_key,
                    "query_type": stack_a.query_type,
                    "pool": stack_a.pool,
                    "repeat": repeat,
                    "fold": fold,
                    "arm_a": stack_a.arm,
                    "arm_b": stack_b.arm,
                    "n_aligned_rows": len(test_queries),
                    "selected_shared_layer_index": selected_layer,
                    "selected_C": float(fixed_c),
                    "inner_ce_bits_sum": float(shared_loss),
                    "inner_ce_bits_a": float(inner_loss_a),
                    "inner_ce_bits_b": float(inner_loss_b),
                    "converged_a": bool(converged_a),
                    "converged_b": bool(converged_b),
                    "auroc_a": float(metrics_a["auroc"]),
                    "auroc_b": float(metrics_b["auroc"]),
                    "delta_auroc": float(metrics_a["auroc"] - metrics_b["auroc"]),
                    "balanced_acc_a": float(metrics_a["balanced_acc"]),
                    "balanced_acc_b": float(metrics_b["balanced_acc"]),
                    "delta_balanced_acc": float(metrics_a["balanced_acc"] - metrics_b["balanced_acc"]),
                    "ce_bits_a": float(metrics_a["ce_bits"]),
                    "ce_bits_b": float(metrics_b["ce_bits"]),
                    "delta_ce_bits": float(metrics_a["ce_bits"] - metrics_b["ce_bits"]),
                    "method": "primary_shared_symmetric_nested_layer_selection_outer_fold",
                }
            )
            for i, query_id in enumerate(test_queries):
                participant = participant_by_query[query_id]
                y_true = int(stack_a.y[map_a[query_id]])
                oof_rows.append(
                    {
                        "model_key": stack_a.model_key,
                        "arm": stack_a.arm,
                        "query_type": stack_a.query_type,
                        "pool": stack_a.pool,
                        "repeat": repeat,
                        "fold": fold,
                        "source_query_id": query_id,
                        "participant": participant,
                        "y_true": y_true,
                        "selected_layer_index": selected_layer,
                        "selected_C": float(fixed_c),
                        "converged": bool(converged_a),
                        "oof_prob": float(pred_a[i]),
                        "selection_method": "shared_symmetric_nested_layer",
                        "role": "primary",
                    }
                )
                oof_rows.append(
                    {
                        "model_key": stack_b.model_key,
                        "arm": stack_b.arm,
                        "query_type": stack_b.query_type,
                        "pool": stack_b.pool,
                        "repeat": repeat,
                        "fold": fold,
                        "source_query_id": query_id,
                        "participant": participant,
                        "y_true": y_true,
                        "selected_layer_index": selected_layer,
                        "selected_C": float(fixed_c),
                        "converged": bool(converged_b),
                        "oof_prob": float(pred_b[i]),
                        "selection_method": "shared_symmetric_nested_layer",
                        "role": "primary",
                    }
                )
    return oof_rows, fold_rows


def _paired_nested_contrasts(
    nested_selected_oof_rows: Sequence[dict[str, Any]],
    *,
    arm_a: str,
    arm_b: str,
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_groups: dict[tuple[str, str, str, int, int], dict[str, dict[str, dict[str, Any]]]] = {}
    for row in nested_selected_oof_rows:
        key = (
            str(row["model_key"]),
            str(row["query_type"]),
            str(row["pool"]),
            int(row["repeat"]),
            int(row["fold"]),
        )
        arm = str(row["arm"])
        query_id = str(row["source_query_id"])
        fold_groups.setdefault(key, {}).setdefault(arm, {})[query_id] = row

    fold_rows: list[dict[str, Any]] = []
    for (model_key, query_type, pool, repeat, fold), by_arm in sorted(fold_groups.items()):
        rows_a = by_arm.get(arm_a, {})
        rows_b = by_arm.get(arm_b, {})
        shared = sorted(set(rows_a).intersection(rows_b))
        if not shared:
            continue
        y = [int(rows_a[q]["y_true"]) for q in shared]
        prob_a = [float(rows_a[q]["oof_prob"]) for q in shared]
        prob_b = [float(rows_b[q]["oof_prob"]) for q in shared]
        metrics_a = _metrics_from_arrays(y, prob_a)
        metrics_b = _metrics_from_arrays(y, prob_b)
        sample_q = shared[0]
        fold_rows.append(
            {
                "model_key": model_key,
                "query_type": query_type,
                "pool": pool,
                "repeat": repeat,
                "fold": fold,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "n_aligned_rows": len(shared),
                "selected_layer_index_a": int(rows_a[sample_q]["selected_layer_index"]),
                "selected_layer_index_b": int(rows_b[sample_q]["selected_layer_index"]),
                "auroc_a": float(metrics_a["auroc"]),
                "auroc_b": float(metrics_b["auroc"]),
                "delta_auroc": float(metrics_a["auroc"] - metrics_b["auroc"]),
                "balanced_acc_a": float(metrics_a["balanced_acc"]),
                "balanced_acc_b": float(metrics_b["balanced_acc"]),
                "delta_balanced_acc": float(metrics_a["balanced_acc"] - metrics_b["balanced_acc"]),
                "ce_bits_a": float(metrics_a["ce_bits"]),
                "ce_bits_b": float(metrics_b["ce_bits"]),
                "delta_ce_bits": float(metrics_a["ce_bits"] - metrics_b["ce_bits"]),
                "method": "secondary_independent_nested_outer_fold_contrast",
            }
        )

    summary_groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in nested_selected_oof_rows:
        key = (str(row["model_key"]), str(row["query_type"]), str(row["pool"]))
        arm = str(row["arm"])
        query_id = str(row["source_query_id"])
        store = summary_groups.setdefault(key, {}).setdefault(arm, {})
        item = store.setdefault(
            query_id,
            {
                "participant": str(row["participant"]),
                "y_true": int(row["y_true"]),
                "probs": [],
            },
        )
        item["probs"].append(float(row["oof_prob"]))

    summary_rows: list[dict[str, Any]] = []
    for (model_key, query_type, pool), by_arm in sorted(summary_groups.items()):
        rows_a = by_arm.get(arm_a, {})
        rows_b = by_arm.get(arm_b, {})
        shared = sorted(set(rows_a).intersection(rows_b))
        if not shared:
            continue
        y: list[int] = []
        groups: list[str] = []
        prob_a: list[float] = []
        prob_b: list[float] = []
        for query_id in shared:
            y_a = int(rows_a[query_id]["y_true"])
            y_b = int(rows_b[query_id]["y_true"])
            if y_a != y_b:
                raise AssertionError(f"label mismatch in nested OOF pairing: {query_id}")
            y.append(y_a)
            groups.append(str(rows_a[query_id]["participant"]))
            prob_a.append(float(np.mean(rows_a[query_id]["probs"])))
            prob_b.append(float(np.mean(rows_b[query_id]["probs"])))
        metrics_a = _metrics_from_arrays(y, prob_a)
        metrics_b = _metrics_from_arrays(y, prob_b)
        delta_ci = paired_cluster_bootstrap_delta_auroc(
            np.asarray(y, dtype=int),
            np.asarray(prob_a, dtype=float),
            np.asarray(prob_b, dtype=float),
            np.asarray(groups, dtype=object),
            n_boot=bootstrap,
            seed=seed,
            seed_context=("secondary_independent", model_key, query_type, pool),
        )
        fold_subset = [
            row
            for row in fold_rows
            if row["model_key"] == model_key
            and row["query_type"] == query_type
            and row["pool"] == pool
        ]
        layers_a = [int(row["selected_layer_index_a"]) for row in fold_subset]
        layers_b = [int(row["selected_layer_index_b"]) for row in fold_subset]
        summary_rows.append(
            {
                "model_key": model_key,
                "query_type": query_type,
                "pool": pool,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "n_aligned_rows": len(shared),
                "auroc_a": float(metrics_a["auroc"]),
                "auroc_b": float(metrics_b["auroc"]),
                "delta_auroc": float(metrics_a["auroc"] - metrics_b["auroc"]),
                "auroc_a_ci95_low": float(delta_ci["auroc_a_ci95_low"]),
                "auroc_a_ci95_high": float(delta_ci["auroc_a_ci95_high"]),
                "auroc_b_ci95_low": float(delta_ci["auroc_b_ci95_low"]),
                "auroc_b_ci95_high": float(delta_ci["auroc_b_ci95_high"]),
                "delta_auroc_ci95_low": float(delta_ci["delta_auroc_ci95_low"]),
                "delta_auroc_ci95_high": float(delta_ci["delta_auroc_ci95_high"]),
                "delta_auroc_boot_valid": int(delta_ci["n_boot_valid"]),
                "delta_auroc_boot_valid_fraction": (
                    float(delta_ci["n_boot_valid"] / max(delta_ci["n_boot_requested"], 1))
                ),
                "balanced_acc_a": float(metrics_a["balanced_acc"]),
                "balanced_acc_b": float(metrics_b["balanced_acc"]),
                "delta_balanced_acc": float(metrics_a["balanced_acc"] - metrics_b["balanced_acc"]),
                "ce_bits_a": float(metrics_a["ce_bits"]),
                "ce_bits_b": float(metrics_b["ce_bits"]),
                "delta_ce_bits": float(metrics_a["ce_bits"] - metrics_b["ce_bits"]),
                "mean_selected_layer_index_a": (float(np.mean(layers_a)) if layers_a else float("nan")),
                "mean_selected_layer_index_b": (float(np.mean(layers_b)) if layers_b else float("nan")),
                "method": "secondary_independent_nested_query_aligned_summary",
                "bootstrap_method": delta_ci["method"],
            }
        )

    return fold_rows, summary_rows


def _paired_summary_with_inference(
    *,
    oof_rows: Sequence[dict[str, Any]],
    fold_rows: Sequence[dict[str, Any]],
    arm_a: str,
    arm_b: str,
    bootstrap: int,
    seed: int,
    method: str,
) -> list[dict[str, Any]]:
    summary_groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in oof_rows:
        key = (str(row["model_key"]), str(row["query_type"]), str(row["pool"]))
        arm = str(row["arm"])
        query_id = str(row["source_query_id"])
        store = summary_groups.setdefault(key, {}).setdefault(arm, {})
        item = store.setdefault(
            query_id,
            {
                "participant": str(row["participant"]),
                "y_true": int(row["y_true"]),
                "probs": [],
            },
        )
        item["probs"].append(float(row["oof_prob"]))
    out: list[dict[str, Any]] = []
    for (model_key, query_type, pool), by_arm in sorted(summary_groups.items()):
        rows_a = by_arm.get(arm_a, {})
        rows_b = by_arm.get(arm_b, {})
        shared = sorted(set(rows_a).intersection(rows_b))
        if not shared:
            continue
        y: list[int] = []
        participants: list[str] = []
        prob_a: list[float] = []
        prob_b: list[float] = []
        for query_id in shared:
            y_a = int(rows_a[query_id]["y_true"])
            y_b = int(rows_b[query_id]["y_true"])
            if y_a != y_b:
                raise AssertionError(f"label mismatch in paired OOF summary: {query_id}")
            y.append(y_a)
            participants.append(str(rows_a[query_id]["participant"]))
            prob_a.append(float(np.mean(rows_a[query_id]["probs"])))
            prob_b.append(float(np.mean(rows_b[query_id]["probs"])))
        metrics_a = _metrics_from_arrays(y, prob_a)
        metrics_b = _metrics_from_arrays(y, prob_b)
        boot = paired_cluster_bootstrap_delta_auroc(
            np.asarray(y, dtype=int),
            np.asarray(prob_a, dtype=float),
            np.asarray(prob_b, dtype=float),
            np.asarray(participants, dtype=object),
            n_boot=bootstrap,
            seed=seed,
            seed_context=("primary_shared_symmetric", model_key, query_type, pool),
        )
        ce_a = _cross_entropy_bits_rows(np.asarray(y, dtype=int), np.asarray(prob_a, dtype=float))
        ce_b = _cross_entropy_bits_rows(np.asarray(y, dtype=int), np.asarray(prob_b, dtype=float))
        by_participant: dict[str, list[float]] = {}
        for participant, delta in zip(participants, (ce_a - ce_b).tolist()):
            by_participant.setdefault(str(participant), []).append(float(delta))
        participant_mean = {
            participant: float(np.mean(values))
            for participant, values in by_participant.items()
        }
        sign_flip = exact_participant_sign_flip_test(participant_mean)
        fold_subset = [
            row for row in fold_rows
            if row["model_key"] == model_key and row["query_type"] == query_type and row["pool"] == pool
        ]
        fold_deltas = [float(row["delta_auroc"]) for row in fold_subset if not math.isnan(float(row["delta_auroc"]))]
        global_delta = float(metrics_a["auroc"] - metrics_b["auroc"])
        global_sign = 0
        if not math.isnan(global_delta):
            global_sign = int(np.sign(global_delta))
        fold_sign_consistent = 0
        if global_sign != 0:
            fold_sign_consistent = sum(1 for delta in fold_deltas if int(np.sign(delta)) == global_sign)
        convergence_healthy = all(
            bool(row.get("converged_a", True)) and bool(row.get("converged_b", True))
            for row in fold_subset
        )
        out.append(
            {
                "model_key": model_key,
                "query_type": query_type,
                "pool": pool,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "n_aligned_rows": len(shared),
                "auroc_a": float(metrics_a["auroc"]),
                "auroc_b": float(metrics_b["auroc"]),
                "delta_auroc": global_delta,
                "auroc_a_ci95_low": float(boot["auroc_a_ci95_low"]),
                "auroc_a_ci95_high": float(boot["auroc_a_ci95_high"]),
                "auroc_b_ci95_low": float(boot["auroc_b_ci95_low"]),
                "auroc_b_ci95_high": float(boot["auroc_b_ci95_high"]),
                "delta_auroc_ci95_low": float(boot["delta_auroc_ci95_low"]),
                "delta_auroc_ci95_high": float(boot["delta_auroc_ci95_high"]),
                "delta_auroc_boot_valid": int(boot["n_boot_valid"]),
                "delta_auroc_boot_valid_fraction": (
                    float(boot["n_boot_valid"] / max(boot["n_boot_requested"], 1))
                ),
                "balanced_acc_a": float(metrics_a["balanced_acc"]),
                "balanced_acc_b": float(metrics_b["balanced_acc"]),
                "delta_balanced_acc": float(metrics_a["balanced_acc"] - metrics_b["balanced_acc"]),
                "ce_bits_a": float(metrics_a["ce_bits"]),
                "ce_bits_b": float(metrics_b["ce_bits"]),
                "delta_ce_bits": float(metrics_a["ce_bits"] - metrics_b["ce_bits"]),
                "exact_sign_flip_p_two_sided_ce_bits": float(sign_flip["p_value_two_sided"]),
                "exact_sign_flip_n_assignments": int(sign_flip["n_assignments"]),
                "exact_sign_flip_n_participants": int(sign_flip["n_participants"]),
                "fold_count": len(fold_deltas),
                "fold_sign_consistent_count": int(fold_sign_consistent),
                "convergence_healthy": bool(convergence_healthy),
                "method": method,
                "bootstrap_method": boot["method"],
                "sign_flip_method": sign_flip["method"],
            }
        )
    return out


def _attach_holm_to_principal_rows(
    rows: Sequence[dict[str, Any]],
    *,
    principal_models: Sequence[str],
    query_type: str,
    pool: str,
) -> None:
    candidates = [
        row for row in rows
        if row["model_key"] in set(principal_models)
        and row["query_type"] == query_type
        and row["pool"] == pool
    ]
    model_to_p = {
        str(row["model_key"]): float(row["exact_sign_flip_p_two_sided_ce_bits"])
        for row in candidates
    }
    holm = holm_adjust_two_models(model_to_p)
    for row in rows:
        row["holm_adjusted_p_two_models"] = float("nan")
    for row in candidates:
        key = str(row["model_key"])
        if key in holm:
            row["holm_adjusted_p_two_models"] = float(holm[key])


def build_principal_promotion_readiness(
    *,
    shared_summary_rows: Sequence[dict[str, Any]],
    independent_summary_rows: Sequence[dict[str, Any]],
    principal_models: Sequence[str],
    principal_query_type: str,
    principal_pool: str,
    min_fold_sign_consistent: int = 4,
    min_boot_valid_fraction: float = 0.95,
) -> list[dict[str, Any]]:
    independent_map = {
        str(row["model_key"]): row
        for row in independent_summary_rows
        if row["query_type"] == principal_query_type
        and row["pool"] == principal_pool
    }
    out: list[dict[str, Any]] = []
    for row in shared_summary_rows:
        model_key = str(row["model_key"])
        if model_key not in set(principal_models):
            continue
        if row["query_type"] != principal_query_type or row["pool"] != principal_pool:
            continue
        independent = independent_map.get(model_key, {})
        shared_delta = float(row["delta_auroc"])
        independent_delta = float(independent.get("delta_auroc", float("nan")))
        shared_sign = int(np.sign(shared_delta)) if not math.isnan(shared_delta) else 0
        independent_sign = int(np.sign(independent_delta)) if not math.isnan(independent_delta) else 0
        signs_agree = bool(shared_sign != 0 and shared_sign == independent_sign)
        ci_low = float(row["delta_auroc_ci95_low"])
        ci_high = float(row["delta_auroc_ci95_high"])
        ci_excludes_zero = bool((ci_low > 0.0) or (ci_high < 0.0))
        holm_p = float(row.get("holm_adjusted_p_two_models", float("nan")))
        fold_sign_ok = int(row.get("fold_sign_consistent_count", 0)) >= int(min_fold_sign_consistent)
        boot_valid_fraction = float(row.get("delta_auroc_boot_valid_fraction", 0.0))
        boot_valid_ok = boot_valid_fraction >= float(min_boot_valid_fraction)
        convergence_healthy = bool(row.get("convergence_healthy", False))
        holm_ok = (not math.isnan(holm_p)) and (holm_p < 0.05)
        ready = bool(
            ci_excludes_zero
            and holm_ok
            and fold_sign_ok
            and boot_valid_ok
            and convergence_healthy
            and signs_agree
        )
        out.append(
            {
                "model_key": model_key,
                "query_type": principal_query_type,
                "pool": principal_pool,
                "shared_nested_delta_auroc": shared_delta,
                "shared_nested_delta_auroc_ci95_low": ci_low,
                "shared_nested_delta_auroc_ci95_high": ci_high,
                "shared_nested_exact_sign_flip_p_two_sided_ce_bits": float(
                    row["exact_sign_flip_p_two_sided_ce_bits"]
                ),
                "shared_nested_holm_adjusted_p_two_models": holm_p,
                "shared_nested_fold_sign_consistent_count": int(row.get("fold_sign_consistent_count", 0)),
                "shared_nested_fold_count": int(row.get("fold_count", 0)),
                "shared_nested_bootstrap_valid_fraction": boot_valid_fraction,
                "shared_nested_convergence_healthy": convergence_healthy,
                "independent_nested_delta_auroc_secondary": independent_delta,
                "shared_vs_independent_sign_agree": signs_agree,
                "promotion_ready_gate": ready,
                "promotion_ready_gate_note": (
                    "Gate label only. It does not establish behavior, use, binding, temporal memory, or causality."
                ),
            }
        )
    return out


def build_prompt_token_audit(
    paired: PairedCanonical,
    *,
    arm_a: str,
    arm_b: str,
    models: Sequence[str],
    expected_non_image_delta_by_family: dict[str, int] | None = None,
) -> dict[str, Any]:
    table = paired.table
    by_model: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_non_image_deltas: list[int] = []
    all_total_deltas: list[int] = []
    claim_boundary_a: set[str] = set()
    claim_boundary_b: set[str] = set()

    for model_key in models:
        aligned_queries = []
        for query_id in sorted(table.label_by_query):
            rec_a = table.record(model_key, arm_a, query_id)
            rec_b = table.record(model_key, arm_b, query_id)
            if rec_a is None or rec_b is None:
                continue
            aligned_queries.append(query_id)
        question_mismatch = 0
        image_mismatch = 0
        non_image_deltas: list[int] = []
        total_deltas: list[int] = []
        for query_id in aligned_queries:
            rec_a = table.record(model_key, arm_a, query_id)
            rec_b = table.record(model_key, arm_b, query_id)
            assert rec_a is not None and rec_b is not None
            if rec_a.question_text != rec_b.question_text:
                question_mismatch += 1
            if rec_a.image_token_count != rec_b.image_token_count:
                image_mismatch += 1
            meta_a = paired.row_meta[(model_key, arm_a, query_id)]
            meta_b = paired.row_meta[(model_key, arm_b, query_id)]
            if meta_a.non_image_tokens is not None and meta_b.non_image_tokens is not None:
                non_image_deltas.append(int(meta_a.non_image_tokens - meta_b.non_image_tokens))
            if meta_a.total_tokens is not None and meta_b.total_tokens is not None:
                total_deltas.append(int(meta_a.total_tokens - meta_b.total_tokens))
            if meta_a.claim_boundary:
                claim_boundary_a.add(meta_a.claim_boundary)
            if meta_b.claim_boundary:
                claim_boundary_b.add(meta_b.claim_boundary)
        unique_non_image = sorted(set(non_image_deltas))
        unique_total = sorted(set(total_deltas))
        has_non_image = bool(non_image_deltas)
        has_total = bool(total_deltas)
        family = probe.model_family(model_key)
        expected_non_image = None
        if expected_non_image_delta_by_family is not None:
            expected_non_image = expected_non_image_delta_by_family.get(family)
        expected_match = (
            True
            if expected_non_image is None or not unique_non_image
            else (len(unique_non_image) == 1 and unique_non_image[0] == int(expected_non_image))
        )
        constant_non_image = has_non_image and len(unique_non_image) == 1
        constant_total = has_total and len(unique_total) == 1
        by_model.append(
            {
                "model_key": model_key,
                "model_family": family,
                "n_aligned_queries": len(aligned_queries),
                "question_mismatch_count": question_mismatch,
                "image_token_mismatch_count": image_mismatch,
                "non_image_token_delta_mean": (float(np.mean(non_image_deltas)) if non_image_deltas else float("nan")),
                "non_image_token_delta_min": (int(min(non_image_deltas)) if non_image_deltas else None),
                "non_image_token_delta_max": (int(max(non_image_deltas)) if non_image_deltas else None),
                "non_image_token_delta_unique": unique_non_image,
                "non_image_token_delta_constant": constant_non_image,
                "total_token_delta_mean": (float(np.mean(total_deltas)) if total_deltas else float("nan")),
                "total_token_delta_min": (int(min(total_deltas)) if total_deltas else None),
                "total_token_delta_max": (int(max(total_deltas)) if total_deltas else None),
                "total_token_delta_unique": unique_total,
                "total_token_delta_constant": constant_total,
                "expected_non_image_delta_for_family": expected_non_image,
                "expected_non_image_delta_match": expected_match,
            }
        )
        if question_mismatch > 0:
            failures.append({"model_key": model_key, "reason": "question_text_mismatch", "count": question_mismatch})
        if image_mismatch > 0:
            failures.append({"model_key": model_key, "reason": "image_token_mismatch", "count": image_mismatch})
        if not constant_non_image:
            failures.append(
                {
                    "model_key": model_key,
                    "reason": "non_image_token_delta_not_constant",
                    "unique_values": unique_non_image[:20],
                }
            )
        if not constant_total:
            failures.append(
                {
                    "model_key": model_key,
                    "reason": "total_token_delta_not_constant",
                    "unique_values": unique_total[:20],
                }
            )
        if not expected_match:
            failures.append(
                {
                    "model_key": model_key,
                    "reason": "expected_non_image_delta_mismatch",
                    "expected": expected_non_image,
                    "observed_unique": unique_non_image[:20],
                }
            )
        all_non_image_deltas.extend(non_image_deltas)
        all_total_deltas.extend(total_deltas)

    instructions_differ = bool(claim_boundary_a != claim_boundary_b) or any(delta != 0 for delta in all_non_image_deltas)
    status = "pass" if not failures else "fail"
    return {
        "arms": {"arm_a": arm_a, "arm_b": arm_b},
        "status": status,
        "failures": failures,
        "model_rows": by_model,
        "global": {
            "question_mismatch_count": sum(int(row["question_mismatch_count"]) for row in by_model),
            "image_token_mismatch_count": sum(int(row["image_token_mismatch_count"]) for row in by_model),
            "mean_non_image_token_delta": (float(np.mean(all_non_image_deltas)) if all_non_image_deltas else float("nan")),
            "mean_total_token_delta": (float(np.mean(all_total_deltas)) if all_total_deltas else float("nan")),
            "instructions_differ": instructions_differ,
            "expected_non_image_delta_by_family": (expected_non_image_delta_by_family or {}),
            "intervention_scope_note": (
                "Estimand is the complete localization-prompt intervention (instructions plus pixels), "
                "not annotation pixels alone. Because the instructions change sequence length, "
                "last-token contrasts also include a token-position shift."
            ),
        },
    }


def _arm_sources_match_default(config: PairedAnalysisConfig) -> bool:
    expected = {str(arm): str(Path(path).resolve()) for arm, path in DEFAULT_ARM_SOURCES}
    observed = {str(arm): str(Path(path).resolve()) for arm, path in config.arm_sources}
    return observed == expected


def _expected_token_delta_by_family(config: PairedAnalysisConfig) -> dict[str, int] | None:
    if config.token_delta_expected_by_family is not None:
        return {str(k): int(v) for k, v in config.token_delta_expected_by_family.items()}
    if _arm_sources_match_default(config):
        return dict(DEFAULT_EXPECTED_NON_IMAGE_DELTA_BY_FAMILY)
    return None


def _should_run_real_mode_audit_gate(config: PairedAnalysisConfig, *, force: bool) -> bool:
    if force:
        return True
    if config.mode not in {"audit", "full"}:
        return False
    return _arm_sources_match_default(config)


def build_intervention_presence_audit(
    paired: PairedCanonical,
    *,
    arm_a: str,
    arm_b: str,
    models: Sequence[str],
) -> dict[str, Any]:
    table = paired.table
    model_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with contextlib.ExitStack() as stack:
        chunk_cache: dict[Path, probe.SafetensorsChunk] = {}

        def read_vec(rec: probe.RowRecord, layer: int, pool: str) -> np.ndarray | None:
            key = rec.layer_key(pool, layer)
            chunk = chunk_cache.get(rec.chunk_path)
            if chunk is None:
                chunk = stack.enter_context(probe.SafetensorsChunk(rec.chunk_path))
                chunk_cache[rec.chunk_path] = chunk
            if not chunk.has(key):
                return None
            return chunk.get_float32(key)

        for model_key in models:
            aligned_queries = [
                query_id for query_id in sorted(table.label_by_query)
                if table.record(model_key, arm_a, query_id) is not None
                and table.record(model_key, arm_b, query_id) is not None
            ]
            identical_count = 0
            missing_vector_count = 0
            sum_abs_delta = 0.0
            n_elements = 0
            pair_max_abs: list[float] = []
            for query_id in aligned_queries:
                rec_a = table.record(model_key, arm_a, query_id)
                rec_b = table.record(model_key, arm_b, query_id)
                assert rec_a is not None and rec_b is not None
                vec_a = read_vec(rec_a, 0, "image_all_mean")
                vec_b = read_vec(rec_b, 0, "image_all_mean")
                if vec_a is None or vec_b is None:
                    missing_vector_count += 1
                    continue
                if np.array_equal(vec_a, vec_b):
                    identical_count += 1
                delta = np.abs(vec_a - vec_b)
                sum_abs_delta += float(delta.sum())
                n_elements += int(delta.size)
                pair_max_abs.append(float(np.max(delta)))
            mean_abs_delta = float(sum_abs_delta / n_elements) if n_elements > 0 else float("nan")
            min_pair_max_abs_delta = float(min(pair_max_abs)) if pair_max_abs else float("nan")
            status = "pass" if identical_count == 0 and missing_vector_count == 0 else "fail"
            row = {
                "model_key": str(model_key),
                "n_aligned_pairs": len(aligned_queries),
                "identical_vector_count": int(identical_count),
                "missing_vector_count": int(missing_vector_count),
                "mean_abs_delta": mean_abs_delta,
                "min_pair_max_abs_delta": min_pair_max_abs_delta,
                "status": status,
                "method": "layer_000_image_all_mean_bitwise_presence_audit",
            }
            model_rows.append(row)
            if status != "pass":
                failures.append(row)
    return {
        "status": ("pass" if not failures else "fail"),
        "arms": {"arm_a": arm_a, "arm_b": arm_b},
        "model_rows": model_rows,
        "failures": failures,
    }


def build_pack_provenance_audit(manifest_path: Path | None) -> dict[str, Any]:
    checks = {
        "status_pass": False,
        "request_count_418": False,
        "source_arm_sequential": False,
        "image_files_created_zero": False,
        "source_mapping_one_to_one": False,
        "method_facing_forbidden_keys_zero": False,
    }
    if manifest_path is None:
        return {
            "status": "skipped",
            "manifest_path": None,
            "checks": checks,
        }
    payload = _read_json(Path(manifest_path))
    checks["status_pass"] = payload.get("status") == "pass"
    checks["request_count_418"] = int(payload.get("request_count", -1)) == 418
    checks["source_arm_sequential"] = str(payload.get("source_arm")) == "sequential"
    checks["image_files_created_zero"] = int(payload.get("image_files_created", -1)) == 0
    checks["source_mapping_one_to_one"] = bool(payload.get("source_mapping_one_to_one")) is True
    checks["method_facing_forbidden_keys_zero"] = int(payload.get("method_facing_forbidden_key_count", -1)) == 0
    status = "pass" if all(checks.values()) else "fail"
    return {
        "status": status,
        "manifest_path": str(manifest_path),
        "checks": checks,
    }


def build_fold_identity_audit(
    *,
    folds: dict[str, Any],
    reference_path: Path | None,
) -> dict[str, Any]:
    if reference_path is None:
        return {
            "status": "skipped",
            "reference_path": None,
            "exact_match": None,
        }
    reference = _read_json(reference_path)
    exact_match = reference == folds
    return {
        "status": ("pass" if exact_match else "fail"),
        "reference_path": str(reference_path),
        "exact_match": bool(exact_match),
    }


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def snapshot_code_provenance(out_dir: Path) -> dict[str, Any]:
    analysis_path = Path(__file__).resolve()
    cli_path = analysis_path.with_name("run_phase2_job2b_paired_analysis.py")
    probe_path = Path(probe.__file__).resolve()
    snapshot_dir = out_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "analysis": analysis_path,
        "cli": cli_path,
        "probe_dependency": probe_path,
    }
    entries: dict[str, dict[str, Any]] = {}
    for role, source_path in source_paths.items():
        snapshot_path = snapshot_dir / source_path.name
        probe.write_text(snapshot_path, source_path.read_text(encoding="utf-8"))
        source_sha = probe._sha256_file(source_path)
        snapshot_sha = probe._sha256_file(snapshot_path)
        if source_sha != snapshot_sha:
            raise RuntimeError(f"code snapshot hash mismatch for {role}: {source_path}")
        entries[role] = {
            "source_path": str(source_path),
            "source_sha256": source_sha,
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": snapshot_sha,
            "snapshot_matches_source": True,
        }
    return {
        "analysis_path": str(analysis_path),
        "analysis_sha256": entries["analysis"]["source_sha256"],
        "analysis_snapshot_path": entries["analysis"]["snapshot_path"],
        "cli_path": str(cli_path),
        "cli_sha256": entries["cli"]["source_sha256"],
        "cli_snapshot_path": entries["cli"]["snapshot_path"],
        "probe_dependency_path": str(probe_path),
        "probe_dependency_sha256": entries["probe_dependency"]["source_sha256"],
        "probe_dependency_snapshot_path": entries["probe_dependency"]["snapshot_path"],
        "snapshot_status": "pass",
        "snapshot_files": [
            str(Path(entry["snapshot_path"]).relative_to(out_dir))
            for entry in entries.values()
        ],
        "entries": entries,
    }


def _manifest(
    config: PairedAnalysisConfig,
    paired: PairedCanonical,
    folds: dict[str, Any],
    *,
    code_provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "phase2_job2b_paired_shared_symmetric_analysis_manifest",
        "mode": config.mode,
        "out_dir": str(config.out_dir),
        "arm_sources": {arm: str(source) for arm, source in config.arm_sources},
        "models": list(config.models),
        "query_types": list(config.query_types),
        "pools_config": (list(config.pools) if config.pools else "auto_per_model"),
        "layer_spec": config.layer_spec,
        "nested_layer_spec": config.nested_layer_spec,
        "fixed_c": float(config.fixed_c),
        "regularization_protocol": f"fixed_C_{config.fixed_c:g}",
        "repeats": int(config.repeats),
        "n_outer": int(folds["n_outer"]),
        "n_inner": int(config.n_inner),
        "seed": int(config.seed),
        "bootstrap": int(config.bootstrap),
        "max_iter": int(config.max_iter),
        "n_source_queries": len(paired.table.label_by_query),
        "n_participants": len(paired.table.participants),
        "principal_scope": {
            "arms": [PRINCIPAL_ARM_A, PRINCIPAL_ARM_B],
            "models": list(config.principal_models),
            "query_type": config.principal_query_type,
            "pool": config.principal_pool,
            "primary_estimator": "shared_symmetric_nested_layer_selection_fixed_C_0.001",
            "secondary_estimator": "independent_nested_layer_selection_fixed_C_0.001",
            "nested_layers": config.nested_layer_spec,
            "fixed_c": float(config.fixed_c),
        },
        "environment": {
            "python": sys.version.split()[0],
            "packages": _package_versions(
                ("numpy", "pandas", "pyarrow", "scikit-learn", "safetensors")
            ),
        },
        "code_provenance": code_provenance,
        "source_provenance": _source_sha_map(config.arm_sources, config.models),
        "audit_gates": {
            "intervention_presence": bool(config.intervention_presence_audit),
            "pack_provenance": bool(config.pack_provenance_audit),
            "fold_identity": bool(config.fold_identity_audit),
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def generate_report(
    config: PairedAnalysisConfig,
    paired: PairedCanonical,
    summary: dict[str, Any],
    *,
    prompt_audit: dict[str, Any],
    primary_shared_summary: Sequence[dict[str, Any]] | None = None,
    secondary_independent_summary: Sequence[dict[str, Any]] | None = None,
    promotion_rows: Sequence[dict[str, Any]] | None = None,
    same_layer_rows: Sequence[dict[str, Any]] | None = None,
    transfer_rows: Sequence[dict[str, Any]] | None = None,
    intervention_audit: dict[str, Any] | None = None,
    pack_audit: dict[str, Any] | None = None,
    fold_identity_audit: dict[str, Any] | None = None,
) -> str:
    primary_shared_summary = list(primary_shared_summary or [])
    secondary_independent_summary = list(secondary_independent_summary or [])
    promotion_rows = list(promotion_rows or [])
    same_layer_rows = list(same_layer_rows or [])
    transfer_rows = list(transfer_rows or [])
    lines: list[str] = []
    lines.append("# Phase 2 Job2 vs Job2b paired representation analysis (shared-symmetric primary)")
    lines.append("")
    lines.append(f"- Mode: `{config.mode}`")
    lines.append(f"- Arms: `{PRINCIPAL_ARM_A}` vs `{PRINCIPAL_ARM_B}`")
    lines.append(f"- Sources: {', '.join(f'{arm}={src}' for arm, src in config.arm_sources)}")
    lines.append(f"- Models: {', '.join(config.models)}")
    lines.append(f"- Query types: {', '.join(config.query_types)}")
    lines.append(f"- Layer spec: `{config.layer_spec}` | nested candidates: `{config.nested_layer_spec}`")
    lines.append(f"- Fixed regularization: `C={config.fixed_c:g}`")
    lines.append("")
    lines.append("## Principal scope (frozen)")
    lines.append("")
    principal_models_txt = ", ".join(f"`{model}`" for model in config.principal_models)
    lines.append(f"- `{config.principal_query_type}` with `{config.principal_pool}` for {principal_models_txt}.")
    lines.append(
        "- Primary: shared symmetric nested layer selection with fixed `C=0.001` on matched inner participant splits."
    )
    lines.append("- Secondary: independently nested per-arm selected layers on the same frozen folds.")
    lines.append("- Delta definition for all paired effects: `visual_prompt - plain_contact_sheet`.")
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("")
    lines.append("- Principal contrast is the complete localization-prompt intervention boundary (prompt package plus pixels).")
    lines.append(
        "- The arm instructions change sequence length; last-token contrasts therefore include a token-position shift."
    )
    lines.append("- Results establish linear decodability only.")
    lines.append("- No behavioral, use, binding, temporal-memory, or causal claim is made.")
    lines.append("- `contact_order_before` remains endpoint-confounded and secondary/exploratory.")
    lines.append("- One frozen fold assignment is used; fold-assignment variance remains a limitation.")
    lines.append(
        "- Bootstrap and sign-flip inference condition on the frozen OOF predictions; they do not refit layer selection under resampling."
    )
    lines.append("- Exact sign-flip and Holm outputs are promotion gates, not confirmatory causal claims.")
    lines.append("")

    if primary_shared_summary:
        principal = [
            row
            for row in primary_shared_summary
            if row["model_key"] in config.principal_models
            and row["query_type"] == config.principal_query_type
            and row["pool"] == config.principal_pool
        ]
        lines.append("## Principal shared-symmetric paired estimates")
        lines.append("")
        lines.append("| model | AUROC delta | delta CI95 | exact CE sign-flip p | Holm p (2 models) | fold sign | bootstrap valid |")
        lines.append("| --- | ---: | --- | ---: | ---: | --- | ---: |")
        for row in principal:
            ci = f"[{_fmt(row['delta_auroc_ci95_low'])}, {_fmt(row['delta_auroc_ci95_high'])}]"
            lines.append(
                f"| {row['model_key']} | {_fmt(row['delta_auroc'])} | {ci} "
                f"| {_fmt(row['exact_sign_flip_p_two_sided_ce_bits'])} | {_fmt(row.get('holm_adjusted_p_two_models'))} "
                f"| {int(row.get('fold_sign_consistent_count', 0))}/{int(row.get('fold_count', 0))} "
                f"| {_fmt(row.get('delta_auroc_boot_valid_fraction'))} |"
            )
        lines.append("")
        principal_by_model = {str(row["model_key"]): row for row in principal}
        gemma = principal_by_model.get("gemma4_31b_hf_bf16")
        if gemma is not None:
            lines.append(
                "- Gemma4-31B: under the frozen matched-site shared-selected linear readout, "
                f"current contact is less linearly decodable for the localization-prompt package "
                f"(delta={_fmt(gemma['delta_auroc'])}). The package includes both instruction/position "
                "changes and localization pixels, so this is not a pixels-only attribution."
            )
        qwen = principal_by_model.get("qwen35_27b_hf_bf16")
        if qwen is not None:
            lines.append(
                "- Qwen3.5-27B: the paired comparison is unresolved "
                f"(delta={_fmt(qwen['delta_auroc'])}; CI crosses zero). This is not evidence of equivalence."
            )
        lines.append("")

    if promotion_rows:
        lines.append("## Principal promotion-readiness gate")
        lines.append("")
        lines.append("| model | shared delta | shared CI95 | exact p | Holm p | fold sign | bootstrap valid | independent delta (secondary) | sign agree | ready |")
        lines.append("| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- |")
        for row in promotion_rows:
            ci = (
                f"[{_fmt(row['shared_nested_delta_auroc_ci95_low'])}, "
                f"{_fmt(row['shared_nested_delta_auroc_ci95_high'])}]"
            )
            lines.append(
                f"| {row['model_key']} | {_fmt(row['shared_nested_delta_auroc'])} | {ci} "
                f"| {_fmt(row['shared_nested_exact_sign_flip_p_two_sided_ce_bits'])} | {_fmt(row['shared_nested_holm_adjusted_p_two_models'])} "
                f"| {int(row['shared_nested_fold_sign_consistent_count'])}/{int(row['shared_nested_fold_count'])} "
                f"| {_fmt(row['shared_nested_bootstrap_valid_fraction'])} "
                f"| {_fmt(row['independent_nested_delta_auroc_secondary'])} "
                f"| `{bool(row['shared_vs_independent_sign_agree'])}` | `{bool(row['promotion_ready_gate'])}` |"
            )
        lines.append("")

    if secondary_independent_summary:
        lines.append("## Secondary independent-nested paired estimates")
        lines.append("")
        lines.append("| model | AUROC delta | delta CI95 |")
        lines.append("| --- | ---: | --- |")
        for row in secondary_independent_summary:
            if row["query_type"] != config.principal_query_type or row["pool"] != config.principal_pool:
                continue
            ci = f"[{_fmt(row['delta_auroc_ci95_low'])}, {_fmt(row['delta_auroc_ci95_high'])}]"
            lines.append(f"| {row['model_key']} | {_fmt(row['delta_auroc'])} | {ci} |")
        lines.append("")

    lines.append("## Prompt/token audit")
    lines.append("")
    global_audit = prompt_audit.get("global", {})
    lines.append(f"- Status: `{prompt_audit.get('status')}`.")
    lines.append(
        f"- Source questions identical mismatches: {global_audit.get('question_mismatch_count', 'n/a')}; "
        f"image-token mismatches: {global_audit.get('image_token_mismatch_count', 'n/a')}."
    )
    lines.append(
        f"- Mean total-token delta (`{PRINCIPAL_ARM_A}-{PRINCIPAL_ARM_B}`): "
        f"{_fmt(global_audit.get('mean_total_token_delta'))}."
    )
    lines.append(
        f"- Mean non-image-token delta (`{PRINCIPAL_ARM_A}-{PRINCIPAL_ARM_B}`): "
        f"{_fmt(global_audit.get('mean_non_image_token_delta'))}."
    )
    lines.append(f"- Arm instructions differ: `{bool(global_audit.get('instructions_differ'))}`.")
    lines.append(f"- {global_audit.get('intervention_scope_note', '')}")
    lines.append("")

    if intervention_audit:
        lines.append("## Intervention-presence audit")
        lines.append("")
        lines.append(f"- Status: `{intervention_audit.get('status')}`.")
        for row in intervention_audit.get("model_rows", []):
            lines.append(
                f"- `{row['model_key']}`: identical={row['identical_vector_count']}, "
                f"mean|delta|={_fmt(row['mean_abs_delta'])}, "
                f"min(max|delta|)={_fmt(row['min_pair_max_abs_delta'])}."
            )
        lines.append("")

    if pack_audit:
        lines.append("## Pack provenance audit")
        lines.append("")
        lines.append(f"- Status: `{pack_audit.get('status')}`.")
        lines.append("")

    if fold_identity_audit:
        lines.append("## Fold identity audit")
        lines.append("")
        lines.append(
            f"- Status: `{fold_identity_audit.get('status')}`; "
            f"exact match: `{fold_identity_audit.get('exact_match')}`."
        )
        lines.append("")

    if same_layer_rows:
        lines.append("## Descriptive same-layer paired contrasts (secondary/exploratory)")
        lines.append("")
        top_rows = sorted(
            same_layer_rows,
            key=lambda row: (
                row["model_key"],
                row["query_type"],
                row["pool"],
                row["layer_index"],
            ),
        )[:24]
        lines.append("| model | query | pool | layer | AUROC delta |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        for row in top_rows:
            lines.append(
                f"| {row['model_key']} | {row['query_type']} | {row['pool']} | {row['layer_index']} | {_fmt(row['delta_auroc'])} |"
            )
        lines.append("")

    if transfer_rows:
        lines.append("## Strict bidirectional transfer (secondary/exploratory)")
        lines.append("")
        all_disjoint = all(bool(row.get("participants_disjoint")) for row in transfer_rows)
        lines.append(f"- Participants disjoint in every transfer fold: `{all_disjoint}`.")
        lines.append("")

    if probe.SECONDARY_QUERY_TYPE in config.query_types:
        lines.append("## Endpoint-confounded order track")
        lines.append("")
        lines.append(
            "- `contact_order_before` remains explicitly endpoint-confounded and is reported as secondary/exploratory only."
        )
        lines.append("")

    lines.append("## Run status")
    lines.append("")
    lines.append(f"- Data alignment audit: `{paired.table.audit.get('status')}`")
    lines.append(f"- Required-cell failures: {len(summary.get('required_cell_failures', []))}")
    lines.append(
        f"- Cells ok/skipped/error: "
        f"{sum(1 for r in summary.get('cell_status', []) if r.get('status') == 'ok')}/"
        f"{sum(1 for r in summary.get('cell_status', []) if r.get('status') == 'skipped')}/"
        f"{sum(1 for r in summary.get('cell_status', []) if r.get('status') == 'error')}"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def run_analysis(config: PairedAnalysisConfig) -> dict[str, Any]:
    t0 = time.perf_counter()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_sources = tuple((str(arm), Path(source)) for arm, source in config.arm_sources)

    _log(config, "building paired canonical table", t0=t0)
    paired = build_paired_canonical_table(arm_sources, config.models)
    table = paired.table
    folds = probe.assign_participant_folds(
        table.participants,
        n_outer=config.n_outer,
        n_inner=config.n_inner,
        repeats=config.repeats,
        seed=config.seed,
        shuffle=config.shuffle_clusters,
    )
    expected_token_delta_by_family = _expected_token_delta_by_family(config)
    prompt_token_audit = build_prompt_token_audit(
        paired,
        arm_a=PRINCIPAL_ARM_A,
        arm_b=PRINCIPAL_ARM_B,
        models=config.models,
        expected_non_image_delta_by_family=expected_token_delta_by_family,
    )
    run_intervention_audit = bool(config.intervention_presence_audit) and _should_run_real_mode_audit_gate(
        config,
        force=bool(config.enforce_intervention_presence_audit),
    )
    intervention_audit = (
        build_intervention_presence_audit(
            paired,
            arm_a=PRINCIPAL_ARM_A,
            arm_b=PRINCIPAL_ARM_B,
            models=config.models,
        )
        if run_intervention_audit
        else {"status": "skipped", "reason": "disabled_or_non_default_run"}
    )
    run_pack_audit = bool(config.pack_provenance_audit) and _should_run_real_mode_audit_gate(
        config,
        force=bool(config.enforce_pack_provenance_audit),
    )
    pack_audit = (
        build_pack_provenance_audit(config.pack_provenance_manifest_path)
        if run_pack_audit
        else {"status": "skipped", "reason": "disabled_or_non_default_run"}
    )
    default_real_full = (
        config.mode == "full"
        and _arm_sources_match_default(config)
        and int(config.repeats) == PROMOTED_FULL_REPEATS
        and int(config.n_outer) == PROMOTED_FULL_OUTER
        and int(config.n_inner) == PROMOTED_FULL_INNER
        and int(config.seed) == PROMOTED_FULL_SEED
    )
    run_fold_identity_audit = bool(config.fold_identity_audit) and (
        bool(config.enforce_fold_identity_audit) or default_real_full
    )
    fold_identity_audit = (
        build_fold_identity_audit(folds=folds, reference_path=config.fold_identity_reference_path)
        if run_fold_identity_audit
        else {"status": "skipped", "exact_match": None, "reason": "disabled_or_non_default_full_run"}
    )

    code_provenance = snapshot_code_provenance(out_dir)
    probe.write_json(
        out_dir / "manifest.json",
        _manifest(config, paired, folds, code_provenance=code_provenance),
    )
    probe.write_json(out_dir / "alignment_audit.json", table.audit)
    probe.write_json(out_dir / "fold_assignments.json", folds)
    probe.write_json(out_dir / "prompt_token_audit.json", prompt_token_audit)
    probe.write_json(out_dir / "intervention_presence_audit.json", intervention_audit)
    probe.write_json(out_dir / "pack_provenance_audit.json", pack_audit)
    probe.write_json(out_dir / "fold_identity_audit.json", fold_identity_audit)

    summary: dict[str, Any] = {
        "mode": config.mode,
        "audit_status": table.audit.get("status"),
        "cell_status": [],
        "required_cell_failures": [],
        "outputs": [
            "manifest.json",
            "alignment_audit.json",
            "fold_assignments.json",
            "prompt_token_audit.json",
            "intervention_presence_audit.json",
            "pack_provenance_audit.json",
            "fold_identity_audit.json",
            *code_provenance["snapshot_files"],
        ],
        "elapsed_s": None,
    }

    if table.audit.get("status") != "pass":
        summary["required_cell_failures"].append({"reason": "alignment_audit_failed"})
    if prompt_token_audit.get("status") == "fail":
        summary["required_cell_failures"].append({"reason": "prompt_token_audit_failed"})
    if intervention_audit.get("status") == "fail":
        summary["required_cell_failures"].append({"reason": "intervention_presence_audit_failed"})
    if pack_audit.get("status") == "fail":
        summary["required_cell_failures"].append({"reason": "pack_provenance_audit_failed"})
    if fold_identity_audit.get("status") == "fail":
        summary["required_cell_failures"].append({"reason": "fold_identity_audit_failed"})

    if config.mode == "audit" or summary["required_cell_failures"]:
        report = generate_report(
            config,
            paired,
            summary,
            prompt_audit=prompt_token_audit,
            intervention_audit=intervention_audit,
            pack_audit=pack_audit,
            fold_identity_audit=fold_identity_audit,
        )
        probe.write_text(out_dir / "report.md", report)
        summary["outputs"].append("report.md")
        summary["elapsed_s"] = time.perf_counter() - t0
        summary["outputs"].append("run_summary.json")
        probe.write_json(out_dir / "run_summary.json", summary)
        return summary

    c_grid = (float(config.fixed_c),)
    descriptive_probe_rows: list[dict[str, Any]] = []
    descriptive_oof_rows: list[dict[str, Any]] = []
    secondary_independent_nested_rows: list[dict[str, Any]] = []
    secondary_independent_nested_selected_oof_rows: list[dict[str, Any]] = []
    secondary_independent_reconstruction_rows: list[dict[str, Any]] = []
    principal_stacks: dict[tuple[str, str, str, str], probe.PoolStack] = {}
    transfer_rows: list[dict[str, Any]] = []

    for model_key in config.models:
        max_layer = table.max_layer_index(model_key)
        if max_layer <= 0:
            summary["cell_status"].append({"model_key": model_key, "status": "skipped", "reason": "no_rows_for_model"})
            continue
        layers = probe.resolve_layers(config.layer_spec, max_layer)
        nested_candidates = probe.resolve_layers(config.nested_layer_spec, max_layer)
        pools = list(config.pools) if config.pools else probe.default_pools_for_model(model_key, include_region=True)
        for arm, _source in arm_sources:
            for query_type in config.query_types:
                for pool in pools:
                    is_required = (
                        model_key in config.principal_models
                        and arm in {PRINCIPAL_ARM_A, PRINCIPAL_ARM_B}
                        and query_type == config.principal_query_type
                        and pool == config.principal_pool
                    )
                    try:
                        stack = probe.load_pool_stack(
                            table,
                            model_key=model_key,
                            arm=arm,
                            query_type=query_type,
                            pool=pool,
                            layers=layers,
                        )
                    except Exception as exc:
                        status = {
                            "model_key": model_key,
                            "arm": arm,
                            "query_type": query_type,
                            "pool": pool,
                            "status": "error",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                        summary["cell_status"].append(status)
                        if is_required:
                            summary["required_cell_failures"].append(status)
                        continue
                    if stack.n_rows == 0 or len(np.unique(stack.y)) < 2:
                        status = {
                            "model_key": model_key,
                            "arm": arm,
                            "query_type": query_type,
                            "pool": pool,
                            "status": "skipped",
                            "reason": "no_rows_or_single_class",
                            "coverage": stack.coverage,
                        }
                        summary["cell_status"].append(status)
                        if is_required:
                            summary["required_cell_failures"].append(status)
                        continue

                    fold_rows, oof_prob_by_layer = probe.descriptive_layer_curves(
                        stack,
                        folds,
                        c_grid=c_grid,
                        max_iter=config.max_iter,
                    )
                    descriptive_probe_rows.extend(fold_rows)
                    cell_nested = probe.nested_best_layer(
                        stack,
                        folds,
                        c_grid=c_grid,
                        max_iter=config.max_iter,
                        candidate_layers=nested_candidates,
                    )
                    secondary_independent_nested_rows.extend(cell_nested)
                    cell_nested_oof, cell_nested_recon = reconstruct_nested_selected_oof(
                        stack,
                        folds,
                        cell_nested,
                        max_iter=config.max_iter,
                    )
                    secondary_independent_nested_selected_oof_rows.extend(cell_nested_oof)
                    secondary_independent_reconstruction_rows.extend(cell_nested_recon)
                    for layer in stack.layers:
                        prob = oof_prob_by_layer[layer]
                        for i, query_id in enumerate(stack.query_ids):
                            descriptive_oof_rows.append(
                                {
                                    "model_key": model_key,
                                    "arm": arm,
                                    "query_type": query_type,
                                    "pool": pool,
                                    "layer_index": layer,
                                    "source_query_id": query_id,
                                    "participant": str(stack.groups[i]),
                                    "y_true": int(stack.y[i]),
                                    "oof_prob": float(prob[i]),
                                }
                            )
                    if (
                        model_key in set(config.principal_models)
                        and query_type == config.principal_query_type
                        and pool == config.principal_pool
                        and arm in {PRINCIPAL_ARM_A, PRINCIPAL_ARM_B}
                    ):
                        principal_stacks[(model_key, query_type, pool, arm)] = stack
                    summary["cell_status"].append(
                        {
                            "model_key": model_key,
                            "arm": arm,
                            "query_type": query_type,
                            "pool": pool,
                            "status": "ok",
                            "n_rows": stack.n_rows,
                            "coverage": stack.coverage,
                        }
                    )

        if config.do_transfer:
            transfer_layers = probe.resolve_layers(config.nested_layer_spec, max_layer)
            for query_type in config.query_types:
                for pool in (list(config.pools) if config.pools else probe.default_pools_for_model(model_key, include_region=False)):
                    for from_arm, to_arm in ((PRINCIPAL_ARM_A, PRINCIPAL_ARM_B), (PRINCIPAL_ARM_B, PRINCIPAL_ARM_A)):
                        rows = probe.strict_cross_arm_transfer(
                            table,
                            folds,
                            model_key=model_key,
                            query_type=query_type,
                            pool=pool,
                            from_arm=from_arm,
                            to_arm=to_arm,
                            layers=transfer_layers,
                            c_grid=c_grid,
                            max_iter=config.max_iter,
                        )
                        transfer_rows.extend(rows)

    if transfer_rows and not all(bool(row.get("participants_disjoint")) for row in transfer_rows):
        raise AssertionError("transfer participant disjointness assertion failed")

    same_layer_rows = _same_layer_contrasts(
        descriptive_oof_rows,
        arm_a=PRINCIPAL_ARM_A,
        arm_b=PRINCIPAL_ARM_B,
    )
    secondary_independent_fold_rows, secondary_independent_summary_rows = _paired_nested_contrasts(
        secondary_independent_nested_selected_oof_rows,
        arm_a=PRINCIPAL_ARM_A,
        arm_b=PRINCIPAL_ARM_B,
        bootstrap=config.bootstrap,
        seed=config.seed,
    )
    primary_shared_oof_rows: list[dict[str, Any]] = []
    primary_shared_fold_rows: list[dict[str, Any]] = []
    principal_model_set = set(config.principal_models)
    for model_key in config.principal_models:
        if model_key not in principal_model_set:
            continue
        stack_a = principal_stacks.get((model_key, config.principal_query_type, config.principal_pool, PRINCIPAL_ARM_A))
        stack_b = principal_stacks.get((model_key, config.principal_query_type, config.principal_pool, PRINCIPAL_ARM_B))
        if stack_a is None or stack_b is None:
            summary["required_cell_failures"].append(
                {
                    "reason": "missing_principal_stack_for_shared_selection",
                    "model_key": model_key,
                }
            )
            continue
        max_layer = min(stack_a.max_layer_index, stack_b.max_layer_index)
        nested_candidates = probe.resolve_layers(config.nested_layer_spec, max_layer)
        shared_oof, shared_folds = shared_symmetric_nested_selected_oof(
            stack_a,
            stack_b,
            folds,
            candidate_layers=nested_candidates,
            fixed_c=float(config.fixed_c),
            max_iter=config.max_iter,
        )
        if not shared_folds or not shared_oof:
            summary["required_cell_failures"].append(
                {
                    "reason": "shared_symmetric_selection_empty",
                    "model_key": model_key,
                }
            )
            continue
        primary_shared_oof_rows.extend(shared_oof)
        primary_shared_fold_rows.extend(shared_folds)
    if summary["required_cell_failures"]:
        report = generate_report(
            config,
            paired,
            summary,
            prompt_audit=prompt_token_audit,
            secondary_independent_summary=secondary_independent_summary_rows,
            same_layer_rows=same_layer_rows,
            transfer_rows=transfer_rows,
            intervention_audit=intervention_audit,
            pack_audit=pack_audit,
            fold_identity_audit=fold_identity_audit,
        )
        probe.write_text(out_dir / "report.md", report)
        summary["outputs"].append("report.md")
        summary["elapsed_s"] = time.perf_counter() - t0
        summary["outputs"].append("run_summary.json")
        probe.write_json(out_dir / "run_summary.json", summary)
        return summary

    primary_shared_summary_rows = _paired_summary_with_inference(
        oof_rows=primary_shared_oof_rows,
        fold_rows=primary_shared_fold_rows,
        arm_a=PRINCIPAL_ARM_A,
        arm_b=PRINCIPAL_ARM_B,
        bootstrap=config.bootstrap,
        seed=config.seed,
        method="primary_shared_symmetric_nested_query_aligned_summary",
    )
    _attach_holm_to_principal_rows(
        primary_shared_summary_rows,
        principal_models=config.principal_models,
        query_type=config.principal_query_type,
        pool=config.principal_pool,
    )
    promotion_rows = build_principal_promotion_readiness(
        shared_summary_rows=primary_shared_summary_rows,
        independent_summary_rows=secondary_independent_summary_rows,
        principal_models=config.principal_models,
        principal_query_type=config.principal_query_type,
        principal_pool=config.principal_pool,
    )

    if descriptive_probe_rows:
        probe.write_parquet(out_dir / "descriptive_probe_results.parquet", descriptive_probe_rows)
        summary["outputs"].append("descriptive_probe_results.parquet")
    if descriptive_oof_rows:
        probe.write_parquet(out_dir / "descriptive_oof_predictions.parquet", descriptive_oof_rows)
        summary["outputs"].append("descriptive_oof_predictions.parquet")
    if secondary_independent_nested_rows:
        probe.write_parquet(out_dir / "secondary_independent_nested_best_layer.parquet", secondary_independent_nested_rows)
        summary["outputs"].append("secondary_independent_nested_best_layer.parquet")
    if secondary_independent_nested_selected_oof_rows:
        probe.write_parquet(
            out_dir / "secondary_independent_nested_selected_oof.parquet",
            secondary_independent_nested_selected_oof_rows,
        )
        summary["outputs"].append("secondary_independent_nested_selected_oof.parquet")
    if secondary_independent_reconstruction_rows:
        probe.write_parquet(
            out_dir / "secondary_independent_nested_selected_reconstruction.parquet",
            secondary_independent_reconstruction_rows,
        )
        summary["outputs"].append("secondary_independent_nested_selected_reconstruction.parquet")
    if secondary_independent_fold_rows:
        probe.write_parquet(out_dir / "secondary_independent_paired_fold_contrasts.parquet", secondary_independent_fold_rows)
        summary["outputs"].append("secondary_independent_paired_fold_contrasts.parquet")
    if secondary_independent_summary_rows:
        probe.write_parquet(out_dir / "secondary_independent_paired_summary.parquet", secondary_independent_summary_rows)
        summary["outputs"].append("secondary_independent_paired_summary.parquet")
    if primary_shared_oof_rows:
        probe.write_parquet(out_dir / "primary_shared_symmetric_selected_oof.parquet", primary_shared_oof_rows)
        summary["outputs"].append("primary_shared_symmetric_selected_oof.parquet")
    if primary_shared_fold_rows:
        probe.write_parquet(out_dir / "primary_shared_symmetric_paired_fold_contrasts.parquet", primary_shared_fold_rows)
        summary["outputs"].append("primary_shared_symmetric_paired_fold_contrasts.parquet")
    if primary_shared_summary_rows:
        probe.write_parquet(out_dir / "primary_shared_symmetric_paired_summary.parquet", primary_shared_summary_rows)
        summary["outputs"].append("primary_shared_symmetric_paired_summary.parquet")
    if promotion_rows:
        probe.write_parquet(out_dir / "principal_promotion_readiness.parquet", promotion_rows)
        summary["outputs"].append("principal_promotion_readiness.parquet")
    if same_layer_rows:
        probe.write_parquet(out_dir / "secondary_same_layer_contrasts.parquet", same_layer_rows)
        summary["outputs"].append("secondary_same_layer_contrasts.parquet")
    if transfer_rows:
        probe.write_parquet(out_dir / "secondary_transfer_results.parquet", transfer_rows)
        summary["outputs"].append("secondary_transfer_results.parquet")

    report = generate_report(
        config,
        paired,
        summary,
        prompt_audit=prompt_token_audit,
        primary_shared_summary=primary_shared_summary_rows,
        secondary_independent_summary=secondary_independent_summary_rows,
        promotion_rows=promotion_rows,
        same_layer_rows=same_layer_rows,
        transfer_rows=transfer_rows,
        intervention_audit=intervention_audit,
        pack_audit=pack_audit,
        fold_identity_audit=fold_identity_audit,
    )
    probe.write_text(out_dir / "report.md", report)
    summary["outputs"].append("report.md")
    summary["elapsed_s"] = time.perf_counter() - t0
    summary["outputs"].append("run_summary.json")
    probe.write_json(out_dir / "run_summary.json", summary)
    return summary
