#!/usr/bin/env python3
"""Offline representation-probe analysis for compact activation capture.

This module is the CPU-only, streaming analysis library behind
``run_phase2_job2_representation_probes.py``. It never mutates raw captures and it
never materializes a persistent fp32/fp16 feature cache. Tensors are read on
demand directly from the safetensors chunk files via an mmap-backed reader, and
at most one (model, pool)-across-layers stack or one layer matrix is held in RAM
at a time.

Scientific contract (enforced in code, restated in the report):

- The analyzer establishes *linear decodability of hand-target contact from
  pooled residual summaries* only. A null probe never means "information is
  absent"; a positive probe never proves use, routing, binding, or causality.
- The compact capture recorded forward passes only. There is no behavior, no
  decision-position state, and the last-prompt-position logits are JSON-format
  logits, so the readout side is structurally absent from this capture. This is
  stated explicitly.
- The primary query type is ``contact_state_current``. ``contact_order_before``
  is a separate secondary track and is never pooled into a headline probe.
- Every split is grouped by participant (one participant == one video here). No
  random row split is ever used. Fold assignments are frozen once and reused for
  all models, arms, features, and layers.

The heavy lifting (probes, nested selection, transfer, permutation, bootstrap)
is exposed as small, individually testable functions.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import struct
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE_RUN = REPO_ROOT / "runs" / "tamia" / "mech-job2-battery-capture" / "364962"

METHOD_NAMES = {
    "probe": "standardscaler_then_l2_logreg_lbfgs",
    "c_selection": "inner_grouped_cv_log_loss",
    "layer_selection": "nested_inner_grouped_cv_log_loss",
    "outer_split": "participant_grouped_kfold_shuffled",
    "inner_split": "participant_grouped_kfold",
    "bootstrap": "participant_cluster_bootstrap_over_oof_predictions",
    "permutation": "fixed_oof_max_over_layers_within_participant_label_permutation",
    "permutation_caveat": (
        "Exploratory fixed-score association diagnostic only: predictions were fit "
        "on true training labels and are not refit under the null. Do not use these "
        "p-values as confirmatory evidence."
    ),
    "object_identity_baseline": "tfidf_question_plus_target_label_grouped_outer_cv",
    "intercept_baseline": "train_fold_class_prior",
    "missingness_baseline": "region_pool_presence_indicator_coverage_audit",
    "transfer": "strict_cross_arm_train_outer_train_test_outer_test_disjoint_participants",
}

DEFAULT_MODEL_KEYS = (
    "qwen35_27b_hf_bf16",
    "gemma4_31b_hf_bf16",
    "gemma4_12b_hf_bf16",
    "gemma4_26b_a4b_hf_bf16",
    "qwen35_4b_hf_bf16",
)
DEFAULT_ARMS = ("visual_prompt", "single_frame", "crop", "sequential")
COMMON_POOLS = ("image_all_mean", "last_token", "text_tail_mean")
QWEN_REGION_POOLS = ("target_mean", "hand_mean", "background_mean")
DEFAULT_QUERY_TYPES = ("contact_state_current", "contact_order_before")
PRIMARY_QUERY_TYPE = "contact_state_current"
SECONDARY_QUERY_TYPE = "contact_order_before"

PRINCIPAL_CONTRAST = ("qwen35_27b_hf_bf16", "gemma4_31b_hf_bf16")
MOE_MODEL_KEYS = ("gemma4_26b_a4b_hf_bf16",)

DEFAULT_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2)
DEFAULT_N_OUTER = 5
DEFAULT_N_INNER = 4
DEFAULT_REPEATS = 10
DEFAULT_BOOTSTRAP = 2000
DEFAULT_PERMUTATIONS = 1000
DEFAULT_MAX_ITER = 300

_LN2 = math.log(2.0)
_PROB_EPS = 1e-6


# ---------------------------------------------------------------------------
# minimal safetensors IO (mmap reader + tiny writer for tests)
# ---------------------------------------------------------------------------


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    return header, 8 + header_len


class SafetensorsChunk:
    """mmap-backed reader that materializes only the requested tensors."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.header, self.data_start = _read_header(self.path)
        self._file = None
        self._mm: mmap.mmap | None = None

    def __enter__(self) -> "SafetensorsChunk":
        self._file = open(self.path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def keys(self) -> list[str]:
        return [key for key in self.header if key != "__metadata__"]

    def has(self, key: str) -> bool:
        return key in self.header and key != "__metadata__"

    def get_float32(self, key: str) -> np.ndarray:
        if self._mm is None:
            raise RuntimeError("SafetensorsChunk must be used as a context manager")
        meta = self.header[key]
        dtype = meta["dtype"]
        start, end = meta["data_offsets"]
        raw = self._mm[self.data_start + start : self.data_start + end]
        if dtype == "F32":
            arr = np.frombuffer(raw, dtype="<f4")
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif dtype == "F64":
            arr = np.frombuffer(raw, dtype="<f8").astype(np.float32)
        elif dtype == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        else:
            raise ValueError(f"unsupported safetensors dtype {dtype!r} for key {key!r}")
        return np.ascontiguousarray(arr.reshape(meta["shape"]), dtype=np.float32)


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], *, dtype: str = "F32") -> None:
    """Minimal safetensors writer (F32 or BF16). Used by tests to build corpora."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {}
    body = bytearray()
    offset = 0
    for key, value in tensors.items():
        array = np.ascontiguousarray(value)
        if dtype == "F32":
            data = array.astype("<f4", copy=False).tobytes()
        elif dtype == "BF16":
            u32 = array.astype(np.float32).view(np.uint32)
            data = (u32 >> 16).astype("<u2").tobytes()
        else:
            raise ValueError(f"unsupported write dtype {dtype!r}")
        header[key] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        body += data
        offset += len(data)
    header_bytes = json.dumps(header).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(body)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# atomic writers
# ---------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    _atomic_write_bytes(Path(path), (json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n").encode("utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not serializable: {type(value)!r}")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    """Write ``rows`` to parquet atomically. Returns True if a file was written."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return True


def write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(Path(path), text.encode("utf-8"))


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(source_run: Path, models: Sequence[str]) -> dict[str, Any]:
    source_run = Path(source_run)
    top_summary = source_run / "capture_summary.json"
    payload: dict[str, Any] = {
        "source_run": str(source_run),
        "capture_summary_sha256": _sha256_file(top_summary),
        "models": {},
    }
    for model_key in models:
        summary_path = source_run / model_key / "capture_summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = {}
        payload["models"][model_key] = {
            "capture_summary_sha256": _sha256_file(summary_path),
            "model_id": summary.get("model_id"),
            "revision": summary.get("revision"),
            "model_snapshot": summary.get("model_snapshot"),
            "dtype": summary.get("dtype"),
            "overall_status": summary.get("overall_status") or summary.get("status"),
        }
    return payload


# ---------------------------------------------------------------------------
# capture schema
# ---------------------------------------------------------------------------


@dataclass
class RowRecord:
    model_key: str
    arm: str
    request_id: str
    source_query_id: str
    participant: str
    video_id: str
    query_type: str
    label: int
    target_label: str
    question_text: str
    chunk_path: Path
    tensor_prefix: str
    image_token_count: int | None
    available_pools: tuple[str, ...]
    hook_modules: tuple[str, ...]
    max_layer_index: int

    def layer_key(self, pool: str, layer_index: int) -> str:
        return f"{self.tensor_prefix}__layer_{layer_index:03d}__{pool}"

    def hook_key(self, module: str) -> str:
        return f"{self.tensor_prefix}__hook__{module}__mean"


def model_family(model_key: str) -> str:
    key = model_key.lower()
    if key.startswith("qwen"):
        return "qwen"
    if key.startswith("gemma"):
        return "gemma"
    return "other"


def _bool_to_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return 1
        if low in ("false", "no", "0"):
            return 0
    return None


def iter_capture_index(capture_index_path: Path) -> Iterable[dict[str, Any]]:
    with open(capture_index_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_model_rows(source_run: Path, model_key: str) -> list[RowRecord]:
    capture_root = Path(source_run) / model_key / "capture"
    index_path = capture_root / "capture_index.jsonl"
    if not index_path.exists():
        return []
    records: list[RowRecord] = []
    for row in iter_capture_index(index_path):
        if row.get("status") != "pass":
            continue
        hidden = row.get("hidden_index") or {}
        label = _bool_to_label(hidden.get("expected_answer"))
        source_query_id = hidden.get("source_query_id")
        participant = hidden.get("source_tier1_participant_id") or hidden.get("source_video_id")
        if label is None or not source_query_id or not participant:
            continue
        tensor_path = row.get("tensor_path")
        if not tensor_path:
            continue
        mask_meta = row.get("mask_meta") or {}
        mask_counts = mask_meta.get("mask_token_counts") or {}
        hidden_capture = row.get("hidden_capture") or {}
        layer_indexes = hidden_capture.get("layer_indexes") or []
        max_layer = max(layer_indexes) if layer_indexes else 0
        query = row.get("query") or {}
        records.append(
            RowRecord(
                model_key=model_key,
                arm=str(row.get("arm")),
                request_id=str(row.get("request_id")),
                source_query_id=str(source_query_id),
                participant=str(participant),
                video_id=str(hidden.get("source_video_id") or participant),
                query_type=str(hidden.get("query_type") or "unknown"),
                label=int(label),
                target_label=str(hidden.get("target_label") or query.get("target_label") or ""),
                question_text=str(query.get("text") or ""),
                chunk_path=capture_root / tensor_path,
                tensor_prefix=str(row.get("tensor_prefix")),
                image_token_count=(int(mask_meta["image_token_count"]) if isinstance(mask_meta.get("image_token_count"), (int, float)) else None),
                available_pools=tuple(sorted(str(k) for k in mask_counts)),
                hook_modules=tuple(row.get("hook_module_names") or ()),
                max_layer_index=int(max_layer),
            )
        )
    return records


@dataclass
class CanonicalTable:
    source_run: Path
    models: tuple[str, ...]
    # (model_key, arm, source_query_id) -> RowRecord
    rows: dict[tuple[str, str, str], RowRecord] = field(default_factory=dict)
    # source_query_id -> canonical label / participant / query_type
    label_by_query: dict[str, int] = field(default_factory=dict)
    participant_by_query: dict[str, str] = field(default_factory=dict)
    query_type_by_query: dict[str, str] = field(default_factory=dict)
    target_by_query: dict[str, str] = field(default_factory=dict)
    question_by_query: dict[str, str] = field(default_factory=dict)
    participants: tuple[str, ...] = ()
    audit: dict[str, Any] = field(default_factory=dict)

    def record(self, model_key: str, arm: str, source_query_id: str) -> RowRecord | None:
        return self.rows.get((model_key, arm, source_query_id))

    def queries_for(self, query_type: str) -> list[str]:
        return sorted(q for q, qt in self.query_type_by_query.items() if qt == query_type)

    def max_layer_index(self, model_key: str) -> int:
        best = 0
        for (mk, _arm, _q), rec in self.rows.items():
            if mk == model_key:
                best = max(best, rec.max_layer_index)
        return best


def build_canonical_table(
    source_run: Path,
    models: Sequence[str],
    *,
    expected_arms: Sequence[str] | None = None,
) -> CanonicalTable:
    """Build the aligned (model x arm x source_query) row table and audit labels."""
    source_run = Path(source_run)
    table = CanonicalTable(source_run=source_run, models=tuple(models))
    # canonical label/participant/query_type are established from the first model
    # that carries a given source_query, then cross-checked against all others.
    label_votes: dict[str, set[int]] = {}
    participant_votes: dict[str, set[str]] = {}
    query_type_votes: dict[str, set[str]] = {}
    target_votes: dict[str, set[str]] = {}
    question_votes: dict[str, set[str]] = {}
    label_mismatches: list[dict[str, Any]] = []
    participant_mismatches: list[dict[str, Any]] = []
    query_type_mismatches: list[dict[str, Any]] = []
    target_mismatches: list[dict[str, Any]] = []
    question_mismatches: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, str]] = []
    per_model_arm_counts: dict[str, dict[str, int]] = {}
    missing_models: list[str] = []

    for model_key in models:
        records = load_model_rows(source_run, model_key)
        if not records:
            missing_models.append(model_key)
            continue
        for rec in records:
            row_key = (model_key, rec.arm, rec.source_query_id)
            if row_key in table.rows:
                duplicate_rows.append(
                    {
                        "model_key": model_key,
                        "arm": rec.arm,
                        "source_query_id": rec.source_query_id,
                    }
                )
                continue
            table.rows[row_key] = rec
            per_model_arm_counts.setdefault(model_key, {}).setdefault(rec.arm, 0)
            per_model_arm_counts[model_key][rec.arm] += 1
            label_votes.setdefault(rec.source_query_id, set()).add(rec.label)
            participant_votes.setdefault(rec.source_query_id, set()).add(rec.participant)
            query_type_votes.setdefault(rec.source_query_id, set()).add(rec.query_type)
            target_votes.setdefault(rec.source_query_id, set()).add(rec.target_label)
            question_votes.setdefault(rec.source_query_id, set()).add(rec.question_text)
            table.target_by_query.setdefault(rec.source_query_id, rec.target_label)
            table.question_by_query.setdefault(rec.source_query_id, rec.question_text)

    for query_id, labels in label_votes.items():
        table.label_by_query[query_id] = sorted(labels)[0]
        if len(labels) > 1:
            label_mismatches.append({"source_query_id": query_id, "labels": sorted(labels)})
    for query_id, parts in participant_votes.items():
        table.participant_by_query[query_id] = sorted(parts)[0]
        if len(parts) > 1:
            participant_mismatches.append({"source_query_id": query_id, "participants": sorted(parts)})
    for query_id, qts in query_type_votes.items():
        table.query_type_by_query[query_id] = sorted(qts)[0]
        if len(qts) > 1:
            query_type_mismatches.append(
                {"source_query_id": query_id, "query_types": sorted(qts)}
            )
    for query_id, targets in target_votes.items():
        if len(targets) > 1:
            target_mismatches.append(
                {"source_query_id": query_id, "target_labels": sorted(targets)}
            )
    for query_id, questions in question_votes.items():
        if len(questions) > 1:
            question_mismatches.append(
                {"source_query_id": query_id, "question_texts": sorted(questions)}
            )

    table.participants = tuple(sorted(set(table.participant_by_query.values())))

    query_type_counts: dict[str, int] = {}
    for qt in table.query_type_by_query.values():
        query_type_counts[qt] = query_type_counts.get(qt, 0) + 1

    # clip sharing between query types must be zero (separate tracks)
    query_type_participants: dict[str, set[str]] = {}
    for query_id, qt in table.query_type_by_query.items():
        query_type_participants.setdefault(qt, set()).add(table.participant_by_query[query_id])

    missing_cells: list[dict[str, Any]] = []
    if expected_arms is not None:
        expected_queries = set(table.label_by_query)
        for model_key in models:
            for arm in expected_arms:
                present = {
                    query_id
                    for (mk, row_arm, query_id) in table.rows
                    if mk == model_key and row_arm == arm
                }
                missing = sorted(expected_queries - present)
                extra = sorted(present - expected_queries)
                if missing or extra:
                    missing_cells.append(
                        {
                            "model_key": model_key,
                            "arm": str(arm),
                            "expected_rows": len(expected_queries),
                            "observed_rows": len(present),
                            "missing_count": len(missing),
                            "missing_examples": missing[:10],
                            "extra_count": len(extra),
                            "extra_examples": extra[:10],
                        }
                    )

    status = "pass"
    if (
        missing_models
        or label_mismatches
        or participant_mismatches
        or query_type_mismatches
        or target_mismatches
        or question_mismatches
        or duplicate_rows
        or missing_cells
    ):
        status = "fail"

    table.audit = {
        "status": status,
        "source_run": str(source_run),
        "models_requested": list(models),
        "models_missing": missing_models,
        "n_source_queries": len(table.label_by_query),
        "n_participants": len(table.participants),
        "participants": list(table.participants),
        "query_type_counts": query_type_counts,
        "per_model_arm_row_counts": per_model_arm_counts,
        "label_mismatches": label_mismatches,
        "participant_mismatches": participant_mismatches,
        "query_type_mismatches": query_type_mismatches,
        "target_mismatches": target_mismatches,
        "question_mismatches": question_mismatches,
        "duplicate_rows": duplicate_rows,
        "missing_cells": missing_cells,
        "expected_arms": list(expected_arms) if expected_arms is not None else None,
        "n_aligned_rows": len(table.rows),
        "primary_query_type": PRIMARY_QUERY_TYPE,
        "secondary_query_type": SECONDARY_QUERY_TYPE,
        "query_type_participant_overlap": {
            qt: sorted(parts) for qt, parts in query_type_participants.items()
        },
        "method_names": METHOD_NAMES,
    }
    return table


# ---------------------------------------------------------------------------
# frozen participant folds
# ---------------------------------------------------------------------------


def _seeded_rng(*parts: Any) -> np.random.Generator:
    payload = "::".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def assign_participant_folds(
    participants: Sequence[str],
    *,
    n_outer: int = DEFAULT_N_OUTER,
    n_inner: int = DEFAULT_N_INNER,
    repeats: int = DEFAULT_REPEATS,
    seed: int = 0,
    shuffle: bool = True,
) -> dict[str, Any]:
    """Freeze deterministic participant -> outer-fold assignments per repeat.

    Returns a JSON-serializable structure reused verbatim for every model, arm,
    feature, and layer so all comparisons are paired.
    """
    parts = sorted(set(str(p) for p in participants))
    n_outer = max(2, min(int(n_outer), len(parts)))
    repeat_assignments: list[dict[str, int]] = []
    for repeat in range(int(repeats)):
        order = list(parts)
        if shuffle:
            rng = _seeded_rng("outer", seed, repeat)
            rng.shuffle(order)
        folds = np.array_split(np.array(order, dtype=object), n_outer)
        mapping: dict[str, int] = {}
        for fold_index, fold_members in enumerate(folds):
            for member in fold_members.tolist():
                mapping[str(member)] = fold_index
        repeat_assignments.append(mapping)
    return {
        "method": METHOD_NAMES["outer_split"],
        "n_outer": n_outer,
        "n_inner": int(n_inner),
        "repeats": int(repeats),
        "seed": int(seed),
        "shuffle": bool(shuffle),
        "n_participants": len(parts),
        "participants": parts,
        "assignments": repeat_assignments,
    }


def inner_fold_splits(train_participants: Sequence[str], *, n_inner: int, seed: int, repeat: int, outer_fold: int) -> list[tuple[list[str], list[str]]]:
    parts = sorted(set(str(p) for p in train_participants))
    n_inner = max(2, min(int(n_inner), len(parts)))
    rng = _seeded_rng("inner", seed, repeat, outer_fold)
    order = list(parts)
    rng.shuffle(order)
    folds = [f.tolist() for f in np.array_split(np.array(order, dtype=object), n_inner)]
    splits: list[tuple[list[str], list[str]]] = []
    for i in range(n_inner):
        val = [str(x) for x in folds[i]]
        train = [str(x) for f in (folds[:i] + folds[i + 1 :]) for x in f]
        if val and train:
            splits.append((train, val))
    return splits


# ---------------------------------------------------------------------------
# streaming feature loading
# ---------------------------------------------------------------------------


@dataclass
class PoolStack:
    """One (model, arm, query_type, pool) stack held across layers.

    Complies with the memory budget: one model/pool across layers. Region pools
    use complete cases (rows missing the vector are dropped, never zero-imputed).
    """

    model_key: str
    arm: str
    query_type: str
    pool: str
    layers: list[int]
    max_layer_index: int
    layer_matrices: dict[int, np.ndarray]  # layer_index -> (n_rows, d) float32
    y: np.ndarray
    groups: np.ndarray  # participant per row
    query_ids: list[str]
    coverage: dict[str, Any]

    @property
    def n_rows(self) -> int:
        return int(self.y.shape[0])


def resolve_layers(layer_spec: str, max_layer_index: int) -> list[int]:
    spec = (layer_spec or "all").strip().lower()
    if spec == "all":
        return list(range(0, max_layer_index + 1))
    if spec in ("coarse", "grid", "normalized"):
        fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
        idxs = sorted({int(round(f * max_layer_index)) for f in fracs})
        return idxs
    out: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if 0 <= idx <= max_layer_index:
            out.append(idx)
    return sorted(set(out)) or [0]


def load_pool_stack(
    table: CanonicalTable,
    *,
    model_key: str,
    arm: str,
    query_type: str,
    pool: str,
    layers: Sequence[int],
) -> PoolStack:
    """Stream the requested (model, arm, query_type, pool) vectors across layers.

    Opens each chunk file once per pass (grouped by chunk), reads only the needed
    tensors, casts to float32, and stacks. Never loads the whole corpus.
    """
    query_ids = table.queries_for(query_type)
    is_region = pool in QWEN_REGION_POOLS
    selected: list[tuple[str, RowRecord]] = []
    missing_pool = 0
    for query_id in query_ids:
        rec = table.record(model_key, arm, query_id)
        if rec is None:
            continue
        if is_region and pool not in rec.available_pools:
            missing_pool += 1
            continue
        selected.append((query_id, rec))

    layers = list(layers)
    layer_matrices: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    y_list: list[int] = []
    group_list: list[str] = []
    kept_query_ids: list[str] = []

    # group rows by chunk file so each chunk is opened once per layer pass
    by_chunk: dict[Path, list[tuple[int, str, RowRecord]]] = {}
    for order_index, (query_id, rec) in enumerate(selected):
        by_chunk.setdefault(rec.chunk_path, []).append((order_index, query_id, rec))

    # first pass: verify every selected row actually has all requested layer keys
    row_vectors: dict[int, dict[int, np.ndarray]] = {}  # order_index -> {layer: vec}
    dropped_missing_layer = 0
    valid_order: list[int] = []
    for chunk_path, entries in by_chunk.items():
        if not chunk_path.exists():
            dropped_missing_layer += len(entries)
            continue
        with SafetensorsChunk(chunk_path) as chunk:
            for order_index, query_id, rec in entries:
                vecs: dict[int, np.ndarray] = {}
                ok = True
                for layer in layers:
                    key = rec.layer_key(pool, layer)
                    if not chunk.has(key):
                        ok = False
                        break
                    vecs[layer] = chunk.get_float32(key).reshape(-1)
                if ok:
                    row_vectors[order_index] = vecs
                    valid_order.append(order_index)
                else:
                    dropped_missing_layer += 1

    valid_order.sort()
    for order_index in valid_order:
        query_id, rec = selected[order_index]
        vecs = row_vectors[order_index]
        for layer in layers:
            layer_matrices[layer].append(vecs[layer])
        y_list.append(table.label_by_query[query_id])
        group_list.append(rec.participant)
        kept_query_ids.append(query_id)

    stacked = {
        layer: (np.vstack(vs).astype(np.float32) if vs else np.zeros((0, 0), dtype=np.float32))
        for layer, vs in layer_matrices.items()
    }
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(group_list, dtype=object)
    coverage = {
        "n_source_queries_in_type": len(query_ids),
        "n_cell_rows_available": len(selected),
        "n_rows_used": int(y.shape[0]),
        "n_missing_pool_rows": missing_pool,
        "n_dropped_missing_layer": dropped_missing_layer,
        "is_region_pool": is_region,
        "coverage_fraction": (len(y_list) / len(query_ids)) if query_ids else 0.0,
        "n_participants": int(len(set(group_list))),
    }
    return PoolStack(
        model_key=model_key,
        arm=arm,
        query_type=query_type,
        pool=pool,
        layers=layers,
        max_layer_index=table.max_layer_index(model_key),
        layer_matrices=stacked,
        y=y,
        groups=groups,
        query_ids=kept_query_ids,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def cross_entropy_bits(y: np.ndarray, prob: np.ndarray) -> float:
    prob = np.clip(prob, _PROB_EPS, 1.0 - _PROB_EPS)
    ll = y * np.log(prob) + (1 - y) * np.log(1.0 - prob)
    return float(-np.mean(ll) / _LN2)


def metric_bundle(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    both_classes = len(np.unique(y)) == 2
    result: dict[str, Any] = {
        "n_rows": int(y.shape[0]),
        "ce_bits": cross_entropy_bits(y, prob) if y.shape[0] else float("nan"),
        "status": "ok" if both_classes else "single_class",
    }
    if both_classes and y.shape[0] > 1:
        result["auroc"] = float(roc_auc_score(y, prob))
        result["balanced_acc"] = float(balanced_accuracy_score(y, (prob >= 0.5).astype(int)))
    else:
        result["auroc"] = float("nan")
        result["balanced_acc"] = float("nan")
    return result


# ---------------------------------------------------------------------------
# probe primitives
# ---------------------------------------------------------------------------


def _fit_scaler_logreg(X_train: np.ndarray, y_train: np.ndarray, *, C: float, max_iter: int) -> tuple[Any, Any, bool]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=int(max_iter))
    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(Xs, y_train)
        for warning in caught:
            if issubclass(warning.category, ConvergenceWarning):
                converged = False
    return scaler, clf, converged


def _predict_prob(scaler: Any, clf: Any, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(scaler.transform(X))[:, 1]


def regularization_protocol(c_grid: Sequence[float]) -> str:
    return (
        f"fixed_C_{float(c_grid[0]):g}"
        if len(c_grid) == 1
        else METHOD_NAMES["c_selection"]
    )


def select_C_inner(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    *,
    c_grid: Sequence[float],
    n_inner: int,
    seed: int,
    repeat: int,
    outer_fold: int,
    max_iter: int,
) -> tuple[float, dict[str, float]]:
    """Select C on inner grouped CV by mean out-of-fold log loss (bits)."""
    if len(c_grid) == 1:
        only = float(c_grid[0])
        return only, {str(only): float("nan")}
    train_parts = sorted(set(groups_train.tolist()))
    splits = inner_fold_splits(train_parts, n_inner=n_inner, seed=seed, repeat=repeat, outer_fold=outer_fold)
    scores: dict[float, list[float]] = {C: [] for C in c_grid}
    if not splits:
        return float(c_grid[len(c_grid) // 2]), {str(C): float("nan") for C in c_grid}
    part_to_idx: dict[str, np.ndarray] = {}
    for part in train_parts:
        part_to_idx[part] = np.where(groups_train == part)[0]
    for inner_train_parts, inner_val_parts in splits:
        tr_idx = np.concatenate([part_to_idx[p] for p in inner_train_parts])
        va_idx = np.concatenate([part_to_idx[p] for p in inner_val_parts])
        if len(np.unique(y_train[tr_idx])) < 2:
            continue
        for C in c_grid:
            scaler, clf, _ = _fit_scaler_logreg(X_train[tr_idx], y_train[tr_idx], C=C, max_iter=max_iter)
            prob = _predict_prob(scaler, clf, X_train[va_idx])
            scores[C].append(cross_entropy_bits(y_train[va_idx], prob))
    mean_scores = {C: (float(np.mean(v)) if v else float("inf")) for C, v in scores.items()}
    best_C = min(mean_scores, key=lambda c: mean_scores[c])
    return float(best_C), {str(C): mean_scores[C] for C in c_grid}


def outer_fold_indices(groups: np.ndarray, mapping: dict[str, int], fold: int) -> tuple[np.ndarray, np.ndarray]:
    test_mask = np.array([mapping.get(str(g), -1) == fold for g in groups])
    train_mask = np.array([mapping.get(str(g), -1) not in (fold, -1) for g in groups])
    return np.where(train_mask)[0], np.where(test_mask)[0]


def descriptive_layer_curves(
    stack: PoolStack,
    folds: dict[str, Any],
    *,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    max_iter: int = DEFAULT_MAX_ITER,
    progress: Any = None,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """Per-layer descriptive OOF curves. Also returns per-layer averaged OOF probs.

    Returns (per_fold_rows, oof_prob_by_layer) where oof_prob_by_layer[layer] is
    the repeat-averaged OOF probability per row (NaN where a row was never a test
    row, which does not happen with full fold coverage).
    """
    per_fold_rows: list[dict[str, Any]] = []
    assignments = folds["assignments"]
    n_layers = len(stack.layers)
    # accumulate OOF probs summed across repeats
    oof_sum = {layer: np.zeros(stack.n_rows, dtype=np.float64) for layer in stack.layers}
    oof_count = {layer: np.zeros(stack.n_rows, dtype=np.float64) for layer in stack.layers}
    seed = int(folds["seed"])
    n_inner = int(folds["n_inner"])
    for repeat, mapping in enumerate(assignments):
        n_outer = folds["n_outer"]
        for fold in range(n_outer):
            train_idx, test_idx = outer_fold_indices(stack.groups, mapping, fold)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            for layer in stack.layers:
                if progress is not None:
                    progress(stack, layer, repeat, fold)
                X = stack.layer_matrices[layer]
                if X.shape[0] == 0:
                    continue
                if len(np.unique(stack.y[train_idx])) < 2:
                    prob = np.full(len(test_idx), float(np.mean(stack.y[train_idx])))
                    C = float("nan")
                    converged = True
                    status = "single_class_train"
                else:
                    C, _ = select_C_inner(
                        X[train_idx], stack.y[train_idx], stack.groups[train_idx],
                        c_grid=c_grid, n_inner=n_inner, seed=seed, repeat=repeat, outer_fold=fold, max_iter=max_iter,
                    )
                    scaler, clf, converged = _fit_scaler_logreg(X[train_idx], stack.y[train_idx], C=C, max_iter=max_iter)
                    prob = _predict_prob(scaler, clf, X[test_idx])
                    status = "ok"
                oof_sum[layer][test_idx] += prob
                oof_count[layer][test_idx] += 1.0
                bundle = metric_bundle(stack.y[test_idx], prob)
                per_fold_rows.append({
                    "model_key": stack.model_key,
                    "arm": stack.arm,
                    "query_type": stack.query_type,
                    "pool": stack.pool,
                    "layer_index": layer,
                    "layer_frac": (layer / stack.max_layer_index) if stack.max_layer_index else 0.0,
                    "repeat": repeat,
                    "fold": fold,
                    "C": C,
                    "converged": converged,
                    "n_rows": bundle["n_rows"],
                    "n_train_rows": int(len(train_idx)),
                    "n_participants": int(len(set(stack.groups[test_idx].tolist()))),
                    "auroc": bundle["auroc"],
                    "ce_bits": bundle["ce_bits"],
                    "balanced_acc": bundle["balanced_acc"],
                    "status": status if bundle["status"] == "ok" else bundle["status"],
                    "method": METHOD_NAMES["probe"],
                    "regularization_protocol": regularization_protocol(c_grid),
                    "is_text_floor": bool(layer == 0 and stack.pool in ("last_token", "text_tail_mean")),
                })
    oof_prob = {}
    for layer in stack.layers:
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(oof_count[layer] > 0, oof_sum[layer] / np.maximum(oof_count[layer], 1e-9), np.nan)
        oof_prob[layer] = avg
    return per_fold_rows, oof_prob


def nested_best_layer(
    stack: PoolStack,
    folds: dict[str, Any],
    *,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    max_iter: int = DEFAULT_MAX_ITER,
    candidate_layers: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Unbiased selected-layer estimate: layer+C chosen on inner folds only."""
    rows: list[dict[str, Any]] = []
    assignments = folds["assignments"]
    seed = int(folds["seed"])
    n_inner = int(folds["n_inner"])
    for repeat, mapping in enumerate(assignments):
        for fold in range(folds["n_outer"]):
            train_idx, test_idx = outer_fold_indices(stack.groups, mapping, fold)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            if len(np.unique(stack.y[train_idx])) < 2:
                continue
            train_parts = sorted(set(stack.groups[train_idx].tolist()))
            inner = inner_fold_splits(train_parts, n_inner=n_inner, seed=seed, repeat=repeat, outer_fold=fold)
            if not inner:
                continue
            # joint (layer, C) selection by inner OOF log loss
            best = None
            selected_candidates = [
                layer
                for layer in (candidate_layers or stack.layers)
                if layer in stack.layer_matrices
            ]
            for layer in selected_candidates:
                X = stack.layer_matrices[layer]
                if X.shape[0] == 0:
                    continue
                for C in c_grid:
                    losses = []
                    for itr_parts, iva_parts in inner:
                        tr = np.concatenate([train_idx[np.isin(stack.groups[train_idx], itr_parts)]])
                        va = np.concatenate([train_idx[np.isin(stack.groups[train_idx], iva_parts)]])
                        if len(np.unique(stack.y[tr])) < 2:
                            continue
                        scaler, clf, _ = _fit_scaler_logreg(X[tr], stack.y[tr], C=C, max_iter=max_iter)
                        prob = _predict_prob(scaler, clf, X[va])
                        losses.append(cross_entropy_bits(stack.y[va], prob))
                    if not losses:
                        continue
                    mean_loss = float(np.mean(losses))
                    if best is None or mean_loss < best[0]:
                        best = (mean_loss, layer, C)
            if best is None:
                continue
            _, sel_layer, sel_C = best
            X = stack.layer_matrices[sel_layer]
            scaler, clf, converged = _fit_scaler_logreg(X[train_idx], stack.y[train_idx], C=sel_C, max_iter=max_iter)
            prob = _predict_prob(scaler, clf, X[test_idx])
            bundle = metric_bundle(stack.y[test_idx], prob)
            rows.append({
                "model_key": stack.model_key,
                "arm": stack.arm,
                "query_type": stack.query_type,
                "pool": stack.pool,
                "repeat": repeat,
                "fold": fold,
                "selected_layer_index": sel_layer,
                "selected_layer_frac": (sel_layer / stack.max_layer_index) if stack.max_layer_index else 0.0,
                "selected_C": sel_C,
                "converged": converged,
                "n_rows": bundle["n_rows"],
                "n_participants": int(len(set(stack.groups[test_idx].tolist()))),
                "auroc": bundle["auroc"],
                "ce_bits": bundle["ce_bits"],
                "balanced_acc": bundle["balanced_acc"],
                "status": bundle["status"],
                "method": METHOD_NAMES["layer_selection"],
                "regularization_protocol": regularization_protocol(c_grid),
            })
    return rows


def cluster_bootstrap_auroc(
    y: np.ndarray,
    prob: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Participant-cluster bootstrap CI over OOF predictions."""
    from sklearn.metrics import roc_auc_score

    mask = ~np.isnan(prob)
    y = np.asarray(y)[mask]
    prob = np.asarray(prob)[mask]
    groups = np.asarray(groups)[mask]
    unique_parts = sorted(set(groups.tolist()))
    part_idx = {p: np.where(groups == p)[0] for p in unique_parts}
    rng = _seeded_rng("bootstrap", seed)
    boot: list[float] = []
    if len(np.unique(y)) == 2 and y.shape[0] > 1:
        point = float(roc_auc_score(y, prob))
    else:
        point = float("nan")
    for _ in range(int(n_boot)):
        chosen = rng.choice(len(unique_parts), size=len(unique_parts), replace=True)
        idx = np.concatenate([part_idx[unique_parts[c]] for c in chosen])
        yb, pb = y[idx], prob[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot.append(float(roc_auc_score(yb, pb)))
    if boot:
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    else:
        lo = hi = float("nan")
    return {"auroc": point, "ci95_low": lo, "ci95_high": hi, "n_boot_valid": len(boot), "method": METHOD_NAMES["bootstrap"]}


def permutation_max_over_layers(
    stack: PoolStack,
    oof_prob_by_layer: dict[int, np.ndarray],
    *,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    """Cheap max-over-layers label-permutation over fixed OOF predictions.

    The permutation map is shared across layers and shuffles labels within each
    participant. This is not a full pipeline-refit permutation (recorded below).
    """
    from sklearn.metrics import roc_auc_score

    layers = [layer for layer in stack.layers if layer in oof_prob_by_layer]
    y = stack.y.astype(int)
    groups = stack.groups
    part_idx = {p: np.where(groups == p)[0] for p in sorted(set(groups.tolist()))}

    def layer_auroc(labels: np.ndarray, prob: np.ndarray) -> float:
        mask = ~np.isnan(prob)
        yl, pl = labels[mask], prob[mask]
        if len(np.unique(yl)) < 2 or yl.shape[0] < 2:
            return float("nan")
        return float(roc_auc_score(yl, pl))

    observed = {layer: layer_auroc(y, oof_prob_by_layer[layer]) for layer in layers}
    obs_values = [v for v in observed.values() if not math.isnan(v)]
    obs_max = max(obs_values) if obs_values else float("nan")

    rng = _seeded_rng("permutation", seed)
    null_max: list[float] = []
    ge_count = 0
    for _ in range(int(n_perm)):
        permuted = y.copy()
        for part, idx in part_idx.items():
            permuted[idx] = rng.permutation(y[idx])
        layer_vals = [layer_auroc(permuted, oof_prob_by_layer[layer]) for layer in layers]
        layer_vals = [v for v in layer_vals if not math.isnan(v)]
        if not layer_vals:
            continue
        m = max(layer_vals)
        null_max.append(m)
        if not math.isnan(obs_max) and m >= obs_max:
            ge_count += 1
    p_value = (1 + ge_count) / (1 + len(null_max)) if null_max else float("nan")
    return {
        "model_key": stack.model_key,
        "arm": stack.arm,
        "query_type": stack.query_type,
        "pool": stack.pool,
        "layers": layers,
        "observed_auroc_by_layer": {str(k): v for k, v in observed.items()},
        "observed_max_auroc": obs_max,
        "n_permutations": int(len(null_max)),
        "p_value_max_stat": p_value,
        "null_max_quantiles": {
            "q50": float(np.percentile(null_max, 50)) if null_max else float("nan"),
            "q95": float(np.percentile(null_max, 95)) if null_max else float("nan"),
            "q99": float(np.percentile(null_max, 99)) if null_max else float("nan"),
        },
        "method": METHOD_NAMES["permutation"],
        "method_caveat": METHOD_NAMES["permutation_caveat"],
    }


# ---------------------------------------------------------------------------
# strict cross-arm transfer
# ---------------------------------------------------------------------------


def strict_cross_arm_transfer(
    table: CanonicalTable,
    folds: dict[str, Any],
    *,
    model_key: str,
    query_type: str,
    pool: str,
    from_arm: str,
    to_arm: str,
    layers: Sequence[int],
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    max_iter: int = DEFAULT_MAX_ITER,
) -> list[dict[str, Any]]:
    """Train on ``from_arm`` (outer-train participants), test ``to_arm`` (outer-test).

    Participants are disjoint between train and test by construction (shared
    frozen folds). This is the only construct called generalization.
    """
    src = load_pool_stack(table, model_key=model_key, arm=from_arm, query_type=query_type, pool=pool, layers=layers)
    dst = load_pool_stack(table, model_key=model_key, arm=to_arm, query_type=query_type, pool=pool, layers=layers)
    rows: list[dict[str, Any]] = []
    if src.n_rows == 0 or dst.n_rows == 0:
        return rows
    seed = int(folds["seed"])
    n_inner = int(folds["n_inner"])
    for repeat, mapping in enumerate(folds["assignments"]):
        for fold in range(folds["n_outer"]):
            src_train, _src_test = outer_fold_indices(src.groups, mapping, fold)
            _dst_train, dst_test = outer_fold_indices(dst.groups, mapping, fold)
            if len(src_train) == 0 or len(dst_test) == 0:
                continue
            if len(np.unique(src.y[src_train])) < 2:
                continue
            # nested layer+C selection on the source arm's inner folds
            train_parts = sorted(set(src.groups[src_train].tolist()))
            inner = inner_fold_splits(train_parts, n_inner=n_inner, seed=seed, repeat=repeat, outer_fold=fold)
            best = None
            for layer in layers:
                Xs = src.layer_matrices[layer]
                if Xs.shape[0] == 0:
                    continue
                for C in c_grid:
                    losses = []
                    for itr, iva in inner:
                        tr = src_train[np.isin(src.groups[src_train], itr)]
                        va = src_train[np.isin(src.groups[src_train], iva)]
                        if len(np.unique(src.y[tr])) < 2:
                            continue
                        scaler, clf, _ = _fit_scaler_logreg(Xs[tr], src.y[tr], C=C, max_iter=max_iter)
                        losses.append(cross_entropy_bits(src.y[va], _predict_prob(scaler, clf, Xs[va])))
                    if losses:
                        m = float(np.mean(losses))
                        if best is None or m < best[0]:
                            best = (m, layer, C)
            if best is None:
                continue
            _, sel_layer, sel_C = best
            scaler, clf, converged = _fit_scaler_logreg(src.layer_matrices[sel_layer][src_train], src.y[src_train], C=sel_C, max_iter=max_iter)
            prob = _predict_prob(scaler, clf, dst.layer_matrices[sel_layer][dst_test])
            bundle = metric_bundle(dst.y[dst_test], prob)
            # verify participant disjointness (guardrail)
            train_participants = set(src.groups[src_train].tolist())
            test_participants = set(dst.groups[dst_test].tolist())
            rows.append({
                "model_key": model_key,
                "query_type": query_type,
                "pool": pool,
                "from_arm": from_arm,
                "to_arm": to_arm,
                "repeat": repeat,
                "fold": fold,
                "selected_layer_index": sel_layer,
                "selected_C": sel_C,
                "converged": converged,
                "n_train_rows": int(len(src_train)),
                "n_test_rows": bundle["n_rows"],
                "auroc": bundle["auroc"],
                "ce_bits": bundle["ce_bits"],
                "balanced_acc": bundle["balanced_acc"],
                "participants_disjoint": bool(train_participants.isdisjoint(test_participants)),
                "status": bundle["status"],
                "method": METHOD_NAMES["transfer"],
                "regularization_protocol": regularization_protocol(c_grid),
            })
    return rows


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def intercept_baseline(table: CanonicalTable, folds: dict[str, Any], *, query_type: str) -> list[dict[str, Any]]:
    query_ids = table.queries_for(query_type)
    y = np.array([table.label_by_query[q] for q in query_ids], dtype=int)
    groups = np.array([table.participant_by_query[q] for q in query_ids], dtype=object)
    rows: list[dict[str, Any]] = []
    for repeat, mapping in enumerate(folds["assignments"]):
        for fold in range(folds["n_outer"]):
            train_idx, test_idx = outer_fold_indices(groups, mapping, fold)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            prior = float(np.mean(y[train_idx])) if len(train_idx) else 0.5
            prob = np.full(len(test_idx), prior)
            bundle = metric_bundle(y[test_idx], prob)
            rows.append({
                "baseline": "intercept_class_prior",
                "query_type": query_type,
                "repeat": repeat,
                "fold": fold,
                "n_rows": bundle["n_rows"],
                "auroc": bundle["auroc"],
                "ce_bits": bundle["ce_bits"],
                "balanced_acc": bundle["balanced_acc"],
                "status": bundle["status"],
                "method": METHOD_NAMES["intercept_baseline"],
            })
    return rows


def object_identity_baseline(table: CanonicalTable, folds: dict[str, Any], *, query_type: str) -> list[dict[str, Any]]:
    """Public-question/object-identity floor: TF-IDF(question + target) grouped CV."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    query_ids = table.queries_for(query_type)
    texts = [f"{table.question_by_query.get(q, '')} || target: {table.target_by_query.get(q, '')}" for q in query_ids]
    y = np.array([table.label_by_query[q] for q in query_ids], dtype=int)
    groups = np.array([table.participant_by_query[q] for q in query_ids], dtype=object)
    rows: list[dict[str, Any]] = []
    for repeat, mapping in enumerate(folds["assignments"]):
        for fold in range(folds["n_outer"]):
            train_idx, test_idx = outer_fold_indices(groups, mapping, fold)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            if len(np.unique(y[train_idx])) < 2:
                prob = np.full(len(test_idx), float(np.mean(y[train_idx])))
            else:
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
                Xtr = vec.fit_transform([texts[i] for i in train_idx])
                Xte = vec.transform([texts[i] for i in test_idx])
                clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)
                clf.fit(Xtr, y[train_idx])
                prob = clf.predict_proba(Xte)[:, 1]
            bundle = metric_bundle(y[test_idx], prob)
            rows.append({
                "baseline": "object_identity_tfidf",
                "query_type": query_type,
                "repeat": repeat,
                "fold": fold,
                "n_rows": bundle["n_rows"],
                "auroc": bundle["auroc"],
                "ce_bits": bundle["ce_bits"],
                "balanced_acc": bundle["balanced_acc"],
                "status": bundle["status"],
                "method": METHOD_NAMES["object_identity_baseline"],
            })
    return rows


def missingness_coverage_baseline(table: CanonicalTable, *, query_type: str, models: Sequence[str], arms: Sequence[str]) -> list[dict[str, Any]]:
    """Coverage-only audit for optional Qwen region pools (no imputation)."""
    query_ids = table.queries_for(query_type)
    rows: list[dict[str, Any]] = []
    for model_key in models:
        if model_family(model_key) != "qwen":
            continue
        for arm in arms:
            for pool in QWEN_REGION_POOLS:
                present = 0
                total = 0
                present_labels: list[int] = []
                for q in query_ids:
                    rec = table.record(model_key, arm, q)
                    if rec is None:
                        continue
                    total += 1
                    if pool in rec.available_pools:
                        present += 1
                        present_labels.append(table.label_by_query[q])
                rows.append({
                    "baseline": "region_pool_missingness_coverage",
                    "model_key": model_key,
                    "arm": arm,
                    "pool": pool,
                    "query_type": query_type,
                    "n_cell_rows": total,
                    "n_present": present,
                    "coverage_fraction": (present / total) if total else 0.0,
                    "present_base_rate": (float(np.mean(present_labels)) if present_labels else float("nan")),
                    "method": METHOD_NAMES["missingness_baseline"],
                })
    return rows


def image_token_confound_table(table: CanonicalTable, *, models: Sequence[str], arms: Sequence[str]) -> list[dict[str, Any]]:
    """Per model/arm min/median/max of mask_meta.image_token_count."""
    rows: list[dict[str, Any]] = []
    for model_key in models:
        for arm in arms:
            counts = [
                rec.image_token_count
                for (mk, a, _q), rec in table.rows.items()
                if mk == model_key and a == arm and rec.image_token_count is not None
            ]
            if not counts:
                continue
            arr = np.array(counts, dtype=float)
            rows.append({
                "model_key": model_key,
                "arm": arm,
                "n_rows": int(arr.shape[0]),
                "image_token_min": int(arr.min()),
                "image_token_median": float(np.median(arr)),
                "image_token_max": int(arr.max()),
            })
    return rows


# ---------------------------------------------------------------------------
# model contrasts
# ---------------------------------------------------------------------------


def default_pools_for_model(model_key: str, *, include_region: bool = True) -> list[str]:
    pools = list(COMMON_POOLS)
    if include_region and model_family(model_key) == "qwen":
        pools.extend(QWEN_REGION_POOLS)
    return pools


def _masked_metric(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    mask = ~np.isnan(np.asarray(prob, dtype=float))
    return metric_bundle(np.asarray(y)[mask], np.asarray(prob, dtype=float)[mask])


def pooled_layer_summary(
    stack: PoolStack,
    oof_prob_by_layer: dict[int, np.ndarray],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Descriptive pooled-OOF per-layer AUROC/CE plus a headline bootstrap CI.

    The bootstrap is computed only for the max-observed-AUROC layer to keep the
    full-run cost bounded; it is descriptive (not the unbiased selected-layer
    estimate, which lives in ``nested_best_layer``).
    """
    per_layer: list[dict[str, Any]] = []
    best_layer = None
    best_auroc = -math.inf
    for layer in stack.layers:
        prob = oof_prob_by_layer.get(layer)
        if prob is None:
            continue
        bundle = _masked_metric(stack.y, prob)
        frac = (layer / stack.max_layer_index) if stack.max_layer_index else 0.0
        per_layer.append({
            "layer_index": layer,
            "layer_frac": frac,
            "auroc": bundle["auroc"],
            "ce_bits": bundle["ce_bits"],
            "balanced_acc": bundle["balanced_acc"],
            "status": bundle["status"],
        })
        if not math.isnan(bundle["auroc"]) and bundle["auroc"] > best_auroc:
            best_auroc = bundle["auroc"]
            best_layer = layer
    headline = None
    if best_layer is not None:
        headline = cluster_bootstrap_auroc(
            stack.y, oof_prob_by_layer[best_layer], stack.groups, n_boot=n_boot, seed=seed
        )
        headline["layer_index"] = best_layer
    return {
        "model_key": stack.model_key,
        "arm": stack.arm,
        "query_type": stack.query_type,
        "pool": stack.pool,
        "coverage": stack.coverage,
        "per_layer": per_layer,
        "headline_bootstrap": headline,
    }


def model_contrast_rows(nested_rows: list[dict[str, Any]], *, model_a: str, model_b: str) -> list[dict[str, Any]]:
    """Paired nested-Theta* contrast between two named checkpoints per cell.

    Never averages families. A4B stays a distinct MoE row elsewhere.
    """
    import pandas as pd

    if not nested_rows:
        return []
    frame = pd.DataFrame(nested_rows)
    out: list[dict[str, Any]] = []
    key_cols = ["arm", "query_type", "pool", "repeat", "fold"]
    a = frame[frame["model_key"] == model_a].set_index(key_cols)
    b = frame[frame["model_key"] == model_b].set_index(key_cols)
    common = a.index.intersection(b.index)
    for idx in common:
        arm, query_type, pool, repeat, fold = idx
        av = a.loc[idx, "auroc"]
        bv = b.loc[idx, "auroc"]
        av = float(av) if np.ndim(av) == 0 else float(np.asarray(av).ravel()[0])
        bv = float(bv) if np.ndim(bv) == 0 else float(np.asarray(bv).ravel()[0])
        out.append({
            "contrast": f"{model_a}__minus__{model_b}",
            "model_a": model_a,
            "model_b": model_b,
            "arm": arm,
            "query_type": query_type,
            "pool": pool,
            "repeat": int(repeat),
            "fold": int(fold),
            "auroc_a": av,
            "auroc_b": bv,
            "delta_auroc": (av - bv) if (not math.isnan(av) and not math.isnan(bv)) else float("nan"),
            "note": "named_checkpoint_contrast_paired_on_shared_folds_not_family_average",
        })
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


CLAIM_BOUNDARY_STATEMENTS = [
    "Job2 current captures establish LINEAR DECODABILITY of hand-target contact "
    "from POOLED SUMMARY vectors only.",
    "A null probe never means information is absent; it means 'not linearly "
    "decodable from this pooled summary at this layer'.",
    "A positive probe never proves use, routing, binding, or causality.",
    "The readout side of Job2 is structurally ABSENT: no behavior, no "
    "decision-position state, and the stored last-prompt-position logits are "
    "JSON-format logits, not the true/false decision token.",
    "contact_state_current is the primary query type; contact_order_before is a "
    "separate secondary track and is never pooled into a headline probe.",
    "Splits are grouped by participant (one participant == one video). Object/"
    "text shortcut baselines are the honest floors, not chance.",
    "Qwen and Gemma have very different image-token budgets; the budget is "
    "reported alongside every cross-family number, never adjusted away.",
    "Qwen-only region pools (target/hand/background) use complete cases with "
    "reported coverage; they are not comparable to Gemma.",
    "The visual_prompt arm is annotation-geometry confounded: a bbox-only "
    "evaluator baseline reaches about 0.80 AUROC, so its model gap is demoted.",
    "contact_order_before is endpoint-confounded: final-frame contact/visibility "
    "alone reaches about 0.96 AUROC, so single-frame decoding is NOT temporal memory.",
    "The broad fixed-OOF permutation table is exploratory, not confirmatory; "
    "principal current/last-token claims use a separate full-refit null.",
    "The promoted run uses one frozen five-fold participant partition. Fold-assignment "
    "variance remains an explicit limitation despite participant-cluster bootstrap.",
]


@dataclass
class AnalysisConfig:
    source_run: Path
    out_dir: Path
    mode: str = "full"  # audit | canary | full
    models: tuple[str, ...] = DEFAULT_MODEL_KEYS
    arms: tuple[str, ...] = DEFAULT_ARMS
    pools: tuple[str, ...] | None = None  # None => auto per model
    query_types: tuple[str, ...] = (PRIMARY_QUERY_TYPE,)
    layer_spec: str = "all"
    nested_layer_spec: str = "coarse"
    repeats: int = DEFAULT_REPEATS
    n_outer: int = DEFAULT_N_OUTER
    n_inner: int = DEFAULT_N_INNER
    c_grid: tuple[float, ...] = DEFAULT_C_GRID
    seed: int = 0
    shuffle_clusters: bool = True
    bootstrap: int = DEFAULT_BOOTSTRAP
    permutations: int = DEFAULT_PERMUTATIONS
    max_iter: int = DEFAULT_MAX_ITER
    do_transfer: bool = True
    do_permutation: bool = True
    do_region_pools: bool = True
    transfer_pairs: tuple[tuple[str, str], ...] = (("sequential", "single_frame"), ("visual_prompt", "sequential"))
    verbose: bool = True


def _log(config: AnalysisConfig, message: str, *, t0: float) -> None:
    if config.verbose:
        print(f"[{time.perf_counter() - t0:8.1f}s] {message}", flush=True)


def _manifest(config: AnalysisConfig, table: CanonicalTable, folds: dict[str, Any]) -> dict[str, Any]:
    analysis_path = Path(__file__).resolve()
    cli_path = analysis_path.with_name("run_phase2_job2_representation_probes.py")
    return {
        "kind": "phase2_job2_representation_probe_manifest",
        "mode": config.mode,
        "source_run": str(config.source_run),
        "out_dir": str(config.out_dir),
        "models": list(config.models),
        "arms": list(config.arms),
        "query_types": list(config.query_types),
        "pools_config": (list(config.pools) if config.pools else "auto_per_model"),
        "layer_spec": config.layer_spec,
        "nested_layer_spec": config.nested_layer_spec,
        "seed": config.seed,
        "shuffle_clusters": config.shuffle_clusters,
        "repeats": config.repeats,
        "n_outer": folds["n_outer"],
        "n_inner": config.n_inner,
        "c_grid": list(config.c_grid),
        "regularization_protocol": regularization_protocol(config.c_grid),
        "bootstrap": config.bootstrap,
        "permutations": config.permutations,
        "max_iter": config.max_iter,
        "n_participants": len(table.participants),
        "n_source_queries": len(table.label_by_query),
        "primary_query_type": PRIMARY_QUERY_TYPE,
        "secondary_query_type": SECONDARY_QUERY_TYPE,
        "principal_contrast": list(PRINCIPAL_CONTRAST),
        "moe_models": list(MOE_MODEL_KEYS),
        "method_names": METHOD_NAMES,
        "code_provenance": {
            "analysis_path": str(analysis_path),
            "analysis_sha256": _sha256_file(analysis_path),
            "cli_path": str(cli_path),
            "cli_sha256": _sha256_file(cli_path),
        },
        "source_provenance": source_provenance(config.source_run, config.models),
        "claim_boundary": CLAIM_BOUNDARY_STATEMENTS,
        "not_supported_from_job2": [
            "behavior / generate() predictions",
            "decision-position true/false logits",
            "attention aggregates",
            "routing or causal claims",
            "V-information (behavioral half)",
        ],
    }


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    """Run the analyzer end-to-end and write all artifacts. Returns a summary."""
    t0 = time.perf_counter()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(config, f"building canonical table from {config.source_run}", t0=t0)
    table = build_canonical_table(
        config.source_run,
        config.models,
        expected_arms=config.arms,
    )
    folds = assign_participant_folds(
        table.participants, n_outer=config.n_outer, n_inner=config.n_inner,
        repeats=config.repeats, seed=config.seed, shuffle=config.shuffle_clusters,
    )

    write_json(out_dir / "manifest.json", _manifest(config, table, folds))
    write_json(out_dir / "data_audit.json", table.audit)
    write_json(out_dir / "fold_assignments.json", folds)

    # confound + coverage tables (metadata-only, always produced)
    confound = image_token_confound_table(table, models=config.models, arms=config.arms)
    missingness: list[dict[str, Any]] = []
    for query_type in config.query_types:
        missingness.extend(
            missingness_coverage_baseline(table, query_type=query_type, models=config.models, arms=config.arms)
        )

    summary: dict[str, Any] = {
        "mode": config.mode,
        "audit_status": table.audit["status"],
        "cell_status": [],
        "required_cell_failures": [],
        "outputs": [],
        "elapsed_s": None,
    }

    if config.mode == "audit" or table.audit["status"] != "pass":
        baseline_rows = missingness + confound_as_baseline(confound)
        if baseline_rows:
            write_parquet(out_dir / "baseline_results.parquet", baseline_rows)
            summary["outputs"].append("baseline_results.parquet")
        report = generate_report(
            config,
            table,
            folds,
            summary,
            confound=confound,
            missingness=missingness,
            baseline_rows=baseline_rows,
        )
        write_text(out_dir / "report.md", report)
        summary["outputs"].extend(["manifest.json", "data_audit.json", "fold_assignments.json", "report.md"])
        summary["elapsed_s"] = time.perf_counter() - t0
        if config.mode != "audit" and table.audit["status"] != "pass":
            summary["required_cell_failures"].append({"reason": "data_audit_failed", "audit": table.audit["status"]})
        return summary

    probe_rows: list[dict[str, Any]] = []
    nested_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    permutation_results: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []

    for model_key in config.models:
        max_layer = table.max_layer_index(model_key)
        if max_layer == 0:
            summary["cell_status"].append({"model_key": model_key, "status": "skipped", "reason": "no_rows_for_model"})
            continue
        layers = resolve_layers(config.layer_spec, max_layer)
        pools = list(config.pools) if config.pools else default_pools_for_model(model_key, include_region=config.do_region_pools)
        for arm in config.arms:
            for query_type in config.query_types:
                for pool in pools:
                    is_required = (
                        query_type == PRIMARY_QUERY_TYPE
                        and pool in COMMON_POOLS
                    )
                    try:
                        stack = load_pool_stack(
                            table, model_key=model_key, arm=arm, query_type=query_type, pool=pool, layers=layers
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        status = {"model_key": model_key, "arm": arm, "query_type": query_type, "pool": pool,
                                  "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
                        summary["cell_status"].append(status)
                        if is_required:
                            summary["required_cell_failures"].append(status)
                        continue
                    if stack.n_rows == 0 or len(np.unique(stack.y)) < 2:
                        reason = "no_rows" if stack.n_rows == 0 else "single_class_cell"
                        status = {"model_key": model_key, "arm": arm, "query_type": query_type, "pool": pool,
                                  "status": "skipped", "reason": reason, "coverage": stack.coverage}
                        summary["cell_status"].append(status)
                        if is_required:
                            summary["required_cell_failures"].append(status)
                        continue

                    _log(config, f"probe {model_key}/{arm}/{query_type}/{pool} n={stack.n_rows} layers={len(layers)}", t0=t0)

                    def _progress(st: PoolStack, layer: int, repeat: int, fold: int) -> None:
                        if config.verbose and layer == st.layers[-1] and fold == 0:
                            _log(config, f"  {st.model_key}/{st.arm}/{st.pool} repeat={repeat} layer={layer}", t0=t0)

                    fold_rows, oof_prob_by_layer = descriptive_layer_curves(
                        stack, folds, c_grid=config.c_grid, max_iter=config.max_iter, progress=_progress
                    )
                    probe_rows.extend(fold_rows)
                    nested_rows.extend(
                        nested_best_layer(
                            stack,
                            folds,
                            c_grid=config.c_grid,
                            max_iter=config.max_iter,
                            candidate_layers=resolve_layers(
                                config.nested_layer_spec,
                                max_layer,
                            ),
                        )
                    )

                    for layer in stack.layers:
                        prob = oof_prob_by_layer[layer]
                        for i, query_id in enumerate(stack.query_ids):
                            oof_rows.append({
                                "model_key": model_key, "arm": arm, "query_type": query_type, "pool": pool,
                                "layer_index": layer, "source_query_id": query_id,
                                "participant": str(stack.groups[i]), "y_true": int(stack.y[i]),
                                "oof_prob": (float(prob[i]) if not math.isnan(prob[i]) else float("nan")),
                            })

                    cell_summaries.append(
                        pooled_layer_summary(stack, oof_prob_by_layer, n_boot=config.bootstrap, seed=config.seed)
                    )

                    if config.do_permutation and is_required:
                        permutation_results.append(
                            permutation_max_over_layers(stack, oof_prob_by_layer, n_perm=config.permutations, seed=config.seed)
                        )

                    summary["cell_status"].append({
                        "model_key": model_key, "arm": arm, "query_type": query_type, "pool": pool,
                        "status": "ok", "n_rows": stack.n_rows, "coverage": stack.coverage,
                    })

        # strict cross-arm transfer (secondary; per model, primary query type, common pools)
        if config.do_transfer:
            for query_type in config.query_types:
                if query_type != PRIMARY_QUERY_TYPE:
                    continue
                for pool in COMMON_POOLS:
                    for from_arm, to_arm in config.transfer_pairs:
                        if from_arm not in config.arms or to_arm not in config.arms:
                            continue
                        try:
                            rows = strict_cross_arm_transfer(
                                table, folds, model_key=model_key, query_type=query_type, pool=pool,
                                from_arm=from_arm, to_arm=to_arm,
                                layers=resolve_layers(config.nested_layer_spec, max_layer),
                                c_grid=config.c_grid, max_iter=config.max_iter,
                            )
                        except Exception as exc:  # pragma: no cover - defensive
                            summary["cell_status"].append({
                                "model_key": model_key, "transfer": f"{from_arm}->{to_arm}", "pool": pool,
                                "status": "skipped", "reason": f"{type(exc).__name__}: {exc}",
                            })
                            continue
                        if rows:
                            transfer_rows.extend(rows)
                        else:
                            summary["cell_status"].append({
                                "model_key": model_key, "transfer": f"{from_arm}->{to_arm}", "pool": pool,
                                "status": "skipped", "reason": "no_transfer_rows",
                            })

    # baselines
    baseline_rows: list[dict[str, Any]] = []
    for query_type in config.query_types:
        baseline_rows.extend(intercept_baseline(table, folds, query_type=query_type))
        baseline_rows.extend(object_identity_baseline(table, folds, query_type=query_type))
    baseline_rows.extend(missingness)
    baseline_rows.extend(confound_as_baseline(confound))

    # model contrasts (principal named-checkpoint contrast)
    contrast_rows = model_contrast_rows(nested_rows, model_a=PRINCIPAL_CONTRAST[0], model_b=PRINCIPAL_CONTRAST[1])

    # write outputs (parquet only when non-empty)
    if probe_rows:
        write_parquet(out_dir / "probe_results.parquet", probe_rows)
        summary["outputs"].append("probe_results.parquet")
    if nested_rows:
        write_parquet(out_dir / "nested_best_layer.parquet", nested_rows)
        summary["outputs"].append("nested_best_layer.parquet")
    if oof_rows:
        write_parquet(out_dir / "oof_predictions.parquet", oof_rows)
        summary["outputs"].append("oof_predictions.parquet")
    if baseline_rows:
        write_parquet(out_dir / "baseline_results.parquet", baseline_rows)
        summary["outputs"].append("baseline_results.parquet")
    if transfer_rows:
        write_parquet(out_dir / "transfer_results.parquet", transfer_rows)
        summary["outputs"].append("transfer_results.parquet")
    if contrast_rows:
        write_parquet(out_dir / "model_contrasts.parquet", contrast_rows)
        summary["outputs"].append("model_contrasts.parquet")
    write_json(out_dir / "permutation_results.json", {
        "method": METHOD_NAMES["permutation"],
        "method_caveat": METHOD_NAMES["permutation_caveat"],
        "n_permutations_config": config.permutations,
        "results": permutation_results,
    })
    summary["outputs"].append("permutation_results.json")

    figures = render_figures(out_dir, cell_summaries)
    summary["figures"] = figures
    summary["outputs"].extend(figures)

    report = generate_report(
        config, table, folds, summary, confound=confound, missingness=missingness,
        cell_summaries=cell_summaries, permutation_results=permutation_results,
        transfer_rows=transfer_rows, contrast_rows=contrast_rows,
        baseline_rows=baseline_rows, nested_rows=nested_rows,
    )
    write_text(out_dir / "report.md", report)
    summary["outputs"].extend(["manifest.json", "data_audit.json", "fold_assignments.json", "report.md"])
    summary["elapsed_s"] = time.perf_counter() - t0
    summary["outputs"].append("run_summary.json")
    write_json(out_dir / "run_summary.json", summary)
    return summary


def confound_as_baseline(confound: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in confound:
        rows.append({
            "baseline": "image_token_budget_confound",
            "model_key": row["model_key"],
            "arm": row["arm"],
            "n_rows": row["n_rows"],
            "image_token_min": row["image_token_min"],
            "image_token_median": row["image_token_median"],
            "image_token_max": row["image_token_max"],
            "method": "mask_meta_image_token_count_min_median_max",
        })
    return rows


# ---------------------------------------------------------------------------
# figures + report
# ---------------------------------------------------------------------------


def render_figures(out_dir: Path, cell_summaries: list[dict[str, Any]]) -> list[str]:
    if not cell_summaries:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figures_dir = Path(out_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # one figure per (model, pool) for the primary query type: AUROC vs layer_frac by arm
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for cell in cell_summaries:
        if not cell.get("per_layer"):
            continue
        key = (cell["model_key"], cell["query_type"], cell["pool"])
        grouped.setdefault(key, []).append(cell)

    for (model_key, query_type, pool), cells in sorted(grouped.items()):
        has_data = any(any(not math.isnan(p["auroc"]) for p in c["per_layer"]) for c in cells)
        if not has_data:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cell in sorted(cells, key=lambda c: c["arm"]):
            fracs = [p["layer_frac"] for p in cell["per_layer"]]
            aurocs = [p["auroc"] for p in cell["per_layer"]]
            ax.plot(fracs, aurocs, marker="o", ms=3, label=cell["arm"])
        ax.axhline(0.5, color="grey", ls="--", lw=0.8)
        ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("normalized layer depth (layer_index / max_layer_index)")
        ax.set_ylabel("OOF AUROC (descriptive)")
        ax.set_title(f"{model_key} | {query_type} | {pool}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        rel = f"figures/auroc_{model_key}__{query_type}__{pool}.png"
        tmp = figures_dir / f".{Path(rel).name}.{os.getpid()}.tmp.png"
        fig.savefig(tmp, dpi=120)
        (figures_dir / Path(rel).name).unlink(missing_ok=True)
        tmp.replace(figures_dir / Path(rel).name)
        plt.close(fig)
        written.append(rel)
    return written


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def generate_report(
    config: AnalysisConfig,
    table: CanonicalTable,
    folds: dict[str, Any],
    summary: dict[str, Any],
    *,
    confound: list[dict[str, Any]],
    missingness: list[dict[str, Any]],
    cell_summaries: list[dict[str, Any]] | None = None,
    permutation_results: list[dict[str, Any]] | None = None,
    transfer_rows: list[dict[str, Any]] | None = None,
    contrast_rows: list[dict[str, Any]] | None = None,
    baseline_rows: list[dict[str, Any]] | None = None,
    nested_rows: list[dict[str, Any]] | None = None,
) -> str:
    cell_summaries = cell_summaries or []
    permutation_results = permutation_results or []
    transfer_rows = transfer_rows or []
    contrast_rows = contrast_rows or []
    baseline_rows = baseline_rows or []
    nested_rows = nested_rows or []

    lines: list[str] = []
    lines.append("# Phase 2 Job2 representation-probe analysis")
    lines.append("")
    lines.append(f"- Mode: `{config.mode}`")
    lines.append(f"- Source run: `{config.source_run}`")
    lines.append(f"- Models: {', '.join(config.models)}")
    lines.append(f"- Arms: {', '.join(config.arms)}")
    lines.append(f"- Query types: {', '.join(config.query_types)}")
    lines.append(f"- Layer spec: `{config.layer_spec}`")
    lines.append(f"- Nested layer-selection candidates: `{config.nested_layer_spec}`")
    lines.append(f"- Outer folds: {folds['n_outer']} | inner folds: {config.n_inner} | repeats: {config.repeats} | seed: {config.seed}")
    lines.append(f"- Bootstrap: {config.bootstrap} | permutations: {config.permutations}")
    lines.append(f"- Regularization protocol: `{regularization_protocol(config.c_grid)}`")
    lines.append(f"- Data audit status: `{table.audit['status']}`")
    lines.append("")

    lines.append("## Claim boundary (read this first)")
    lines.append("")
    for statement in CLAIM_BOUNDARY_STATEMENTS:
        lines.append(f"- {statement}")
    lines.append("")

    lines.append("## Alignment and grouping")
    lines.append("")
    lines.append(f"- Source queries: {len(table.label_by_query)}; participants (== videos): {len(table.participants)}.")
    lines.append(f"- Query-type counts: {table.audit['query_type_counts']}.")
    lines.append(f"- Label mismatches across arms/models: {len(table.audit['label_mismatches'])}.")
    lines.append(f"- Participant mismatches: {len(table.audit['participant_mismatches'])}.")
    lines.append(f"- Query-type mismatches: {len(table.audit.get('query_type_mismatches', []))}.")
    lines.append(f"- Duplicate aligned rows: {len(table.audit.get('duplicate_rows', []))}.")
    lines.append(f"- Missing required model/arm/query cells: {len(table.audit.get('missing_cells', []))}.")
    lines.append("- Folds are grouped by participant, frozen once, and reused for every model, arm, pool, and layer.")
    lines.append("")

    lines.append("## Image-token budget confound (per model / arm)")
    lines.append("")
    lines.append("| model | arm | n | min | median | max |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in confound:
        lines.append(
            f"| {row['model_key']} | {row['arm']} | {row['n_rows']} | {row['image_token_min']} "
            f"| {_fmt(row['image_token_median'])} | {row['image_token_max']} |"
        )
    lines.append("")
    lines.append("Qwen and Gemma see very different visual-token budgets at 'the same arm'; "
                 "this budget is a confound reported here and never adjusted away.")
    lines.append("")

    if missingness:
        lines.append("## Qwen region-pool coverage (complete cases, no imputation)")
        lines.append("")
        lines.append("| model | arm | pool | present | total | coverage |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for row in missingness:
            lines.append(
                f"| {row['model_key']} | {row['arm']} | {row['pool']} | {row['n_present']} "
                f"| {row['n_cell_rows']} | {_fmt(row['coverage_fraction'])} |"
            )
        lines.append("")

    if cell_summaries:
        lines.append("## Descriptive best-layer OOF AUROC (unadjusted post-selection CI)")
        lines.append("")
        lines.append("These are DESCRIPTIVE peaks and the bootstrap CI conditions on the observed "
                     "argmax layer; it is not selection-adjusted and must not be used as an inferential "
                     "best-layer CI. The unbiased selected-layer estimate is in "
                     "`nested_best_layer.parquet`. Layer-0 `last_token`/`text_tail_mean` rows are the embedding/text floor.")
        lines.append("")
        lines.append("| model | arm | query | pool | best layer | AUROC | CI95 | rows | coverage |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |")
        for cell in cell_summaries:
            hb = cell.get("headline_bootstrap") or {}
            cov = cell.get("coverage", {})
            ci = f"[{_fmt(hb.get('ci95_low'))}, {_fmt(hb.get('ci95_high'))}]" if hb else "n/a"
            lines.append(
                f"| {cell['model_key']} | {cell['arm']} | {cell['query_type']} | {cell['pool']} "
                f"| {_fmt(hb.get('layer_index'))} | {_fmt(hb.get('auroc'))} | {ci} "
                f"| {cov.get('n_rows_used', '?')} | {_fmt(cov.get('coverage_fraction'))} |"
            )
        lines.append("")

    if nested_rows:
        import pandas as pd

        lines.append("## Nested selected-layer estimate")
        lines.append("")
        lines.append(
            "Layer and regularization are selected inside each outer training fold. "
            f"Candidate layers follow `{config.nested_layer_spec}`. "
            "The table reports the mean outer-fold metric; fold rows remain in "
            "`nested_best_layer.parquet`."
        )
        lines.append("")
        frame = pd.DataFrame(nested_rows)
        agg = frame.groupby(
            ["model_key", "arm", "query_type", "pool"], as_index=False
        ).agg(
            mean_outer_fold_auroc=("auroc", "mean"),
            mean_outer_fold_ce_bits=("ce_bits", "mean"),
            mean_selected_layer_frac=("selected_layer_frac", "mean"),
            outer_fold_rows=("fold", "count"),
        )
        lines.append("| model | arm | query | pool | mean fold AUROC | mean CE bits | selected depth | fold rows |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
        for _, row in agg.iterrows():
            lines.append(
                f"| {row['model_key']} | {row['arm']} | {row['query_type']} | {row['pool']} "
                f"| {_fmt(float(row['mean_outer_fold_auroc']))} "
                f"| {_fmt(float(row['mean_outer_fold_ce_bits']))} "
                f"| {_fmt(float(row['mean_selected_layer_frac']))} "
                f"| {int(row['outer_fold_rows'])} |"
            )
        lines.append("")

    if permutation_results:
        lines.append("## Exploratory fixed-score max-over-layers permutation")
        lines.append("")
        lines.append(f"Method: `{METHOD_NAMES['permutation']}`. Caveat: {METHOD_NAMES['permutation_caveat']}")
        lines.append("")
        lines.append("| model | arm | pool | observed max AUROC | p (max-stat) | null q95 |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for res in permutation_results:
            lines.append(
                f"| {res['model_key']} | {res['arm']} | {res['pool']} | {_fmt(res['observed_max_auroc'])} "
                f"| {_fmt(res['p_value_max_stat'])} | {_fmt(res['null_max_quantiles']['q95'])} |"
            )
        lines.append("")

    if transfer_rows:
        import pandas as pd

        lines.append("## Strict cross-arm transfer (train outer-train arm A, test outer-test arm B)")
        lines.append("")
        frame = pd.DataFrame(transfer_rows)
        agg = frame.groupby(["model_key", "from_arm", "to_arm", "pool"], as_index=False)["auroc"].mean()
        all_disjoint = bool(frame["participants_disjoint"].all())
        lines.append(f"- Participants disjoint in every transfer fold: `{all_disjoint}` (guardrail).")
        lines.append("")
        lines.append("| model | from | to | pool | mean AUROC |")
        lines.append("| --- | --- | --- | --- | ---: |")
        for _, row in agg.iterrows():
            lines.append(
                f"| {row['model_key']} | {row['from_arm']} | {row['to_arm']} | {row['pool']} | {_fmt(float(row['auroc']))} |"
            )
        lines.append("")

    if contrast_rows:
        import pandas as pd

        lines.append(f"## Principal contrast: {PRINCIPAL_CONTRAST[0]} vs {PRINCIPAL_CONTRAST[1]}")
        lines.append("")
        frame = pd.DataFrame(contrast_rows)
        agg = frame.groupby(["arm", "query_type", "pool"], as_index=False)["delta_auroc"].mean()
        lines.append("Paired named-checkpoint nested-Theta* delta on shared source rows and folds. "
                     "Gemma26-A4B is a distinct MoE row; no family averages are formed.")
        lines.append("")
        lines.append("| arm | query | pool | mean delta AUROC (A-B) |")
        lines.append("| --- | --- | --- | ---: |")
        for _, row in agg.iterrows():
            lines.append(f"| {row['arm']} | {row['query_type']} | {row['pool']} | {_fmt(float(row['delta_auroc']))} |")
        lines.append("")

    lines.append("## Shortcut baselines and floors")
    lines.append("")
    metric_baselines = [
        row
        for row in baseline_rows
        if row.get("baseline") in {"intercept_class_prior", "object_identity_tfidf"}
        and "auroc" in row
    ]
    if metric_baselines:
        import pandas as pd

        frame = pd.DataFrame(metric_baselines)
        agg = frame.groupby(["baseline", "query_type"], as_index=False).agg(
            mean_outer_fold_auroc=("auroc", "mean"),
            mean_outer_fold_ce_bits=("ce_bits", "mean"),
            fold_rows=("fold", "count"),
        )
        lines.append("| baseline | query | mean fold AUROC | mean CE bits | fold rows |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for _, row in agg.iterrows():
            lines.append(
                f"| {row['baseline']} | {row['query_type']} "
                f"| {_fmt(float(row['mean_outer_fold_auroc']))} "
                f"| {_fmt(float(row['mean_outer_fold_ce_bits']))} "
                f"| {int(row['fold_rows'])} |"
            )
        lines.append("")
    lines.append("- Intercept / class-prior baseline (chance floor).")
    lines.append("- Object-identity / public-question baseline: TF-IDF(question + target) with the identical grouped outer folds, train-only fit.")
    lines.append("- Embedding/text floor: layer-0 `last_token` and `text_tail_mean` probe rows, flagged `is_text_floor` in `probe_results.parquet`.")
    lines.append("- Missingness-only coverage audit for optional Qwen region pools.")
    lines.append("- Image-token-budget confound table (min/median/max per model/arm).")
    lines.append("")
    lines.append("The fixed-OOF permutation above is exploratory only. Confirmatory evidence for "
                 "the principal current-state last-token cells comes from the separate full-refit "
                 "permutation artifact; all other permutation-backed existence statements remain descriptive.")
    lines.append("")

    lines.append("## Cell status")
    lines.append("")
    n_ok = sum(1 for s in summary.get("cell_status", []) if s.get("status") == "ok")
    n_skipped = sum(1 for s in summary.get("cell_status", []) if s.get("status") == "skipped")
    n_error = sum(1 for s in summary.get("cell_status", []) if s.get("status") == "error")
    lines.append(f"- ok: {n_ok} | skipped: {n_skipped} | error: {n_error}")
    lines.append(f"- required-cell failures: {len(summary.get('required_cell_failures', []))}")
    lines.append("")

    return "\n".join(lines) + "\n"
