#!/usr/bin/env python3
"""Hash-bound behavioral inference for the re-anchored EgoHOS 2AFC battery.

Workers see only public request packs. Full analysis uses the contract-bound
analyzer, the only component that opens the hidden evaluator index. Canary
workers consume only public rows; the coordinator may hash the sealed evaluator
file for integrity but never parses, joins, or emits outcome metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ARMS = ("real", "blank_full", "wrong_time")
CHOICES = ("A", "B")
# Legacy fixture/export only. Runtime provenance is always derived from the
# operational contract through _contract_preregistration_sha256().
EXPECTED_PREREGISTRATION_SHA256 = (
    "bd0f021fd510d873f388075a3d4c0e0d138da07fc6da8215cc0ff16b1d09279f"
)
EXPECTED_TARGET_COMPONENT_RULE = (
    "within the chosen class mask, select the largest 8-connected component with "
    "area at least 64 pixels; break exact area ties by lexicographically smallest "
    "top,left,bottom,right bounding box; render only that component and record its "
    "hidden fingerprint"
)
COMMON_ALLOCATION_MEAN_ROLE = (
    "descriptive_advisory_union_window_absent_rank_buckets_zero_filled"
)
FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "answer_position",
    "binary_answer",
    "canonical_answer_json",
    "class",
    "class_id",
    "correct",
    "correctness",
    "expected_answer",
    "first_order",
    "hidden_label",
    "label",
    "label_role",
    "role",
    "second_order",
    "target_role",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_now_precise() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            yield row


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def contains_forbidden_public_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_PUBLIC_KEYS:
                return True
            if contains_forbidden_public_key(child):
                return True
    elif isinstance(value, list):
        return any(contains_forbidden_public_key(child) for child in value)
    return False


def strict_parse_choice(raw_text: str) -> dict[str, Any]:
    """Parse exactly one canonical A/B JSON object without recovery heuristics."""
    try:
        payload = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        return {"valid": False, "answer": None, "error": f"json_decode:{exc.msg}"}
    if not isinstance(payload, dict):
        return {"valid": False, "answer": None, "error": "not_object"}
    expected_keys = {"answer", "abstain", "abstention_reason", "confidence"}
    if set(payload) != expected_keys:
        return {"valid": False, "answer": None, "error": "schema_keys"}
    if payload.get("answer") not in CHOICES:
        return {"valid": False, "answer": None, "error": "answer_not_A_or_B"}
    if type(payload.get("abstain")) is not bool:
        return {"valid": False, "answer": None, "error": "abstain_not_boolean"}
    if payload.get("abstention_reason") is not None and not isinstance(
        payload.get("abstention_reason"), str
    ):
        return {"valid": False, "answer": None, "error": "bad_abstention_reason"}
    confidence = payload.get("confidence")
    if type(confidence) not in {int, float} or not math.isfinite(float(confidence)):
        return {"valid": False, "answer": None, "error": "bad_confidence"}
    if not 0.0 <= float(confidence) <= 1.0:
        return {"valid": False, "answer": None, "error": "confidence_out_of_range"}
    if payload["abstain"] is False and payload["abstention_reason"] is not None:
        return {"valid": False, "answer": None, "error": "unexpected_abstention_reason"}
    return {
        "valid": True,
        "answer": str(payload["answer"]),
        "abstain": bool(payload["abstain"]),
        "abstention_reason": payload["abstention_reason"],
        "confidence": float(confidence),
        "error": None,
    }


def score_candidate_suffix_from_logits(
    *, base_input_ids: Sequence[int], full_input_ids: Sequence[int], logits: Any
) -> dict[str, Any]:
    """CPU/GPU-neutral exact causal suffix scoring used by tests and workers."""
    from research.lifecycle_memory.phase2_job2c_behavioral_companion import (
        score_candidate_suffix_from_logits as score_suffix,
    )

    return score_suffix(
        base_input_ids=base_input_ids,
        full_input_ids=full_input_ids,
        logits=logits,
    )


def _contract_preregistration_sha256(contract: Mapping[str, Any]) -> str:
    binding = contract.get("governing_preregistration")
    if not isinstance(binding, dict):
        raise ValueError("operational contract lacks preregistration binding")
    expected = str(binding.get("sha256") or "").lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("operational contract has an invalid preregistration SHA-256")
    return expected


def _load_contract(contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = contract_path.expanduser().resolve()
    contract = read_json(contract_path)
    if contract.get("kind") != "phase2_reanchored_job_a_operational_contract":
        raise ValueError("invalid Job A operational contract kind")
    binding = contract.get("governing_preregistration")
    if not isinstance(binding, dict):
        raise ValueError("operational contract lacks preregistration binding")
    prereg_path = (REPO_ROOT / str(binding.get("path") or "")).resolve()
    if not is_within(prereg_path, REPO_ROOT) or not prereg_path.is_file():
        raise ValueError("governing preregistration path is missing or escaped")
    expected = _contract_preregistration_sha256(contract)
    if sha256_file(prereg_path) != expected:
        raise ValueError("stale preregistration hash: governing file mismatch")
    prereg = read_json(prereg_path)
    if prereg.get("kind") != "phase2_reanchored_evidence_failure_localization_prespec":
        raise ValueError("invalid governing preregistration kind")
    if tuple(prereg.get("candidate_completions", {})) != CHOICES:
        raise ValueError("governing candidate choice order drift")
    if tuple(prereg.get("data", {}).get("arms", [])) != ARMS:
        raise ValueError("governing arm order drift")
    if prereg.get("prompt_validity_gate") is not None:
        raise ValueError("stale binary validity gate reappeared in governing preregistration")
    if (
        prereg.get("data", {}).get("selection", {}).get("target_component_rule")
        != EXPECTED_TARGET_COMPONENT_RULE
    ):
        raise ValueError("governing largest-component target rule drift")

    rights = contract.get("governing_rights_ledger")
    if not isinstance(rights, dict):
        raise ValueError("operational contract lacks rights-ledger binding")
    rights_path = (REPO_ROOT / str(rights.get("path") or "")).resolve()
    if not is_within(rights_path, REPO_ROOT) or not rights_path.is_file():
        raise ValueError("rights ledger path is missing or escaped")
    if sha256_file(rights_path) != str(rights.get("sha256") or ""):
        raise ValueError("rights ledger hash mismatch")
    return contract, prereg


def _load_model_plan(path: Path, prereg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_json(path.expanduser().resolve())
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("model plan must contain exactly four rows")
    expected = {
        row["model_key"]: (row["model_id"], row["revision"])
        for row in prereg.get("models", [])
    }
    observed_workers: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("model plan rows must be objects")
        model_key = str(raw.get("model_key") or "")
        analysis_model_key = str(raw.get("analysis_model_key") or model_key)
        worker_index = int(raw.get("worker_index", -1))
        if analysis_model_key not in expected:
            raise ValueError(f"unexpected analysis model key: {analysis_model_key}")
        if (str(raw.get("model_id") or ""), str(raw.get("revision") or "")) != expected[
            analysis_model_key
        ]:
            raise ValueError(f"model identity drift for {analysis_model_key}")
        if raw.get("dtype") != "bf16":
            raise ValueError(f"dtype drift for {model_key}")
        if worker_index not in range(4) or worker_index in observed_workers:
            raise ValueError("worker indexes must be the unique integers 0..3")
        observed_workers.add(worker_index)
        normalized.append(
            {
                "model_key": model_key,
                "analysis_model_key": analysis_model_key,
                "model_id": str(raw["model_id"]),
                "revision": str(raw["revision"]),
                "dtype": "bf16",
                "worker_index": worker_index,
                "canary_microbatch_size": int(raw.get("canary_microbatch_size", 0)),
            }
        )
        if normalized[-1]["canary_microbatch_size"] < 1:
            raise ValueError(f"invalid canary microbatch size for {model_key}")
    operational_keys = [row["model_key"] for row in normalized]
    if len(set(operational_keys)) != 4:
        raise ValueError("operational model keys must be unique")
    analysis_counts: dict[str, int] = {}
    for row in normalized:
        key = row["analysis_model_key"]
        analysis_counts[key] = analysis_counts.get(key, 0) + 1
    expected_replicas = 4 // len(expected) if len(expected) in {2, 4} else 0
    if (
        observed_workers != set(range(4))
        or set(analysis_counts) != set(expected)
        or expected_replicas < 1
        or set(analysis_counts.values()) != {expected_replicas}
    ):
        raise ValueError("model plan does not evenly cover the frozen analysis models")
    return sorted(normalized, key=lambda row: row["worker_index"])


def _resolve_public_path(data_dir: Path, raw_path: str) -> Path:
    path = (data_dir / raw_path).resolve()
    if not is_within(path, data_dir):
        raise ValueError(f"public input path escaped data root: {raw_path}")
    if not path.is_file():
        raise FileNotFoundError(f"missing public input: {path}")
    return path


def _image_identity(path: Path, cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    if path in cache:
        return cache[path]
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        width, height = image.size
    identity = {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "width": int(width),
        "height": int(height),
    }
    cache[path] = identity
    return identity


def request_identity(
    request: dict[str, Any], *, data_dir: Path, image_cache: dict[Path, dict[str, Any]]
) -> dict[str, Any]:
    if contains_forbidden_public_key(request):
        raise ValueError("public request crossed the hidden-label firewall")
    item_id = str(request.get("item_id") or request.get("trial_id") or "")
    request_id = str(request.get("request_id") or "")
    if not item_id or not request_id:
        raise ValueError("public request lacks item_id/trial_id or request_id")
    system_prompt = request.get("system_prompt")
    user_prompt = request.get("user_prompt")
    if not isinstance(system_prompt, str) or not system_prompt:
        raise ValueError("public request lacks system_prompt")
    if not isinstance(user_prompt, str) or not user_prompt:
        raise ValueError("public request lacks user_prompt")
    inputs = request.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 3:
        raise ValueError("2AFC public request must contain exactly three image inputs")
    expected_kinds = ("full_frame", "reference_crop_a", "reference_crop_b")
    input_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(inputs):
        if not isinstance(raw, dict) or not raw.get("path"):
            raise ValueError("invalid public image input")
        if str(raw.get("kind") or "") != expected_kinds[index]:
            raise ValueError("2AFC public image input order/kind drift")
        relative = str(raw["path"])
        path = _resolve_public_path(data_dir, relative)
        observed = _image_identity(path, image_cache)
        if raw.get("width") is not None and int(raw["width"]) != observed["width"]:
            raise ValueError(f"image width metadata mismatch: {relative}")
        if raw.get("height") is not None and int(raw["height"]) != observed["height"]:
            raise ValueError(f"image height metadata mismatch: {relative}")
        input_rows.append(
            {
                "index": index,
                "path": relative,
                "kind": str(raw.get("kind") or ""),
                **observed,
            }
        )
    return {
        "item_id": item_id,
        "request_id": request_id,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
        "ordered_inputs": input_rows,
        "serialization_sha256": canonical_sha256(
            {
                "content_order": ["system_prompt", "input_1_full", "input_2_crop_A", "input_3_crop_B", "user_prompt"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "ordered_image_sha256": [row["sha256"] for row in input_rows],
            }
        ),
    }


def _pack_requests(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("requests")
    else:
        rows = payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("public scoring pack must contain a request list")
    return rows


def _manifest_count(manifest: dict[str, Any], names: Sequence[str]) -> int | None:
    for name in names:
        value = manifest.get(name)
        if type(value) is int:
            return int(value)
    return None


def _verify_hash_manifest(data_dir: Path, path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("kind") != "phase2_reanchored_egohos_job_a_2afc_hash_manifest":
        raise ValueError("invalid battery hash-manifest kind")
    if payload.get("algorithm") != "sha256":
        raise ValueError("battery hash manifest does not use SHA-256")
    entries = payload.get("entries")
    if not isinstance(entries, list) or payload.get("entry_count") != len(entries):
        raise ValueError("battery hash-manifest entry count drift")
    observed_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("battery hash-manifest row is not an object")
        relative = str(entry.get("path") or "")
        target = (data_dir / relative).resolve()
        if not relative or relative in observed_paths or not is_within(target, data_dir):
            raise ValueError("battery hash-manifest path is duplicate, empty, or escaped")
        if not target.is_file():
            raise FileNotFoundError(f"hash-manifest input is missing: {relative}")
        if entry.get("bytes") != target.stat().st_size:
            raise ValueError(f"hash-manifest byte count drift: {relative}")
        if entry.get("sha256") != sha256_file(target):
            raise ValueError(f"hash-manifest SHA-256 drift: {relative}")
        observed_paths.add(relative)
    actual_paths = {
        str(candidate.relative_to(data_dir))
        for candidate in data_dir.rglob("*")
        if candidate.is_file()
        and candidate.resolve() != path.resolve()
    }
    if observed_paths != actual_paths:
        missing = sorted(observed_paths - actual_paths)
        extra = sorted(actual_paths - observed_paths)
        raise ValueError(
            "battery hash manifest does not cover the exact data tree: "
            f"missing_count={len(missing)} extra_count={len(extra)} "
            f"missing_sample={missing[:8]} extra_sample={extra[:8]}"
        )
    return payload


def _code_snapshot_sha256(contract_path: Path, model_plan_path: Path) -> str:
    job_dir = model_plan_path.resolve().parent
    analyzer_name = str(read_json(contract_path).get("analyzer_relative_path") or "")
    analyzer_path = (
        (REPO_ROOT / analyzer_name).resolve()
        if analyzer_name
        else REPO_ROOT / "research/lifecycle_memory/analyze_phase2_reanchored_behavior.py"
    )
    paths = [
        Path(__file__).resolve(),
        analyzer_path,
        REPO_ROOT / "research/lifecycle_memory/phase2_job3_heldout_confirmation.py",
        REPO_ROOT / "research/lifecycle_memory/phase2_job2c_behavioral_companion.py",
        REPO_ROOT / "docs/research/lifecycle_memory/phase 2 research/jobs/job0/run_phase1_qwen27b_hf.py",
        contract_path.resolve(),
        model_plan_path.resolve(),
        job_dir / "run_tamia_reanchored_jobA.sh",
        job_dir / "tamia_jobA_worker.sh",
        job_dir / "job_tamia_reanchored_jobA_canary_h100.sh",
        job_dir / "job_tamia_reanchored_jobA_full_h100.sh",
        job_dir / "verify_tamia_reanchored_jobA.sh",
        job_dir / "submit_canary.sh",
        job_dir / "submit_full.sh",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing transitive runtime source: {path}")
        try:
            identity = str(path.relative_to(REPO_ROOT))
        except ValueError:
            identity = str(path.resolve())
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_batch_profile(
    path: Path,
    *,
    model_plan_path: Path,
    contract_path: Path,
    model_plan: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = read_json(path)
    if payload.get("kind") != "phase2_reanchored_job_a_batch_profile":
        raise ValueError("invalid canary batch-profile kind")
    if payload.get("status") != "pass" or payload.get("source_mode") != "canary":
        raise ValueError("full mode requires a passing canary batch profile")
    contract = read_json(contract_path)
    preregistration_sha256 = _contract_preregistration_sha256(contract)
    if payload.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError("batch profile preregistration drift")
    if payload.get("model_plan_sha256") != sha256_file(model_plan_path):
        raise ValueError("batch profile model-plan drift")
    if payload.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("batch profile contract drift")
    if payload.get("code_snapshot_sha256") != _code_snapshot_sha256(
        contract_path, model_plan_path
    ):
        raise ValueError("batch profile code-snapshot drift")
    for field, environment in (
        ("sif_sha256", "SIF_SHA256"),
        ("release_manifest_sha256", "RELEASE_MANIFEST_SHA256"),
        ("data_manifest_sha256", "DATA_MANIFEST_SHA256"),
        ("hash_manifest_sha256", "HASH_MANIFEST_SHA256"),
    ):
        expected = str(os.environ.get(environment) or "").lower()
        if len(expected) != 64 or payload.get(field) != expected:
            raise ValueError(f"batch profile {field} drift")
    rows = payload.get("models")
    if not isinstance(rows, list):
        raise ValueError("batch profile lacks model rows")
    if payload.get("canary_measurements_sha256") != canonical_sha256(rows):
        raise ValueError("batch profile canary measurement hash drift")
    fresh = payload.get("fresh_canary_provenance")
    if (
        not isinstance(fresh, dict)
        or fresh.get("status") != "pass"
        or fresh.get("required_rows_present_at_start") != 0
    ):
        raise ValueError("batch profile does not prove a fresh canary")
    fresh_models = fresh.get("models")
    if not isinstance(fresh_models, dict):
        raise ValueError("batch profile lacks per-model fresh-canary receipts")
    by_key = {str(row.get("model_key") or ""): row for row in rows if isinstance(row, dict)}
    if len(rows) != len(by_key) or set(by_key) != {row["model_key"] for row in model_plan}:
        raise ValueError("batch profile model set drift")
    if set(fresh_models) != set(by_key) or any(
        not isinstance(row, dict)
        or row.get("fresh_canary") is not True
        or row.get("rows_present_at_start") != [0]
        for row in fresh_models.values()
    ):
        raise ValueError("batch profile canary was resumed or pre-populated")
    inference_distributions: list[dict[str, Any]] = []
    for model_key, row in by_key.items():
        if int(row.get("effective_batch_size", 0)) < 1:
            raise ValueError(f"invalid effective batch size for {model_key}")
        planned = next(
            model["canary_microbatch_size"]
            for model in model_plan
            if model["model_key"] == model_key
        )
        if int(row["effective_batch_size"]) > int(planned):
            raise ValueError(f"batch profile exceeds tested canary size for {model_key}")
        if row.get("strategy") not in {"batched", "sequential"}:
            raise ValueError(f"invalid batch strategy for {model_key}")
        inference_utilization = row.get("gpu_inference_window_utilization_distribution")
        if (
            not isinstance(inference_utilization, dict)
            or inference_utilization.get("status") != "pass"
        ):
            raise ValueError(f"invalid inference-window telemetry for {model_key}")
        _finite_metric(
            f"{model_key}.inference_mean_pct",
            inference_utilization.get("mean_pct"),
            lower=0.0,
            upper=100.0,
        )
        _finite_metric(
            f"{model_key}.inference_p90_pct",
            inference_utilization.get("p90_pct"),
            lower=0.0,
            upper=100.0,
        )
        sample_count = inference_utilization.get("sample_count")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError(f"invalid inference-window sample count for {model_key}")
        inference_distributions.append(inference_utilization)
    gate = payload.get("utilization_tail_gate")
    if not isinstance(gate, dict) or gate.get("status") != "pass":
        raise ValueError("full mode requires a passing utilization/tail gate")
    schedule = payload.get("schedule")
    if not isinstance(schedule, dict) or schedule.get("kind") != (
        "phase2_reanchored_job_a_balanced_full_schedule_v2"
    ):
        raise ValueError("batch profile lacks the frozen balanced schedule")
    if schedule.get("sha256") != canonical_sha256(
        {key: value for key, value in schedule.items() if key != "sha256"}
    ):
        raise ValueError("balanced schedule hash drift")
    thresholds = contract.get("canary", {}).get("utilization_tail_gate", {})
    gate_thresholds = gate.get("thresholds")
    if gate_thresholds != thresholds:
        raise ValueError("utilization/tail threshold binding drift")
    common_utilization = payload.get("common_allocation_gpu_utilization")
    if (
        not isinstance(common_utilization, dict)
        or common_utilization.get("status") != "pass"
        or common_utilization.get("mean_pct")
        != gate.get("common_allocation_mean_gpu_utilization_pct")
    ):
        raise ValueError("common allocation utilization binding drift")
    selection_count = int(contract["battery"]["required_trial_count"])
    if int(schedule.get("selection_count", -1)) != selection_count:
        raise ValueError("balanced schedule selection-count drift")
    expected_schedule = build_balanced_full_schedule(
        [
            {
                **by_key[str(plan_row["model_key"])],
                "worker_index": int(plan_row["worker_index"]),
                "analysis_model_key": str(
                    plan_row.get("analysis_model_key") or plan_row["model_key"]
                ),
            }
            for plan_row in model_plan
        ],
        selection_count=selection_count,
        arm_count=len(ARMS),
    )
    if schedule != expected_schedule:
        raise ValueError("balanced schedule does not match the bound canary measurements")
    runtime_estimate = payload.get("full_runtime_estimate")
    if not isinstance(runtime_estimate, dict) or runtime_estimate.get(
        "fits_configured_full_walltime"
    ) is not True or int(runtime_estimate.get("configured_full_walltime_s", 0)) != 10800:
        raise ValueError("batch profile does not prove the three-hour walltime gate")
    expected_gate = evaluate_utilization_tail_gate(
        common_allocation_mean_pct=common_utilization.get("mean_pct"),
        minimum_worker_inference_mean_pct=min(
            float(row["mean_pct"]) for row in inference_distributions
        ),
        minimum_worker_inference_p90_pct=min(
            float(row["p90_pct"]) for row in inference_distributions
        ),
        minimum_worker_inference_sample_count=min(
            int(row["sample_count"]) for row in inference_distributions
        ),
        projected_rank_occupancy_fraction=schedule.get(
            "projected_aggregate_rank_occupancy_fraction"
        ),
        projected_rank_tail_idle_fraction=schedule.get(
            "projected_max_rank_tail_idle_fraction"
        ),
        recommended_walltime_s=int(
            runtime_estimate.get(
                "recommended_walltime_s_rounded_to_5_minutes", 0
            )
        ),
        configured_walltime_s=int(
            runtime_estimate.get(
                "configured_full_walltime_s", 0
            )
        ),
        thresholds=thresholds,
    )
    if gate != expected_gate:
        raise ValueError("utilization/tail gate recomputation drift")
    projected_makespan = _finite_metric(
        "schedule.projected_makespan_s",
        schedule.get("projected_makespan_s"),
        lower=0.0,
        upper=10800.0,
    )
    expected_recommended = int(
        math.ceil((projected_makespan * 1.20 + 600.0) / 300.0) * 300
    )
    if (
        runtime_estimate.get("balanced_schedule_projection_s") != projected_makespan
        or runtime_estimate.get("recommended_walltime_s_rounded_to_5_minutes")
        != expected_recommended
        or runtime_estimate.get("fits_configured_full_walltime")
        is not (expected_recommended <= 10800)
    ):
        raise ValueError("batch profile walltime recomputation drift")
    ranks = schedule.get("ranks")
    if not isinstance(ranks, list) or {row.get("rank") for row in ranks} != set(range(4)):
        raise ValueError("balanced schedule must contain ranks 0..3 exactly once")
    assignments = [
        assignment
        for rank_row in ranks
        for assignment in rank_row.get("assignments", [])
        if isinstance(assignment, dict)
    ]
    assignment_ids = [str(row.get("assignment_id") or "") for row in assignments]
    if not assignment_ids or len(assignment_ids) != len(set(assignment_ids)):
        raise ValueError("balanced schedule assignment IDs are missing or duplicated")
    for rank_row in ranks:
        for assignment in rank_row.get("assignments", []):
            if int(assignment.get("rank", -1)) != int(rank_row["rank"]):
                raise ValueError("balanced schedule rank binding drift")
            model_key = str(assignment.get("model_key") or "")
            if model_key not in by_key:
                raise ValueError("balanced schedule contains an unknown model")
            if int(assignment.get("effective_batch_size", 0)) != int(
                by_key[model_key]["effective_batch_size"]
            ) or assignment.get("strategy") != by_key[model_key]["strategy"]:
                raise ValueError("balanced schedule batch-profile binding drift")
    plan_by_instance = {row["model_key"]: row for row in model_plan}
    analysis_model_keys = sorted(
        {
            str(row.get("analysis_model_key") or row["model_key"])
            for row in model_plan
        }
    )
    for analysis_model_key in analysis_model_keys:
        intervals = sorted(
            (
                int(row.get("selection_start", -1)),
                int(row.get("selection_stop", -1)),
            )
            for row in assignments
            if str(
                plan_by_instance[str(row.get("model_key") or "")].get(
                    "analysis_model_key"
                )
                or row.get("model_key")
            )
            == analysis_model_key
        )
        cursor = 0
        for start, stop in intervals:
            if start != cursor or stop <= start or stop > selection_count:
                raise ValueError(
                    f"balanced schedule overlap/gap for {analysis_model_key}"
                )
            cursor = stop
        if cursor != selection_count:
            raise ValueError(
                f"balanced schedule incomplete for {analysis_model_key}"
            )
    return payload


def build_balanced_full_schedule(
    model_reports: Sequence[dict[str, Any]],
    *,
    selection_count: int,
    arm_count: int,
) -> dict[str, Any]:
    """Freeze a deterministic two-wave schedule from canary-only throughput.

    Every model starts on its canonical rank. Within each fast/slow rank pair, the
    fast rank may take a contiguous tail shard of the slow model after finishing
    its own model. Splits are aligned to the two-order frame pair.
    """
    if selection_count < 2 or selection_count % 2:
        raise ValueError("balanced scheduling requires an even selection count")
    if arm_count < 1 or len(model_reports) != 4:
        raise ValueError("balanced scheduling requires four model reports")
    reports = {str(row["model_key"]): dict(row) for row in model_reports}
    if len(reports) != 4:
        raise ValueError("balanced scheduling model keys are not unique")
    for row in reports.values():
        if int(row.get("worker_index", -1)) not in range(4):
            raise ValueError("balanced scheduling lacks canonical worker indices")
        if float(row.get("decisions_per_second") or 0.0) <= 0.0:
            raise ValueError("balanced scheduling requires positive throughput")

    analysis_groups: dict[str, list[str]] = {}
    for model_key, row in reports.items():
        analysis_key = str(row.get("analysis_model_key") or model_key)
        analysis_groups.setdefault(analysis_key, []).append(model_key)

    if len(analysis_groups) == 2 and {len(rows) for rows in analysis_groups.values()} == {2}:
        split = (selection_count // 4) * 2
        if split <= 0 or split >= selection_count:
            raise ValueError("replicated-anchor schedule cannot form paired half-shards")
        rank_rows: list[dict[str, Any]] = []
        for analysis_key in sorted(analysis_groups):
            instances = sorted(
                analysis_groups[analysis_key],
                key=lambda key: int(reports[key]["worker_index"]),
            )
            for shard_index, model_key in enumerate(instances):
                row = reports[model_key]
                start, stop = ((0, split), (split, selection_count))[shard_index]
                duration = float(row.get("load_time_s") or 0.0) + (
                    (stop - start) * arm_count / float(row["decisions_per_second"])
                )
                assignment = {
                    "assignment_id": f"{analysis_key}.shard-{shard_index}-of-2",
                    "rank": int(row["worker_index"]),
                    "model_key": model_key,
                    "analysis_model_key": analysis_key,
                    "shard_index": shard_index,
                    "shard_count": 2,
                    "selection_start": start,
                    "selection_stop": stop,
                    "effective_batch_size": int(row["effective_batch_size"]),
                    "strategy": str(row["strategy"]),
                    "projected_start_s": 0.0,
                    "projected_finish_s": duration,
                }
                rank_rows.append(
                    {
                        "rank": int(row["worker_index"]),
                        "assignments": [assignment],
                        "projected_runtime_s": duration,
                    }
                )
        rank_rows.sort(key=lambda row: int(row["rank"]))
        makespan = max(float(row["projected_runtime_s"]) for row in rank_rows)
        busy = sum(float(row["projected_runtime_s"]) for row in rank_rows)
        payload = {
            "kind": "phase2_reanchored_job_a_balanced_full_schedule_v2",
            "selection_count": selection_count,
            "arm_count": arm_count,
            "construction": "two_anchor_models_two_fixed_counterbalance_aligned_half_shards_each",
            "adaptation_during_full_run": False,
            "ranks": rank_rows,
            "projected_makespan_s": makespan,
            "projected_aggregate_rank_occupancy_fraction": busy / (4.0 * makespan),
            "projected_max_rank_tail_idle_fraction": max(
                (makespan - float(row["projected_runtime_s"])) / makespan
                for row in rank_rows
            ),
        }
        payload["sha256"] = canonical_sha256(payload)
        return payload

    def inference_seconds(row: dict[str, Any], count: int) -> float:
        return count * arm_count / float(row["decisions_per_second"])

    totals = {
        key: float(row.get("load_time_s") or 0.0)
        + inference_seconds(row, selection_count)
        for key, row in reports.items()
    }
    ordered = sorted(reports, key=lambda key: (totals[key], key))
    pairs = ((ordered[0], ordered[-1]), (ordered[1], ordered[-2]))
    assignments_by_rank: dict[int, list[dict[str, Any]]] = {rank: [] for rank in range(4)}
    split_tail: dict[str, tuple[int, int] | None] = {key: None for key in reports}
    helper_for: dict[str, str] = {}
    for fast_key, slow_key in pairs:
        fast_total = totals[fast_key]
        slow_inference = inference_seconds(reports[slow_key], selection_count)
        fraction = max(0.0, min(0.45, (slow_inference - fast_total) / (2.0 * slow_inference)))
        helper_frames = int(round((selection_count // 2) * fraction))
        if helper_frames >= max(8, int(math.ceil((selection_count // 2) * 0.05))):
            helper_count = helper_frames * 2
            split_tail[slow_key] = (selection_count - helper_count, selection_count)
            helper_for[slow_key] = fast_key

    for model_key, row in reports.items():
        rank = int(row["worker_index"])
        tail = split_tail[model_key]
        stop = tail[0] if tail else selection_count
        shard_count = 2 if tail else 1
        assignments_by_rank[rank].append(
            {
                "assignment_id": f"{model_key}.shard-0-of-{shard_count}",
                "rank": rank,
                "model_key": model_key,
                "shard_index": 0,
                "shard_count": shard_count,
                "selection_start": 0,
                "selection_stop": stop,
                "effective_batch_size": int(row["effective_batch_size"]),
                "strategy": str(row["strategy"]),
            }
        )
    for slow_key, tail in split_tail.items():
        if tail is None:
            continue
        helper_key = helper_for[slow_key]
        helper_rank = int(reports[helper_key]["worker_index"])
        row = reports[slow_key]
        assignments_by_rank[helper_rank].append(
            {
                "assignment_id": f"{slow_key}.shard-1-of-2",
                "rank": helper_rank,
                "model_key": slow_key,
                "shard_index": 1,
                "shard_count": 2,
                "selection_start": tail[0],
                "selection_stop": tail[1],
                "effective_batch_size": int(row["effective_batch_size"]),
                "strategy": str(row["strategy"]),
            }
        )

    rank_rows = []
    for rank in range(4):
        elapsed = 0.0
        projected = []
        for assignment in assignments_by_rank[rank]:
            row = reports[assignment["model_key"]]
            duration = float(row.get("load_time_s") or 0.0) + inference_seconds(
                row, assignment["selection_stop"] - assignment["selection_start"]
            )
            projected.append(
                {
                    **assignment,
                    "projected_start_s": elapsed,
                    "projected_finish_s": elapsed + duration,
                }
            )
            elapsed += duration
        rank_rows.append(
            {"rank": rank, "assignments": projected, "projected_runtime_s": elapsed}
        )
    makespan = max(float(row["projected_runtime_s"]) for row in rank_rows)
    busy = sum(float(row["projected_runtime_s"]) for row in rank_rows)
    payload = {
        "kind": "phase2_reanchored_job_a_balanced_full_schedule_v2",
        "selection_count": selection_count,
        "arm_count": arm_count,
        "construction": "two_wave_fast_rank_takes_contiguous_slow_model_tail",
        "adaptation_during_full_run": False,
        "ranks": rank_rows,
        "projected_makespan_s": makespan,
        "projected_aggregate_rank_occupancy_fraction": busy / (4.0 * makespan),
        "projected_max_rank_tail_idle_fraction": max(
            (makespan - float(row["projected_runtime_s"])) / makespan for row in rank_rows
        ),
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def preflight(
    *,
    data_dir: Path,
    run_dir: Path,
    model_plan_path: Path,
    contract_path: Path,
    mode: str,
    limit_frames: int,
    batch_profile_path: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"canary", "full"}:
        raise ValueError("mode must be canary or full")
    data_dir = data_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    model_plan_path = model_plan_path.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    contract, prereg = _load_contract(contract_path)
    preregistration_sha256 = _contract_preregistration_sha256(contract)
    model_plan = _load_model_plan(model_plan_path, prereg)
    battery = contract["battery"]
    expected_frames = int(battery["required_frame_count"])
    expected_trials = int(battery["required_trial_count"])
    canary_frames = int(contract["canary"]["frame_count"])
    if mode == "canary" and int(limit_frames) != canary_frames:
        raise ValueError(f"canary requires --limit-frames={canary_frames}")
    if mode == "full" and int(limit_frames) != 0:
        raise ValueError("full mode requires --limit-frames=0")
    batch_profile: dict[str, Any] | None = None
    batch_profile_sha256: str | None = None
    if mode == "full":
        if batch_profile_path is None:
            raise ValueError("full mode requires --batch-profile from a passing canary")
        batch_profile = _load_batch_profile(
            batch_profile_path,
            model_plan_path=model_plan_path,
            contract_path=contract_path,
            model_plan=model_plan,
        )
        batch_profile_sha256 = sha256_file(batch_profile_path.expanduser().resolve())
    elif batch_profile_path is not None:
        raise ValueError("canary mode must not consume a prior batch profile")

    manifest_path = (data_dir / str(battery["manifest_relative_path"])).resolve()
    hash_manifest_path = (
        data_dir / str(battery["hash_manifest_relative_path"])
    ).resolve()
    hidden_path = (data_dir / str(battery["hidden_evaluator_relative_path"])).resolve()
    if not is_within(manifest_path, data_dir) or not manifest_path.is_file():
        raise FileNotFoundError("battery manifest is missing or escaped")
    if not is_within(hash_manifest_path, data_dir) or not hash_manifest_path.is_file():
        raise FileNotFoundError("battery hash manifest is missing or escaped")
    if not is_within(hidden_path, data_dir) or not hidden_path.is_file():
        raise FileNotFoundError("hidden evaluator index is missing or escaped")
    hash_manifest = _verify_hash_manifest(data_dir, hash_manifest_path)
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("battery manifest must be an object")
    if manifest.get("kind") != "phase2_reanchored_egohos_job_a_2afc_manifest":
        raise ValueError("invalid battery manifest kind")
    if manifest.get("status") != "production_construction_complete":
        raise ValueError("battery is not a completed production construction")
    if manifest.get("production_eligible") is not True:
        raise ValueError("battery is not production eligible")
    if manifest.get("selection_only") is not False:
        raise ValueError("selection-only battery cannot be scored")
    if manifest.get("limited_nonproduction_mode") is not False:
        raise ValueError("limited nonproduction battery cannot be scored")
    if manifest.get("governing_prespec_sha256") != preregistration_sha256:
        raise ValueError("battery governing preregistration drift")
    frame_count = _manifest_count(
        manifest, ("frame_level_unit_count", "frame_count", "selected_frame_count")
    )
    trial_count = _manifest_count(manifest, ("trial_count", "question_count", "request_count"))
    participant_count = _manifest_count(
        manifest, ("participant_cluster_count", "participant_count", "unique_participant_count")
    )
    if frame_count != expected_frames or trial_count != expected_trials:
        raise ValueError("battery manifest count drift")
    if participant_count is None or participant_count < int(battery["required_participant_count"]):
        raise ValueError("battery manifest does not prove the participant minimum")
    if manifest.get("counterbalanced_trials_per_frame") != int(
        battery["required_trials_per_frame"]
    ):
        raise ValueError("battery counterbalance multiplicity drift")
    if manifest.get("request_count_all_arms") != expected_trials * len(ARMS):
        raise ValueError("battery all-arm request count drift")
    if manifest.get("answer_counts_hidden") != {
        "A": expected_frames,
        "B": expected_frames,
    }:
        raise ValueError("battery does not prove exact 800/800 A/B counterbalancing")
    class_allocation = manifest.get("class_allocation")
    if not isinstance(class_allocation, dict):
        raise ValueError("battery manifest lacks ontology class allocation")
    for role, expected_classes in (
        ("first_order", {"3", "4", "5"}),
        ("second_order", {"6", "7", "8"}),
    ):
        counts = class_allocation.get(role)
        if (
            not isinstance(counts, dict)
            or set(counts) != expected_classes
            or any(type(value) is not int or value < 0 for value in counts.values())
            or sum(counts.values()) != expected_frames
            or (expected_frames == 800 and any(value <= 0 for value in counts.values()))
        ):
            raise ValueError(f"battery {role} class-support count drift")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("battery manifest lacks output paths")
    if outputs.get("hidden_evaluator_index") != battery["hidden_evaluator_relative_path"]:
        raise ValueError("battery hidden-evaluator path drift")
    if outputs.get("hash_manifest") != battery["hash_manifest_relative_path"]:
        raise ValueError("battery hash-manifest path drift")
    if outputs.get("public_request_packs") != battery["pack_relative_paths"]:
        raise ValueError("battery public-pack paths drift")
    selection_audit_relative = str(outputs.get("selection_audit") or "")
    selection_audit_path = (data_dir / selection_audit_relative).resolve()
    if (
        not selection_audit_relative
        or not is_within(selection_audit_path, data_dir)
        or not selection_audit_path.is_file()
        or manifest.get("selection_audit_sha256") != sha256_file(selection_audit_path)
    ):
        raise ValueError("battery component-provenance selection-audit binding drift")
    hash_entries = {
        str(row["path"]): row
        for row in hash_manifest["entries"]
        if isinstance(row, dict) and row.get("path")
    }
    if selection_audit_relative not in hash_entries:
        raise ValueError("component-provenance selection audit is absent from hash manifest")
    analysis_unit = manifest.get("analysis_unit_contract")
    if not isinstance(analysis_unit, dict) or analysis_unit.get(
        "primary_evidence_estimand"
    ) != (
        "intersection_union_test_requiring_positive_participant_equal_real_minus_"
        "blank_full_and_real_minus_wrong_time_components"
    ):
        raise ValueError("battery primary analysis-unit contract drift")
    ontology = manifest.get("ontology_contract")
    if (
        not isinstance(ontology, dict)
        or ontology.get("primary_task")
        != "annotation_native_first_vs_second_order_interacting_object_2afc"
        or ontology.get("single_target_binary_direct_contact_primary") is not False
    ):
        raise ValueError("battery ontology contract drift")

    image_cache: dict[Path, dict[str, Any]] = {}
    identities_by_arm: dict[str, list[dict[str, Any]]] = {}
    pack_hashes: dict[str, str] = {}
    for arm in ARMS:
        pack_path = (data_dir / str(battery["pack_relative_paths"][arm])).resolve()
        if not is_within(pack_path, data_dir) or not pack_path.is_file():
            raise FileNotFoundError(f"missing public scoring pack for {arm}")
        payload = read_json(pack_path)
        if not isinstance(payload, dict) or payload.get(
            "kind"
        ) != "phase2_reanchored_egohos_job_a_2afc_request_pack":
            raise ValueError(f"invalid public pack kind for {arm}")
        if payload.get("arm") != arm:
            raise ValueError(f"pack arm drift for {arm}")
        if payload.get("request_count") != expected_trials:
            raise ValueError(f"pack request count metadata drift for {arm}")
        if payload.get("frame_level_unit_count") != expected_frames:
            raise ValueError(f"pack frame count metadata drift for {arm}")
        if payload.get("counterbalanced_trials_per_frame") != int(
            battery["required_trials_per_frame"]
        ):
            raise ValueError(f"pack counterbalance metadata drift for {arm}")
        requests = _pack_requests(payload)
        if len(requests) != expected_trials:
            raise ValueError(f"pack count drift for {arm}")
        prompt_text = str(prereg["generation"]["prompt_text"])
        identities: list[dict[str, Any]] = []
        request_map: dict[str, dict[str, Any]] = {}
        for request in requests:
            if request.get("arm") != arm:
                raise ValueError(f"public request arm drift in {arm}")
            if str(request.get("user_prompt") or "") != prompt_text:
                raise ValueError(f"governing 2AFC user prompt drift in {arm}")
            identity = request_identity(request, data_dir=data_dir, image_cache=image_cache)
            if identity["item_id"] in request_map:
                raise ValueError(f"duplicate item ID in {arm}")
            request_map[identity["item_id"]] = request
            identities.append(identity)
        identities_by_arm[arm] = identities
        pack_hashes[arm] = sha256_file(pack_path)

    canonical_ids = [row["item_id"] for row in identities_by_arm["real"]]
    if len(set(canonical_ids)) != expected_trials:
        raise ValueError("real pack item IDs are not unique")
    for arm in ARMS[1:]:
        if [row["item_id"] for row in identities_by_arm[arm]] != canonical_ids:
            raise ValueError("public pack item order/alignment drift")

    identity_map: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        identity_map[arm] = {row["item_id"]: row for row in identities_by_arm[arm]}
    for item_id in canonical_ids:
        real = identity_map["real"][item_id]
        for arm in ARMS:
            row = identity_map[arm][item_id]
            if row["system_prompt_sha256"] != real["system_prompt_sha256"] or row[
                "user_prompt_sha256"
            ] != real["user_prompt_sha256"]:
                raise ValueError("prompt drift across arms")
            if [entry["sha256"] for entry in row["ordered_inputs"][1:]] != [
                entry["sha256"] for entry in real["ordered_inputs"][1:]
            ]:
                raise ValueError("crop pixels changed across evidence arms")
    if len(canonical_ids) != expected_frames * int(battery["required_trials_per_frame"]):
        raise ValueError("public rows cannot form the required adjacent counterbalance pairs")
    pair_item_ids = [
        canonical_ids[index : index + 2] for index in range(0, len(canonical_ids), 2)
    ]
    if len(pair_item_ids) != expected_frames or any(len(pair) != 2 for pair in pair_item_ids):
        raise ValueError("counterbalance pairs are not exactly 800 adjacent row pairs")
    for arm in ARMS:
        for pair_index, item_ids in enumerate(pair_item_ids):
            first = identity_map[arm][item_ids[0]]["ordered_inputs"]
            second = identity_map[arm][item_ids[1]]["ordered_inputs"]
            if first[0]["sha256"] != second[0]["sha256"]:
                raise ValueError(
                    f"counterbalanced full frame changed in {arm}:pair-{pair_index}"
                )
            if not (
                first[1]["sha256"] == second[2]["sha256"]
                and first[2]["sha256"] == second[1]["sha256"]
                and first[1]["sha256"] != first[2]["sha256"]
            ):
                raise ValueError(f"crop order is not an exact swap in {arm}:pair-{pair_index}")

    full_sizes = {
        (row["ordered_inputs"][0]["width"], row["ordered_inputs"][0]["height"])
        for arm in ARMS
        for row in identities_by_arm[arm]
    }
    crop_sizes = {
        (entry["width"], entry["height"])
        for arm in ARMS
        for row in identities_by_arm[arm]
        for entry in row["ordered_inputs"][1:]
    }
    rendering = manifest.get("rendering_contract")
    if not isinstance(rendering, dict):
        raise ValueError("battery manifest lacks fixed rendering geometry")
    expected_full_size = (
        int(rendering.get("full_frame", {}).get("width", 0)),
        int(rendering.get("full_frame", {}).get("height", 0)),
    )
    expected_crop_size = (
        int(rendering.get("reference_crop", {}).get("width", 0)),
        int(rendering.get("reference_crop", {}).get("height", 0)),
    )
    if full_sizes != {expected_full_size} or crop_sizes != {expected_crop_size}:
        raise ValueError("fixed full-frame or crop-canvas geometry gate failed")

    selected_pair_count = expected_frames if mode == "full" else int(limit_frames)
    selected_item_ids = {
        item_id for pair in pair_item_ids[:selected_pair_count] for item_id in pair
    }
    selection_rows = []
    for source_index, item_id in enumerate(canonical_ids):
        if item_id not in selected_item_ids:
            continue
        selection_rows.append(
            {
                "selection_index": len(selection_rows),
                "counterbalance_pair_index": source_index // 2,
                "item_id": item_id,
                "request_id_by_arm": {
                    arm: identity_map[arm][item_id]["request_id"] for arm in ARMS
                },
                "public_identity_by_arm": {
                    arm: identity_map[arm][item_id] for arm in ARMS
                },
            }
        )

    code_sha = _code_snapshot_sha256(contract_path, model_plan_path)
    selection_manifest = {
        "kind": "phase2_reanchored_job_a_evaluator_unjoined_selection_manifest",
        "status": "pass",
        "mode": mode,
        "frame_count": selected_pair_count,
        "trial_count": len(selection_rows),
        "arms": list(ARMS),
        "selection": selection_rows,
        "data_manifest_sha256": sha256_file(manifest_path),
        "hash_manifest_sha256": sha256_file(hash_manifest_path),
        "hidden_evaluator_sha256": sha256_file(hidden_path),
        "public_pack_sha256": pack_hashes,
        "model_plan_sha256": sha256_file(model_plan_path),
        "contract_sha256": sha256_file(contract_path),
        "preregistration_sha256": preregistration_sha256,
        "code_snapshot_sha256": code_sha,
        "full_frame_size": list(expected_full_size),
        "crop_canvas_size": list(expected_crop_size),
        "model_plan": model_plan,
        "batch_profile": batch_profile,
        "batch_profile_sha256": batch_profile_sha256,
        "claim_boundary": "Evaluator-unjoined routing and byte-identity manifest. The coordinator verifies the sealed evaluator byte hash but does not parse labels; the pair index is derived solely from public row order.",
    }
    if contains_forbidden_public_key(selection_manifest):
        raise RuntimeError("internal error: selection manifest crossed the label firewall")
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_path = run_dir / "selection_manifest.json"
    write_json(selection_path, selection_manifest)
    report = {
        "kind": "phase2_reanchored_job_a_preflight_report",
        "status": "pass",
        "mode": mode,
        "data_dir": str(data_dir),
        "frame_count": selected_pair_count,
        "trial_count": len(selection_rows),
        "participant_count_from_frozen_manifest": participant_count,
        "unique_image_count_hashed": len(image_cache),
        "full_frame_size": list(expected_full_size),
        "crop_canvas_size": list(expected_crop_size),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "preregistration_sha256": preregistration_sha256,
        "hidden_evaluator_integrity_hash_verified": True,
        "hidden_labels_parsed_or_joined": False,
        "batch_profile_sha256": batch_profile_sha256,
        "created_at": utc_now(),
    }
    write_json(run_dir / "preflight_report.json", report)
    return report


def _load_selection(
    *,
    data_dir: Path,
    selection_path: Path,
    model_plan_path: Path,
    contract_path: Path,
    verify_hidden_evaluator: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    contract, prereg = _load_contract(contract_path)
    preregistration_sha256 = _contract_preregistration_sha256(contract)
    model_plan = _load_model_plan(model_plan_path, prereg)
    selection = read_json(selection_path)
    if selection.get("kind") != "phase2_reanchored_job_a_evaluator_unjoined_selection_manifest":
        raise ValueError("invalid selection manifest kind")
    expected = {
        "status": "pass",
        "arms": list(ARMS),
        "model_plan_sha256": sha256_file(model_plan_path),
        "contract_sha256": sha256_file(contract_path),
        "preregistration_sha256": preregistration_sha256,
        "code_snapshot_sha256": _code_snapshot_sha256(contract_path, model_plan_path),
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"selection manifest binding drift: {key}")
    battery = contract["battery"]
    manifest_path = data_dir / str(battery["manifest_relative_path"])
    hash_manifest_path = data_dir / str(battery["hash_manifest_relative_path"])
    if selection.get("data_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("battery manifest changed after preflight")
    if selection.get("hash_manifest_sha256") != sha256_file(hash_manifest_path):
        raise ValueError("battery hash manifest changed after preflight")
    if verify_hidden_evaluator:
        hidden_path = data_dir / str(battery["hidden_evaluator_relative_path"])
        if selection.get("hidden_evaluator_sha256") != sha256_file(hidden_path):
            raise ValueError("hidden evaluator changed after preflight")
    for arm in ARMS:
        path = data_dir / str(battery["pack_relative_paths"][arm])
        if selection.get("public_pack_sha256", {}).get(arm) != sha256_file(path):
            raise ValueError(f"public pack changed after preflight: {arm}")
    if contains_forbidden_public_key(selection):
        raise ValueError("selection manifest crossed the label firewall")
    return selection, prereg, model_plan


def _load_public_pack_maps(
    *, data_dir: Path, contract: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        path = data_dir / str(contract["battery"]["pack_relative_paths"][arm])
        rows = _pack_requests(read_json(path))
        arm_map: dict[str, dict[str, Any]] = {}
        for row in rows:
            item_id = str(row.get("item_id") or row.get("trial_id") or "")
            if not item_id or item_id in arm_map:
                raise ValueError(f"missing or duplicate item ID in {arm}")
            arm_map[item_id] = row
        maps[arm] = arm_map
    return maps


def _strict_greedy(
    *, model: Any, runtime: Any, processor: Any, base_inputs: Any, max_new_tokens: int
) -> dict[str, Any]:
    import torch

    job2c = __import__(
        "research.lifecycle_memory.phase2_job2c_behavioral_companion",
        fromlist=["_decode_generated_tokens"],
    )
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **base_inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )
    raw_text = job2c._decode_generated_tokens(runtime, processor, base_inputs, output_ids)
    parsed = strict_parse_choice(raw_text)
    return {
        "status": "pass",
        "raw_text": raw_text,
        "strict_choice": parsed,
        "latency_s": time.perf_counter() - started,
    }


def _score_request(
    *,
    request: dict[str, Any],
    data_dir: Path,
    model_key: str,
    model: Any,
    runtime: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from research.lifecycle_memory import phase2_job2c_behavioral_companion as job2c
    from research.lifecycle_memory import phase2_job3_heldout_confirmation as job3

    image_paths = [
        _resolve_public_path(data_dir, str(row["path"])) for row in request["inputs"]
    ]
    base_inputs, serialization = job3._prepare_universal_base(
        runtime=runtime,
        processor=processor,
        system_prompt=str(request["system_prompt"]),
        user_prompt=str(request["user_prompt"]),
        image_paths=image_paths,
        device=device,
    )
    base_ids = job2c._input_ids_list(base_inputs)
    if not base_ids:
        raise RuntimeError("base serialization produced no input IDs")
    prefix_text, prefix_ids, answer_ids = _answer_token_spec(tokenizer, prereg)
    scoring_base_inputs, prefix_trace = job2c._append_exact_candidate_continuation(
        base_inputs, prefix_ids
    )
    if job2c._visual_identity_audit(base_inputs, scoring_base_inputs).get("status") != "pass":
        raise RuntimeError("frozen answer prefix changed visual tensors")
    scoring_base_ids = job2c._input_ids_list(scoring_base_inputs)
    candidate_rows: dict[str, dict[str, Any]] = {}
    for choice in CHOICES:
        candidate_text = str(prereg["candidate_completions"][choice])
        candidate_ids = answer_ids[choice]
        full_inputs, append_trace = job2c._append_exact_candidate_continuation(
            scoring_base_inputs, candidate_ids
        )
        visual_audit = job2c._visual_identity_audit(scoring_base_inputs, full_inputs)
        if visual_audit.get("status") != "pass":
            raise RuntimeError("candidate continuation changed visual tensors")
        with torch.inference_mode():
            output = model(**full_inputs)
        scored = score_candidate_suffix_from_logits(
            base_input_ids=scoring_base_ids,
            full_input_ids=job2c._input_ids_list(full_inputs),
            logits=job2c._extract_logits(output),
        )
        if scored.get("status") != "pass":
            raise RuntimeError(f"candidate suffix scoring failed for {choice}: {scored}")
        if not math.isfinite(float(scored["sum_logprob"])):
            raise RuntimeError("candidate score is nonfinite")
        candidate_rows[choice] = {
            "status": "pass",
            "candidate_text_sha256": hashlib.sha256(candidate_text.encode("utf-8")).hexdigest(),
            "candidate_construction": "frozen_shared_prefix_then_exact_answer_token_append",
            "shared_prefix_sha256": hashlib.sha256(prefix_text.encode("utf-8")).hexdigest(),
            "shared_prefix_token_ids": list(prefix_ids),
            "combined_prefix_and_answer_token_ids": list(prefix_ids + candidate_ids),
            "token_boundary_status": "pass",
            "shared_prefix_append_trace": prefix_trace,
            "token_count": int(scored["token_count"]),
            "scored_token_ids": list(scored["scored_token_ids"]),
            "token_logprobs": list(scored["token_logprobs"]),
            "sum_logprob": float(scored["sum_logprob"]),
            "mean_logprob": float(scored["mean_logprob"]),
            "append_trace": append_trace,
            "visual_identity_status": visual_audit["status"],
        }
        del output, full_inputs
    greedy = _strict_greedy(
        model=model,
        runtime=runtime,
        processor=processor,
        base_inputs=base_inputs,
        max_new_tokens=int(prereg["generation"]["max_new_tokens"]),
    )
    return {
        "model_key": model_key,
        "serialization": serialization,
        "teacher_forced": {"status": "pass", "candidate_rows": candidate_rows},
        "greedy": greedy,
    }


def _answer_token_spec(
    tokenizer: Any, prereg: Mapping[str, Any]
) -> tuple[str, list[int], dict[str, list[int]]]:
    from research.lifecycle_memory import phase2_job2c_behavioral_companion as job2c

    prefix_text = str(prereg.get("candidate_shared_prefix") or "")
    if not prefix_text:
        raise RuntimeError("governing preregistration lacks a frozen answer prefix")
    prefix_ids = job2c._encode_candidate_text(tokenizer, prefix_text)
    if not prefix_ids:
        raise RuntimeError("frozen answer prefix encoded to no tokens")
    answer_ids: dict[str, list[int]] = {}
    for choice in CHOICES:
        answer_text = str(prereg["candidate_completions"][choice])
        ids = job2c._encode_candidate_text(tokenizer, answer_text)
        if len(ids) != 1:
            raise RuntimeError(
                f"answer {choice} must encode as exactly one token, observed {len(ids)}"
            )
        combined = job2c._encode_candidate_text(tokenizer, prefix_text + answer_text)
        if combined != prefix_ids + ids:
            raise RuntimeError(
                f"answer {choice} crosses the frozen prefix token boundary"
            )
        answer_ids[choice] = ids
    if answer_ids["A"] == answer_ids["B"]:
        raise RuntimeError("A and B encoded to the same answer token")
    return prefix_text, prefix_ids, answer_ids


def _universal_messages(
    *, system_prompt: str, user_prompt: str, image_paths: Sequence[Path]
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                *[{"type": "image", "image": str(path)} for path in image_paths],
                {"type": "text", "text": user_prompt},
            ],
        }
    ]


def _prepare_batch_base(
    *,
    requests: Sequence[dict[str, Any]],
    data_dir: Path,
    runtime: Any,
    processor: Any,
    device: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    from research.lifecycle_memory import run_phase2_mech_job1_instrument_gates as gates

    conversations: list[list[dict[str, Any]]] = []
    traces: list[dict[str, Any]] = []
    for request in requests:
        image_paths = [
            _resolve_public_path(data_dir, str(row["path"])) for row in request["inputs"]
        ]
        conversations.append(
            _universal_messages(
                system_prompt=str(request["system_prompt"]),
                user_prompt=str(request["user_prompt"]),
                image_paths=image_paths,
            )
        )
        traces.append(
            {
                "status": "pass",
                "serialization_id": "batched_single_user_system_images_user_v1",
                "image_count": len(image_paths),
                "system_prompt_sha256": hashlib.sha256(
                    str(request["system_prompt"]).encode("utf-8")
                ).hexdigest(),
                "user_prompt_sha256": hashlib.sha256(
                    str(request["user_prompt"]).encode("utf-8")
                ).hexdigest(),
            }
        )
    inputs = runtime._apply_chat_template_direct(
        processor,
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    input_ids = gates._input_get(inputs, "input_ids")
    attention_mask = gates._input_get(inputs, "attention_mask")
    if input_ids is None or not hasattr(input_ids, "dim") or input_ids.dim() != 2:
        raise RuntimeError("batched processor produced invalid input_ids")
    if int(input_ids.shape[0]) != len(requests):
        raise RuntimeError("batched processor batch dimension drift")
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise RuntimeError("batched attention-mask shape drift")
        if not bool(attention_mask.bool().all().item()):
            raise RuntimeError(
                "batched prompt lengths differ; exact suffix placement would cross padding"
            )
    return runtime._to_device(inputs, device), traces


def _append_batch_candidate(
    base_inputs: Any, candidate_token_ids: Sequence[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from research.lifecycle_memory import run_phase2_mech_job1_instrument_gates as gates

    items = {str(key): value for key, value in gates._input_items(base_inputs)}
    input_ids = items.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2:
        raise RuntimeError("batched exact continuation requires rank-2 input_ids")
    batch_size, base_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    suffix_ids = [int(value) for value in candidate_token_ids]
    if not suffix_ids:
        raise RuntimeError("empty candidate suffix")
    suffix = torch.tensor(
        [suffix_ids] * batch_size,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    out = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in items.items()
    }
    out["input_ids"] = torch.cat([input_ids, suffix], dim=-1)
    extended = ["input_ids"]
    for key, fill_value in (
        ("attention_mask", 1),
        ("mm_token_type_ids", 0),
        ("token_type_ids", 0),
    ):
        value = items.get(key)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(input_ids.shape):
            raise RuntimeError(f"cannot safely extend batched sequence field {key}")
        extension = torch.full(
            (batch_size, len(suffix_ids)),
            fill_value,
            dtype=value.dtype,
            device=value.device,
        )
        out[key] = torch.cat([value, extension], dim=-1)
        extended.append(key)
    for key, value in items.items():
        if key in extended:
            continue
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and int(value.shape[-1]) == base_len:
            raise RuntimeError(f"unsupported batched sequence-aligned field {key}")
    return out, {
        "candidate_construction": "batched_base_generation_prompt_exact_token_append",
        "batch_size": batch_size,
        "base_input_length": base_len,
        "candidate_token_count": len(suffix_ids),
        "sequence_fields_extended": extended,
    }


def _score_batch_suffix(
    *, base_input_ids: Any, full_input_ids: Any, logits: Any
) -> list[dict[str, Any]]:
    import torch

    if not all(isinstance(value, torch.Tensor) for value in (base_input_ids, full_input_ids, logits)):
        raise RuntimeError("batched scoring requires tensors")
    if base_input_ids.dim() != 2 or full_input_ids.dim() != 2 or logits.dim() != 3:
        raise RuntimeError("batched scoring tensor rank drift")
    if int(base_input_ids.shape[0]) != int(full_input_ids.shape[0]) or int(logits.shape[0]) != int(
        full_input_ids.shape[0]
    ):
        raise RuntimeError("batched scoring batch dimension drift")
    rows: list[dict[str, Any]] = []
    for index in range(int(base_input_ids.shape[0])):
        rows.append(
            score_candidate_suffix_from_logits(
                base_input_ids=[int(value) for value in base_input_ids[index].detach().cpu().tolist()],
                full_input_ids=[int(value) for value in full_input_ids[index].detach().cpu().tolist()],
                logits=logits[index : index + 1],
            )
        )
    return rows


def _strict_greedy_batch(
    *, model: Any, processor: Any, base_inputs: Any, max_new_tokens: int
) -> list[dict[str, Any]]:
    import torch
    from research.lifecycle_memory import run_phase2_mech_job1_instrument_gates as gates

    input_ids = gates._input_get(base_inputs, "input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2:
        raise RuntimeError("batched generation requires rank-2 input_ids")
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **base_inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )
    if not isinstance(output_ids, torch.Tensor) or output_ids.dim() != 2:
        raise RuntimeError("batched generation output rank drift")
    generated = output_ids[:, int(input_ids.shape[1]) :]
    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(generated, skip_special_tokens=True)
    elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "batch_decode"):
        decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
    else:
        raise RuntimeError("processor cannot batch-decode generated tokens")
    if len(decoded) != int(input_ids.shape[0]):
        raise RuntimeError("batched decode count drift")
    elapsed = time.perf_counter() - started
    return [
        {
            "status": "pass",
            "raw_text": str(text).strip(),
            "strict_choice": strict_parse_choice(str(text).strip()),
            "latency_s": elapsed,
            "batch_shared_latency": True,
        }
        for text in decoded
    ]


def _score_requests_batch(
    *,
    requests: Sequence[dict[str, Any]],
    data_dir: Path,
    model_key: str,
    model: Any,
    runtime: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    prereg: dict[str, Any],
) -> list[dict[str, Any]]:
    import torch
    from research.lifecycle_memory import phase2_job2c_behavioral_companion as job2c
    from research.lifecycle_memory import run_phase2_mech_job1_instrument_gates as gates

    if len(requests) < 2:
        return [
            _score_request(
                request=requests[0],
                data_dir=data_dir,
                model_key=model_key,
                model=model,
                runtime=runtime,
                processor=processor,
                tokenizer=tokenizer,
                device=device,
                prereg=prereg,
            )
        ]
    base_inputs, serialization_rows = _prepare_batch_base(
        requests=requests,
        data_dir=data_dir,
        runtime=runtime,
        processor=processor,
        device=device,
    )
    prefix_text, prefix_ids, answer_ids = _answer_token_spec(tokenizer, prereg)
    scoring_base_inputs, prefix_trace = _append_batch_candidate(base_inputs, prefix_ids)
    if job2c._visual_identity_audit(base_inputs, scoring_base_inputs).get("status") != "pass":
        raise RuntimeError("batched frozen answer prefix changed visual tensors")
    scoring_base_ids = gates._input_get(scoring_base_inputs, "input_ids")
    candidate_by_choice: dict[str, list[dict[str, Any]]] = {}
    for choice in CHOICES:
        candidate_text = str(prereg["candidate_completions"][choice])
        candidate_ids = answer_ids[choice]
        full_inputs, append_trace = _append_batch_candidate(
            scoring_base_inputs, candidate_ids
        )
        visual_audit = job2c._visual_identity_audit(scoring_base_inputs, full_inputs)
        if visual_audit.get("status") != "pass":
            raise RuntimeError("batched candidate continuation changed visual tensors")
        with torch.inference_mode():
            output = model(**full_inputs)
        logits = job2c._extract_logits(output)
        full_ids = gates._input_get(full_inputs, "input_ids")
        scored_rows = _score_batch_suffix(
            base_input_ids=scoring_base_ids,
            full_input_ids=full_ids,
            logits=logits,
        )
        normalized: list[dict[str, Any]] = []
        for scored in scored_rows:
            if scored.get("status") != "pass" or not math.isfinite(float(scored["sum_logprob"])):
                raise RuntimeError(f"batched candidate scoring failed for {choice}: {scored}")
            normalized.append(
                {
                    "status": "pass",
                    "candidate_text_sha256": hashlib.sha256(candidate_text.encode("utf-8")).hexdigest(),
                    "candidate_construction": "frozen_shared_prefix_then_exact_answer_token_append",
                    "shared_prefix_sha256": hashlib.sha256(prefix_text.encode("utf-8")).hexdigest(),
                    "shared_prefix_token_ids": list(prefix_ids),
                    "combined_prefix_and_answer_token_ids": list(prefix_ids + candidate_ids),
                    "token_boundary_status": "pass",
                    "shared_prefix_append_trace": prefix_trace,
                    "token_count": int(scored["token_count"]),
                    "scored_token_ids": list(scored["scored_token_ids"]),
                    "token_logprobs": list(scored["token_logprobs"]),
                    "sum_logprob": float(scored["sum_logprob"]),
                    "mean_logprob": float(scored["mean_logprob"]),
                    "append_trace": append_trace,
                    "visual_identity_status": visual_audit["status"],
                }
            )
        candidate_by_choice[choice] = normalized
        del output, logits, full_inputs
    greedy_rows = _strict_greedy_batch(
        model=model,
        processor=processor,
        base_inputs=base_inputs,
        max_new_tokens=int(prereg["generation"]["max_new_tokens"]),
    )
    return [
        {
            "model_key": model_key,
            "serialization": serialization_rows[index],
            "teacher_forced": {
                "status": "pass",
                "candidate_rows": {
                    choice: candidate_by_choice[choice][index] for choice in CHOICES
                },
            },
            "greedy": greedy_rows[index],
        }
        for index in range(len(requests))
    ]


def _prefetch_requests(requests: Sequence[dict[str, Any]], data_dir: Path) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    paths = [
        _resolve_public_path(data_dir, str(raw["path"]))
        for request in requests
        for raw in request["inputs"]
    ]
    unique_paths = list(dict.fromkeys(paths))
    started = time.perf_counter()

    def warm(path: Path) -> int:
        with path.open("rb") as handle:
            return len(handle.read())

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(unique_paths)))) as executor:
        byte_counts = list(executor.map(warm, unique_paths))
    return {
        "path_count": len(unique_paths),
        "bytes": sum(byte_counts),
        "elapsed_s": time.perf_counter() - started,
    }


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "gpu" in message)


def row_hash(row: dict[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in row.items() if key != "row_sha256"})


def validate_completed_row(
    row: dict[str, Any], *, expected_common: dict[str, Any]
) -> list[str]:
    issues = [key for key, value in expected_common.items() if row.get(key) != value]
    if row.get("kind") != "phase2_reanchored_job_a_worker_row" or row.get("status") != "pass":
        issues.append("row_contract")
    if (
        row.get("arm") not in ARMS
        or not row.get("item_id")
        or type(row.get("counterbalance_pair_index")) is not int
        or int(row["counterbalance_pair_index"]) < 0
    ):
        issues.append("row_identity")
    if row.get("row_sha256") != row_hash(row):
        issues.append("row_hash")
    candidates = row.get("teacher_forced", {}).get("candidate_rows", {})
    if set(candidates) != set(CHOICES):
        issues.append("candidate_choices")
    else:
        token_counts = []
        answer_token_ids: dict[str, list[int]] = {}
        prefix_token_ids: dict[str, list[int]] = {}
        answer_token_contract = bool(
            str(expected_common.get("candidate_shared_prefix") or "")
        )
        for choice in CHOICES:
            candidate = candidates[choice]
            if candidate.get("status") != "pass" or candidate.get("visual_identity_status") != "pass":
                issues.append(f"candidate_{choice}_status")
            if answer_token_contract and (
                candidate.get("candidate_construction")
                != "frozen_shared_prefix_then_exact_answer_token_append"
            ):
                issues.append(f"candidate_{choice}_construction")
            expected_text = str(expected_common.get(f"candidate_{choice}_text") or "")
            if expected_text:
                expected_sha256 = hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
                if candidate.get("candidate_text_sha256") != expected_sha256:
                    issues.append(f"candidate_{choice}_text")
            token_count = candidate.get("token_count")
            if answer_token_contract:
                if type(token_count) is not int or token_count != 1:
                    issues.append(f"candidate_{choice}_token_count")
                else:
                    token_counts.append(token_count)
                scored_ids = candidate.get("scored_token_ids")
                prefix_ids = candidate.get("shared_prefix_token_ids")
                combined_ids = candidate.get("combined_prefix_and_answer_token_ids")
                safe_scored_ids = scored_ids if isinstance(scored_ids, list) else []
                safe_prefix_ids = prefix_ids if isinstance(prefix_ids, list) else []
                if (
                    not isinstance(scored_ids, list)
                    or len(scored_ids) != 1
                    or not all(type(value) is int for value in scored_ids)
                ):
                    issues.append(f"candidate_{choice}_answer_token_ids")
                else:
                    answer_token_ids[choice] = scored_ids
                if (
                    not isinstance(prefix_ids, list)
                    or not prefix_ids
                    or not all(type(value) is int for value in prefix_ids)
                ):
                    issues.append(f"candidate_{choice}_prefix_token_ids")
                else:
                    prefix_token_ids[choice] = prefix_ids
                if (
                    candidate.get("token_boundary_status") != "pass"
                    or not isinstance(combined_ids, list)
                    or combined_ids != safe_prefix_ids + safe_scored_ids
                ):
                    issues.append(f"candidate_{choice}_token_boundary")
            expected_prefix = str(expected_common.get("candidate_shared_prefix") or "")
            if expected_prefix:
                expected_prefix_sha256 = hashlib.sha256(
                    expected_prefix.encode("utf-8")
                ).hexdigest()
                if candidate.get("shared_prefix_sha256") != expected_prefix_sha256:
                    issues.append(f"candidate_{choice}_shared_prefix")
            value = candidate.get("sum_logprob")
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                issues.append(f"candidate_{choice}_score")
        if (
            answer_token_contract
            and len(token_counts) == len(CHOICES)
            and len(set(token_counts)) != 1
        ):
            issues.append("candidate_token_count_mismatch")
        if answer_token_contract and len(answer_token_ids) == len(CHOICES):
            if answer_token_ids["A"] == answer_token_ids["B"]:
                issues.append("candidate_answer_token_collision")
        if answer_token_contract and len(prefix_token_ids) == len(CHOICES):
            if prefix_token_ids["A"] != prefix_token_ids["B"]:
                issues.append("candidate_shared_prefix_token_mismatch")
    greedy = row.get("greedy")
    if not isinstance(greedy, dict) or greedy.get("status") != "pass":
        issues.append("greedy_execution")
    return sorted(set(issues))


def _load_model_bundle(model_row: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    from research.lifecycle_memory.phase2_job3_heldout_confirmation import _load_model_bundle

    return _load_model_bundle(
        model_row=model_row,
        dtype="bf16",
        local_files_only=True,
    )


def run_worker(
    *,
    data_dir: Path,
    run_dir: Path,
    selection_path: Path,
    model_plan_path: Path,
    contract_path: Path,
    worker_index: int,
    resume: bool,
    model_key: str | None = None,
    assignment_id: str | None = None,
    selection_start: int | None = None,
    selection_stop: int | None = None,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    selection_path = selection_path.expanduser().resolve()
    model_plan_path = model_plan_path.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    selection, prereg, model_plan = _load_selection(
        data_dir=data_dir,
        selection_path=selection_path,
        model_plan_path=model_plan_path,
        contract_path=contract_path,
    )
    if int(worker_index) not in range(4):
        raise ValueError("worker index must be 0..3")
    if selection["mode"] == "canary":
        if any(value is not None for value in (model_key, assignment_id, selection_start, selection_stop)):
            raise ValueError("canary workers may not override their canonical assignment")
        model_row = model_plan[int(worker_index)]
        if model_row["worker_index"] != int(worker_index):
            raise ValueError("model plan worker topology drift")
        assignment_id = f"{model_row['model_key']}.canary"
        selection_start = 0
        selection_stop = len(selection["selection"])
        model_dir = run_dir / "models" / model_row["model_key"]
    else:
        if None in (model_key, assignment_id, selection_start, selection_stop):
            raise ValueError("full workers require an exact frozen schedule assignment")
        model_rows = {row["model_key"]: row for row in model_plan}
        if model_key not in model_rows:
            raise ValueError("full worker schedule references an unknown model")
        model_row = model_rows[str(model_key)]
        schedule_rows = {
            str(row["assignment_id"]): row
            for rank_row in selection["batch_profile"]["schedule"]["ranks"]
            for row in rank_row["assignments"]
        }
        scheduled = schedule_rows.get(str(assignment_id))
        expected_schedule = {
            "rank": int(worker_index),
            "model_key": str(model_key),
            "selection_start": int(selection_start),
            "selection_stop": int(selection_stop),
        }
        if scheduled is None or any(scheduled.get(key) != value for key, value in expected_schedule.items()):
            raise ValueError("full worker assignment does not match the frozen schedule")
        model_dir = run_dir / "models" / model_row["model_key"] / "shards" / str(assignment_id)
    assert assignment_id is not None and selection_start is not None and selection_stop is not None
    if not (0 <= int(selection_start) < int(selection_stop) <= len(selection["selection"])):
        raise ValueError("worker selection interval is invalid")
    contract, _ = _load_contract(contract_path)
    pack_maps = _load_public_pack_maps(data_dir=data_dir, contract=contract)

    rows_path = model_dir / "rows.jsonl"
    summary_path = model_dir / "summary.json"
    progress_path = model_dir / "progress.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    selection_sha = sha256_file(selection_path)
    expected_common = {
        "model_key": model_row["model_key"],
        "model_id": model_row["model_id"],
        "revision": model_row["revision"],
        "dtype": "bf16",
        "worker_index": int(worker_index),
        "assignment_id": str(assignment_id),
        "selection_start": int(selection_start),
        "selection_stop": int(selection_stop),
        "selection_manifest_sha256": selection_sha,
        "preregistration_sha256": selection["preregistration_sha256"],
        "code_snapshot_sha256": selection["code_snapshot_sha256"],
        "candidate_A_text": prereg["candidate_completions"]["A"],
        "candidate_B_text": prereg["candidate_completions"]["B"],
        "candidate_shared_prefix": prereg.get("candidate_shared_prefix"),
    }
    completed: set[tuple[str, str]] = set()
    if rows_path.exists() and rows_path.stat().st_size:
        if not resume:
            raise ValueError("worker rows already exist; pass --resume or use a fresh run directory")
        for row in iter_jsonl(rows_path):
            issues = validate_completed_row(row, expected_common=expected_common)
            if issues:
                raise ValueError(f"resume row failed validation: {issues}")
            key = (str(row["item_id"]), str(row["arm"]))
            if key in completed:
                raise ValueError(f"duplicate resume key: {key}")
            completed.add(key)

    assigned_selection = selection["selection"][int(selection_start) : int(selection_stop)]
    expected_rows = len(assigned_selection) * len(ARMS)
    summary = {
        "kind": "phase2_reanchored_job_a_worker_summary",
        "status": "loading",
        **expected_common,
        "mode": selection["mode"],
        "expected_row_count": expected_rows,
        "resume": bool(resume),
        "rows_present_at_start": len(completed),
        "started_at": utc_now(),
    }
    write_json(summary_path, summary)
    write_json(
        progress_path,
        {"status": "loading", "rows_complete": len(completed), "rows_expected": expected_rows, "updated_at": utc_now()},
    )
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    runtime, model, processor, tokenizer, device, provenance = _load_model_bundle(model_row)
    load_elapsed_s = time.perf_counter() - load_start
    if selection["mode"] == "canary":
        effective_batch_size = int(model_row["canary_microbatch_size"])
        batch_strategy = "batched" if effective_batch_size > 1 else "sequential"
    else:
        profile_rows = {
            str(row["model_key"]): row for row in selection["batch_profile"]["models"]
        }
        profile = profile_rows[model_row["model_key"]]
        effective_batch_size = int(profile["effective_batch_size"])
        batch_strategy = str(profile["strategy"])
        if batch_strategy == "sequential" and effective_batch_size != 1:
            raise ValueError("sequential full-run batch profile must use size one")
    summary.update(
        {
            "status": "running",
            "load_elapsed_s": load_elapsed_s,
            "runtime_provenance": provenance,
            "requested_microbatch_size": int(model_row["canary_microbatch_size"]),
            "effective_batch_size": effective_batch_size,
            "batch_strategy": batch_strategy,
        }
    )
    write_json(summary_path, summary)

    image_cache: dict[Path, dict[str, Any]] = {}
    started = time.perf_counter()
    new_rows = 0
    tasks: list[tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]] = []
    for selection_row in assigned_selection:
        item_id = str(selection_row["item_id"])
        for arm in ARMS:
            key = (item_id, arm)
            if key in completed:
                continue
            request = pack_maps[arm].get(item_id)
            if request is None:
                raise ValueError(f"selected request missing from {arm}: {item_id}")
            observed_identity = request_identity(
                request, data_dir=data_dir, image_cache=image_cache
            )
            expected_identity = selection_row["public_identity_by_arm"][arm]
            if observed_identity != expected_identity:
                raise ValueError(f"public request identity changed after preflight: {arm}/{item_id}")
            tasks.append((selection_row, arm, request, observed_identity))

    batch_events: list[dict[str, Any]] = []
    prefetch_totals = {"path_count": 0, "bytes": 0, "elapsed_s": 0.0}
    task_index = 0
    gpu_scoring_window_started_at = utc_now_precise()
    gpu_scoring_started = time.perf_counter()
    while task_index < len(tasks):
        actual_batch_size = 1 if batch_strategy == "sequential" else effective_batch_size
        batch_tasks = tasks[task_index : task_index + actual_batch_size]
        batch_requests = [task[2] for task in batch_tasks]
        prefetch = _prefetch_requests(batch_requests, data_dir)
        for key in prefetch_totals:
            prefetch_totals[key] += prefetch[key]
        batch_started = time.perf_counter()
        try:
            if batch_strategy == "batched" and len(batch_requests) > 1:
                results = _score_requests_batch(
                    requests=batch_requests,
                    data_dir=data_dir,
                    model_key=model_row["model_key"],
                    model=model,
                    runtime=runtime,
                    processor=processor,
                    tokenizer=tokenizer,
                    device=device,
                    prereg=prereg,
                )
            else:
                results = [
                    _score_request(
                        request=request,
                        data_dir=data_dir,
                        model_key=model_row["model_key"],
                        model=model,
                        runtime=runtime,
                        processor=processor,
                        tokenizer=tokenizer,
                        device=device,
                        prereg=prereg,
                    )
                    for request in batch_requests
                ]
        except Exception as exc:
            if selection["mode"] != "canary" or effective_batch_size <= 1:
                raise
            old_size = effective_batch_size
            if _is_cuda_oom(exc):
                effective_batch_size = max(1, effective_batch_size // 2)
                batch_strategy = "batched" if effective_batch_size > 1 else "sequential"
                event_kind = "oom_halve"
            else:
                effective_batch_size = 1
                batch_strategy = "sequential"
                event_kind = "processor_or_semantic_fallback"
            batch_events.append(
                {
                    "kind": event_kind,
                    "from_batch_size": old_size,
                    "to_batch_size": effective_batch_size,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "task_index": task_index,
                    "recorded_at": utc_now(),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        batch_elapsed = time.perf_counter() - batch_started
        if len(results) != len(batch_tasks):
            raise RuntimeError("batch result count drift")
        for (selection_row, arm, request, observed_identity), result in zip(
            batch_tasks, results, strict=True
        ):
            item_id = str(selection_row["item_id"])
            row = {
                "kind": "phase2_reanchored_job_a_worker_row",
                "status": "pass",
                **expected_common,
                "mode": selection["mode"],
                "selection_index": int(selection_row["selection_index"]),
                "item_id": item_id,
                "counterbalance_pair_index": int(
                    selection_row["counterbalance_pair_index"]
                ),
                "arm": arm,
                "request_id": str(request["request_id"]),
                "serialization_sha256": observed_identity["serialization_sha256"],
                "ordered_input_sha256": [
                    entry["sha256"] for entry in observed_identity["ordered_inputs"]
                ],
                "teacher_forced": result["teacher_forced"],
                "greedy": result["greedy"],
                "batch_strategy": batch_strategy,
                "effective_batch_size": len(batch_tasks),
                "batch_shared_elapsed_s": batch_elapsed,
                "completed_at": utc_now(),
            }
            row["row_sha256"] = row_hash(row)
            append_jsonl_durable(rows_path, row)
            completed.add((item_id, arm))
            new_rows += 1
        task_index += len(batch_tasks)
        last_selection, last_arm, _, _ = batch_tasks[-1]
        write_json(
            progress_path,
            {
                "status": "running",
                "model_key": model_row["model_key"],
                "current_item_id": str(last_selection["item_id"]),
                "current_arm": last_arm,
                "rows_complete": len(completed),
                "rows_expected": expected_rows,
                "rows_new": new_rows,
                "effective_batch_size": effective_batch_size,
                "batch_strategy": batch_strategy,
                "batch_fallback_count": len(batch_events),
                "elapsed_s": time.perf_counter() - started,
                "updated_at": utc_now(),
            },
        )
    if len(completed) != expected_rows:
        raise RuntimeError("worker row union is incomplete")
    gpu_scoring_elapsed_s = time.perf_counter() - gpu_scoring_started
    gpu_scoring_window_finished_at = utc_now_precise()
    inference_elapsed_s = time.perf_counter() - started
    if torch.cuda.is_available():
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        total_gpu_memory = int(torch.cuda.get_device_properties(0).total_memory)
    else:
        peak_allocated = peak_reserved = total_gpu_memory = 0
    summary.update(
        {
            "status": "pass",
            "rows_complete": len(completed),
            "rows_new": new_rows,
            "rows_sha256": sha256_file(rows_path),
            "elapsed_s": inference_elapsed_s,
            "decisions_per_second": new_rows / inference_elapsed_s if inference_elapsed_s else None,
            "effective_batch_size": effective_batch_size,
            "batch_strategy": batch_strategy,
            "batch_fallback": batch_events,
            "prefetch": prefetch_totals,
            "gpu_scoring_window_started_at": gpu_scoring_window_started_at,
            "gpu_scoring_window_finished_at": gpu_scoring_window_finished_at,
            "gpu_scoring_window_elapsed_s": gpu_scoring_elapsed_s,
            "peak_gpu_memory": {
                "max_allocated_bytes": peak_allocated,
                "max_reserved_bytes": peak_reserved,
                "total_bytes": total_gpu_memory,
                "max_reserved_fraction": peak_reserved / total_gpu_memory if total_gpu_memory else None,
            },
            "finished_at": utc_now(),
        }
    )
    write_json(summary_path, summary)
    write_json(
        progress_path,
        {
            "status": "pass",
            "model_key": model_row["model_key"],
            "rows_complete": len(completed),
            "rows_expected": expected_rows,
            "updated_at": utc_now(),
        },
    )
    return summary


def _quantile(values: Sequence[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * proportion
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _parse_telemetry_timestamp(raw: Any) -> datetime:
    value = str(raw or "").strip()
    if not value:
        raise ValueError("missing telemetry timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("telemetry timestamp lacks a timezone")
    return parsed.astimezone(timezone.utc)


def _read_gpu_utilization_samples(path: Path) -> dict[str, Any]:
    import csv

    if not path.is_file():
        return {"status": "missing", "sample_count": 0, "samples": []}
    samples: list[tuple[datetime, float]] = []
    invalid_rows: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            raw = row.get("utilization_gpu_pct")
            try:
                timestamp = _parse_telemetry_timestamp(row.get("timestamp"))
                value = float(str(raw).strip())
                if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                    raise ValueError("GPU utilization is outside [0, 100]")
                samples.append((timestamp, value))
            except (TypeError, ValueError, OverflowError):
                invalid_rows.append(row_number)
    if invalid_rows:
        return {
            "status": "fail",
            "sample_count": len(samples),
            "invalid_row_count": len(invalid_rows),
            "invalid_row_numbers": invalid_rows[:20],
            "samples": samples,
        }
    if not samples:
        return {"status": "fail", "sample_count": 0, "samples": []}
    samples.sort(key=lambda row: row[0])
    return {"status": "pass", "sample_count": len(samples), "samples": samples}


def _gpu_utilization_summary(
    path: Path,
    *,
    start_at: str | None = None,
    end_at: str | None = None,
) -> dict[str, Any]:
    parsed = _read_gpu_utilization_samples(path)
    if parsed["status"] != "pass":
        return {key: value for key, value in parsed.items() if key != "samples"}
    start = _parse_telemetry_timestamp(start_at) if start_at else None
    end = _parse_telemetry_timestamp(end_at) if end_at else None
    if start is not None and end is not None and end < start:
        return {"status": "fail", "sample_count": 0, "reason": "invalid_window"}
    values = [
        value
        for timestamp, value in parsed["samples"]
        if (start is None or timestamp >= start) and (end is None or timestamp <= end)
    ]
    if not values:
        return {"status": "fail", "sample_count": 0, "reason": "empty_window"}
    return {
        "status": "pass",
        "sample_count": len(values),
        "mean_pct": sum(values) / len(values),
        "p10_pct": _quantile(values, 0.10),
        "p50_pct": _quantile(values, 0.50),
        "p90_pct": _quantile(values, 0.90),
        "minimum_pct": min(values),
        "maximum_pct": max(values),
        "fraction_at_least_90_pct": sum(value >= 90.0 for value in values) / len(values),
        "window_started_at": start_at,
        "window_finished_at": end_at,
    }


def _common_allocation_gpu_utilization_summary(
    paths: Sequence[Path],
    *,
    interval_seconds: float,
) -> dict[str, Any]:
    if len(paths) != 4 or not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
        raise ValueError("common allocation telemetry requires four paths and a positive interval")
    parsed = [_read_gpu_utilization_samples(path) for path in paths]
    if any(row["status"] != "pass" for row in parsed):
        return {
            "status": "fail",
            "rank_statuses": [row["status"] for row in parsed],
        }
    allocation_start = min(row["samples"][0][0] for row in parsed)
    allocation_end = max(row["samples"][-1][0] for row in parsed)
    bucket_count = max(
        1,
        int(math.floor((allocation_end - allocation_start).total_seconds() / interval_seconds))
        + 1,
    )
    rank_buckets: list[dict[int, list[float]]] = []
    for row in parsed:
        buckets: dict[int, list[float]] = {}
        for timestamp, value in row["samples"]:
            bucket = int(
                round((timestamp - allocation_start).total_seconds() / interval_seconds)
            )
            if 0 <= bucket < bucket_count:
                buckets.setdefault(bucket, []).append(value)
        rank_buckets.append(buckets)
    zero_filled_values: list[float] = []
    observed_rank_bucket_count = 0
    for bucket in range(bucket_count):
        for buckets in rank_buckets:
            values = buckets.get(bucket, [])
            if values:
                observed_rank_bucket_count += 1
                zero_filled_values.append(sum(values) / len(values))
            else:
                zero_filled_values.append(0.0)
    return {
        "status": "pass",
        "definition": "fixed_interval_rank_buckets_over_union_window_with_absent_rank_buckets_zero_filled",
        "allocation_window_started_at": allocation_start.isoformat(),
        "allocation_window_finished_at": allocation_end.isoformat(),
        "interval_seconds": interval_seconds,
        "bucket_count": bucket_count,
        "rank_bucket_count": bucket_count * 4,
        "observed_rank_bucket_count": observed_rank_bucket_count,
        "zero_filled_rank_bucket_count": bucket_count * 4 - observed_rank_bucket_count,
        "mean_pct": sum(zero_filled_values) / len(zero_filled_values),
    }


def _finite_metric(name: str, value: Any, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or not lower <= parsed <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return parsed


def evaluate_utilization_tail_gate(
    *,
    common_allocation_mean_pct: float,
    minimum_worker_inference_mean_pct: float,
    minimum_worker_inference_p90_pct: float,
    minimum_worker_inference_sample_count: int,
    projected_rank_occupancy_fraction: float,
    projected_rank_tail_idle_fraction: float,
    recommended_walltime_s: int,
    configured_walltime_s: int,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate measured inference utilization and projected rank occupancy."""
    common_mean = _finite_metric(
        "common_allocation_mean_pct", common_allocation_mean_pct, lower=0.0, upper=100.0
    )
    worker_mean = _finite_metric(
        "minimum_worker_inference_mean_pct",
        minimum_worker_inference_mean_pct,
        lower=0.0,
        upper=100.0,
    )
    worker_p90 = _finite_metric(
        "minimum_worker_inference_p90_pct",
        minimum_worker_inference_p90_pct,
        lower=0.0,
        upper=100.0,
    )
    occupancy = _finite_metric(
        "projected_rank_occupancy_fraction",
        projected_rank_occupancy_fraction,
        lower=0.0,
        upper=1.0,
    )
    tail_idle = _finite_metric(
        "projected_rank_tail_idle_fraction",
        projected_rank_tail_idle_fraction,
        lower=0.0,
        upper=1.0,
    )
    if isinstance(minimum_worker_inference_sample_count, bool) or not isinstance(
        minimum_worker_inference_sample_count, int
    ) or minimum_worker_inference_sample_count < 0:
        raise ValueError("minimum_worker_inference_sample_count must be a nonnegative integer")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (recommended_walltime_s, configured_walltime_s)
    ):
        raise ValueError("walltimes must be positive integer seconds")
    minimum_common_mean = _finite_metric(
        "threshold.minimum_common_allocation_mean_pct",
        thresholds["minimum_common_allocation_mean_pct"],
        lower=0.0,
        upper=100.0,
    )
    minimum_inference_mean = _finite_metric(
        "threshold.minimum_worker_inference_mean_pct",
        thresholds["minimum_worker_inference_mean_pct"],
        lower=0.0,
        upper=100.0,
    )
    minimum_inference_p90 = _finite_metric(
        "threshold.minimum_worker_inference_p90_pct",
        thresholds["minimum_worker_inference_p90_pct"],
        lower=0.0,
        upper=100.0,
    )
    minimum_samples_raw = thresholds["minimum_worker_inference_sample_count"]
    if isinstance(minimum_samples_raw, bool) or not isinstance(minimum_samples_raw, int):
        raise ValueError("minimum worker inference sample threshold must be an integer")
    minimum_samples = minimum_samples_raw
    if minimum_samples < 1:
        raise ValueError("minimum worker inference sample threshold must be positive")
    minimum_occupancy = _finite_metric(
        "threshold.minimum_projected_rank_occupancy_fraction",
        thresholds["minimum_projected_rank_occupancy_fraction"],
        lower=0.0,
        upper=1.0,
    )
    maximum_tail = _finite_metric(
        "threshold.maximum_projected_rank_tail_idle_fraction",
        thresholds["maximum_projected_rank_tail_idle_fraction"],
        lower=0.0,
        upper=1.0,
    )
    common_mean_role = str(thresholds["common_allocation_mean_role"])
    if common_mean_role != COMMON_ALLOCATION_MEAN_ROLE:
        raise ValueError(
            "common allocation utilization role must remain descriptive and advisory"
        )
    issues: list[str] = []
    advisories: list[str] = []
    if common_mean < minimum_common_mean:
        advisories.append("common_allocation_mean_gpu_utilization")
    if worker_mean < minimum_inference_mean:
        issues.append("worker_inference_mean_gpu_utilization")
    if worker_p90 < minimum_inference_p90:
        issues.append("worker_inference_p90_gpu_utilization")
    if minimum_worker_inference_sample_count < minimum_samples:
        issues.append("worker_inference_sample_count")
    if occupancy < minimum_occupancy:
        issues.append("projected_rank_occupancy_fraction")
    if tail_idle > maximum_tail:
        issues.append("projected_rank_tail_idle_fraction")
    if recommended_walltime_s > configured_walltime_s:
        issues.append("three_hour_walltime")
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "advisories": advisories,
        "common_allocation_mean_gpu_utilization_pct": common_mean,
        "common_allocation_mean_role": common_mean_role,
        "minimum_worker_inference_mean_gpu_utilization_pct": worker_mean,
        "minimum_worker_inference_p90_gpu_utilization_pct": worker_p90,
        "minimum_worker_inference_sample_count": minimum_worker_inference_sample_count,
        "projected_aggregate_rank_occupancy_fraction": occupancy,
        "projected_max_rank_tail_idle_fraction": tail_idle,
        "thresholds": dict(thresholds),
    }


def validate_exact_shard_key_union(
    *,
    selection_rows: Sequence[dict[str, Any]],
    assignments: Sequence[dict[str, Any]],
    rows_by_assignment: dict[str, Sequence[dict[str, Any]]],
) -> list[str]:
    """Return deterministic overlap/gap issues for one model's shard union."""
    issues: list[str] = []
    all_keys: list[tuple[str, str]] = []
    for assignment in assignments:
        assignment_id = str(assignment["assignment_id"])
        rows = rows_by_assignment.get(assignment_id, [])
        observed = [(str(row.get("item_id") or ""), str(row.get("arm") or "")) for row in rows]
        expected = {
            (str(selection_row["item_id"]), arm)
            for selection_row in selection_rows[
                int(assignment["selection_start"]) : int(assignment["selection_stop"])
            ]
            for arm in ARMS
        }
        if len(observed) != len(set(observed)):
            issues.append(f"duplicate_within:{assignment_id}")
        if set(observed) != expected:
            issues.append(f"key_union:{assignment_id}")
        all_keys.extend(observed)
    expected_all = {
        (str(selection_row["item_id"]), arm)
        for selection_row in selection_rows
        for arm in ARMS
    }
    if len(all_keys) != len(set(all_keys)):
        issues.append("cross_shard_duplicate")
    if set(all_keys) != expected_all:
        issues.append("cross_shard_union")
    return sorted(set(issues))


def audit_replicated_anchor_union(
    *,
    selection_rows: Sequence[dict[str, Any]],
    instances: Sequence[dict[str, Any]],
    instance_keys: dict[str, Sequence[tuple[str, str]]],
    instance_assignments: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Audit the exact two-instance union for one analysis model."""
    merged_keys = [
        key
        for instance in instances
        for key in instance_keys.get(str(instance["model_key"]), [])
    ]
    intervals = sorted(
        (
            int(assignment["selection_start"]),
            int(assignment["selection_stop"]),
        )
        for instance in instances
        for assignment in instance_assignments.get(str(instance["model_key"]), [])
    )
    expected = {
        (str(selection_row["item_id"]), arm)
        for selection_row in selection_rows
        for arm in ARMS
    }
    issues: list[str] = []
    if len(instances) not in {1, 2}:
        issues.append("instance_count")
    cursor = 0
    for start, stop in intervals:
        if start != cursor or stop <= start or stop > len(selection_rows):
            issues.append("cross_instance_interval_coverage")
            break
        cursor = stop
    if cursor != len(selection_rows):
        issues.append("cross_instance_interval_coverage")
    if len(merged_keys) != len(set(merged_keys)):
        issues.append("cross_instance_duplicate")
    if set(merged_keys) != expected:
        issues.append("cross_instance_union")
    issues = sorted(set(issues))
    status = "fail" if issues else (
        "not_applicable_single_instance" if len(instances) == 1 else "pass"
    )
    return {
        "status": status,
        "issues": issues,
        "instance_model_keys": sorted(str(row["model_key"]) for row in instances),
        "intervals": [list(interval) for interval in intervals],
        "observed_row_count": len(merged_keys),
        "expected_row_count": len(expected),
    }


def technical_verify(
    *,
    data_dir: Path,
    run_dir: Path,
    selection_path: Path,
    model_plan_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    selection, prereg, model_plan = _load_selection(
        data_dir=data_dir.expanduser().resolve(),
        selection_path=selection_path.expanduser().resolve(),
        model_plan_path=model_plan_path.expanduser().resolve(),
        contract_path=contract_path.expanduser().resolve(),
    )
    contract, _ = _load_contract(contract_path.expanduser().resolve())
    expected_rows = len(selection["selection"]) * len(ARMS)
    full_decisions_per_model = int(
        contract["battery"]["required_trial_count"]
    ) * len(ARMS)
    model_reports: list[dict[str, Any]] = []
    instance_keys: dict[str, list[tuple[str, str]]] = {}
    instance_assignments: dict[str, list[dict[str, Any]]] = {}
    all_pass = True
    for model_row in model_plan:
        model_dir = run_dir / "models" / model_row["model_key"]
        issues: list[str] = []
        if selection["mode"] == "canary":
            assignments = [
                {
                    "assignment_id": f"{model_row['model_key']}.canary",
                    "rank": model_row["worker_index"],
                    "selection_start": 0,
                    "selection_stop": len(selection["selection"]),
                    "artifact_dir": model_dir,
                }
            ]
        else:
            assignments = []
            for rank_row in selection["batch_profile"]["schedule"]["ranks"]:
                for assignment in rank_row["assignments"]:
                    if assignment["model_key"] == model_row["model_key"]:
                        assignments.append(
                            {
                                **assignment,
                                "artifact_dir": model_dir / "shards" / assignment["assignment_id"],
                            }
                        )
            assignments.sort(key=lambda row: int(row["selection_start"]))
        if not assignments:
            issues.append("missing_frozen_assignments")
        all_rows: list[dict[str, Any]] = []
        rows_by_assignment: dict[str, list[dict[str, Any]]] = {}
        assignment_summaries: list[dict[str, Any]] = []
        for assignment in assignments:
            artifact_dir = Path(assignment["artifact_dir"])
            summary_path = artifact_dir / "summary.json"
            rows_path = artifact_dir / "rows.jsonl"
            assigned_count = int(assignment["selection_stop"]) - int(
                assignment["selection_start"]
            )
            assigned_expected_rows = assigned_count * len(ARMS)
            if not summary_path.is_file() or not rows_path.is_file():
                issues.append(f"missing_worker_artifacts:{assignment['assignment_id']}")
                continue
            summary = read_json(summary_path)
            assignment_summaries.append(summary)
            if summary.get("status") != "pass" or summary.get("rows_complete") != assigned_expected_rows:
                issues.append(f"worker_summary:{assignment['assignment_id']}")
            expected_common = {
                "model_key": model_row["model_key"],
                "model_id": model_row["model_id"],
                "revision": model_row["revision"],
                "dtype": "bf16",
                "worker_index": int(assignment["rank"]),
                "assignment_id": str(assignment["assignment_id"]),
                "selection_start": int(assignment["selection_start"]),
                "selection_stop": int(assignment["selection_stop"]),
                "selection_manifest_sha256": sha256_file(selection_path),
                "preregistration_sha256": selection["preregistration_sha256"],
                "code_snapshot_sha256": selection["code_snapshot_sha256"],
                "candidate_A_text": prereg["candidate_completions"]["A"],
                "candidate_B_text": prereg["candidate_completions"]["B"],
                "candidate_shared_prefix": prereg.get("candidate_shared_prefix"),
            }
            rows = list(iter_jsonl(rows_path))
            rows_by_assignment[str(assignment["assignment_id"])] = rows
            if len(rows) != assigned_expected_rows:
                issues.append(f"row_count:{assignment['assignment_id']}")
            assignment_keys: set[tuple[str, str]] = set()
            for row in rows:
                issues.extend(validate_completed_row(row, expected_common=expected_common))
                key = (str(row.get("item_id") or ""), str(row.get("arm") or ""))
                if key in assignment_keys:
                    issues.append(f"duplicate_row:{assignment['assignment_id']}")
                assignment_keys.add(key)
            assigned_selection = selection["selection"][
                int(assignment["selection_start"]) : int(assignment["selection_stop"])
            ]
            assigned_expected_keys = {
                (str(selection_row["item_id"]), arm)
                for selection_row in assigned_selection
                for arm in ARMS
            }
            if assignment_keys != assigned_expected_keys:
                issues.append(f"row_union:{assignment['assignment_id']}")
            all_rows.extend(rows)
        keys = [(str(row.get("item_id") or ""), str(row.get("arm") or "")) for row in all_rows]
        expected_keys = {
            (str(selection_row["item_id"]), arm)
            for assignment in assignments
            for selection_row in selection["selection"][
                int(assignment["selection_start"]) : int(assignment["selection_stop"])
            ]
            for arm in ARMS
        }
        if len(keys) != len(set(keys)):
            issues.append("cross_shard_duplicate_row")
        if set(keys) != expected_keys:
            issues.append("cross_shard_row_union")
        instance_keys[model_row["model_key"]] = keys
        instance_assignments[model_row["model_key"]] = assignments
        utilization_by_rank = {
            str(rank): _gpu_utilization_summary(
                run_dir / "telemetry" / f"gpu_worker_{rank}.csv"
            )
            for rank in sorted({int(row["rank"]) for row in assignments})
        }
        if any(row.get("status") != "pass" for row in utilization_by_rank.values()):
            issues.append("gpu_utilization_telemetry")
        summary = assignment_summaries[0] if assignment_summaries else {}
        inference_utilization = (
            _gpu_utilization_summary(
                run_dir
                / "telemetry"
                / f"gpu_worker_{int(model_row['worker_index'])}.csv",
                start_at=summary.get("gpu_scoring_window_started_at"),
                end_at=summary.get("gpu_scoring_window_finished_at"),
            )
            if selection["mode"] == "canary"
            else {"status": "not_applicable"}
        )
        if selection["mode"] == "canary" and inference_utilization.get("status") != "pass":
            issues.append("gpu_inference_window_telemetry")
        rows_present_at_start = [
            value
            for value in (
                row.get("rows_present_at_start") for row in assignment_summaries
            )
            if type(value) is int
        ]
        fresh_canary = (
            selection["mode"] != "canary"
            or (
                len(rows_present_at_start) == len(assignment_summaries) == 1
                and rows_present_at_start == [0]
            )
        )
        if not fresh_canary:
            issues.append("canary_rows_present_at_start")
        for field in (
            "load_elapsed_s",
            "decisions_per_second",
            "peak_gpu_memory",
            "effective_batch_size",
            "batch_strategy",
            "batch_fallback",
            "prefetch",
            "gpu_scoring_window_started_at",
            "gpu_scoring_window_finished_at",
            "gpu_scoring_window_elapsed_s",
        ):
            if selection["mode"] == "canary" and summary.get(field) is None:
                issues.append(f"missing_summary_{field}")
        throughput = summary.get("decisions_per_second")
        if selection["mode"] == "canary" and (
            type(throughput) not in {int, float}
            or not math.isfinite(float(throughput))
            or float(throughput) <= 0.0
        ):
            issues.append("invalid_canary_throughput")
        estimated_full_inference_s = (
            full_decisions_per_model / float(throughput)
            if type(throughput) in {int, float} and float(throughput) > 0.0
            else None
        )
        if selection["mode"] == "full" and not issues:
            order = {arm: index for index, arm in enumerate(ARMS)}
            all_rows.sort(key=lambda row: (int(row["selection_index"]), order[str(row["arm"])]))
            atomic_write_text(
                model_dir / "rows.jsonl",
                "".join(
                    json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                    for row in all_rows
                ),
            )
            write_json(
                model_dir / "aggregate_summary.json",
                {
                    "kind": "phase2_reanchored_job_a_model_shard_aggregate",
                    "status": "pass",
                    "model_key": model_row["model_key"],
                    "assignment_ids": [row["assignment_id"] for row in assignments],
                    "row_count": len(all_rows),
                    "rows_sha256": sha256_file(model_dir / "rows.jsonl"),
                    "exact_one_result_per_key": True,
                },
            )
        issues = sorted(set(issues))
        if issues:
            all_pass = False
        model_reports.append(
            {
                "model_key": model_row["model_key"],
                "analysis_model_key": model_row["analysis_model_key"],
                "worker_index": model_row["worker_index"],
                "status": "pass" if not issues else "fail",
                "issues": sorted(set(issues)),
                "load_time_s": summary.get("load_elapsed_s"),
                "decisions_per_second": summary.get("decisions_per_second"),
                "peak_gpu_memory": summary.get("peak_gpu_memory"),
                "gpu_worker_window_utilization_distribution": utilization_by_rank.get(
                    str(model_row["worker_index"]), {"status": "missing"}
                ),
                "gpu_inference_window_utilization_distribution": inference_utilization,
                "gpu_utilization_by_assigned_rank": utilization_by_rank,
                "requested_microbatch_size": summary.get("requested_microbatch_size"),
                "effective_batch_size": summary.get("effective_batch_size"),
                "strategy": summary.get("batch_strategy"),
                "batch_fallback": summary.get("batch_fallback"),
                "prefetch": summary.get("prefetch"),
                "estimated_full_inference_s": estimated_full_inference_s,
                "rows_present_at_start": rows_present_at_start,
                "fresh_canary": fresh_canary if selection["mode"] == "canary" else None,
            }
        )
    replicated_anchor_union_audit: dict[str, Any] = {
        "status": "not_applicable_canary",
        "models": {},
    }
    if selection["mode"] == "full":
        replicated_anchor_union_audit = {"status": "pass", "models": {}}
        grouped_plan: dict[str, list[dict[str, Any]]] = {}
        for model_row in model_plan:
            grouped_plan.setdefault(model_row["analysis_model_key"], []).append(
                model_row
            )
        for analysis_model_key, instances in sorted(grouped_plan.items()):
            audit = audit_replicated_anchor_union(
                selection_rows=selection["selection"],
                instances=instances,
                instance_keys=instance_keys,
                instance_assignments=instance_assignments,
            )
            replicated_anchor_union_audit["models"][analysis_model_key] = audit
            issues = audit["issues"]
            if issues:
                all_pass = False
                replicated_anchor_union_audit["status"] = "fail"
                affected = {instance["model_key"] for instance in instances}
                for report in model_reports:
                    if report["model_key"] in affected:
                        report["status"] = "fail"
                        report["issues"] = sorted(
                            set(report["issues"])
                            | {f"analysis_model_union:{issue}" for issue in issues}
                        )
    full_runtime_estimate = None
    schedule = None
    utilization_tail_gate = None
    common_allocation_utilization = None
    if selection["mode"] == "canary" and all_pass:
        schedule = build_balanced_full_schedule(
            model_reports,
            selection_count=int(contract["battery"]["required_trial_count"]),
            arm_count=len(ARMS),
        )
        measured_projection_s = float(schedule["projected_makespan_s"])
        conservative_projection_s = measured_projection_s * 1.20 + 600.0
        recommended_seconds = int(math.ceil(conservative_projection_s / 300.0) * 300)
        thresholds = contract.get("canary", {}).get("utilization_tail_gate", {})
        common_allocation_utilization = _common_allocation_gpu_utilization_summary(
            [run_dir / "telemetry" / f"gpu_worker_{rank}.csv" for rank in range(4)],
            interval_seconds=float(thresholds["telemetry_interval_seconds"]),
        )
        if common_allocation_utilization.get("status") != "pass":
            raise ValueError("common allocation GPU telemetry is invalid")
        inference_rows = [
            row["gpu_inference_window_utilization_distribution"]
            for row in model_reports
        ]
        utilization_tail_gate = evaluate_utilization_tail_gate(
            common_allocation_mean_pct=float(common_allocation_utilization["mean_pct"]),
            minimum_worker_inference_mean_pct=min(
                float(row["mean_pct"]) for row in inference_rows
            ),
            minimum_worker_inference_p90_pct=min(
                float(row["p90_pct"]) for row in inference_rows
            ),
            minimum_worker_inference_sample_count=min(
                int(row["sample_count"]) for row in inference_rows
            ),
            projected_rank_occupancy_fraction=float(
                schedule["projected_aggregate_rank_occupancy_fraction"]
            ),
            projected_rank_tail_idle_fraction=float(
                schedule["projected_max_rank_tail_idle_fraction"]
            ),
            recommended_walltime_s=recommended_seconds,
            configured_walltime_s=10800,
            thresholds=thresholds,
        )
        gate_issues = list(utilization_tail_gate["issues"])
        if gate_issues:
            all_pass = False
        full_runtime_estimate = {
            "full_decisions_per_model": full_decisions_per_model,
            "execution": "frozen_two_wave_balanced_schedule",
            "balanced_schedule_projection_s": measured_projection_s,
            "overhead_rule": "20% throughput margin plus 600 seconds for staging, verification, sealed-label analysis, and copyback",
            "recommended_walltime_s_rounded_to_5_minutes": recommended_seconds,
            "configured_full_walltime_s": 10800,
            "fits_configured_full_walltime": recommended_seconds <= 10800,
        }
    report = {
        "kind": "phase2_reanchored_job_a_technical_verify",
        "status": "pass" if all_pass else "fail",
        "mode": selection["mode"],
        "frame_count": selection["frame_count"],
        "trial_count": selection["trial_count"],
        "expected_rows_per_model": expected_rows,
        "replicated_anchor_union_audit": replicated_anchor_union_audit,
        "models": model_reports,
        "frozen_balanced_schedule": schedule,
        "common_allocation_gpu_utilization": common_allocation_utilization,
        "utilization_tail_gate": utilization_tail_gate,
        "full_runtime_estimate_from_canary": full_runtime_estimate,
        "labels_joined": False,
        "created_at": utc_now(),
    }
    write_json(run_dir / "technical_verify.json", report)
    if selection["mode"] == "canary" and all_pass:
        runtime_bindings = {
            "sif_sha256": str(os.environ.get("SIF_SHA256") or "").lower(),
            "release_manifest_sha256": str(
                os.environ.get("RELEASE_MANIFEST_SHA256") or ""
            ).lower(),
            "data_manifest_sha256": selection["data_manifest_sha256"],
            "hash_manifest_sha256": selection["hash_manifest_sha256"],
        }
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in runtime_bindings.values()
        ):
            raise ValueError("canary runtime identity bindings are missing or malformed")
        profile_models = [
            {
                "model_key": row["model_key"],
                "effective_batch_size": row["effective_batch_size"],
                "strategy": row["strategy"],
                "load_time_s": row["load_time_s"],
                "decisions_per_second": row["decisions_per_second"],
                "peak_gpu_memory": row["peak_gpu_memory"],
                "gpu_worker_window_utilization_distribution": row[
                    "gpu_worker_window_utilization_distribution"
                ],
                "gpu_inference_window_utilization_distribution": row[
                    "gpu_inference_window_utilization_distribution"
                ],
                "batch_fallback": row["batch_fallback"],
                "estimated_full_inference_s": row["estimated_full_inference_s"],
            }
            for row in model_reports
        ]
        profile = {
            "kind": "phase2_reanchored_job_a_batch_profile",
            "status": "pass",
            "source_mode": "canary",
            "preregistration_sha256": selection["preregistration_sha256"],
            "model_plan_sha256": sha256_file(model_plan_path),
            "contract_sha256": sha256_file(contract_path),
            "code_snapshot_sha256": selection["code_snapshot_sha256"],
            **runtime_bindings,
            "selection_manifest_sha256": sha256_file(selection_path),
            "canary_measurements_sha256": canonical_sha256(profile_models),
            "fresh_canary_provenance": {
                "status": "pass",
                "required_rows_present_at_start": 0,
                "models": {
                    row["model_key"]: {
                        "fresh_canary": row["fresh_canary"],
                        "rows_present_at_start": row["rows_present_at_start"],
                    }
                    for row in model_reports
                },
            },
            "models": profile_models,
            "schedule": schedule,
            "common_allocation_gpu_utilization": common_allocation_utilization,
            "utilization_tail_gate": utilization_tail_gate,
            "full_runtime_estimate": full_runtime_estimate,
            "created_at": utc_now(),
        }
        write_json(run_dir / "batch_profile.json", profile)
        report["batch_profile"] = str(run_dir / "batch_profile.json")
        report["batch_profile_sha256"] = sha256_file(run_dir / "batch_profile.json")
        write_json(run_dir / "technical_verify.json", report)
    return report


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    pre = commands.add_parser("preflight")
    pre.add_argument("--data-dir", type=Path, required=True)
    pre.add_argument("--run-dir", type=Path, required=True)
    pre.add_argument("--model-plan", type=Path, required=True)
    pre.add_argument("--contract", type=Path, required=True)
    pre.add_argument("--mode", choices=("canary", "full"), required=True)
    pre.add_argument("--limit-frames", type=int, required=True)
    pre.add_argument("--batch-profile", type=Path)

    worker = commands.add_parser("run-worker")
    worker.add_argument("--data-dir", type=Path, required=True)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--selection-manifest", type=Path, required=True)
    worker.add_argument("--model-plan", type=Path, required=True)
    worker.add_argument("--contract", type=Path, required=True)
    worker.add_argument("--worker-index", type=int, choices=range(4), required=True)
    worker.add_argument("--model-key")
    worker.add_argument("--assignment-id")
    worker.add_argument("--selection-start", type=int)
    worker.add_argument("--selection-stop", type=int)
    worker.add_argument("--resume", action="store_true")

    verify = commands.add_parser("technical-verify")
    verify.add_argument("--data-dir", type=Path, required=True)
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--selection-manifest", type=Path, required=True)
    verify.add_argument("--model-plan", type=Path, required=True)
    verify.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        payload = preflight(
            data_dir=args.data_dir,
            run_dir=args.run_dir,
            model_plan_path=args.model_plan,
            contract_path=args.contract,
            mode=args.mode,
            limit_frames=args.limit_frames,
            batch_profile_path=args.batch_profile,
        )
    elif args.command == "run-worker":
        payload = run_worker(
            data_dir=args.data_dir,
            run_dir=args.run_dir,
            selection_path=args.selection_manifest,
            model_plan_path=args.model_plan,
            contract_path=args.contract,
            worker_index=args.worker_index,
            resume=args.resume,
            model_key=args.model_key,
            assignment_id=args.assignment_id,
            selection_start=args.selection_start,
            selection_stop=args.selection_stop,
        )
    else:
        payload = technical_verify(
            data_dir=args.data_dir,
            run_dir=args.run_dir,
            selection_path=args.selection_manifest,
            model_plan_path=args.model_plan,
            contract_path=args.contract,
        )
    _print(payload)
    return 0 if payload.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
