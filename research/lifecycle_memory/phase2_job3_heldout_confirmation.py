#!/usr/bin/env python3
"""Fail-closed runtime for the burned-data scoring pilot.

Workers consume only public scoring packs and a label-free selection manifest.
The evaluator index is opened only by successful pilot aggregation; canary and
generic full-mode aggregation remain label-free.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

ARM_PCS = "pcs_e"
ARM_MPP = "mpp_e"
ARM_VP = "vp_e"
ARM_ALT = "mpp_e_alternate_descriptive_nonconfirmatory"
ARM_ORDER = (ARM_PCS, ARM_MPP, ARM_VP, ARM_ALT)

SERIALIZATION_ID = "single_user_system_text_then_images_then_user_text_v1"
CANDIDATE_CONSTRUCTION_NATIVE = "native_assistant_chat_template"
CANDIDATE_CONSTRUCTION_EXACT_APPEND = "base_generation_prompt_exact_token_append"
FROZEN_CANDIDATE_COMPLETIONS = {
    "false": '{"answer":false,"abstain":false,"abstention_reason":null,"confidence":1.0}',
    "true": '{"answer":true,"abstain":false,"abstention_reason":null,"confidence":1.0}',
}
FROZEN_MODEL_PLAN = (
    (
        "gemma4_31b_hf_bf16",
        "google/gemma-4-31B-it",
        "3548789868c5356dbf307c98e6f609007b82b3eb",
    ),
    (
        "qwen35_27b_hf_bf16",
        "Qwen/Qwen3.5-27B",
        "fc05daec18b0a78c049392ed2e771dde82bdf654",
    ),
)
GEMMA_NATIVE_PREFIX_REWRITE_SIGNATURE = {
    "id": "job2c_367860_gemma31_all_1362_candidates_v1",
    "source_job_id": "367860",
    "candidate_observation_count": 1362,
    "attempts": [
        {
            "continue_final_message": True,
            "full_minus_base_tokens": 19,
            "prefix_check": False,
            "exact_candidate_suffix": False,
            "visual_exact_identity": True,
        },
        {
            "continue_final_message": False,
            "full_minus_base_tokens": 21,
            "prefix_check": False,
            "exact_candidate_suffix": False,
            "visual_exact_identity": True,
        },
    ],
}
WORKER_TO_MODEL = {
    0: FROZEN_MODEL_PLAN[0][0],
    1: FROZEN_MODEL_PLAN[0][0],
    2: FROZEN_MODEL_PLAN[1][0],
    3: FROZEN_MODEL_PLAN[1][0],
}
WORKER_TO_SHARD = {0: 0, 1: 1, 2: 0, 3: 1}

PACK_RELATIVE_PATHS = {
    arm: f"public/scoring/{arm}_scoring_request_pack.json" for arm in ARM_ORDER
}
HIDDEN_RELATIVE_PATH = "hidden/evaluator_index.json"

EXPECTED_MANIFEST_KIND = "phase2_job3_egohos_burned_construction_manifest"
EXPECTED_MANIFEST_STATUS = "construction_pass_scoring_ready"
EXPECTED_SCORING_STATUS = "emitted_all_rows_eligible"
EXPECTED_ITEM_COUNT = 120
EXPECTED_CLASS_COUNTS = {str(class_id): 20 for class_id in range(3, 9)}
PILOT_ACCURACY_FLOOR = 0.65
PILOT_WILSON_FLOOR = 0.50
ONE_SIDED_95_Z = 1.6448536269514722

JOB2C_TRANSITIVE_RUNTIME_FILES = (
    "research/lifecycle_memory/phase2_job2c_behavioral_companion.py",
    "research/lifecycle_memory/run_phase2_job2c_behavioral_companion.py",
    "docs/research/lifecycle_memory/phase 2 research/jobs/job0/run_phase1_qwen27b_hf.py",
    "research/lifecycle_memory/run_phase2_mech_job1_instrument_gates.py",
    "research/lifecycle_memory/run_phase2_mech_job2_battery_capture.py",
    "research/lifecycle_memory/phase2_job2_probe_analysis.py",
    "research/lifecycle_memory/phase2_job2b_paired_analysis.py",
    "research/lifecycle_memory/build_epic_visor_dynamic_smoke.py",
    "research/lifecycle_memory/run_epic_visor_dynamic_llamacpp_metal.py",
    "research/lifecycle_memory/run_epic_visor_dynamic_vlm_smoke.py",
    "research/lifecycle_memory/protocol_v3_3rscan_visual_vlm.py",
)
EXPECTED_WORKER_RUNTIME_VERSIONS = {
    "accelerate": "1.14.0",
    "huggingface_hub": "1.22.0",
    "torch": "2.12.1+cu126",
    "transformers": "5.13.0",
}
EXPECTED_WORKER_RUNTIME_BY_MODEL = {
    "gemma4_31b_hf_bf16": {
        "adapter": "generic_generate",
        "model_class": "Gemma4ForConditionalGeneration",
        "processor_class": "Gemma4Processor",
        "tokenizer_class": "GemmaTokenizer",
        "snapshot_suffix": (
            "models--google--gemma-4-31B-it/snapshots/"
            "3548789868c5356dbf307c98e6f609007b82b3eb"
        ),
    },
    "qwen35_27b_hf_bf16": {
        "adapter": "generic_generate",
        "model_class": "Qwen3_5ForConditionalGeneration",
        "processor_class": "Qwen3VLProcessor",
        "tokenizer_class": "Qwen2Tokenizer",
        "snapshot_suffix": (
            "models--Qwen--Qwen3.5-27B/snapshots/"
            "fc05daec18b0a78c049392ed2e771dde82bdf654"
        ),
    },
}
EXPECTED_WORKER_GPU_NAME = "NVIDIA H100 80GB HBM3"

# Job 375739 completed all four frozen inference shards before an evaluator-only
# label-role normalization bug stopped aggregation. Compatibility is exact and
# pilot-only: no arbitrary or future worker snapshot can enter this path.
APPROVED_PILOT_WORKER_SNAPSHOT_COMPATIBILITY = {
    "f6b137448192183587ed6d0bb2fe3d6c90a6ff2913649e290f542238a296522c": {
        "source_job_id": "375739",
        "scope": "evaluator_only_negative_role_normalization",
        "reason": (
            "Workers completed before aggregation rejected the audited "
            "negative_human_audited_no_direct_contact role."
        ),
        "source_job_manifest_sha256": "6f09b557b94abf36235eb248b9a4bb9c1683c44bda695e1eb0bcf5d191569dd6",
        "selection_manifest_sha256": "df4b7d4447aa700532950d0b50b5908e5ff374c3fa21e6c04ec662838c19365c",
        "worker_artifact_sha256": {
            "0": {
                "rows": "e5ffae17ee387bbace871c91ea62e6bac671f33c552061d627ef05115ce2cdbc",
                "summary": "48a6ba72b1f41c61fd84b1e46b3ca8564a27b7879eeb4cbf035e1adf5ca28f89",
            },
            "1": {
                "rows": "b4dd8d39c1eb16973cd4920fb40e84316e2e89451afd83eb69faeb4ae1cab88e",
                "summary": "a80ef36bf75ac3f28d0671fb894b49a28cd7178aa9449c9017747736bb84f1e9",
            },
            "2": {
                "rows": "bad96ae6b6e3481c668aa244a67bab988a5b103e01b64fd7c1807529c51fba5f",
                "summary": "dfca7e3d1f2be2564e63e2a704c781171675c779185a2ca9c1abd7a374ed1ceb",
            },
            "3": {
                "rows": "1ef5b36c908a53504e138461330d92b829ffb38b00f496e03efb3444280a88af",
                "summary": "aab5fc372514d0f82a8ab31301cdfc478d1bda3008fb7d1440b41254b7495c68",
            },
        },
    }
}

FORBIDDEN_LABEL_KEYS = {
    "answer_count",
    "canonical_answer_json",
    "class_id",
    "contact",
    "correct",
    "correctness",
    "expected_answer",
    "hidden_label",
    "label_role",
    "target_label",
}
CANARY_FORBIDDEN_METRIC_TERMS = (
    "accuracy",
    "ce_bits",
    "cross_entropy",
    "effect_direction",
    "renderer_effect",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_with_sha256(path: Path) -> tuple[Any, str]:
    with path.open("rb") as handle:
        content = handle.read()
    digest = hashlib.sha256(content).hexdigest()
    return json.loads(content.decode("utf-8")), digest


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            yield row


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _atomic_write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _append_jsonl_atomic(path: Path, row: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    _atomic_write_text(path, existing + json.dumps(row, sort_keys=True) + "\n")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _job2c_transitive_snapshot_sha256() -> str:
    chunks: list[str] = []
    for relative in JOB2C_TRANSITIVE_RUNTIME_FILES:
        path = REPO_ROOT / relative
        chunks.append(f"### {relative}\n")
        try:
            chunks.append(path.read_text(encoding="utf-8"))
            chunks.append("\n")
        except Exception as exc:
            chunks.append(f"<unreadable:{type(exc).__name__}>\n")
    return _sha256_text("".join(chunks))


def _job2c_runtime() -> Any:
    from research.lifecycle_memory import phase2_job2c_behavioral_companion

    return phase2_job2c_behavioral_companion


def _gate_runtime() -> Any:
    from research.lifecycle_memory import run_phase2_mech_job1_instrument_gates

    return run_phase2_mech_job1_instrument_gates


def _code_snapshot_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "research" / "lifecycle_memory" / "run_phase2_job3_heldout_confirmation.py",
    )
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing runtime source for code hash: {path}")
        rows.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256_file(path)})
    return _canonical_sha256(
        {
            "job3_runtime_files": rows,
            "reused_job2c_transitive_snapshot_sha256": _job2c_transitive_snapshot_sha256(),
        }
    )


def _gate(name: str, passed: bool, detail: Any | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"gate": name, "status": "pass" if passed else "fail", "blocking": True}
    if detail is not None:
        row["detail"] = detail
    return row


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _assert_public_path(path: Path, data_dir: Path) -> None:
    if not _is_within(path, data_dir):
        raise ValueError(f"path escapes data directory: {path}")
    relative_parts = {part.lower() for part in path.resolve().relative_to(data_dir.resolve()).parts}
    if "hidden" in relative_parts or path.name == "evaluator_index.json":
        raise ValueError(f"worker path enters evaluator-only data: {path}")


def _contains_forbidden_label_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_LABEL_KEYS:
                return True
            if _contains_forbidden_label_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_label_key(item) for item in value)
    return False


def _load_model_plan(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError("model plan must be a JSON list")
    rows: list[dict[str, Any]] = []
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError("model plan rows must be objects")
        rows.append(
            {
                "model_key": str(raw.get("model_key") or ""),
                "model_id": str(raw.get("model_id") or ""),
                "revision": str(raw.get("revision") or ""),
                "dtype": str(raw.get("dtype") or ""),
                "worker_indexes": list(raw.get("worker_indexes") or []),
            }
        )
    observed = tuple((row["model_key"], row["model_id"], row["revision"]) for row in rows)
    if len(observed) != len(FROZEN_MODEL_PLAN) or set(observed) != set(FROZEN_MODEL_PLAN):
        raise ValueError(f"model plan drift: expected exactly {FROZEN_MODEL_PLAN}, got {observed}")
    by_key = {row["model_key"]: row for row in rows}
    expected_workers = {
        "gemma4_31b_hf_bf16": [0, 1],
        "qwen35_27b_hf_bf16": [2, 3],
    }
    for model_key, workers in expected_workers.items():
        row = by_key[model_key]
        if row["dtype"] != "bf16" or row["worker_indexes"] != workers:
            raise ValueError(
                f"model plan dtype/worker topology drift for {model_key}: "
                f"dtype={row['dtype']!r}, workers={row['worker_indexes']!r}"
            )
    return [by_key[model_key] for model_key, _, _ in FROZEN_MODEL_PLAN]


def _load_prespec(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("prespec must be a JSON object")
    if payload.get("kind") != "phase2_job3_heldout_confirmation_runtime_prespec":
        raise ValueError("invalid Job3 prespec kind")
    arm_rows = payload.get("arms")
    if not isinstance(arm_rows, list) or not all(isinstance(row, dict) for row in arm_rows):
        raise ValueError("prespec arms must be objects")
    if tuple(str(row.get("arm") or "") for row in arm_rows) != ARM_ORDER:
        raise ValueError("prespec arm order drift")
    if payload.get("candidate_completions") != FROZEN_CANDIDATE_COMPLETIONS:
        raise ValueError("prespec candidate byte strings drift")
    prompt_serialization = payload.get("prompt_serialization")
    if not isinstance(prompt_serialization, dict) or prompt_serialization.get("id") != SERIALIZATION_ID:
        raise ValueError("prespec prompt serialization drift")
    generation = payload.get("generation")
    if (
        not isinstance(generation, dict)
        or generation.get("do_sample") is not False
        or int(generation.get("max_new_tokens") or 0) <= 0
    ):
        raise ValueError("prespec generation contract drift")
    construction = payload.get("candidate_construction")
    if not isinstance(construction, dict) or tuple(construction.get("allowed_modes", [])) != (
        CANDIDATE_CONSTRUCTION_NATIVE,
        CANDIDATE_CONSTRUCTION_EXACT_APPEND,
    ):
        raise ValueError("prespec candidate construction policy drift")
    if (
        construction.get("audited_gemma_native_prefix_rewrite_signature")
        != GEMMA_NATIVE_PREFIX_REWRITE_SIGNATURE
    ):
        raise ValueError("prespec Gemma native prefix-rewrite signature drift")
    gates_payload = payload.get("pilot_task_validity_gate")
    if not isinstance(gates_payload, dict):
        raise ValueError("prespec missing pilot_task_validity_gate")
    if float(gates_payload.get("minimum_greedy_accuracy", -1)) != PILOT_ACCURACY_FLOOR:
        raise ValueError("prespec class accuracy floor drift")
    if float(gates_payload.get("minimum_one_sided_wilson_lower_bound", -1)) != PILOT_WILSON_FLOOR:
        raise ValueError("prespec Wilson floor drift")
    if float(gates_payload.get("wilson_z", -1)) != ONE_SIDED_95_Z:
        raise ValueError("prespec Wilson z drift")
    if gates_payload.get("strict_lower_bound_operator") != ">":
        raise ValueError("prespec Wilson operator drift")
    if not isinstance(payload.get("data_binding"), dict):
        raise ValueError("prespec missing data_binding")
    data_binding = payload["data_binding"]
    expected_binding = {
        "required_manifest_status": EXPECTED_MANIFEST_STATUS,
        "required_scoring_status": EXPECTED_SCORING_STATUS,
        "required_item_count": EXPECTED_ITEM_COUNT,
        "required_class_counts": EXPECTED_CLASS_COUNTS,
        "required_flags": {
            "burned": True,
            "dataset_role": "construction_pilot",
            "non_confirmatory": True,
            "rows_frozen": False,
            "scoring_permitted": True,
        },
    }
    for key, expected_value in expected_binding.items():
        if data_binding.get(key) != expected_value:
            raise ValueError(f"prespec data_binding drift for {key}")
    for key in ("manifest_sha256", "construction_audit_sha256", "evaluator_index_sha256"):
        value = str(data_binding.get(key) or "")
        if len(value) != 64:
            raise ValueError(f"prespec data pin missing or malformed: {key}")
    for row in arm_rows:
        arm = str(row["arm"])
        if row.get("scoring_pack") != PACK_RELATIVE_PATHS[arm] or len(str(row.get("sha256") or "")) != 64:
            raise ValueError(f"prespec scoring-pack pin drift for {arm}")
    visibility = payload.get("analysis_visibility")
    if not isinstance(visibility, dict) or not isinstance(
        visibility.get("canary_forbidden_outputs"), list
    ):
        raise ValueError("prespec missing canary output firewall")
    return payload


def _prespec_arm_rows(prespec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["arm"]): row for row in prespec["arms"]}


def _canary_forbidden_terms(prespec: dict[str, Any]) -> tuple[str, ...]:
    configured = tuple(
        str(value).lower()
        for value in prespec["analysis_visibility"]["canary_forbidden_outputs"]
    )
    return tuple(dict.fromkeys((*CANARY_FORBIDDEN_METRIC_TERMS, *configured)))


def _validate_mode_slice(mode: str, request_offset: int, limit_items: int) -> None:
    if mode not in {"canary", "pilot", "full"}:
        raise ValueError("mode must be canary, pilot, or full")
    if request_offset < 0 or limit_items < 0:
        raise ValueError("request offset and item limit must be non-negative")
    if mode == "canary" and (request_offset != 0 or limit_items != 4):
        raise ValueError("canary requires --request-offset=0 and --limit-items=4")
    if mode == "pilot" and (request_offset != 0 or limit_items != 0):
        raise ValueError("pilot requires --request-offset=0 and --limit-items=0")


def _pack_path(data_dir: Path, arm: str) -> Path:
    return data_dir / PACK_RELATIVE_PATHS[arm]


def _load_pack(data_dir: Path, arm: str) -> tuple[dict[str, Any], Path]:
    path = _pack_path(data_dir, arm)
    _assert_public_path(path, data_dir)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid public pack: {path}")
    return payload, path


def _resolve_input_path(data_dir: Path, raw_path: str) -> Path:
    path = (data_dir / raw_path).resolve()
    _assert_public_path(path, data_dir)
    return path


def _serialization_identity(
    *, system_prompt: str, user_prompt: str, ordered_image_hashes: Sequence[str]
) -> str:
    return _canonical_sha256(
        {
            "serialization_id": SERIALIZATION_ID,
            "content_order": ["system_prompt_text", "images", "user_prompt_text"],
            "system_prompt_sha256": _sha256_text(system_prompt),
            "user_prompt_sha256": _sha256_text(user_prompt),
            "ordered_image_sha256": list(ordered_image_hashes),
        }
    )


def _request_identity(request: dict[str, Any], *, data_dir: Path) -> dict[str, Any]:
    system_prompt = request.get("system_prompt")
    user_prompt = request.get("user_prompt")
    if not isinstance(system_prompt, str) or not system_prompt:
        raise ValueError("request missing exact system_prompt")
    if not isinstance(user_prompt, str) or not user_prompt:
        raise ValueError("request missing exact user_prompt")
    inputs = request.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("request has no public image inputs")
    input_rows: list[dict[str, Any]] = []
    hashes: list[str] = []
    for raw in inputs:
        if not isinstance(raw, dict) or not raw.get("path"):
            raise ValueError("invalid request input")
        relative = str(raw["path"])
        path = _resolve_input_path(data_dir, relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing request image: {path}")
        digest = _sha256_file(path)
        hashes.append(digest)
        input_rows.append(
            {
                "path": relative,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "kind": str(raw.get("kind") or ""),
                "width": raw.get("width"),
                "height": raw.get("height"),
            }
        )
    return {
        "request_id": str(request.get("request_id") or ""),
        "system_prompt_sha256": _sha256_text(system_prompt),
        "user_prompt_sha256": _sha256_text(user_prompt),
        "ordered_inputs": input_rows,
        "serialization_id": SERIALIZATION_ID,
        "serialization_sha256": _serialization_identity(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            ordered_image_hashes=hashes,
        ),
    }


def preflight(
    *,
    data_dir: Path,
    run_dir: Path,
    model_plan_path: Path,
    prespec_path: Path,
    mode: str,
    request_offset: int,
    limit_items: int,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    model_plan_path = model_plan_path.expanduser().resolve()
    prespec_path = prespec_path.expanduser().resolve()
    _validate_mode_slice(mode, int(request_offset), int(limit_items))
    model_plan = _load_model_plan(model_plan_path)
    prespec = _load_prespec(prespec_path)
    data_binding = prespec["data_binding"]
    required_flags = data_binding["required_flags"]
    prespec_arms = _prespec_arm_rows(prespec)

    gates_out: list[dict[str, Any]] = []
    manifest_path = data_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest_checks = {
        "full_original_manifest": manifest.get("worker_data_role") is None
        and manifest.get("source_manifest_sha256") is None,
        "kind": manifest.get("kind") == EXPECTED_MANIFEST_KIND,
        "construction_status": manifest.get("construction_status") == data_binding["required_manifest_status"],
        "scoring_export_status": manifest.get("scoring_export_status") == data_binding["required_scoring_status"],
        "selected_count": manifest.get("selected_count") == data_binding["required_item_count"],
        "class_counts": manifest.get("class_counts") == data_binding["required_class_counts"],
        "rows_frozen": manifest.get("rows_frozen") == required_flags["rows_frozen"],
        "scoring_export_unblocked": manifest.get("scoring_export_block_reason") is None,
        "manifest_sha256": _sha256_file(manifest_path) == data_binding["manifest_sha256"],
    }
    gates_out.append(_gate("scoring_ready_manifest", all(manifest_checks.values()), manifest_checks))

    pinned_sidecars = {
        "construction_audit": (
            data_dir / "audits" / "construction_audit.json",
            data_binding["construction_audit_sha256"],
        ),
        "evaluator_index": (
            data_dir / HIDDEN_RELATIVE_PATH,
            data_binding["evaluator_index_sha256"],
        ),
    }
    sidecar_checks: dict[str, Any] = {}
    for name, (path, expected_sha) in pinned_sidecars.items():
        sidecar_checks[name] = {
            "exists": path.is_file(),
            "sha256": _sha256_file(path) == expected_sha if path.is_file() else False,
        }
    gates_out.append(
        _gate(
            "prespec_data_sidecar_pins",
            all(all(checks.values()) for checks in sidecar_checks.values()),
            sidecar_checks,
        )
    )

    packs: dict[str, dict[str, Any]] = {}
    pack_hashes: dict[str, str] = {}
    pack_checks: dict[str, Any] = {}
    requests_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ARM_ORDER:
        payload, path = _load_pack(data_dir, arm)
        packs[arm] = payload
        pack_hashes[arm] = _sha256_file(path)
        rows = payload.get("requests")
        requests_by_arm[arm] = rows if isinstance(rows, list) else []
        checks = {
            "prespec_relative_path": str(path.relative_to(data_dir)) == prespec_arms[arm]["scoring_pack"],
            "prespec_sha256": pack_hashes[arm] == prespec_arms[arm]["sha256"],
            "arm": payload.get("arm") == arm,
            "request_count": payload.get("request_count") == data_binding["required_item_count"],
            "actual_count": len(requests_by_arm[arm]) == data_binding["required_item_count"],
            **{f"flag_{key}": payload.get(key) == value for key, value in required_flags.items()},
        }
        for row in requests_by_arm[arm]:
            checks["row_contract"] = checks.get("row_contract", True) and bool(
                isinstance(row, dict)
                and row.get("arm") == arm
                and all(row.get(key) == value for key, value in required_flags.items())
                and not _contains_forbidden_label_key(row)
            )
        pack_checks[arm] = checks
    gates_out.append(
        _gate(
            "public_scoring_pack_contract",
            all(all(checks.values()) for checks in pack_checks.values()),
            pack_checks,
        )
    )

    item_orders = {
        arm: [str(row.get("item_id") or "") for row in requests_by_arm[arm]] for arm in ARM_ORDER
    }
    canonical_order = item_orders[ARM_PCS]
    alignment_ok = bool(canonical_order) and len(set(canonical_order)) == data_binding["required_item_count"]
    alignment_ok = alignment_ok and all(item_orders[arm] == canonical_order for arm in ARM_ORDER)
    gates_out.append(_gate("four_pack_item_alignment", alignment_ok, {arm: len(v) for arm, v in item_orders.items()}))

    maps_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    duplicate_requests: dict[str, list[str]] = {}
    for arm in ARM_ORDER:
        arm_map: dict[str, dict[str, Any]] = {}
        duplicates: list[str] = []
        for row in requests_by_arm[arm]:
            item_id = str(row.get("item_id") or "")
            if not item_id or item_id in arm_map:
                duplicates.append(item_id)
            arm_map[item_id] = row
        maps_by_arm[arm] = arm_map
        if duplicates:
            duplicate_requests[arm] = duplicates
    gates_out.append(_gate("unique_public_item_ids", not duplicate_requests, duplicate_requests))

    selected_ids = canonical_order[int(request_offset) :]
    if int(limit_items) > 0:
        selected_ids = selected_ids[: int(limit_items)]
    selection_rows: list[dict[str, Any]] = []
    identity_failures: list[dict[str, Any]] = []
    for selection_index, item_id in enumerate(selected_ids):
        identities: dict[str, Any] = {}
        request_ids: dict[str, str] = {}
        for arm in ARM_ORDER:
            request = maps_by_arm.get(arm, {}).get(item_id)
            try:
                if request is None:
                    raise ValueError("missing aligned request")
                identity = _request_identity(request, data_dir=data_dir)
                if not identity["request_id"]:
                    raise ValueError("missing request_id")
                identities[arm] = identity
                request_ids[arm] = identity["request_id"]
            except Exception as exc:
                identity_failures.append(
                    {"item_id": item_id, "arm": arm, "error": f"{type(exc).__name__}: {exc}"}
                )
        selection_rows.append(
            {
                "selection_index": selection_index,
                "item_id": item_id,
                "request_id_by_arm": request_ids,
                "public_identity_by_arm": identities,
            }
        )
    gates_out.append(_gate("public_file_and_hash_validation", not identity_failures, identity_failures[:20]))
    expected_selection_count = (
        4
        if mode == "canary"
        else data_binding["required_item_count"]
        if mode == "pilot"
        else len(selected_ids)
    )
    gates_out.append(
        _gate(
            "mode_selection_count",
            len(selection_rows) == expected_selection_count,
            {"mode": mode, "expected": expected_selection_count, "observed": len(selection_rows)},
        )
    )

    selection_manifest = {
        "kind": "phase2_job3_label_free_selection_manifest",
        "status": "pass" if all(g["status"] == "pass" for g in gates_out) else "fail",
        "mode": mode,
        "request_offset": int(request_offset),
        "limit_items": int(limit_items),
        "selection_count": len(selection_rows),
        "selection": selection_rows,
        "arms": list(ARM_ORDER),
        "prompt_serialization_id": SERIALIZATION_ID,
        "data_manifest_relative_path": "manifest.json",
        "data_manifest_sha256": _sha256_file(manifest_path),
        "construction_audit_sha256": _sha256_file(pinned_sidecars["construction_audit"][0]),
        "evaluator_index_sha256": _sha256_file(pinned_sidecars["evaluator_index"][0]),
        "pack_relative_paths": PACK_RELATIVE_PATHS,
        "pack_sha256_by_arm": pack_hashes,
        "model_plan_sha256": _sha256_file(model_plan_path),
        "prespec_sha256": _sha256_file(prespec_path),
        "code_snapshot_sha256": _code_snapshot_sha256(),
        "model_plan": model_plan,
        "worker_topology": {
            str(worker): {"model_key": WORKER_TO_MODEL[worker], "shard_index": WORKER_TO_SHARD[worker], "shard_count": 2}
            for worker in range(4)
        },
        "claim_boundary": "Label-free public routing manifest; no evaluator labels or class fields.",
    }
    if _contains_forbidden_label_key(selection_manifest):
        raise RuntimeError("internal error: selection manifest crossed the label firewall")
    audit = {
        "kind": "phase2_job3_preflight_audit",
        "status": selection_manifest["status"],
        "mode": mode,
        "data_dir": str(data_dir),
        "run_dir": str(run_dir),
        "selection_manifest_path": str(run_dir / "selection_manifest.json"),
        "selection_count": len(selection_rows),
        "gates": gates_out,
        "blocking_failure_count": sum(g["status"] != "pass" for g in gates_out),
        "generated_at": _utc_now_iso(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "selection_manifest.json", selection_manifest)
    _write_json(run_dir / "preflight_summary.json", audit)
    return audit


def _load_selection_manifest(
    *,
    selection_manifest_path: Path,
    data_dir: Path,
    model_plan_path: Path,
    prespec_path: Path,
    expected_code_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    path = selection_manifest_path.expanduser().resolve()
    manifest = _read_json(path)
    prespec = _load_prespec(prespec_path)
    if manifest.get("kind") != "phase2_job3_label_free_selection_manifest":
        raise ValueError("invalid selection manifest kind")
    if manifest.get("status") != "pass":
        raise ValueError("selection manifest did not pass preflight")
    if tuple(manifest.get("arms", [])) != ARM_ORDER:
        raise ValueError("selection manifest arm drift")
    if manifest.get("prompt_serialization_id") != SERIALIZATION_ID:
        raise ValueError("selection manifest serialization drift")
    if manifest.get("model_plan_sha256") != _sha256_file(model_plan_path):
        raise ValueError("selection manifest model-plan hash mismatch")
    if manifest.get("prespec_sha256") != _sha256_file(prespec_path):
        raise ValueError("selection manifest prespec hash mismatch")
    for key in ("construction_audit_sha256", "evaluator_index_sha256"):
        if manifest.get(key) != prespec["data_binding"].get(key):
            raise ValueError(f"selection manifest evaluator binding mismatch for {key}")
    expected_code_snapshot_sha256 = expected_code_snapshot_sha256 or _code_snapshot_sha256()
    if manifest.get("code_snapshot_sha256") != expected_code_snapshot_sha256:
        raise ValueError("selection manifest code hash mismatch")
    worker_manifest_path = data_dir / "manifest.json"
    worker_manifest = _read_json(worker_manifest_path)
    if worker_manifest.get("worker_data_role") == "public_method_facing_only":
        if worker_manifest.get("source_manifest_sha256") != manifest.get("data_manifest_sha256"):
            raise ValueError("public worker-stage manifest source hash mismatch")
    elif manifest.get("data_manifest_sha256") != _sha256_file(worker_manifest_path):
        raise ValueError("selection manifest data hash mismatch")
    pack_hashes = manifest.get("pack_sha256_by_arm")
    if not isinstance(pack_hashes, dict):
        raise ValueError("selection manifest missing pack hashes")
    for arm in ARM_ORDER:
        path = _pack_path(data_dir, arm)
        _assert_public_path(path, data_dir)
        if pack_hashes.get(arm) != _sha256_file(path):
            raise ValueError(f"public pack hash mismatch for {arm}")
    rows = manifest.get("selection")
    if not isinstance(rows, list) or manifest.get("selection_count") != len(rows):
        raise ValueError("selection manifest count mismatch")
    if _contains_forbidden_label_key(manifest):
        raise ValueError("selection manifest contains evaluator label fields")
    item_ids = [str(row.get("item_id") or "") for row in rows if isinstance(row, dict)]
    if len(item_ids) != len(rows) or not all(item_ids) or len(item_ids) != len(set(item_ids)):
        raise ValueError("selection manifest item IDs are missing or duplicated")
    return manifest


def _worker_assignment(worker_index: int) -> tuple[str, int]:
    if worker_index not in WORKER_TO_MODEL:
        raise ValueError("worker-index must be one of 0, 1, 2, 3")
    return WORKER_TO_MODEL[worker_index], WORKER_TO_SHARD[worker_index]


def _public_request_for_selection(
    *,
    data_dir: Path,
    arm: str,
    selection: dict[str, Any],
    pack_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_id = str(selection["request_id_by_arm"].get(arm) or "")
    request = pack_map.get(request_id)
    if request is None:
        raise ValueError(f"public request missing for {arm}:{request_id}")
    if str(request.get("item_id") or "") != str(selection.get("item_id") or ""):
        raise ValueError("public request item join mismatch")
    observed = _request_identity(request, data_dir=data_dir)
    expected = selection["public_identity_by_arm"].get(arm)
    if observed != expected:
        raise ValueError(f"public request identity drift for {arm}:{request_id}")
    return request, observed


def _universal_user_messages(
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


def _prepare_universal_base(
    *, runtime: Any, processor: Any, system_prompt: str, user_prompt: str, image_paths: Sequence[Path], device: Any
) -> tuple[Any, dict[str, Any]]:
    gates = _gate_runtime()
    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError("processor does not provide a chat template")
    messages = _universal_user_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=image_paths,
    )
    inputs = runtime._apply_chat_template_direct(
        processor,
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if gates._input_get(inputs, "input_ids") is None:
        raise RuntimeError("universal prompt serialization produced no input_ids")
    inputs = runtime._to_device(inputs, device)
    return inputs, {
        "status": "pass",
        "serialization_id": SERIALIZATION_ID,
        "content_order": ["system_prompt_text", "images", "user_prompt_text"],
        "system_prompt_sha256": _sha256_text(system_prompt),
        "user_prompt_sha256": _sha256_text(user_prompt),
        "image_count": len(image_paths),
    }


def _prepare_candidate_with_policy(
    *,
    runtime: Any,
    processor: Any,
    device: Any,
    model_key: str,
    base_messages: Sequence[dict[str, Any]],
    base_inputs: Any,
    tokenizer: Any,
    candidate_text: str,
) -> tuple[Any, list[int], list[int], dict[str, Any], str, dict[str, Any]]:
    job2c = _job2c_runtime()
    gates = _gate_runtime()
    base_ids = job2c._input_ids_list(base_inputs)
    candidate_ids = job2c._encode_candidate_text(tokenizer, candidate_text)
    if not base_ids or not candidate_ids:
        raise RuntimeError("candidate policy requires nonempty base and standalone candidate tokens")
    attempts: list[dict[str, Any]] = []
    accepted: tuple[Any, list[int], dict[str, Any], dict[str, Any]] | None = None
    candidate_messages = [*base_messages, {"role": "assistant", "content": candidate_text}]
    for continue_final in (True, False):
        attempt: dict[str, Any] = {
            "name": f"native_assistant_continue_{str(continue_final).lower()}",
            "candidate_construction": CANDIDATE_CONSTRUCTION_NATIVE,
            "continue_final_message": continue_final,
        }
        try:
            inputs = runtime._apply_chat_template_direct(
                processor,
                candidate_messages,
                add_generation_prompt=False,
                continue_final_message=continue_final,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if gates._input_get(inputs, "input_ids") is None:
                attempts.append({**attempt, "status": "skip", "reason": "missing_input_ids"})
                continue
            inputs = runtime._to_device(inputs, device)
            full_ids = job2c._input_ids_list(inputs)
            prefix_ok = full_ids[: len(base_ids)] == base_ids
            suffix_ok = prefix_ok and full_ids[len(base_ids) :] == candidate_ids
            visual_audit = job2c._visual_identity_audit(base_inputs, inputs)
            audit = {
                **attempt,
                "status": "prepared",
                "input_ids_present": bool(full_ids),
                "prefix_check": prefix_ok,
                "exact_candidate_suffix": suffix_ok,
                "visual_exact_identity": visual_audit.get("status") == "pass",
                "base_input_length": len(base_ids),
                "full_input_length": len(full_ids),
            }
            attempts.append(audit)
            if (
                accepted is None
                and prefix_ok
                and suffix_ok
                and visual_audit.get("status") == "pass"
            ):
                accepted = (inputs, full_ids, visual_audit, audit)
        except Exception as exc:
            attempts.append(
                {**attempt, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}
            )

    if accepted is not None:
        inputs, full_ids, visual_audit, selected = accepted
        trace = {
            "policy": "native_preference_with_gemma_prefix_rewrite_fallback_only",
            "attempts": attempts,
            "selected_attempt": selected,
            "selected_construction": CANDIDATE_CONSTRUCTION_NATIVE,
        }
        return (
            inputs,
            full_ids,
            candidate_ids,
            visual_audit,
            CANDIDATE_CONSTRUCTION_NATIVE,
            trace,
        )

    audited_prefix_rewrite = _matches_gemma_native_prefix_rewrite_signature(attempts)
    if model_key != "gemma4_31b_hf_bf16" or not audited_prefix_rewrite:
        raise RuntimeError(
            f"native candidate policy failed for {model_key}; exact append is not permitted"
        )

    full_inputs, append_trace = job2c._append_exact_candidate_continuation(
        base_inputs, candidate_ids
    )
    full_ids = job2c._input_ids_list(full_inputs)
    visual_audit = job2c._visual_identity_audit(base_inputs, full_inputs)
    prefix_ok = full_ids[: len(base_ids)] == base_ids
    suffix_ok = prefix_ok and full_ids[len(base_ids) :] == candidate_ids
    fallback_attempt = {
        "name": CANDIDATE_CONSTRUCTION_EXACT_APPEND,
        "status": "prepared",
        "candidate_construction": CANDIDATE_CONSTRUCTION_EXACT_APPEND,
        "activation_reason": "both_gemma_native_variants_show_audited_prefix_rewrite",
        "audited_rewrite_signature_id": GEMMA_NATIVE_PREFIX_REWRITE_SIGNATURE["id"],
        "prefix_check": prefix_ok,
        "exact_candidate_suffix": suffix_ok,
        "visual_exact_identity": visual_audit.get("status") == "pass",
        "append_trace": append_trace,
    }
    attempts.append(fallback_attempt)
    if not prefix_ok or not suffix_ok or visual_audit.get("status") != "pass":
        raise RuntimeError("Gemma exact-append fallback failed prefix, suffix, or visual gate")
    trace = {
        "policy": "native_preference_with_gemma_prefix_rewrite_fallback_only",
        "attempts": attempts,
        "selected_attempt": fallback_attempt,
        "selected_construction": CANDIDATE_CONSTRUCTION_EXACT_APPEND,
        "audited_rewrite_signature_id": GEMMA_NATIVE_PREFIX_REWRITE_SIGNATURE["id"],
    }
    return (
        full_inputs,
        full_ids,
        candidate_ids,
        visual_audit,
        CANDIDATE_CONSTRUCTION_EXACT_APPEND,
        trace,
    )


def _score_candidate_with_policy(
    *,
    runtime: Any,
    model: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    model_key: str,
    base_messages: Sequence[dict[str, Any]],
    base_inputs: Any,
    candidate_label: str,
    candidate_text: str,
) -> dict[str, Any]:
    import torch

    job2c = _job2c_runtime()
    base_ids = job2c._input_ids_list(base_inputs)
    (
        full_inputs,
        full_ids,
        candidate_ids,
        visual_audit,
        construction,
        policy_trace,
    ) = _prepare_candidate_with_policy(
        runtime=runtime,
        processor=processor,
        device=device,
        model_key=model_key,
        base_messages=base_messages,
        base_inputs=base_inputs,
        tokenizer=tokenizer,
        candidate_text=candidate_text,
    )
    with torch.inference_mode():
        output = model(**full_inputs)
    scored = job2c.score_candidate_suffix_from_logits(
        base_input_ids=base_ids,
        full_input_ids=full_ids,
        logits=job2c._extract_logits(output),
    )
    if scored.get("status") != "pass":
        raise RuntimeError(f"candidate score failed: {scored.get('reason')}")
    return {
        "status": "pass",
        "candidate_label": candidate_label,
        "candidate_text": candidate_text,
        "candidate_construction": construction,
        "prefix_check": True,
        "exact_candidate_suffix": True,
        "visual_exact_identity": True,
        "visual_identity_audit": visual_audit,
        "token_count": scored["token_count"],
        "scored_token_ids": scored["scored_token_ids"],
        "sum_logprob": scored["sum_logprob"],
        "mean_logprob": scored["mean_logprob"],
        "candidate_policy_trace": policy_trace,
    }


def _score_public_request(
    *,
    runtime: Any,
    model: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    request: dict[str, Any],
    data_dir: Path,
    prespec: dict[str, Any],
    model_key: str,
) -> dict[str, Any]:
    job2c = _job2c_runtime()
    system_prompt = str(request["system_prompt"])
    user_prompt = str(request["user_prompt"])
    image_paths = [_resolve_input_path(data_dir, str(row["path"])) for row in request["inputs"]]
    base_inputs, serialization_trace = _prepare_universal_base(
        runtime=runtime,
        processor=processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=image_paths,
        device=device,
    )
    base_messages = _universal_user_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=image_paths,
    )
    candidates = {
        label: _score_candidate_with_policy(
            runtime=runtime,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            model_key=model_key,
            base_messages=base_messages,
            base_inputs=base_inputs,
            candidate_label=label,
            candidate_text=text,
        )
        for label, text in FROZEN_CANDIDATE_COMPLETIONS.items()
    }
    greedy = job2c._run_greedy(
        model=model,
        runtime=runtime,
        processor=processor,
        base_inputs=base_inputs,
        max_new_tokens=int(prespec["generation"]["max_new_tokens"]),
    )
    parsed = greedy.get("parsed") if isinstance(greedy, dict) else None
    if not isinstance(parsed, dict) or type(parsed.get("answer")) is not bool or greedy.get("parse_error"):
        raise RuntimeError("greedy output did not parse to a Boolean answer")
    return {
        "serialization": serialization_trace,
        "teacher_forced": {"status": "pass", "candidate_rows": candidates},
        "greedy": greedy,
    }


def _load_model_bundle(
    *, model_row: dict[str, str], dtype: str, local_files_only: bool
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    gates = _gate_runtime()
    runtime = gates._load_job0_runtime(REPO_ROOT)
    gpu_snapshot = runtime._assert_single_visible_gpu()
    snapshot = gates._snapshot_provenance(
        model_row["model_id"], model_row["revision"], local_files_only=local_files_only
    )
    model, processor, device, adapter, tokenizer = runtime._load_model(
        model_id=model_row["model_id"],
        dtype_name=dtype,
        trust_remote_code=True,
        local_files_only=local_files_only,
        revision=model_row["revision"],
        force_single_gpu_device_map=True,
    )
    if tokenizer is None:
        tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("model runtime did not provide a tokenizer")
    provenance = {
        "gpu_snapshot": gpu_snapshot,
        "model_snapshot": snapshot,
        "adapter": adapter,
        "processor_class": type(processor).__name__,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "runtime_versions": gates._runtime_versions(),
    }
    return runtime, model, processor, tokenizer, device, provenance


def _row_hash(row: dict[str, Any]) -> str:
    return _canonical_sha256({key: value for key, value in row.items() if key != "row_sha256"})


def _matches_gemma_native_prefix_rewrite_signature(
    attempts: Sequence[dict[str, Any]],
) -> bool:
    expected_attempts = GEMMA_NATIVE_PREFIX_REWRITE_SIGNATURE["attempts"]
    if len(attempts) != len(expected_attempts):
        return False
    for observed, expected in zip(attempts, expected_attempts, strict=True):
        base_length = observed.get("base_input_length")
        full_length = observed.get("full_input_length")
        if not isinstance(base_length, int) or not isinstance(full_length, int):
            return False
        if (
            observed.get("candidate_construction")
            != CANDIDATE_CONSTRUCTION_NATIVE
            or observed.get("status") != "prepared"
            or observed.get("input_ids_present") is not True
            or observed.get("continue_final_message")
            is not expected["continue_final_message"]
            or full_length - base_length != expected["full_minus_base_tokens"]
            or observed.get("prefix_check") is not expected["prefix_check"]
            or observed.get("exact_candidate_suffix")
            is not expected["exact_candidate_suffix"]
            or observed.get("visual_exact_identity")
            is not expected["visual_exact_identity"]
        ):
            return False
    return True


def _candidate_policy_trace_valid(candidate: dict[str, Any], model_key: str) -> bool:
    construction = candidate.get("candidate_construction")
    trace = candidate.get("candidate_policy_trace")
    if not isinstance(trace, dict) or trace.get("selected_construction") != construction:
        return False
    attempts = trace.get("attempts")
    selected = trace.get("selected_attempt")
    if not isinstance(attempts, list) or not isinstance(selected, dict):
        return False
    if construction == CANDIDATE_CONSTRUCTION_NATIVE:
        return bool(
            len(attempts) == 2
            and selected.get("candidate_construction") == construction
            and selected.get("status") == "prepared"
            and selected.get("prefix_check") is True
            and selected.get("exact_candidate_suffix") is True
            and selected.get("visual_exact_identity") is True
        )
    if construction != CANDIDATE_CONSTRUCTION_EXACT_APPEND:
        return False
    if model_key != "gemma4_31b_hf_bf16" or len(attempts) != 3:
        return False
    native_attempts = attempts[:2]
    fallback = attempts[2]
    return bool(
        _matches_gemma_native_prefix_rewrite_signature(native_attempts)
        and trace.get("audited_rewrite_signature_id")
        == GEMMA_NATIVE_PREFIX_REWRITE_SIGNATURE["id"]
        and selected == fallback
        and fallback.get("candidate_construction") == construction
        and fallback.get("status") == "prepared"
        and fallback.get("activation_reason")
        == "both_gemma_native_variants_show_audited_prefix_rewrite"
        and fallback.get("audited_rewrite_signature_id")
        == GEMMA_NATIVE_PREFIX_REWRITE_SIGNATURE["id"]
        and fallback.get("prefix_check") is True
        and fallback.get("exact_candidate_suffix") is True
        and fallback.get("visual_exact_identity") is True
    )


def _validate_completed_row(
    row: dict[str, Any], *, expected: dict[str, Any]
) -> list[str]:
    issues: list[str] = []
    if row.get("kind") != "phase2_job3_worker_row":
        issues.append("row_kind")
    for key, value in expected.items():
        if row.get(key) != value:
            issues.append(f"{key}_mismatch")
    if row.get("status") != "pass":
        issues.append("row_status")
    if row.get("row_sha256") != _row_hash(row):
        issues.append("row_hash")
    serialization = row.get("serialization")
    if not (
        isinstance(serialization, dict)
        and serialization.get("status") == "pass"
        and serialization.get("serialization_id") == SERIALIZATION_ID
    ):
        issues.append("serialization")
    teacher = row.get("teacher_forced")
    candidates = teacher.get("candidate_rows") if isinstance(teacher, dict) else None
    if not (
        isinstance(teacher, dict)
        and teacher.get("status") == "pass"
        and isinstance(candidates, dict)
        and set(candidates) == set(FROZEN_CANDIDATE_COMPLETIONS)
    ):
        issues.append("teacher_forced_candidates")
    else:
        for label, expected_text in FROZEN_CANDIDATE_COMPLETIONS.items():
            candidate = candidates[label]
            finite = all(
                isinstance(candidate.get(key), (int, float)) and math.isfinite(float(candidate[key]))
                for key in ("sum_logprob", "mean_logprob")
            )
            if not (
                candidate.get("status") == "pass"
                and candidate.get("candidate_text") == expected_text
                and candidate.get("prefix_check") is True
                and candidate.get("exact_candidate_suffix") is True
                and candidate.get("visual_exact_identity") is True
                and candidate.get("candidate_label") == label
                and _candidate_policy_trace_valid(candidate, str(row.get("model_key") or ""))
                and int(candidate.get("token_count") or 0) > 0
                and finite
            ):
                issues.append(f"candidate_{label}")
    greedy = row.get("greedy")
    parsed = greedy.get("parsed") if isinstance(greedy, dict) else None
    confidence = parsed.get("confidence") if isinstance(parsed, dict) else None
    if not (
        isinstance(greedy, dict)
        and greedy.get("status") == "pass"
        and isinstance(parsed, dict)
        and type(parsed.get("answer")) is bool
        and parsed.get("abstain") is False
        and parsed.get("abstention_reason") is None
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(float(confidence))
        and 0.0 <= float(confidence) <= 1.0
        and not greedy.get("parse_error")
    ):
        issues.append("greedy_parse")
    return issues


def run_worker(
    *,
    data_dir: Path,
    run_dir: Path,
    selection_manifest_path: Path,
    model_plan_path: Path,
    prespec_path: Path,
    worker_index: int,
    dtype: str,
    local_files_only: bool,
    resume: bool,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    model_plan_path = model_plan_path.expanduser().resolve()
    prespec_path = prespec_path.expanduser().resolve()
    model_plan = _load_model_plan(model_plan_path)
    prespec = _load_prespec(prespec_path)
    if dtype != "bf16":
        raise ValueError("Job3 workers require --dtype=bf16")
    selection = _load_selection_manifest(
        selection_manifest_path=selection_manifest_path,
        data_dir=data_dir,
        model_plan_path=model_plan_path,
        prespec_path=prespec_path,
    )
    model_key, shard_index = _worker_assignment(int(worker_index))
    model_row = next(row for row in model_plan if row["model_key"] == model_key)
    if model_row["dtype"] != dtype or int(worker_index) not in model_row["worker_indexes"]:
        raise ValueError("worker dtype or model-plan worker assignment drift")
    assigned = [
        row for row in selection["selection"] if int(row["selection_index"]) % 2 == shard_index
    ]

    pack_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARM_ORDER:
        payload, _ = _load_pack(data_dir, arm)
        pack_maps[arm] = {
            str(row.get("request_id") or ""): row
            for row in payload.get("requests", [])
            if isinstance(row, dict) and row.get("request_id")
        }
    for item in assigned:
        for arm in ARM_ORDER:
            _public_request_for_selection(
                data_dir=data_dir, arm=arm, selection=item, pack_map=pack_maps[arm]
            )

    worker_dir = run_dir / "workers" / f"worker_{int(worker_index)}"
    rows_path = worker_dir / "rows.jsonl"
    summary_path = worker_dir / "summary.json"
    progress_path = worker_dir / "progress.json"
    worker_dir.mkdir(parents=True, exist_ok=True)

    expected_common = {
        "worker_index": int(worker_index),
        "model_key": model_key,
        "model_id": model_row["model_id"],
        "revision": model_row["revision"],
        "dtype": dtype,
        "selection_manifest_sha256": _sha256_file(selection_manifest_path),
        "model_plan_sha256": _sha256_file(model_plan_path),
        "prespec_sha256": _sha256_file(prespec_path),
        "code_snapshot_sha256": _code_snapshot_sha256(),
    }
    existing_keys: set[tuple[str, str]] = set()
    if rows_path.exists() and rows_path.stat().st_size:
        if not resume:
            raise ValueError("worker output exists; use --resume or a fresh run directory")
        for row in _iter_jsonl(rows_path):
            issues = _validate_completed_row(row, expected=expected_common)
            if issues:
                raise ValueError(f"resume row failed validation: {issues}")
            key = (str(row.get("item_id") or ""), str(row.get("arm") or ""))
            if key in existing_keys:
                raise ValueError(f"duplicate resume key: {key}")
            existing_keys.add(key)

    summary = {
        "kind": "phase2_job3_worker_summary",
        "status": "loading",
        **expected_common,
        "shard_index": shard_index,
        "shard_count": 2,
        "assigned_item_count": len(assigned),
        "expected_row_count": len(assigned) * len(ARM_ORDER),
        "dtype": dtype,
        "local_files_only": bool(local_files_only),
        "resume": bool(resume),
        "started_at": _utc_now_iso(),
    }
    _write_json(summary_path, summary)
    _write_json(progress_path, {"status": "loading", "rows_complete": len(existing_keys), "updated_at": _utc_now_iso()})

    load_start = time.perf_counter()
    runtime, model, processor, tokenizer, device, runtime_provenance = _load_model_bundle(
        model_row=model_row, dtype=dtype, local_files_only=bool(local_files_only)
    )
    summary["load_elapsed_s"] = time.perf_counter() - load_start
    summary["runtime_provenance"] = runtime_provenance
    summary["status"] = "running"
    _write_json(summary_path, summary)

    new_rows = 0
    skipped_rows = 0
    started = time.perf_counter()
    for item in assigned:
        for arm in ARM_ORDER:
            key = (str(item["item_id"]), arm)
            if key in existing_keys:
                skipped_rows += 1
                continue
            request, identity = _public_request_for_selection(
                data_dir=data_dir, arm=arm, selection=item, pack_map=pack_maps[arm]
            )
            row_start = time.perf_counter()
            result = _score_public_request(
                runtime=runtime,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                device=device,
                request=request,
                data_dir=data_dir,
                prespec=prespec,
                model_key=model_key,
            )
            row = {
                "kind": "phase2_job3_worker_row",
                "status": "pass",
                **expected_common,
                "shard_index": shard_index,
                "selection_index": int(item["selection_index"]),
                "item_id": str(item["item_id"]),
                "arm": arm,
                "request_id": str(request["request_id"]),
                "system_prompt_sha256": identity["system_prompt_sha256"],
                "user_prompt_sha256": identity["user_prompt_sha256"],
                "ordered_input_sha256": [entry["sha256"] for entry in identity["ordered_inputs"]],
                "serialization_sha256": identity["serialization_sha256"],
                "serialization": result["serialization"],
                "teacher_forced": result["teacher_forced"],
                "greedy": result["greedy"],
                "elapsed_s": time.perf_counter() - row_start,
            }
            row["row_sha256"] = _row_hash(row)
            _append_jsonl_atomic(rows_path, row)
            existing_keys.add(key)
            new_rows += 1
            _write_json(
                progress_path,
                {
                    "status": "running",
                    "rows_complete": len(existing_keys),
                    "rows_expected": summary["expected_row_count"],
                    "new_rows": new_rows,
                    "skipped_rows": skipped_rows,
                    "elapsed_s": time.perf_counter() - started,
                    "updated_at": _utc_now_iso(),
                },
            )

    if len(existing_keys) != summary["expected_row_count"]:
        raise RuntimeError("worker completed with an incomplete row union")
    summary.update(
        {
            "status": "pass",
            "rows_complete": len(existing_keys),
            "rows_new": new_rows,
            "rows_skipped": skipped_rows,
            "rows_sha256": _sha256_file(rows_path),
            "elapsed_s": time.perf_counter() - started,
            "finished_at": _utc_now_iso(),
        }
    )
    _write_json(summary_path, summary)
    _write_json(progress_path, {"status": "pass", "rows_complete": len(existing_keys), "updated_at": _utc_now_iso()})
    return summary


def _runtime_provenance_issues(
    provenance: Any, *, model_row: dict[str, Any], worker_index: int
) -> list[str]:
    if not isinstance(provenance, dict):
        return ["runtime_provenance_missing"]
    expected = EXPECTED_WORKER_RUNTIME_BY_MODEL[model_row["model_key"]]
    issues: list[str] = []
    for key in ("adapter", "model_class", "processor_class", "tokenizer_class"):
        if provenance.get(key) != expected[key]:
            issues.append(key)
    if provenance.get("runtime_versions") != EXPECTED_WORKER_RUNTIME_VERSIONS:
        issues.append("runtime_versions")

    snapshot = provenance.get("model_snapshot")
    if not isinstance(snapshot, dict):
        issues.append("model_snapshot")
    else:
        snapshot_path = str(snapshot.get("snapshot_path") or "")
        if not (
            snapshot.get("status") == "pass"
            and snapshot.get("requested_revision") == model_row["revision"]
            and snapshot.get("resolved_snapshot_revision") == model_row["revision"]
            and snapshot_path.endswith(str(expected["snapshot_suffix"]))
        ):
            issues.append("model_snapshot")

    gpu = provenance.get("gpu_snapshot")
    if not isinstance(gpu, dict):
        issues.append("gpu_snapshot")
    elif not (
        gpu.get("cuda_available") is True
        and gpu.get("cuda_device_count") == 1
        and gpu.get("cuda_visible_devices") == str(worker_index)
        and gpu.get("cuda_device_names") == [EXPECTED_WORKER_GPU_NAME]
    ):
        issues.append("gpu_snapshot")
    return issues


def _technical_validate(
    *,
    data_dir: Path,
    run_dir: Path,
    selection_manifest_path: Path,
    model_plan_path: Path,
    prespec_path: Path,
    mode: str,
    worker_code_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    worker_code_snapshot_sha256 = worker_code_snapshot_sha256 or _code_snapshot_sha256()
    model_plan = _load_model_plan(model_plan_path)
    _load_prespec(prespec_path)
    selection = _load_selection_manifest(
        selection_manifest_path=selection_manifest_path,
        data_dir=data_dir,
        model_plan_path=model_plan_path,
        prespec_path=prespec_path,
        expected_code_snapshot_sha256=worker_code_snapshot_sha256,
    )
    if selection.get("mode") != mode:
        raise ValueError("aggregate mode differs from selection manifest mode")

    expected: dict[tuple[str, str, str], int] = {}
    model_by_worker: dict[int, dict[str, Any]] = {}
    for worker in range(4):
        model_key, shard = _worker_assignment(worker)
        model_by_worker[worker] = next(row for row in model_plan if row["model_key"] == model_key)
        for item in selection["selection"]:
            if int(item["selection_index"]) % 2 == shard:
                for arm in ARM_ORDER:
                    expected[(model_key, str(item["item_id"]), arm)] = worker

    load_failures: list[dict[str, Any]] = []
    row_failures: list[dict[str, Any]] = []
    teacher_failures: list[dict[str, Any]] = []
    token_failures: list[dict[str, Any]] = []
    greedy_failures: list[dict[str, Any]] = []
    duplicate_keys: list[dict[str, Any]] = []
    unexpected_keys: list[dict[str, Any]] = []
    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    selection_sha = _sha256_file(selection_manifest_path)
    selection_by_item = {
        str(row["item_id"]): row for row in selection["selection"] if isinstance(row, dict)
    }
    for worker in range(4):
        worker_dir = run_dir / "workers" / f"worker_{worker}"
        summary_path = worker_dir / "summary.json"
        rows_path = worker_dir / "rows.jsonl"
        if not summary_path.exists() or not rows_path.exists():
            load_failures.append({"worker_index": worker, "reason": "missing_worker_outputs"})
            continue
        summary = _read_json(summary_path)
        model_row = model_by_worker[worker]
        shard_index = WORKER_TO_SHARD[worker]
        assigned_item_count = sum(
            int(item["selection_index"]) % 2 == shard_index
            for item in selection["selection"]
        )
        expected_row_count = assigned_item_count * len(ARM_ORDER)
        runtime_provenance = summary.get("runtime_provenance")
        provenance_issues = _runtime_provenance_issues(
            runtime_provenance,
            model_row=model_row,
            worker_index=worker,
        )
        if not (
            summary.get("kind") == "phase2_job3_worker_summary"
            and summary.get("status") == "pass"
            and summary.get("worker_index") == worker
            and summary.get("model_key") == model_row["model_key"]
            and summary.get("model_id") == model_row["model_id"]
            and summary.get("revision") == model_row["revision"]
            and summary.get("dtype") == "bf16"
            and summary.get("selection_manifest_sha256") == selection_sha
            and summary.get("model_plan_sha256") == _sha256_file(model_plan_path)
            and summary.get("prespec_sha256") == _sha256_file(prespec_path)
            and summary.get("code_snapshot_sha256") == worker_code_snapshot_sha256
            and summary.get("rows_sha256") == _sha256_file(rows_path)
            and summary.get("shard_index") == shard_index
            and summary.get("shard_count") == 2
            and summary.get("assigned_item_count") == assigned_item_count
            and summary.get("expected_row_count") == expected_row_count
            and summary.get("rows_complete") == expected_row_count
            and type(summary.get("rows_new")) is int
            and type(summary.get("rows_skipped")) is int
            and summary["rows_new"] + summary["rows_skipped"]
            == expected_row_count
            and summary.get("local_files_only") is True
            and not provenance_issues
        ):
            load_failures.append(
                {
                    "worker_index": worker,
                    "reason": "summary_or_model_provenance",
                    "provenance_issues": provenance_issues,
                }
            )
        common = {
            "worker_index": worker,
            "model_key": model_row["model_key"],
            "model_id": model_row["model_id"],
            "revision": model_row["revision"],
            "dtype": "bf16",
            "selection_manifest_sha256": selection_sha,
            "model_plan_sha256": _sha256_file(model_plan_path),
            "prespec_sha256": _sha256_file(prespec_path),
            "code_snapshot_sha256": worker_code_snapshot_sha256,
        }
        for row in _iter_jsonl(rows_path):
            key = (str(row.get("model_key") or ""), str(row.get("item_id") or ""), str(row.get("arm") or ""))
            if key in observed:
                duplicate_keys.append({"key": list(key), "worker_index": worker})
                continue
            observed[key] = row
            if expected.get(key) != worker:
                unexpected_keys.append({"key": list(key), "worker_index": worker})
            selected = selection_by_item.get(key[1])
            row_expected = dict(common)
            row_expected["shard_index"] = shard_index
            if isinstance(selected, dict):
                row_expected["selection_index"] = selected.get("selection_index")
            issues = _validate_completed_row(row, expected=row_expected)
            identity = (
                selected.get("public_identity_by_arm", {}).get(key[2])
                if isinstance(selected, dict)
                else None
            )
            if not isinstance(identity, dict):
                issues.append("selection_identity_missing")
            else:
                identity_checks = {
                    "request_id": identity.get("request_id"),
                    "system_prompt_sha256": identity.get("system_prompt_sha256"),
                    "user_prompt_sha256": identity.get("user_prompt_sha256"),
                    "ordered_input_sha256": [
                        entry.get("sha256") for entry in identity.get("ordered_inputs", [])
                    ],
                    "serialization_sha256": identity.get("serialization_sha256"),
                }
                for field, expected_value in identity_checks.items():
                    if row.get(field) != expected_value:
                        issues.append(f"{field}_mismatch")
            if issues:
                row_failures.append({"key": list(key), "worker_index": worker, "issues": issues})
                if any(issue.startswith("candidate_") for issue in issues):
                    token_failures.append({"key": list(key), "issues": issues})
                    teacher_failures.append({"key": list(key), "issues": issues})
                elif "teacher_forced_candidates" in issues:
                    teacher_failures.append({"key": list(key), "issues": issues})
                if "greedy_parse" in issues:
                    greedy_failures.append({"key": list(key)})
    missing_keys = [list(key) for key in sorted(set(expected) - set(observed))]
    gates_out = [
        _gate("worker_load_and_revision", not load_failures, load_failures[:20]),
        _gate("worker_union_complete", not missing_keys and not unexpected_keys, {"missing": missing_keys[:20], "unexpected": unexpected_keys[:20]}),
        _gate("worker_union_unique", not duplicate_keys, duplicate_keys[:20]),
        _gate("worker_rows_valid", not row_failures, row_failures[:20]),
        _gate("teacher_forced_scoring", not teacher_failures, teacher_failures[:20]),
        _gate("candidate_token_boundaries", not token_failures, token_failures[:20]),
        _gate("greedy_parsing", not greedy_failures, greedy_failures[:20]),
    ]
    return {
        "kind": "phase2_job3_label_free_technical_summary",
        "status": "pass" if all(g["status"] == "pass" for g in gates_out) else "fail",
        "mode": mode,
        "worker_code_snapshot_sha256": worker_code_snapshot_sha256,
        "evaluator_code_snapshot_sha256": _code_snapshot_sha256(),
        "selection_count": selection["selection_count"],
        "expected_worker_row_count": len(expected),
        "observed_worker_row_count": len(observed),
        "load_failure_count": len(load_failures),
        "worker_row_failure_count": len(row_failures),
        "teacher_forced_failure_count": len(teacher_failures),
        "token_boundary_failure_count": len(token_failures),
        "greedy_parse_failure_count": len(greedy_failures),
        "missing_union_count": len(missing_keys),
        "duplicate_union_count": len(duplicate_keys),
        "unexpected_union_count": len(unexpected_keys),
        "gates": gates_out,
        "rows": observed,
    }


def one_sided_wilson_lower_bound(successes: int, total: int, z: float = ONE_SIDED_95_Z) -> float:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid Wilson count")
    proportion = successes / total
    z2 = z * z
    center = proportion + z2 / (2 * total)
    radius = z * math.sqrt((proportion * (1 - proportion) + z2 / (4 * total)) / total)
    return (center - radius) / (1 + z2 / total)


def _hidden_label_map(
    data_dir: Path, *, evaluator_payload: Any | None = None
) -> dict[str, dict[str, Any]]:
    path = (data_dir / HIDDEN_RELATIVE_PATH).resolve()
    if not _is_within(path, data_dir) or path.name != "evaluator_index.json":
        raise ValueError("invalid evaluator index path")
    payload = evaluator_payload
    if payload is None:
        payload, _ = _read_json_with_sha256(path)
    if not (
        payload.get("burned") is True
        and payload.get("non_confirmatory") is True
        and payload.get("rows_frozen") is False
        and payload.get("answer_count") == EXPECTED_ITEM_COUNT
    ):
        raise ValueError("evaluator index burned-pilot contract failed")
    rows = payload.get("answers")
    if not isinstance(rows, list):
        raise ValueError("evaluator index answers missing")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("item_id") or "") if isinstance(row, dict) else ""
        if not item_id or item_id in out:
            raise ValueError("evaluator index item IDs missing or duplicated")
        if row.get("scoring_eligible") is not True:
            raise ValueError("evaluator row is not scoring eligible")
        canonical_text = str(row.get("canonical_answer_json") or "")
        if canonical_text not in set(FROZEN_CANDIDATE_COMPLETIONS.values()):
            raise ValueError("evaluator row canonical answer is not a frozen completion")
        try:
            canonical = json.loads(canonical_text)
        except json.JSONDecodeError as exc:
            raise ValueError("evaluator row lacks canonical binary answer JSON") from exc
        if not (
            isinstance(canonical, dict)
            and type(canonical.get("answer")) is bool
            and canonical.get("abstain") is False
            and canonical.get("abstention_reason") is None
        ):
            raise ValueError("evaluator row canonical answer is not a scored Boolean")
        expected_answer = bool(canonical["answer"])
        expected_role = (
            "positive" if expected_answer else "negative_human_audited_no_direct_contact"
        )
        if row.get("label_role") != expected_role:
            raise ValueError("evaluator role disagrees with canonical binary answer")
        normalized = dict(row)
        normalized["binary_answer"] = expected_answer
        normalized["binary_class"] = "positive" if expected_answer else "negative"
        out[item_id] = normalized
    return out


def _validate_evaluator_side_bindings(
    *, data_dir: Path, prespec: dict[str, Any], selection: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    expected = prespec["data_binding"]
    paths = {
        "construction_audit_sha256": data_dir / "audits/construction_audit.json",
        "evaluator_index_sha256": data_dir / HIDDEN_RELATIVE_PATH,
    }
    observed: dict[str, str] = {}
    evaluator_payload: dict[str, Any] | None = None
    for key, path in paths.items():
        resolved = path.resolve()
        if not _is_within(resolved, data_dir) or not resolved.is_file():
            raise ValueError(f"evaluator binding path missing or escaped: {path}")
        if key == "evaluator_index_sha256":
            loaded, digest = _read_json_with_sha256(resolved)
            if not isinstance(loaded, dict):
                raise ValueError("evaluator index must be a JSON object")
            evaluator_payload = loaded
        else:
            digest = _sha256_file(resolved)
        if digest != expected.get(key) or digest != selection.get(key):
            raise ValueError(f"evaluator-side hash binding mismatch for {key}")
        observed[key] = digest
    if evaluator_payload is None:
        raise ValueError("evaluator index was not loaded from its hash-bound bytes")
    return observed, evaluator_payload


def _validate_compatible_source_artifacts(
    *,
    run_dir: Path,
    selection_manifest_path: Path,
    compatibility: dict[str, Any] | None,
) -> None:
    if compatibility is None:
        return
    checks = {
        "source_job_manifest": (
            run_dir / "source_job_manifest.json",
            compatibility.get("source_job_manifest_sha256"),
        ),
        "selection_manifest": (
            selection_manifest_path,
            compatibility.get("selection_manifest_sha256"),
        ),
    }
    worker_hashes = compatibility.get("worker_artifact_sha256")
    if not isinstance(worker_hashes, dict) or set(worker_hashes) != {"0", "1", "2", "3"}:
        raise ValueError("compatible source worker artifact allowlist is malformed")
    for worker in range(4):
        expected = worker_hashes[str(worker)]
        checks[f"worker_{worker}_rows"] = (
            run_dir / "workers" / f"worker_{worker}" / "rows.jsonl",
            expected.get("rows"),
        )
        checks[f"worker_{worker}_summary"] = (
            run_dir / "workers" / f"worker_{worker}" / "summary.json",
            expected.get("summary"),
        )
    for name, (path, expected_hash) in checks.items():
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"compatible source hash pin is malformed for {name}")
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ValueError(f"compatible source artifact mismatch for {name}")


def _resolve_worker_snapshot_compatibility(
    *, mode: str, compatible_worker_code_snapshot_sha256: str | None
) -> tuple[str, dict[str, Any] | None]:
    current = _code_snapshot_sha256()
    if not compatible_worker_code_snapshot_sha256:
        return current, None
    if mode != "pilot":
        raise ValueError("worker snapshot compatibility is pilot-only")
    compatibility = APPROVED_PILOT_WORKER_SNAPSHOT_COMPATIBILITY.get(
        compatible_worker_code_snapshot_sha256
    )
    if compatibility is None:
        raise ValueError("worker code snapshot is not in the exact compatibility allowlist")
    return compatible_worker_code_snapshot_sha256, dict(compatibility)


def _pilot_metrics(
    *,
    data_dir: Path,
    selection: dict[str, Any],
    rows: dict[tuple[str, str, str], dict[str, Any]],
    evaluator_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    hidden = _hidden_label_map(data_dir, evaluator_payload=evaluator_payload)
    selected_ids = [str(row["item_id"]) for row in selection["selection"]]
    missing_labels = sorted(set(selected_ids) - set(hidden))
    extra_labels = sorted(set(hidden) - set(selected_ids))
    join_failures: list[dict[str, Any]] = []
    if missing_labels:
        join_failures.append({"reason": "missing_labels", "item_ids": missing_labels[:20]})
    if extra_labels:
        join_failures.append({"reason": "extra_labels", "count": len(extra_labels)})

    joined: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], list[bool]] = {}
    for model_key in (FROZEN_MODEL_PLAN[0][0], FROZEN_MODEL_PLAN[1][0]):
        for item_id in selected_ids:
            worker_row = rows.get((model_key, item_id, ARM_PCS))
            label = hidden.get(item_id)
            if worker_row is None or label is None:
                join_failures.append({"model_key": model_key, "item_id": item_id, "reason": "pcs_join_missing"})
                continue
            expected_request_id = str(label.get("request_ids", {}).get(ARM_PCS) or "")
            if worker_row.get("request_id") != expected_request_id:
                join_failures.append(
                    {"model_key": model_key, "item_id": item_id, "reason": "pcs_request_join_mismatch"}
                )
                continue
            parsed = worker_row["greedy"]["parsed"]
            observed_answer = parsed.get("answer")
            expected_answer = bool(label["binary_answer"])
            if type(observed_answer) is not bool:
                join_failures.append({"model_key": model_key, "item_id": item_id, "reason": "greedy_not_boolean"})
                continue
            correct = observed_answer == expected_answer
            class_name = str(label["binary_class"])
            counts.setdefault((model_key, class_name), []).append(correct)
            joined.append(
                {
                    "model_key": model_key,
                    "item_id": item_id,
                    "arm": ARM_PCS,
                    "class": class_name,
                    "greedy_answer": observed_answer,
                    "correct": correct,
                }
            )

    metric_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for model_key in (FROZEN_MODEL_PLAN[0][0], FROZEN_MODEL_PLAN[1][0]):
        for class_name in ("positive", "negative"):
            values = counts.get((model_key, class_name), [])
            successes = sum(values)
            total = len(values)
            accuracy = successes / total if total else None
            lower = one_sided_wilson_lower_bound(successes, total) if total else None
            passed = bool(
                accuracy is not None
                and lower is not None
                and accuracy >= PILOT_ACCURACY_FLOOR
                and lower > PILOT_WILSON_FLOOR
            )
            metric_rows.append(
                {
                    "model_key": model_key,
                    "class": class_name,
                    "successes": successes,
                    "total": total,
                    "greedy_class_accuracy": accuracy,
                    "one_sided_95_wilson_lower_bound": lower,
                    "accuracy_floor": PILOT_ACCURACY_FLOOR,
                    "wilson_strict_floor": PILOT_WILSON_FLOOR,
                    "status": "pass" if passed else "fail",
                }
            )
            gate_rows.append(_gate(f"pcs_e_{model_key}_{class_name}", passed, metric_rows[-1]))
    return {"rows": metric_rows, "join_failure_count": len(join_failures)}, joined, [
        _gate("hidden_label_join", not join_failures, join_failures[:20]), *gate_rows
    ]


def _strip_private_rows(technical: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in technical.items() if key != "rows"}


def _exact_key_failures(actual: dict[str, Any], expected_keys: set[str]) -> list[str]:
    actual_keys = set(actual)
    return [
        *[f"missing_key:{key}" for key in sorted(expected_keys - actual_keys)],
        *[f"unexpected_key:{key}" for key in sorted(actual_keys - expected_keys)],
    ]


def aggregate(
    *,
    data_dir: Path,
    run_dir: Path,
    selection_manifest_path: Path,
    model_plan_path: Path,
    prespec_path: Path,
    mode: str,
    compatible_worker_code_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    model_plan_path = model_plan_path.expanduser().resolve()
    prespec_path = prespec_path.expanduser().resolve()
    if mode not in {"canary", "pilot", "full"}:
        raise ValueError("mode must be canary, pilot, or full")
    worker_code_snapshot_sha256, compatibility = _resolve_worker_snapshot_compatibility(
        mode=mode,
        compatible_worker_code_snapshot_sha256=compatible_worker_code_snapshot_sha256,
    )
    _validate_compatible_source_artifacts(
        run_dir=run_dir,
        selection_manifest_path=selection_manifest_path,
        compatibility=compatibility,
    )
    prespec = _load_prespec(prespec_path)
    canary_forbidden_terms = _canary_forbidden_terms(prespec)
    technical = _technical_validate(
        data_dir=data_dir,
        run_dir=run_dir,
        selection_manifest_path=selection_manifest_path,
        model_plan_path=model_plan_path,
        prespec_path=prespec_path,
        mode=mode,
        worker_code_snapshot_sha256=worker_code_snapshot_sha256,
    )
    aggregate_dir = run_dir / "aggregate"
    public_technical = _strip_private_rows(technical)
    public_technical["generated_at"] = _utc_now_iso()
    public_technical["worker_snapshot_compatibility"] = compatibility
    if mode in {"canary", "full"} and _scan_forbidden_metric_terms(
        public_technical, canary_forbidden_terms
    ):
        raise RuntimeError("label-free technical summary crossed the metric firewall")
    _write_json(aggregate_dir / "technical_summary.json", public_technical)
    if technical["status"] != "pass":
        summary = {
            "kind": "phase2_job3_aggregate_summary",
            "status": "fail",
            "mode": mode,
            "inference_status": "withheld_due_to_technical_failure",
            "technical_summary_sha256": _sha256_file(aggregate_dir / "technical_summary.json"),
        }
        _write_json(aggregate_dir / "aggregate_summary.json", summary)
        return summary

    selection = _load_selection_manifest(
        selection_manifest_path=selection_manifest_path,
        data_dir=data_dir,
        model_plan_path=model_plan_path,
        prespec_path=prespec_path,
        expected_code_snapshot_sha256=worker_code_snapshot_sha256,
    )
    if mode in {"canary", "full"}:
        summary = {
            "kind": "phase2_job3_label_free_aggregate_summary",
            "status": "pass",
            "mode": mode,
            "label_state": "withheld",
            "technical_summary_sha256": _sha256_file(aggregate_dir / "technical_summary.json"),
            "selection_count": selection["selection_count"],
            "worker_row_count": technical["observed_worker_row_count"],
            "metric_firewall_status": "pass",
            "generated_at": _utc_now_iso(),
        }
        serialized = json.dumps(summary, sort_keys=True).lower()
        if any(term in serialized for term in canary_forbidden_terms):
            raise RuntimeError("label-free aggregate crossed the metric firewall")
        _write_json(aggregate_dir / "aggregate_summary.json", summary)
        return summary

    evaluator_bindings, evaluator_payload = _validate_evaluator_side_bindings(
        data_dir=data_dir,
        prespec=prespec,
        selection=selection,
    )
    metrics, joined_rows, pilot_gates = _pilot_metrics(
        data_dir=data_dir,
        selection=selection,
        rows=technical["rows"],
        evaluator_payload=evaluator_payload,
    )
    zero_failure_gate = _gate(
        "zero_technical_and_join_failures",
        all(
            int(technical[key]) == 0
            for key in (
                "load_failure_count",
                "worker_row_failure_count",
                "teacher_forced_failure_count",
                "token_boundary_failure_count",
                "greedy_parse_failure_count",
                "missing_union_count",
                "duplicate_union_count",
                "unexpected_union_count",
            )
        )
        and metrics["join_failure_count"] == 0,
    )
    all_gates = [zero_failure_gate, *pilot_gates]
    summary = {
        "kind": "phase2_job3_burned_pilot_aggregate_summary",
        "status": "pass" if all(g["status"] == "pass" for g in all_gates) else "fail",
        "mode": "pilot",
        "burned": True,
        "non_confirmatory": True,
        "rows_frozen": False,
        "analysis_scope": "PCS-E greedy class-conditional construction-pilot gates only",
        "renderer_comparison_performed": False,
        "arm_selection_performed": False,
        "technical_summary_sha256": _sha256_file(aggregate_dir / "technical_summary.json"),
        "worker_code_snapshot_sha256": worker_code_snapshot_sha256,
        "evaluator_code_snapshot_sha256": _code_snapshot_sha256(),
        "worker_snapshot_compatibility": compatibility,
        "evaluator_side_bindings": evaluator_bindings,
        "pcs_e_metrics": metrics["rows"],
        "gates": all_gates,
        "generated_at": _utc_now_iso(),
    }
    _write_jsonl(aggregate_dir / "pilot_pcs_e_joined_rows.jsonl", joined_rows)
    summary["joined_rows_sha256"] = _sha256_file(aggregate_dir / "pilot_pcs_e_joined_rows.jsonl")
    _write_json(aggregate_dir / "aggregate_summary.json", summary)
    return summary


def _scan_forbidden_metric_terms(
    value: Any, terms: Sequence[str] = CANARY_FORBIDDEN_METRIC_TERMS
) -> list[str]:
    text = json.dumps(value, sort_keys=True).lower()
    return [term for term in terms if term in text]


def classify_recovery_outcome(
    *,
    aggregate_status: Any,
    verification_status: Any,
    verification_gates: Any,
    aggregate_cli_status: int,
    verify_cli_status: int,
) -> str:
    if not isinstance(verification_gates, list) or not all(
        isinstance(row, dict) for row in verification_gates
    ):
        raise ValueError("verification gates are missing or malformed")
    gate_by_name = {
        str(row.get("gate")): row.get("status") for row in verification_gates
    }
    if len(gate_by_name) != len(verification_gates) or "" in gate_by_name:
        raise ValueError("verification gate names are missing or duplicated")
    expected_gate_names = {
        "fresh_technical_validation",
        "technical_summary_contract",
        "aggregate_status",
        "aggregate_mode",
        "pilot_scope",
        "pilot_aggregate_contract",
    }
    if set(gate_by_name) != expected_gate_names:
        raise ValueError("verification recovery gate set drifted")
    required_structural_gates = expected_gate_names - {"aggregate_status"}
    if any(gate_by_name[name] != "pass" for name in required_structural_gates):
        raise ValueError("a structural recovery verification gate failed")

    if aggregate_status == "pass":
        if aggregate_cli_status != 0 or verify_cli_status != 0:
            raise ValueError("task-validity pass disagrees with aggregate/verify CLI exits")
        if verification_status != "pass" or any(
            status != "pass" for status in gate_by_name.values()
        ):
            raise ValueError("task-validity pass disagrees with verification gates")
        return "completed_task_validity_pass"
    if aggregate_status == "fail":
        if aggregate_cli_status != 1 or verify_cli_status != 1:
            raise ValueError("task-validity fail disagrees with aggregate/verify CLI exits")
        if verification_status != "fail":
            raise ValueError("task-validity fail must yield a failed verification summary")
        failed_gates = {name for name, status in gate_by_name.items() if status != "pass"}
        if failed_gates != {"aggregate_status"}:
            raise ValueError(
                "scientific gate failure was mixed with contract failures: "
                f"{sorted(failed_gates)}"
            )
        return "completed_task_validity_fail"
    raise ValueError(f"unexpected aggregate task-validity status: {aggregate_status!r}")


def verify(
    *,
    data_dir: Path,
    run_dir: Path,
    model_plan_path: Path,
    prespec_path: Path,
    mode: str,
    compatible_worker_code_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    model_plan_path = model_plan_path.expanduser().resolve()
    prespec_path = prespec_path.expanduser().resolve()
    prespec = _load_prespec(prespec_path)
    worker_code_snapshot_sha256, compatibility = _resolve_worker_snapshot_compatibility(
        mode=mode,
        compatible_worker_code_snapshot_sha256=compatible_worker_code_snapshot_sha256,
    )
    _validate_compatible_source_artifacts(
        run_dir=run_dir,
        selection_manifest_path=run_dir / "selection_manifest.json",
        compatibility=compatibility,
    )
    canary_forbidden_terms = _canary_forbidden_terms(prespec)
    selection_path = run_dir / "selection_manifest.json"
    technical = _technical_validate(
        data_dir=data_dir,
        run_dir=run_dir,
        selection_manifest_path=selection_path,
        model_plan_path=model_plan_path,
        prespec_path=prespec_path,
        mode=mode,
        worker_code_snapshot_sha256=worker_code_snapshot_sha256,
    )
    aggregate_path = run_dir / "aggregate" / "aggregate_summary.json"
    if not aggregate_path.exists():
        raise FileNotFoundError("aggregate_summary.json is missing")
    aggregate_summary = _read_json(aggregate_path)
    technical_path = run_dir / "aggregate" / "technical_summary.json"
    if not technical_path.exists():
        raise FileNotFoundError("technical_summary.json is missing")
    written_technical = _read_json(technical_path)
    expected_public_technical = _strip_private_rows(technical)
    technical_contract_failures = _exact_key_failures(
        written_technical,
        set(expected_public_technical) | {"generated_at", "worker_snapshot_compatibility"},
    )
    technical_contract_failures.extend(
        [
        key
        for key, value in expected_public_technical.items()
        if written_technical.get(key) != value
        ]
    )
    if written_technical.get("worker_snapshot_compatibility") != compatibility:
        technical_contract_failures.append("worker_snapshot_compatibility")
    if not isinstance(written_technical.get("generated_at"), str):
        technical_contract_failures.append("generated_at")
    gates_out = [
        _gate("fresh_technical_validation", technical["status"] == "pass"),
        _gate(
            "technical_summary_contract",
            not technical_contract_failures,
            sorted(set(technical_contract_failures)),
        ),
        _gate("aggregate_status", aggregate_summary.get("status") == "pass"),
        _gate("aggregate_mode", aggregate_summary.get("mode") == mode),
    ]
    if mode in {"canary", "full"}:
        forbidden = _scan_forbidden_metric_terms(aggregate_summary, canary_forbidden_terms)
        gates_out.append(_gate("label_free_metric_firewall", not forbidden, forbidden))
        gates_out.append(_gate("labels_withheld", aggregate_summary.get("label_state") == "withheld"))
    else:
        selection = _load_selection_manifest(
            selection_manifest_path=selection_path,
            data_dir=data_dir,
            model_plan_path=model_plan_path,
            prespec_path=prespec_path,
            expected_code_snapshot_sha256=worker_code_snapshot_sha256,
        )
        evaluator_bindings, evaluator_payload = _validate_evaluator_side_bindings(
            data_dir=data_dir,
            prespec=prespec,
            selection=selection,
        )
        metrics, joined_rows, pilot_gates = _pilot_metrics(
            data_dir=data_dir,
            selection=selection,
            rows=technical["rows"],
            evaluator_payload=evaluator_payload,
        )
        zero_failure_gate = _gate(
            "zero_technical_and_join_failures",
            all(
                int(technical[key]) == 0
                for key in (
                    "load_failure_count",
                    "worker_row_failure_count",
                    "teacher_forced_failure_count",
                    "token_boundary_failure_count",
                    "greedy_parse_failure_count",
                    "missing_union_count",
                    "duplicate_union_count",
                    "unexpected_union_count",
                )
            )
            and metrics["join_failure_count"] == 0,
        )
        expected_pilot_gates = [zero_failure_gate, *pilot_gates]
        joined_path = run_dir / "aggregate" / "pilot_pcs_e_joined_rows.jsonl"
        joined_contract = bool(
            joined_path.is_file()
            and list(_iter_jsonl(joined_path)) == joined_rows
            and aggregate_summary.get("joined_rows_sha256") == _sha256_file(joined_path)
        )
        expected_fields = {
            "kind": "phase2_job3_burned_pilot_aggregate_summary",
            "status": (
                "pass"
                if all(gate["status"] == "pass" for gate in expected_pilot_gates)
                else "fail"
            ),
            "mode": "pilot",
            "burned": True,
            "non_confirmatory": True,
            "rows_frozen": False,
            "analysis_scope": "PCS-E greedy class-conditional construction-pilot gates only",
            "renderer_comparison_performed": False,
            "arm_selection_performed": False,
            "technical_summary_sha256": _sha256_file(technical_path),
            "worker_code_snapshot_sha256": worker_code_snapshot_sha256,
            "evaluator_code_snapshot_sha256": _code_snapshot_sha256(),
            "worker_snapshot_compatibility": compatibility,
            "evaluator_side_bindings": evaluator_bindings,
            "pcs_e_metrics": metrics["rows"],
            "gates": expected_pilot_gates,
        }
        aggregate_contract_failures = [
            key
            for key, value in expected_fields.items()
            if aggregate_summary.get(key) != value
        ]
        aggregate_contract_failures.extend(
            _exact_key_failures(
                aggregate_summary,
                set(expected_fields) | {"generated_at", "joined_rows_sha256"},
            )
        )
        if not isinstance(aggregate_summary.get("generated_at"), str):
            aggregate_contract_failures.append("generated_at")
        if not joined_contract:
            aggregate_contract_failures.append("joined_rows")
        gates_out.append(
            _gate(
                "pilot_scope",
                aggregate_summary.get("renderer_comparison_performed") is False
                and aggregate_summary.get("arm_selection_performed") is False,
            )
        )
        gates_out.append(
            _gate(
                "pilot_aggregate_contract",
                not aggregate_contract_failures,
                sorted(set(aggregate_contract_failures)),
            )
        )
    summary = {
        "kind": "phase2_job3_verification_summary",
        "status": "pass" if all(g["status"] == "pass" for g in gates_out) else "fail",
        "mode": mode,
        "gates": gates_out,
        "selection_manifest_sha256": _sha256_file(selection_path),
        "aggregate_summary_sha256": _sha256_file(aggregate_path),
        "worker_code_snapshot_sha256": worker_code_snapshot_sha256,
        "evaluator_code_snapshot_sha256": _code_snapshot_sha256(),
        "worker_snapshot_compatibility": compatibility,
        "verified_at": _utc_now_iso(),
    }
    if mode in {"canary", "full"} and _scan_forbidden_metric_terms(
        summary, canary_forbidden_terms
    ):
        raise RuntimeError("label-free verification crossed the metric firewall")
    _write_json(run_dir / "verify_summary.json", summary)
    return summary
