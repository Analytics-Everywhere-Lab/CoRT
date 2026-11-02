#!/usr/bin/env python3
"""Dependency-free login-node metadata gate for PatchAudit held-out closure.

The login node checks only immutable JSON/JSONL metadata and file existence.
Image decoding, recursive asset hashing, model imports, token geometry, and
hooks are reserved for the compute-node canary.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


CONTRACT_KIND = "patchaudit_heldout_component_closure_job_contract"
PRESPEC_KIND = "patchaudit_heldout_component_closure_preregistration"
FROZEN_STATUS = (
    "prospectively_frozen_before_any_jobE_component_patch_outcome"
)
MODEL_KEYS = ("qwen35_27b_hf_bf16", "gemma4_31b_hf_bf16")
MODEL_SPECS = {
    "qwen35_27b_hf_bf16": (
        "Qwen/Qwen3.5-27B",
        "fc05daec18b0a78c049392ed2e771dde82bdf654",
        [0],
    ),
    "gemma4_31b_hf_bf16": (
        "google/gemma-4-31B-it",
        "3548789868c5356dbf307c98e6f609007b82b3eb",
        [1, 2, 3],
    ),
}
ARMS = ("real", "blank_full", "wrong_time")
CORRUPTIONS = ("blank_full", "wrong_time")
EXPECTED_USER_PROMPT = (
    "Input 1 is a full frame. Inputs 2 and 3 are unmarked target-only reference "
    "crops A and B. Which referenced target is the first-order interacting "
    "object, meaning the object directly manipulated by a visible hand rather "
    "than a second-order object interacting through it? Return exactly one JSON "
    "object with answer A or B plus the frozen abstention and confidence keys."
)
FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "correct_choice",
    "label",
    "class_id",
    "target_role",
    "first_order",
    "second_order",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MetadataGateError(RuntimeError):
    """Raised when a lightweight immutable-input contract does not hold."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataGateError(f"cannot read JSON {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise MetadataGateError(
                        f"non-object JSONL row at {path}:{line_number}"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataGateError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetadataGateError(f"{label} must be an object")
    return value


def _safe_relative(root: Path, raw: Any, label: str) -> Path:
    relative = Path(str(raw))
    if relative.is_absolute() or ".." in relative.parts:
        raise MetadataGateError(f"{label} is not a safe relative path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MetadataGateError(f"{label} escapes the data root") from exc
    return resolved


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            keys.update(str(key) for key in current)
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return keys


def validate_contract(
    *, repo_dir: Path, contract_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    contract = _mapping(read_json(contract_path), "contract")
    if contract.get("kind") != CONTRACT_KIND or contract.get("version") != 1:
        raise MetadataGateError("invalid Job E contract identity")
    prespec_record = _mapping(contract.get("prespec"), "contract prespec")
    prespec_path = _safe_relative(
        repo_dir, prespec_record.get("path"), "prespec path"
    )
    expected_prespec_sha = str(prespec_record.get("sha256") or "")
    if not SHA256_RE.fullmatch(expected_prespec_sha):
        raise MetadataGateError("contract has no valid prespec SHA-256")
    observed_prespec_sha = sha256_file(prespec_path)
    if observed_prespec_sha != expected_prespec_sha:
        raise MetadataGateError("Job E prespec SHA-256 drifted")
    prespec = _mapping(read_json(prespec_path), "prespec")
    if (
        prespec.get("kind") != PRESPEC_KIND
        or prespec.get("version") != 1
        or prespec.get("analysis_status") != FROZEN_STATUS
        or tuple(prespec.get("corruptions") or ()) != CORRUPTIONS
    ):
        raise MetadataGateError("invalid Job E frozen prespec")
    observed_models = [
        (
            str(row.get("model_key") or ""),
            str(row.get("model_id") or ""),
            str(row.get("revision") or ""),
        )
        for row in prespec.get("models", [])
        if isinstance(row, Mapping)
    ]
    expected_models = [
        (key, MODEL_SPECS[key][0], MODEL_SPECS[key][1])
        for key in MODEL_KEYS
    ]
    if observed_models != expected_models:
        raise MetadataGateError("Job E model or pinned revision drifted")
    completion = _mapping(contract.get("completion"), "completion")
    if (
        completion.get("expected_requests_per_model") != 92
        or completion.get("expected_outcomes_per_request_by_model")
        != {
            "qwen35_27b_hf_bf16": 24,
            "gemma4_31b_hf_bf16": 128,
        }
        or completion.get("expected_full_outcome_rows") != 13_984
        or contract.get("canary", {}).get("expected_outcome_rows") != 408
    ):
        raise MetadataGateError("Job E Cartesian inventory drifted")
    execution = _mapping(contract.get("execution"), "execution")
    expected_execution = {
        "account": "REPLACE_WITH_ACCOUNT",
        "gpu": "h100",
        "gpu_count": 4,
        "nodes": 1,
        "tasks": 4,
        "host_memory": "480G",
    }
    if any(execution.get(key) != value for key, value in expected_execution.items()):
        raise MetadataGateError("Job E TamIA whole-node contract drifted")
    if execution.get("rank_allocation") != {
        "qwen35_27b_hf_bf16": [0],
        "gemma4_31b_hf_bf16": [1, 2, 3],
    }:
        raise MetadataGateError("Job E rank allocation drifted")
    return (
        dict(contract),
        dict(prespec),
        {
            "contract_sha256": sha256_file(contract_path),
            "prespec_sha256": observed_prespec_sha,
        },
    )


def validate_model_plan(path: Path) -> dict[str, Any]:
    plan = read_json(path)
    if not isinstance(plan, list) or len(plan) != 2:
        raise MetadataGateError("model plan must contain exactly two rows")
    observed = [
        (
            str(row.get("model_key") or ""),
            str(row.get("model_id") or ""),
            str(row.get("revision") or ""),
            row.get("ranks"),
            row.get("shards"),
        )
        for row in plan
        if isinstance(row, Mapping)
    ]
    expected = [
        (
            key,
            MODEL_SPECS[key][0],
            MODEL_SPECS[key][1],
            MODEL_SPECS[key][2],
            len(MODEL_SPECS[key][2]),
        )
        for key in MODEL_KEYS
    ]
    if observed != expected:
        raise MetadataGateError("model plan identity, revision, or ranks drifted")
    return {
        "status": "pass",
        "sha256": sha256_file(path),
        "model_keys": list(MODEL_KEYS),
    }


def _asset_specs(
    row: Mapping[str, Any], arm: str, data_dir: Path
) -> list[dict[str, Any]]:
    arms = _mapping(row.get("arms"), "public row arms")
    arm_value = _mapping(arms.get(arm), f"public arm {arm}")
    inputs = arm_value.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 3:
        raise MetadataGateError(f"arm {arm} must declare three inputs")
    expected_geometry = ((896, 672), (336, 336), (336, 336))
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(inputs):
        spec = _mapping(raw, f"arm {arm} input {index}")
        path = _safe_relative(
            data_dir, spec.get("path"), f"arm {arm} input path"
        )
        if not path.is_file():
            raise MetadataGateError(f"declared asset is missing: {path}")
        declared_sha = str(spec.get("sha256") or "")
        if not SHA256_RE.fullmatch(declared_sha):
            raise MetadataGateError("asset has no valid declared SHA-256")
        if (spec.get("width"), spec.get("height")) != expected_geometry[index]:
            raise MetadataGateError(f"arm {arm} input geometry drifted")
        result.append({"path": str(path), "sha256": declared_sha})
    return result


def validate_public_metadata(
    *,
    data_dir: Path,
    mechanism_path: Path,
    selection_path: Path,
    public_path: Path,
    prespec: Mapping[str, Any],
) -> dict[str, Any]:
    data = _mapping(prespec.get("data"), "prespec data")
    expected_hashes = {
        mechanism_path: str(data["source_mechanism_manifest_sha256"]),
        selection_path: str(data["selection_manifest_sha256"]),
        public_path: str(data["public_requests_sha256"]),
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise MetadataGateError(f"frozen metadata SHA-256 drifted: {path}")
    mechanism = _mapping(read_json(mechanism_path), "source mechanism")
    selection = _mapping(read_json(selection_path), "selection manifest")
    selected = _mapping(selection.get("selected"), "selected cohort")
    overlap = _mapping(selection.get("overlap_audit"), "selection overlap audit")
    if (
        mechanism.get("status") != "complete_production"
        or mechanism.get("frame_count") != 96
        or selection.get("status") != "pass"
        or selected.get("frame_count") != 46
        or selected.get("request_count") != 92
        or selected.get("participant_cluster_count") != 46
        or any(
            not isinstance(value, Mapping)
            or int(value.get("overlap_count") or 0) != 0
            for value in overlap.values()
        )
    ):
        raise MetadataGateError("source or Job-D-disjoint selection contract drifted")
    rows = read_jsonl(public_path)
    if len(rows) != 92:
        raise MetadataGateError("selected public manifest must have 92 rows")
    request_ids = [str(row.get("request_id") or "") for row in rows]
    if len(set(request_ids)) != 92 or "" in request_ids:
        raise MetadataGateError("selected request IDs are duplicated or empty")
    if request_ids != [str(value) for value in selected.get("request_ids", [])]:
        raise MetadataGateError("selected public request order drifted")
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    asset_paths: set[str] = set()
    for row in rows:
        leaked = FORBIDDEN_PUBLIC_KEYS.intersection(_nested_keys(row))
        if leaked:
            raise MetadataGateError(f"public metadata leaks labels: {sorted(leaked)}")
        if row.get("schema_version") != 1:
            raise MetadataGateError("public schema version drifted")
        if row.get("user_prompt") != EXPECTED_USER_PROMPT:
            raise MetadataGateError("public prompt drifted")
        frame_id = str(row.get("frame_unit_id") or "")
        order_id = str(row.get("order_id") or "")
        peer = str(row.get("counterbalanced_request_id") or "")
        if not frame_id or order_id not in {"order_0", "order_1"} or not peer:
            raise MetadataGateError("counterbalance identity is incomplete")
        by_frame[frame_id].append(row)
        for arm in ARMS:
            asset_paths.update(
                spec["path"] for spec in _asset_specs(row, arm, data_dir)
            )
    if len(by_frame) != 46:
        raise MetadataGateError("selected public manifest must have 46 frame pairs")
    for frame_id, pair in by_frame.items():
        if len(pair) != 2:
            raise MetadataGateError(f"frame {frame_id} is not a pair")
        first, second = sorted(pair, key=lambda row: str(row["order_id"]))
        if (
            first["counterbalanced_request_id"] != second["request_id"]
            or second["counterbalanced_request_id"] != first["request_id"]
        ):
            raise MetadataGateError(f"frame {frame_id} peer join is not reciprocal")
        for arm in ARMS:
            first_assets = _asset_specs(first, arm, data_dir)
            second_assets = _asset_specs(second, arm, data_dir)
            if first_assets[0]["sha256"] != second_assets[0]["sha256"]:
                raise MetadataGateError(f"frame {frame_id} full frame changed by order")
            if [row["sha256"] for row in first_assets[1:]] != [
                second_assets[2]["sha256"],
                second_assets[1]["sha256"],
            ]:
                raise MetadataGateError(f"frame {frame_id} crops are not swapped")
    return {
        "status": "pass",
        "source_mechanism_sha256": sha256_file(mechanism_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "public_manifest_sha256": sha256_file(public_path),
        "request_count": len(rows),
        "frame_count": len(by_frame),
        "participant_cluster_count": selected["participant_cluster_count"],
        "declared_asset_file_count": len(asset_paths),
        "asset_files_exist": True,
        "asset_bytes_opened": False,
        "exact_asset_hashing_deferred_to_compute_canary": True,
    }


def verify_metadata(
    *,
    repo_dir: Path,
    contract_path: Path,
    model_plan_path: Path,
    data_dir: Path,
    mechanism_path: Path,
    selection_path: Path,
    public_path: Path,
) -> dict[str, Any]:
    repo_dir = repo_dir.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    model_plan_path = model_plan_path.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    mechanism_path = mechanism_path.expanduser().resolve()
    selection_path = selection_path.expanduser().resolve()
    public_path = public_path.expanduser().resolve()
    contract, prespec, hashes = validate_contract(
        repo_dir=repo_dir, contract_path=contract_path
    )
    return {
        "kind": "patchaudit_component_closure_login_metadata_preflight",
        "status": "pass",
        "hidden_index_opened": False,
        "contract": hashes,
        "model_plan": validate_model_plan(model_plan_path),
        "public_metadata": validate_public_metadata(
            data_dir=data_dir,
            mechanism_path=mechanism_path,
            selection_path=selection_path,
            public_path=public_path,
            prespec=prespec,
        ),
        "expected_full_outcome_rows": contract["completion"][
            "expected_full_outcome_rows"
        ],
        "heavy_runtime_imports": False,
        "compute_canary_required": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-plan", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mechanism-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        receipt = verify_metadata(
            repo_dir=args.repo_dir,
            contract_path=args.contract,
            model_plan_path=args.model_plan,
            data_dir=args.data_dir,
            mechanism_path=args.mechanism_manifest,
            selection_path=args.selection_manifest,
            public_path=args.public_manifest,
        )
        if args.out:
            args.out.expanduser().resolve().write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except MetadataGateError as exc:
        print(f"FATAL MetadataGateError: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
