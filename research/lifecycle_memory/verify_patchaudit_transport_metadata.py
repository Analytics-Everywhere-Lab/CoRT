#!/usr/bin/env python3
"""Dependency-free login-node preflight for PatchAudit transport decomposition.

This verifier intentionally uses only the Python standard library. Image
decoding, recursive asset hashing, model imports, and runtime hook checks belong
inside the Apptainer canary on the compute node.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


CONTRACT_KIND = "patchaudit_transport_decomposition_job_contract"
PRESPEC_KIND = "patchaudit_transport_decomposition_preregistration"
MECHANISM_KIND = "phase2_reanchored_egohos_mechanism_subset_manifest"
FROZEN_STATUS = (
    "prospectively_frozen_before_any_new_transport_decomposition_outcome"
)
MODEL_KEYS = ("qwen35_27b_hf_bf16", "gemma4_31b_hf_bf16")
MODEL_REVISIONS = {
    "qwen35_27b_hf_bf16": (
        "Qwen/Qwen3.5-27B",
        "fc05daec18b0a78c049392ed2e771dde82bdf654",
    ),
    "gemma4_31b_hf_bf16": (
        "google/gemma-4-31B-it",
        "3548789868c5356dbf307c98e6f609007b82b3eb",
    ),
}
ARMS = ("real", "blank_full", "wrong_time")
CORRUPTIONS = ("blank_full", "wrong_time")
OPERATORS = (
    "single_full_frame",
    "persistent_full_frame",
    "complement_once",
    "full_sequence_once",
)
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
    """Raised when the lightweight immutable-input contract does not hold."""


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


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetadataGateError(f"{label} must be an object")
    return value


def _safe_relative_path(root: Path, raw: Any, label: str) -> Path:
    relative = Path(str(raw))
    if relative.is_absolute() or ".." in relative.parts:
        raise MetadataGateError(f"{label} is not a safe relative path: {relative}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MetadataGateError(f"{label} escapes its root: {relative}") from exc
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


def _asset_specs(row: Mapping[str, Any], arm: str, data_dir: Path) -> list[dict[str, Any]]:
    arms = _require_mapping(row.get("arms"), "public row arms")
    arm_value = _require_mapping(arms.get(arm), f"public row arm {arm}")
    raw_inputs = arm_value.get("inputs")
    if not isinstance(raw_inputs, list) or len(raw_inputs) != 3:
        raise MetadataGateError(f"arm {arm} must have exactly three inputs")
    expected_geometry = ((896, 672), (336, 336), (336, 336))
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_inputs):
        spec = _require_mapping(raw, f"arm {arm} input {index}")
        path = _safe_relative_path(
            data_dir, spec.get("path"), f"arm {arm} input {index} path"
        )
        if not path.is_file():
            raise MetadataGateError(f"declared asset is missing: {path}")
        observed_geometry = (spec.get("width"), spec.get("height"))
        if observed_geometry != expected_geometry[index]:
            raise MetadataGateError(
                f"arm {arm} input {index} geometry drift: {observed_geometry}"
            )
        declared_sha = str(spec.get("sha256") or "").lower()
        if not SHA256_RE.fullmatch(declared_sha):
            raise MetadataGateError(
                f"arm {arm} input {index} has no valid declared SHA-256"
            )
        result.append(
            {
                "path": str(path),
                "sha256": declared_sha,
                "width": int(spec["width"]),
                "height": int(spec["height"]),
            }
        )
    return result


def validate_contract(
    *, repo_dir: Path, contract_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    contract = _require_mapping(read_json(contract_path), "contract")
    if contract.get("kind") != CONTRACT_KIND or contract.get("version") != 1:
        raise MetadataGateError("invalid transport contract kind or version")
    prespec_record = _require_mapping(contract.get("prespec"), "contract prespec")
    prespec_path = _safe_relative_path(
        repo_dir, prespec_record.get("path"), "prespec path"
    )
    expected_prespec_sha = str(prespec_record.get("sha256") or "").lower()
    if not SHA256_RE.fullmatch(expected_prespec_sha):
        raise MetadataGateError("contract has no valid prespec SHA-256")
    observed_prespec_sha = sha256_file(prespec_path)
    if observed_prespec_sha != expected_prespec_sha:
        raise MetadataGateError(
            "transport prespec SHA-256 differs from the frozen contract"
        )
    prespec = _require_mapping(read_json(prespec_path), "prespec")
    if (
        prespec.get("kind") != PRESPEC_KIND
        or prespec.get("version") != 1
        or prespec.get("analysis_status") != FROZEN_STATUS
    ):
        raise MetadataGateError("invalid transport prespec identity or frozen status")
    if tuple(prespec.get("corruptions") or ()) != CORRUPTIONS:
        raise MetadataGateError("transport corruption inventory drifted")
    operators = _require_mapping(prespec.get("operators"), "prespec operators")
    if set(operators) != set(OPERATORS):
        raise MetadataGateError("transport operator inventory drifted")
    models = prespec.get("models")
    if not isinstance(models, list):
        raise MetadataGateError("prespec models must be a list")
    observed_models = [
        (
            str(row.get("model_key") or ""),
            str(row.get("model_id") or ""),
            str(row.get("revision") or ""),
        )
        for row in models
        if isinstance(row, Mapping)
    ]
    expected_models = [
        (key, MODEL_REVISIONS[key][0], MODEL_REVISIONS[key][1])
        for key in MODEL_KEYS
    ]
    if observed_models != expected_models:
        raise MetadataGateError("transport model inventory or revision drifted")
    depth_plan = _require_mapping(prespec.get("depth_plan"), "prespec depth plan")
    depth_counts: set[int] = set()
    for model_key in MODEL_KEYS:
        model_depth = _require_mapping(
            depth_plan.get(model_key), f"depth plan {model_key}"
        )
        depths = model_depth.get("ordered_post_block_indexes")
        if (
            not isinstance(depths, list)
            or len(depths) != len(set(depths))
            or any(type(value) is not int or value < 0 for value in depths)
        ):
            raise MetadataGateError(f"invalid ordered depth plan for {model_key}")
        depth_counts.add(len(depths))
    completion = _require_mapping(contract.get("completion"), "contract completion")
    expected_per_request = (
        len(CORRUPTIONS) * 2 * next(iter(depth_counts)) * len(OPERATORS)
        if len(depth_counts) == 1
        else -1
    )
    if expected_per_request != completion.get("expected_outcomes_per_request"):
        raise MetadataGateError("per-request Cartesian outcome count drifted")
    expected_full = (
        len(MODEL_KEYS)
        * int(completion.get("expected_requests_per_model") or 0)
        * expected_per_request
    )
    if expected_full != completion.get("expected_full_outcome_rows"):
        raise MetadataGateError("full Cartesian outcome count drifted")
    execution = _require_mapping(contract.get("execution"), "contract execution")
    required_execution = {
        "account": "REPLACE_WITH_ACCOUNT",
        "gpu": "h100",
        "gpu_count": 4,
        "nodes": 1,
        "tasks": 4,
        "tasks_per_model": 2,
        "host_memory": "480G",
    }
    drift = {
        key: {"expected": expected, "observed": execution.get(key)}
        for key, expected in required_execution.items()
        if execution.get(key) != expected
    }
    if drift:
        raise MetadataGateError(f"TamIA execution contract drift: {drift}")
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
    if not isinstance(plan, list):
        raise MetadataGateError("model plan must be a list")
    observed = [
        (
            str(row.get("model_key") or ""),
            str(row.get("model_id") or ""),
            str(row.get("revision") or ""),
        )
        for row in plan
        if isinstance(row, Mapping)
    ]
    expected = [
        (key, MODEL_REVISIONS[key][0], MODEL_REVISIONS[key][1])
        for key in MODEL_KEYS
    ]
    if observed != expected:
        raise MetadataGateError("model plan identity or pinned revision drifted")
    return {
        "status": "pass",
        "sha256": sha256_file(path),
        "model_keys": list(MODEL_KEYS),
    }


def validate_public_metadata(
    *, data_dir: Path, mechanism_path: Path, public_path: Path
) -> dict[str, Any]:
    expected_mechanism = (data_dir / "mechanism/manifest.json").resolve()
    expected_public = (
        data_dir / "mechanism/public/mechanism_requests.jsonl"
    ).resolve()
    if mechanism_path != expected_mechanism or public_path != expected_public:
        raise MetadataGateError("mechanism/public paths do not match the frozen layout")
    mechanism = _require_mapping(read_json(mechanism_path), "mechanism manifest")
    expected_manifest_values = {
        "kind": MECHANISM_KIND,
        "schema_version": 1,
        "status": "complete_production",
        "production_eligible": True,
        "frame_count": 80,
        "trial_count": 160,
        "participant_cluster_count": 80,
        "counterbalanced_trials_per_frame": 2,
        "arms": list(ARMS),
        "images_per_arm": 3,
    }
    drift = {
        key: {"expected": expected, "observed": mechanism.get(key)}
        for key, expected in expected_manifest_values.items()
        if mechanism.get(key) != expected
    }
    if drift:
        raise MetadataGateError(f"mechanism manifest metadata drift: {drift}")
    if mechanism.get("fixed_geometry") != {
        "full_frame": {"width": 896, "height": 672},
        "reference_crop": {"width": 336, "height": 336},
    }:
        raise MetadataGateError("mechanism fixed geometry drifted")
    public_schema = _require_mapping(
        mechanism.get("public_schema"), "mechanism public schema"
    )
    if public_schema.get("evaluator_fields_present") is not False:
        raise MetadataGateError("mechanism public schema breaks the label firewall")
    files = _require_mapping(mechanism.get("files"), "mechanism files")
    public_record = _require_mapping(
        files.get("public_requests"), "mechanism public file record"
    )
    observed_public_sha = sha256_file(public_path)
    if (
        public_record.get("path")
        != "mechanism/public/mechanism_requests.jsonl"
        or public_record.get("rows") != 160
        or public_record.get("sha256") != observed_public_sha
    ):
        raise MetadataGateError("mechanism public file record drifted")

    rows = read_jsonl(public_path)
    if len(rows) != 160:
        raise MetadataGateError(f"expected 160 public rows, found {len(rows)}")
    request_ids = [str(row.get("request_id") or "") for row in rows]
    if "" in request_ids or len(set(request_ids)) != len(request_ids):
        raise MetadataGateError("public request IDs are missing or duplicated")
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    declared_assets: set[str] = set()
    for row in rows:
        leaked = FORBIDDEN_PUBLIC_KEYS.intersection(_nested_keys(row))
        if leaked:
            raise MetadataGateError(
                f"public row leaks evaluator keys: {sorted(leaked)}"
            )
        if row.get("schema_version") != 1:
            raise MetadataGateError("public row schema version drifted")
        if row.get("user_prompt") != EXPECTED_USER_PROMPT:
            raise MetadataGateError("public row prompt differs from the frozen prompt")
        order_id = str(row.get("order_id") or "")
        frame_id = str(row.get("frame_unit_id") or "")
        peer_id = str(row.get("counterbalanced_request_id") or "")
        if order_id not in {"order_0", "order_1"} or not frame_id or not peer_id:
            raise MetadataGateError("public counterbalance metadata is incomplete")
        by_frame[frame_id].append(row)
        for arm in ARMS:
            declared_assets.update(
                asset["path"] for asset in _asset_specs(row, arm, data_dir)
            )
    if len(by_frame) != 80:
        raise MetadataGateError(f"expected 80 frame pairs, found {len(by_frame)}")
    for frame_id, pair in by_frame.items():
        if len(pair) != 2 or {
            str(row.get("order_id") or "") for row in pair
        } != {"order_0", "order_1"}:
            raise MetadataGateError(f"frame {frame_id} is not one balanced pair")
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
                raise MetadataGateError(
                    f"frame {frame_id} full-frame declaration differs by order"
                )
            if [asset["sha256"] for asset in first_assets[1:]] != [
                second_assets[2]["sha256"],
                second_assets[1]["sha256"],
            ]:
                raise MetadataGateError(
                    f"frame {frame_id} crop declarations are not an A/B permutation"
                )
        for row in pair:
            real_assets = _asset_specs(row, "real", data_dir)
            for corruption in CORRUPTIONS:
                corrupt_assets = _asset_specs(row, corruption, data_dir)
                if [asset["sha256"] for asset in corrupt_assets[1:]] != [
                    asset["sha256"] for asset in real_assets[1:]
                ]:
                    raise MetadataGateError(
                        f"frame {frame_id} crop declaration changed in {corruption}"
                    )
    return {
        "status": "pass",
        "manifest_sha256": sha256_file(mechanism_path),
        "public_sha256": observed_public_sha,
        "trial_count": len(rows),
        "frame_unit_count": len(by_frame),
        "declared_asset_file_count": len(declared_assets),
        "asset_files_exist": True,
        "asset_bytes_opened": False,
        "image_decoder_imported": False,
        "exact_asset_hashing_deferred_to_compute_canary": True,
    }


def verify_metadata(
    *,
    repo_dir: Path,
    contract_path: Path,
    model_plan_path: Path,
    data_dir: Path,
    mechanism_path: Path,
    public_path: Path,
) -> dict[str, Any]:
    repo_dir = repo_dir.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    model_plan_path = model_plan_path.expanduser().resolve()
    mechanism_path = mechanism_path.expanduser().resolve()
    public_path = public_path.expanduser().resolve()
    contract, prespec, hashes = validate_contract(
        repo_dir=repo_dir, contract_path=contract_path
    )
    return {
        "kind": "patchaudit_transport_login_metadata_preflight",
        "status": "pass",
        "hidden_index_opened": False,
        "contract": hashes,
        "model_plan": validate_model_plan(model_plan_path),
        "public_metadata": validate_public_metadata(
            data_dir=data_dir,
            mechanism_path=mechanism_path,
            public_path=public_path,
        ),
        "models": [row["model_key"] for row in prespec["models"]],
        "expected_full_outcome_rows": contract["completion"][
            "expected_full_outcome_rows"
        ],
        "heavy_runtime_imports": False,
        "deep_preflight_deferred_to_compute_canary": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-plan", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mechanism-manifest", type=Path, required=True)
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
