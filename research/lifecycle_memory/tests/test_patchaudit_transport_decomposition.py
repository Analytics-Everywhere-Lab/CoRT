from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from research.lifecycle_memory import (
    analyze_patchaudit_transport_decomposition as analysis,
)
from research.lifecycle_memory import reanchored_release
from research.lifecycle_memory import run_patchaudit_transport_decomposition as runner


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = (
    REPO_ROOT
    / "docs/research/lifecycle_memory/phase 2 research/jobs/"
    "patchaudit_jobD_transport_decomposition/patch_contract.json"
)


def _result_rows(request_id: str, count: int = 64) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "model_key": "qwen35_27b_hf_bf16",
                "request_id": request_id,
                "corruption": runner.CORRUPTIONS[(index // 32) % 2],
                "direction": ("rescue", "damage")[(index // 16) % 2],
                "start_layer": (24, 28, 32, 36)[(index // 4) % 4],
                "operator": runner.OPERATORS[index % 4],
            }
        )
    return rows


def _synthetic_rows_for_records(
    *,
    prespec: dict[str, object],
    model_key: str,
    rank: int,
    records: list[dict[str, str]],
    operator_errors: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    errors = operator_errors or {
        "single_full_frame": 0.5,
        "persistent_full_frame": 0.25,
        "complement_once": 0.4,
        "full_sequence_once": 0.0,
    }
    layers = prespec["depth_plan"][model_key]["ordered_post_block_indexes"]  # type: ignore[index]
    rows: list[dict[str, object]] = []
    for record in records:
        for corruption in runner.CORRUPTIONS:
            for direction in ("rescue", "damage"):
                source_arm, recipient_arm = (
                    ("real", corruption)
                    if direction == "rescue"
                    else (corruption, "real")
                )
                for layer in layers:
                    for operator in runner.OPERATORS:
                        error = errors[operator]
                        recipient_margin = (
                            0.0 if direction == "rescue" else 2.0
                        )
                        donor_margin = (
                            2.0 if direction == "rescue" else 0.0
                        )
                        patched_margin = (
                            2.0 - error
                            if direction == "rescue"
                            else error
                        )
                        rows.append(
                            {
                                "kind": "patchaudit_transport_outcome",
                                "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
                                "model_key": model_key,
                                "rank": rank,
                                **record,
                                "corruption": corruption,
                                "direction": direction,
                                "source_arm": source_arm,
                                "recipient_arm": recipient_arm,
                                "start_layer": layer,
                                "operator": operator,
                                "real_margin_a_minus_b_bits": 2.0,
                                "corrupt_margin_a_minus_b_bits": 0.0,
                                "recipient_margin_a_minus_b_bits": recipient_margin,
                                "donor_margin_a_minus_b_bits": donor_margin,
                                "patched_margin_a_minus_b_bits": patched_margin,
                                "baseline_donor_gap_abs_bits": 2.0,
                                "patched_donor_error_abs_bits": error,
                                "audit": {
                                    "status": "pass",
                                    "all_fired_once": True,
                                    "all_removed": True,
                                    "all_finite": True,
                                },
                            }
                        )
    return rows


def _write_rank_artifacts(
    *,
    run_dir: Path,
    rank: int,
    records: list[dict[str, str]],
    rows: list[dict[str, object]],
    immutable: dict[str, str],
    canary: bool,
) -> None:
    rank_dir = run_dir / f"rank_{rank:03d}"
    outcomes_path = rank_dir / "outcomes.jsonl"
    completed_path = rank_dir / "completed_requests.jsonl"
    runner.atomic_write_jsonl(outcomes_path, rows)
    rows_by_request: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_request.setdefault(str(row["request_id"]), []).append(row)
    binding_sha = f"{rank + 7:064x}"
    receipts = [
        {
            "kind": "patchaudit_transport_request_receipt",
            "request_id": record["request_id"],
            "result_row_count": len(rows_by_request[record["request_id"]]),
            "result_batch_sha256": runner.sha256_json(
                rows_by_request[record["request_id"]]
            ),
            "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
            "resume_binding_sha256": binding_sha,
        }
        for record in records
    ]
    runner.atomic_write_jsonl(completed_path, receipts)
    model_key = runner.EXPECTED_MODEL_BY_RANK[rank]
    runner.atomic_write_json(
        rank_dir / "worker_manifest.json",
        {
            "kind": "patchaudit_transport_worker_manifest",
            "status": "pass",
            "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
            "model_key": model_key,
            "rank": rank,
            "world_size": 4,
            "selected_request_count": len(records),
            "selected_request_records": records,
            "selected_request_records_sha256": runner.sha256_json(records),
            "hidden_index_opened": False,
        },
    )
    runner.atomic_write_json(
        rank_dir / "worker_summary.json",
        {
            "kind": "patchaudit_transport_worker_summary",
            "status": "pass",
            "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
            "model_key": model_key,
            "rank": rank,
            "world_size": 4,
            "requests_completed": len(records),
            "requests_assigned": len(records),
            "outcome_rows": len(rows),
            "completion_receipt_rows": len(receipts),
            "outcomes_sha256": runner.sha256_file(outcomes_path),
            "completed_requests_sha256": runner.sha256_file(completed_path),
            "resume_binding_sha256": binding_sha,
            "hidden_index_opened": False,
            "peak_cuda_memory_bytes": 1024,
            "fresh_request_elapsed_s": [10.0 + rank],
            "load_elapsed_s": 20.0,
            "immutable_hashes": immutable,
        },
    )
    if canary:
        runner.atomic_write_json(
            rank_dir / "canary_audits.json",
            {
                "kind": "patchaudit_transport_canary_audits",
                "answer_token_equivalence_gate": {
                    "status": "pass",
                    "maximum_absolute_difference_bits": 0.0,
                },
                "self_checks": [
                    {"operator": operator, "status": "pass"}
                    for operator in runner.OPERATORS
                ],
            },
        )


def _write_job_manifest(
    *,
    run_dir: Path,
    mode: str,
    immutable: dict[str, str],
    canary_dir: Path | None = None,
) -> None:
    canary_fields: dict[str, object] = {}
    if mode == "full":
        if canary_dir is None:
            raise AssertionError("full synthetic job requires bound canary artifacts")
        canary_dir.mkdir(parents=True, exist_ok=True)
        canary_aggregate_path = canary_dir / "aggregate_summary.json"
        runner.atomic_write_json(
            canary_aggregate_path,
            {
                "kind": "patchaudit_transport_aggregate",
                "status": "pass",
                "mode": "canary",
                "immutable_hashes": immutable,
            },
        )
        canary_aggregate_sha = runner.sha256_file(canary_aggregate_path)
        canary_profile_path = canary_dir / "runtime_profile.json"
        runner.atomic_write_json(
            canary_profile_path,
            {
                "kind": "patchaudit_transport_runtime_profile",
                "status": "pass",
                "canary_aggregate_sha256": canary_aggregate_sha,
                "immutable_hashes": immutable,
            },
        )
        canary_fields = {
            "canary_profile_path": str(canary_profile_path),
            "canary_profile_sha256": runner.sha256_file(canary_profile_path),
            "canary_aggregate_path": str(canary_aggregate_path),
            "canary_aggregate_sha256": canary_aggregate_sha,
        }
    runner.atomic_write_json(
        run_dir / "job_manifest.json",
        {
            "kind": "patchaudit_transport_tamia_job_manifest",
            "status": "running",
            "mode": mode,
            **immutable,
            "worker_hidden_bind": False,
            "hidden_index_staged_before_technical_gate": False,
            **canary_fields,
        },
    )


def _write_passing_telemetry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "timestamp,index,utilization_gpu_pct,memory_used_mib,power_w\n"
        + "".join(
            f"2026-07-26T00:00:0{sample}Z,{gpu},50,1024,300\n"
            for sample in range(4)
            for gpu in range(4)
        ),
        encoding="utf-8",
    )


def _prepare_canary_run(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, str], dict[str, object]]:
    _, prespec, hashes = runner.verify_contract(
        repo_dir=REPO_ROOT,
        contract_path=CONTRACT,
    )
    data_dir = (
        REPO_ROOT
        / "runs/research/lifecycle_memory/phase_2/"
        "reanchored_answer_token_battery_v3_2026_07_22"
    )
    public_path = data_dir / "mechanism/public/mechanism_requests.jsonl"
    mechanism_path = data_dir / "mechanism/manifest.json"
    if not public_path.is_file():
        pytest.skip("exact frozen mechanism battery is unavailable")
    public = runner.read_jsonl(public_path)
    immutable = {
        "release_manifest_sha256": "1" * 64,
        "code_snapshot_sha256": "2" * 64,
        "contract_sha256": hashes["contract_sha256"],
        "prespec_sha256": hashes["prespec_sha256"],
        "mechanism_manifest_sha256": runner.sha256_file(mechanism_path),
        "public_manifest_sha256": runner.sha256_file(public_path),
        "model_plan_sha256": "5" * 64,
        "sif_sha256": "6" * 64,
    }
    run_dir = tmp_path / "canary"
    for rank in range(4):
        public_record = public[rank % 2]
        records = [
            {
                "request_id": str(public_record["request_id"]),
                "frame_unit_id": str(public_record["frame_unit_id"]),
                "order_id": str(public_record["order_id"]),
            }
        ]
        rows = _synthetic_rows_for_records(
            prespec=prespec,
            model_key=runner.EXPECTED_MODEL_BY_RANK[rank],
            rank=rank,
            records=records,
        )
        _write_rank_artifacts(
            run_dir=run_dir,
            rank=rank,
            records=records,
            rows=rows,
            immutable=immutable,
            canary=True,
        )
    _write_job_manifest(run_dir=run_dir, mode="canary", immutable=immutable)
    telemetry = run_dir / "telemetry.csv"
    _write_passing_telemetry(telemetry)
    return run_dir, public_path, telemetry, immutable, prespec


def test_frozen_contract_and_prespec_are_hash_bound() -> None:
    contract, prespec, hashes = runner.verify_contract(
        repo_dir=REPO_ROOT,
        contract_path=CONTRACT,
    )
    assert contract["completion"]["expected_full_outcome_rows"] == 20_480
    assert prespec["technical_gates"]["self_patch_max_abs_error_bits"] == 0.001
    assert len(hashes["contract_sha256"]) == 64
    assert len(hashes["prespec_sha256"]) == 64


def test_shards_preserve_counterbalanced_pairs() -> None:
    rows = []
    for frame in range(80):
        for order in range(2):
            rows.append(
                {
                    "frame_unit_id": f"frame-{frame:03d}",
                    "order_id": f"order_{order}",
                    "request_id": f"request-{frame:03d}-{order}",
                }
            )
    left = runner.shard_rows(rows, shard_index=0, shard_count=2)
    right = runner.shard_rows(rows, shard_index=1, shard_count=2)
    assert len(left) == len(right) == 80
    assert {row["request_id"] for row in left}.isdisjoint(
        {row["request_id"] for row in right}
    )
    for selected in (left, right):
        counts: dict[str, int] = {}
        for row in selected:
            frame = str(row["frame_unit_id"])
            counts[frame] = counts.get(frame, 0) + 1
        assert set(counts.values()) == {2}


def test_resume_sanitizer_discards_partial_request(tmp_path: Path) -> None:
    outcomes = tmp_path / "outcomes.jsonl"
    receipts = tmp_path / "completed_requests.jsonl"
    complete_rows = _result_rows("complete")
    partial_rows = _result_rows("partial", count=7)
    runner.atomic_write_jsonl(outcomes, [*complete_rows, *partial_rows])
    runner.atomic_write_jsonl(
        receipts,
        [
            {
                "kind": "patchaudit_transport_request_receipt",
                "request_id": "complete",
                "result_row_count": 64,
                "result_batch_sha256": runner.sha256_json(complete_rows),
                "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
                "resume_binding_sha256": "a" * 64,
            },
            {
                "kind": "patchaudit_transport_request_receipt",
                "request_id": "partial",
                "result_row_count": 64,
                "result_batch_sha256": "b" * 64,
                "output_schema_version": runner.OUTPUT_SCHEMA_VERSION,
                "resume_binding_sha256": "a" * 64,
            },
        ],
    )
    result = runner.sanitize_resume(
        outcomes_path=outcomes,
        completed_path=receipts,
        expected_binding_sha256="a" * 64,
        expected_request_ids={"complete", "partial"},
        expected_rows_per_request=64,
    )
    assert result["completed_request_ids"] == ["complete"]
    assert result["kept_rows"] == 64
    assert result["dropped_rows"] == 7
    assert len(runner.read_jsonl(outcomes)) == 64
    assert len(runner.read_jsonl(receipts)) == 1


def test_pristine_worker_initializes_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = (
        REPO_ROOT
        / "runs/research/lifecycle_memory/phase_2/"
        "reanchored_answer_token_battery_v3_2026_07_22"
    )
    mechanism_path = data_dir / "mechanism/manifest.json"
    public_path = data_dir / "mechanism/public/mechanism_requests.jsonl"
    model_plan_path = (
        REPO_ROOT
        / "docs/research/lifecycle_memory/phase 2 research/jobs/"
        "patchaudit_jobD_transport_decomposition/model_plan.json"
    )
    if not public_path.is_file():
        pytest.skip("exact frozen mechanism battery is unavailable")
    _, _, hashes = runner.verify_contract(
        repo_dir=REPO_ROOT,
        contract_path=CONTRACT,
    )
    env = {
        "RELEASE_MANIFEST_SHA256": "1" * 64,
        "CODE_SNAPSHOT_SHA256": "2" * 64,
        "CONTRACT_SHA256": hashes["contract_sha256"],
        "PRESPEC_SHA256": hashes["prespec_sha256"],
        "MECHANISM_MANIFEST_SHA256": runner.sha256_file(mechanism_path),
        "PUBLIC_MANIFEST_SHA256": runner.sha256_file(public_path),
        "MODEL_PLAN_SHA256": runner.sha256_file(model_plan_path),
        "SIF_SHA256": "6" * 64,
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    def stop_before_model_load(**_kwargs: object) -> object:
        raise RuntimeError("model-load-sentinel")

    monkeypatch.setattr(runner._base, "_load_anchor_model", stop_before_model_load)
    run_dir = tmp_path / "fresh"
    with pytest.raises(RuntimeError, match="model-load-sentinel"):
        runner.worker_command(
            argparse.Namespace(
                repo_dir=REPO_ROOT,
                contract=CONTRACT,
                data_dir=data_dir,
                mechanism_manifest=mechanism_path,
                public_manifest=public_path,
                model_plan=model_plan_path,
                run_dir=run_dir,
                rank=0,
                world_size=4,
                dtype="bfloat16",
                seed=20260726,
                canary=True,
                local_files_only=True,
                stop_file=None,
                max_runtime_seconds=None,
                deadline_epoch_seconds=None,
            )
        )
    rank_dir = run_dir / "rank_000"
    assert (rank_dir / "worker_manifest.json").is_file()
    assert (rank_dir / "outcomes.jsonl").read_text(encoding="utf-8") == ""
    assert (rank_dir / "completed_requests.jsonl").read_text(encoding="utf-8") == ""


def test_outcome_gate_recomputes_directional_margin_algebra() -> None:
    _, prespec, _ = runner.verify_contract(
        repo_dir=REPO_ROOT,
        contract_path=CONTRACT,
    )
    public_record = {
        "request_id": "request",
        "frame_unit_id": "frame",
        "order_id": "order_0",
    }
    rows = _synthetic_rows_for_records(
        prespec=prespec,
        model_key="qwen35_27b_hf_bf16",
        rank=0,
        records=[public_record],
    )
    damage = next(row for row in rows if row["direction"] == "damage")
    runner._validate_outcome_row(
        damage,
        rank=0,
        model_key="qwen35_27b_hf_bf16",
        public_record=public_record,
    )
    swapped = dict(damage)
    swapped["recipient_margin_a_minus_b_bits"] = 0.0
    swapped["donor_margin_a_minus_b_bits"] = 2.0
    with pytest.raises(runner.GateError, match="margin algebra"):
        runner._validate_outcome_row(
            swapped,
            rank=0,
            model_key="qwen35_27b_hf_bf16",
            public_record=public_record,
        )
    falsified_error = dict(damage)
    falsified_error["patched_donor_error_abs_bits"] = 0.0
    with pytest.raises(runner.GateError, match="margin algebra"):
        runner._validate_outcome_row(
            falsified_error,
            rank=0,
            model_key="qwen35_27b_hf_bf16",
            public_record=public_record,
        )


def _synthetic_joined(prespec: dict[str, object]) -> list[dict[str, object]]:
    errors = {
        "qwen35_27b_hf_bf16": {
            "single_full_frame": 1.2,
            "persistent_full_frame": 0.6,
            "complement_once": 1.1,
            "full_sequence_once": 0.0,
        },
        "gemma4_31b_hf_bf16": {
            "single_full_frame": 1.2,
            "persistent_full_frame": 1.1,
            "complement_once": 0.6,
            "full_sequence_once": 0.0,
        },
    }
    rows: list[dict[str, object]] = []
    for model in prespec["models"]:  # type: ignore[index]
        model_key = str(model["model_key"])
        layers = prespec["depth_plan"][model_key]["ordered_post_block_indexes"]  # type: ignore[index]
        for cluster in range(80):
            for order in range(2):
                request_id = f"{model_key}-c{cluster:03d}-o{order}"
                for corruption in runner.CORRUPTIONS:
                    for direction in ("rescue", "damage"):
                        for layer in layers:
                            for operator in runner.OPERATORS:
                                rows.append(
                                    {
                                        "model_key": model_key,
                                        "request_id": request_id,
                                        "frame_unit_id": f"frame-{cluster:03d}",
                                        "ego4d_participant_id": f"cluster-{cluster:03d}",
                                        "corruption": corruption,
                                        "direction": direction,
                                        "start_layer": layer,
                                        "operator": operator,
                                        "patched_donor_error_bits": errors[model_key][
                                            operator
                                        ],
                                        "recipient_donor_gap_bits": 2.0,
                                    }
                                )
    return rows


def test_primary_analysis_recovers_prespecified_taxonomy() -> None:
    _, prespec, _ = runner.verify_contract(
        repo_dir=REPO_ROOT,
        contract_path=CONTRACT,
    )
    rows = _synthetic_joined(prespec)
    cells, taxonomy = analysis.primary_analysis(
        rows,
        prespec=prespec,
        draws=400,
        seed=123,
    )
    assert len(cells) == 4
    assert (
        taxonomy["qwen35_27b_hf_bf16"]["call"]
        == "clamp_increment_supported_complement_unresolved"
    )
    assert (
        taxonomy["gemma4_31b_hf_bf16"]["call"]
        == "complement_increment_supported_clamp_unresolved"
    )
    for cell in cells:
        if cell["positive"]:
            assert cell["estimate_delta_rho"] == pytest.approx(0.3)
            assert cell["holm_adjusted_pvalue"] < 0.05
        else:
            assert cell["estimate_delta_rho"] == pytest.approx(0.05)


def test_hidden_join_rejects_worker_label_leak() -> None:
    public = [
        {
            "request_id": f"r{index}",
            "frame_unit_id": f"f{index // 2}",
            "order_id": "order_0",
        }
        for index in range(160)
    ]
    hidden = [
        {
            "request_id": f"r{index}",
            "correct_choice": "A",
            "participant_cluster_id": f"p{index // 2}",
            "ego4d_participant_id": str(index // 2),
        }
        for index in range(160)
    ]
    raw = [
        {
            "request_id": "r0",
            "frame_unit_id": "f0",
            "order_id": "order_0",
            "correct_choice": "A",
            "recipient_margin_a_minus_b_bits": 0.0,
            "donor_margin_a_minus_b_bits": 1.0,
            "patched_margin_a_minus_b_bits": 0.5,
            "patched_donor_error_abs_bits": 0.5,
            "baseline_donor_gap_abs_bits": 1.0,
        }
    ]
    with pytest.raises(analysis.AnalysisError, match="leaked hidden fields"):
        analysis._join_hidden(
            rows=raw,
            public_rows=public,
            hidden_rows=hidden,
        )


def test_tamia_package_is_release_selected_and_fail_closed() -> None:
    job_dir = Path(
        "docs/research/lifecycle_memory/phase 2 research/jobs/"
        "patchaudit_jobD_transport_decomposition"
    )
    selected = {
        path.relative_to(REPO_ROOT)
        for path in reanchored_release.selected_files(REPO_ROOT)
    }
    assert Path(
        "research/lifecycle_memory/run_patchaudit_transport_decomposition.py"
    ) in selected
    assert Path(
        "research/lifecycle_memory/analyze_patchaudit_transport_decomposition.py"
    ) in selected
    assert any(path.is_relative_to(job_dir) for path in selected)

    batch = (
        REPO_ROOT / job_dir / "job_tamia_patchaudit_jobD_transport_h100.sh"
    ).read_text(encoding="utf-8")
    submit = (REPO_ROOT / job_dir / "submit_tamia.sh").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --account=REPLACE_WITH_ACCOUNT" in batch
    assert "#SBATCH --gpus=h100:4" in batch
    assert "#SBATCH --ntasks-per-node=4" in batch
    assert "#SBATCH --cpus-per-task=12" in batch
    assert "#SBATCH --mem=480G" in batch
    assert "#SBATCH --signal=B:USR1@300" in batch
    assert not any(
        line.startswith("#SBATCH --partition=") for line in batch.splitlines()
    )
    assert "--gpus-per-task=h100:1" in batch
    assert "--gpu-bind=single:1" in batch
    assert "--max-runtime-seconds" in batch
    assert "deadline_watchdog" in batch
    assert 'kill -USR1 "$parent_pid"' in batch
    assert "partial_checkpoint" in batch
    assert "require-hidden-outside-data-dir" in batch
    assert "--public-manifest" in batch
    assert "technical_gate_passed_before_hidden_stage" in batch
    assert "hidden_index_staged_after_technical_gate" in batch
    assert "forbidden evaluator/source variable reached worker" in batch
    assert "worker_visible_hidden_scan_status" in batch
    assert "hidden evaluator artifact reached a worker-visible tree" in batch
    assert "HF_HUB_OFFLINE=1" in batch
    assert "TRANSFORMERS_OFFLINE=1" in batch
    assert 'sha256sum "$SIF_PATH"' in batch
    assert 'sha256sum "$SIF_PATH"' not in submit
    assert "SUBMISSION_SIF_SHA256" in submit
    assert "--comment=\"$SUBMISSION_TRANSACTION_TAG\"" in submit
    assert "--time=\"$WALLTIME\"" in submit


def test_analysis_end_to_end_on_frozen_inventory(tmp_path: Path) -> None:
    data_dir = (
        REPO_ROOT
        / "runs/research/lifecycle_memory/phase_2/"
        "reanchored_answer_token_battery_v3_2026_07_22"
    )
    public_path = data_dir / "mechanism/public/mechanism_requests.jsonl"
    hidden_path = data_dir / "mechanism/hidden/mechanism_evaluator_index.jsonl"
    mechanism_path = data_dir / "mechanism/manifest.json"
    if not all(path.is_file() for path in (public_path, hidden_path, mechanism_path)):
        pytest.skip("exact frozen mechanism battery is unavailable")
    contract, prespec, hashes = runner.verify_contract(
        repo_dir=REPO_ROOT,
        contract_path=CONTRACT,
    )
    public = runner.read_jsonl(public_path)
    error_by_model = {
        "qwen35_27b_hf_bf16": {
            "single_full_frame": 1.2,
            "persistent_full_frame": 0.6,
            "complement_once": 1.1,
            "full_sequence_once": 0.0,
        },
        "gemma4_31b_hf_bf16": {
            "single_full_frame": 1.2,
            "persistent_full_frame": 1.1,
            "complement_once": 0.6,
            "full_sequence_once": 0.0,
        },
    }
    run_dir = tmp_path / "run"
    model_plan_path = (
        REPO_ROOT
        / "docs/research/lifecycle_memory/phase 2 research/jobs/"
        "patchaudit_jobD_transport_decomposition/model_plan.json"
    )
    immutable = {
        "release_manifest_sha256": "1" * 64,
        "code_snapshot_sha256": "2" * 64,
        "contract_sha256": hashes["contract_sha256"],
        "prespec_sha256": hashes["prespec_sha256"],
        "mechanism_manifest_sha256": runner.sha256_file(mechanism_path),
        "public_manifest_sha256": runner.sha256_file(public_path),
        "model_plan_sha256": runner.sha256_file(model_plan_path),
        "sif_sha256": "6" * 64,
    }
    records_by_rank: dict[int, list[dict[str, str]]] = {
        rank: [] for rank in range(4)
    }
    for model_index, _model in enumerate(prespec["models"]):
        for request_index, request in enumerate(public):
            rank = model_index * 2 + request_index % 2
            records_by_rank[rank].append(
                {
                    "request_id": str(request["request_id"]),
                    "frame_unit_id": str(request["frame_unit_id"]),
                    "order_id": str(request["order_id"]),
                }
            )
    rows_by_rank: dict[int, list[dict[str, object]]] = {}
    for rank, records in records_by_rank.items():
        model_key = runner.EXPECTED_MODEL_BY_RANK[rank]
        rows = _synthetic_rows_for_records(
            prespec=prespec,
            model_key=model_key,
            rank=rank,
            records=records,
            operator_errors=error_by_model[model_key],
        )
        rows_by_rank[rank] = rows
        _write_rank_artifacts(
            run_dir=run_dir,
            rank=rank,
            records=records,
            rows=rows,
            immutable=immutable,
            canary=False,
        )
    _write_job_manifest(
        run_dir=run_dir,
        mode="full",
        immutable=immutable,
        canary_dir=tmp_path / "bound_canary",
    )
    telemetry = run_dir / "telemetry.csv"
    _write_passing_telemetry(telemetry)
    all_rows = [row for rows in rows_by_rank.values() for row in rows]
    assert len(all_rows) == contract["completion"]["expected_full_outcome_rows"]
    assert (
        runner.aggregate_command(
            argparse.Namespace(
                repo_dir=REPO_ROOT,
                contract=CONTRACT,
                run_dir=run_dir,
                public_manifest=public_path,
                telemetry_csv=telemetry,
                canary=False,
            )
        )
        == 0
    )
    aggregate_path = run_dir / "aggregate_summary.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["status"] == "pass"
    evaluator_hidden = (
        tmp_path / "evaluator_only" / "mechanism_evaluator_index.jsonl"
    )
    evaluator_hidden.parent.mkdir(parents=True)
    evaluator_hidden.write_bytes(hidden_path.read_bytes())
    job_manifest_path = run_dir / "job_manifest.json"
    job_manifest = json.loads(job_manifest_path.read_text(encoding="utf-8"))
    job_manifest.update(
        {
            "technical_gate_passed_before_hidden_stage": True,
            "technical_aggregate_sha256": runner.sha256_file(aggregate_path),
            "hidden_index_staged_after_technical_gate": True,
            "hidden_index_outside_worker_data_dir": True,
            "hidden_index_stage_path": str(evaluator_hidden),
        }
    )
    runner.atomic_write_json(job_manifest_path, job_manifest)
    out_dir = run_dir / "analysis"
    result = analysis.analyze_command(
        argparse.Namespace(
            repo_dir=REPO_ROOT,
            contract=CONTRACT,
            run_dir=run_dir,
            data_dir=data_dir,
            mechanism_manifest=mechanism_path,
            public_manifest=public_path,
            hidden_index=evaluator_hidden,
            out_dir=out_dir,
            bootstrap_draws=10_000,
            seed=20260726,
            require_hidden_outside_data_dir=True,
        )
    )
    assert result == 0
    payload = json.loads(
        (out_dir / "transport_analysis.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "pass"
    assert payload["raw_outcome_row_count"] == 20_480
    assert (
        payload["primary_holm_family"]["taxonomy"]["qwen35_27b_hf_bf16"][
            "call"
        ]
        == "clamp_increment_supported_complement_unresolved"
    )
    assert (
        payload["primary_holm_family"]["taxonomy"]["gemma4_31b_hf_bf16"][
            "call"
        ]
        == "complement_increment_supported_clamp_unresolved"
    )
    assert (out_dir / "paper_transport_table.tex").is_file()
    assert (out_dir / "transport_profile_plot.csv").is_file()
    assert (out_dir / "analysis_manifest.json").is_file()


def test_canary_aggregate_binds_runtime_profile_to_immutable_inputs(
    tmp_path: Path,
) -> None:
    run_dir, public_path, telemetry, immutable, _ = _prepare_canary_run(
        tmp_path
    )
    result = runner.aggregate_command(
        argparse.Namespace(
            repo_dir=REPO_ROOT,
            contract=CONTRACT,
            run_dir=run_dir,
            public_manifest=public_path,
            telemetry_csv=telemetry,
            canary=True,
        )
    )
    assert result == 0
    aggregate = json.loads(
        (run_dir / "aggregate_summary.json").read_text(encoding="utf-8")
    )
    profile = json.loads(
        (run_dir / "runtime_profile.json").read_text(encoding="utf-8")
    )
    assert aggregate["status"] == "pass"
    assert aggregate["immutable_hashes"] == immutable
    assert profile["immutable_hashes"] == immutable
    assert profile["planned_chunk_count"] >= 1


def test_aggregate_rejects_noncartesian_operator_inventory(
    tmp_path: Path,
) -> None:
    run_dir, public_path, telemetry, immutable, _ = _prepare_canary_run(
        tmp_path
    )
    rank_dir = run_dir / "rank_000"
    manifest = json.loads(
        (rank_dir / "worker_manifest.json").read_text(encoding="utf-8")
    )
    rows = runner.read_jsonl(rank_dir / "outcomes.jsonl")
    rows[0]["operator"] = "unregistered_operator"
    _write_rank_artifacts(
        run_dir=run_dir,
        rank=0,
        records=manifest["selected_request_records"],
        rows=rows,
        immutable=immutable,
        canary=True,
    )
    result = runner.aggregate_command(
        argparse.Namespace(
            repo_dir=REPO_ROOT,
            contract=CONTRACT,
            run_dir=run_dir,
            public_manifest=public_path,
            telemetry_csv=telemetry,
            canary=True,
        )
    )
    assert result == 2
    aggregate = json.loads(
        (run_dir / "aggregate_summary.json").read_text(encoding="utf-8")
    )
    assert aggregate["status"] == "fail"
    assert any(
        failure.startswith("worker_artifact_contract:0")
        for failure in aggregate["failures"]
    )


def test_aggregate_rejects_missing_full_sequence_control(
    tmp_path: Path,
) -> None:
    run_dir, public_path, telemetry, immutable, _ = _prepare_canary_run(
        tmp_path
    )
    rank_dir = run_dir / "rank_000"
    manifest = json.loads(
        (rank_dir / "worker_manifest.json").read_text(encoding="utf-8")
    )
    rows = [
        row
        for row in runner.read_jsonl(rank_dir / "outcomes.jsonl")
        if row["operator"] != "full_sequence_once"
    ]
    _write_rank_artifacts(
        run_dir=run_dir,
        rank=0,
        records=manifest["selected_request_records"],
        rows=rows,
        immutable=immutable,
        canary=True,
    )
    result = runner.aggregate_command(
        argparse.Namespace(
            repo_dir=REPO_ROOT,
            contract=CONTRACT,
            run_dir=run_dir,
            public_manifest=public_path,
            telemetry_csv=telemetry,
            canary=True,
        )
    )
    assert result == 2
    aggregate = json.loads(
        (run_dir / "aggregate_summary.json").read_text(encoding="utf-8")
    )
    assert aggregate["status"] == "fail"
    assert aggregate["full_sequence_row_count"] < 64


def test_canary_requires_answer_token_equivalence_and_telemetry(
    tmp_path: Path,
) -> None:
    run_dir, public_path, telemetry, _, _ = _prepare_canary_run(tmp_path)
    audit_path = run_dir / "rank_000/canary_audits.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["answer_token_equivalence_gate"]["status"] = "fail"
    runner.atomic_write_json(audit_path, audit)
    assert (
        runner.aggregate_command(
            argparse.Namespace(
                repo_dir=REPO_ROOT,
                contract=CONTRACT,
                run_dir=run_dir,
                public_manifest=public_path,
                telemetry_csv=telemetry,
                canary=True,
            )
        )
        == 2
    )

    run_dir_2, public_path_2, _, _, _ = _prepare_canary_run(
        tmp_path / "missing_telemetry"
    )
    assert (
        runner.aggregate_command(
            argparse.Namespace(
                repo_dir=REPO_ROOT,
                contract=CONTRACT,
                run_dir=run_dir_2,
                public_manifest=public_path_2,
                telemetry_csv=run_dir_2 / "missing.csv",
                canary=True,
            )
        )
        == 2
    )


def test_production_analysis_rejects_bootstrap_override_before_unblinding(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        analysis.AnalysisError,
        match="frozen bootstrap configuration",
    ):
        analysis.analyze_command(
            argparse.Namespace(
                repo_dir=REPO_ROOT,
                contract=CONTRACT,
                run_dir=tmp_path / "not_read",
                data_dir=tmp_path / "not_read",
                mechanism_manifest=tmp_path / "not_read.json",
                public_manifest=tmp_path / "not_read.jsonl",
                hidden_index=tmp_path / "must_not_be_read.jsonl",
                out_dir=tmp_path / "out",
                bootstrap_draws=100,
                seed=321,
                require_hidden_outside_data_dir=True,
            )
        )


def test_full_job_requires_hash_bound_passing_canary(tmp_path: Path) -> None:
    immutable = {
        "release_manifest_sha256": "1" * 64,
        "code_snapshot_sha256": "2" * 64,
    }
    with pytest.raises(runner.GateError, match="bound canary"):
        runner.validate_bound_canary(
            job_manifest={},
            immutable_hashes=immutable,
        )
    canary_dir = tmp_path / "canary"
    run_dir = tmp_path / "full"
    _write_job_manifest(
        run_dir=run_dir,
        mode="full",
        immutable=immutable,
        canary_dir=canary_dir,
    )
    job = json.loads(
        (run_dir / "job_manifest.json").read_text(encoding="utf-8")
    )
    receipt = runner.validate_bound_canary(
        job_manifest=job,
        immutable_hashes=immutable,
    )
    assert receipt["status"] == "pass"
    assert receipt["canary_aggregate_sha256"] == job["canary_aggregate_sha256"]


def test_hidden_join_uses_declared_ego4d_participant_field() -> None:
    public = [
        {
            "request_id": f"r{index}",
            "frame_unit_id": f"f{index // 2}",
            "order_id": f"order_{index % 2}",
        }
        for index in range(160)
    ]
    hidden = [
        {
            "request_id": f"r{index}",
            "correct_choice": "A",
            "ego4d_participant_id": str(index // 2),
        }
        for index in range(160)
    ]
    raw = [
        {
            "request_id": "r0",
            "frame_unit_id": "f0",
            "order_id": "order_0",
            "recipient_margin_a_minus_b_bits": 0.0,
            "donor_margin_a_minus_b_bits": 1.0,
            "patched_margin_a_minus_b_bits": 0.5,
            "patched_donor_error_abs_bits": 0.5,
            "baseline_donor_gap_abs_bits": 1.0,
        }
    ]
    joined = analysis._join_hidden(
        rows=raw,
        public_rows=public,
        hidden_rows=hidden,
    )
    assert joined[0]["ego4d_participant_id"] == "0"
