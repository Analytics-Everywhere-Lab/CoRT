from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from research.lifecycle_memory import reanchored_release


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_REL = Path(
    "research/lifecycle_memory/run_phase2_reanchored_answer_token_causal_patch.py"
)
ANALYZER_REL = Path(
    "research/lifecycle_memory/analyze_phase2_reanchored_answer_token_causal_patch.py"
)
PRESPEC_REL = Path(
    "docs/research/lifecycle_memory/phase 2 research/preregistration/"
    "reanchored_answer_token_causal_intervention_v1_2026_07_23.json"
)
JOB_DIR_REL = Path(
    "docs/research/lifecycle_memory/phase 2 research/jobs/"
    "reanchored_jobCR_answer_token_causal_patch"
)
PRESPEC_SHA256 = (
    "21dfd51486c95e370dac2e5c9cc297dd897a136ad7772286467525ee06047703"
)


def _probe(source: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_answer_token_runner_scores_both_choices_with_one_forward() -> None:
    probe = _probe(
        r'''
from types import SimpleNamespace
import math
import torch

from research.lifecycle_memory import (
    run_phase2_reanchored_answer_token_causal_patch as runner,
)

slot_map = runner.ImageSlotMap(
    image_token_id=99,
    sequence_length=3,
    slots=tuple(
        runner.SlotBoundary(
            slot_index=index,
            slot_name=runner.IMAGE_SLOT_NAMES[index],
            decoder_positions=(index,),
            decoder_start=index,
            decoder_stop=index + 1,
            decoder_count=1,
            premerge_count=1,
        )
        for index in range(3)
    ),
    grid_triplets=((1, 1, 1),) * 3,
    processor_signature_sha256="a" * 64,
)
shared_inputs = {"input_ids": torch.tensor([[8, 9, 10]])}
choices = {
    choice: runner.PreparedChoice(
        choice=choice,
        inputs=shared_inputs,
        base_length=3,
        full_ids=[8, 9, 10, token_id],
        candidate_ids=[token_id],
        slot_map=slot_map,
        construction="one_shared_prefix_forward_next_token_score",
        policy_trace_sha256="b" * 64,
    )
    for choice, token_id in (("A", 1), ("B", 2))
}
prepared = runner.PreparedArm(
    arm="real",
    base_inputs=shared_inputs,
    base_messages=[],
    base_ids=[8, 9, 10],
    question_position=1,
    question_token_audit={"status": "pass", "index": 1},
    slot_map=slot_map,
    choices=choices,
    serialization={},
    asset_receipts=[],
)

class Model:
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        logits = torch.zeros(1, 3, 4)
        logits[0, 2] = torch.tensor([-1.0, 3.0, 1.0, 0.0])
        return SimpleNamespace(logits=logits)

model = Model()
scores, forwards = runner.forward_plain_scores(model, prepared)
expected = torch.log_softmax(torch.tensor([-1.0, 3.0, 1.0, 0.0]), dim=-1)
assert forwards == 1
assert model.calls == 1
assert math.isclose(scores["A"], float(expected[1]), rel_tol=0.0, abs_tol=1e-7)
assert math.isclose(scores["B"], float(expected[2]), rel_tol=0.0, abs_tol=1e-7)
assert runner._engine.REQUIRED_CANARY_ANSWER_TOKEN_EQUIVALENCE is True
assert runner._engine.PROJECTION_FORMULA_VERSION == 5
'''
    )
    assert probe.returncode == 0, probe.stderr or probe.stdout


def test_job_cr_prespec_and_behavioral_admission_are_exact() -> None:
    prespec_path = REPO_ROOT / PRESPEC_REL
    assert hashlib.sha256(prespec_path.read_bytes()).hexdigest() == PRESPEC_SHA256
    prespec = json.loads(prespec_path.read_text(encoding="utf-8"))
    assert prespec["status"].startswith("prospectively_frozen")
    assert prespec["candidate_completions"] == {"A": "A", "B": "B"}
    assert prespec["data"]["mechanism_subset"]["frame_count"] == 80
    assert prespec["job_c"]["primary_patch_families"] == [
        "vision_tower_final",
        "projector_or_merger_output",
        "decoder_visual_token_residual",
    ]

    summary = (
        REPO_ROOT
        / "runs/tamia/reanchored-jobAR-token-full/380145/analysis/"
        "analysis_summary.json"
    )
    manifest = summary.with_name("analysis_manifest.json")
    if not summary.is_file() or not manifest.is_file():
        pytest.skip("local Job A-R result bundle is unavailable")
    probe = _probe(
        f'''
from pathlib import Path
from research.lifecycle_memory import (
    analyze_phase2_reanchored_answer_token_causal_patch as analysis,
)
gate = analysis.load_job_a_promotion_gate(
    summary_path=Path({str(summary)!r}),
    manifest_path=Path({str(manifest)!r}),
    expected_summary_sha256=(
        "8ebee5db19e8d123789cfb73bd6d6f5a7cd55a0f9473af8d01dfd6e3ef23a9c2"
    ),
)
assert gate["status"] == "pass", gate
assert gate["frame_count"] == 320
assert all(
    gate["models"][model]["promotion_status"] == "pass"
    for model in ("qwen35_27b_hf_bf16", "gemma4_31b_hf_bf16")
)
assert analysis.PRIMARY_STAGE_FAMILIES == (
    "vision_tower_final",
    "projector_or_merger_output",
    "decoder_visual_token_residual",
)
assert analysis.REPORT_EARLIEST_CAUSAL_SITE is False
assert analysis.ANALYSIS_KIND == (
    "phase2_reanchored_answer_token_causal_patch_analysis"
)
'''
    )
    assert probe.returncode == 0, probe.stderr or probe.stdout


def test_job_cr_tamia_package_is_release_selected_and_fail_closed() -> None:
    selected = {
        path.relative_to(REPO_ROOT)
        for path in reanchored_release.selected_files(REPO_ROOT)
    }
    assert RUNNER_REL in selected
    assert ANALYZER_REL in selected
    assert PRESPEC_REL in selected
    assert any(path.is_relative_to(JOB_DIR_REL) for path in selected)

    batch = (
        REPO_ROOT
        / JOB_DIR_REL
        / "job_tamia_reanchored_jobCR_answer_token_causal_patch_h100.sh"
    ).read_text(encoding="utf-8")
    submit = (REPO_ROOT / JOB_DIR_REL / "submit_tamia.sh").read_text(
        encoding="utf-8"
    )
    package_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / JOB_DIR_REL).glob("*"))
        if path.is_file()
    )
    assert "#SBATCH --gpus=h100:4" in batch
    assert "#SBATCH --ntasks-per-node=4" in batch
    assert "#SBATCH --cpus-per-task=12" in batch
    assert "#SBATCH --mem=480G" in batch
    assert not any(
        line.startswith("#SBATCH --partition=") for line in batch.splitlines()
    )
    assert "--expected-frames 80" in batch
    assert "--gpus-per-task=h100:1" in batch
    assert "--gpu-bind=single:1" in batch
    assert "HF_HUB_OFFLINE=1" in batch
    assert "TRANSFORMERS_OFFLINE=1" in batch
    assert "projection_formula_version\"] == 5" in batch
    assert "projection_formula_version\"] == 5" in submit
    assert "phase2_reanchored_answer_token_confirmation_analysis" in submit
    assert "Job C-R requires both promoted anchors" in submit
    assert 'p["sif_sha256"] == expected_sif_sha256' in submit
    assert 'p["sif_sha256"] == sha(sif_path)' not in submit
    assert 'sha256sum "$SIF_PATH"' not in submit
    assert "launcher must bind the compute-verified SIF SHA-256" in submit
    assert "reanchored-jobC-causal-patch" not in package_text
    assert PRESPEC_SHA256 in package_text
