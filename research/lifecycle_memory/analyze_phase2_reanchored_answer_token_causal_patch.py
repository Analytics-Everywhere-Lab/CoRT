#!/usr/bin/env python3
"""Analyze the separately frozen answer-token causal intervention."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.lifecycle_memory import (
    run_phase2_reanchored_answer_token_causal_patch as runner,
)
from research.lifecycle_memory import analyze_phase2_reanchored_causal_patch as _analysis
from research.lifecycle_memory.analyze_phase2_reanchored_causal_patch import *  # noqa: F403


JOB_A_SUMMARY_KIND = "phase2_reanchored_answer_token_confirmation_analysis"
JOB_A_MANIFEST_KIND = (
    "phase2_reanchored_answer_token_confirmation_analysis_manifest"
)
BEHAVIORAL_ELIGIBILITY_FRAME_COUNT = 320
BEHAVIORAL_ELIGIBILITY_PRESPEC_SHA256 = (
    "920f441efc5faf4de0070a61cbf64132de7b1ebe2828bea9781e3b8cfa9715ba"
)
BEHAVIORAL_ELIGIBILITY_RESULT_KEY = (
    "answer_token_320_frame_promotion_gate"
)
BEHAVIORAL_ELIGIBILITY_CALL_GATE_KEY = (
    "answer_token_320_frame_model_promotion_gate"
)
PRIMARY_STAGE_FAMILIES = (
    "vision_tower_final",
    "projector_or_merger_output",
    "decoder_visual_token_residual",
)
REPORT_EARLIEST_CAUSAL_SITE = False
ANALYSIS_KIND = "phase2_reanchored_answer_token_causal_patch_analysis"
DEFAULT_SEED = 20260723
ANALYSIS_CLAIM_BOUNDARY = (
    "A positive primary call identifies an intervention-specific bidirectional "
    "causal bottleneck under the frozen full-frame pathway replacement: the "
    "matched clean state rescues a corrupted execution, the reverse state "
    "damages the clean execution, and both exceed unrelated and norm-matched "
    "controls. The jointly selected decoder site is the strongest tested coarse "
    "site after correction, not an earliest, unique, or historical failure "
    "origin. Final-question-token patches are downstream diagnostics only. A "
    "null means no detected effect under this intervention, not absence of "
    "information."
)


def _configure_analysis() -> None:
    values = {
        "runner": runner,
        "JOB_A_SUMMARY_KIND": JOB_A_SUMMARY_KIND,
        "JOB_A_MANIFEST_KIND": JOB_A_MANIFEST_KIND,
        "BEHAVIORAL_ELIGIBILITY_FRAME_COUNT": BEHAVIORAL_ELIGIBILITY_FRAME_COUNT,
        "BEHAVIORAL_ELIGIBILITY_PRESPEC_SHA256": (
            BEHAVIORAL_ELIGIBILITY_PRESPEC_SHA256
        ),
        "BEHAVIORAL_ELIGIBILITY_RESULT_KEY": BEHAVIORAL_ELIGIBILITY_RESULT_KEY,
        "BEHAVIORAL_ELIGIBILITY_CALL_GATE_KEY": (
            BEHAVIORAL_ELIGIBILITY_CALL_GATE_KEY
        ),
        "PRIMARY_STAGE_FAMILIES": PRIMARY_STAGE_FAMILIES,
        "REPORT_EARLIEST_CAUSAL_SITE": REPORT_EARLIEST_CAUSAL_SITE,
        "ANALYSIS_KIND": ANALYSIS_KIND,
        "DEFAULT_SEED": DEFAULT_SEED,
        "ANALYSIS_CLAIM_BOUNDARY": ANALYSIS_CLAIM_BOUNDARY,
    }
    for name, value in values.items():
        setattr(_analysis, name, value)


_configure_analysis()


def main() -> int:
    _configure_analysis()
    return _analysis.main()


if __name__ == "__main__":
    raise SystemExit(main())
