"""Run explicit CPU-only suites with all model-framework imports prohibited."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from runtime.verify_source import BlockModelImports, HEAVY

SUITES = (
    "test_source",
    "research.lifecycle_memory.tests.test_role_aware_killarney_runtime",
    "research.lifecycle_memory.tests.test_role_aware_transfer",
    "research.lifecycle_memory.tests.test_temporal_forecasting_protocol",
    "research.lifecycle_memory.tests.test_temporal_forecasting_data",
    "research.lifecycle_memory.tests.test_temporal_dense_analysis",
    "research.lifecycle_memory.tests.test_temporal_recognition_analysis",
    "research.lifecycle_memory.tests.test_temporal_followup_analysis",
)


def main():
    sys.meta_path.insert(0, BlockModelImports())
    from research.lifecycle_memory import role_aware_transfer as transfer
    from research.lifecycle_memory.tests.test_role_aware_transfer import TransferTests
    adapters = [ROOT / transfer.REFERENCE_ADAPTERS / f"{transfer.CASE}.seed{s}.final.safetensors"
                for s in transfer.SEEDS]
    if not all(path.is_file() for path in adapters):
        # Preserve upstream tests and gates; the public weights directory does
        # not populate the historical prepared-release paths these tests expect.
        for name in (
            "test_canary_end_to_end_all_requests_and_repeats_no_labels",
            "test_freeze_exact_assets_mode_and_no_source_changes",
            "test_freeze_receipt_provenance_and_offline_environment_restoration",
            "test_full_fake_counts_and_real_estimator_individual_decisions",
            "test_invalid_worker_is_rejected_before_full_evaluator_check",
            "test_timed_drift_stops_worker_and_saves_failure",
            "test_verifier_rejects_resealed_missing_duplicate_and_changed_evidence",
        ):
            method = unittest.skip("requires historical prepared-release adapter paths")(getattr(TransferTests, name))
            setattr(TransferTests, name, method)
    suite = unittest.defaultTestLoader.loadTestsFromNames(SUITES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    versions = {}
    for name in ("numpy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"kind": "synthetic_cpu_tests_not_experiment_results", "suites": list(SUITES),
        "python": sys.version.split()[0], "packages": versions,
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
        "heavy_imports": sorted(HEAVY.intersection(sys.modules)),
        "model_or_gpu_execution": False, "network_used": False}
    if "--report" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--report") + 1])
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if result.wasSuccessful() and not report["heavy_imports"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
