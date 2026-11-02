"""Standard-library-only source and synthetic scientific contract checks."""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import export_source, verify_source
from research.lifecycle_memory import patchaudit_component_partition as component
from research.lifecycle_memory import role_aware_policy, role_aware_transfer
from research.lifecycle_memory.temporal_forecasting import protocol, recognition_protocol, followup_protocol

POLICY = {"version": 1, "kind": "matched_static_reuse",
          "historical_static_sha256": "a" * 64, "temporal_exposure_sha256": "b" * 64}


class SourceTests(unittest.TestCase):
    def test_editorial_changes_are_comment_only(self):
        source = '# workflow note\n"""Old documentation."""\nx = "unchanged"\n'
        edits = [
            {"kind": "public_comment_edit", "before": "# workflow note", "after": "# Technical note"},
            {"kind": "public_comment_edit", "before": '"""Old documentation."""',
             "after": '"""Public documentation."""'},
        ]
        actual = export_source.apply_comment_edits(source, edits)
        self.assertIn("# Technical note", actual)
        self.assertIn("Public documentation.", actual)
        self.assertEqual(export_source.executable_ast(source), export_source.executable_ast(actual))
        self.assertEqual(export_source.apply_comment_edits(source, []), source)

    def test_editorial_changes_reject_code_and_ambiguous_matches(self):
        for source, before, after in (
            ('x = "old"\n', '"old"', '"new"'),
            ('# note\nx = 1\n', '# note', '# note\ny = 2'),
            ('# note\n# note\nx = 1\n', '# note', '# revised'),
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                export_source.apply_comment_edits(source, [
                    {"kind": "public_comment_edit", "before": before, "after": after}])

    def test_editorial_spans_support_unicode(self):
        source = 'def f(): """Position \\u03c1."""; return 1\n'.replace('\\u03c1', '\u03c1')
        before = '"""Position \u03c1."""'
        edited = export_source.apply_comment_edits(source, [
            {"kind": "public_comment_edit", "before": before, "after": '"""Recovery."""'}])
        self.assertEqual(export_source.executable_ast(source), export_source.executable_ast(edited))

    def test_editorial_spans_preserve_form_feed_offsets(self):
        source = 'x = 1\f\n# comment\n'
        spans = export_source.comment_spans(source)
        self.assertEqual([source[start:end] for start, end in spans], ['# comment'])

    def test_editorial_changes_reject_private_paths(self):
        private = '/scratch/' + 'a/example/private'
        with self.assertRaises(ValueError):
            export_source.apply_comment_edits('# note\nx = 1\n', [
                {"kind": "public_comment_edit", "before": '# note', "after": '# ' + private}])

    def test_hashes_syntax_and_every_local_import(self):
        result = verify_source.verify()
        self.assertGreater(result["local_import_statements"], 100)

    def test_no_private_paths_accounts_or_legacy_credentials(self):
        manifest = json.loads((ROOT / "runtime/source_manifest.json").read_text())
        patterns = (r"/(?:Users|home)/[A-Za-z][^\s\"']*/", r"/scratch/[a-z]/[A-Za-z0-9_-]+/",
                    r"\b(?:aip|rrg|def)-[A-Za-z][A-Za-z0-9_-]*", r"_gemini_api_key|_load_local_env_file")
        for entry in manifest["files"]:
            if entry["original_path"] is None:
                continue
            text = (ROOT / entry["path"]).read_text()
            for pattern in patterns:
                with self.subTest(path=entry["path"], pattern=pattern):
                    self.assertIsNone(re.search(pattern, text))
        self.assertFalse((ROOT / "research/lifecycle_memory/role_aware_killarney_transfer.py").exists())

    def test_namespace_import_without_model_backends(self):
        code = """import importlib, sys
from runtime.verify_source import BlockModelImports, HEAVY
sys.meta_path.insert(0, BlockModelImports())
for name in ('role_aware_policy', 'role_aware_transfer', 'run_role_aware_transfer',
             'role_aware_killarney_runtime', 'temporal_forecasting.cli',
             'temporal_forecasting.recognition_launch', 'temporal_forecasting.followup_prepare'):
    importlib.import_module('research.lifecycle_memory.' + name)
assert not HEAVY.intersection(sys.modules)
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dynamic_helper_is_present(self):
        from runtime.export_source import JOB0
        self.assertTrue((ROOT / JOB0).is_file())

    def test_preparation_roots_and_dependency_closure(self):
        from runtime.export_source import PACKAGE, PREPARATION_MODULES
        manifest = json.loads((ROOT / "runtime/source_manifest.json").read_text())
        for module in PREPARATION_MODULES:
            self.assertIn(PACKAGE + module + ".py", manifest["roots"])
        for module in ("phase2_job3_egohos_census", "phase2_job3_egohos_burned_items",
                       "phase2_job3_marker_renderer", "phase2_job3_negative_review_export",
                       "phase2_job3_negative_replacement_review_export", "temporal_matched_media",
                       "temporal_matched_preparation", "temporal_forecasting_census",
                       "temporal_forecasting_static_overlap"):
            self.assertTrue((ROOT / (PACKAGE + module + ".py")).is_file(), module)

    def test_private_selection_is_not_a_compatibility_config(self):
        from runtime.export_source import PRIVATE_SELECTION
        manifest = json.loads((ROOT / "runtime/source_manifest.json").read_text())
        self.assertFalse((ROOT / PRIVATE_SELECTION).exists())
        self.assertNotIn(PRIVATE_SELECTION, {entry["path"] for entry in manifest["files"]})
        summary = json.loads((ROOT / "protocols/construction/heldout_selection_documentary_summary.json").read_text())
        self.assertTrue(summary["not_an_execution_manifest"])
        self.assertNotEqual(summary["kind"], "patchaudit_jobE_heldout_selection")
        selected = summary["original_record"]["selected"]
        self.assertEqual((selected["frame_count"], selected["request_count"]), (46, 92))
        for key in ("frame_unit_ids", "request_ids"):
            self.assertNotIn(key, selected)
            self.assertRegex(selected[key + "_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(verify_source.digest(ROOT / "protocols/construction/heldout_selection_documentary_summary.json"),
                            summary["original_sha256"])

    def test_construction_documents_do_not_publish_identity_or_path_fields(self):
        forbidden = {"frame_unit_ids", "request_ids", "reviewer", "reviewer_id", "reviewer_ids",
                     "path", "source_path", "s3_uri", "account", "container"}

        def check(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertNotRegex(value, r"/(?:Users|home|scratch)/|s3://|https?://")

        paths = list((ROOT / "protocols/construction").glob("*.json"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            with self.subTest(path=path.name):
                value = json.loads(path.read_text())
                check(value)
                self.assertTrue(value["not_an_execution_manifest"])
                self.assertRegex(value["provenance"]["archive_member_sha256"], r"^[0-9a-f]{64}$")

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"),
                         "held-out analyzer imports require NumPy and Pillow")
    def test_documentary_summary_cannot_replace_bound_selection(self):
        from research.lifecycle_memory import analyze_patchaudit_heldout_component_closure as analyzer
        path = ROOT / "protocols/construction/heldout_selection_documentary_summary.json"
        summary = json.loads(path.read_text())
        with self.assertRaisesRegex(analyzer.AnalysisError, "selection manifest hash drifted"):
            analyzer._validate_selection(selection_path=path, public_path=ROOT / "missing-private-requests.jsonl",
                prespec={"data": {"selection_manifest_sha256": summary["original_sha256"]}})

    def test_optional_scipy_detection_is_explicit(self):
        with patch.object(verify_source.importlib.util, "find_spec", return_value=None):
            for module in verify_source.SCIPY_MODULES:
                self.assertEqual(verify_source.missing_optional_dependencies(module), ["scipy"])
            self.assertEqual(verify_source.missing_optional_dependencies("temporal_dense_current_data"), [])

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"),
                         "smoke imports require NumPy and Pillow")
    def test_smoke_reports_missing_scipy_without_loading_it(self):
        code = """import importlib.util, json, sys
from runtime import verify_source
find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args, **kwargs: None if name == 'scipy' else find_spec(name, *args, **kwargs)
result = verify_source.smoke()
assert len(result['optional_dependency_skips']) == 4
assert 'scipy' not in sys.modules
assert all(item['missing_dependencies'] == ['scipy'] for item in result['optional_dependency_skips'])
print(json.dumps(result))
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("numpy", "PIL", "scipy")),
                         "static preparation checks require NumPy, Pillow and optional SciPy")
    def test_static_components_exclusions_and_bound_input_failures(self):
        import numpy as np
        from research.lifecycle_memory import build_phase2_reanchored_egohos_battery as builder
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[2:10, 2:10] = 3
        mask[16:24, 16:24] = 3
        selected = builder._select_target_component(mask, 3)
        self.assertEqual((selected.area, selected.bbox_tlbr), (64, (2, 2, 10, 10)))
        self.assertEqual(selected.qualifying_component_count, 2)
        self.assertIsNone(builder._select_target_component(mask, 4))
        burned = builder._burned_identifiers([{"candidate_id": "synthetic-candidate"}],
            [{"candidate_id": "synthetic-candidate", "frame_id": "synthetic-frame", "split": "train"}])
        self.assertEqual(burned["frame_keys"], {"train|synthetic-frame"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "burned evaluator index path is required"):
                builder.build_reanchored_egohos_battery(repo_root=root, data_root=root,
                    inventory_path=root / "inventory", candidate_sidecar_path=root / "candidates",
                    out_dir=root / "output")
            prespec = root / "prespec.json"
            prespec.write_text("{}\n")
            with self.assertRaisesRegex(builder.ConstructionError, "prespec SHA-256 mismatch"):
                builder._validate_prespec(prespec_path=prespec, candidate_sidecar_path=root / "candidates",
                    burned_evaluator_path=root / "exclusions", prompt_prespec_path=root / "prompt",
                    enforce_frozen_bindings=True)

    def test_dense_preparation_synthetic_alignment_and_fail_closed_selection(self):
        from research.lifecycle_memory import temporal_dense_current_data as dense
        targets = [index / 4 for index in range(9)]
        aligned = dense.choose_pts(list(range(9)), "1/4", 0, targets)
        self.assertEqual([aligned[slot]["pts"] for slot in dense.SCENES], list(range(9)))
        self.assertEqual(dense._proofs(None), {})
        with self.assertRaisesRegex(ValueError, "NineTargetsRequired"):
            dense.choose_pts(list(range(9)), "1/4", 0, targets[:-1])
        with self.assertRaisesRegex(ValueError, "PrecedingFrameUnavailable"):
            dense.choose_pts([], "1/4", 0, targets)
        with self.assertRaisesRegex(ValueError, "Full96InsufficientWearers"):
            dense.select_current_rows([], {"participants": []}, stage="current")

    def test_cpu_cli_help(self):
        for name in ("role_aware_killarney_runtime", "run_role_aware_transfer",
                     "temporal_forecasting.cli", "temporal_forecasting.recognition_launch",
                     "temporal_forecasting.followup_prepare", "temporal_dense_current_data"):
            with self.subTest(module=name):
                result = subprocess.run([sys.executable, "-B", "-m", "research.lifecycle_memory." + name,
                                         "--help"], cwd=ROOT, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_component_partition_and_all_fifteen_unions(self):
        part = component.build_component_partition(sequence_length=926, active_positions=tuple(range(926)),
            image_slot_positions=(tuple(range(48, 636)), tuple(range(638, 738)), tuple(range(740, 840))),
            literal_user_prompt_start=841, literal_user_prompt_count=73, answer_prefix_token_count=3)
        self.assertEqual(part.full_frame.count, 588)
        self.assertEqual(part.complement.count, 338)
        self.assertEqual(len(component.factorial_component_unions(part).cells), 15)
        with self.assertRaises(component.ComponentPartitionError):
            component.build_component_partition(sequence_length=4, active_positions=(0, 0, 1),
                image_slot_positions=((0,), (1,), (2,)), literal_user_prompt_start=3,
                literal_user_prompt_count=1, answer_prefix_token_count=1)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "transport imports require Pillow")
    def test_transport_partition_rejects_overlap(self):
        from research.lifecycle_memory import patchaudit_transport_operators as transport
        with self.assertRaises(transport.TransportGateError):
            transport.position_partition(sequence_length=4, active_positions=(0, 1, 2, 3),
                                         full_frame_positions=(0, 0))

    def test_practical_variants_and_unfrozen_transfer(self):
        self.assertEqual(len(role_aware_policy.variant_specs()), 26)
        self.assertEqual(len(role_aware_transfer.variant_specs()), 5)
        config = role_aware_transfer.source_config()
        role_aware_transfer.validate_config(config)
        self.assertEqual(config["status"], "awaiting_freeze")
        self.assertFalse(config["training_allowed"])
        config["training_allowed"] = True
        with self.assertRaises(ValueError):
            role_aware_transfer.validate_config(config)

    def test_current_and_recognition_protocol_gates(self):
        current = protocol.dense_current_protocol(POLICY)
        self.assertEqual(current["fixed_cuts"], [22, 34])
        for task in recognition_protocol.TASKS:
            recognition = recognition_protocol.recognition_protocol(task, POLICY)
            protocol.validate_protocol(recognition)
            self.assertEqual(recognition["scene_count"], 17)
            broken = deepcopy(recognition)
            broken["scene_count"] = 16
            with self.assertRaises(ValueError):
                protocol.validate_protocol(broken)

    def test_followup_pins_permutations_and_canary(self):
        source = recognition_protocol.recognition_protocol(recognition_protocol.TASKS[0], POLICY)
        pins = {key: "c" * 64 for key in followup_protocol.PIN_FIELDS}
        value = followup_protocol.followup_protocol(source, **pins)
        self.assertTrue(value["followup"]["new_canary_required"])
        self.assertEqual(len(followup_protocol.SUBSETS), 15)
        for condition in ("S1", "S2"):
            order = followup_protocol.scene_permutation(source["task"], "synthetic-episode", condition)
            self.assertEqual(sorted(order), list(range(17)))
            self.assertEqual((order[0], order[-1]), (0, 16))
        with self.assertRaises(ValueError):
            followup_protocol.verify_followup_dataset(value, "d" * 64)
        value["followup"]["new_canary_required"] = False
        with self.assertRaises(ValueError):
            followup_protocol.validate_followup_protocol(value)


if __name__ == "__main__":
    unittest.main()
