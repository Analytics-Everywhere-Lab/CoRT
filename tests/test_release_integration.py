"""Regression checks for replay bindings, release inventory, and the toy example."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from reproduce_all import PACKAGES, RECEIPT, digest, inputs, run_replay, verified_receipt
from rebuild_public_results import ADDITIONAL_REPLAY
from verify_release import verify
from examples.position_replacement import replace_rows


class ReplayBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "analysis").mkdir()
        (self.root / "analysis/example.py").write_text("# synthetic fixture\n")
        (self.root / "manifests").mkdir()
        records = {}
        for name, spec in PACKAGES.items():
            path = self.root / spec["report"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"status": "pass"}\n')
            records[name] = dict(spec, status="pass", sha256=digest(path))
        self.receipt = {"status": "pass", "input_sha256": inputs(self.root), "packages": records}
        self.save()

    def save(self):
        (self.root / RECEIPT).write_text(json.dumps(self.receipt))

    def test_valid_binding(self):
        self.assertIsNotNone(verified_receipt(self.root))

    def test_changed_code_invalidates_binding(self):
        (self.root / "analysis/example.py").write_text("# changed\n")
        self.assertIsNone(verified_receipt(self.root))

    def test_new_input_invalidates_binding(self):
        path = self.root / "results/replay_temporal/new.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        self.assertIsNone(verified_receipt(self.root))

    def test_changed_report_invalidates_binding(self):
        (self.root / PACKAGES["temporal"]["report"]).write_text('{"status": "fail"}')
        self.assertIsNone(verified_receipt(self.root))

    def test_missing_package_invalidates_binding(self):
        del self.receipt["packages"]["temporal"]
        self.save()
        self.assertIsNone(verified_receipt(self.root))

    def test_missing_receipt_is_unverified(self):
        (self.root / RECEIPT).unlink()
        self.assertIsNone(verified_receipt(self.root))

    def test_zero_exit_without_fresh_report_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            run_replay(PACKAGES['temporal'], lambda script: None, self.root)

    def test_failed_fresh_report_is_rejected(self):
        spec = PACKAGES['temporal']
        def runner(script):
            (self.root / spec['report']).write_text('{"status": "fail"}')
        with self.assertRaises(ValueError):
            run_replay(spec, runner, self.root)

    def test_unreplayed_paper_items_not_certified(self):
        for name in ('Tables S33-S34', 'Table S35; Section S14.4', 'Table S29; Section S14.2', 'Table 1',
                     'Tables S17-S18; S20; S23', 'Table S36'):
            self.assertNotIn(name, ADDITIONAL_REPLAY)
        self.assertEqual(ADDITIONAL_REPLAY['Tables S30-S32'], 'practical')


class InventoryTests(unittest.TestCase):
    def test_unlisted_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "release.txt"
            path.write_text("synthetic fixture\n")
            manifest = {"files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}]}
            (root / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest))
            self.assertEqual(verify(root), [])
            (root / "unexpected.txt").write_text("not inventoried")
            self.assertIn("Unlisted release file: unexpected.txt", verify(root))

    def test_symlink_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "RELEASE_MANIFEST.json").write_text('{"files": []}')
            (root / "link").symlink_to("absent")
            self.assertTrue(any("Symlinks" in item for item in verify(root)))


class ReplacementExampleTests(unittest.TestCase):
    def test_preserves_recipient_outside_patch(self):
        recipient = np.zeros((5, 3))
        donor = np.arange(15).reshape(5, 3)
        result = replace_rows(recipient, donor, [1, 3])
        np.testing.assert_array_equal(result[[1, 3]], donor[[1, 3]])
        np.testing.assert_array_equal(result[[0, 2, 4]], recipient[[0, 2, 4]])
        np.testing.assert_array_equal(recipient, np.zeros((5, 3)))

    def test_invalid_indices_rejected(self):
        states = np.zeros((5, 3))
        for positions in ([1.5], [True], [1, 1], [-1], [5], [[1]]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                replace_rows(states, states, positions)


if __name__ == "__main__":
    unittest.main()
