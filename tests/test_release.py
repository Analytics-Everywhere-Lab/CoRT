"""Portable checks for the public artifact, without model or dataset access."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", "__pycache__", "outputs", ".pytest_cache"}


def release_files():
    return [p for p in ROOT.rglob("*") if (p.is_file() or p.is_symlink())
            and not any(part in IGNORED for part in p.relative_to(ROOT).parts)]


class PublicReleaseTests(unittest.TestCase):
    def weight_entries(self):
        manifest = json.loads((ROOT / "adapters/manifest.json").read_text())
        self.assertIs(manifest["weights_distributed"], True)
        self.assertEqual(manifest["weights_directory"], "weights")
        entries = {}
        for entry in manifest["checkpoints"]:
            name = entry["filename"]
            self.assertEqual(Path(name).name, name)
            self.assertEqual(Path(name).suffix, ".safetensors")
            relative = Path("adapters/weights") / name
            self.assertNotIn(relative, entries)
            entries[relative] = entry
        return entries

    def assert_released_weight(self, path, entries):
        relative = path.relative_to(ROOT)
        self.assertIn(relative, entries, f"Unlisted model weight: {relative}")
        for component in (path, path.parent, path.parent.parent):
            self.assertFalse(component.is_symlink(), f"Symlink in weight path: {relative}")
        entry = entries[relative]
        self.assertEqual(path.stat().st_size, entry["bytes"])
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"])
        return raw

    def test_python_syntax(self):
        for path in release_files():
            if path.suffix == ".py":
                with self.subTest(path=str(path.relative_to(ROOT))):
                    ast.parse(path.read_text(), filename=str(path))

    def test_json_is_finite(self):
        def reject_constant(value):
            raise ValueError(f"Nonfinite JSON value: {value}")
        for path in release_files():
            if path.suffix == ".json":
                with self.subTest(path=str(path.relative_to(ROOT))):
                    json.loads(path.read_text(), parse_constant=reject_constant)

    def test_no_model_weights_or_source_media(self):
        entries = self.weight_entries()
        excluded = {".mp4", ".mov", ".avi", ".jpg", ".jpeg", ".webp",
                    ".safetensors", ".pt", ".pth", ".ckpt", ".sif", ".pem", ".key"}
        for path in release_files():
            with self.subTest(path=str(path.relative_to(ROOT))):
                if path.suffix.lower() == ".safetensors":
                    self.assert_released_weight(path, entries)
                else:
                    self.assertNotIn(path.suffix.lower(), excluded)
                self.assertLess(path.stat().st_size, 10 * 1024 * 1024)

    def test_no_credentials_or_private_machine_paths(self):
        entries = self.weight_entries()
        patterns = [
            r"AKIA[A-Z0-9]{16}",
            r"ASIA[A-Z0-9]{16}",
            r"gh[pousr]_[A-Za-z0-9]{30,}",
            r"github_pat_[A-Za-z0-9_]{40,}",
            r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",
            r"/(?:Users|home)/[A-Za-z][^\s\"']*/",
            r"/scratch/[a-z]/[A-Za-z0-9_-]+/",
        ]
        for path in release_files():
            if path.suffix in {".png", ".pdf", ".woff", ".woff2"}:
                continue
            if path.suffix.lower() == ".safetensors":
                raw = self.assert_released_weight(path, entries)
                self.assertGreaterEqual(len(raw), 8)
                size = struct.unpack("<Q", raw[:8])[0]
                self.assertGreaterEqual(size, 2)
                self.assertLessEqual(size, 16384)
                self.assertLessEqual(8 + size, len(raw))
                # Parse and re-encode to expose JSON-escaped private strings too.
                header = json.loads(raw[8:8 + size])
                self.assertIsInstance(header, dict)
                text = json.dumps(header, ensure_ascii=False)
            else:
                try:
                    text = path.read_text()
                except UnicodeDecodeError:
                    self.fail(f"Unexpected binary file: {path.relative_to(ROOT)}")
            for pattern in patterns:
                with self.subTest(path=str(path.relative_to(ROOT)), pattern=pattern):
                    self.assertIsNone(re.search(pattern, text))


if __name__ == "__main__":
    unittest.main()
