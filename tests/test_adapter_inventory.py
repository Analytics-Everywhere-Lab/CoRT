"""Verify released adapter bytes and safetensors validation using only the stdlib."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from export_adapter_inventory import inspect

# Canonical JSON digest of the original, metadata-only manifest's checkpoint list.
ORIGINAL_CHECKPOINTS_SHA256 = "ebe3983551404c3abc1324c859bb0a97c100a42f906762c67db8419476110e47"


class AdapterInventoryTests(unittest.TestCase):
    def test_inventory(self):
        data = json.loads((ROOT / "adapters/manifest.json").read_text())
        entries = data["checkpoints"]
        self.assertEqual(len(entries), 16)
        self.assertEqual(data["checkpoint_count"], 16)
        self.assertEqual(data["total_checkpoint_bytes"], 2759024)
        self.assertEqual(len({entry["sha256"] for entry in entries}), 16)
        self.assertEqual(sum(entry["bytes"] for entry in entries), 2759024)
        self.assertIs(data["weights_distributed"], True)
        self.assertEqual(data["weights_directory"], "weights")
        canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), ORIGINAL_CHECKPOINTS_SHA256)
        self.assertEqual({entry["seed"] for entry in entries}, {17, 29})
        weights = ROOT / "adapters/weights"
        self.assertTrue(weights.is_dir())
        self.assertFalse(weights.is_symlink())
        expected_paths = {weights / entry["filename"] for entry in entries}
        self.assertEqual(set(weights.iterdir()), expected_paths)
        self.assertEqual(set((ROOT / "adapters").rglob("*.safetensors")), expected_paths)
        self.assertEqual(sum(path.stat().st_size for path in expected_paths), 2759024)
        for entry in entries:
            with self.subTest(filename=entry["filename"]):
                self.assertEqual(entry["parameters"], 43008)
                self.assertEqual(entry["rank"], 4)
                self.assertEqual(entry["training_steps"], 256)
                self.assertEqual(entry["cut"], 34)
                self.assertRegex(entry["sha256"], r"^[a-f0-9]{64}$")
                self.assertEqual(Path(entry["filename"]).name, entry["filename"])
                path = weights / entry["filename"]
                self.assertFalse(path.is_symlink())
                raw = path.read_bytes()
                self.assertEqual(len(raw), entry["bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"])
                self.assertEqual(inspect(path), entry)


class AdapterInspectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "native__correct_L34_CQ.seed17.final.safetensors"
        self.header = {
            "__metadata__": {"steps": "256", "layer_index": "34", "positions": "CQ",
                             "seed": "17", "rank": "4", "selection": "fixed_final_step",
                             "compression_id": "native", "canary": "False",
                             "target": "native__correct_L34_CQ", "legacy_driver_target": "l34_cq"},
            "down.weight": {"dtype": "F32", "shape": [4, 5376], "data_offsets": [0, 86016]},
            "up.weight": {"dtype": "F32", "shape": [5376, 4], "data_offsets": [86016, 172032]},
        }
        self.payload = struct.pack("<f", 0.25) * 43008

    def write_checkpoint(self, header=None, payload=None):
        encoded = json.dumps(self.header if header is None else header).encode()
        encoded += b" " * (-len(encoded) % 8)
        raw = struct.pack("<Q", len(encoded)) + encoded
        raw += self.payload if payload is None else payload
        self.path.write_bytes(raw)
        return raw

    def test_valid_checkpoint_returns_exact_entry(self):
        raw = self.write_checkpoint()
        self.assertEqual(inspect(self.path), {
            "filename": self.path.name, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "compression": "native", "positions": "CQ", "seed": 17,
            "cut": 34, "rank": 4, "training_steps": 256, "selection": "fixed_final_step",
            "parameters": 43008,
            "tensors": {name: {"dtype": value["dtype"], "shape": value["shape"]}
                        for name, value in self.header.items() if name != "__metadata__"},
        })

    def test_rejects_extra_missing_or_changed_metadata(self):
        variants = [{**self.header["__metadata__"], "extra": "unexpected"}]
        for key in self.header["__metadata__"]:
            missing = dict(self.header["__metadata__"])
            del missing[key]
            variants.extend([missing, {**self.header["__metadata__"], key: "unexpected"}])
        for metadata in variants:
            with self.subTest(metadata=metadata):
                self.write_checkpoint({**self.header, "__metadata__": metadata})
                with self.assertRaises(ValueError):
                    inspect(self.path)

    def test_rejects_tensor_inventory_and_geometry_changes(self):
        variants = []
        for name in ("down.weight", "up.weight"):
            missing = copy.deepcopy(self.header)
            del missing[name]
            variants.append((f"missing {name}", missing))
            wrong_shape = [5376, 4] if name == "down.weight" else [4, 5376]
            for field, value in (("dtype", "I32"), ("shape", wrong_shape),
                                 ("data_offsets", [0, 172032]), ("extra", "unexpected")):
                changed = copy.deepcopy(self.header)
                changed[name][field] = value
                variants.append((f"{name} {field}", changed))
        variants.append(("extra tensor", {**self.header, "extra.weight": self.header["up.weight"]}))
        for label, header in variants:
            with self.subTest(case=label):
                self.write_checkpoint(header)
                with self.assertRaises(ValueError):
                    inspect(self.path)

    def test_rejects_nonfinite_parameters_in_either_tensor(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for offset in (0, 86016, len(self.payload) - 4):
                with self.subTest(value=value, offset=offset):
                    payload = bytearray(self.payload)
                    struct.pack_into("<f", payload, offset, value)
                    self.write_checkpoint(payload=payload)
                    with self.assertRaises(ValueError):
                        inspect(self.path)

    def test_rejects_truncation_and_trailing_payload(self):
        raw = self.write_checkpoint()
        header_end = 8 + struct.unpack("<Q", raw[:8])[0]
        cases = {"empty": b"", "prefix": raw[:7], "header": raw[:header_end - 1],
                 "no payload": raw[:header_end], "partial float": raw[:-1],
                 "missing float": raw[:-4], "trailing byte": raw + b"\0",
                 "trailing float": raw + struct.pack("<f", 0.0)}
        for label, malformed in cases.items():
            with self.subTest(case=label):
                self.path.write_bytes(malformed)
                with self.assertRaises(ValueError):
                    inspect(self.path)

    def test_rejects_symlink(self):
        self.write_checkpoint()
        directory = self.path.parent / "links"
        directory.mkdir()
        link = directory / self.path.name
        link.symlink_to(self.path)
        with self.assertRaises(ValueError):
            inspect(link)


if __name__ == "__main__":
    unittest.main()
