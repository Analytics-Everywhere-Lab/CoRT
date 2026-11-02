"""Self-contained preparation tests; no dataset, image decoder or models needed."""

import base64
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from research.lifecycle_memory.temporal_forecasting import data


# A real 1x1 PNG, not a synthetic scientific dataset or an image decoder stub.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)


class TemporalForecastingDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.image = self.source / "future_answer_mug.png"
        self.image.write_bytes(PNG)
        self.observations = self.source / "observations.jsonl"
        self.labels = self.source / "labels.jsonl"
        self.output = self.root / "release"
        self.exposure = self.source / "exposure.json"
        self.rows = [self.observation()]
        self.keys = [self.label()]
        self.write_inputs()

    def observation(self, episode="source_episode", split="pilot", suffix=""):
        return {
            "episode_id": episode, "cluster_id": "source_wearer" + suffix,
            "video_id": "source_video" + suffix, "split": split,
            "candidate_pair_id": "source_pair" + suffix,
            "pair_frozen_before_label_review": True, "annotation_source": "ego4d_sta",
            "observation_times_s": {"past1": 0, "past2": 1, "current": 2,
                                    "candidate0": 2, "candidate1": 2},
            "timestamps_verified": True, "candidates_visible_at_cutoff": True,
            "event_id": "source_event" + suffix, "source": "restricted source description",
            "images": {slot: {"path": self.image.name} for slot in data.SLOTS},
        }

    def label(self, episode="source_episode", **changes):
        return {"episode_id": episode, "correct_candidate": 1, "time_to_contact_s": 0.5,
                "label_status": "official", **changes}

    def write_jsonl(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def write_inputs(self):
        self.write_jsonl(self.observations, self.rows)
        self.write_jsonl(self.labels, self.keys)

    def registry(self, **fields):
        self.exposure.write_text(json.dumps({"version": 1, "participants": [],
                                             "videos": [], **fields}), encoding="utf-8")
        return self.exposure

    def prepare(self, exposure=None):
        return data.prepare_dataset(self.observations, self.labels, self.output, exposure)

    def rejected(self, exposure=None):
        with self.assertRaises((ValueError, OSError)):
            self.prepare(exposure)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".release.prepare*")), [])

    def manifest(self):
        return json.loads((self.output / "manifest.json").read_text())

    def save_manifest(self, manifest):
        (self.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def rewrite_metadata(self, filename, rows):
        path = self.output / filename
        self.write_jsonl(path, rows)
        manifest = self.manifest()
        key = "public_sha256" if filename == "public.jsonl" else "evaluator_sha256"
        manifest[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_manifest(manifest)

    def test_embedded_png_is_a_valid_one_pixel_fixture(self):
        self.assertEqual(PNG[:8], b"\x89PNG\r\n\x1a\n")
        offset = 8
        chunks = {}
        while offset < len(PNG):
            length = struct.unpack(">I", PNG[offset:offset + 4])[0]
            chunk = PNG[offset + 4:offset + 8 + length]
            checksum = struct.unpack(">I", PNG[offset + 8 + length:offset + 12 + length])[0]
            self.assertEqual(zlib.crc32(chunk), checksum)
            chunks[chunk[:4]] = chunk[4:]
            offset += length + 12
        self.assertEqual(struct.unpack(">II", chunks[b"IHDR"][:8]), (1, 1))
        self.assertEqual(len(zlib.decompress(chunks[b"IDAT"])), 3)

    def test_round_trip_opaque_portable_private_and_sources_unchanged(self):
        self.rows[0]["images"]["past1"]["sha256"] = hashlib.sha256(PNG).hexdigest()
        self.keys[0].update(source="private official annotation", event_id="source_event")
        self.write_inputs()
        before = {path.name: path.read_bytes() for path in self.source.iterdir()}
        manifest = self.prepare()
        result = data.validate_dataset(self.output, verify_assets=True)
        self.assertEqual(set(result), {"manifest", "rows"})
        self.assertEqual(manifest, result["manifest"])
        self.assertEqual(manifest["counts"], {"episodes": 1, "participants": 1, "videos": 1})
        self.assertEqual(len(manifest["asset_inventory"]), 1)
        public = (self.output / "public.jsonl").read_text()
        for secret in ("source_episode", "source_wearer", "source_video", "source_pair",
                       "source_event", "future_answer_mug", "time_to_contact_s",
                       "correct_candidate", "label_status", "restricted source description"):
            self.assertNotIn(secret, public)
            self.assertNotIn('"' + secret + '"', json.dumps(result))
        private = json.loads((self.output / "evaluator.jsonl").read_text())
        self.assertEqual(private["episode_id"], result["rows"][0]["episode_id"])
        self.assertEqual(private["event_id"], result["rows"][0]["event_id"])
        self.assertEqual(private["correct_candidate"], 1)
        self.assertEqual((self.output / "evaluator.jsonl").stat().st_mode & 0o777, 0o600)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.source.iterdir()})
        relocated = self.root / "portable"
        self.output.rename(relocated)
        shutil.rmtree(self.source)
        self.assertEqual(data.validate_dataset(relocated, verify_assets=True), result)

    def test_splits_grouping_hashes_and_explicit_attrition(self):
        self.rows += [self.observation("e2", "pilot"),
                      self.observation("e3", "exploration", "2"),
                      self.observation("e4", "confirmation", "3")]
        self.keys += [self.label("e2", label_status="audited"), self.label("e3"), self.label("e4")]
        self.write_inputs()
        manifest = self.prepare(self.registry())
        pilot = data.validate_dataset(self.output, split="pilot")["rows"]
        self.assertEqual(len(pilot), 2)
        self.assertEqual(pilot[0]["cluster_id"], pilot[1]["cluster_id"])
        self.assertEqual(manifest["counts"]["participants"], 3)
        self.assertEqual(manifest["splits"]["pilot"]["public_sha256"],
                         hashlib.sha256(data._lines(pilot)).hexdigest())
        audit = manifest["provenance"]["attrition"]
        self.assertEqual(audit["accepted"], 4)
        self.assertEqual(audit["rejected"], 0)
        self.assertEqual(audit["label_status_counts"], {"official": 3, "audited": 1})
        with self.assertRaises(ValueError):
            data.validate_dataset(self.output, split="test")

    def test_reject_label_errors(self):
        cases = [[], [self.label("extra")], [self.label(), self.label()],
                 [self.label(), self.label("extra")]]
        cases += [[self.label(**{field: value})] for field, values in {
            "correct_candidate": [True, -1, 2, 1.0, "1"],
            "time_to_contact_s": [0, -1, True, "0.5", None, float("nan"), float("inf")],
            "label_status": ["guessed", "synthetic", None],
        }.items() for value in values]
        cases += [[{"episode_id": "source_episode"}], [self.label(answer="mug")]]
        for labels in cases:
            with self.subTest(labels=labels):
                self.keys = labels
                self.write_inputs()
                self.rejected()

    def test_timestamps_and_manual_attestations_are_required(self):
        for field in ("observation_times_s", "timestamps_verified", "candidates_visible_at_cutoff"):
            self.rows = [self.observation()]
            del self.rows[0][field]
            self.write_inputs()
            self.rejected()
        for field in ("timestamps_verified", "candidates_visible_at_cutoff"):
            for value in (False, 1, "true", None):
                with self.subTest(field=field, value=value):
                    self.rows = [self.observation()]
                    self.rows[0][field] = value
                    self.write_inputs()
                    self.rejected()

    def test_timestamps_require_exact_slots_and_finite_nonnegative_numbers(self):
        for slot in data.SLOTS:
            for value in (-1, True, "1", None, float("nan"), float("inf"), float("-inf")):
                with self.subTest(slot=slot, value=value):
                    self.rows = [self.observation()]
                    self.rows[0]["observation_times_s"][slot] = value
                    self.write_inputs()
                    self.rejected()
            self.rows = [self.observation()]
            del self.rows[0]["observation_times_s"][slot]
            self.write_inputs()
            self.rejected()
        self.rows = [self.observation()]
        self.rows[0]["observation_times_s"]["future_contact"] = 2.5
        self.write_inputs()
        self.rejected()

    def test_timestamp_order_and_non_cutoff_candidates_rejected(self):
        cases = [{"past1": 1}, {"past1": 1.5}, {"past2": 0}, {"past2": 2},
                 {"current": 0.5}, {"candidate0": 2.01}, {"candidate1": 3},
                 {"candidate0": 0}, {"candidate1": 1.5},
                 {"candidate0": 1.9999999999999998},
                 {"candidate0": 1, "candidate1": 1}]
        for changes in cases:
            with self.subTest(changes=changes):
                self.rows = [self.observation()]
                self.rows[0]["observation_times_s"].update(changes)
                self.write_inputs()
                self.rejected()

    def test_valid_observation_times_are_public_without_future_labels(self):
        self.rows[0]["observation_times_s"].update(past2=0.25, candidate0=2.0, candidate1=2.0)
        self.write_inputs()
        self.prepare()
        row = data.validate_dataset(self.output, verify_assets=True)["rows"][0]
        self.assertEqual(row["observation_times_s"], self.rows[0]["observation_times_s"])
        self.assertIs(row["timestamps_verified"], True)
        self.assertIs(row["candidates_visible_at_cutoff"], True)
        self.assertNotIn("time_to_contact_s", row)

    def test_validation_rechecks_times_even_when_metadata_rehashed(self):
        self.prepare()
        original = data.validate_dataset(self.output)["rows"]
        for slot, value in (("past1", 1), ("candidate0", 3), ("past2", -1),
                            ("candidate0", 1), ("candidate1", 1.5)):
            rows = json.loads(json.dumps(original))
            rows[0]["observation_times_s"][slot] = value
            self.rewrite_metadata("public.jsonl", rows)
            with self.assertRaises(ValueError):
                data.validate_dataset(self.output, split="exploration")

    def test_reject_leaking_or_invalid_observation_fields(self):
        mutations = [(field, value) for field, value in {
            "correct_candidate": 1, "time_to_contact_s": 0.5, "target_name": "mug",
            "narration": "future action", "split": "test", "cluster_id": "unknown",
            "video_id": "", "annotation_source": "synthetic",
            "pair_frozen_before_label_review": 1,
        }.items()]
        for field, value in mutations:
            with self.subTest(field=field):
                self.rows = [self.observation()]
                self.rows[0][field] = value
                self.write_inputs()
                self.rejected()
        self.rows = [self.observation()]
        self.rows[0]["images"]["current"]["target_name"] = "mug"
        self.write_inputs()
        self.rejected()

    def test_missing_slots_and_duplicate_or_empty_observations(self):
        for rows in ([], [self.observation(), self.observation()]):
            self.rows = rows
            self.write_inputs()
            self.rejected()
        for slot in data.SLOTS:
            self.rows = [self.observation()]
            del self.rows[0]["images"][slot]
            self.write_inputs()
            self.rejected()

    def test_cross_split_identity_overlap(self):
        for field in ("cluster_id", "video_id", "candidate_pair_id", "event_id"):
            with self.subTest(field=field):
                self.rows = [self.observation(), self.observation("e2", "exploration", "2")]
                self.rows[1][field] = self.rows[0][field]
                self.keys = [self.label(), self.label("e2")]
                self.write_inputs()
                self.rejected()

    def test_confirmation_requires_explicit_registry_and_rejects_exposure(self):
        self.rows[0]["split"] = "confirmation"
        self.write_inputs()
        self.rejected()
        for field, value in {
            "participants": "source_wearer", "videos": "source_video",
            "episodes": "source_episode", "events": "source_event",
            "candidate_pairs": "source_pair", "asset_sha256": hashlib.sha256(PNG).hexdigest(),
        }.items():
            with self.subTest(field=field):
                self.rejected(self.registry(**{field: [value]}))
        for value in ({"wearers": []}, {"version": 1, "participants": []},
                      {"version": True, "participants": [], "videos": []},
                      {"version": 1, "participants": {}, "videos": []},
                      {"version": 1, "participants": ["p", "p"], "videos": []}):
            self.exposure.write_text(json.dumps(value))
            self.rejected(self.exposure)

    def test_invalid_registry_is_not_ignored_for_pilot(self):
        self.rejected(self.registry(old_schema=[]))

    def test_asset_escape_symlink_extension_and_hash(self):
        external = self.root / "outside.png"
        external.write_bytes(PNG)
        symlink = self.source / "link.png"
        symlink.symlink_to(self.image)
        directory_link = self.source / "linked"
        directory_link.symlink_to(self.source, target_is_directory=True)
        unsupported = self.source / "image.svg"
        unsupported.write_bytes(PNG)
        empty = self.source / "empty.png"
        empty.touch()
        for path in ("../outside.png", str(external), "link.png", "linked/" + self.image.name,
                     "image.svg", "empty.png", "missing.png", "."):
            with self.subTest(path=path):
                self.rows[0]["images"]["current"] = {"path": path}
                self.write_inputs()
                self.rejected()
        self.rows[0]["images"]["current"] = {"path": str(self.image), "sha256": "0" * 64}
        self.write_inputs()
        self.rejected()

    def test_absolute_confined_source_paths_are_supported(self):
        self.rows[0]["images"]["current"]["path"] = str(self.image.resolve())
        self.write_inputs()
        self.prepare()
        data.validate_dataset(self.output, verify_assets=True)

    def test_event_label_mismatch_rejected(self):
        self.keys[0]["event_id"] = "different_event"
        self.write_inputs()
        self.rejected()

    def test_json_duplicate_keys_and_blank_lines_rejected(self):
        original = self.observations.read_text()
        for raw in ('{"episode_id":"a","episode_id":"b"}\n', original + "\n", "[]\n"):
            self.observations.write_text(raw)
            self.rejected()

    def test_physical_tampering_and_optional_hashing(self):
        manifest = self.prepare()
        asset = self.output / manifest["asset_inventory"][0]["path"]
        asset.write_bytes(bytes([PNG[0] ^ 1]) + PNG[1:])
        data.validate_dataset(self.output)
        with self.assertRaisesRegex(ValueError, "hash"):
            data.validate_dataset(self.output, verify_assets=True)
        asset.write_bytes(PNG + b"extra")
        with self.assertRaisesRegex(ValueError, "size"):
            data.validate_dataset(self.output)
        asset.unlink()
        with self.assertRaisesRegex(ValueError, "inventory"):
            data.validate_dataset(self.output)

    def test_extra_assets_and_asset_symlinks_rejected_without_hashing(self):
        manifest = self.prepare()
        extra = self.output / "assets" / "extra.png"
        extra.write_bytes(PNG)
        with self.assertRaises(ValueError):
            data.validate_dataset(self.output)
        extra.unlink()
        asset = self.output / manifest["asset_inventory"][0]["path"]
        asset.unlink()
        asset.symlink_to(self.image)
        with self.assertRaises(ValueError):
            data.validate_dataset(self.output)

    def test_metadata_hashes_always_verified(self):
        self.prepare()
        for filename in ("public.jsonl", "evaluator.jsonl"):
            with self.subTest(filename=filename):
                path = self.output / filename
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "hash"):
                    data.validate_dataset(self.output)
                path.write_bytes(original)

    def test_manifest_inventory_splits_counts_and_attrition_are_validated(self):
        self.prepare()
        original = self.manifest()
        for field in ("counts", "splits", "asset_inventory", "version"):
            changed = dict(original)
            changed[field] = []
            self.save_manifest(changed)
            with self.assertRaises(ValueError):
                data.validate_dataset(self.output)
        self.save_manifest(original)
        changed = self.manifest()
        changed["provenance"]["attrition"]["rejected"] = 1
        self.save_manifest(changed)
        with self.assertRaisesRegex(ValueError, "attrition"):
            data.validate_dataset(self.output)

    def test_public_semantic_validation_even_after_rehashing(self):
        self.prepare()
        original = data.validate_dataset(self.output)["rows"]
        mutations = [("correct_candidate", 1), ("episode_id", "answer_mug"),
                     ("source", "target mug"), ("pair_frozen_before_label_review", False)]
        for field, value in mutations:
            rows = json.loads(json.dumps(original))
            rows[0][field] = value
            self.rewrite_metadata("public.jsonl", rows)
            with self.assertRaises(ValueError):
                data.validate_dataset(self.output, split="exploration")

    def test_private_schema_validated_even_after_rehashing(self):
        self.prepare()
        labels = [json.loads((self.output / "evaluator.jsonl").read_text())]
        labels[0]["time_to_contact_s"] = -1
        self.rewrite_metadata("evaluator.jsonl", labels)
        with self.assertRaises(ValueError):
            data.validate_dataset(self.output)

    def test_exposure_is_rechecked_during_validation(self):
        self.rows[0]["split"] = "confirmation"
        self.write_inputs()
        self.prepare(self.registry())
        row = data.validate_dataset(self.output)["rows"][0]
        manifest = self.manifest()
        manifest["provenance"]["exposure"]["participants"] = [row["cluster_id"]]
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "exposed"):
            data.validate_dataset(self.output, split="pilot")
        manifest["provenance"].update(exposure=None, exposure_sha256=None)
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "requires"):
            data.validate_dataset(self.output)

    def test_existing_readonly_or_empty_output_never_modified(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):
            self.prepare()
        sentinel = self.output / "sentinel"
        sentinel.write_bytes(b"unchanged")
        self.output.chmod(0o500)
        try:
            with self.assertRaises(ValueError):
                self.prepare()
            self.assertEqual(sentinel.read_bytes(), b"unchanged")
        finally:
            self.output.chmod(0o700)

    def test_broken_output_symlink_never_replaced(self):
        self.output.symlink_to(self.root / "nonexistent")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertTrue(self.output.is_symlink())

    def test_failed_copy_or_publication_leaves_no_partial_release(self):
        for operation in ("copyfile", "rename"):
            target = data.shutil if operation == "copyfile" else data.os
            with self.subTest(operation=operation), patch.object(
                target, operation, side_effect=OSError("injected failure")
            ):
                self.rejected()
        self.assertEqual(self.image.read_bytes(), PNG)

    def test_copy_corruption_fails_before_publication(self):
        def corrupt(source, target):
            Path(target).write_bytes(b"corrupted")

        with patch.object(data.shutil, "copyfile", side_effect=corrupt):
            self.rejected()

    def test_output_appearing_during_build_is_not_overwritten(self):
        real_validate = data.validate_dataset

        def race(*args, **kwargs):
            result = real_validate(*args, **kwargs)
            self.output.mkdir()
            (self.output / "other-owner").write_bytes(b"keep")
            return result

        with patch.object(data, "validate_dataset", side_effect=race):
            with self.assertRaises(ValueError):
                self.prepare()
        self.assertEqual((self.output / "other-owner").read_bytes(), b"keep")
        self.assertEqual(list(self.root.glob(".release.prepare*")), [])

    def test_existing_publication_lock_is_not_removed(self):
        lock = self.root / ".release.prepare.lock"
        lock.write_bytes(b"other-owner")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(lock.read_bytes(), b"other-owner")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
