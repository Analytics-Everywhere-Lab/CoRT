"""Bounded CPU-only contract tests; no transformers or model downloads."""
import ast
from collections import Counter, UserDict
import copy
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from research.lifecycle_memory.application_pilot_compression import CompressionPlan, compression_context
from research.lifecycle_memory.decoder_token_matrix import (
    Arm, Stage, build_arms, build_schedule, compact_states, make_schedule,
    matrix_context, protection_receipt, remap_positions,
)


torch.set_num_threads(1)


def fixture(production=False, kernel=1):
    grids = ((19, 14), (16, 16), (16, 16)) if production else ((4, 2), (2, 2), (2, 2))
    cursor, slots, image_positions = 1, [], []
    for width, height in grids:
        slots.append(tuple(range(cursor, cursor + width * height)))
        cursor += width * height
        image_positions.append([(x, y) for y in range(height * kernel) for x in range(width * kernel)])
    max_patches = max(map(len, image_positions))
    coords = torch.tensor([p + [(-1, -1)] * (max_patches - len(p)) for p in image_positions])
    q = tuple(range(cursor, cursor + (75 if production else 4)))
    cursor += len(q)
    r = (0,) + tuple(range(cursor, cursor + (64 if production else 2)))
    cursor += len(r) - 1
    p = tuple(range(cursor, cursor + (3 if production else 2)))
    n = cursor + len(p)

    def group(positions):
        return SimpleNamespace(positions=positions, count=len(positions))

    partition = SimpleNamespace(sequence_length=n, image_slots=tuple(map(group, slots)),
                                full_frame=group(slots[0]), candidate_crops=group(slots[1] + slots[2]),
                                literal_user_prompt=group(q), structural_remainder=group(r),
                                answer_prefix=group(p))
    return partition, coords


def payload_fixture():
    partition, coords = fixture(production=True, kernel=3)
    ids = torch.full((1, 921), 10)
    payload = list(partition.full_frame.positions + partition.candidate_crops.positions)
    ids[0, payload] = 258880
    ids[0, 0] = 2
    boundaries = list(partition.structural_remainder.positions[1:7])
    ids[0, boundaries] = torch.tensor([255999, 258882] * 3)
    prepared = SimpleNamespace(pilot_partition=partition, choices={"A": SimpleNamespace(
        inputs={"input_ids": ids, "image_position_ids": coords})})
    tokenizer = SimpleNamespace(all_special_ids=[2, 255999, 258882, 258880, 107],
                                bos_token_id=2, boi_token_id=255999, eoi_token_id=258882,
                                image_token_id=258880)
    return prepared, tokenizer


class Block(torch.nn.Module):
    def __init__(self, local=False, fail=False):
        super().__init__()
        self.self_attn = SimpleNamespace(layer_type="sliding_attention" if local else "full_attention")
        self.seen = None
        self.fail = fail

    def forward(self, hidden_states, per_layer_input=None, **kwargs):
        self.seen = kwargs
        if self.fail:
            raise RuntimeError("intentional block failure")
        return hidden_states + 1


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.partition, self.coords = fixture()
        self.n = self.partition.sequence_length

    def schedule(self, stages=(), protected=()):
        return build_schedule(partition=self.partition, image_position_ids=self.coords,
                              arm=Arm("toy", tuple(stages)), protected_positions=protected, pooling_kernel=1)

    def side_inputs(self):
        return dict(position_ids=torch.arange(self.n)[None],
                    position_embeddings=(torch.arange(self.n * 2).view(1, self.n, 2).float(),) * 2,
                    attention_mask=torch.arange(self.n**2).view(1, 1, self.n, self.n),
                    cache_position=torch.arange(self.n), past_key_values=None, use_cache=False)

    def run_layers(self, layers, hidden=None, side=None):
        hidden = torch.arange(self.n * 2).view(1, self.n, 2).float() if hidden is None else hidden
        side = self.side_inputs() if side is None else side
        for layer in layers:
            hidden = layer(hidden, None, **side)
        return hidden

    def assert_clean(self, layers):
        for layer in layers:
            self.assertFalse(layer._forward_hooks)
            self.assertFalse(layer._forward_pre_hooks)
            self.assertFalse(hasattr(layer, "_decoder_token_matrix_active"))

    def test_inventory_counts_order_memberships_and_no_duplicate_conditions(self):
        arms = build_arms()
        self.assertEqual(arms, build_arms(config={"ignored": True}))
        self.assertEqual(json.loads(json.dumps(arms)), arms)
        self.assertEqual(len(arms), 71)
        self.assertEqual(len({a["id"] for a in arms}), 71)
        self.assertEqual(len({json.dumps(a["stages"], sort_keys=True) for a in arms}), 71)
        self.assertEqual(Counter(a["family"] for a in arms),
                         {"native": 1, "late_subset": 30, "F_only": 22, "F_drop_all": 4,
                          "progressive_F": 4, "early_C": 2, "early_C_late_F": 4, "matched_total_C": 4})
        subsets = [a for a in arms if a["family"] == "late_subset"]
        expected = {subset for n in range(1, 5) for subset in itertools.combinations("FCQR", n)}
        self.assertEqual({tuple(a["subset"]) for a in subsets}, expected)
        for method in ("merge", "prune"):
            shared = next(a for a in subsets if a["subset"] == ["F"] and a["stages"][0]["method"] == method)
            self.assertEqual(shared["stages"][0]["targets"], {"F": 133})
            self.assertEqual(shared["group_memberships"], ["late_subset", "F_only"])
        for arm in subsets:
            if arm["stages"][0]["method"] == "merge" and set(arm["subset"]) & set("QR"):
                self.assertIn("QRprune", arm["id"])
                for role in set(arm["subset"]) & set("QR"):
                    self.assertEqual(arm["operators_by_stage"][0][role], "position_spread_prune")

    def test_all_71_production_plans_and_crop_balance(self):
        partition, coords = fixture(production=True, kernel=3)
        for arm in build_arms():
            with self.subTest(arm=arm["id"]):
                schedule = build_schedule(partition=partition, image_position_ids=coords,
                                          arm=arm, protected_positions=[0, 854], pooling_kernel=3)
                self.assertEqual(schedule.original_length, 921)
                for stage in schedule.stages:
                    self.assertEqual(stage.keep, tuple(sorted(set(stage.keep))))
                    self.assertTrue(set(partition.answer_prefix.positions).issubset(stage.keep))
                    for group in stage.groups:
                        self.assertEqual(group[-1], max(group))
                        if len(group) > 1:
                            self.assertEqual(sum(set(group).issubset(slot.positions) for slot in partition.image_slots), 1)
                    if "C" in stage.role_counts:
                        per_slot = stage.role_counts["C"]["per_slot_retained"]
                        actual = [sum(p in stage.keep for p in slot.positions)
                                  for slot in partition.image_slots[1:]]
                        self.assertEqual(actual, per_slot)
                        self.assertLessEqual(abs(per_slot[0] - per_slot[1]), 1)
                        if arm["family"] == "matched_total_C":
                            self.assertEqual(per_slot, [190, 189])
                            self.assertEqual(len(stage.groups), 788)
                        elif arm["family"] in ("early_C", "early_C_late_F"):
                            self.assertEqual(per_slot[0], per_slot[1])

    def test_protection_counts_text_pruning_and_duplicate_remap(self):
        q = self.partition.literal_user_prompt.positions
        r = self.partition.structural_remainder.positions
        protected = [q[0], q[0], r[1]]
        schedule = self.schedule([Stage(1, fractions={"Q": .5, "R": .5})], protected)
        stage = schedule.stages[0]
        self.assertEqual(stage.role_counts["Q"]["eligible_before"], 3)
        self.assertEqual(stage.role_counts["Q"]["eligible_retained"], 2)
        self.assertEqual(stage.role_counts["R"]["protected"], 2)
        self.assertEqual(stage.role_counts["R"]["eligible_before"], 1)
        self.assertEqual(stage.role_counts["Q"]["effective_method"], "prune")
        self.assertTrue(all(len(group) == 1 for group in stage.groups))
        for p in set(protected) | set(self.partition.answer_prefix.positions) | {0, self.n - 1}:
            self.assertIn((p,), stage.groups)
        dropped = next(p for p in q if p not in stage.keep)
        self.assertEqual(schedule.remap_positions([dropped, dropped], 1), (None, None))
        self.assertEqual(remap_positions(schedule, [q[0], q[0]], -1), (q[0], q[0]))
        with self.assertRaisesRegex(ValueError, "zero eligible"):
            self.schedule([Stage(1, fractions={"R": .5})], r)

    def test_visual_protected_member_cannot_enter_mean(self):
        schedule = self.schedule([Stage(0, targets={"F": 2})], [1])
        stage = schedule.stages[0]
        self.assertEqual(stage.role_counts["F"]["after"], 3)
        self.assertIn((1,), stage.groups)
        self.assertTrue(all(1 not in group for group in stage.groups if len(group) > 1))

    def test_progressive_mass_weights_and_latest_original_representatives(self):
        schedule = self.schedule([Stage(0, targets={"F": 5}), Stage(2, targets={"F": 2})])
        x = torch.arange(self.n, dtype=torch.float32).square().view(1, self.n, 1)
        first = compact_states(x, schedule.stages[0])
        second = compact_states(first, schedule.stages[1])
        unequal = False
        for i, group in enumerate(schedule.stages[1].groups):
            self.assertAlmostEqual(second[0, i, 0].item(), x[0, list(group), 0].mean().item(), places=5)
            mapped = schedule.remap_positions(group + group[:1], 2)
            self.assertEqual(mapped, (i,) * (len(group) + 1))
            reduction = schedule.stages[1].reductions[i]
            unequal |= len({mass for _, mass in reduction}) > 1
        self.assertTrue(unequal, "fixture must detect unweighted mean-of-means")
        self.assertTrue(torch.equal(second[:, -1], x[:, -1]))
        # Evolved values from intervening blocks, not frozen original states.
        evolved = first + torch.arange(first.shape[1])[None, :, None]
        result = compact_states(evolved, schedule.stages[1])
        for i, reduction in enumerate(schedule.stages[1].reductions):
            total = sum(mass for _, mass in reduction)
            expected = sum(evolved[:, source] * (mass / total) for source, mass in reduction)
            torch.testing.assert_close(result[:, i], expected)

    def test_full_F_drop_and_prune_never_averages(self):
        for target in (0, 3):
            schedule = self.schedule([Stage(0, targets={"F": target}, method="prune")])
            x = torch.arange(self.n * 2).view(1, self.n, 2).float()
            stage = schedule.stages[0]
            self.assertTrue(torch.equal(compact_states(x, stage), x[:, stage.keep]))
            self.assertEqual(len(stage.groups), self.n - 8 + target)
            self.assertEqual(sum(p in stage.keep for p in self.partition.full_frame.positions), target)
            if target == 0:
                self.assertEqual(schedule.remap_positions(self.partition.full_frame.positions, 0), (None,) * 8)
        dropped = schedule.remap_positions(self.partition.full_frame.positions, 0)
        self.assertEqual(sum(x is None for x in dropped), 5)
        with self.assertRaisesRegex(ValueError, "zero retention"):
            self.schedule([Stage(0, targets={"F": 0})])

    def test_bfloat16_merge_preserves_untouched_rows_and_dtype(self):
        stage = self.schedule([Stage(0, targets={"F": 3})]).stages[0]
        x = torch.arange(self.n * 3).view(1, self.n, 3).to(torch.bfloat16)
        result = compact_states(x, stage)
        self.assertEqual(result.dtype, x.dtype)
        for i, group in enumerate(stage.groups):
            if len(group) == 1:
                self.assertTrue(torch.equal(result[:, i], x[:, group[0]]))
            else:
                expected = x[:, list(group)].float().mean(1).to(torch.bfloat16)
                torch.testing.assert_close(result[:, i], expected)

    def test_multistage_masks_rope_cache_positions_and_all_60_lengths(self):
        schedule = self.schedule([Stage(22, targets={"F": 5}), Stage(34, targets={"F": 2})])
        layers = torch.nn.ModuleList([Block(local=bool(i % 2)) for i in range(60)])
        side = self.side_inputs()
        original = side["attention_mask"].clone()
        with matrix_context(layers, schedule) as audit:
            output = self.run_layers(layers, side=side)
        self.assertEqual(audit["compression_count"], 2)
        self.assertEqual(audit["stage_counts"], {"22": 1, "34": 1})
        self.assertEqual(len(audit["sequence_lengths"]), 60)
        for i in range(60):
            expected = self.n if i <= 22 else self.n - (3 if i <= 34 else 6)
            self.assertEqual(audit["sequence_lengths"][str(i)], expected)
        for layer_index, stage in ((23, schedule.stages[0]), (35, schedule.stages[1])):
            seen = layers[layer_index].seen
            keep = list(stage.keep)
            self.assertEqual(seen["position_ids"].tolist(), [keep])
            self.assertEqual(seen["cache_position"].tolist(), keep)
            self.assertTrue(torch.equal(seen["position_embeddings"][0], side["position_embeddings"][0][:, keep]))
            self.assertTrue(torch.equal(seen["attention_mask"], original[:, :, keep][:, :, :, keep]))
        self.assertTrue(torch.equal(original, side["attention_mask"]))
        self.assertEqual(output.shape[1], self.n - 6)
        self.assert_clean(layers)

    def test_native_and_keep_all_bit_exact_no_side_input_changes(self):
        for stages in ([], [Stage(1, targets={"F": 8})]):
            schedule = self.schedule(stages)
            layers = torch.nn.ModuleList([Block() for _ in range(4)])
            side = self.side_inputs()
            baseline = self.run_layers(layers, side=side)
            with matrix_context(layers, schedule) as audit:
                actual = self.run_layers(layers, side=side)
            self.assertTrue(torch.equal(baseline, actual))
            self.assertEqual(audit["sequence_lengths"], {str(i): self.n for i in range(4)})
            self.assertIs(layers[-1].seen["attention_mask"], side["attention_mask"])
            self.assertIs(layers[-1].seen["position_embeddings"], side["position_embeddings"])
            self.assert_clean(layers)

    def test_implicit_full_mask_only_and_bad_side_geometry(self):
        schedule = self.schedule([Stage(0, targets={"F": 4})])
        layers = [Block(), Block()]
        side = self.side_inputs()
        side["attention_mask"] = None
        with matrix_context(layers, schedule):
            self.run_layers(layers, side=side)
        self.assertIsNone(layers[1].seen["attention_mask"])
        layers[1].self_attn.layer_type = "sliding_attention"
        with self.assertRaisesRegex(ValueError, "sliding mask"):
            with matrix_context(layers, schedule):
                self.run_layers(layers, side=side)
        self.assert_clean(layers)
        for key, bad in (("attention_mask", torch.ones(1, 1, 2, 2)),
                         ("position_ids", torch.arange(2)[None]),
                         ("position_embeddings", None)):
            side = self.side_inputs()
            side[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                with matrix_context(layers, schedule):
                    self.run_layers(layers, side=side)
            self.assert_clean(layers)

    def test_callbacks_preserved_on_exception_and_repeated_run_rejected(self):
        layers = [Block(), Block(fail=True), Block()]
        schedule = self.schedule([Stage(0, targets={"F": 4})])
        called = []
        handle = layers[0].register_forward_hook(lambda *args: called.append(True))
        existing = dict(layers[0]._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with matrix_context(layers, schedule):
                self.run_layers(layers)
        self.assertEqual(dict(layers[0]._forward_hooks), existing)
        self.assertEqual(called, [True])
        handle.remove()
        self.assert_clean(layers)
        layers[1].fail = False
        with self.assertRaisesRegex(ValueError, "one ordered forward"):
            with matrix_context(layers, schedule):
                self.run_layers(layers)
                self.run_layers(layers)
        self.assert_clean(layers)
        with self.assertRaisesRegex(ValueError, "exactly one forward"):
            with matrix_context(layers, schedule):
                pass
        self.assert_clean(layers)

    def test_cache_batch_and_stacked_contexts_rejected(self):
        layers = [Block(), Block()]
        schedule = self.schedule([Stage(0, targets={"F": 4})])
        for key, bad in (("past_key_values", object()), ("past_key_value", object()),
                         ("use_cache", True), ("per_layer_input", torch.ones(1))):
            side = self.side_inputs()
            side[key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unsupported"):
                with matrix_context(layers, schedule):
                    self.run_layers(layers, side=side)
            self.assert_clean(layers)
        with self.assertRaisesRegex(ValueError, "geometry"):
            with matrix_context(layers, schedule):
                self.run_layers(layers, hidden=torch.zeros(2, self.n, 2))
        with matrix_context(layers, schedule):
            with self.assertRaisesRegex(ValueError, "stacked"):
                with matrix_context(layers, schedule):
                    pass
            self.run_layers(layers)
        legacy = CompressionPlan(self.n, (), tuple(range(self.n)), tuple(range(self.n)))
        with compression_context(layers, legacy, cut=0):
            with self.assertRaisesRegex(ValueError, "stacked"):
                with matrix_context(layers, schedule):
                    pass
            self.run_layers(layers)
        with self.assertRaisesRegex(ValueError, "stacked"):
            with matrix_context(layers, schedule):
                with compression_context(layers, legacy, cut=0):
                    self.run_layers(layers)
        self.assert_clean(layers)

    def test_receipt_hash_and_geometry_are_stable_and_detached(self):
        schedule = self.schedule([Stage(0, targets={"F": 4})])
        receipt = schedule.receipt()
        digest = receipt.pop("sha256")
        encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertEqual(hashlib.sha256(encoded.encode("ascii")).hexdigest(), digest)
        self.assertEqual(schedule.receipt(), self.schedule([Stage(0, targets={"F": 4})]).receipt())
        receipt["arm"]["id"] = "edited"
        self.assertEqual(schedule.arm["id"], "toy")
        self.assertNotEqual(schedule.receipt()["sha256"], self.schedule([Stage(0, targets={"F": 4})], [1]).receipt()["sha256"])

    def test_invalid_partition_geometry_stages_and_remapping(self):
        bad = copy.deepcopy(self.partition)
        bad.full_frame.positions += (bad.full_frame.positions[-1],)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_schedule(partition=bad, image_position_ids=self.coords, arm=build_arms()[0], protected_positions=[])
        for stages in ([Stage(0, targets={"P": 1})], [Stage(0, targets={"F": 9})],
                       [Stage(2, targets={"F": 4}), Stage(2, targets={"F": 2})],
                       [Stage(60, targets={"F": 4})], [Stage(True, targets={"F": 4})],
                       [Stage(0, targets={"F": 2}, fractions={"F": .5})],
                       [Stage(0, fractions={"Q": float("nan")})],
                       [Stage(0, targets={"F": -1})], [Stage(0, targets={"F": 2}, method="bogus")]):
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                self.schedule(stages)
        coords = self.coords.clone()
        coords[0, 1] = coords[0, 0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_schedule(partition=self.partition, image_position_ids=coords,
                           arm=Arm("bad", (Stage(0, targets={"F": 4}),)), protected_positions=[], pooling_kernel=1)
        schedule = self.schedule()
        with self.assertRaises(ValueError):
            schedule.remap_positions([self.n], 0)
        with self.assertRaises(ValueError):
            schedule.remap_positions([0], -2)

    def test_prepared_wrapper_requires_and_unions_special_token_evidence(self):
        partition, coords = fixture(production=True)
        ids = torch.full((1, 921), 10)
        specials = [0, 780, 854, 860]
        ids[0, specials] = 99
        prepared = SimpleNamespace(pilot_partition=partition,
                                   choices={"A": SimpleNamespace(inputs={"input_ids": ids, "image_position_ids": coords})})
        arm = Arm("QR", (Stage(0, fractions={"Q": .5, "R": .5}),))
        with self.assertRaisesRegex(ValueError, "provide tokenizer"):
            make_schedule(prepared, arm, 1)
        tokenizer = SimpleNamespace(all_special_ids=[99], bos_token_id=99)
        schedule = make_schedule(prepared, arm, 1, tokenizer=tokenizer, protected_positions=[855])
        self.assertTrue(set(specials + [855]).issubset(schedule.protected_positions))
        self.assertEqual(schedule.stages[0].role_counts["R"]["protected"], 4)
        explicit = make_schedule(prepared, arm, 1, special_token_ids=[99])
        self.assertTrue(set(specials).issubset(explicit.protected_positions))
        make_schedule(prepared, arm, 1, protected_positions=specials)
        make_schedule(prepared, build_arms()[0], 1)
        ids[0, list(partition.structural_remainder.positions)] = 99
        with self.assertRaisesRegex(ValueError, "zero eligible"):
            make_schedule(prepared, arm, 1, tokenizer=tokenizer)

    def test_all_71_wrapper_plans_allow_special_image_payload_only_in_slots(self):
        prepared, tokenizer = payload_fixture()
        partition = prepared.pilot_partition
        proof = protection_receipt(prepared, tokenizer=tokenizer)
        self.assertEqual(proof["eligible_counts"], {"F": 266, "C": 512, "Q": 75, "R": 58, "P": 0})
        self.assertEqual(len(proof["image_payload_positions"]), 778)
        self.assertTrue(set(partition.structural_remainder.positions[:7]).issubset(proof["protected_positions"]))
        for arm in build_arms():
            with self.subTest(arm=arm["id"]):
                schedule = make_schedule(prepared, arm, tokenizer=tokenizer)
                self.assertEqual(schedule.geometry["protection"], proof)
                self.assertEqual(list(schedule.protected_positions), proof["protected_positions"])
                for stage in schedule.stages:
                    self.assertTrue(set(proof["protected_positions"]).issubset(stage.keep))
                    if "F" in stage.role_counts:
                        self.assertEqual(stage.role_counts["F"]["protected"], 0)
                    if "C" in stage.role_counts:
                        self.assertEqual(stage.role_counts["C"]["protected"], 0)
                if arm["id"] == "F133_L34_merge":
                    self.assertEqual(len(schedule.stages[0].groups), 788)

    def test_other_visual_specials_and_out_of_slot_payload_ids_stay_protected(self):
        prepared, tokenizer = payload_fixture()
        p = prepared.pilot_partition
        ids = prepared.choices["A"].inputs["input_ids"]
        outside = [p.literal_user_prompt.positions[0], p.structural_remainder.positions[7]]
        ids[0, outside] = tokenizer.image_token_id
        foreign = [p.full_frame.positions[0], p.candidate_crops.positions[0], p.full_frame.positions[1]]
        ids[0, foreign] = torch.tensor([tokenizer.boi_token_id, tokenizer.eoi_token_id, 107])
        explicit = p.full_frame.positions[2]
        proof = protection_receipt(prepared, tokenizer=tokenizer, protected_positions=[explicit])
        self.assertTrue(set(outside + foreign + [explicit]).issubset(proof["protected_positions"]))
        self.assertTrue(set(outside + foreign).isdisjoint(proof["image_payload_positions"]))
        self.assertIn(explicit, proof["image_payload_positions"])
        self.assertEqual(proof["eligible_counts"]["F"], 263)
        self.assertEqual(proof["eligible_counts"]["C"], 511)
        arm = Arm("mixed", (Stage(34, targets={"F": 133, "C": 379}, fractions={"Q": .5, "R": .5}),))
        schedule = make_schedule(prepared, arm, tokenizer=tokenizer, protected_positions=[explicit])
        for pos in outside + foreign + [explicit]:
            self.assertIn((pos,), schedule.stages[0].groups)

    def test_payload_id_fallback_override_and_boundary_collision(self):
        prepared, tokenizer = payload_fixture()
        fallback = protection_receipt(prepared, special_token_ids=tokenizer.all_special_ids)
        self.assertEqual(fallback, protection_receipt(prepared, tokenizer=tokenizer))
        positions = fallback["image_payload_positions"]
        prepared.choices["A"].inputs["input_ids"][0, positions] = 7777
        explicit = protection_receipt(prepared, special_token_ids=tokenizer.all_special_ids, image_token_id=7777)
        tokenizer.image_token_id = 7777
        self.assertEqual(explicit, protection_receipt(prepared, tokenizer=tokenizer))
        self.assertEqual(explicit["eligible_counts"]["F"], 266)
        self.assertNotEqual(explicit["sha256"], fallback["sha256"])
        with self.assertRaisesRegex(ValueError, "collides"):
            protection_receipt(prepared, tokenizer=tokenizer, image_token_id=tokenizer.boi_token_id)

    def test_pinned_source_signature_and_shared_kv_dict_passthrough(self):
        path = Path("/tmp/mani-pilot-transformers513/transformers/models/gemma4/modeling_gemma4.py")
        if not path.is_file():
            self.skipTest("local pinned Transformers source not installed")
        tree = ast.parse(path.read_text())
        decoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Gemma4TextDecoderLayer")
        forward = next(n for n in decoder.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        self.assertEqual([arg.arg for arg in forward.args.args],
                         ["self", "hidden_states", "per_layer_input", "shared_kv_states",
                          "position_embeddings", "attention_mask", "position_ids", "past_key_values"])
        text = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Gemma4TextModel")
        call = next(n for n in ast.walk(text) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "decoder_layer")
        self.assertEqual([n.id for n in call.args], ["hidden_states", "per_layer_input"])
        self.assertEqual({kw.arg for kw in call.keywords}, {None, "shared_kv_states", "position_embeddings",
                                                           "attention_mask", "position_ids", "past_key_values"})
        layers = [Block(), Block(), Block()]
        shared = UserDict({"full_attention": (torch.ones(1), torch.ones(1))})
        side = self.side_inputs()
        side["shared_kv_states"] = shared
        with matrix_context(layers, self.schedule([Stage(0, targets={"F": 4})])):
            self.run_layers(layers, side=side)
        self.assertTrue(all(layer.seen["shared_kv_states"] is shared for layer in layers))
        self.assert_clean(layers)


if __name__ == "__main__":
    unittest.main()
