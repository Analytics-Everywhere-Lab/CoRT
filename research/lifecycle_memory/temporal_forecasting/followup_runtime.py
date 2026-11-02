"""Explicit opt-in runtime for the recognition factorial/order follow-up.

The parent must validate the schema-5 protocol, suite and protocol identity
before choosing this class. Its constructor still takes the immutable version-3
recognition input contract, never an execution protocol. No new arms are enabled
on HFBackend. Imports and partition tests need only the standard library.

Row audits contain public input identities only. File hashes are SHA256 of the
staged bytes; other *_sha256 fields use common.digest (canonical JSON). In a
patch row, frame_audit/token_partition describe the recipient; donor_* describe
the donor. scene_permutation indexes source_frame_keys, including repeated index
16 for B4. Token hashes cover the token-ID list at the corresponding positions.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from pathlib import Path

from . import recognition_protocol as recognition
from . import runtime as rt
from .common import digest


CONDITIONS = ("B0", "B4", "S1", "S2", "RV")
_CANONICAL_SUBSETS = tuple("+".join(names) for size in range(1, 5)
                           for names in combinations("CQPR", size))


def _subsets():
    from .followup_protocol import SUBSETS

    rt._require(isinstance(SUBSETS, tuple) and len(SUBSETS) == 15
                and all(isinstance(name, str) for name in SUBSETS)
                and set(SUBSETS) == set(_CANONICAL_SUBSETS),
                "follow-up requires all 15 canonical C/Q/P/R subsets")
    return SUBSETS


def _scene_permutation(task, episode_id, condition):
    from .followup_protocol import scene_permutation

    rt._require(isinstance(condition, str) and condition in CONDITIONS,
                "follow-up behavior supports B0/B4/S1/S2/RV only")
    rt._require(isinstance(episode_id, str) and bool(episode_id.strip()),
                "follow-up requires a nonempty episode_id")
    identity = list(range(recognition.SCENE_COUNT))

    def checked(name):
        values = scene_permutation(task, episode_id, name)
        rt._require(isinstance(values, list) and len(values) == 17
                    and all(type(value) is int and 0 <= value < 17 for value in values),
                    "follow-up permutation must contain 17 integer indices in 0..16")
        if name == "B4":
            rt._require(values == [16] * 17, "B4 must repeat only the final frame")
        else:
            rt._require(sorted(values) == identity, "scene permutation is not a bijection")
            if name == "B0":
                rt._require(values == identity, "B0 scene order must be unchanged")
            elif name == "RV":
                rt._require(values == identity[::-1], "RV must reverse exactly the 17 scene frames")
            else:
                rt._require(values != identity and values[0] == 0 and values[-1] == 16,
                            "shuffle must be nonidentity with first/final frames fixed")
        return list(values)

    values = checked(condition)
    if condition in ("S1", "S2"):
        other = checked("S2" if condition == "S1" else "S1")
        rt._require(values != other, "S1 and S2 must be distinct permutations")
    return values


def slot_plan(condition, order, contract, episode_id):
    """Keep scene roles positional and candidates outside the scene permutation."""
    recognition.validate_recognition_contract(contract)
    rt._require(type(order) is int and order in (0, 1), "order must be 0 or 1")
    indices = _scene_permutation(contract["task"], episode_id, condition)
    return (*zip((*recognition.HISTORY_ROLES, "V"),
                 (recognition.SCENE_SLOTS[index] for index in indices)),
            ("CA", f"candidate{order}"), ("CB", f"candidate{1 - order}"))


def token_partition(prepared, contract):
    """Recheck native boundaries and emit auditable per-request token identities."""
    recognition.validate_recognition_contract(contract)
    ids, mask = (rt._row(prepared["inputs"][key]) for key in ("input_ids", "attention_mask"))
    groups, roles = prepared["positions"], prepared["slot_roles"]
    rt._require(roles == [*recognition.HISTORY_ROLES, "V", "CA", "CB"],
                "follow-up requires all 19 positional image slots")
    rebuilt = rt.partition_positions(ids, mask, {role: groups[role] for role in roles},
                                     groups["Q"], groups["P"], contract=contract)
    for subset in _subsets():
        rebuilt[subset] = rt.select_positions(rebuilt, subset)
    rt._require(groups == rebuilt, "follow-up native partition drift")
    readout = prepared["readout_position"]
    rt._require(type(readout) is int and readout == len(ids) - 1
                and readout in groups["P"] and readout not in groups["C"] + groups["Q"] + groups["R"],
                "follow-up readout must be in P and excluded from CQR")
    rt._require(all(groups[key] for key in "CQPR"), "follow-up requires nonempty C/Q/P/R classes")
    rt._require(sorted(groups["H"] + groups["V"] + groups["K"]) == groups["A"],
                "H/V/K must disjointly exhaust all active positions")
    return {
        "input_ids_sha256": digest(ids), "attention_mask_sha256": digest(mask),
        "cardinalities": {key: len(values) for key, values in groups.items()},
        "position_sha256": {key: digest(values) for key, values in groups.items()},
        "token_ids_sha256": {key: digest([ids[index] for index in values])
                             for key, values in groups.items()},
        "disjoint_exhaustive": True, "readout_in_P": True, "readout_excluded_from_CQR": True,
    }


class FollowupHFBackend(rt.HFBackend):
    """Same loader and bounded full-forward operator; narrowly extended arm gates."""

    def __init__(self, model_key, model_path, device="cuda:0", attention_backend="eager", *, contract=None):
        recognition.validate_recognition_contract(contract)
        super().__init__(model_key, model_path, device, attention_backend, contract=contract)

    def _slot_plan(self, episode, order, condition):
        return slot_plan(condition, order, self.contract, episode.get("episode_id"))

    def _scope_key(self, episode, order):
        return (*super()._scope_key(episode, order),
                tuple(episode["images"][key].get("sha256") for key in recognition.SLOTS))

    def _activate_cache(self, episode, order, conditions, cut=None):
        recognition.validate_recognition_contract(self.contract)
        return super()._activate_cache(episode, order, conditions, cut)

    def _frame_audit(self, episode, order, condition):
        plan = self._slot_plan(episode, order, condition)
        source_keys = list(recognition.SCENE_SLOTS)
        scene_keys, candidate_keys = [key for _, key in plan[:17]], [key for _, key in plan[17:]]
        indices = [source_keys.index(key) for key in scene_keys]
        hashes = {}
        for key in recognition.SLOTS:
            spec = episode["images"][key]
            path = Path(spec["path"])
            rt._require(path.is_absolute(), "episode image paths must be absolute")
            hashes[key] = rt._sha256(path)
            rt._require(hashes[key] == spec.get("sha256"), f"staged image hash differs: {key}")
        source_hashes = [hashes[key] for key in source_keys]
        scene_hashes, candidate_hashes = ([hashes[key] for key in keys]
                                         for keys in (scene_keys, candidate_keys))
        endpoints = scene_hashes[0] == source_hashes[0] and scene_hashes[-1] == source_hashes[-1]
        content = sorted(scene_hashes) == sorted(source_hashes)
        if condition in ("S1", "S2"):
            rt._require(endpoints and content, "shuffle changed frame content or endpoints")
        return {
            "condition": condition, "seed": recognition.SEED, "task": self.contract["task"],
            "episode_id": episode["episode_id"], "scene_permutation": indices,
            "source_frame_keys": source_keys, "source_frame_sha256": source_hashes,
            "scene_frame_keys": scene_keys, "scene_frame_sha256": scene_hashes,
            "candidate_keys": candidate_keys, "candidate_sha256": candidate_hashes,
            "scene_order_sha256": digest({"indices": indices, "keys": scene_keys, "sha256": scene_hashes}),
            "candidate_order_sha256": digest({"keys": candidate_keys, "sha256": candidate_hashes}),
            "endpoints_preserved": endpoints, "content_multiset_preserved": content,
            "prompt_chronology_violated": condition in ("S1", "S2", "RV"),
        }

    def _prepare(self, episode, order, condition):
        # All input files are parent-staged and immutable. Cache reuse is bound
        # to their stat identities and declared hashes, not only episode IDs.
        frame_audit = self._frame_audit(episode, order, condition)
        prepared = super()._prepare(episode, order, condition)
        for subset in _subsets():
            prepared["positions"][subset] = rt.select_positions(prepared["positions"], subset)
        prepared["frame_audit"] = frame_audit
        prepared["token_partition"] = token_partition(prepared, self.contract)
        return prepared

    def _align(self, donor, recipient):
        super()._align(donor, recipient)
        rt._require(donor["token_partition"] == recipient["token_partition"],
                    "donor/recipient token hashes or cardinalities differ")
        for key in ("source_frame_keys", "source_frame_sha256", "candidate_keys", "candidate_sha256"):
            rt._require(donor["frame_audit"][key] == recipient["frame_audit"][key],
                        f"donor/recipient input identity differs: {key}")

    def _result(self, scores, prepared, started):
        return {**super()._result(scores, prepared, started),
                "frame_audit": deepcopy(prepared["frame_audit"]),
                "token_partition": deepcopy(prepared["token_partition"])}

    def _validate_patch_request(self, donor_condition, recipient_condition, cut, position_set, self_patch):
        recognition.validate_recognition_contract(self.contract)
        rt._require(rt._model_alias(self.model_key) == "gemma31", "follow-up supports Gemma31 only")
        rt._require(type(cut) is int and cut == 34 and cut < len(self.layers),
                    "follow-up requires fixed cut 34")
        rt._require(type(self_patch) is bool, "self_patch must be boolean")
        if self_patch:
            rt._require((donor_condition, recipient_condition, position_set)
                        in (("B0", "B0", "A"), ("B4", "B4", "A")),
                        "follow-up self patch must be B0-to-B0 or B4-to-B4 A")
        else:
            rt._require((donor_condition, recipient_condition) in (("B0", "B4"), ("B4", "B0"))
                        and position_set in (*_subsets(), "A"),
                        "follow-up patches are B0/B4 canonical C/Q/P/R subsets or A only")

    def patch(self, episode, order, donor_condition, recipient_condition, cut: int,
              position_set: str, self_patch=False) -> dict:
        result = super().patch(episode, order, donor_condition, recipient_condition,
                               cut, position_set, self_patch=self_patch)
        donor, recipient = (self._prepared_cache[name] for name in (donor_condition, recipient_condition))
        selected = rt.select_positions(recipient["positions"], position_set)
        ids = rt._row(recipient["inputs"]["input_ids"])
        return {**result, "donor_frame_audit": deepcopy(donor["frame_audit"]),
                "donor_token_partition": deepcopy(donor["token_partition"]),
                "patch_selection": {
                    "positions": list(selected), "cardinality": len(selected),
                    "positions_sha256": digest(selected),
                    "token_ids_sha256": digest([ids[index] for index in selected]),
                    "includes_readout": recipient["readout_position"] in selected,
                }}
