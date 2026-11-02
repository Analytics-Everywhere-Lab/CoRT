"""Sealed factorial/order follow-ups on unchanged recognition input datasets."""

from copy import deepcopy
from itertools import combinations
import re

from .common import digest
from .recognition_protocol import (
    CONDITIONS as SOURCE_CONDITIONS, FIXED_CUTS as SOURCE_CUTS,
    STUDY as SOURCE_STUDY, TASKS, SEED, recognition_contract,
)

STUDY = "temporal-recognition-followup-v1"
SUITE = "recognition-factorial-order-v1"
CONDITIONS = ("B0", "B4", "S1", "S2", "RV")
FIXED_CUTS = (34,)
CLASSES = ("C", "Q", "P", "R")
SUBSETS = tuple("+".join(s) for n in range(1, 5) for s in combinations(CLASSES, n))
ROWS_PER_EPISODE = 78
CUT_SELECTION = "L34 fixed from prior static study; no follow-up outcome selection"
SPEC = {
    "suite": SUITE, "subsets": list(SUBSETS), "cut": 34,
    "shuffle_seed": SEED, "shuffle_rule": "sha256-ranked-interior-v1",
    "shuffle_endpoints_fixed": True, "permutations_paired_across_candidate_orders": True,
    "transformed_label_policy": "original-answer retention; no inferred transformed gold",
    "raw_interaction": "gold-oriented m(CQ)-m(C)-m(Q)+m(empty), directions separate",
    "relative_guard_bits": .01, "bootstrap_unit": "wearer",
    "main_order_contrast": "B0-minus-within-clip-mean(S1,S2)",
    "new_canary_required": True, "rows_per_episode": ROWS_PER_EPISODE,
}
PIN_FIELDS = ("source_dataset_manifest_sha256", "source_result_set_sha256",
              "design_sha256", "human_validation_sha256")


def is_followup(value):
    return (isinstance(value, dict) and type(value.get("schema_version")) is int
            and value["schema_version"] == 5 and value.get("study") == STUDY)


def verify_followup_dataset(protocol, manifest_sha256):
    if is_followup(protocol):
        validate_followup_protocol(protocol)
        if protocol["source_dataset_manifest_sha256"] != manifest_sha256:
            raise ValueError("follow-up dataset differs from its frozen source")


def followup_protocol(source_protocol, **pins):
    from .recognition_protocol import validate_recognition_protocol

    validate_recognition_protocol(source_protocol)
    if source_protocol["schema_version"] != 4:
        raise ValueError("follow-up source must be the original schema-4 recognition protocol")
    result = deepcopy(source_protocol)
    result.update(schema_version=5, study=STUDY, conditions=list(CONDITIONS),
                  fixed_cuts=list(FIXED_CUTS), cut_selection=CUT_SELECTION,
                  followup=deepcopy(SPEC), **pins)
    return validate_followup_protocol(result)


def validate_followup_protocol(protocol):
    from .recognition_protocol import validate_recognition_protocol

    if not is_followup(protocol):
        raise ValueError("follow-up requires schema 5 and its own study identity")
    if (protocol.get("followup") != SPEC or digest(protocol["followup"]) != digest(SPEC)
            or protocol.get("conditions") != list(CONDITIONS)
            or protocol.get("fixed_cuts") != list(FIXED_CUTS)
            or any(type(x) is not int for x in protocol.get("fixed_cuts", []))
            or protocol.get("cut_selection") != CUT_SELECTION):
        raise ValueError("follow-up design drift")
    for key in PIN_FIELDS:
        if not isinstance(protocol.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", protocol[key]):
            raise ValueError(f"follow-up requires immutable {key}")
    source = deepcopy(protocol)
    for key in (*PIN_FIELDS, "followup"):
        source.pop(key)
    source.update(schema_version=4, study=SOURCE_STUDY, conditions=list(SOURCE_CONDITIONS),
                  fixed_cuts=list(SOURCE_CUTS), cut_selection="fixed cuts 22,34; no outcome-based selection")
    validate_recognition_protocol(source)
    return protocol


def scene_permutation(task, episode_id, condition):
    """Slot-to-source indices; independent of candidate order and evaluator labels."""
    if task not in TASKS or not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("invalid task/episode for a frozen permutation")
    if condition not in CONDITIONS:
        raise ValueError("unknown follow-up condition")
    if condition == "B0":
        return list(range(17))
    if condition == "B4":
        return [16] * 17
    if condition == "RV":
        return list(reversed(range(17)))
    forbidden = [list(range(1, 16))]
    if condition == "S2":
        forbidden.append(scene_permutation(task, episode_id, "S1")[1:-1])
    for attempt in range(100):
        middle = sorted(range(1, 16), key=lambda index: digest({
            "seed": SEED, "task": task, "episode_id": episode_id,
            "condition": condition, "attempt": attempt, "source_index": index}))
        if middle not in forbidden:
            return [0, *middle, 16]
    raise ValueError("could not construct a distinct nonidentity frozen shuffle")


def followup_inventory(rows, protocol, stage, freeze=None):
    from .protocol import GEMMA
    from .recognition_protocol import selected_recognition_episodes

    validate_followup_protocol(protocol)
    if freeze is not None:
        raise ValueError("follow-up protocol itself freezes the design; no selected-cut freeze")
    episodes = selected_recognition_episodes(rows, protocol, stage)
    contract = recognition_contract(protocol)
    result = []

    def add(episode, order, kind, **kwargs):
        task = {"episode_id": episode["episode_id"], "cluster_id": episode["cluster_id"],
                "model_key": GEMMA, "order": order, "kind": kind,
                "task": contract["task"], "contract_sha256": digest(contract), **kwargs}
        if "split" in episode:
            task["split"] = episode["split"]
        task["task_id"] = digest({"protocol": protocol, "task": task})
        result.append(task)

    for episode in episodes:
        for order in (0, 1):
            for condition in CONDITIONS:
                add(episode, order, "behavior", condition=condition)
            for donor, recipient in (("B0", "B4"), ("B4", "B0")):
                for positions in (*SUBSETS, "A"):
                    add(episode, order, "patch", donor_condition=donor,
                        recipient_condition=recipient, cut=34, position_set=positions)
            for condition in ("B0", "B4"):
                add(episode, order, "self", donor_condition=condition,
                    recipient_condition=condition, cut=34, position_set="A")
    if len(result) != ROWS_PER_EPISODE * len(episodes) or len({t["task_id"] for t in result}) != len(result):
        raise ValueError("invalid follow-up inventory cardinality")
    return result
