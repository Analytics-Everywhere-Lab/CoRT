"""Frozen task inventories, independent of scheduler and GPU count."""

from __future__ import annotations

from itertools import combinations
from copy import deepcopy
import re

from .common import digest
from .recognition_protocol import is_recognition, recognition_protocol


MODELS = {
    "gemma4_31b_hf_bf16": {"repo": "google/gemma-4-31B-it", "revision": "3548789868c5356dbf307c98e6f609007b82b3eb", "layers": 60},
    "qwen35_27b_hf_bf16": {"repo": "Qwen/Qwen3.5-27B", "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654", "layers": 64},
}
GEMMA = "gemma4_31b_hf_bf16"
QWEN = "qwen35_27b_hf_bf16"
STAGES = ("pilot", "exploration", "confirmation")
FORECAST = "forecast_next_interaction"
CURRENT = "current_interaction"
DENSE_STUDY = "temporal-current-dense-v1"
DENSE_HISTORY_OFFSETS_S = (-2, -1.75, -1.5, -1.25, -1, -.75, -.5, -.25, 0)
DENSE_FIXED_CUTS = (22, 34)
DENSE_CONDITIONS = ("B0", "B1", "B4", "B8")
DENSE_SEED = 20261023
DENSE_PATCHING = "fixed_current_history_v1"
# Exact task question from run_phase2_reanchored_answer_token_causal_patch.py.
STATIC_CURRENT_QUESTION = (
    "Which referenced target is the first-order interacting object, meaning the "
    "object directly manipulated by a visible hand rather than a second-order "
    "object interacting through it?"
)
PROMPTS = {
    FORECAST: (
        "The final two images are candidate objects A and B, respectively. Any "
        "preceding images are observed scene frames in the presented order. "
        "Which candidate object will the camera wearer interact with next after "
        "the observation cutoff? Predict the next annotated object interaction. "
        "Answer with only A or B."
    ),
    CURRENT: (
        "The final two images are candidate objects A and B, respectively. Any "
        "preceding images are observed scene frames in the presented order. "
        "The task concerns the current observation time. "
        + STATIC_CURRENT_QUESTION + " Answer with only A or B."
    ),
}


def validate_contract(contract):
    if isinstance(contract, dict) and type(contract.get("version")) is int and contract["version"] == 3:
        from .recognition_protocol import validate_recognition_contract

        return validate_recognition_contract(contract)
    if (not isinstance(contract, dict) or type(contract.get("version")) is not int
            or contract["version"] not in (1, 2)):
        raise ValueError("invalid matched task contract")
    dense = contract["version"] == 2
    fields = {"version", "task", "prompt", "exposure_policy"}
    if dense:
        fields |= {"history_offsets_s", "patching"}
    if set(contract) != fields:
        raise ValueError("invalid matched task contract")
    task = contract["task"]
    if not isinstance(task, str) or task not in PROMPTS or contract["prompt"] != PROMPTS[task]:
        raise ValueError("unsupported task or prompt drift")
    if dense and (task != CURRENT or contract["patching"] != DENSE_PATCHING
                  or not isinstance(contract["history_offsets_s"], list)
                  or any(type(x) not in (int, float) for x in contract["history_offsets_s"])
                  or contract["history_offsets_s"] != list(DENSE_HISTORY_OFFSETS_S)):
        raise ValueError("invalid dense current history/patching contract")
    policy = contract["exposure_policy"]
    if (not isinstance(policy, dict) or set(policy) !=
            {"version", "kind", "historical_static_sha256", "temporal_exposure_sha256"}
            or type(policy["version"]) is not int or policy["version"] != 1
            or policy["kind"] != "matched_static_reuse"):
        raise ValueError("explicit versioned matched exposure policy required")
    for key in ("historical_static_sha256", "temporal_exposure_sha256"):
        if not isinstance(policy[key], str) or not re.fullmatch(r"[0-9a-f]{64}", policy[key]):
            raise ValueError("exposure policy requires raw registry SHA256 pins")
    return contract


def protocol_contract(protocol):
    if type(protocol.get("schema_version")) is int and protocol["schema_version"] in (4, 5):
        from .recognition_protocol import recognition_contract

        return recognition_contract(protocol)
    if protocol.get("schema_version") == 1:
        if any(key in protocol for key in ("task", "prompt", "exposure_policy")):
            raise ValueError("task/policy changes require a new versioned protocol")
        return None
    if type(protocol.get("schema_version")) is not int or protocol["schema_version"] not in (2, 3):
        raise ValueError("unsupported protocol schema")
    dense = protocol["schema_version"] == 3
    fields = ("task", "prompt", "exposure_policy") + (("history_offsets_s", "patching") if dense else ())
    return validate_contract({"version": 2 if dense else 1,
                              **{key: protocol.get(key) for key in fields}})


def is_dense_current(protocol_or_contract):
    """Recognize the versioned dense geometry, never just a CURRENT task label."""
    if not isinstance(protocol_or_contract, dict):
        return False
    try:
        if "schema_version" in protocol_or_contract:
            if protocol_or_contract.get("study") != DENSE_STUDY:
                return False
            contract = protocol_contract(protocol_or_contract)
        else:
            contract = validate_contract(protocol_or_contract)
    except ValueError:
        return False
    return contract is not None and contract["version"] == 2


def validate_rows_contract(rows, contract):
    for row in rows:
        if contract is None:
            if "task" in row or "contract_sha256" in row:
                raise ValueError("matched rows require a matched protocol")
        elif row.get("task") != contract["task"] or row.get("contract_sha256") != digest(contract):
            raise ValueError("row task/policy contract mismatch")


def validate_dataset_protocol(manifest, protocol):
    validate_protocol(protocol)
    if manifest.get("contract") != protocol_contract(protocol):
        raise ValueError("dataset/protocol task or exposure policy mismatch")


def matched_protocol(task, exposure_policy):
    contract = validate_contract({"version": 1, "task": task, "prompt": PROMPTS.get(task),
                                  "exposure_policy": deepcopy(exposure_policy)})
    result = default_protocol()
    result.update(schema_version=2, study="temporal-matched-v1",
                  **{key: contract[key] for key in ("task", "prompt", "exposure_policy")})
    if task == CURRENT:
        result.update(primary_models=[], primary_family="descriptive_only", alpha=None,
                      candidate_policy="preserved audited static pair; no new outcome-based selection")
    return validate_protocol(result)


def dense_current_protocol(exposure_policy):
    """Bounded descriptive current task; no exploration or selected cuts."""
    result = matched_protocol(CURRENT, exposure_policy)
    result.update(schema_version=3, study=DENSE_STUDY, models={GEMMA: deepcopy(MODELS[GEMMA])},
                  history_offsets_s=list(DENSE_HISTORY_OFFSETS_S), patching=DENSE_PATCHING,
                  seed=DENSE_SEED, current_limit=96, current_canary_limit=4,
                  fixed_cuts=list(DENSE_FIXED_CUTS),
                  cut_selection="fixed cuts 22,34; no outcome-based selection")
    return validate_protocol(result)


def default_protocol():
    return {
        "schema_version": 1,
        "study": "temporal-forecasting-v2",
        "models": deepcopy(MODELS),
        "attention_backend": "sdpa",
        "dtype": "bfloat16",
        "seed": 20261023,
        "bootstrap_draws": 19999,
        "pilot_limit": 100,
        "exploration_limit": 60,
        "confirmation_limit": None,
        "minimum_confirmation_clusters": 46,
        "survey_stride": 2,
        "patch_corruption": "B4",
        "confirmation_secondary_corruption": "B9",
        "primary_models": [GEMMA, QWEN],
        "primary_family": "history_logloss_IUT_and_current_V_raw_reduction_per_model_Holm4",
        "alpha": 0.05,
        "pruning_enabled": False,
        "candidate_policy": "both crops fixed from observed input before outcome-label review",
        "cut_selection": "exploration only; deterministic maximizer; lowest-cut tie break",
    }


def validate_protocol(protocol):
    if type(protocol.get("schema_version")) is int and protocol["schema_version"] in (4, 5):
        from .recognition_protocol import validate_recognition_protocol

        return validate_recognition_protocol(protocol)
    contract = protocol_contract(protocol)
    current = contract is not None and contract["task"] == CURRENT
    dense = contract is not None and contract["version"] == 2
    if protocol.get("models") != ({GEMMA: MODELS[GEMMA]} if dense else MODELS):
        raise ValueError("protocol/model pins differ from supported v1 schema")
    if contract and protocol.get("study") != (DENSE_STUDY if dense else "temporal-matched-v1"):
        raise ValueError("matched contract requires a new matched study")
    if dense:
        for key, expected in (("seed", DENSE_SEED), ("current_limit", 96), ("current_canary_limit", 4)):
            if type(protocol.get(key)) is not int or protocol[key] != expected:
                raise ValueError(f"dense current {key} is frozen to {expected}")
        cuts = protocol.get("fixed_cuts")
        if (not isinstance(cuts, list) or any(type(c) is not int for c in cuts)
                or cuts != list(DENSE_FIXED_CUTS)
                or protocol.get("cut_selection") != "fixed cuts 22,34; no outcome-based selection"):
            raise ValueError("dense current requires fixed cuts 22,34 and no cut selection")
    for name in ("pilot_limit", "exploration_limit", "minimum_confirmation_clusters", "survey_stride", "bootstrap_draws"):
        value = protocol.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"invalid {name}")
    if protocol["survey_stride"] != 2 or protocol.get("patch_corruption") != "B4":
        raise ValueError("changing survey geometry requires a new protocol version")
    if protocol.get("confirmation_secondary_corruption") != "B9":
        raise ValueError("confirmation includes the frozen gray-history robustness comparison")
    if protocol.get("attention_backend") != "sdpa" or protocol.get("dtype") != "bfloat16":
        raise ValueError("runtime mode is frozen to BF16/SDPA")
    if protocol.get("primary_models") != ([] if current else [GEMMA, QWEN]):
        raise ValueError("this new study includes both models in its primary family")
    if protocol.get("pruning_enabled") is not False:
        raise ValueError("temporal pruning has no verified runtime adapter; keep it disabled")
    if (protocol.get("alpha") != (None if current else .05)
            or protocol.get("primary_family") != ("descriptive_only" if current else default_protocol()["primary_family"])):
        raise ValueError("the implemented primary family is fixed to four claims and alpha=.05")
    if contract:
        expected = ("preserved audited static pair; no new outcome-based selection" if current
                    else default_protocol()["candidate_policy"])
        if protocol.get("candidate_policy") != expected:
            raise ValueError("candidate policy differs from the task contract")
    return protocol


def selected_episodes(rows, protocol, stage):
    if type(protocol.get("schema_version")) is int and protocol["schema_version"] in (4, 5):
        from .recognition_protocol import selected_recognition_episodes

        return selected_recognition_episodes(rows, protocol, stage)
    if protocol.get("schema_version") == 3:
        validate_protocol(protocol)
        if stage not in ("current", "current-canary"):
            raise ValueError(f"unknown dense current stage {stage}")
        seen_episodes = set()
        for row in rows:
            if (row.get("split") not in STAGES
                    or any(not isinstance(row.get(key), str) or not row[key]
                           for key in ("episode_id", "cluster_id"))):
                raise ValueError("dense current requires public split and episode/cluster provenance")
            if row["episode_id"] in seen_episodes:
                raise ValueError("duplicate dense current episode")
            seen_episodes.add(row["episode_id"])
        available = sorted(rows, key=lambda row: (
            digest({"seed": protocol["seed"], "episode": row["episode_id"]}), row["episode_id"]))
        selected, clusters = [], set()
        limit = protocol["current_canary_limit" if stage.endswith("-canary") else "current_limit"]
        for row in available:
            if row["cluster_id"] not in clusters:
                selected.append(row)
                clusters.add(row["cluster_id"])
                if len(selected) == limit:
                    break
        if not selected or (stage == "current" and len(selected) < limit):
            minimum = 1 if stage.endswith("-canary") else limit
            raise ValueError(f"{stage} requires {minimum} independent clusters; found {len(selected)}")
        return selected
    split = stage.removesuffix("-canary")
    if split not in STAGES:
        raise ValueError(f"unknown stage {stage}")
    available = sorted((row for row in rows if row["split"] == split),
                       key=lambda row: digest({"seed": protocol["seed"], "episode": row["episode_id"]}))
    limit = protocol.get(f"{split}_limit")
    selected = available if limit is None else available[:limit]
    if not selected:
        raise ValueError(f"no data for {split}")
    if stage.endswith("-canary"):
        # Enough episode groups to exercise eight workers without making task
        # identities depend on a particular site's GPU count.
        return selected[:8 if split == "pilot" else 4]
    if limit is not None and len(selected) < limit:
        raise ValueError(f"{split} requires {limit} episodes; found {len(selected)}")
    if split == "confirmation" and len({row["cluster_id"] for row in selected}) < protocol["minimum_confirmation_clusters"]:
        raise ValueError("confirmation has insufficient independent participant clusters")
    return selected


def inventory(rows, protocol, stage, freeze=None):
    if type(protocol.get("schema_version")) is int and protocol["schema_version"] in (4, 5):
        from .recognition_protocol import recognition_inventory

        return recognition_inventory(rows, protocol, stage, freeze)
    validate_protocol(protocol)
    contract = protocol_contract(protocol)
    validate_rows_contract(rows, contract)
    current = contract is not None and contract["task"] == CURRENT
    dense = is_dense_current(contract)
    split, canary = stage.removesuffix("-canary"), stage.endswith("-canary")
    episodes = selected_episodes(rows, protocol, stage)
    models = [GEMMA] if split == "pilot" or dense else list(MODELS)
    if dense and freeze is not None:
        raise ValueError("dense current uses fixed cuts; selection freeze is unsupported")
    tasks = []

    def add(episode, model, order, kind, **kwargs):
        task = {"episode_id": episode["episode_id"], "cluster_id": episode["cluster_id"],
                "model_key": model, "order": order, "kind": kind, **kwargs}
        if dense:
            task["split"] = episode["split"]
        if contract:
            task.update(task=contract["task"], contract_sha256=digest(contract))
        task["task_id"] = digest({"protocol": protocol, "task": task})
        tasks.append(task)

    for episode in episodes:
        for model in models:
            if dense:
                cuts = list(DENSE_FIXED_CUTS)
            elif current:
                if split == "confirmation" and (not freeze or freeze.get("cuts") != {}):
                    raise ValueError("current companion requires a frozen selection with no cuts")
                cuts = []
            elif split == "confirmation":
                if not freeze or model not in freeze.get("cuts", {}):
                    raise ValueError("confirmation requires an explicit frozen design")
                cuts = sorted(set(freeze["cuts"][model].values()))
                if any(type(c) is not int or not 0 <= c < MODELS[model]["layers"] for c in cuts):
                    raise ValueError("invalid frozen cut")
            else:
                cuts = list(range(0, MODELS[model]["layers"], protocol["survey_stride"]))
            if canary and split != "confirmation" and not current:
                cuts = [0, MODELS[model]["layers"] // 2, MODELS[model]["layers"] - 2]
            for order in (0, 1):
                conditions = (DENSE_CONDITIONS if dense else
                              ("B0", "B1", "B4", "B5") if split == "pilot" else tuple(f"B{i}" for i in range(10)))
                for condition in conditions:
                    add(episode, model, order, "behavior", condition=condition)
                if split == "pilot" or (current and not dense):
                    continue
                for cut in cuts:
                    directions = [("B0", "B4"), ("B4", "B0")]
                    if split == "confirmation":
                        directions += [("B0", "B9"), ("B9", "B0")]
                    for donor, recipient in directions:
                        for position_set in ("H", "V", "K", "A"):
                            add(episode, model, order, "patch", donor_condition=donor,
                                recipient_condition=recipient, cut=cut, position_set=position_set)
                    # The same replacement path must preserve native output when source=target.
                    add(episode, model, order, "self", donor_condition="B0", recipient_condition="B0",
                        cut=cut, position_set="A")
                if split == "confirmation":
                    late = freeze["cuts"][model]["late_cut"]
                    for n in range(1, 5):
                        for subset in combinations("CQPR", n):
                            for donor, recipient in (("B0", "B4"), ("B4", "B0")):
                                add(episode, model, order, "patch", donor_condition=donor,
                                    recipient_condition=recipient, cut=late,
                                    position_set="+".join(subset))
                    if protocol.get("pruning_enabled") and model == GEMMA:
                        for cut in sorted({freeze["cuts"][model]["early_cut"], late}):
                            add(episode, model, order, "prune", condition="B0", cut=cut, position_set="H")
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task construction")
    return tasks


def assign_tasks(tasks, workers):
    """Model-sticky episode groups avoid per-task model reloads, at any GPU count."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    models = sorted({t["model_key"] for t in tasks})
    assignments = [[] for _ in range(workers)]
    if workers < len(models):
        for index, model in enumerate(models):
            assignments[index % workers].extend(t for t in tasks if t["model_key"] == model)
        return assignments
    model_ranks = {model: [i for i in range(workers) if i % len(models) == j] for j, model in enumerate(models)}
    groups = {}
    for task in tasks:
        groups.setdefault((task["model_key"], task["episode_id"]), []).append(task)
    for (model, _), group in sorted(groups.items()):
        rank = min(model_ranks[model], key=lambda i: (len(assignments[i]), i))
        assignments[rank].extend(group)
    return assignments
