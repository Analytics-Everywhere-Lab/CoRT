"""Frozen retrospective recognition tasks, separate from dense-current results."""

from collections import Counter
from copy import deepcopy

from .common import digest


NEW_INTERACTION = "new_interaction_recognition"
INTERACTION_ORDER = "interaction_order_recognition"
TASKS = (NEW_INTERACTION, INTERACTION_ORDER)
STUDY = "temporal-recognition-v1"
SEED = 20261025
BOOTSTRAP_DRAWS = 19999
SCENE_COUNT = 17
HISTORY_ROLES = tuple(f"H{i}" for i in range(1, SCENE_COUNT))
HISTORY_KEYS = tuple(f"past{i}" for i in range(1, SCENE_COUNT))
SCENE_SLOTS = (*HISTORY_KEYS, "current")
SLOTS = (*SCENE_SLOTS, "candidate0", "candidate1")
CONDITIONS = ("B0", "B1", "B4", "B2")
FIXED_CUTS = (22, 34)
PATCHING = "recognition_history_v1"
CANDIDATE_POLICY = "two visually audited annotated event objects; crops fixed across conditions"
CUT_SELECTION = "fixed cuts 22,34; no outcome-based selection"
_PREFIX = (
    "The final two images are candidate objects A and B, respectively. "
    "Any preceding images are observed scene frames in chronological order. "
    "The task concerns only the shown sequence, not a future interaction. "
)
PROMPTS = {
    NEW_INTERACTION: _PREFIX + (
        "Which candidate object does a visible hand begin directly manipulating during the shown sequence? "
        "Choose the object whose manipulation starts within the sequence, rather than an object already "
        "being manipulated at its beginning. Answer with only A or B."
    ),
    INTERACTION_ORDER: _PREFIX + (
        "Which candidate object does a visible hand begin directly manipulating first during the shown sequence? "
        "Answer with only A or B."
    ),
}
CONTRACT_FIELDS = ("task", "prompt", "exposure_policy", "scene_count", "patching")


def validate_recognition_contract(contract):
    from .protocol import CURRENT, PROMPTS as legacy_prompts, validate_contract

    if (not isinstance(contract, dict) or set(contract) != {"version", *CONTRACT_FIELDS}
            or type(contract.get("version")) is not int or contract["version"] != 3):
        raise ValueError("invalid recognition contract")
    task = contract["task"]
    if not isinstance(task, str) or task not in TASKS or contract["prompt"] != PROMPTS[task]:
        raise ValueError("unsupported recognition task or prompt drift")
    if (type(contract["scene_count"]) is not int or contract["scene_count"] != SCENE_COUNT
            or contract["patching"] != PATCHING):
        raise ValueError("recognition requires 17 scenes and fixed history patching")
    # Reuse the exact raw-registry policy validator, not a weaker interpretation.
    validate_contract({"version": 1, "task": CURRENT, "prompt": legacy_prompts[CURRENT],
                       "exposure_policy": contract["exposure_policy"]})
    return contract


def recognition_contract(protocol):
    if type(protocol.get("schema_version")) is not int or protocol["schema_version"] not in (4, 5):
        raise ValueError("recognition requires protocol schema 4 or follow-up schema 5")
    return validate_recognition_contract({"version": 3, **{key: protocol.get(key) for key in CONTRACT_FIELDS}})


def is_recognition(value):
    if not isinstance(value, dict):
        return False
    try:
        if "schema_version" in value:
            from .followup_protocol import is_followup

            if value.get("study") != STUDY and not is_followup(value):
                return False
            recognition_contract(value)
        else:
            validate_recognition_contract(value)
    except ValueError:
        return False
    return True


def recognition_protocol(task, exposure_policy, *, episode_limit=96, minimum_clusters=4):
    from .protocol import GEMMA, MODELS

    if not isinstance(task, str) or task not in TASKS:
        raise ValueError("unsupported recognition task")
    contract = validate_recognition_contract({
        "version": 3, "task": task, "prompt": PROMPTS.get(task),
        "exposure_policy": deepcopy(exposure_policy), "scene_count": SCENE_COUNT, "patching": PATCHING,
    })
    return validate_recognition_protocol({
        "schema_version": 4, "study": STUDY,
        **{key: contract[key] for key in CONTRACT_FIELDS},
        "models": {GEMMA: deepcopy(MODELS[GEMMA])}, "attention_backend": "sdpa", "dtype": "bfloat16",
        "seed": SEED, "bootstrap_draws": BOOTSTRAP_DRAWS, "episode_limit": episode_limit,
        "minimum_clusters": minimum_clusters, "max_episodes_per_cluster": 3, "canary_limit": 4,
        "conditions": list(CONDITIONS), "fixed_cuts": list(FIXED_CUTS), "cut_selection": CUT_SELECTION,
        "primary_models": [], "primary_family": "descriptive_only", "alpha": None,
        "pruning_enabled": False, "candidate_policy": CANDIDATE_POLICY,
    })


def validate_recognition_protocol(protocol):
    from .protocol import GEMMA, MODELS

    if type(protocol.get("schema_version")) is int and protocol["schema_version"] == 5:
        from .followup_protocol import validate_followup_protocol

        return validate_followup_protocol(protocol)

    recognition_contract(protocol)
    fixed = {
        "study": STUDY, "models": {GEMMA: MODELS[GEMMA]}, "attention_backend": "sdpa", "dtype": "bfloat16",
        "seed": SEED, "bootstrap_draws": BOOTSTRAP_DRAWS, "max_episodes_per_cluster": 3, "canary_limit": 4,
        "conditions": list(CONDITIONS), "fixed_cuts": list(FIXED_CUTS), "cut_selection": CUT_SELECTION,
        "primary_models": [], "primary_family": "descriptive_only", "alpha": None,
        "pruning_enabled": False, "candidate_policy": CANDIDATE_POLICY,
    }
    required = {"schema_version", *CONTRACT_FIELDS, *fixed, "episode_limit", "minimum_clusters", "bootstrap_draws"}
    if not required <= protocol.keys() or protocol.keys() - required - {"episode_ids", "bootstrap_seed"}:
        raise ValueError("invalid recognition protocol fields")
    for key, value in fixed.items():
        if type(protocol.get(key)) is not type(value) or protocol[key] != value:
            raise ValueError(f"recognition protocol drift: {key}")
    for key in ("seed", "max_episodes_per_cluster", "canary_limit", "episode_limit", "minimum_clusters", "bootstrap_draws"):
        if type(protocol[key]) is not int:
            raise ValueError(f"recognition requires integer {key}")
    if (not 4 <= protocol["episode_limit"] <= 96
            or not 4 <= protocol["minimum_clusters"] <= protocol["episode_limit"]):
        raise ValueError("invalid recognition episode/cluster limits")
    if any(type(cut) is not int for cut in protocol["fixed_cuts"]):
        raise ValueError("recognition cuts must be integer block indices")
    if "bootstrap_seed" in protocol and (type(protocol["bootstrap_seed"]) is not int or protocol["bootstrap_seed"] != SEED):
        raise ValueError("recognition bootstrap seed drift")
    if "episode_ids" in protocol:
        ids = protocol["episode_ids"]
        # Analysis binds either the entire frozen cohort or the four-wearer
        # canary; its phase-specific coverage gate distinguishes these scopes.
        if (not isinstance(ids, list) or len(ids) not in (protocol["episode_limit"], protocol["canary_limit"])
                or any(not isinstance(x, str) or not x.strip() for x in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("invalid recognition analysis episode IDs")
    return protocol


def selected_recognition_episodes(rows, protocol, stage):
    from .protocol import validate_rows_contract

    validate_recognition_protocol(protocol)
    if stage not in ("recognition", "recognition-canary"):
        raise ValueError(f"unknown recognition stage {stage}")
    validate_rows_contract(rows, recognition_contract(protocol))
    if len(rows) != protocol["episode_limit"]:
        raise ValueError("recognition requires exactly episode_limit published rows")
    episodes, clusters = set(), Counter()
    for row in rows:
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("episode_id", "cluster_id")):
            raise ValueError("recognition requires episode/cluster provenance")
        if row["episode_id"] in episodes:
            raise ValueError("duplicate recognition episode")
        episodes.add(row["episode_id"])
        clusters[row["cluster_id"]] += 1
    if len(clusters) < protocol["minimum_clusters"] or max(clusters.values()) > 3:
        raise ValueError("recognition cluster coverage or maximum three episodes per wearer violated")
    available = sorted(rows, key=lambda row: (
        digest({"seed": SEED, "episode": row["episode_id"]}), row["episode_id"]))
    if stage == "recognition":
        return available
    selected, seen = [], set()
    for row in available:
        if row["cluster_id"] not in seen:
            selected.append(row)
            seen.add(row["cluster_id"])
            if len(selected) == 4:
                return selected
    raise ValueError("recognition canary requires four distinct wearers")


def recognition_inventory(rows, protocol, stage, freeze=None):
    from .protocol import GEMMA

    if type(protocol.get("schema_version")) is int and protocol["schema_version"] == 5:
        from .followup_protocol import followup_inventory

        return followup_inventory(rows, protocol, stage, freeze)

    episodes = selected_recognition_episodes(rows, protocol, stage)
    if freeze is not None:
        raise ValueError("recognition uses fixed cuts; selection freeze is unsupported")
    contract = recognition_contract(protocol)
    tasks = []

    def add(episode, order, kind, **kwargs):
        task = {"episode_id": episode["episode_id"], "cluster_id": episode["cluster_id"],
                "model_key": GEMMA, "order": order, "kind": kind, "task": contract["task"],
                "contract_sha256": digest(contract), **kwargs}
        if "split" in episode:
            task["split"] = episode["split"]
        task["task_id"] = digest({"protocol": protocol, "task": task})
        tasks.append(task)

    for episode in episodes:
        for order in (0, 1):
            for condition in CONDITIONS:
                add(episode, order, "behavior", condition=condition)
            for cut in FIXED_CUTS:
                for donor, recipient in (("B0", "B4"), ("B4", "B0")):
                    for group in ("H", "V", "K", "A"):
                        add(episode, order, "patch", donor_condition=donor, recipient_condition=recipient,
                            cut=cut, position_set=group)
                add(episode, order, "self", donor_condition="B0", recipient_condition="B0", cut=cut, position_set="A")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate recognition task construction")
    return tasks
