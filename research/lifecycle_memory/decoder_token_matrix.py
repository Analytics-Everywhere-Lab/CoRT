"""Deterministic raw-post-block compression for the non-cached Gemma31 pilot.

Targets count *eligible current rows*, with protected rows retained in addition.
Fractions use ceil(fraction * eligible current rows). C targets are split within
one row between its two processor slots. Merge arms always prune Q/R; receipts disclose
the effective per-role operators. No hidden states or labels select geometry.

The pure builder supports small CPU fixtures. ``make_schedule`` additionally
checks the frozen 921/F266/C512/Q75/P3/R65 production partition. Callers of the
pure builder must supply all structural specials in protected_positions;
P, position zero (BOS), and the final readout are protected automatically.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
import hashlib
import itertools
import json
import math
from typing import Any, Mapping

from research.lifecycle_memory.application_pilot_compression import spatial_coordinates


ROLES = ("F", "C", "Q", "R")
ROLE_FIELDS = {"F": "full_frame", "C": "candidate_crops",
               "Q": "literal_user_prompt", "P": "answer_prefix",
               "R": "structural_remainder"}
PINNED_COUNTS = {"F": 266, "C": 512, "Q": 75, "P": 3, "R": 65}
# Gemma4Config and the frozen Gemma31 config distinguish soft image payload
# placeholders from the hard BOI/EOI boundary tokens around each image.
PINNED_IMAGE_TOKEN_ID = 258880
PINNED_BOUNDARY_IDS = (2, 255999, 258882)  # BOS, BOI, EOI


def _receipt(record):
    # Return a detached, JSON-native tree, hashing everything except the hash.
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result = json.loads(encoded)
    result["sha256"] = hashlib.sha256(encoded.encode("ascii")).hexdigest()
    return result


@dataclass(frozen=True)
class Stage:
    cut: int
    targets: dict[str, int] = field(default_factory=dict)
    fractions: dict[str, float] = field(default_factory=dict)
    method: str = "merge"

    @property
    def roles(self):
        return tuple(role for role in ROLES if role in self.targets or role in self.fractions)

    def as_dict(self):
        return {"cut": self.cut, "method": self.method,
                **({"targets": dict(self.targets)} if self.targets else {}),
                **({"fractions": dict(self.fractions)} if self.fractions else {})}


@dataclass(frozen=True)
class Arm:
    id: str
    stages: tuple[Stage, ...]
    family: str = "custom"

    def as_dict(self):
        return {"id": self.id, "family": self.family,
                "stages": [stage.as_dict() for stage in self.stages]}


def build_arms(config=None) -> list[dict]:
    """Frozen 71-arm inventory; shared F34/133 retains both memberships.

    config is accepted for runner compatibility; it cannot alter the inventory.
    """
    arms = [dict(id="native", family="native", stages=[],
                 group_memberships=["native"], subset=[])]

    def add(family, stages, subset=()):
        parts = []
        for stage in stages:
            amounts = "_".join(f"{r}{stage.targets[r]}" if r in stage.targets else f"{r}half"
                               for r in stage.roles)
            parts.append(f"{amounts}_L{stage.cut}_{stage.method}")
        identifier = "__".join(parts)
        if any(s.method == "merge" and set(s.roles) & {"Q", "R"} for s in stages):
            identifier += "_QRprune"
        item = Arm(identifier, tuple(stages), family).as_dict()
        item.update(group_memberships=[family], subset=list(subset))
        item["operators_by_stage"] = [
            {r: "spatial_group_mean" if s.method == "merge" and r in ("F", "C")
             else "position_spread_prune" if r in ("Q", "R") else "spatial_spread_prune"
             for r in s.roles} for s in stages]
        arms.append(item)
        return item

    for size in range(1, 5):
        for subset in itertools.combinations(ROLES, size):
            for method in ("merge", "prune"):
                # Canonical exact F133 is shared with the target/depth sweep.
                stage = (Stage(34, targets={"F": 133}, method=method) if subset == ("F",)
                         else Stage(34, fractions={r: .5 for r in subset}, method=method))
                item = add("late_subset", [stage], subset)
                if subset == ("F",):
                    item["group_memberships"].append("F_only")
                    item["subset_fraction"] = .5
    for cut in (22, 26, 30, 34):
        for target in (200, 133, 67):
            for method in ("merge", "prune"):
                if (cut, target) != (34, 133):
                    add("F_only", [Stage(cut, targets={"F": target}, method=method)])
    for cut in (22, 26, 30, 34):
        add("F_drop_all", [Stage(cut, targets={"F": 0}, method="prune")])
    for cut in (22, 26):
        for target in (133, 67):
            add("progressive_F", [Stage(cut, targets={"F": 200}),
                                  Stage(34, targets={"F": target})])
    for target_c in (448, 384):
        early = Stage(22, targets={"C": target_c})
        add("early_C", [early])
        for target_f in (133, 67):
            add("early_C_late_F", [early, Stage(34, targets={"F": target_f})])
    for cut in (22, 34):
        for method in ("merge", "prune"):
            add("matched_total_C", [Stage(cut, targets={"C": 379}, method=method)])
    assert len(arms) == len({a["id"] for a in arms}) == 71
    return arms


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _positions(values, n, label):
    result = tuple(values)
    for p in result:
        _integer(p, label)
        if p >= n:
            raise ValueError(f"{label} position outside sequence")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{label} positions must be sorted and unique")
    return result


def _partition(partition):
    n = _integer(partition.sequence_length, "sequence_length", 1)
    roles = {r: _positions(getattr(partition, name).positions, n, r)
             for r, name in ROLE_FIELDS.items()}
    all_positions = [p for positions in roles.values() for p in positions]
    if sorted(all_positions) != list(range(n)):
        raise ValueError("roles must be disjoint and exhaustive; padding is unsupported")
    slots = tuple(_positions(s.positions, n, "image slot") for s in partition.image_slots)
    if len(slots) != 3 or slots[0] != roles["F"] or tuple(sorted(slots[1] + slots[2])) != roles["C"]:
        raise ValueError("image slots must partition F and the two C crops")
    return n, roles, slots


def _stages(arm):
    if isinstance(arm, Arm):
        arm = arm.as_dict()
    if not isinstance(arm, Mapping) or not isinstance(arm.get("id"), str) or not arm["id"]:
        raise ValueError("arm requires a nonempty id")
    result = []
    for raw in arm["stages"]:
        if not isinstance(raw, Mapping) or set(raw) - {"cut", "targets", "fractions", "method"}:
            raise ValueError("unknown stage fields")
        stage = Stage(**raw)
        _integer(stage.cut, "cut")
        if stage.cut >= 60 or (result and stage.cut <= result[-1].cut):
            raise ValueError("cuts must increase strictly within raw blocks 0..59")
        if stage.method not in ("merge", "prune"):
            raise ValueError("unknown method")
        if not isinstance(stage.targets, dict) or not isinstance(stage.fractions, dict):
            raise ValueError("targets and fractions must be role mappings")
        selected = set(stage.targets) | set(stage.fractions)
        if not selected or selected - set(ROLES) or set(stage.targets) & set(stage.fractions):
            raise ValueError("roles need exactly one target or fraction; P is protected")
        for target in stage.targets.values():
            _integer(target, "target")
        for fraction in stage.fractions.values():
            if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not 0 <= fraction <= 1:
                raise ValueError("fraction must be finite and in [0, 1]")
        result.append(stage)
    return arm, tuple(result)


def _spread_seeds(points, count):
    """Farthest-first spatial coverage with stable first/tie selection."""
    if not count:
        return []
    seeds = [0]
    distances = [float("inf")] * len(points)
    chosen = {0}
    while len(seeds) < count:
        x, y = points[seeds[-1]]
        for i, (px, py) in enumerate(points):
            distances[i] = min(distances[i], (x - px)**2 + (y - py)**2)
        best = max((i for i in range(len(points)) if i not in chosen),
                   key=lambda i: (distances[i], -i))
        seeds.append(best)
        chosen.add(best)
    return seeds


def _group_rows(rows, target, method, coordinates):
    if target > len(rows):
        raise ValueError("target exceeds eligible current rows")
    if target == len(rows):
        return rows
    if not target:
        if method == "merge" and rows:
            raise ValueError("zero retention requires pruning")
        return []
    if coordinates is None:
        return [rows[((2 * i + 1) * len(rows)) // (2 * target)] for i in range(target)]
    points = [tuple(sum(coordinates[p][axis] for p in row) / len(row) for axis in (0, 1))
              for row in rows]
    seeds = _spread_seeds(points, target)
    if method == "prune":
        return [rows[i] for i in seeds]
    clusters = {seed: list(rows[seed]) for seed in seeds}
    for i, point in enumerate(points):
        if i not in clusters:
            nearest = min(seeds, key=lambda s: (sum((a - b)**2 for a, b in zip(point, points[s])), s))
            clusters[nearest].extend(rows[i])
    return [tuple(sorted(cluster)) for cluster in clusters.values()]


@dataclass(frozen=True)
class ScheduledStage:
    cut: int
    method: str
    input_groups: tuple[tuple[int, ...], ...]
    groups: tuple[tuple[int, ...], ...]
    # Output row -> contributing input indices and their original-token masses.
    reductions: tuple[tuple[tuple[int, int], ...], ...]
    role_counts: dict[str, dict[str, Any]]

    @property
    def keep(self):
        return tuple(group[-1] for group in self.groups)


@dataclass(frozen=True)
class Schedule:
    original_length: int
    arm: dict
    protected_positions: tuple[int, ...]
    stages: tuple[ScheduledStage, ...]
    geometry: dict

    def groups_at(self, at_cut):
        _integer(at_cut, "at_cut", -1)
        groups = tuple((p,) for p in range(self.original_length))
        for stage in self.stages:
            if stage.cut > at_cut:
                break
            groups = stage.groups
        return groups

    def remap_positions(self, original_positions, at_cut):
        """Post-cut current indices; merged members alias, pruned members are None.

        Input order and duplicates are preserved. Use at_cut=-1 for raw input.
        """
        mapping = {p: i for i, group in enumerate(self.groups_at(at_cut)) for p in group}
        result = []
        for p in original_positions:
            _integer(p, "original position")
            if p >= self.original_length:
                raise ValueError("original position outside sequence")
            result.append(mapping.get(p))
        return tuple(result)

    def receipt(self):
        return _receipt({"schema": "decoder-token-matrix-v1", **asdict(self)})


def remap_positions(schedule, original_positions, at_cut):
    return schedule.remap_positions(original_positions, at_cut)


def build_schedule(*, partition, image_position_ids, arm, protected_positions, pooling_kernel=3):
    """Build without model access; supplied protection covers structural specials.

    Image payload placeholders are not structural specials. This pure API has
    no token IDs and therefore never removes explicit position protection.
    """
    n, roles, slots = _partition(partition)
    arm, stages = _stages(arm)
    _integer(pooling_kernel, "pooling_kernel", 1)
    supplied = list(protected_positions)
    for p in supplied:
        _integer(p, "protected position")
        if p >= n:
            raise ValueError("protected position outside sequence")
    protected = set(supplied) | set(roles["P"]) | {0, n - 1}
    coordinates = {}
    if any(set(s.roles) & {"F", "C"} for s in stages):
        for slot, positions in enumerate(slots):
            coords = spatial_coordinates(image_position_ids, slot, pooling_kernel, len(positions))
            coordinates.update(zip(positions, coords))
    groups = tuple((p,) for p in range(n))
    plans = []
    for stage in stages:
        before = groups
        updated = list(groups)
        counts = {}
        for role in stage.roles:
            role_positions = set(roles[role])
            eligible = [row for row in before if row[-1] in role_positions and row[-1] not in protected]
            if role == "R" and not eligible:
                raise ValueError("selected R has zero eligible rows after protection")
            target = (stage.targets[role] if role in stage.targets
                      else math.ceil(stage.fractions[role] * len(eligible)))
            method = stage.method if role in ("F", "C") else "prune"
            if role == "C":
                parts = [[row for row in eligible if row[-1] in set(slot)] for slot in slots[1:]]
            else:
                parts = [eligible]
            budgets = [target // len(parts) + int(i < target % len(parts)) for i in range(len(parts))]
            selected = []
            for part, budget in zip(parts, budgets):
                selected.extend(_group_rows(part, budget, method,
                                            coordinates if role in ("F", "C") else None))
            eligible_reps = {row[-1] for row in eligible}
            updated = [row for row in updated if row[-1] not in eligible_reps] + selected
            protected_count = len(role_positions & protected)
            counts[role] = {"original": len(role_positions), "protected": protected_count,
                            "eligible_before": len(eligible), "eligible_retained": target,
                            "after": target + protected_count, "effective_method": method,
                            "per_slot_retained": budgets}
        groups = tuple(sorted(updated, key=lambda row: row[-1]))
        owner = {p: i for i, row in enumerate(before) for p in row}
        reductions = tuple(tuple((i, len(before[i])) for i in sorted({owner[p] for p in row}))
                           for row in groups)
        plans.append(ScheduledStage(stage.cut, stage.method, before, groups, reductions, counts))
    geometry = {"pooling_kernel": pooling_kernel, "image_slots": slots,
                "coordinates": sorted(coordinates.items()), "roles": roles,
                "clustering": "farthest-first-centroid-nearest-v1",
                "representative": "latest-original-member", "weighting": "original-member-mass"}
    return Schedule(n, json.loads(json.dumps(arm)), tuple(sorted(protected)), tuple(plans), geometry)


def protection_receipt(prepared, *, protected_positions=None, tokenizer=None,
                       special_token_ids=None, image_token_id=None, require_evidence=True):
    """Shared, hashed boundary-vs-payload audit for runtime and schedule callers.

    Pass tokenizer=engine.tokenizer when prepared does not carry one. A supplied
    tokenizer/special_token_ids protects every matching position EXCEPT verified
    image payload occurrences inside F/C slots. BOI/EOI, text specials, P, BOS,
    and readout stay protected. Explicit protected_positions are never removed,
    including image positions. Without tokenizer/special IDs, explicit positions
    assert a complete structural-special audit; missing evidence fails closed.

    The payload ID comes from image_token_id, tokenizer.image_token_id, or the
    frozen Gemma31 config (258880), in that order. Returns protected_positions
    and per-role eligible counts so the runtime need not duplicate this policy.
    """
    partition = prepared.pilot_partition
    n, roles, _ = _partition(partition)
    if n != 921 or {r: len(p) for r, p in roles.items()} != PINNED_COUNTS:
        raise ValueError("frozen Gemma31 partition changed")
    inputs = prepared.choices["A"].inputs
    ids = inputs["input_ids"].detach().cpu().tolist()
    if len(ids) != 1 or len(ids[0]) != n:
        raise ValueError("only a single complete prompt is supported")
    tokenizer = tokenizer if tokenizer is not None else getattr(prepared, "tokenizer", None)
    specials = set(special_token_ids) if special_token_ids is not None else set()
    boundaries = set(PINNED_BOUNDARY_IDS)
    if tokenizer is not None:
        specials.update(tokenizer.all_special_ids)
        for name in ("bos_token_id", "boi_token_id", "eoi_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(tokenizer, name, None)
            if value is not None:
                boundaries.add(value)
    if require_evidence and protected_positions is None and tokenizer is None and special_token_ids is None:
        raise ValueError("provide tokenizer, special_token_ids, or complete protected_positions")
    if image_token_id is None:
        image_token_id = getattr(tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = PINNED_IMAGE_TOKEN_ID
    _integer(image_token_id, "image_token_id")
    if image_token_id in boundaries:
        raise ValueError("image payload ID collides with a structural boundary ID")
    specials.update(boundaries)
    specials.add(image_token_id)
    visual = set(roles["F"]) | set(roles["C"])
    payload = {i for i in visual if ids[0][i] == image_token_id}
    explicit = tuple(() if protected_positions is None else protected_positions)
    structural = {i for i, token in enumerate(ids[0]) if token in specials and i not in payload}
    for position in explicit:
        _integer(position, "protected position")
        if position >= n:
            raise ValueError("protected position outside sequence")
    protected = set(explicit) | structural | set(roles["P"]) | {0, n - 1}
    evidence = {"policy": "structural-specials-except-verified-image-payload-v1",
                "image_token_id": image_token_id, "special_token_ids": sorted(specials),
                "image_payload_positions": sorted(payload),
                "explicit_protected_positions": sorted(set(explicit)),
                "structural_special_positions": sorted(structural),
                "protected_positions": sorted(protected),
                "eligible_counts": {role: len(set(positions) - protected)
                                    for role, positions in roles.items()}}
    return _receipt(evidence)


def make_schedule(prepared, arm, pooling_kernel=3, *, protected_positions=None,
                  tokenizer=None, special_token_ids=None, image_token_id=None):
    """Production schedule using the shared structural-vs-payload protection audit."""
    arm, stages = _stages(arm)
    evidence = protection_receipt(prepared, protected_positions=protected_positions,
                                  tokenizer=tokenizer, special_token_ids=special_token_ids,
                                  image_token_id=image_token_id, require_evidence=bool(stages))
    schedule = build_schedule(partition=prepared.pilot_partition,
                              image_position_ids=prepared.choices["A"].inputs.get("image_position_ids"),
                              arm=arm, protected_positions=evidence["protected_positions"],
                              pooling_kernel=pooling_kernel)
    return replace(schedule, geometry={**schedule.geometry, "protection": evidence})


def compact_states(hidden, stage):
    """Mass-weight evolved summaries, not an unweighted mean of group means."""
    import torch
    if hidden.ndim != 3 or hidden.shape[1] != len(stage.input_groups):
        raise ValueError("unexpected raw decoder output geometry")
    if stage.groups == stage.input_groups:
        return hidden
    # A representative remains an input representative under nested grouping.
    source = {row[-1]: i for i, row in enumerate(stage.input_groups)}
    keep = torch.tensor([source[p] for p in stage.keep], device=hidden.device)
    result = hidden.index_select(1, keep)
    merged_rows = []
    for dest, reduction in enumerate(stage.reductions):
        if len(reduction) > 1:
            merged_rows.append((dest, reduction))
    if merged_rows:
        # Padded gathers avoid CUDA atomic scatter-add nondeterminism.
        width = max(len(reduction) for _, reduction in merged_rows)
        indices, weights = [], []
        for _, reduction in merged_rows:
            total = sum(mass for _, mass in reduction)
            padding = width - len(reduction)
            indices.extend([i for i, _ in reduction] + [reduction[0][0]] * padding)
            weights.append([mass / total for _, mass in reduction] + [0.] * padding)
        index = torch.tensor(indices, device=hidden.device)
        weight = torch.tensor(weights, dtype=torch.float32, device=hidden.device)
        values = hidden.index_select(1, index).float().reshape(hidden.shape[0], len(merged_rows), width, hidden.shape[2])
        means = (values * weight[None, :, :, None]).sum(2)
        merged = torch.tensor([dest for dest, _ in merged_rows], device=hidden.device)
        result.index_copy_(1, merged, means.to(hidden.dtype))
    return result


def _legacy_context_present(layer):
    return any("compression_context.<locals>" in getattr(hook, "__qualname__", "")
               and getattr(hook, "__module__", "").endswith("application_pilot_compression")
               for hooks in (layer._forward_hooks, layer._forward_pre_hooks)
               for hook in hooks.values())


@contextmanager
def matrix_context(layers, schedule):
    """Single batch-one forward; never nest with another compression context.

    Native side inputs must be in the original sequence coordinate system on
    every call. Attention masks and RoPE are sliced, never recomputed/reindexed.
    The audit records input lengths for every block (all 60 in production).
    """
    import torch
    layers = tuple(layers)
    if not layers or len({id(layer) for layer in layers}) != len(layers):
        raise ValueError("decoder layers must be nonempty and distinct")
    if schedule.stages and schedule.stages[-1].cut >= len(layers):
        raise ValueError("stage cut outside supplied layers")
    marker = "_decoder_token_matrix_active"
    if any(hasattr(layer, marker) or _legacy_context_present(layer) for layer in layers):
        raise ValueError("compression contexts must not be stacked")
    plans = {stage.cut: stage for stage in schedule.stages}
    audit = {"sequence_lengths": {}, "output_lengths": {}, "stage_counts": {},
             "compression_count": 0, "schedule_sha256": schedule.receipt()["sha256"]}
    active = tuple(range(schedule.original_length))
    next_block = 0
    handles = []

    def before(index, module, args, kwargs):
        nonlocal next_block
        if index != next_block:
            raise ValueError("one ordered forward only; repeated/generation calls unsupported")
        if _legacy_context_present(module):
            raise ValueError("compression contexts must not be stacked")
        # The pinned signature accepts optional per_layer_input as argument 2.
        if len(args) > 2 or (len(args) == 2 and args[1] is not None):
            raise ValueError("only hidden_states and optional None may be positional")
        if (any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value", "per_layer_input"))
                or kwargs.get("use_cache", False)):
            raise ValueError("KV cache/generation/per-layer inputs unsupported")
        h = args[0] if args else kwargs["hidden_states"]
        if not torch.is_tensor(h) or h.ndim != 3 or h.shape[0] != 1 or h.shape[1] != len(active):
            raise ValueError(f"unexpected input geometry at block {index}")
        audit["sequence_lengths"][str(index)] = int(h.shape[1])
        next_block += 1
        if len(active) == schedule.original_length:
            return args, kwargs
        keep = torch.tensor(active, device=h.device)
        kwargs = dict(kwargs)

        def restrict(value, axis):
            if not torch.is_tensor(value) or value.ndim < abs(axis) or value.shape[axis] != schedule.original_length:
                raise ValueError("side input must have original sequence coordinates")
            return value.index_select(axis, keep.to(value.device))

        kwargs["position_ids"] = restrict(kwargs.get("position_ids"), -1)
        rope = kwargs.get("position_embeddings")
        if not isinstance(rope, (tuple, list)) or len(rope) != 2:
            raise ValueError("missing original RoPE pair")
        kwargs["position_embeddings"] = tuple(restrict(x, -2) for x in rope)
        if kwargs.get("cache_position") is not None:
            kwargs["cache_position"] = restrict(kwargs["cache_position"], -1)
        mask = kwargs.get("attention_mask")
        if mask is None:
            if getattr(getattr(module, "self_attn", None), "layer_type", None) != "full_attention":
                raise ValueError("missing explicit multimodal sliding mask")
        else:
            if not torch.is_tensor(mask) or mask.ndim != 4:
                raise ValueError("unexpected native attention-mask shape")
            kwargs["attention_mask"] = restrict(restrict(mask, -2), -1)
        return args, kwargs

    def after(index, module, args, output):
        nonlocal active
        if not torch.is_tensor(output) or output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != len(active):
            raise ValueError("unexpected raw decoder output geometry")
        if index in plans:
            stage = plans[index]
            if str(index) in audit["stage_counts"]:
                raise ValueError("stage fired more than once")
            output = compact_states(output, stage)
            active = stage.keep
            audit["stage_counts"][str(index)] = 1
            audit["compression_count"] += 1
        audit["output_lengths"][str(index)] = int(output.shape[1])
        return output

    try:
        for index, layer in enumerate(layers):
            setattr(layer, marker, True)
            def pre(module, args, kwargs, index=index):
                return before(index, module, args, kwargs)
            def post(module, args, output, index=index):
                return after(index, module, args, output)
            handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(layer.register_forward_hook(post))
        yield audit
        if len(audit["output_lengths"]) != len(layers) or audit["compression_count"] != len(plans):
            raise ValueError("hooks did not complete exactly one forward and every stage")
    finally:
        for handle in handles:
            handle.remove()
        for layer in layers:
            if hasattr(layer, marker):
                delattr(layer, marker)
