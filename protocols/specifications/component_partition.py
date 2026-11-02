#!/usr/bin/env python3
"""Audited token-component partitions for PatchAudit.

The module is intentionally standard-library-only.  It turns processor-proved
token boundaries into an immutable partition receipt, factorial component
unions, and deterministic candidate-crop cardinality controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from typing import Any, Mapping, Sequence


COMPONENT_GROUP_NAMES = (
    "full_frame",
    "candidate_crops",
    "literal_user_prompt",
    "answer_prefix",
    "structural_remainder",
)
NONVISUAL_GROUP_NAMES = (
    "literal_user_prompt",
    "answer_prefix",
    "structural_remainder",
)
FACTORIAL_GROUP_NAMES = (
    "candidate_crops",
    "literal_user_prompt",
    "answer_prefix",
    "structural_remainder",
)


class ComponentPartitionError(ValueError):
    """Raised when a token partition cannot prove its audit contract."""


def _sha256_positions(values: Sequence[int]) -> str:
    encoded = json.dumps(
        list(values), ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _strict_integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComponentPartitionError(f"{label} must be an integer")
    if value < minimum:
        raise ComponentPartitionError(f"{label} must be at least {minimum}")
    return value


def _strict_positions(
    values: Sequence[int], *, sequence_length: int, label: str
) -> tuple[int, ...]:
    positions = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in positions
    ):
        raise ComponentPartitionError(f"{label} must contain only integers")
    if positions != tuple(sorted(set(positions))):
        raise ComponentPartitionError(f"{label} must be sorted and unique")
    if any(value < 0 or value >= sequence_length for value in positions):
        raise ComponentPartitionError(
            f"{label} contains an out-of-range position for "
            f"sequence_length={sequence_length}"
        )
    return positions


@dataclass(frozen=True)
class PositionGroup:
    """One hash-bound token-position group."""

    name: str
    positions: tuple[int, ...]
    count: int
    sha256: str
    structurally_absent: bool

    @classmethod
    def from_positions(
        cls, name: str, positions: Sequence[int]
    ) -> "PositionGroup":
        canonical = tuple(positions)
        return cls(
            name=name,
            positions=canonical,
            count=len(canonical),
            sha256=_sha256_positions(canonical),
            structurally_absent=not canonical,
        )


@dataclass(frozen=True)
class ComponentPartition:
    """A disjoint, exhaustive partition of all active decoder positions."""

    sequence_length: int
    active: PositionGroup
    complement: PositionGroup
    padding: PositionGroup
    image_slots: tuple[PositionGroup, PositionGroup, PositionGroup]
    full_frame: PositionGroup
    candidate_crops: PositionGroup
    literal_user_prompt: PositionGroup
    answer_prefix: PositionGroup
    structural_remainder: PositionGroup
    literal_user_prompt_start: int
    literal_user_prompt_count: int
    answer_prefix_start: int
    answer_prefix_token_count: int

    @property
    def active_positions(self) -> tuple[int, ...]:
        return self.active.positions

    @property
    def complement_positions(self) -> tuple[int, ...]:
        return self.complement.positions

    @property
    def active_count(self) -> int:
        return self.active.count

    @property
    def complement_count(self) -> int:
        return self.complement.count

    @property
    def active_sha256(self) -> str:
        return self.active.sha256

    @property
    def complement_sha256(self) -> str:
        return self.complement.sha256

    def component(self, name: str) -> PositionGroup:
        if name not in COMPONENT_GROUP_NAMES:
            raise ComponentPartitionError(f"unknown component group: {name}")
        return getattr(self, name)

    def components(self) -> tuple[PositionGroup, ...]:
        return tuple(self.component(name) for name in COMPONENT_GROUP_NAMES)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reject_overlaps(groups: Sequence[PositionGroup]) -> None:
    owners: dict[int, str] = {}
    collisions: list[str] = []
    for group in groups:
        for position in group.positions:
            previous = owners.get(position)
            if previous is not None:
                collisions.append(f"{position}:{previous}+{group.name}")
            else:
                owners[position] = group.name
    if collisions:
        raise ComponentPartitionError(
            "component groups overlap: " + ", ".join(collisions[:12])
        )


def _validate_exhaustive_partition(
    *, active: PositionGroup, components: Sequence[PositionGroup]
) -> None:
    _reject_overlaps(components)
    union = tuple(
        sorted(position for group in components for position in group.positions)
    )
    if union != active.positions:
        missing = sorted(set(active.positions) - set(union))
        extra = sorted(set(union) - set(active.positions))
        raise ComponentPartitionError(
            f"component partition is not exhaustive: missing={missing[:12]} "
            f"extra={extra[:12]}"
        )


def build_component_partition(
    *,
    sequence_length: int,
    active_positions: Sequence[int],
    image_slot_positions: Sequence[Sequence[int]],
    literal_user_prompt_start: int,
    literal_user_prompt_count: int,
    answer_prefix_token_count: int,
) -> ComponentPartition:
    """Build and prove the token-position component partition.

    ``image_slot_positions`` must contain exactly three processor-proved sets:
    the full-frame slot followed by the two candidate-crop slots.  The answer
    prefix is the contiguous final ``answer_prefix_token_count`` active tokens,
    matching the existing append-only answer-scoring contract.
    """

    length = _strict_integer(sequence_length, label="sequence_length", minimum=1)
    active_positions_tuple = _strict_positions(
        active_positions, sequence_length=length, label="active_positions"
    )
    if not active_positions_tuple:
        raise ComponentPartitionError("active_positions must be nonempty")
    if len(image_slot_positions) != 3:
        raise ComponentPartitionError(
            "image_slot_positions must contain exactly three image-slot sets"
        )

    slot_values = tuple(
        _strict_positions(
            values,
            sequence_length=length,
            label=f"image_slot_positions[{slot_index}]",
        )
        for slot_index, values in enumerate(image_slot_positions)
    )
    if not slot_values[0]:
        raise ComponentPartitionError("full-frame image slot is missing or empty")
    if not slot_values[1] or not slot_values[2]:
        raise ComponentPartitionError(
            "both candidate-crop image slots must be present and nonempty"
        )

    image_slots = (
        PositionGroup.from_positions("image_slot_0_full_frame", slot_values[0]),
        PositionGroup.from_positions("image_slot_1_candidate_crop", slot_values[1]),
        PositionGroup.from_positions("image_slot_2_candidate_crop", slot_values[2]),
    )
    _reject_overlaps(image_slots)
    active_set = set(active_positions_tuple)
    for slot in image_slots:
        if not set(slot.positions) <= active_set:
            raise ComponentPartitionError(
                f"{slot.name} is not a subset of active_positions"
            )

    prompt_start = _strict_integer(
        literal_user_prompt_start, label="literal_user_prompt_start"
    )
    prompt_count = _strict_integer(
        literal_user_prompt_count,
        label="literal_user_prompt_count",
        minimum=1,
    )
    prompt_stop = prompt_start + prompt_count
    if prompt_stop > length:
        raise ComponentPartitionError(
            "literal user prompt boundary extends beyond sequence_length"
        )
    prompt_positions = tuple(range(prompt_start, prompt_stop))
    if not set(prompt_positions) <= active_set:
        raise ComponentPartitionError(
            "literal user prompt boundary is not contiguous within active_positions"
        )

    prefix_count = _strict_integer(
        answer_prefix_token_count,
        label="answer_prefix_token_count",
        minimum=1,
    )
    if prefix_count > len(active_positions_tuple):
        raise ComponentPartitionError(
            "answer prefix is larger than the active sequence"
        )
    prefix_positions = active_positions_tuple[-prefix_count:]
    prefix_start = prefix_positions[0]
    if prefix_positions != tuple(range(prefix_start, prefix_start + prefix_count)):
        raise ComponentPartitionError(
            "answer prefix boundary is not contiguous at the active-sequence tail"
        )
    if prefix_start < prompt_stop:
        raise ComponentPartitionError(
            "answer prefix begins before the literal user prompt ends"
        )

    full_frame = PositionGroup.from_positions("full_frame", slot_values[0])
    candidate_crops = PositionGroup.from_positions(
        "candidate_crops", tuple(sorted(slot_values[1] + slot_values[2]))
    )
    literal_user_prompt = PositionGroup.from_positions(
        "literal_user_prompt", prompt_positions
    )
    answer_prefix = PositionGroup.from_positions("answer_prefix", prefix_positions)
    assigned = (
        set(full_frame.positions)
        | set(candidate_crops.positions)
        | set(literal_user_prompt.positions)
        | set(answer_prefix.positions)
    )
    structural_remainder = PositionGroup.from_positions(
        "structural_remainder",
        tuple(
            position
            for position in active_positions_tuple
            if position not in assigned
        ),
    )
    active = PositionGroup.from_positions("active", active_positions_tuple)
    components = (
        full_frame,
        candidate_crops,
        literal_user_prompt,
        answer_prefix,
        structural_remainder,
    )
    _validate_exhaustive_partition(active=active, components=components)

    complement_positions = tuple(
        position
        for position in active_positions_tuple
        if position not in set(full_frame.positions)
    )
    complement = PositionGroup.from_positions("complement", complement_positions)
    expected_complement = tuple(
        sorted(
            position
            for group in components[1:]
            for position in group.positions
        )
    )
    if complement.positions != expected_complement:
        raise ComponentPartitionError(
            "complement is not the exhaustive union of non-full-frame components"
        )
    padding = PositionGroup.from_positions(
        "padding",
        tuple(position for position in range(length) if position not in active_set),
    )

    return ComponentPartition(
        sequence_length=length,
        active=active,
        complement=complement,
        padding=padding,
        image_slots=image_slots,
        full_frame=full_frame,
        candidate_crops=candidate_crops,
        literal_user_prompt=literal_user_prompt,
        answer_prefix=answer_prefix,
        structural_remainder=structural_remainder,
        literal_user_prompt_start=prompt_start,
        literal_user_prompt_count=prompt_count,
        answer_prefix_start=prefix_start,
        answer_prefix_token_count=prefix_count,
    )


# A concise alias for callers using the token-position ``position_partition`` style.
component_partition = build_component_partition


@dataclass(frozen=True)
class FactorialCell:
    """One unique nonempty position set in the component factorial."""

    cell_index: int
    cell_id: str
    included_groups: tuple[str, ...]
    positions: tuple[int, ...]
    count: int
    sha256: str
    requested_group_sets: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class FactorialDesign:
    """Deduplicated full factorial with structural-absence receipts."""

    component_order: tuple[str, ...]
    structurally_absent_groups: tuple[str, ...]
    requested_combination_count: int
    nonempty_requested_combination_count: int
    empty_requested_group_sets: tuple[tuple[str, ...], ...]
    duplicate_requested_combination_count: int
    cells: tuple[FactorialCell, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def factorial_component_unions(
    partition: ComponentPartition,
    *,
    component_order: Sequence[str] = FACTORIAL_GROUP_NAMES,
) -> FactorialDesign:
    """Return all nonempty component unions, deduplicated by position set."""

    order = tuple(str(name) for name in component_order)
    if not order or len(set(order)) != len(order):
        raise ComponentPartitionError("factorial component order is empty or duplicated")
    if any(name not in COMPONENT_GROUP_NAMES for name in order):
        raise ComponentPartitionError("factorial component order contains an unknown group")
    groups = {name: partition.component(name) for name in order}
    aliases_by_positions: dict[tuple[int, ...], list[tuple[str, ...]]] = {}
    empty_requests: list[tuple[str, ...]] = []
    requested_count = 0
    for width in range(1, len(order) + 1):
        for requested in itertools.combinations(order, width):
            requested_count += 1
            positions = tuple(
                sorted(
                    {
                        position
                        for name in requested
                        for position in groups[name].positions
                    }
                )
            )
            if not positions:
                empty_requests.append(requested)
                continue
            aliases_by_positions.setdefault(positions, []).append(requested)

    ordered_items = sorted(
        aliases_by_positions.items(),
        key=lambda item: (
            len(item[0]),
            item[0],
            item[1][0],
        ),
    )
    cells: list[FactorialCell] = []
    for cell_index, (positions, requested_group_sets) in enumerate(
        ordered_items, start=1
    ):
        included = tuple(
            name
            for name in order
            if groups[name].positions
            and set(groups[name].positions) <= set(positions)
        )
        digest = _sha256_positions(positions)
        cells.append(
            FactorialCell(
                cell_index=cell_index,
                cell_id=f"component_factorial_{cell_index:02d}_{digest[:12]}",
                included_groups=included,
                positions=positions,
                count=len(positions),
                sha256=digest,
                requested_group_sets=tuple(requested_group_sets),
            )
        )

    nonempty_requested = requested_count - len(empty_requests)
    duplicate_count = nonempty_requested - len(cells)
    if requested_count != (2 ** len(order)) - 1:
        raise ComponentPartitionError("factorial request count is incomplete")
    if any(not cell.positions for cell in cells):
        raise ComponentPartitionError("factorial contains an empty union")
    if len({cell.positions for cell in cells}) != len(cells):
        raise ComponentPartitionError("factorial position sets were not deduplicated")

    return FactorialDesign(
        component_order=order,
        structurally_absent_groups=tuple(
            name for name in order if groups[name].structurally_absent
        ),
        requested_combination_count=requested_count,
        nonempty_requested_combination_count=nonempty_requested,
        empty_requested_group_sets=tuple(empty_requests),
        duplicate_requested_combination_count=duplicate_count,
        cells=tuple(cells),
    )


build_factorial_unions = factorial_component_unions


@dataclass(frozen=True)
class CandidateCropControl:
    """A deterministic same-cardinality crop-token control receipt."""

    namespace: str
    model: str
    group: str
    replicate: int
    requested_count: int
    positions: tuple[int, ...]
    count: int
    sha256: str
    candidate_crop_pool_count: int
    candidate_crop_pool_sha256: str
    seed_sha256: str
    structurally_absent: bool


def deterministic_candidate_crop_sample(
    *,
    candidate_crop_positions: Sequence[int],
    sample_count: int,
    namespace: str,
    model: str,
    group: str,
    replicate: int = 0,
) -> CandidateCropControl:
    """Sample crop positions deterministically from a namespace-bound seed."""

    pool = tuple(candidate_crop_positions)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in pool):
        raise ComponentPartitionError(
            "candidate_crop_positions must contain only integers"
        )
    if pool != tuple(sorted(set(pool))):
        raise ComponentPartitionError(
            "candidate_crop_positions must be sorted and unique"
        )
    count = _strict_integer(sample_count, label="sample_count")
    if count > len(pool):
        raise ComponentPartitionError(
            f"requested sample_count={count} exceeds candidate-crop pool "
            f"size={len(pool)}"
        )
    labels = {
        "namespace": str(namespace),
        "model": str(model),
        "group": str(group),
        "replicate": _strict_integer(replicate, label="replicate"),
    }
    if any(not value for key, value in labels.items() if key != "replicate"):
        raise ComponentPartitionError(
            "namespace, model, and group must all be nonempty"
        )
    sampling_labels = {
        key: value for key, value in labels.items() if key != "replicate"
    }
    seed_bytes = json.dumps(
        sampling_labels, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    seed_sha256 = hashlib.sha256(seed_bytes).hexdigest()
    ranked_pool = tuple(
        sorted(
            pool,
            key=lambda position: hashlib.sha256(
                f"{seed_sha256}|{position}".encode("ascii")
            ).hexdigest(),
        )
    )
    if count:
        offset = int(labels["replicate"]) % len(ranked_pool)
        sampled = tuple(
            sorted(
                ranked_pool[(offset + index) % len(ranked_pool)]
                for index in range(count)
            )
        )
    else:
        sampled = ()
    return CandidateCropControl(
        namespace=labels["namespace"],
        model=labels["model"],
        group=labels["group"],
        replicate=labels["replicate"],
        requested_count=count,
        positions=sampled,
        count=len(sampled),
        sha256=_sha256_positions(sampled),
        candidate_crop_pool_count=len(pool),
        candidate_crop_pool_sha256=_sha256_positions(pool),
        seed_sha256=seed_sha256,
        structurally_absent=count == 0,
    )


def sample_candidate_crop_control(
    partition: ComponentPartition,
    *,
    namespace: str,
    model: str,
    group: str,
) -> CandidateCropControl:
    """Build one crop-token control matching a named nonvisual component."""

    if group not in NONVISUAL_GROUP_NAMES:
        raise ComponentPartitionError(
            f"same-cardinality crop controls require a nonvisual group, got {group}"
        )
    target = partition.component(group)
    return deterministic_candidate_crop_sample(
        candidate_crop_positions=partition.candidate_crops.positions,
        sample_count=target.count,
        namespace=namespace,
        model=model,
        group=group,
    )


def nonvisual_candidate_crop_controls(
    partition: ComponentPartition, *, namespace: str, model: str
) -> Mapping[str, CandidateCropControl]:
    """Return deterministic controls for every named nonvisual component."""

    return {
        group: sample_candidate_crop_control(
            partition, namespace=namespace, model=model, group=group
        )
        for group in NONVISUAL_GROUP_NAMES
    }
