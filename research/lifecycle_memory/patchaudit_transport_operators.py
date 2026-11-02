#!/usr/bin/env python3
"""Audited decoder transport operators for PatchAudit.

This module deliberately does not modify the frozen answer-token causal-patch
engine. It reuses its tensor extraction and module-resolution contracts while
adding explicit position-indexed capture, one-shot patches, and multi-layer
persistent clamps.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterator, Mapping, Sequence

from research.lifecycle_memory import run_phase2_reanchored_causal_patch as _base


class TransportGateError(RuntimeError):
    """Raised when an operator cannot prove its intervention contract."""


@dataclass(frozen=True)
class TransportSite:
    site_id: str
    module_name: str
    layer_index: int


@dataclass(frozen=True)
class PositionPartition:
    sequence_length: int
    active_positions: tuple[int, ...]
    full_frame_positions: tuple[int, ...]
    complement_positions: tuple[int, ...]
    padding_positions: tuple[int, ...]
    active_sha256: str
    full_frame_sha256: str
    complement_sha256: str


@dataclass(frozen=True)
class CapturedState:
    site_id: str
    layer_index: int
    full_frame: Any
    active_sequence: Any | None
    sequence_axis: int
    full_frame_count: int
    active_count: int


@dataclass(frozen=True)
class PatchInstruction:
    site: TransportSite
    source_selected: Any
    recipient_positions: tuple[int, ...]
    operator: str


@dataclass
class TransportHookAudit:
    site_id: str
    layer_index: int
    operator: str
    status: str = "pending"
    fire_count: int = 0
    sequence_axis: int | None = None
    sequence_length: int | None = None
    selected_count: int = 0
    preserved_count: int = 0
    selected_positions_sha256: str | None = None
    source_shape: list[int] | None = None
    recipient_shape: list[int] | None = None
    patched_shape: list[int] | None = None
    changed_selected_value_count: int = 0
    max_abs_change_selected: float = 0.0
    max_abs_change_preserved: float = 0.0
    finite: bool = False
    hook_count_before: int | None = None
    hook_count_after: int | None = None
    hook_removed: bool = False
    error: str | None = None

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_ints(values: Sequence[int]) -> str:
    raw = json.dumps(list(values), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _strict_positions(
    values: Sequence[int], *, sequence_length: int, label: str
) -> tuple[int, ...]:
    positions = tuple(int(value) for value in values)
    if tuple(sorted(set(positions))) != positions:
        raise TransportGateError(f"{label} must be sorted and unique")
    if any(value < 0 or value >= sequence_length for value in positions):
        raise TransportGateError(
            f"{label} contains an out-of-range position for T={sequence_length}"
        )
    return positions


def position_partition(
    *,
    sequence_length: int,
    active_positions: Sequence[int],
    full_frame_positions: Sequence[int],
) -> PositionPartition:
    """Prove that full-frame and complement partition active positions."""
    if sequence_length <= 0:
        raise TransportGateError("sequence length must be positive")
    active = _strict_positions(
        active_positions, sequence_length=sequence_length, label="active_positions"
    )
    full_frame = _strict_positions(
        full_frame_positions,
        sequence_length=sequence_length,
        label="full_frame_positions",
    )
    active_set = set(active)
    full_set = set(full_frame)
    if not active:
        raise TransportGateError("active position set is empty")
    if not full_frame or not full_set <= active_set:
        raise TransportGateError(
            "full-frame positions must be a non-empty subset of active positions"
        )
    complement = tuple(value for value in active if value not in full_set)
    padding = tuple(value for value in range(sequence_length) if value not in active_set)
    if full_set & set(complement):
        raise TransportGateError("full-frame and complement positions overlap")
    if full_set | set(complement) != active_set:
        raise TransportGateError(
            "full-frame and complement positions do not partition active positions"
        )
    return PositionPartition(
        sequence_length=sequence_length,
        active_positions=active,
        full_frame_positions=full_frame,
        complement_positions=complement,
        padding_positions=padding,
        active_sha256=_sha256_ints(active),
        full_frame_sha256=_sha256_ints(full_frame),
        complement_sha256=_sha256_ints(complement),
    )


def active_positions_from_inputs(inputs: Mapping[str, Any], sequence_length: int) -> tuple[int, ...]:
    """Return non-padding positions for a batch-size-one processor result."""
    import torch

    raw = inputs.get("attention_mask")
    if raw is None:
        return tuple(range(sequence_length))
    if not torch.is_tensor(raw):
        raise TransportGateError("attention_mask is not a tensor")
    mask = raw.detach()
    if mask.ndim != 2 or int(mask.shape[0]) != 1:
        raise TransportGateError(
            f"attention_mask must have shape [1,T], observed {list(mask.shape)}"
        )
    if int(mask.shape[1]) != sequence_length:
        raise TransportGateError(
            "attention-mask length differs from the proved decoder sequence length"
        )
    return tuple(
        int(value)
        for value in torch.nonzero(mask[0] != 0, as_tuple=False).reshape(-1).tolist()
    )


def _index_tensor(positions: Sequence[int], *, device: Any) -> Any:
    import torch

    return torch.tensor(list(positions), dtype=torch.long, device=device)


def _sequence_axis(tensor: Any, sequence_length: int) -> int:
    shape = list(tensor.shape)
    candidates = [
        axis
        for axis, size in enumerate(shape[:-1])
        if int(size) == int(sequence_length)
    ]
    if len(candidates) != 1:
        raise TransportGateError(
            f"cannot uniquely identify sequence axis T={sequence_length} in {shape}"
        )
    return candidates[0]


@contextmanager
def capture_transport_states(
    *,
    model: Any,
    sites: Sequence[TransportSite],
    partition: PositionPartition,
    detailed_site_ids: Sequence[str],
) -> Iterator[dict[str, CapturedState]]:
    """Capture native donor states from exactly one unpatched forward."""
    captures: dict[str, CapturedState] = {}
    handles: list[Any] = []
    hook_counts_before: dict[str, int] = {}
    detailed = set(str(value) for value in detailed_site_ids)
    expected = {site.site_id for site in sites}
    if len(expected) != len(sites):
        raise TransportGateError("capture sites contain duplicate site IDs")
    if not detailed <= expected:
        raise TransportGateError("detailed capture site is not in the site inventory")
    module_names = [site.module_name for site in sites]
    if len(set(module_names)) != len(module_names):
        raise TransportGateError("capture sites contain duplicate module paths")

    for site in sites:
        module = _base.module_by_path(model, site.module_name)
        hook_counts_before[site.site_id] = len(
            getattr(module, "_forward_hooks", {})
        )

        def make_hook(current: TransportSite):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                if current.site_id in captures:
                    raise TransportGateError(
                        f"capture hook fired more than once: {current.site_id}"
                    )
                tensor, _ = _base.extract_patch_tensor(output)
                axis = _sequence_axis(tensor, partition.sequence_length)
                full_index = _index_tensor(
                    partition.full_frame_positions, device=tensor.device
                )
                active_index = _index_tensor(
                    partition.active_positions, device=tensor.device
                )
                full = tensor.index_select(axis, full_index).detach().clone()
                active = (
                    tensor.index_select(axis, active_index).detach().clone()
                    if current.site_id in detailed
                    else None
                )
                captures[current.site_id] = CapturedState(
                    site_id=current.site_id,
                    layer_index=current.layer_index,
                    full_frame=full,
                    active_sequence=active,
                    sequence_axis=axis,
                    full_frame_count=len(partition.full_frame_positions),
                    active_count=len(partition.active_positions),
                )

            return hook

        handles.append(module.register_forward_hook(make_hook(site)))
    try:
        yield captures
    finally:
        for handle in reversed(handles):
            handle.remove()
        removal_failures = []
        for site in sites:
            module = _base.module_by_path(model, site.module_name)
            observed = len(getattr(module, "_forward_hooks", {}))
            expected_count = hook_counts_before[site.site_id]
            if observed != expected_count:
                removal_failures.append(
                    f"{site.site_id}:before={expected_count}:after={observed}"
                )
        if removal_failures:
            raise TransportGateError(
                "capture hook removal failed: " + "; ".join(removal_failures)
            )


def assert_complete_capture(
    captures: Mapping[str, CapturedState],
    sites: Sequence[TransportSite],
    *,
    detailed_site_ids: Sequence[str],
) -> None:
    import torch

    expected = {site.site_id for site in sites}
    observed = set(captures)
    if observed != expected:
        raise TransportGateError(
            f"capture inventory mismatch: missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}"
        )
    for site_id in detailed_site_ids:
        if captures[site_id].active_sequence is None:
            raise TransportGateError(f"detailed capture is missing for {site_id}")
    finite_checks: list[tuple[str, Any]] = []
    for site in sites:
        capture = captures[site.site_id]
        if int(capture.full_frame.shape[capture.sequence_axis]) != int(
            capture.full_frame_count
        ):
            raise TransportGateError(
                f"full-frame capture count drift at {site.site_id}"
            )
        finite_checks.append(
            (f"{site.site_id}:full_frame", torch.isfinite(capture.full_frame).all())
        )
        if capture.active_sequence is not None:
            if int(
                capture.active_sequence.shape[capture.sequence_axis]
            ) != int(capture.active_count):
                raise TransportGateError(
                    f"active-sequence capture count drift at {site.site_id}"
                )
            finite_checks.append(
                (
                    f"{site.site_id}:active_sequence",
                    torch.isfinite(capture.active_sequence).all(),
                )
            )
    if finite_checks:
        values = (
            torch.stack([value for _label, value in finite_checks])
            .detach()
            .cpu()
            .tolist()
        )
        failed = [
            label
            for (label, _value), finite in zip(finite_checks, values)
            if not bool(finite)
        ]
        if failed:
            raise TransportGateError(
                f"native capture contains nonfinite state: {failed[:8]}"
            )


def selected_from_capture(
    capture: CapturedState,
    *,
    operator: str,
    partition: PositionPartition,
) -> Any:
    """Select the donor tensor required by one operator."""
    if operator in {"single_full_frame", "persistent_full_frame"}:
        return capture.full_frame
    active = capture.active_sequence
    if active is None:
        raise TransportGateError(
            f"operator {operator} requires a detailed capture at {capture.site_id}"
        )
    if operator == "full_sequence_once":
        return active
    if operator == "complement_once":
        import torch

        active_offset = {
            position: offset
            for offset, position in enumerate(partition.active_positions)
        }
        offsets = [active_offset[position] for position in partition.complement_positions]
        index = torch.tensor(offsets, dtype=torch.long, device=active.device)
        return active.index_select(capture.sequence_axis, index)
    raise TransportGateError(f"unknown operator: {operator}")


def recipient_positions_for_operator(
    operator: str, partition: PositionPartition
) -> tuple[int, ...]:
    if operator in {"single_full_frame", "persistent_full_frame"}:
        return partition.full_frame_positions
    if operator == "complement_once":
        return partition.complement_positions
    if operator == "full_sequence_once":
        return partition.active_positions
    raise TransportGateError(f"unknown operator: {operator}")


@contextmanager
def patch_transport_outputs(
    *,
    model: Any,
    instructions: Sequence[PatchInstruction],
    sequence_length: int,
) -> Iterator[list[TransportHookAudit]]:
    """Install an exact one-shot patch or ordered persistent clamp."""
    import torch

    if not instructions:
        raise TransportGateError("patch instruction list is empty")
    site_ids = [row.site.site_id for row in instructions]
    module_names = [row.site.module_name for row in instructions]
    if len(set(site_ids)) != len(site_ids):
        raise TransportGateError("patch instructions contain duplicate site IDs")
    if len(set(module_names)) != len(module_names):
        raise TransportGateError("patch instructions contain duplicate module paths")
    audits = [
        TransportHookAudit(
            site_id=row.site.site_id,
            layer_index=row.site.layer_index,
            operator=row.operator,
        )
        for row in instructions
    ]
    handles: list[Any] = []
    pending_metrics: list[tuple[TransportHookAudit, Any, Any]] = []

    for instruction, audit in zip(instructions, audits):
        module = _base.module_by_path(model, instruction.site.module_name)
        audit.hook_count_before = len(getattr(module, "_forward_hooks", {}))
        positions = tuple(instruction.recipient_positions)

        def make_hook(
            current: PatchInstruction,
            current_audit: TransportHookAudit,
            current_positions: tuple[int, ...],
        ):
            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                try:
                    current_audit.fire_count += 1
                    if current_audit.fire_count != 1:
                        raise TransportGateError(
                            f"hook fired more than once: {current.site.site_id}"
                        )
                    tensor, location = _base.extract_patch_tensor(output)
                    axis = _sequence_axis(tensor, sequence_length)
                    selected = _strict_positions(
                        current_positions,
                        sequence_length=sequence_length,
                        label=f"{current.site.site_id}.recipient_positions",
                    )
                    index = _index_tensor(selected, device=tensor.device)
                    preserved = tuple(
                        value for value in range(sequence_length) if value not in set(selected)
                    )
                    recipient = tensor.index_select(axis, index)
                    source = current.source_selected.to(
                        device=tensor.device, dtype=tensor.dtype
                    )
                    current_audit.sequence_axis = axis
                    current_audit.sequence_length = sequence_length
                    current_audit.selected_count = len(selected)
                    current_audit.preserved_count = len(preserved)
                    current_audit.selected_positions_sha256 = _sha256_ints(selected)
                    current_audit.source_shape = list(source.shape)
                    current_audit.recipient_shape = list(recipient.shape)
                    if source.shape != recipient.shape:
                        raise TransportGateError(
                            f"shape mismatch at {current.site.site_id}: "
                            f"source={list(source.shape)} recipient={list(recipient.shape)}"
                        )
                    patched = tensor.clone()
                    patched.index_copy_(axis, index, source)
                    selected_delta = patched.index_select(axis, index) - recipient
                    current_audit.patched_shape = list(patched.shape)
                    current_audit.max_abs_change_preserved = 0.0
                    delta_abs = selected_delta.detach().float().abs()
                    pending_metrics.append(
                        (
                            current_audit,
                            delta_abs,
                            torch.isfinite(source).all(),
                        )
                    )
                    return _base.replace_patch_tensor(output, location, patched)
                except Exception as exc:
                    current_audit.status = "fail"
                    current_audit.error = f"{type(exc).__name__}: {exc}"
                    raise

            return hook

        handles.append(
            module.register_forward_hook(make_hook(instruction, audit, positions))
        )
    try:
        yield audits
    finally:
        for handle in reversed(handles):
            handle.remove()
        if pending_metrics:
            metric_rows = []
            for _audit, delta_abs, source_finite in pending_metrics:
                maximum = (
                    delta_abs.max()
                    if delta_abs.numel()
                    else torch.zeros((), device=delta_abs.device)
                )
                changed = (delta_abs.reshape(-1) > 0).sum().to(torch.float64)
                metric_rows.append(
                    torch.stack(
                        [
                            maximum.to(torch.float64),
                            changed,
                            source_finite.to(torch.float64),
                        ]
                    )
                )
            values = torch.stack(metric_rows).detach().cpu().tolist()
            for (audit, _delta, _finite), metrics in zip(
                pending_metrics, values
            ):
                audit.max_abs_change_selected = float(metrics[0])
                audit.changed_selected_value_count = int(metrics[1])
                audit.finite = bool(metrics[2])
                if audit.status != "fail":
                    if audit.finite:
                        audit.status = "pass"
                    else:
                        audit.status = "fail"
                        audit.error = "source selected state was nonfinite"
        for instruction, audit in zip(instructions, audits):
            module = _base.module_by_path(model, instruction.site.module_name)
            audit.hook_count_after = len(getattr(module, "_forward_hooks", {}))
            audit.hook_removed = audit.hook_count_after == audit.hook_count_before
            if not audit.hook_removed and audit.status != "fail":
                audit.status = "fail"
                audit.error = "hook was not removed"


def validate_patch_audits(
    audits: Sequence[TransportHookAudit],
    *,
    expected_site_ids: Sequence[str],
    allow_noop: bool,
) -> None:
    expected = list(expected_site_ids)
    observed = [audit.site_id for audit in audits]
    if observed != expected:
        raise TransportGateError(
            f"patch audit order mismatch: expected={expected} observed={observed}"
        )
    failures: list[str] = []
    changed_any = any(audit.changed_selected_value_count > 0 for audit in audits)
    for audit in audits:
        if audit.status != "pass":
            failures.append(f"{audit.site_id}:status={audit.status}:{audit.error}")
        if audit.fire_count != 1:
            failures.append(f"{audit.site_id}:fire_count={audit.fire_count}")
        if not audit.hook_removed:
            failures.append(f"{audit.site_id}:hook_not_removed")
        if not audit.finite:
            failures.append(f"{audit.site_id}:nonfinite")
        if audit.max_abs_change_preserved != 0.0:
            failures.append(f"{audit.site_id}:outside_change")
        if (
            not allow_noop
            and audit.operator != "persistent_full_frame"
            and audit.selected_count > 0
            and audit.changed_selected_value_count == 0
        ):
            failures.append(f"{audit.site_id}:matched_intervention_noop")
    if not allow_noop and not changed_any:
        failures.append("entire_intervention_was_noop")
    if failures:
        raise TransportGateError("patch audit failed: " + "; ".join(failures))
