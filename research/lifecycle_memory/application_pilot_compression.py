"""Sequence-shortening pilot for the pinned, non-KV-sharing Gemma4-31B.

No generation/cache support. Geometry follows Gemma4VisionPooler's ordering.
Pairs are averaged and placed at the later member's original position. Masks
and RoPE are restricted to those representative coordinates, never reindexed.
Expansion broadcasts evolved summaries, without retaining original states.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Any


# group, raw output cut, expansion input, operator. Insertion order is frozen.
COMPRESSION_ARMS = {
    "native": (None, None, None, None),
    "merge_F_L34": ("F", 34, None, "merge"),
    "merge_F_L34_expand47": ("F", 34, 47, "merge"),
    "merge_F_L22": ("F", 22, None, "merge"),
    "merge_C_L34": ("C", 34, None, "merge"),
    "merge_C_L22": ("C", 22, None, "merge"),
    "prune_F_L34": ("F", 34, None, "prune"),
}
ROLE_DEPTH_WEIGHTS = {"merge_F_L34": 1, "merge_C_L34": -1,
                      "merge_F_L22": -1, "merge_C_L22": 1}


@dataclass(frozen=True)
class CompressionPlan:
    original_length: int
    pairs: tuple[tuple[int, int], ...]
    keep: tuple[int, ...]
    expand_index: tuple[int, ...]
    method: str = "merge"

    @property
    def compact_length(self):
        return len(self.keep)

    def receipt(self):
        record = asdict(self)
        record["sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        return record


def spatial_coordinates(image_position_ids, slot: int, pooling_kernel: int, expected_count: int):
    """Recover the exact sorted non-padding pool-bin order, not a guessed grid."""
    coords = image_position_ids[slot].detach().cpu().tolist()
    valid = [(int(x), int(y)) for x, y in coords if x >= 0 and y >= 0]
    if not valid or len(set(valid)) != len(valid):
        raise ValueError("missing/duplicate processor spatial coordinates")
    width = max(x for x, _ in valid) + 1
    if width % pooling_kernel:
        raise ValueError("processor width not divisible by pooling kernel")
    bins = {}
    for x, y in valid:
        key = (x // pooling_kernel, y // pooling_kernel)
        bins[key] = bins.get(key, 0) + 1
    if any(n != pooling_kernel**2 for n in bins.values()):
        raise ValueError("partial pool bins are outside the pilot geometry")
    ordered = sorted(bins, key=lambda xy: xy[0] + (width // pooling_kernel) * xy[1])
    if len(ordered) != expected_count:
        raise ValueError(f"processor pool geometry {len(ordered)} != slot count {expected_count}")
    return ordered


def make_plan(*, sequence_length, slots, image_position_ids, pooling_kernel=3,
              group="F", remove_count=133, method="merge"):
    if method not in ("merge", "prune") or group not in ("F", "C"):
        raise ValueError("unknown compression operator")
    candidates = []
    for slot in ((0,) if group == "F" else (1, 2)):
        positions = tuple(slots[slot])
        coords = spatial_coordinates(image_position_ids, slot, pooling_kernel, len(positions))
        lookup = dict(zip(coords, positions))
        # Pair vertical neighbours where possible; horizontal pairs cover an
        # odd leftover row. Every pair stays within one processor image slot.
        used = set()
        for x, y in sorted(coords, key=lambda xy: (xy[1], xy[0])):
            if (x, y) in used:
                continue
            other = next((q for q in ((x, y + 1), (x + 1, y)) if q in lookup and q not in used), None)
            if other is None:
                continue
            used.update(((x, y), other))
            a, b = sorted((lookup[x, y], lookup[other]))
            candidates.append((slot, a, b))
    if len(candidates) < remove_count or remove_count < 0:
        raise ValueError("not enough disjoint spatial pairs for fixed removal count")
    # Fixed spatially spread role control, independent of hidden states/labels.
    if group == "C":
        candidates.sort(key=lambda p: hashlib.sha256(f"application-pilot-v1:{p}".encode()).hexdigest())
    pairs = tuple(sorted((a, b) for _, a, b in candidates[:remove_count]))
    removed = {a for a, _ in pairs}
    if len(removed) != remove_count or any(sequence_length - 1 in p for p in pairs):
        raise ValueError("invalid merge or readout position selected")
    keep = tuple(i for i in range(sequence_length) if i not in removed)
    compact_index = {p: i for i, p in enumerate(keep)}
    representative = {a: b for a, b in pairs}
    expand_index = tuple(compact_index[representative.get(p, p)] for p in range(sequence_length))
    return CompressionPlan(sequence_length, pairs, keep, expand_index, method)


def compact_states(hidden, plan: CompressionPlan):
    import torch
    keep = torch.tensor(plan.keep, device=hidden.device)
    result = hidden.index_select(1, keep)
    if plan.method == "merge" and plan.pairs:
        left = torch.tensor([p[0] for p in plan.pairs], device=hidden.device)
        right = torch.tensor([p[1] for p in plan.pairs], device=hidden.device)
        dest = torch.tensor([plan.keep.index(p[1]) for p in plan.pairs], device=hidden.device)
        means = ((hidden.index_select(1, left).float() + hidden.index_select(1, right).float()) * .5).to(hidden.dtype)
        result.index_copy_(1, dest, means)
    return result


def expand_states(hidden, plan):
    import torch
    return hidden.index_select(1, torch.tensor(plan.expand_index, device=hidden.device))


def architecture_gate(model, *, production=True):
    text = model.model.language_model
    config = text.config
    if production and (config.num_hidden_layers != 60 or config.hidden_size != 5376):
        raise ValueError("not the pinned Gemma31 decoder")
    if getattr(config, "num_kv_shared_layers", 0) or getattr(config, "hidden_size_per_layer_input", 0):
        raise ValueError("KV-sharing/per-layer-input architectures are outside this pilot")
    if getattr(config, "enable_moe_block", False):
        raise ValueError("MoE is outside this pilot")
    if config._attn_implementation != "sdpa":
        raise ValueError("pilot requires the same SDPA backend for every arm")
    return text.layers


@contextmanager
def compression_context(layers, plan, *, cut, expand_before=None):
    """Restrict each native block's side inputs without rewriting its math."""
    import torch
    if not (0 <= cut < len(layers) - 1):
        raise ValueError("invalid compression cut")
    if expand_before is not None and not cut + 1 < expand_before < len(layers):
        raise ValueError("invalid expansion cut")
    audit: dict[str, Any] = {"sequence_lengths": {}, "compression_count": 0, "expansion_count": 0}
    handles = []
    state = {"compact": False}

    def shorten(module, args, output):
        if not torch.is_tensor(output) or output.ndim != 3 or output.shape[1] != plan.original_length:
            raise ValueError("unexpected raw decoder output geometry")
        audit["compression_count"] += 1
        state["compact"] = True
        return compact_states(output, plan)

    def before(index, module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if index == expand_before:
            h = expand_states(h, plan)
            if args:
                args = (h, *args[1:])
            else:
                kwargs = {**kwargs, "hidden_states": h}
            state["compact"] = False
            audit["expansion_count"] += 1
        expected = plan.compact_length if state["compact"] else plan.original_length
        if h.shape[1] != expected:
            raise ValueError(f"block {index} length {h.shape[1]} != {expected}")
        audit["sequence_lengths"][str(index)] = int(h.shape[1])
        if not state["compact"] or plan.compact_length == plan.original_length:
            return args, kwargs
        if kwargs.get("past_key_values") is not None or (len(args) > 1 and args[1] is not None) or kwargs.get("per_layer_input") is not None:
            raise ValueError("cached decoding/per-layer inputs unsupported")
        keep = torch.tensor(plan.keep, device=h.device)
        kwargs = dict(kwargs)
        pos = kwargs.get("position_ids")
        if pos is None or pos.shape[-1] != plan.original_length:
            raise ValueError("missing original position coordinates")
        kwargs["position_ids"] = pos.index_select(-1, keep)
        rope = kwargs["position_embeddings"]
        kwargs["position_embeddings"] = tuple(x.index_select(-2, keep) for x in rope)
        mask = kwargs.get("attention_mask")
        if mask is None:
            # keep is strictly increasing, so causality over retained original
            # positions is plain causality over compact rows. Leaving the mask
            # implicit keeps native SDPA's is_causal kernel for every arm. A
            # missing local multimodal mask is not safe to guess.
            if module.self_attn.layer_type != "full_attention":
                raise ValueError("missing explicit multimodal sliding mask")
            if any(b <= a for a, b in zip(plan.keep, plan.keep[1:])):
                raise ValueError("retained positions must be strictly increasing")
        else:
            if not torch.is_tensor(mask) or mask.ndim != 4 or mask.shape[-2:] != (plan.original_length, plan.original_length):
                raise ValueError("unexpected native attention-mask shape")
            kwargs["attention_mask"] = mask.index_select(-2, keep).index_select(-1, keep)
        return args, kwargs

    try:
        handles.append(layers[cut].register_forward_hook(shorten))
        for index, layer in enumerate(layers):
            def hook(module, args, kwargs, index=index):
                return before(index, module, args, kwargs)
            handles.append(layer.register_forward_pre_hook(hook, with_kwargs=True))
        yield audit
        if audit["compression_count"] != 1 or len(audit["sequence_lengths"]) != len(layers):
            raise ValueError("compression hooks did not execute exactly once")
        if audit["expansion_count"] != int(expand_before is not None):
            raise ValueError("expansion hook count mismatch")
    finally:
        for handle in handles:
            handle.remove()


def estimated_decoder_flops(config, sequence_lengths):
    """Dense matmul estimate only: excludes vision, norms, softmax, and overhead."""
    total = 0
    for index, n in enumerate(sequence_lengths):
        glob = config.layer_types[index] == "full_attention"
        d = config.hidden_size
        head = config.global_head_dim if glob else config.head_dim
        q = config.num_attention_heads * head
        kvheads = getattr(config, "num_global_key_value_heads", None) if glob else config.num_key_value_heads
        kv = (kvheads or config.num_key_value_heads) * head
        k_eq_v = bool(getattr(config, "attention_k_eq_v", False) and glob)
        projections = 2 * n * d * (2 * q + (1 if k_eq_v else 2) * kv)
        mlp = 6 * n * d * config.intermediate_size
        # SDPA may skip causal triangles; this is the dense upper-bound work.
        attention = 4 * config.num_attention_heads * n * n * head
        total += projections + mlp + attention
    return int(total)
