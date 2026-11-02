"""Bounded, frozen-base steering pilot; no model loading, data splitting, or CLI.

Cuts are zero-based *post-block* indices. Callbacks own preprocessing and A/B
scoring, with use_cache=False. They must not supply oracle/donor information,
mutate the base (including through .data), or perform their own optimizer steps.
The driver never receives validation/test data and never selects a checkpoint.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Callable, ContextManager, Iterator, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpointing


@dataclass(frozen=True)
class SteeringTarget:
    name: str
    layer_index: int
    positions: str


TARGETS = {
    "l34_cq": SteeringTarget("l34_cq", 34, "C+Q"),
    "l22_cq": SteeringTarget("l22_cq", 22, "C+Q"),
    "l34_readout": SteeringTarget("l34_readout", 34, "readout"),
    "l34_f": SteeringTarget("l34_f", 34, "F"),
}
PILOT_SEEDS = (17, 29)


class LowRankSteering(nn.Module):
    """Bias-free rank-r correction with exactly 2 * hidden_size * rank weights.

    forward returns the corrected activation, not just a delta. Parameters and
    arithmetic remain FP32 even under autocast; only the delta is cast to the
    activation dtype. Move with .to(device), not .half()/.bfloat16(). The smooth
    RMS bound has nonzero derivative at zero (unlike normalizing a zero delta).
    Before runtime rounding, delta RMS is at most 10% of input RMS.
    """

    def __init__(self, hidden_size: int, rank: int = 4):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (hidden_size, rank)):
            raise ValueError("hidden_size and rank must be positive integers")
        self.hidden_size = hidden_size
        self.rank = rank
        self.down = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: Tensor) -> Tensor:
        if not hidden.is_floating_point() or hidden.shape[-1] != self.hidden_size:
            raise ValueError("expected floating hidden states of the configured width")
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError("steering parameters must remain float32")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            value = hidden.float()
            scale = value.square().mean(dim=-1, keepdim=True).sqrt()
            normalized = value / scale.clamp_min(1e-6)
            raw = self.up(self.down(normalized))
            delta = 0.1 * scale * raw / (1 + raw.square().mean(-1, keepdim=True)).sqrt()
        return hidden + delta.to(hidden.dtype)


@contextmanager
def _frozen_eval(model: nn.Module) -> Iterator[None]:
    parameters = tuple(model.parameters())
    flags = tuple(p.requires_grad for p in parameters)
    modes = tuple((module, module.training) for module in model.modules())
    if any(p.grad is not None for p in parameters):
        raise ValueError("base parameters must have no pre-existing gradients")
    try:
        for p in parameters:
            p.requires_grad_(False)
        model.eval()
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


@dataclass
class SteeringReceipt:
    layer_index: int
    selected_positions: tuple[int, ...]
    applications: int = 0
    checkpointed_blocks: int = 0


def _validate_architecture(model: nn.Module, layers: Sequence[nn.Module]) -> None:
    config = getattr(model, "config", None)
    config = getattr(config, "text_config", config)
    if config is not None:
        for name in ("num_kv_shared_layers", "hidden_size_per_layer_input"):
            if getattr(config, name, 0) != 0:
                raise ValueError(f"unsupported hidden side channel: {name}")
        if getattr(config, "num_hidden_layers", len(layers)) != len(layers):
            raise ValueError("layers must contain the complete decoder")
    if any(getattr(m, "gradient_checkpointing", False) for m in model.modules()):
        raise ValueError("disable native gradient checkpointing before steering")
    if any(getattr(m, "is_kv_shared_layer", False) for layer in layers for m in layer.modules()):
        raise ValueError("unsupported hidden side channel: KV-consuming layer")


@contextmanager
def steering_context(
    model: nn.Module,
    layers: Sequence[nn.Module],
    layer_index: int,
    positions: Sequence[int],
    adapter: LowRankSteering,
    training: bool = False,
) -> Iterator[SteeringReceipt]:
    """One batch-one forward per context, with exact instance-forward restoration.

    Tensor or tuple/list-with-hidden-first block outputs are supported. Untargeted
    values are copied unchanged. Prefix blocks execute under no_grad; the entire
    cut activation is detached before intervention. Only suffix block forwards
    are nonreentrantly checkpointed, so backward never replays the intervention.
    Gemma's unused last-local/global KV exports are disabled temporarily: even
    with zero KV consumers, 5.13 otherwise retains them in mutable replay kwargs.
    Keep the context active through backward so recomputation sees frozen eval
    blocks. Eval/dropout and requires_grad flags are restored on exit. Not
    thread-safe; concurrent use of a model is out
    of scope. Training must not run under torch.inference_mode().
    """
    layers = tuple(layers)
    positions = tuple(positions)
    if type(layer_index) is not int or not 0 <= layer_index < len(layers):
        raise ValueError("invalid zero-based layer_index")
    if (not positions or any(type(p) is not int or p < 0 for p in positions)
            or len(set(positions)) != len(positions)):
        raise ValueError("positions must be nonempty, unique, nonnegative integers")
    if len({id(layer) for layer in layers}) != len(layers):
        raise ValueError("shared decoder block objects are unsupported")
    modules = {id(module) for module in model.modules()}
    if any(id(layer) not in modules for layer in layers):
        raise ValueError("layers must belong to model")
    if any(id(p) in {id(q) for q in model.parameters()} for p in adapter.parameters()):
        raise ValueError("adapter must be separate from the frozen model")
    if getattr(model, "_pilot_steering_active", False):
        raise ValueError("nested steering contexts are unsupported")
    if training and torch.is_inference_mode_enabled():
        raise ValueError("training cannot run in inference_mode")
    _validate_architecture(model, layers)
    receipt = SteeringReceipt(layer_index, positions)
    originals = [(layer, "forward" in layer.__dict__, layer.__dict__.get("forward"),
                  layer.forward) for layer in layers]
    kv_exports = [(module, module.store_full_length_kv) for layer in layers
                  for module in layer.modules() if hasattr(module, "store_full_length_kv")]
    active_existed = "_pilot_steering_active" in model.__dict__
    active_before = model.__dict__.get("_pilot_steering_active")

    def wrap(index: int, original: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("return_shared_kv_states", False):
                raise ValueError("shared KV export is unsupported during steering")
            if kwargs.get("use_cache", False) or any(
                kwargs.get(name) is not None for name in ("past_key_value", "past_key_values")
            ):
                raise ValueError("steering requires use_cache=False and no KV cache")
            if index > layer_index:
                if training:
                    receipt.checkpointed_blocks += 1
                    # Capture the bound forward, not layer.__call__ or a live hook.
                    with torch.enable_grad():
                        return checkpointing.checkpoint(
                            original, *args, use_reentrant=False, **kwargs
                        )
                return original(*args, **kwargs)
            with torch.no_grad():
                output = original(*args, **kwargs)
            if index != layer_index:
                return output
            if receipt.applications:
                raise RuntimeError("selected cut executed more than once")
            hidden = output if isinstance(output, Tensor) else output[0]
            if (not isinstance(hidden, Tensor) or hidden.ndim != 3
                    or hidden.shape[0] != 1 or max(positions) >= hidden.shape[1]):
                raise ValueError("expected batch-one [1, sequence, hidden] at the cut")
            hidden = hidden.detach()
            selected = torch.tensor(positions, device=hidden.device, dtype=torch.long)
            with torch.set_grad_enabled(training):
                corrected = adapter(hidden.index_select(1, selected))
                steered = hidden.index_copy(1, selected, corrected)
            receipt.applications += 1
            if isinstance(output, Tensor):
                return steered
            if isinstance(output, tuple):
                return (steered, *output[1:])
            if isinstance(output, list):
                return [steered, *output[1:]]
            raise TypeError("unsupported decoder output type")
        return run

    try:
        with _frozen_eval(model):
            model._pilot_steering_active = True
            for module, _ in kv_exports:
                module.store_full_length_kv = False
            for index, (layer, _, _, original) in enumerate(originals):
                layer.forward = wrap(index, original)
            yield receipt
            if receipt.applications != 1:
                raise RuntimeError("selected cut must execute exactly once")
    finally:
        for module, value in kv_exports:
            module.store_full_length_kv = value
        for layer, existed, value, _ in originals:
            if existed:
                layer.forward = value
            else:
                layer.__dict__.pop("forward", None)
        if active_existed:
            model._pilot_steering_active = active_before
        else:
            model.__dict__.pop("_pilot_steering_active", None)


@dataclass(frozen=True)
class TrainingRow:
    """row is label-free method input, passed verbatim to prepare(row).

    The caller owns split/identity validation and must exclude oracle fields
    from row. Keeping the supervised label separate prevents this driver from
    passing it to either method callback; arbitrary callbacks cannot be audited
    for hidden side channels by this module.
    """

    row_id: str
    row: Any
    label: int
    split: str


@dataclass(frozen=True)
class StepReceipt:
    step: int
    examples_seen: int
    loss: float
    gradient_norm: float
    update_norm: float
    finite_gradients: bool
    nonzero_gradient: bool
    nonzero_update: bool
    frozen_base_checked: bool


@dataclass(frozen=True)
class TrainingResult:
    adapter: LowRankSteering
    target: SteeringTarget
    checkpoint: Path
    seed: int
    parameter_count: int
    data_order: tuple[str, ...]
    receipts: tuple[StepReceipt, ...]
    final_update_norm: float
    optimizer_steps: int
    canary: bool = False
    accumulation_steps: int = 4
    microbatch_size: int = 1
    learning_rate: float = 1e-3
    clip_norm: float = 1.0
    weight_decay: float = 0.01
    checkpoint_selection: str = "fixed_final_step"
    test_rows_seen: int = 0
    labels_passed_to_callbacks: bool = False
    oracle_policy: str = "label-free inputs; caller must audit opaque callbacks"
    frozen_check: str = "parameter identity/version/requires_grad/grad; excludes .data writes"


def _base_snapshot(model: nn.Module) -> tuple[tuple[str, nn.Parameter, int], ...]:
    return tuple((name, p, p._version) for name, p in model.named_parameters())


def _check_base(model: nn.Module, snapshot: tuple[tuple[str, nn.Parameter, int], ...]) -> None:
    current = tuple(model.named_parameters())
    if len(current) != len(snapshot) or any(
        name != old_name or p is not old or p._version != version
        or p.requires_grad or p.grad is not None
        for (name, p), (old_name, old, version) in zip(current, snapshot)
    ):
        raise RuntimeError("frozen base parameter integrity check failed")
    if any(module.training for module in model.modules()):
        raise RuntimeError("base must remain in eval mode (dropout disabled)")


def _difference_norm(parameters: Sequence[Tensor], before: Sequence[Tensor]) -> float:
    return math.sqrt(sum(float((p.detach() - old).square().sum())
                         for p, old in zip(parameters, before)))


def train_adapter(
    model: nn.Module,
    rows: Sequence[TrainingRow],
    *,
    hidden_size: int,
    target: str,
    prepare: Callable[[Any], Any],
    forward: Callable[[Any, LowRankSteering, SteeringTarget, bool], Tensor | ContextManager[Tensor]],
    checkpoint_dir: str | Path,
    rank: int = 4,
    seed: int = 17,
    optimizer_steps: int = 256,
    canary: bool = False,
    device: torch.device | str | None = None,
    early_report: Callable[[StepReceipt], None] | None = None,
    report_every: int = 10,
) -> TrainingResult:
    """Fit one target with matched initialization/order and a fixed final save.

    Defaults: 256 AdamW steps (1,024 examples), accumulation 4, microbatch 1, lr=1e-3,
    weight_decay=.01, clip=1.0, rank=4, seeds 17/29. Smaller positive budgets
    (<=256) require canary=True; production runs always use exactly 256 steps.
    Sampling exhausts each shuffled pass before reshuffling, so a 1,010-row
    training pool is covered once plus 14 examples from the next pass.
    prepare receives TrainingRow.row only; forward receives prepared input,
    adapter, TARGETS[target], training=True and returns a context manager yielding
    differentiable A/B logits of shape [2] or [1,2]. The driver calls backward
    INSIDE that context: wrap steering_context around the yield. Plain tensors
    are also accepted for callbacks without a managed forward lifetime.
    No validation/test callback exists. early_report runs after step 1, every
    report_every steps, and the final step; its return value is ignored.
    Reports are immutable and contain no model or labels. Exceptions abort with
    no checkpoint. Use the same rows/seed/budget/rank for all four targets.

    device is the cut device (explicit for sharded models), otherwise the first
    base parameter's device. Python/Torch RNG states are restored on exit;
    preprocessing must be deterministic or use these RNGs, not an unseeded NumPy
    generator. Thread counts are owned by the runner, not changed globally here.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {tuple(TARGETS)}")
    if torch.is_inference_mode_enabled():
        raise ValueError("training cannot run in inference_mode")
    if type(canary) is not bool:
        raise ValueError("canary must be a boolean")
    if type(optimizer_steps) is not int or not 1 <= optimizer_steps <= 256:
        raise ValueError("optimizer_steps must be in [1,256]")
    if type(rank) is not int or rank != 4:
        raise ValueError("the matched pilot fixes rank=4")
    if type(seed) is not int or seed not in PILOT_SEEDS:
        raise ValueError("the matched pilot fixes seeds to 17 and 29")
    if not canary and optimizer_steps != 256:
        raise ValueError("production fixes 256 steps; shorter runs require canary=True")
    if type(report_every) is not int or report_every <= 0:
        raise ValueError("report_every must be positive")
    rows = tuple(rows)
    if not rows or any(row.split != "train" for row in rows):
        raise ValueError("only explicitly train-split rows are accepted")
    if any(type(row.label) is not int or row.label not in (0, 1) for row in rows):
        raise ValueError("training labels must be integer A/B indices 0 or 1")
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError("row_id must uniquely identify each training example/order")
    try:
        from safetensors.torch import save_file
    except ImportError:
        save_file = None
    directory = Path(checkpoint_dir)
    stem = f"{target}.seed{seed}{'.canary' if canary else ''}.final"
    if any((directory / f"{stem}{suffix}").exists() for suffix in (".pt", ".safetensors")):
        raise FileExistsError("refusing to overwrite a previous final checkpoint")
    path = directory / f"{stem}{'.safetensors' if save_file else '.pt'}"
    if device is None:
        device = next(model.parameters(), torch.empty(0)).device
    device = torch.device(device)
    rng_state = random.getstate()
    order_rng = random.Random(seed)
    devices = {p.device.index for p in model.parameters() if p.device.type == "cuda"}
    if device.type == "cuda":
        devices.add(device.index if device.index is not None else torch.cuda.current_device())
    devices = sorted(devices)
    receipts: list[StepReceipt] = []
    data_order: list[str] = []
    try:
        with torch.random.fork_rng(devices=devices), _frozen_eval(model):
            random.seed(seed)
            # Do not initialize other workers' GPU contexts just to preserve RNGs.
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                with torch.cuda.device(index):
                    torch.cuda.manual_seed(seed)
            adapter = LowRankSteering(hidden_size, rank).to(device=device)
            parameters = tuple(adapter.parameters())
            initial = tuple(p.detach().clone() for p in parameters)
            optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0.01)
            snapshot = _base_snapshot(model)
            order: list[int] = []
            for step in range(1, optimizer_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                before = tuple(p.detach().clone() for p in parameters)
                loss_total = 0.0
                for _ in range(4):
                    if not order:
                        order = list(range(len(rows)))
                        order_rng.shuffle(order)
                    row = rows[order.pop()]
                    data_order.append(row.row_id)
                    prepared = prepare(row.row)
                    with torch.enable_grad():
                        result = forward(prepared, adapter, TARGETS[target], True)
                        with (nullcontext(result) if isinstance(result, Tensor) else result) as logits:
                            if (not isinstance(logits, Tensor) or logits.shape not in ((2,), (1, 2))
                                    or not logits.requires_grad or not torch.isfinite(logits).all()):
                                raise ValueError("forward must return finite differentiable A/B logits")
                            loss = F.cross_entropy(logits.float().reshape(1, 2),
                                                   torch.tensor([row.label], device=logits.device))
                            if not torch.isfinite(loss):
                                raise FloatingPointError("nonfinite training loss")
                            (loss / 4).backward()
                            loss_total += float(loss.detach()) / 4
                    _check_base(model, snapshot)
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError("missing or nonfinite adapter gradients")
                gradient_norm = float(nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in parameters):
                    raise FloatingPointError("nonfinite adapter parameters")
                update_norm = _difference_norm(parameters, before)
                _check_base(model, snapshot)
                receipt = StepReceipt(step, step * 4, loss_total, gradient_norm, update_norm,
                                      True, gradient_norm > 0, update_norm > 0, True)
                receipts.append(receipt)
                if early_report and (step == 1 or step % report_every == 0 or step == optimizer_steps):
                    early_report(receipt)
            _check_base(model, snapshot)
            final_update = _difference_norm(parameters, initial)
            if not any(r.nonzero_gradient and r.nonzero_update for r in receipts) or final_update == 0:
                raise RuntimeError("adapter did not learn a nonzero correction")
            optimizer.zero_grad(set_to_none=True)
            state = {name: value.detach().cpu().contiguous() for name, value in adapter.state_dict().items()}
            directory.mkdir(parents=True, exist_ok=True)
            if save_file:
                save_file(state, str(path), metadata={"target": target, "selection": "fixed_final_step",
                                                    "steps": str(optimizer_steps), "seed": str(seed),
                                                    "rank": str(rank), "hidden_size": str(hidden_size),
                                                    "canary": str(canary)})
            else:
                # Own-origin direct state only; load with torch.load(..., weights_only=True).
                with path.open("xb") as stream:
                    torch.save(state, stream)
    finally:
        random.setstate(rng_state)
    return TrainingResult(adapter, TARGETS[target], path, seed,
                          sum(p.numel() for p in parameters), tuple(data_order),
                          tuple(receipts), final_update, optimizer_steps, canary)
