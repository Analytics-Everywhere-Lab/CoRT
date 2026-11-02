"""Independent requests using geometry-only plans, with an explicit cache lifecycle.

Create one cache per engine/configuration. Call ``benchmark_measurement`` with
``phase="warmup"`` for every geometry/arm, then ``cache.seal()`` before timed
calls. Finish with ``cache.close()`` (or use the cache as a context manager).
Warmup returns first-use cold timings separately; ``cache.report()`` contains
detached build/cold timing records, never request inputs or model outputs.

Schedules are logically immutable: the engine and callbacks must not modify
them. Full content and receipt checks at seal/teardown detect mutation without
rehashing plans on the hot path. Only structural, JSON-compatible dataclass
schedules are accepted; tensors, prepared requests, and opaque objects are not.
This is a single-threaded helper, not a concurrent cache or a compiler for model
graphs. Schedule construction is permitted only during explicit warmup.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import json
import time


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _structure(value):
    """Inspect actual plan fields as well as receipts, rejecting opaque payloads."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _structure(getattr(value, field.name)) for field in fields(value)}
    if type(value) in (dict,):
        if any(type(key) is not str for key in value):
            raise TypeError("schedule metadata keys must be strings")
        return {key: _structure(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_structure(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError("cached schedules must contain structural metadata only")


@dataclass(frozen=True)
class CachedPlan:
    """A borrowed immutable plan; no prepared input is retained by this handle."""

    plan: object
    geometry_sha256: str
    schedule_sha256: str
    cache_hit: bool
    plan_construction_ms: float


class CachedScheduleCache:
    """Bounded geometry cache with no eviction or timed-phase cache misses.

    ``checksum_fn`` defaults to the runner's checksum, and ``receipt_fn`` to
    its plan_receipt. They are injectable for CPU tests. All geometry fields
    are hashed, and canonical content is also compared to detect colliding keys.
    Arm definitions are snapshotted at construction and checked on every lookup.
    """

    def __init__(self, arms, *, max_geometries, clock=time.perf_counter,
                 checksum_fn=None, receipt_fn=None):
        if type(max_geometries) is not int or max_geometries < 1:
            raise ValueError("max_geometries must be a positive integer")
        if checksum_fn is None or receipt_fn is None:
            from research.lifecycle_memory.run_decoder_token_matrix import checksum, plan_receipt
            checksum_fn = checksum if checksum_fn is None else checksum_fn
            receipt_fn = plan_receipt if receipt_fn is None else receipt_fn
        self._checksum = checksum_fn
        self._receipt = receipt_fn
        self._clock = clock
        self._max_geometries = max_geometries
        self._arms = {}
        for arm in arms:
            arm_id = arm["id"]
            if not isinstance(arm_id, str) or arm_id in self._arms:
                raise ValueError("arm IDs must be unique strings")
            self._arms[arm_id] = _canonical(arm)
        if not self._arms:
            raise ValueError("at least one arm is required")
        self._entries = {}
        self._warmed = set()
        self._cold = {}
        self._state = "warmup"

    def _snapshot(self, plan):
        content = _canonical(_structure(plan))
        receipt = self._receipt(plan)
        return content, _canonical(receipt), self._checksum(receipt)

    def get(self, engine, prepared, arm, *, warmup=False):
        """Compute geometry and retrieve one plan; build only in warmup state.

        Geometry computation, checksum verification, and content comparison are
        deliberately part of every lookup, including native and timed hits.
        """
        if self._state != ("warmup" if warmup else "sealed"):
            raise RuntimeError("cache must be warming for builds or sealed for timed lookup")
        arm_id = arm["id"]
        if self._arms.get(arm_id) != _canonical(arm):
            raise ValueError("unknown or changed arm definition")
        key, geometry = engine.geometry(prepared)
        canonical = _canonical(geometry)
        if key != self._checksum(geometry):
            raise ValueError("geometry checksum is invalid")
        entry = self._entries.get(key)
        if entry is not None and entry["geometry"] != canonical:
            raise RuntimeError("geometry checksum collision")
        if entry is None:
            if not warmup:
                raise RuntimeError("unseen geometry in sealed cache")
            if len(self._entries) >= self._max_geometries:
                raise RuntimeError("geometry cache capacity exceeded")
            entry = {"geometry": canonical, "plans": {}}
        cached = entry["plans"].get(arm_id)
        hit = cached is not None
        build_ms = 0.0
        if not hit:
            if not warmup:
                raise RuntimeError("unseen arm in sealed cache")
            started = self._clock()
            # Detached metadata prevents caller-owned arm/geometry dictionaries
            # from becoming aliases into a cached schedule.
            plan = engine.schedule(prepared, json.loads(self._arms[arm_id]),
                                   json.loads(canonical))
            content, receipt, digest = self._snapshot(plan)
            build_ms = 1000 * (self._clock() - started)
            cached = {"plan": plan, "content": content, "receipt": receipt,
                      "digest": digest, "build_ms": build_ms}
            entry["plans"][arm_id] = cached
            self._entries[key] = entry
        return CachedPlan(cached["plan"], key, cached["digest"], hit, build_ms)

    def _record_warmup(self, handle, arm_id, timings):
        pair = (handle.geometry_sha256, arm_id)
        first = pair not in self._warmed
        if first:
            # Only scalar timing metadata is retained, never callback outputs.
            self._cold[pair] = dict(timings)
        self._warmed.add(pair)
        return first

    def verify_integrity(self):
        """Expensive audit outside request timing, also available before sealing."""
        for entry in self._entries.values():
            for cached in entry["plans"].values():
                if self._snapshot(cached["plan"]) != (
                        cached["content"], cached["receipt"], cached["digest"]):
                    raise RuntimeError("cached schedule mutation detected")

    def seal(self):
        """Require successful forward warmups for every declared arm/geometry."""
        if self._state != "warmup" or not self._entries:
            raise RuntimeError("only a nonempty warming cache can be sealed")
        self.verify_integrity()
        required = {(key, arm_id) for key in self._entries for arm_id in self._arms}
        if not required <= self._warmed:
            raise RuntimeError("all geometry/arm pairs must complete forward warmup")
        self._state = "sealed"

    def report(self):
        """Detached, JSON-compatible one-time build and first-use measurements."""
        return {"state": self._state, "geometry_count": len(self._entries),
                "max_geometries": self._max_geometries,
                "geometries": {key: json.loads(entry["geometry"])
                               for key, entry in self._entries.items()},
                "plans": [{"geometry_sha256": key, "arm_id": arm_id,
                           "schedule_sha256": cached["digest"],
                           "receipt": json.loads(cached["receipt"]),
                           "plan_construction_ms": cached["build_ms"],
                           "cold_timing": dict(self._cold.get((key, arm_id), {}))}
                          for key, entry in self._entries.items()
                          for arm_id, cached in entry["plans"].items()]}

    def close(self):
        """Audit teardown, closing even if a corrupt plan is discovered."""
        try:
            self.verify_integrity()
        finally:
            self._state = "closed"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def benchmark_measurement(engine, row, data_dir, arm, cache, *, phase="timed",
                          forward=None, postprocess=None, clock=time.perf_counter,
                          measure=None, synchronize=None):
    """Return ``(result, metrics)`` for an independently prepared request.

    ``forward(engine, prepared, schedule)`` defaults to matrix_forward; a
    correction runner can supply its own forward using that exact cached plan.
    ``postprocess(result, prepared, schedule)`` runs inside full-wall timing.
    Supply the same callbacks in warmup and timed phases. ``measure(torch, fn)``
    follows pilot.measure's return-value-plus-metrics contract. No torch import
    occurs here; CPU tests inject measure, synchronize, and a seconds clock.

    Model wall time encloses the measurement wrapper and final synchronization;
    pilot's own elapsed/CUDA/memory metrics remain in ``forward_metrics``.
    Lookup wall time includes one-time construction on warmup misses, so
    plan_construction_ms is a *subset*, not an extra additive timing component.
    """
    if phase not in ("warmup", "timed"):
        raise ValueError("phase must be warmup or timed")
    expected_state = "warmup" if phase == "warmup" else "sealed"
    if cache._state != expected_state:
        raise RuntimeError("benchmark phase does not match cache lifecycle")
    if measure is None:
        from research.lifecycle_memory.run_decoder_application_pilots import measure
    if synchronize is None:
        synchronize = engine.torch.cuda.synchronize
    if forward is None:
        forward = lambda engine, prepared, schedule: engine.matrix_forward(prepared, schedule)

    synchronize()
    start = clock()
    prepared = engine.prepare(row, data_dir)
    synchronize()
    prepared_at = clock()
    handle = cache.get(engine, prepared, arm, warmup=phase == "warmup")
    synchronize()
    looked_up_at = clock()
    result, forward_metrics = measure(
        getattr(engine, "torch", None), lambda: forward(engine, prepared, handle.plan))
    synchronize()
    forwarded_at = clock()
    if postprocess is not None:
        result = postprocess(result, prepared, handle.plan)
    synchronize()
    finished = clock()
    timings = {"preparation_ms": 1000 * (prepared_at - start),
               "plan_lookup_ms": 1000 * (looked_up_at - prepared_at),
               "model_forward_ms": 1000 * (forwarded_at - looked_up_at),
               "postprocess_ms": 1000 * (finished - forwarded_at),
               "end_to_end_elapsed_ms": 1000 * (finished - start)}
    first_use = (cache._record_warmup(handle, arm["id"], timings)
                 if phase == "warmup" else False)
    return result, {**timings, "forward_metrics": dict(forward_metrics),
                    "phase": phase, "first_use": first_use,
                    "timing_classification": ("cold_first_use" if first_use else
                                              "warmup" if phase == "warmup" else "warm_cached"),
                    "first_use_cold_ms": timings["end_to_end_elapsed_ms"] if first_use else None,
                    "plan_cache_hit": handle.cache_hit,
                    "plan_construction_ms": handle.plan_construction_ms,
                    "geometry_sha256": handle.geometry_sha256,
                    "schedule_sha256": handle.schedule_sha256,
                    "independent_preparation": True,
                    "timing_scope": "synchronized independent prepare + cached lookup + forward + postprocess"}
