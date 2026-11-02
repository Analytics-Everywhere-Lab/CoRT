"""Evaluator-only temporal study analysis; never import this into method workers.

Rows contain natural A/B log probabilities and raw A-minus-B margins in bits.
Labels are read, never modified. Confirmation uses protocol['cuts'] (also
accepting protocol['freeze']['cuts']);
it never selects endpoints. Optional episode_ids / patch_episode_ids in the
protocol (or freeze) specify outcome-blind inventories, otherwise labels define
both inventories.
Both orders and both models are required in each primary inventory. A completely
absent patch inventory fails, rather than silently selecting complete cases.
"""

from collections import defaultdict
from itertools import combinations
import math


GEMMA = "gemma4_31b_hf_bf16"
QWEN = "qwen35_27b_hf_bf16"
LAYERS = {GEMMA: 60, QWEN: 64}
MODEL_ALIASES = {"gemma31": GEMMA, "qwen27": QWEN}
CONDITIONS = tuple(f"B{i}" for i in range(10))
SUBSETS = tuple("".join(s) for n in range(1, 5) for s in combinations("CQPR", n))


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values):
    return math.fsum(values) / len(values)


def _cluster_values(records):
    """records = (cluster, episode, order, scalar); orders then episodes equally."""
    episodes = defaultdict(lambda: defaultdict(list))
    for cluster, episode, order, value in records:
        episodes[(cluster, episode)][order].append(value)
    clusters = defaultdict(list)
    for (cluster, _), orders in episodes.items():
        clusters[cluster].append(_mean([_mean(v) for v in orders.values()]))
    return {c: _mean(v) for c, v in sorted(clusters.items())}


def _bootstrap(columns, draws, seed):
    """Paired resampling, bounded index/sample temporaries, shared cluster order."""
    import numpy as np

    x = np.asarray(columns, dtype=float).T
    n, k = x.shape
    means = np.empty((draws, k))
    errors = np.empty((draws, k))
    rng = np.random.default_rng(seed)
    batch = max(1, min(256, 1_000_000 // max(1, n * k)))
    for start in range(0, draws, batch):
        stop = min(draws, start + batch)
        sample = x[rng.integers(0, n, size=(stop - start, n))]
        means[start:stop] = sample.mean(axis=1)
        errors[start:stop] = sample.std(axis=1, ddof=1) / math.sqrt(n) if n > 1 else 0
    return means, errors


def _interval(values):
    import numpy as np

    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def _summary(values, draws, seed, test=False):
    """Percentile mean CI descriptively; bootstrap-t CI for primary contrasts."""
    import numpy as np

    x = list(values.values())
    if not x:
        return {"estimate": None, "ci95": None, "p_value": None, "valid": False,
                "reason": "no_clusters", "n_clusters": 0}
    theta = _mean(x)
    result = {"estimate": theta, "n_clusters": len(x), "wearer_values": values,
              "ci95": None, "p_value": None, "valid": False}
    if len(x) < 2:
        result["reason"] = "fewer_than_two_clusters"
        return result
    means, errors = _bootstrap([x], draws, seed)
    result.update(ci95=_interval(means[:, 0]), valid=True, ci_method="cluster_percentile")
    result["leave_one_wearer_out_range"] = [min((_mean(x[:i] + x[i + 1:]) for i in range(len(x)))),
                                            max((_mean(x[:i] + x[i + 1:]) for i in range(len(x))))]
    if not test:
        return result
    se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    invalid = int(np.count_nonzero((errors[:, 0] <= 0) | ~np.isfinite(errors[:, 0])))
    result.update(standard_error=se, invalid_bootstrap_draws=invalid,
                  ci95=None, ci_method="null_centered_cluster_bootstrap_t")
    if se <= 0 or not math.isfinite(se) or invalid:
        result.update(valid=False, reason="zero_or_invalid_standard_error")
        return result
    t_star = (means[:, 0] - theta) / errors[:, 0]
    quantiles = _interval(t_star)
    result.update(p_value=float((1 + np.count_nonzero(t_star >= theta / se)) / (draws + 1)),
                  ci95=[theta - quantiles[1] * se, theta - quantiles[0] * se])
    return result


def _ratio(numerator, denominator, draws, seed, guard):
    """Ratio of wearer means, with paired denominator resampling and no dropping."""
    import numpy as np

    keys = sorted(numerator)
    n = [numerator[k] for k in keys]
    d = [denominator[k] for k in keys]
    result = {"estimate": None, "ci95": None, "stable": False, "guard_bits": guard,
              "denominator": _mean(d) if d else None}
    if len(keys) < 2:
        result["reason"] = "fewer_than_two_clusters"
        return result
    means, _ = _bootstrap([n, d], draws, seed)
    lower = _interval(means[:, 1])[0]
    bad = float(np.mean(means[:, 1] <= guard))
    result.update(denominator_ci95=_interval(means[:, 1]), unstable_draw_fraction=bad)
    if _mean(d) <= guard or lower <= guard or bad > .01 or np.any(means[:, 1] <= 0):
        result["reason"] = "unstable_denominator"
        return result
    result.update(estimate=_mean(n) / _mean(d), ci95=_interval(means[:, 0] / means[:, 1]), stable=True)
    return result


def _patch_values(row):
    donor = row["donor_margin_ab_bits"]
    base = abs(row["recipient_margin_ab_bits"] - donor)
    return base - abs(row["margin_ab_bits"] - donor), base


def _direction(row):
    return f"{row.get('donor_condition')}->{row.get('recipient_condition')}"


def select_cuts(rows):
    """Exploration-only API: maximize wearer-equal forward raw T, ties lower.

    Caller must supply exploration rows. Explicit non-exploration phase tags are
    rejected, and every eligible cut/set must cover the same episodes/orders.
    """
    if any(r.get("phase", "exploration") != "exploration" for r in rows):
        raise ValueError("cut selection requires exploration rows")
    rows = [dict(r, model_key=MODEL_ALIASES.get(r.get("model_key"), r.get("model_key"))) for r in rows]
    result = {}
    for model, layers in LAYERS.items():
        patches = [r for r in rows if r.get("model_key") == model and r.get("kind") == "patch"
                   and _direction(r) == "B0->B4"]
        if not patches:
            continue
        universe = {(r["cluster_id"], r["episode_id"], o) for r in patches for o in (0, 1)}
        chosen = {}
        for name, positions, low, high in (("early_cut", "H", .1, .45),
                                           ("carrier_cut", "V", .1, .8),
                                           ("late_cut", "K", .5, .8)):
            groups = defaultdict(list)
            for row in patches:
                if row.get("position_set") == positions and low * layers <= row["cut"] <= high * layers:
                    groups[row["cut"]].append(row)
            scores = []
            for cut, group in groups.items():
                cells = [(r["cluster_id"], r["episode_id"], r["order"]) for r in group]
                if len(cells) != len(set(cells)) or set(cells) != universe or any(r.get("status") != "pass" for r in group):
                    raise ValueError(f"incomplete exploration cells: {model}/{positions}/{cut}")
                values = _cluster_values([(*cell, _patch_values(r)[0]) for cell, r in zip(cells, group)])
                scores.append((_mean(list(values.values())), cut))
            if not scores:
                raise ValueError(f"no eligible exploration cuts: {model}/{positions}")
            chosen[name] = max(scores, key=lambda item: (item[0], -item[1]))[1]
        result[model] = chosen
    return result


def _holm(claims, alpha=.05):
    """Missing/invalid claims remain in the fixed four-member family at p=1."""
    ordered = sorted(claims, key=lambda c: (c["p_value"] if c["p_value"] is not None else 1, c["id"]))
    running = 0.0
    stopped = False
    for rank, claim in enumerate(ordered):
        p = claim["p_value"] if claim["p_value"] is not None else 1.0
        running = max(running, min(1.0, (len(claims) - rank) * p))
        threshold = alpha / (len(claims) - rank)
        reject = not stopped and claim["valid"] and claim["positive"] and p <= threshold
        stopped = stopped or not reject
        claim.update(holm_adjusted_p=running, holm_threshold=threshold, reject=reject)


def _claim(model, name, components):
    valid = bool(components) and all(c["valid"] for c in components.values())
    return {"id": f"{model}:{name}", "model_key": model, "claim": name,
            "components": components, "valid": valid,
            "positive": bool(components) and all(c["estimate"] is not None and c["estimate"] > 0 for c in components.values()),
            "p_value": max(c["p_value"] for c in components.values()) if valid else None}


def _validate(rows, labels):
    label_map = {}
    for label in labels:
        episode = label["episode_id"]
        if episode in label_map or label.get("correct_candidate") not in (0, 1):
            raise ValueError("duplicate episode label or invalid correct_candidate")
        ttc = label.get("time_to_contact_s")
        if ttc is not None and (not _number(ttc) or ttc < 0):
            raise ValueError("invalid time_to_contact_s")
        label_map[episode] = label
    tasks, cells, clusters = set(), set(), {}
    for r in rows:
        task = r["task_id"]
        if task in tasks:
            raise ValueError(f"duplicate task_id: {task}")
        tasks.add(task)
        if r.get("model_key") not in LAYERS or r.get("kind") not in ("behavior", "patch", "self", "prune"):
            raise ValueError("unknown model or row kind")
        if r.get("order") not in (0, 1) or r.get("episode_id") not in label_map:
            raise ValueError("invalid order or missing evaluator label")
        if not isinstance(r.get("cluster_id"), str) or not r["cluster_id"]:
            raise ValueError("cluster_id must identify a wearer")
        episode = r["episode_id"]
        if episode in clusters and clusters[episode] != r["cluster_id"]:
            raise ValueError("episode assigned to multiple wearers")
        clusters[episode] = r["cluster_id"]
        cell = tuple(r.get(k) for k in ("kind", "model_key", "episode_id", "order",
                                        "donor_condition", "recipient_condition", "cut", "position_set"))
        cell += (r.get("condition") if r["kind"] in ("behavior", "prune") else None,)
        if cell in cells:
            raise ValueError("duplicate logical cell")
        cells.add(cell)
        if r.get("status") != "pass":
            continue
        fields = ("a_logprob", "b_logprob", "margin_ab_bits")
        if r["kind"] in ("patch", "self"):
            fields += ("donor_margin_ab_bits", "recipient_margin_ab_bits")
            if r.get("donor_condition") not in CONDITIONS or r.get("recipient_condition") not in CONDITIONS:
                raise ValueError("invalid patch conditions")
        elif r.get("condition") not in CONDITIONS:
            raise ValueError("invalid behavior/prune condition")
        if any(not _number(r.get(f)) for f in fields):
            raise ValueError("missing or nonfinite score")
        if r["a_logprob"] > 0 or r["b_logprob"] > 0:
            raise ValueError("log probabilities must be natural logs <= 0")
        if not math.isclose((r["a_logprob"] - r["b_logprob"]) / math.log(2), r["margin_ab_bits"], abs_tol=1e-5, rel_tol=1e-5):
            raise ValueError("A/B log probabilities disagree with bit margin")
        if r["kind"] != "behavior":
            if type(r.get("cut")) is not int or not 0 <= r["cut"] < LAYERS[r["model_key"]]:
                raise ValueError("invalid decoder cut")
    return label_map


def _behavior_metrics(row, label):
    a_is_gold = row["order"] == label["correct_candidate"]
    gold, other = (row["a_logprob"], row["b_logprob"]) if a_is_gold else (row["b_logprob"], row["a_logprob"])
    delta = other - gold
    loss = max(delta, 0) + math.log1p(math.exp(-abs(delta)))
    return {"loss_nats": loss, "accuracy": float(gold > other) if gold != other else .5,
            "gold_margin_bits": row["margin_ab_bits"] * (1 if a_is_gold else -1)}


def _records(rows, function):
    return _cluster_values([(r["cluster_id"], r["episode_id"], r["order"], function(r)) for r in rows])


def _factorial(group, expected, draws, seed, guard):
    index = {(r["episode_id"], r["order"], r["position_set"]): r for r in group}
    missing = [(e, o, s) for e in sorted(expected) for o in (0, 1) for s in SUBSETS
               if (e, o, s) not in index or index[(e, o, s)].get("status") != "pass"]
    if missing:
        return {"status": "incomplete", "missing_cells": missing, "shapley": None}
    values = {s: _records([index[(e, o, s)] for e in sorted(expected) for o in (0, 1)],
                          lambda r: _patch_values(r)[0]) for s in SUBSETS}
    baseline = _records([index[(e, o, "C")] for e in sorted(expected) for o in (0, 1)], lambda r: _patch_values(r)[1])
    for e in expected:
        for o in (0, 1):
            reference = index[(e, o, "C")]
            if any(any(not math.isclose(index[(e, o, s)][f], reference[f], abs_tol=1e-8)
                       for f in ("donor_margin_ab_bits", "recipient_margin_ab_bits")) for s in SUBSETS):
                return {"status": "inconsistent_baselines", "shapley": None}
    values[""] = {c: 0.0 for c in baseline}
    allocation = {}
    for player in "CQPR":
        phi = {c: 0.0 for c in baseline}
        others = [p for p in "CQPR" if p != player]
        for size in range(4):
            weight = math.factorial(size) * math.factorial(3 - size) / math.factorial(4)
            for subset in combinations(others, size):
                before = "".join(p for p in "CQPR" if p in subset)
                after = "".join(p for p in "CQPR" if p in subset or p == player)
                for c in phi:
                    phi[c] += weight * (values[after][c] - values[before][c])
        allocation[player] = phi
    cq_sum = {c: allocation["C"][c] + allocation["Q"][c] for c in baseline}
    interaction = {c: values["CQ"][c] - values["C"][c] - values["Q"][c] for c in baseline}
    return {"status": "complete", "coalitions": {s: _summary(values[s], draws, seed) for s in SUBSETS},
            "shapley": {p: {"raw_T_bits": _summary(v, draws, seed),
                             "recovery": _ratio(v, baseline, draws, seed, guard),
                             "share": _ratio(v, values["CQPR"], draws, seed, guard)} for p, v in allocation.items()},
            "CQ_coalition_recovery": _ratio(values["CQ"], baseline, draws, seed, guard),
            "C_plus_Q_shapley_share": _ratio(cq_sum, values["CQPR"], draws, seed, guard),
            "CQ_interaction_raw_T_bits": _summary(interaction, draws, seed),
            "efficiency_residual_bits": _mean([sum(allocation[p][c] for p in "CQPR") - values["CQPR"][c] for c in baseline])}


def _current_companion(rows, labels, protocol, phase, draws, contract):
    from .data import _labels

    if phase not in ("pilot", "exploration", "confirmation"):
        raise ValueError("current companion has no factorial stage")
    expected = set(protocol.get("episode_ids", [label["episode_id"] for label in labels]))
    label_map = _labels(labels, expected, contract)
    if any(row["kind"] != "behavior" for row in rows):
        raise ValueError("current companion supports descriptive behavior only")
    models = [GEMMA] if phase == "pilot" else list(LAYERS)
    conditions = ("B0", "B1", "B4", "B5") if phase == "pilot" else CONDITIONS
    cells = {(m, e, o, c) for m in models for e in expected for o in (0, 1) for c in conditions}
    index = {(r["model_key"], r["episode_id"], r["order"], r["condition"]): r for r in rows}
    if set(index) - cells:
        raise ValueError("unexpected current companion cells")
    missing = sorted(cell for cell in cells if cell not in index or index[cell].get("status") != "pass")
    seed = protocol.get("bootstrap_seed", protocol["seed"])
    summaries, contrasts = [], []
    for model in models:
        for condition in conditions:
            required = {(model, e, o, condition) for e in expected for o in (0, 1)}
            if not expected or any(cell in missing for cell in required):
                continue
            group = [index[cell] for cell in sorted(required)]
            summaries.append({"model_key": model, "condition": condition,
                              "metrics": {metric: _summary(_records(group,
                                  lambda r: _behavior_metrics(r, label_map[r["episode_id"]])[metric]), draws, seed)
                                  for metric in ("loss_nats", "accuracy", "gold_margin_bits")}})
        for control in ("B1", "B4"):
            required = {(model, e, o, c) for e in expected for o in (0, 1) for c in ("B0", control)}
            if not expected or any(cell in missing for cell in required):
                continue
            records = []
            for episode in sorted(expected):
                for order in (0, 1):
                    baseline, history = (index[model, episode, order, c] for c in (control, "B0"))
                    improvement = (_behavior_metrics(baseline, label_map[episode])["loss_nats"]
                                   - _behavior_metrics(history, label_map[episode])["loss_nats"])
                    records.append((history["cluster_id"], episode, order, improvement))
            contrasts.append({"model_key": model, "control": control, "descriptive_only": True,
                              "history_loss_improvement_nats": _summary(_cluster_values(records), draws, seed)})
    return {"phase": phase, "task": contract["task"], "status": "descriptive",
            "data_complete": bool(expected) and not missing, "missing_cells": missing,
            "primary_complete": False, "primary_family": [], "family_size": 0,
            "cuts": {}, "behavior": summaries, "history_contrasts": contrasts,
            "warnings": ["Matched static companion only; no forecasting primary tests or TTC claims.",
                         "Static labels/pairs and prior exposure are inherited, not fresh confirmation."],
            "claim_boundary": "Descriptive history association on the audited static current-interaction task."}


def analyze(rows: list[dict], labels: list[dict], protocol: dict, phase: str, draws: int = 19999) -> dict:
    """Return JSON-safe scientific summaries; incomplete primary cells fail closed.

    Only confirmation can reject claims. Other phases report observed effects,
    not an automatic scientific freeze. Bootstrap draws are shared by sorted
    wearer identity when cohorts coincide; NumPy is loaded only at evaluation.
    """
    if protocol.get("schema_version") == 3:
        from .dense_analysis import analyze_dense_current
        return analyze_dense_current(rows, labels, protocol, phase, draws=draws)
    if phase not in ("pilot", "exploration", "exploration_factorial", "confirmation"):
        raise ValueError("unknown phase: use pilot, exploration, exploration_factorial, or confirmation")
    if not isinstance(draws, int) or draws < 2:
        raise ValueError("draws must be >= 2")
    # Task enumeration can spell unions C+Q; canonicalize a copy, never inputs.
    rows = [dict(r) for r in rows]
    for r in rows:
        r["model_key"] = MODEL_ALIASES.get(r.get("model_key"), r.get("model_key"))
        position = r.get("position_set") or ""
        if "+" in position:
            parts = position.split("+")
            if len(parts) != len(set(parts)) or any(p not in ("C", "Q", "P", "R") for p in parts):
                raise ValueError("invalid factorial position union")
            r["position_set"] = "".join(p for p in "CQPR" if p in parts)
    label_map = _validate(rows, labels)
    if any(r.get("phase", phase) != phase for r in rows):
        raise ValueError("mixed analysis phases")
    from .protocol import CURRENT, protocol_contract, validate_protocol, validate_rows_contract

    contract = protocol_contract(protocol) if "schema_version" in protocol or "task" in protocol else None
    validate_rows_contract(rows, contract)
    if contract:
        validate_protocol(protocol)
        from .data import _labels

        _labels(labels, [label["episode_id"] for label in labels], contract)
        if contract["task"] == CURRENT:
            return _current_companion(rows, labels, protocol, phase, draws, contract)
    seed = protocol.get("bootstrap_seed", protocol.get("seed", 0))
    freeze = protocol.get("freeze") or {}
    guard = freeze.get("rho_guard_bits", protocol.get("rho_guard_bits", .01))
    if not _number(guard) or guard < 0:
        raise ValueError("invalid recovery guard")
    cuts = protocol.get("cuts", freeze.get("cuts", {}))
    if protocol.get("cuts") and freeze.get("cuts") and protocol["cuts"] != freeze["cuts"]:
        raise ValueError("conflicting top-level and freeze cuts")
    cuts = {MODEL_ALIASES.get(model, model): model_cuts for model, model_cuts in cuts.items()}
    if phase == "exploration":
        cuts = select_cuts(rows)
    if phase in ("confirmation", "exploration_factorial"):
        for model, n in LAYERS.items():
            for name in ("early_cut", "carrier_cut", "late_cut"):
                cut = cuts.get(model, {}).get(name)
                if type(cut) is not int or not 0 <= cut < n:
                    raise ValueError(f"{phase} requires frozen/supplied {model}/{name}")
    expected = set(freeze.get("episode_ids", protocol.get("episode_ids", label_map)))
    patch_expected = set(freeze.get("patch_episode_ids", protocol.get("patch_episode_ids", expected)))
    if not expected <= label_map.keys() or not patch_expected <= label_map.keys():
        raise ValueError("inventory episodes require evaluator labels")
    warnings = ["H/K reversal is descriptive, not route proof or native necessity.",
                "TTC bins use official variable event times, not paired-horizon effects.",
                "Recovery targets donor behavior, not correctness; 0.01-bit guard is proposed unless frozen."]
    passed = [r for r in rows if r.get("status") == "pass"]
    behavior = [r for r in passed if r["kind"] == "behavior"]
    behavior_index = {(r["model_key"], r["episode_id"], r["order"], r["condition"]): r for r in behavior}
    summaries = []
    bins = ((0, .5), (.5, 1), (1, 2), (2, math.inf))
    for model in LAYERS:
        for condition in CONDITIONS:
            group = [r for r in behavior if r["model_key"] == model and r["condition"] == condition]
            if not group:
                continue
            strata = [("all", group)] + [(f"[{lo},{hi})", [r for r in group if label_map[r["episode_id"]].get("time_to_contact_s") is not None
                        and lo <= label_map[r["episode_id"]]["time_to_contact_s"] < hi]) for lo, hi in bins]
            for stratum, selected in strata:
                if selected:
                    summaries.append({"model_key": model, "condition": condition, "ttc_bin": stratum,
                                      "n_rows": len(selected), "metrics": {metric: _summary(_records(selected,
                                          lambda r: _behavior_metrics(r, label_map[r["episode_id"]])[metric]), draws, seed)
                                          for metric in ("loss_nats", "accuracy", "gold_margin_bits")}})
    claims, missing, pilot = [], [], {}
    for model in LAYERS:
        absent = [(model, e, o, c) for e in sorted(expected) for o in (0, 1) for c in CONDITIONS[:5]
                  if (model, e, o, c) not in behavior_index]
        missing.extend(absent)
        components = {}
        if expected:
            for control in CONDITIONS[1:5]:
                if any((model, e, o, c) not in behavior_index for e in expected for o in (0, 1) for c in ("B0", control)):
                    continue
                rec = []
                for e in sorted(expected):
                    for o in (0, 1):
                        base = behavior_index[(model, e, o, "B0")]
                        other = behavior_index[(model, e, o, control)]
                        gain = _behavior_metrics(other, label_map[e])["loss_nats"] - _behavior_metrics(base, label_map[e])["loss_nats"]
                        rec.append((base["cluster_id"], e, o, gain))
                components[f"{control}-B0"] = _summary(_cluster_values(rec), draws, seed, test=True)
        history = _claim(model, "history_benefit", components)
        if len(components) != 4:
            history.update(valid=False, p_value=None)
        claims.append(history)
        carrier = cuts.get(model, {}).get("carrier_cut")
        patch_index = {(r["episode_id"], r["order"]): r for r in passed if r["kind"] == "patch"
                       and r["model_key"] == model and _direction(r) == "B0->B4"
                       and r["position_set"] == "V" and r["cut"] == carrier}
        absent_patch = [(model, e, o, "V", carrier) for e in sorted(patch_expected) for o in (0, 1)
                        if (e, o) not in patch_index]
        missing.extend(absent_patch)
        if not patch_expected:
            missing.append((model, "missing_patch_inventory"))
        primary = [patch_index[(e, o)] for e in sorted(patch_expected) for o in (0, 1)] if patch_expected and not absent_patch else []
        comp = {"V_T_bits": _summary(_records(primary, lambda r: _patch_values(r)[0]), draws, seed, test=True)} if primary else {}
        claims.append(_claim(model, "V_carrier", comp))
        gap_rows = [behavior_index[(model, e, o, "B0")] for e in sorted(expected) for o in (0, 1)
                    if (model, e, o, "B0") in behavior_index and (model, e, o, "B4") in behavior_index]
        gap = _records(gap_rows, lambda r: abs(r["margin_ab_bits"] - behavior_index[(model, r["episode_id"], r["order"], "B4")]["margin_ab_bits"]))
        gains = {k: v["estimate"] for k, v in components.items()}
        pilot[model] = {"observed_history_gain_nats": gains, "D0_bits": _summary(gap, draws, seed),
                        "recommendation": ("review_go" if len(gains) == 4 and all(v > 0 for v in gains.values())
                                           else "review_partial_pilot" if gains and all(v > 0 for v in gains.values())
                                           else "no_go_or_insufficient_history_evidence"),
                        "automatic_scientific_freeze": False}
    _holm(claims)
    for claim in claims:
        claim["confirmatory"] = phase == "confirmation"
        if phase != "confirmation":
            claim["reject"] = False
    profiles = []
    groups = defaultdict(list)
    factorial_groups = defaultdict(list)
    for r in rows:
        if r["kind"] == "patch":
            key = (r["model_key"], _direction(r), r["cut"])
            if r["position_set"] in SUBSETS:
                factorial_groups[key].append(r)
            if r.get("status") == "pass":
                groups[(*key, r["position_set"])].append(r)
    for (model, direction, cut, positions), group in sorted(groups.items()):
        t = _records(group, lambda r: _patch_values(r)[0])
        d = _records(group, lambda r: _patch_values(r)[1])
        profiles.append({"model_key": model, "direction": direction, "cut": cut, "position_set": positions,
                         "T_bits": _summary(t, draws, seed), "D0_bits": _summary(d, draws, seed),
                         "rho": _ratio(t, d, draws, seed, guard)})
    factorial = [{"model_key": model, "direction": direction, "cut": cut,
                  **_factorial(group, patch_expected, draws, seed, guard)}
                 for (model, direction, cut), group in sorted(factorial_groups.items())]
    reversal = []
    for model in LAYERS:
        early = cuts.get(model, {}).get("early_cut")
        late = cuts.get(model, {}).get("late_cut")
        for direction in ("B0->B4", "B4->B0"):
            required = [(early, "H"), (early, "K"), (late, "H"), (late, "K")]
            indexed = {(cut, pos): {(r["episode_id"], r["order"]): r
                       for r in groups.get((model, direction, cut, pos), [])} for cut, pos in required}
            cells = {(e, o) for e in patch_expected for o in (0, 1)}
            if not cells or any(set(indexed[k]) != cells for k in required):
                continue
            contrasts = defaultdict(list)
            for e, o in sorted(cells):
                eh, ek, lh, lk = [indexed[k][(e, o)] for k in required]
                values = {"early_H_minus_K": _patch_values(eh)[0] - _patch_values(ek)[0],
                          "late_K_minus_H": _patch_values(lk)[0] - _patch_values(lh)[0],
                          "early_H": _patch_values(eh)[0], "late_K": _patch_values(lk)[0]}
                for name, value in values.items():
                    contrasts[name].append((eh["cluster_id"], e, o, value))
            reversal.append({"model_key": model, "direction": direction, "descriptive_only": True,
                             "contrasts": {k: _summary(_cluster_values(v), draws, seed) for k, v in contrasts.items()}})
    pruning = []
    for key in sorted({(r["model_key"], r["condition"], r["cut"], r["position_set"]) for r in passed if r["kind"] == "prune"}):
        model, condition, cut, positions = key
        group = [r for r in passed if r["kind"] == "prune" and (r["model_key"], r["condition"], r["cut"], r["position_set"]) == key]
        runtime = [r for r in group if _number(r.get("elapsed_s")) and r["elapsed_s"] >= 0]
        pruning.append({"model_key": model, "condition": condition, "cut": cut, "position_set": positions,
                        "accuracy": _summary(_records(group, lambda r: _behavior_metrics(r, label_map[r["episode_id"]])["accuracy"]), draws, seed),
                        "elapsed_s": _summary(_records(runtime, lambda r: r["elapsed_s"]), draws, seed),
                        "runtime_rows": len(runtime), "n_rows": len(group), "speedup": None})
    if pruning:
        warnings.append("Pruning is descriptive; no speedup comparison without paired hardware/runtime provenance.")
    controls = [{"task_id": r["task_id"], "kind": r["kind"],
                 "absolute_residual_bits": abs(r["margin_ab_bits"] - r["recipient_margin_ab_bits"])
                 if r["kind"] == "self" else abs(r["margin_ab_bits"] - r["donor_margin_ab_bits"])}
                for r in passed if r["kind"] == "self" or (r["kind"] == "patch" and r["position_set"] == "A")]
    if any(r.get("status") != "pass" for r in rows):
        warnings.append("Failed rows retained as missing, never substituted in primary inference.")
    complete = bool(expected) and not missing
    if not complete:
        for claim in claims:
            claim["reject"] = False
    status = ("complete" if complete else "incomplete") if phase == "confirmation" else "descriptive"
    if phase == "exploration_factorial":
        status = "complete" if factorial and all(f["status"] == "complete" for f in factorial) else "incomplete"
    return {"phase": phase, "status": status, "primary_complete": complete,
            "missing_primary_cells": missing if phase == "confirmation" else [],
            "primary_family": claims, "family_size": 4,
            "cuts": cuts, "bootstrap": {"draws": draws, "seed": seed, "method": "null_centered_studentized_cluster"},
            "behavior": summaries, "patch_profiles": profiles, "factorial": factorial, "HK_reversal": reversal,
            "pilot_go_no_go": pilot, "pruning": pruning, "technical_controls": controls,
            "warnings": warnings, "claim_boundary": "V transfer is output control, not proof of H->V routing."}
