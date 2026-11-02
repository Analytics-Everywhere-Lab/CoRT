"""Descriptive recognition analysis and label-free pre-receipt control checks.

Candidate orders are averaged within episode, episodes within participant,
then participants are resampled with equal weight. No selection uses outcomes.
"""

from collections import Counter

from .analysis import (
    GEMMA, MODEL_ALIASES, _behavior_metrics, _cluster_values, _number,
    _patch_values, _records, _validate,
)
from .dense_analysis import (
    CONTROL_TOLERANCE_BITS, CUTS, DIRECTIONS, POSITIONS, ROWS_PER_EPISODE,
    _cell, _cell_sort, _controls, _coverage, _describe, _recovery, _valid_score,
)
from .recognition_protocol import CONDITIONS, is_recognition


def _expected_cells(episodes):
    cells = set()
    for episode in episodes:
        for order in (0, 1):
            for condition in CONDITIONS:
                cells.add((episode, order, "behavior", condition, None, None, None, None))
            for cut in CUTS:
                for donor, recipient in DIRECTIONS:
                    for positions in POSITIONS:
                        cells.add((episode, order, "patch", None, donor, recipient, cut, positions))
                cells.add((episode, order, "self", None, "B0", "B0", cut, "A"))
    return cells


def _row_inventory(rows):
    tasks, index, episodes = set(), {}, {}
    for row in rows:
        if (MODEL_ALIASES.get(row.get("model_key"), row.get("model_key")) != GEMMA
                or row.get("kind") not in ("behavior", "patch", "self")
                or type(row.get("order")) is not int or row["order"] not in (0, 1)
                or any(not isinstance(row.get(k), str) or not row[k].strip()
                       for k in ("episode_id", "cluster_id", "task_id"))):
            raise ValueError("invalid recognition row identity")
        cell = _cell(row)
        if row["task_id"] in tasks or cell in index:
            raise ValueError("duplicate recognition task or cell")
        tasks.add(row["task_id"])
        index[cell] = row
        episode, cluster = row["episode_id"], row["cluster_id"]
        if episode in episodes and episodes[episode] != cluster:
            raise ValueError("episode assigned to multiple wearers")
        episodes[episode] = cluster
        if row["kind"] != "behavior" and (
                type(row.get("cut")) is not int
                or row.get("condition", row["recipient_condition"]) != row["recipient_condition"]):
            raise ValueError("invalid recognition patch cut or condition")
        if row.get("status") == "pass":
            if not _valid_score(row):
                raise ValueError("invalid recognition score")
            if row["kind"] != "behavior" and any(
                    not _number(row.get(f"{role}_margin_ab_bits")) for role in ("donor", "recipient")):
                raise ValueError("invalid recognition patch baseline")
    counts = Counter(episodes.values())
    if any(n > 3 for n in counts.values()):
        raise ValueError("recognition permits at most three episodes per wearer")
    if index.keys() - _expected_cells(episodes):
        raise ValueError("unexpected recognition cells")
    return index, episodes, counts


def validate_recognition_technical(rows, expected_episodes=None):
    """Check complete 44-cell inventories and dense runtime certificates, no labels.

    Coordinators supply the planned episode IDs, including wholly missing ones.
    Without that inventory this checks only observed episodes, not study scope.
    """
    result = {"status": "fail", "passed": False, "checks": [], "failures": [],
              "n_checks": 0, "failed_checks": 0, "failure_count": 0,
              "tolerance_bits": CONTROL_TOLERANCE_BITS}
    try:
        rows = [dict(row) for row in rows]
        index, episodes, counts = _row_inventory(rows)
        expected = set(episodes if expected_episodes is None else expected_episodes)
        if set(episodes) - expected:
            raise ValueError("unexpected recognition episodes")
        coverage = _coverage(_expected_cells(expected), index)
        coverage.update(observed_clusters=len(counts), episodes_per_cluster=dict(counts))
        result["coverage"] = coverage
        if not coverage["complete"]:
            result["failures"].append({"kind": "inventory", "errors": ["incomplete_recognition_inventory"]})
        behavior = {(r["episode_id"], r["order"], r["condition"]): r
                    for r in rows if r["kind"] == "behavior"}
        result["checks"] = _controls(rows, behavior)
        result["failures"].extend(c for c in result["checks"] if not c["passed"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        result["failures"].append({"kind": "validation", "errors": [str(error)]})
    result.update(n_checks=len(result["checks"]),
                  failed_checks=sum(not c["passed"] for c in result["checks"]),
                  failure_count=len(result["failures"]))
    result["passed"] = not result["failures"]
    result["status"] = "pass" if result["passed"] else "fail"
    return result


def analyze_recognition(rows, labels, protocol, phase, draws=19999):
    """Report all fixed comparisons; incomplete groups never become complete cases."""
    from .data import _labels
    from .protocol import protocol_contract, validate_protocol, validate_rows_contract

    if phase not in ("recognition", "recognition-canary"):
        raise ValueError("unsupported recognition phase")
    if type(draws) is not int or draws < 2:
        raise ValueError("draws must be an integer >= 2")
    validate_protocol(protocol)
    if not is_recognition(protocol):
        raise ValueError("recognition requires schema 4 / contract version 3")
    contract = protocol_contract(protocol)
    rows = [dict(row, model_key=MODEL_ALIASES.get(row.get("model_key"), row.get("model_key"))) for row in rows]
    _validate(rows, labels)
    validate_rows_contract(rows, contract)
    label_map = _labels(labels, [label["episode_id"] for label in labels], contract)
    episodes = protocol.get("episode_ids", list(label_map))
    if (not isinstance(episodes, list) or any(not isinstance(e, str) or not e for e in episodes)
            or len(episodes) != len(set(episodes)) or not set(episodes) <= label_map.keys()):
        raise ValueError("invalid or unlabeled recognition inventory")
    expected = set(episodes)
    index, episode_clusters, cluster_counts = _row_inventory(rows)
    if any(row.get("phase", phase) != phase for row in rows):
        raise ValueError("mixed recognition analysis phases")
    cells = _expected_cells(expected)
    if index.keys() - cells:
        raise ValueError("unexpected recognition inventory cells")
    canary = phase == "recognition-canary"
    scope_complete = (len(expected) == 4 and len(cluster_counts) == 4 if canary else
                      len(expected) == protocol["episode_limit"]
                      and len(cluster_counts) >= protocol["minimum_clusters"])
    coverage = _coverage(cells, index)
    coverage.update(scope="canary" if canary else "full", scope_complete=scope_complete,
                    expected_episodes=len(expected), observed_episodes=len(episode_clusters),
                    observed_clusters=len(cluster_counts), episodes_per_cluster=dict(cluster_counts),
                    rows_per_order=22, rows_per_episode=ROWS_PER_EPISODE,
                    full_episodes=protocol["episode_limit"], canary_clusters=4,
                    counts_by_kind=dict(Counter(r["kind"] for r in rows)))
    coverage["complete"] = coverage["complete"] and scope_complete
    technical = validate_recognition_technical(rows, expected)
    controls_pass = technical["passed"] and scope_complete
    seed, guard = protocol["seed"], .01
    behavior_index = {(r["episode_id"], r["order"], r["condition"]): r
                      for r in rows if r["kind"] == "behavior"}

    def group(required):
        info = _coverage(required, index)
        info["complete"] = info["complete"] and scope_complete
        return info, [index[cell] for cell in sorted(required, key=_cell_sort)] if info["complete"] else []

    def metrics(selected, function):
        return _describe(_records(selected, function), draws, seed)

    def gold_sign(row):
        return 1 if row["order"] == label_map[row["episode_id"]]["correct_candidate"] else -1

    behavior, contrasts, profiles = [], [], []
    for condition in CONDITIONS:
        info, selected = group({c for c in cells if c[2] == "behavior" and c[3] == condition})
        behavior.append({"model_key": GEMMA, "condition": condition, "coverage": info,
                         "metrics": {metric: metrics(selected, lambda r, m=metric:
                             _behavior_metrics(r, label_map[r["episode_id"]])[m])
                             for metric in ("accuracy", "loss_nats", "gold_margin_bits")} if selected else {}})
    for control in ("B4", "B1", "B2"):
        info, selected = group({c for c in cells if c[2] == "behavior" and c[3] in ("B0", control)})
        values = {}
        if selected:
            for metric, name in (("accuracy", "accuracy_improvement"),
                                 ("loss_nats", "loss_improvement_nats"),
                                 ("gold_margin_bits", "gold_margin_improvement_bits")):
                records = []
                for episode in sorted(expected):
                    for order in (0, 1):
                        base, other = (behavior_index[episode, order, c] for c in ("B0", control))
                        delta = (_behavior_metrics(base, label_map[episode])[metric]
                                 - _behavior_metrics(other, label_map[episode])[metric])
                        records.append((base["cluster_id"], episode, order, -delta if metric == "loss_nats" else delta))
                values[name] = _describe(_cluster_values(records), draws, seed)
        contrasts.append({"model_key": GEMMA, "reference": "B0", "control": control,
                          "positive_favors": "B0", "descriptive_only": True, "coverage": info, "metrics": values})
    for donor, recipient in DIRECTIONS:
        for cut in CUTS:
            for positions in POSITIONS:
                info, selected = group({c for c in cells if c[2] == "patch"
                                        and c[4:] == (donor, recipient, cut, positions)})
                profile = {"model_key": GEMMA, "direction": f"{donor}->{recipient}", "cut": cut,
                           "position_set": positions, "coverage": info, "descriptive_only": True,
                           "technical_controls_pass": controls_pass, "metrics": {}, "recovery": None}
                if selected and controls_pass:
                    functions = {
                        "signed_margin_movement_bits": lambda r: r["margin_ab_bits"] - r["recipient_margin_ab_bits"],
                        "signed_donor_discrepancy_bits": lambda r: r["donor_margin_ab_bits"] - r["recipient_margin_ab_bits"],
                        "gold_signed_margin_movement_bits": lambda r: gold_sign(r) * (r["margin_ab_bits"] - r["recipient_margin_ab_bits"]),
                        "gold_signed_donor_discrepancy_bits": lambda r: gold_sign(r) * (r["donor_margin_ab_bits"] - r["recipient_margin_ab_bits"]),
                        "absolute_donor_discrepancy_bits": lambda r: _patch_values(r)[1],
                        "absolute_residual_reduction_bits": lambda r: _patch_values(r)[0],
                    }
                    profile["metrics"] = {name: metrics(selected, fn) for name, fn in functions.items()}
                    profile["by_order"] = [{"order": order, "metrics": {
                        name: metrics([r for r in selected if r["order"] == order], fn)
                        for name, fn in functions.items()}} for order in (0, 1)]
                    profile["recovery"] = _recovery(
                        _records(selected, lambda r: _patch_values(r)[0]),
                        _records(selected, lambda r: _patch_values(r)[1]), draws, seed, guard)
                    profile["recovery"].update(n_rows=len(selected), excluded_rows=0,
                        small_denominator_rows=sum(_patch_values(r)[1] <= guard for r in selected),
                        zero_denominator_rows=sum(_patch_values(r)[1] == 0 for r in selected))
                else:
                    profile["reason"] = "incomplete_inventory" if not selected else "technical_controls_failed"
                profiles.append(profile)
    complete = coverage["complete"] and controls_pass
    return {
        "phase": phase, "task": contract["task"], "study": protocol["study"],
        "status": "descriptive" if complete else "incomplete", "descriptive_only": True,
        "data_complete": coverage["complete"], "analysis_complete": complete,
        "primary_complete": False, "primary_family": [], "family_size": 0,
        "cuts": {GEMMA: list(CUTS)}, "coverage": coverage,
        "design": {"scene_count": 17,
                   "conditions": {"B0": "17 chronological scene frames", "B1": "last scene only",
                                  "B4": "last scene repeated 17 times", "B2": "first scene only"},
                   "position_sets": {"H": "first 16 scene frames", "V": "last scene frame",
                                     "K": "all non-scene positions", "A": "all active positions"}},
        "bootstrap": {"draws": draws, "seed": seed, "method": "paired_cluster_percentile",
                      "confidence_level": .95, "unit": "participant", "orders_paired": True,
                      "aggregation": ["candidate_orders", "episodes", "participant"],
                      "episodes_equal_within_participant": True, "participants_equal": True},
        "metric_definitions": {"accuracy": "forced A/B choice; ties count as 0.5",
                               "loss_nats": "negative log probability conditional on A or B",
                               "gold_margin_bits": "correct-minus-incorrect candidate margin in bits",
                               "recovery": "ratio of participant means, not mean row-wise recovery; not clipped"},
        "behavior": behavior, "history_contrasts": contrasts, "patch_profiles": profiles,
        "technical_controls": {**technical, "passed": controls_pass, "status": "pass" if controls_pass else "fail"},
        "warnings": [
            "Descriptive only: no hypothesis tests, multiplicity correction, or confirmatory claim.",
            "Both candidate orders and all eligible episodes contribute equally within each participant.",
            "Null and negative recognition benefits are valid outcomes, not exclusion criteria.",
            "Small donor discrepancies retain all clusters; unstable recovery intervals are withheld.",
            "Raw A/B movement can cancel across orders; gold-oriented and by-order summaries are included.",
            "Canary intervals are technical diagnostics, not scientific evidence.",
            "Row validation does not audit pixels, event annotations, or hidden-state geometry.",
        ],
        "claim_boundary": "Retrospective interaction recognition and directional output control under substitution only; no forecasting, native necessity, information routing, or universal depth-reversal claim.",
    }
