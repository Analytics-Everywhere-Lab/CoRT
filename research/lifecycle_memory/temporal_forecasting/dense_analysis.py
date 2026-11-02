"""Dense-current evaluator analysis and a label-free coordinator control gate.

The shared protocol validator owns schema-3 / contract-v2 validation. This
module owns the fixed 44-cell inventory and statistics; never import it into a
method worker or pass evaluator labels to runtime. Cuts are not selected here.
Both candidate orders are averaged within each wearer before resampling.
validate_dense_technical uses only public result rows, never evaluator labels.
"""

from collections import Counter
import math

from .analysis import (
    GEMMA, MODEL_ALIASES, _behavior_metrics, _bootstrap, _cluster_values,
    _interval, _mean, _number, _patch_values, _records, _summary, _validate,
)


CONDITIONS = ("B0", "B1", "B4", "B8")
CUTS = (22, 34)
POSITIONS = ("H", "V", "K", "A")
DIRECTIONS = (("B0", "B4"), ("B4", "B0"))
CONTROL_TOLERANCE_BITS = .001
ROWS_PER_EPISODE = 44
SCORE_FIELDS = ("a_logprob", "b_logprob", "margin_ab_bits")


def _describe(values, draws, seed):
    result = _summary(values, draws, seed)
    # The reusable helper includes a dormant testing field, not used here.
    result.pop("p_value", None)
    return result


def _cell(row):
    return (row["episode_id"], row["order"], row["kind"],
            row.get("condition") if row["kind"] == "behavior" else None,
            row.get("donor_condition"), row.get("recipient_condition"),
            row.get("cut"), row.get("position_set"))


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


def _cell_sort(cell):
    return tuple("" if value is None else str(value) for value in cell)


def _coverage(cells, index):
    missing = sorted(cells - index.keys(), key=_cell_sort)
    failed = sorted((cell for cell in cells & index.keys()
                     if index[cell].get("status") != "pass"), key=_cell_sort)
    return {"complete": bool(cells) and not missing and not failed,
            "expected_rows": len(cells), "present_rows": len(cells & index.keys()),
            "passed_rows": len(cells) - len(missing) - len(failed),
            "missing_rows": len(missing), "failed_rows": len(failed),
            "missing_cells": missing, "failed_cells": failed}


def _recovery(numerator, denominator, draws, seed, guard):
    """Retain all clusters; guard the ratio, never select on donor discrepancy.

    Small row/cluster discrepancies are counted separately by the caller. They
    are not removed from this estimand. If any resampled denominator is too
    small, withhold the CI instead of reporting a conditional-on-survival CI.
    """
    if numerator.keys() != denominator.keys():
        raise ValueError("recovery requires paired cluster numerator/denominator")
    keys = sorted(denominator)
    n, d = [numerator[k] for k in keys], [denominator[k] for k in keys]
    result = {"estimate": None, "ci95": None, "stable": False,
              "guard_bits": guard, "n_clusters": len(keys), "excluded_clusters": 0,
              "small_denominator_clusters": sum(value <= guard for value in d),
              "zero_denominator_clusters": sum(value == 0 for value in d),
              "numerator_mean_bits": _mean(n) if n else None,
              "denominator_mean_bits": _mean(d) if d else None,
              "bootstrap_draws": draws, "valid_bootstrap_draws": 0,
              "excluded_bootstrap_draws": 0,
              "estimand": "mean_cluster_absolute_residual_reduction / mean_cluster_absolute_donor_discrepancy"}
    if not keys:
        result.update(reason="no_clusters", bootstrap_draws=0)
        return result
    if _mean(d) <= guard:
        result["reason"] = "small_or_zero_denominator"
        result["bootstrap_draws"] = 0
        return result
    result["estimate"] = _mean(n) / _mean(d)
    if len(keys) < 2:
        result.update(reason="fewer_than_two_clusters", bootstrap_draws=0)
        return result
    means, _ = _bootstrap([n, d], draws, seed)
    excluded = int((means[:, 1] <= guard).sum())
    result.update(valid_bootstrap_draws=draws - excluded,
                  excluded_bootstrap_draws=excluded,
                  denominator_ci95=_interval(means[:, 1]))
    if excluded:
        result["reason"] = "small_or_zero_bootstrap_denominator"
        return result
    result.update(ci95=_interval(means[:, 0] / means[:, 1]), stable=True,
                  ci_method="paired_cluster_percentile_ratio_of_means")
    return result


def _valid_score(score):
    return (isinstance(score, dict) and all(_number(score.get(k)) for k in SCORE_FIELDS)
            and score["a_logprob"] <= 0 and score["b_logprob"] <= 0
            and math.isclose((score["a_logprob"] - score["b_logprob"]) / math.log(2),
                             score["margin_ab_bits"], abs_tol=1e-5, rel_tol=1e-5))


def _runtime_evidence(row, references):
    """Require the runtime's certificates, not just top-level success flags."""
    errors = []
    technical, controls = row.get("technical"), row.get("controls")
    if not isinstance(technical, dict):
        return ["missing_or_invalid_technical_report"]
    for key in ("alignment_pass", "self_patch_pass", "full_A_pass", "full_forward_reference",
                "selected_rows_equal", "preserved_rows_equal"):
        if technical.get(key) is not True:
            errors.append(f"technical.{key}")
    if technical.get("suffix_replay") is not False:
        errors.append("technical.suffix_replay")
    if type(technical.get("hook_calls")) is not int or technical["hook_calls"] != 1:
        errors.append("technical.hook_calls")
    if type(technical.get("forward_count")) is not int or technical["forward_count"] < 1:
        errors.append("technical.forward_count")
    reused = technical.get("controls_reused")
    if not isinstance(reused, dict) or any(type(reused.get(k)) is not bool for k in
                                           ("recipient_self", "donor_self", "full_A")):
        errors.append("technical.controls_reused")
    for key in ("self_max_drift_bits", "full_closure_max_drift_bits"):
        if not _number(technical.get(key)) or not 0 <= technical[key] <= CONTROL_TOLERANCE_BITS:
            errors.append(f"technical.{key}")
    if not _number(technical.get("tolerance_bits")) or technical["tolerance_bits"] != CONTROL_TOLERANCE_BITS:
        errors.append("technical.tolerance_bits")
    if not isinstance(controls, dict):
        return errors + ["missing_or_invalid_controls"]
    required = {"recipient_self": "recipient", "donor_self": "donor", "full_A": "donor"}
    if row["kind"] == "self":
        required["selected_self"] = "recipient"
    elif "selected_self" not in controls or controls["selected_self"] is not None:
        errors.append("controls.selected_self_must_be_null_for_patch")
    for name, role in required.items():
        certificate = controls.get(name)
        if not isinstance(certificate, dict):
            errors.append(f"controls.{name}.missing")
            continue
        if certificate.get("passed") is not True:
            errors.append(f"controls.{name}.passed")
        absolute = certificate.get("absolute_error_bits")
        drift = certificate.get("max_drift_bits")
        valid_absolute = isinstance(absolute, dict) and all(
            _number(absolute.get(k)) and 0 <= absolute[k] <= CONTROL_TOLERANCE_BITS for k in SCORE_FIELDS)
        if not valid_absolute:
            errors.append(f"controls.{name}.absolute_error_bits")
        if (not _number(drift) or not 0 <= drift <= CONTROL_TOLERANCE_BITS
                or (valid_absolute and not math.isclose(drift, max(absolute[k] for k in SCORE_FIELDS), abs_tol=1e-12))):
            errors.append(f"controls.{name}.max_drift_bits")
        score, reference = certificate.get("scores"), references.get(role)
        if not _valid_score(score):
            errors.append(f"controls.{name}.scores")
        elif reference is not None and any(
                abs(score[k] - reference[k]) / (1 if k == "margin_ab_bits" else math.log(2))
                > CONTROL_TOLERANCE_BITS for k in SCORE_FIELDS):
            errors.append(f"controls.{name}.native_score_mismatch")
        if name == "full_A":
            if type(certificate.get("hook_calls")) is not int or certificate["hook_calls"] != 1:
                errors.append("controls.full_A.hook_calls")
            for key in ("selected_rows_equal", "preserved_rows_equal"):
                if certificate.get(key) is not True:
                    errors.append(f"controls.full_A.{key}")
    return errors


def _controls(rows, behavior_index):
    """Check native baselines and self/A closure against independent behavior.

    Natural-log A/B differences are converted to bits, matching runtime's
    0.001-bit tolerance. Margin-only closure can miss common-mode score drift.
    """
    checks = []
    for row in rows:
        if row["kind"] == "behavior":
            continue
        check = {"task_id": row["task_id"], "kind": row["kind"],
                 "episode_id": row["episode_id"], "order": row["order"],
                 "cut": row["cut"], "position_set": row["position_set"],
                 "direction": f"{row['donor_condition']}->{row['recipient_condition']}",
                 "passed": False, "errors": []}
        if row.get("status") != "pass":
            check["errors"].append("row_failed")
            checks.append(check)
            continue
        references = {}
        for role in ("donor", "recipient"):
            reference = behavior_index.get((row["episode_id"], row["order"], row[f"{role}_condition"]))
            if reference is None or reference.get("status") != "pass":
                check["errors"].append(f"missing_{role}_native_behavior")
                continue
            references[role] = reference
            error = abs(row[f"{role}_margin_ab_bits"] - reference["margin_ab_bits"])
            check[f"{role}_baseline_error_bits"] = error
            if error > CONTROL_TOLERANCE_BITS:
                check["errors"].append(f"{role}_baseline_mismatch")
            if f"{role}_score" not in row:
                check["errors"].append(f"missing_{role}_score")
            else:
                score = row[f"{role}_score"]
                if not isinstance(score, dict) or any(not _number(score.get(k)) for k in
                        ("a_logprob", "b_logprob", "margin_ab_bits")):
                    check["errors"].append(f"invalid_{role}_score")
                elif any(abs(score[k] - reference[k]) / (1 if k == "margin_ab_bits" else math.log(2))
                         > CONTROL_TOLERANCE_BITS for k in ("a_logprob", "b_logprob", "margin_ab_bits")):
                    check["errors"].append(f"{role}_score_mismatch")
        if row["kind"] == "self" or row["position_set"] == "A":
            target = references.get("recipient" if row["kind"] == "self" else "donor")
            if target is not None:
                errors = {key: abs(row[key] - target[key]) / (1 if key == "margin_ab_bits" else math.log(2))
                          for key in ("a_logprob", "b_logprob", "margin_ab_bits")}
                check.update(absolute_error_bits=errors, max_drift_bits=max(errors.values()))
                if max(errors.values()) > CONTROL_TOLERANCE_BITS:
                    check["errors"].append("native_reproduction_failed")
        check["errors"].extend(_runtime_evidence(row, references))
        check["passed"] = not check["errors"]
        checks.append(check)
    return checks


def validate_dense_technical(rows: list[dict]) -> dict:
    """Label-free pre-receipt gate for real or simulated dense result rows.

    Require 44 cells per observed episode, independent native behavior scores,
    and mandatory runtime control certificates. The execution caller must also
    bind the observed episodes to its planned inventory: absent whole episodes
    cannot be discovered from result rows alone. No evaluator or bootstrap API
    is accessed. Invalid input returns a failure report rather than a pass.
    """
    result = {"status": "fail", "passed": False, "checks": [], "failures": [],
              "n_checks": 0, "failed_checks": 0, "failure_count": 0,
              "tolerance_bits": CONTROL_TOLERANCE_BITS}
    try:
        rows = [dict(r) for r in rows]
        tasks, cells, clusters = set(), set(), {}
        for row in rows:
            if (MODEL_ALIASES.get(row.get("model_key"), row.get("model_key")) != GEMMA
                    or row.get("kind") not in ("behavior", "patch", "self")
                    or type(row.get("order")) is not int or row["order"] not in (0, 1)
                    or any(not isinstance(row.get(k), str) or not row[k]
                           for k in ("episode_id", "cluster_id", "task_id"))):
                raise ValueError("invalid dense technical row identity")
            cell = _cell(row)
            if row["task_id"] in tasks or cell in cells:
                raise ValueError("duplicate dense technical task or cell")
            tasks.add(row["task_id"])
            cells.add(cell)
            episode, cluster = row["episode_id"], row["cluster_id"]
            if episode in clusters and clusters[episode] != cluster:
                raise ValueError("episode assigned to multiple wearers")
            clusters[episode] = cluster
            if not _valid_score(row):
                raise ValueError("invalid dense technical score")
            if row["kind"] != "behavior" and (
                    type(row.get("cut")) is not int or
                    any(not _number(row.get(f"{role}_margin_ab_bits")) for role in ("donor", "recipient"))):
                raise ValueError("invalid dense technical patch baseline or cut")
        if len(set(clusters.values())) != len(clusters):
            raise ValueError("dense technical rows require one episode per wearer")
        expected = _expected_cells(clusters)
        index = {_cell(row): row for row in rows}
        if cells - expected:
            raise ValueError("unexpected dense technical cells")
        coverage = _coverage(expected, index)
        result["coverage"] = coverage
        if not coverage["complete"]:
            result["failures"].append({"kind": "inventory", "errors": ["incomplete_dense_technical_inventory"]})
        behavior_index = {(r["episode_id"], r["order"], r["condition"]): r
                          for r in rows if r["kind"] == "behavior"}
        result["checks"] = _controls(rows, behavior_index)
        result["failures"].extend(c for c in result["checks"] if not c["passed"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        result["failures"].append({"kind": "validation", "errors": [str(error)]})
    result.update(n_checks=len(result["checks"]),
                  failed_checks=sum(not c["passed"] for c in result["checks"]),
                  failure_count=len(result["failures"]))
    result["passed"] = not result["failures"]
    result["status"] = "pass" if result["passed"] else "fail"
    return result


def analyze_dense_current(rows: list[dict], labels: list[dict], protocol: dict,
                          phase: str, draws: int = 19999) -> dict:
    """Return JSON-safe descriptive results; no testing family or cut selection.

    Inventory is protocol['episode_ids'] (or freeze['episode_ids']), falling
    back to evaluator labels, never observed successful rows. Full runs require
    96 episodes/wearers; 'current-canary' permits one through four. Missing
    cells suppress the affected summary, not just the incomplete wearer. A
    failed technical control suppresses all patch profiles but not behavior.
    Shared protocol/data validators must support schema 3 and contract v2.
    """
    from .data import _labels
    from .protocol import CURRENT, protocol_contract, validate_protocol, validate_rows_contract

    if phase not in ("current", "current-canary"):
        raise ValueError("unsupported dense-current phase")
    if type(draws) is not int or draws < 2:
        raise ValueError("draws must be an integer >= 2")
    if type(protocol.get("schema_version")) is not int or protocol["schema_version"] != 3 or protocol.get("study") != "temporal-current-dense-v1":
        raise ValueError("dense-current requires schema 3 / temporal-current-dense-v1")
    validate_protocol(protocol)
    contract = protocol_contract(protocol)
    if contract is None or contract.get("version") != 2 or contract.get("task") != CURRENT:
        raise ValueError("dense-current requires current task contract v2")
    rows = [dict(row, model_key=MODEL_ALIASES.get(row.get("model_key"), row.get("model_key"))) for row in rows]
    _validate(rows, labels)
    validate_rows_contract(rows, contract)
    label_map = _labels(labels, [label["episode_id"] for label in labels], contract)
    freeze = protocol.get("freeze") or {}
    if "episode_ids" in protocol and "episode_ids" in freeze and protocol["episode_ids"] != freeze["episode_ids"]:
        raise ValueError("conflicting episode inventories")
    episodes = protocol.get("episode_ids", freeze.get("episode_ids", list(label_map)))
    if not isinstance(episodes, (list, tuple)) or any(not isinstance(e, str) or not e for e in episodes) or len(episodes) != len(set(episodes)):
        raise ValueError("episode inventory must contain unique identifiers")
    expected = set(episodes)
    if not expected <= label_map.keys():
        raise ValueError("inventory episodes require evaluator labels")
    seed = protocol.get("bootstrap_seed", protocol.get("seed", 0))
    guard = freeze.get("rho_guard_bits", protocol.get("rho_guard_bits", .01))
    if type(seed) is not int or seed < 0 or not _number(guard) or guard < 0:
        raise ValueError("invalid bootstrap seed or recovery guard")
    wearer_episodes = {}
    for row in rows:
        if row["model_key"] != GEMMA or row["kind"] not in ("behavior", "patch", "self"):
            raise ValueError("dense-current supports Gemma31 behavior/patch/self only")
        if type(row.get("order")) is not int or row.get("phase", phase) != phase:
            raise ValueError("invalid candidate order or mixed analysis phases")
        wearer = row["cluster_id"]
        if wearer in wearer_episodes and wearer_episodes[wearer] != row["episode_id"]:
            raise ValueError("dense-current requires one episode per wearer")
        wearer_episodes[wearer] = row["episode_id"]
        if row["kind"] != "behavior":
            if type(row.get("cut")) is not int or row.get("condition", row["recipient_condition"]) != row["recipient_condition"]:
                raise ValueError("invalid patch cut or condition")
    cells = _expected_cells(expected)
    index = {_cell(row): row for row in rows}
    if len(index) != len(rows) or index.keys() - cells:
        raise ValueError("unexpected or duplicate dense-current cells")
    coverage = _coverage(cells, index)
    canary = phase == "current-canary"
    scope_complete = 1 <= len(expected) <= 4 if canary else len(expected) == 96
    coverage.update(scope="canary" if canary else "full", scope_complete=scope_complete,
                    expected_episodes=len(expected), observed_episodes=len({r["episode_id"] for r in rows}),
                    observed_clusters=len(wearer_episodes), rows_per_order=22,
                    rows_per_episode=ROWS_PER_EPISODE, full_episodes=96, canary_max_episodes=4,
                    counts_by_kind=dict(Counter(r["kind"] for r in rows)))
    coverage["complete"] = coverage["complete"] and scope_complete
    behavior_index = {(r["episode_id"], r["order"], r["condition"]): r for r in rows if r["kind"] == "behavior"}
    controls = _controls(rows, behavior_index)
    control_cells = {cell for cell in cells if cell[2] == "self" or (cell[2] == "patch" and cell[-1] == "A")}
    control_coverage = _coverage(control_cells, index)
    controls_pass = scope_complete and control_coverage["complete"] and all(c["passed"] for c in controls)

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
        info, selected = group({cell for cell in cells if cell[2] == "behavior" and cell[3] == condition})
        behavior.append({"model_key": GEMMA, "condition": condition, "coverage": info,
                         "metrics": {metric: metrics(selected, lambda r, m=metric:
                             _behavior_metrics(r, label_map[r["episode_id"]])[m])
                             for metric in ("accuracy", "loss_nats", "gold_margin_bits")} if selected else {}})
    for control in ("B1", "B4", "B8"):
        info, selected = group({cell for cell in cells if cell[2] == "behavior" and cell[3] in ("B0", control)})
        values = {}
        if selected:
            for metric in ("accuracy", "loss_nats"):
                records = []
                for episode in sorted(expected):
                    for order in (0, 1):
                        base, other = (behavior_index[episode, order, c] for c in ("B0", control))
                        base_value = _behavior_metrics(base, label_map[episode])[metric]
                        other_value = _behavior_metrics(other, label_map[episode])[metric]
                        gain = base_value - other_value if metric == "accuracy" else other_value - base_value
                        records.append((base["cluster_id"], episode, order, gain))
                values["accuracy_improvement" if metric == "accuracy" else "loss_improvement_nats"] = _describe(_cluster_values(records), draws, seed)
        contrasts.append({"model_key": GEMMA, "reference": "B0", "control": control,
                          "comparison": "chronological_order" if control == "B8" else "scene_history_benefit",
                          "positive_favors": "B0", "descriptive_only": True, "coverage": info, "metrics": values})
    for donor, recipient in DIRECTIONS:
        for cut in CUTS:
            for positions in POSITIONS:
                info, selected = group({cell for cell in cells if cell[2] == "patch" and cell[4:] == (donor, recipient, cut, positions)})
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
                    # Raw A-minus-B signs may cancel across orders. Preserve
                    # the requested raw statistic and expose each order too.
                    profile["by_order"] = [{"order": order, "metrics": {
                        name: metrics([r for r in selected if r["order"] == order], fn)
                        for name, fn in functions.items()}} for order in (0, 1)]
                    numerator = _records(selected, lambda r: _patch_values(r)[0])
                    denominator = _records(selected, lambda r: _patch_values(r)[1])
                    profile["recovery"] = _recovery(numerator, denominator, draws, seed, guard)
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
        "design": {"history_offsets_s": list(contract["history_offsets_s"]),
                   "conditions": {"B0": "nine chronological scene frames",
                                  "B1": "current scene frame only",
                                  "B4": "current scene frame repeated nine times",
                                  "B8": "eight shuffled history frames, current frame last"},
                   "position_sets": {"H": "eight history frames", "V": "current scene frame",
                                     "K": "non-scene positions", "A": "all active positions"}},
        "bootstrap": {"draws": draws, "seed": seed, "method": "paired_cluster_percentile",
                      "confidence_level": .95, "unit": "wearer", "orders_paired": True},
        "metric_definitions": {"accuracy": "A/B forced choice; ties count as 0.5",
                               "loss_nats": "negative log probability conditional on A or B",
                               "gold_margin_bits": "correct-minus-incorrect candidate log probability in bits",
                               "gold_signed_margin_movement_bits": "gold-oriented patched minus recipient margin; positive improves the correct candidate margin",
                               "gold_signed_donor_discrepancy_bits": "gold-oriented donor minus recipient margin; positive means donor has the better correct candidate margin",
                               "recovery": "ratio of cluster means, not mean row-wise recovery; not clipped"},
        "behavior": behavior, "history_contrasts": contrasts, "patch_profiles": profiles,
        "technical_controls": {"passed": controls_pass, "tolerance_bits": CONTROL_TOLERANCE_BITS,
                               "coverage": control_coverage, "failed_checks": sum(not c["passed"] for c in controls),
                               "checks": controls},
        "warnings": [
            "Descriptive only: no hypothesis tests, multiplicity correction, or confirmatory claim.",
            "Audited static pairs/labels and prior exposure are inherited, not fresh data.",
            "Null or negative current-task history benefits are valid outcomes, not exclusion criteria.",
            "Raw signed A/B movements can cancel across orders; gold-oriented and order-specific summaries are included.",
            "Small donor discrepancies are counted, not removed; unstable recovery CIs are withheld.",
            "Canary intervals with few wearers are diagnostic only; one wearer has no CI.",
            "Row/score validation does not audit source pixels, timestamps, or hidden-state geometry.",
        ],
        "claim_boundary": "Current-interaction association and directional output control under substitution only; no forecasting, native necessity, information routing, or universal depth-reversal claim.",
    }
