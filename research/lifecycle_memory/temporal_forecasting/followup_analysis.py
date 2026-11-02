"""Evaluator-only factorial/order follow-up, with a label-free technical gate.

No model or media loading occurs here. The protocol is imported lazily so that
method workers need not import evaluator code. All statistics use the same
sorted wearer cohort and bootstrap seed: reinitializing the proven helpers
replays identical wearer draws across subsets, directions, and contrasts.
"""

from collections import Counter
from itertools import combinations
import math
import re

from .analysis import (
    GEMMA, MODEL_ALIASES, _behavior_metrics, _cluster_values, _mean, _number,
    _patch_values, _ratio, _records,
)
from .common import digest
from .dense_analysis import (
    CONTROL_TOLERANCE_BITS, DIRECTIONS, _cell, _cell_sort, _controls, _coverage,
    _describe, _recovery, _valid_score,
)
from .recognition_protocol import NEW_INTERACTION, INTERACTION_ORDER, SCENE_SLOTS


SEED = 20261025
GUARD_BITS = .01
K = "C+Q+P+R"
COHORTS = {NEW_INTERACTION: (37, 20), INTERACTION_ORDER: (27, 14)}
STAGES = ("recognition", "recognition-canary")


def _expected_cells(episodes):
    from .followup_protocol import CONDITIONS, SUBSETS

    cells = set()
    for episode in episodes:
        for order in (0, 1):
            for condition in CONDITIONS:
                cells.add((episode, order, "behavior", condition, None, None, None, None))
            for donor, recipient in DIRECTIONS:
                for subset in (*SUBSETS, "A"):
                    cells.add((episode, order, "patch", None, donor, recipient, 34, subset))
            for condition in ("B0", "B4"):
                cells.add((episode, order, "self", None, condition, condition, 34, "A"))
    return cells


def _episode_ids(values):
    if isinstance(values, (str, bytes)) or values is None:
        raise ValueError("planned episode inventory is required")
    ids = list(values)
    if (not ids or any(not isinstance(e, str) or not e.strip() for e in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("expected episodes must be nonempty unique identifiers")
    return set(ids)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _row_inventory(rows):
    index, tasks, episodes = {}, set(), {}
    for row in rows:
        if (MODEL_ALIASES.get(row.get("model_key"), row.get("model_key")) != GEMMA
                or row.get("kind") not in ("behavior", "patch", "self")
                or type(row.get("order")) is not int or row["order"] not in (0, 1)
                or any(not isinstance(row.get(k), str) or not row[k].strip()
                       for k in ("episode_id", "cluster_id", "task_id"))):
            raise ValueError("invalid follow-up row identity")
        cell = _cell(row)
        if row["task_id"] in tasks or cell in index:
            raise ValueError("duplicate follow-up task or cell")
        tasks.add(row["task_id"])
        index[cell] = row
        episode, wearer = row["episode_id"], row["cluster_id"]
        if episode in episodes and episodes[episode] != wearer:
            raise ValueError("episode assigned to multiple wearers")
        episodes[episode] = wearer
        if row["kind"] != "behavior" and (
                type(row.get("cut")) is not int
                or row.get("condition", row.get("recipient_condition")) != row.get("recipient_condition")):
            raise ValueError("invalid follow-up patch cut or condition")
        if row.get("status") == "pass":
            if not _valid_score(row):
                raise ValueError("invalid follow-up score")
            if row["kind"] != "behavior" and any(
                    not _number(row.get(f"{role}_margin_ab_bits"))
                    or not _valid_score(row.get(f"{role}_score")) for role in ("donor", "recipient")):
                raise ValueError("invalid follow-up patch baseline")
    counts = Counter(episodes.values())
    if any(n > 3 for n in counts.values()):
        raise ValueError("follow-up permits at most three episodes per wearer")
    return index, episodes, counts


def _position_check(row, behavior):
    """Recheck the saved processor partition, including prefix/readout exclusion."""
    from .followup_protocol import SUBSETS

    errors = []
    positions = row.get("positions")
    if not isinstance(positions, dict):
        return ["missing_positions"]
    required = (*[f"H{i}" for i in range(1, 17)], "V", "CA", "CB", "H",
                "C", "Q", "P", "R", "K", "A", "F_video",
                *(name for name in SUBSETS if "+" in name))
    if set(positions) != set(required):
        errors.append("unexpected_or_missing_position_groups")
    for key in required:
        value = positions.get(key)
        if (not isinstance(value, list) or not value
                or any(type(p) is not int or p < 0 for p in value)
                or value != sorted(set(value))):
            errors.append(f"positions.{key}")
    if errors:
        return errors
    groups = {key: set(positions[key]) for key in required}
    primitives = [groups[key] for key in (*required[:19], "Q", "P", "R")]
    # The primitive spans, unlike H/C/K unions, must be disjoint and exhaust A.
    if sum(map(len, primitives)) != len(set().union(*primitives)):
        errors.append("overlapping_position_classes")
    if set().union(*primitives) != groups["A"]:
        errors.append("position_classes_do_not_cover_active")
    unions = {"H": set().union(*(groups[f"H{i}"] for i in range(1, 17))),
              "C": groups["CA"] | groups["CB"],
              "F_video": groups["H"] | groups["V"],
              "K": set().union(*(groups[p] for p in "CQPR"))}
    for key, value in unions.items():
        if groups[key] != value:
            errors.append(f"positions.{key}_union")
    if groups["F_video"] & groups["K"] or groups["F_video"] | groups["K"] != groups["A"]:
        errors.append("scene_non_scene_partition")
    readout = row.get("readout_position")
    if (type(readout) is not int or readout != max(groups["A"])
            or readout not in groups["P"] or readout in groups["C"] | groups["Q"] | groups["R"]
            or positions["P"] != list(range(min(groups["P"]), max(groups["A"]) + 1))
            or max(groups["Q"]) >= min(groups["P"])):
        errors.append("invalid_prefix_or_readout")
    answers = row.get("answer_token_ids")
    if (not isinstance(answers, dict) or set(answers) != {"A", "B"}
            or any(type(v) is not int or v < 0 for v in answers.values())
            or answers.get("A") == answers.get("B")):
        errors.append("invalid_answer_token_ids")
    for name, value in positions.items():
        if "+" in name and set(name.split("+")) <= set("CQPR"):
            union = sorted(set().union(*(groups[p] for p in name.split("+"))))
            if value != union:
                errors.append(f"positions.{name}_union")
    reference = behavior.get((row["episode_id"], row["order"], "B0"))
    if reference is not None:
        for key in ("positions", "readout_position", "answer_token_ids"):
            if row.get(key) != reference.get(key):
                errors.append(f"native_alignment.{key}")
    return errors


def _baseline_checks(rows):
    """Require identical native margins for every subset, not merely close gaps."""
    references, checks = {}, []
    for row in rows:
        if row["kind"] != "patch" or row.get("status") != "pass":
            continue
        key = (row["episode_id"], row["order"], row["donor_condition"], row["recipient_condition"])
        reference = references.setdefault(key, row)
        fields = ("donor_margin_ab_bits", "recipient_margin_ab_bits")
        errors = [f"inconsistent_subset_{field}" for field in fields
                  if row[field] != reference[field]]
        checks.append({"task_id": row["task_id"], "kind": "shared_denominator",
                       "passed": not errors, "errors": errors})
    return checks


def _frame_check(row, audit, condition, behavior):
    from .followup_protocol import scene_permutation

    if not isinstance(audit, dict):
        return ["missing_frame_audit"]
    errors = []
    expected = scene_permutation(row["task"], row["episode_id"], condition)
    indices = audit.get("scene_permutation")
    if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
            or indices != expected):
        errors.append("frame_audit.scene_permutation")
    for key, value in (("condition", condition), ("task", row["task"]),
                       ("episode_id", row["episode_id"]), ("seed", SEED)):
        if type(audit.get(key)) is not type(value) or audit[key] != value:
            errors.append(f"frame_audit.{key}")
    source = audit.get("source_frame_sha256")
    candidates = audit.get("candidate_sha256")
    if not isinstance(source, list) or len(source) != 17 or not all(_sha(h) for h in source):
        return errors + ["frame_audit.source_frame_sha256"]
    if not isinstance(candidates, list) or len(candidates) != 2 or not all(_sha(h) for h in candidates):
        return errors + ["frame_audit.candidate_sha256"]
    scene_keys = [SCENE_SLOTS[i] for i in expected]
    scene_hashes = [source[i] for i in expected]
    candidate_keys = [f"candidate{row['order']}", f"candidate{1 - row['order']}"]
    fields = {
        "source_frame_keys": list(SCENE_SLOTS), "scene_frame_keys": scene_keys,
        "scene_frame_sha256": scene_hashes, "candidate_keys": candidate_keys,
        "scene_order_sha256": digest({"indices": expected, "keys": scene_keys, "sha256": scene_hashes}),
        "candidate_order_sha256": digest({"keys": candidate_keys, "sha256": candidates}),
    }
    for key, value in fields.items():
        if audit.get(key) != value:
            errors.append(f"frame_audit.{key}")
    flags = {"endpoints_preserved": scene_hashes[0] == source[0] and scene_hashes[-1] == source[-1],
             "content_multiset_preserved": sorted(scene_hashes) == sorted(source),
             "prompt_chronology_violated": condition in ("S1", "S2", "RV")}
    for key, value in flags.items():
        if audit.get(key) is not value:
            errors.append(f"frame_audit.{key}")
    # Bind physical inputs across orders as well as across the five conditions.
    reference = behavior.get((row["episode_id"], 0, "B0"), {}).get("frame_audit", {})
    if isinstance(reference, dict) and reference:
        if source != reference.get("source_frame_sha256"):
            errors.append("frame_audit.source_frames_changed")
        ref_candidates = reference.get("candidate_sha256")
        if isinstance(ref_candidates, list) and len(ref_candidates) == 2:
            if candidates != (ref_candidates if row["order"] == 0 else ref_candidates[::-1]):
                errors.append("frame_audit.candidate_crops_changed")
    return errors


def _token_check(row, behavior):
    audit, positions = row.get("token_partition"), row.get("positions")
    if not isinstance(audit, dict) or not isinstance(positions, dict):
        return ["missing_token_partition"]
    errors = []
    for key in ("input_ids_sha256", "attention_mask_sha256"):
        if not _sha(audit.get(key)):
            errors.append(f"token_partition.{key}")
    for key in ("disjoint_exhaustive", "readout_in_P", "readout_excluded_from_CQR"):
        if audit.get(key) is not True:
            errors.append(f"token_partition.{key}")
    for key in ("cardinalities", "position_sha256", "token_ids_sha256"):
        mapping = audit.get(key)
        if not isinstance(mapping, dict) or mapping.keys() != positions.keys():
            errors.append(f"token_partition.{key}")
            continue
        for group, values in positions.items():
            if not isinstance(values, list):
                errors.append(f"token_partition.{group}.positions")
            elif key == "cardinalities":
                if type(mapping[group]) is not int or mapping[group] != len(values):
                    errors.append(f"token_partition.{group}.cardinality")
            elif key == "position_sha256":
                if mapping[group] != digest(values):
                    errors.append(f"token_partition.{group}.position_sha256")
            elif not _sha(mapping[group]):
                errors.append(f"token_partition.{group}.token_ids_sha256")
    reference = behavior.get((row["episode_id"], row["order"], "B0"))
    if reference is not None and audit != reference.get("token_partition"):
        errors.append("native_token_partition_alignment")
    if row["kind"] == "behavior":
        return errors
    if row.get("donor_token_partition") != audit:
        errors.append("donor_recipient_token_partition_alignment")
    selection = row.get("patch_selection")
    if not isinstance(selection, dict):
        return errors + ["missing_patch_selection"]
    names = ("A",) if row["position_set"] == "A" else row["position_set"].split("+")
    if any(not isinstance(positions.get(name), list) for name in names):
        return errors + ["invalid_patch_selection_classes"]
    selected = sorted({p for name in names for p in positions[name]})
    actual = selection.get("positions")
    if (not selected or not isinstance(actual, list) or any(type(p) is not int for p in actual)
            or actual != selected):
        errors.append("patch_selection.positions")
    if type(selection.get("cardinality")) is not int or selection["cardinality"] != len(selected):
        errors.append("patch_selection.cardinality")
    if selection.get("positions_sha256") != digest(selected):
        errors.append("patch_selection.positions_sha256")
    if not _sha(selection.get("token_ids_sha256")):
        errors.append("patch_selection.token_ids_sha256")
    token_hashes = audit.get("token_ids_sha256")
    if isinstance(token_hashes, dict):
        for name, values in positions.items():
            if values == selected and selection.get("token_ids_sha256") != token_hashes.get(name):
                errors.append("patch_selection.named_group_token_hash")
    if selection.get("includes_readout") is not (row.get("readout_position") in selected):
        errors.append("patch_selection.includes_readout")
    return errors


def validate_followup_technical(rows, expected_episodes):
    """Fail-closed 78-cell, real-runtime, label-free gate over planned episodes.

Full scope is exactly 37/20 or 27/14 clips/wearers; canary scope is four clips
from distinct wearers. This checks row evidence, not pixels, GPU telemetry,
sealed-release provenance, or the coordinator's independently pinned receipts.
"""
    result = {"status": "fail", "passed": False, "checks": [], "failures": [],
              "n_checks": 0, "failed_checks": 0, "failure_count": 0,
              "labels_opened": False, "tolerance_bits": CONTROL_TOLERANCE_BITS}
    try:
        rows = [dict(row) for row in rows]
        expected = _episode_ids(expected_episodes)
        index, episodes, counts = _row_inventory(rows)
        cells = _expected_cells(expected)
        if index.keys() - cells:
            raise ValueError("unexpected follow-up cells or episodes")
        task_names = {r.get("task") for r in rows}
        if len(task_names) != 1 or next(iter(task_names)) not in COHORTS:
            raise ValueError("missing or mixed follow-up recognition tasks")
        task = next(iter(task_names))
        full_clips, full_wearers = COHORTS[task]
        canary = len(expected) == 4
        scope_complete = (len(counts) == 4 if canary else
                          len(expected) == full_clips and len(counts) == full_wearers)
        phases = {r["phase"] for r in rows if "phase" in r}
        if phases and phases != {"recognition-canary" if canary else "recognition"}:
            raise ValueError("mixed or incorrect follow-up phases")
        coverage = _coverage(cells, index)
        coverage.update(scope="canary" if canary else "full", scope_complete=scope_complete,
                        expected_episodes=len(expected), observed_episodes=len(episodes),
                        observed_clusters=len(counts), episodes_per_cluster=dict(counts),
                        rows_per_episode=78, rows_per_order=39,
                        counts_by_kind=dict(Counter(r["kind"] for r in rows)))
        coverage["complete"] = coverage["complete"] and scope_complete
        result["coverage"] = coverage
        if not coverage["complete"]:
            result["failures"].append({"kind": "inventory", "errors": ["incomplete_followup_inventory"]})
        behavior = {(r["episode_id"], r["order"], r["condition"]): r
                    for r in rows if r["kind"] == "behavior"}
        result["checks"] = _controls(rows, behavior) + _baseline_checks(rows)
        identity = rows[0].get("identity") if rows else None
        contract_hashes = {r.get("contract_sha256") for r in rows}
        if len(contract_hashes) != 1 or not all(_sha(h) for h in contract_hashes):
            raise ValueError("missing or mixed input contract hashes")
        for row in rows:
            errors = []
            row_identity = row.get("identity")
            if (not isinstance(row_identity, dict) or row_identity.get("simulation") is not False
                    or not _sha(row_identity.get("protocol_sha256"))):
                errors.append("missing_real_protocol_identity")
            if row_identity != identity:
                errors.append("mixed_runtime_identities")
            if row.get("simulation", False) is not False:
                errors.append("simulation_not_accepted")
            if row.get("status") != "pass":
                errors.append("row_failed")
            errors.extend(_position_check(row, behavior))
            errors.extend(_token_check(row, behavior))
            condition = row["condition"] if row["kind"] == "behavior" else row["recipient_condition"]
            errors.extend(_frame_check(row, row.get("frame_audit"), condition, behavior))
            if row["kind"] != "behavior":
                errors.extend("donor_" + error for error in _frame_check(
                    row, row.get("donor_frame_audit"), row["donor_condition"], behavior))
            result["checks"].append({"task_id": row["task_id"], "kind": "input_positions",
                                     "passed": not errors, "errors": errors})
        result["failures"].extend(c for c in result["checks"] if not c["passed"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        result["failures"].append({"kind": "validation", "errors": [str(error)]})
    result.update(n_checks=len(result["checks"]),
                  failed_checks=sum(not c["passed"] for c in result["checks"]),
                  failure_count=len(result["failures"]))
    result["passed"] = not result["failures"]
    result["status"] = "pass" if result["passed"] else "fail"
    return result


def _finite_paired(numerator, denominator):
    if numerator.keys() != denominator.keys() or any(
            not _number(v) for values in (numerator, denominator) for v in values.values()):
        raise ValueError("ratios require finite paired wearer estimates")


def _paired_ratio(numerator, denominator, draws):
    _finite_paired(numerator, denominator)
    return _ratio(numerator, denominator, draws, SEED, GUARD_BITS)


def _absolute_recovery(numerator, denominator, draws):
    _finite_paired(numerator, denominator)
    return _recovery(numerator, denominator, draws, SEED, GUARD_BITS)


def _relative_ratio(numerator, k_effect, native, draws):
    native_guard = _absolute_recovery(k_effect, native, draws)
    result = _paired_ratio(numerator, k_effect, draws)
    result.update(native_guard_passed=native_guard["stable"],
                  k_effect_guard_passed=result["stable"],
                  denominator_units="bits", excluded_clusters=0)
    if not native_guard["stable"]:
        result.update(estimate=None, ci95=None, stable=False, reason="native_donor_gap_guard_failed")
    return result


def _difference(left, right):
    if left.keys() != right.keys():
        raise ValueError("contrast requires the identical wearer cohort")
    return {key: left[key] - right[key] for key in sorted(left)}


def _factorial(rows, labels, donor, recipient, draws):
    from .followup_protocol import SUBSETS

    selected = [r for r in rows if r["kind"] == "patch"
                and (r["donor_condition"], r["recipient_condition"]) == (donor, recipient)]
    by_subset = {s: [r for r in selected if r["position_set"] == s] for s in (*SUBSETS, "A")}
    by_cell = {(r["episode_id"], r["order"], r["position_set"]): r for r in selected}
    t = {s: _records(group, lambda r: _patch_values(r)[0]) for s, group in by_subset.items()}
    d = _records(by_subset[K], lambda r: _patch_values(r)[1])
    t[""] = dict.fromkeys(d, 0.)

    def describe(values):
        return _describe(values, draws, SEED)

    def recovery(values):
        return _absolute_recovery(values, d, draws)

    profiles = []
    for subset, group in by_subset.items():
        rho = recovery(t[subset])
        rho.update(n_rows=len(group), excluded_rows=0,
                   small_denominator_rows=sum(_patch_values(r)[1] <= GUARD_BITS for r in group),
                   zero_denominator_rows=sum(_patch_values(r)[1] == 0 for r in group))
        profiles.append({"model_key": GEMMA, "direction": f"{donor}->{recipient}", "cut": 34,
                         "position_set": subset, "recovery": rho,
                         "T_bits": describe(t[subset]), "D0_bits": describe(d)})
    differences = {}
    for subset in ("Q+P+R", "C+Q+R"):
        delta = _difference(t[K], t[subset])
        differences[f"K-minus-{subset.replace('+', '')}"] = {
            "T_difference_bits": describe(delta), "recovery_difference": recovery(delta), "paired": True}
    interaction = _difference(_difference(t["C+Q"], t["C"]), t["Q"])
    raw = []
    for row in by_subset["C+Q"]:
        episode, order = row["episode_id"], row["order"]
        sign = 1 if order == labels[episode]["correct_candidate"] else -1
        value = sign * (row["margin_ab_bits"] - by_cell[episode, order, "C"]["margin_ab_bits"]
                        - by_cell[episode, order, "Q"]["margin_ab_bits"] + row["recipient_margin_ab_bits"])
        raw.append((row["cluster_id"], episode, order, value))
    allocation = {}
    for player in "CQPR":
        phi = dict.fromkeys(d, 0.)
        others = [p for p in "CQPR" if p != player]
        for size in range(4):
            weight = math.factorial(size) * math.factorial(3 - size) / math.factorial(4)
            for subset in combinations(others, size):
                before = "+".join(p for p in "CQPR" if p in subset)
                after = "+".join(p for p in "CQPR" if p in subset or p == player)
                for wearer in phi:
                    phi[wearer] += weight * (t[after][wearer] - t[before][wearer])
        allocation[player] = phi
    residuals = {w: math.fsum(allocation[p][w] for p in "CQPR") - t[K][w] for w in d}
    verified = all(math.isclose(math.fsum(allocation[p][w] for p in "CQPR"), t[K][w],
                                abs_tol=1e-10, rel_tol=1e-10) for w in d)
    if not verified:
        raise ValueError("Shapley efficiency failed")
    shapley = {p: {"T_bits": describe(phi), "value": recovery(phi),
                   "share": _relative_ratio(phi, t[K], d, draws)} for p, phi in allocation.items()}
    k_recovery = recovery(t[K])
    value_sum = (math.fsum(v["value"]["estimate"] for v in shapley.values())
                 if all(v["value"]["estimate"] is not None for v in shapley.values()) else None)
    share_sum = (math.fsum(v["share"]["estimate"] for v in shapley.values())
                 if all(v["share"]["stable"] for v in shapley.values()) else None)
    if ((value_sum is not None and not math.isclose(value_sum, k_recovery["estimate"], abs_tol=1e-10))
            or (share_sum is not None and not math.isclose(share_sum, 1., abs_tol=1e-10))):
        raise ValueError("Shapley value/share sum failed")
    return {"model_key": GEMMA, "direction": f"{donor}->{recipient}", "cut": 34,
            "status": "complete", "empty_recovery": 0., "subsets": profiles,
            "paired_differences": differences,
            "relative_recoveries": {f"{s.replace('+', '')}/K": _relative_ratio(t[s], t[K], d, draws)
                                    for s in ("Q+P+R", "C+Q+R", "C+Q")},
            "CQ_interaction": {"raw_gold_margin_bits": describe(_cluster_values(raw)),
                               "recovery": recovery(interaction), "T_bits": describe(interaction),
                               "raw_baseline": "recipient_native_margin", "recovery_baseline": 0.},
            "shapley": shapley,
            "shapley_efficiency": {"verified": verified, "max_abs_residual_bits": max(map(abs, residuals.values())),
                                   "value_sum": value_sum, "K_recovery": k_recovery["estimate"],
                                   "share_sum": share_sum},
            "excluded_rows": 0, "excluded_clusters": 0}


def _behavior(rows, labels, episodes, draws):
    from .followup_protocol import CONDITIONS

    index = {(r["episode_id"], r["order"], r["condition"]): r
             for r in rows if r["kind"] == "behavior"}
    behavior = []
    for condition in CONDITIONS:
        transformed = condition in ("S1", "S2", "RV")
        selected = [index[e, o, condition] for e in sorted(episodes) for o in (0, 1)]
        names = (("original_answer_agreement", "accuracy"),
                 ("original_label_loss_nats", "loss_nats"),
                 ("original_label_margin_bits", "gold_margin_bits")) if transformed else (
                     ("accuracy", "accuracy"), ("loss_nats", "loss_nats"), ("gold_margin_bits", "gold_margin_bits"))
        behavior.append({"model_key": GEMMA, "condition": condition,
                         "transformed_input": transformed, "truth_accuracy": not transformed,
                         "metrics": {name: _describe(_records(selected, lambda r, m=metric:
                             _behavior_metrics(r, labels[r["episode_id"]])[m]), draws, SEED)
                             for name, metric in names}})
    contrasts = []
    for name, conditions in (("mean(S1,S2)", ("S1", "S2")), ("S1", ("S1",)),
                             ("S2", ("S2",)), ("RV", ("RV",)), ("B4", ("B4",))):
        metrics = {}
        for metric, output in (("accuracy", "original_answer_agreement_difference"),
                               ("loss_nats", "original_label_loss_difference_nats"),
                               ("gold_margin_bits", "original_label_margin_difference_bits")):
            records = []
            for episode in sorted(episodes):
                for order in (0, 1):
                    base = index[episode, order, "B0"]
                    other = _mean([_behavior_metrics(index[episode, order, c], labels[episode])[metric]
                                   for c in conditions])
                    value = _behavior_metrics(base, labels[episode])[metric] - other
                    records.append((base["cluster_id"], episode, order, value))
            metrics[output] = _describe(_cluster_values(records), draws, SEED)
        contrasts.append({"reference": "B0", "control": name, "contrast": f"B0-minus-{name}",
                          "main_order_diagnostic": name == "mean(S1,S2)", "paired": True,
                          "metrics": metrics, "loss_difference_positive_favors": "control",
                          "agreement_and_margin_positive_favor": "B0"})
    return behavior, contrasts


def analyze_followup(rows, labels, protocol, stage, draws=19999):
    """Return descriptive results, or a technical-only canary without labels.

Small draw overrides are for bounded numeric tests; production execution must
enforce the frozen 19,999 draws. A failed cell suppresses scientific summaries
for the entire cohort. Unstable ratios are withheld, not treated as exclusions
or technical failures. Inputs and evaluator labels are never modified.
"""
    from .followup_protocol import validate_followup_protocol
    from .protocol import protocol_contract, validate_rows_contract

    if stage not in STAGES:
        raise ValueError("unsupported follow-up stage")
    if type(draws) is not int or draws < 2:
        raise ValueError("draws must be an integer >= 2")
    validate_followup_protocol(protocol)
    contract = protocol_contract(protocol)
    if (contract.get("version") != 3 or protocol["seed"] != SEED
            or protocol["episode_limit"] != COHORTS[contract["task"]][0]
            or protocol["minimum_clusters"] != COHORTS[contract["task"]][1]):
        raise ValueError("follow-up requires the unchanged input contract and full frozen cohort")
    expected = _episode_ids(protocol.get("episode_ids"))
    if (stage == "recognition-canary" and len(expected) != 4
            or stage == "recognition" and len(expected) != protocol["episode_limit"]):
        raise ValueError("follow-up stage/episode scope mismatch")
    rows = [dict(row) for row in rows]
    validate_rows_contract(rows, contract)
    if any(row.get("phase", stage) != stage for row in rows):
        raise ValueError("mixed follow-up analysis phases")
    technical = validate_followup_technical(rows, expected)
    # Execution adds these analysis-only fields after sealing the protocol.
    allowed_hashes = {digest({k: v for k, v in protocol.items() if k not in removed})
                      for removed in ((), ("episode_ids",), ("bootstrap_seed",),
                                      ("episode_ids", "bootstrap_seed"))}
    if any(isinstance(r.get("identity"), dict)
           and r["identity"].get("protocol_sha256") not in allowed_hashes for r in rows):
        raise ValueError("follow-up result protocol identity mismatch")
    canary = stage == "recognition-canary"
    complete = technical["passed"]
    result = {"phase": stage, "task": contract["task"], "study": protocol["study"],
              "status": ("technical_pass" if canary else "descriptive") if complete else "incomplete",
              "analysis_complete": complete, "data_complete": technical.get("coverage", {}).get("complete", False),
              "scientific_analysis_performed": False, "labels_opened": False,
              "descriptive_only": True, "primary_complete": False, "primary_family": [], "family_size": 0,
              "cuts": {GEMMA: [34]}, "coverage": technical.get("coverage", {}),
              "technical_controls": technical, "behavior": [], "order_contrasts": [], "factorial": [],
              "patch_profiles": [],
              "bootstrap": {"draws": draws, "seed": SEED, "unit": "wearer",
                            "method": "paired_cluster_percentile", "confidence_level": .95,
                            "joint_resamples": True, "aggregation": ["candidate_orders", "clips", "wearers"],
                            "clips_equal_within_wearer": True, "wearers_equal": True,
                            "shuffles_averaged_within_clip": True},
              "warnings": ["Exploratory, descriptive reuse of previously seen clips; not confirmatory.",
                           "S1/S2/RV violate the unchanged prompt's chronological-order assertion.",
                           "Corrupted-input scores measure retention of the original answer, not transformed truth accuracy.",
                           "Reversal never flips labels; it changes endpoints and event semantics.",
                           "The two specified shuffles do not estimate every possible permutation.",
                           "Recovery measures donor output control, not native necessity or information routing.",
                           "Unstable ratios are withheld without dropping clips, wearers, or bootstrap draws.",
                           "Canaries are label-free technical diagnostics, not scientific evidence."],
              "metric_definitions": {
                  "recovery": "wearer-mean residual reduction T(S) / wearer-mean absolute native donor gap; not clipped",
                  "relative_recovery": "T(S)/T(K), guarded in bits after the native donor-gap guard",
                  "raw_interaction": "gold-oriented m(CQ)-m(C)-m(Q)+recipient native margin, paired per example",
                  "recovery_interaction": "v(CQ)-v(C)-v(Q)+v(empty), with v(empty)=0",
                  "shapley": "four-player allocation in recovery units; shares normalize by stable v(K)",
                  "coalition": "CQ coalition recovery is not the sum of C and Q Shapley values",
                  "loss_contrast": "B0 loss minus control loss; negative favors B0"}}
    if canary or not complete:
        return result
    from .data import _labels

    label_map = _labels(labels, expected, contract)
    result.update(labels_opened=True, scientific_analysis_performed=True)
    result["behavior"], result["order_contrasts"] = _behavior(rows, label_map, expected, draws)
    result["factorial"] = [_factorial(rows, label_map, donor, recipient, draws) for donor, recipient in DIRECTIONS]
    result["patch_profiles"] = [profile for direction in result["factorial"] for profile in direction["subsets"]]
    return result
