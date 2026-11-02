#!/usr/bin/env python3
"""Project allowlisted temporal statistics from an explicit offline upstream."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'current': 'runs/temporal_current/fir_full_439956_2026_10_25/results_bundle',
    'recognition_new_interaction': 'runs/temporal_recognition/fir_results_2026_10_25_v1/new_interaction/run-v2/full',
    'recognition_interaction_order': 'runs/temporal_recognition/fir_results_2026_10_25_v1/interaction_order/run-v2/full',
    'followup_new_interaction': 'runs/temporal_recognition/followup_fir_results_2026_10_25_v1/new_interaction/run-v1/full',
    'followup_interaction_order': 'runs/temporal_recognition/followup_fir_results_2026_10_25_v1/interaction_order/run-v1/full',
}
PUBLIC = {
    'current': ('temporal_current.json', []),
    'recognition_new_interaction': ('temporal_recognition.json', ['tasks', 'new_interaction']),
    'recognition_interaction_order': ('temporal_recognition.json', ['tasks', 'interaction_order']),
    'followup_new_interaction': ('temporal_followup.json', ['analyses', 'new_interaction']),
    'followup_interaction_order': ('temporal_followup.json', ['analyses', 'interaction_order']),
}
INVENTORY = {
    'current': (96, 306, 16, 0),
    'recognition_new_interaction': (20, 309, 16, 0),
    'recognition_interaction_order': (14, 309, 16, 0),
    'followup_new_interaction': (20, 174, 92, 2),
    'followup_interaction_order': (14, 174, 92, 2),
}
METRICS = {
    'accuracy', 'gold_margin_bits', 'loss_nats', 'accuracy_improvement',
    'gold_margin_improvement_bits', 'loss_improvement_nats',
    'original_answer_agreement', 'original_label_loss_nats', 'original_label_margin_bits',
    'original_answer_agreement_difference', 'original_label_loss_difference_nats',
    'original_label_margin_difference_bits', 'signed_margin_movement_bits',
    'signed_donor_discrepancy_bits', 'gold_signed_margin_movement_bits',
    'gold_signed_donor_discrepancy_bits', 'absolute_donor_discrepancy_bits',
    'absolute_residual_reduction_bits',
}
NUMERIC_FIELDS = {
    'estimate', 'ci95', 'n_clusters', 'valid', 'leave_one_wearer_out_range',
    'bootstrap_draws', 'denominator_ci95', 'denominator_mean_bits',
    'excluded_bootstrap_draws', 'excluded_clusters', 'excluded_rows', 'guard_bits',
    'n_rows', 'numerator_mean_bits', 'small_denominator_clusters',
    'small_denominator_rows', 'stable', 'valid_bootstrap_draws',
    'zero_denominator_clusters', 'zero_denominator_rows', 'denominator',
    'k_effect_guard_passed', 'native_guard_passed', 'unstable_draw_fraction',
}
ENUMS = {
    'ci_method': {'cluster_percentile', 'paired_cluster_percentile_ratio_of_means'},
    'estimand': {'mean_cluster_absolute_residual_reduction / mean_cluster_absolute_donor_discrepancy'},
    'denominator_units': {'bits'},
    'reason': {'no_clusters', 'fewer_than_two_clusters', 'small_or_zero_denominator',
               'small_or_zero_bootstrap_denominator', 'unstable_denominator',
               'native_donor_gap_guard_failed'},
}
LABELS = {
    'condition': {'B0', 'B1', 'B2', 'B4', 'B8', 'S1', 'S2', 'RV'},
    'control': {'B1', 'B2', 'B4', 'B8', 'S1', 'S2', 'RV', 'mean(S1,S2)'},
    'reference': {'B0'}, 'direction': {'B0->B4', 'B4->B0'},
    'position_set': {'H', 'V', 'K', 'A', 'C', 'Q', 'P', 'R', 'C+Q', 'C+P',
                     'C+R', 'Q+P', 'Q+R', 'P+R', 'C+Q+P', 'C+Q+R', 'C+P+R',
                     'Q+P+R', 'C+Q+P+R'},
    'cut': {22, 34}, 'order': {0, 1},
}
ROW_ONLY = {'n_rows', 'excluded_rows', 'small_denominator_rows', 'zero_denominator_rows'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(value, path):
    for key in path:
        value = value[key]
    return value


def read_json(path):
    def reject(value):
        raise ValueError('nonfinite JSON constant')
    return json.loads(path.read_text(), parse_constant=reject)


def numeric_tree(value):
    if value is None or type(value) is bool:
        return True
    if type(value) in (int, float):
        return math.isfinite(value)
    return isinstance(value, list) and all(numeric_tree(v) for v in value)


def reference(statistic):
    require(not (statistic.keys() - NUMERIC_FIELDS - ENUMS.keys() - {'wearer_values'}),
            'unrecognized statistical field')
    result = {}
    for key, value in statistic.items():
        if key in NUMERIC_FIELDS:
            require(numeric_tree(value), 'non-numeric reference')
            result[key] = value
        elif key in ENUMS:
            require(value in ENUMS[key], 'unrecognized statistical enum')
            result[key] = value
    return result


def labels(record):
    result = {}
    for key, allowed in LABELS.items():
        if key in record:
            require(record[key] in allowed, 'unrecognized scientific label')
            result[key] = record[key]
    return result


def public_analysis(name, root=ROOT):
    filename, path = PUBLIC[name]
    return resolve(read_json(root / 'results' / 'aggregate' / filename), path)


def has_public_reference(path):
    # The historical aggregates intentionally omit candidate-order strata.
    return 'by_order' not in path


def public_reference(name, expected):
    # Follow-up aggregates also omit leave-one-participant-out ranges.
    return {k: v for k, v in expected.items()
            if not (name.startswith('followup_') and k == 'leave_one_wearer_out_range')}


def project_analysis(name, original, public):
    """Construct a new schema; never recursively scrub an upstream object."""
    bootstrap = original['bootstrap']
    require(type(bootstrap['draws']) is int and bootstrap['draws'] >= 2, 'invalid draw count')
    require(type(bootstrap['seed']) is int and bootstrap['seed'] >= 0, 'invalid seed')
    require(all(public['bootstrap'][key] == bootstrap[key] for key in ('draws', 'seed')),
            'bootstrap settings differ from public aggregates')
    data = {'schema_version': 1, 'name': name,
            'bootstrap': {k: bootstrap[k] for k in ('draws', 'seed')},
            'anonymous_index_order': [], 'vectors': [], 'ratios': [], 'factorials': []}
    cohort = None
    lookup = {}

    def checked_reference(path):
        expected = reference(resolve(original, path))
        if has_public_reference(path):
            published = resolve(public, path)
            require(all(k in published and published[k] == v for k, v in public_reference(name, expected).items()),
                    'source statistics differ from public aggregates')
        return expected

    def vector(path, context):
        nonlocal cohort
        saved = resolve(original, path)['wearer_values']
        keys = sorted(saved)
        require(list(saved) == keys, 'saved vector order is not the historical sorted order')
        if cohort is None:
            cohort = keys
            data['anonymous_index_order'] = list(range(len(keys)))
        require(keys == cohort, 'unpaired participant cohort')
        require(keys and all(type(saved[k]) in (int, float) and math.isfinite(saved[k]) for k in keys),
                'invalid participant vector')
        index = len(data['vectors'])
        require(tuple(path) not in lookup, 'duplicate vector path')
        lookup[tuple(path)] = index
        data['vectors'].append({'path': path, 'labels': context, 'values': [saved[k] for k in keys],
                                'public_reference_available': has_public_reference(path),
                                'reference': checked_reference(path)})
        return index

    def ratio(path, numerator, denominator, native=None):
        item = {'path': path, 'numerator': numerator, 'denominator': denominator,
                'kind': 'absolute' if native is None else 'relative',
                'reference': checked_reference(path)}
        if native is not None:
            item['native'] = native
        data['ratios'].append(item)

    def metrics(path, context):
        values = resolve(original, path)
        require(set(values) <= METRICS, 'unrecognized metric')
        return {key: vector(path + [key], context) for key in values}

    for section in ('behavior', 'history_contrasts', 'order_contrasts'):
        for i, record in enumerate(original.get(section, [])):
            metrics([section, i, 'metrics'], labels(record))

    def profile(path, record):
        context = labels(record)
        if 'metrics' in record:
            indices = metrics(path + ['metrics'], context)
            for j, order in enumerate(record['by_order']):
                metrics(path + ['by_order', j, 'metrics'], {**context, **labels(order)})
            n = indices['absolute_residual_reduction_bits']
            d = indices['absolute_donor_discrepancy_bits']
        else:
            n = vector(path + ['T_bits'], context)
            d = vector(path + ['D0_bits'], context)
        ratio(path + ['recovery'], n, d)
        return {'position_set': record['position_set'], 'numerator': n, 'denominator': d}

    for i, record in enumerate(original['patch_profiles']):
        profile(['patch_profiles', i], record)

    for i, record in enumerate(original.get('factorial', [])):
        base = ['factorial', i]
        context = labels(record)
        subsets = [profile(base + ['subsets', j], p) for j, p in enumerate(record['subsets'])]
        t = {p['position_set']: p['numerator'] for p in subsets}
        d = subsets[0]['denominator']
        k = t['C+Q+P+R']
        derived = {}
        for contrast in ('K-minus-QPR', 'K-minus-CQR'):
            p = base + ['paired_differences', contrast]
            n = vector(p + ['T_difference_bits'], context)
            derived[contrast] = n
            ratio(p + ['recovery_difference'], n, d)
        p = base + ['CQ_interaction']
        vector(p + ['raw_gold_margin_bits'], context)
        n = vector(p + ['T_bits'], context)
        derived['CQ_interaction'] = n
        ratio(p + ['recovery'], n, d)
        for subset in ('Q+P+R', 'C+Q+R', 'C+Q'):
            ratio(base + ['relative_recoveries', subset.replace('+', '') + '/K'], t[subset], k, d)
        for player in 'CQPR':
            p = base + ['shapley', player]
            n = vector(p + ['T_bits'], context)
            derived[player] = n
            ratio(p + ['value'], n, d)
            ratio(p + ['share'], n, k, d)
        efficiency = record['shapley_efficiency']
        require(set(efficiency) == {'verified', 'max_abs_residual_bits', 'value_sum', 'K_recovery', 'share_sum'},
                'unexpected efficiency summary')
        require(all(numeric_tree(v) for v in efficiency.values()), 'invalid efficiency summary')
        require(efficiency == resolve(public, base + ['shapley_efficiency']), 'public efficiency mismatch')
        data['factorials'].append({'path': base, 'labels': context, 'subsets': subsets,
                                   'derived': derived, 'reference': efficiency})

    # Detect omissions, including any future statistic added outside known sections.
    def count_vectors(value):
        if isinstance(value, dict):
            return int('wearer_values' in value) + sum(count_vectors(v) for v in value.values())
        return sum(count_vectors(v) for v in value) if isinstance(value, list) else 0
    require(len(data['vectors']) == count_vectors(original), 'not all saved vectors were projected')
    return data


def project_technical(checks):
    rows = checks['checks']
    require(checks['n_checks'] == len(rows), 'technical count mismatch')
    require(all(type(c['passed']) is bool for c in rows), 'invalid technical pass flag')
    require(all(c['kind'] in {'patch', 'self', 'input_positions', 'shared_denominator'} for c in rows),
            'unexpected technical check kind')
    counts = Counter(c['kind'] for c in rows)
    passed = Counter(c['kind'] for c in rows if c['passed'])
    require(checks['failed_checks'] == sum(not c['passed'] for c in rows), 'failed check mismatch')
    return {'n_checks': len(rows), 'passed_checks': sum(passed.values()),
            'by_kind': {k: {'count': counts[k], 'passed': passed[k]} for k in sorted(counts)},
            'verification': 'saved_check_flags_only_not_runtime_reexecution'}


def project_permutations(records, checks):
    """Match private clip inventories before replacing keys by local ordinals."""
    observed = {c['episode_id'] for c in checks['checks'] if 'episode_id' in c}
    keyed = {r['episode_id']: r for r in records}
    require(len(keyed) == len(records) and set(keyed) == observed, 'permutation clip inventory mismatch')
    output = []
    for ordinal, key in enumerate(sorted(observed)):
        permutations = keyed[key]['permutations']
        require(set(permutations) == {'B0', 'B4', 'RV', 'S1', 'S2'}, 'unexpected permutation condition')
        for condition, values in permutations.items():
            require(isinstance(values, list) and len(values) == 17 and
                    all(type(v) is int and 0 <= v < 17 for v in values), 'invalid frame indices')
            if condition == 'B4':
                require(values == [16] * 17, 'invalid last-frame repeat')
            else:
                require(sorted(values) == list(range(17)), 'invalid frame permutation')
            if condition == 'B0':
                require(values == list(range(17)), 'invalid original order')
            if condition == 'RV':
                require(values == list(reversed(range(17))), 'invalid reversal')
            if condition in {'S1', 'S2'}:
                require(values[0] == 0 and values[-1] == 16, 'shuffle moved an endpoint')
        output.append({'clip_ordinal': ordinal, 'permutations': permutations})
    return output


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def export(upstream, output, public_root=ROOT):
    upstream, output = Path(upstream).resolve(), Path(output).resolve()
    require(upstream != output and upstream not in output.parents, 'output must not be inside upstream')
    datasets, snapshots = {}, {}

    def snapshot(path):
        snapshots[path] = path.read_bytes()
        value = read_json(path)
        require(path.read_bytes() == snapshots[path], 'source changed during read')
        return value

    for name, relative in SOURCES.items():
        original = snapshot(upstream / relative / 'analysis.json')
        data = project_analysis(name, original, public_analysis(name, public_root))
        require((len(data['anonymous_index_order']), len(data['vectors']), len(data['ratios']),
                 len(data['factorials'])) == INVENTORY[name], 'historical inventory mismatch')
        checks = snapshot(upstream / relative / 'technical_checks.json')
        data['technical_summary'] = project_technical(checks)
        if name.startswith('followup_'):
            task = name.removeprefix('followup_')
            path = Path('runs/temporal_recognition/followup_fir_deployment_2026_10_25_v1') / task / 'permutations.json'
            data['frozen_frame_orders'] = project_permutations(snapshot(upstream / path), checks)
        datasets[name] = data
    # Validate every dataset before writing any public output.
    require(all(path.read_bytes() == content for path, content in snapshots.items()), 'upstream source changed')
    for name, data in datasets.items():
        write_json(output / (name + '.json'), data)
    return {'datasets': len(datasets), 'vectors': sum(len(d['vectors']) for d in datasets.values()),
            'values': sum(len(v['values']) for d in datasets.values() for v in d['vectors']),
            'recovery_profiles': sum(sum(r['path'][0] == 'patch_profiles' for r in d['ratios'])
                                     for d in datasets.values()),
            'technical_checks': sum(d['technical_summary']['n_checks'] for d in datasets.values()),
            'frozen_clip_orders': sum(len(d.get('frozen_frame_orders', [])) for d in datasets.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path, help='Read-only research repository root')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'replay_temporal')
    args = parser.parse_args()
    print(json.dumps(export(args.upstream, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
