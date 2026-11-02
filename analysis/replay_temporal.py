#!/usr/bin/env python3
"""Offline, single-worker replay of anonymous temporal sufficient statistics."""
from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import combinations
import json
import math
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from analysis.export_temporal_replay import (
    ENUMS, INVENTORY, PUBLIC, ROW_ONLY, has_public_reference, public_analysis, public_reference,
    read_json, require, resolve, write_json,
)
from research.lifecycle_memory.temporal_forecasting.analysis import _ratio
from research.lifecycle_memory.temporal_forecasting.dense_analysis import _describe, _recovery

TOLERANCE = 2e-12
K = 'C+Q+P+R'
VECTOR_REFERENCE = {'ci95', 'ci_method', 'estimate', 'leave_one_wearer_out_range', 'n_clusters', 'valid'}
ABSOLUTE_REFERENCE = {
    'bootstrap_draws', 'ci95', 'ci_method', 'denominator_ci95', 'denominator_mean_bits',
    'estimand', 'estimate', 'excluded_bootstrap_draws', 'excluded_clusters', 'guard_bits',
    'n_clusters', 'numerator_mean_bits', 'small_denominator_clusters', 'stable',
    'valid_bootstrap_draws', 'zero_denominator_clusters',
}
RELATIVE_REFERENCE = {
    'ci95', 'denominator', 'denominator_ci95', 'denominator_units', 'estimate',
    'excluded_clusters', 'guard_bits', 'k_effect_guard_passed', 'native_guard_passed',
    'stable', 'unstable_draw_fraction',
}
EFFICIENCY_REFERENCE = {'K_recovery', 'max_abs_residual_bits', 'share_sum', 'value_sum', 'verified'}
EXPECTED_COMPARISONS = {
    'current': 2076, 'recognition_new_interaction': 2094, 'recognition_interaction_order': 2094,
    'followup_new_interaction': 2628, 'followup_interaction_order': 2544,
}


def validate_reference(reference, fields):
    """Schema 1 describes the frozen, fully available historical references."""
    require(isinstance(reference, dict) and set(reference) == fields, 'incomplete or unexpected reference fields')
    for key, value in reference.items():
        if key in {'ci95', 'denominator_ci95', 'leave_one_wearer_out_range'}:
            require(isinstance(value, list) and len(value) == 2
                    and all(type(v) in (int, float) and math.isfinite(v) for v in value)
                    and value[0] <= value[1], 'invalid historical reference interval')
        elif key in {'valid', 'stable', 'verified', 'k_effect_guard_passed', 'native_guard_passed'}:
            require(type(value) is bool, 'invalid reference flag')
        elif key in ENUMS:
            require(isinstance(value, str) and value in ENUMS[key], 'invalid reference method')
        else:
            require(type(value) in (int, float) and math.isfinite(value), 'invalid reference number')


def expected_paths(name):
    """Structural inventory is independent of the supplied reference records."""
    vectors, ratios = set(), {}
    followup = name.startswith('followup_')
    basic = ('accuracy', 'gold_margin_bits', 'loss_nats')
    transformed = ('original_answer_agreement', 'original_label_margin_bits', 'original_label_loss_nats')
    for i in range(5 if followup else 4):
        vectors.update(('behavior', i, 'metrics', m) for m in (transformed if followup and i >= 2 else basic))
    if followup:
        metrics = ('original_answer_agreement_difference', 'original_label_loss_difference_nats',
                   'original_label_margin_difference_bits')
        section, count = 'order_contrasts', 5
    else:
        metrics = ('accuracy_improvement', 'loss_improvement_nats')
        if name != 'current':
            metrics += ('gold_margin_improvement_bits',)
        section, count = 'history_contrasts', 3
    vectors.update((section, i, 'metrics', m) for i in range(count) for m in metrics)
    patch_metrics = ('absolute_donor_discrepancy_bits', 'absolute_residual_reduction_bits',
                     'gold_signed_donor_discrepancy_bits', 'gold_signed_margin_movement_bits',
                     'signed_donor_discrepancy_bits', 'signed_margin_movement_bits')
    for i in range(32 if followup else 16):
        base = ('patch_profiles', i)
        ratios[(*base, 'recovery')] = 'profile'
        if followup:
            vectors.update((*base, m) for m in ('T_bits', 'D0_bits'))
        else:
            vectors.update((*base, 'metrics', m) for m in patch_metrics)
            vectors.update((*base, 'by_order', j, 'metrics', m) for j in range(2) for m in patch_metrics)
    for i in range(2 if followup else 0):
        base = ('factorial', i)
        for j in range(16):
            vectors.update((*base, 'subsets', j, m) for m in ('T_bits', 'D0_bits'))
            ratios[(*base, 'subsets', j, 'recovery')] = 'profile'
        for contrast in ('K-minus-QPR', 'K-minus-CQR'):
            vectors.add((*base, 'paired_differences', contrast, 'T_difference_bits'))
            ratios[(*base, 'paired_differences', contrast, 'recovery_difference')] = 'absolute'
        vectors.update((*base, 'CQ_interaction', m) for m in ('T_bits', 'raw_gold_margin_bits'))
        ratios[(*base, 'CQ_interaction', 'recovery')] = 'absolute'
        for subset in ('QPR/K', 'CQR/K', 'CQ/K'):
            ratios[(*base, 'relative_recoveries', subset)] = 'relative'
        for player in 'CQPR':
            vectors.add((*base, 'shapley', player, 'T_bits'))
            ratios[(*base, 'shapley', player, 'value')] = 'absolute'
            ratios[(*base, 'shapley', player, 'share')] = 'relative'
    return vectors, ratios


class Comparison:
    """Count checked numbers separately from exact equality and CI availability."""

    def __init__(self):
        self.numeric_values = 0
        self.exact_numeric_values = 0
        self.ci95_intervals = 0
        self.withheld_ci95 = 0
        self.max_abs_difference = 0.0
        self.differences = []

    def check(self, actual, expected, path=()):
        if isinstance(expected, dict):
            for key, value in expected.items():
                if not isinstance(actual, dict) or key not in actual:
                    self.differences.append({'path': list(path) + [key], 'reason': 'missing_field'})
                    continue
                if key == 'ci95':
                    self.ci95_intervals += int(value is not None)
                    self.withheld_ci95 += int(value is None)
                self.check(actual[key], value, (*path, key))
        elif isinstance(expected, list):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                self.differences.append({'path': list(path), 'reason': 'shape_mismatch'})
            else:
                for i, value in enumerate(expected):
                    self.check(actual[i], value, (*path, i))
        elif type(expected) in (int, float):
            self.numeric_values += 1
            valid = type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
            delta = abs(actual - expected) if valid else math.inf
            self.exact_numeric_values += int(valid and actual == expected)
            if valid:
                self.max_abs_difference = max(self.max_abs_difference, delta)
            if delta > TOLERANCE:
                self.differences.append({'path': list(path), 'reason': 'numeric_mismatch'})
        elif type(actual) is not type(expected) or actual != expected:
            self.differences.append({'path': list(path), 'reason': 'value_mismatch'})

    def result(self):
        return {**vars(self), 'passed': not self.differences}


@lru_cache(maxsize=4096)
def describe(values, draws, seed):
    result = _describe(dict(enumerate(values)), draws, seed)
    return {k: v for k, v in result.items() if k != 'wearer_values'}


@lru_cache(maxsize=4096)
def recovery(numerator, denominator, draws, seed, guard):
    return _recovery(dict(enumerate(numerator)), dict(enumerate(denominator)), draws, seed, guard)


@lru_cache(maxsize=4096)
def relative(numerator, denominator, native, draws, seed, guard):
    result = _ratio(dict(enumerate(numerator)), dict(enumerate(denominator)), draws, seed, guard)
    native_guard = recovery(denominator, native, draws, seed, guard)
    result.update(native_guard_passed=native_guard['stable'], k_effect_guard_passed=result['stable'],
                  denominator_units='bits', excluded_clusters=0)
    if not native_guard['stable']:
        result.update(estimate=None, ci95=None, stable=False, reason='native_donor_gap_guard_failed')
    return result


def difference(left, right):
    require(len(left) == len(right), 'unpaired contrast')
    return tuple(a - b for a, b in zip(left, right))


def factorial_vectors(subsets):
    """Use the historical operation order, including Shapley accumulation order."""
    expected = {'+'.join(s) for n in range(1, 5) for s in combinations('CQPR', n)} | {'A'}
    require(set(subsets) == expected, 'incomplete factorial subset inventory')
    n = len(subsets[K])
    require(all(len(v) == n for v in subsets.values()), 'unpaired factorial vectors')
    t = {**subsets, '': (0.0,) * n}
    result = {'K-minus-QPR': difference(t[K], t['Q+P+R']),
              'K-minus-CQR': difference(t[K], t['C+Q+R']),
              'CQ_interaction': difference(difference(t['C+Q'], t['C']), t['Q'])}
    for player in 'CQPR':
        phi = [0.0] * n
        others = [p for p in 'CQPR' if p != player]
        for size in range(4):
            weight = math.factorial(size) * math.factorial(3 - size) / math.factorial(4)
            for subset in combinations(others, size):
                before = '+'.join(p for p in 'CQPR' if p in subset)
                after = '+'.join(p for p in 'CQPR' if p in subset or p == player)
                for i in range(n):
                    phi[i] += weight * (t[after][i] - t[before][i])
        result[player] = tuple(phi)
    return result


def validate(data):
    require(data['schema_version'] == 1 and data['name'] in PUBLIC, 'unsupported replay dataset')
    order = data['anonymous_index_order']
    require(len(order) >= 2 and all(type(i) is int for i in order)
            and order == list(range(len(order))), 'anonymous index order changed')
    require((len(order), len(data['vectors']), len(data['ratios']), len(data['factorials']))
            == INVENTORY[data['name']], 'historical inventory mismatch')
    draws, seed = data['bootstrap']['draws'], data['bootstrap']['seed']
    require(type(draws) is int and draws >= 2 and type(seed) is int and seed >= 0, 'invalid bootstrap settings')
    paths = set()
    vector_paths, ratio_paths = expected_paths(data['name'])
    for record in data['vectors']:
        validate_reference(record.get('reference'), VECTOR_REFERENCE)
        require(type(record['public_reference_available']) is bool
                and record['public_reference_available'] == has_public_reference(record['path']),
                'invalid public comparison scope')
        require(tuple(record['path']) not in paths, 'duplicate vector path')
        paths.add(tuple(record['path']))
        values = record['values']
        require(len(values) == len(order) and all(type(v) in (int, float) and math.isfinite(v) for v in values),
                'invalid vector shape or nonfinite value')
        require(record['reference']['n_clusters'] == len(order), 'cluster count mismatch')
    require(paths == vector_paths, 'historical vector path coverage mismatch')
    count = len(data['vectors'])
    require(count > 0, 'empty vector inventory')
    paths = set()
    for ratio in data['ratios']:
        path = tuple(ratio['path'])
        require(path in ratio_paths and path not in paths, 'historical ratio path coverage mismatch')
        paths.add(path)
        category = ratio_paths[path]
        require(ratio['kind'] == ('relative' if category == 'relative' else 'absolute'), 'wrong ratio kind')
        fields = RELATIVE_REFERENCE if category == 'relative' else ABSOLUTE_REFERENCE
        validate_reference(ratio.get('reference'), fields | ROW_ONLY if category == 'profile' else fields)
        for key in ('numerator', 'denominator') + (('native',) if ratio['kind'] == 'relative' else ()):
            require(type(ratio[key]) is int and 0 <= ratio[key] < count, 'invalid paired vector reference')
        guard = ratio['reference']['guard_bits']
        require(type(guard) in (int, float) and math.isfinite(guard) and guard >= 0, 'invalid ratio guard')
        if ratio['kind'] == 'absolute':
            require(all(v >= 0 for v in data['vectors'][ratio['denominator']]['values']),
                    'negative absolute donor discrepancy')
    require(paths == set(ratio_paths), 'historical ratio path coverage mismatch')
    paths = set()
    for factorial in data['factorials']:
        validate_reference(factorial.get('reference'), EFFICIENCY_REFERENCE)
        path = tuple(factorial['path'])
        require(path not in paths, 'duplicate factorial path')
        paths.add(path)
        for subset in factorial['subsets']:
            for key in ('numerator', 'denominator'):
                require(type(subset[key]) is int and 0 <= subset[key] < count,
                        'invalid factorial vector reference')
        require(all(type(i) is int and 0 <= i < count for i in factorial['derived'].values()),
                'invalid derived vector reference')
    require(paths == {('factorial', i) for i in range(INVENTORY[data['name']][3])},
            'historical factorial path coverage mismatch')


def replay(data, public=None):
    validate(data)
    comparison, projection = Comparison(), Comparison()
    values = [tuple(v['values']) for v in data['vectors']]
    draws, seed = data['bootstrap']['draws'], data['bootstrap']['seed']
    derived_count = 0
    for factorial in data['factorials']:
        subsets = factorial['subsets']
        require(len({s['position_set'] for s in subsets}) == len(subsets), 'duplicate subset')
        native = values[subsets[0]['denominator']]
        require(all(values[s['denominator']] == native for s in subsets), 'unpaired factorial denominators')
        derived = factorial_vectors({s['position_set']: values[s['numerator']] for s in subsets})
        require(set(derived) == set(factorial['derived']), 'incomplete derived vector inventory')
        for name, calculated in derived.items():
            index = factorial['derived'][name]
            comparison.check(list(calculated), list(values[index]), (*factorial['path'], 'derived', name))
            values[index] = calculated
            derived_count += 1

    summaries, ratios, efficiencies = [], [], []
    for index, record in enumerate(data['vectors']):
        actual = describe(values[index], draws, seed)
        comparison.check(actual, record['reference'], tuple(record['path']))
        if public is not None and record['public_reference_available']:
            projection.check(resolve(public, record['path']), public_reference(data['name'], record['reference']),
                             tuple(record['path']))
        summaries.append({'path': record['path'], 'statistics': actual})
    ratio_by_path = {}
    row_fields = 0
    for record in data['ratios']:
        n, d = values[record['numerator']], values[record['denominator']]
        guard = record['reference']['guard_bits']
        actual = (recovery(n, d, draws, seed, guard) if record['kind'] == 'absolute'
                  else relative(n, d, values[record['native']], draws, seed, guard))
        expected = {k: v for k, v in record['reference'].items() if k not in ROW_ONLY}
        row_fields += len(record['reference'].keys() & ROW_ONLY)
        comparison.check(actual, expected, tuple(record['path']))
        if public is not None:
            projection.check(resolve(public, record['path']), record['reference'], tuple(record['path']))
        ratios.append({'path': record['path'], 'statistics': actual})
        require(tuple(record['path']) not in ratio_by_path, 'duplicate ratio path')
        ratio_by_path[tuple(record['path'])] = actual

    for factorial in data['factorials']:
        base = tuple(factorial['path'])
        k = next(s for s in factorial['subsets'] if s['position_set'] == K)
        total = values[k['numerator']]
        allocations = {p: values[factorial['derived'][p]] for p in 'CQPR'}
        residuals = [math.fsum(allocations[p][i] for p in 'CQPR') - total[i] for i in range(len(total))]
        value_stats = [ratio_by_path[(*base, 'shapley', p, 'value')] for p in 'CQPR']
        share_stats = [ratio_by_path[(*base, 'shapley', p, 'share')] for p in 'CQPR']
        actual = {
            'verified': all(math.isclose(math.fsum(allocations[p][i] for p in 'CQPR'), total[i],
                                         abs_tol=1e-10, rel_tol=1e-10) for i in range(len(total))),
            'max_abs_residual_bits': max(map(abs, residuals)),
            'value_sum': math.fsum(s['estimate'] for s in value_stats)
                         if all(s['estimate'] is not None for s in value_stats) else None,
            'share_sum': math.fsum(s['estimate'] for s in share_stats)
                         if all(s['stable'] for s in share_stats) else None,
            'K_recovery': recovery(total, values[k['denominator']], draws, seed, .01)['estimate'],
        }
        comparison.check(actual, factorial['reference'], (*base, 'shapley_efficiency'))
        if public is not None:
            projection.check(resolve(public, [*base, 'shapley_efficiency']), factorial['reference'], base)
        efficiencies.append({'path': list(base), 'statistics': actual})

    require(comparison.numeric_values == EXPECTED_COMPARISONS[data['name']]
            and comparison.ci95_intervals == len(data['vectors']) + len(data['ratios'])
            and comparison.withheld_ci95 == 0, 'incomplete historical comparison coverage')

    return {'bootstrap': data['bootstrap'], 'vector_count': len(values),
            'vector_value_count': sum(map(len, values)), 'derived_vector_count': derived_count,
            'ratio_count': len(ratios),
            'recovery_profile_count': sum(r['path'][0] == 'patch_profiles' for r in ratios),
            'vector_summaries_in_public_aggregates': sum(v['public_reference_available'] for v in data['vectors']),
            'vector_summaries_original_source_only': sum(not v['public_reference_available'] for v in data['vectors']),
            'row_diagnostic_fields_copied_not_recomputed': row_fields,
            'comparison': comparison.result(),
            'public_aggregate_comparison': projection.result() if public is not None else None,
            'vector_summaries': summaries, 'ratios': ratios, 'factorial_efficiency': efficiencies}


def run(directory=ROOT / 'results' / 'replay_temporal', public_root=ROOT):
    aggregate_paths = {Path(public_root) / 'results' / 'aggregate' / filename for filename, _ in PUBLIC.values()}
    snapshots = {path: path.read_bytes() for path in aggregate_paths}
    results, datasets = {}, {}
    for name in PUBLIC:
        data = read_json(Path(directory) / (name + '.json'))
        require(data['name'] == name, 'dataset filename/name mismatch')
        validate(data)
        datasets[name] = data
    for name, data in datasets.items():
        results[name] = replay(data, public_analysis(name, public_root))
    totals = {k: sum(r[k] for r in results.values()) for k in (
        'vector_count', 'vector_value_count', 'derived_vector_count', 'ratio_count',
        'recovery_profile_count', 'row_diagnostic_fields_copied_not_recomputed',
        'vector_summaries_in_public_aggregates', 'vector_summaries_original_source_only')}
    totals.update({key: sum(r['comparison'][key] for r in results.values()) for key in (
        'numeric_values', 'exact_numeric_values', 'ci95_intervals', 'withheld_ci95')})
    unchanged = all(path.read_bytes() == content for path, content in snapshots.items())
    passed = unchanged and all(r['comparison']['passed'] and r['public_aggregate_comparison']['passed']
                               for r in results.values())
    return {'schema_version': 1, 'numpy_version': np.__version__, 'numeric_workers': 1,
            'comparison_tolerance': TOLERANCE, 'passed': passed, 'status': 'pass' if passed else 'fail',
            'counts': totals,
            'coverage': {'datasets': len(results), 'all_exported_estimates_and_ci95_checked': True,
                         'all_temporal_runtime_outputs_recomputed': False,
                         'public_aggregates_unchanged': unchanged},
            'scope': 'saved_participant_summaries_and_paired_temporal_ratios',
            'not_recomputed': ['raw_accuracy_or_gold_margin_scoring', 'within_clip_or_participant_aggregation',
                               'row_level_denominator_diagnostics', 'runtime_technical_checks',
                               'frame_alignment_or_selection_qa'],
            'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'results' / 'replay_temporal')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'rebuilt' / 'temporal_replay.json')
    args = parser.parse_args()
    try:
        report = run(args.data)
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        report = {'schema_version': 1, 'status': 'fail', 'passed': False, 'counts': {},
                  'coverage': {'all_exported_estimates_and_ci95_checked': False},
                  'validation_error': type(error).__name__}
        if type(error) is ValueError:
            report['reason'] = str(error)
    write_json(args.output, report)
    print(json.dumps({'status': report['status'], **report['counts']}, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
