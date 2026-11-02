#!/usr/bin/env python3
"""Offline supplementary static replay. No model, network, or private inputs."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/replay_static_supplement'
MODELS = ('gemma4_31b_hf_bf16', 'qwen35_27b_hf_bf16')
OPS = ('empty', 'C', 'Q', 'CQ')
CONTRAST = np.array([1., -1., -1., 1.])
FILES = {'metric_cells.csv', 'metric_reference.json', 'behavior_reference.json',
         'exploratory.json', 'gemma_controls.json', 'sensitivity_reference.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(array, dtype):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


def compare(actual, expected, path='result'):
    """Compare saved references, including shape, finite values and key inventory."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), path + ' keys')
        for key in expected:
            compare(actual[key], expected[key], path + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path + ' shape')
        for i, item in enumerate(expected):
            compare(actual[i], item, path + '/' + str(i))
    elif isinstance(expected, bool):
        require(type(actual) is bool and actual == expected, path)
    elif isinstance(expected, (int, float)):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), path)
    else:
        require(actual == expected, path)


def vector(value, n, nonnegative=False):
    require(isinstance(value, list) and len(value) == n
            and all(type(v) in (int, float) for v in value), 'bad vector shape/type')
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (n,) and np.isfinite(array).all(), 'nonfinite vector')
    require(not nonnegative or (array >= 0).all(), 'negative discrepancy')
    return array


def indexes(n, seed, draws=10000):
    require(type(seed) is int and type(draws) is int and draws == 10000, 'bootstrap configuration')
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def rho(v, n, idx=None):
    require(set(v) == {'error_bits', 'gap_bits'}, 'unexpected vector fields')
    error = vector(v['error_bits'], n, True)
    gap = vector(v['gap_bits'], n, True)
    require((gap > 1e-12).all(), 'nonpositive donor gap')
    if idx is None:
        return float(1 - float(error.mean()) / float(gap.mean()))
    return 1 - error[idx].mean(axis=1) / gap[idx].mean(axis=1)


def interval(point, boot):
    return {'estimate': float(point), 'ci95': np.quantile(boot, [.025, .975]).tolist()}


def holm(tails):
    result, running = {}, 0.
    for i, (name, value) in enumerate(sorted(tails.items(), key=lambda kv: (kv[1], kv[0]))):
        running = max(running, min(1., (len(tails)-i)*value))
        result[name] = running
    return result


def metric_replay(rows, reference):
    fields = {'corruption', 'direction', 'cluster_ordinal', 'raw_interaction_bits',
              *('m_' + op + '_bits' for op in OPS), *('error_' + op + '_bits' for op in OPS)}
    require(len(rows) == 322 and all(set(r) == fields for r in rows), 'metric CSV shape')
    require(len(reference['cells']) == 7, 'metric stratum inventory')
    idx = indexes(46, reference['bootstrap']['seed'], reference['bootstrap']['draws'])
    require(array_digest(idx, '<i8') == reference['bootstrap']['index_sha256_little_endian_int64'],
            'historical metric draw index hash')
    cells, paired, used = [], [], set()
    for saved in reference['cells']:
        key = (saved['corruption'], saved['direction'])
        require(key not in used, 'duplicate metric stratum')
        used.add(key)
        selected = [r for r in rows if (r['corruption'], r['direction']) == key]
        require([int(r['cluster_ordinal']) for r in selected] == list(range(1, 47)),
                'metric ordinal order')
        margins = np.array([[float(r['m_' + op + '_bits']) for op in OPS] for r in selected])
        errors = np.array([[float(r['error_' + op + '_bits']) for op in OPS] for r in selected])
        require(np.isfinite(margins).all() and np.isfinite(errors).all()
                and (errors >= 0).all() and (errors[:, 0] > 1e-12).all(), 'metric values')
        raw = margins @ CONTRAST
        compare(raw.tolist(), [float(r['raw_interaction_bits']) for r in selected])
        raw_boot = raw[idx].mean(axis=1)
        recovery = 1 - errors.mean(axis=0) / errors[:, 0].mean()
        recovery_boot = np.column_stack([1 - errors[:, i][idx].mean(axis=1) /
                                         errors[:, 0][idx].mean(axis=1) for i in range(4)]) @ CONTRAST
        both = np.column_stack([raw_boot, recovery_boot])
        count = 2 * (2 if key[0] == 'all' else 1) * (2 if key[1] == 'all' else 1)
        cell = {'corruption': key[0], 'direction': key[1], 'participant_cluster_count': 46,
                'rows_per_cluster': count, 'paired_rows': 46*count,
                'raw_interaction_bits': float(raw.mean()),
                'raw_ci95_bits': np.quantile(raw_boot, [.025, .975]).tolist(),
                'recovery_interaction': float(recovery @ CONTRAST),
                'recovery_ci95': np.quantile(recovery_boot, [.025, .975]).tolist(),
                'cluster_equal_baseline_gap_bits': float(errors[:, 0].mean()),
                'cluster_equal_margin_bits': dict(zip(OPS, margins.mean(axis=0).tolist())),
                'recovery_set_values': dict(zip(OPS, recovery.tolist())),
                'paired_bootstrap_covariance_raw_recovery': np.cov(both, rowvar=False, ddof=1).tolist(),
                'cluster_margin_covariance_operator_order': list(OPS),
                'cluster_margin_covariance_bits_squared': np.cov(margins, rowvar=False, ddof=1).tolist()}
        compare(cell, saved)
        cells.append(cell)
        paired.append(both)
    require(all((r['corruption'], r['direction']) in used for r in rows), 'extra metric stratum')
    require(array_digest(np.stack(paired, axis=1), '<f8') == reference['paired_draws_sha256'],
            'historical paired interaction draw hash')
    return {'cells': cells, 'saved_index_and_paired_draw_hashes_match': True}


def python_quantile(values, p):
    ordered = sorted(values)
    position = (len(ordered)-1)*p
    lo, hi = math.floor(position), math.ceil(position)
    fraction = position-lo
    return ordered[lo]*(1-fraction) + ordered[hi]*fraction


def python_bootstrap(values, seed):
    rng = random.Random(seed)
    return [sum([rng.choice(values) for _ in values])/len(values) for _ in range(10000)]


def behavior_replay(data, references):
    results = {}
    for model in MODELS:
        values = data[model]['cluster_vectors']
        require(set(values) == {'real', 'blank', 'wrong', 'pair'}, 'behavior vector inventory')
        cells, tails = {}, {}
        for name, v in values.items():
            a = vector(v, 82)
            require((a >= (0 if name in ('real', 'pair') else -1)).all()
                    and (a <= 1).all(), 'behavior range')
            point = sum(v)/len(v)
            boot = python_bootstrap(v, 20260729 if name == 'pair' else 20260722)
            cells[name] = {'estimate': point,
                           'ci95': [python_quantile(boot, .025), python_quantile(boot, .975)],
                           'one_sided_lower_95': python_quantile(boot, .05)}
        for name, seed in (('blank', 20260733), ('wrong', 20260751), ('pair', 20260765)):
            null = .25 if name == 'pair' else 0.
            effect = cells[name]['estimate']-null
            centered = [v-effect for v in values[name]]
            tails[name] = (1+sum(t-null >= effect-1e-15 for t in python_bootstrap(centered, seed)))/10001
        result = {'cells': cells, 'tails': tails, 'conjunction': max(tails.values())}
        compare(result, references[model])
        compare({k: v['estimate'] for k, v in cells.items()}, data[model]['estimate'])
        compare({k: v['ci95'][0] if k == 'real' else v['one_sided_lower_95']
                 for k, v in cells.items()}, data[model]['lower'])
        compare(tails, data[model]['tails'])
        results[model] = result
    return results


def endpoint_replay(bundle, expected_count, n):
    require(bundle['cluster_ordinal'] == list(range(n)) and len(bundle['cells']) == expected_count,
            'endpoint shape/ordinal inventory')
    results, keys = [], set()
    for cell in bundle['cells']:
        label = cell['label']
        key = tuple(sorted(label.items()))
        require(key not in keys, 'duplicate endpoint')
        keys.add(key)
        idx = indexes(n, cell['seed'])
        v = cell['vector']
        result = {**interval(rho(v, n), rho(v, n, idx)),
                  'mean_error_bits': float(vector(v['error_bits'], n, True).mean()),
                  'mean_gap_bits': float(vector(v['gap_bits'], n, True).mean())}
        compare(result, cell['reference'])
        results.append({'label': label, **result})
    return results


def exploratory_replay(bundle, public):
    require(bundle['raw_outcomes'] == 20480, 'exploratory inventory')
    endpoints = endpoint_replay(bundle, 128, 80)
    require(len(bundle['screen']) == 4, 'four-test family required')
    screens = []
    for cell in bundle['screen']:
        idx = indexes(80, cell['seed'])
        a, b = cell['comparator'], cell['single']
        require(a['gap_bits'] == b['gap_bits'], 'screen pairing')
        boot = rho(a, 80, idx)-rho(b, 80, idx)
        result = {**interval(rho(a, 80)-rho(b, 80), boot),
                  'tail': (1+int(np.count_nonzero(boot <= 0)))/10001,
                  'comparator_rho': rho(a, 80), 'single_rho': rho(b, 80)}
        screens.append({'name': cell['name'], **result})
    adjusted = holm({r['name']: r['tail'] for r in screens})
    for row, cell in zip(screens, bundle['screen']):
        row['holm_tail'] = adjusted[row['name']]
        row['positive'] = bool(row['holm_tail'] < .05 and row['estimate'] >= .1)
        compare({k: v for k, v in row.items() if k != 'name'}, cell['reference'])
    profile_cells, profiles = [], {}
    by_key = {tuple(r['label'][k] for k in ('model_key', 'start_layer', 'operator', 'corruption', 'direction')):
              r['estimate'] for r in endpoints}
    for model, layers in (('qwen', [24, 28, 32, 36]), ('gemma', [22, 26, 30, 34])):
        full_model = next(m for m in MODELS if m.startswith(model))
        profiles[model] = {}
        for op in ('single_full_frame', 'complement_once'):
            profile = []
            for layer in layers:
                ratios = []
                for corruption in ('blank_full', 'wrong_time'):
                    for direction in ('rescue', 'damage'):
                        value = by_key[(full_model, layer, op, corruption, direction)]
                        ratios.append(value)
                        profile_cells.append({'model': model, 'layer': layer, 'operator': op,
                                              'corruption': corruption, 'direction': direction, 'rho': value})
                profile.append(float(np.mean(ratios)))
            profiles[model][op] = profile
    compare(profiles, public['profiles'])
    compare(profile_cells, public['profile_cells'])
    # The clamp identity is checked on input vectors, not inferred from rounded summaries.
    vectors = {tuple(c['label'][k] for k in ('model_key', 'start_layer', 'operator', 'corruption', 'direction')):
               c['vector'] for c in bundle['cells']}
    for key, v in vectors.items():
        if key[2] == 'single_full_frame':
            require(v == vectors[(*key[:2], 'persistent_full_frame', *key[3:])], 'clamp identity')
    return {'endpoints': endpoints, 'screen': screens, 'profiles': profiles,
            'profile_cells': profile_cells, 'clamp_vectors_identical': True}


def sensitivity_replay(cutpair, reference):
    require(cutpair['cluster_index'] == list(range(46)), 'cutpair ordinal inventory')
    v = cutpair['vectors']
    for item in v.values():
        rho(item, 46)
    gap = vector(v['early_F']['gap_bits'], 46, True)
    require(all(item['gap_bits'] == gap.tolist() for item in v.values()), 'cutpair pairing')
    e = {k: vector(v[k]['error_bits'], 46, True) for k in ('early_F', 'early_K', 'late_F', 'late_K')}
    idx = indexes(46, 20260908)
    require(array_digest(idx, '<i8') == reference['family_A']['draw_indices_sha256'], 'sensitivity draw hash')
    cells = {}
    for name, z in (('early22_F_over_K_materiality', e['early_K']-e['early_F']-.1*gap),
                    ('late34_K_over_F_materiality', e['late_F']-e['late_K']-.1*gap)):
        point = float(z.mean())
        boot = z[idx].mean(axis=1)
        dev = boot-point
        count = int(np.count_nonzero(dev >= point))
        cells[name] = {'estimate': point, 'null': 0., 'observed_effect_above_null': point,
                       'descriptive_percentile_ci95': np.quantile(boot, [.025, .975]).tolist(),
                       'basic_one_sided_lower_95': point-float(np.quantile(dev, .95)),
                       'basic_one_sided_lower_97_5': point-float(np.quantile(dev, .975)),
                       'centered_upper_tail_count': count, 'one_sided_centered_add_one_p': (count+1)/10001,
                       'cluster_count': 46, 'draws': 10000}
    adjusted = holm({k: c['one_sided_centered_add_one_p'] for k, c in cells.items()})
    for k, c in cells.items():
        c.update(posthoc_holm_p=adjusted[k], posthoc_holm_below_0_05=adjusted[k] < .05)
    family = {'status': 'POSTHOC', 'family_size': 2, 'multiplicity': 'Holm within this family only',
              'cells': cells, 'draw_indices_sha256': array_digest(idx, '<i8')}
    compare(family, reference['family_A'])
    loco = []
    for i in range(46):
        keep = np.arange(46) != i
        row = {k: float(1-a[keep].mean()/gap[keep].mean()) for k, a in e.items()}
        row['early_F_minus_K'] = row['early_F']-row['early_K']
        row['late_K_minus_F'] = row['late_K']-row['late_F']
        loco.append({'omitted_ordinal': i, **row})
    summaries = {k: [float(np.min([r[k] for r in loco])), float(np.median([r[k] for r in loco])),
                     float(np.max([r[k] for r in loco]))] for k in loco[0] if k != 'omitted_ordinal'}
    compare(summaries, reference['loco_min_median_max'])
    return {'family_A': family, 'loco': loco, 'loco_min_median_max': summaries,
            'loco_reference_scope': 'saved_min_median_max_only_not_historical_per_omission_rows'}


def verify_inputs(directory=DATA, root=ROOT):
    manifest = read(directory/'manifest.json')
    require(manifest['schema_version'] == 1 and set(manifest['files']) == FILES, 'manifest inventory')
    for name, expected in manifest['files'].items():
        require(digest(directory/name) == expected, 'input hash mismatch: ' + name)
    required = {'behavior.json', 'exploratory_depth.json', 'exploratory_endpoints.json',
                'static_endpoints.json', 'gemma_supplementary.json', 'gemma_cutpair.json'}
    require(set(manifest['existing_files']) == required, 'existing dependency inventory')
    for name, expected in manifest['existing_files'].items():
        folder = 'replay' if name == 'gemma_cutpair.json' else 'aggregate'
        require(digest(root/'results'/folder/name) == expected, 'existing input hash mismatch: ' + name)
    return manifest


def run(directory=DATA, root=ROOT):
    verify_inputs(directory, root)
    aggregate = root/'results/aggregate'
    with (directory/'metric_cells.csv').open(newline='') as handle:
        metric = metric_replay(list(csv.DictReader(handle)), read(directory/'metric_reference.json'))
    compare(metric['cells'], read(aggregate/'gemma_supplementary.json')['metric_cells'])
    behavior = behavior_replay(read(aggregate/'behavior.json'), read(directory/'behavior_reference.json'))
    exploratory = exploratory_replay(read(directory/'exploratory.json'), read(aggregate/'exploratory_depth.json'))
    controls = read(directory/'gemma_controls.json')
    require(controls['window_outcomes'] == 4416 and controls['endpoint_outcomes'] == 1472,
            'Gemma outcome inventory')
    windows = endpoint_replay(controls['windows'], 12, 46)
    endpoints = endpoint_replay(controls['endpoints'], 16, 46)
    sensitivity = sensitivity_replay(read(root/'results/replay/gemma_cutpair.json'),
                                     read(directory/'sensitivity_reference.json'))
    return {'schema_version': 1, 'status': 'pass', 'interval_status': 'historical_finite_draw_replay',
            'counts': {'metric_rows': 322, 'metric_strata': 7, 'behavior_models': 2,
                       'behavior_clusters_per_model': 82, 'exploratory_outcomes': 20480,
                       'exploratory_clusters': 80, 'exploratory_endpoints': 128,
                       'exploratory_screen_tests': 4, 'window_cluster_records': 552,
                       'gemma_windows': 12, 'stratified_cluster_records': 736,
                       'gemma_stratified_endpoints': 16, 'sensitivity_tests': 2, 'loco_rows': 46},
            'coverage': {'metric_dependence': 'seven_strata_and_exact_saved_draw_hashes',
                         'behavior': 'original_random_Random_full_intervals_and_tails',
                         'exploratory': 'four_cut_profiles_middle_cut_four_test_family_clamp_full_sequence',
                         'gemma_windows': 'twelve_deterministic_overlapping_selected_windows_not_random_reference',
                         'gemma_stratified_endpoints': 'F_K_by_corruption_direction_at_two_cuts',
                         'sensitivity': 'current_family_A_only',
                         'loco': '46_recomputed_rows_saved_min_median_max_reference_only'},
            'comparison': 'rtol=atol=1e-12; metric draw arrays SHA-256 exact',
            'metric_dependence': metric, 'behavior': behavior, 'exploratory': exploratory,
            'gemma_windows': windows, 'gemma_stratified_endpoints': endpoints,
            'sensitivity': sensitivity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check without writing the rebuilt report')
    parser.add_argument('--output', type=Path, default=ROOT/'results/rebuilt/static_supplement_replay.json')
    args = parser.parse_args()
    try:
        result = run()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result = {'schema_version': 1, 'status': 'fail', 'counts': {}, 'coverage': {},
                  'error_type': type(exc).__name__}
    if not args.check:
        destination = args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'counts': result['counts']}))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
