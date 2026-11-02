#!/usr/bin/env python3
"""Offline replay from anonymous cluster sufficient statistics; requires NumPy."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'results' / 'replay'
SUBSETS = [''.join(c for i, c in enumerate('CQPR') if m & (1 << i))
           for m in range(16)]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def summary(point, draws):
    return {'estimate': float(point),
            'ci95': np.quantile(draws, [.025, .975], method='linear').tolist()}


def shapley(values):
    result = np.zeros((*values.shape[:-1], 4))
    for player in range(4):
        for mask in range(16):
            if not mask & (1 << player):
                n = mask.bit_count()
                weight = math.factorial(n) * math.factorial(3-n) / 24
                result[..., player] += weight * (values[..., mask | (1 << player)] - values[..., mask])
    return result


def mobius(values):
    result = values.copy()
    for bit in range(4):
        for mask in range(16):
            if mask & (1 << bit):
                result[..., mask] -= result[..., mask ^ (1 << bit)]
    return result


def analyze(data, seed_override=None):
    require(data['cluster_index'] == list(range(46)), 'ordinal order was modified')
    require(data['rows_per_cluster'] == 8, 'wrong within-cluster inventory')
    expected_ops = ({e+'_'+o for e in ('early','late') for o in ('F','K','A')}
                    if data['kind'].endswith('cutpair') else set(SUBSETS[1:]))
    if data['kind'] == 'qwen_factorial':
        expected_ops |= {'F','A'}
    require(set(data['vectors']) == expected_ops, 'incomplete operator inventory')
    gaps = [v['gap_bits'] for v in data['vectors'].values()]
    require(all(np.array_equal(gaps[0], g) for g in gaps), 'unpaired denominators')
    seed = data['seed'] if seed_override is None else seed_override
    indexes = np.random.default_rng(seed).integers(0, 46, size=(data['draws'], 46))
    points, boot, raw = {}, {}, {}
    for name, vector in data['vectors'].items():
        error = np.asarray(vector['error_bits'], dtype=np.float64)
        baseline = np.asarray(vector['gap_bits'], dtype=np.float64)
        require(error.shape == baseline.shape == (46,), 'bad vector shape')
        require(np.all(np.isfinite(error)) and np.all(np.isfinite(baseline)), 'nonfinite vector')
        require(np.all(error >= 0) and np.all(baseline >= 0), 'negative absolute discrepancy')
        denom = baseline[indexes].mean(axis=1)
        require(baseline.mean() > 1e-12 and np.all(denom > 1e-12), 'donor-gap guard failed')
        if data['kind'] != 'qwen_factorial':
            require(np.all(baseline > 1e-12), 'Job E individual-gap guard failed')
        points[name] = 1 - float(error.mean()) / float(baseline.mean())
        boot[name] = 1 - error[indexes].mean(axis=1) / denom
        raw[name] = {'error_bits': float(error.mean()), 'gap_bits': float(baseline.mean())}
    result = {'seed': seed, 'draws': data['draws'], 'raw_means': raw,
              'interval_status': 'historical_finite_draw_replay' if seed_override is None else 'independently_regenerated',
              'operators': {k: summary(v, boot[k]) for k, v in points.items()}}
    if data['kind'].endswith('cutpair'):
        ds = {e: points[e+'_K'] - points[e+'_F'] for e in ('early', 'late')}
        bs = {e: boot[e+'_K'] - boot[e+'_F'] for e in ('early', 'late')}
        result['contrasts'] = {e: summary(ds[e], bs[e]) for e in ds}
        result['cross_cut'] = summary(ds['late']-ds['early'], bs['late']-bs['early'])
        if data['kind'] == 'gemma_cutpair':
            tails = {'late': (1 + int(np.count_nonzero(bs['late'] <= .1))) / (data['draws']+1),
                     'early': (1 + int(np.count_nonzero(bs['early'] >= -.1))) / (data['draws']+1)}
            adjusted, running = {}, 0.
            for i, e in enumerate(sorted(tails, key=lambda e: (tails[e], 0 if e == 'late' else 1))):
                running = max(running, min(1., (2-i)*tails[e]))
                adjusted[e] = running
            result['primary'] = {e: {'tail': tails[e], 'holm_tail': adjusted[e],
                                     'passes': adjusted[e] < .05} for e in tails}
            result['holm_lower_bounds'] = {'early': float(np.quantile(-bs['early'], .05)),
                                           'late': float(np.quantile(bs['late'], .025))}
    else:
        values = np.asarray([0.] + [points[s] for s in SUBSETS[1:]])
        samples = np.column_stack([np.zeros(data['draws'])] + [boot[s] for s in SUBSETS[1:]])
        phi, phi_b = shapley(values), shapley(samples)
        mu, mu_b = mobius(values), mobius(samples)
        require(np.max(abs(phi_b.sum(axis=1)-samples[:,15])) < 1e-12, 'Shapley efficiency')
        for m in range(16):
            require(abs(sum(mu[t] for t in range(16) if t & m == t)-values[m]) < 1e-12, 'Mobius inversion')
        result['shapley'] = {c: {**summary(phi[i], phi_b[:,i]), 'share': float(phi[i]/values[15])}
                             for i, c in enumerate('CQPR')}
        result['mobius'] = {s: summary(mu[m], mu_b[:,m]) for m, s in enumerate(SUBSETS) if m}
        result['efficiency_residual'] = float(phi.sum()-values[15])
        if data['kind'] == 'qwen_factorial':
            guard = float(np.quantile(boot['CQPR'], .025)) > .1
            result['K_guard'] = guard
            contrasts = {'noncandidate_below_75pct_K': (.75*points['CQPR']-points['QPR'], .75*boot['CQPR']-boot['QPR']),
                         'prefix_excluded_above_75pct_K': (points['CQR']-.75*points['CQPR'], boot['CQR']-.75*boot['CQPR'])}
            result['primary'] = {k: {**summary(p, b), 'lower_97_5': float(np.quantile(b,.025)),
                                     'supported': bool(guard and np.quantile(b,.025) > 0)} for k,(p,b) in contrasts.items()}
            valid = guard and points['CQPR'] > .1 and np.all(boot['CQPR'] > .1)
            result['fractions'] = {
                'CQ_recovery_fraction': summary(points['CQ']/points['CQPR'], boot['CQ']/boot['CQPR']) if valid else None,
                'CQ_shapley_fraction': summary((phi[0]+phi[1])/values[15], (phi_b[:,0]+phi_b[:,1])/boot['CQPR']) if valid else None}
    return result


def compare(actual, expected, prefix=''):
    errors = []
    for key, value in expected.items():
        path = prefix + '/' + key
        if isinstance(value, dict):
            errors.extend(compare(actual[key], value, path))
        elif isinstance(value, (float, int, list)) and not isinstance(value, bool):
            delta = float(np.max(np.abs(np.asarray(actual[key])-np.asarray(value))))
            if delta > 2e-12:
                errors.append({'quantity': path, 'actual': actual[key], 'saved': value, 'max_abs_difference': delta})
        elif actual[key] != value:
            errors.append({'quantity': path, 'actual': actual[key], 'saved': value})
    return errors


def main():
    if hasattr(signal, 'alarm'):
        signal.alarm(180)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, help='Independent sensitivity intervals, NOT historical replay')
    args = parser.parse_args()
    inventory = json.loads((DATA/'legacy_coverage.json').read_text())
    results, differences = {}, []
    for name, metadata in inventory['datasets'].items():
        path = DATA/metadata['file']
        require(hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256'], 'data hash mismatch')
        results[name] = analyze(json.loads(path.read_text()), args.seed)
    if args.seed is None:
        reference = DATA/'expected.json'
        require(hashlib.sha256(reference.read_bytes()).hexdigest() == inventory['expected_sha256'], 'reference hash mismatch')
        expected = json.loads(reference.read_text())
        differences = compare(results, expected)
    report = {'numpy_version': np.__version__, 'comparison_tolerance': 2e-12,
              'historical_comparison_performed': args.seed is None,
              'discrepancies': differences, 'results': results}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'datasets': len(results), 'historical_comparison_performed': args.seed is None,
                      'discrepancies': differences}, indent=2))
    return bool(differences)


if __name__ == '__main__':
    raise SystemExit(main())
