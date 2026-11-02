#!/usr/bin/env python3
"""Curator-only allowlisted export; requires an explicit read-only upstream root."""
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
from statistics import mean
import zipfile

from replay_static_supplement import (
    DATA, ROOT, MODELS, array_digest, compare, digest, read, require, run,
)
from export_aggregates import IDENTITY_KEY, PRIVATE
import numpy as np

ARCHIVE = ('papers/patchaudit_final_revision_2026_09_08/packaging/'
           'patchaudit_anonymous_scientific_supplement_2026_09_08_v2.zip')
PHASE2 = 'runs/research/lifecycle_memory/phase_2'
AUDIT = ('papers/scene_to_task_roles_revision_cycle_2026_10_18/audits/'
         'number_provenance_2026_10_18/depth_statistics')
LABELS = ('model_key', 'start_layer', 'operator', 'corruption', 'direction')


def privacy(value):
    if isinstance(value, dict):
        for key, item in value.items():
            require(not IDENTITY_KEY.search(key) and not PRIVATE.search(key), 'private export key')
            privacy(item)
    elif isinstance(value, list):
        for item in value:
            privacy(item)
    elif isinstance(value, str):
        require(not PRIVATE.search(value), 'private export value')


def write_json(path, value):
    privacy(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_outcomes(upstream, run_folder, hidden_path, cluster_field, expected_count, n):
    hidden_rows = rows(upstream/hidden_path)
    hidden = {r['request_id']: r for r in hidden_rows}
    require(len(hidden) == len(hidden_rows), 'duplicate evaluator key')
    result, seen = [], set()
    files = sorted((upstream/run_folder).glob('rank_*/outcomes.jsonl'))
    require(len(files) == 4, 'expected four outcome shards')
    for path in files:
        for r in rows(path):
            key = tuple(r[k] for k in (*LABELS, 'request_id'))
            require(key not in seen, 'duplicate outcome')
            seen.add(key)
            h = hidden[r['request_id']]
            require(r['order_id'] == h['order_id'], 'counterbalance join mismatch')
            error = abs(r['patched_margin_a_minus_b_bits']-r['donor_margin_a_minus_b_bits'])
            gap = abs(r['recipient_margin_a_minus_b_bits']-r['donor_margin_a_minus_b_bits'])
            compare(error, r['patched_donor_error_abs_bits'])
            compare(gap, r['baseline_donor_gap_abs_bits'])
            # Identities remain in memory only, to recover the historical sorted order.
            result.append({**{k: r[k] for k in LABELS}, 'cluster': h[cluster_field],
                           'frame': r['frame_unit_id'], 'error': error, 'gap': gap})
    require(len(result) == expected_count, 'raw outcome count')
    clusters = sorted({r['cluster'] for r in result})
    require(len(clusters) == n, 'cluster count')
    require(all(len({r['frame'] for r in result if r['cluster'] == c}) == 1 for c in clusters),
            'one-frame-per-cluster assumption')
    return result, clusters, [digest(p) for p in files]


def project_vector(outcomes, clusters, selector, count):
    grouped = defaultdict(list)
    for row in outcomes:
        if all(row[k] in v if isinstance(v, list) else row[k] == v for k, v in selector.items()):
            grouped[row['cluster']].append(row)
    require(sorted(grouped) == clusters and all(len(grouped[c]) == count for c in clusters),
            'selected within-cluster inventory')
    return {'error_bits': [mean(r['error'] for r in grouped[c]) for c in clusters],
            'gap_bits': [mean(r['gap'] for r in grouped[c]) for c in clusters]}


def endpoint_cell(saved, outcomes, clusters, count, label=None):
    selector = label if label is not None else {k: saved[k] for k in LABELS}
    return {'label': selector, 'seed': saved['bootstrap_seed'], 'rows_per_cluster': count,
            'vector': project_vector(outcomes, clusters, selector, count),
            'reference': {'estimate': saved['rho'], 'ci95': saved['ci95'],
                          'mean_error_bits': saved['cluster_equal_mean_patched_donor_error_bits'],
                          'mean_gap_bits': saved['cluster_equal_mean_recipient_donor_gap_bits']}}


def export(upstream):
    upstream = upstream.resolve(strict=True)
    require(upstream != ROOT, 'upstream must be separate from the public checkout')
    DATA.mkdir(parents=True, exist_ok=True)
    source_receipts = {}
    # Copy only the already-anonymous CSV; never extract the archive wholesale.
    with zipfile.ZipFile(upstream/ARCHIVE) as archive:
        base = 'analysis/posthoc_metric_dependence/'
        raw = archive.read(base+'cluster_cells.csv')
        metric_rows = list(csv.DictReader(io.StringIO(raw.decode())))
        require(len(metric_rows) == 322, 'metric row count')
        for row in metric_rows:
            privacy(row)
        saved = json.loads(archive.read(base+'results.json'))
        with np.load(io.BytesIO(archive.read(base+'paired_bootstrap.npz')), allow_pickle=False) as draws:
            require(draws['cluster_indices'].shape == (10000, 46)
                    and draws['paired_interactions'].shape == (10000, 7, 2), 'saved metric draw shape')
            require(array_digest(draws['cluster_indices'], '<i8') ==
                    saved['bootstrap']['index_sha256_little_endian_int64'], 'saved metric index hash')
            metric_reference = {'bootstrap': saved['bootstrap'], 'cells': saved['cells'],
                                'paired_draws_sha256': array_digest(draws['paired_interactions'], '<f8')}
        compare(saved['cells'], read(ROOT/'results/aggregate/gemma_supplementary.json')['metric_cells'])
        (DATA/'metric_cells.csv').write_bytes(raw)
        write_json(DATA/'metric_reference.json', metric_reference)
        sensitivity = json.loads(archive.read('analysis/posthoc_sensitivity/results.json'))['family_A']
        archived_vectors = json.loads(archive.read('analysis/posthoc_sensitivity/cluster_sufficient_statistics.json'))['family_A']
        current = read(ROOT/'results/replay/gemma_cutpair.json')['vectors']
        for column, op in (('dF22', 'early_F'), ('dK22', 'early_K'),
                           ('dF34', 'late_F'), ('dK34', 'late_K'), ('d0', 'early_F')):
            compare([r[column] for r in archived_vectors['rows']],
                    current[op]['gap_bits' if column == 'd0' else 'error_bits'])
    source_receipts['metric_archive_sha256'] = digest(upstream/ARCHIVE)

    # Full intervals are taken from the original saved analysis, not the new replay.
    behavior_path = upstream/'runs/tamia/reanchored-jobAR-token-full/380145/analysis/analysis_summary.json'
    behavior = {}
    for m in read(behavior_path)['models']:
        model = m['model_key']
        require(model in MODELS, 'unexpected behavioral model')
        cells = {}
        for short, key in (('real', 'real_capability'), ('pair', 'primary_counterbalanced_pair_correctness')):
            cells[short] = {k: m[key][k] for k in ('estimate', 'ci95', 'one_sided_lower_95')}
        effect = m['primary_intersection_evidence_effect']
        for short, key in (('blank', 'real_minus_blank_full'), ('wrong', 'real_minus_wrong_time')):
            cells[short] = {'estimate': effect[key], 'ci95': effect['component_ci95'][key],
                            'one_sided_lower_95': effect['component_one_sided_lower_95'][key]}
        tests = m['confirmatory_centered_cluster_bootstrap']
        tails = {short: tests[key] for short, key in
                 (('blank', 'real_minus_blank_full_p'), ('wrong', 'real_minus_wrong_time_p'),
                  ('pair', 'counterbalanced_pair_correctness_above_paired_chance_p'))}
        behavior[model] = {'cells': cells, 'tails': tails, 'conjunction': tests['model_conjunctive_p']}
    write_json(DATA/'behavior_reference.json', behavior)
    source_receipts['behavior_analysis_sha256'] = digest(behavior_path)

    droot = 'runs/tamia/patchaudit-jobD-transport/383819'
    dhidden = PHASE2+'/reanchored_answer_token_battery_v2_2026_07_22/mechanism/hidden/mechanism_evaluator_index.jsonl'
    drows, dclusters, dhashes = load_outcomes(upstream, droot, dhidden, 'ego4d_participant_id', 20480, 80)
    dpath = upstream/droot/'analysis/transport_analysis.json'
    dsaved = read(dpath)
    public = read(ROOT/'results/aggregate/exploratory_endpoints.json')
    compare(dsaved['endpoint_summaries'], public['endpoint_summaries'])
    compare(dsaved['primary_holm_family'], public['primary_holm_family'])
    exploratory = {'cluster_ordinal': list(range(80)), 'raw_outcomes': 20480,
                   'cells': [endpoint_cell(s, drows, dclusters, 2) for s in dsaved['endpoint_summaries']],
                   'screen': []}
    for c in dsaved['primary_holm_family']['cells']:
        selector = {'model_key': c['model_key'], 'start_layer': c['central_post_block_indexes']}
        exploratory['screen'].append({
            'name': c['cell_id'], 'seed': c['bootstrap_seed'], 'rows_per_cluster': 16,
            'comparator': project_vector(drows, dclusters, {**selector, 'operator': c['comparator_operator']}, 16),
            'single': project_vector(drows, dclusters, {**selector, 'operator': c['reference_operator']}, 16),
            'reference': {'estimate': c['estimate_delta_rho'], 'ci95': c['ci95'],
                          'tail': c['one_sided_bootstrap_pvalue'], 'holm_tail': c['holm_adjusted_pvalue'],
                          'comparator_rho': c['comparator_rho'], 'single_rho': c['single_rho'],
                          'positive': c['positive']}})
    write_json(DATA/'exploratory.json', exploratory)
    source_receipts['exploratory_analysis_sha256'] = digest(dpath)
    source_receipts['exploratory_outcome_sha256'] = dhashes

    eroot = 'runs/tamia/patchaudit-jobE-component-closure/383933'
    ehidden = PHASE2+'/reanchored_egohos_battery_v3_2026_07_21/mechanism/hidden/mechanism_evaluator_index.jsonl'
    # Count is the complete saved run, although only the selected numerical cells are exported.
    epath = upstream/eroot/'analysis/component_closure_analysis.json'
    esaved = read(epath)
    erows, eclusters, ehashes = load_outcomes(upstream, eroot, ehidden, 'participant_cluster_id',
                                           esaved['raw_outcome_row_count'], 46)
    public = read(ROOT/'results/aggregate/static_endpoints.json')
    compare(esaved['cardinality_benchmarks'], public['cardinality_benchmarks'])
    compare(esaved['endpoint_summaries'], public['endpoint_summaries'])
    selected = [s for s in esaved['endpoint_summaries'] if s['model_key'] == MODELS[0]
                and s['operator'] in ('F', 'K', 'CQPR')]
    benchmarks = esaved['cardinality_benchmarks']
    controls = {'window_outcomes': 4416, 'endpoint_outcomes': 1472,
                'windows': {'cluster_ordinal': list(range(46)), 'cells': [
                    endpoint_cell(s, erows, eclusters, 8, {'model_key': benchmarks['model_key'],
                                  'start_layer': benchmarks['start_layer'], 'operator': s['operator']})
                    for s in benchmarks['cells']]},
                'endpoints': {'cluster_ordinal': list(range(46)),
                              'cells': [endpoint_cell(s, erows, eclusters, 2) for s in selected]}}
    write_json(DATA/'gemma_controls.json', controls)
    source_receipts['gemma_analysis_sha256'] = digest(epath)
    source_receipts['gemma_outcome_sha256'] = ehashes

    checks_path = upstream/AUDIT/'checks.json'
    checks = read(checks_path)
    if isinstance(checks, dict):
        checks = checks['checks']
    checks = {c['check_id']: c for c in checks}
    loco = {op: [float(checks['loco_'+op+'_'+str(i)]['expected_values'][0]) for i in (1, 2, 3)]
            for op in ('early_F', 'early_K', 'late_F', 'late_K', 'early_F_minus_K', 'late_K_minus_F')}
    write_json(DATA/'sensitivity_reference.json', {'family_A': sensitivity, 'loco_min_median_max': loco})
    source_receipts['loco_saved_audit_sha256'] = digest(checks_path)
    existing = {p.name: digest(p) for p in [ROOT/'results/aggregate'/name for name in
                ('behavior.json', 'exploratory_depth.json', 'exploratory_endpoints.json',
                 'static_endpoints.json', 'gemma_supplementary.json')] + [ROOT/'results/replay/gemma_cutpair.json']}
    names = ('metric_cells.csv', 'metric_reference.json', 'behavior_reference.json',
             'exploratory.json', 'gemma_controls.json', 'sensitivity_reference.json')
    manifest = {'schema_version': 1, 'files': {name: digest(DATA/name) for name in names},
                'existing_files': existing, 'source_receipts': source_receipts,
                'ordering': 'Historical analysis order before anonymous ordinal assignment; no identity mapping exported.',
                'counts': {'metric_rows': 322, 'metric_strata': 7, 'exploratory_outcomes': 20480,
                           'exploratory_clusters': 80, 'window_cluster_records': 552,
                           'endpoint_cluster_records': 736}}
    write_json(DATA/'manifest.json', manifest)
    result = run()
    write_json(ROOT/'results/rebuilt/static_supplement_replay.json', result)
    print(json.dumps({'status': 'pass', 'exported_files': len(names)+1}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path)
    export(parser.parse_args().upstream)
