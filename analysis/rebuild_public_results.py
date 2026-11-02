#!/usr/bin/env python3
"""Offline CPU rebuild: exact static replay plus explicitly aggregate-only tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import signal

from replay_cluster_sufficient_stats import analyze, compare, np, require
from export_aggregates import scalar_count
from reproduce_all import PACKAGES, verified_receipt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
AGGREGATE = 'aggregate_only_not_exact_ci_replay'
EXACT = 'historical_finite_draw_replay_252_reference_values'
REPLAY_NAMES = ('gemma_cutpair', 'qwen_cutpair', 'gemma_factorial', 'qwen_factorial')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def csv_write(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def numeric_rows(value, pointer=''):
    if isinstance(value, dict):
        for key, item in value.items():
            key = key.replace('~', '~0').replace('/', '~1')
            yield from numeric_rows(item, pointer + '/' + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from numeric_rows(item, pointer + '/' + str(index))
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        yield {'json_pointer': pointer, 'value': value}


# Main numbering follows the reviewed manuscript identified by paper_reference.json.
# Supplement labels remain stable. A mapping can have several evidence files.
MAP = [
    ('Table 1', 'tab:instantiation', 'behavioral', 'behavior', 'Model geometry and group counts'),
    ('Section 5.1; Table S21', 'tab:behavioral-contract', 'behavioral', 'behavior', 'Real accuracy; blank/wrong are paired gains, not control accuracy'),
    ('Figure 3', 'fig:transport', 'exploratory_depth', 'exploratory_depth', 'Four-cut curves; averages of four corruption/direction recoveries, not pooled recovery'),
    ('Table S12', 'tab:exploratory-primary-family', 'exploratory_depth', 'exploratory_depth', 'Four-test exploratory screen, including null clamp results'),
    ('Table 2; Figure 4', 'tab:jobe-cuts;fig:headline', 'cutpair', 'gemma_cutpair;qwen_cutpair', 'Fixed-cut operators, contrasts and Gemma Holm family'),
    ('Table 3; Table S8', 'tab:clip-behavior;tab:clip-behavior-full', 'temporal_current', 'temporal_current', 'Current object behavior and contrasts; the clip-minus-shuffled contrast is also replayed in the temporal package'),
    ('Table 3; Table S8', 'tab:clip-behavior;tab:clip-behavior-full', 'temporal_onset_order', 'temporal_recognition;temporal_followup', 'New interaction and interaction order behavior; altered order is original-answer agreement; the paired clip-minus-shuffled and clip-minus-reversed contrasts are also replayed in the temporal package'),
    ('Figure 5; Table S9', 'fig:evidence-over-time;tab:clip-patching-full', 'temporal_current', 'temporal_current', 'H/V/K/A at L22 and L34 in both directions'),
    ('Figure 5; Table S9', 'fig:evidence-over-time;tab:clip-patching-full', 'temporal_onset_order', 'temporal_recognition', 'H/V/K/A at L22 and L34 in both directions'),
    ('Table 4; Figure 6; Table S1', 'tab:parallel-role-comparisons;fig:factorial-combinations;tab:factorial-complete', 'gemma_factorial', 'gemma_factorial', 'All 15 nonempty C/Q/P/R unions'),
    ('Table 4; Figure 6; Table S4', 'tab:parallel-role-comparisons;fig:factorial-combinations;tab:qwen-subsets', 'qwen_factorial', 'qwen_factorial', 'All 15 unions and F/A controls; planned contrasts'),
    ('Table 5; Table S2; Table S3', 'tab:shapley-comparison;tab:shapley-allocation;tab:mobius', 'gemma_factorial', 'gemma_factorial', 'Shapley allocations/shares and Mobius interactions'),
    ('Table 5; Tables S5-S7', 'tab:shapley-comparison;tab:qwen-shapley;tab:qwen-mobius;tab:qwen-fractions', 'qwen_factorial', 'qwen_factorial', 'Shapley, Mobius and recovery/allocation fractions'),
    ('Figure 7; Table S10', 'fig:clip-roles;tab:clip-factorial-full', 'temporal_followup', 'temporal_followup', 'L34 15 unions, both directions, paired differences, interactions and Shapley; both tasks'),
    ('Table 6; Figure 8; Tables S25-S28', 'tab:application-main;fig:practical-tradeoffs;tab:killarney-decisions;tab:killarney-sources;tab:killarney-complete;tab:killarney-followup-complete', 'practical_removal_trained_and_scene_only', 'practical_final', 'Removal-trained and scene-only E4 sessions; each has its own native baseline; individual policies distinct from seed means'),
    ('Tables S14-S16', 'tab:raw-discrepancy-components;tab:baseline-discrepancies;tab:corruption-direction-recovery', 'cutpair', 'static_endpoints', 'Endpoint mean gaps/residuals and corruption/direction recovery; raw-component intervals regenerated separately, not among 252 checked values'),
    ('Table S19; Section 5.3', 'tab:metric-dependence', 'gemma_factorial', 'gemma_supplementary', 'Metric dependence, cardinality controls and CQ/K ratio'),
    ('Tables S17-S18; S20; S23', 'tab:dataset-provenance;tab:prompt-positions;tab:group-lineage;tab:confirmation-overlap', 'behavioral', 'behavior', 'Aggregate counts, token geometry and group overlaps; no identities'),
    ('Table S22', 'tab:freeze-chronology', 'cutpair', 'legacy_coverage;static_endpoints', 'Confirmation chronology counts: 13,984 outcomes and 1,472 full-active controls; documentary counts, not a numeric result reconstruction'),
    ('Table S24', 'tab:posthoc-sensitivity', 'exploratory_depth', 'exploratory_depth', 'Post hoc sensitivity on confirmation cohorts; see sensitivity rather than exploratory profiles'),
    ('Tables S30-S32', 'tab:application-all;tab:application-means;tab:application-sources', 'practical_earlier_benchmark', 'practical_compression', 'Earlier E4 benchmark session: deletion depth, compression and correction conditions, and seed means'),
    ('Tables S33-S34', 'tab:frozen-bias-levels;tab:frozen-bias-contrasts', 'practical_earlier_benchmark', 'bias_baseline', 'Post hoc answer-bias offset comparisons; not confirmation'),
    ('Table S29; Section S14.2', 'tab:application-cost-boundary', 'practical_earlier_benchmark', 'practical_followup_public', 'Documentary cost-measurement boundary and saved setup records; not an independently replayed cost benchmark'),
    ('Table S35; Section S14.4', 'tab:application-ties', 'practical_earlier_benchmark', 'practical_followup_public', 'Accuracy tie sensitivity for the earlier benchmark session'),
    ('Table S36', 'tab:reporting-intervals', 'reporting_intervals', 'reporting_intervals', 'Saved post hoc intervals, with private unit lists removed'),
    ('Table S13', 'tab:loco', 'cutpair', 'gemma_cutpair', 'LOCO min/median/max generated from anonymous vectors; not in historical 252-value comparison'),
]

# Only these paper items have statistics covered by the additional replayers.
ADDITIONAL_REPLAY = {
    'Section 5.1; Table S21': 'static_supplement',
    'Figure 3': 'static_supplement',
    'Table S12': 'static_supplement',
    'Table 3; Table S8': 'temporal',
    'Figure 5; Table S9': 'temporal',
    'Figure 7; Table S10': 'temporal',
    'Table 6; Figure 8; Tables S25-S28': 'practical',
    'Tables S14-S16': 'static_supplement',
    'Table S19; Section 5.3': 'static_supplement',
    'Table S24': 'static_supplement',
    'Tables S30-S32': 'practical',
    'Table S13': 'static_supplement',
}


def verify_inputs():
    legacy = read(RESULTS / 'replay/legacy_coverage.json')
    actual = {}
    for name in REPLAY_NAMES:
        meta = legacy['datasets'][name]
        path = RESULTS / 'replay' / meta['file']
        require(sha(path) == meta['sha256'], 'Replay input hash mismatch: ' + name)
        actual[name] = analyze(read(path))
    expected_path = RESULTS / 'replay/expected.json'
    require(sha(expected_path) == legacy['expected_sha256'], 'Expected reference hash mismatch')
    expected = read(expected_path)
    require(scalar_count(expected) == 252, 'Unexpected static reference inventory')
    differences = compare(actual, expected)
    require(not differences, 'Historical replay mismatch: ' + str(differences))
    receipts = read(RESULTS / 'source_receipts.json')['exports']
    for receipt in receipts:
        path = ROOT / receipt['file']
        require(path.is_file() and sha(path) == receipt['sha256'], 'Aggregate hash mismatch: ' + receipt['name'])
    return legacy, actual, receipts


def diagnostics():
    d = read(RESULTS / 'replay/gemma_cutpair.json')
    rng = np.random.default_rng(d['seed'])
    indices = rng.integers(0, 46, size=(d['draws'], 46))
    gap = np.array(d['vectors']['early_F']['gap_bits'])
    gb = gap[indices].mean(axis=1)
    loco, components = {}, {}
    for name in ('early_F', 'early_K', 'late_F', 'late_K'):
        error = np.array(d['vectors'][name]['error_bits'])
        loco[name] = np.array([1 - np.delete(error, i).mean() / np.delete(gap, i).mean() for i in range(46)])
        eb = error[indices].mean(axis=1)
        components[name] = {'residual_bits': float(error.mean()), 'residual_ci95': np.quantile(eb, [.025, .975]).tolist(),
                            'reduction_bits': float(gap.mean()-error.mean()), 'reduction_ci95': np.quantile(gb-eb, [.025, .975]).tolist()}
    loco['early_F_minus_K'] = loco['early_F'] - loco['early_K']
    loco['late_K_minus_F'] = loco['late_K'] - loco['late_F']
    return {'status': 'derived_from_static_vectors_not_in_252_reference_check',
            'loco_spread': {k: {'min': float(v.min()), 'median': float(np.median(v)), 'max': float(v.max())} for k,v in loco.items()},
            'raw_components': components,
            'baseline': {'n_clusters': 46, 'min': float(gap.min()), 'median': float(np.median(gap)),
                         'max': float(gap.max()), 'range': float(np.ptp(gap)), 'mean': float(gap.mean()),
                         'ci95': np.quantile(gb, [.025, .975]).tolist()}}


def figure_data(data, replay):
    out = {}
    temporal_ci_status = 'historical_participant_vector_replay' if verified_receipt() else AGGREGATE
    out['figure3'] = [dict(r, interval_status='no_intervals_in_paper') for r in data['exploratory_depth']['profile_cells']]
    out['figure4'] = [dict(model=m, operator=k, estimate=v['estimate'], ci_low=v['ci95'][0], ci_high=v['ci95'][1], interval_status=EXACT)
                      for m in ('gemma', 'qwen') for k,v in replay[m+'_cutpair']['operators'].items()]
    profiles = {'current_state': data['temporal_current']['patch_profiles']}
    profiles.update({k: v['patch_profiles'] for k,v in data['temporal_recognition']['tasks'].items()})
    out['figure5'] = [dict(task=t, direction=r['direction'], cut=r['cut'], position_set=r['position_set'],
                         estimate=r['recovery']['estimate'], ci_low=r['recovery']['ci95'][0], ci_high=r['recovery']['ci95'][1], interval_status=temporal_ci_status)
                      for t, rows in profiles.items() for r in rows]
    out['figure6'] = [dict(model=m, subset=k, estimate=v['estimate'], ci_low=v['ci95'][0], ci_high=v['ci95'][1], interval_status=EXACT)
                      for m in ('gemma', 'qwen') for k,v in replay[m+'_factorial']['operators'].items() if k not in ('F', 'A')]
    out['figure7'] = []
    for task, analysis in data['temporal_followup']['analyses'].items():
        for f in analysis['factorial']:
            if f['direction'] != 'B0->B4':
                continue
            for row in f['subsets']:
                if row['position_set'] in ('C+Q+P+R', 'Q+P+R', 'C+Q+R'):
                    r = row['recovery']
                    out['figure7'].append(dict(task=task, direction=f['direction'], cut=f['cut'], position_set=row['position_set'],
                                               estimate=r['estimate'], ci_low=r['ci95'][0], ci_high=r['ci95'][1], interval_status=temporal_ci_status))
    out['figure8'] = []
    for session, d in data['practical_final']['benchmarks'].items():
        for name, metric in d['metrics'].items():
            timing = d['timing'].get(name)
            out['figure8'].append(dict(session=session, condition=name, accuracy_percent=100*metric['accuracy']['estimate'],
                                      warm_request_ms=timing['request_ms'] if timing else None, native_request_ms=d['timing']['native']['request_ms'],
                                      request_percent_native=100*timing['request_ms']/d['timing']['native']['request_ms'] if timing else None,
                                      timing_status='measured' if timing else 'not_available_in_source',
                                      interval_status='point_estimates_saved_aggregates'))
    require(len(out['figure7']) == 6, 'Figure 7 must have exactly six bars')
    return out


def rebuild(output):
    legacy, replay, receipts = verify_inputs()
    additional = verified_receipt()
    data = {r['name']: read(ROOT/r['file']) for r in receipts}
    dump(output / 'static_replay.json', {'numeric_reference_scalars': 252, 'discrepancies': [], 'results': replay})
    dump(output / 'static_diagnostics.json', diagnostics())
    for receipt in receipts:
        csv_write(output / (receipt['name'] + '_numbers.csv'), list(numeric_rows(data[receipt['name']])))
    figures = figure_data(data, replay)
    for name, rows in figures.items():
        csv_write(ROOT / 'figures' / (name + '_data.csv'), rows)
    dump(ROOT / 'figures/figure7_values.json', {
        'source_sha256': next(r['sources'][0]['sha256'] for r in receipts if r['name']=='temporal_followup'),
        'interval_status': figures['figure7'][0]['interval_status'], 'plotted': figures['figure7']})
    mapping = []
    by_name = {r['name']: r for r in receipts}
    for paper_item, label, family, names, note in MAP:
        files, hashes = [], []
        all_replay = True
        for name in names.split(';'):
            if name in REPLAY_NAMES:
                files.append('results/replay/' + name + '.json')
                hashes.extend(legacy['source_sha256'].values())
            elif name == 'legacy_coverage':
                all_replay = False
                files.append('results/replay/legacy_coverage.json')
                hashes.append(legacy['source_sha256']['job_e_analysis'])
            else:
                all_replay = False
                files.append(by_name[name]['file'])
                hashes.extend(x['sha256'] for x in by_name[name]['sources'])
        status = EXACT if all_replay else AGGREGATE
        if paper_item == 'Table S13':
            status = 'derived_static_loco_not_in_252_reference_check'
        package = ADDITIONAL_REPLAY.get(paper_item)
        mapping.append(dict(paper_item=paper_item, label=label, family=family,
                            evidence_files=';'.join(files), source_sha256=';'.join(sorted(set(hashes))),
                            interval_status=status, note=note,
                            additional_replay_report=PACKAGES[package]['report'] if package else '',
                            additional_replay_guide=PACKAGES[package]['guide'] if package else '',
                            additional_replay_status=('verified_selected_statistics' if additional else 'run_reproduce_all') if package else ''))
    csv_write(RESULTS / 'paper_results_map.csv', mapping)
    families = {}
    for row in mapping:
        families.setdefault(row['family'], []).append(row['paper_item'])
    coverage = {
        'schema_version': 2, 'scope': 'all_current_numeric_experiment_families; not every manuscript claim',
        'current_paper_reference': read(RESULTS / 'paper_reference.json'),
        'paper_numbering': 'Main and supplemental numbering follows the reviewed manuscript reference.',
        'families': families, 'aggregate_exports': receipts,
        'replay': {'datasets': list(REPLAY_NAMES), 'numeric_reference_scalars': 252,
                   'expected_sha256': legacy['expected_sha256'], 'tolerance': 2e-12, 'discrepancies': [],
                   'ci_status': EXACT, 'anonymous_clusters_per_dataset': 46,
                   'scope_note': 'These four original static datasets retain the 252-value reference check. Additional anonymous inputs and replay coverage are listed separately.'},
        'additional_replay': {
            'status': 'verified' if additional else 'not_currently_verified_run_reproduce_all',
            'packages': PACKAGES,
            'receipt': 'manifests/replay_verification.json',
            'scope': 'Selected historical numerical statistics; package reports identify exact coverage and exclusions. Aggregate source files remain unchanged.'},
        'aggregate_policy': {'ci_status': AGGREGATE,
                             'numeric_values': 'Exact JSON scalar projection from original saved summaries; paths and identity metadata removed; no rounded manuscript transcription.',
                             'limits': 'Aggregate files alone do not support paired bootstrap replay. Separate anonymous input packages provide the statistics identified in additional_replay.'},
        'additional_static_diagnostics': 'LOCO and raw components derived from preserved vectors; not among the 252 historical reference scalars.',
        'exclusions': ['Source media and manuscript media/PDFs', 'Base-model weights/checkpoints and activation tensors',
                       'Subject/video/trial/frame/recording identifiers, identity lists and original filesystem paths',
                       'Raw trial outcomes and timing records', 'GPU inference and retraining',
                       'Non-numeric conceptual Figures 1-2; documentary Tables S11, S22, S29 are not numeric result reconstructions'],
        'commands': {'rebuild': 'python analysis/rebuild_public_results.py',
                     'all_replays': 'python analysis/reproduce_all.py',
                     'figures_optional': 'python analysis/render_public_figures.py',
                     'replay_only': 'python analysis/replay_cluster_sufficient_stats.py',
                     'tests': 'python -m unittest discover -s tests -p test_results.py -v'},
        'resource_limits': {'numeric_threads': 1, 'wall_seconds': 180, 'network_required': False},
    }
    dump(RESULTS / 'coverage.json', coverage)
    dump(output / 'verification.json', {'static_reference_scalars': 252, 'static_discrepancies': [],
                                       'aggregate_exports': len(receipts), 'aggregate_numeric_scalars': sum(r['numeric_scalars'] for r in receipts),
                                       'figure_data_files': len(figures), 'map_rows': len(mapping), 'numpy_version': np.__version__,
                                       'temporal_and_practical_ci_replayed': bool(additional),
                                       'additional_scope': 'selected_statistics_not_every_runtime_field',
                                       'additional_receipt': 'manifests/replay_verification.json'})
    print(json.dumps(read(output / 'verification.json'), indent=2))


if __name__ == '__main__':
    if hasattr(signal, 'alarm'):
        signal.alarm(180)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RESULTS/'rebuilt')
    args = parser.parse_args()
    rebuild(args.output)
