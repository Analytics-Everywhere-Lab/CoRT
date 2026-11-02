"""Portable temporal replay checks; optional explicit read-only source audit."""
from copy import deepcopy
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from analysis.export_aggregates import IDENTITY_KEY, PRIVATE
from analysis.export_temporal_replay import (
    INVENTORY, PUBLIC, ROOT, SOURCES, export, project_analysis, project_permutations,
    public_analysis, read_json, reference, resolve,
)
from analysis.replay_temporal import (
    Comparison, describe, factorial_vectors, recovery, relative, replay, run, validate,
)

DATA = ROOT / 'results' / 'replay_temporal'


class TemporalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = {name: read_json(DATA / (name + '.json')) for name in PUBLIC}
        cls.report = run()

    def test_all_historical_estimates_and_intervals(self):
        report = self.report
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(report['counts']['vector_count'], 1272)
        self.assertEqual(report['counts']['vector_value_count'], 45798)
        self.assertEqual(report['counts']['recovery_profile_count'], 112)
        self.assertEqual(report['counts']['ratio_count'], 232)
        self.assertEqual(report['counts']['derived_vector_count'], 28)
        self.assertEqual(report['counts']['ci95_intervals'], 1504)
        self.assertEqual(report['counts']['numeric_values'], 11436)
        self.assertEqual(report['counts']['withheld_ci95'], 0)
        for result in report['results'].values():
            self.assertEqual(result['comparison']['differences'], [])
            self.assertEqual(result['public_aggregate_comparison']['differences'], [])
        self.assertTrue(report['coverage']['public_aggregates_unchanged'])
        self.assertFalse(report['coverage']['all_temporal_runtime_outputs_recomputed'])

    def test_report_is_deterministic(self):
        self.assertEqual(self.report, run())
        saved = read_json(ROOT / 'results' / 'rebuilt' / 'temporal_replay.json')
        self.assertEqual(saved['status'], self.report['status'])
        self.assertEqual(saved['counts']['numeric_values'], self.report['counts']['numeric_values'])
        self.assertEqual(saved['counts']['ci95_intervals'], self.report['counts']['ci95_intervals'])

    def test_projection_boundaries(self):
        self.assertEqual(self.report['counts']['vector_summaries_in_public_aggregates'], 696)
        self.assertEqual(self.report['counts']['vector_summaries_original_source_only'], 576)
        self.assertEqual(self.report['counts']['row_diagnostic_fields_copied_not_recomputed'], 704)
        self.assertEqual(sum(d['technical_summary']['n_checks'] for d in self.datasets.values()), 19200)
        self.assertEqual(sum(d['technical_summary']['passed_checks'] for d in self.datasets.values()), 19200)
        self.assertEqual(sum(len(d.get('frozen_frame_orders', [])) for d in self.datasets.values()), 64)

    def test_no_identity_fields_or_private_strings(self):
        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertIsNone(IDENTITY_KEY.search(key), key)
                    self.assertNotIn(key, {'source', 'source_files', 'source_hashes', 'provenance', 'wearer_values'})
                    check(key)
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
            elif isinstance(value, str):
                self.assertIsNone(PRIVATE.search(value), value)
                self.assertIsNone(re.search(r'\b[a-f0-9]{32,64}\b', value), value)
        check(self.report)
        for dataset in self.datasets.values():
            check(dataset)

    def test_indices_and_frozen_orders(self):
        for name, data in self.datasets.items():
            self.assertEqual(data['anonymous_index_order'], list(range(INVENTORY[name][0])))
            frozen = data.get('frozen_frame_orders', [])
            self.assertEqual([r['clip_ordinal'] for r in frozen], list(range(len(frozen))))
            for record in frozen:
                for condition, values in record['permutations'].items():
                    self.assertEqual(len(values), 17)
                    self.assertTrue(all(type(i) is int for i in values))
                    self.assertEqual(values if condition == 'B4' else sorted(values),
                                     [16] * 17 if condition == 'B4' else list(range(17)))

    def test_rejects_reordered_or_malformed_vectors(self):
        base = self.datasets['current']
        mutations = (
            lambda d: d['anonymous_index_order'].reverse(),
            lambda d: d['vectors'][0]['values'].pop(),
            lambda d: d['vectors'][0]['values'].__setitem__(0, float('nan')),
            lambda d: d['vectors'][0]['values'].__setitem__(0, True),
            lambda d: d['ratios'][0].__setitem__('denominator', -1),
            lambda d: d['vectors'].pop(),
            lambda d: d['bootstrap'].__setitem__('seed', True),
        )
        for mutate in mutations:
            data = deepcopy(base)
            mutate(data)
            with self.assertRaises(ValueError):
                validate(data)

    def test_changed_value_or_seed_is_detected(self):
        data = deepcopy(self.datasets['followup_interaction_order'])
        data['vectors'][0]['values'][0] += .1
        self.assertFalse(replay(data)['comparison']['passed'])
        data = deepcopy(self.datasets['followup_interaction_order'])
        data['bootstrap']['seed'] += 1
        self.assertFalse(replay(data)['comparison']['passed'])

    def test_reordered_candidate_order_with_missing_ci_is_rejected_before_replay(self):
        datasets = deepcopy(self.datasets)
        data = datasets['recognition_interaction_order']
        record = next(v for v in data['vectors'] if 'by_order' in v['path'])
        record['values'].reverse()
        del record['reference']['ci95']
        with patch('analysis.replay_temporal.read_json', side_effect=lambda p: datasets[p.stem]), \
                patch('analysis.replay_temporal.describe', side_effect=AssertionError('comparison started')):
            with self.assertRaisesRegex(ValueError, 'reference fields'):
                run()

    def test_every_reference_field_is_mandatory(self):
        data = deepcopy(self.datasets['followup_interaction_order'])
        representatives = [data['vectors'][0], data['factorials'][0]]
        schemas = set()
        for record in data['ratios']:
            schema = frozenset(record['reference'])
            if schema not in schemas:
                representatives.append(record)
                schemas.add(schema)
        for record in representatives:
            reference = record['reference']
            for key in list(reference):
                with self.subTest(path=record['path'], omitted=key):
                    value = reference.pop(key)
                    try:
                        with self.assertRaisesRegex(ValueError, 'reference fields'):
                            validate(data)
                    finally:
                        reference[key] = value
            record['reference'] = {}
            with self.assertRaisesRegex(ValueError, 'reference fields'):
                validate(data)
            record['reference'] = reference

    def test_reference_shapes_types_and_path_coverage_are_required(self):
        base = self.datasets['recognition_interaction_order']
        for replacement in (None, [], [0.], [0., 1., 2.], [False, 1.], [0., float('nan')]):
            data = deepcopy(base)
            record = next(v for v in data['vectors'] if 'by_order' in v['path'])
            record['reference']['ci95'] = replacement
            with self.subTest(ci95=replacement), self.assertRaisesRegex(ValueError, 'reference interval'):
                validate(data)
        for key, replacement in (('estimate', None), ('estimate', float('nan')), ('valid', 1),
                                 ('ci_method', 'unknown'), ('extra', 0)):
            data = deepcopy(base)
            data['vectors'][0]['reference'][key] = replacement
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate(data)
        data = deepcopy(base)
        record = next(v for v in data['vectors'] if 'by_order' in v['path'])
        record['path'][3] = 2
        with self.assertRaisesRegex(ValueError, 'path coverage'):
            validate(data)
        data = deepcopy(base)
        data['ratios'][1] = deepcopy(data['ratios'][0])
        with self.assertRaisesRegex(ValueError, 'path coverage'):
            validate(data)

    def test_public_aggregate_change_is_detected(self):
        data = self.datasets['followup_interaction_order']
        public = public_analysis(data['name'])
        resolve(public, data['vectors'][0]['path'])['estimate'] += .1
        self.assertFalse(replay(data, public)['public_aggregate_comparison']['passed'])

    def test_pooled_ratio_not_individual_ratios(self):
        n, d = (1., 3., 4.), (1., 9., 10.)
        actual = recovery(n, d, 101, 123, .01)
        self.assertEqual(actual['estimate'], (math.fsum(n) / len(n)) / (math.fsum(d) / len(d)))
        self.assertNotAlmostEqual(actual['estimate'], math.fsum(a / b for a, b in zip(n, d)) / 3)
        changed_pairing = recovery(tuple(reversed(n)), d, 101, 123, .01)
        self.assertEqual(actual['estimate'], changed_pairing['estimate'])
        self.assertNotEqual(actual['ci95'], changed_pairing['ci95'])

    def test_guards_withhold_without_deleting_draws(self):
        zero = recovery((1., 2.), (0., 0.), 101, 123, .01)
        self.assertIsNone(zero['estimate'])
        self.assertIsNone(zero['ci95'])
        unstable = recovery((1., 2.), (0., 1.), 101, 123, .01)
        self.assertIsNotNone(unstable['estimate'])
        self.assertIsNone(unstable['ci95'])
        self.assertFalse(unstable['stable'])
        self.assertEqual(unstable['bootstrap_draws'], 101)
        self.assertEqual(unstable['valid_bootstrap_draws'] + unstable['excluded_bootstrap_draws'], 101)
        self.assertGreater(unstable['excluded_bootstrap_draws'], 0)
        self.assertEqual(unstable['excluded_clusters'], 0)
        guarded = relative((1., 2.), (1., 2.), (0., 0.), 101, 123, .01)
        self.assertIsNone(guarded['estimate'])
        self.assertFalse(guarded['native_guard_passed'])
        self.assertTrue(guarded['k_effect_guard_passed'])

    def test_shapley_recomputed_from_subset_values(self):
        data = self.datasets['followup_new_interaction']
        for block in data['factorials']:
            subsets = {s['position_set']: data['vectors'][s['numerator']]['values'] for s in block['subsets']}
            calculated = factorial_vectors(subsets)
            for key, index in block['derived'].items():
                self.assertEqual(list(calculated[key]), data['vectors'][index]['values'])
            incomplete = dict(subsets)
            incomplete.pop('C')
            with self.assertRaises(ValueError):
                factorial_vectors(incomplete)

    def test_comparison_does_not_treat_bool_as_number(self):
        comparison = Comparison()
        comparison.check(True, 1)
        self.assertFalse(comparison.result()['passed'])
        comparison = Comparison()
        comparison.check(0.5 + 1e-13, 0.5)
        self.assertTrue(comparison.result()['passed'])
        self.assertEqual(comparison.exact_numeric_values, 0)
        comparison = Comparison()
        comparison.check(0.5, float('nan'))
        self.assertFalse(comparison.result()['passed'])

    def test_exporter_rejects_non_allowlisted_data(self):
        stat = {**describe((.2, .4, .9), 19, 123), 'wearer_values': {'a': .2, 'b': .4, 'c': .9}}
        source = {'bootstrap': {'draws': 19, 'seed': 123},
                  'behavior': [{'condition': 'B0', 'metrics': {'accuracy': stat}}], 'patch_profiles': []}
        projected = project_analysis('current', source, source)
        self.assertEqual(projected['vectors'][0]['values'], [.2, .4, .9])
        for mutation in ('identity', 'order', 'boolean', 'prose'):
            bad = deepcopy(source)
            item = bad['behavior'][0]['metrics']['accuracy']
            if mutation == 'identity':
                bad['behavior'][0]['condition'] = 'private-person'
            elif mutation == 'order':
                item['wearer_values'] = {'c': .9, 'a': .2, 'b': .4}
            elif mutation == 'boolean':
                item['wearer_values']['a'] = True
            else:
                item['reviewer_notes'] = 'private prose'
            with self.assertRaises(ValueError):
                project_analysis('current', bad, bad)

    def test_permutation_inventory_must_match(self):
        orders = {'B0': list(range(17)), 'B4': [16] * 17, 'RV': list(reversed(range(17))),
                  'S1': [0, *reversed(range(1, 16)), 16], 'S2': list(range(17))}
        records = [{'episode_id': 'example-b', 'permutations': orders},
                   {'episode_id': 'example-a', 'permutations': orders}]
        checks = {'checks': [{'episode_id': 'example-a'}, {'episode_id': 'example-b'}]}
        exported = project_permutations(records, checks)
        self.assertEqual([v['clip_ordinal'] for v in exported], [0, 1])
        with self.assertRaises(ValueError):
            project_permutations(records[:1], checks)
        with self.assertRaises(ValueError):
            project_permutations(records + records[:1], checks)

    def test_cli_output_and_validation_failure(self):
        with tempfile.TemporaryDirectory(dir=DATA) as temporary:
            output = Path(temporary) / 'report.json'
            command = [sys.executable, str(ROOT / 'analysis' / 'replay_temporal.py'), '--output', str(output)]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(read_json(output), self.report)
            result = subprocess.run(command + ['--data', temporary], capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(read_json(output)['status'], 'fail')
            self.assertFalse(read_json(output)['coverage']['all_exported_estimates_and_ci95_checked'])


@unittest.skipUnless(os.environ.get('CORT_TEMPORAL_UPSTREAM'), 'explicit offline upstream not supplied')
class TemporalSourceAuditTests(unittest.TestCase):
    def test_every_value_and_reference_matches_original(self):
        upstream = Path(os.environ['CORT_TEMPORAL_UPSTREAM'])
        total = 0
        for name, relative in SOURCES.items():
            original = read_json(upstream / relative / 'analysis.json')
            data = read_json(DATA / (name + '.json'))
            for record in data['vectors']:
                source = resolve(original, record['path'])
                ordered = [source['wearer_values'][k] for k in sorted(source['wearer_values'])]
                self.assertEqual(record['values'], ordered)
                self.assertEqual(record['reference'], reference(source))
                total += len(ordered)
            for record in data['ratios']:
                self.assertEqual(record['reference'], reference(resolve(original, record['path'])))
        self.assertEqual(total, 45798)

    def test_export_is_reproducible_and_read_only(self):
        upstream = Path(os.environ['CORT_TEMPORAL_UPSTREAM'])
        paths = [upstream / root / filename for root in SOURCES.values()
                 for filename in ('analysis.json', 'technical_checks.json')]
        paths += [upstream / 'runs/temporal_recognition/followup_fir_deployment_2026_10_25_v1' /
                  task / 'permutations.json' for task in ('new_interaction', 'interaction_order')]
        snapshots = {p: p.read_bytes() for p in paths}
        with tempfile.TemporaryDirectory(dir=DATA) as temporary:
            counts = export(upstream, Path(temporary))
            self.assertEqual(counts['vectors'], 1272)
            for name in PUBLIC:
                self.assertEqual((Path(temporary) / (name + '.json')).read_bytes(),
                                 (DATA / (name + '.json')).read_bytes())
        self.assertTrue(all(p.read_bytes() == content for p, content in snapshots.items()))


if __name__ == '__main__':
    unittest.main()
