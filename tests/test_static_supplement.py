"""Offline supplementary replay, historical-reference and privacy regressions."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis'))
from export_aggregates import IDENTITY_KEY, PRIVATE
import replay_static_supplement as replay
import numpy as np


class StaticSupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = replay.run()
        cls.exploratory = replay.read(replay.DATA/'exploratory.json')
        cls.metric_reference = replay.read(replay.DATA/'metric_reference.json')
        with (replay.DATA/'metric_cells.csv').open(newline='') as handle:
            cls.metric_rows = list(csv.DictReader(handle))

    def test_saved_report_and_coverage(self):
        self.assertEqual(self.result, replay.read(ROOT/'results/rebuilt/static_supplement_replay.json'))
        self.assertEqual(self.result['status'], 'pass')
        counts = self.result['counts']
        self.assertEqual((counts['metric_rows'], counts['metric_strata']), (322, 7))
        self.assertEqual((counts['exploratory_outcomes'], counts['exploratory_clusters']), (20480, 80))
        self.assertEqual((counts['window_cluster_records'], counts['stratified_cluster_records']), (552, 736))
        self.assertEqual(counts['loco_rows'], 46)

    def test_bad_vector_shape_fails(self):
        data = copy.deepcopy(self.exploratory)
        data['cells'][0]['vector']['error_bits'].pop()
        with self.assertRaisesRegex(ValueError, 'shape'):
            replay.endpoint_replay(data, 128, 80)

    def test_nonfinite_negative_and_string_values_fail(self):
        for value in (float('nan'), float('inf'), -1., '0.5', True):
            data = copy.deepcopy(self.exploratory['cells'][0]['vector'])
            data['error_bits'][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                replay.rho(data, 80)
        data['error_bits'][0] = 0.
        data['gap_bits'][0] = 0.
        with self.assertRaisesRegex(ValueError, 'gap'):
            replay.rho(data, 80)

    def test_invalid_input_hash_fails(self):
        with tempfile.TemporaryDirectory(dir=replay.DATA) as temp:
            directory = Path(temp)
            manifest = replay.read(replay.DATA/'manifest.json')
            for name in manifest['files']:
                (directory/name).write_bytes((replay.DATA/name).read_bytes())
            (directory/'manifest.json').write_text(json.dumps(manifest))
            path = directory/'metric_cells.csv'
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                replay.verify_inputs(directory)

    def test_historical_draw_hash_fails(self):
        reference = copy.deepcopy(self.metric_reference)
        reference['paired_draws_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'paired interaction draw hash'):
            replay.metric_replay(self.metric_rows, reference)
        reference = copy.deepcopy(self.metric_reference)
        reference['bootstrap']['index_sha256_little_endian_int64'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'draw index hash'):
            replay.metric_replay(self.metric_rows, reference)

    def test_metric_ordinal_and_schema_fail(self):
        for mutate in ('ordinal', 'field', 'count'):
            rows = copy.deepcopy(self.metric_rows)
            if mutate == 'ordinal':
                rows[0]['cluster_ordinal'] = '2'
            elif mutate == 'field':
                rows[0]['extra'] = 'private'
            else:
                rows.pop()
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                replay.metric_replay(rows, self.metric_reference)

    def test_wrong_saved_mean_and_ci_fail(self):
        for field in ('estimate', 'ci95', 'mean_gap_bits', 'mean_error_bits'):
            data = copy.deepcopy(self.exploratory)
            if field == 'ci95':
                data['cells'][0]['reference'][field][0] += .01
            else:
                data['cells'][0]['reference'][field] += .01
            with self.subTest(field=field), self.assertRaises(ValueError):
                replay.endpoint_replay(data, 128, 80)

    def test_ratio_of_means_not_mean_of_ratios(self):
        v = {'error_bits': [1., 1.], 'gap_bits': [1., 9.]}
        self.assertEqual(replay.rho(v, 2), .8)
        self.assertNotAlmostEqual(replay.rho(v, 2), float(np.mean([0., 8/9])))
        index = np.array([[0, 0], [1, 1], [0, 1]])
        np.testing.assert_allclose(replay.rho(v, 2, index), [0., 8/9, .8], rtol=0, atol=1e-15)

    def test_seed_change_keeps_mean_but_rejects_historical_ci(self):
        data = copy.deepcopy(self.exploratory)
        first = data['cells'][0]
        self.assertAlmostEqual(replay.rho(first['vector'], 80), first['reference']['estimate'])
        first['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'ci95'):
            replay.endpoint_replay(data, 128, 80)

    def test_behavior_matches_original_full_intervals(self):
        reference = replay.read(replay.DATA/'behavior_reference.json')
        self.assertEqual(self.result['behavior'], reference)
        gemma = self.result['behavior'][replay.MODELS[0]]
        self.assertEqual(gemma['cells']['real']['ci95'], [0.7124350319396051, 0.8498718786295006])
        self.assertEqual(gemma['tails']['wrong'], 4/10001)
        data = replay.read(ROOT/'results/aggregate/behavior.json')
        data[replay.MODELS[0]]['cluster_vectors']['real'].reverse()
        with self.assertRaises(ValueError):
            replay.behavior_replay(data, reference)

    def test_current_sensitivity_and_loco(self):
        result = self.result['sensitivity']
        self.assertNotIn('family_B', result)
        self.assertEqual([r['omitted_ordinal'] for r in result['loco']], list(range(46)))
        source = replay.read(ROOT/'results/replay/gemma_cutpair.json')['vectors']
        for i, row in enumerate(result['loco']):
            for op in ('early_F', 'early_K', 'late_F', 'late_K'):
                error = np.delete(source[op]['error_bits'], i)
                gap = np.delete(source[op]['gap_bits'], i)
                self.assertEqual(row[op], float(1-error.mean()/gap.mean()))
        self.assertTrue(all(r['early_F_minus_K'] > 0 and r['late_K_minus_F'] > 0 for r in result['loco']))

    def test_privacy_and_text_only_outputs(self):
        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertIsNone(IDENTITY_KEY.search(key))
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
        for path in list(replay.DATA.glob('*')) + [ROOT/'results/rebuilt/static_supplement_replay.json']:
            self.assertIn(path.suffix, ('.json', '.csv'))
            text = path.read_text()
            self.assertIsNone(PRIVATE.search(text))
            self.assertNotIn('family_B', text)
            if path.suffix == '.json':
                check(json.loads(text, parse_constant=lambda x: self.fail(x)))

    def test_cli_output_and_idempotence(self):
        paths = [p for folder in ('results/aggregate', 'results/replay', 'results/replay_static_supplement')
                 for p in (ROOT/folder).glob('*') if p.is_file()]
        before = {str(p): replay.digest(p) for p in paths}
        with tempfile.TemporaryDirectory(dir=replay.DATA) as temp:
            path = Path(temp)/'report.json'
            command = [sys.executable, '-B', str(ROOT/'analysis/replay_static_supplement.py'), '--output', str(path)]
            subprocess.run(command, check=True, capture_output=True, timeout=120, cwd=ROOT)
            first = path.read_bytes()
            subprocess.run(command, check=True, capture_output=True, timeout=120, cwd=ROOT)
            self.assertEqual(path.read_bytes(), first)
            self.assertEqual(json.loads(first), self.result)
        self.assertEqual(before, {str(p): replay.digest(p) for p in paths})


if __name__ == '__main__':
    unittest.main()
