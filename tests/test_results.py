"""CPU-only result integrity, coverage, privacy and replay regression checks."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from export_aggregates import IDENTITY_KEY, PRIVATE, scalar_count, scrub
from rebuild_public_results import AGGREGATE, REPLAY_NAMES, figure_data, read, verify_inputs
from replay_cluster_sufficient_stats import analyze, compare
from reproduce_all import PACKAGES

ORIGINAL_HASHES = {
    'gemma_cutpair.json': 'ef3fe656a753b009ba423a0af7fd0f469b544c1219c6f00ed9cc3e505cc7fdc6',
    'qwen_cutpair.json': '44a97b47839c02dda790436f2e1f099bb61f5d8c0d010eded03ed1e5f7bce504',
    'gemma_factorial.json': '3f0939bda6a9aac0fac92ad1b8763d0188a60e68ce920fa7a2f4f2123adf812a',
    'qwen_factorial.json': 'e8c6d72429a5b848c172613e9a70d3b2e0bfadd238579206de824133c048d37e',
    'expected.json': 'e634598bf6c08d39848fa4826837dccae91b45f0efd2c5b1d1cc754802a02baf',
}


def hashes(paths):
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from keys(v)


class ResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy, cls.replay, cls.receipts = verify_inputs()
        cls.data = {r['name']: read(ROOT/r['file']) for r in cls.receipts}

    def test_original_static_bytes_preserved(self):
        for name, expected in ORIGINAL_HASHES.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT/'results/replay'/name).read_bytes()).hexdigest(), expected)

    def test_252_historical_references(self):
        expected = read(ROOT/'results/replay/expected.json')
        self.assertEqual(scalar_count(expected), 252)
        self.assertEqual(compare(self.replay, expected), [])

    def test_bad_vector_shape_rejected(self):
        data = read(ROOT/'results/replay/gemma_cutpair.json')
        data['vectors']['early_F']['error_bits'].pop()
        with self.assertRaises(ValueError):
            analyze(data)

    def test_independent_seed_not_historical_replay(self):
        data = read(ROOT/'results/replay/gemma_cutpair.json')
        independent = analyze(data, seed_override=71)
        self.assertEqual(independent['interval_status'], 'independently_regenerated')
        self.assertEqual(independent['operators']['early_F']['estimate'], self.replay['gemma_cutpair']['operators']['early_F']['estimate'])
        self.assertNotEqual(independent['operators']['early_F']['ci95'], self.replay['gemma_cutpair']['operators']['early_F']['ci95'])

    def test_aggregate_inventory_complete(self):
        actual = {p.relative_to(ROOT).as_posix() for p in (ROOT/'results/aggregate').glob('*.json')}
        self.assertEqual(actual, {r['file'] for r in self.receipts})
        for r in self.receipts:
            self.assertEqual(r['numeric_scalars'], scalar_count(self.data[r['name']]))
            self.assertEqual(r['ci_status'], AGGREGATE)
            for source in r['sources']:
                self.assertRegex(source['sha256'], r'^[a-f0-9]{64}$')

    def test_no_identity_keys_or_private_values(self):
        for directory in ('results', 'figures'):
            for path in (ROOT/directory).rglob('*'):
                if not path.is_file() or path.suffix == '.png':
                    continue
                with self.subTest(path=path.relative_to(ROOT)):
                    text = path.read_text()
                    # Only these explicit, existing public documentation links are allowed.
                    for spec in PACKAGES.values():
                        self.assertTrue((ROOT/spec['guide']).is_file())
                        text = text.replace(spec['guide'], 'PUBLIC_GUIDE')
                    self.assertIsNone(PRIVATE.search(text))
                    if path.suffix == '.json':
                        d = json.loads(text, parse_constant=lambda x: self.fail(x))
                        self.assertFalse([k for k in keys(d) if IDENTITY_KEY.search(k)])

    def test_privacy_projection_preserves_numbers_and_hashes(self):
        digest = 'a'*64
        fake = {'estimate': .12345678901234567, 'source_hashes': {'private-location': digest},
                'missing_unit_ids': ['restricted-identity'], 'ci95': [.01,.9]}
        clean = scrub(fake)
        self.assertEqual(clean['estimate'], fake['estimate'])
        self.assertEqual(clean['ci95'], fake['ci95'])
        self.assertEqual(clean['source_hashes_anonymous'], [{'sha256': digest}])
        self.assertNotIn('missing_unit_ids', clean)

    def test_all_current_families_and_mapping_files(self):
        required = {'behavioral', 'exploratory_depth', 'cutpair', 'gemma_factorial', 'qwen_factorial',
                    'temporal_current', 'temporal_onset_order', 'temporal_followup',
                    'practical_removal_trained_and_scene_only', 'practical_earlier_benchmark'}
        coverage = read(ROOT/'results/coverage.json')
        self.assertTrue(required <= set(coverage['families']))
        with (ROOT/'results/paper_results_map.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(required <= {r['family'] for r in rows})
        for row in rows:
            for filename in row['evidence_files'].split(';'):
                self.assertTrue((ROOT/filename).is_file(), filename)
            if row['family'].startswith(('temporal', 'practical')):
                self.assertEqual(row['interval_status'], AGGREGATE)

    def test_current_paper_reference(self):
        coverage = read(ROOT/'results/coverage.json')
        reference = coverage['current_paper_reference']
        self.assertEqual(reference['text_sha256'], '28a345c310b6650e857ba75f9d18aa34933f4705cfafe3186fe86a5bc559879c')
        self.assertEqual(reference['numeric_figures'], [3,4,5,6,7,8])

    def test_temporal_inventory_both_directions(self):
        tasks = self.data['temporal_recognition']['tasks']
        self.assertEqual(set(tasks), {'new_interaction','interaction_order'})
        for task in tasks.values():
            self.assertEqual({r['direction'] for r in task['patch_profiles']}, {'B0->B4','B4->B0'})
        for task in self.data['temporal_followup']['analyses'].values():
            self.assertEqual(len(task['factorial']), 2)
            for factorial in task['factorial']:
                labels = {r['position_set'] for r in factorial['subsets']}
                self.assertEqual(len(labels - {'A'}), 15)
                self.assertIn('A', labels)

    def test_figure7_saved_values(self):
        figures = figure_data(self.data, self.replay)
        points = {(r['task'],r['position_set']): r['estimate'] for r in figures['figure7']}
        expected = {
            ('new_interaction','C+Q+P+R'): .4812168478901339,
            ('new_interaction','Q+P+R'): .3167494743894233,
            ('new_interaction','C+Q+R'): .3754067817225226,
            ('interaction_order','C+Q+P+R'): .5866986699303365,
            ('interaction_order','Q+P+R'): .2966862523062199,
            ('interaction_order','C+Q+R'): .28681387628640287,
        }
        self.assertEqual(points, expected)

    def test_practical_sessions_and_decisions(self):
        d = self.data['practical_final']['benchmarks']
        self.assertEqual(set(d), {'combined','followup'})
        self.assertNotEqual(d['combined']['timing']['native']['request_ms'], d['followup']['timing']['native']['request_ms'])
        checks = d['followup']['decision']['co_primaries']
        self.assertTrue(checks['seed17']['upper_left_pass'])
        self.assertFalse(checks['seed29']['upper_left_pass'])
        self.assertFalse(d['followup']['decision']['both_seeds_pass'])
        for v in d.values():
            self.assertEqual(v['counts']['frames'],447)

    def test_missing_timing_not_fabricated(self):
        rows = figure_data(self.data,self.replay)['figure8']
        missing = [r for r in rows if r['timing_status']=='not_available_in_source']
        self.assertTrue(missing)
        self.assertTrue(all(r['warm_request_ms'] is None and r['request_percent_native'] is None for r in missing))

    def test_rebuild_idempotent_inputs_immutable(self):
        inputs = list((ROOT/'results/aggregate').glob('*.json')) + list((ROOT/'results/replay').glob('*.json')) + [ROOT/'results/source_receipts.json']
        before = hashes(inputs)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        cmd = [sys.executable,'-B','analysis/rebuild_public_results.py']
        subprocess.run(cmd,cwd=ROOT,env=env,check=True,capture_output=True,timeout=180)
        outputs = list((ROOT/'results').rglob('*')) + list((ROOT/'figures').glob('*.csv')) + list((ROOT/'figures').glob('*.json'))
        first = hashes(outputs)
        subprocess.run(cmd,cwd=ROOT,env=env,check=True,capture_output=True,timeout=180)
        self.assertEqual(first, hashes(outputs))
        self.assertEqual(before, hashes(inputs))

    @unittest.skipUnless(importlib.util.find_spec('matplotlib') and importlib.util.find_spec('PIL'), 'optional figure dependencies not installed')
    def test_figure_render_idempotent_and_nonblank(self):
        from PIL import Image
        cmd = [sys.executable,'-B','analysis/render_public_figures.py']
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        immutable = list((ROOT/'results/aggregate').glob('*.json')) + [ROOT/'results/source_receipts.json']
        before = hashes(immutable)
        subprocess.run(cmd,cwd=ROOT,env=env,check=True,capture_output=True,timeout=120)
        paths = sorted((ROOT/'figures/rendered').glob('figure*.png'))
        self.assertEqual(len(paths),6)
        first = hashes(paths)
        for path in paths:
            with Image.open(path) as im:
                self.assertGreater(im.width,1000)
                self.assertGreater(im.height,500)
                self.assertTrue(any(lo != hi for lo,hi in im.convert('RGB').getextrema()))
        subprocess.run(cmd,cwd=ROOT,env=env,check=True,capture_output=True,timeout=120)
        self.assertEqual(first,hashes(paths))
        self.assertEqual(before,hashes(immutable))


if __name__ == '__main__':
    unittest.main()
