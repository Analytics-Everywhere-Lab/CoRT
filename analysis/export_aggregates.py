"""Curator-only projection of approved aggregate sources, supplied as JSON on stdin.

No source locations are saved. Each specification has name, family and parts;
each part has path, optional pointer, optional keys, and optional destination.
This is not a general anonymizer: inputs must already be aggregate summaries.
"""
import hashlib
import json
from pathlib import Path
import re
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = re.compile(
    r"(?:/(?:Users|home|scratch|project|private|tmp)/|"
    r"(?:runs|docs|papers|research|data)/|recording_[a-f0-9]{8,}|"
    r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"
)
IDENTITY_KEY = re.compile(
    r"(?:^|_)(?:subject|participant|wearer|video|trial|episode|recording|unit|sample|frame|cluster)_ids?$"
)
DROP = {'source_files', 'source_hashes', 'source_analysis', 'source',
        'saved_hook_receipts', 'worked_example', 'job_id', 'python_executable'}
OMIT = object()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(data, pointer):
    for part in pointer.strip('/').split('/') if pointer else []:
        part = part.replace('~1', '/').replace('~0', '~')
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def scrub(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in {'source_files', 'source_hashes'} and isinstance(item, dict):
                # Preserve provenance digests, never the private dictionary keys.
                hashes = []
                for record in item.values():
                    if isinstance(record, str) and re.fullmatch(r'[a-f0-9]{64}', record):
                        hashes.append({'sha256': record})
                    elif isinstance(record, dict) and 'sha256' in record:
                        hashes.append({k: record[k] for k in ('sha256', 'bytes') if k in record})
                out[key + '_anonymous'] = hashes
                continue
            if PRIVATE.search(key) or IDENTITY_KEY.search(key):
                continue
            if key in DROP and not (key == 'source' and item in ('escape', 'thu_read', 'youtube')):
                continue
            clean = scrub(item)
            if clean is not OMIT:
                out[key] = clean
        return out
    if isinstance(value, list):
        items = [scrub(x) for x in value]
        # Never silently reindex a list containing private identities.
        return OMIT if any(x is OMIT for x in items) else items
    if isinstance(value, str) and PRIVATE.search(value):
        return OMIT
    return value


def scalar_count(value):
    if isinstance(value, dict):
        return sum(map(scalar_count, value.values()))
    if isinstance(value, list):
        return sum(map(scalar_count, value))
    return int(isinstance(value, (int, float)) and not isinstance(value, bool))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def export(specs):
    receipts = []
    for spec in specs:
        assert re.fullmatch(r'[a-z_]+', spec['name'])
        output, sources = {}, []
        for part in spec['parts']:
            path = Path(part['path'])
            before = digest(path)
            original = json.loads(path.read_text())
            selected = resolve(original, part.get('pointer', ''))
            if 'keys' in part:
                selected = {k: selected[k] for k in part['keys']}
            selected = scrub(selected)
            assert selected is not OMIT
            if part.get('destination'):
                output[part['destination']] = selected
            else:
                assert not (output.keys() & selected.keys())
                output.update(selected)
            assert digest(path) == before, 'Source changed during read'
            sources.append({'sha256': before, 'json_pointer': part.get('pointer', ''),
                            'selected_keys': part.get('keys', ['*']),
                            'destination': part.get('destination', '')})
        destination = ROOT / 'results' / 'aggregate' / (spec['name'] + '.json')
        write_json(destination, output)
        receipts.append({'name': spec['name'], 'family': spec['family'],
                         'file': destination.relative_to(ROOT).as_posix(),
                         'sha256': digest(destination), 'sources': sources,
                         'numeric_scalars': scalar_count(output),
                         'point_status': 'saved_aggregate_exact_projection',
                         'ci_status': 'aggregate_only_not_exact_ci_replay',
                         'verification': 'selected values copied without rounding; source bytes unchanged'})
    write_json(ROOT / 'results' / 'source_receipts.json', {'schema_version': 1, 'exports': receipts})
    print(json.dumps({'aggregate_exports': len(receipts),
                      'numeric_scalars': sum(r['numeric_scalars'] for r in receipts)}))


if __name__ == '__main__':
    signal.alarm(90)
    export(json.load(sys.stdin))
