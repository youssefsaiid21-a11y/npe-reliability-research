"""Fetch/check pinned public artifacts. Never execute or modify upstream sources."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
RUNTIME = ROOT / 'reproduction.nosync'


def verify(data, entry):
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if blob != entry['git_blob_sha1']:
        raise ValueError('Upstream Git blob mismatch: ' + entry['path'])
    if entry.get('sha256') and hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('SHA-256 mismatch: ' + entry['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['RW', 'SEIRb', 'GB', 'all'], default='RW')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((BASE / 'source-manifest.json').read_text())
    model_files = {'RW': {'RandomWalk3.voc', 'RandomWalk3.vpmx'},
                   'SEIRb': {'SEIRb.voc', 'SEIRbNosubs.vpmx'},
                   'GB': {'GeneralizedBass.voc', 'GeneralizedBass.vpmx'}}
    records, missing = [], []
    for entry in manifest['files']:
        rel = entry['path']
        models = list(model_files) if args.model == 'all' else [args.model]
        selected = (rel.startswith('setup_functions/') or rel in {'LICENSE', 'Main.ipynb'}
                    or any(rel in model_files[model] or rel.startswith('density_estimators/' + model + '_') for model in models))
        if not selected:
            continue
        target = RUNTIME / 'upstream' / rel
        candidates = [target, ROOT / 'verification/upstream' / rel,
                      ROOT / 'reproduction/reference' / rel]
        data = None
        for candidate in candidates:
            if candidate.is_file():
                candidate_data = candidate.read_bytes()
                verify(candidate_data, entry)
                data = candidate_data
                break
        if data is None:
            if args.offline:
                missing.append(rel)
                continue
            url = ('https://raw.githubusercontent.com/ali-akhavan89/'
                   'SD_Neural_Estimation_Benchmarking/' + manifest['commit'] + '/' + rel)
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read()
            verify(data, entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        records.append({'path': rel, 'sha256': hashlib.sha256(data).hexdigest(),
                        'git_blob_sha1': entry['git_blob_sha1'], 'bytes': len(data)})
    result = {'source_commit': manifest['commit'], 'model': args.model,
              'verified_files': records, 'missing': missing,
              'status': 'ready' if not missing else 'incomplete'}
    (BASE / 'source-checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'model': args.model, 'verified': len(records), 'missing': missing}))
    if missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
