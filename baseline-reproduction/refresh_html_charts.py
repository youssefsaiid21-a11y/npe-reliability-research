"""Embed compact histograms from our verified local run into the offline guide.

Reads our generated NPZ and JSON only. It never reads the benchmarking paper.
"""
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
from checkpoint_runtime import prior_bounds

BASE = Path(__file__).resolve().parent
summary = json.loads((BASE / 'original-observation-summary.json').read_text())
artifact = BASE.parent / 'reproduction.nosync/results/original-observation/posterior-samples.npz'
if hashlib.sha256(artifact.read_bytes()).hexdigest() != summary['local_array_sha256']:
    raise ValueError('Local sample arrays do not match the recorded run summary.')
names, low, high = prior_bounds('RW')
if names != summary['parameter_order']:
    raise ValueError('Unexpected parameter order.')
records = {item['checkpoint']: item for item in summary['checkpoints']}
data = {'created': summary['created_utc'], 'seed': summary['seed'],
        'draws': records['1k']['posterior_draws'], 'source_commit': summary['source_commit'],
        'parameters': []}
with np.load(artifact, allow_pickle=False) as arrays:
    for i, name in enumerate(names):
        parameter = {'name': name, 'low': low[i], 'high': high[i],
                     'truth': float(arrays['original_truth'].reshape(-1)[i]),
                     'histograms': {}, 'summaries': {}}
        for budget in ['1k', '10k', '100k']:
            samples = arrays['samples_' + budget]
            if samples.shape != (records[budget]['posterior_draws'], len(names)):
                raise ValueError('Unexpected sample-array shape.')
            column = samples[:, i]
            if not np.isfinite(column).all() or not ((column >= low[i]) & (column <= high[i])).all():
                raise ValueError('Invalid sample values.')
            density, edges = np.histogram(column, bins=40, range=(low[i], high[i]), density=True)
            parameter['centres'] = ((edges[:-1] + edges[1:]) / 2).tolist()
            parameter['histograms'][budget] = density.round(8).tolist()
            result = records[budget]['parameters'][i]
            if result['name'] != name:
                raise ValueError('Unexpected summary parameter order.')
            parameter['summaries'][budget] = {
                'mean': result['mean'], 'interval': result['equal_tailed_95_interval'],
                'inside': result['truth_inside_interval_for_this_example']}
        data['parameters'].append(parameter)
target = BASE / 'codebase-guide.html'
text = target.read_text()
block = '<script id="posterior-data" type="application/json">' + json.dumps(data).replace('<', '\\u003c') + '</script>'
pattern = r'<script id="posterior-data" type="application/json">.*?</script>'
text, count = re.subn(pattern, lambda match: block, text, flags=re.DOTALL)
if count != 1:
    raise ValueError('Expected exactly one chart-data placeholder.')
target.write_text(text)
print('Embedded compact histograms for 4 parameters and 3 checkpoints; no raw arrays or paper embedded.')
