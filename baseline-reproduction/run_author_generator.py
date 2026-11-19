"""Run only the manuscript authors' CPU data generator, never an NPE training script.

The replication archive must already be available locally. Do not publish it or
its private download link. GitHub checkpoints are a separately versioned source.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNTIME = BASE.parent / 'reproduction.nosync'
ARCHIVE_SHA256 = 'fdfbe324b0dc260651780ed5b9e813b85ac78f35f36a234002cc7ee9aba54f9d'
FOLDERS = {'RW': '04_Random_Walk', 'GB': '07_Generalized_Bass', 'SEIRb': '09_SEIRb'}
BACKENDS = {'RW': 'torch', 'GB': 'torch', 'SEIRb': 'cpp'}


def main():
    import numpy as np
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=list(FOLDERS))
    parser.add_argument('--cases', type=int, default=16)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error('--cases must be positive')
    out = args.output.resolve()
    if out.exists():
        parser.error('Output exists; use a new directory to preserve previous evidence')
    archive = RUNTIME / 'replication-author.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError('Replication archive identity mismatch')
    root = RUNTIME / 'manuscript-replication'
    prefix = FOLDERS[args.model] + '/'
    # Check all released source files in the selected model before executing.
    with zipfile.ZipFile(archive) as source:
        members = [x for x in source.infolist() if x.filename.startswith(prefix) and not x.is_dir()]
        for member in members:
            path = (root / member.filename).resolve()
            if not path.is_relative_to(root.resolve()):
                raise RuntimeError('Unsafe archive member')
            if path.read_bytes() != source.read(member):
                raise RuntimeError('Author source changed: ' + member.filename)
    script = root / FOLDERS[args.model] / 'synth_data_gen_pysdtorch.py'
    command = [sys.executable, str(script), '--num-datasets', str(args.cases),
               '--seed', str(args.seed), '--workers', '1',
               '--max-attempts', str(max(200, args.cases * 100)), '--output-root', str(out)]
    if args.model == 'GB':
        command += ['--backend', 'torch']
    elif args.model == 'SEIRb':
        command += ['--backend', 'cpp', '--cpp-threads', '1']
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    process = subprocess.run(command, cwd=script.parent, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'generator.log').write_text(process.stdout)
    print(process.stdout, end='', flush=True)
    if process.returncode:
        raise RuntimeError('Author generator failed; log retained; no successful receipt written')
    files = sorted((out / 'true_data' / 'true_series_tab').glob('data_v*.tab'))
    truths = sorted((out / 'true_data' / 'true_params').glob('true_param_v*.cin'))
    if len(files) != args.cases or len(truths) != args.cases:
        raise RuntimeError('Author output count mismatch')
    records = []
    bounds = {}
    for line in (script.parent / 'Synth_Data_Gen.voc').read_text().splitlines():
        match = re.fullmatch(r'\s*([-+\d.eE]+)\s*<=\s*(.*?)\s*<=\s*([-+\d.eE]+)\s*', line)
        if match:
            bounds[match[2]] = (float(match[1]), float(match[3]))
    if not bounds:
        raise RuntimeError('No evaluation parameter bounds found')
    time_axis = None
    for data_path in files:
        match = re.fullmatch(r'data_v(\d+)\.tab', data_path.name)
        index = int(match[1])
        truth_path = out / 'true_data' / 'true_params' / f'true_param_v{index:02d}.cin'
        if not truth_path.is_file():
            raise RuntimeError('Series has no matching parameter file')
        data = pd.read_csv(data_path, sep='\t')
        if not np.isfinite(data.to_numpy(dtype=float)).all() or not np.all(np.diff(data.iloc[:, 0]) > 0):
            raise RuntimeError('Non-finite data or invalid time axis')
        if time_axis is None:
            time_axis = data.iloc[:, 0].to_numpy()
        elif not np.array_equal(time_axis, data.iloc[:, 0].to_numpy()):
            raise RuntimeError('Datasets have inconsistent observation times')
        values = {}
        for line in truth_path.read_text().splitlines():
            key, value = line.split('=', 1)
            values[key.strip()] = float(value)
        if not np.isfinite(list(values.values())).all():
            raise RuntimeError('Non-finite ground-truth parameters')
        for key, (low, high) in bounds.items():
            if key not in values or not low - 1e-7 <= values[key] <= high + 1e-7:
                raise RuntimeError('Missing or out-of-bounds true parameter: ' + key)
        records.append(dict(series=str(data_path.relative_to(out)),
                            parameters=str(truth_path.relative_to(out)),
                            series_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                            parameters_sha256=hashlib.sha256(truth_path.read_bytes()).hexdigest()))
    result = dict(model=args.model, cases=args.cases, seed=args.seed,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  status='author CPU data generation succeeded; paper NPE reproduction NOT established',
                  archive_sha256=ARCHIVE_SHA256, verified_source_files=len(members),
                  script=str(script.relative_to(RUNTIME)), workers=1, numerical_threads=2,
                  engine='author PySDTorch, ' + BACKENDS[args.model] + ' CPU backend',
                  training_executed=False, selection='Unmodified author generator defaults, including screening and stratification where enabled',
                  rows_per_dataset=len(data), data_columns=list(data.columns[1:]),
                  parameter_bounds=bounds,
                  checkpoint_alignment='Unresolved: these are manuscript-release models; available GitHub weights have different configurations',
                  records=records)
    (out / 'generation-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Verified {args.cases} paired time-series / parameter files. No NPE training executed.')


if __name__ == '__main__':
    main()
