"""Our simulation coordinator: verify → generate → check → record.

The model equations live in the authors' files. We call their data generator
and keep each time series paired with the parameters that produced it.
We do not train or load NPE here.
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
MODELS = {
    'RW': {'name': 'Random Walk', 'folder': '04_Random_Walk', 'backend': 'torch'},
    'SEIRb': {'name': 'SEIRb', 'folder': '09_SEIRb', 'backend': 'cpp'},
    'GB': {'name': 'Generalized Bass', 'folder': '07_Generalized_Bass', 'backend': 'torch'},
}


def fingerprint(path):
    """We use a fingerprint to notice changed files, not to score an experiment."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(model):
    """We check the archive and original files before executing the generator."""
    archive = RUNTIME / 'replication-author.zip'
    if not archive.is_file():
        raise RuntimeError('The local author replication archive is missing. See SIMULATION_SETUP.md.')
    if fingerprint(archive) != ARCHIVE_SHA256:
        raise RuntimeError('Replication archive identity mismatch')
    root = RUNTIME / 'manuscript-replication'
    prefix = MODELS[model]['folder'] + '/'
    with zipfile.ZipFile(archive) as source:
        members = [x for x in source.infolist() if x.filename.startswith(prefix) and not x.is_dir()]
        if not members:
            raise RuntimeError('Selected model is absent from the archive')
        for member in members:
            path = (root / member.filename).resolve()
            if not path.is_relative_to(root.resolve()):
                raise RuntimeError('Unsafe archive member')
            if not path.is_file() or path.read_bytes() != source.read(member):
                raise RuntimeError('Author source missing or changed: ' + member.filename)
    script = root / MODELS[model]['folder'] / 'synth_data_gen_pysdtorch.py'
    return script, len(members)


def read_bounds(path):
    """Read the allowed parameter ranges from the author's control file."""
    bounds = {}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r'\s*([-+\d.eE]+)\s*<=\s*(.*?)\s*<=\s*([-+\d.eE]+)\s*', line)
        if match:
            bounds[match[2]] = (float(match[1]), float(match[3]))
    if not bounds:
        raise RuntimeError('No evaluation parameter bounds found')
    return bounds


def run_generator(script, model, cases, seed, output):
    """We run only the simulator, with visible progress and a persistent log."""
    # We name the simulation script explicitly. The archive also contains NPE
    # training scripts, but they are outside this command's execution path.
    command = [sys.executable, str(script), '--num-datasets', str(cases),
               '--seed', str(seed), '--workers', '1',
               '--max-attempts', str(max(200, cases * 100)), '--output-root', str(output)]
    if model == 'GB':
        command += ['--backend', 'torch']
    elif model == 'SEIRb':
        # This is the authors' CPU C++ backend, not the Windows Vensim DLL.
        command += ['--backend', 'cpp', '--cpp-threads', '1']
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with (output / 'generator.log').open('w') as log:
        process = subprocess.Popen(command, cwd=script.parent, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            if process.wait():
                raise RuntimeError('Author generator failed; see generator.log')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()


def validate_outputs(output, cases, bounds):
    """Check graph/parameter pairing, numbers, columns, times and parameter limits."""
    import numpy as np
    import pandas as pd
    files = sorted((output / 'true_data/true_series_tab').glob('data_v*.tab'))
    truths = sorted((output / 'true_data/true_params').glob('true_param_v*.cin'))
    if len(files) != cases or len(truths) != cases:
        raise RuntimeError('Author output count mismatch')
    records, indices = [], set()
    time_axis, columns = None, None
    for data_path in files:
        match = re.fullmatch(r'data_v(\d+)\.tab', data_path.name)
        if not match:
            raise RuntimeError('Unrecognized dataset filename')
        index = int(match[1])
        indices.add(index)
        truth_path = output / 'true_data/true_params' / f'true_param_v{index:02d}.cin'
        if not truth_path.is_file():
            raise RuntimeError('Series has no matching parameter file')
        data = pd.read_csv(data_path, sep='\t')
        if len(data) < 2 or len(data.columns) < 2 or data.columns[0].lower() != 'time':
            raise RuntimeError('Dataset needs a Time column and at least two observations')
        if not np.isfinite(data.to_numpy(dtype=float)).all() or not np.all(np.diff(data.iloc[:, 0]) > 0):
            raise RuntimeError('Non-finite data or invalid time axis')
        if time_axis is None:
            time_axis, columns = data.iloc[:, 0].to_numpy(), list(data.columns)
        elif not np.array_equal(time_axis, data.iloc[:, 0].to_numpy()) or columns != list(data.columns):
            raise RuntimeError('Datasets have inconsistent observation times or columns')
        values = {}
        for line in truth_path.read_text().splitlines():
            key, value = line.split('=', 1)
            key = key.strip()
            if key in values:
                raise RuntimeError('Duplicate parameter: ' + key)
            values[key] = float(value)
        if not np.isfinite(list(values.values())).all():
            raise RuntimeError('Non-finite ground-truth parameters')
        for key, (low, high) in bounds.items():
            # The author writes rounded parameter values. We allow that tiny
            # rounding error; we do not accept a different parameter range.
            if key not in values or not low - 1e-7 <= values[key] <= high + 1e-7:
                raise RuntimeError('Missing or out-of-bounds true parameter: ' + key)
        records.append(dict(series=str(data_path.relative_to(output)),
                            parameters=str(truth_path.relative_to(output)),
                            series_sha256=fingerprint(data_path), parameters_sha256=fingerprint(truth_path)))
    if indices != set(range(1, cases + 1)):
        raise RuntimeError('Dataset numbering has gaps or duplicate identifiers')
    return dict(rows_per_dataset=len(time_axis), data_columns=columns[1:], records=records)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main(argv=None):
    """The whole workflow: verify → generate → check → save a receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=list(MODELS))
    parser.add_argument('--cases', type=int, default=16)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cases < 1:
        parser.error('--cases must be positive')
    output = args.output.resolve()
    if not output.is_relative_to((RUNTIME / 'results').resolve()):
        parser.error('Generated data must stay under reproduction.nosync/results')
    if output.exists():
        parser.error('Output exists; choose a new directory to preserve evidence')

    # 1. We establish which original files and parameter ranges we are using.
    script, verified_count = verify_source(args.model)
    bounds = read_bounds(script.parent / 'Synth_Data_Gen.voc')
    # We reserve a fresh folder before starting. A second run cannot overwrite
    # it, and an interrupted run cannot look like a previous successful one.
    output.mkdir(parents=True, exist_ok=False)
    state = dict(model=args.model, cases=args.cases, seed=args.seed, status='running',
                 created_utc=datetime.now(timezone.utc).isoformat(), training_executed=False)
    write_json(output / 'run-status.json', state)
    try:
        # 2. The authors' generator does the actual simulation work.
        run_generator(script, args.model, args.cases, args.seed, output)
        # 3. We check every graph/answer pair before calling this a result.
        checks = validate_outputs(output, args.cases, bounds)
        receipt = dict(state, status='author CPU data generation succeeded; paper NPE reproduction NOT established',
                       archive_sha256=ARCHIVE_SHA256, verified_source_files=verified_count,
                       script=str(script.relative_to(RUNTIME)), workers=1, numerical_threads=2,
                       engine='author PySDTorch, ' + MODELS[args.model]['backend'] + ' CPU backend',
                       selection='Unmodified author generator defaults, including screening and stratification where enabled',
                       parameter_bounds=bounds, checkpoint_alignment='Waiting for manuscript pretrained networks from authors', **checks)
        # 4. We record the settings and exact files so we can revisit the run.
        write_json(output / 'generation-receipt.json', receipt)
        write_json(output / 'run-status.json', dict(state, status='complete'))
    except BaseException as error:
        write_json(output / 'run-status.json', dict(state, status='failed', error=str(error)))
        raise
    print(f'Checked {args.cases} paired time-series / parameter files. No NPE training executed.')
    return output


if __name__ == '__main__':
    main()
