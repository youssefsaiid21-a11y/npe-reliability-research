"""Rerun the authors' saved RW example. This is not aggregate paper replication."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['VECLIB_MAXIMUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error('--draws must be positive')
    import importlib.metadata
    import numpy as np
    import torch
    from sbi.inference.posteriors.direct_posterior import DirectPosterior
    from sbi.utils import BoxUniform
    from checkpoint_runtime import load_checkpoint, prior_bounds, RUNTIME, UPSTREAM
    from prepare_source import verify
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    manifest = json.loads((BASE / 'source-manifest.json').read_text())
    source = {x['path']: x for x in manifest['files']}
    verify((UPSTREAM / 'setup_functions/pre_experiment.py').read_bytes(),
           source['setup_functions/pre_experiment.py'])
    names, low, high = prior_bounds('RW')
    low_t, high_t = torch.tensor(low), torch.tensor(high)
    prior = BoxUniform(low=low_t, high=high_t, device='cpu')
    directory = RUNTIME / 'results/original-observation'
    directory.mkdir(parents=True, exist_ok=True)
    records, arrays = [], {}
    reference_x = reference_truth = None
    for budget in ['1k', '10k', '100k']:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        network, config = load_checkpoint('RW', budget)
        if list(config['venParameters']) != names:
            raise RuntimeError('Checkpoint/VOC parameter order mismatch')
        for name, lo, hi in zip(names, low, high):
            if tuple(map(float, config['venParameters'][name])) != (lo, hi):
                raise RuntimeError('Checkpoint/VOC prior mismatch: ' + name)
        x = config['x_o'].detach().cpu().float()
        truth = config['true_parameter'].detach().cpu().reshape(-1)
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[2] != 1:
            raise RuntimeError('Unexpected saved observation shape: ' + str(x.shape))
        if truth.shape != low_t.shape or not torch.isfinite(truth).all():
            raise RuntimeError('Unexpected original truth')
        n_times = len(config['Time_points'])
        raw = x[0, :n_times, :].numpy().T
        from setup_functions.pre_experiment import generate_summary_statistics
        stats_config = dict(config, plot=False)  # Presentation-only runtime adapter.
        manual = generate_summary_statistics(raw, stats_config)
        expected = torch.from_numpy(manual.T.copy()).float().unsqueeze(0)
        saved = x[:, n_times:, :]
        if saved.shape != expected.shape:
            raise RuntimeError('Checkpoint/manual-summary shape mismatch')
        difference = float(torch.max(torch.abs(expected - saved)))
        if not torch.allclose(saved, expected, atol=1e-5, rtol=1e-5):
            raise RuntimeError('Original preprocessing did not reproduce checkpoint features')
        if reference_x is None:
            reference_x, reference_truth = x.clone(), truth.clone()
        elif not torch.equal(reference_x, x) or not torch.equal(reference_truth, truth):
            raise RuntimeError('Checkpoints do not contain the same original example')
        # The original preprocessing module seeds RNGs when imported. Reset
        # after imports so a user-supplied seed is honored for every checkpoint.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        start = time.perf_counter()
        posterior = DirectPosterior(posterior_estimator=network, prior=prior, device='cpu')
        draws = posterior.sample((args.draws,), x=x, show_progress_bars=False).detach().cpu()
        elapsed = time.perf_counter() - start
        if tuple(draws.shape) != (args.draws, len(names)) or not torch.isfinite(draws).all():
            raise RuntimeError('Invalid posterior samples')
        if not ((draws >= low_t) & (draws <= high_t)).all():
            raise RuntimeError('Posterior samples outside original prior')
        samples = draws.numpy()
        quantiles = np.quantile(samples, [0.025, 0.5, 0.975], axis=0)
        arrays['samples_' + budget] = samples
        record = {'checkpoint': budget, 'posterior_draws': args.draws,
                  'sampling_seconds': elapsed, 'manual_summary_max_abs_difference': difference,
                  'parameters': [{'name': name, 'truth': float(truth[i]),
                                  'mean': float(samples[:, i].mean()),
                                  'median': float(quantiles[1, i]),
                                  'equal_tailed_95_interval': [float(quantiles[0, i]), float(quantiles[2, i])],
                                  'truth_inside_interval_for_this_example': bool(quantiles[0, i] <= truth[i] <= quantiles[2, i])}
                                 for i, name in enumerate(names)]}
        records.append(record)
        print(budget, 'passed:', args.draws, 'finite in-prior samples;', round(elapsed, 3), 'seconds', flush=True)
    arrays['original_observation_and_features'] = reference_x.numpy()
    arrays['original_truth'] = reference_truth.numpy()
    artifact = directory / 'posterior-samples.npz'
    np.savez_compressed(artifact, **arrays)
    summary = {'status': 'original_saved_observation_inference_passed',
               'aggregate_paper_reproduction': 'not established',
               'forward_simulations_run': 0, 'training_runs': 0,
               'created_utc': datetime.now(timezone.utc).isoformat(), 'source_commit': manifest['commit'],
               'seed': args.seed, 'cpu_threads': 2, 'architecture': platform.machine(),
               'python_version': platform.python_version(),
               'packages': {name: importlib.metadata.version(name) for name in ['torch', 'sbi', 'numpy', 'scipy']},
               'parameter_order': names, 'original_input_shape': list(reference_x.shape),
               'source_hash_checks': True,
               'runtime_adapters': ['CUDA tensor storage remapped to CPU', 'evaluation mode',
                                    'two CPU threads', 'interactive plotting disabled for summary check'],
               'checkpoints': records,
               'local_array_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
               'limitations': ['One saved example is not an empirical coverage study.',
                              'The original simulator has not been run by this command.',
                              'CPU/runtime changes do not guarantee identical random draws to the published run.']}
    (BASE / 'original-observation-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Saved compact evidence; posterior arrays remain local.')


if __name__ == '__main__':
    main()
