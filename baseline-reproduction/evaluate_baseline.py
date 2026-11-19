"""Evaluate fixed author NPE on generated data. No training path is imported or called."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')


STRUCTURAL_PARAMETERS = {
    'RW': ['Drift', 'InitS'],
    'GB': ['b1', 'b2', 'Imitation q', 'Initial Adopters', 'Innovation p'],
    'SEIRb': ['IFR', 'Patient_Zero_Arrival_Time', 'PWRisk_Gamma',
              'Sensitivity_to_death_Alpha', 'Time_to_perceive_Tp',
              'Time_to_reduce_risk', 'Transmission_intensity_Beta'],
}


def score(samples, truth, widths, std):
    from setup_functions import validation_calculations as metrics
    import torch
    levels = [.5, .75, .95]
    error, errors = metrics.calculate_coverage_prob(samples, truth, levels, len(truth))
    return dict(normalized_rmse=float(metrics.calculate_rmse(samples, truth, widths, torch.as_tensor(std))),
                interval_score=float(metrics.calculate_agg_int_score(samples, truth, std, levels, len(truth))),
                coverage_fraction_error=float(error), coverage_errors_by_level=list(map(float, errors)),
                posterior_contraction=float(metrics.calculate_agg_post_cont(samples, widths**2 / 12)))


def main():
    import numpy as np
    import torch
    from sbi.inference.posteriors.direct_posterior import DirectPosterior
    from sbi.utils import BoxUniform
    from checkpoint_runtime import load_checkpoint, prior_bounds
    from generate_baseline_data import digest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--budget', choices=['1k', '10k', '100k'], default='100k')
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=4242)
    parser.add_argument('--support', choices=['author-gaussian', 'voc-box'], default='author-gaussian')
    args = parser.parse_args()
    if args.draws < 2:
        parser.error('At least two posterior draws are required')
    destination = args.directory / f'evaluation-{args.budget}-{args.support}'
    if destination.exists():
        parser.error('Output already exists; choose a new dataset directory to preserve evidence')
    metadata = json.loads((args.directory / 'data-metadata.json').read_text())
    artifact = args.directory / 'evaluation-data.npz'
    if digest(artifact) != metadata['artifact_sha256']:
        raise RuntimeError('Dataset hash mismatch')
    torch.set_num_threads(2)
    network, config = load_checkpoint(metadata['model'], args.budget)
    names, low, high = prior_bounds(metadata['model'])
    if names != metadata['parameter_order'] or names != list(config['venParameters']):
        raise RuntimeError('Parameter order mismatch')
    lo, hi = torch.tensor(low), torch.tensor(high)
    width = hi - lo
    gaussian = torch.distributions.MultivariateNormal((lo + hi) / 2, torch.diag((width / 4)**2))
    prior = gaussian if args.support == 'author-gaussian' else BoxUniform(lo, hi)
    posterior = DirectPosterior(posterior_estimator=network, prior=prior, device='cpu')
    # Match the normalizer implemented in the pinned run_training entry point.
    torch.manual_seed(42)
    normalizer = torch.clamp(gaussian.sample((10000,)), min=torch.where(lo == 0, lo + .0001, lo), max=hi).std(0)
    with np.load(artifact, allow_pickle=False) as data:
        truth = data['theta'].copy()
        observations = data['observations'].copy()
    if observations.shape != (metadata['cases'], *config['x_o'].shape[1:]):
        raise RuntimeError('Checkpoint input shape mismatch')
    if not np.isfinite(observations).all() or not np.isfinite(truth).all():
        raise RuntimeError('Non-finite input')
    samples = []
    for i, observation in enumerate(observations):
        torch.manual_seed(args.seed + i)
        draws = posterior.sample((args.draws,), x=torch.from_numpy(observation[None]), show_progress_bars=False).detach().cpu().numpy()
        if draws.shape != (args.draws, len(names)) or not np.isfinite(draws).all():
            raise RuntimeError('Invalid posterior samples')
        samples.append(draws)
        print(f'{metadata["model"]} {args.budget}: inferred {i + 1}/{len(truth)}', flush=True)
    samples = np.stack(samples)
    levels = [.5, .75, .95]
    intervals = np.quantile(samples, [.025, .975], axis=1)
    per_parameter_coverage = ((truth >= intervals[0]) & (truth <= intervals[1])).mean(0)
    rmse = np.sqrt(np.mean((samples.mean(1) - truth)**2, axis=0))
    result = dict(model=metadata['model'], budget=args.budget, cases=len(truth), draws=args.draws,
                  seed=args.seed, source_commit=metadata['source_commit'], dataset_sha256=metadata['artifact_sha256'],
                  status='fresh-data simulator-port pilot; not paper replication',
                  support=args.support, support_explanation='Gaussian follows pinned run_training; checkpoint lineage still unverified. Box option is explicitly a different support restriction.',
                  credible_levels=levels, prior_std_normalizer=normalizer.tolist(),
                  metrics=score(samples, truth, width.numpy(), normalizer.numpy()),
                  parameters=[dict(name=name, rmse=float(rmse[j]), coverage_95=float(per_parameter_coverage[j])) for j, name in enumerate(names)],
                  limitations=metadata['deviations'] + ['Small pilot; no manuscript matching or pass/fail claim. Public entry-point scoring differs from manuscript scoring.'])
    indices = [names.index(name) for name in STRUCTURAL_PARAMETERS[metadata['model']]]
    structural_width = width.numpy()[indices]
    result['structural_parameter_names'] = STRUCTURAL_PARAMETERS[metadata['model']]
    result['structural_metrics_uniform_normalizer'] = score(samples[:, :, indices], truth[:, indices], structural_width, structural_width / np.sqrt(12))
    result['metric_definitions'] = {'metrics': 'All parameters, clipped Gaussian standard deviation: pinned public run_training convention',
                                  'structural_metrics_uniform_normalizer': 'Exclude noise scales/correlation times, normalize by VOC uniform standard deviation: manuscript-oriented diagnostic; model/release/selection alignment still pending'}
    destination.mkdir()
    arrays = destination / 'posterior-samples.npz'
    np.savez_compressed(arrays, samples=samples)
    result['posterior_array_sha256'] = digest(arrays)
    (destination / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
