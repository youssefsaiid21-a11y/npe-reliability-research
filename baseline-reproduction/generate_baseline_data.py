"""Generate known parameters and fresh data from exported author equations; never train NPE.

PySD is a different execution engine. Generated data remain a simulator-port pilot
until comparison with native Vensim and manuscript/release alignment is complete.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
BASE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_simulator(model, allow_pink_expansion=False):
    import pysd
    from checkpoint_runtime import RUNTIME, UPSTREAM
    from prepare_source import verify
    receipts = json.loads((BASE / 'model-export-receipts.json').read_text())
    receipt = receipts[model]
    manifest = json.loads((BASE / 'source-manifest.json').read_text())
    entry = next(e for e in manifest['files'] if e['path'] == receipt['package'])
    verify((UPSTREAM / receipt['package']).read_bytes(), entry)
    source = RUNTIME / 'exported-models' / receipt['exported_model']
    if digest(source) != receipt['exported_sha256']:
        raise RuntimeError('Export changed: inspect and update provenance before running')
    if model == 'GB':
        if not allow_pink_expansion:
            raise RuntimeError('Generalized Bass requires RANDOM PINK NOISE. Use --allow-pink-noise-expansion only for the explicitly provisional vendor-formula port.')
        # Expand the stateful formula published by a Vensim administrator.
        # This is an opt-in candidate, NOT certified equivalent to the binary.
        # https://www.ventanasystems.co.uk/forum/viewtopic.php?t=8294
        import re
        text, count = re.subn(r'RANDOM PINK NOISE\(1,sp1, tsp1, NSeed \)',
                              'Pilot pink noise', source.read_text())
        if count != 1:
            raise RuntimeError('Unexpected pink-noise source; expansion refused')
        equations = '''
Pilot pink noise = INTEG(
 (Pilot scaled white noise-Pilot pink noise)/tsp1,
 1+sp1*RANDOM NORMAL(-6,6,0,1,NSeed))
 ~ dmnl ~ |
Pilot scaled white noise =
 1+sp1*SQRT((2-TIME STEP/tsp1)/(TIME STEP/tsp1))*RANDOM NORMAL(-6,6,0,1,NSeed)
 ~ dmnl ~ |
'''
        marker = '********************************************************'
        if marker not in text:
            raise RuntimeError('Missing model equation section boundary')
        text = text.replace(marker, equations + '\n' + marker, 1)
        source = RUNTIME / 'adapted-models' / 'GB_pink_expanded.mdl'
        source.parent.mkdir(exist_ok=True)
        source.write_text(text)
        receipt = dict(receipt, adaptation='Opt-in stateful pink-noise expansion; not native-engine certified',
                       adaptation_reference='https://www.ventanasystems.co.uk/forum/viewtopic.php?t=8294',
                       adapted_sha256=digest(source))
    # Translate the verified MDL again; do not trust a stale generated Python file.
    return pysd.read_vensim(str(source)), receipt


def parameter_overrides(simulator, config, values):
    import re
    normalize = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
    lookup = {}
    for key in simulator.namespace:
        normalized = normalize(key)
        if normalized in lookup:
            raise RuntimeError('Ambiguous model variable mapping: ' + key)
        lookup[normalized] = key
    settings = dict(zip(config['venParameters'], map(float, values)))
    settings.update(zip(config['constants'], config['constant_vals']))
    return {lookup[normalize(key)]: value for key, value in settings.items()}


def simulate(simulator, config, theta, seed):
    import numpy as np
    # PySD random functions use NumPy's generator and ignore Vensim stream IDs.
    # Seed each trajectory explicitly. This does not promise identical Vensim draws.
    np.random.seed(int(seed))
    overrides = parameter_overrides(simulator, config, theta)
    if config['venModelName'] == 'GeneralizedBass':
        # Vensim has a fixed DrivingNoiseSeed independent of trajectory NSeed.
        # Preserve that separation: do not redraw price/advertising per case.
        import pandas as pd
        from scipy.stats import truncnorm
        times = np.arange(0, float(config['Time_points'][-1]) + .0625 / 2, .0625)
        forcing_rng = np.random.RandomState(10000000)
        innovations = truncnorm.rvs(-10, 90, size=(len(times), 2), random_state=forcing_rng)
        overrides.update({'Price P': pd.Series(10 + innovations[:, 0], index=times),
                          'Advertising A': pd.Series(100 + 10 * innovations[:, 1], index=times)})
    return simulator.run(
        params=overrides,
        initial_condition='original', return_columns=config['outputs'],
        return_timestamps=config['Time_points']).to_numpy(dtype=np.float64)


def main():
    import numpy as np
    import torch
    from checkpoint_runtime import load_checkpoint, prior_bounds, RUNTIME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['RW', 'SEIRb', 'GB'], required=True)
    parser.add_argument('--cases', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20261119)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-pink-noise-expansion', action='store_true')
    args = parser.parse_args()
    if args.cases < 1:
        parser.error('--cases must be positive')
    torch.set_num_threads(2)
    _, config = load_checkpoint(args.model, '100k')
    simulator, receipt = load_simulator(args.model, args.allow_pink_noise_expansion)
    from setup_functions.pre_experiment import generate_summary_statistics
    names, low, high = prior_bounds(args.model)
    if list(config['venParameters']) != names:
        raise RuntimeError('Parameter order differs between checkpoint and VOC')
    rng = np.random.default_rng(args.seed)
    requested = rng.uniform(low, high, (args.cases, len(names)))
    # Same boundary guard as author refit_prior_bounds. Retain BOTH labels.
    adjusted_low = np.where(np.asarray(low) == 0, 0.0001, low)
    theta = np.clip(requested, adjusted_low, high)
    seeds = rng.integers(1, 2**31 - 1, args.cases)
    raw, inputs = [], []
    for i, (parameters, seed) in enumerate(zip(theta, seeds)):
        series = simulate(simulator, config, parameters, seed)
        summaries = generate_summary_statistics(series.T, dict(config, plot=False)).T
        observation = np.concatenate([series, summaries], axis=0).astype(np.float32)
        if observation.shape != tuple(config['x_o'].shape[1:]) or not np.isfinite(observation).all():
            raise RuntimeError('Invalid model output/features; no silent rejection or replacement')
        raw.append(series)
        inputs.append(observation)
        print(f'{args.model}: generated {i + 1}/{args.cases}', flush=True)
    out = args.output or RUNTIME / 'results' / (args.model + '-fresh-pilot')
    out.mkdir(parents=True, exist_ok=False)
    artifact = out / 'evaluation-data.npz'
    np.savez_compressed(artifact, requested_theta=requested, theta=theta,
                        raw=np.stack(raw), observations=np.stack(inputs), seeds=seeds)
    metadata = dict(model=args.model, seed=args.seed, cases=args.cases,
                    parameter_order=names, outputs=config['outputs'],
                    time_points=np.asarray(config['Time_points']).tolist(),
                    source_commit=json.loads((BASE / 'source-manifest.json').read_text())['commit'],
                    export=receipt, engine='PySD', engine_version=__import__('pysd').__version__,
                    numpy_version=np.__version__, torch_version=torch.__version__,
                    artifact_sha256=digest(artifact),
                    selection='Uniform VOC draws, author boundary guard, no behavior filtering; follows pinned generate_true_parameters, not certified manuscript selection',
                    status='fresh-data simulator-port pilot; not paper replication',
                    deviations=['NumPy RNG, not Vensim RNG or stream-ID implementation',
                                'Native stochastic equivalence and manuscript selection not certified'])
    if args.model == 'GB':
        metadata['deviations'].extend(['Stateful vendor-formula pink-noise expansion is provisional',
                                       'Fixed driving-noise stream preserved across cases, but NumPy draws differ from Vensim'])
    (out / 'data-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Saved ' + str(artifact), flush=True)


if __name__ == '__main__':
    main()
