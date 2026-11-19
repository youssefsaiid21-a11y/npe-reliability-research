"""Scientific sanity checks for the author-equation ports; not native Vensim certification."""
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')


def main():
    import numpy as np
    import torch
    from checkpoint_runtime import load_checkpoint, RUNTIME
    from generate_baseline_data import load_simulator, simulate, parameter_overrides
    torch.set_num_threads(2)
    result = {'status': 'sanity checks only; native-engine equivalence pending', 'checks': {}}
    rw, _ = load_simulator('RW')
    _, config = load_checkpoint('RW', '100k')
    t = np.asarray(config['Time_points'])
    deterministic = simulate(rw, config, [.2, 4, 0, 0], 17)[:, 0]
    error = float(np.max(np.abs(deterministic - (4 + .2 * t))))
    assert error < 1e-10, error
    theta = [.2, 4, .3, .15]
    first = simulate(rw, config, theta, 17)
    again = simulate(rw, config, theta, 17)
    changed = simulate(rw, config, theta, 18)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, changed)
    increments = []
    for seed in range(60):
        series = simulate(rw, config, [.2, 4, .3, 0], seed)[:, 0]
        increments.extend(np.diff(series) - .2)
    mean, std = float(np.mean(increments)), float(np.std(increments))
    assert abs(mean) < .02 and abs(std - .3) < .02, (mean, std)
    result['checks']['RW'] = dict(linear_limit_max_error=error, replay_exact=True,
                                 changed_seed_changes_data=True, innovation_mean=mean,
                                 innovation_std=std, expected_innovation_std=.3)
    seir, _ = load_simulator('SEIRb')
    _, config = load_checkpoint('SEIRb', '100k')
    values = config['true_parameter'].numpy().copy()
    for j, name in enumerate(config['venParameters']):
        if name in ['Death_Measurement_Noise_Scale', 'Onset_Measurement_Noise_Scale',
                    'Recovery_Measurement_Noise_Scale', 'Noise_Standard_Deviation_1']:
            values[j] = 0
    overrides = parameter_overrides(seir, config, values)
    columns = ['S', 'E', 'I', 'Recovered', 'Dead']
    stocks = seir.run(params=overrides, initial_condition='original',
                      return_columns=columns, return_timestamps=config['Time_points'])
    total = stocks.to_numpy().sum(1)
    times = np.asarray(config['Time_points'])
    arrival = values[list(config['venParameters']).index('Patient_Zero_Arrival_Time')]
    # Source starts with population-1 and admits one patient at the arrival step.
    expected = 999999 + (times >= np.ceil(arrival / .25) * .25)
    error = float(np.max(np.abs(total - expected)))
    assert error < 1e-6 and stocks.to_numpy().min() >= -1e-8, (error, stocks.min())
    first = simulate(seir, config, values, 17)
    second = simulate(seir, config, values, 999)
    assert np.array_equal(first, second), 'Noiseless model should ignore seed'
    result['checks']['SEIRb'] = dict(population_balance_max_error=error,
                                    nonnegative_stocks=True, noiseless_seed_invariance=True)
    gb, receipt = load_simulator('GB', allow_pink_expansion=True)
    _, config = load_checkpoint('GB', '100k')
    values = config['true_parameter'].numpy().copy()
    for j, name in enumerate(config['venParameters']):
        if name in ['SM1', 'SM2', 'sp1']:
            values[j] = 0
    first = simulate(gb, config, values, 17)
    second = simulate(gb, config, values, 999)
    assert np.array_equal(first, second), 'Fixed driving noise must not change with NSeed'
    assert np.isfinite(first).all() and first.min() >= 0
    overrides = parameter_overrides(gb, config, values)
    # Disable sensitivity to price/advertising to check the Bass stock equations.
    overrides.update({'b1': 0, 'b2': 0})
    stocks = gb.run(params=overrides, initial_condition='original',
                    return_columns=['Adopters N', 'Potential Adopter P', 'PN1'],
                    return_timestamps=config['Time_points'])
    error = float(np.max(np.abs(stocks.iloc[:, :2].sum(axis=1) - 1000000)))
    assert error < 1e-6 and np.all(stocks['PN1'] == 1)
    assert (np.diff(stocks['Adopters N']) >= 0).all()
    result['checks']['GB'] = dict(population_balance_max_error=error,
                                 fixed_driving_noise_preserved=True,
                                 noiseless_pink_multiplier_one=True, nondecreasing_adopters=True,
                                 adaptation=receipt['adaptation'])
    path = RUNTIME / 'results' / 'simulator-sanity-checks.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
