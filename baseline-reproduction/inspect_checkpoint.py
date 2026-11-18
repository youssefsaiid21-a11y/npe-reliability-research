"""Inspect a verified author checkpoint without running a forward simulator."""
import argparse
import json
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import torch
from checkpoint_runtime import load_checkpoint, prior_bounds

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', choices=['RW', 'SEIRb', 'GB'], default='RW')
parser.add_argument('--budget', choices=['1k', '10k', '100k'], default='1k')
args = parser.parse_args()
torch.set_num_threads(2)
network, config = load_checkpoint(args.model, args.budget)
names, low, high = prior_bounds(args.model)
result = {'model': args.model, 'budget': args.budget, 'network_class': type(network).__name__,
          'config_keys': sorted(config), 'parameter_names': names, 'prior_low': low, 'prior_high': high,
          'observation_shape': list(config['x_o'].shape) if 'x_o' in config else None,
          'truth_shape': list(config['true_parameter'].shape) if config.get('true_parameter') is not None else None,
          'scientific_status': 'checkpoint inspection only; no fresh simulations or paper-table replication'}
print(json.dumps(result, indent=2))
