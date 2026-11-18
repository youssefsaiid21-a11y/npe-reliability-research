"""CPU adapter for the authors' hash-verified, trusted checkpoint files."""
import io
import json
import pickle
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNTIME = BASE.parent / 'reproduction.nosync'
UPSTREAM = RUNTIME / 'upstream'


def load_checkpoint(model, budget):
    import torch
    from prepare_source import verify
    manifest = json.loads((BASE / 'source-manifest.json').read_text())
    entries = {x['path']: x for x in manifest['files']}
    rel = f'density_estimators/{model}_1r_NPE_{budget}_NN.pickle'
    for path in [rel, 'setup_functions/classes.py', 'setup_functions/__init__.py']:
        verify((UPSTREAM / path).read_bytes(), entries[path])
    sys.path.insert(0, str(UPSTREAM))

    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda data: torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
            return super().find_class(module, name)

    # Pickle can execute code: load only the fixed, verified author artifacts.
    with (UPSTREAM / rel).open('rb') as stream:
        network = CPUUnpickler(stream).load()
    network = network.to('cpu').eval()
    embeddings = [m for m in network.modules() if hasattr(m, 'vensimInputs')]
    if len(embeddings) != 1:
        raise RuntimeError(f'Expected one original embedding configuration, found {len(embeddings)}')
    config = embeddings[0].vensimInputs
    config['device'] = 'cpu'  # Runtime placement only; do not change weights/equations.
    return network, config


def prior_bounds(model):
    import re
    names, low, high = [], [], []
    file = {'RW': 'RandomWalk3.voc', 'SEIRb': 'SEIRb.voc', 'GB': 'GeneralizedBass.voc'}[model]
    for line in (UPSTREAM / file).read_text().splitlines():
        match = re.fullmatch(r'\s*([-+\d.eE]+)<=(.+?)<=([-+\d.eE]+)\s*', line)
        if match:
            names.append(match[2].strip())
            low.append(float(match[1]))
            high.append(float(match[3]))
    if not names:
        raise RuntimeError('No original prior bounds found')
    return names, low, high
