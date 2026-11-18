"""Plot our local checkpoint rerun; this is not a copied paper figure."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
summary = json.loads((base / 'original-observation-summary.json').read_text())
directory = base.parent / 'reproduction.nosync/results/original-observation'
with np.load(directory / 'posterior-samples.npz') as arrays:
    truth = arrays['original_truth'].reshape(-1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for index, (name, ax) in enumerate(zip(summary['parameter_order'], axes.flat)):
        for budget in ['1k', '10k', '100k']:
            ax.hist(arrays['samples_' + budget][:, index], bins=55, density=True,
                    histtype='step', linewidth=1.4, label=budget + ' training checkpoint')
        ax.axvline(truth[index], color='black', linestyle='--', label='Saved truth')
        ax.set(xlabel=name, ylabel='Posterior density')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Random Walk — original saved example\nCPU checkpoint inference check; aggregate paper replication pending')
    fig.savefig(directory / 'posterior-comparison.png', dpi=170)
    plt.close(fig)
print('Local figure saved; no paper figure or manuscript was loaded.')
