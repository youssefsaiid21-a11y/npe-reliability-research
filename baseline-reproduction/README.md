# Matched-model baseline reproduction

Start with [the reproduction plan](REPRODUCTION_PLAN.md). Random Walk is the first execution target; SEIRb and Generalized Bass follow after source/runtime validation. Earlier results recorded on another Mac are historical evidence, not results from this setup.

## Environment

This thesis uses an isolated native Python 3.12 environment. The general-purpose `python3` installation is separate. Initial compatibility pins use PyTorch 2.6.0 and sbi 0.23.3; the recovered historical Intel runtime used a different PyTorch version. The exact environment is recorded after installation.

From the repository root:

```sh
uv venv --python 3.12 reproduction.nosync/.venv
uv pip install --python reproduction.nosync/.venv/bin/python -r baseline-reproduction/requirements.in
python3 baseline-reproduction/prepare_source.py --model RW
reproduction.nosync/.venv/bin/python baseline-reproduction/inspect_checkpoint.py
reproduction.nosync/.venv/bin/python baseline-reproduction/run_original_observation.py
```

Only hash-verified artifacts from the pinned author repository are loaded. Checkpoints are Python pickles and must not be replaced with untrusted downloads. No upstream training or Vensim wrapper is automatically executed by these commands.

Open `Thesis.code-workspace` in VS Code to use the configured local thesis interpreter. GitHub saves contain our scripts, configuration and compact evidence; local inputs, environments and posterior arrays are excluded.
