# Matched-model baseline reproduction

**19 November update:** the manuscript authors' CPU generators now run on this Mac for RW, SEIRb and GB. See [the simple simulation setup](SIMULATION_SETUP.md). Data generation is ready; matching the manuscript's trained checkpoints remains unresolved. No NPE training is performed.

Open the [visual HTML guide](codebase-guide.html) in a browser for interactive diagrams and charts, or read the [detailed codebase guide](CODEBASE_GUIDE.md), then [the reproduction plan](REPRODUCTION_PLAN.md). The HTML works offline. Its compact histograms come from our verified local run; regenerate them with `reproduction.nosync/.venv/bin/python baseline-reproduction/refresh_html_charts.py` when the local arrays match the recorded summary hash. Random Walk is the first execution target; SEIRb and Generalized Bass follow after source/runtime validation. Earlier results recorded on another Mac are historical evidence, not results from this setup.

## Environment

This thesis uses an isolated native Python 3.12 environment. The general-purpose `python3` installation is separate. Initial compatibility pins use PyTorch 2.6.0 and sbi 0.23.3; the recovered historical Intel runtime used a different PyTorch version. The exact installed native environment is recorded in `environment-lock.txt`; restore it with the sync command below.

From the repository root:

```sh
uv venv --python 3.12 reproduction.nosync/.venv
uv pip sync --python reproduction.nosync/.venv/bin/python baseline-reproduction/environment-lock.txt
python3 baseline-reproduction/prepare_source.py --model RW
reproduction.nosync/.venv/bin/python baseline-reproduction/inspect_checkpoint.py
reproduction.nosync/.venv/bin/python baseline-reproduction/run_original_observation.py
```

Only hash-verified artifacts from the pinned author repository are loaded. Checkpoints are Python pickles and must not be replaced with untrusted downloads. No upstream training or Vensim wrapper is automatically executed by these commands.

Open `Thesis.code-workspace` in VS Code to use the configured local thesis interpreter. GitHub saves contain our scripts, configuration and compact evidence; local inputs, environments and posterior arrays are excluded.

## Verified on this Mac

On 18 November 2026, the native ARM64 environment loaded all three original RW checkpoints and produced 10,000 valid draws within the VOC-box support imposed by our harness per checkpoint. Original-feature recomputation passed with maximum absolute error 0.0000057221. See [the compact run summary](original-observation-summary.json) for versions, seed, timing and parameter intervals.

This is a single saved-example inference rerun. No new training or forward simulations were performed, and aggregate paper replication remains pending.

To create a local figure from our generated samples:

```sh
reproduction.nosync/.venv/bin/python baseline-reproduction/plot_original_observation.py
```

The generated arrays and image stay under the ignored `reproduction.nosync/results/original-observation` directory. They are not the benchmarking paper or its figures.

## Run inside VS Code

Use **Terminal → Run Task** and select a `Thesis:` task, or use the explicit terminal commands above. The tasks are manual and use the tested local Python executable. The Python/debugger editor extension download encountered a Marketplace DNS error; this does not prevent the workspace tasks or command-line runs.

## Prior alignment remains unresolved

Our current wrapper uses a bounded uniform support from the `.voc` limits. The authors’ notebook entry point constructs a Gaussian prior, while another source path uses a box prior. The saved checkpoint’s exact training provenance still needs to be established. The completed run is a bounded-support compatibility check, not a certified match to every published inference setting. See the [guide’s prior explanation](CODEBASE_GUIDE.md#10-the-unresolved-prior-distinction).
