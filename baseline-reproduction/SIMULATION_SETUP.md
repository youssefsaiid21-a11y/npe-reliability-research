# Generate data, then estimate parameters

**Data generation works on this Mac for all three models. Paper reproduction has not passed yet. No NPE training is performed.**

There are two separate tools:

1. The mechanistic model receives known parameters and generates a time series.
2. The authors' pretrained NPE receives that time series and estimates the parameters.

We have working data generators and working GitHub NPE checkpoints. We still need to match the model release and trained network used for the paper results.

## What actually ran on 19 November 2026

| Model | Datasets from the manuscript authors' own CPU code | Values per observed series |
|---|---:|---:|
| Random Walk | 500 | 101 |
| SEIRb | 16 | 301 |
| Generalized Bass | 16 | 301 |

Each dataset is paired with known parameters. Output counts, finite values, time axes, parameter pairing and parameter bounds were checked. Repeating an eight-case Random Walk run produced 16 byte-identical parameter/data files.

These generators retain the authors' default screening and, where enabled, stratification. The engine is the authors' released PySDTorch: Torch CPU for Random Walk and Generalized Bass, C++ CPU for SEIRb. One worker process and at most two numerical threads are used.

## Run it

From the project root, use a **new output directory**:

```sh
reproduction.nosync/.venv/bin/python baseline-reproduction/run_author_generator.py \
  --model RW --cases 500 --seed 31 \
  --output reproduction.nosync/results/our-new-RW-run
```

Change `RW` to `SEIRb` or `GB` for the other models. Start with a small case count when checking a change.

Our shorter command is `./thesis generate --model RW --cases 16`. It chooses a fresh run folder automatically. In VS Code, **Terminal → Run Task → Our thesis: generate data** asks for the model, case count and an optional run-folder name. [Our file guide](START_HERE.md) explains the execution path.

Outputs:

- `true_data/true_params/true_param_v01.cin`: known parameters for dataset 1.
- `true_data/true_series_tab/data_v01.tab`: its time series, readable as a table. GB files also include driving inputs.
- `generator.log`: execution log.
- `generation-receipt.json`: source identity, settings and paired file fingerprints.

The wrapper checks the local archive and author source files before running **only** `synth_data_gen_pysdtorch.py`. Do not run the archive's `NPE_CPU.py` or `NPE_with_manual_sum_GPU.py`: those are training programs, outside our fixed-estimator study.

## Do we need Vensim?

**Not for these working author CPU generators.** They execute the equations through Python/Torch or compiled C++.

Vensim remains useful for inspecting diagrams and checking CPU simulation against the original engine. PLE can export the packages, but its subscript restriction prevents native SEIRb simulation. A university Professional/DSS license may help with native validation. The original GitHub `venpy` DLL wrapper needs a compatible Windows DLL setup; Mac software alone would not make it work unchanged.

## Why paper reproduction is still pending

The GitHub weights and manuscript replication archive are different releases:

- GitHub Random Walk uses 100 observation times and independent process shocks; the manuscript model saves 101 times and has explicitly autocorrelated process shocks.
- GitHub Generalized Bass estimates nine parameters, including noise correlation time, and uses different parameter limits from the manuscript release, which fixes that correlation time.
- The manuscript ZIP contains code and models, but **no saved network weights**. The public GitHub main branch and releases were checked; no newer release assets were found.
- The paper excludes noise-related parameters from its parameter comparison; the older public entry point scores all parameters and uses a different normalization convention.

Using another release's weights would be a cross-release test. We will not crop observations, relabel parameters or alter priors silently and call that paper reproduction.

The useful request to the professor/authors is: **the pretrained NPE checkpoints used for the benchmarking paper's parameter-recovery table, with matching model version, preprocessing, parameter ordering and posterior construction settings.** We will not train replacements.

## Interim NPE diagnostics

Before locating the authors' manuscript CPU generator, small pilots used exports of the GitHub packages and the fixed GitHub 100k networks: 16 RW cases, eight SEIRb cases and eight GB cases. GB required an explicit, opt-in vendor-formula pink-noise expansion that is not certified against native Vensim.

Those results are in [fresh-pilot-summary.json](fresh-pilot-summary.json). They **do not** use the manuscript datasets above and do not establish agreement with the paper. They preserve their engine/release limitations.

`generate_baseline_data.py`, `evaluate_baseline.py` and `validate_simulators.py` are for those separate diagnostics; `run_author_generator.py` is the primary manuscript data-generation command.

## Source and privacy

- GitHub revision: `d74a866f0476c07e93d786d297f90c0706281c33`.
- Replication ZIP SHA-256: `fdfbe324b0dc260651780ed5b9e813b85ac78f35f36a234002cc7ee9aba54f9d`.
- Native Python 3.12, PyTorch 2.6.0, sbi 0.23.3; installed dependencies in `environment-lock.txt`.
- Upstream files remain unchanged. Archive, exports, models, checkpoints, datasets, private links and papers stay local under ignored paths.
- GitHub receives our wrappers, documentation and compact evidence only.

See [runtime evidence](simulation-environment-summary.json) for counts and checks.
