# Reproduce the published baseline first

Scope: NPE baselines for the thesis models — Random Walk first, then SEIRb and Generalized Bass. Keep misspecification experiments on hold until baseline compatibility and calibration are established.

## 1. Freeze and verify the reference

Use the authors' public repository at revision `d74a866f0476c07e93d786d297f90c0706281c33`. Check source/model/checkpoint hashes against `source-manifest.json`. Preserve original files; adapters live in this folder. Keep third-party binary assets and the runtime in the ignored `reproduction.nosync` folder.

Separate the GitHub notebook's saved output, the manuscript's numerical targets, the OSF release and any earlier readable Random Walk models. They are not interchangeable baselines. Private manuscript material stays local.

## 2. Verify original-observation inference

Load the three original Random Walk checkpoints (1k/10k/100k) on CPU, preserving their weights and embedded observation. Record the runtime, versions, seed, prior, parameter ordering and hashes. Generate 10,000 posterior draws per checkpoint, check shape/finiteness/prior support, and recompute the authors' manual statistics against the saved features.

This is a compatibility/inference check on one saved example. It does not establish aggregate coverage or reproduce a paper table.

## 3. Establish the original prior and forward simulator

Resolve the prior/support distinction first: our compatibility harness uses VOC-box support, but the notebook’s `run_training()` path constructs a Gaussian prior. Other source paths use a box prior. Determine the saved estimators’ training lineage and the role of clipping before describing our wrapper as a matched prior. Do not change the recorded numerical samples while investigating this discrepancy.


Validate the checkpoint-compatible compiled RandomWalk3 model and the authors' preprocessing. Vensim PLE is installed on this Mac; its presence does not make the authors' Windows DLL/venpy batch wrapper available on macOS. A supported Windows Vensim runtime is the direct route for that wrapper. An author-source Python port is a separate route requiring documented equation, noise and numerical equivalence checks before use.

Do not substitute the earlier RandomWalk2 model or the OSF correlated-noise release and call it a matched baseline. No Windows machine, license purchase or generic simulator is silently assumed.

## 4. Reproduce validation and metrics

Audit `generate_true_parameters`, simulator filtering/selection, original priors, noise handling and parameter subsets before freezing the independent validation suite. The pinned notebook configures 500 validation datasets and 10,000 posterior draws per dataset. Match the actual manuscript target definitions and aggregation rather than importing those settings as proof of the paper's experimental conditions.

Create a target register with: source location, model/release, estimation method, training budget, parameter subset, metric definition, published value and uncertainty/rounding tolerance. Compare parameter and outcome metrics separately, retain failures and selection counts, and report Monte Carlo uncertainty. No numerical paper target has yet been certified from the private manuscript.

## 5. Expand and decide readiness

After Random Walk passes the compatibility and matched-model gates, repeat for SEIRb and Generalized Bass. Only then return to the frozen-estimator misspecification design. Do not infer model robustness from an original-example rerun or a cross-release pilot.

## Evidence and progress

- `source-checks.json`: verified local public artifacts.
- `baseline-config.json`: starting allocation/configuration; not experimental findings.
- `environment-lock.txt`: exact installed native environment when setup completes.
- `original-observation-summary.json`: created only after a successful fresh inference run.
- `../execution/`: dated progress and unresolved runtime constraints.

Commit project-authored scripts and compact validated summaries to GitHub after meaningful work. Never publish private manuscripts, raw conversations, credentials, environments, model binaries or large posterior arrays.

## References

- [Authors' pinned repository](https://github.com/ali-akhavan89/SD_Neural_Estimation_Benchmarking/tree/d74a866f0476c07e93d786d297f90c0706281c33)
- [Vensim DLL platform documentation](https://www.vensim.com/documentation/dss_dll.html)
- [venpy limitations](https://github.com/pbreach/venpy#limitations)
