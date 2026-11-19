# NPE Reliability Under Model Misspecification

An experimental design for evaluating how a frozen neural posterior estimator behaves when a dynamic model differs from the simulator used during training.

**Status: author CPU data generation works for all three thesis models; paper replication remains pending.** We generated 500 Random Walk datasets and 16 each for SEIRb and Generalized Bass, paired with known parameters. Small fixed-NPE diagnostics are recorded separately. Matching pretrained weights to the manuscript model release remains unresolved. We will not train or retrain NPE. See [the simple simulation setup](baseline-reproduction/SIMULATION_SETUP.md).

## Start here

| Artifact | Purpose |
| --- | --- |
| [Practical overview](experimental-design/START_HERE.md) | Models, perturbations, evaluation and execution order |
| [Full protocol](experimental-design/EXPERIMENTAL_DESIGN.md) | Prespecified design, assumptions and analysis |
| [HTML overview](experimental-design/experimental-design.html) | Download and open locally for a visual walkthrough |
| [Review log](experimental-design/REVIEW_LOG.md) | AI-assisted design review, revisions and limitations |
| [Sample-size planner](experimental-design/planning.py) | Allocation arithmetic with hypothetical or supplied pilot counts |
| [Runtime assessment](execution/2026-11-12-runtime-assessment.md) | Simulator compatibility and execution prerequisites |

## Research question

How do controlled changes to a simulator affect parameter coverage, uncertainty width, recovery error and inference failures when the estimator remains fixed?

The selected models are Random Walk, behavioral SEIR (SEIRb), and Generalized Bass. Each comparison preserves the baseline parameterization and changes a prespecified mechanism. Conclusions remain conditional on the validated simulator and checkpoint.

## Run the planning tool

The planner uses Python's standard library; it does not require the simulator or a machine-learning environment.

```sh
python3 experimental-design/planning.py
```

The default output is explicitly hypothetical. For allocation from observed pilot counts:

```sh
python3 experimental-design/planning.py --pilot-total 50 --common-success 48 --discordant 5
```

These calculations are planning checks, not experimental findings.

## Simulator and provenance

The intended baseline is the [Rahmandad/Akhavan simulator pipeline](https://github.com/ali-akhavan89/SD_Neural_Estimation_Benchmarking), using its Vensim assets, original preprocessing and compatible frozen checkpoints. The recorded checkpoint-compatible revision is `d74a866f0476c07e93d786d297f90c0706281c33`.

Runtime/source compatibility, editable equations and matched-model calibration must be verified before experiments. See the full protocol for the execution gates and scientific references.

## Scope

This repository contains project-authored protocol material, planning code and progress notes. Private conversations, unpublished source manuscripts, third-party checkpoints, installers, credentials and large environments are excluded. Historical assessments retain their original dates and do not establish that an experiment has since run.

## Current execution priority

Reproduce the matched-model baseline before misspecification experiments. Start with the [baseline reproduction setup](baseline-reproduction/README.md) and [execution plan](baseline-reproduction/REPRODUCTION_PLAN.md). The benchmarking paper, appendix, archive and extracted manuscript text remain local and are not tracked.

For a file-by-file explanation and the unresolved prior/support distinction, read the [codebase guide](baseline-reproduction/CODEBASE_GUIDE.md).
