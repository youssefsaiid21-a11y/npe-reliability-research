# NPE reliability in dynamic models

Research progress for Random Walk, behavioral SEIR, and Generalized Bass.

The current protocol studies controlled misspecification with a frozen neural
posterior estimator. It requires the Rahmandad/Akhavan GitHub simulator pipeline
as the baseline, with source/checkpoint and numerical validation before tests.

- `experimental-design/experimental-design.html`: offline overview.
- `experimental-design/EXPERIMENTAL_DESIGN.md`: full reviewed protocol.
- `experimental-design/REVIEW_LOG.md`: review decisions and limitations.
- `experimental-design/planning.py`: sample-size planning arithmetic.
- `execution/`: runtime investigations and subsequent run reports.

## Current scientific status

The design is reviewed. This session has not yet produced a model simulation,
training run, new posterior inference result, or misspecification result.
Historical reports are separate evidence and are not relabeled as new results.

## Baseline source

https://github.com/ali-akhavan89/SD_Neural_Estimation_Benchmarking

Checkpoint/source-compatible revision currently recorded:
`d74a866f0476c07e93d786d297f90c0706281c33`.

This repository tracks our own code, protocol, and progress. Raw private chats,
unpublished manuscripts, third-party checkpoints, installers, large environments,
and credentials are excluded. Intended remote visibility: private.
