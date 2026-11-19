# Research workspace

## Our writing style

We write our project documentation, comments and explanations in a shared voice, using "we" and "our". We use plain language and comments that explain purpose and reasoning. We preserve the authors' original source verbatim so our source checks remain valid. baseline-reproduction/START_HERE.md and the `./thesis` command are the main entry points; older GitHub diagnostics remain separate references.

Save research code, evidence, reports, and results within this Research Project directory. Keep large Python environments and caches in reproduction.nosync to avoid iCloud Desktop sync overhead. Limit numerical work to two CPU threads by default while we work. Preserve upstream source files; put adaptations in separate scripts. Distinguish checkpoint compatibility tests, cross-release experiments, baseline reproduction, and manuscript-level replication. Record seeds, dependencies, source revisions, and unresolved deviations.

## Thesis progress and GitHub

We save thesis progress to GitHub after meaningful completed changes. Use the existing `origin` remote and main branch; inspect changes, commit project-authored code/documentation and compact verified summaries, push, and verify the remote revision. Preserve our uncommitted work. Do not claim a push succeeded without checking.

The remote is public. Do not commit private manuscripts, raw conversations, credentials, third-party checkpoints/model binaries, Python environments or large generated arrays. Maintain the tracked-file allowlist. Keep large local runtime assets in reproduction.nosync.

Current priority: reproduce the authors' matched-model baseline results before running model-misspecification experiments. Start with Random Walk, then SEIRb and Generalized Bass. Keep saved-observation inference, fresh-data calibration, paper-table reproduction and new robustness experiments separate.

## Fixed pretrained NPE only

We use the authors' pretrained NPE without training or retraining. Do not launch author scripts named NPE_CPU.py or NPE_with_manual_sum_GPU.py; those train networks. Use the released CPU simulator/data generators independently. Do not silently connect different simulator/checkpoint releases or crop/relabel observations to claim paper replication. The 19 November setup found working manuscript CPU generators but no saved weights in that replication archive. See baseline-reproduction/SIMULATION_SETUP.md for the current release-alignment issue.

## Explicit private-paper restriction

Do not commit or push the benchmarking paper, appendix, paper ZIP archive, or extracted manuscript text. We agreed on 2026-11-18 that the benchmarking paper must stay local. These files remain local, including when arranging VS Code files or preparing reproduction materials. Check the actual Git index before every thesis push; do not rely solely on ignore rules.
