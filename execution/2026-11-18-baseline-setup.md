# Baseline reproduction setup — 18 November 2026

The current priority is to reproduce matched-model NPE baselines for Random Walk, SEIRb and Generalized Bass before misspecification experiments.

## Completed setup

- Synchronized the thesis checkout with the existing GitHub remote without overwriting local user work.
- Added standing instructions to save completed thesis work to GitHub.
- Added an explicit prohibition on publishing the benchmarking paper, appendix, ZIP and extracted manuscript text. No such files are tracked; ignore rules were checked.
- Created a separate `baseline-reproduction` folder with a reproduction plan, target register, pinned source manifest, configuration, verified artifact preparation and CPU checkpoint adapters.
- Verified all 19 required Random Walk source/model/checkpoint files against the pinned Git blob hashes. Original author files remain unchanged.
- Created an isolated native Python 3.12 environment under the ignored `reproduction.nosync` directory. Scientific dependencies are being installed; their successful imports and the checkpoint runner still require validation.
- Opened the plan, configuration and runner in VS Code with a thesis-specific workspace and interpreter setting.

## Scientific status

No fresh forward simulation, training run, saved-observation inference or aggregate paper result has yet been completed by this setup. Historical Intel results remain separate evidence. The Mac Vensim PLE application is installed; the author's Windows DLL/venpy batch wrapper remains an unresolved simulator route on this Mac.

The first executable milestone is hash-verified original-observation inference for the three RW training budgets. That check must not be described as aggregate manuscript reproduction.
