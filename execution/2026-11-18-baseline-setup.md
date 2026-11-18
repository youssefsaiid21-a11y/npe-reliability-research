# Baseline reproduction setup — 18 November 2026

The current priority is to reproduce matched-model NPE baselines for Random Walk, SEIRb and Generalized Bass before misspecification experiments.

## Completed setup

- Synchronized the thesis checkout with the existing GitHub remote without overwriting local user work.
- Added standing instructions to save completed thesis work to GitHub.
- Added an explicit prohibition on publishing the benchmarking paper, appendix, ZIP and extracted manuscript text. No such files are tracked; ignore rules were checked.
- Created a separate `baseline-reproduction` folder with a reproduction plan, target register, pinned source manifest, configuration, verified artifact preparation and CPU checkpoint adapters.
- Verified all 19 required Random Walk source/model/checkpoint files against the pinned Git blob hashes. Original author files remain unchanged.
- Created an isolated native Python 3.12 environment under the ignored `reproduction.nosync` directory. Scientific dependencies installed successfully. The native interpreter is Python 3.12.16, with torch 2.6.0, sbi 0.23.3, NumPy 1.26.4 and SciPy 1.12.0. Exact transitive versions are saved in `baseline-reproduction/environment-lock.txt`.
- Opened the plan, configuration and runner in VS Code with a thesis-specific workspace and interpreter setting.

## Scientific status

The original saved-observation inference check passed for RW 1k/10k/100k checkpoints, with 10,000 finite in-prior posterior draws each. The saved features matched recomputation within 0.0000057221. No fresh forward simulation, training run or aggregate paper reproduction has been completed. Historical Intel results remain separate evidence. The Mac Vensim PLE application is installed; the author's Windows DLL/venpy batch wrapper remains an unresolved simulator route on this Mac.

The first executable milestone is hash-verified original-observation inference for the three RW training budgets. That check must not be described as aggregate manuscript reproduction.

## Fresh local evidence

`baseline-reproduction/original-observation-summary.json` records our fresh native run. Original inputs, posterior arrays and our generated comparison image remain local under `reproduction.nosync/results/original-observation`. The figure was generated from our rerun and does not copy or load a paper figure. The benchmarking paper and extracted manuscript remain excluded from Git.

VS Code is open with the thesis workspace. The Python-environment extension installed, but the Python/debugger/Pylance bundle could not finish because `ms-python.gallery.vsassets.io` failed DNS resolution. Native command-line inference works. Manual workspace tasks provide source verification, inference and plotting without requiring those optional editor extensions; no task runs automatically on folder open.
