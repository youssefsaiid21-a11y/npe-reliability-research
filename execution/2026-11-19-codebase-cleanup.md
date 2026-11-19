# Our simpler code path

We now start with `./thesis`, which always uses our tested local Python environment.

- `generate` runs the authors' data generator through our coordinator.
- `show` reads one checked time-series/parameter pair.
- `files` explains the execution order.
- `status` separates working data generation from the pending manuscript NPE.
- `learn` builds and opens our local visual lesson, using actual checked examples.

We split the coordinator into named functions for source verification, simulation, output checks and recording. Progress is streamed to the terminal and log. Each run reserves a fresh folder; failures receive an explicit failure status. We added checks for inconsistent series columns, repeated parameter names and incorrect dataset numbering.

Our lesson explains the files first. The known values shown with a graph are simulation inputs, not NPE estimates. The template is shared code; the filled page and raw examples stay under our ignored runtime directory.

Our default VS Code workspace shows the active workflow. The reference workspace retains the broader view. We use a shared voice in our documentation and comments, and preserve the authors' original source.

## Verification

- Ten focused checks passed, including incorrect pairing, invalid truth values, time/column mismatches, non-finite observations, duplicate parameter names, failed simulations and modified lesson inputs.
- Two-case generation completed for RW, SEIRb and GB through the new command.
- All four parameter/data files in each repeated SEIRb and GB run were byte-identical to the corresponding pre-refactor run.
- Existing run folders were preserved; the launcher worked from another directory.
- Browser checks passed for all three models, dataset/series switching, answer reveal/reset, pending-NPE wording, the template fallback and mobile layout.
- Our documentation and comments were checked for a consistent shared voice.

We have not reproduced the manuscript's NPE results. We are waiting for its corresponding pretrained networks and evaluation metadata, and we do not train replacements. No model-misspecification experiment was launched during this cleanup.
