# Independent design review

Date: 12 November 2026. Reviewer: a fresh sub-agent launched without the parent conversation, instructed to inspect the written protocol and local project rules independently. Its role was critique, not implementation. It was explicitly told not to approve merely because approval was requested.

## Round 1: REVISE

Material issues:

1. The primary coverage estimand under failed runs was ambiguous between conditional coverage, pairwise common success, and all-ID operational performance.
2. Sample-size selection lacked a definite precision/power target, protection against noisy pilot variance, and a deterministic budget-cap fallback.

Revisions:

- Defined primary coverage differences conditional on success in ALL FOUR conditions per model, using the same selected population at every departure level. Specified its selection limitation and mandatory all-ID successful-and-covers companion.
- Set an approximate +/-3-percentage-point adjusted interval-half-width planning target, separate from the 5-point practically important coverage-loss scale.
- Added exact-binomial one-sided bounds for pilot success/discordance, a fixed allocation formula, 1,000 minimum/3,000 cap, and prespecified cap/failure behavior.
- Specified paired-data score confidence intervals and boundary handling rather than independent-proportion or zero-width intervals.

## Round 2: REVISE

Remaining issue: pilot-based allocation estimates must refer to the same simulator/interventions/sampling/failure rules as the final test design.

Revision: material redesign requires a new independent allocation pilot under the frozen revised candidate, or justified conservative bounds documented before main testing. Added explicit pilot-revision costs and mechanism-engagement checks without silently excluding weakly informative cases.

The reviewer also inspected the planning helper's binomial inversion direction and allocation formulas and found them internally consistent.

## Round 3: NO UNRESOLVED MATERIAL DESIGN FLAWS

Final reviewer verdict:

> The protocol now coherently specifies interventions, shared coefficient estimands, failure selection, paired comparisons, multiplicity, precision allocation, reference attribution, and computational limits.

Approval is limited to design coherence. It is NOT a guarantee that no undiscovered flaw exists or a claim of empirical validity/execution readiness. Exact source/checkpoint compatibility, simulator semantics, numerical behavior, baseline adequacy, posterior-draw stability, runtime, and reference feasibility remain explicitly unverified execution gates.

## Completed verification

- Read local model/reproduction evidence and currently accessible research chats. User directly confirmed RW, SEIRb, and GB; the latest Overleaf source itself was not read.
- Ran deterministic planning arithmetic and checked Clopper-Pearson boundary formulas against closed forms, plus minimum/cap/failure branches. Results in planning-checks.json.
- No actual model simulations, training, posterior inference, checkpoint unpickling, numerical reproduction, or baseline calibration were run.

The study should proceed to source/engineering verification before any production simulation allocation is treated as approved to execute.

## Simulator-source clarification and HTML overview

The user explicitly required the Rahmandad/Akhavan GitHub simulator. Section 2 now makes its pinned Vensim assets/wrapper/preprocessing/checkpoint pipeline the required baseline, with editable source and runtime checks before alterations. Equal NSeed values alone do not prove aligned noise; a declared independent-noise fallback requires a new allocation pilot and transparent selection/efficiency interpretation.

Reviewer verdict on the clarification: **NO UNRESOLVED MATERIAL DESIGN FLAWS introduced by this clarification.** This did not validate repository execution or source equivalence.

`experimental-design.html` is a standalone local overview of the protocol. No external assets are required; the GitHub citation opens only when clicked. Its HTML structure and relative document links were checked. No model experiments were run.
