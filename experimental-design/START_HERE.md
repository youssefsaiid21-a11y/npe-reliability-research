# Recommended experiment in practical terms

- Confirmed models: Random Walk, SEIRb, Generalized Bass. Read EXPERIMENTAL_DESIGN.md for the authoritative reviewed protocol.
- Explicit source requirement: use the Rahmandad/Akhavan GitHub Vensim simulator assets, wrapper, preprocessing and compatible checkpoints as the baseline. Verify runtime/source compatibility; obtain editable equations for structural changes. Do not substitute generic textbook simulators.
- Use one fixed, baseline-trained NPE estimator per model. Keep parameters, prior, observation grid, preprocessing, and estimator fixed across conditions.
- RW: baseline plus process-shock correlations 0.25, 0.50, 0.75, only if baseline iid Gaussian shocks are source-confirmed. Preserve shock variance and startup distribution.
- SEIRb: baseline one-stage risk-perception lag versus 2/3/4 stages at the same mean perception delay, only if the baseline lag is source-confirmed first order.
- GB: baseline linear imitation versus three increasing curvature settings, preserving imitation strength at 50% adoption, marketing inputs, and source noise.
- Each perturbation must nest the exact baseline. None should merely move a freely estimated parameter within its training prior.
- First run engineering checks, then 50 paired pilot dataset IDs per model. One unchanged candidate across all models costs 600 condition-datasets. Material redesign requires a new allocation pilot.
- Pilot determines the fixed main allocation: 1,000–3,000 independent IDs per model, four conditions per ID (12,000–36,000 main condition-datasets overall). The rule targets approximate adjusted +/-3-point precision for the three prespecified largest-departure coverage contrasts, with a declared cap fallback.
- Start posterior-draw testing at 2,000 draws per dataset; compare 10,000 on prespecified pilot cases. More posterior draws do not substitute for independent datasets.
- Primary parameters: RW Drift; SEIRb transmission Beta; GB imitation q. All other parameters get separate secondary results.
- Measure parameter-wise coverage, width, recovery error, and failures. Primary paired coverage uses IDs successful in all four conditions; always co-report all-ID successful-and-covers performance so selection cannot hide failures.
- Add costly predictive simulations and validated reference inference on prespecified small subsets. Do not let secondary analyses consume the budget before the core comparison works.
- Keep SBC for matched-model validation. Under altered models, report truth coverage and robustness. A wrong-model posterior can fail truth recovery even if NPE approximates that posterior perfectly.
- Keep one checkpoint's conclusions conditional on that checkpoint. Do not claim general NPE behavior, real-world prevalence, or a causal training-budget effect from this minimum study.
- Implement in order of verified source availability; proposed default RW -> GB -> SEIRb. Respect two CPU threads and preserve upstream files.
- The independent reviewer accepted the design after three rounds. The experiment has not been run; source/version mismatches and baseline calibration must be resolved first.

Files:

- EXPERIMENTAL_DESIGN.md — full protocol, equations, allocation, metrics, failure treatment, reference strategy, source gates and citations.
- REVIEW_LOG.md — independent review issues and resolutions.
- planning.py — dependency-free allocation calculator. With no arguments, prints hypothetical scenarios. With `--pilot-total 50 --common-success S --discordant D`, computes allocation from real pilot counts after the protocol is frozen.
- planning-checks.json — arithmetic verification only, not scientific validation.
