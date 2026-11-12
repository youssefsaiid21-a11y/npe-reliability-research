# Experimental design: NPE reliability under model misspecification

Status: REVIEWED DESIGN — independent reviewer found no unresolved material design flaws after three review rounds on 12 November 2026. Execution prerequisites remain open. No model simulations, posterior inference, or training have been executed for this design. See REVIEW_LOG.md for the review history and its limits.

## 1. Scope and decision

The user confirmed Random Walk (RW), behavioral SEIR (SEIRb), and Generalized Bass (GB) as the three selected models on 12 November. The latest structure is in Overleaf; that document itself has not been read in this session. The local `npe_experiment_vertical.html` independently names the same three models. This protocol addresses the confirmed model set and the existing project aim: reliability of a fixed NPE estimator when the generating mechanism differs from its assumed simulator.

**Recommended core:** one fixed baseline estimator per model; one interpretable misspecification axis per model; baseline plus three departures; paired datasets; parameter-wise uncertainty and recovery. Avoid a large factorial of architectures, sample sizes, perturbations, and training seeds.

**Question:** Under controlled departures in noise persistence, behavioral-delay shape, and imitation-response shape, when do frozen NPE estimates lose truth coverage, become misleadingly concentrated, or compensate through other parameters?

These are different departures, not interchangeable units of severity. There is no presumption that increasing a perturbation monotonically worsens inference. The study is synthetic: it establishes performance under declared mechanisms, not prevalence of these mechanisms in real applications.

## 2. What must be verified before execution

**Simulator choice explicitly requested by the user:** use Rahmandad/Akhavan's published GitHub simulator assets and accompanying pipeline as the baseline, pinned to the checkpoint-compatible revision. Repository: https://github.com/ali-akhavan89/SD_Neural_Estimation_Benchmarking . The intended baseline path is the model's Vensim asset -> `setup_functions/vensim_simulation_model.py` / `venpy` -> original preprocessing -> compatible frozen checkpoint. Preserve upstream files and put adaptations in separate scripts/model copies.

The repository includes compiled `RandomWalk3.vpmx`, `SEIRbNosubs.vpmx`, and `GeneralizedBass.vpmx`, their VOC configurations, and density estimators. A compiled baseline may be executable without exposing the editable equations needed for structural interventions. Obtain author-supplied editable source or a verified equivalent implementation before changing SEIRb delay shape or Bass imitation curvature. Do not silently replace these with generic textbook models. A port or newly trained estimator is an explicitly approved alternative branch, not the default.

The published wrapper assumes a suitable Vensim/venpy runtime and notebook Windows paths; it has not been shown to run unchanged on this ARM Mac. It also modifies zero prior bounds and writes shared sensitivity filenames. Record requested and actually applied parameters, resolve the resulting effective generating distribution, and isolate run directories before concurrency. Validate CPU/runtime and license availability rather than assuming a Python wrapper makes the compiled engine portable.

For common random numbers, equal `NSeed` values alone do not establish aligned streams after a structural change. Verify process/measurement noise-path alignment or use a source-faithful explicit stream interface. If alignment cannot be established, retain paired parameter IDs with independently generated noise, rerun the allocation pilot under that scheme, and disclose that the CRN efficiency claim does not apply. The primary paired estimand and variance-based allocation still apply to that declared joint coupling; this fallback changes the coupling and may change all-four-success selection, so its pilot and population must be reported accordingly.

Keep three evidence levels separate: original-example checkpoint inference; a new controlled experiment; manuscript-level reproduction. The historical report records successful RW original-example inference on a different Intel Mac. Its referenced run scripts, result arrays, lockfile, and `reproduction.nosync` are absent from the currently inspected actual project folder. The six RW/SEIRb checkpoints are present, but current ARM runtime compatibility is unverified.

Create a source manifest for EACH model/checkpoint:

- Pinned source revision, file hashes, actual executable equations, solver and internal step, output times, observation channels/units, parameter order, prior bounds and transformations, initialization, filtering/rejection, process/measurement noise laws, clipping/truncation, preprocessing, and checkpoint configuration.
- Source pin currently recorded: `d74a866f0476c07e93d786d297f90c0706281c33`. Do not combine different releases without equivalence evidence.
- Resolve RW 100 versus 101 observation times; SEIRb 12 VOC parameters versus 11 in the manuscript, and 300 versus 301 time points; GB 9 VOC parameters versus 8 displayed in the manuscript example, and different manuscript/VOC prior bounds.
- Obtain readable executable SEIRb/GB equations or author-approved validated ports. Their public compiled models and labels alone do not establish source equivalence.
- Resolve GB process factor `xi` versus `1+xi` in the simplified manuscript versus general noise definition, and positive-rate handling. Use executable behavior, not an ambiguous figure, as the implementation specification.
- The readable older RW model has multiplicative bounded-normal measurement noise. A standard additive linear-Gaussian Kalman likelihood is NOT automatically a valid reference for that model or the compiled checkpoint model.
- Run checkpoint input-shape, finite-output, parameter-order, prior-support, and deterministic/noise-zero sanity checks. Check normalization near zero means/variances; do not silently replace undefined summaries with zeros.
- Checkpoint priors, preprocessing, and observation dimensions are authoritative for checkpoint inference. A newly trained transparent model is an alternative study branch, not an unnoticed substitution for the original baseline.

**Source gate:** no main simulation until every item needed for the selected model branch is documented. If a source cannot be matched, request the author's simulator or explicitly choose a new implementation and baseline-trained estimator. User/supervisor approval is needed to change the claimed study scope; it is not needed for routine read-only verification.

## 3. Data-generating mechanisms

For model m, let theta_i be a draw from the EXACT training joint's parameter mechanism and u_i its exogenous random streams. Generate x_i,delta = Simulator_m(theta_i, u_i; delta). Run the SAME frozen q_m(theta | x) on every condition. Keep prior, parameter meaning, observation channels/grid, output length, measurement noise, solver, preprocessing, and estimator fixed across conditions unless that feature is the declared intervention.

Parameter/data rejection used in training must be understood. Ordinary SBC requires sampling the same joint distribution the estimator targets, including any selection. Arbitrarily filtering validation trajectories or truncating priors while retaining an unadjusted checkpoint can invalidate its calibration interpretation. Baseline-dependent descriptive strata are allowed but must be labeled conditional summaries.

### RW: process-noise persistence

If the verified baseline has iid Gaussian process shocks with standard deviation sigma_p, use candidate rho = {0, 0.25, 0.50, 0.75}.

First USED shock: eta_1 = sigma_p * epsilon_1.

Subsequent shocks: eta_t = rho * eta_(t-1) + sigma_p * sqrt(1-rho^2) * epsilon_t, t >= 2, with independent standard normal epsilon_t.

Each used shock has variance sigma_p^2; Cov(eta_t, eta_(t-k)) = sigma_p^2 rho^k. At rho=0 the original iid construction is recovered using the same streams. Preserve S0 and whether the first observation occurs before the first state update. This changes persistence rather than marginal shock amplitude; the accumulated-state variance SHOULD change as a consequence of persistence.

Do not initialize eta_0=0 and immediately apply the stationary innovation scale: that creates a rho-dependent early variance deficit. If the actual training model instead has nonzero autocorrelation, smoothing initialized at zero, or materially truncated process shocks, this specific AR experiment does not source-match. Revise this module to nest the actual baseline and re-review before execution; do not silently replace its startup convention or noise law.

Primary target: Drift. Other structural/nuisance parameters remain secondary. Preserve the original measurement-noise equation, even when multiplicative.

### SEIRb: shape of the behavioral perception delay

If the verified baseline risk-perception mechanism is a first-order lag with mean Tp and input D(t), replace ONLY that lag with an n-stage chain:

- dR1/dt = (n/Tp) * (D - R1)
- dRj/dt = (n/Tp) * (R_(j-1) - Rj), j=2,...,n
- The existing behavioral response receives Rn instead of R1.

Candidate n = {1, 2, 3, 4}. n=1 is exactly the original lag; each chain has mean delay Tp and delay-kernel variance Tp^2/n. Initialize every stage at the baseline perception state. Keep the subsequent behavior/reduction mechanism, clinical compartments, noise, mortality reporting, and observations unchanged.

Changing Tp inside its estimated prior is not misspecification; changing the unmodeled delay shape is. Tp retains the meaning of mean perception delay. n is a known DGP intervention, not an extra parameter passed to the old estimator.

Primary target: transmission intensity Beta, subject to source-confirmed naming. Tp, sensitivity Alpha, and risk-reduction time are key secondary compensation parameters.

Check the actual input, lag order, initialization, and solver. For an explicit Euler implementation, n*dt/Tp <= 1 is a useful positivity condition for the isolated lag; complete coupled-model stability still needs testing. Do not alter the baseline integration step merely to make a perturbation run without assessing compatibility. If the original lag is not first order, redesign this module to nest the true baseline before execution.

### GB: nonlinear imitation response

Let f=N/M, where N is latent adopters and M is market potential. The verified simplified baseline includes an imitation pressure p + q*f, with marketing, susceptible-pool, and process-noise multipliers elsewhere.

Replace ONLY p+q*f with p+q*h_a(f), where h_a(f)=f*exp[a*(f-0.5)]. Candidate a={0, 0.5, 1.0, 2.0}.

a is dimensionless; a=0 recovers the baseline exactly. At f=0.5, h_a(f)=0.5 in every condition, anchoring the imitation contribution q/2. q retains inverse-time units and this anchor interpretation. For a>=0 and f in [0,1], the response is nonnegative and increasing. Preserve price/advertising paths and multiplier, market potential, initial adopters, all noise, channels, and timing.

Primary target: imitation q. Secondary: innovation p, initial adopters, and marketing sensitivities. This is a specified change in adoption-pressure shape; an omitted advertising shock would confound an additional mechanism and is postponed.

Only apply h_a to the source model's actual latent adoption fraction. Do not clip f or rates differently by condition. If the original stochastic model can leave [0,M], establish its native boundary rules and the valid extension domain; report those failures rather than inventing new clipping. At a>0, fitted old-model p/q may be compensating pseudo-true quantities; distinguish recovery of generating shared coefficients from optimal fit of the wrong model.

## 4. Estimands and interpretation

- Primary per model: change in the 95% equal-tailed marginal interval's truth coverage for Drift / Beta / q relative to the paired baseline, conditional on a fixed common-success population defined next. Choose these targets before observing test outcomes. Report the other parameters separately, not pooled.
- Let S_i,delta=1 mean a valid observation, preprocessing, and a valid posterior interval were obtained under the frozen retry/sampling rules. Define A_i=1 only if S_i,delta=1 for ALL FOUR conditions in that model. The primary estimand is E[I_i,delta-I_i,0 | A_i=1], with the largest-departure contrast prespecified as confirmatory. The same A population is used at every level within a model. It is a selected joint-success population, not prior-averaged unconditional coverage, and it can change if the intervention set changes. Report its fraction and parameter/dynamics composition. If no common-success cases exist, this estimand is not estimable; report the failure outcome instead.
- Mandatory all-ID companion: O_i,delta=S_i,delta*I_i,delta, defined as zero when no valid interval exists. Report E[O_i,delta-O_i,0] over EVERY scheduled ID, together with each failure category and successful-condition coverage. This is successful-and-covers performance, not Bayesian posterior coverage. Never headline a favorable conditional result while hiding deterioration in this operational outcome. Baseline SBC evaluated only after numerical/inference failures cannot certify unconditional calibration; show their selection explicitly.
- Truth means the shared DGP coefficient with the explicitly preserved meaning above. Do not assign truth to the omitted intervention itself or equate a pseudo-true assumed-model coefficient with the DGP coefficient.
- Misspecified truth coverage is an operational robustness measure. Failure to cover the DGP coefficient is not automatically a numerical failure of NPE: an exact posterior under the wrong model can also miscover it.
- Baseline SBC checks the matched generating distribution. Misspecified ranks may be shown descriptively but are not standard SBC with a uniformity guarantee.
- Conclusions with a single checkpoint are conditional on that checkpoint and its training implementation. Three named checkpoints at different training sizes do not identify a causal training-budget effect.

## 5. Replication and efficient allocation

Use independent, disjoint seed namespaces for engineering checks, pilot, main test, optional diagnostic-threshold calibration, and reference-case selection.

1. Engineering: a small set of deterministic/zero-noise and source-validity checks; never count these as statistical replication.
2. Independent pilot: 50 dataset IDs per model, paired across all four conditions (600 condition-datasets total). Check validity, covariance of paired outcomes, sampling stability, solver resolution, preprocessing, and actual runtime. Pilot outcomes may change the final design, but final settings must be frozen before generating main data; exclude pilot observations from main estimates.
   - Pilot allocation counts must describe the SAME candidate used for main testing. If pilot findings materially change the intervention set, simulator, preprocessing, checkpoint, posterior-sampling rule, retry/timeout rule, or another feature affecting success/discordance, freeze the revised candidate and obtain an independent 50-ID allocation pilot under it. Do not reuse old counts as if generated by the revised design. Alternatively use explicitly justified conservative probability bounds and document that amendment before any main outcomes. Engineering/tuning attempts remain separate and are not pooled into allocation counts. The 600 pilot count is the minimum for one unchanged candidate per model; revisions add pilot cost.
   - Check whether the intervention actually engages its mechanism: RW requires non-negligible process noise; SEIRb requires epidemic/behavioral feedback activity; GB requires enough adoption progression to reveal imitation curvature. Quantify these regimes using prespecified baseline quantities and retain the full intended generating population. Report weakly engaged regimes rather than silently filtering them away or choosing departures that maximize posterior failure. If scientific scope requires a restricted regime, revise the target/training distribution and re-review its calibration interpretation before main execution.
3. Main default: 1,000 independent dataset IDs PER MODEL, each paired across its four conditions. Total 12,000 condition-datasets. Different model parameter spaces use different draws. Baseline is shared across the three contrasts within a model, not regenerated for each contrast.
4. Posterior sampling: candidate 2,000 draws per condition-dataset. Compare with 10,000 on 20 prespecified pilot condition-cases per model. Inspect quantile, interval-inclusion, rank, and metric stability. Pick and freeze a global per-model draw count before the main run; extend it for all cases under a prespecified convergence rule if required, not only cases with attractive results.
5. Optional predictive analysis: preselect 50 main dataset IDs per model independently of outcomes; examine baseline and largest departure with 50 posterior predictive replicates each. That adds 15,000 forward simulations. Fewer replicates are acceptable only if statistics are stable; 50 replicates do not support precise 95% tail-band claims.
6. Reference inference: initially preselect 12 independent main IDs per model, compare all four conditions when feasible (144 reference fits). This is a diagnostic subset, not a population estimate. Prototype on two cases per model before committing, since reference inference may dominate cost.

Posterior draws are NOT independent data-generating experiments. Increasing 2,000 to 10,000 draws cannot replace increasing independent dataset IDs.

At 1,000 successful independent cases, coverage MCSE near c=.95 is approximately .0069 (0.69 percentage points), and worst-case near c=.5 is .0158. These are single-cell MCSEs, not uncertainty of paired differences or simultaneous bands. Use the actual denominator N_A for common-success inference.

For a paired coverage contrast within A, define D_i=I_i,delta-I_i,0. Estimate mean(D) and SE=sd(D)/sqrt(N_A). Primary intervals use a paired-binary score interval (Tango or Newcombe's PAIRED-data method), with per-model alpha=.05/3 for the three confirmatory contrasts. Do not use an independent-proportions formula or a zero-width Wald/bootstrap interval when discordances are zero. Validate the interval implementation on boundary cases before analysis. For secondary continuous metrics, use dataset-level paired resampling; preserve all conditions and parameters from one ID together. Shared baseline contrasts are correlated. Never bootstrap posterior draws as if they were different datasets.

Freeze this deterministic precision rule BEFORE the pilot: target an approximate adjusted 95% interval half-width h=.03 for the primary coverage difference; .05 is the practically important coverage-loss scale, NOT a claim of 80% power. With pilot size n=50, let s=n_A be common-success IDs and d the number of those IDs whose coverage indicators disagree between baseline and the largest departure. Let delta=.05/6. Compute L_A as a one-sided Clopper-Pearson lower bound for success probability: BetaQuantile(delta;s,n-s+1), with zero lower bound at s=0. Compute U_r as a one-sided Clopper-Pearson upper bound for conditional discordance probability: BetaQuantile(1-delta;d+1,s-d), with upper bound one at d=s. The six one-sided allowances cover three success probabilities and three discordance probabilities against noisy pilot estimates. Since Var(D|A)<=E[D^2|A]=r, U_r is a conservative variance planning bound. Use the exact-binomial bounds rather than an overly conservative generic bounded-variable bound to avoid needlessly increasing the main simulation count.

Let z=Phi^-1(1-.05/(2*3)) (approximately 2.394), K_required=ceil(z^2*U_r/h^2), and N_required=ceil(K_required/L_A). For each model choose N_m=max(1000,min(3000,N_required)). If n_A<25 or L_A=0, stop the feasibility gate and revise the pipeline; do not launch main testing. This rule plans for common-success loss and declares a minimum, cap, and finite-pilot uncertainty safeguard. It is an approximate precision allocation, not an exact guarantee on future score-interval width.

If N_required>3000, run exactly the predeclared cap only if the study can accept precision-limited inference; label that model accordingly before main data generation and report predicted and achieved widths. If the cap is unaffordable, stop and revise this protocol before generating main data rather than silently lowering N. No repeated outcome-dependent extension of the main test is permitted. Primary inference is three prespecified largest-departure contrasts; intermediate levels and other parameters are descriptive with clearly labeled pointwise intervals. Report score-interval achieved widths even if the planning target is missed.

Use common parameter draws and separately aligned process and measurement base streams across conditions; different dataset IDs remain independent. Split seeds by purpose so extra draws from a changed module do not shift measurement or unrelated process streams. For an n-stage deterministic lag, reuse the same exogenous streams; do not invent additional independent noise streams. CRN is not guaranteed to lower variance: inspect paired covariance in the pilot and retain pairing for interpretation even if its efficiency gain is absent.

## 6. Outcomes and uncertainty

- Primary: 95% marginal truth coverage and paired change for the three target parameters.
- Secondary parameter-wise: 50/80/90/95% coverage curves, interval width divided by original prior width, bias and RMSE normalized by prior width, posterior SD, and standardized error where posterior SD is finite and positive. Zero/undefined posterior SD is a failure/flag, not a divide-by-zero replacement.
- Misleading concentration: show coverage and width jointly. Optionally report the prespecified event “truth outside the 95% interval AND interval width less than half the prior width,” with sensitivity at one-quarter and three-quarters. This is an operational descriptive flag, not a universal definition of overconfidence.
- Show structural and noise parameters separately; no pooled coverage across incomparable parameters. Report prior averages and prespecified baseline-dynamics strata (e.g. informative diffusion versus nearly flat); do not select strata after seeing posterior failures.
- Predictive subset: use the baseline simulator at posterior draws to assess adequacy of the model the estimator actually assumes. Compare independent posterior predictive summaries against the observed trajectory: RW increment autocorrelation and trend; SEIRb peak timing/size and interwave structure; GB adoption-rate peak and early/late adoption shape. Calibrate statistics relative to source units and independent pilot scales. Do not substitute latent unobserved quantities for observable diagnostic claims.
- These predictive checks reuse observed data and are descriptive; do not label raw posterior predictive p-values as uniform frequentist tests. A detection claim needs separate null-calibration data and final held-out false-positive/true-positive evaluation. Detection/repair is optional, not required for the main thesis.
- Record simulator failure, nonphysical state, undefined preprocessing, posterior exception/timeout, invalid support, and reference failure separately. Preserve every scheduled ID, runtime, retries, and reason. Freeze the retry/timeout budgets in the pilot; a valid source-defined trajectory boundary is not a numerical failure merely because it looks unusual.
- Report failure rate on all scheduled IDs; condition-specific truth coverage among successful runs; the primary ALL-FOUR common-success contrasts; and all-ID successful-and-covers probability/contrasts defined in section 4. Keep these denominators explicit. Condition-specific successful coverage is not a substitute for the primary paired population.

## 7. Reference inference and error attribution

Three distinct quantities:

- q0(theta|x): frozen NPE posterior.
- p0(theta|x): posterior under the exact SAME assumed baseline model/prior/observations as the checkpoint.
- p_delta(theta|x): posterior under the actual altered DGP with its intervention known (optional richer-model comparator).

q0 versus p0 isolates approximation discrepancies for that dataset. p0 versus truth evaluates the wrong model's inferential behavior. p_delta is a separate comparison with different assumptions; do not call it the same reference.

Only build a likelihood/reference when the exact transition, observation, truncation, and boundary laws are established. A standard Kalman filter is admissible only for a separately verified additive linear-Gaussian model, not merely because the model is called Random Walk. For intractable stochastic likelihoods, a validated particle likelihood and particle-MCMC may be possible but are a distinct expensive task. Generic Gaussian residual least squares is not a gold-standard posterior for a different noise law.

Validate any MCMC reference with dispersed chains, rank-normalized R-hat (target <1.01), bulk/tail ESS (starting target >=1,000 for reported scalar quantities), MCSE relative to the contrast of interest, mixing/trace checks, and likelihood-estimator stability if particle methods are used. These are screening targets, not proof of a correct posterior. Retry budget and failed-reference reporting must be fixed. Never delete difficult cases or call a nonconverged fit ground truth.

Compare quantiles/marginal distributions and selected joint dependence; an optional cross-validated classifier two-sample test is diagnostic, with matched sample sizes and its power limitations declared. Predictive fit alone does not verify parameter recovery; SBC marginal ranks alone do not establish full posterior correctness.

If a valid reference cannot be obtained within budget, the main robustness study can still run after source/baseline checks, but claims must be about the frozen estimator-plus-assumed-model pipeline. Attribution of a failure specifically to neural approximation remains unresolved. A separate linear-Gaussian toy can validate analysis code, but cannot substitute for a reference to the selected source model.

## 8. Numerical and baseline acceptance gates

- At intervention zero, the perturbed simulator must recover the source baseline under common streams to solver/source tolerance, including startup and noise conventions. Test this before sampling a large prior ensemble.
- Check state conservation and boundaries: SEIR population balances and nonnegative clinically meaningful rates; Bass stock-flow balance, adopted/potential population rules; RW zero-noise drift and declared measurement behavior. Preserve stochastic source semantics; do not silently repair negative values differently across conditions.
- Compare dt and dt/2 on prespecified pilot trajectories for every intervention. If the original discretization is part of a checkpoint's training distribution, report its numerical behavior and separate source-reproduction from a refined continuous-time experiment. Do not claim a refined port is checkpoint-equivalent without evidence.
- Baseline checks: parameter-wise SBC/rank plots with Monte Carlo uncertainty, coverage-width/recovery summaries, and available reference checks. Uniformity alone can miss an uninformative prior-like estimator.
- Predeclare an operational adequacy tolerance for key targets with supervisor input; an initial candidate is +/-3 percentage points around 95% prior-averaged coverage, accompanied by Monte Carlo intervals. N=1,000 cannot establish extremely tight calibration across every parameter. If baseline reliability is insufficient, investigate/fix the baseline before attributing new failures to misspecification.
- If the original checkpoint cannot meet the declared baseline scope, either narrow claims to its observed conditional behavior or train a new baseline estimator on a transparent simulator. Any new architecture/hyperparameter/prior decisions use only training/validation/pilot data; freeze before final test.
- If using one checkpoint, report checkpoint-conditional conclusions. Optional independent baseline training seeds (e.g. three) can assess training variability; they are not silently averaged as independent test datasets and do not have to be included in the minimum study.

## 9. Compute plan and artifacts

Preserve upstream code; implement adaptations separately under this project. Put large environments/caches in `reproduction.nosync`. Respect the project default of two CPU threads. Benchmark on this ARM device; do not extrapolate the old Intel report to SEIRb or model training.

Cache immutable theta, seeds/streams, raw observations, preprocessing version, checkpoint hash, exact simulation settings, and per-ID metrics. Resume at ID boundaries; append results atomically and avoid shared simulator filenames. Source Vensim wrapper uses common working-directory filenames, so isolate work directories before any concurrency.

Save posterior draws in float32 only if appropriate for summaries; keep sensitive computations/quantiles at suitable precision. With 2,000 draws, N=1,000 per model, four conditions and parameter counts 4/12/9 (the latter two provisional source mappings), draw arrays alone require about 800 MB total before metadata/compression; all models at the 3,000 cap require about 2.4 GB. Recompute arithmetic after source mapping. Pilot and predictive/reference results have separate directories.

Estimate wall time by summing measured simulation, preprocessing, inference, predictive, reference, and I/O costs. Posterior predictive/refit work can dominate the apparently cheap core. Set a budget before main execution; cut optional predictive/reference breadth before reducing independent replication without acknowledging its uncertainty cost. Never promise a runtime before measuring the actual models.

Minimum outputs: source manifest, frozen configuration, pilot report, dataset registry and failure log, parameter-wise coverage/width and recovery results, paired contrast tables with MC uncertainty, perturbation-response figures, predictive/reference subset report if feasible, and explicit limits on attribution/generalization.

Suggested implementation order: source verification -> RW pipeline and pilot -> GB module and pilot -> SEIRb module and pilot -> freeze common analysis/N -> main runs -> subset diagnostics -> interpretation. This orders the initial engineering from simpler to more coupled models, subject to which validated simulators are available. Stop early at a failed gate rather than training or simulating a large unmatched model.

## 10. Public methodological sources

- Morris, White & Crowther (2019), ADEMP and Monte Carlo uncertainty: https://discovery.ucl.ac.uk/id/eprint/10066118/1/2019%20-%20Morris%20-%20simulation%20studies%20tutorial%20-%20stat%20med.pdf
- Talts et al., SBC: https://arxiv.org/abs/1804.06788
- Modrak et al., SBC limitations and extensions: https://arxiv.org/abs/2211.02383
- Glasserman & Yao, common random numbers: https://business.columbia.edu/sites/default/files-efs/pubfiles/4261/glasserman_yao_guidelines.pdf
- Lueckmann et al. (2021), SBI benchmarking and reference comparison: https://proceedings.mlr.press/v130/lueckmann21a.html
- Ward et al. (2022), robust NPE/misspecification: https://papers.neurips.cc/paper_files/paper/2022/hash/db0eac6747e3631eb91095cd76065611-Abstract-Conference.html
- Vehtari et al., MCMC diagnostics: https://arxiv.org/abs/1903.08008
- Tango (1998), score intervals for paired proportions: https://pubmed.ncbi.nlm.nih.gov/9595618/
- Newcombe (1998), PAIRED binary proportions (not the independent-proportions paper): https://pubmed.ncbi.nlm.nih.gov/9839354/
- Public pinned implementation: https://github.com/ali-akhavan89/SD_Neural_Estimation_Benchmarking/tree/d74a866f0476c07e93d786d297f90c0706281c33

Private local manuscript/evidence are retained locally. The manuscript says not to cite or distribute it without author permission; use it to resolve private implementation details and cite public primary methods for an outward-facing protocol.
