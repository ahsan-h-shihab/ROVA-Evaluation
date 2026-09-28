# Revision Analysis Plan — REV-001

**Date written:** 2026-09-29.
**Status:** written **after** the MAIN-001 and ABLATION-001 results and the first-round peer
review were known, and **before** any analysis specified here was executed. This is not a
pre-registration of the original study. The commit that adds this file, tagged
`revision-001-analysis-plan`, fixes its content; changes made before execution are recorded
with dates in §10. The exact code that executes the plan is tagged `revision-001-frozen`.

Companion record: [`PREREGISTRATION_DEVIATIONS.md`](PREREGISTRATION_DEVIATIONS.md).

Every analysis below is labelled:
**(R) restored** — committed in the original internal protocol but omitted in the executed
study; **(N) new** — not in the original protocol; **(S) sensitivity** — a re-analysis of
existing data or of existing tuned configurations.

---

## 1. Fixed elements (inherited unchanged)

| Element | Value (source) |
|---|---|
| Implementation | `dtcoa_val/` at tag `ablation-001-frozen` (tree-identical on `main`); **not modified**. All revision code is additive, in `rova_revision/`. |
| Environments | `favorable_hmm_coupled` (N=10), `unfavorable_oscillation_uncoupled` (N=1), `collapse_moderate_coupled` (N=6) (`environments.py`) |
| Cost-ratio grid | c ∈ {0.5, 1.5, 3.5, 8.0, 20.0}, U = 1, H = c |
| Seeds | 0–59 |
| Outer folds | `kfold_split(range(60), 5, default_rng(0))` |
| Inner folds (fold k) | `kfold_split(outer_train_k, 3, default_rng(k))` |
| Tuning budget | K_out = 5, K_in = 3, 80 random-search candidates per outer fold, drawn by the frozen `sample_config` from `default_rng(0)` |
| Selection | minimum mean inner-fold validation regret (`nanargmin`; ties → earliest-sampled candidate) |
| Primary metric | mean per-agent-step authorization regret per scenario instance (per-seed regret) |
| Tail metric | CVaR-95 of per-seed regret per outer fold (co-primary, descriptive) |
| Statistics | frozen `dtcoa_val/analysis/stats.py`: `wilcoxon_paired`, `cliffs_delta`, `holm_bonferroni`, `bca_bootstrap_ci` (5000 resamples, `default_rng(0)`) |
| Negligibility band | \|Cliff's δ\| ≤ 0.147 |
| Comparison rule | A vs B: "A lower-regret" iff Holm-significant Wilcoxon **and** \|δ\| > 0.147 **and** δ(A,B) < 0; symmetric for B; otherwise "no detected difference" |
| Regimes | low = {0.5, 1.5}; transition = {3.5}; high = {8.0, 20.0} |

### 1.1 Analytic facts fixed before execution

- **F1 (oracle structure).** The oracle loss L(a; r, h) = π(a)·h·H + (1−π(a))·r·U with
  π(a) = a/5 is linear in π, so the per-step optimum is always a ∈ {0, 5} (ties → 0).
  Consequently a policy's per-step regret equals |hH − rU|·|π(a) − π(a\*)|. This is a
  property of the evaluation loss, derived from `oracles.py`, not a result.
- **F2 (world seeding).** `run_trial` seeds each simulated world with
  `trial_seed(seed, 0, tag)`. In MAIN-001 and ABLATION-001 the tag contains the method name,
  so different methods were evaluated on different, independently drawn worlds for the same
  seed index. Within one tag, the random stream is action-independent, so two policies run
  with the same (seed, tag) face common random numbers.
- **F3 (regret basis).** Reported regret is computed from the expected per-step loss at the
  realized effective latent state (`runner.py`), not from realized harm events; realized
  events drive the failure dynamics only.

## 2. Execution environment and gates

**E-1. Machine.** All runs execute on the lab workstation that produced MAIN-001:
AMD Ryzen 9 9950X (16 cores / 32 threads), Ubuntu, Python 3.13.14, numpy 2.5.1,
scipy 1.18.0, single-threaded BLAS/OpenMP per process. `rova_revision/env_guard.py`
verifies CPU model, logical-CPU count, host name, interpreter and library versions,
thread variables, the git commit (must equal `revision-001-frozen`) and a clean working
tree. A mismatch aborts the run unless an explicit, written approval is supplied, which is
then recorded verbatim in `run_metadata.json`. No run is executed on any other machine.

**E-2. Gate R0 — reproduction of archived results (must pass first).** On the target
machine, replay all 45 MAIN-001 tuning units' outer-fold-0 test evaluations (540 scenario
instances) with the archived configurations and tags, and require
|replayed − archived fold mean regret| < 1e-9 for every unit. Recompute the archived BCa
intervals for mean per-seed regret for all 45 units and require exact agreement with
`stats_summary.json`. Any failure aborts the run; no revision output is valid without a
passing gate.

**E-3. Timing pilot.** Before the full run, a timing pilot executes the learned-policy and
Q-learning tuning loops for one outer fold, five candidates and one inner fold on
**non-experimental seeds 1000–1059** with tags prefixed `REV1_PILOT`. It is used only for
run-time extrapolation; its regret values are not analysed or reported.

## 3. World-seeding conventions and tag registry

- **Own worlds (frozen convention).** A tuning campaign evaluates its selected configuration
  on outer-test seeds with tag `<campaign_tag>_out{k}_test`.
- **Common evaluation worlds (CEW, new).** For paired comparisons, every method's per-fold
  tuned configuration is additionally evaluated on outer-test seeds of fold k with the
  method-independent tag `REV1_CEW__{env}__c{c}_out{k}_test`. All methods then face
  identical worlds (common random numbers), so Wilcoxon signed-rank pairs are genuine
  matched pairs.
- **Archived worlds.** Where stated, a new policy is also evaluated with an archived
  method's tag, to pair it with that method's archived per-seed results.

| Experiment | Tag pattern |
|---|---|
| R1 own worlds | `REV1__{env}__c{c}__always{0,5}_out{k}_test` |
| CEW | `REV1_CEW__{env}__c{c}_out{k}_test` |
| R3 / R4 campaigns | base `REV1__{env}__c{c}__{learned,qlearn}`; training `…_out{k}_infold{j}_train`, `…_out{k}_final_train`; validation `…_out{k}_hpo{i}_infold{j}`; test `…_out{k}_test` |
| R5a | `REV1_ADV__{profile}__{env}__c{c}_out{k}_test` (run with and without the adversary) |
| R5b | base `REV1_ADVT__{profile}__{env}__c{c}__{method}`; CEW `REV1_ADVT_CEW__{profile}__{env}__c{c}_out{k}_test` |
| R7 | reduced instances `REV1_DP__{env}__N{n}__c{c}`; full-size collapse uses the CEW tags |
| R8 / R10 / R12 | base `REV1__{env}__c{c}__{static_ext, ewma_ext, clairvoyant, rova_single}` |
| R9a | base `REV1_SENS__{K}__{env}__c{c}__{method}`; CEW `REV1_SENS_CEW__{K}__{env}__c{c}_out{k}_test` |
| R11 | archived ROVA test tags `{env}__c{c}__rova_out{k}_test` |
| R13 | archived campaign tags `{env}__c{c}__{method}` (deliberate: same validation and test worlds as the archived random search; outputs written to a separate directory) |

## 4. Experiments

Methods referred to below: **static, ewma, rova** = archived MAIN-001 tuned configurations;
**A0** = always full supervision (level 0); **A5** = always high autonomy (level 5);
**learned**, **qlearn**, **static_ext**, **ewma_ext**, **clairvoyant**, **rova_single**,
**\*_tpe** as defined below.

### R1 (R) — Constant reference policies
Policies A0 and A5 (no parameters, no tuning) evaluated (i) on own worlds, (ii) on the CEW,
and (iii) on each archived method's test worlds, for all 15 cells and all 60 seeds. Every
evaluation records per-agent-step oracle decisions, so the fraction of agent-steps with
a\* = 0 (full supervision) is reported per cell, along each method's trajectory. For every
cell, the **surviving-margin ratio**
G = (R̄_A0 − R̄_rova) / (R̄_best − R̄_rova), with R̄ the CEW mean regret and "best" the lower-regret
tuned baseline in that cell, is reported with a paired BCa interval over CEW worlds; G is
reported as "not defined" when R̄_best ≤ R̄_rova. Interpretation fixed now: G ≥ 1 — the anchor
is no better than the best tuned baseline; 0 < G < 1 — the anchor closes a fraction 1 − G of
the baseline-to-ROVA gap; G ≤ 0 — the anchor is at least as good as ROVA.

### R2 (S/R) — Archived-data and matched-world re-analysis
(a) BCa intervals (mean and median per-seed regret) for every method and cell. (b) Tuning
curves at 20/40/80 candidates for every tuning campaign. (c) Variance decomposition of
per-seed regret into between-outer-fold and within-fold components (one-way random effects;
ICC). (d) **Matched-world re-analysis:** static, ewma and rova tuned configurations
re-evaluated on the CEW; per-cell family {rova–static, rova–ewma, static–ewma} with the
§1 rule; labels compared with the published verdict table. (e) Independent-samples
sensitivity for the published own-world comparisons: two-sided Mann–Whitney U
(`method="asymptotic"`, continuity correction). (f) Holm correction over wider families
(§5.3). (g) Critical-difference diagrams per regime on the CEW (Friedman, Nemenyi).

### R3 (R) — Learned cost-sensitive authorization policy (`learned`)
A policy trained by direct optimization of the evaluation loss, using only the observation
channels available to the threshold baselines.
- **Features (per agent, per step):** missing channels carried forward (initial value 0.6,
  as in the baselines) → z (6); exponential smoothing e = β·e + (1−β)·z with initial 0.6 (6);
  population standard deviation of the composite mean(z) over the last ≤5 steps (1). Degree 1
  uses these 13 features; degree 2 adds all 91 products of pairs (with squares). Features are
  standardized with training-set statistics.
- **Objective:** cost-proportionate weighted logistic regression. Target y = restrict if
  Δ = h·H − r·U > 0; weight |Δ|; L2 penalty on non-intercept weights; L-BFGS-B from zero.
  By F1 this is a convex surrogate of the exact regret of a two-action policy. The latent
  state is used only to compute training targets; the deployed policy sees observations only.
- **Decision:** level 0 if the score is positive, otherwise level 5.
- **Training data:** round 0 rolls out a uniformly random level per agent-step (its own
  random stream, independent of the world stream) on the training worlds; if
  `onpolicy_rounds` = 1, the fitted policy is rolled out once more on the same worlds, the
  data are pooled and the model is refit.
- **Search space (frozen sampler kinds, in this order):** `beta` loguniform-complement
  [1e-3, 0.999]; `degree` int [1, 2]; `l2` loguniform [1e-6, 1]; `onpolicy_rounds` int [0, 1].
- **Protocol:** identical nested cross-validation (§1); training on inner-train seeds,
  validation on inner-validation seeds, refit on the full outer-train set, test on outer-test
  seeds; plus CEW evaluation of each fold's fitted model.

### R4 (R) — Tabular Q-learning (`qlearn`)
- **State:** (bin of composite mean(z), bin of its exponential smoothing with factor β),
  `n_bins` equal-width bins on [0, 1]; one Q-table shared across agents; initial Q = 0.
- **Reward:** −L(a; r, h) for the chosen level (training only). One-step Q-learning with
  learning rate `lr`, discount `gamma`, terminal at the horizon; ε-greedy exploration with
  ε decaying linearly from `eps0` to 0.01 over training; greedy at test (ties → lowest level).
- **Search space:** `n_bins` int [3, 10]; `beta` loguniform-complement [1e-3, 0.999];
  `lr` loguniform [1e-3, 0.5]; `gamma` uniform [0, 0.95]; `eps0` uniform [0.05, 0.5];
  `epochs` int [1, 3]. Same nested cross-validation and CEW evaluation as R3.
- **Stability gate (fixed now):** (G1) every Q-value finite; (G2) every Q-value within
  [−L_max/(1−γ), 0], L_max = max(H, U); (G3) retraining the selected configuration of
  fold 0 in every cell reproduces the Q-table bit-for-bit. If any check fails, `qlearn` is
  reported as having failed the gate and is excluded from comparative tables; its outputs are
  still archived. Training-seed sensitivity (two further exploration seeds per fold) is
  reported but is not part of the gate.

### R5 (N) — Adversarial scenarios (the simulator's three scripted profiles)
Profiles, exactly as defined in the frozen code (`experiments/campaign.py`):
`ramp_defect` (agents {0}, defect_t 150, spike 0.85, fake obs 0.9); `collusion`
(agents {0,1,2}, defect_t 150, spike 0.85, fake obs 0.9); `poisoning` (agents {0,1},
defect_t 0, spike 0.6, fake obs 0.9). A profile is run only in environments that contain
all of its agents: `ramp_defect` in all three environments, `collusion` and `poisoning` in
the two coupled environments (7 environment–profile pairs × 5 cost ratios).
- **R5a (primary; deployment shift):** all methods' clean-tuned per-fold configurations and
  A0/A5 are evaluated on outer-test seeds with the adversary active, and on the same worlds
  without it. Outcomes: regret (primary), harm rate, unauthorized-unsafe rate.
- **R5b (secondary; adversary-aware tuning):** static, ewma and rova re-tuned with the
  adversary present (identical protocol), then evaluated on the adversarial CEW with A0/A5.

### R6 (R) — Measured decision cost
Per-agent-step wall-clock time of `policy.step` for every method (tuned configurations or
fitted models from this run) at N ∈ {1, 6, 10, 100, 1000} agents, 250 steps, fixed synthetic
observations, 2 warm-up and 7 timed repetitions, single thread, with no other workload on
the machine. Reported: median and inter-quartile range, the simulator's own per-step cost
for context, the ratio to static, and the protocol's compute-conditional flag
(ratio > 3). Tuning cost is reported in simulated trajectories per method.

### R7 (R) — Sequential (dynamic-programming) oracle in coupled environments
Given a world's exogenous trajectories (base reliability, base hazard, graph), the only
action-dependent state is each agent's time since its last failure (τ + 1 values per agent).
Exact backward induction over this state, restricted to levels {0, 5} per agent (valid
because the expected cost-to-go is multilinear in the permission fractions), gives the
optimal expected total loss V_opt; exact policy evaluation of the myopic oracle gives
V_myopic.
- **Reduced instances:** N = 2…6 for both coupled environments (for collapse, N = 6 is the
  full evaluated size), 5 cost ratios, 60 seeds. Reported: per-agent-step gap
  (V_myopic − V_opt)/(T·N) and the disagreement rate between the two policies' actions.
- **Full-size collapse environment, method level:** V_opt for every CEW world; each method's
  regret against the sequential reference, (Σ L(a) − V_opt)/(T·N), next to its regret
  against the myopic reference; comparison labels recomputed.
- **Decision rule for the protocol's "substantial disagreement" clause (fixed now):** the
  disagreement is substantial if any cell-level comparison label among static, ewma, rova,
  A0, A5 and learned changes when the sequential reference replaces the myopic one. If so,
  the sequential-reference results are reported as authoritative for that environment.
  The favorable environment (N = 10) is beyond exact solution; only its reduced instances
  are reported.

### R8 (N) — Extended baseline search space (full restriction reachable)
`static_ext` and `ewma_ext`: the frozen search spaces plus `cap_code` int [0, 24], mapped to
a maximum permitted level cap = min(cap_code, 5), i.e. P(no cap) = 0.8 and P(cap = k) = 0.04
for k = 0…4. cap = 0 is exactly always-full-supervision. Identical protocol; reported
separately from the published comparison.

### R9 (N) — Simulator-assumption sensitivity
Three single-factor observation perturbations, currently disabled in all environments:
K1 missing-not-at-random (`p_miss` = 0.10, `mnar` = True);
K2 observation delay (`delay` = 3 steps);
K3 conflicting channel (`conflict_frac` = 1/6, one channel inverted).
- **R9a (primary):** static, ewma and rova re-tuned under each perturbation (identical
  protocol) and evaluated on the perturbation's CEW with A0/A5; per-cell labels (§1 rule).
- **R9b (secondary):** clean-tuned configurations of every method evaluated on the same
  perturbed CEW worlds.

### R10 (R) — Clairvoyant cost-sensitive threshold (`clairvoyant`)
Exactly as designed before MAIN-001: `OracleLite` (thresholds on true reliability, blind
to hazard) with `CLAIRVOYANT_SEARCH_SPACE`, tuned per cost ratio by the frozen
`nested_cv_hpo`. A reference expressing available headroom, not a competitor.

### R11 (R) — Penalty-magnitude sweep (behavioral containment)
For every cell and outer fold, the archived ROVA configuration is evaluated on that fold's
outer-test worlds (archived tags) with `prop_scale` swept over
{0, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 5} ∪ {tuned value}, other parameters fixed at their
tuned values; likewise for `stab_scale`; plus both set to 0. The tuned point must reproduce
the archived per-seed test regrets (< 1e-9), which checks the sweep. Reported: dose–response
curves and, per cell, tuned-versus-zero comparisons over the 60 paired instances (§1 rule).

### R12 (R) — Band-vector ablation (`rova_single`)
ROVA with the tuned 5-band vector replaced by a single tuned threshold b (uniform
[0.02, 0.98], all five bands equal to b), so authorization is either level 0 or level 5.
Identical protocol.

### R13 (R) — Optimizer robustness re-run (`*_tpe`)
Tree-structured Parzen Estimator in place of random search for static, ewma and rova, same
80-candidate budget: candidates 1–20 are the frozen random-search candidates (so the
20-candidate tuning-curve point must reproduce the archived value exactly — a built-in
check); candidates 21–80 are proposed by TPE (good fraction γ = 0.25, 24 candidates scored
by the density ratio, per-dimension Parzen estimators with a uniform prior component, own
random stream). Reported: verdict family under TPE-tuned configurations and optimizer
comparisons, on the CEW.

## 5. Statistical analysis

### 5.1 Per-cell comparison families (Holm within family)
On the CEW (paired Wilcoxon, Cliff's δ, §1 rule):
F1 {rova–static, rova–ewma, static–ewma}; F2 {rova, static, ewma} × {A0, A5};
F3 {rova, static, ewma} × {learned, qlearn}; F4 {rova–static_ext, rova–ewma_ext,
static_ext–static, ewma_ext–ewma}; F5 {rova–clairvoyant, rova–rova_single};
F6 {rova_tpe–static_tpe, rova_tpe–ewma_tpe, static_tpe–ewma_tpe};
F7 {rova_tpe–rova, static_tpe–static, ewma_tpe–ewma}.
The same families are applied within R5a, R5b and R9a where the methods exist.

### 5.2 Intervals
BCa (frozen function, 5000 resamples, `default_rng(0)`) for mean and median regret per
method and cell, and for mean paired differences on the CEW. Pooled-seed CVaR-95 is reported
with a percentile bootstrap interval and flagged as low-precision (three tail instances).

### 5.3 Multiplicity sensitivity
S1: Holm across the 45 published MAIN-001 tests. S2: Holm across all tests in families
F1–F7 over all cells. For each, cell labels that change relative to within-family Holm are
listed.

## 6. Reporting rules (fixed now)

1. Every output specified here is reported, in the manuscript or its supplementary material,
   whatever its direction — including cells where ROVA does not outperform a reference.
2. No equivalence claims. Results without a detected difference are reported as such, with
   effect sizes and intervals.
3. No narrative is chosen before results exist; conclusions are written to the observed
   results.
4. A failing unit is re-run once; a persistent failure is reported, never silently dropped.
   A failing gate (§2) invalidates the run.
5. Anything not specified here is labelled exploratory.

## 7. Outputs and provenance

`revision_out/` contains `run_metadata.json` (environment guard, git commit, host, versions,
thread settings, start/end times), per-unit checkpoints and evaluation records, logs,
`analysis/` (tables, figures), a manifest of all units with SHA-256 digests, and
`SHA256SUMS`. The directory is archived as a single `.tar.gz` with its SHA-256.

## 8. Compute estimate and tiers (Ryzen 9 9950X; single-threaded units)

Basis: MAIN-001 ≈ 15 core-hours for 45 tuning units (≈ 20 core-min/unit), ≈ 0.07 s per
250-step instance. Refined by the timing pilot (§2).

| Tier | Experiments | Core-hours (est.) |
|---|---|---|
| A | R0, R1, R2, R3, R4, R5a, R6, R7, R8, R9, R10 | 80–140 |
| B | R5b, R11, R12, R13 | 50–70 |
| Total | | 130–210 (≈ 9–14 h wall-clock with 15 workers) |

No experiment requires a GPU.

## 9. Considered and not executed

| Item | Reason |
|---|---|
| Full 2⁴ factorial re-tuned ablation (protocol §7) | The manuscript makes no component-interaction claims and none are requested. With 5 outer folds per cell and 6 cells per regime, interaction contrasts would be estimated with at least the variance of the one-at-a-time effects, for which per-cell tests already have no power; the design cannot license interaction claims. Recorded as a limitation. |
| Bayesian ROPE layer (protocol §8) | No ROPE width was ever specified; choosing one after the results are known would be an unconstrained post-hoc choice. |
| TOST equivalence (protocol §8) | Triggered only by equivalence claims, and none will be made (§6). |
| Validation on public LLM-agent trajectory benchmarks | The regret endpoint requires, for every step, the loss of every authorization level (counterfactual outcomes) and a variable harm-to-utility cost ratio. Logged trajectory benchmarks record one action per step; LLM-judged scores would reintroduce a mechanism-dependent score. Mitigation: R9. |
| Direct comparison with LLM guardrail systems | They judge natural-language actions with a language model and output allow/block; the simulator has no text and no language model, and inserting one would replace the independent oracle. Mitigation: R3, R4, R1. |
| Richer learned policies (recurrent) | Optional in the protocol; R3 and R4 cover direct loss optimization and reinforcement learning. |

## 10. Amendment log

| Date | Change | Reason |
|---|---|---|
| 2026-09-29 | Initial version | — |
