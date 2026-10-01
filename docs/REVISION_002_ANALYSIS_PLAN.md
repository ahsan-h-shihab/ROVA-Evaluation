# Revision Analysis Plan — REV-002 (A1 tuning convergence; A2 fold-clustered re-analysis)

**Date written:** 2026-10-01.
**Status:** written **after** the REV-001 results (`revision_out.tar.gz`, SHA-256
`63dae057d63c4917d90391fd9da72279f5e909ffb1a1b7e192999f55d28a6b10`) were known and audited,
and **before** either analysis specified here was executed. It is not a pre-registration. The
commit that adds this file, tagged `revision-002-analysis-plan`, fixes its content; changes
made before execution are recorded with dates in §8. The exact code that executes the plan is
tagged `revision-002-frozen`.

Both analyses are labelled **(L4) specified after the REV-001 results**. In any report they
are presented as such and are never described as pre-registered or as part of the REV-001
plan ([`REVISION_ANALYSIS_PLAN.md`](REVISION_ANALYSIS_PLAN.md)).

Nothing here modifies a REV-001 result. REV-001 outputs are read-only inputs; every new output
is written to new directories, separate from the REV-001 archive.

---

## 1. Why these two analyses

**A1.** REV-001 tuned the same three methods (static threshold, EWMA threshold, ROVA) under
three tuning protocols with different outcomes in the high-cost cells: random search with 80
candidates (published protocol; 8 cells where ROVA has lower regret than both baselines),
random search with 200 candidates and five inner folds (6 such cells), and TPE with 80
candidates (no such cell; 7 cells where both baselines have lower regret). None of these
protocols was shown to have converged: under TPE, best inner-validation regret was still
decreasing between candidates 40 and 80 for ROVA in most low- and mid-cost cells and for the
baselines at low cost ratios. A1 asks whether the verdict stabilizes as the optimizer
approaches convergence, and what it is.

**A2.** REV-001's per-world tests condition on the five tuned configurations of each method
(one per outer fold). Per-world regrets of the tuned baselines are strongly clustered by
fold (ICC up to 0.96 in REV-001). A2 quantifies which differences also hold across tuning
runs, treating the fold-level configuration as a random effect.

---

## 2. A1 — tuning convergence (TPE continued from 80 to 320 candidates)

### 2.1 Design (fixed)

| Item | Specification |
|---|---|
| Methods | static, EWMA, ROVA; frozen policy classes and frozen search spaces (as in REV-001 R13). No other method |
| Cells | all 15: environments `favorable_hmm_coupled`, `unfavorable_oscillation_uncoupled`, `collapse_moderate_coupled` × c ∈ {0.5, 1.5, 3.5, 8.0, 20.0} |
| Nested cross-validation | frozen: K_out = 5, K_in = 3, seeds 0–59, the frozen outer and inner splits, inner-validation world tags identical to REV-001 R13 (`{env}__c{c}__{method}_out{k}_hpo{i}_infold{j}`) |
| Optimizer | the REV-001 R13 TPE implementation, unchanged (good fraction γ = 0.25; 24 candidates scored by the density ratio; per-dimension Parzen estimators with a uniform prior component; sampler stream seeded by `trial_seed(k, 0, "{tag}__tpe")`), run for **320** candidates per outer fold |
| Startup candidates | candidates 1–20 are the frozen random-search candidates. They are drawn from `default_rng(0)` with **80 draws per outer fold**, exactly as in REV-001 (the frozen code sizes this draw by the 80-candidate budget); the draw is not resized for 320 candidates, so every outer fold receives the same startup candidates as in REV-001 |
| Selection at budget b | the configuration with the lowest mean inner-validation regret among candidates 1…b (`nanargmin`; ties → earliest candidate), for b ∈ {80, 160, 320} |
| Recorded tuning curves | all 320 inner scores per outer fold; best-so-far at b ∈ {20, 40, 80, 120, 160, 240, 320} |
| Evaluation | each selected configuration (3 methods × 3 budgets) on the REV-001 common evaluation worlds (world tag `REV1_CEW__{env}__c{c}_out{k}_test`, unchanged), so A1 results are matched world-by-world to every REV-001 method; and the 80-budget selection additionally on its archived own-world test tag (`{env}__c{c}__{method}_out{k}_test`) for reproduction check RC4 |
| Execution | the MAIN-001/REV-001 workstation (AMD Ryzen 9 9950X), CPU only, 15 single-threaded worker processes, from a clean checkout of `revision-002-frozen` |

### 2.2 Gates and reproduction checks

All must pass. A failed environment gate, R0 gate or input check stops the run before any
tuning. A failed reproduction check RC1–RC5 stops the run before the evaluation and analysis
stages and invalidates A1; it is reported, not repaired.

| ID | Check | Pass condition |
|---|---|---|
| G-ENV | REV-001 execution-environment gate with its archived reference, bound to the tag `revision-002-frozen` | every hard and soft item OK, no written approval: CPU AMD Ryzen 9 9950X, 16 physical / 32 logical cores, 1 socket; kernel 6.17.0-40-generic; glibc 2.42; Python 3.13.14; numpy 2.5.1; scipy 1.18.0; matplotlib 3.11.0; scipy-openblas 0.3.33; 15 workers; clean tree at the frozen tag; one thread per process; no GPU framework loaded |
| G-R0 | REV-001 gate R0 (45 archived MAIN-001 units replayed; 45 archived BCa intervals recomputed) | all 45 replays with maximum absolute difference exactly 0.0, all 45 intervals identical |
| G-IN | REV-001 outputs used as inputs | the file `SHA256SUMS` of REV-001 `revision_out/` has SHA-256 `12a72b84d635aee108950808fd8d0f7d901c56be0289b63ddf05c0f325577cc4`; every one of its 1,883 entries verifies; the REV-001 output verifier passes |
| RC1 | per method, cell and outer fold: inner scores of candidates 1–80 versus REV-001 R13 | exact equality |
| RC2 | per outer fold: index and configuration selected at b = 80 versus REV-001 R13 | identical |
| RC3 | per-world common-world regrets of the 80-budget selections versus the REV-001 records of `static_tpe`, `ewma_tpe`, `rova_tpe` (same seeds, same world tags) | exact equality |
| RC4 | own-world test regrets of the 80-budget selection versus REV-001 R13 `test_regrets` | exact equality |
| RC5 | best-so-far at candidate 20 versus the archived MAIN-001 random-search tuning curve | absolute difference < 1e-9 (as REV-001 R13) |
| RC6 | completeness | every A1 unit done (1 gate, 45 tuning campaigns, 15 evaluations, 1 analysis), no failure after one automatic retry |

### 2.3 Convergence criterion (fixed now; not changed after any A1 result is seen)

B(b) denotes the best mean inner-validation regret among candidates 1…b, for one method, cell
and outer fold.

1. An outer fold is **converged** iff **B(160) − B(320) ≤ max(0.05 × B(160), 0.0002)**.
   - 0.05: at most a 5% relative improvement over the second half of the budget.
   - 0.0002 (regret per agent-step): less than half the smallest absolute mean difference
     that carries a non-negligible label in any REV-001 comparison among static, EWMA and
     ROVA under random search or TPE (0.00045: ROVA-TPE versus EWMA-TPE, collapse, c = 0.5).
2. A method is **converged in a cell** iff all five outer folds are converged.
3. A cell is **converged** iff static, EWMA and ROVA are all converged in it.
4. Reported descriptively, not part of the criterion: whether the verdict at 160 equals the
   verdict at 320; whether the selected configuration changes between 160 and 320;
   B(240) − B(320).

### 2.4 Analyses (all on the common evaluation worlds, n = 60 per cell)

Comparison rule as in the REV-001 plan §1 (Wilcoxon signed-rank, Holm within family, Cliff's
δ with |δ| > 0.147 for a non-negligible label); cell verdict by the REV-001 verdict rule
(ROVA versus both baselines in the same direction). Additionally, Holm across **all A1 tests**
(the "A1 family"), reported as the A1 counterpart of the REV-001 protocol-family layer and
used as the significance criterion of the decision rules below. BCa intervals as in REV-001
(5,000 resamples, `default_rng(0)`).

| ID | Content | Budgets |
|---|---|---|
| H1(b) | ROVA–static, ROVA–EWMA, static–EWMA, all TPE-tuned at budget b; cell verdict(b) | 80, 160, 320 |
| H2 | each method at 320 versus always-full-supervision (A0, REV-001) | 320 |
| H3 | each method at 320 versus the REV-001 learned policy and versus REV-001 Q-learning | 320 |
| H4 | each method at 320 versus the same method at 80 (budget effect) | — |
| H5 | each method at 320 versus the same method's published random-search configurations (REV-001 common-world records) | — |
| G | surviving-margin ratio (A0 − ROVA@320) / (min(static@320, EWMA@320) − ROVA@320) with the REV-001 interval rule (interval not reported if the denominator is non-positive in any resample) | 320 |
| DR | DR1, DR2 with crossover location, DR8 (REV-001 operationalizations) for ROVA@320 versus static@320, EWMA@320 and A0 | 320 |
| Curves | per method and cell, best-so-far over candidates 1…320 (mean over outer folds and per fold); the archived random-search tuning curve (recorded at 20, 40 and 80 candidates only) shown alongside | — |
| Behaviour | authorization-level fractions of the 320 selections; oracle full-supervision fraction | 320 |
| A2 on A1 | the statistics of §3.4 for H1(320), H2, H3 and H4 | — |

### 2.5 Reporting rules (fixed now)

1. Every A1 output is reported, whatever its direction.
2. Every cell's verdict at 320 is reported together with its convergence status. The term
   "converged-budget verdict" is used only for converged cells; other cells are reported as
   "not converged at 320 candidates", with their verdict shown.
3. No budget beyond 320 candidates, no other optimizer seed, and no change of the
   convergence criterion after A1 results are seen.
4. A1 results that qualify or reverse a conclusion drawn from REV-001 are reported as such.

### 2.6 Compute

Basis: REV-001 R13 unit medians at 80 candidates (static 612 s, EWMA 614 s, ROVA 931 s).
At 320 candidates: ≈ 36 core-hours for the 45 tuning campaigns, ≈ 0.2 core-hours for the 15
evaluations, ≈ 10 minutes of analysis; ≈ 2.7 h wall-clock with 15 workers (budget ≤ 4 h).
No GPU.

---

## 3. A2 — fold-clustered statistical re-analysis

### 3.1 Nature

A post-run, deterministic statistical re-analysis of stored REV-001 per-world regrets. It runs
no simulation and imports no simulator code (enforced by a test that blocks those imports).
It may run on any machine; the machine and library versions are recorded.

### 3.2 Inputs

The REV-001 archive `revision_out.tar.gz` (SHA-256 above), extracted into a new directory;
`SHA256SUMS` must have the SHA-256 given in §2.2 G-IN and all 1,883 entries must verify.
Data: the per-world common-world records of the REV-001 evaluation units (`regret`, `seed`,
`outer_fold`, world tag); REV-001 per-world labels, Wilcoxon p-values, Cliff's δ and BCa
intervals from `analysis/summary.json`.

### 3.3 Comparisons (fixed list)

Per cell (15 cells), pairs (A, B) with paired differences d_i = regret_A,i − regret_B,i:

- F1 ROVA–static, ROVA–EWMA
- F2 ROVA–A0, static–A0, EWMA–A0
- F3 ROVA–learned, ROVA–Q-learning
- F4 ROVA–static_ext, ROVA–ewma_ext
- F5 ROVA–rova_single, ROVA–clairvoyant
- F6 ROVA_TPE–static_TPE, ROVA_TPE–EWMA_TPE
- F7 ROVA_TPE–ROVA, static_TPE–static, EWMA_TPE–EWMA
- F8 abl_no_X–abl_full for X ∈ {memory, decay, propagation, stability}

That is 20 pairs × 15 cells = 300 comparisons, plus the surviving-margin ratio G in every
cell where REV-001 reports it as defined. After A1, the same code is applied to H1(320), H2,
H3 and H4.

### 3.4 Statistics (per comparison)

1. Point estimates as in REV-001: mean of d; Cliff's δ(A, B).
2. **Two-stage cluster bootstrap (primary A2 interval).** 5,000 resamples; a fresh
   `numpy.random.default_rng(0)` per comparison. Each resample draws 5 outer folds with
   replacement and, within each drawn fold, 12 world indices with replacement; the same
   indices are used for A and B, so pairing is preserved. Statistics: mean of d, and Cliff's
   δ between the resampled A and B values. 95% percentile intervals.
3. Fold-only cluster bootstrap (secondary): resample the 5 outer folds with replacement and
   keep all worlds of each drawn fold; 95% percentile interval of the mean of d.
4. Exact fold-level sign-flip test: m_k = mean of d within outer fold k; two-sided p = share
   of the 32 sign vectors s with |mean(s·m)| ≥ |mean(m)|. The smallest attainable p is
   0.0625; it is reported descriptively and never used for a label.
5. ICC(1) of d by outer fold (one-way ANOVA estimator, truncated at 0); design effect
   DE = 1 + 11·ICC; effective sample size 60/DE.
6. G: two-stage bootstrap of numerator and denominator jointly (same resample indices for all
   methods); the interval of G is reported only if the denominator is positive in every
   resample, otherwise the interval of the denominator is reported.

### 3.5 Interpretation rules (fixed now)

1. A2 never changes a REV-001 label; the per-world labels remain the pre-specified result.
2. A difference is described as **generalizing across tuning runs** only if the two-stage 95%
   interval of the mean of d excludes 0 in the direction of the per-world label. Otherwise it
   is described as holding for the five tuned configurations and as not established across
   tuning runs.

### 3.6 Outputs and determinism

Outputs are written to a new directory outside the repository and separate from the REV-001
archive: the per-comparison table, the G table, a summary, an automatically generated report,
run metadata (commit, library versions, operating system, CPU, input hashes) and
`SHA256SUMS`. The analysis is run twice from the frozen tag; the two output sets must be
byte-identical. The re-application inside A1 uses the same code; any difference due to the
different library versions of the two machines is reported.

---

## 4. Code-verification policy before the freeze

Unit tests and plumbing checks may run on a development machine using only non-experimental
seeds (≥ 1000), shrunken worlds and synthetic data. Such outputs are used only to check that
the code runs and are deleted; they are not inspected as results. Any such computation is
listed in §8. No study seed is simulated outside the workstation run, except replays of
already-archived evaluations performed by the frozen REV-001 gate code.

## 5. Outputs and provenance (A1)

A new output directory (default name `a1_out/`), separate from REV-001's `revision_out/`:
run metadata with the environment-gate record of every invocation, the git commit, start and
end times; per-unit results with result digests; logs; `analysis/` (summary, tables, figures,
report, the A2 statistics on A1); `SHA256SUMS`; archived as a single `.tar.gz` with its SHA-256.
A verifier checks checksums, unit digests, completeness, the gates, the reproduction checks,
the analysis digest, the environment record of every invocation, a single commit, and that the
convergence constants recorded in the output equal those of §2.3.

## 6. What this plan does not do

It does not change the REV-001 plan, outputs or labels; it does not add methods, cells,
optimizers or budgets beyond those stated; it does not use a GPU.

## 7. Relation to the manuscript

Both analyses are reported as revision analyses specified after the REV-001 results. Their
outcomes are reported in full, in either direction.

## 8. Amendment log

| Date | Change | Reason |
|---|---|---|
| 2026-10-01 | Initial version | — |
| 2026-10-01 | **Implementation details (no change of design).** (i) The fold-only bootstrap of §3.4 item 3 uses its own `default_rng(0)`, drawing `integers(0, 5, (5000, 5))`; the two-stage bootstrap draws the fold choices `integers(0, 5, (5000, 5))` and then the within-fold positions `integers(0, 12, (5000, 5, 12))` from one fresh `default_rng(0)` per comparison. (ii) A2 analyses the cells present in the REV-001 summary and refuses to complete unless all 15 are present. (iii) A1 records the startup-draw size, the convergence constants and the budgets in its output; the verifier compares them with §2.3. (iv) The A1 environment gate is the REV-001 gate with the frozen-tag check bound to `revision-002-frozen`; G-R0 additionally requires every replay difference to be exactly 0.0 (the REV-001 gate code tolerates 1e-9). | Records how the plan is implemented. |
| 2026-10-01 | **Code verification before the freeze (disclosure).** On the development laptop, before the freeze: the REV-002 unit tests (non-experimental smoke seeds 1000–1005, shrunken worlds, synthetic data); the REV-001 test suite; one REV-001 `smoke` run (seeds 1000–1005, horizon 20, 3 agents) to create a plumbing input, whose gate code replays one archived MAIN-001 evaluation, as in REV-001; A1 `smoke` runs on that input, including deliberate falsification runs (a changed REV-001 inner score, a changed common-world regret, an input file changed without updating its checksum, a changed A1 output: each was detected — reproduction failure, gate failure or verification failure); a refused A1 `full` start (environment gate failed; no unit run); A2 on the smoke input twice (byte-identical outputs). No study seed was simulated; no REV-001 study output was analysed. All these outputs were used only to check the code and were deleted. | Complete disclosure of pre-freeze computation (§4). |
