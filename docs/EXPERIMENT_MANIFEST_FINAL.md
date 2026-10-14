# Experiment manifest — MAIN-001 (frozen, rev. 5)

**Frozen:** 2026-07-04, after two same-day revisions (rev. 4: claim-traceability review; rev. 5: cost-ratio grid
re-derived from the intended claim). After freezing, changes only by an explicit, dated decision recorded in
`CHANGELOG.md` with its scientific consequence.

Wording condensed on 2026-10-15; scope, grid, budget and dates are unchanged, and one stale number is corrected
(marked below). The frozen text as first published is in tag `v1.0` (`docs/EXPERIMENT_MANIFEST_FINAL.md`, git blob
`1ee393ddde210c0b2f4f22b953a83f2ae9485ab7`).

## Intended claim and grid

When the manifest was frozen no manuscript text existed beyond the target claim sentence; the scope follows the
claim analyses of the internal minimum-evidence and claim-granularity reviews. The claim does **not** locate a
crossover point c\*. It is that *"the runtime-governance versus baseline verdict changes across the operating-cost
spectrum, and survives the containment check, under matched, fairly tuned, nested-CV-validated evaluation."*

The cost-ratio grid therefore went from 7 to **5 points**, `[0.5, 1.5, 3.5, 8.0, 20.0]`, geometrically spaced over
the same range. Five is the smallest grid at which a permutation-style test of a non-random pattern across grid
points can reach p < 0.05 (best case p = 2/n!: 0.083 for n = 4, 0.017 for n = 5). Per-point power needs no more
points: PILOT-001 found effects large relative to per-point noise at 60 seeds. The points are spread over the whole
range, not clustered near a presumed transition, because no location is claimed.

**Scope.** Mechanism attribution (all ablations), the full factorial ablation, an RL-trained baseline and a
clairvoyant per-cost-ratio analysis are future work. Distribution shift is outside the required launch scope (rev. 4)
and reserved for revision; its code path is kept correct (its hard-coded cost ratio pointed at the removed 3.33 grid
point and now uses 3.5, the nearest remaining point).

## Experiment units

**45 nested-CV HPO units:** 3 methods × 3 environments × 5 cost ratios.
- Methods: `StaticThreshold`, `EWMA`, `DTCOATunable` (ROVA), all through the same `nested_cv_hpo` procedure with a
  matched budget and matched search-space conventions (band vectors of identical dimension and range).
- Environments (`environments.py`): `favorable_hmm_coupled`, `unfavorable_oscillation_uncoupled`,
  `collapse_moderate_coupled`.
- Cost-ratio grid (`environments.py`, rev. 5): **0.5, 1.5, 3.5, 8.0, 20.0**.
- Budget (PILOT-001 showed it cannot safely be reduced): K_out = 5, K_in = 3, n_trials = 80, seeds = 60.

**Obtained without extra compute:**
- containment report (DR4) per environment and cost ratio, from `prop_scale`/`stab_scale` of ROVA's tuned
  configurations;
- CVaR-95 co-primary metric, collected per fold;
- BCa bootstrap intervals (`stats.describe()`);
- Wilcoxon + Cliff's δ + Holm, the primary pairwise statistic per cost ratio;
- Friedman + Holm as a secondary omnibus summary only (the protocol cautions that its blocks are correlated).

**Distribution shift (2 evaluations) — removed from the required launch scope in rev. 4.** Reserved for revision:
ROVA and EWMA tuned on `favorable_hmm_coupled` at cost ratio 3.5 (rev. 5; formerly 3.33) and evaluated without
re-tuning on `unfavorable_oscillation_uncoupled`. It would support a generalisation claim ("holds beyond the tuning
regime") that the intended claim does not make. The code path (`distribution_shift.py`, wired into
`run_pipeline.py`) remains ready and cheap to run (it reuses tuned configurations; no new HPO).

**DP-oracle validation (complete):** the myopic oracle equals the sequential oracle in
`unfavorable_oscillation_uncoupled`, the only single-agent uncoupled environment. Required, but reported as an
appendix-level methodological result.

## Excluded (future work)

- Ablation study / mechanism attribution (`ablation_study.py` exists but is not called by `run_pipeline.py`)
- Full 2⁴ factorial ablation
- RL-trained baseline
- Clairvoyant per-cost-ratio threshold
- Multi-agent DP oracle
- Bayesian-optimisation robustness re-run
- Bayesian ROPE
- Environments beyond the frozen 3 and cost-ratio points beyond the frozen 5 *(corrected 2026-10-15: first published
  as "frozen 7", left over from rev. 4; the rev. 5 grid in `environments.py` and in MAIN-001's `config.json` has 5
  points)*

## Compute

Benchmarked at full-scale environment parameters: about 2 h single-threaded per unit, so **45 × 2 h ≈ 90 h**
single-threaded (126 h with the rev. 4 seven-point grid). Units are independent (`run_pipeline.py --num-shards N`
shards them deterministically); with 16–32 parallel processes this is a **3–6 h** job on a multi-core machine or a
small cluster.

## Launch

```
python3 -m dtcoa_val.experiments.run_pipeline --budget full \
    --out /path/to/results --shard-index $I --num-shards $N
```
Run once per shard index 0..N−1, then run shard 0 once more after all shards finish: it completes the DP-oracle
validation and distribution-shift steps, which depend on other shards' outputs and detect when these are ready
(see the `[wait]` log lines of `run_pipeline.py`).

## Completion

MAIN-001 is complete only when all raw and processed outputs, configurations, seeds, figures and tables have been
copied from `--out` to `experiments/MAIN-001_core_comparison/`, `EXPERIMENT_INDEX.md` has been updated, every
referenced artifact exists, and reproduction from the archived artifacts alone has been confirmed.
