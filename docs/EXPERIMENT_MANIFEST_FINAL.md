# EXPERIMENT MANIFEST — MAIN-001 (FROZEN, rev. 5)

**Frozen:** 2026-07-04. Revised 2026-07-04 (round 4: manuscript-claim
traceability audit; round 5: cost-ratio grid re-derived from the actual
intended claim — see `CLAIM_GRANULARITY_AUDIT.md`). Do not modify after
this point except by an explicit new PI decision, logged as a new dated
entry in CHANGELOG.md with scientific consequence stated.

**Note on "manuscript":** no manuscript text currently exists beyond the
target claim sentence below — the only prior manuscript draft is confirmed
stale/invalid (`latex_submission/WARNING_STALE_MANUSCRIPT.md`). This
manifest is scoped against `AE_MINIMUM_EVIDENCE_REVIEW.md`'s claim analysis
and `CLAIM_GRANULARITY_AUDIT.md`'s granularity re-derivation, the most
recent explicit statements of intended claims.

**Rev. 5 clarification of the primary claim:** the manuscript does NOT
intend to claim a precise, located crossover point c*. The intended claim
is narrower: *"the runtime-governance versus baseline verdict changes
across the operating-cost spectrum, and survives the containment check,
under matched, fairly tuned, nested-CV-validated evaluation."* This is
weaker than a located-crossover claim and requires less grid resolution.

**Cost-ratio grid re-derived accordingly:** 7 points → **5 points**
`[0.5, 1.5, 3.5, 8.0, 20.0]`, geometrically spaced across the same overall
range. Basis: 5 is the combinatorial floor below which a permutation-style
test of "the pattern across grid points is non-random" cannot reach
p<0.05 even in the best case (exact best-case p = 2/n!; n=4 gives 0.083,
n=5 gives 0.017). Per-point statistical power does not require more points
— PILOT-001 found effect sizes large relative to per-point noise at n=60
seeds. Placement is geometric across the full range (not clustered near any
presumed transition location) specifically because the claim does not
assert a location. Full reasoning: `CLAIM_GRANULARITY_AUDIT.md`.

**Scope decision this manifest encodes:** mechanism attribution (all
ablations), full factorial ablation, RL-trained baseline, and clairvoyant
per-cost-ratio analysis are future work. Distribution-shift evaluation is
also out of required launch scope (round 4) — reserved for revision, code
path retained and kept correct (fixed a bug in this round: its hardcoded
shift cost-ratio pointed at the now-removed 3.33 grid point, corrected to
3.5, the nearest surviving point).

## Experiment units

**45 nested-CV-HPO units:** 3 methods (StaticThreshold, EWMA, ROVA) × 3
environments × 5 cost ratios (was 7 — see rev. 5 note above).

- Methods: `StaticThreshold`, `EWMA`, `DTCOATunable` (ROVA) — all via the
  identical `nested_cv_hpo` procedure, matched budget, matched search-space
  conventions (band vectors identical dimensionality/range across methods).
- Environments (frozen, `environments.py`): `favorable_hmm_coupled`,
  `unfavorable_oscillation_uncoupled`, `collapse_moderate_coupled`.
- Cost-ratio grid (frozen, rev. 5, `environments.py`): **0.5, 1.5, 3.5, 8.0,
  20.0** — re-derived from the actual claim (transition existence, not
  location); see `CLAIM_GRANULARITY_AUDIT.md`.
- Budget (frozen, confirmed by PILOT-001 not safe to reduce): K_out=5,
  K_in=3, n_trials=80, seeds=60.

**Free byproducts of the above (no additional compute):**
- Containment report (DR4) per (environment, cost-ratio) — reads
  `prop_scale`/`stab_scale` out of ROVA's own tuned configs.
- CVaR-95 co-primary metric — already collected per nested_cv_hpo fold.
- BCa bootstrap CIs — computed by `stats.describe()` on any collected array.
- Wilcoxon + Cliff's δ + Holm (primary pairwise statistic, per cost ratio).
- Friedman + Holm (secondary/supporting omnibus summary only, per protocol's
  own caveat about correlated blocks).

**2 distribution-shift evaluations — REMOVED FROM REQUIRED LAUNCH SCOPE (rev. 4).**
Reserved for revision: ROVA and EWMA, tuned on `favorable_hmm_coupled` at
cost ratio 3.5 (updated rev. 5 — nearest surviving grid point after the old
3.33 was removed), evaluated without re-tuning on
`unfavorable_oscillation_uncoupled`. Rationale: supports a generalization
claim ("holds beyond the tuning regime") that the current manuscript claim
does not make — the claim is explicitly scoped to tested conditions.
Code path (`distribution_shift.py`, wired in `run_pipeline.py`) stays ready;
cheap to run on request (reuses already-tuned configs, no new HPO) if a
reviewer asks for it in the first round.

**1 DP-oracle validation (already complete):** certified myopic oracle ==
sequential oracle for `unfavorable_oscillation_uncoupled` (the only
single-agent-uncoupled environment in the grid). Required, but reported as
an appendix/methodological footnote, not a headline result.

## Explicitly excluded from this manifest (future work)

- Ablation study / mechanism attribution (`ablation_study.py` — module
  exists, not invoked by `run_pipeline.py`)
- Full 2⁴ factorial ablation
- RL-trained baseline
- Clairvoyant per-cost-ratio threshold
- Multi-agent DP oracle
- Bayesian-optimization robustness re-run
- Bayesian ROPE
- Additional environments beyond the frozen 3, additional cost-ratio points
  beyond the frozen 7

## Compute estimate

Benchmarked (not guessed) at true full-scale environment parameters: ~2h
single-threaded per nested-CV-HPO unit → **45 units × 2h ≈ 90h
single-threaded** (down from 126h at the rev-4 7-point grid — the grid
re-derivation is the first net compute reduction to survive the full audit
chain). Every unit is independent (`run_pipeline.py --num-shards N` shards
deterministically across them) — with 16-32-way parallelism this is a
**~3-6h wall-clock job**, feasible overnight on a multi-core machine or
small cluster. This cannot run inside the chat environment (no tool call
survives more than a few minutes, no state persists across turns) — it
needs to be launched on real compute.

## Launch command (once execution venue is confirmed)

```
python3 -m dtcoa_val.experiments.run_pipeline --budget full \
    --out /path/to/results --shard-index $I --num-shards $N
```
Run once per shard index 0..N-1 (parallel processes/machines), then re-run
shard 0 once more after all shards finish to finalize the DP-oracle
validation and distribution-shift steps (these depend on other shards'
outputs and self-detect readiness — see `run_pipeline.py`'s `[wait]` log
lines).

## Completion protocol

Per the Experiment Completion Protocol, MAIN-001 is not complete until: all
raw/processed outputs, configs, seeds, and figures/tables are copied out of
`--out` into this project's `experiments/MAIN-001_core_comparison/`
directory; `EXPERIMENT_INDEX.md` is updated; every referenced artifact is
verified to exist; and reproducibility is confirmed using only the archived
artifacts (not this conversation).
