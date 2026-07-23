# Changelog

## 2026-07-04

- Implemented `dtcoa_tunable.py`: continuous penalty magnitudes replacing
  frozen `dtcoa.py`'s integer/binary penalties. Validated against frozen
  reference at matching defaults.
- Implemented `search_space.py`: matched HPO search spaces for ROVA,
  StaticThreshold, EWMA (clairvoyant space added then later removed, see
  below).
- Implemented `nested_hpo.py`: random-search HPO + K_out x K_in nested CV,
  CVaR co-primary metric, checkpointed resumability. Verified resumability
  bit-identical across interrupted/resumed runs.
- Implemented `stats.py` additions: BCa bootstrap CI (replacing percentile
  CI as the default in `describe()`), Friedman+Holm multi-method test.
  Verified against synthetic skewed/separated data.
- Implemented `dp_oracle.py`: single-agent sequential (DP-optimal) oracle.
  FINDING: certified numerically identical to the myopic oracle across all
  6 regime types (max discrepancy ~1e-15), via counterfactual-perturbation
  verification, not assumption. This is a real scientific result, not
  scaffolding.
- Implemented `environments.py`: 3 frozen environments (favorable/
  unfavorable/structural-contrast) + 7-point sparse cost-ratio grid.
- Implemented `ablation_study.py`: one-at-a-time re-tuned ablations +
  containment (DR4) report. **Later descoped from the pipeline** (see PI
  review round 3) but retained as a module for future-work use.
- Implemented `distribution_shift.py`: minimal train->shift evaluation,
  reusing already-tuned configs.
- Implemented `run_pipeline.py`: top-level pipeline over methods x
  environments x cost-ratios, with manifest-based resumability and
  deterministic sharding for parallel execution. Verified: full pilot run,
  interrupted-and-resumed, and 2-shard parallel execution all produce
  identical, non-duplicated, non-corrupted results.
- **PI review round 2:** reclassified full ablation battery and
  clairvoyant threshold from essential to revision-contingent; kept a
  1-environment ablation spot-check in the minimum set.
- **PI review round 3:** mechanism attribution (all ablations, including
  the spot-check), full ablations, RL baseline, and clairvoyant analysis
  moved to future work in full. Pipeline refactored accordingly:
  - Removed `run_ablation_study` call from `run_pipeline.py`'s main loop;
    ROVA now runs via plain `nested_cv_hpo`, same as every other method.
  - Containment report is now computed as a free byproduct of ROVA's own
    tuned configs (no separate ablation run needed for it).
  - Removed `OracleLite`/clairvoyant from the methods dict and imports.
  - **Bug fix:** `distribution_shift` previously only evaluated ROVA under
    shift, making the delta non-comparative. Fixed to evaluate ROVA AND
    EWMA (baseline) under the same shift, using their respective
    already-tuned configs — no new HPO required.
  - Regression-tested: pilot-scale run (reduced env, reduced budget)
    confirms all of the above works end to end after the refactor, and both
    `distribution_shift_report__rova.json` and
    `distribution_shift_report__ewma.json` are produced correctly.
- **PILOT-001 executed and archived:** budget-sensitivity assessment across
  n_trials, K_out, K_in, seeds, cost-ratio grid density. All five tested
  reductions rejected on evidence (see
  `experiments/PILOT-001_budget_sensitivity/REPORT.md`). No budget
  parameter changed as a result.
- `EXPERIMENT_MANIFEST_FINAL.md` frozen: the minimum campaign for TMLR
  submission, per PI review round 3 scoping.

## 2026-07-04 (round 4)

- Verified project understanding explicitly at user's request (no changes
  from this step; confirmed evaluation-methodology framing, ROVA-as-object-
  of-study, and flagged that "the manuscript" does not yet exist as text).
- Produced `AE_MINIMUM_EVIDENCE_REVIEW.md`: reasoned from manuscript-claim
  necessity alone (TMLR AE perspective), independent of the frozen protocol
  and sunk implementation. Confirmed required set (E1/E2/E3/E5/E6/E7),
  appendix-level item (E4, DP-oracle validation), future-work items
  (E8 ablations, E9 RL baseline, E10 clairvoyant threshold — unchanged from
  round 3), and one new finding: E11 (distribution-shift) does not support
  any claim the current target sentence makes.
- **`EXPERIMENT_MANIFEST_FINAL.md` revised to rev. 4:** distribution-shift
  evaluation removed from MAIN-001's required launch scope, reserved for
  revision. Code path retained, unchanged, ready to run on request — no
  implementation change, documentation-only revision per explicit
  instruction not to implement/launch this round.
- `IMPLEMENTATION_STATUS.md` and `TODO.md` updated accordingly.

## 2026-07-04 (round 5)

- Ran the claim-granularity audit requested (`CLAIM_GRANULARITY_AUDIT.md`):
  traced each design choice (environment count/identity, cost-ratio grid
  count/placement) to manuscript-sentence necessity vs. evidentiary-strength
  choice vs. protocol inheritance. Flagged one genuine unresolved ambiguity:
  whether the manuscript intends a located-crossover (c*) claim, which would
  change the correct grid-density classification.
- Author resolved the ambiguity: no located-crossover claim intended; the
  claim is narrower ("verdict changes across the operating-cost spectrum").
- Re-derived the cost-ratio grid from that weaker claim using statistical-
  power/combinatorial reasoning: 5 points is the floor below which a
  permutation-style test of "the pattern across grid points is non-random"
  cannot reach p<0.05 even in the best case (exact best-case p=2/n!; n=4
  gives 0.083, n=5 gives 0.017). Per-point power does not require more
  points (PILOT-001 effect sizes already well-resolved at n=60 seeds vs.
  noise). Placement: geometric spacing across the existing range, not
  clustered near any presumed location.
- **Implemented:** `environments.py` `COST_RATIO_GRID` changed from
  `[0.5, 1.0, 2.0, 3.33, 5.0, 10.0, 20.0]` (7 points) to
  `[0.5, 1.5, 3.5, 8.0, 20.0]` (5 points).
- **Regression-tested** (Milestone Audit Protocol step 5): ran the full
  pilot-budget pipeline end to end after the change. **Caught a real bug:**
  `run_pipeline.py`'s `SHIFT_COST_RATIO = 3.33` no longer matched any grid
  point, causing the (already revision-deferred) distribution-shift step to
  wait indefinitely on a prerequisite that could never complete. Fixed to
  `3.5` (nearest surviving point). Re-ran pilot: clean, 63/63 units
  completed, no errors.
- `EXPERIMENT_MANIFEST_FINAL.md` → rev. 5, `IMPLEMENTATION_STATUS.md`,
  `TODO.md` updated. Consequence: MAIN-001 compute reduced from 126h to
  90h single-threaded (~3-6h wall-clock parallelized) — first net compute
  reduction to survive the full audit chain. No prior scientific conclusion
  affected (nothing run at full scale yet).

## 2026-07-04 (round 6 — venue reassessment)

- Target venue reassessed: IEEE Access, not TMLR. Produced
  `IEEE_ACCESS_VENUE_FIT_ASSESSMENT.md`: re-evaluated every currently
  excluded item against IEEE Access's actual bar (technical soundness +
  adequate evidence + reproducibility, NOT novelty-gated). Finding: this
  bar is lower/different than TMLR's on exactly the dimension that drove
  the earlier Case-A/ablation contingency risk (novelty-sensitivity to an
  "unattributed, expected" result) — that risk is materially reduced under
  IEEE Access. No other excluded item's risk changes. **No manifest
  changes required** — rev. 5 scope (45-unit MAIN-001) already sufficient.
  One new, non-experimental risk surfaced: claims should be scoped to the
  tested synthetic framework in the writeup (a writing task, not an
  experimental gap).

## 2026-07-04 (round 7 — execution runbook)

- Produced `MAIN-001_EXECUTION_RUNBOOK.md`: pure operational runbook for
  executing the frozen rev. 5 / 45-unit MAIN-001 campaign. No scope,
  experiment, or protocol changes. Covers execution order, sharding
  (N = min(15, cores)), checkpoint/resume procedure, expected directory
  structure, artifact preservation (registers MAIN-001 as one
  EXPERIMENT_INDEX.md entry per the PILOT-001 precedent), logging,
  failure recovery, intermediate validation checkpoints (~25%/50%/100%),
  final validation, and 8 explicit, checkable completion criteria.
  Distribution-shift explicitly excluded from completion criteria
  (revision-stage per frozen scope) even though the pipeline's finalization
  step will attempt it automatically — its output, if produced, is
  optional/bonus, not required.

## 2026-07-04 (round 8 — reproducibility audit)

- Reproducibility audit of 3 proposed runbook additions, none changing
  scope/claims/budget:
  1. Code version identifier: no git repo existed yet. Initialized one on
     `code/` and created the frozen reference commit
     `7f624ca27a86fbbbe67afc68ebb886a4edbb67a2`. Runbook section 0.1 now
     requires cloning (not copying) this commit to the compute host and
     verifying the hash before launch.
  2. Execution environment capture: added runbook section 0.2 (capture
     python version + pip freeze + uname on the actual compute host at
     launch time -- explicitly not pre-populated from this sandbox).
  3. Seed-set verification: added runbook section 0.3, a zero-cost
     assertion (60 unique seeds, exactly 0..59) run before launch, output
     archived as seeds_verified.json. Verified the check itself works
     correctly before prescribing it.
- Updated runbook sections 5 (archival must include git_commit in
  config.json, environment.txt, seeds_verified.json), 9 (final validation
  now checks all three), and 10 (completion criteria now include all
  three) accordingly.
- No experiments, analyses, statistical tests, or hyperparameter changes
  introduced. Documentation/reproducibility-infrastructure only.

## 2026-07-08 (MAIN-001 execution)

- **Executed MAIN-001** per `MAIN-001_EXECUTION_RUNBOOK.md` on the frozen
  rev. 5 manifest, code commit `7f624ca27a86fbbbe67afc68ebb886a4edbb67a2`
  (verified at launch). Host: AMD Ryzen 9 9950X (16C/32T), CPU-only. Section 0
  pre-flight passed (grid `[0.5,1.5,3.5,8.0,20.0]`, SHIFT_COST_RATIO 3.5,
  numpy/scipy, seed-set 0..59, environment captured).
- Launched 15 shards (`N=min(15,cores)`), one work-unit each, detached via
  `setsid`, BLAS/OMP threads pinned to 1 (scipy-openblas — prevents 15-way
  oversubscription). Wall-clock ≈ 1h; **zero `[FAIL]`**.
- **Result:** 61/61 required completion entries; 45/45 HPO checkpoints at
  exactly 5 folds with finite regret + CVaR; 15/15 containment reports;
  DP-oracle `certified_equivalent=true` (max discrepancy 4.5e-14). Full stats
  pass (BCa, Wilcoxon+Cliff's δ+Holm per cost-ratio, Friedman+Holm, CVaR,
  containment) computed and saved. **Primary claim supported:** ROVA-vs-baseline
  verdict changes across the cost spectrum in all 3 environments.
- **Deviation (infrastructure, no scientific-data impact):** discovered a
  lost-update race on the shared `manifest.json` — `run_pipeline._mark_done` +
  `utils.save_json` do non-atomic/unlocked read-modify-write; at 15 shards the
  live manifest lost updates (ended 4/61). Latent bug (parallel path validated
  only at 2 shards previously). Per-unit checkpoints are single-writer and
  unaffected. Reported to and resolved by user decision ("continue +
  reconstruct"): manifest rebuilt deterministically from authoritative
  artifacts (`raw/rebuild_and_validate.py`, validates 61 required while
  assembling); raced file preserved as `raw/manifest.live_raced.json`.
  **No scientific-scope, hyperparameter, environment, cost-grid, or protocol
  change.** Root-cause fix of the IO path is a future code change, out of this
  execution's scope.
- Distribution-shift (`rova`, `ewma`) produced by the finalize step as an
  optional/bonus artifact — revision-stage, not a completion criterion.
- Archived to `experiments/MAIN-001_core_comparison/` (raw/, analysis.py,
  config.json, seeds.json, REPORT.md, stats_summary.json); registered in
  `EXPERIMENT_INDEX.md`; reproducibility verified (fresh `analysis.py` from
  `raw/` alone → identical sha256).

## 2026-07-09 (ABLATION-001 execution) / 2026-07-10 (archival)

- Branched `ablation-001` from the frozen anchor tag `main-001-frozen`
  (`7f624ca`); created that tag on the MAIN-001 commit. **MAIN-001 untouched
  throughout** (archive re-verified 88/88 at every step).
- **FIX-1:** kept `e_min` tunable in the `no_memory`/`no_decay` ablations — it
  feeds the propagation path independently of the memory/decay switch. Verified
  by falsification (active in coupled envs; **exactly** inert, spread 0.0, in
  the single-agent uncoupled env).
- Added ABLATION-001 tooling (producing commit `4533c61`, tag
  `ablation-001-frozen`): standalone driver `run_ablation.py` with race-free
  per-shard atomic manifests (FIX-5); read-only distributional-consistency
  validator `validate_consistency.py`; DR3 descriptive `dr3_attribution.py`;
  DR3 regime inference `dr3_regime_inference.py`. All with tests.
- **DR3 method (design review):** the frozen per-cell test has *zero* power at
  K_out=5 (verified: even a maximal signal is unflaggable); naive fold-pooling
  is pseudoreplicative (folds nest in cells, cells in 3 environments). Adopted a
  pre-specified two-stage cell-level analysis — Cliff's δ (scale-free),
  environment-clustered inference, exact within-cell sign-flip permutation
  (Option B primary + Option C sensitivity); effect sizes primary; Holm + BH
  both reported; "not detected" ≠ "inert".
- **Executed:** 15 shards via `systemd-run --user` (one isolated transient
  service per cell, single-threaded BLAS, no oversubscription), ~2h13m, 15-way,
  zero failures. Structural audit PASS (75/75 checkpoints, 375/375 folds).
- **Consistency validator (hard gate): PASS 15/15** — ablation-`full` agrees
  with archived MAIN-001 ROVA within Monte-Carlo noise (no harness drift).
- **FINDING:** memory, propagation, and stability are load-bearing specifically
  in the high-cost regime where ROVA wins (Cliff's δ>0, env-CI excludes 0,
  BH-FDR significant); decay not detected at this budget; the contributing set
  differs by regime (H2 descriptive, permutation p≤0.006) — the mechanistic
  "why" for MAIN-001's cost-ratio-dependent verdict. Containment (DR4) passes in
  the win regime (`prop_scale`/`stab_scale` bounded away from zero). Honest
  ceiling: nothing survives Holm-FWER at ~6 cells/regime (design limit).
- Post-processing kept raw byte-immutable (identical digest pre/post); all
  derived analyses written to a new location.
- Archived to `experiments/ABLATION-001_mechanism_attribution/` (raw/, analysis/,
  config.json, REPORT.md, reproduce.sh, SHA256SUMS [132 files], immutable/
  tarball + `.sha256`); registered in `EXPERIMENT_INDEX.md`; reproducibility
  verified (raw immutable pre/post; derived regenerable from `raw/` at
  `4533c61`; tarball + fresh extraction verify 132/132).
