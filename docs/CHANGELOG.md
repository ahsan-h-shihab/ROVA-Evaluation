# Changelog

The wording of this changelog was condensed on 2026-10-15 and revised on 2026-10-16. The text as
first published is in tag
`v1.0` for the entries up to 2026-07-10 (`docs/CHANGELOG.md`, git blob `baf9b4ef1f73dce939528057b7609c38601476bf`)
and in tag `v2.0` for the entries from 2026-09-29 (git blob `971fb5f136289e5fc6f123ec133855ec3cfb16d6`).

## 2026-07-04 — implementation and scope

- `dtcoa_tunable.py`: continuous penalty magnitudes replacing the integer/binary penalties of the frozen
  `dtcoa.py`; validated against the frozen reference at matching defaults.
- `search_space.py`: matched HPO search spaces for ROVA, StaticThreshold and EWMA (a clairvoyant space was added and
  later removed).
- `nested_hpo.py`: random-search HPO with K_out × K_in nested cross-validation, CVaR co-primary metric and
  checkpointed resumption; resumed runs are bit-identical to uninterrupted ones.
- `stats.py`: BCa bootstrap intervals (now the default in `describe()`) and a Friedman + Holm multi-method test,
  checked on synthetic skewed and separated data.
- `dp_oracle.py`: single-agent sequential (DP-optimal) oracle. It is numerically identical to the myopic oracle in
  all 6 regime types (maximum discrepancy ~1e-15), established by counterfactual perturbation rather than assumed.
- `environments.py`: three frozen environments (favorable, unfavorable, structural contrast) and a 7-point sparse
  cost-ratio grid (reduced to 5 points in round 5).
- `ablation_study.py`: one-at-a-time re-tuned ablations and the containment (DR4) report; later removed from the
  pipeline (round 3) and kept as a module.
- `distribution_shift.py`: train→shift evaluation reusing tuned configurations.
- `run_pipeline.py`: pipeline over methods × environments × cost ratios with manifest-based resumption and
  deterministic sharding. A full pilot run, an interrupted-and-resumed run and a 2-shard run gave identical,
  complete results.
- Scope review, round 2: the full ablation battery and the clairvoyant threshold reclassified from essential to
  revision-contingent; a 1-environment ablation spot-check kept.
- Scope review, round 3: mechanism attribution (all ablations, including the spot-check), the RL baseline and the
  clairvoyant analysis moved to future work. Pipeline changes: ROVA runs through plain `nested_cv_hpo` like every
  other method; the containment report is computed from ROVA's own tuned configurations; `OracleLite`/clairvoyant
  removed. Bug fix: `distribution_shift` evaluated only ROVA under shift; it now evaluates ROVA and EWMA under the
  same shift with their own tuned configurations. A pilot-scale regression run produced both
  `distribution_shift_report__rova.json` and `distribution_shift_report__ewma.json`.
- PILOT-001 executed and archived: budget sensitivity over n_trials, K_out, K_in, seeds and grid density. All five
  tested reductions were rejected on the evidence
  ([`REPORT.md`](../experiments/PILOT-001_budget_sensitivity/REPORT.md)); no budget parameter changed.
- `EXPERIMENT_MANIFEST_FINAL.md` frozen as the minimum campaign (round-3 scope).

## 2026-07-04 — rounds 4 to 8

- Round 4. Framing confirmed (the study evaluates an evaluation methodology, with ROVA as its object); no manuscript
  text existed yet. A minimum-evidence review (internal note) assessed each intended claim on its own evidence
  needs, independently of the frozen protocol and of the work already done: a required set, the DP-oracle
  validation as an appendix item, ablations, RL baseline and clairvoyant threshold as future work (as in round 3),
  and one new finding — the distribution-shift evaluation supports none of the claims in the target sentence. The
  manifest (rev. 4) therefore moved distribution shift out of MAIN-001's required launch scope and reserved it for
  revision; its code path was kept unchanged.
- Round 5. A claim-granularity review (internal note) traced each design choice to the intended claim. The author
  resolved its one open question: no located crossover point is claimed, only that the verdict changes across the
  operating-cost spectrum. The cost-ratio grid was re-derived from that claim: 5 points is the smallest grid at which
  a permutation-style test of a non-random pattern across grid points can reach p < 0.05 (best case p = 2/n!:
  0.083 for n = 4, 0.017 for n = 5); per-point power needs no more points (PILOT-001 effects are well resolved at 60
  seeds); placement is geometric across the existing range. `COST_RATIO_GRID` changed from
  `[0.5, 1.0, 2.0, 3.33, 5.0, 10.0, 20.0]` to `[0.5, 1.5, 3.5, 8.0, 20.0]`. The regression run caught a bug:
  `SHIFT_COST_RATIO = 3.33` no longer matched a grid point and the deferred distribution-shift step waited
  indefinitely; fixed to 3.5 and re-run cleanly (63/63 units). Manifest → rev. 5. MAIN-001 compute fell from 126 h
  to 90 h single-threaded (about 3–6 h wall-clock in parallel); nothing had yet run at full scale.
- Round 6. Target journal changed to IEEE Access. The excluded items were re-assessed for that journal; none was
  brought back into scope, and the rev. 5 manifest was kept unchanged. One writing requirement was recorded: claims
  in the paper should be scoped to the tested synthetic framework.
- Round 7. `MAIN-001_EXECUTION_RUNBOOK.md`: operational runbook for the frozen rev. 5 campaign (45 units) — order,
  sharding (N = min(15, cores)), checkpoint and resume, directory layout, archival (one `EXPERIMENT_INDEX.md` entry,
  as for PILOT-001), logging, recovery, validation at about 25/50/100 %, final validation and 8 checkable completion
  criteria. Distribution shift is not a completion criterion although the finalisation step attempts it.
- Round 8. Reproducibility additions, none changing scope, claims or budget: the code was put under git and frozen
  at `7f624ca27a86fbbbe67afc68ebb886a4edbb67a2` (runbook section 0.1 requires cloning this commit on the compute
  host and verifying the hash); the Python version, `pip freeze` and `uname` are captured on the compute host at
  launch (section 0.2); a seed assertion (60 unique seeds, exactly 0..59), tested before it was added to the runbook, is
  archived as `seeds_verified.json` (section 0.3). Sections 5, 9 and 10 now require `git_commit` in `config.json`,
  `environment.txt` and `seeds_verified.json`.

## 2026-07-08 — MAIN-001 execution

- Executed per the runbook on the rev. 5 manifest at commit `7f624ca27a86fbbbe67afc68ebb886a4edbb67a2` (verified at
  launch) on an AMD Ryzen 9 9950X (16 cores / 32 threads), CPU only. Pre-flight passed (grid
  `[0.5, 1.5, 3.5, 8.0, 20.0]`, `SHIFT_COST_RATIO` 3.5, numpy/scipy, seeds 0..59, environment captured).
- 15 shards, one work unit each, detached with `setsid`, BLAS/OMP threads pinned to 1 (scipy-openblas; prevents
  15-way oversubscription); about 1 h wall-clock; no failures.
- Results: 61/61 required completion entries; 45/45 HPO checkpoints with exactly 5 folds and finite regret and CVaR;
  15/15 containment reports; DP oracle `certified_equivalent=true` (maximum discrepancy 4.5e-14). Statistics (BCa,
  Wilcoxon + Cliff's δ + Holm per cost ratio, Friedman + Holm, CVaR, containment) computed. The ROVA-versus-baseline
  verdict changed across the cost spectrum in all 3 environments.
- Deviation (infrastructure; no effect on scientific data): `run_pipeline._mark_done` and `utils.save_json` update the
  shared `manifest.json` without locking, and at 15 shards the live manifest lost updates (4/61 entries; the parallel
  path had been tested only with 2 shards). Per-unit checkpoints are single-writer and were unaffected. On the
  project's decision of 2026-07-08 ("continue + reconstruct"; `authorized_by` in `config.json`) the manifest was
  rebuilt from the per-unit artifacts by `raw/rebuild_and_validate.py` (which validates the 61 required entries);
  the raced file is kept as `raw/manifest.live_raced.json`. No change of scope, hyperparameters, environments, grid
  or protocol; fixing the I/O path was left for later code.
- The finalisation step also produced the distribution-shift reports for `rova` and `ewma` (optional; not a
  completion criterion).
- Archived in `experiments/MAIN-001_core_comparison/` (raw/, analysis.py, config.json, seeds.json, REPORT.md,
  stats_summary.json) and registered in `EXPERIMENT_INDEX.md`; re-running `analysis.py` on `raw/` alone reproduced
  the outputs byte for byte.

## 2026-07-09 to 2026-07-10 — ABLATION-001 execution and archival

- Branch `ablation-001` created from tag `main-001-frozen` (`7f624ca`), which was set on the MAIN-001 commit;
  MAIN-001 was not modified (its archive re-verified 88/88 at every step).
- FIX-1: `e_min` stays tunable in the `no_memory` and `no_decay` ablations because it feeds the propagation path
  independently of memory and decay; checked by falsification (active in the coupled environments, exactly inert
  — spread 0.0 — in the single-agent uncoupled one).
- ABLATION-001 tooling (producing commit `4533c61`; tag `ablation-001-frozen`): driver
  `run_ablation.py` with atomic per-shard manifests (FIX-5); read-only distributional-consistency validator
  `validate_consistency.py`; DR3 descriptive analysis `dr3_attribution.py`; DR3 regime inference
  `dr3_regime_inference.py`; all with tests.
- DR3 method: the frozen per-cell test has no power at K_out = 5 (even a maximal signal cannot be flagged), and
  pooling folds is pseudoreplication (folds are nested in cells, cells in 3 environments). A two-stage cell-level
  analysis was specified in advance: Cliff's δ, environment-clustered inference, exact within-cell sign-flip
  permutation (Option B primary, Option C sensitivity); effect sizes primary; Holm and BH both reported; "not
  detected" is not "inert".
- Executed as 15 shards under `systemd-run --user` (one transient service per cell, single-threaded BLAS), about
  2 h 13 min, no failures. Structural audit passed (75/75 checkpoints, 375/375 folds). The consistency validator
  (hard gate) passed 15/15: the ablation `full` branch agrees with archived MAIN-001 ROVA within Monte-Carlo noise.
- Finding at the time: memory, propagation and stability are load-bearing in the high-cost regime where ROVA wins
  (Cliff's δ > 0, environment-clustered interval excludes 0, BH-FDR significant); decay was not detected at this
  budget; the contributing set differs by regime (H2, descriptive; permutation p ≤ 0.006). Containment (DR4) holds in
  that regime (`prop_scale`/`stab_scale` bounded away from zero). Limitation: nothing survives Holm-FWER with about 6
  cells per regime. The revision's factorial ablation (REV-001) reaches a different conclusion and supersedes this
  finding.
- Post-processing left `raw/` byte-identical (same digest before and after) and wrote derived analyses elsewhere.
- Archived in `experiments/ABLATION-001_mechanism_attribution/` (raw/, analysis/, config.json, REPORT.md,
  reproduce.sh, SHA256SUMS over 132 files, immutable tarball with `.sha256`) and registered in `EXPERIMENT_INDEX.md`;
  the derived analyses regenerate from `raw/` at tag `ablation-001-frozen`, and the tarball and a fresh extraction
  verify 132/132.

## 2026-09-29 — revision: protocol, deviation record, REV-001 plan and code

- The original internal evaluation protocol published verbatim (`docs/PRE_REGISTERED_PROTOCOL.md`; never registered
  externally) with the deviation record (`docs/PREREGISTRATION_DEVIATIONS.md`): each commitment, what MAIN-001 and
  ABLATION-001 executed, and what the revision restores.
- Dated REV-001 analysis plan (`docs/REVISION_ANALYSIS_PLAN.md`), tagged `revision-001-analysis-plan` (`8dbb054`) with its initial version before any revision analysis ran. Its dated amendments, made
  before execution, are logged in its section 10 and included at tag `revision-001-frozen`.
- Revision code (`rova_revision/`) frozen at tag `revision-001-frozen` (`053a1fa`).
- Correction: each method's evaluation worlds were seeded from a method-specific tag, so the published paired
  comparisons did not pair worlds (deviation record, section 3). The revision re-runs the published analysis code
  on matched worlds; archived files are not modified.

## 2026-09-30 to 2026-10-01 — REV-001 execution

- Executed on the lab workstation from `revision-001-frozen` with 15 workers, 2026-09-30 19:57 to 2026-10-01 08:54
  (UTC+05:00). Environment gate and reproduction gate R0 passed; no failed units. Output archive SHA-256 `63dae057…`
  (`experiments/REV-001_revision_analyses/`).

## 2026-10-01 — REV-002: plan, A1, A2

- Dated REV-002 analysis plan (`docs/REVISION_002_ANALYSIS_PLAN.md`), tagged `revision-002-analysis-plan`
  (`58affad`) before A1 or A2 ran; code tagged `revision-002-frozen` (`226186c`).
- A2: fold-clustered re-analysis of the stored REV-001 per-world regrets, no simulation; a second run reproduced every
  file byte for byte (`experiments/REV-002_A2_cluster_reanalysis/`).
- A1: tuning convergence, executed on the lab workstation from `revision-002-frozen`, 2026-10-01 16:39 to 19:24
  (UTC+05:00); no failed units. Output archive SHA-256 `35459e1e…` (`experiments/REV-002_A1_tuning_convergence/`).

## 2026-10-02 — exploratory post-run analysis P1

- P1, a behavioural comparison with always-full-supervision, specified after the A1 and A2 results
  (`docs/REVISION_002_POSTRUN_ANALYSES.md`) and reported as exploratory. It reads stored outputs only; commit
  `f094e20` (`experiments/REV-002_P1_behaviour_a0/`).

## 2026-10-02 to 2026-10-03 — figures

- `rova_revision/paper_figures.py`: the paper's figures, drawn from the stored REV-001 and A1 tables (presentation
  only).

## 2026-10-06 — v2.0 release preparation

- Revision outputs added under `experiments/`, indexed with digests and verification commands in
  `experiments/REVISION_OUTPUTS.md`.
- `.gitattributes`: files are checked out byte for byte, so the checksum manifests also verify on Windows checkouts
  with `core.autocrlf` enabled (previously all 88 MAIN-001 and 132 ABLATION-001 entries failed there).
- Documents written for v1.0, including the entries above this revision section, are kept as written; where the
  revision analyses reach a different conclusion, they are superseded.

## 2026-10-15 — documentation cleanup

- Wording of `docs/CHANGELOG.md`, `docs/EXPERIMENT_MANIFEST_FINAL.md`, `docs/HOSTILE_SELF_REVIEW.md` and
  `docs/PROVENANCE_AUDIT.md` condensed; one factual correction in the manifest (the frozen grid has 5 points, not 7).
- Licence of the Inter fonts bundled with the Figure 1 source added (`figures/LICENSE-Inter-OFL.txt`).
- No archived output, checksum manifest or code behaviour changed. Not part of tag `v2.0`, which is
  unchanged; released as v2.0.1.
