# Experiment Index

The MAIN-001, PILOT-001 and ABLATION-001 entries were written when those
experiments completed and are kept as written. Where the revision analyses
(REV-001, REV-002, below) reach a different conclusion, those entries are
superseded; see `docs/PREREGISTRATION_DEVIATIONS.md` and
`experiments/REVISION_OUTPUTS.md`.

## MAIN-001
- **Objective:** Frozen rev. 5 core comparison — 45 nested-CV-HPO units
  (3 methods × 3 environments × 5 cost ratios) supporting the primary claim
  that the ROVA-vs-baseline verdict changes across the operating-cost
  spectrum and survives the containment (DR4) check, under matched, fairly
  tuned, nested-CV-validated evaluation.
- **Date:** 2026-07-08
- **Status:** Complete
- **Output directory:** `experiments/MAIN-001_core_comparison/`
- **Configuration:** `experiments/MAIN-001_core_comparison/config.json`
  (git commit `7f624ca27a86fbbbe67afc68ebb886a4edbb67a2`, verified)
- **Seed file:** `experiments/MAIN-001_core_comparison/seeds.json` (0..59)
- **Summary:** 61/61 required completion entries; 45/45 checkpoints at 5 folds
  (finite regret + CVaR); 15/15 containment reports; DP-oracle
  `certified_equivalent=true` (max discrepancy 4.5e-14); zero failures.
  Verdict changes across the cost spectrum in all 3 environments
  (low cost → ROVA does not beat baselines; high cost → ROVA wins decisively,
  Holm-significant, Cliff's δ→−1.0). Full detail:
  `experiments/MAIN-001_core_comparison/REPORT.md`.
- **Deviation:** 15-shard lost-update race on the shared `manifest.json`
  (non-atomic/unlocked read-modify-write) — infrastructure only, zero
  scientific-data impact (per-unit checkpoints are single-writer). Manifest
  reconstructed and validated from authoritative artifacts
  (`raw/rebuild_and_validate.py`); raced file preserved as
  `raw/manifest.live_raced.json`. No scope/hyperparameter/protocol change.
- **Reproducibility:** Verified — `analysis.py` regenerates `stats_summary.json`
  from `raw/*` and `../../code/` alone in a fresh process, byte-identical
  (identical sha256, root path excluded).

## PILOT-001
- **Objective:** Evidence-based check of whether HPO trials/K_out/K_in/seeds/
  cost-ratio-grid can be reduced below frozen full-budget values without
  risking a change in scientific conclusions.
- **Date:** 2026-07-04
- **Status:** Complete
- **Output directory:** `experiments/PILOT-001_budget_sensitivity/`
- **Configuration:** `experiments/PILOT-001_budget_sensitivity/config.json`
- **Seed file:** `experiments/PILOT-001_budget_sensitivity/seeds.json`
- **Summary:** All 5 tested reductions (n_trials→40, K_in→2, K_out→3,
  seeds→30, cost-ratio grid→3pt) produced consistent, same-direction
  degradation in selected-policy quality (13.7-23.1% worse pooled regret) or
  revealed non-monotonic structure a sparser grid would miss. **No budget
  parameter reduced.** Full detail: `experiments/PILOT-001_budget_sensitivity/REPORT.md`.
- **Reproducibility:** Verified — `analysis.py` regenerates every reported
  number from `raw/*.json` and `../../code/` alone, in a fresh process, with
  no dependence on conversation history or any other project state.

## ABLATION-001
- **Objective:** Mechanism attribution (DR3) — 75 nested-CV-HPO campaigns
  (full + 4 one-at-a-time re-tuned ablations × 3 environments × 5 cost ratios)
  determining which ROVA components (memory, decay, propagation, stability) are
  load-bearing and whether the contributing set differs by cost-ratio regime,
  under the frozen MAIN-001 protocol.
- **Date:** 2026-07-09 (executed); 2026-07-10 (archived)
- **Status:** Complete
- **Output directory:** `experiments/ABLATION-001_mechanism_attribution/`
- **Configuration:** `experiments/ABLATION-001_mechanism_attribution/config.json`
  (git commit `4533c61aa2e252698fd0c31d5bc1f9b9d04c18f0`, tag
  `ablation-001-frozen`, branched from `main-001-frozen`/`7f624ca`; verified)
- **Seed file:** `experiments/ABLATION-001_mechanism_attribution/raw/seeds_verified.json` (0..59)
- **Summary:** 90/90 completion entries (75 branch HPO units + 15 ablation +
  15 containment reports); 375/375 checkpoints finite at 5 folds; structural
  audit PASS (0 problems); cross-experiment distributional-consistency validator
  **PASS 15/15 (hard gate)**; DR3 recompute-vs-driver mismatches 0. Result: in
  the high-cost regime memory/propagation/stability are load-bearing (Cliff's
  δ>0, env-cluster CI excludes 0, BH-FDR significant); decay not detected at this
  budget; the contributing set differs by regime (H2 descriptive, permutation
  p≤0.006). Containment (DR4) passes in the win regime. Full detail:
  `experiments/ABLATION-001_mechanism_attribution/REPORT.md`.
- **Immutable archive:** `immutable/ABLATION-001_mechanism_attribution_20260710.tar.gz`
  (+ `.sha256`); tree checksums `SHA256SUMS` (132 files). Tarball and a fresh
  extraction both verify (132/132).
- **Reproducibility:** Verified — raw byte-immutable across the entire
  post-processing pipeline (identical digest pre/post); derived analyses
  regenerate from `raw/` + `../../code/` at commit `4533c61` via `reproduce.sh`
  (validator → DR3 descriptive → DR3 regime). Determinism: same-tag reruns
  reproduce checkpoints bit-for-bit.

## REV-001
- **Objective:** Revision analyses specified in `docs/REVISION_ANALYSIS_PLAN.md`:
  the protocol commitments restored according to
  `docs/PREREGISTRATION_DEVIATIONS.md`, and the matched-world correction of the
  published paired analyses.
- **Date:** 2026-09-30 to 2026-10-01
- **Status:** Complete
- **Output:** `experiments/REV-001_revision_analyses/revision_out.tar.gz`
  (SHA-256 `63dae057d63c4917d90391fd9da72279f5e909ffb1a1b7e192999f55d28a6b10`)
- **Plan and code:** tags `revision-001-analysis-plan` (`8dbb054`) and
  `revision-001-frozen` (`053a1fa`)
- **Seeds:** 0..59 (the frozen study seeds)
- **Summary:** `analysis/REPORT.md` inside the archive.
- **Verification:** `rova_revision.verify_outputs` (see
  `experiments/REVISION_OUTPUTS.md`).

## REV-002 A1 (tuning convergence)
- **Objective:** TPE tuning continued to 320 candidates, with selections at 80,
  160 and 320 (`docs/REVISION_002_ANALYSIS_PLAN.md`).
- **Date:** 2026-10-01
- **Status:** Complete
- **Output:** `experiments/REV-002_A1_tuning_convergence/a1_out.tar.gz`
  (SHA-256 `35459e1e096135531849a3a96d3f7501392c8d4f10474f00bc83913baa908e03`)
- **Plan and code:** tags `revision-002-analysis-plan` (`58affad`) and
  `revision-002-frozen` (`226186c`)
- **Summary:** `analysis/REPORT.md` inside the archive.
- **Verification:** `rova_revision.verify_a1`.

## REV-002 A2 (fold-clustered re-analysis)
- **Objective:** Re-analysis of the stored REV-001 per-world regrets with the
  outer fold as the cluster (plan, section 3). No simulation.
- **Date:** 2026-10-01
- **Status:** Complete
- **Output:** `experiments/REV-002_A2_cluster_reanalysis/` (`SHA256SUMS`)
- **Code:** tag `revision-002-frozen` (`226186c`)
- **Summary:** `A2_REPORT.md`.
- **Reproducibility:** a second run reproduced every file byte for byte.

## REV-002 P1 (exploratory)
- **Objective:** Behavioural comparison of the tuned policies with
  always-full-supervision, specified after the A1 and A2 results
  (`docs/REVISION_002_POSTRUN_ANALYSES.md`). No simulation.
- **Date:** 2026-10-02
- **Status:** Complete
- **Output:** `experiments/REV-002_P1_behaviour_a0/` (`SHA256SUMS`)
- **Code:** commit `f094e20`
- **Summary:** `BEHAVIOUR_A0_REPORT.md`.
- **Reproducibility:** a second run reproduced every file byte for byte.
