# Experiment Index

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
