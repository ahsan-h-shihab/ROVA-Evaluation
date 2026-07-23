# ABLATION-001 — Mechanism Attribution (DR3) — Final Report

**Status:** complete · **Executed:** 2026-07-09 · **Archived:** 2026-07-10
**Producing commit:** `4533c61` (branch `ablation-001`, tag `ablation-001-frozen`),
branched from the frozen MAIN-001 anchor `main-001-frozen` (`7f624ca`).

## Objective
Attribute ROVA's behaviour to its components (memory, decay, propagation,
stability) via one-at-a-time re-tuned ablations under the **frozen MAIN-001
protocol**, and determine whether the contributing set differs between the
low-cost regime (ROVA does not win) and the high-cost regime (ROVA wins).

## Design & execution
- Cells = 3 frozen environments × 5 frozen cost ratios = 15. Per cell: full
  ROVA + 4 one-at-a-time ablations, each independently re-tuned at the frozen
  budget (K_out=5, K_in=3, n_trials=80, n_seeds=60). e_min kept tunable in the
  memory/decay ablations (FIX-1).
- 75 nested-CV-HPO campaigns; 15-way parallel via `systemd-run --user`
  (isolated services, single-threaded BLAS, no oversubscription); ~2h13m.

## Completion & integrity (structural audit: PASS, 0 problems)
- 75/75 checkpoints, 375/375 folds, all finite, budget 5/3/80 in every unit.
- 15 ablation reports + 15 containment reports; 15 shard manifests; union 90/90.
- Seeds 0..59. No reboot during the run; zero ablation-related journal errors.

## Cross-experiment consistency (HARD GATE: PASS 15/15)
Every ablation-`full` cell agrees with archived MAIN-001 ROVA within
Monte-Carlo noise (CI overlap in all 15 cells). No harness drift. Ablation-full
is an independent Monte-Carlo draw (different run tag), so agreement is
distributional, not bit-identical — as designed.

## DR3 attribution — primary result (Cliff's δ; δ>0 ⇒ removing the component worsens regret)

| Mechanism | Low-cost (loses): δ [env-CI], Option-C p | High-cost (wins): δ [env-CI], Option-C p |
|---|---|---|
| memory | −0.58 [−0.96,−0.34] p=0.062 | **+0.26 [+0.20,+0.32] p=0.031** (6/6) |
| propagation | −0.38 [−0.49,−0.29] p=0.031 | **+0.34 [+0.27,+0.39] p=0.031** (6/6) |
| stability | −0.22 [−0.45,+0.06] p=0.125 | **+0.27 [+0.12,+0.37] p=0.031** (4/6) |
| decay | +0.12 [−0.08,+0.35] p=0.250 | +0.25 [−0.09,+0.43] p=0.125 (not detected) |

**In the high-cost regime, memory, propagation, and stability are load-bearing**
(positive δ, env-cluster CIs excluding 0). **Decay is not detected at this
budget.** In the low-cost regime the same mechanisms are null or negative
(removing memory/propagation *improves* regret).

## H2 — regime dependence (pre-specified DESCRIPTIVE contrast)

| Mechanism | δ(high)−δ(low) | permutation p |
|---|---|---|
| memory | +0.84 | 0.002 |
| propagation | +0.72 | 0.002 |
| stability | +0.49 | 0.006 |
| decay | +0.12 | 0.433 |

The contributing set differs by regime: memory, propagation, and stability
flip from non-contributing/harmful (low cost) to load-bearing (high cost) — the
mechanistic "why" for MAIN-001's cost-ratio-dependent verdict.

## Statistical honesty
- **Holm-FWER:** nothing survives in either regime — the pre-registered
  ~6-cells-per-regime power ceiling (a property of the frozen 3-env × 5-cost
  design, not the analysis).
- **BH-FDR (high regime):** memory, propagation, stability survive; decay does not.
- **BH-FDR (low regime):** nothing survives.
- Non-flagged components are "not detected at this budget," **never** "inert."
- Per-cell attribution has zero power at K_out=5 (documented); inference is at
  the regime level (Option B primary + Option C sensitivity).

## Containment (DR4)
In the high-cost regime neither `prop_scale` nor `stab_scale` collapses to zero
in any cell (prop medians 0.002–0.096, stab 0.001–0.026) → the attributed
mechanisms are genuinely used, not a containment artefact. Two isolated cells
outside the win regime hit the floor (unfavorable c0.5 prop→0; favorable c3.5
stab→0). Consistent with MAIN-001's containment pass.

## Limitations
Synthetic environments; marginal (one-at-a-time, non-factorial) attribution;
limited per-cell power (regime-level inference only); 3-environment replication
ceiling; H2 is an exploratory contrast reported descriptively.

## Artifacts
- `raw/` — immutable experiment output (checkpoints, per-shard manifests, per-cell
  reports, environment, seeds; `manifest.json` = in-run auto, possibly partial).
- `analysis/` — derived: `consolidated_manifest.json` (authoritative, 90/90),
  `consistency_report.json`, `dr3_descriptive/`, `dr3_regime/`.
- `config.json` — provenance + reproducibility metadata. `SHA256SUMS` — tree
  checksums. `immutable/` — tarball + checksum. `reproduce.sh` — regeneration.
