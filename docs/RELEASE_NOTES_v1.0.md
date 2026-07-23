# ROVA-Evaluation v1.0 — release notes

First public release of the complete evaluation artifact for *"The Verdict Is
in the Protocol: A Circularity-Controlled, Cost-Ratio-Resolved Evaluation of
Runtime Governance for Agentic AI"* (IEEE Access, under review, 2026).

## Release summary

* **MAIN-001** — core cost-ratio comparison: config, seeds, 45 raw HPO
  checkpoints, containment / oracle-certification / distribution-shift reports
* **ABLATION-001** — mechanism attribution: config, seeds, 75 raw checkpoints,
  regime-level inference
* **PILOT-001** — budget-sensitivity pilot
* **Replay package** (`reproduce/`) — numpy-only, deterministic, self-validating
* **SHA256 checksums** — 88-file (MAIN-001) and 132-file (ABLATION-001)
  integrity manifests, plus checksummed immutable archives as release assets
* **Verdict tables** — archived v1 (`stats_summary.json`) and corrected
  symmetric v2 artifacts with the derivation script (`verdicts/`)
* **Analysis scripts** — per-experiment analysis and regime-level DR3 inference
* **Environment specification** — exact pinned freeze (`requirements.txt`,
  per-experiment `environment.txt`)
* **Documentation** — experiment index, execution runbook, provenance and
  self-audit reports, figure sources

## Contents

* Frozen reference implementation (`dtcoa_val/`) with the two annotated
  experiment tags cited in the paper's Appendix D.1:
  * `main-001-frozen` = `7f624ca27a86fbbbe67afc68ebb886a4edbb67a2`
  * `ablation-001-frozen` = `4533c61aa2e252698fd0c31d5bc1f9b9d04c18f0`
* Raw per-cell checkpoints, configs, seed sets, and integrity manifests for
  MAIN-001 (88-file SHA256 manifest), ABLATION-001 (132-file manifest), and the
  PILOT-001 budget-sensitivity pilot.
* Both verdict tables — archived v1 and corrected symmetric v2 — plus
  `verdicts/verdicts_v2.py`, which re-derives v2 from the archived statistics
  (exactly one of 15 labels differs; see `verdicts/README.md`).
* Self-contained replay package (`reproduce/`) that verifies the
  execution-derived claims from archived configs and seeds (numpy-only,
  deterministic, self-validating to 1e-9).
* Provenance and self-audit documentation (`docs/`).

## Release assets (immutable archives)

| Asset | SHA-256 |
|---|---|
| `MAIN-001_core_comparison_20260708.tar.gz` | `c76493f16757793c625e802e24cb86b19c5c73ecec0614ba7ac00c0d02fdebce` |
| `ABLATION-001_mechanism_attribution_20260710.tar.gz` | `33cb516894972716c984459913d49d67b621587db4f796c3080587b37f123cd9` |

## Hardware / runtime

The core comparison was executed on a single AMD Ryzen 9 9950X workstation
(16 physical cores, 59 GiB RAM, Ubuntu) as 15 single-threaded shards with BLAS
and OpenMP threads pinned to one per process; the 45-unit campaign completes in
about one hour of CPU wall-clock, no GPU.

## Known notes and errata (documentation-level; no result is affected)

* **Identical containment reports at c = 8.0 and c = 20.0
  (collapse_moderate_coupled).** The two files are byte-identical because the
  tuned propagation/stability scales coincided at both cost ratios; the
  underlying runs are distinct (their HPO checkpoints differ). Recorded here to
  preempt the natural question.
* **Composite-observation wording.** The paper describes the baselines'
  composite as `nanmean(o)`; the archived implementation imputes missing
  channels by last-value carry-forward and then takes a plain mean (its own
  docstring also says "nanmean"). Behaviorally identical in all evaluated
  environments (p_miss = 0 everywhere) and parallel to ROVA's carry-forward.
* **MNAR scaling flag.** The 1.5× missingness scaling under high hazard is
  gated by an `mnar` config flag (default off); the paper describes the
  mechanism without mentioning the flag. Inactive in all evaluated runs
  (p_miss = 0).
* **Signal-poisoning `spike=0.6`.** The configured hazard-spike value for the
  signal-poisoning adversary profile is inert in code (the hazard override
  applies only to the ramp-defect and collusion profiles), consistent with the
  paper's description of that adversary as emitting high observations "without
  changing true hazard." No adversary profile was active in any evaluated run.
