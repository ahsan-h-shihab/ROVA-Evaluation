# Corrected (v2, symmetric-rule) verdict table — MAIN-001

Extracted from the archived per-cell statistics
(`../experiments/MAIN-001_core_comparison/stats_summary.json`); regenerable at
any time with `python verdicts/verdicts_v2.py`.

Cliff's δ convention: **δ < 0 ⇒ ROVA better** (lower regret). The p columns are
**raw Wilcoxon signed-rank p-values**; significance for the verdict is assessed
against Holm step-down thresholds within each cell's method family.
n = 60 seeds per cell.

| Environment | c | δ (vs Static) | p (vs Static) | δ (vs EWMA) | p (vs EWMA) | v2 verdict |
|---|---|---|---|---|---|---|
| favorable_hmm_coupled | 0.5 | +0.98 | 1.6e-11 | +0.78 | 2.6e-09 | ROVA_loses |
| favorable_hmm_coupled | 1.5 | +1.00 | 1.6e-11 | +0.42 | 9.3e-06 | ROVA_loses |
| favorable_hmm_coupled | 3.5 | −0.28 | 7.4e-03 | −0.69 | 1.3e-09 | ROVA_wins |
| favorable_hmm_coupled | 8.0 | −1.00 | 1.6e-11 | −1.00 | 1.6e-11 | ROVA_wins |
| favorable_hmm_coupled | 20.0 | −1.00 | 1.6e-11 | −1.00 | 1.6e-11 | ROVA_wins |
| unfavorable_oscillation_uncoupled | 0.5 | −0.05 | 2.9e-01 | +0.15 | 6.5e-02 | mixed |
| unfavorable_oscillation_uncoupled | 1.5 | +0.98 | 1.6e-11 | +0.56 | 2.9e-06 | ROVA_loses |
| unfavorable_oscillation_uncoupled | 3.5 | −0.25 | 5.0e-02 | −0.68 | 6.0e-09 | mixed |
| unfavorable_oscillation_uncoupled | 8.0 | −1.00 | 1.6e-11 | −1.00 | 1.6e-11 | ROVA_wins |
| unfavorable_oscillation_uncoupled | 20.0 | −1.00 | 1.6e-11 | −1.00 | 1.6e-11 | ROVA_wins |
| collapse_moderate_coupled | 0.5 | +0.37 | 2.5e-05 | +0.02 | 4.7e-01 | mixed |
| collapse_moderate_coupled | 1.5 | +0.62 | 1.2e-08 | −0.11 | 1.5e-01 | mixed |
| collapse_moderate_coupled | 3.5 | +0.42 | 5.7e-04 | −0.47 | 6.2e-06 | mixed |
| collapse_moderate_coupled | 8.0 | −1.00 | 1.6e-11 | −1.00 | 1.6e-11 | ROVA_wins |
| collapse_moderate_coupled | 20.0 | −1.00 | 1.6e-11 | −1.00 | 1.6e-11 | ROVA_wins |

v1 → v2 delta (from the archived statistics): exactly one label changes —
`collapse_moderate_coupled` c = 0.5: `ROVA_loses` → `mixed`. Everything else is
identical under both rules.
