# PILOT-001: Budget Sensitivity Assessment — Report

**Objective:** determine whether HPO trials (80), outer folds (5), inner folds (3),
seed count (60), or cost-ratio grid density (7 points) can be reduced without
risking a change in the paper's scientific conclusions.

**Method:** ran the exact frozen HPO/CV procedure parameters on a
speed-reduced environment (horizon=40, n_agents=3 vs. frozen horizon=250,
n_agents=10) to make direct comparison tractable. See `config.json` for the
explicit caveat this implies (reduced scale is plausibly a *harder*, more
conservative test, not an easier one).

## Findings

| Parameter | Finding | Conclusion |
|---|---|---|
| n_trials (80) | 2/5 outer folds still improving 21-29% between trial 40 and 80 (within-run tuning curve); independent n_trials=40 rerun scored 17.4% worse | **Keep at 80** |
| K_out (5) | Reducing to 3 cuts the paired-test sample size for the primary Wilcoxon/Cliff's δ statistic, and degraded mean regret 23.1% | **Keep at 5** |
| K_in (3) | Reducing to 2 selected systematically worse configs: 17.5% worse pooled test regret | **Keep at 3** |
| seeds (60) | Reducing to 30 gave a worse point estimate (13.7%) AND a non-narrower CI — consistent with under-powered selection, not just added noise | **Keep at 60** |
| cost-ratio grid (7 pts) | 3-point spot check (c=0.5, 3.33, 5.0) is non-monotonic (0.008 → 0.094 → 0.057) — real curvature a sparser grid could miss | **Keep at 7** |

## Bottom line

Every tested reduction produced a consistent, same-direction degradation in
selected-policy quality, not just wider CIs from less data. This is the
specific failure mode ("mismatched tuning disadvantages one method") the
whole evaluation protocol exists to prevent. No budget parameter is reduced
as a result of this pilot.

## Reproduction

```
cd experiments/PILOT-001_budget_sensitivity
python3 analysis.py
```
Requires only `raw/*.json` (in this directory) and the `dtcoa_val` package
at `../../code/`. No other project state or conversation history is needed.
Verified: output matches this report exactly, from a fresh process, using
only these two inputs.
