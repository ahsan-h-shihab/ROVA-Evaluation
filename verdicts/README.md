# Verdict tables: archived (v1) vs corrected symmetric (v2)

The paper's per-cell governance-vs-baseline verdicts (Table `tab:verdict`) use
the **v2 symmetric rule**. The archived experiment output
(`../experiments/MAIN-001_core_comparison/stats_summary.json`) carries **v1**
labels computed at freeze time. Both are preserved here, with the derivation
script, so the correction is fully transparent and re-executable.

## The two rules

* **v1 (archived, asymmetric)** — implemented in the archived
  `experiments/MAIN-001_core_comparison/analysis.py`:
  `ROVA_wins` required lower median regret AND a Holm-significant Wilcoxon
  result against **both** baselines; `ROVA_loses` required only a higher median
  against both baselines (no significance or effect-size requirement).
  The asymmetry over-declared losses.
* **v2 (corrected, symmetric)** — used in the manuscript: a definite verdict
  (`ROVA_wins` or `ROVA_loses`) requires, **in the same direction against both
  baselines**: (i) a Holm-significant Wilcoxon signed-rank result (the archived
  step-down decision, `holm_reject`), and (ii) a non-negligible effect size,
  |Cliff's δ| > 0.147. Otherwise `mixed` (no detected difference).

## Files

| File | Content |
|---|---|
| `verdicts_v2.py` | Derives the v2 table from the archived `stats_summary.json` (stdlib only) and reports the v1→v2 delta. |
| `verdicts_v2.json` | Output of the script against the archived statistics: all 15 cells with v1 and v2 labels and per-baseline Cliff's δ. |
| `verdict_table_v2_symmetric.md` | The corrected table in human-readable form, matching the manuscript. |

## Verified property

Running `python verdicts/verdicts_v2.py` from the repository root reproduces
the manuscript table and confirms the delta documented in the paper's
Reproducibility Statement (Appendix D.3): **exactly one** of the 15 cell labels
changes between the rules — `collapse_moderate_coupled`, c = 0.5:
`ROVA_loses` → `mixed` (ROVA is higher-median there but not significant and
negligible in effect vs the EWMA baseline: δ = +0.02, raw Wilcoxon p = 0.47).
All effect sizes, all p-values, the containment result, and the overall outcome
are identical under both rules.
