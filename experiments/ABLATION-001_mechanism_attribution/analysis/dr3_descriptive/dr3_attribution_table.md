# ABLATION-001 DR3 Mechanism Attribution (DESCRIPTIVE per-cell)

> WARNING: the per-cell 'A'/'-' flags below use the frozen per-cell test, which has ZERO statistical power at K_out=5 (it can never flag a component even under a maximal signal). This table is DESCRIPTIVE (effect sizes) only. The valid regime-level attribution is in dr3_regime_inference (Option B + C).

Cells: 15/15. 'A' = per-cell Holm flag (structurally always '-' at this budget); '-' = not detected at this budget (NOT evidence of inertness).

| Mechanism | collapse_moderate_coupled__c0.5 | collapse_moderate_coupled__c1.5 | collapse_moderate_coupled__c3.5 | collapse_moderate_coupled__c8.0 | collapse_moderate_coupled__c20.0 | favorable_hmm_coupled__c0.5 | favorable_hmm_coupled__c1.5 | favorable_hmm_coupled__c3.5 | favorable_hmm_coupled__c8.0 | favorable_hmm_coupled__c20.0 | unfavorable_oscillation_uncoupled__c0.5 | unfavorable_oscillation_uncoupled__c1.5 | unfavorable_oscillation_uncoupled__c3.5 | unfavorable_oscillation_uncoupled__c8.0 | unfavorable_oscillation_uncoupled__c20.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| memory | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| decay | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| propagation | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| stability | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

## Regime-stratified rollup (H2) — DESCRIPTIVE COUNTS ONLY

> These are counts of the zero-power per-cell flags; they are NOT the H2 inference. See dr3_regime_inference for the valid test.

| Mechanism | low (c<=1.5) | transition (c=3.5) | high (c>=8.0) |
|---|---|---|---|
| memory | 0/6 | 0/3 | 0/6 |
| decay | 0/6 | 0/3 | 0/6 |
| propagation | 0/6 | 0/3 | 0/6 |
| stability | 0/6 | 0/3 | 0/6 |

Low-regime contributing set: []

High-regime contributing set: []

Sets differ between regimes (H2): False

_Sets are built from 'attributed' cells only; components absent from a set are NOT detected at this budget, not proven inert._
