# ABLATION-001 DR3 Regime-Level Attribution (Option B + C)

Primary effect: cliffs_delta (ablation vs full trial regrets). Inference unit: cell (n<=6/regime); environment-level reported as descriptive robustness.
Cells: 15. _not detected at this budget (NOT evidence of inertness)_

## Per component x regime (Cliff's delta; delta>0 => removal worsens regret)

| Mechanism | Regime | n | mean delta | env-cluster 95% CI | sign p | Option-C perm p | non-negligible cells |
|---|---|---|---|---|---|---|---|
| memory | low | 6 | -0.581 | [-0.958, -0.343] | 0.219 | 0.062 | 5/6 |
| memory | high | 6 | 0.255 | [0.201, 0.316] | 0.031 | 0.031 | 6/6 |
| memory | transition | 3 | -0.295 | [-0.707, 0.114] | 1.000 | 0.500 | 2/3 |
| decay | low | 6 | 0.121 | [-0.078, 0.347] | 0.688 | 0.250 | 4/6 |
| decay | high | 6 | 0.246 | [-0.087, 0.426] | 0.688 | 0.125 | 4/6 |
| decay | transition | 3 | -0.087 | [-0.252, 0.078] | 1.000 | 0.500 | 1/3 |
| propagation | low | 6 | -0.383 | [-0.490, -0.287] | 0.031 | 0.031 | 6/6 |
| propagation | high | 6 | 0.337 | [0.265, 0.393] | 0.031 | 0.031 | 6/6 |
| propagation | transition | 3 | 0.230 | [0.114, 0.349] | 0.250 | 0.250 | 2/3 |
| stability | low | 6 | -0.222 | [-0.451, 0.060] | 0.219 | 0.125 | 4/6 |
| stability | high | 6 | 0.272 | [0.115, 0.370] | 0.031 | 0.031 | 4/6 |
| stability | transition | 3 | -0.041 | [-0.206, 0.082] | 1.000 | 1.000 | 1/3 |

## H2 (does the contributing set differ between regimes?) — DESCRIPTIVE

| Mechanism | mean delta low | mean delta high | high-low | perm p (descriptive) |
|---|---|---|---|---|
| memory | -0.581 | 0.255 | 0.837 | 0.002 |
| decay | 0.121 | 0.246 | 0.124 | 0.433 |
| propagation | -0.383 | 0.337 | 0.720 | 0.002 |
| stability | -0.222 | 0.272 | 0.494 | 0.006 |

## Multiplicity (Option-C p across the 4-component family)

| Regime | Mechanism | p | Holm reject | BH reject |
|---|---|---|---|---|
| low | memory | 0.062 | False | False |
| low | decay | 0.250 | False | False |
| low | propagation | 0.031 | False | False |
| low | stability | 0.125 | False | False |
| high | memory | 0.031 | False | True |
| high | decay | 0.125 | False | False |
| high | propagation | 0.031 | False | True |
| high | stability | 0.031 | False | True |

_Environment-level (n=3) effects are in the JSON as descriptive robustness only._
