"""Reproduces every number in REPORT.md from raw/*.json alone.
Run from this directory: python3 analysis.py
Requires only the dtcoa_val package (see ../../code/) on the path -- no
other project state, and no conversation history, is needed.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))
import numpy as np
from dtcoa_val.analysis.stats import bca_bootstrap_ci

RAW = os.path.join(os.path.dirname(__file__), "raw")


def load(tag):
    with open(os.path.join(RAW, f"{tag}.json")) as f:
        return json.load(f)["results"]


def summarize(name, res):
    r = [f["test_mean_regret"] for f in res]
    pooled = [x for f in res for x in f["test_regrets"]]
    ci = bca_bootstrap_ci(pooled)
    print(f"{name}: n_folds={len(res)} per-fold={[round(x,4) for x in r]} "
          f"pooled_mean={round(ci[0],4)} CI=({round(ci[1],4)},{round(ci[2],4)}) "
          f"width={round(ci[2]-ci[1],4)}")
    return pooled, ci


if __name__ == "__main__":
    A = load("pilotA")
    B = load("pilotB_trials40")
    C = load("pilotC_kin2")
    D = load("pilotD_kout3")
    E = load("pilotE_seeds30")
    G05 = load("pilotG_c0.5")
    G5 = load("pilotG_c5.0")

    print("=== Tuning curve convergence (Run A, per outer fold) ===")
    for fold in A:
        print(f"fold {fold['outer_fold']}: ", {k: round(v, 4) for k, v in sorted(fold["tuning_curve"].items())})

    print("\n=== Sensitivity comparisons (all vs Run A reference) ===")
    pA, ciA = summarize("A (ref: trials=80,Kin=3,Kout=5,seeds=60)", A)
    pB, ciB = summarize("B (trials=40)", B)
    pC, ciC = summarize("C (Kin=2)", C)
    pD, ciD = summarize("D (Kout=3)", D)
    pE, ciE = summarize("E (seeds=30)", E)

    print("\n=== Relative degradation vs reference ===")
    for name, ci in [("B (trials40)", ciB), ("C (Kin2)", ciC), ("D (Kout3)", ciD), ("E (seeds30)", ciE)]:
        rel = abs(ci[0] - ciA[0]) / ciA[0] * 100
        print(f"{name}: mean_diff={round(ci[0]-ciA[0],4)} ({round(rel,1)}% relative)")

    print("\n=== Cost-ratio grid density check ===")
    for name, res in [("c=0.5", G05), ("c=3.33 (Run A)", A), ("c=5.0", G5)]:
        r = [f["test_mean_regret"] for f in res]
        print(f"{name}: mean_regret={round(np.mean(r),4)} std_across_folds={round(np.std(r,ddof=1),4)}")
