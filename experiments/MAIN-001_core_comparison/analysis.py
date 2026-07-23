"""MAIN-001 full statistics pass. Reproduces every summary number from the
authoritative raw artifacts alone (per-unit HPO checkpoints + containment
reports + DP-oracle JSON) and the frozen dtcoa_val stats module -- no
conversation state, no live manifest.

Frozen statistical protocol (EXPERIMENT_MANIFEST_FINAL.md rev.5):
  - Unit of analysis: scenario-instance seed. Pooled per-seed regret =
    concatenation of per-fold test_regrets (n=60), per-seed aligned across
    methods (identical outer-fold split, seed0=0, for every method).
  - PRIMARY pairwise: Wilcoxon signed-rank + Cliff's delta, Holm-corrected
    across the 3 method pairs, per (environment, cost-ratio).
  - SECONDARY omnibus: Friedman + Holm (supporting only; correlated-block
    caveat per protocol).
  - CVaR-95 co-primary: per-outer-fold tail metric (5 values/method).
  - BCa bootstrap CIs on pooled per-seed regret.
  - Containment (DR4) summary across all 15 (env, cost) ROVA tuned configs.

Usage:  python3 analysis.py [DATA_ROOT]
DATA_ROOT must contain hpo_checkpoints/ and the *_containment_report.json /
*_dp_oracle_validation.json files. Defaults to the script directory, else ./raw.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from dtcoa_val.analysis.stats import (
    bca_bootstrap_ci, wilcoxon_paired, cliffs_delta, holm_bonferroni, friedman_holm, describe,
)
from dtcoa_val.experiments.environments import ENVIRONMENTS, COST_RATIO_GRID

METHODS = ["static", "ewma", "rova"]


def _resolve_root():
    here = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1:
        return os.path.abspath(sys.argv[1])
    if os.path.isdir(os.path.join(here, "hpo_checkpoints")):
        return here
    if os.path.isdir(os.path.join(here, "raw", "hpo_checkpoints")):
        return os.path.join(here, "raw")
    return here


ROOT = _resolve_root()
CK = os.path.join(ROOT, "hpo_checkpoints")


def load_ckpt(env, c, method):
    with open(os.path.join(CK, f"{env}__c{c}__{method}.json")) as f:
        return json.load(f)["results"]


def pooled_regret(res):
    return [x for fold in res for x in fold["test_regrets"]]


def per_fold_cvar(res):
    return [fold["test_cvar95_regret"] for fold in res]


def analyze_cell(env, c):
    reg = {m: pooled_regret(load_ckpt(env, c, m)) for m in METHODS}
    cvar = {m: per_fold_cvar(load_ckpt(env, c, m)) for m in METHODS}
    n = len(reg["rova"])
    assert all(len(reg[m]) == n for m in METHODS), "pooled per-seed vectors misaligned"

    # descriptives (BCa) on pooled per-seed regret; CVaR describe on per-fold
    desc = {m: describe(reg[m]) for m in METHODS}
    cvar_desc = {m: describe(cvar[m]) for m in METHODS}

    # PRIMARY: pairwise Wilcoxon + Cliff's delta, Holm across the 3 pairs
    pairs = [("rova", "static"), ("rova", "ewma"), ("static", "ewma")]
    pvals = [wilcoxon_paired(reg[a], reg[b])[1] for a, b in pairs]
    holm = holm_bonferroni(pvals)
    primary = []
    for (a, b), (_, p, thr, reject) in zip(pairs, holm):
        primary.append({
            "pair": f"{a}_vs_{b}",
            "wilcoxon_p": p, "holm_threshold": thr, "holm_reject": bool(reject),
            "cliffs_delta": cliffs_delta(reg[a], reg[b]),
            "median_diff": float(np.median(reg[a]) - np.median(reg[b])),
        })

    # SECONDARY: Friedman + Holm omnibus (supporting only)
    secondary = friedman_holm({m: reg[m] for m in METHODS})

    # verdict direction: does ROVA beat BOTH baselines (lower regret) with Holm-sig?
    rova_better = {}
    for base in ("static", "ewma"):
        rec = next(x for x in primary if x["pair"] == f"rova_vs_{base}")
        rova_better[base] = {"lower_median_regret": rec["median_diff"] < 0,
                             "holm_significant": rec["holm_reject"]}
    verdict = "ROVA_wins" if all(rova_better[b]["lower_median_regret"] and
                                 rova_better[b]["holm_significant"] for b in ("static", "ewma")) \
        else ("ROVA_loses" if all(not rova_better[b]["lower_median_regret"] for b in ("static", "ewma"))
              else "mixed")

    return {"n_seeds": n, "regret_desc": desc, "cvar_desc": cvar_desc,
            "primary_pairwise": primary, "secondary_friedman": secondary,
            "rova_vs_baselines": rova_better, "verdict": verdict}


def containment_summary():
    rows = []
    for env in ENVIRONMENTS:
        for c in COST_RATIO_GRID:
            p = os.path.join(ROOT, f"{env}__c{c}__containment_report.json")
            d = json.load(open(p))
            rows.append({"env": env, "cost_ratio": c,
                         "prop_collapsed_to_zero": d.get("prop_collapsed_to_zero"),
                         "stab_collapsed_to_zero": d.get("stab_collapsed_to_zero"),
                         "prop_scale_median": d.get("prop_scale_median"),
                         "stab_scale_median": d.get("stab_scale_median")})
    return rows


def dp_summary():
    p = os.path.join(ROOT, "unfavorable_oscillation_uncoupled__dp_oracle_validation.json")
    d = json.load(open(p))
    return {"certified_equivalent": d.get("certified_equivalent"),
            "max_discrepancy": d.get("max_discrepancy") or d.get("max_abs_discrepancy")}


def main():
    out = {"root": ROOT, "cost_ratio_grid": COST_RATIO_GRID, "environments": ENVIRONMENTS, "cells": {}}
    print(f"# MAIN-001 statistics pass\n# data root: {ROOT}\n# grid: {COST_RATIO_GRID}\n")
    for env in ENVIRONMENTS:
        print(f"\n===== ENVIRONMENT: {env} =====")
        out["cells"][env] = {}
        for c in COST_RATIO_GRID:
            cell = analyze_cell(env, c)
            out["cells"][env][str(c)] = cell
            rd = cell["regret_desc"]
            print(f"\n-- cost_ratio={c}  (n={cell['n_seeds']} seeds)  VERDICT={cell['verdict']}")
            for m in METHODS:
                d = rd[m]
                print(f"   {m:6s} regret mean={d['mean']:.4f} BCa=({d['ci_lo']:.4f},{d['ci_hi']:.4f}) "
                      f"median={d['median']:.4f} | CVaR95 mean={cell['cvar_desc'][m]['mean']:.4f}")
            for rec in cell["primary_pairwise"]:
                print(f"   [primary] {rec['pair']:16s} Wilcoxon p={rec['wilcoxon_p']:.2e} "
                      f"Holm{'*' if rec['holm_reject'] else ' '} delta={rec['cliffs_delta']:+.3f} "
                      f"med_diff={rec['median_diff']:+.4f}")
            sf = cell["secondary_friedman"]
            print(f"   [secondary] Friedman chi2={sf['friedman_stat']:.2f} p={sf['friedman_p']:.2e}")

    print("\n\n===== VERDICT ACROSS COST SPECTRUM (per environment) =====")
    for env in ENVIRONMENTS:
        verdicts = [(c, out["cells"][env][str(c)]["verdict"]) for c in COST_RATIO_GRID]
        changed = len(set(v for _, v in verdicts)) > 1
        print(f"{env}: " + "  ".join(f"c{c}:{v}" for c, v in verdicts)
              + f"   -> verdict_changes_across_spectrum={changed}")
        out["cells"][env]["_verdict_sequence"] = verdicts
        out["cells"][env]["_verdict_changes"] = changed

    print("\n\n===== CONTAINMENT (DR4) SUMMARY =====")
    cs = containment_summary()
    out["containment"] = cs
    for r in cs:
        print(f"   {r['env']:32s} c{r['cost_ratio']:<5} prop0={r['prop_collapsed_to_zero']!s:5} "
              f"stab0={r['stab_collapsed_to_zero']!s:5} "
              f"prop_med={r['prop_scale_median']} stab_med={r['stab_scale_median']}")

    print("\n===== DP-ORACLE VALIDATION =====")
    dp = dp_summary()
    out["dp_oracle"] = dp
    print(f"   certified_equivalent={dp['certified_equivalent']}  max_discrepancy={dp['max_discrepancy']:.2e}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nwrote stats_summary.json")


if __name__ == "__main__":
    main()
