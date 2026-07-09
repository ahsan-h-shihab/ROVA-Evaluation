"""ABLATION-001 distributional-consistency validator (FIX-4).

Confirms that the ablation "full" branch and the archived MAIN-001 ROVA agree
*within Monte-Carlo noise* per cell -- NOT bit-for-bit. Per-trial simulation
seeding derives from a hash that includes the run tag (utils.trial_seed), and
the ablation-full tag (`__full`) necessarily differs from MAIN-001's ROVA tag
(`__rova`), so the two are independent Monte-Carlo draws of the same estimand.
Demanding exact equality would fail on a correct run; the correct invariant is
distributional (see VALIDATOR_DESIGN / handoff I11).

Check per cell (recomputed from RAW per-fold checkpoints, never from derived
reports):
  - primary:   confidence-interval overlap of the mean per-fold test regret
               (t-based CI, small-sample), and
  - secondary: a relative-difference guard that also lets a cell pass when both
               distributions are near-zero with degenerate CIs.
A cell PASSES iff the CIs overlap OR the relative difference of means is within
tolerance. Any cell failing BOTH is genuine drift and makes the whole check
exit nonzero (fail loudly).

This module is READ-ONLY with respect to MAIN-001: it only reads the archived
raw checkpoints and never writes into the MAIN-001 tree.

Usage:
    python -m dtcoa_val.experiments.validate_consistency --ablation-out <dir>
    python -m dtcoa_val.experiments.validate_consistency --ablation-out <dir> \
        --main-raw <path-to-MAIN-001/raw/hpo_checkpoints> --rel-tol 0.15
"""
import argparse
import os
import sys

import numpy as np
from scipy import stats

from dtcoa_val.utils import load_json
from dtcoa_val.experiments.environments import ENVIRONMENTS, COST_RATIO_GRID

FLOOR = 1e-9  # guards relative-difference when regret is ~0

# Default location of the archived MAIN-001 raw ROVA checkpoints, resolved
# relative to this file: code/dtcoa_val/experiments/ -> repo code root -> sibling
# experiments/ tree. Overridable via --main-raw.
_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MAIN_RAW = os.path.normpath(os.path.join(
    _CODE_ROOT, "..", "experiments", "MAIN-001_core_comparison", "raw", "hpo_checkpoints"))


def _fold_regrets(ckpt):
    """Per-outer-fold test_mean_regret list from a nested_cv_hpo checkpoint."""
    return [float(r["test_mean_regret"]) for r in ckpt["results"]]


def _mean_ci(vals, alpha=0.05):
    """Small-sample t-based CI for the mean. Returns (mean, lo, hi)."""
    a = np.asarray(vals, float)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return mean, mean, mean
    se = float(a.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        return mean, mean, mean
    t = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return mean, mean - t * se, mean + t * se


def _overlap(ci_a, ci_m):
    _, alo, ahi = ci_a
    _, mlo, mhi = ci_m
    return not (ahi < mlo or mhi < alo)


def compare_distributions(ablation_regrets, main_regrets, rel_tol):
    """Core distributional check for one cell. Returns a diagnostic dict with
    a boolean `pass`. Symmetric and recomputed from the raw per-fold values."""
    ci_a = _mean_ci(ablation_regrets)
    ci_m = _mean_ci(main_regrets)
    mean_a, mean_m = ci_a[0], ci_m[0]
    rel = abs(mean_a - mean_m) / max(abs(mean_m), FLOOR)
    overlap = _overlap(ci_a, ci_m)
    passed = bool(overlap or rel <= rel_tol)
    return {
        "n_folds_ablation": len(ablation_regrets),
        "n_folds_main": len(main_regrets),
        "mean_ablation_full": mean_a,
        "mean_main_rova": mean_m,
        "ci_ablation_full": [ci_a[1], ci_a[2]],
        "ci_main_rova": [ci_m[1], ci_m[2]],
        "ci_overlap": overlap,
        "relative_difference": rel,
        "rel_tol": rel_tol,
        "pass": passed,
    }


def validate_cell(ablation_full_ckpt_path, main_rova_ckpt_path, rel_tol):
    a = load_json(ablation_full_ckpt_path)
    m = load_json(main_rova_ckpt_path)
    diag = compare_distributions(_fold_regrets(a), _fold_regrets(m), rel_tol)
    diag["ablation_budget"] = {k: a.get(k) for k in ("K_out", "K_in", "n_trials")}
    diag["main_budget"] = {k: m.get(k) for k in ("K_out", "K_in", "n_trials")}
    diag["budget_match"] = diag["ablation_budget"] == diag["main_budget"]
    return diag


def validate_experiment(ablation_out, main_raw=DEFAULT_MAIN_RAW, rel_tol=0.15,
                        envs=None, grid=None, verbose=True):
    """Validate every cell. Returns (all_pass, report_dict). Never writes to
    main_raw."""
    envs = list(envs or ENVIRONMENTS)
    grid = list(grid or COST_RATIO_GRID)
    ckpt_dir = os.path.join(ablation_out, "hpo_checkpoints")
    cells = {}
    problems = []
    for env in envs:
        for c in grid:
            tag = f"{env}__c{c}"
            a_path = os.path.join(ckpt_dir, f"{tag}__full.json")
            m_path = os.path.join(main_raw, f"{tag}__rova.json")
            if not os.path.exists(a_path):
                problems.append(f"MISSING ablation-full checkpoint: {tag}")
                continue
            if not os.path.exists(m_path):
                problems.append(f"MISSING MAIN-001 rova checkpoint: {tag}")
                continue
            diag = validate_cell(a_path, m_path, rel_tol)
            cells[tag] = diag
            if not diag["pass"]:
                problems.append(
                    f"DRIFT {tag}: mean_full={diag['mean_ablation_full']:.5f} "
                    f"mean_rova={diag['mean_main_rova']:.5f} "
                    f"rel={diag['relative_difference']:.3f} overlap={diag['ci_overlap']}")
            if not diag["budget_match"]:
                problems.append(f"BUDGET MISMATCH {tag}: {diag['ablation_budget']} "
                                f"vs {diag['main_budget']}")
            if verbose:
                mark = "PASS" if diag["pass"] and diag["budget_match"] else "FAIL"
                print(f"[{mark}] {tag}: full={diag['mean_ablation_full']:.5f} "
                      f"rova={diag['mean_main_rova']:.5f} rel={diag['relative_difference']:.3f} "
                      f"overlap={diag['ci_overlap']}")
    all_pass = len(problems) == 0 and len(cells) == len(envs) * len(grid)
    report = {
        "all_pass": all_pass,
        "n_cells_checked": len(cells),
        "n_cells_expected": len(envs) * len(grid),
        "rel_tol": rel_tol,
        "main_raw": main_raw,
        "problems": problems,
        "cells": cells,
    }
    return all_pass, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-out", required=True,
                    help="ABLATION-001 output dir (must contain hpo_checkpoints/).")
    ap.add_argument("--main-raw", default=DEFAULT_MAIN_RAW,
                    help="Dir with archived MAIN-001 {cell}__rova.json checkpoints (read-only).")
    ap.add_argument("--rel-tol", type=float, default=0.15)
    ap.add_argument("--report-out", default=None,
                    help="Optional path to write the JSON report (in the ablation tree).")
    args = ap.parse_args()
    ok, report = validate_experiment(args.ablation_out, args.main_raw, args.rel_tol)
    if args.report_out:
        from dtcoa_val.utils import save_json
        save_json(args.report_out, report)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"({report['n_cells_checked']}/{report['n_cells_expected']} cells, "
          f"{len(report['problems'])} problems)")
    sys.exit(0 if ok else 1)
