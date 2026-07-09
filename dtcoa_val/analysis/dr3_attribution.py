"""ABLATION-001 cross-cell DR3 attribution analysis (FIX-3).

Recomputes mechanism attribution from the RAW per-fold ablation checkpoints
(never from the driver's saved per-cell reports), so it is an independent
verification layer as well as the manuscript-output generator. It produces:

  1. a per-cell attribution table (component -> attributed / not-detected,
     with Cliff's delta and Holm-corrected p),
  2. a machine-readable cross-cell summary JSON,
  3. a manuscript attribution table (Markdown),
  4. a regime-stratified rollup (low vs high cost-ratio) addressing H2:
     does the contributing component set differ between the regime where ROVA
     does not win and the regime where it wins?

Statistical method is the frozen ablation_report (Wilcoxon signed-rank +
Cliff's delta, Holm-corrected across the component family) -- reused verbatim,
recomputed from raw. Per-cell power is limited by the number of outer folds:
a component that is not flagged is reported as "not detected at this budget,"
NEVER as "inert." This wording is enforced in every generated artifact.

Regime boundary (documented design constant, NOT a runtime read of MAIN-001):
MAIN-001's frozen verdict table shows ROVA does not win at cost ratios
{0.5, 1.5}, wins unambiguously at {8.0, 20.0}, with 3.5 as the transition.
We therefore stratify low = c <= LOW_REGIME_MAX, high = c >= HIGH_REGIME_MIN,
and report the transition cost ratio separately so nothing is hidden.

Read-only with respect to MAIN-001; reads only the ABLATION-001 output tree.
"""
import argparse
import os

from dtcoa_val.utils import load_json, save_json
from dtcoa_val.experiments.environments import ENVIRONMENTS, COST_RATIO_GRID
from dtcoa_val.experiments.ablation_study import ablation_report, containment_report

BRANCHES = ["full", "no_memory", "no_decay", "no_propagation", "no_stability"]
COMPONENTS = ["no_memory", "no_decay", "no_propagation", "no_stability"]
_MECHANISM = {"no_memory": "memory", "no_decay": "decay",
              "no_propagation": "propagation", "no_stability": "stability"}

# See module docstring: grounded in MAIN-001's frozen verdict table, encoded as
# a constant so this analysis never reads MAIN-001 outputs at runtime.
LOW_REGIME_MAX = 1.5
HIGH_REGIME_MIN = 8.0

NOT_DETECTED = "not detected at this budget (NOT evidence of inertness)"


def regime_of(c):
    if c <= LOW_REGIME_MAX:
        return "low"
    if c >= HIGH_REGIME_MIN:
        return "high"
    return "transition"


def _load_cell(ckpt_dir, env, c):
    """Assemble {branch: per-fold results} from raw checkpoints for one cell."""
    out = {}
    for b in BRANCHES:
        p = os.path.join(ckpt_dir, f"{env}__c{c}__{b}.json")
        if not os.path.exists(p):
            return None
        out[b] = load_json(p)["results"]
    return out


def recompute_cell(ckpt_dir, env, c):
    """Recompute DR3 + containment for one cell from raw. Returns None if the
    cell's checkpoints are incomplete."""
    res = _load_cell(ckpt_dir, env, c)
    if res is None:
        return None
    dr3 = ablation_report(res)
    cont = containment_report(res["full"])
    return {"dr3": dr3, "containment": cont}


def build_summary(ablation_out, envs=None, grid=None, driver_report_dir=None):
    """Recompute every cell; cross-check against the driver's saved reports when
    present; return the machine-readable summary dict."""
    envs = list(envs or ENVIRONMENTS)
    grid = list(grid or COST_RATIO_GRID)
    ckpt_dir = os.path.join(ablation_out, "hpo_checkpoints")
    driver_report_dir = driver_report_dir or ablation_out

    cells = {}
    missing = []
    mismatches = []
    for env in envs:
        for c in grid:
            tag = f"{env}__c{c}"
            rec = recompute_cell(ckpt_dir, env, c)
            if rec is None:
                missing.append(tag)
                continue
            cells[tag] = {
                "env": env, "cost_ratio": c, "regime": regime_of(c),
                "attribution": {
                    comp: {
                        "mechanism": _MECHANISM[comp],
                        "attributed": bool(rec["dr3"][comp]["significant_degradation"]),
                        "cliffs_delta_vs_full": rec["dr3"][comp]["cliffs_delta_vs_full"],
                        "holm_p": rec["dr3"][comp]["holm_p"],
                        "interpretation": ("contributes (significant, non-negligible "
                                           "degradation when removed)"
                                           if rec["dr3"][comp]["significant_degradation"]
                                           else NOT_DETECTED),
                    } for comp in COMPONENTS
                },
                "containment": {
                    "prop_scale_median": rec["containment"]["prop_scale_median"],
                    "stab_scale_median": rec["containment"]["stab_scale_median"],
                    "prop_collapsed_to_zero": rec["containment"]["prop_collapsed_to_zero"],
                    "stab_collapsed_to_zero": rec["containment"]["stab_collapsed_to_zero"],
                },
            }
            # Independent-verification cross-check vs the driver's saved report.
            drv_path = os.path.join(driver_report_dir, f"{tag}__ablation_report.json")
            if os.path.exists(drv_path):
                drv = load_json(drv_path)
                for comp in COMPONENTS:
                    a = bool(drv[comp]["significant_degradation"])
                    b = cells[tag]["attribution"][comp]["attributed"]
                    if a != b:
                        mismatches.append(f"{tag}/{comp}: driver={a} recomputed={b}")

    rollup = _regime_rollup(cells)
    return {
        "n_cells": len(cells),
        "n_expected": len(envs) * len(grid),
        "missing_cells": missing,
        "recompute_vs_driver_mismatches": mismatches,
        "regime_boundary": {"low_max": LOW_REGIME_MAX, "high_min": HIGH_REGIME_MIN,
                            "transition_note": "cost ratios strictly between are 'transition'"},
        "wording_policy": NOT_DETECTED,
        # The per-cell 'attributed' flags below come from the frozen per-cell
        # test (Wilcoxon over 5 folds + Holm across 4 components), which has
        # STRUCTURALLY ZERO power at K_out=5 (min achievable p = 0.0625 > the
        # 0.0125 Holm threshold) -- it can never flag a component even under a
        # maximal signal. This module is therefore DESCRIPTIVE ONLY (per-cell
        # effect sizes + containment). The authoritative regime-level attribution
        # inference is dr3_regime_inference.py (Option B primary + Option C).
        "per_cell_test_power_warning": (
            "per-cell 'attributed' flags have zero power at K_out=5; descriptive "
            "only. Use dr3_regime_inference for valid attribution."),
        "authoritative_inference": "dtcoa_val.analysis.dr3_regime_inference",
        "regime_rollup": rollup,
        "cells": cells,
    }


def _regime_rollup(cells):
    """For each component, count attributed cells per regime. The 'contributing
    set' of a regime is the components attributed in >=1 of its cells; the H2
    question is whether these sets differ between low and high."""
    rollup = {}
    for comp in COMPONENTS:
        counts = {"low": [0, 0], "transition": [0, 0], "high": [0, 0]}  # [attributed, total]
        for cell in cells.values():
            reg = cell["regime"]
            counts[reg][1] += 1
            if cell["attribution"][comp]["attributed"]:
                counts[reg][0] += 1
        rollup[comp] = {
            "mechanism": _MECHANISM[comp],
            "low_attributed": counts["low"][0], "low_total": counts["low"][1],
            "transition_attributed": counts["transition"][0], "transition_total": counts["transition"][1],
            "high_attributed": counts["high"][0], "high_total": counts["high"][1],
        }
    low_set = sorted(_MECHANISM[c] for c in COMPONENTS if rollup[c]["low_attributed"] > 0)
    high_set = sorted(_MECHANISM[c] for c in COMPONENTS if rollup[c]["high_attributed"] > 0)
    return {
        "per_component": rollup,
        "low_regime_contributing_set": low_set,
        "high_regime_contributing_set": high_set,
        "sets_differ_between_regimes": low_set != high_set,
        "H2_note": ("Sets are built from 'attributed' cells only; components "
                    "absent from a set are NOT detected at this budget, not proven inert."),
    }


def render_markdown_table(summary):
    """Manuscript attribution table: rows = mechanisms, cols = cells, marks =
    A (attributed) / - (not detected)."""
    cells = summary["cells"]
    tags = sorted(cells, key=lambda t: (cells[t]["env"], cells[t]["cost_ratio"]))
    lines = ["# ABLATION-001 DR3 Mechanism Attribution (DESCRIPTIVE per-cell)\n",
             "> WARNING: the per-cell 'A'/'-' flags below use the frozen per-cell "
             "test, which has ZERO statistical power at K_out=5 (it can never flag "
             "a component even under a maximal signal). This table is DESCRIPTIVE "
             "(effect sizes) only. The valid regime-level attribution is in "
             "dr3_regime_inference (Option B + C).\n",
             f"Cells: {summary['n_cells']}/{summary['n_expected']}. "
             f"'A' = per-cell Holm flag (structurally always '-' at this budget); "
             f"'-' = {NOT_DETECTED}.\n",
             "| Mechanism | " + " | ".join(tags) + " |",
             "|" + "---|" * (len(tags) + 1)]
    for comp in COMPONENTS:
        row = [_MECHANISM[comp]]
        for t in tags:
            row.append("A" if cells[t]["attribution"][comp]["attributed"] else "-")
        lines.append("| " + " | ".join(row) + " |")
    r = summary["regime_rollup"]
    lines += ["\n## Regime-stratified rollup (H2) — DESCRIPTIVE COUNTS ONLY\n",
              "> These are counts of the zero-power per-cell flags; they are NOT "
              "the H2 inference. See dr3_regime_inference for the valid test.\n",
              "| Mechanism | low (c<=1.5) | transition (c=3.5) | high (c>=8.0) |",
              "|---|---|---|---|"]
    for comp in COMPONENTS:
        pc = r["per_component"][comp]
        lines.append(f"| {pc['mechanism']} | {pc['low_attributed']}/{pc['low_total']} | "
                     f"{pc['transition_attributed']}/{pc['transition_total']} | "
                     f"{pc['high_attributed']}/{pc['high_total']} |")
    lines += [f"\nLow-regime contributing set: {r['low_regime_contributing_set']}",
              f"\nHigh-regime contributing set: {r['high_regime_contributing_set']}",
              f"\nSets differ between regimes (H2): {r['sets_differ_between_regimes']}",
              f"\n_{r['H2_note']}_"]
    return "\n".join(lines) + "\n"


def run(ablation_out, out_dir=None):
    out_dir = out_dir or os.path.join(ablation_out, "dr3_analysis")
    os.makedirs(out_dir, exist_ok=True)
    summary = build_summary(ablation_out)
    save_json(os.path.join(out_dir, "dr3_summary.json"), summary)
    md = render_markdown_table(summary)
    with open(os.path.join(out_dir, "dr3_attribution_table.md"), "w") as f:
        f.write(md)
    return summary, out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-out", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    summary, out_dir = run(args.ablation_out, args.out_dir)
    print(f"DR3 analysis written to {out_dir}")
    print(f"cells: {summary['n_cells']}/{summary['n_expected']}  "
          f"missing: {len(summary['missing_cells'])}  "
          f"recompute-vs-driver mismatches: {len(summary['recompute_vs_driver_mismatches'])}")
    r = summary["regime_rollup"]
    print(f"low set:  {r['low_regime_contributing_set']}")
    print(f"high set: {r['high_regime_contributing_set']}")
    print(f"H2 sets differ: {r['sets_differ_between_regimes']}")
