#!/usr/bin/env python3
"""Derive the corrected (v2, symmetric) verdict table from the archived
MAIN-001 per-cell statistics.

The archived ``stats_summary.json`` carries the v1 verdict labels computed by
``analysis.py`` at freeze time.  The v1 rule was asymmetric: "ROVA_wins"
required Holm-significant superiority against both baselines, but "ROVA_loses"
required only a higher median against both baselines (no significance or
effect-size requirement), which over-declared losses.

The v2 (symmetric) rule used in the manuscript requires, IN THE SAME DIRECTION
against BOTH baselines:
  (i)  a Holm-significant Wilcoxon signed-rank result (the archived
       ``holm_reject`` decision, step-down within the cell's 3-pair method
       family), and
  (ii) a non-negligible effect size, |Cliff's delta| > 0.147.
Otherwise the cell is "mixed" (no detected difference).

Running this script against the archived ``stats_summary.json`` reproduces the
manuscript's verdict table (Table: tab:verdict) and reports the v1 -> v2
delta.  Expected result, documented in the reproducibility statement (D.3):
exactly one cell changes, ``collapse_moderate_coupled c=0.5``,
ROVA_loses -> mixed; every other label, all effect sizes, and all p-values are
unchanged.

Usage:
    python verdicts/verdicts_v2.py [path/to/stats_summary.json]
Default path: experiments/MAIN-001_core_comparison/stats_summary.json
(relative to the repository root).

Only the Python standard library is required.
"""
import json
import sys
import os

NEGLIGIBLE = 0.147  # |Cliff's delta| at or below this is negligible (Romano et al.)


def v2_verdict(cell):
    """Symmetric verdict for one cell dict from stats_summary.json."""
    sides = {}
    for base in ("static", "ewma"):
        rec = next(x for x in cell["primary_pairwise"] if x["pair"] == f"rova_vs_{base}")
        d = rec["cliffs_delta"]
        sides[base] = {
            "delta": d,
            "holm_reject": bool(rec["holm_reject"]),
            "non_negligible": abs(d) > NEGLIGIBLE,
            "direction": "rova_better" if d < 0 else "rova_worse",
        }
    def definite(direction):
        return all(
            s["holm_reject"] and s["non_negligible"] and s["direction"] == direction
            for s in sides.values()
        )
    if definite("rova_better"):
        return "ROVA_wins", sides
    if definite("rova_worse"):
        return "ROVA_loses", sides
    return "mixed", sides


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        "experiments", "MAIN-001_core_comparison", "stats_summary.json")
    with open(path) as f:
        summary = json.load(f)

    rows, diffs = [], []
    for env, by_c in summary["cells"].items():
        for c, cell in by_c.items():
            if c.startswith("_"):
                continue
            v1 = cell["verdict"]
            v2, sides = v2_verdict(cell)
            rows.append({"environment": env, "cost_ratio": float(c),
                         "v1_verdict": v1, "v2_verdict": v2,
                         "delta_vs_static": sides["static"]["delta"],
                         "delta_vs_ewma": sides["ewma"]["delta"]})
            if v1 != v2:
                diffs.append((env, c, v1, v2))

    print(f"{'Environment':<36} {'c':>5}  {'v1':<10} {'v2':<10}")
    for r in rows:
        marker = "  <-- changed" if r["v1_verdict"] != r["v2_verdict"] else ""
        print(f"{r['environment']:<36} {r['cost_ratio']:>5}  "
              f"{r['v1_verdict']:<10} {r['v2_verdict']:<10}{marker}")
    print(f"\n{len(rows)} cells; v1 -> v2 changes: {len(diffs)}")
    for env, c, v1, v2 in diffs:
        print(f"  {env} c={c}: {v1} -> {v2}")

    out = {"rule": ("definite verdict requires, in the same direction vs BOTH "
                    "baselines: Holm-significant Wilcoxon (archived holm_reject) "
                    f"AND |Cliff's delta| > {NEGLIGIBLE}; otherwise mixed"),
           "source": path, "cells": rows,
           "v1_to_v2_changes": [
               {"environment": e, "cost_ratio": c, "v1": a, "v2": b}
               for e, c, a, b in diffs]}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "verdicts_v2.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
