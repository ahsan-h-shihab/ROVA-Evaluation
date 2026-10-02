"""Manuscript figures drawn from stored REV-001 and A1 tables (presentation only; no
simulation, no new statistics).

    python -m rova_revision.paper_figures --rev001-dir REV001/revision_out --a1-dir A1/a1_out --out FIGDIR
"""
import argparse
import csv
import json
import os

import numpy as np

ENVS = [("favorable_hmm_coupled", "favorable (coupled, N = 10)"),
        ("unfavorable_oscillation_uncoupled", "unfavorable (uncoupled, N = 1)"),
        ("collapse_moderate_coupled", "collapse (coupled, N = 6)")]
GRID = [0.5, 1.5, 3.5, 8.0, 20.0]
SERIES = [  # (source, method, label, colour, linestyle, marker)
    ("rev", "A0", "always full supervision (A0)", "#000000", ":", "s"),
    ("rev", "static", "static, random search 80", "#0072B2", "--", "o"),
    ("rev", "ewma", "EWMA, random search 80", "#E69F00", "--", "o"),
    ("rev", "rova", "ROVA, random search 80", "#009E73", "--", "o"),
    ("a1", "static_tpe320", "static, TPE 320", "#0072B2", "-", "^"),
    ("a1", "ewma_tpe320", "EWMA, TPE 320", "#E69F00", "-", "^"),
    ("a1", "rova_tpe320", "ROVA, TPE 320", "#009E73", "-", "^"),
    ("rev", "learned", "learned (cost-sensitive)", "#CC79A7", "-", "D"),
]


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def regret_figure(rev001_dir, a1_dir, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = {}
    for r in _rows(os.path.join(rev001_dir, "analysis", "tables", "cew_methods.csv")):
        data[("rev", r["method"], r["env"], float(r["c"]))] = (float(r["mean"]), json.loads(r["mean_ci"]))
    for r in _rows(os.path.join(a1_dir, "analysis", "tables", "A1_methods.csv")):
        data[("a1", r["method"], r["env"], float(r["c"]))] = (float(r["mean"]), json.loads(r["mean_ci"]))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, (env, title) in zip(axes, ENVS):
        for src, m, lab, col, ls, mk in SERIES:
            pts = [data.get((src, m, env, c)) for c in GRID]
            y = np.array([p[0] for p in pts])
            lo = np.array([p[1][0] for p in pts])
            hi = np.array([p[1][1] for p in pts])
            ax.errorbar(GRID, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], color=col, ls=ls, marker=mk,
                        ms=3.5, lw=1.1, capsize=2, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(GRID)
        ax.set_xticklabels([str(c) for c in GRID])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("cost ratio c = H/U", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="major", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("mean authorization regret (log scale)", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_regret_by_protocol.{ext}"), dpi=300)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rev001-dir", required=True)
    ap.add_argument("--a1-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    regret_figure(a.rev001_dir, a.a1_dir, a.out)
    print("written", a.out)


if __name__ == "__main__":
    main()
