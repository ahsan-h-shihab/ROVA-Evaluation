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


# --------------------------------------------------------------------------- #
# Appendix figures redrawn from the stored REV-001 summary (presentation only).
# Each function also writes the exact values it plots to fig_data_<name>.json so
# the figure can be checked against summary.json without reading the drawing.
# --------------------------------------------------------------------------- #
C_COLOURS = {0.5: "#0072B2", 1.5: "#56B4E9", 3.5: "#009E73", 8.0: "#E69F00", 20.0: "#D55E00"}
C_MARKERS = {0.5: "o", 1.5: "s", 3.5: "^", 8.0: "D", 20.0: "v"}
COMPONENT_LABEL = {"memory": "memory", "decay": "decay", "propagation": "propagation", "stability": "stability"}
METHOD_LABELS = {
    "A0": "A0 (always full supervision)", "A5": "A5 (always high autonomy)",
    "learned": "Learned (cost-sensitive)", "qlearn": "Q-learning", "clairvoyant": "Clairvoyant threshold",
    "static": "Static, RS-80", "ewma": "EWMA, RS-80", "rova": "ROVA, RS-80",
    "static_tpe": "Static, TPE-80", "ewma_tpe": "EWMA, TPE-80", "rova_tpe": "ROVA, TPE-80",
    "static_pb": "Static, PB", "ewma_pb": "EWMA, PB", "rova_pb": "ROVA, PB",
    "static_ext": "Static-cap", "ewma_ext": "EWMA-cap", "rova_single": "ROVA-single",
    "ewma_out": "EWMA, threshold only", "rova_out": "ROVA, band vector only",
    "abl_full": "ROVA, ablation study (full)",
}
GROUP_COLOUR = {"reference": "#000000", "static": "#0072B2", "ewma": "#E69F00", "rova": "#009E73",
                "ablation": "#7FBFA6"}


def method_label(m):
    if m in METHOD_LABELS:
        return METHOD_LABELS[m]
    if m.startswith("abl_no_"):
        off = m[len("abl_no_"):]
        return "ROVA − memory, decay" if off == "memory" else f"ROVA − {COMPONENT_LABEL[off]}"
    if m.startswith("fx_"):
        return "ROVA − " + ", ".join(COMPONENT_LABEL[x] for x in m[3:].split("-"))
    raise KeyError(m)


def method_group(m):
    if m in ("A0", "A5", "learned", "qlearn", "clairvoyant"):
        return "reference"
    if m.startswith(("abl_no_", "fx_")):
        return "ablation"
    return m.split("_")[0] if m.split("_")[0] in ("static", "ewma", "rova") else "rova"


def _summary(rev001_dir):
    with open(os.path.join(rev001_dir, "analysis", "summary.json"), encoding="utf-8") as f:
        return json.load(f)


def _save(fig, out, name, data):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    with open(os.path.join(out, f"fig_data_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)


def cd_figure(rev001_dir, out, regime="high"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = _summary(rev001_dir)["cd"][regime]
    ranks = d["average_ranks"]
    order = sorted(ranks, key=lambda m: (ranks[m], m))
    k = len(order)
    fig, ax = plt.subplots(figsize=(3.5, 0.165 * k + 0.95))
    for i, m in enumerate(order):
        col = GROUP_COLOUR[method_group(m)]
        ax.plot([1, ranks[m]], [i, i], color="0.85", lw=0.6, zorder=1)
        ax.plot([ranks[m]], [i], "o", color=col, ms=3.6, zorder=2)
        ax.text(ranks[m] + 0.7, i, f"{ranks[m]:.2f}", va="center", fontsize=5.5, color="0.25")
    best = ranks[order[0]]
    cd = d["critical_difference"]
    ax.axvspan(best, best + cd, color="0.92", zorder=0)
    ax.plot([best, best + cd], [-1.3, -1.3], color="k", lw=1.6, solid_capstyle="butt")
    ax.text(best + cd + 0.4, -1.3, f"CD = {cd:.2f} (Nemenyi, 0.05)", va="center", fontsize=6)
    ax.set_yticks(range(k))
    ax.set_yticklabels([method_label(m) for m in order], fontsize=6)
    ax.set_ylim(k - 0.4, -2.2)
    ax.set_xlim(0.5, k + 4.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("average rank (1 = lowest regret)", fontsize=7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [Line2D([], [], ls="", marker="o", ms=3.6, color=GROUP_COLOUR[g], label=lab)
               for g, lab in (("reference", "reference policies"), ("static", "Static"), ("ewma", "EWMA"),
                              ("rova", "ROVA"), ("ablation", "ROVA ablations"))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.2, -0.075), ncol=3, fontsize=5.5, frameon=False, columnspacing=0.8, handletextpad=0.2)
    fig.tight_layout()
    _save(fig, out, f"fig_R2g_cd_{regime}",
          {"regime": regime, "average_ranks": ranks, "critical_difference": cd, "blocks": d["blocks"],
           "order": order, "labels": {m: method_label(m) for m in order}})
    plt.close(fig)


SWEEP_TICKS = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 5.0]
SWEEP_TICKLABELS = ["0", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1", "5"]


def sweep_figure(rev001_dir, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sw = _summary(rev001_dir)["sweep"]
    rows = (("prop_scale", "propagation"), ("stab_scale", "stability"))
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3), sharex=True)
    plotted = {}
    for j, (env, title) in enumerate(ENVS):
        for i, (p, plabel) in enumerate(rows):
            ax = axes[i][j]
            for c in GRID:
                cell = sw.get(f"{env}|{c}", {})
                if cell.get("status") != "ok":
                    continue
                pts = cell["dose_response"][p]
                x = [q["value"] for q in pts]
                y = [q["mean"] for q in pts]
                lo = [q["ci"][0] for q in pts]
                hi = [q["ci"][1] for q in pts]
                ax.fill_between(x, lo, hi, color=C_COLOURS[c], alpha=0.15, lw=0)
                ax.plot(x, y, color=C_COLOURS[c], marker=C_MARKERS[c], ms=2.8, lw=1.0, label=f"c = {c:g}")
                plotted[f"{env}|{c}|{p}"] = {"value": x, "mean": y, "ci_lo": lo, "ci_hi": hi}
            ax.set_xscale("symlog", linthresh=1e-3, linscale=0.5)
            ax.set_yscale("log")
            ax.set_xticks(SWEEP_TICKS)
            ax.set_xticklabels(SWEEP_TICKLABELS)
            ax.tick_params(labelsize=6.5)
            ax.grid(True, which="major", lw=0.3, alpha=0.5)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if i == 1:
                ax.set_xlabel("penalty magnitude", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{plabel} penalty varied\nmean regret (log scale)", fontsize=7)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out, "fig_R11_penalty_sweep", plotted)
    plt.close(fig)


def dp_figure(rev001_dir, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dp = _summary(rev001_dir)["dp_reduced"]
    envs = [e for e in ENVS if any(r["env"] == e[0] for r in dp)]
    fig, axes = plt.subplots(1, len(envs), figsize=(3.5, 2.5), sharey=True)
    plotted = {}
    for ax, (env, title) in zip(axes, envs):
        for c in GRID:
            rs = sorted((r for r in dp if r["env"] == env and r["c"] == c and r.get("status") == "ok"),
                        key=lambda r: r["N"])
            if not rs:
                continue
            n = [r["N"] for r in rs]
            y = [r["gap_per_agent_step_mean"] for r in rs]
            lo = [r["gap_ci"][0] for r in rs]
            hi = [r["gap_ci"][1] for r in rs]
            ax.errorbar(n, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], color=C_COLOURS[c],
                        marker=C_MARKERS[c], ms=2.8, lw=1.0, capsize=1.5, elinewidth=0.6, label=f"c = {c:g}")
            plotted[f"{env}|{c}"] = {"N": n, "gap_mean": y, "ci_lo": lo, "ci_hi": hi}
        ax.set_yscale("symlog", linthresh=1e-6, linscale=0.4)
        ax.set_yticks([0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
        ax.set_ylim(-2e-7, 1e-2)
        ax.set_xticks(sorted({r["N"] for r in dp}))
        ax.set_title(title.split(" (")[0] + " (coupled)", fontsize=7.5)
        ax.set_xlabel("agents N", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, which="major", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("gap per agent-step\n(symlog; linear below $10^{-6}$)", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6, frameon=False, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, out, "fig_R7_dp_gap_vs_N", plotted)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rev001-dir", required=True)
    ap.add_argument("--a1-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=["regret", "appendix"], default=None,
                    help="draw only the main-text regret figure or only the appendix figures")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.only in (None, "regret"):
        regret_figure(a.rev001_dir, a.a1_dir, a.out)
    if a.only in (None, "appendix"):
        cd_figure(a.rev001_dir, a.out)
        sweep_figure(a.rev001_dir, a.out)
        dp_figure(a.rev001_dir, a.out)
    print("written", a.out)


if __name__ == "__main__":
    main()
