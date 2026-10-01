"""A1 analysis (docs/REVISION_002_ANALYSIS_PLAN.md, sections 2.3–2.5).

Reads A1 unit results and REV-001 common-world records; never runs a simulation.
Writes ``<out>/analysis/``: summary.json, CSV tables, the convergence-curve
figure and an auto-generated REPORT.md that lists results without interpretation.
"""
import csv
import hashlib
import io
import json
import os

import numpy as np

from . import config as C
from .a1_config import A1_METHODS, CONV_ABS, CONV_REL, fold_converged, method_name
from .a1_units import a1_uid, enumerate_a1_units, load_result
from .analysis import g_ratio
from .clustered import analyze_pair, generalization_flag, load_rev001_cew
from .decision_rules import pairwise_rules, protocol_family_holm
from .stathelp import aligned, describe_vec, family, trajectory_stats, verdict
from .units import _curve_at, load_archived


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def families_for(a1prof):
    fams = {}
    for b in a1prof.select_budgets:
        r, s, e = method_name("rova", b), method_name("static", b), method_name("ewma", b)
        fams[f"H1_{b}"] = [(r, s), (r, e), (s, e)]
    fin = {m: method_name(m, a1prof.b_final) for m in A1_METHODS}
    rep = {m: method_name(m, a1prof.b_repro) for m in A1_METHODS}
    fams["H2"] = [(fin[m], "A0") for m in A1_METHODS[::-1]]
    fams["H3"] = [(fin[m], ref) for m in A1_METHODS[::-1] for ref in ("learned", "qlearn")]
    fams["H4"] = [(fin[m], rep[m]) for m in A1_METHODS[::-1]]
    fams["H5"] = [(fin[m], m) for m in A1_METHODS[::-1]]
    return fams


A2_FAMILIES = ("H2", "H3", "H4")


def convergence(a1prof, out_dir):
    rows, cells = [], {}
    for env in C.ENV_NAMES:
        for c in a1prof.base.grid:
            cell_ok = True
            for m in A1_METHODS:
                res = load_result(out_dir, a1_uid(2, "a1_campaign", method=m, env=env, c=c))
                if res is None:
                    cell_ok = False
                    rows.append({"env": env, "c": c, "method": m, "status": "missing"})
                    continue
                m_ok = True
                for f in res["results"]:
                    s = np.asarray(f["inner_scores"], float)
                    B = {b: float(np.nanmin(s[:b])) for b in sorted(set(a1prof.curve_budgets) | set(a1prof.select_budgets))}
                    conv = fold_converged(B[a1prof.b_half], B[a1prof.b_final])
                    m_ok &= conv
                    sel = f["selections"]
                    rows.append({"env": env, "c": c, "method": m, "outer_fold": f["outer_fold"],
                                 **{f"B{b}": v for b, v in B.items()},
                                 "improvement_half_to_final": B[a1prof.b_half] - B[a1prof.b_final],
                                 "threshold": max(CONV_REL * B[a1prof.b_half], CONV_ABS),
                                 "fold_converged": conv,
                                 "selection_changed_half_to_final": sel[str(a1prof.b_half)]["best_index"]
                                 != sel[str(a1prof.b_final)]["best_index"],
                                 "selected_index_final": sel[str(a1prof.b_final)]["best_index"]})
                cells.setdefault(f"{env}|{c}", {})[m] = bool(m_ok)
                cell_ok &= m_ok
            cells.setdefault(f"{env}|{c}", {})["cell_converged"] = bool(cell_ok)
    return rows, cells


def curves(a1prof, out_dir):
    out = {}
    for env in C.ENV_NAMES:
        for c in a1prof.base.grid:
            for m in A1_METHODS:
                res = load_result(out_dir, a1_uid(2, "a1_campaign", method=m, env=env, c=c))
                if res is None:
                    continue
                per_fold = [np.minimum.accumulate(np.asarray(f["inner_scores"], float)).tolist() for f in res["results"]]
                rec = {"mean_best_so_far": np.mean(np.array(per_fold), axis=0).tolist(), "per_fold": per_fold}
                if a1prof.scientific:
                    arch = load_archived(env, c, m)
                    rec["random_search_archived"] = {str(b): float(np.mean([_curve_at(f, b) for f in arch]))
                                                     for b in C.SUB_BUDGETS_FULL}
                out[f"{m}|{env}|{c}"] = rec
    return out


def reproduction(a1prof, out_dir):
    gate = load_result(out_dir, a1_uid(1, "a1_gate")) or {}
    camp, ev = [], []
    plan = enumerate_a1_units(a1prof)
    n_camp = sum(u["kind"] == "a1_campaign" for u in plan)
    n_ev = sum(u["kind"] == "a1_eval" for u in plan)
    for u in plan:
        res = load_result(out_dir, u["uid"])
        if u["kind"] == "a1_campaign":
            camp.append({"uid": u["uid"], "ok": bool(res and res["reproduction"]["ok"])})
        elif u["kind"] == "a1_eval":
            ev.append({"uid": u["uid"], "ok": bool(res and res["reproduction"]["ok"])})
    return {"gate_passed": bool(gate.get("passed")), "r0_exact_zero": gate.get("r0_exact_zero"),
            "campaign_checks_ok": all(x["ok"] for x in camp) and len(camp) == n_camp,
            "eval_checks_ok": all(x["ok"] for x in ev) and len(ev) == n_ev,
            "n_campaign_units": n_camp, "n_eval_units": n_ev,
            "failed": [x["uid"] for x in camp + ev if not x["ok"]]}


def analyze_cells(a1prof, out_dir, rdir):
    fams_def = families_for(a1prof)
    cells, raw, a2_rows = {}, {}, []
    for env in C.ENV_NAMES:
        for c in a1prof.base.grid:
            key = f"{env}|{c}"
            res = load_result(out_dir, a1_uid(3, "a1_eval", env=env, c=c))
            if res is None:
                cells[key] = {"status": "missing unit"}
                continue
            recs = dict(load_rev001_cew(rdir, env, c))
            recs.update(res["records"])
            data, keys, bad = aligned(recs)
            if bad:
                cells[key] = {"status": "misaligned", "misaligned_methods": bad}
                continue
            shown = [method_name(m, b) for m in A1_METHODS for b in a1prof.select_budgets] + \
                    ["A0", "A5", "learned", "qlearn", "static", "ewma", "rova", "static_tpe", "ewma_tpe", "rova_tpe"]
            methods = {}
            for name in shown:
                if name in data:
                    methods[name] = describe_vec(data[name], a1prof.n_boot)
                    methods[name].update(trajectory_stats(recs[name]))
            fams = {f: family(p, data, a1prof.n_boot) for f, p in fams_def.items()}
            verdicts = {str(b): verdict(fams[f"H1_{b}"], f"_tpe{b}") for b in a1prof.select_budgets}
            fin = {m: method_name(m, a1prof.b_final) for m in A1_METHODS}
            g = g_ratio({"A0": data["A0"], "rova": data[fin["rova"]], "static": data[fin["static"]],
                         "ewma": data[fin["ewma"]]}, a1prof.n_boot)
            cells[key] = {"status": "ok", "n_worlds": len(keys), "methods": methods, "families": fams,
                          "verdicts_by_budget": verdicts, "G_final": g}
            raw[(env, c)] = {"data": data}
            folds = np.array([k[0][0] for k in keys])
            for fname in [f"H1_{a1prof.b_final}", *A2_FAMILIES]:
                for row in fams[fname]:
                    if "code" not in row:
                        continue
                    r = analyze_pair(data[row["a"]], data[row["b"]], folds, a1prof.n_boot)
                    a2_rows.append({"env": env, "c": c, "family": fname, "a": row["a"], "b": row["b"],
                                    "label": row["label"], "code": row["code"], **r,
                                    "generalization": generalization_flag(row["code"], r["two_stage_mean_diff_ci"])})
    return cells, raw, a2_rows


def _cell(v):
    return json.dumps(v) if isinstance(v, (list, dict)) else ("" if v is None else v)


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _cell(r.get(k)) for k in keys})
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(buf.getvalue())


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def make_figure(fdir, a1prof, curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    envs, grid = C.ENV_NAMES, list(a1prof.base.grid)
    fig, axes = plt.subplots(len(envs), len(grid), figsize=(3.2 * len(grid), 2.6 * len(envs)), squeeze=False)
    colors = {"static": "#0072B2", "ewma": "#E69F00", "rova": "#009E73"}
    for i, env in enumerate(envs):
        for j, c in enumerate(grid):
            ax = axes[i][j]
            for m in A1_METHODS:
                rec = curve.get(f"{m}|{env}|{c}")
                if not rec:
                    continue
                y = np.asarray(rec["mean_best_so_far"])
                ax.plot(np.arange(1, len(y) + 1), np.maximum(y, 1e-6), color=colors[m], lw=1.2, label=f"{m} (TPE)")
                rs = rec.get("random_search_archived")
                if rs:
                    ax.plot([int(b) for b in rs], [max(v, 1e-6) for v in rs.values()], "o--", color=colors[m],
                            ms=3, lw=0.7, alpha=0.8, label=f"{m} (random search, archived)")
            for b in (a1prof.b_repro, a1prof.b_half):
                ax.axvline(b, color="0.6", lw=0.6, ls=":")
            ax.set_yscale("log")
            ax.set_title(f"{env.split('_')[0]}, c={c}", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == len(envs) - 1:
                ax.set_xlabel("candidates", fontsize=7)
            if j == 0:
                ax.set_ylabel("best inner regret (mean of folds)", fontsize=7)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fdir, f"A1_convergence_curves.{ext}"), dpi=200)
    plt.close(fig)


def _fmt(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    if isinstance(x, list):
        return "[" + ", ".join(_fmt(v, nd) for v in x) + "]"
    return str(x)


def write_report(path, S, a1prof):
    L = ["# A1 tuning-convergence report", "",
         "Auto-generated from unit results; lists results without interpretation. "
         "Plan: docs/REVISION_002_ANALYSIS_PLAN.md (specified after the REV-001 results).", "",
         f"Convergence criterion: fold converged iff B({a1prof.b_half}) - B({a1prof.b_final}) <= "
         f"max({CONV_REL} * B({a1prof.b_half}), {CONV_ABS}).", "", "## Reproduction", ""]
    L += [f"- {k}: {v}" for k, v in S["reproduction"].items()]
    L += ["", "## Verdicts by budget and convergence status", "",
          "| env | c | " + " | ".join(f"verdict@{b}" for b in a1prof.select_budgets)
          + " | static conv | ewma conv | rova conv | cell converged | G@final | G CI |", "|---" * (len(a1prof.select_budgets) + 8) + "|"]
    for key, cell in S["cells"].items():
        cv = S["convergence_cells"].get(key, {})
        if cell.get("status") != "ok":
            L.append(f"| {key} | {cell.get('status')} |")
            continue
        env, c = key.split("|")
        g = cell["G_final"]
        L.append(f"| {env} | {c} | " + " | ".join(_fmt(cell["verdicts_by_budget"].get(str(b))) for b in a1prof.select_budgets)
                 + f" | {cv.get('static')} | {cv.get('ewma')} | {cv.get('rova')} | {cv.get('cell_converged')} | "
                 f"{_fmt(g.get('G'))} | {_fmt(g.get('G_ci'))} |")
    L += ["", "## Pairwise comparisons (within-family Holm; P = A1-family Holm)", "",
          "| env | c | family | a | b | δ | p | mean diff [BCa] | label | P |", "|---" * 10 + "|"]
    for key, cell in S["cells"].items():
        if cell.get("status") != "ok":
            continue
        env, c = key.split("|")
        for fname, rows in cell["families"].items():
            for r in rows:
                if "wilcoxon_p" not in r:
                    continue
                L.append(f"| {env} | {c} | {fname} | {r['a']} | {r['b']} | {r['cliffs_delta']:+.3f} | {r['wilcoxon_p']:.2g} | "
                         f"{_fmt(r['mean_diff'])} {_fmt(r['mean_diff_ci'])} | {r['label']} | {r.get('protocol_family_reject')} |")
    L += ["", "## Decision rules at the final budget (A1-family significance)", ""]
    for k, v in S["decision_rules"].items():
        L.append(f"- {k}: {v.get('DR2')}; DR8 {v.get('DR8_no_practically_significant_difference')}; "
                 + "; ".join(f"c={c}: {pc['DR1']}" for c, pc in v.get("per_cost_ratio", {}).items())
                 + ("; crossovers " + json.dumps({e: [x['crossing_c'] for x in xs] for e, xs in v.get('DR2_crossovers', {}).items()})
                    if v.get("DR2_crossovers") else ""))
    L += ["", "## A2 statistics on A1 comparisons", "",
          "| env | c | family | pair | label | two-stage CI (d) | sign-flip p | n_eff | generalization |", "|---" * 9 + "|"]
    for r in S["a2_on_a1"]:
        L.append(f"| {r['env']} | {r['c']} | {r['family']} | {r['a']}–{r['b']} | {r['label']} | "
                 f"{_fmt(r['two_stage_mean_diff_ci'])} | {_fmt(r['sign_flip_p'], 3)} | {_fmt(r['effective_n'], 3)} | {r['generalization']} |")
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def run_a1_analysis(a1prof, out_dir, rdir):
    adir = os.path.join(out_dir, "analysis")
    tdir, fdir = os.path.join(adir, "tables"), os.path.join(adir, "figures")
    for d in (tdir, fdir):
        os.makedirs(d, exist_ok=True)
    errors = []
    S = {"profile": a1prof.name, "scientific": a1prof.scientific,
         "constants": {"CONV_REL": CONV_REL, "CONV_ABS": CONV_ABS, "n_total": a1prof.n_total,
                       "select_budgets": list(a1prof.select_budgets), "curve_budgets": list(a1prof.curve_budgets),
                       "frozen_startup_draw_per_fold": a1prof.base.n_trials, "tpe_startup": a1prof.base.tpe_startup}}
    S["reproduction"] = reproduction(a1prof, out_dir)
    conv_rows, conv_cells = convergence(a1prof, out_dir)
    S["convergence_cells"] = conv_cells
    S["convergence_counts"] = {
        "cells_converged": sum(v.get("cell_converged", False) for v in conv_cells.values()),
        "method_cells_converged": {m: sum(v.get(m, False) for v in conv_cells.values()) for m in A1_METHODS}}
    cells, raw, a2_rows = analyze_cells(a1prof, out_dir, rdir)
    S["a1_family"] = protocol_family_holm([cells])
    S["cells"] = cells
    fin = {m: method_name(m, a1prof.b_final) for m in A1_METHODS}
    pairs = [(fin["rova"], fin["static"]), (fin["rova"], fin["ewma"]), (fin["rova"], "A0")]
    try:
        S["decision_rules"] = pairwise_rules(cells, raw, pairs, list(a1prof.base.grid), C.ENV_NAMES, a1prof.n_boot)
    except Exception as e:  # recorded, never silently dropped
        S["decision_rules"] = {}
        errors.append({"section": "decision_rules", "error": repr(e)})
    S["a2_on_a1"] = a2_rows
    curve = curves(a1prof, out_dir)
    S["curves"] = curve
    S["section_errors"] = errors
    try:
        make_figure(fdir, a1prof, curve)
    except Exception as e:
        errors.append({"section": "figure", "error": repr(e)})

    _write_csv(os.path.join(tdir, "A1_convergence.csv"), conv_rows)
    vrows, frows, mrows = [], [], []
    for key, cell in cells.items():
        env, c = key.split("|")
        if cell.get("status") != "ok":
            vrows.append({"env": env, "c": c, "status": cell.get("status")})
            continue
        cv = conv_cells.get(key, {})
        vrows.append({"env": env, "c": c, **{f"verdict_{b}": v for b, v in cell["verdicts_by_budget"].items()},
                      "cell_converged": cv.get("cell_converged"), **{f"{m}_converged": cv.get(m) for m in A1_METHODS},
                      "G_final": cell["G_final"].get("G"), "G_final_ci": cell["G_final"].get("G_ci"),
                      "G_status": cell["G_final"].get("status")})
        for fname, rows in cell["families"].items():
            for r in rows:
                frows.append({"env": env, "c": c, "family": fname,
                              **{k: r.get(k) for k in ("a", "b", "n", "wilcoxon_p", "holm_threshold", "holm_reject",
                                                       "protocol_family_reject", "cliffs_delta", "mean_diff", "mean_diff_ci",
                                                       "cvar95_diff", "cvar95_diff_ci_percentile", "label", "missing")}})
        for name, d in cell["methods"].items():
            mrows.append({"env": env, "c": c, "method": name, **{k: d.get(k) for k in (
                "n", "mean", "mean_ci", "median", "median_ci", "cvar95_pooled", "oracle_full_supervision_fraction",
                "action_fractions_levels_0_to_5")}})
    _write_csv(os.path.join(tdir, "A1_verdicts.csv"), vrows)
    _write_csv(os.path.join(tdir, "A1_families.csv"), frows)
    _write_csv(os.path.join(tdir, "A1_methods.csv"), mrows)
    _write_csv(os.path.join(tdir, "A1_A2_clustered.csv"), a2_rows)

    S_json = json.loads(json.dumps(S, default=_jsonable))
    spath = os.path.join(adir, "summary.json")
    with open(spath, "w", newline="\n", encoding="utf-8") as f:
        json.dump(S_json, f, indent=1, default=_jsonable)
    try:
        write_report(os.path.join(adir, "REPORT.md"), S_json, a1prof)
    except Exception as e:
        errors.append({"section": "report", "error": repr(e)})
    return {"summary_path": "analysis/summary.json", "summary_sha256": _sha256(spath), "section_errors": errors,
            "reproduction": S["reproduction"], "convergence_counts": S["convergence_counts"]}
