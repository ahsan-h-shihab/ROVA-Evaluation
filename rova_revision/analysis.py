"""REV-001 statistical analysis (plan sections 4 and 5).

Reads unit results only and never runs a simulation. Writes
``<out>/analysis/summary.json``, CSV tables, figures and an auto-generated
``REPORT.md`` that lists results without interpretation.

Standalone (e.g. after copying a finished run elsewhere):
    python -m rova_revision.analysis --profile full --out revision_out
"""
import argparse
import csv
import hashlib
import json
import os

import numpy as np
from scipy import stats as sps

from dtcoa_val.analysis.stats import (bca_bootstrap_ci, bootstrap_ci, cliffs_delta, holm_bonferroni,
                                      wilcoxon_paired)
from dtcoa_val.experiments.nested_hpo import _cvar
from dtcoa_val.experiments.search_space import sample_config

from . import config as C
from .archive import REPO, inner_folds, load_archived, load_stats_summary, outer_folds
from .training import LEARNED_SPACE, QLEARN_SPACE
from .units import CLEAN_METHODS, ROBUSTNESS_METHODS, _curve_at, enumerate_units, load_result, uid

NEGLIGIBLE = 0.147
REGIMES = {"low": (0.5, 1.5), "transition": (3.5,), "high": (8.0, 20.0)}
FAMILIES = {
    "F1": [("rova", "static"), ("rova", "ewma"), ("static", "ewma")],
    "F2": [(m, a) for m in ("rova", "static", "ewma") for a in ("A0", "A5")],
    "F3": [(m, l) for m in ("rova", "static", "ewma") for l in ("learned", "qlearn")],
    "F4": [("rova", "static_ext"), ("rova", "ewma_ext"), ("static_ext", "static"), ("ewma_ext", "ewma")],
    "F5": [("rova", "clairvoyant"), ("rova", "rova_single")],
    "F6": [("rova_tpe", "static_tpe"), ("rova_tpe", "ewma_tpe"), ("static_tpe", "ewma_tpe")],
    "F7": [("rova_tpe", "rova"), ("static_tpe", "static"), ("ewma_tpe", "ewma")],
}
# Plan R7 decision rule: labels among static, ewma, rova, A0, A5 and learned.
DECISION_FAMILIES = {
    "F1": FAMILIES["F1"],
    "F2": FAMILIES["F2"],
    "F3_learned": [(m, "learned") for m in ("rova", "static", "ewma")],
}
FIG_METHODS = ["A0", "A5", "static", "ewma", "rova", "learned", "qlearn", "clairvoyant"]


# --------------------------------------------------------------------------- #
# Statistics helpers (frozen functions only)
# --------------------------------------------------------------------------- #
def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def describe_vec(x, n_boot):
    x = np.asarray(x, float)
    m, lo, hi = bca_bootstrap_ci(x, n_boot=n_boot)
    md, mlo, mhi = bca_bootstrap_ci(x, n_boot=n_boot, statistic=np.median)
    cv, clo, chi = bootstrap_ci(x, n_boot=n_boot, statistic=lambda v: _cvar(v, 0.95))
    return {"n": int(len(x)), "mean": m, "mean_ci": [lo, hi], "median": md, "median_ci": [mlo, mhi],
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "cvar95_pooled": cv, "cvar95_pooled_ci_percentile": [clo, chi],
            "cvar95_tail_instances": int(len(x) - int(np.ceil(0.95 * len(x)))),
            "cvar95_note": "low precision: percentile bootstrap over the pooled per-seed tail"}


def compare(xa, xb, n_boot):
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    stat, p = wilcoxon_paired(xa, xb)
    diff = xa - xb
    m, lo, hi = bca_bootstrap_ci(diff, n_boot=n_boot)
    return {"n": int(len(xa)), "wilcoxon_stat": _f(stat), "wilcoxon_p": float(p),
            "cliffs_delta": cliffs_delta(xa, xb), "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
            "mean_diff": m, "mean_diff_ci": [lo, hi],
            "median_diff": float(np.median(xa) - np.median(xb)), "median_of_paired_diff": float(np.median(diff))}


def rule(a, b, reject, delta):
    """Plan section 1 comparison rule."""
    if reject and abs(delta) > NEGLIGIBLE:
        return ("a_lower", f"{a} lower-regret") if delta < 0 else ("b_lower", f"{b} lower-regret")
    return ("none", "no detected difference")


def family(pairs, data, n_boot):
    rows = []
    for a, b in pairs:
        if a in data and b in data:
            rows.append({"a": a, "b": b, **compare(data[a], data[b], n_boot)})
        else:
            rows.append({"a": a, "b": b, "missing": [m for m in (a, b) if m not in data]})
    avail = [r for r in rows if "missing" not in r]
    if avail:
        for r, (_i, _p, thr, rej) in zip(avail, holm_bonferroni([r["wilcoxon_p"] for r in avail])):
            r["holm_threshold"], r["holm_reject"] = thr, bool(rej)
            r["code"], r["label"] = rule(r["a"], r["b"], rej, r["cliffs_delta"])
            r["family_size"] = len(avail)
    return rows


def verdict(rows, sfx=""):
    """Cell verdict with the manuscript's symmetric rule (ROVA versus both baselines)."""
    by = {(r["a"], r["b"]): r for r in rows}
    sides = [by.get((f"rova{sfx}", f"static{sfx}")), by.get((f"rova{sfx}", f"ewma{sfx}"))]
    if any(s is None or "code" not in s for s in sides):
        return None
    codes = {s["code"] for s in sides}
    if codes == {"a_lower"}:
        return "ROVA_wins"
    if codes == {"b_lower"}:
        return "ROVA_loses"
    return "mixed"


def _key(r):
    return (int(r["outer_fold"]), int(r["seed"]))


def aligned(recs_by_method, key="regret"):
    """Per-method vectors in (outer fold, seed) order; methods whose worlds
    differ from the first method's are returned as misaligned, not used."""
    ref, data, bad = None, {}, []
    for m, recs in recs_by_method.items():
        rs = sorted(recs, key=_key)
        ks = [(_key(r), r.get("tag")) for r in rs]
        if ref is None:
            ref = ks
        if ks != ref:
            bad.append(m)
            continue
        data[m] = np.array([r[key] for r in rs], float)
    return data, ref or [], bad


def trajectory_stats(recs):
    n = sum(int(r["n_agent_steps"]) for r in recs)
    ast = np.sum([r["astar_counts"] for r in recs], axis=0)
    act = np.sum([r["a_counts"] for r in recs], axis=0)
    mean = lambda k: float(np.mean([r[k] for r in recs]))
    return {"agent_steps": n,
            "oracle_full_supervision_fraction": float(ast[0] / n) if n else None,
            "oracle_level5_fraction": float(ast[5] / n) if n else None,
            "action_fractions_levels_0_to_5": (act / n).tolist() if n else None,
            "harm_rate_mean": mean("harm_rate"), "unauth_unsafe_rate_mean": mean("unauth_unsafe_rate"),
            "overrestrict_mean": mean("overrestrict"), "sysfail_rate_mean": mean("sysfail_rate"),
            "esc_rate_mean": mean("esc_rate")}


def variance_decomposition(x, groups):
    """One-way random-effects decomposition over outer folds; ICC(1)."""
    x, g = np.asarray(x, float), np.asarray(groups)
    levels = np.unique(g)
    k, n = len(levels), len(x)
    if k < 2 or n <= k:
        return {"status": "not estimable"}
    sizes = np.array([np.sum(g == l) for l in levels])
    means = np.array([x[g == l].mean() for l in levels])
    ssb = float(np.sum(sizes * (means - x.mean()) ** 2))
    ssw = float(sum(np.sum((x[g == l] - m) ** 2) for l, m in zip(levels, means)))
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - np.sum(sizes ** 2) / n) / (k - 1)
    var_b = max(0.0, (msb - msw) / n0)
    tot = var_b + msw
    F = msb / msw if msw > 0 else np.nan
    return {"groups": int(k), "n": int(n), "ms_between": msb, "ms_within": msw,
            "var_between": var_b, "var_within": msw, "icc1": var_b / tot if tot > 0 else None,
            "F": _f(F), "p": _f(sps.f.sf(F, k - 1, n - k)) if np.isfinite(F) else None}


def curve_summary(folds, budgets):
    per_fold = [{str(b): _curve_at(f, b) for b in budgets} for f in folds]
    mean = {}
    for b in budgets:
        v = [pf[str(b)] for pf in per_fold if pf[str(b)] is not None]
        mean[str(b)] = float(np.mean(v)) if v else None
    return {"per_fold": per_fold, "mean_over_folds": mean}


def g_ratio(data, n_boot):
    """Plan R1 surviving-margin ratio with a paired BCa interval over worlds."""
    need = ("A0", "rova", "static", "ewma")
    if any(m not in data for m in need):
        return {"status": "missing methods", "missing": [m for m in need if m not in data]}
    best = "static" if data["static"].mean() <= data["ewma"].mean() else "ewma"
    a0, rv, bb = data["A0"], data["rova"], data[best]
    num, den = float(a0.mean() - rv.mean()), float(bb.mean() - rv.mean())
    out = {"best_baseline": best, "mean_A0": float(a0.mean()), "mean_rova": float(rv.mean()),
           "mean_best_baseline": float(bb.mean()), "numerator": num, "denominator": den}
    if den <= 0:
        out["status"] = "not defined: best tuned baseline mean regret <= ROVA mean regret"
        return out
    n = len(rv)
    idx = np.arange(n, dtype=float)

    def stat(ix):
        ix = ix.astype(int)
        d = bb[ix].mean() - rv[ix].mean()
        return (a0[ix].mean() - rv[ix].mean()) / d if d > 0 else np.nan

    rng = np.random.default_rng(0)  # the resamples bca_bootstrap_ci draws
    nonpos = sum(1 for _ in range(n_boot) if (lambda ix: bb[ix].mean() - rv[ix].mean())(
        rng.choice(idx, size=n, replace=True).astype(int)) <= 0)
    g = num / den
    out.update({"status": "defined", "G": g, "resamples_with_nonpositive_denominator": int(nonpos)})
    dm, dlo, dhi = bca_bootstrap_ci(bb - rv, n_boot=n_boot)
    out["denominator_ci"] = [dlo, dhi]
    if nonpos == 0:
        _g, lo, hi = bca_bootstrap_ci(idx, n_boot=n_boot, statistic=stat)
        out["G_ci"] = [lo, hi]
    else:
        out["G_ci"] = None
        out["G_ci_note"] = "not reported: the denominator is non-positive in some bootstrap resamples"
    if g >= 1:
        out["interpretation"] = "G >= 1: the anchor is no better than the best tuned baseline"
    elif g > 0:
        out["interpretation"] = f"0 < G < 1: the anchor closes a fraction {1 - g:.4f} of the baseline-to-ROVA gap"
    else:
        out["interpretation"] = "G <= 0: the anchor is at least as good as ROVA"
    return out


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _published_verdicts():
    with open(os.path.join(REPO, "verdicts", "verdicts_v2.json")) as f:
        cells = json.load(f)["cells"]
    return {(r["environment"], float(r["cost_ratio"])): r["v2_verdict"] for r in cells}


def archived_reanalysis(cells, n_boot):
    """R2 (a), (b), (c), (e) and S1 on the archived MAIN-001 per-seed results."""
    summary = load_stats_summary()
    published = _published_verdicts()
    out, s1_rows = {}, []
    for env, c in cells:
        arch = {m: load_archived(env, c, m) for m in C.ARCHIVED_METHODS}
        reg = {m: np.array([x for f in arch[m] for x in f["test_regrets"]], float) for m in arch}
        grp = {m: np.array([f["outer_fold"] for f in arch[m] for _ in f["test_regrets"]]) for m in arch}
        cs = summary["cells"][env][str(c)]
        methods = {}
        for m in arch:
            d = describe_vec(reg[m], n_boot)
            ref = cs["regret_desc"][m]
            d["mean_ci_matches_stats_summary"] = (bool(d["mean"] == ref["mean"] and
                                                       d["mean_ci"] == [ref["ci_lo"], ref["ci_hi"]])
                                                  if n_boot == 5000 else None)
            d["variance_decomposition"] = variance_decomposition(reg[m], grp[m])
            d["tuning_curve"] = curve_summary(arch[m], C.SUB_BUDGETS_FULL)
            methods[m] = d
        pub = []
        for a, b in FAMILIES["F1"]:
            rec = next(x for x in cs["primary_pairwise"] if x["pair"] == f"{a}_vs_{b}")
            code, lab = rule(a, b, rec["holm_reject"], rec["cliffs_delta"])
            row = {"a": a, "b": b, "wilcoxon_p": rec["wilcoxon_p"], "holm_reject": bool(rec["holm_reject"]),
                   "cliffs_delta": rec["cliffs_delta"], "code": code, "label": lab,
                   "recomputed_p_matches": bool(wilcoxon_paired(reg[a], reg[b])[1] == rec["wilcoxon_p"])}
            pub.append(row)
            s1_rows.append({"env": env, "c": c, **row})
        mw = []
        for a, b in FAMILIES["F1"]:
            u, p = sps.mannwhitneyu(reg[a], reg[b], alternative="two-sided", method="asymptotic",
                                    use_continuity=True)
            mw.append({"a": a, "b": b, "U": float(u), "p": float(p), "cliffs_delta": cliffs_delta(reg[a], reg[b])})
        for r, (_i, _p, thr, rej) in zip(mw, holm_bonferroni([r["p"] for r in mw])):
            r["holm_threshold"], r["holm_reject"] = thr, bool(rej)
            r["code"], r["label"] = rule(r["a"], r["b"], rej, r["cliffs_delta"])
        out[f"{env}|{c}"] = {"env": env, "c": c, "methods": methods, "published_pairs": pub,
                             "published_verdict_v2": published.get((env, float(c))),
                             "published_pairs_verdict_recomputed": verdict(pub),
                             "mann_whitney": mw, "mann_whitney_verdict": verdict(mw)}
    changes = []
    for r, (_i, _p, thr, rej) in zip(s1_rows, holm_bonferroni([r["wilcoxon_p"] for r in s1_rows])):
        code, lab = rule(r["a"], r["b"], rej, r["cliffs_delta"])
        r.update({"S1_holm_threshold": thr, "S1_reject": bool(rej), "S1_code": code, "S1_label": lab})
        if code != r["code"]:
            changes.append({"env": r["env"], "c": r["c"], "pair": f"{r['a']}_vs_{r['b']}",
                            "published": r["label"], "S1": lab})
    s1_verdicts = {}
    for key, cell in out.items():
        rows = [{"a": r["a"], "b": r["b"], "code": r["S1_code"]} for r in s1_rows
                if f"{r['env']}|{r['c']}" == key]
        s1_verdicts[key] = verdict(rows)
        cell["S1_verdict"] = s1_verdicts[key]
    return out, {"n_tests": len(s1_rows), "label_changes": changes, "verdicts": s1_verdicts}


def cew_cell(out_dir, env, c, excluded, n_boot):
    res = load_result(out_dir, uid(3, "eval_cew", env=env, c=c))
    if res is None:
        return {"status": "missing unit"}, None
    recs = res["records"]
    data, keys, bad = aligned(recs)
    methods = {}
    for m in recs:
        if m in data:
            methods[m] = {**describe_vec(data[m], n_boot), **trajectory_stats(recs[m]),
                          "excluded_from_comparisons": m in excluded}
    cmp_data = {m: v for m, v in data.items() if m not in excluded}
    fams = {f: family(p, cmp_data, n_boot) for f, p in FAMILIES.items()}
    return ({"status": "ok", "n_worlds": len(keys), "misaligned_methods": bad, "missing_methods": res["missing"],
             "methods": methods, "families": fams, "verdict": verdict(fams["F1"]),
             "verdict_tpe": verdict(fams["F6"], "_tpe"), "G": g_ratio(data, n_boot)},
            {"records": recs, "data": data})


def anchor_cell(profile, out_dir, env, c, n_boot):
    res = load_result(out_dir, uid(2, "anchor_eval", env=env, c=c))
    if res is None:
        return {"status": "missing unit"}
    out = {"status": "ok", "own_worlds": {}}
    for A in ("A0", "A5"):
        recs = res[A]["own"]
        out["own_worlds"][A] = {**describe_vec([r["regret"] for r in recs], n_boot), **trajectory_stats(recs)}
    if not profile.scientific:
        out["archived_worlds"] = "not applicable (non-scientific profile)"
        return out
    folds = outer_folds(profile.seeds, profile.K_out)
    rows, per_method = [], {}
    for m in C.ARCHIVED_METHODS:
        arch = load_archived(env, c, m)
        by_seed = {s: arch[k]["test_regrets"][i] for k in range(profile.K_out) for i, s in enumerate(folds[k][1])}
        data = {}
        for A in ("A0", "A5"):
            recs = sorted(res[A][f"on_{m}_worlds"], key=_key)
            data[A] = np.array([r["regret"] for r in recs], float)
            data[m] = np.array([by_seed[int(r["seed"])] for r in recs], float)
            per_method[f"{A}_on_{m}_worlds"] = {"mean": float(data[A].mean()),
                                                "oracle_full_supervision_fraction":
                                                    trajectory_stats(recs)["oracle_full_supervision_fraction"]}
        for A in ("A0", "A5"):
            rows.append({"a": m, "b": A, "worlds": f"archived {m} test worlds", **compare(data[m], data[A], n_boot)})
    for r, (_i, _p, thr, rej) in zip(rows, holm_bonferroni([r["wilcoxon_p"] for r in rows])):
        r["holm_threshold"], r["holm_reject"] = thr, bool(rej)
        r["code"], r["label"] = rule(r["a"], r["b"], rej, r["cliffs_delta"])
        r["family_size"] = len(rows)
    out["archived_worlds"] = {"family": rows, "anchor_summaries": per_method}
    return out


def trainable_cells(profile, out_dir, n_boot):
    out, gate_cells, complete, ok = {}, [], True, True
    for env in C.ENV_NAMES:
        for c in profile.grid:
            for m in ("learned", "qlearn"):
                res = load_result(out_dir, uid(2, "campaign_trainable", method=m, env=env, c=c))
                key = f"{m}|{env}|{c}"
                if res is None:
                    out[key] = {"status": "missing unit"}
                    if m == "qlearn":
                        complete = False
                    continue
                folds = res["results"]
                pooled = [x for f in folds for x in f["test_regrets"]]
                rec = {"status": "ok", "own_worlds": describe_vec(pooled, n_boot),
                       "best_configs": [f["best_config"] for f in folds],
                       "tuning_curve": curve_summary(folds, profile.sub_budgets)}
                if m == "learned":
                    diags = [d for f in folds for d in f["model"].get("fit_diagnostics", [])]
                    rec["fits"] = {"n": len(diags), "converged": sum(bool(d.get("converged")) for d in diags),
                                   "degenerate": sum(bool(d.get("degenerate")) for d in diags)}
                else:
                    g1 = all(f["gate"]["G1_finite"] for f in folds)
                    g2 = all(f["gate"]["G2_bounded"] for f in folds)
                    g3 = bool(folds[0]["gate"].get("G3_deterministic"))
                    sens = []
                    for f in folds:
                        vals = [f["test_mean_regret"]] + [s["test_mean_regret"] for s in f["training_seed_sensitivity"]]
                        sens.append({"outer_fold": f["outer_fold"], "primary": vals[0], "alternatives": vals[1:],
                                     "range": float(max(vals) - min(vals)), "sd": float(np.std(vals, ddof=1))})
                    rec.update({"G1_finite": g1, "G2_bounded": g2, "G3_deterministic_fold0": g3,
                                "training_seed_sensitivity": sens,
                                "q_ranges": [[f["gate"]["q_min"], f["gate"]["q_max"]] for f in folds]})
                    gate_cells.append({"env": env, "c": c, "G1": g1, "G2": g2, "G3": g3})
                    ok = ok and g1 and g2 and g3
                out[key] = rec
    status = "passed" if (ok and complete) else ("failed" if complete else "incomplete")
    return out, {"status": status, "cells": gate_cells,
                 "consequence": None if status == "passed" else
                 "qlearn excluded from comparative tables (plan R4); outputs archived"}


def adversarial_cells(profile, out_dir, excluded, n_boot):
    r5a, r5b = {}, {}
    for adv in C.ADVERSARY_PROFILES:
        for env in C.ENV_NAMES:
            if not C.adversary_valid(adv, env, profile):
                continue
            for c in profile.grid:
                key = f"{adv}|{env}|{c}"
                res = load_result(out_dir, uid(3, "eval_adv", adv=adv, env=env, c=c))
                if res is None:
                    r5a[key] = {"status": "missing unit"}
                else:
                    att, ka, bad_a = aligned(res["attacked"])
                    cln, kc, bad_c = aligned(res["clean"])
                    methods = {}
                    for m in att:
                        rec = {"attacked": {**describe_vec(att[m], n_boot), **trajectory_stats(res["attacked"][m])},
                               "excluded_from_comparisons": m in excluded}
                        if m in cln:
                            rec["clean_same_worlds"] = {**describe_vec(cln[m], n_boot),
                                                        **trajectory_stats(res["clean"][m])}
                            dm, dlo, dhi = bca_bootstrap_ci(att[m] - cln[m], n_boot=n_boot)
                            rec["degradation_attacked_minus_clean"] = {"mean": dm, "ci": [dlo, dhi]}
                        methods[m] = rec
                    cmp_a = {m: v for m, v in att.items() if m not in excluded}
                    cmp_c = {m: v for m, v in cln.items() if m not in excluded}
                    fams = {f: family(FAMILIES[f], cmp_a, n_boot) for f in ("F1", "F2", "F3", "F4", "F5")}
                    r5a[key] = {"status": "ok", "adversary": adv, "env": env, "c": c, "n_worlds": len(ka),
                                "keys_match_clean": ka == kc, "misaligned": bad_a + bad_c,
                                "missing_methods": res["missing"], "methods": methods, "families_attacked": fams,
                                "verdict_attacked": verdict(fams["F1"]),
                                "verdict_clean_same_worlds": verdict(family(FAMILIES["F1"], cmp_c, n_boot))}
                res = load_result(out_dir, uid(3, "eval_advt_cew", adv=adv, env=env, c=c))
                if res is None:
                    r5b[key] = {"status": "missing unit"}
                else:
                    data, keys, bad = aligned(res["records"])
                    fams = {f: family(FAMILIES[f], data, n_boot) for f in ("F1", "F2")}
                    r5b[key] = {"status": "ok", "adversary": adv, "env": env, "c": c, "n_worlds": len(keys),
                                "misaligned": bad, "missing_methods": res["missing"],
                                "methods": {m: {**describe_vec(v, n_boot), **trajectory_stats(res["records"][m])}
                                            for m, v in data.items()},
                                "families": fams, "verdict": verdict(fams["F1"])}
    return r5a, r5b


def sensitivity_cells(profile, out_dir, excluded, clean_verdicts, n_boot):
    r9a, r9b = {}, {}
    for pert in C.PERTURBATIONS:
        for env in C.ENV_NAMES:
            for c in profile.grid:
                key = f"{pert}|{env}|{c}"
                res = load_result(out_dir, uid(3, "eval_sens_cew", pert=pert, env=env, c=c))
                if res is None:
                    r9a[key] = r9b[key] = {"status": "missing unit"}
                    continue
                data, keys, bad = aligned(res["records"])
                retuned = {m: v for m, v in data.items() if not m.startswith("clean:")}
                clean = {m.split(":", 1)[1]: v for m, v in data.items() if m.startswith("clean:")}
                clean.update({a: data[a] for a in ("A0", "A5") if a in data})
                clean_cmp = {m: v for m, v in clean.items() if m not in excluded}
                fa = {f: family(FAMILIES[f], retuned, n_boot) for f in ("F1", "F2")}
                fb = {f: family(FAMILIES[f], clean_cmp, n_boot) for f in ("F1", "F2", "F3", "F4", "F5")}
                common = {"perturbation": pert, "env": env, "c": c, "n_worlds": len(keys), "misaligned": bad,
                          "missing_methods": res["missing"]}
                r9a[key] = {"status": "ok", **common,
                            "methods": {m: {**describe_vec(v, n_boot), **trajectory_stats(res["records"][m])}
                                        for m, v in retuned.items()},
                            "families": fa, "verdict": verdict(fa["F1"]),
                            "unperturbed_cew_verdict": clean_verdicts.get((env, c))}
                r9b[key] = {"status": "ok", **common,
                            "methods": {m: {**describe_vec(v, n_boot),
                                            "excluded_from_comparisons": m in excluded} for m, v in clean.items()},
                            "families": fb, "verdict": verdict(fb["F1"]),
                            "unperturbed_cew_verdict": clean_verdicts.get((env, c))}
    return r9a, r9b


def dp_reduced(profile, out_dir, n_boot):
    rows = []
    for env, sizes in profile.dp_sizes.items():
        for n in sizes:
            for c in profile.grid:
                res = load_result(out_dir, uid(2, "dp_reduced", env=env, n=n, c=c))
                if res is None:
                    rows.append({"env": env, "N": n, "c": c, "status": "missing unit"})
                    continue
                recs = res["records"]
                g = np.array([r["gap_per_agent_step"] for r in recs], float)
                dis = np.array([r["disagreement_rate"] for r in recs], float)
                vo = np.array([r["V_opt"] for r in recs], float)
                vm = np.array([r["V_myopic"] for r in recs], float)
                m, lo, hi = bca_bootstrap_ci(g, n_boot=n_boot)
                pos = vo > 0
                rows.append({"env": env, "N": n, "c": c, "status": "ok", "n_worlds": len(recs),
                             "T": recs[0]["T"], "tau": recs[0]["tau"],
                             "gap_per_agent_step_mean": m, "gap_ci": [lo, hi], "gap_min": float(g.min()),
                             "gap_max": float(g.max()), "fraction_worlds_gap_gt_1e-12": float(np.mean(g > 1e-12)),
                             "relative_gap_mean": float(np.mean((vm[pos] - vo[pos]) / vo[pos])) if pos.any() else None,
                             "disagreement_rate_mean": float(dis.mean()), "disagreement_rate_max": float(dis.max())})
    return rows


def dp_sequential(profile, out_dir, raw_cew, excluded, n_boot):
    env = C.DP_FULL_SIZE_ENV
    cells, changes = {}, []
    for c in profile.grid:
        dp = load_result(out_dir, uid(2, "dp_cew", c=c))
        raw = raw_cew.get((env, c))
        if dp is None or raw is None:
            cells[str(c)] = {"status": "missing unit"}
            continue
        V = {(r["tag"], int(r["seed"])): r for r in dp["records"]}
        seq, myo, problems = {}, {}, []
        oracle_gap = None
        for m, recs in raw["records"].items():
            rs = sorted(recs, key=_key)
            vals = []
            for r in rs:
                d = V.get((r["tag"], int(r["seed"])))
                if d is None:
                    problems.append(f"{m}: no DP world for {r['tag']} seed {r['seed']}")
                    break
                if int(r["n_agent_steps"]) != d["T"] * d["N"]:
                    problems.append(f"{m}: agent-step count mismatch at seed {r['seed']}")
                vals.append((r["sum_loss"] - d["V_opt"]) / r["n_agent_steps"])
            if len(vals) == len(rs):
                seq[m] = np.array(vals, float)
                myo[m] = np.array([r["regret"] for r in rs], float)
                if oracle_gap is None:
                    oracle_gap = np.array([(V[(r["tag"], int(r["seed"]))]["V_myopic"] -
                                            V[(r["tag"], int(r["seed"]))]["V_opt"]) /
                                           (V[(r["tag"], int(r["seed"]))]["T"] * V[(r["tag"], int(r["seed"]))]["N"])
                                           for r in rs], float)
        methods = {m: {"sequential_regret": describe_vec(seq[m], n_boot),
                       "myopic_regret_mean": float(myo[m].mean()),
                       "fraction_worlds_negative_sequential_regret": float(np.mean(seq[m] < 0)),
                       "excluded_from_comparisons": m in excluded} for m in seq}
        seq_c = {m: v for m, v in seq.items() if m not in excluded}
        myo_c = {m: v for m, v in myo.items() if m not in excluded}
        dec_seq = {f: family(p, seq_c, n_boot) for f, p in DECISION_FAMILIES.items()}
        dec_myo = {f: family(p, myo_c, n_boot) for f, p in DECISION_FAMILIES.items()}
        cell_changes = []
        for f in DECISION_FAMILIES:
            for rs_, rm_ in zip(dec_seq[f], dec_myo[f]):
                if "code" in rs_ and "code" in rm_ and rs_["code"] != rm_["code"]:
                    cell_changes.append({"c": c, "family": f, "pair": f"{rs_['a']}_vs_{rs_['b']}",
                                         "myopic_reference": rm_["label"], "sequential_reference": rs_["label"]})
        changes += cell_changes
        full = {f: family(FAMILIES[f], seq_c, n_boot) for f in ("F1", "F2", "F3", "F4", "F5")}
        cells[str(c)] = {"status": "ok" if not problems else "alignment problems", "problems": problems,
                         "n_worlds": len(oracle_gap) if oracle_gap is not None else 0,
                         "myopic_oracle_sequential_regret": describe_vec(oracle_gap, n_boot)
                         if oracle_gap is not None else None,
                         "methods": methods, "decision_families_sequential": dec_seq,
                         "decision_families_myopic": dec_myo, "label_changes": cell_changes,
                         "families_sequential": full, "verdict_sequential": verdict(full["F1"])}
    complete = all(v.get("status") == "ok" for v in cells.values())
    decision = ("substantial disagreement: sequential-reference results authoritative for " + env
                if changes else "no label change: disagreement not substantial") if complete else "incomplete"
    return {"env": env, "cells": cells, "label_changes": changes, "decision": decision}


def sweep_cells(profile, out_dir, n_boot):
    out = {}
    for env in C.ENV_NAMES:
        for c in profile.grid:
            key = f"{env}|{c}"
            res = load_result(out_dir, uid(2, "sweep", env=env, c=c))
            if res is None:
                out[key] = {"status": "missing unit"}
                continue
            vec, tuned_vals = {}, []
            for row in sorted(res["rows"], key=lambda r: r["outer_fold"]):
                vec.setdefault((row["point"], row["param"], row["value"]), []).extend(row["regrets"])
                if row["point"] == "tuned":
                    tuned_vals.append({"outer_fold": row["outer_fold"], "prop_scale": row["prop_scale"],
                                       "stab_scale": row["stab_scale"]})
            dose = {"prop_scale": [], "stab_scale": []}
            for (point, param, value), v in vec.items():
                if point in ("prop", "stab"):
                    m, lo, hi = bca_bootstrap_ci(np.array(v, float), n_boot=n_boot)
                    dose[param].append({"value": value, "mean": m, "ci": [lo, hi]})
            for p in dose:
                dose[p].sort(key=lambda r: r["value"])
            zd = {"tuned": vec.get(("tuned", None, None)), "prop_scale=0": vec.get(("prop", "prop_scale", 0.0)),
                  "stab_scale=0": vec.get(("stab", "stab_scale", 0.0)), "both=0": vec.get(("both_zero", "both", 0.0))}
            zd = {k: np.array(v, float) for k, v in zd.items() if v is not None}
            fam = family([("tuned", "prop_scale=0"), ("tuned", "stab_scale=0"), ("tuned", "both=0")], zd, n_boot)
            repro = res.get("reproduction") or []
            out[key] = {"status": "ok", "env": env, "c": c, "tuned_values": tuned_vals, "dose_response": dose,
                        "tuned_vs_zero": fam,
                        "reproduction_ok": all(r["ok"] for r in repro) if repro else None,
                        "reproduction_max_abs_diff": max((r["max_abs_diff"] for r in repro), default=None)}
    return out


def tpe_cells(profile, out_dir):
    out = {}
    budgets = tuple(sorted(set(profile.sub_budgets) | {profile.tpe_startup}))
    for env in C.ENV_NAMES:
        for c in profile.grid:
            for m in C.ARCHIVED_METHODS:
                key = f"{m}|{env}|{c}"
                res = load_result(out_dir, uid(2, "campaign_tpe", method=m, env=env, c=c))
                if res is None:
                    out[key] = {"status": "missing unit"}
                    continue
                rec = {"status": "ok", "tpe_curve": curve_summary(res["results"], budgets),
                       "best_index_per_fold": [f["best_index"] for f in res["results"]],
                       "startup_reproduction": res.get("startup_reproduction")}
                if profile.scientific:
                    arch = load_archived(env, c, m)
                    sr = res.get("startup_reproduction") or []
                    rec["random_search_curve"] = curve_summary(arch, budgets)
                    rec["best_inner_tpe_minus_random_search"] = [
                        float(t["best_inner_score"] - a["best_inner_score"]) for t, a in zip(res["results"], arch)]
                    rec["startup_reproduction_ok"] = bool(sr) and all(x["ok"] for x in sr)
                out[key] = rec
    return out


def campaign_curves(profile, out_dir):
    """R2 (b) for campaigns run by the revision."""
    out = {}
    for u in enumerate_units(profile):
        if u["kind"] not in ("campaign_frozen", "campaign_sens", "campaign_advt"):
            continue
        res = load_result(out_dir, u["uid"])
        if res is not None:
            out[u["uid"]] = curve_summary(res["results"], C.SUB_BUDGETS_FULL)
    return out


def tuning_cost(profile, out_dir):
    """Simulated trajectories per tuning campaign (plan R6) and measured unit durations."""
    folds = outer_folds(profile.seeds, profile.K_out)
    val = [sum(len(iv) for _t, iv in inner_folds(tr, profile.K_in, k)) for k, (tr, _te) in enumerate(folds)]
    trn = [sum(len(it) for it, _v in inner_folds(tr, profile.K_in, k)) for k, (tr, _te) in enumerate(folds)]
    otr = [len(tr) for tr, _te in folds]
    tst = [len(te) for _tr, te in folds]
    rs = sum(profile.n_trials * v + t for v, t in zip(val, tst))
    out = {"random_search_or_tpe_per_campaign": rs, "trainable": {}}
    for m, space in (("learned", LEARNED_SPACE), ("qlearn", QLEARN_SPACE)):
        rng = np.random.default_rng(0)
        cands = [[sample_config(space, rng) for _ in range(profile.n_trials)] for _ in range(profile.K_out)]
        per_cell = {}
        for env in C.ENV_NAMES:
            for c in profile.grid:
                res = load_result(out_dir, uid(2, "campaign_trainable", method=m, env=env, c=c))
                if res is None:
                    continue
                tot = 0
                for k, f in enumerate(res["results"]):
                    best = f["best_config"]
                    if m == "learned":
                        tot += trn[k] + sum(int(x["onpolicy_rounds"]) for x in cands[k]) * trn[k]
                        tot += otr[k] * (1 + int(best["onpolicy_rounds"]))
                    else:
                        tot += sum(int(x["epochs"]) for x in cands[k]) * trn[k] + int(best["epochs"]) * otr[k]
                    tot += profile.n_trials * val[k] + tst[k]
                per_cell[f"{env}|{c}"] = tot
        out["trainable"][m] = per_cell
    durations = {}
    for u in enumerate_units(profile):
        p = os.path.join(out_dir, "units", u["uid"], "done.json")
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            name = u["kind"] + (":" + u["params"]["method"] if "method" in u["params"] else "")
            durations.setdefault(name, []).append(d["duration_s"])
    out["unit_wall_seconds"] = {k: {"units": len(v), "total": float(np.sum(v)), "median": float(np.median(v))}
                                for k, v in sorted(durations.items())}
    return out


def cd_diagrams(profile, raw_cew, excluded):
    out = {}
    for regime, cs in REGIMES.items():
        cells = [(e, c) for e in C.ENV_NAMES for c in cs if c in profile.grid and (e, c) in raw_cew]
        if not cells:
            continue
        methods = [m for m in CLEAN_METHODS if m not in excluded and all(m in raw_cew[x]["data"] for x in cells)]
        if len(methods) < 3:
            out[regime] = {"status": "fewer than 3 methods"}
            continue
        M = np.column_stack([np.concatenate([raw_cew[x]["data"][m] for x in cells]) for m in methods])
        ranks = np.apply_along_axis(sps.rankdata, 1, M)
        avg = ranks.mean(axis=0)
        chi2, p = sps.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
        k, N = len(methods), M.shape[0]
        q = float(sps.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2))
        cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
        out[regime] = {"status": "ok", "cells": [f"{e}|{c}" for e, c in cells], "blocks": int(N),
                       "methods": methods, "average_ranks": dict(zip(methods, avg.tolist())),
                       "friedman_chi2": float(chi2), "friedman_p": float(p), "nemenyi_q_0.05": q,
                       "critical_difference": float(cd)}
    return out


def multiplicity_s2(cew):
    rows = []
    for key, cell in cew.items():
        if cell.get("status") != "ok":
            continue
        for f, fam in cell["families"].items():
            for r in fam:
                if "code" in r:
                    rows.append((key, f, r))
    changes = []
    if rows:
        hb = holm_bonferroni([r["wilcoxon_p"] for _k, _f, r in rows])
        for (key, f, r), (_i, _p, thr, rej) in zip(rows, hb):
            code, lab = rule(r["a"], r["b"], rej, r["cliffs_delta"])
            if code != r["code"]:
                changes.append({"cell": key, "family": f, "pair": f"{r['a']}_vs_{r['b']}",
                                "within_family": r["label"], "S2": lab})
    return {"n_tests": len(rows), "label_changes": changes}


def completeness(profile, out_dir):
    units = enumerate_units(profile)
    done, failed, missing = [], [], []
    for u in units:
        d = os.path.join(out_dir, "units", u["uid"])
        if os.path.exists(os.path.join(d, "done.json")):
            done.append(u["uid"])
        elif os.path.exists(os.path.join(d, "failed.json")):
            failed.append(u["uid"])
        else:
            missing.append(u["uid"])
    own = uid(5, "analysis")
    done, failed, missing = ([x for x in lst if x != own] for lst in (done, failed, missing))
    return {"units_planned_excluding_analysis": len(units) - 1, "done": len(done), "failed": failed,
            "not_run": missing}


# --------------------------------------------------------------------------- #
# Output: tables, figures, report
# --------------------------------------------------------------------------- #
def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, default=_jsonable) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})


def _family_rows(prefix_cols, fams):
    rows = []
    for f, fam in fams.items():
        for r in fam:
            rows.append({**prefix_cols, "family": f, "a": r["a"], "b": r["b"],
                         **{k: r.get(k) for k in ("n", "wilcoxon_p", "holm_threshold", "holm_reject", "cliffs_delta",
                                                  "mean_diff", "mean_diff_ci", "median_diff", "label", "missing")}})
    return rows


def write_tables(tdir, S):
    os.makedirs(tdir, exist_ok=True)
    rows, frows, vrows = [], [], []
    for key, cell in S["cew"].items():
        env, c = key.split("|")
        if cell.get("status") != "ok":
            continue
        for m, d in cell["methods"].items():
            rows.append({"env": env, "c": c, "method": m, **{k: d[k] for k in (
                "n", "mean", "mean_ci", "median", "median_ci", "cvar95_pooled", "cvar95_pooled_ci_percentile",
                "oracle_full_supervision_fraction", "action_fractions_levels_0_to_5", "harm_rate_mean",
                "unauth_unsafe_rate_mean", "excluded_from_comparisons")}})
        frows += _family_rows({"env": env, "c": c}, cell["families"])
        arch = S["archived"].get(key, {})
        vrows.append({"env": env, "c": c, "published_v2": arch.get("published_verdict_v2"),
                      "cew": cell["verdict"], "cew_tpe": cell["verdict_tpe"],
                      "mann_whitney_archived": arch.get("mann_whitney_verdict"),
                      "S1_archived": arch.get("S1_verdict"), "G": cell["G"].get("G"),
                      "G_ci": cell["G"].get("G_ci"), "G_status": cell["G"].get("status")})
    _write_csv(os.path.join(tdir, "cew_methods.csv"), rows)
    _write_csv(os.path.join(tdir, "cew_families.csv"), frows)
    _write_csv(os.path.join(tdir, "verdicts.csv"), vrows)
    for name, block in (("R5a_adversarial_transfer", "r5a"), ("R5b_adversarial_retuned", "r5b"),
                        ("R9a_sensitivity_retuned", "r9a"), ("R9b_sensitivity_transfer", "r9b")):
        mrows, frs = [], []
        for key, cell in S[block].items():
            if cell.get("status") != "ok":
                continue
            fams = cell.get("families") or cell.get("families_attacked")
            frs += _family_rows({"cell": key}, fams)
            for m, d in cell["methods"].items():
                base = d.get("attacked", d)
                mrows.append({"cell": key, "method": m, "mean": base["mean"], "mean_ci": base["mean_ci"],
                              "harm_rate_mean": base.get("harm_rate_mean"),
                              "unauth_unsafe_rate_mean": base.get("unauth_unsafe_rate_mean"),
                              "clean_same_worlds_mean": d.get("clean_same_worlds", {}).get("mean"),
                              "degradation": d.get("degradation_attacked_minus_clean")})
        _write_csv(os.path.join(tdir, f"{name}_methods.csv"), mrows)
        _write_csv(os.path.join(tdir, f"{name}_families.csv"), frs)
    _write_csv(os.path.join(tdir, "R7_dp_reduced.csv"), S["dp_reduced"])
    seq_rows = []
    for c, cell in S["dp_sequential"]["cells"].items():
        for m, d in cell.get("methods", {}).items():
            seq_rows.append({"c": c, "method": m, "sequential_mean": d["sequential_regret"]["mean"],
                             "sequential_ci": d["sequential_regret"]["mean_ci"],
                             "myopic_mean": d["myopic_regret_mean"]})
    _write_csv(os.path.join(tdir, "R7_dp_sequential_methods.csv"), seq_rows)
    sw = []
    for key, cell in S["sweep"].items():
        if cell.get("status") != "ok":
            continue
        for p, pts in cell["dose_response"].items():
            for pt in pts:
                sw.append({"cell": key, "param": p, "value": pt["value"], "mean": pt["mean"], "ci": pt["ci"]})
    _write_csv(os.path.join(tdir, "R11_sweep_dose_response.csv"), sw)
    ov = []
    for N, row in (S.get("overhead") or {}).get("per_N", {}).items():
        for m, d in row.items():
            ov.append({"N": N, "method": m, **d})
    _write_csv(os.path.join(tdir, "R6_overhead.csv"), ov)


def make_figures(fdir, S, profile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fdir, exist_ok=True)
    errors = []

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(fdir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    def series(block, key_fn, methods, getter):
        return {m: [(c, getter(block.get(key_fn(c), {}), m)) for c in profile.grid] for m in methods}

    def cew_get(cell, m):
        d = cell.get("methods", {}).get(m) if cell.get("status") == "ok" else None
        return (d["mean"], d["mean_ci"]) if d else None

    def panel(ax, data, title):
        for m, pts in data.items():
            pts = [(c, v) for c, v in pts if v is not None]
            if not pts:
                continue
            xs = [c for c, _ in pts]
            ys = [v[0] for _, v in pts]
            err = np.array([[v[0] - v[1][0], v[1][1] - v[0]] for _, v in pts]).T
            ax.errorbar(xs, ys, yerr=np.clip(err, 0, None), marker="o", ms=3, capsize=2, label=m)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("cost ratio c", fontsize=8)
        ax.tick_params(labelsize=7)

    try:
        fig, axes = plt.subplots(1, len(C.ENV_NAMES), figsize=(4 * len(C.ENV_NAMES), 3.2))
        for ax, env in zip(np.atleast_1d(axes), C.ENV_NAMES):
            panel(ax, series(S["cew"], lambda c: f"{env}|{c}", FIG_METHODS, cew_get), env)
        np.atleast_1d(axes)[0].set_ylabel("mean regret (CEW, BCa 95%)", fontsize=8)
        np.atleast_1d(axes)[-1].legend(fontsize=6)
        save(fig, "cew_regret_vs_cost_ratio")
    except Exception as e:
        errors.append(f"cew figure: {e!r}")

    try:
        pairs = [(a, e) for a in C.ADVERSARY_PROFILES for e in C.ENV_NAMES if C.adversary_valid(a, e, profile)]
        fig, axes = plt.subplots(1, len(pairs), figsize=(3.2 * len(pairs), 3.0), squeeze=False)

        def att_get(cell, m):
            d = cell.get("methods", {}).get(m) if cell.get("status") == "ok" else None
            return (d["attacked"]["mean"], d["attacked"]["mean_ci"]) if d else None
        for ax, (a, e) in zip(axes[0], pairs):
            panel(ax, series(S["r5a"], lambda c: f"{a}|{e}|{c}", FIG_METHODS[:6], att_get), f"{a} / {e}")
        axes[0][0].set_ylabel("mean regret, adversary active", fontsize=8)
        axes[0][-1].legend(fontsize=6)
        save(fig, "R5a_adversarial_transfer")
    except Exception as e:
        errors.append(f"adversarial figure: {e!r}")

    try:
        perts = list(C.PERTURBATIONS)
        fig, axes = plt.subplots(len(perts), len(C.ENV_NAMES), figsize=(4 * len(C.ENV_NAMES), 2.8 * len(perts)),
                                 squeeze=False)
        for i, pt in enumerate(perts):
            for j, e in enumerate(C.ENV_NAMES):
                panel(axes[i][j], series(S["r9a"], lambda c: f"{pt}|{e}|{c}", ["A0", "A5", "static", "ewma", "rova"],
                                         cew_get), f"{pt} / {e}")
        axes[0][-1].legend(fontsize=6)
        save(fig, "R9a_sensitivity_retuned")
    except Exception as e:
        errors.append(f"sensitivity figure: {e!r}")

    try:
        envs = list(profile.dp_sizes)
        fig, axes = plt.subplots(1, len(envs), figsize=(4 * len(envs), 3.0), squeeze=False)
        for ax, env in zip(axes[0], envs):
            for c in profile.grid:
                pts = [(r["N"], r["gap_per_agent_step_mean"]) for r in S["dp_reduced"]
                       if r["env"] == env and r["c"] == c and r.get("status") == "ok"]
                if pts:
                    ax.plot(*zip(*sorted(pts)), marker="o", ms=3, label=f"c={c}")
            ax.set_title(env, fontsize=8)
            ax.set_xlabel("agents N", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0][0].set_ylabel("myopic-oracle gap per agent-step", fontsize=8)
        axes[0][-1].legend(fontsize=6)
        save(fig, "R7_dp_gap_vs_N")
    except Exception as e:
        errors.append(f"dp figure: {e!r}")

    try:
        fig, axes = plt.subplots(2, len(C.ENV_NAMES), figsize=(4 * len(C.ENV_NAMES), 5.6), squeeze=False)
        for j, e in enumerate(C.ENV_NAMES):
            for i, p in enumerate(("prop_scale", "stab_scale")):
                ax = axes[i][j]
                for c in profile.grid:
                    cell = S["sweep"].get(f"{e}|{c}", {})
                    if cell.get("status") != "ok":
                        continue
                    pts = cell["dose_response"][p]
                    ax.plot([x["value"] for x in pts], [x["mean"] for x in pts], marker="o", ms=3, label=f"c={c}")
                ax.set_xscale("symlog", linthresh=1e-3)
                ax.set_title(f"{e}: {p}", fontsize=8)
                ax.tick_params(labelsize=7)
        axes[0][-1].legend(fontsize=6)
        save(fig, "R11_penalty_sweep")
    except Exception as e:
        errors.append(f"sweep figure: {e!r}")

    for regime, d in S["cd"].items():
        if d.get("status") != "ok":
            continue
        try:
            ranks = d["average_ranks"]
            k = len(ranks)
            fig, ax = plt.subplots(figsize=(7, 0.35 * k + 1.2))
            order = sorted(ranks, key=ranks.get)
            for i, m in enumerate(order):
                ax.plot([ranks[m]], [i], "o", color="k", ms=4)
                ax.text(ranks[m] + 0.08, i, f"{m} ({ranks[m]:.2f})", va="center", fontsize=7)
            ax.plot([1, 1 + d["critical_difference"]], [-1, -1], color="r", lw=2)
            ax.text(1, -1.5, f"CD = {d['critical_difference']:.3f} (Nemenyi, 0.05)", fontsize=7)
            ax.set_xlim(0.8, k + 0.5)
            ax.set_yticks([])
            ax.invert_yaxis()
            ax.set_xlabel("average rank (1 = lowest regret)", fontsize=8)
            ax.set_title(f"{regime} regime: Friedman p = {d['friedman_p']:.2e}, {d['blocks']} worlds", fontsize=8)
            save(fig, f"R2g_cd_{regime}")
        except Exception as e:
            errors.append(f"cd figure {regime}: {e!r}")
    return errors


def _fmt(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_fmt(v, nd) for v in x) + "]"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}g}"
    return str(x)


def write_report(path, S, profile):
    L = [f"# REV-001 analysis report ({profile.name} profile)", "",
         "Auto-generated from unit results by `rova_revision/analysis.py`. It lists results without "
         "interpretation. Plan: `docs/REVISION_ANALYSIS_PLAN.md`.", ""]
    if not profile.scientific:
        L += ["**NON-SCIENTIFIC PROFILE — plumbing check on non-experimental seeds and shrunken worlds. "
              "No value in this report is a result.**", ""]
    comp = S["completeness"]
    L += ["## Completeness", "", f"- units planned (excluding this analysis): "
          f"{comp['units_planned_excluding_analysis']}; done: {comp['done']}; "
          f"failed: {len(comp['failed'])}; not run: {len(comp['not_run'])}"]
    L += [f"  - failed: `{u}`" for u in comp["failed"]] + [f"  - not run: `{u}`" for u in comp["not_run"][:50]]
    L += ["", f"- Q-learning stability gate: **{S['qlearn_gate']['status']}**",
          f"- sequential-oracle decision (plan R7): **{S['dp_sequential']['decision']}**", ""]
    L += ["## Verdicts per cell", "",
          "| env | c | published v2 | CEW | CEW (TPE-tuned) | Mann–Whitney (archived) | S1 Holm-45 (archived) | G | G 95% CI |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, cell in S["cew"].items():
        env, c = key.split("|")
        a = S["archived"].get(key, {})
        g = cell.get("G", {}) if cell.get("status") == "ok" else {}
        L.append(f"| {env} | {c} | {_fmt(a.get('published_verdict_v2'))} | {_fmt(cell.get('verdict'))} | "
                 f"{_fmt(cell.get('verdict_tpe'))} | {_fmt(a.get('mann_whitney_verdict'))} | "
                 f"{_fmt(a.get('S1_verdict'))} | {_fmt(g.get('G'))} | {_fmt(g.get('G_ci'))} |")
    L += ["", "## Mean regret on the common evaluation worlds (BCa 95%)", "",
          "| env | c | method | mean | 95% CI | median | oracle a*=0 fraction |", "|---|---|---|---|---|---|---|"]
    for key, cell in S["cew"].items():
        if cell.get("status") != "ok":
            continue
        env, c = key.split("|")
        for m, d in cell["methods"].items():
            L.append(f"| {env} | {c} | {m}{' (excluded)' if d['excluded_from_comparisons'] else ''} | "
                     f"{_fmt(d['mean'])} | {_fmt(d['mean_ci'])} | {_fmt(d['median'])} | "
                     f"{_fmt(d['oracle_full_supervision_fraction'])} |")
    L += ["", "## Comparison families on the common evaluation worlds", "",
          "| env | c | family | A | B | δ(A,B) | p | Holm | label |", "|---|---|---|---|---|---|---|---|---|"]
    for key, cell in S["cew"].items():
        if cell.get("status") != "ok":
            continue
        env, c = key.split("|")
        for f, fam in cell["families"].items():
            for r in fam:
                if "missing" in r:
                    L.append(f"| {env} | {c} | {f} | {r['a']} | {r['b']} | – | – | – | missing: {', '.join(r['missing'])} |")
                else:
                    L.append(f"| {env} | {c} | {f} | {r['a']} | {r['b']} | {r['cliffs_delta']:+.3f} | "
                             f"{r['wilcoxon_p']:.2e} | {'reject' if r['holm_reject'] else '–'} | {r['label']} |")
    L += ["", "## Adversarial scenarios (R5a transfer; R5b re-tuned)", "",
          "| profile / env / c | R5a verdict (attacked) | R5a verdict (same worlds, clean) | R5b verdict |",
          "|---|---|---|---|"]
    for key in S["r5a"]:
        a, b = S["r5a"][key], S["r5b"].get(key, {})
        L.append(f"| {key} | {_fmt(a.get('verdict_attacked'))} | {_fmt(a.get('verdict_clean_same_worlds'))} | "
                 f"{_fmt(b.get('verdict'))} |")
    L += ["", "## Simulator-assumption sensitivity (R9a re-tuned; R9b transfer)", "",
          "| perturbation / env / c | R9a verdict | R9b verdict | unperturbed CEW verdict |", "|---|---|---|---|"]
    for key in S["r9a"]:
        a, b = S["r9a"][key], S["r9b"].get(key, {})
        L.append(f"| {key} | {_fmt(a.get('verdict'))} | {_fmt(b.get('verdict'))} | "
                 f"{_fmt(a.get('unperturbed_cew_verdict'))} |")
    L += ["", "## Sequential oracle (R7)", "", "Reduced instances:", "",
          "| env | N | c | gap per agent-step (mean) | 95% CI | max | disagreement (mean) |", "|---|---|---|---|---|---|---|"]
    for r in S["dp_reduced"]:
        if r.get("status") == "ok":
            L.append(f"| {r['env']} | {r['N']} | {r['c']} | {_fmt(r['gap_per_agent_step_mean'])} | {_fmt(r['gap_ci'])} | "
                     f"{_fmt(r['gap_max'])} | {_fmt(r['disagreement_rate_mean'])} |")
    L += ["", f"Full-size {S['dp_sequential']['env']}: label changes under the sequential reference: "
          f"{len(S['dp_sequential']['label_changes'])}", ""]
    for ch in S["dp_sequential"]["label_changes"]:
        L.append(f"- c={ch['c']} {ch['family']} {ch['pair']}: {ch['myopic_reference']} → {ch['sequential_reference']}")
    L += ["", "## Penalty sweep (R11): tuned versus zero", "", "| env / c | comparison | δ | p | label | reproduction ok |",
          "|---|---|---|---|---|---|"]
    for key, cell in S["sweep"].items():
        if cell.get("status") != "ok":
            continue
        for r in cell["tuned_vs_zero"]:
            if "code" in r:
                L.append(f"| {key} | {r['a']} vs {r['b']} | {r['cliffs_delta']:+.3f} | {r['wilcoxon_p']:.2e} | "
                         f"{r['label']} | {_fmt(cell['reproduction_ok'])} |")
    L += ["", "## Decision cost (R6)", ""]
    ov = S.get("overhead") or {}
    for N, row in ov.get("per_N", {}).items():
        L.append(f"- N={N}: " + "; ".join(f"{m} {d['median_ns_per_agent_step']:.0f} ns"
                                          + (f" (×{d['ratio_to_static']:.2f})" if "ratio_to_static" in d else "")
                                          for m, d in row.items() if "median_ns_per_agent_step" in d))
    L += ["", "## Multiplicity", "", f"- S1 (Holm over {S['S1']['n_tests']} published tests): "
          f"{len(S['S1']['label_changes'])} pairwise label changes",
          f"- S2 (Holm over {S['S2']['n_tests']} CEW tests in F1–F7): {len(S['S2']['label_changes'])} label changes", ""]
    L += ["## Consistency checks", ""]
    for k, v in S["consistency"].items():
        L.append(f"- {k}: {_fmt(v)}")
    if S.get("figure_errors"):
        L += ["", "## Figure errors", ""] + [f"- {e}" for e in S["figure_errors"]]
    if S.get("section_errors"):
        L += ["", "## Analysis section errors", ""] + [f"- {e['section']}: {e['error']}" for e in S["section_errors"]]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _safe(errors, name, fn, *args, default=None):
    """Runs one analysis section; an exception is recorded, not propagated, so a
    defect in one section cannot discard the others."""
    try:
        return fn(*args)
    except Exception as e:
        import traceback
        errors.append({"section": name, "error": repr(e), "traceback": traceback.format_exc()})
        return default


def _consistency(profile, S, n_boot):
    archived, sweep, tpe = S["archived"], S["sweep"], S["tpe"]
    ok_sweep = [v["reproduction_ok"] for v in sweep.values() if v.get("status") == "ok"]
    return {
        "gate_passed": (S["gate"] or {}).get("passed"),
        "cew_misaligned_methods": {k: v["misaligned_methods"] for k, v in S["cew"].items()
                                   if v.get("status") == "ok" and v["misaligned_methods"]},
        "archived_bca_matches_stats_summary": (all(m["mean_ci_matches_stats_summary"] for a in archived.values()
                                                   for m in a["methods"].values()) if n_boot == 5000 else None),
        "archived_wilcoxon_p_recomputed_equal": all(r["recomputed_p_matches"] for a in archived.values()
                                                    for r in a["published_pairs"]),
        "archived_pairs_reproduce_published_verdicts": all(a["published_pairs_verdict_recomputed"] ==
                                                           a["published_verdict_v2"] for a in archived.values()),
        "sweep_tuned_point_reproduces_archive": (bool(ok_sweep) and all(ok_sweep)) if profile.scientific else None,
        "tpe_startup_reproduces_archive": (all(v.get("startup_reproduction_ok") for v in tpe.values()
                                               if v.get("status") == "ok") if profile.scientific else None),
        "dp_alignment_problems": {c: v["problems"] for c, v in S["dp_sequential"]["cells"].items()
                                  if v.get("problems")},
        "r5a_attacked_and_clean_worlds_match": all(v["keys_match_clean"] for v in S["r5a"].values()
                                                   if v.get("status") == "ok"),
    }


def run_analysis(profile, out_dir):
    n_boot = profile.n_boot
    adir = os.path.join(out_dir, "analysis")
    os.makedirs(adir, exist_ok=True)
    E = []
    gate_res = load_result(out_dir, uid(1, "gate"))
    trainable, qgate = _safe(E, "trainable", trainable_cells, profile, out_dir, n_boot,
                             default=({}, {"status": "error", "cells": []}))
    excluded = set() if qgate["status"] == "passed" else {"qlearn"}

    cells = [(e, c) for e in C.ENV_NAMES for c in profile.grid]
    arch_cells = [(e, c) for e in C.ENV_NAMES for c in C.GRID] if profile.scientific else cells[:1]
    archived, s1 = _safe(E, "archived", archived_reanalysis, arch_cells, n_boot,
                         default=({}, {"n_tests": 0, "label_changes": [], "verdicts": {}}))
    cew, raw = {}, {}
    for env, c in cells:
        summ, r = _safe(E, f"cew {env}|{c}", cew_cell, out_dir, env, c, excluded, n_boot,
                        default=({"status": "error"}, None))
        cew[f"{env}|{c}"] = summ
        if r is not None:
            raw[(env, c)] = r
    clean_verdicts = {(e, c): cew[f"{e}|{c}"].get("verdict") for e, c in cells}
    anchors = {f"{e}|{c}": _safe(E, f"anchor {e}|{c}", anchor_cell, profile, out_dir, e, c, n_boot,
                                 default={"status": "error"}) for e, c in cells}
    r5a, r5b = _safe(E, "adversarial", adversarial_cells, profile, out_dir, excluded, n_boot, default=({}, {}))
    r9a, r9b = _safe(E, "sensitivity", sensitivity_cells, profile, out_dir, excluded, clean_verdicts, n_boot,
                     default=({}, {}))
    S = {
        "profile": profile.name, "scientific": profile.scientific, "n_boot": n_boot,
        "gate": {"passed": gate_res.get("passed"), "scope": gate_res.get("scope")} if gate_res else None,
        "completeness": _safe(E, "completeness", completeness, profile, out_dir,
                              default={"units_planned_excluding_analysis": None, "done": None, "failed": [],
                                       "not_run": []}),
        "qlearn_gate": qgate, "excluded_from_comparisons": sorted(excluded),
        "archived": archived, "S1": s1, "cew": cew, "anchors": anchors, "trainable": trainable,
        "r5a": r5a, "r5b": r5b, "r9a": r9a, "r9b": r9b,
        "dp_reduced": _safe(E, "dp_reduced", dp_reduced, profile, out_dir, n_boot, default=[]),
        "dp_sequential": _safe(E, "dp_sequential", dp_sequential, profile, out_dir, raw, excluded, n_boot,
                               default={"env": C.DP_FULL_SIZE_ENV, "cells": {}, "label_changes": [],
                                        "decision": "error"}),
        "sweep": _safe(E, "sweep", sweep_cells, profile, out_dir, n_boot, default={}),
        "tpe": _safe(E, "tpe", tpe_cells, profile, out_dir, default={}),
        "campaign_tuning_curves": _safe(E, "campaign_curves", campaign_curves, profile, out_dir, default={}),
        "tuning_cost": _safe(E, "tuning_cost", tuning_cost, profile, out_dir, default={}),
        "overhead": load_result(out_dir, uid(4, "overhead")),
        "cd": _safe(E, "cd", cd_diagrams, profile, raw, excluded, default={}),
    }
    S["S2"] = _safe(E, "S2", multiplicity_s2, cew, default={"n_tests": 0, "label_changes": []})
    S["consistency"] = _safe(E, "consistency", _consistency, profile, S, n_boot, default={})
    S["figure_errors"] = _safe(E, "figures", make_figures, os.path.join(adir, "figures"), S, profile,
                               default=["figure generation failed; see section_errors"])
    _safe(E, "tables", write_tables, os.path.join(adir, "tables"), S)
    S["section_errors"] = E
    spath = os.path.join(adir, "summary.json")
    with open(spath, "w") as f:
        json.dump(S, f, indent=1, default=_jsonable)
    _safe(E, "report", write_report, os.path.join(adir, "REPORT.md"), S, profile)
    with open(spath, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {"summary_path": os.path.relpath(spath, out_dir).replace(os.sep, "/"), "summary_sha256": digest,
            "completeness": S["completeness"], "qlearn_gate": qgate["status"],
            "dp_decision": S["dp_sequential"]["decision"], "consistency": S["consistency"],
            "figure_errors": S["figure_errors"],
            "section_errors": [{"section": e["section"], "error": e["error"]} for e in E]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(C.PROFILES), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    res = run_analysis(C.PROFILES[args.profile], os.path.abspath(args.out))
    print(json.dumps(res, indent=1, default=_jsonable))


if __name__ == "__main__":
    main()
