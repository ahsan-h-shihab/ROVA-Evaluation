"""Statistical helpers shared by the analysis modules (plan section 5).

Only the frozen functions of ``dtcoa_val/analysis/stats.py`` produce tests and
intervals; this module arranges them. Additions required by the protocol:
the CVaR-95 difference of every paired comparison (percentile bootstrap over
matched worlds, flagged low precision) and the secondary Bayesian ROPE layer
(posterior of Cliff's delta under the Bayesian bootstrap of the matched worlds,
with the protocol's own negligibility band as the region of practical
equivalence).
"""
import numpy as np
from scipy import stats as sps

from dtcoa_val.analysis.stats import (bca_bootstrap_ci, bootstrap_ci, cliffs_delta, holm_bonferroni,
                                      wilcoxon_paired)
from dtcoa_val.experiments.nested_hpo import _cvar

NEGLIGIBLE = 0.147


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


def rope_posterior(xa, xb, n_post, band=NEGLIGIBLE):
    """P(delta < -band), P(|delta| <= band), P(delta > band) for Cliff's delta(a, b)
    under Dirichlet(1,...,1) weights on the matched worlds (Bayesian bootstrap)."""
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    S = np.sign(xa[:, None] - xb[None, :])
    W = np.random.default_rng(0).dirichlet(np.ones(len(xa)), size=n_post)
    d = ((W @ S) * W).sum(axis=1)
    return {"p_a_lower": float(np.mean(d < -band)), "p_rope": float(np.mean(np.abs(d) <= band)),
            "p_b_lower": float(np.mean(d > band)), "posterior_mean_delta": float(d.mean()),
            "rope": [-band, band], "n_post": int(n_post)}


def compare(xa, xb, n_boot, n_post=0):
    """Paired comparison on matched worlds: Wilcoxon, Cliff's delta, BCa of the mean
    difference, CVaR-95 difference, and (if n_post) the Bayesian ROPE posterior."""
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    stat, p = wilcoxon_paired(xa, xb)
    diff = xa - xb
    m, lo, hi = bca_bootstrap_ci(diff, n_boot=n_boot)
    idx = np.arange(len(xa), dtype=float)
    cv, clo, chi = bootstrap_ci(idx, n_boot=n_boot,
                                statistic=lambda ix: _cvar(xa[ix.astype(int)], 0.95) - _cvar(xb[ix.astype(int)], 0.95))
    out = {"n": int(len(xa)), "wilcoxon_stat": _f(stat), "wilcoxon_p": float(p),
           "cliffs_delta": cliffs_delta(xa, xb), "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
           "mean_diff": m, "mean_diff_ci": [lo, hi],
           "median_diff": float(np.median(xa) - np.median(xb)), "median_of_paired_diff": float(np.median(diff)),
           "cvar95_diff": cv, "cvar95_diff_ci_percentile": [clo, chi]}
    if n_post:
        out["rope"] = rope_posterior(xa, xb, n_post)
    return out


def rule(a, b, reject, delta):
    """Plan section 1 comparison rule (within-family Holm)."""
    if reject and abs(delta) > NEGLIGIBLE:
        return ("a_lower", f"{a} lower-regret") if delta < 0 else ("b_lower", f"{b} lower-regret")
    return ("none", "no detected difference")


def family(pairs, data, n_boot, n_post=0):
    rows = []
    for a, b in pairs:
        if a in data and b in data:
            rows.append({"a": a, "b": b, **compare(data[a], data[b], n_boot, n_post)})
        else:
            rows.append({"a": a, "b": b, "missing": [m for m in (a, b) if m not in data]})
    holm_rows(rows)
    return rows


def holm_rows(rows):
    avail = [r for r in rows if "wilcoxon_p" in r]
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
