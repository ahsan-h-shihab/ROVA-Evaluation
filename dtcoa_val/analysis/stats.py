"""Reusable statistics: bootstrap CIs, Wilcoxon signed-rank, Holm-Bonferroni,
Cliff's delta effect size, descriptives. Used by ALL experiments (no per-experiment stats hacks)."""
import numpy as np
from scipy import stats as sps

def bootstrap_ci(x, n_boot=5000, alpha=0.05, statistic=np.mean, rng=None):
    """Percentile bootstrap CI. Retained for backward compatibility with v1/v2
    campaign code; new analyses should use bca_bootstrap_ci (TMLR-quality
    reviewers expect bias-corrected-and-accelerated intervals, not plain
    percentile ones, per PRE_REGISTERED_PROTOCOL.md §8)."""
    x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    rng = rng or np.random.default_rng(0)
    boots = np.array([statistic(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return (float(statistic(x)), float(lo), float(hi))


def bca_bootstrap_ci(x, n_boot=5000, alpha=0.05, statistic=np.mean, rng=None):
    """Bias-corrected-and-accelerated (BCa) bootstrap CI, per
    PRE_REGISTERED_PROTOCOL.md §8. Corrects the plain percentile bootstrap
    for both median bias (z0) and skewness (acceleration, via jackknife).
    Falls back to the percentile interval if the acceleration/bias-correction
    is degenerate (e.g. n too small, or the statistic is exactly constant).
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return bootstrap_ci(x, n_boot, alpha, statistic, rng)
    rng = rng or np.random.default_rng(0)
    theta_hat = float(statistic(x))
    boots = np.array([statistic(rng.choice(x, size=n, replace=True)) for _ in range(n_boot)])

    # Bias-correction z0
    prop_less = np.mean(boots < theta_hat)
    if prop_less <= 0.0 or prop_less >= 1.0:
        return bootstrap_ci(x, n_boot, alpha, statistic, rng)
    z0 = sps.norm.ppf(prop_less)

    # Acceleration via jackknife
    jack = np.array([statistic(np.delete(x, i)) for i in range(n)])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a_hat = num / den if den != 0 else 0.0

    z_alpha_lo = sps.norm.ppf(alpha / 2)
    z_alpha_hi = sps.norm.ppf(1 - alpha / 2)

    def _adj(z_a):
        denom = 1 - a_hat * (z0 + z_a)
        if denom == 0:
            return np.nan
        return sps.norm.cdf(z0 + (z0 + z_a) / denom)

    lo_pct = _adj(z_alpha_lo)
    hi_pct = _adj(z_alpha_hi)
    if np.isnan(lo_pct) or np.isnan(hi_pct):
        return bootstrap_ci(x, n_boot, alpha, statistic, rng)
    lo, hi = np.percentile(boots, [100 * lo_pct, 100 * hi_pct])
    return (theta_hat, float(lo), float(hi))


def friedman_holm(method_scores):
    """Friedman test across >2 methods evaluated on matched blocks (same
    seeds/cost-ratios), with Holm post-hoc pairwise correction, per
    PRE_REGISTERED_PROTOCOL.md §8.

    Args:
        method_scores: dict of method_name -> 1D array-like, all the SAME
            length and block-aligned (e.g. per-seed or per-cost-ratio paired
            values). At least 3 methods required.

    Returns:
        dict with 'friedman_stat', 'friedman_p', and 'holm_pairwise':
        list of {method_a, method_b, wilcoxon_p, holm_reject}. Caveat (per
        protocol §8): the Friedman omnibus assumes independent blocks, which
        does not strictly hold when comparing conditions on the same scenario
        instances; the per-pair Wilcoxon+Holm results carry the primary
        evidentiary weight, the omnibus is reported as a supporting summary.
    """
    names = list(method_scores.keys())
    mat = np.array([np.asarray(method_scores[k], float) for k in names])  # methods x blocks
    if mat.shape[0] < 3:
        raise ValueError("friedman_holm requires at least 3 methods")
    stat, p = sps.friedmanchisquare(*mat)

    pairwise = []
    pvals = []
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            w = wilcoxon_paired(mat[i], mat[j])
            pvals.append(w[1])
            pairs.append((names[i], names[j]))
    hb = holm_bonferroni(pvals)
    for (a, b), (_, p_i, thr, reject) in zip(pairs, hb):
        pairwise.append({"method_a": a, "method_b": b, "wilcoxon_p": p_i,
                          "holm_threshold": thr, "holm_reject": reject})
    return {"friedman_stat": float(stat), "friedman_p": float(p), "holm_pairwise": pairwise}

def wilcoxon_paired(a, b):
    """Paired Wilcoxon signed-rank; returns (stat, p). NaN-safe."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0): return (0.0, 1.0)
    try:
        s, p = sps.wilcoxon(a, b, zero_method="wilcox", correction=False)
        return (float(s), float(p))
    except ValueError:
        return (np.nan, 1.0)

def holm_bonferroni(pvals, alpha=0.05):
    """Returns list of (idx, p, adjusted_threshold, reject) preserving input order."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p)
    out = [None]*m; prev_reject = True
    for rank, idx in enumerate(order):
        thr = alpha/(m-rank)
        reject = prev_reject and (p[idx] <= thr)
        prev_reject = reject
        out[idx] = (int(idx), float(p[idx]), float(thr), bool(reject))
    return out

def cliffs_delta(a, b):
    """Nonparametric effect size in [-1,1]. >0 means a tends to exceed b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a)==0 or len(b)==0: return np.nan
    gt = sum((ai > b).sum() for ai in a); lt = sum((ai < b).sum() for ai in a)
    return float((gt - lt) / (len(a)*len(b)))

def describe(x):
    x = np.asarray(x, float)
    m, lo, hi = bca_bootstrap_ci(x)
    return {"n": int(len(x)), "mean": m, "ci_lo": lo, "ci_hi": hi,
            "std": float(np.std(x, ddof=1)) if len(x)>1 else 0.0,
            "median": float(np.median(x)) if len(x) else np.nan,
            "min": float(np.min(x)) if len(x) else np.nan,
            "max": float(np.max(x)) if len(x) else np.nan}
