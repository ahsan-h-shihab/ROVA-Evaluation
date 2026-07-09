"""ABLATION-001 DR3 regime-level inference (Option B primary + Option C sensitivity).

Implements the pre-specified two-stage attribution analysis, replacing the naive
per-cell significance test (which has structurally zero power at K_out=5) and the
count-of-per-cell-flags rollup (which inherits that vacuity). See the design
review: pooling raw fold-differences is pseudoreplicative; the honest inference
unit is the CELL (clustered by environment).

Stage 1 (within cell, uses all folds/trials -> precise scale-free per-cell effect):
  primary effect = Cliff's delta of ablation-vs-full nested-CV test regrets,
  computed on the concatenated per-fold test_regrets. delta>0 => removing the
  component worsens regret => contributes. Robustness measures also computed:
  fold-level Cliff's delta (5v5), standardized mean difference, relative regret
  increase (flagged unstable near regret 0).

Stage 2 (across cells within a cost-ratio regime):
  Option B: one-sample sign test + Wilcoxon signed-rank on the 6 cell-level
            deltas (H0: median delta = 0); environment-clustered bootstrap CI.
  Option C: exact within-cell sign-flip permutation (statistic = mean cell
            delta; null = all 2^n sign patterns) -- uses delta magnitudes, so
            finer-grained than the sign test at the same n=6 exchangeability.
  Robustness: environment-level (n=3) deltas, DESCRIPTIVE only.
  H2: low-vs-high delta contrast via exact label permutation (descriptive).
  Multiplicity: uncorrected, Holm (FWER) and Benjamini-Hochberg (FDR) across the
  4-component family. Effect sizes are primary; every non-flagged component is
  reported as "not detected at this budget", never "inert".

READ-ONLY with respect to MAIN-001; reads only the ABLATION-001 raw checkpoints.
Deterministic: permutations are enumerated exactly; the bootstrap uses a fixed
seed -> reproducible from the raw checkpoints alone.
"""
import argparse
import itertools
import os

import numpy as np
from scipy import stats

from dtcoa_val.utils import load_json, save_json
from dtcoa_val.experiments.environments import ENVIRONMENTS, COST_RATIO_GRID
from dtcoa_val.analysis.stats import cliffs_delta, holm_bonferroni

COMPONENTS = ["no_memory", "no_decay", "no_propagation", "no_stability"]
_MECHANISM = {"no_memory": "memory", "no_decay": "decay",
              "no_propagation": "propagation", "no_stability": "stability"}
# Regime boundary: MAIN-001 verdict table (documented constant, not a runtime
# read of MAIN-001). low = c<=1.5, high = c>=8.0, 3.5 = transition.
LOW_REGIME_MAX = 1.5
HIGH_REGIME_MIN = 8.0
NEGLIGIBLE = 0.147  # |delta| below this is a negligible effect (matches frozen report)
BOOTSTRAP_SEED = 0
N_BOOTSTRAP = 10000
NOT_DETECTED = "not detected at this budget (NOT evidence of inertness)"


def regime_of(c):
    if c <= LOW_REGIME_MAX:
        return "low"
    if c >= HIGH_REGIME_MIN:
        return "high"
    return "transition"


# --------------------------------------------------------------------------- #
# Stage 1: per-cell standardized effects.
# --------------------------------------------------------------------------- #
def _branch_regrets(ckpt_dir, env, c, branch):
    """Return (trial_regrets concatenated across folds, per-fold mean regrets)."""
    p = os.path.join(ckpt_dir, f"{env}__c{c}__{branch}.json")
    if not os.path.exists(p):
        return None, None
    res = load_json(p)["results"]
    trial = np.array([r for fold in res for r in fold["test_regrets"]], float)
    fold_means = np.array([fold["test_mean_regret"] for fold in res], float)
    return trial, fold_means


def cell_effects(ckpt_dir, env, c):
    """Per-cell standardized effect (primary + robustness) for each component."""
    full_trial, full_fold = _branch_regrets(ckpt_dir, env, c, "full")
    if full_trial is None:
        return None
    out = {}
    for comp in COMPONENTS:
        ab_trial, ab_fold = _branch_regrets(ckpt_dir, env, c, comp)
        if ab_trial is None:
            return None
        mean_full = float(full_trial.mean())
        # pooled sd from fold means (generalization-level dispersion), spurious
        # pairing (F3) -> pooled, not paired, sd.
        pooled_sd = float(np.sqrt((full_fold.var(ddof=1) + ab_fold.var(ddof=1)) / 2)) \
            if len(full_fold) > 1 else 0.0
        out[comp] = {
            "mechanism": _MECHANISM[comp],
            "cliffs_delta": float(cliffs_delta(ab_trial, full_trial)),          # PRIMARY
            "cliffs_delta_fold": float(cliffs_delta(ab_fold, full_fold)),       # robustness
            "smd": (float((ab_trial.mean() - mean_full) / pooled_sd)
                    if pooled_sd > 0 else float("nan")),                        # robustness
            "relative_increase": (float((ab_trial.mean() - mean_full) / mean_full)
                                  if abs(mean_full) > 1e-6 else float("nan")),  # robustness (unstable near 0)
            "mean_full": mean_full,
            "mean_ablation": float(ab_trial.mean()),
        }
    return out


def stage1(ckpt_dir, envs, grid):
    """Assemble per-(component, cell) primary delta and metadata."""
    cells = {}
    missing = []
    for env in envs:
        for c in grid:
            eff = cell_effects(ckpt_dir, env, c)
            tag = f"{env}__c{c}"
            if eff is None:
                missing.append(tag)
                continue
            cells[tag] = {"env": env, "cost_ratio": c, "regime": regime_of(c), "effects": eff}
    return cells, missing


# --------------------------------------------------------------------------- #
# Stage 2 statistics.
# --------------------------------------------------------------------------- #
def sign_test(deltas):
    """Two-sided exact binomial sign test, H0: median = 0 (drop exact zeros)."""
    d = np.asarray(deltas, float)
    pos = int((d > 0).sum()); neg = int((d < 0).sum()); n = pos + neg
    if n == 0:
        return {"n": 0, "n_pos": 0, "p": 1.0}
    p = float(min(1.0, 2 * stats.binom.cdf(min(pos, neg), n, 0.5)))
    return {"n": n, "n_pos": pos, "p": p}


def signed_rank(deltas):
    d = np.asarray(deltas, float)
    if np.allclose(d, 0) or len(d) < 2:
        return {"p": 1.0}
    try:
        _, p = stats.wilcoxon(d, zero_method="wilcox", correction=False)
        return {"p": float(p)}
    except ValueError:
        return {"p": 1.0}


def signflip_permutation(deltas):
    """Option C: exact within-cell sign-flip permutation. Statistic = mean(delta);
    null = every 2^n sign assignment. Two-sided p on |mean|. Deterministic."""
    d = np.asarray(deltas, float)
    n = len(d)
    if n == 0:
        return {"p": 1.0, "n_perm": 0, "stat": float("nan")}
    obs = abs(float(d.mean()))
    if n <= 22:  # exact enumeration (frozen grid: <=6/regime -> 64)
        ge = sum(abs(float((d * np.array(s)).mean())) >= obs - 1e-12
                 for s in itertools.product((-1, 1), repeat=n))
        return {"p": ge / (2 ** n), "n_perm": 2 ** n, "stat": float(d.mean()), "exact": True}
    rng = np.random.default_rng(BOOTSTRAP_SEED)  # deterministic fallback
    draws = rng.choice((-1, 1), size=(N_BOOTSTRAP, n))
    ge = int((np.abs((draws * d).mean(axis=1)) >= obs - 1e-12).sum())
    return {"p": (ge + 1) / (N_BOOTSTRAP + 1), "n_perm": N_BOOTSTRAP, "stat": float(d.mean()), "exact": False}


def env_cluster_bootstrap_ci(cell_deltas_by_env, alpha=0.05):
    """Percentile CI for the mean cell delta, resampling ENVIRONMENTS (clusters).
    Deterministic (fixed seed)."""
    envs = list(cell_deltas_by_env)
    if not envs:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = []
    for _ in range(N_BOOTSTRAP):
        pick = rng.choice(len(envs), size=len(envs), replace=True)
        vals = [v for i in pick for v in cell_deltas_by_env[envs[i]]]
        means.append(np.mean(vals))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def h2_contrast(low_deltas, high_deltas):
    """Descriptive low-vs-high contrast via exact label permutation of the pooled
    cell deltas. Statistic = mean(high) - mean(low)."""
    lo = list(low_deltas); hi = list(high_deltas)
    pooled = np.array(lo + hi, float); nlow = len(lo); n = len(pooled)
    if nlow == 0 or len(hi) == 0:
        return {"p": float("nan"), "delta_high_minus_low": float("nan")}
    obs = float(np.mean(hi) - np.mean(lo))
    from math import comb
    if comb(n, nlow) <= 200000:  # exact (frozen grid: C(12,6)=924)
        ge = 0; tot = 0
        for idx in itertools.combinations(range(n), nlow):
            mask = np.zeros(n, bool); mask[list(idx)] = True
            ge += abs(float(pooled[~mask].mean() - pooled[mask].mean())) >= abs(obs) - 1e-12
            tot += 1
        return {"p": ge / tot, "delta_high_minus_low": obs, "n_perm": tot, "exact": True}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ge = 0
    for _ in range(N_BOOTSTRAP):
        perm = rng.permutation(n)
        ge += abs(float(pooled[perm[nlow:]].mean() - pooled[perm[:nlow]].mean())) >= abs(obs) - 1e-12
    return {"p": (ge + 1) / (N_BOOTSTRAP + 1), "delta_high_minus_low": obs, "n_perm": N_BOOTSTRAP, "exact": False}


# --------------------------------------------------------------------------- #
# Assembly.
# --------------------------------------------------------------------------- #
def _regime_component(cells, comp, regime):
    sel = {t: cell for t, cell in cells.items() if cell["regime"] == regime}
    deltas = [cell["effects"][comp]["cliffs_delta"] for cell in sel.values()]
    by_env = {}
    for cell in sel.values():
        by_env.setdefault(cell["env"], []).append(cell["effects"][comp]["cliffs_delta"])
    env_level = {e: float(np.mean(v)) for e, v in by_env.items()}
    return {
        "n_cells": len(deltas),
        "cell_deltas": deltas,
        "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
        "median_delta": float(np.median(deltas)) if deltas else float("nan"),
        "sign_test": sign_test(deltas),
        "signed_rank": signed_rank(deltas),
        "permutation_optionC": signflip_permutation(deltas),
        "env_cluster_bootstrap_ci": env_cluster_bootstrap_ci(by_env),
        "env_level_mean_delta": env_level,          # n=3, descriptive only
        "non_negligible_cells": int(sum(abs(d) > NEGLIGIBLE for d in deltas)),
    }


def build(ckpt_dir, envs=None, grid=None):
    envs = list(envs or ENVIRONMENTS); grid = list(grid or COST_RATIO_GRID)
    cells, missing = stage1(ckpt_dir, envs, grid)
    regimes = ["low", "high", "transition"]
    per_component = {}
    for comp in COMPONENTS:
        per_component[comp] = {"mechanism": _MECHANISM[comp], "by_regime": {}}
        for reg in regimes:
            per_component[comp]["by_regime"][reg] = _regime_component(cells, comp, reg)
        low_d = per_component[comp]["by_regime"]["low"]["cell_deltas"]
        high_d = per_component[comp]["by_regime"]["high"]["cell_deltas"]
        per_component[comp]["H2_low_vs_high"] = h2_contrast(low_d, high_d)

    # Multiplicity across the 4-component family, per regime, on the Option-C p.
    multiplicity = {}
    for reg in ["low", "high"]:
        pvals = [per_component[c]["by_regime"][reg]["permutation_optionC"]["p"] for c in COMPONENTS]
        holm = holm_bonferroni(pvals)
        bh = _benjamini_hochberg(pvals)
        multiplicity[reg] = {
            COMPONENTS[i]: {"p_uncorrected": pvals[i],
                            "holm_reject": bool(holm[i][3]), "holm_threshold": holm[i][2],
                            "bh_reject": bool(bh[i])}
            for i in range(len(COMPONENTS))
        }

    return {
        "n_cells": len(cells), "missing_cells": missing,
        "primary_effect": "cliffs_delta (ablation vs full trial regrets)",
        "inference_unit": "cell (n<=6/regime); environment-level reported as descriptive robustness",
        "regime_boundary": {"low_max": LOW_REGIME_MAX, "high_min": HIGH_REGIME_MIN},
        "multiplicity_note": "Effect sizes are primary; p reported uncorrected + Holm(FWER) + BH(FDR).",
        "wording_policy": NOT_DETECTED,
        "per_component": per_component,
        "multiplicity": multiplicity,
        "cells": cells,
    }


def _benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    reject = np.zeros(m, bool); thr_max = -1
    for rank, idx in enumerate(order, start=1):
        if p[idx] <= alpha * rank / m:
            thr_max = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= thr_max:
            reject[idx] = True
    return reject.tolist()


def render_markdown(summary):
    L = ["# ABLATION-001 DR3 Regime-Level Attribution (Option B + C)\n",
         f"Primary effect: {summary['primary_effect']}. Inference unit: {summary['inference_unit']}.",
         f"Cells: {summary['n_cells']}. _{summary['wording_policy']}_\n",
         "## Per component x regime (Cliff's delta; delta>0 => removal worsens regret)\n",
         "| Mechanism | Regime | n | mean delta | env-cluster 95% CI | sign p | Option-C perm p | non-negligible cells |",
         "|---|---|---|---|---|---|---|---|"]
    for comp in COMPONENTS:
        for reg in ["low", "high", "transition"]:
            r = summary["per_component"][comp]["by_regime"][reg]
            ci = r["env_cluster_bootstrap_ci"]
            L.append(f"| {_MECHANISM[comp]} | {reg} | {r['n_cells']} | {r['mean_delta']:.3f} | "
                     f"[{ci[0]:.3f}, {ci[1]:.3f}] | {r['sign_test']['p']:.3f} | "
                     f"{r['permutation_optionC']['p']:.3f} | {r['non_negligible_cells']}/{r['n_cells']} |")
    L += ["\n## H2 (does the contributing set differ between regimes?) — DESCRIPTIVE\n",
          "| Mechanism | mean delta low | mean delta high | high-low | perm p (descriptive) |",
          "|---|---|---|---|---|"]
    for comp in COMPONENTS:
        lo = summary["per_component"][comp]["by_regime"]["low"]["mean_delta"]
        hi = summary["per_component"][comp]["by_regime"]["high"]["mean_delta"]
        h2 = summary["per_component"][comp]["H2_low_vs_high"]
        L.append(f"| {_MECHANISM[comp]} | {lo:.3f} | {hi:.3f} | {h2['delta_high_minus_low']:.3f} | {h2['p']:.3f} |")
    L += ["\n## Multiplicity (Option-C p across the 4-component family)\n",
          "| Regime | Mechanism | p | Holm reject | BH reject |", "|---|---|---|---|---|"]
    for reg in ["low", "high"]:
        for comp in COMPONENTS:
            m = summary["multiplicity"][reg][comp]
            L.append(f"| {reg} | {_MECHANISM[comp]} | {m['p_uncorrected']:.3f} | "
                     f"{m['holm_reject']} | {m['bh_reject']} |")
    L.append("\n_Environment-level (n=3) effects are in the JSON as descriptive robustness only._")
    return "\n".join(L) + "\n"


def run(ablation_out, out_dir=None):
    out_dir = out_dir or os.path.join(ablation_out, "dr3_analysis")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(ablation_out, "hpo_checkpoints")
    summary = build(ckpt_dir)
    save_json(os.path.join(out_dir, "dr3_regime_inference.json"), summary)
    with open(os.path.join(out_dir, "dr3_regime_inference.md"), "w") as f:
        f.write(render_markdown(summary))
    return summary, out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-out", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    summary, out_dir = run(args.ablation_out, args.out_dir)
    print(f"DR3 regime inference written to {out_dir} ({summary['n_cells']} cells, "
          f"{len(summary['missing_cells'])} missing)")
