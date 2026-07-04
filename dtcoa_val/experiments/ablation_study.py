"""One-at-a-time, re-tuned ablation study (protocol §7, frozen-scope version
-- full factorial is future work, see PROJECT_SCOPE_FREEZE.md) and the
containment/DR4 report (does the tuned-optimal penalty magnitude collapse
toward zero?).

Each ablation is independently re-tuned under the SAME nested-CV-HPO budget
as the full method -- re-tuning is required per protocol §7, or removing a
component gets confounded with simply de-tuning everything else.
"""
from dtcoa_val.dtcoa_tunable import DTCOATunable
from dtcoa_val.experiments.nested_hpo import nested_cv_hpo
from dtcoa_val.experiments.search_space import ROVA_SEARCH_SPACE, ROVA_ABLATIONS, make_ablation_factory


def run_ablation_study(cfg, seeds, checkpoint_dir=None, tag_prefix="", **hpo_kwargs):
    """Runs the full-method HPO plus all four one-at-a-time ablations, each
    independently re-tuned. Returns {"full": [...], "no_memory": [...], ...}
    -- each value is the list of per-outer-fold nested_cv_hpo results.
    Extra hpo_kwargs (K_out, K_in, n_trials, seed0) are forwarded to every
    nested_cv_hpo call, e.g. for a reduced-budget pilot run.
    """
    results = {}
    results["full"] = nested_cv_hpo(
        DTCOATunable, ROVA_SEARCH_SPACE, cfg, seeds,
        tag=f"{tag_prefix}full", checkpoint_dir=checkpoint_dir, **hpo_kwargs,
    )
    for name, spec in ROVA_ABLATIONS.items():
        factory = make_ablation_factory(DTCOATunable, spec["fixed_kwargs"])
        results[name] = nested_cv_hpo(
            factory, spec["search_space"], cfg, seeds,
            tag=f"{tag_prefix}{name}", checkpoint_dir=checkpoint_dir, **hpo_kwargs,
        )
    return results


def ablation_report(ablation_results):
    """DR3 (mechanism attribution): compare each ablation's outer-fold test
    regrets against the full method's, via Wilcoxon + Cliff's delta, Holm
    corrected across the family of ablation comparisons."""
    import numpy as np
    from dtcoa_val.analysis.stats import wilcoxon_paired, cliffs_delta, holm_bonferroni

    full = ablation_results["full"]
    full_regrets = [r["test_mean_regret"] for r in full]
    names = [n for n in ablation_results if n != "full"]
    pvals, deltas = [], []
    for name in names:
        ab_regrets = [r["test_mean_regret"] for r in ablation_results[name]]
        w = wilcoxon_paired(ab_regrets, full_regrets)
        d = cliffs_delta(ab_regrets, full_regrets)
        pvals.append(w[1])
        deltas.append(d)
    hb = holm_bonferroni(pvals)
    report = {}
    for i, name in enumerate(names):
        _, p, thr, reject = hb[i]
        report[name] = {
            "cliffs_delta_vs_full": deltas[i],
            "holm_p": p,
            "holm_threshold": thr,
            "significant_degradation": bool(reject and deltas[i] > 0.147),
            "note": (
                "DR3: significant degradation when this component is removed "
                "-> contributes to performance"
                if reject and deltas[i] > 0.147 else
                "DR3: no significant/practically-meaningful degradation detected "
                "-> component's contribution not established at this budget"
            ),
        }
    return report


def containment_report(full_results):
    """DR4 containment check: does the tuned-optimal penalty magnitude
    collapse toward zero (an expressiveness/containment effect, no distinct
    mechanism) or stay bounded away from zero (a genuine mechanism)?
    Uses the loguniform_with_zero floor value 0.0 as the reference for
    'collapsed'."""
    import numpy as np

    prop_scales = [r["best_config"]["prop_scale"] for r in full_results]
    stab_scales = [r["best_config"]["stab_scale"] for r in full_results]
    return {
        "prop_scale_per_fold": prop_scales,
        "stab_scale_per_fold": stab_scales,
        "prop_scale_median": float(np.median(prop_scales)),
        "stab_scale_median": float(np.median(stab_scales)),
        "prop_collapsed_to_zero": bool(np.median(prop_scales) < 1e-6),
        "stab_collapsed_to_zero": bool(np.median(stab_scales) < 1e-6),
        "note": (
            "If BOTH collapse to ~0 across folds while the full method still "
            "weakly dominates baselines, report per DR4: expressiveness/"
            "containment effect, not a mechanism -- report margin and compute "
            "cost only, do not claim a mechanism."
        ),
    }
