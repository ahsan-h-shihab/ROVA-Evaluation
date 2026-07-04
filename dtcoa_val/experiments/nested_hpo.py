"""Matched-budget random-search HPO with nested cross-validation.

Implements PRE_REGISTERED_PROTOCOL.md §3 (HPO) and §4 (nested CV):
- Identical HPO algorithm (random search) and identical trial budget for
  every method (§3.4).
- The unit of cross-validation is the scenario instance (a seed), not a
  timestep within an instance (§4, last bullet).
- K_out=5 outer folds estimate generalization; K_in=5 inner folds select
  hyperparameters within each outer training set; each outer test fold is
  touched exactly once, after tuning is complete for that fold (§4).
- Tuning curve (best-so-far vs. trials used) is recorded at the pre-registered
  sub-budgets so budget-dependence is visible rather than hidden (§3.6).

This module deliberately does NOT decide what counts as "success" (DR1-DR8
in the protocol) — it only produces the per-outer-fold test regret/CVaR
distributions and the selected configs. Decision rules are applied downstream
against the aggregated statistics (see analysis/stats.py additions).
"""
import numpy as np
from dtcoa_val.experiments.runner import run_trial
from dtcoa_val.experiments.search_space import sample_config

# Frozen PI-scoped budget (see PROJECT_SCOPE_FREEZE.md): K_out=5, K_in=3,
# n_trials=80. Full-spec values (K_in=5, n_trials=200) are future work.
DEFAULT_K_OUT = 5
DEFAULT_K_IN = 3
DEFAULT_N_TRIALS = 80
SUB_BUDGETS = (20, 40, 80)


def kfold_split(seeds, k, rng):
    """Split a list of scenario-instance seeds into k folds; return list of
    (train_seeds, test_seeds) tuples, one per fold, covering all seeds."""
    seeds = list(seeds)
    rng.shuffle(seeds)
    folds = [seeds[i::k] for i in range(k)]
    out = []
    for i in range(k):
        test = folds[i]
        train = [s for j, f in enumerate(folds) if j != i for s in f]
        out.append((train, test))
    return out


def _cvar(losses, q=0.95):
    """CVaR at the q-th percentile: mean of the worst (1-q) fraction of losses."""
    losses = np.sort(np.asarray(losses, float))
    n = len(losses)
    if n == 0:
        return float("nan")
    cutoff = int(np.ceil(q * n))
    tail = losses[cutoff:] if cutoff < n else losses[-1:]
    return float(np.mean(tail))


def _eval_config(policy_class, config, cfg, seeds, tag):
    trials = [run_trial(cfg, policy_class(**config), seed=s, tag=tag) for s in seeds]
    regrets = [t["regret"] for t in trials]
    return {
        "mean_regret": float(np.mean(regrets)),
        "cvar95_regret": _cvar(regrets, 0.95),
        "regrets": regrets,
    }


def nested_cv_hpo(
    policy_class,
    search_space,
    cfg,
    all_seeds,
    K_out=DEFAULT_K_OUT,
    K_in=DEFAULT_K_IN,
    n_trials=DEFAULT_N_TRIALS,
    seed0=0,
    tag="",
    checkpoint_dir=None,
):
    """Returns a list of per-outer-fold result dicts:
    {outer_fold, best_config, tuning_curve (per sub-budget best-so-far mean
    regret on the inner-validation average), test_mean_regret, test_cvar95,
    test_regrets}
    """
    import os
    from dtcoa_val.utils import save_json, load_json

    master_rng = np.random.default_rng(seed0)
    outer_folds = kfold_split(all_seeds, K_out, np.random.default_rng(seed0))

    ckpt_path = None
    results = []
    done_folds = set()
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"{tag}.json")
        if os.path.exists(ckpt_path):
            prior = load_json(ckpt_path)
            results = prior.get("results", [])
            done_folds = {r["outer_fold"] for r in results}

    for out_i, (outer_train, outer_test) in enumerate(outer_folds):
        if out_i in done_folds:
            continue
        inner_rng = np.random.default_rng(seed0 * 1_000_003 + out_i)
        inner_folds = kfold_split(outer_train, K_in, np.random.default_rng(seed0 * 7 + out_i))

        candidate_configs = [sample_config(search_space, master_rng) for _ in range(n_trials)]
        config_scores = np.full(n_trials, np.nan)
        running_best = np.inf
        tuning_curve = {}

        for i, conf in enumerate(candidate_configs):
            val_scores = []
            for fold_j, (_inner_train, inner_val) in enumerate(inner_folds):
                r = _eval_config(
                    policy_class, conf, cfg, inner_val,
                    tag=f"{tag}_out{out_i}_hpo{i}_infold{fold_j}",
                )
                val_scores.append(r["mean_regret"])
            score = float(np.mean(val_scores))
            config_scores[i] = score
            running_best = min(running_best, score)
            if (i + 1) in SUB_BUDGETS:
                tuning_curve[i + 1] = running_best

        best_idx = int(np.nanargmin(config_scores))
        best_config = candidate_configs[best_idx]

        test_eval = _eval_config(
            policy_class, best_config, cfg, outer_test, tag=f"{tag}_out{out_i}_test"
        )

        results.append({
            "outer_fold": out_i,
            "outer_train_n": len(outer_train),
            "outer_test_n": len(outer_test),
            "best_config": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in best_config.items()
            },
            "best_inner_score": float(config_scores[best_idx]),
            "tuning_curve": tuning_curve,
            "test_mean_regret": test_eval["mean_regret"],
            "test_cvar95_regret": test_eval["cvar95_regret"],
            "test_regrets": test_eval["regrets"],
        })
        if ckpt_path is not None:
            save_json(ckpt_path, {"tag": tag, "K_out": K_out, "K_in": K_in,
                                   "n_trials": n_trials, "results": sorted(results, key=lambda r: r["outer_fold"])})
    return sorted(results, key=lambda r: r["outer_fold"])
