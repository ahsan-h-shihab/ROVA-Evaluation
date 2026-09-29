"""Random-search nested cross-validation that records every candidate (plan R14).

Reproduces ``dtcoa_val.experiments.nested_hpo.nested_cv_hpo`` exactly (same
candidate stream from ``default_rng(seed0)`` drawn fold by fold, same outer and
inner splits, same validation and test tags, same ``nanargmin`` selection) and
additionally records every candidate's inner-validation score, so tuning
curves can be read at any sub-budget (the protocol's 25/50/100/200). All
candidates are drawn before the first fold, so a run resumed from its per-fold
checkpoint proposes the same candidates. Equality with the frozen function is
checked by a unit test.
"""
import json
import os

import numpy as np

from dtcoa_val.experiments.nested_hpo import _eval_config, kfold_split
from dtcoa_val.experiments.search_space import sample_config


def nested_cv_hpo_record(policy_class, search_space, cfg, all_seeds, K_out, K_in, n_trials, sub_budgets,
                         seed0=0, tag="", checkpoint_path=None):
    master_rng = np.random.default_rng(seed0)
    outer = kfold_split(all_seeds, K_out, np.random.default_rng(seed0))
    candidates = [[sample_config(search_space, master_rng) for _ in range(n_trials)] for _ in range(K_out)]
    results = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            results = json.load(f)["results"]
    done = {r["outer_fold"] for r in results}
    for out_i, (outer_train, outer_test) in enumerate(outer):
        if out_i in done:
            continue
        inner = kfold_split(outer_train, K_in, np.random.default_rng(seed0 * 7 + out_i))
        scores = np.full(n_trials, np.nan)
        best, curve = np.inf, {}
        for i, conf in enumerate(candidates[out_i]):
            vals = [_eval_config(policy_class, conf, cfg, iv, tag=f"{tag}_out{out_i}_hpo{i}_infold{j}")["mean_regret"]
                    for j, (_it, iv) in enumerate(inner)]
            scores[i] = float(np.mean(vals))
            best = min(best, scores[i])
            if (i + 1) in sub_budgets:
                curve[i + 1] = best
        bi = int(np.nanargmin(scores))
        bc = candidates[out_i][bi]
        test = _eval_config(policy_class, bc, cfg, outer_test, tag=f"{tag}_out{out_i}_test")
        results.append({
            "outer_fold": out_i, "outer_train_n": len(outer_train), "outer_test_n": len(outer_test),
            "best_config": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in bc.items()},
            "best_index": bi, "best_inner_score": float(scores[bi]), "inner_scores": scores.tolist(),
            "tuning_curve": curve, "test_mean_regret": test["mean_regret"],
            "test_cvar95_regret": test["cvar95_regret"], "test_regrets": test["regrets"],
        })
        results.sort(key=lambda r: r["outer_fold"])
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            tmp = checkpoint_path + ".tmp"
            with open(tmp, "w", newline="\n") as f:
                json.dump({"tag": tag, "K_out": K_out, "K_in": K_in, "n_trials": n_trials, "results": results}, f)
            os.replace(tmp, checkpoint_path)
    return sorted(results, key=lambda r: r["outer_fold"])
