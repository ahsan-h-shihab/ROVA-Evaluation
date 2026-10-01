"""A1: the REV-001 TPE tuning run (``tpe.py``, plan R13) continued to more candidates
(docs/REVISION_002_ANALYSIS_PLAN.md, section 2.1).

The loop is ``nested_cv_tpe`` with three differences, none of which changes the
first ``base.n_trials`` candidates:
- it runs ``n_total`` candidates per outer fold;
- the frozen startup draw stays at ``base.n_trials`` candidates per fold (the
  frozen code sizes that draw by the budget; resizing it would change the startup
  candidates of every fold after the first);
- it records every inner score, the selection at each budget in
  ``select_budgets`` (``nanargmin`` of the first b scores, ties to the earliest),
  and the own-world test evaluation of the selection at the reproduction budget.
"""
import json
import os

import numpy as np

from dtcoa_val.experiments.nested_hpo import _eval_config
from dtcoa_val.experiments.search_space import sample_config
from dtcoa_val.utils import make_rng, trial_seed

from .archive import inner_folds, outer_folds
from .tpe import TPESampler


def _plain(conf):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in conf.items()}


def nested_cv_tpe_extended(policy_class, space, cfg, seeds, tag, a1prof, checkpoint_path=None):
    base = a1prof.base
    master_rng = np.random.default_rng(0)
    folds = outer_folds(seeds, base.K_out)
    frozen = [[sample_config(space, master_rng) for _ in range(base.n_trials)]
              for _ in range(base.K_out)]
    results = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ck = json.load(f)
        if ck.get("tag") != tag or ck.get("n_total") != a1prof.n_total:
            raise RuntimeError(f"checkpoint {checkpoint_path} belongs to another run")
        results = ck["results"]
    done = {r["outer_fold"] for r in results}
    curve_at = set(a1prof.curve_budgets) | {base.tpe_startup}
    for out_i, (outer_train, outer_test) in enumerate(folds):
        if out_i in done:
            continue
        ifolds = inner_folds(outer_train, base.K_in, out_i)
        sampler = TPESampler(space, make_rng(trial_seed(out_i, 0, f"{tag}__tpe")))
        history, curve, best = [], {}, np.inf
        for i in range(a1prof.n_total):
            conf = frozen[out_i][i] if i < base.tpe_startup else sampler.propose(history)
            vals = [_eval_config(policy_class, conf, cfg, ival,
                                 tag=f"{tag}_out{out_i}_hpo{i}_infold{j}")["mean_regret"]
                    for j, (_it, ival) in enumerate(ifolds)]
            score = float(np.mean(vals))
            history.append((conf, score))
            best = min(best, score)
            if (i + 1) in curve_at:
                curve[i + 1] = best
        scores = np.array([s for _, s in history])
        selections = {}
        for b in a1prof.select_budgets:
            idx = int(np.nanargmin(scores[:b]))
            selections[str(b)] = {"best_index": idx, "best_config": _plain(history[idx][0]),
                                  "best_inner_score": float(scores[idx])}
        rb = a1prof.b_repro
        test = _eval_config(policy_class, history[selections[str(rb)]["best_index"]][0], cfg, outer_test,
                            tag=f"{tag}_out{out_i}_test")
        results.append({
            "outer_fold": out_i,
            "inner_scores": scores.tolist(),
            "tuning_curve": curve,
            "selections": selections,
            "repro_budget": rb,
            "repro_test_mean_regret": test["mean_regret"],
            "repro_test_regrets": test["regrets"],
        })
        results.sort(key=lambda r: r["outer_fold"])
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            tmp = checkpoint_path + ".tmp"
            with open(tmp, "w", newline="\n") as f:
                json.dump({"tag": tag, "n_total": a1prof.n_total, "results": results}, f)
            os.replace(tmp, checkpoint_path)
    return sorted(results, key=lambda r: r["outer_fold"])
