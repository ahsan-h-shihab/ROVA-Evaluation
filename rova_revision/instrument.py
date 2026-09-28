"""Evaluation through the frozen runner with per-step oracle instrumentation.

``run_trial`` accepts an ``auth`` oracle and calls ``auth.regret(a, r, h)`` once
per agent-step. ``RecordingOracle`` returns exactly the frozen value and also
accumulates the policy's loss, the oracle's loss and both action histograms,
so no part of the frozen dynamics is re-implemented.
"""
import numpy as np

from dtcoa_val.experiments.runner import run_trial
from dtcoa_val.oracles.oracles import AuthorizationOracle


class RecordingOracle(AuthorizationOracle):
    def __init__(self, H=1.0, U=0.3):
        super().__init__(H=H, U=U)
        self.n = 0
        self.sum_loss = 0.0
        self.sum_opt = 0.0
        self.astar_counts = np.zeros(6, int)
        self.a_counts = np.zeros(6, int)

    def regret(self, a, r, h):
        a_star, l_star = self.optimal(r, h)
        loss = self.loss(a, r, h)
        self.n += 1
        self.sum_loss += loss
        self.sum_opt += l_star
        self.astar_counts[a_star] += 1
        self.a_counts[int(a)] += 1
        return float(loss - l_star)


def evaluate_trial(cfg, policy, seed, tag):
    """One scenario instance through the frozen runner; returns a flat record."""
    orc = RecordingOracle(H=cfg.get("H", 1.0), U=cfg.get("U", 0.3))
    out = run_trial(cfg, policy, seed=seed, tag=tag, auth=orc)
    return {
        "seed": int(seed),
        "regret": float(out["regret"]),
        "harm_rate": float(out["harm_rate"]),
        "overrestrict": float(out["overrestrict"]),
        "unauth_unsafe_rate": float(out["unauth_unsafe_rate"]),
        "sysfail_rate": float(out["sysfail_rate"]),
        "esc_rate": float(out["esc_rate"]),
        "sum_loss": float(orc.sum_loss),
        "sum_opt": float(orc.sum_opt),
        "n_agent_steps": int(orc.n),
        "astar_counts": orc.astar_counts.tolist(),
        "a_counts": orc.a_counts.tolist(),
    }


def evaluate_folds(cfg, make_policy_for_fold, fold_seeds, tag_for_fold):
    """Evaluate a method fold by fold.

    make_policy_for_fold(k) -> zero-argument factory of a fresh policy
    fold_seeds: list of (k, seeds)
    tag_for_fold(k) -> world tag for fold k
    """
    records = []
    for k, seeds in fold_seeds:
        factory = make_policy_for_fold(k)
        tag = tag_for_fold(k)
        for s in seeds:
            rec = evaluate_trial(cfg, factory(), s, tag)
            rec["outer_fold"] = int(k)
            rec["tag"] = tag
            records.append(rec)
    return records
