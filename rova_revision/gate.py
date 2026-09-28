"""Gate R0: the execution environment must reproduce archived MAIN-001 results.

Replays outer-fold-0 test evaluations of archived tuning units with their
archived configurations and tags, and recomputes the archived BCa intervals.
Any mismatch fails the gate; a failed gate invalidates the whole run.
"""
from dtcoa_val.analysis.stats import describe
from dtcoa_val.baselines.baselines import EWMA, StaticThreshold
from dtcoa_val.dtcoa_tunable import DTCOATunable
from dtcoa_val.experiments.environments import cfg_for
from dtcoa_val.experiments.nested_hpo import _eval_config

from .archive import load_archived, load_stats_summary, outer_folds
from .config import ARCHIVED_METHODS, ENV_NAMES, GRID, archived_tag

TOL = 1e-9
ARCHIVED_CLASSES = {"static": StaticThreshold, "ewma": EWMA, "rova": DTCOATunable}


def replay_fold0(env, c, method):
    fold0 = load_archived(env, c, method)[0]
    seeds = outer_folds(range(60), 5)[0][1]
    ev = _eval_config(ARCHIVED_CLASSES[method], dict(fold0["best_config"]), cfg_for(env, c), seeds,
                      tag=f"{archived_tag(env, c, method)}_out0_test")
    per_seed = max(abs(a - b) for a, b in zip(ev["regrets"], fold0["test_regrets"]))
    diff = abs(ev["mean_regret"] - fold0["test_mean_regret"])
    return {"env": env, "c": c, "method": method, "replayed_mean": ev["mean_regret"],
            "archived_mean": fold0["test_mean_regret"], "abs_diff": diff,
            "max_per_seed_abs_diff": per_seed, "ok": bool(diff < TOL and per_seed < TOL)}


def bca_reproduction(env, c, method, summary):
    pooled = [x for fold in load_archived(env, c, method) for x in fold["test_regrets"]]
    d = describe(pooled)
    ref = summary["cells"][env][str(c)]["regret_desc"][method]
    ok = all(d[k] == ref[k] for k in ("mean", "ci_lo", "ci_hi"))
    return {"env": env, "c": c, "method": method, "ok": bool(ok),
            "recomputed": {k: d[k] for k in ("mean", "ci_lo", "ci_hi")},
            "archived": {k: ref[k] for k in ("mean", "ci_lo", "ci_hi")}}


def run_gate(profile):
    units = [(e, c, m) for e in ENV_NAMES for c in GRID for m in ARCHIVED_METHODS]
    if not profile.scientific:
        units = units[:1]
    summary = load_stats_summary()
    replay = [replay_fold0(*u) for u in units]
    bca = [bca_reproduction(*u, summary) for u in units]
    passed = all(r["ok"] for r in replay) and all(b["ok"] for b in bca)
    return {"passed": bool(passed), "n_units": len(units), "tolerance": TOL,
            "replay": replay, "bca": bca, "scope": "all 45 units" if profile.scientific else "1 unit (smoke)"}
