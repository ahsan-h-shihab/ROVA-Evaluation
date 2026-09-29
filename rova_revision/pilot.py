"""Timing pilot (plan section 2, E-3).

Times candidate evaluation for one outer fold, five candidates and one inner
fold of each tuning method, on NON-EXPERIMENTAL seeds 1000-1059 with tags
prefixed REV1_PILOT, in each environment at full size; times instance
evaluation and one full-size exhaustive DP solve; extrapolates to the unit
plan. Regret values are not reported.
"""
import time

import numpy as np

from dtcoa_val.experiments.nested_hpo import _eval_config
from dtcoa_val.experiments.search_space import sample_config

from . import config as C
from .archive import inner_folds, outer_folds
from .coupled_dp import solve_world
from .policies import ConstantLevel, LearnedCSPolicy, QLearnPolicy
from .training import LEARNED_SPACE, QLEARN_SPACE, fit_learned, fit_qlearn
from .units import CEW_TIER_METHODS, FROZEN_CAMPAIGNS, PREREG_BUDGET_METHODS, ROBUSTNESS_METHODS

N_CAND = 5
TIMED = ("static", "ewma", "rova", "clairvoyant", "rova_single", "static_ext", "ewma_ext")


def _time_frozen(method, cfg, tag, ival):
    cls, space = FROZEN_CAMPAIGNS[method]
    rng = np.random.default_rng(0)
    t0 = time.time()
    for i in range(N_CAND):
        _eval_config(cls, sample_config(space, rng), cfg, ival, tag=f"{tag}_hpo{i}")
    return (time.time() - t0) / N_CAND


def _time_trainable(method, cfg, tag, itrain, ival):
    space = LEARNED_SPACE if method == "learned" else QLEARN_SPACE
    fit = fit_learned if method == "learned" else fit_qlearn
    pol = LearnedCSPolicy if method == "learned" else QLearnPolicy
    rng = np.random.default_rng(0)
    cache = {}
    t0 = time.time()
    fit(sample_config(space, rng), cfg, itrain, f"{tag}_train", cache)
    t_first = time.time() - t0
    t0 = time.time()
    for i in range(N_CAND):
        model = fit(sample_config(space, rng), cfg, itrain, f"{tag}_train", cache)
        _eval_config(lambda m=model: pol(m), {}, cfg, ival, tag=f"{tag}_hpo{i}")
    return (time.time() - t0) / N_CAND, t_first


def _timing_key(m):
    if m in TIMED:
        return m
    base = PREREG_BUDGET_METHODS.get(m, m)
    if base.startswith(("fx_", "abl_", "rova")):
        return "rova"
    return "ewma" if base.startswith("ewma") else "static"


def run_pilot(profile, units):
    folds = outer_folds(profile.seeds, profile.K_out)
    outer_train, outer_test = folds[0]
    itrain, ival = inner_folds(outer_train, profile.K_in, 0)[0]
    per_env = {}
    for env in C.ENV_NAMES:
        cfg = C.cell_cfg(profile, env, 3.5)
        tag = f"{profile.tag_prefix}__{env}__pilot"
        row = {m: {"sec_per_candidate_fold": _time_frozen(m, cfg, f"{tag}_{m}", ival)} for m in TIMED}
        for m in ("learned", "qlearn"):
            per, first = _time_trainable(m, cfg, f"{tag}_{m}", list(itrain), ival)
            row[m] = {"sec_per_candidate_fold": per, "sec_first_fit_incl_cache": first}
        t0 = time.time()
        _eval_config(lambda: ConstantLevel(5), {}, cfg, list(outer_test), tag=f"{tag}_eval")
        row["_sec_per_instance"] = (time.time() - t0) / len(outer_test)
        per_env[env] = row
    t0 = time.time()
    dp_rec = solve_world(C.cell_cfg(profile, C.DP_FULL_SIZE_ENV, 3.5), 1000, f"{profile.tag_prefix}_DP_pilot")
    dp_sec = time.time() - t0
    K, n_full = dp_rec["tau"] + 1, dp_rec["N"]

    def dp_scale(n):
        return (K ** n * 6 ** n) / (K ** n_full * 6 ** n_full)

    k_out, k_in, n_tr = profile.K_out, profile.K_in, profile.n_trials
    n_seeds = len(profile.seeds)
    estimate = {}
    for u in units:
        kind, p = u["kind"], u["params"]
        env = p.get("env", C.ENV_NAMES[0])
        row = per_env.get(env, per_env[C.ENV_NAMES[0]])
        inst = row["_sec_per_instance"]
        if kind in ("campaign_frozen", "campaign_sens", "campaign_advt", "campaign_tpe"):
            sec = k_out * n_tr * k_in * row[_timing_key(p["method"])]["sec_per_candidate_fold"] + n_seeds * inst
        elif kind == "campaign_prereg_budget":
            # every candidate is evaluated on all outer-training seeds whatever K_in is
            sec = k_out * profile.prereg_n_trials * k_in * row[_timing_key(p["method"])]["sec_per_candidate_fold"]
        elif kind == "campaign_trainable":
            m = p["method"]
            sec = k_out * (n_tr * k_in * row[m]["sec_per_candidate_fold"]
                           + (k_in + 1) * row[m]["sec_first_fit_incl_cache"])
            if m == "qlearn":
                sec += k_out * 3 * row[m]["sec_first_fit_incl_cache"]
        elif kind == "anchor_eval":
            sec = 8 * n_seeds * inst
        elif kind in ("eval_cew", "eval_cew_b", "eval_cew_c"):
            tier = {"eval_cew": "A", "eval_cew_b": "B", "eval_cew_c": "C"}[kind]
            sec = len(CEW_TIER_METHODS[tier]) * n_seeds * inst
        elif kind == "eval_adv":
            sec = 2 * len(ROBUSTNESS_METHODS) * n_seeds * inst
        elif kind == "eval_advt_cew":
            sec = 5 * n_seeds * inst
        elif kind == "eval_sens_cew":
            sec = (5 + len(ROBUSTNESS_METHODS)) * n_seeds * inst
        elif kind == "eval_shift_cew":
            sec = len(ROBUSTNESS_METHODS) * k_out * n_seeds * per_env[C.SHIFT_EVAL_ENV]["_sec_per_instance"]
        elif kind == "sweep":
            sec = 20 * n_seeds * inst
        elif kind == "dp_reduced":
            sec = n_seeds * (dp_sec * dp_scale(p["n"]) + 0.5)
        elif kind == "dp_cew":
            sec = n_seeds * dp_sec
        else:
            sec = 60.0
        estimate.setdefault(u["tier"], 0.0)
        estimate[u["tier"]] += sec
    core_hours = {t: s / 3600.0 for t, s in sorted(estimate.items())}
    return {"note": "timing only; non-experimental seeds; regret values not reported",
            "per_env_timings": per_env, "dp_full_size_sec_per_world": dp_sec,
            "estimated_core_hours_by_tier": core_hours,
            "estimated_total_core_hours": sum(core_hours.values()),
            "estimated_wall_hours_15_workers": sum(core_hours.values()) / 15.0,
            "note_wall": "lower bound: the longest single units (full-size DP, protocol-budget campaigns) "
                         "can extend the tail"}
