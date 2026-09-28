"""Work units: enumeration (plan sections 4 and 8) and execution.

Stages: 1 gate; 2 tuning campaigns and evaluations that need no new tuning;
3 evaluations of tuned configurations; 4 decision-cost benchmark (alone);
5 analysis. Unit identifiers are built only by ``uid()`` so producers and
consumers agree.
"""
import json
import os

from dtcoa_val.baselines.baselines import EWMA, OracleLite, StaticThreshold
from dtcoa_val.dtcoa_tunable import DTCOATunable
from dtcoa_val.experiments.nested_hpo import _eval_config, nested_cv_hpo
from dtcoa_val.experiments.search_space import (BAND_DIM, BAND_HIGH, BAND_LOW, CLAIRVOYANT_SEARCH_SPACE,
                                                EWMA_SEARCH_SPACE, ROVA_SEARCH_SPACE,
                                                STATIC_THRESHOLD_SEARCH_SPACE)

from . import config as C
from .archive import load_archived, outer_folds
from .coupled_dp import solve_world
from .gate import run_gate
from .instrument import evaluate_folds
from .overhead import benchmark
from .policies import (CappedEWMA, CappedStaticThreshold, ConstantLevel, LearnedCSPolicy,
                       QLearnPolicy, rova_single)
from .tpe import nested_cv_tpe
from .training import nested_cv_trainable

# --------------------------------------------------------------------------- #
# Search spaces added by the revision (plan R8, R12)
# --------------------------------------------------------------------------- #
BAND = ("band_vector", BAND_DIM, BAND_LOW, BAND_HIGH)
CAP = ("int_uniform", 0, 24)
STATIC_EXT_SPACE = {"thr": BAND, "cap_code": CAP}
EWMA_EXT_SPACE = {"beta": EWMA_SEARCH_SPACE["beta"], "thr": BAND, "cap_code": CAP}
ROVA_SINGLE_SPACE = {}
for _k, _v in ROVA_SEARCH_SPACE.items():
    ROVA_SINGLE_SPACE["b" if _k == "bands" else _k] = ("uniform", BAND_LOW, BAND_HIGH) if _k == "bands" else _v

FROZEN_CAMPAIGNS = {
    "static": (StaticThreshold, STATIC_THRESHOLD_SEARCH_SPACE),
    "ewma": (EWMA, EWMA_SEARCH_SPACE),
    "rova": (DTCOATunable, ROVA_SEARCH_SPACE),
    "static_ext": (CappedStaticThreshold, STATIC_EXT_SPACE),
    "ewma_ext": (CappedEWMA, EWMA_EXT_SPACE),
    "clairvoyant": (OracleLite, CLAIRVOYANT_SEARCH_SPACE),
    "rova_single": (rova_single, ROVA_SINGLE_SPACE),
}
SWEEP_GRID = [0.0, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 5.0]
CLEAN_METHODS = ["static", "ewma", "rova", "A0", "A5", "learned", "qlearn", "static_ext", "ewma_ext",
                 "clairvoyant", "rova_single", "static_tpe", "ewma_tpe", "rova_tpe"]
ROBUSTNESS_METHODS = ["static", "ewma", "rova", "A0", "A5", "learned", "qlearn", "static_ext", "ewma_ext",
                      "clairvoyant", "rova_single"]


def uid(stage, kind, **p):
    parts = [f"S{stage}", kind] + [str(p[k]) for k in sorted(p)]
    return "__".join(parts)


def unit_dir(out_dir, u):
    return os.path.join(out_dir, "units", u)


def load_result(out_dir, u):
    path = os.path.join(unit_dir(out_dir, u), "result.json")
    if not os.path.exists(os.path.join(unit_dir(out_dir, u), "done.json")) or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def enumerate_units(profile, tiers=("A", "B")):
    units = []

    def add(stage, tier, kind, cost, **p):
        units.append({"uid": uid(stage, kind, **p), "stage": stage, "tier": tier,
                      "kind": kind, "cost": cost, "params": p})

    add(1, "A", "gate", 1)
    for env in C.ENV_NAMES:
        n = C.cell_cfg(profile, env, 1.0).get("n_agents", 1)
        for c in profile.grid:
            add(2, "A", "anchor_eval", 2 * n, env=env, c=c)
            add(2, "A", "campaign_trainable", 60 * n, method="learned", env=env, c=c)
            add(2, "A", "campaign_trainable", 120 * n, method="qlearn", env=env, c=c)
            for m in ("static_ext", "ewma_ext", "clairvoyant"):
                add(2, "A", "campaign_frozen", 15 * n, method=m, env=env, c=c)
            for pert in C.PERTURBATIONS:
                for m in C.ARCHIVED_METHODS:
                    add(2, "A", "campaign_sens", 20 * n, pert=pert, method=m, env=env, c=c)
            add(2, "B", "campaign_frozen", 20 * n, method="rova_single", env=env, c=c)
            for m in C.ARCHIVED_METHODS:
                add(2, "B", "campaign_tpe", 22 * n, method=m, env=env, c=c)
            for adv in C.ADVERSARY_PROFILES:
                if C.adversary_valid(adv, env, profile):
                    for m in C.ARCHIVED_METHODS:
                        add(2, "B", "campaign_advt", 20 * n, adv=adv, method=m, env=env, c=c)
            add(2, "B", "sweep", 3 * n, env=env, c=c)
            add(3, "A", "eval_cew", 6 * n, env=env, c=c)
            for pert in C.PERTURBATIONS:
                add(3, "A", "eval_sens_cew", 6 * n, pert=pert, env=env, c=c)
            for adv in C.ADVERSARY_PROFILES:
                if C.adversary_valid(adv, env, profile):
                    add(3, "A", "eval_adv", 8 * n, adv=adv, env=env, c=c)
                    add(3, "B", "eval_advt_cew", 2 * n, adv=adv, env=env, c=c)
    for env, sizes in profile.dp_sizes.items():
        for n in sizes:
            for c in profile.grid:
                add(2, "A", "dp_reduced", 3 ** n, env=env, n=n, c=c)
    for c in profile.grid:
        add(2, "A", "dp_cew", 3 ** 6, c=c)
    add(4, "A", "overhead", 1)
    add(5, "A", "analysis", 1)
    return [u for u in units if u["tier"] in tiers]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fold_seeds(profile):
    return [(k, list(f[1])) for k, f in enumerate(outer_folds(profile.seeds, profile.K_out))]


def _config_maker(cls, per_fold_configs):
    return lambda k: (lambda: cls(**dict(per_fold_configs[k])))


def _model_maker(cls, per_fold_models):
    return lambda k: (lambda: cls(per_fold_models[k]))


def _constant_maker(level):
    return lambda k: (lambda: ConstantLevel(level))


def method_makers(profile, out_dir, env, c, names):
    """Per-fold policy factories for clean-tuned methods; missing ones are listed."""
    makers, missing = {}, []
    for name in names:
        try:
            if name in ("A0", "A5"):
                makers[name] = _constant_maker(int(name[1]))
            elif name in C.ARCHIVED_METHODS:
                arch = load_archived(env, c, name)
                makers[name] = _config_maker(FROZEN_CAMPAIGNS[name][0],
                                             [arch[k]["best_config"] for k in range(profile.K_out)])
            elif name in ("learned", "qlearn"):
                res = load_result(out_dir, uid(2, "campaign_trainable", method=name, env=env, c=c))
                if res is None:
                    missing.append(name)
                    continue
                cls = LearnedCSPolicy if name == "learned" else QLearnPolicy
                makers[name] = _model_maker(cls, [f["model"] for f in res["results"]])
            elif name.endswith("_tpe"):
                base = name[:-4]
                res = load_result(out_dir, uid(2, "campaign_tpe", method=base, env=env, c=c))
                if res is None:
                    missing.append(name)
                    continue
                makers[name] = _config_maker(FROZEN_CAMPAIGNS[base][0], [f["best_config"] for f in res["results"]])
            else:
                res = load_result(out_dir, uid(2, "campaign_frozen", method=name, env=env, c=c))
                if res is None:
                    missing.append(name)
                    continue
                makers[name] = _config_maker(FROZEN_CAMPAIGNS[name][0], [f["best_config"] for f in res["results"]])
        except FileNotFoundError:
            missing.append(name)
    return makers, missing


def _evaluate_all(profile, cfg, makers, tag_for_fold):
    fs = _fold_seeds(profile)
    return {name: evaluate_folds(cfg, mk, fs, tag_for_fold) for name, mk in makers.items()}


def _frozen_campaign(profile, method, cfg, tag):
    cls, space = FROZEN_CAMPAIGNS[method]
    res = nested_cv_hpo(cls, space, cfg, list(profile.seeds), K_out=profile.K_out, K_in=profile.K_in,
                        n_trials=profile.n_trials, seed0=0, tag=tag, checkpoint_dir=None)
    return {"tag": tag, "method": method, "results": res}


def _curve_at(fold, n):
    curve = fold.get("tuning_curve", {})
    return curve.get(n, curve.get(str(n)))


# --------------------------------------------------------------------------- #
# Executors: (profile, params, unit_dir, out_dir) -> JSON-serialisable result
# --------------------------------------------------------------------------- #
def ex_gate(profile, p, udir, out_dir):
    return run_gate(profile)


def ex_anchor_eval(profile, p, udir, out_dir):
    env, c = p["env"], p["c"]
    cfg = C.cell_cfg(profile, env, c)
    fs = _fold_seeds(profile)
    out = {}
    for level in (0, 5):
        mk = _constant_maker(level)
        rec = {"own": evaluate_folds(cfg, mk, fs, lambda k, L=level: C.own_world_tag(profile, env, c, f"always{L}", k))}
        for m in C.ARCHIVED_METHODS:
            rec[f"on_{m}_worlds"] = evaluate_folds(cfg, mk, fs, lambda k, m=m: f"{C.archived_tag(env, c, m)}_out{k}_test")
        out[f"A{level}"] = rec
    return out


def ex_campaign_trainable(profile, p, udir, out_dir):
    env, c, m = p["env"], p["c"], p["method"]
    tag = C.campaign_tag(profile, env, c, m)
    res = nested_cv_trainable(m, C.cell_cfg(profile, env, c), list(profile.seeds), tag, profile,
                              checkpoint_path=os.path.join(udir, "checkpoint.json"))
    return {"tag": tag, "method": m, "results": res}


def ex_campaign_frozen(profile, p, udir, out_dir):
    env, c, m = p["env"], p["c"], p["method"]
    return _frozen_campaign(profile, m, C.cell_cfg(profile, env, c), C.campaign_tag(profile, env, c, m))


def ex_campaign_sens(profile, p, udir, out_dir):
    env, c, m, pert = p["env"], p["c"], p["method"], p["pert"]
    cfg = C.cell_cfg(profile, env, c, **C.PERTURBATIONS[pert])
    out = _frozen_campaign(profile, m, cfg, C.sens_campaign_tag(profile, pert, env, c, m))
    out["perturbation"] = pert
    return out


def ex_campaign_advt(profile, p, udir, out_dir):
    env, c, m, adv = p["env"], p["c"], p["method"], p["adv"]
    cfg = C.cell_cfg(profile, env, c, adversary=C.ADVERSARY_PROFILES[adv])
    out = _frozen_campaign(profile, m, cfg, C.advt_campaign_tag(profile, adv, env, c, m))
    out["adversary"] = adv
    return out


def ex_campaign_tpe(profile, p, udir, out_dir):
    env, c, m = p["env"], p["c"], p["method"]
    cls, space = FROZEN_CAMPAIGNS[m]
    tag = C.archived_tag(env, c, m)
    res = nested_cv_tpe(cls, space, C.cell_cfg(profile, env, c), list(profile.seeds), tag, profile,
                        checkpoint_path=os.path.join(udir, "checkpoint.json"))
    check = None
    if profile.scientific:
        arch = load_archived(env, c, m)
        check = []
        for k in range(profile.K_out):
            a, b = _curve_at(arch[k], profile.tpe_startup), _curve_at(res[k], profile.tpe_startup)
            check.append({"outer_fold": k, "archived": a, "rerun": b,
                          "abs_diff": None if a is None or b is None else abs(a - b),
                          "ok": a is not None and b is not None and abs(a - b) < 1e-9})
    return {"tag": tag, "method": m, "results": res, "startup_reproduction": check}


def ex_sweep(profile, p, udir, out_dir):
    env, c = p["env"], p["c"]
    cfg = C.cell_cfg(profile, env, c)
    arch = load_archived(env, c, "rova")
    points = [("tuned", None, None), ("both_zero", "both", 0.0)]
    points += [("prop", "prop_scale", g) for g in SWEEP_GRID] + [("stab", "stab_scale", g) for g in SWEEP_GRID]
    rows, repro = [], []
    for k, seeds in _fold_seeds(profile):
        conf = dict(arch[k]["best_config"])
        tag = f"{C.archived_tag(env, c, 'rova')}_out{k}_test"
        for label, key, val in points:
            cc = dict(conf)
            if key == "both":
                cc["prop_scale"], cc["stab_scale"] = 0.0, 0.0
            elif key:
                cc[key] = val
            ev = _eval_config(DTCOATunable, cc, cfg, seeds, tag)
            rows.append({"outer_fold": k, "point": label, "param": key, "value": val,
                         "prop_scale": cc["prop_scale"], "stab_scale": cc["stab_scale"],
                         "mean_regret": ev["mean_regret"], "regrets": ev["regrets"], "seeds": seeds})
            if label == "tuned" and profile.scientific:
                d = max(abs(a - b) for a, b in zip(ev["regrets"], arch[k]["test_regrets"]))
                repro.append({"outer_fold": k, "max_abs_diff": d, "ok": d < 1e-9})
    return {"rows": rows, "reproduction": repro, "grid": SWEEP_GRID}


def ex_eval_cew(profile, p, udir, out_dir):
    env, c = p["env"], p["c"]
    makers, missing = method_makers(profile, out_dir, env, c, CLEAN_METHODS)
    recs = _evaluate_all(profile, C.cell_cfg(profile, env, c), makers, lambda k: C.cew_tag(profile, env, c, k))
    return {"records": recs, "missing": missing}


def ex_eval_adv(profile, p, udir, out_dir):
    env, c, adv = p["env"], p["c"], p["adv"]
    makers, missing = method_makers(profile, out_dir, env, c, ROBUSTNESS_METHODS)
    tag_fn = lambda k: C.adv_tag(profile, adv, env, c, k)
    attacked = _evaluate_all(profile, C.cell_cfg(profile, env, c, adversary=C.ADVERSARY_PROFILES[adv]), makers, tag_fn)
    clean = _evaluate_all(profile, C.cell_cfg(profile, env, c), makers, tag_fn)
    return {"attacked": attacked, "clean": clean, "missing": missing}


def ex_eval_advt_cew(profile, p, udir, out_dir):
    env, c, adv = p["env"], p["c"], p["adv"]
    makers, missing = {"A0": _constant_maker(0), "A5": _constant_maker(5)}, []
    for m in C.ARCHIVED_METHODS:
        res = load_result(out_dir, uid(2, "campaign_advt", adv=adv, method=m, env=env, c=c))
        if res is None:
            missing.append(m)
            continue
        makers[m] = _config_maker(FROZEN_CAMPAIGNS[m][0], [f["best_config"] for f in res["results"]])
    cfg = C.cell_cfg(profile, env, c, adversary=C.ADVERSARY_PROFILES[adv])
    recs = _evaluate_all(profile, cfg, makers, lambda k: C.advt_cew_tag(profile, adv, env, c, k))
    return {"records": recs, "missing": missing}


def ex_eval_sens_cew(profile, p, udir, out_dir):
    env, c, pert = p["env"], p["c"], p["pert"]
    makers, missing = {"A0": _constant_maker(0), "A5": _constant_maker(5)}, []
    for m in C.ARCHIVED_METHODS:
        res = load_result(out_dir, uid(2, "campaign_sens", pert=pert, method=m, env=env, c=c))
        if res is None:
            missing.append(m)
            continue
        makers[m] = _config_maker(FROZEN_CAMPAIGNS[m][0], [f["best_config"] for f in res["results"]])
    clean, clean_missing = method_makers(profile, out_dir, env, c, [m for m in ROBUSTNESS_METHODS if m not in ("A0", "A5")])
    for name, mk in clean.items():
        makers[f"clean:{name}"] = mk
    missing += [f"clean:{m}" for m in clean_missing]
    cfg = C.cell_cfg(profile, env, c, **C.PERTURBATIONS[pert])
    recs = _evaluate_all(profile, cfg, makers, lambda k: C.sens_cew_tag(profile, pert, env, c, k))
    return {"records": recs, "missing": missing}


def ex_dp_reduced(profile, p, udir, out_dir):
    env, n, c = p["env"], p["n"], p["c"]
    cfg = C.cell_cfg(profile, env, c, n_agents=n)
    return {"records": [solve_world(cfg, s, C.dp_tag(profile, env, n, c)) for s in profile.seeds]}


def ex_dp_cew(profile, p, udir, out_dir):
    env, c = C.DP_FULL_SIZE_ENV, p["c"]
    cfg = C.cell_cfg(profile, env, c)
    recs = []
    for k, seeds in _fold_seeds(profile):
        for s in seeds:
            rec = solve_world(cfg, s, C.cew_tag(profile, env, c, k))
            rec["outer_fold"] = k
            recs.append(rec)
    return {"env": env, "c": c, "records": recs}


def ex_overhead(profile, p, udir, out_dir):
    env, c = C.ENV_NAMES[0], profile.grid[-1]
    makers, missing = method_makers(profile, out_dir, env, c,
                                    ["static", "ewma", "rova", "A0", "A5", "learned", "qlearn", "static_ext",
                                     "ewma_ext", "clairvoyant", "rova_single"])
    policies = {name: mk(0) for name, mk in makers.items()}
    policies["rova_default"] = lambda: DTCOATunable()
    res = benchmark(policies, list(profile.overhead_n), profile.overhead_reps,
                    cfg_factory=lambda N: C.cell_cfg(profile, env, c, n_agents=N),
                    needs_truth=("clairvoyant",))
    res.update({"configs_from": {"env": env, "c": c, "outer_fold": 0}, "missing": missing})
    return res


def ex_analysis(profile, p, udir, out_dir):
    from .analysis import run_analysis
    return run_analysis(profile, out_dir)


EXECUTORS = {
    "gate": ex_gate, "anchor_eval": ex_anchor_eval, "campaign_trainable": ex_campaign_trainable,
    "campaign_frozen": ex_campaign_frozen, "campaign_sens": ex_campaign_sens,
    "campaign_advt": ex_campaign_advt, "campaign_tpe": ex_campaign_tpe, "sweep": ex_sweep,
    "eval_cew": ex_eval_cew, "eval_adv": ex_eval_adv, "eval_advt_cew": ex_eval_advt_cew,
    "eval_sens_cew": ex_eval_sens_cew, "dp_reduced": ex_dp_reduced, "dp_cew": ex_dp_cew,
    "overhead": ex_overhead, "analysis": ex_analysis,
}
RESUMABLE_KINDS = {"campaign_trainable", "campaign_tpe"}
