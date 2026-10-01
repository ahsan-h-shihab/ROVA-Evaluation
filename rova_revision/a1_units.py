"""A1 work units (docs/REVISION_002_ANALYSIS_PLAN.md, section 2).

Stages: 1 gate (R0 replay + REV-001 input verification); 2 the 45 extended TPE
tuning campaigns, each checking RC1, RC2, RC4 and RC5 against REV-001; 3 the 15
common-world evaluations, each checking RC3; 4 analysis. REV-001 outputs are read
from the directory recorded in ``<out>/a1_inputs.json`` and are never written.
"""
import json
import os

from . import config as C
from .a1_config import A1_METHODS, method_name
from .clustered import verify_rev001_dir
from .gate import run_gate
from .instrument import evaluate_folds
from .tpe_ext import nested_cv_tpe_extended
from .units import FROZEN_CAMPAIGNS, _config_maker, _curve_at, _fold_seeds, load_archived, load_result, uid

INPUTS_FILE = "a1_inputs.json"


def a1_uid(stage, kind, **p):
    return "A1__" + uid(stage, kind, **p)


def enumerate_a1_units(a1prof):
    units = []

    def add(stage, kind, cost, **p):
        units.append({"uid": a1_uid(stage, kind, **p), "stage": stage, "tier": "A1", "kind": kind,
                      "cost": cost, "params": p})

    add(1, "a1_gate", 1)
    for env in C.ENV_NAMES:
        for c in a1prof.base.grid:
            for m in A1_METHODS:
                add(2, "a1_campaign", 40 if m == "rova" else 26, method=m, env=env, c=c)
            add(3, "a1_eval", 1, env=env, c=c)
    add(4, "a1_analysis", 1)
    return units


def rev001_dir(out_dir):
    with open(os.path.join(out_dir, INPUTS_FILE)) as f:
        return json.load(f)["rev001_dir"]


def load_a1(out_dir, u):
    return load_result(out_dir, u)


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #
def ex_a1_gate(a1prof, p, udir, out_dir):
    base = a1prof.base
    r0 = run_gate(base)
    exact = all(r["abs_diff"] == 0.0 and r["max_per_seed_abs_diff"] == 0.0 for r in r0["replay"])
    rdir = rev001_dir(out_dir)
    inputs = verify_rev001_dir(rdir)
    rev_verify = None
    if a1prof.scientific:
        from .verify_outputs import verify
        rev_verify = verify(base, rdir)
        inputs_ok = inputs["passed"] and rev_verify["passed"]
    else:  # plumbing check against a REV-001 smoke output: entries only, no fixed fingerprint
        inputs_ok = inputs["entries"] > 0 and not inputs["failed_entries"]
    return {"passed": bool(r0["passed"] and exact and inputs_ok), "r0": r0, "r0_exact_zero": bool(exact),
            "rev001_inputs": inputs, "rev001_verify_outputs": rev_verify, "rev001_dir": rdir,
            "scope": "scientific" if a1prof.scientific else "smoke"}


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #
def _plainjson(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _same(a, b):
    return json.dumps(a, sort_keys=True, default=_plainjson) == json.dumps(b, sort_keys=True, default=_plainjson)


def ex_a1_campaign(a1prof, p, udir, out_dir):
    base = a1prof.base
    env, c, m = p["env"], p["c"], p["method"]
    cls, space = FROZEN_CAMPAIGNS[m]
    tag = C.archived_tag(env, c, m)
    res = nested_cv_tpe_extended(cls, space, C.cell_cfg(base, env, c), list(base.seeds), tag, a1prof,
                                 checkpoint_path=os.path.join(udir, "checkpoint.json"))
    ref = load_result(rev001_dir(out_dir), uid(2, "campaign_tpe", method=m, env=env, c=c))
    rb = str(a1prof.b_repro)
    checks = []
    for k, fold in enumerate(res):
        r = ref["results"][k] if ref else None
        sel = fold["selections"][rb]
        row = {"outer_fold": k}
        if r is None:
            row.update({"RC1": False, "RC2": False, "RC4": False, "note": "REV-001 result missing"})
        else:
            prefix = fold["inner_scores"][:a1prof.b_repro]
            row["RC1"] = prefix == r["inner_scores"]
            row["RC1_max_abs_diff"] = (max(abs(a - b) for a, b in zip(prefix, r["inner_scores"]))
                                       if len(prefix) == len(r["inner_scores"]) else None)
            row["RC2"] = sel["best_index"] == r["best_index"] and _same(sel["best_config"], r["best_config"])
            row["RC4"] = fold["repro_test_regrets"] == r["test_regrets"]
        if a1prof.scientific:
            a = _curve_at(load_archived(env, c, m)[k], base.tpe_startup)
            b = _curve_at(fold, base.tpe_startup)
            row["RC5"] = a is not None and b is not None and abs(a - b) < 1e-9
            row["RC5_abs_diff"] = None if a is None or b is None else abs(a - b)
        checks.append(row)
    keys = ["RC1", "RC2", "RC4"] + (["RC5"] if a1prof.scientific else [])
    ok = bool(ref) and all(row.get(k) is True for row in checks for k in keys)
    return {"tag": tag, "method": m, "env": env, "c": c, "n_total": a1prof.n_total, "results": res,
            "reproduction": {"checks": checks, "keys": keys, "ok": ok}}


# --------------------------------------------------------------------------- #
# Stage 3
# --------------------------------------------------------------------------- #
def a1_makers(a1prof, out_dir, env, c):
    makers, missing = {}, []
    for m in A1_METHODS:
        res = load_result(out_dir, a1_uid(2, "a1_campaign", method=m, env=env, c=c))
        if res is None:
            missing.append(m)
            continue
        for b in a1prof.select_budgets:
            cfgs = [f["selections"][str(b)]["best_config"] for f in res["results"]]
            makers[method_name(m, b)] = _config_maker(FROZEN_CAMPAIGNS[m][0], cfgs)
    return makers, missing


def ex_a1_eval(a1prof, p, udir, out_dir):
    base = a1prof.base
    env, c = p["env"], p["c"]
    makers, missing = a1_makers(a1prof, out_dir, env, c)
    cfg = C.cell_cfg(base, env, c)
    recs = {name: evaluate_folds(cfg, mk, _fold_seeds(base), lambda k: C.cew_tag(base, env, c, k))
            for name, mk in makers.items()}
    ref = load_result(rev001_dir(out_dir), uid(3, "eval_cew_b", env=env, c=c))
    checks = []
    for m in A1_METHODS:
        name = method_name(m, a1prof.b_repro)
        a, r = recs.get(name), (ref or {}).get("records", {}).get(f"{m}_tpe")
        if a is None or r is None:
            checks.append({"method": m, "RC3": False, "note": "missing records"})
            continue
        ka = [(x["outer_fold"], x["seed"], x["tag"]) for x in a]
        kr = [(x["outer_fold"], x["seed"], x["tag"]) for x in r]
        same_worlds = ka == kr
        diffs = [abs(x["regret"] - y["regret"]) for x, y in zip(a, r)] if same_worlds else []
        checks.append({"method": m, "same_worlds": same_worlds,
                       "RC3": bool(same_worlds and all(x["regret"] == y["regret"] for x, y in zip(a, r))),
                       "max_abs_diff": max(diffs) if diffs else None})
    ok = not missing and all(ch["RC3"] for ch in checks)
    return {"env": env, "c": c, "records": recs, "missing": missing, "reproduction": {"checks": checks, "ok": ok}}


def ex_a1_analysis(a1prof, p, udir, out_dir):
    from .a1_analysis import run_a1_analysis
    return run_a1_analysis(a1prof, out_dir, rev001_dir(out_dir))


A1_EXECUTORS = {"a1_gate": ex_a1_gate, "a1_campaign": ex_a1_campaign, "a1_eval": ex_a1_eval,
                "a1_analysis": ex_a1_analysis}
A1_RESUMABLE_KINDS = {"a1_campaign"}
