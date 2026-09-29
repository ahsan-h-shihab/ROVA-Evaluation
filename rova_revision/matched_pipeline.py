"""Matched-world correction of the published analyses (plan section 3, R2(d)).

The published MAIN-001 statistics (``experiments/MAIN-001_core_comparison/
analysis.py``), the symmetric verdict rule (``verdicts/verdicts_v2.py``) and the
ABLATION-001 attribution analyses (``dtcoa_val/analysis/dr3_attribution.py``,
``dr3_regime_inference.py``) are re-run UNCHANGED on the common-evaluation-world
records, written in the archived checkpoint layout. Only the inputs change: every
method's per-fold tuned configuration evaluated on worlds shared by all methods.
The archived files are never modified; outputs go to ``analysis/matched_world``.
"""
import importlib.util
import json
import os
import sys

import numpy as np

from dtcoa_val.analysis import dr3_attribution, dr3_regime_inference
from dtcoa_val.experiments.nested_hpo import _cvar

from .archive import (ABL_REGIME_PUBLISHED, MAIN001, REPO, VERDICTS_V2, load_archived, load_archived_ablation,
                      load_stats_summary)

MAIN_METHODS = ("static", "ewma", "rova")
ABL_VARIANTS = ("full", "no_memory", "no_decay", "no_propagation", "no_stability")


def _load_script(name, path):
    saved = sys.argv
    sys.argv = [path]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def checkpoint_view(records, archived_folds):
    """Archived checkpoint layout built from CEW records (fold order, seed order of the split)."""
    by_fold = {}
    for r in records:
        by_fold.setdefault(int(r["outer_fold"]), []).append(r)
    out = []
    for fold in archived_folds:
        k = int(fold["outer_fold"])
        if k not in by_fold:
            continue
        rs = by_fold[k]
        order = {int(s): i for i, s in enumerate(fold.get("_seeds", []))}
        rs = sorted(rs, key=lambda r: order.get(int(r["seed"]), int(r["seed"])))
        reg = [float(r["regret"]) for r in rs]
        out.append({"outer_fold": k, "best_config": fold["best_config"], "test_regrets": reg,
                    "test_mean_regret": float(np.mean(reg)), "test_cvar95_regret": _cvar(reg, 0.95),
                    "seeds": [int(r["seed"]) for r in rs], "tags": sorted({r["tag"] for r in rs})})
    return out


def write_views(view_dir, cew_records, cells, fold_seeds):
    """Write MAIN-001 and ABLATION-001 checkpoint views of the CEW records."""
    main_dir, abl_dir = os.path.join(view_dir, "main001"), os.path.join(view_dir, "ablation001")
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(abl_dir, exist_ok=True)
    written = {"main001": [], "ablation001": []}
    for env, c in cells:
        recs = cew_records.get((env, c), {})
        for m in MAIN_METHODS:
            if m in recs:
                arch = [dict(f, _seeds=fold_seeds.get(int(f["outer_fold"]), [])) for f in load_archived(env, c, m)]
                with open(os.path.join(main_dir, f"{env}__c{c}__{m}.json"), "w", newline="\n") as f:
                    json.dump({"results": checkpoint_view(recs[m], arch)}, f, default=_jsonable)
                written["main001"].append(f"{env}__c{c}__{m}")
        for v in ABL_VARIANTS:
            name = f"abl_{v}"
            if name in recs:
                arch = [dict(f, _seeds=fold_seeds.get(int(f["outer_fold"]), [])) for f in load_archived_ablation(env, c, v)]
                with open(os.path.join(abl_dir, f"{env}__c{c}__{v}.json"), "w", newline="\n") as f:
                    json.dump({"results": checkpoint_view(recs[name], arch)}, f, default=_jsonable)
                written["ablation001"].append(f"{env}__c{c}__{v}")
    return main_dir, abl_dir, written


def rerun_main001(main_dir, cells):
    """Frozen MAIN-001 per-cell statistics and the v2 verdict rule on matched worlds."""
    mod = _load_script("main001_analysis_frozen", os.path.join(MAIN001, "analysis.py"))
    ver = _load_script("verdicts_v2_frozen", os.path.join(REPO, "verdicts", "verdicts_v2.py"))
    mod.CK = main_dir
    published = load_stats_summary()["cells"]
    with open(VERDICTS_V2) as f:
        pub_v2 = {(r["environment"], float(r["cost_ratio"])): r["v2_verdict"] for r in json.load(f)["cells"]}
    out, changes = {}, []
    for env, c in cells:
        if not all(os.path.exists(os.path.join(main_dir, f"{env}__c{c}__{m}.json")) for m in MAIN_METHODS):
            out[f"{env}|{c}"] = {"status": "missing"}
            continue
        cell = mod.analyze_cell(env, c)
        v2, _sides = ver.v2_verdict(cell)
        pub = published[env][str(c)]
        row = {"status": "ok", "matched": cell, "matched_v1_verdict": cell["verdict"], "matched_v2_verdict": v2,
               "published_v1_verdict": pub["verdict"], "published_v2_verdict": pub_v2.get((env, float(c)))}
        pair_changes = []
        for rec in cell["primary_pairwise"]:
            old = next(x for x in pub["primary_pairwise"] if x["pair"] == rec["pair"])
            pair_changes.append({"pair": rec["pair"], "published_p": old["wilcoxon_p"], "matched_p": rec["wilcoxon_p"],
                                 "published_holm_reject": old["holm_reject"], "matched_holm_reject": rec["holm_reject"],
                                 "published_delta": old["cliffs_delta"], "matched_delta": rec["cliffs_delta"]})
        row["pairs"] = pair_changes
        out[f"{env}|{c}"] = row
        if v2 != row["published_v2_verdict"]:
            changes.append({"env": env, "c": c, "published_v2": row["published_v2_verdict"], "matched_v2": v2})
    return {"cells": out, "v2_verdict_changes": changes,
            "note": "frozen analysis.py analyze_cell and verdicts_v2.v2_verdict, unchanged, on matched-world inputs"}


def rerun_ablation(abl_dir, envs, grid):
    """Frozen ABLATION-001 per-cell attribution and regime-level inference on matched worlds."""
    per_cell = {}
    for env in envs:
        for c in grid:
            rec = dr3_attribution.recompute_cell(abl_dir, env, c)
            per_cell[f"{env}|{c}"] = rec if rec is not None else {"status": "missing"}
    regime = dr3_regime_inference.build(abl_dir, envs=envs, grid=grid)
    published = None
    if os.path.exists(ABL_REGIME_PUBLISHED):
        with open(ABL_REGIME_PUBLISHED) as f:
            published = json.load(f)
    comparison = []
    if published:
        for comp, blk in regime["per_component"].items():
            for reg, r in blk["by_regime"].items():
                old = published["per_component"][comp]["by_regime"].get(reg, {})
                comparison.append({"component": comp, "regime": reg,
                                   "published_mean_delta": old.get("mean_delta"), "matched_mean_delta": r["mean_delta"],
                                   "published_optionC_p": (old.get("permutation_optionC") or {}).get("p"),
                                   "matched_optionC_p": r["permutation_optionC"]["p"]})
    return {"per_cell_frozen_ablation_report": per_cell, "regime_inference": regime,
            "regime_markdown": dr3_regime_inference.render_markdown(regime), "published_vs_matched": comparison,
            "note": "frozen dr3_attribution.recompute_cell and dr3_regime_inference.build, unchanged, on matched worlds"}
