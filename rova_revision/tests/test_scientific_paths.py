"""Exercises analysis paths that only the scientific profile reaches, using
SYNTHETIC unit results (never simulation output) written to a temporary
directory. Archived MAIN-001 files are read only to check seed alignment.

    PYTHONPATH=. python -m rova_revision.tests.test_scientific_paths
"""
import json
import os
import shutil
import tempfile

import numpy as np

from rova_revision import analysis as AN
from rova_revision import config as C
from rova_revision import units as UN
from rova_revision.archive import load_archived, outer_folds
from rova_revision.config import FULL

ENV, CC = "collapse_moderate_coupled", 3.5
FOLDS = [(k, list(f[1])) for k, f in enumerate(outer_folds(FULL.seeds, FULL.K_out))]


def _write(out, u, result):
    d = os.path.join(out, "units", u)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(result, f)
    open(os.path.join(d, "done.json"), "w").close()


def _rec(k, s, tag, regret, sum_loss=None, n=1500):
    return {"seed": s, "outer_fold": k, "tag": tag, "regret": regret, "harm_rate": 0.1, "overrestrict": 0.2,
            "unauth_unsafe_rate": 0.0, "sysfail_rate": 0.0, "esc_rate": 0.1,
            "sum_loss": regret * n if sum_loss is None else sum_loss, "sum_opt": 0.0, "n_agent_steps": n,
            "astar_counts": [n // 2, 0, 0, 0, 0, n - n // 2], "a_counts": [n, 0, 0, 0, 0, 0]}


def _recs(tag_fn, rng, shift=0.0):
    return [_rec(k, s, tag_fn(k), float(rng.uniform(0.1, 0.2) + shift)) for k, seeds in FOLDS for s in seeds]


def test_anchor_pairs_with_every_archived_seed():
    out = tempfile.mkdtemp(prefix="rova_sci_")
    try:
        rng = np.random.default_rng(6001)
        res = {}
        for A in ("A0", "A5"):
            res[A] = {"own": _recs(lambda k, A=A: C.own_world_tag(FULL, ENV, CC, f"always{A[1]}", k), rng)}
            for m in C.ARCHIVED_METHODS:
                res[A][f"on_{m}_worlds"] = _recs(lambda k, m=m: f"{C.archived_tag(ENV, CC, m)}_out{k}_test", rng)
        _write(out, UN.uid(2, "anchor_eval", env=ENV, c=CC), res)
        a = AN.anchor_cell(FULL, out, ENV, CC, 100)
        fam = a["archived_worlds"]["family"]
        assert len(fam) == 6 and all("label" in r for r in fam)
        for r in fam:
            pooled = [x for f in load_archived(ENV, CC, r["a"]) for x in f["test_regrets"]]
            assert abs(r["mean_a"] - float(np.mean(pooled))) < 1e-12 and r["n"] == 60
    finally:
        shutil.rmtree(out)


def test_sweep_and_tpe_scientific_paths():
    out = tempfile.mkdtemp(prefix="rova_sci_")
    try:
        rng = np.random.default_rng(6002)
        rows, repro = [], []
        for k, seeds in FOLDS:
            pts = [("tuned", None, None), ("both_zero", "both", 0.0)]
            pts += [("prop", "prop_scale", g) for g in UN.SWEEP_GRID] + [("stab", "stab_scale", g) for g in UN.SWEEP_GRID]
            for label, key, val in pts:
                rows.append({"outer_fold": k, "point": label, "param": key, "value": val, "prop_scale": 0.1,
                             "stab_scale": 0.2, "mean_regret": 0.1, "seeds": seeds,
                             "regrets": (rng.uniform(0.1, 0.2, len(seeds)) + (0.0 if label == "tuned" else 0.05)).tolist()})
            repro.append({"outer_fold": k, "max_abs_diff": 0.0, "ok": True})
        _write(out, UN.uid(2, "sweep", env=ENV, c=CC), {"rows": rows, "reproduction": repro, "grid": UN.SWEEP_GRID})
        sw = AN.sweep_cells(FULL, out, 100)
        cell = sw[f"{ENV}|{CC}"]
        assert cell["status"] == "ok" and cell["reproduction_ok"] is True
        assert [p["value"] for p in cell["dose_response"]["prop_scale"]] == sorted(UN.SWEEP_GRID)
        assert all(r["code"] == "a_lower" for r in cell["tuned_vs_zero"]), cell["tuned_vs_zero"]
        assert sum(v.get("status") == "missing unit" for v in sw.values()) == len(sw) - 1

        arch = load_archived(ENV, CC, "static")
        for ok_last in (True, False):
            results = [{"outer_fold": k, "best_config": {}, "best_index": 3,
                        "best_inner_score": a["best_inner_score"] - 0.01,
                        "tuning_curve": {"20": a["tuning_curve"]["20"], "40": 0.1, "80": 0.09}}
                       for k, a in enumerate(arch)]
            sr = [{"outer_fold": k, "ok": ok_last or k < 4} for k in range(5)]
            _write(out, UN.uid(2, "campaign_tpe", method="static", env=ENV, c=CC),
                   {"results": results, "startup_reproduction": sr})
            t = AN.tpe_cells(FULL, out)[f"static|{ENV}|{CC}"]
            assert t["startup_reproduction_ok"] is ok_last
            assert all(abs(d + 0.01) < 1e-12 for d in t["best_inner_tpe_minus_random_search"])
    finally:
        shutil.rmtree(out)


def test_dp_substantial_disagreement_branch():
    out = tempfile.mkdtemp(prefix="rova_sci_")
    try:
        rng = np.random.default_rng(6003)
        env = C.DP_FULL_SIZE_ENV
        raw = {}
        for c in FULL.grid:
            dp = []
            recs = {m: [] for m in ("rova", "static", "ewma", "A0", "A5", "learned")}
            for k, seeds in FOLDS:
                tag = C.cew_tag(FULL, env, c, k)
                for s in seeds:
                    vopt = float(rng.uniform(100, 110))
                    dp.append({"seed": s, "tag": tag, "N": 6, "T": 250, "V_opt": vopt, "V_myopic": vopt + 5})
                    base = float(rng.uniform(0.0, 0.01))
                    # myopic reference: rova lower; sequential reference (total loss): rova higher
                    loss = {"rova": vopt + 30, "static": vopt + 10, "ewma": vopt + 20, "A0": vopt + 40,
                            "A5": vopt + 50, "learned": vopt + 25}
                    myo = {"rova": base, "static": base + 0.05, "ewma": base + 0.06, "A0": base + 0.07,
                           "A5": base + 0.08, "learned": base + 0.02}
                    for m in recs:
                        recs[m].append(_rec(k, s, tag, myo[m] + rng.normal(0, 1e-4), sum_loss=loss[m] +
                                            rng.normal(0, 0.1)))
            _write(out, UN.uid(2, "dp_cew", c=c), {"env": env, "c": c, "records": dp})
            raw[(env, c)] = {"records": recs}
        res = AN.dp_sequential(FULL, out, raw, set(), 100)
        assert res["decision"].startswith("substantial"), res["decision"]
        ch = {(x["family"], x["pair"]) for x in res["label_changes"]}
        assert ("F1", "rova_vs_static") in ch
        assert all(not v["problems"] for v in res["cells"].values())
    finally:
        shutil.rmtree(out)


def test_qlearn_gate_failure_excludes():
    out = tempfile.mkdtemp(prefix="rova_sci_")
    try:
        for e in C.ENV_NAMES:
            for c in FULL.grid:
                folds = []
                for k in range(5):
                    gate = {"G1_finite": True, "G2_bounded": not (e == ENV and c == CC and k == 2),
                            "q_min": -1.0, "q_max": 0.0}
                    if k == 0:
                        gate["G3_deterministic"] = True
                    folds.append({"outer_fold": k, "test_regrets": [0.1] * 12, "test_mean_regret": 0.1,
                                  "best_config": {}, "tuning_curve": {}, "gate": gate,
                                  "training_seed_sensitivity": [{"seed_offset": 100, "test_mean_regret": 0.11},
                                                                {"seed_offset": 200, "test_mean_regret": 0.12}]})
                _write(out, UN.uid(2, "campaign_trainable", method="qlearn", env=e, c=c), {"results": folds})
        _t, gate = AN.trainable_cells(FULL, out, 50)
        assert gate["status"] == "failed" and "excluded" in gate["consequence"]
    finally:
        shutil.rmtree(out)


def test_g_ratio_unstable_denominator():
    rng = np.random.default_rng(6004)
    rv = rng.uniform(0.1, 0.2, 60)
    noise = rng.normal(0, 0.5, 60)
    d = {"rova": rv, "static": rv + noise - noise.mean() + 0.01, "ewma": rv + 1.0, "A0": rv + 0.005}
    g = AN.g_ratio(d, 300)
    assert g["status"] == "defined" and g["resamples_with_nonpositive_denominator"] > 0 and g["G_ci"] is None


def test_run_analysis_on_sparse_full_output_and_section_errors():
    out = tempfile.mkdtemp(prefix="rova_sci_")
    saved = AN.archived_reanalysis, AN.sweep_cells
    try:
        AN.archived_reanalysis = lambda cells, n_boot: ({}, {"n_tests": 0, "label_changes": [], "verdicts": {}})
        res = AN.run_analysis(FULL, out)
        assert res["section_errors"] == [], res["section_errors"]
        assert os.path.exists(os.path.join(out, "analysis", "REPORT.md"))
        assert res["completeness"]["done"] == 0 and len(res["completeness"]["not_run"]) == 592

        def boom(*a):
            raise RuntimeError("synthetic section failure")
        AN.sweep_cells = boom
        res = AN.run_analysis(FULL, out)
        assert [e["section"] for e in res["section_errors"]] == ["sweep"], res["section_errors"]
    finally:
        AN.archived_reanalysis, AN.sweep_cells = saved
        shutil.rmtree(out)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
