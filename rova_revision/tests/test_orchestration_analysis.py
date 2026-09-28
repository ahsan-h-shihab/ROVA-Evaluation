"""Tests of the orchestrator and of the analysis helpers.

Synthetic data and non-experimental seeds only. Run from the repository root:
    PYTHONPATH=. python -m rova_revision.tests.test_orchestration_analysis
"""
import json
import os
import shutil
import tempfile

import numpy as np

from rova_revision import run_revision as RR
from rova_revision import units as UN
from rova_revision.analysis import (aligned, dp_sequential, family, g_ratio, rule, variance_decomposition,
                                    verdict)
from rova_revision.config import SMOKE


# --------------------------------------------------------------------------- #
# Test executors (top level so that spawned workers can import them)
# --------------------------------------------------------------------------- #
def _flaky(profile, p, udir, out_dir):
    marker = os.path.join(out_dir, f"flaky_{p['name']}.marker")
    if not os.path.exists(marker):
        open(marker, "w").close()
        raise RuntimeError("first attempt fails by design")
    return {"value": 1}


def _fail(profile, p, udir, out_dir):
    raise RuntimeError("always fails by design")


def _ok(profile, p, udir, out_dir):
    return {"value": p["name"]}


def _register():
    UN.EXECUTORS.update({"t_flaky": _flaky, "t_fail": _fail, "t_ok": _ok})


def _worker(unit, profile_name, out_dir):
    _register()
    return RR.execute_unit(unit, profile_name, out_dir)


def _unit(kind, name):
    return {"uid": f"S9__{kind}__{name}", "stage": 9, "tier": "A", "kind": kind, "cost": 1,
            "params": {"name": name}}


def _tmp():
    return tempfile.mkdtemp(prefix="rova_rev_test_")


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def test_serial_retry_failure_and_resume():
    _register()
    out = _tmp()
    try:
        us = [_unit("t_flaky", "a"), _unit("t_fail", "b"), _unit("t_ok", "c")]
        res = RR.run_stage_serial(us, "smoke", out)
        assert res["failed"] == ["S9__t_fail__b"], res
        assert RR.unit_is_done(out, us[0]) and RR.unit_is_done(out, us[2])
        assert not RR.unit_is_done(out, us[1])
        assert os.path.exists(os.path.join(out, "units", us[1]["uid"], "failed.json"))
        # a done unit is skipped on resume; a tampered result is detected and re-run
        with open(os.path.join(out, "units", us[2]["uid"], "result.json"), "w") as f:
            json.dump({"value": "tampered"}, f)
        assert not RR.unit_is_done(out, us[2])
        RR.run_stage_serial([us[2]], "smoke", out)
        with open(os.path.join(out, "units", us[2]["uid"], "result.json")) as f:
            assert json.load(f) == {"value": "c"}
    finally:
        shutil.rmtree(out)


def test_parallel_retry_and_failure():
    out = _tmp()
    try:
        us = [_unit("t_flaky", "p"), _unit("t_fail", "q"), _unit("t_ok", "r")]
        res = RR.run_stage_parallel(us, "smoke", out, workers=2, fn=_worker)
        assert res["failed"] == ["S9__t_fail__q"], res
        assert RR.unit_is_done(out, us[0]) and RR.unit_is_done(out, us[2])
        res2 = RR.run_stage_parallel(us, "smoke", out, workers=2, fn=_worker)
        assert res2["skipped_done"] == 2 and res2["failed"] == ["S9__t_fail__q"], res2
    finally:
        shutil.rmtree(out)


def test_rerun_forces_execution():
    _register()
    out = _tmp()
    try:
        u = _unit("t_ok", "z")
        RR.run_stage_serial([u], "smoke", out)
        first = json.load(open(os.path.join(out, "units", u["uid"], "done.json")))["finished"]
        import time
        time.sleep(1.1)
        RR.run_stage_serial([u], "smoke", out, rerun=True)
        second = json.load(open(os.path.join(out, "units", u["uid"], "done.json")))["finished"]
        assert first != second
    finally:
        shutil.rmtree(out)


def test_output_inside_repository_rejected():
    from rova_revision.env_guard import REPO
    try:
        RR.main(["--profile", "smoke", "--out", os.path.join(REPO, "revision_out_should_not_exist")])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("an output directory inside the repository must be rejected")
    assert not os.path.exists(os.path.join(REPO, "revision_out_should_not_exist"))


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def test_rule_and_verdict():
    assert rule("x", "y", True, -0.5)[0] == "a_lower"
    assert rule("x", "y", True, 0.5)[0] == "b_lower"
    assert rule("x", "y", True, 0.147)[0] == "none"
    assert rule("x", "y", False, -0.9)[0] == "none"
    rows = [{"a": "rova", "b": "static", "code": "a_lower"}, {"a": "rova", "b": "ewma", "code": "a_lower"}]
    assert verdict(rows) == "ROVA_wins"
    rows[1]["code"] = "b_lower"
    assert verdict(rows) == "mixed"
    rows[0]["code"] = "b_lower"
    assert verdict(rows) == "ROVA_loses"
    assert verdict(rows[:1]) is None


def test_family_holm_and_missing():
    rng = np.random.default_rng(5001)
    base = rng.uniform(0.1, 0.2, 60)
    data = {"rova": base, "static": base + 0.05 + rng.normal(0, 0.005, 60), "ewma": base.copy()}
    rows = family([("rova", "static"), ("rova", "ewma"), ("static", "A0")], data, 200)
    assert rows[0]["code"] == "a_lower" and rows[0]["holm_reject"]
    assert rows[1]["code"] == "none" and rows[1]["wilcoxon_p"] == 1.0
    assert rows[2]["missing"] == ["A0"] and "code" not in rows[2]
    assert rows[0]["family_size"] == 2


def test_g_ratio():
    rng = np.random.default_rng(5002)
    rv = rng.uniform(0.1, 0.2, 60)
    noise = lambda: rng.normal(0, 1e-4, 60)
    d = {"rova": rv, "A0": rv + 0.5 + noise(), "static": rv + 1.0 + noise(), "ewma": rv + 2.0 + noise()}
    g = g_ratio(d, 300)
    assert g["best_baseline"] == "static" and abs(g["G"] - 0.5) < 1e-2, g
    lo, hi = g["G_ci"]
    assert lo <= g["G"] <= hi and hi - lo < 1e-2, g
    assert "closes a fraction" in g["interpretation"]
    d["static"] = rv - 0.1
    assert g_ratio(d, 300)["status"].startswith("not defined")


def test_variance_decomposition():
    rng = np.random.default_rng(5003)
    groups = np.repeat(np.arange(5), 12)
    same = rng.normal(0, 1, 60)
    sep = same * 0.01 + groups * 10.0
    assert variance_decomposition(sep, groups)["icc1"] > 0.99
    flat = variance_decomposition(rng.normal(0, 1, 60), groups)
    assert flat["icc1"] < 0.5 and flat["var_between"] >= 0.0


def test_aligned_detects_different_worlds():
    rec = lambda s, tag, v: {"outer_fold": 0, "seed": s, "tag": tag, "regret": v}
    recs = {"a": [rec(1, "T", 0.1), rec(2, "T", 0.2)], "b": [rec(2, "T", 0.3), rec(1, "T", 0.4)],
            "c": [rec(1, "U", 0.1), rec(2, "U", 0.2)]}
    data, keys, bad = aligned(recs)
    assert bad == ["c"] and list(data["b"]) == [0.4, 0.3]


def test_dp_sequential_detects_missing_world():
    out = _tmp()
    try:
        c = SMOKE.grid[0]
        u = UN.uid(2, "dp_cew", c=c)
        os.makedirs(os.path.join(out, "units", u))
        dp = {"records": [{"seed": 1000, "tag": "W", "N": 2, "T": 10, "V_opt": 1.0, "V_myopic": 1.5}]}
        with open(os.path.join(out, "units", u, "result.json"), "w") as f:
            json.dump(dp, f)
        open(os.path.join(out, "units", u, "done.json"), "w").close()
        r = lambda tag: {"outer_fold": 0, "seed": 1000, "tag": tag, "sum_loss": 3.0, "n_agent_steps": 20,
                         "regret": 0.05}
        raw = {("collapse_moderate_coupled", c): {"records": {"rova": [r("W")], "static": [r("X")]}}}
        res = dp_sequential(SMOKE, out, raw, set(), 50)
        cell = res["cells"][str(c)]
        assert any("no DP world" in p for p in cell["problems"]), cell["problems"]
        assert abs(cell["methods"]["rova"]["sequential_regret"]["mean"] - (3.0 - 1.0) / 20) < 1e-12
    finally:
        shutil.rmtree(out)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
