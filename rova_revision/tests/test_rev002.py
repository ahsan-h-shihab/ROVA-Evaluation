"""REV-002 tests (docs/REVISION_002_ANALYSIS_PLAN.md): A1 extended TPE and orchestration
guards; A2 statistics and its isolation from the simulator.

Non-experimental seeds (the smoke profile, seeds >= 1000), shrunken worlds and
synthetic data only.
    PYTHONPATH=. python -m rova_revision.tests.test_rev002
"""
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# A1
# --------------------------------------------------------------------------- #
def _tiny(n_total, select):
    from rova_revision.a1_config import A1Profile
    from rova_revision.config import SMOKE
    return A1Profile(name="t", base=SMOKE, n_total=n_total, select_budgets=select,
                     curve_budgets=tuple(sorted(set(select))), scientific=False, n_boot=50)


def _run_ext(method, n_total, select, env="favorable_hmm_coupled", c=0.5):
    from rova_revision import config as C
    from rova_revision.tpe_ext import nested_cv_tpe_extended
    from rova_revision.units import FROZEN_CAMPAIGNS
    cls, space = FROZEN_CAMPAIGNS[method]
    prof = _tiny(n_total, select)
    return nested_cv_tpe_extended(cls, space, C.cell_cfg(prof.base, env, c), list(prof.base.seeds),
                                  C.archived_tag(env, c, method), prof)


def _run_frozen(method, env="favorable_hmm_coupled", c=0.5):
    from rova_revision import config as C
    from rova_revision.config import SMOKE
    from rova_revision.tpe import nested_cv_tpe
    from rova_revision.units import FROZEN_CAMPAIGNS
    cls, space = FROZEN_CAMPAIGNS[method]
    return nested_cv_tpe(cls, space, C.cell_cfg(SMOKE, env, c), list(SMOKE.seeds), C.archived_tag(env, c, method), SMOKE)


_CACHE = {}


def _cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def test_T1_extended_at_frozen_budget_reproduces_frozen_tpe():
    for method in ("static", "rova"):
        fro = _cached(("frozen", method), lambda: _run_frozen(method))
        ext = _cached(("ext4", method), lambda: _run_ext(method, 4, (4, 4, 4)))
        for f, e in zip(fro, ext):
            assert e["inner_scores"] == f["inner_scores"], method
            s = e["selections"]["4"]
            assert s["best_index"] == f["best_index"]
            assert json.dumps(s["best_config"], sort_keys=True) == json.dumps(f["best_config"], sort_keys=True)
            assert e["repro_test_regrets"] == f["test_regrets"]
            assert e["repro_test_mean_regret"] == f["test_mean_regret"]
            for b in (2, 4):
                assert e["tuning_curve"][b] == f["tuning_curve"][b]


def test_T2_prefix_property():
    for method in ("static", "rova"):
        long_ = _cached(("ext12", method), lambda: _run_ext(method, 12, (4, 8, 12)))
        mid = _cached(("ext8", method), lambda: _run_ext(method, 8, (4, 8, 8)))
        short = _cached(("ext4", method), lambda: _run_ext(method, 4, (4, 4, 4)))
        for l, m, s in zip(long_, mid, short):
            assert l["inner_scores"][:8] == m["inner_scores"]
            assert l["inner_scores"][:4] == s["inner_scores"]
            assert l["selections"]["8"] == m["selections"]["8"]
            assert l["selections"]["4"] == s["selections"]["4"]
            assert l["repro_test_regrets"] == s["repro_test_regrets"]
            sc = np.asarray(l["inner_scores"])
            for b in (4, 8, 12):
                assert l["selections"][str(b)]["best_index"] == int(np.nanargmin(sc[:b]))


def test_T3_startup_draw_not_resized():
    """The startup candidates of every outer fold equal the frozen run's; resizing the
    draw to the extended budget would change those of folds after the first."""
    from dtcoa_val.experiments.search_space import sample_config
    from rova_revision.units import FROZEN_CAMPAIGNS
    space = FROZEN_CAMPAIGNS["static"][1]

    def draw(n):
        rng = np.random.default_rng(0)
        return [[sample_config(space, rng) for _ in range(n)] for _ in range(2)]
    a, b = draw(4), draw(12)
    assert json.dumps(a[0][:2], default=list) == json.dumps(b[0][:2], default=list)
    assert json.dumps(a[1][:2], default=list) != json.dumps(b[1][:2], default=list)  # why the draw stays at 4
    fro = _cached(("frozen", "static"), lambda: _run_frozen("static"))
    ext = _cached(("ext12", "static"), lambda: _run_ext("static", 12, (4, 8, 12)))
    for f, e in zip(fro, ext):
        assert e["inner_scores"][:2] == f["inner_scores"][:2]


def test_T4_convergence_classifier():
    from rova_revision.rev002_constants import CONV_ABS, CONV_REL, fold_converged
    assert CONV_REL == 0.05 and CONV_ABS == 0.0002
    assert fold_converged(1.0, 0.951)          # 4.9% relative
    assert not fold_converged(1.0, 0.94)       # 6%
    assert fold_converged(0.001, 0.0008)       # 0.0002 absolute floor
    assert not fold_converged(0.001, 0.00079)
    assert fold_converged(0.0, 0.0)
    assert fold_converged(0.5, 0.5)


def test_T5_world_tags_are_rev001():
    from rova_revision import config as C
    from rova_revision.a1_config import A1_FULL
    from rova_revision.config import FULL
    assert A1_FULL.base is FULL and FULL.tag_prefix == "REV1"
    assert A1_FULL.base.n_trials == 80 and A1_FULL.base.tpe_startup == 20
    assert A1_FULL.n_total == 320 and A1_FULL.select_budgets == (80, 160, 320)
    for env in C.ENV_NAMES:
        for c in FULL.grid:
            for k in range(5):
                assert C.cew_tag(FULL, env, c, k) == f"REV1_CEW__{env}__c{c}_out{k}_test"
            assert C.archived_tag(env, c, "rova") == f"{env}__c{c}__rova"


def test_T6_unit_plan_and_verifier_on_empty_output():
    from rova_revision.a1_config import A1_FULL
    from rova_revision.a1_units import enumerate_a1_units
    from rova_revision.verify_a1 import verify
    units = enumerate_a1_units(A1_FULL)
    kinds = {k: sum(u["kind"] == k for u in units) for k in ("a1_gate", "a1_campaign", "a1_eval", "a1_analysis")}
    assert kinds == {"a1_gate": 1, "a1_campaign": 45, "a1_eval": 15, "a1_analysis": 1}
    assert len({u["uid"] for u in units}) == 62
    tmp = tempfile.mkdtemp()
    try:
        res = verify(A1_FULL, tmp)
        assert not res["passed"] and not any(res["checks"].values())
    finally:
        shutil.rmtree(tmp)


def test_T7_output_location_refusals():
    from rova_revision import run_a1
    tmp = tempfile.mkdtemp()
    try:
        for out, rev in ((os.path.join(REPO, "_a1_out_inside_repo"), tmp),
                         (os.path.join(tmp, "a1_out"), tmp)):
            try:
                run_a1.main(["--profile", "smoke", "--out", out, "--rev001", rev])
            except SystemExit as e:
                assert e.code == 2
            else:
                raise AssertionError("run_a1 accepted a forbidden output location")
            assert not os.path.exists(os.path.join(REPO, "_a1_out_inside_repo"))
    finally:
        shutil.rmtree(tmp)


# --------------------------------------------------------------------------- #
# A2
# --------------------------------------------------------------------------- #
METHODS = ["static", "ewma", "rova", "A0", "A5", "learned", "qlearn", "static_ext", "ewma_ext", "clairvoyant",
           "rova_single", "static_tpe", "ewma_tpe", "rova_tpe", "abl_full", "abl_no_memory", "abl_no_decay",
           "abl_no_propagation", "abl_no_stability"]


def _fixture(root, env="favorable_hmm_coupled", c=8.0):
    rng = np.random.default_rng(12345)
    recs = {}
    for j, m in enumerate(METHODS):
        base = 0.02 * (j + 1)
        recs[m] = [{"seed": 1000 + i, "outer_fold": i // 12, "tag": f"SYN_out{i // 12}",
                    "regret": float(base + 0.01 * (i // 12) + 0.002 * rng.standard_normal())} for i in range(60)]
    split = {"S3__eval_cew__": METHODS[:10] + METHODS[14:], "S3__eval_cew_b__": METHODS[10:14], "S3__eval_cew_c__": []}
    for pre, ms in split.items():
        d = os.path.join(root, "units", f"{pre}{c}__{env}")
        os.makedirs(d)
        with open(os.path.join(d, "result.json"), "w") as f:
            json.dump({"records": {m: recs[m] for m in ms}, "missing": []}, f)
    os.makedirs(os.path.join(root, "analysis"))
    with open(os.path.join(root, "analysis", "summary.json"), "w") as f:
        json.dump({"cew": {f"{env}|{c}": {"families": {"F1": [{"a": "rova", "b": "static", "label": "rova lower-regret",
                                                               "code": "a_lower", "wilcoxon_p": 0.001,
                                                               "cliffs_delta": -0.9}]},
                                          "G": {"status": "defined", "G": 0.1}}}}, f)
    return env, c


BLOCKED = ["dtcoa_val.sim_core", "dtcoa_val.experiments", "dtcoa_val.oracles", "dtcoa_val.baselines",
           "dtcoa_val.dtcoa_tunable", "dtcoa_val.dtcoa", "dtcoa_val.environments", "dtcoa_val.utils",
           "rova_revision.units", "rova_revision.instrument", "rova_revision.config", "rova_revision.archive",
           "rova_revision.tpe", "rova_revision.tpe_ext", "rova_revision.training", "rova_revision.policies",
           "rova_revision.coupled_dp", "rova_revision.gate", "rova_revision.a1_units", "rova_revision.a1_config",
           "rova_revision.stathelp", "rova_revision.analysis"]


def test_T8_a2_runs_with_simulator_imports_blocked():
    tmp = tempfile.mkdtemp()
    try:
        env, c = _fixture(tmp)
        code = f"""
import importlib.abc, sys
BLOCKED = {BLOCKED!r}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in BLOCKED):
            raise ImportError("blocked by test: " + name)
        return None
sys.meta_path.insert(0, Block())
from rova_revision import clustered
rows, g = clustered.run_rev001({tmp!r}, n_boot=200, cells=[({env!r}, {c!r})])
assert len(rows) == 20 and len(g) == 1, (len(rows), len(g))
loaded = [m for m in sys.modules if any(m == b or m.startswith(b + ".") for b in BLOCKED)]
assert not loaded, loaded
print("isolated-ok")
"""
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": REPO})
        assert out.returncode == 0 and "isolated-ok" in out.stdout, out.stderr[-2000:]
    finally:
        shutil.rmtree(tmp)


def test_T9_a2_known_answers_and_determinism():
    from rova_revision import clustered as A
    folds = np.repeat(np.arange(5), 12)
    x = np.linspace(0.1, 0.7, 60)
    r = A.analyze_pair(x, x.copy(), folds, n_boot=300)
    assert r["two_stage_mean_diff_ci"] == [0.0, 0.0] and r["fold_only_mean_diff_ci"] == [0.0, 0.0]
    assert r["cliffs_delta"] == 0.0 and r["sign_flip_p"] == 1.0
    p, n = A.sign_flip_p([0.1, 0.2, 0.3, 0.4, 0.5])
    assert n == 32 and abs(p - 0.0625) < 1e-12
    d = np.repeat([0.1, 0.2, 0.3, 0.4, 0.5], 12)
    assert abs(A.icc1(d, folds) - 1.0) < 1e-12
    F, _ = A.fold_structure(folds)
    idx = A.two_stage_indices(F, 40)
    for row in idx:
        blocks = row.reshape(5, 12)
        for blk in blocks:
            assert len(set(folds[blk])) == 1     # each block of 12 comes from one fold
    assert np.array_equal(idx, A.two_stage_indices(F, 40))   # deterministic
    rng = np.random.default_rng(7)
    xa, xb = rng.random(60), rng.random(60)
    r1, r2 = A.analyze_pair(xa, xb, folds, 300), A.analyze_pair(xa, xb, folds, 300)
    assert json.dumps(r1) == json.dumps(r2)
    lo, hi = r1["two_stage_delta_ci"]
    assert -1.0 <= lo <= hi <= 1.0
    g = A.g_clustered({"A0": xa, "rova": xa.copy(), "static": xa + 1.0, "ewma": xa + 2.0}, folds, 200)
    assert g["status"] == "defined" and g["G"] == 0.0 and g["two_stage_G_ci"] == [0.0, 0.0]
    assert A.generalization_flag("a_lower", [-0.2, -0.1]) == "generalizes across tuning runs"
    assert A.generalization_flag("a_lower", [-0.2, 0.1]) == "not established across tuning runs"
    assert A.generalization_flag("b_lower", [0.1, 0.2]) == "generalizes across tuning runs"
    assert A.generalization_flag("none", [0.1, 0.2]).startswith("no per-world label")


def test_T10_constants_match_plan_text():
    from rova_revision import rev002_constants as K
    with open(os.path.join(REPO, "docs", "REVISION_002_ANALYSIS_PLAN.md"), encoding="utf-8") as f:
        plan = f.read()
    assert "B(160) − B(320) ≤ max(0.05 × B(160), 0.0002)" in plan
    for h in (K.REV001_ARCHIVE_SHA256, K.REV001_SUMS_SHA256):
        assert h in plan
    assert "1,883" in plan and K.REV001_SUMS_ENTRIES == 1883
    assert K.A2_N_BOOT == 5000 and K.A2_SEED == 0 and K.FROZEN_TAG_REV002 == "revision-002-frozen"


def test_T11_a1_guard_binds_rev002_tag():
    from rova_revision import a1_guard, env_guard
    info = a1_guard.collect(workers=15)
    assert info["git_frozen_tag_name"] == "revision-002-frozen"
    ref = env_guard.reference()
    res = env_guard.evaluate(info, ref)
    names = {c["item"] for c in res["hard_checks"]}
    assert "git_commit_is_frozen_tag" in names and "git_working_tree_clean" in names
    fake = dict(info, git_head="0" * 40)
    assert not env_guard.evaluate(fake, ref)["passed"]


def main():
    tests = [v for k, v in sorted(globals().items(), key=lambda kv: kv[0]) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
