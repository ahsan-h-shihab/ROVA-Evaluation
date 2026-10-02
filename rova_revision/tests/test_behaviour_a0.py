"""Tests for the post-run behavioural comparison (docs/REVISION_002_POSTRUN_ANALYSES.md, P1).
Synthetic data only.
    PYTHONPATH=. python -m rova_revision.tests.test_behaviour_a0
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

from rova_revision.tests.test_rev002 import BLOCKED, REPO

ENV, C = "favorable_hmm_coupled", 8.0


def _rec(seed, fold, regret, l0, steps=10):
    return {"seed": seed, "outer_fold": fold, "tag": f"T{fold}", "regret": regret, "n_agent_steps": steps,
            "a_counts": [l0, 0, 0, 0, 0, steps - l0], "astar_counts": [9, 0, 0, 0, 0, 1]}


def _fixture(root):
    worlds = [(i, i // 2) for i in range(4)]
    a0 = [_rec(s, k, 0.5, 10) for s, k in worlds]
    rova = [_rec(s, k, 0.5 if s < 3 else 0.4, 10 if s < 3 else 7) for s, k in worlds]
    other = [_rec(s, k, 0.3, 6) for s, k in worlds]
    tier = {"S3__eval_cew__": {"A0": a0, "rova": rova, "static": other, "ewma": other},
            "S3__eval_cew_b__": {}, "S3__eval_cew_c__": {}}
    for pre, recs in tier.items():
        d = os.path.join(root, "rev", "units", f"{pre}{C}__{ENV}")
        os.makedirs(d)
        json.dump({"records": recs}, open(os.path.join(d, "result.json"), "w"))
    d = os.path.join(root, "a1", "units", f"A1__S3__a1_eval__{C}__{ENV}")
    os.makedirs(d)
    a1 = {f"{m}_tpe{b}": (rova if m == "rova" else other) for m in ("rova", "static", "ewma") for b in (80, 160, 320)}
    json.dump({"records": a1}, open(os.path.join(d, "result.json"), "w"))


def test_counts_on_synthetic_cell():
    from rova_revision import behaviour_a0 as B
    tmp = tempfile.mkdtemp()
    try:
        _fixture(tmp)
        rows = {r["method"]: r for r in B.cell_rows(B.load_cell(os.path.join(tmp, "rev"), os.path.join(tmp, "a1"), ENV, C), ENV, C)}
        r = rows["rova"]
        assert r["worlds_all_full_supervision"] == 3 and r["regret_equals_A0_on_those_worlds"]
        assert abs(r["fraction_agent_steps_level0"] - 37 / 40) < 1e-12
        assert r["folds_with_all_worlds_full_supervision"] == 1
        assert abs(r["mean_regret_minus_A0"] + 0.025) < 1e-12
        assert rows["static"]["worlds_all_full_supervision"] == 0
        assert abs(rows["static_tpe320"]["oracle_fraction_level0"] - 0.9) < 1e-12
    finally:
        shutil.rmtree(tmp)


def test_runs_with_simulator_imports_blocked():
    tmp = tempfile.mkdtemp()
    try:
        _fixture(tmp)
        code = f"""
import importlib.abc, sys
BLOCKED = {BLOCKED!r}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in BLOCKED):
            raise ImportError("blocked by test: " + name)
        return None
sys.meta_path.insert(0, Block())
from rova_revision import behaviour_a0 as B
rows = B.cell_rows(B.load_cell({os.path.join(tmp, 'rev')!r}, {os.path.join(tmp, 'a1')!r}, {ENV!r}, {C!r}), {ENV!r}, {C!r})
assert len(rows) == 12
assert not [m for m in sys.modules if any(m == b or m.startswith(b + ".") for b in BLOCKED)]
print("isolated-ok")
"""
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": REPO})
        assert out.returncode == 0 and "isolated-ok" in out.stdout, out.stderr[-2000:]
    finally:
        shutil.rmtree(tmp)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
