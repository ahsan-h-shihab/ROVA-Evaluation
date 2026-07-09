"""FIX-4 (ABLATION-001) validator tests. Run:
    python3 -m dtcoa_val.tests.test_ablation_consistency

Verifies the distributional-consistency validator discriminates correctly:
- passes on genuinely consistent data (identical arrays; two independent
  Monte-Carlo draws of the same estimand under different run tags), and
- fails loudly on genuine drift (a systematically shifted distribution),
and that it is READ-ONLY with respect to the MAIN-001 tree.
"""
import hashlib
import os
import tempfile

import numpy as np

from dtcoa_val.utils import save_json
from dtcoa_val.experiments.environments import cfg_for
from dtcoa_val.experiments.search_space import ROVA_SEARCH_SPACE
from dtcoa_val.experiments.nested_hpo import nested_cv_hpo
from dtcoa_val.dtcoa_tunable import DTCOATunable
from dtcoa_val.experiments.validate_consistency import (
    compare_distributions, validate_experiment,
)


def test_identical_distributions_pass():
    x = [0.10, 0.12, 0.09, 0.11, 0.10]
    assert compare_distributions(x, list(x), rel_tol=0.15)["pass"]


def test_disjoint_distributions_fail():
    a = [0.10, 0.11, 0.09, 0.10, 0.12]
    m = [0.80, 0.82, 0.79, 0.81, 0.83]  # far apart, non-overlapping CIs
    d = compare_distributions(a, m, rel_tol=0.15)
    assert not d["pass"] and not d["ci_overlap"]


def _draw(tag):
    cfg = cfg_for("unfavorable_oscillation_uncoupled", 8.0, horizon=60)
    seeds = list(range(24))
    res = nested_cv_hpo(DTCOATunable, ROVA_SEARCH_SPACE, cfg, seeds,
                        tag=tag, K_out=4, K_in=2, n_trials=12)
    return [r["test_mean_regret"] for r in res]


def test_independent_draws_pass():
    """Two independent MC draws (different tags, same estimand) must agree
    distributionally -- reproduces handoff I11 from live code (deterministic
    given the tags, so this is a regression test, not a flaky one)."""
    a, m = _draw("draw__full"), _draw("draw__rova")
    assert compare_distributions(a, m, rel_tol=0.15)["pass"], (a, m)


def test_drift_detected():
    """A systematically shifted independent draw must be rejected."""
    m = _draw("draw__rova")
    drifted = [x * 1.5 + 0.1 for x in _draw("draw__full")]
    assert not compare_distributions(drifted, m, rel_tol=0.15)["pass"]


def _write_ckpt(path, regrets):
    save_json(path, {"K_out": len(regrets), "K_in": 3, "n_trials": 80,
                     "results": [{"outer_fold": i, "test_mean_regret": r}
                                 for i, r in enumerate(regrets)]})


def _dir_hash(d):
    h = hashlib.sha256()
    for root, _, files in os.walk(d):
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(f.encode())
            h.update(open(p, "rb").read())
    return h.hexdigest()


def test_validate_experiment_passes_and_is_readonly():
    env, c = "collapse_moderate_coupled", 8.0
    with tempfile.TemporaryDirectory() as abl, tempfile.TemporaryDirectory() as main:
        _write_ckpt(os.path.join(abl, "hpo_checkpoints", f"{env}__c{c}__full.json"),
                    [0.20, 0.22, 0.19, 0.21, 0.20])
        _write_ckpt(os.path.join(main, f"{env}__c{c}__rova.json"),
                    [0.21, 0.20, 0.22, 0.19, 0.21])
        before = _dir_hash(main)
        ok, rep = validate_experiment(abl, main_raw=main, rel_tol=0.15,
                                      envs=[env], grid=[c], verbose=False)
        assert ok and rep["cells"][f"{env}__c{c}"]["pass"]
        assert _dir_hash(main) == before, "validator must not write into MAIN-001 tree"


def test_validate_experiment_fails_on_drift():
    env, c = "collapse_moderate_coupled", 8.0
    with tempfile.TemporaryDirectory() as abl, tempfile.TemporaryDirectory() as main:
        _write_ckpt(os.path.join(abl, "hpo_checkpoints", f"{env}__c{c}__full.json"),
                    [0.60, 0.62, 0.59, 0.61, 0.60])   # drifted high
        _write_ckpt(os.path.join(main, f"{env}__c{c}__rova.json"),
                    [0.20, 0.21, 0.19, 0.20, 0.22])
        ok, rep = validate_experiment(abl, main_raw=main, rel_tol=0.15,
                                      envs=[env], grid=[c], verbose=False)
        assert not ok and rep["problems"], "drift must fail loudly"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL FIX-4 VALIDATOR TESTS PASSED")
