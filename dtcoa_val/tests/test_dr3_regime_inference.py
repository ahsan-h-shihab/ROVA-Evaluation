"""DR3 regime-inference tests (Option B + C). Run:
    python3 -m dtcoa_val.tests.test_dr3_regime_inference

Validates the statistics against known closed-form values and end-to-end
recovery of a planted attribution effect from synthetic RAW checkpoints, plus
determinism (reproducible from raw alone).
"""
import os
import tempfile

import numpy as np

from dtcoa_val.utils import save_json
from dtcoa_val.analysis import dr3_regime_inference as R


# ---- closed-form statistic checks ---------------------------------------- #
def test_signflip_permutation_known_values():
    # 6 identical positive deltas -> only all-+ and all-- reach |mean| -> 2/64.
    r = R.signflip_permutation([0.5] * 6)
    assert abs(r["p"] - 2 / 64) < 1e-9 and r["exact"]
    # a delta of 0 among them cannot change the extreme count
    assert R.signflip_permutation([1, 1, 1])["p"] == 2 / 8


def test_sign_test_known_values():
    assert abs(R.sign_test([1, 1, 1, 1, 1, 1])["p"] - 2 * (1 / 64)) < 1e-9
    assert R.sign_test([1, -1, 1, -1])["p"] == 1.0


def test_benjamini_hochberg():
    # smallest p=0.01 passes 0.05*1/4=0.0125; others fail
    assert R._benjamini_hochberg([0.01, 0.30, 0.60, 0.90]) == [True, False, False, False]
    assert R._benjamini_hochberg([0.9, 0.9, 0.9, 0.9]) == [False, False, False, False]


# ---- stage-1 effect on synthetic checkpoints ----------------------------- #
def _write_branch(ck, env, c, branch, per_fold_trials):
    res = [{"outer_fold": i, "test_regrets": list(tr),
            "test_mean_regret": float(np.mean(tr))} for i, tr in enumerate(per_fold_trials)]
    save_json(os.path.join(ck, f"{env}__c{c}__{branch}.json"), {"results": res})


def _cell(ck, env, c, worse=()):
    """full = 0.20 everywhere; branches in `worse` set to 0.50 (delta=+1)."""
    folds_full = [[0.20] * 4 for _ in range(5)]
    _write_branch(ck, env, c, "full", folds_full)
    for comp in R.COMPONENTS:
        val = 0.50 if comp in worse else 0.20
        _write_branch(ck, env, c, comp, [[val] * 4 for _ in range(5)])


def test_end_to_end_recovery_high_regime_only():
    """Plant a strong 'memory' effect in HIGH cells only; expect it recovered in
    high (delta=+1, small perm p), absent in low, and H2 flags the difference."""
    envs = list(R.ENVIRONMENTS); grid = [0.5, 1.5, 8.0, 20.0]  # 2 low + 2 high per env
    with tempfile.TemporaryDirectory() as out:
        ck = os.path.join(out, "hpo_checkpoints")
        for env in envs:
            for c in grid:
                _cell(ck, env, c, worse=("no_memory",) if R.regime_of(c) == "high" else ())
        s = R.build(ck, envs=envs, grid=grid)

        hi = s["per_component"]["no_memory"]["by_regime"]["high"]
        lo = s["per_component"]["no_memory"]["by_regime"]["low"]
        assert hi["n_cells"] == 6 and lo["n_cells"] == 6
        assert abs(hi["mean_delta"] - 1.0) < 1e-9      # removal strictly worsens
        assert abs(lo["mean_delta"] - 0.0) < 1e-9
        assert abs(hi["permutation_optionC"]["p"] - 2 / 64) < 1e-9
        assert hi["sign_test"]["p"] < 0.05 and lo["sign_test"]["p"] == 1.0
        assert hi["non_negligible_cells"] == 6 and lo["non_negligible_cells"] == 0
        # env-cluster bootstrap CI for high excludes 0 (all envs show the effect)
        assert hi["env_cluster_bootstrap_ci"][0] > 0.5
        # H2: contributing set differs (memory high but not low)
        h2 = s["per_component"]["no_memory"]["H2_low_vs_high"]
        assert h2["delta_high_minus_low"] == 1.0 and h2["p"] < 0.01
        # a null component shows nothing anywhere
        nul = s["per_component"]["no_stability"]["by_regime"]["high"]
        assert abs(nul["mean_delta"]) < 1e-9 and nul["permutation_optionC"]["p"] == 1.0


def test_determinism_reproducible_from_raw():
    envs = list(R.ENVIRONMENTS); grid = [0.5, 8.0]
    with tempfile.TemporaryDirectory() as out:
        ck = os.path.join(out, "hpo_checkpoints")
        for env in envs:
            for c in grid:
                _cell(ck, env, c, worse=("no_decay",) if c == 8.0 else ())
        a = R.build(ck, envs=envs, grid=grid)
        b = R.build(ck, envs=envs, grid=grid)
        import json
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("ALL DR3 REGIME-INFERENCE TESTS PASSED")
