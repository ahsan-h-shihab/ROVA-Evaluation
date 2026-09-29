"""Tests of the decision rules, factorial analysis and statistical helpers
(synthetic data with known answers).

    PYTHONPATH=. python -m rova_revision.tests.test_decision_rules
"""
import numpy as np

from rova_revision import config as C
from rova_revision.decision_rules import (collect_rows, crossover, dr3, factorial_cell, pairwise_rules,
                                          protocol_family_holm)
from rova_revision.stathelp import compare, rope_posterior


def test_crossover_located_between_grid_points():
    rng = np.random.default_rng(7001)
    grid = [0.5, 1.5, 3.5, 8.0, 20.0]
    true_gap = {c: 0.1 * (np.log(c) - np.log(2.5)) for c in grid}  # zero at c = 2.5
    diffs = {c: true_gap[c] + rng.normal(0, 0.01, 60) for c in grid}
    out = crossover(diffs, grid, 1000)
    assert len(out) == 1 and out[0]["bracket"] == [1.5, 3.5]
    assert abs(out[0]["crossing_c"] - 2.5) < 0.2, out
    lo, hi = out[0]["ci_percentile"]
    assert lo < 2.5 < hi and out[0]["fraction_resamples_crossing_in_bracket"] > 0.99


def _cell(delta, reject, mean_a, mean_b, ci=0.01):
    return {"status": "ok",
            "methods": {"A": {"mean_ci": [mean_a - ci, mean_a + ci]}, "B": {"mean_ci": [mean_b - ci, mean_b + ci]}},
            "families": {"F1": [{"a": "A", "b": "B", "wilcoxon_p": 1e-6 if reject else 0.5, "cliffs_delta": delta,
                                 "protocol_family_reject": reject, "code": "x"}]}}


def test_dr1_superiority_and_reversal_clause():
    envs, grid = ["e1", "e2", "e3"], [1.0]
    cells = {"e1|1.0": _cell(-0.8, True, 0.1, 0.2), "e2|1.0": _cell(-0.6, True, 0.1, 0.2),
             "e3|1.0": _cell(0.05, False, 0.15, 0.15)}
    out = pairwise_rules(cells, {}, [("A", "B")], grid, envs, 200)["A_vs_B"]
    assert out["per_cost_ratio"]["1.0"]["DR1"] == "A superior"
    cells["e3|1.0"] = _cell(0.3, True, 0.2, 0.1)  # significant reversal in the third environment
    out = pairwise_rules(cells, {}, [("A", "B")], grid, envs, 200)["A_vs_B"]
    assert out["per_cost_ratio"]["1.0"]["DR1"] == "no superiority claim"
    cells = {"e1|1.0": _cell(-0.8, True, 0.1, 0.2), "e2|1.0": _cell(-0.8, True, 0.1, 0.105, ci=0.01),
             "e3|1.0": _cell(0.0, False, 0.15, 0.15)}
    out = pairwise_rules(cells, {}, [("A", "B")], grid, envs, 200)["A_vs_B"]
    assert out["per_cost_ratio"]["1.0"]["DR1"] == "no superiority claim"  # CIs overlap in e2
    assert out["per_cost_ratio"]["1.0"]["DR6_environment_specific"] is False
    cells = {e: _cell(0.05, False, 0.1, 0.1) for e in ("e1|1.0", "e2|1.0", "e3|1.0")}
    out = pairwise_rules(cells, {}, [("A", "B")], grid, envs, 200)["A_vs_B"]
    assert out["DR8_no_practically_significant_difference"] is True


def _factorial_data(effect_of_propagation, n=60, seed=7002):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.1, 0.2, n)
    data = {}
    for s in C.ARCHITECTURES:
        data[C.architecture_name(s)] = base + (effect_of_propagation if "propagation" in s else 0.0) \
            + rng.normal(0, 1e-4, n)
    return data


def test_factorial_cell_effects_and_dr3():
    data = _factorial_data(0.1)
    cell = factorial_cell(data, 300, 0)
    assert len(cell["removal_tests"]) == 28 and len(cell["structurally_identical"]) == 4
    main = cell["effects"]["main"]
    assert abs(main["propagation"]["mean"] - 0.1) < 1e-3 and abs(main["memory"]["mean"]) < 1e-3
    assert all(abs(v["mean"]) < 1e-3 for v in cell["effects"]["two_way"].values())
    rows = [r for r in cell["removal_tests"] if r["component"] == "propagation"]
    assert len(rows) == 8 and all(r["cliffs_delta"] > 0.9 for r in rows)
    envs, grid = ["e1", "e2", "e3"], [1.0]
    by_cell = {}
    for e in envs:
        fc = factorial_cell(_factorial_data(0.1, seed=7010 + envs.index(e)), 300, 0)
        for r in fc["removal_tests"]:
            r["protocol_family_reject"] = r["wilcoxon_p"] < 1e-3
        by_cell[f"{e}|1.0"] = fc
    out = dr3(by_cell, grid, envs)
    assert out["propagation"]["1.0"]["outcome"] == "contributes"
    assert out["memory"]["1.0"]["outcome"] == "not detected at this budget"
    assert out["decay"]["1.0"]["settings_evaluated"] == 8


def test_rope_and_cvar():
    rng = np.random.default_rng(7003)
    a = rng.uniform(0, 1, 60)
    r = rope_posterior(a, a.copy(), 500)
    assert r["p_rope"] == 1.0
    r = rope_posterior(a, a + 5.0, 500)
    assert r["p_a_lower"] == 1.0
    c = compare(a + 0.1, a, 300, 0)
    assert abs(c["cvar95_diff"] - 0.1) < 1e-12 and c["cvar95_diff_ci_percentile"][0] <= 0.1 + 1e-12


def test_protocol_family_holm_skips_underscored_duplicates():
    rows = [{"a": "x", "b": "y", "wilcoxon_p": p, "code": "none", "cliffs_delta": 0.0} for p in (1e-6, 0.02, 0.5)]
    dup = {"a": "x", "b": "y", "wilcoxon_p": 1e-9, "code": "none", "cliffs_delta": 0.0}
    section = {"cell": {"families": {"F1": rows[:2]}, "_duplicate": [dup]}, "other": [rows[2]]}
    assert len(collect_rows(section)) == 3
    res = protocol_family_holm([section])
    assert res["n_tests"] == 3 and "protocol_family_reject" not in dup
    assert rows[0]["protocol_family_reject"] and not rows[2]["protocol_family_reject"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
