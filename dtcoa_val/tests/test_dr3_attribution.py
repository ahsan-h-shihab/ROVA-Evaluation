"""FIX-3 (ABLATION-001) DR3 cross-cell analysis tests. Run:
    python3 -m dtcoa_val.tests.test_dr3_attribution

Covers regime stratification, the H2 rollup, honest wording, and the
independent recompute-vs-driver cross-check.
"""
import os
import tempfile

from dtcoa_val.utils import save_json
from dtcoa_val.analysis import dr3_attribution as D


def test_regime_of_boundaries():
    assert D.regime_of(0.5) == "low"
    assert D.regime_of(1.5) == "low"
    assert D.regime_of(3.5) == "transition"
    assert D.regime_of(8.0) == "high"
    assert D.regime_of(20.0) == "high"


def _cell(env, c, attributed):
    return {"env": env, "cost_ratio": c, "regime": D.regime_of(c),
            "attribution": {comp: {"mechanism": D._MECHANISM[comp],
                                   "attributed": comp in attributed}
                            for comp in D.COMPONENTS}}


def test_regime_rollup_h2_sets_differ():
    cells = {
        "e__c0.5": _cell("e", 0.5, set()),                 # low: none
        "e__c1.5": _cell("e", 1.5, {"no_memory"}),          # low: memory
        "e__c8.0": _cell("e", 8.0, {"no_memory", "no_propagation"}),  # high
        "e__c20.0": _cell("e", 20.0, {"no_propagation"}),   # high
    }
    r = D._regime_rollup(cells)
    assert r["low_regime_contributing_set"] == ["memory"]
    assert r["high_regime_contributing_set"] == ["memory", "propagation"]
    assert r["sets_differ_between_regimes"] is True
    assert r["per_component"]["no_memory"]["low_attributed"] == 1
    assert r["per_component"]["no_propagation"]["high_attributed"] == 2


def test_markdown_wording_and_marks():
    cells = {"e__c8.0": _cell("e", 8.0, {"no_memory"})}
    summary = {"cells": cells, "n_cells": 1, "n_expected": 1,
               "regime_rollup": D._regime_rollup(cells)}
    md = D.render_markdown_table(summary)
    assert "not detected at this budget" in md
    assert "not proven inert" in md.lower() or "not evidence of inert" in D.NOT_DETECTED.lower()
    assert "| memory | A |" in md   # attributed mark
    assert "| decay | - |" in md    # not-detected mark


def _write_branch(ckpt_dir, env, c, branch, regrets):
    results = []
    for i, r in enumerate(regrets):
        d = {"outer_fold": i, "test_mean_regret": r}
        if branch == "full":
            d["best_config"] = {"prop_scale": 0.5, "stab_scale": 0.4}
        results.append(d)
    save_json(os.path.join(ckpt_dir, f"{env}__c{c}__{branch}.json"), {"results": results})


def test_build_summary_recompute_and_crosscheck():
    env, c = "collapse_moderate_coupled", 8.0
    with tempfile.TemporaryDirectory() as abl:
        ck = os.path.join(abl, "hpo_checkpoints")
        # All branches ~equal -> nothing attributed (equal arrays -> p=1.0).
        base = [0.20, 0.21, 0.19, 0.20, 0.22]
        for b in D.BRANCHES:
            _write_branch(ck, env, c, b, base)

        # Matching driver report (all not-attributed) -> 0 mismatches.
        drv = {comp: {"significant_degradation": False} for comp in D.COMPONENTS}
        save_json(os.path.join(abl, f"{env}__c{c}__ablation_report.json"), drv)
        summary = D.build_summary(abl, envs=[env], grid=[c])
        assert summary["n_cells"] == 1 and not summary["missing_cells"]
        assert summary["recompute_vs_driver_mismatches"] == []
        for comp in D.COMPONENTS:
            att = summary["cells"][f"{env}__c{c}"]["attribution"][comp]
            assert att["attributed"] is False
            assert "not detected" in att["interpretation"]

        # Conflicting driver report -> cross-check must flag the mismatch.
        drv["no_memory"]["significant_degradation"] = True
        save_json(os.path.join(abl, f"{env}__c{c}__ablation_report.json"), drv)
        summary2 = D.build_summary(abl, envs=[env], grid=[c])
        assert any("no_memory" in m for m in summary2["recompute_vs_driver_mismatches"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL FIX-3 DR3 TESTS PASSED")
