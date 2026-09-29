"""Matched-world design (plan section 3): a seed must generate the identical
simulated world for every compared method.

Non-experimental seeds (>= 5000) and shrunken worlds only.
    PYTHONPATH=. python -m rova_revision.tests.test_matched_worlds
"""
import json
import os
import shutil
import tempfile

import numpy as np

import dtcoa_val.experiments.runner as runner_mod
from dtcoa_val.baselines.baselines import EWMA, StaticThreshold
from dtcoa_val.dtcoa_tunable import DTCOATunable
from dtcoa_val.experiments.environments import cfg_for
from dtcoa_val.experiments.nested_hpo import nested_cv_hpo
from dtcoa_val.experiments.runner import run_trial
from dtcoa_val.experiments.search_space import ROVA_ABLATIONS, ROVA_SEARCH_SPACE, STATIC_THRESHOLD_SEARCH_SPACE

from rova_revision import config as C
from rova_revision import units as UN
from rova_revision.config import FULL, SMOKE
from rova_revision.hpo_record import nested_cv_hpo_record
from rova_revision.policies import ConstantLevel

SEEDS = [5300, 5301]


class LoggingRNG:
    """Delegates to a numpy Generator and logs every draw: method, arguments, values."""

    def __init__(self, rng, log):
        self._rng, self._log = rng, log

    def __getattr__(self, name):
        attr = getattr(self._rng, name)
        if not callable(attr):
            return attr

        def wrapped(*a, **k):
            out = attr(*a, **k)
            self._log.append((name, repr(a), repr(sorted(k.items())), np.asarray(out).tobytes()))
            return out
        return wrapped


def draw_log(cfg, policy, seed, tag):
    log = []
    orig = runner_mod.make_rng
    runner_mod.make_rng = lambda s: LoggingRNG(orig(s), log)
    try:
        out = run_trial(cfg, policy, seed=seed, tag=tag)
    finally:
        runner_mod.make_rng = orig
    return log, out


def _policies():
    return [ConstantLevel(0), ConstantLevel(5), StaticThreshold(), EWMA(), DTCOATunable()]


def _cfgs():
    out = []
    for env in C.ENV_NAMES:
        out.append((env, cfg_for(env, 3.5, horizon=60)))
    out.append(("adversary", cfg_for("favorable_hmm_coupled", 3.5, horizon=60, adversary=C.ADVERSARY_PROFILES["collusion"])))
    for name, pert in C.PERTURBATIONS.items():
        out.append((name, cfg_for("collapse_moderate_coupled", 3.5, horizon=60, **pert)))
    return out


def test_all_policies_consume_the_identical_random_stream_on_a_shared_tag():
    for label, cfg in _cfgs():
        tag = C.cew_tag(FULL, label, 3.5, 0)
        for s in SEEDS:
            logs, regrets = [], []
            for pol in _policies():
                log, out = draw_log(cfg, pol, s, tag)
                logs.append(log)
                regrets.append(out["regret"])
            assert all(lg == logs[0] for lg in logs[1:]), label
            assert len(logs[0]) > 100
            assert len(set(np.round(regrets, 12))) > 1, "policies must actually act differently"


def test_archived_method_specific_tags_gave_different_worlds():
    """Documents the defect being corrected: MAIN-001 test worlds were seeded with
    method-specific tags, so the same seed index produced different worlds."""
    for env in C.ENV_NAMES:
        cfg = cfg_for(env, 3.5, horizon=60)
        tags = [f"{C.archived_tag(env, 3.5, m)}_out0_test" for m in C.ARCHIVED_METHODS]
        logs = [draw_log(cfg, ConstantLevel(5), SEEDS[0], t)[0] for t in tags]
        assert logs[0] != logs[1] and logs[0] != logs[2] and logs[1] != logs[2]


def test_same_policy_under_two_method_labels():
    cfg = cfg_for("collapse_moderate_coupled", 8.0, horizon=60)
    cew = C.cew_tag(FULL, "collapse_moderate_coupled", 8.0, 0)
    a = [run_trial(cfg, ConstantLevel(5), seed=s, tag=cew)["regret"] for s in SEEDS]
    b = [run_trial(cfg, ConstantLevel(5), seed=s, tag=cew)["regret"] for s in SEEDS]
    assert a == b
    own = [run_trial(cfg, ConstantLevel(5), seed=s, tag=f"{C.archived_tag('collapse_moderate_coupled', 8.0, 'x')}_out0_test")["regret"]
           for s in SEEDS]
    own2 = [run_trial(cfg, ConstantLevel(5), seed=s, tag=f"{C.archived_tag('collapse_moderate_coupled', 8.0, 'y')}_out0_test")["regret"]
            for s in SEEDS]
    assert own != own2


def test_comparison_tags_contain_no_method_name():
    names = set(UN.CLEAN_METHODS) | {"static", "ewma", "rova", "always0", "always5", "full"}
    for prof in (FULL, SMOKE):
        tags = []
        for env in C.ENV_NAMES:
            for c in prof.grid:
                for k in range(prof.K_out):
                    tags.append(C.cew_tag(prof, env, c, k))
                    tags += [C.adv_tag(prof, a, env, c, k) for a in C.ADVERSARY_PROFILES]
                    tags += [C.advt_cew_tag(prof, a, env, c, k) for a in C.ADVERSARY_PROFILES]
                    tags += [C.sens_cew_tag(prof, p, env, c, k) for p in C.PERTURBATIONS]
        tags += [C.shift_cew_tag(prof, k) for k in range(prof.K_out)]
        for t in tags:
            fields = set(t.replace("__", "|").replace("_out", "|").split("|"))
            assert not (fields & names), (t, fields & names)


def _keys(recs):
    return sorted((r["outer_fold"], r["seed"], r["tag"]) for r in recs)


def test_eval_units_pair_every_method_on_identical_worlds():
    tmp = tempfile.mkdtemp(prefix="rova_mw_")
    try:
        res = UN.EXECUTORS["eval_cew"](SMOKE, {"env": "favorable_hmm_coupled", "c": 0.5}, tmp, tmp)
        recs = res["records"]
        assert {"static", "ewma", "rova", "A0", "A5"} <= set(recs)
        assert set(C.ABLATION_ARCHIVED_METHODS) <= set(recs), recs.keys()
        ref = _keys(recs["static"])
        assert all(_keys(v) == ref for v in recs.values())
        assert {t for _k, _s, t in ref} == {C.cew_tag(SMOKE, "favorable_hmm_coupled", 0.5, k) for k in range(SMOKE.K_out)}
        sh = UN.EXECUTORS["eval_shift_cew"](SMOKE, {}, tmp, tmp)["records"]
        ref = _keys(sh["static"])
        assert all(_keys(v) == ref for v in sh.values()) and len(ref) == SMOKE.K_out * len(SMOKE.seeds)
    finally:
        shutil.rmtree(tmp)


def test_factorial_registry():
    assert len(C.FACTORIAL_CELLS) == 16 and len(C.ARCHITECTURES) == 12
    assert len(C.FACTORIAL_NEW_METHODS) == 7 and len(C.ABLATION_ARCHIVED_METHODS) == 5
    for name, spec in ROVA_ABLATIONS.items():
        off = C.ARCHIVED_ABLATION_VARIANTS[name]
        dropped = {p for x in off for p in C.COMPONENT_PARAMS[x]}
        assert set(spec["search_space"]) == set(ROVA_SEARCH_SPACE) - dropped, name
    attr = {"use_trust": "use_trust", "use_decay": "use_decay", "use_propagation": "use_prop",
            "use_stability": "use_stab"}
    for m in C.FACTORIAL_NEW_METHODS:
        factory, space = UN.architecture_spec(m)
        off = UN.ARCH_BY_NAME[m]
        assert list(space) == [k for k in ROVA_SEARCH_SPACE if k not in {p for x in off for p in C.COMPONENT_PARAMS[x]}]
        pol = factory(**{k: (np.sort(np.random.default_rng(1).uniform(0.02, 0.98, 5)) if k == "bands" else 0.5)
                         for k in space if k != "k"})
        for x in C.COMPONENTS:
            assert getattr(pol, attr[C.COMPONENT_SWITCH[x]]) == (x not in off), (m, x)


def test_memory_off_makes_decay_inert():
    """With memory off, decay on and decay off act identically (why 16 cells hold 12 architectures)."""
    rng = np.random.default_rng(5500)
    for _ in range(20):
        conf = dict(e_min=rng.uniform(0.05, 0.6), context_risk=rng.uniform(0, 0.5),
                    bands=np.sort(rng.uniform(0.02, 0.98, 5)), prop_scale=rng.uniform(0, 5),
                    stab_scale=rng.uniform(0, 5), s_min=rng.uniform(0, 0.95))
        a = DTCOATunable(use_trust=False, use_decay=True, **conf)
        b = DTCOATunable(use_trust=False, use_decay=False, **conf)
        a.reset(4)
        b.reset(4)
        for _t in range(80):
            O = rng.uniform(0, 1, (4, 6))
            O[rng.random((4, 6)) < 0.1] = np.nan
            xa, ea = a.step(O.copy())
            xb, eb = b.step(O.copy())
            assert np.array_equal(xa, xb) and np.array_equal(ea, eb)


def test_hpo_record_reproduces_frozen_nested_cv_hpo():
    cfg = cfg_for("collapse_moderate_coupled", 3.5, horizon=20, n_agents=3)
    seeds = list(range(5600, 5609))
    for K_in, n_trials in ((3, 5), (2, 4)):
        frozen = nested_cv_hpo(StaticThreshold, STATIC_THRESHOLD_SEARCH_SPACE, cfg, seeds, K_out=3, K_in=K_in,
                               n_trials=n_trials, seed0=0, tag="unit_hpo")
        tmp = tempfile.mkdtemp(prefix="rova_hpo_")
        try:
            ck = os.path.join(tmp, "ck.json")
            rec = nested_cv_hpo_record(StaticThreshold, STATIC_THRESHOLD_SEARCH_SPACE, cfg, seeds, K_out=3,
                                       K_in=K_in, n_trials=n_trials, sub_budgets=(20, 40, 80), tag="unit_hpo",
                                       checkpoint_path=ck)
            with open(ck) as f:
                d = json.load(f)
            d["results"] = [r for r in d["results"] if r["outer_fold"] != 1]
            with open(ck, "w") as f:
                json.dump(d, f)
            resumed = nested_cv_hpo_record(StaticThreshold, STATIC_THRESHOLD_SEARCH_SPACE, cfg, seeds, K_out=3,
                                           K_in=K_in, n_trials=n_trials, sub_budgets=(20, 40, 80), tag="unit_hpo",
                                           checkpoint_path=ck)
        finally:
            shutil.rmtree(tmp)
        for a, b, r in zip(frozen, rec, resumed):
            for key in ("outer_fold", "best_config", "best_inner_score", "test_regrets", "test_mean_regret",
                        "tuning_curve"):
                assert json.dumps(a[key], sort_keys=True) == json.dumps(b[key], sort_keys=True), key
                assert json.dumps(b[key], sort_keys=True) == json.dumps(r[key], sort_keys=True), key


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
