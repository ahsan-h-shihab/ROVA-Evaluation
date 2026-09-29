"""Unit tests for the revision package.

All tests use NON-EXPERIMENTAL seeds (>= 5000) and shrunken worlds; they check
correctness and determinism only and produce no reported numbers.
Run: python -m rova_revision.tests.test_core_modules
"""
import copy

import numpy as np

from dtcoa_val.baselines.baselines import EWMA, OracleLite, StaticThreshold
from dtcoa_val.dtcoa_tunable import DTCOATunable
from dtcoa_val.experiments.environments import cfg_for
from dtcoa_val.experiments.nested_hpo import nested_cv_hpo
from dtcoa_val.experiments.runner import run_trial
from dtcoa_val.experiments.search_space import ROVA_SEARCH_SPACE, STATIC_THRESHOLD_SEARCH_SPACE
from dtcoa_val.oracles.oracles import AuthorizationOracle
from dtcoa_val.utils import make_rng

from rova_revision.config import SMOKE
from rova_revision.coupled_dp import CoupledDP, FullGridDP, solve_naive_all_levels, world
from rova_revision.instrument import RecordingOracle, evaluate_trial
from rova_revision.policies import (CappedEWMA, CappedStaticThreshold, ConstantLevel,
                                    LearnedCSPolicy, ObsFeatures, QLearnPolicy, RandomLevels,
                                    fit_cpwlr, rova_single)
from rova_revision.tpe import TPESampler, nested_cv_tpe
from rova_revision.training import nested_cv_trainable

TEST_SEEDS = list(range(5000, 5006))


def small_cfg(env="favorable_hmm_coupled", c=3.5, **kw):
    base = dict(horizon=25, n_agents=3, graph_density=0.6, coupling=0.3)
    base.update(kw)
    return cfg_for(env, c, **base)


def test_recording_oracle_is_transparent():
    """RecordingOracle must leave every frozen output bit-identical."""
    for cfg in (small_cfg(), small_cfg("unfavorable_oscillation_uncoupled", 0.5, n_agents=1, coupling=0.0),
                small_cfg(p_miss=0.1, mnar=True, delay=2)):
        for make in (lambda: StaticThreshold(), lambda: DTCOATunable(), lambda: ConstantLevel(5),
                     lambda: EWMA(), lambda: RandomLevels(make_rng(3))):
            for s in TEST_SEEDS[:3]:
                ref = run_trial(cfg, make(), seed=s, tag="unit_test")
                orc = RecordingOracle(H=cfg["H"], U=cfg["U"])
                new = run_trial(cfg, make(), seed=s, tag="unit_test", auth=orc)
                for k in ("regret", "harm_rate", "overrestrict", "unauth_unsafe_rate", "sysfail_rate", "esc_rate"):
                    assert ref[k] == new[k], (k, ref[k], new[k])
                assert orc.n == cfg["horizon"] * cfg.get("n_agents", 1)
                assert abs((orc.sum_loss - orc.sum_opt) / orc.n - ref["regret"]) < 1e-12


def test_oracle_is_bang_bang():
    rng = np.random.default_rng(5001)
    for _ in range(2000):
        H, U = rng.uniform(0.1, 20), rng.uniform(0.1, 2)
        a, _ = AuthorizationOracle(H=H, U=U).optimal(rng.uniform(), rng.uniform())
        assert a in (0, 5)


def _random_learned_model(degree, rng):
    d = ObsFeatures(0.8, degree).dim
    return {"beta": 0.8, "degree": degree, "mu": rng.normal(size=d).tolist(),
            "sd": (1 + rng.random(d)).tolist(), "theta": rng.normal(size=d + 1).tolist()}


def test_deployed_policies_ignore_latent_state():
    rng = np.random.default_rng(5002)
    makers = [lambda: ConstantLevel(0), lambda: CappedStaticThreshold(cap_code=3),
              lambda: CappedEWMA(cap_code=2), lambda: rova_single(0.5),
              lambda: LearnedCSPolicy(_random_learned_model(1, np.random.default_rng(1))),
              lambda: LearnedCSPolicy(_random_learned_model(2, np.random.default_rng(2))),
              lambda: QLearnPolicy({"n_bins": 4, "beta": 0.5, "Q": np.random.default_rng(3).normal(size=(16, 6)).tolist()})]
    O = rng.uniform(size=(30, 4, 6))
    O[rng.random(O.shape) < 0.1] = np.nan
    for make in makers:
        p1, p2 = make(), make()
        p1.reset(4)
        p2.reset(4)
        for t in range(30):
            a1, _ = p1.step(O[t], truth=None)
            a2, _ = p2.step(O[t], truth={"r": rng.uniform(size=4), "h": rng.uniform(size=4)})
            assert np.array_equal(a1, a2)


def test_capped_baselines():
    rng = np.random.default_rng(5003)
    O = rng.uniform(size=(20, 5, 6))
    for code, expect_zero in ((0, True), (24, False)):
        p, ref = CappedStaticThreshold(cap_code=code), StaticThreshold()
        p.reset(5)
        ref.reset(5)
        for t in range(20):
            a, _ = p.step(O[t])
            b, _ = ref.step(O[t])
            assert (np.all(a == 0)) if expect_zero else np.array_equal(a, b)


def test_rova_single_is_binary():
    cfg = small_cfg()
    p = rova_single(0.4, prop_scale=0.0, stab_scale=0.0)
    out = run_trial(cfg, p, seed=5004, tag="unit_test")
    levels = set(np.unique(np.round(out["_a_mean"] * cfg["n_agents"]).astype(int)).tolist())
    rec = RecordingOracle(H=cfg["H"], U=cfg["U"])
    run_trial(cfg, rova_single(0.4, prop_scale=0.0, stab_scale=0.0), seed=5004, tag="unit_test", auth=rec)
    assert rec.a_counts[1:5].sum() == 0, rec.a_counts


def test_fit_cpwlr():
    rng = np.random.default_rng(5005)
    X = rng.normal(size=(4000, 3))
    delta = X[:, 0] - 0.5 * X[:, 1]
    mu, sd, theta, diag = fit_cpwlr(X, delta, 1e-6)
    pred = ((X - mu) / sd) @ theta[:-1] + theta[-1] > 0
    assert diag["converged"] and np.mean(pred == (delta > 0)) > 0.97
    _, _, th, dg = fit_cpwlr(X, np.abs(delta) + 0.1, 1e-3)
    assert dg["degenerate"] and th[-1] > 0


class _Recorder:
    """Records the configurations nested_cv_hpo instantiates, in order."""
    log = []

    def __init__(self, **config):
        _Recorder.log.append(copy.deepcopy(config))
        self.p = StaticThreshold(**config)

    def reset(self, N):
        self.p.reset(N)

    def step(self, O, truth=None):
        return self.p.step(O, truth)


def test_candidate_streams_match_frozen():
    """The revision's candidate draws equal the frozen nested_cv_hpo draws."""
    cfg = small_cfg(horizon=5)
    seeds = list(range(5000, 5006))
    _Recorder.log = []
    nested_cv_hpo(_Recorder, STATIC_THRESHOLD_SEARCH_SPACE, cfg, seeds, K_out=2, K_in=2, n_trials=3, tag="unit_test")
    # nested_cv_hpo evaluates each candidate on each inner fold's validation seeds, then the best on test
    seen = []
    for c in _Recorder.log:
        key = tuple(np.round(np.asarray(c["thr"], float), 12))
        if key not in seen:
            seen.append(key)
    from dtcoa_val.experiments.search_space import sample_config
    rng = np.random.default_rng(0)
    expected = [tuple(np.round(sample_config(STATIC_THRESHOLD_SEARCH_SPACE, rng)["thr"], 12)) for _ in range(6)]
    assert seen[:6] == expected


def test_nested_trainable_learned_deterministic():
    cfg = small_cfg(horizon=15)
    r1 = nested_cv_trainable("learned", cfg, list(range(5000, 5006)), "unit_learned", SMOKE)
    r2 = nested_cv_trainable("learned", cfg, list(range(5000, 5006)), "unit_learned", SMOKE)
    assert len(r1) == SMOKE.K_out
    assert [f["test_regrets"] for f in r1] == [f["test_regrets"] for f in r2]
    assert all(np.isfinite(f["test_mean_regret"]) for f in r1)


def test_nested_trainable_qlearn_gate():
    cfg = small_cfg(horizon=15)
    res = nested_cv_trainable("qlearn", cfg, list(range(5000, 5006)), "unit_qlearn", SMOKE)
    assert res[0]["gate"]["G1_finite"] and res[0]["gate"]["G2_bounded"]
    assert res[0]["gate"]["G3_deterministic"]
    assert len(res[0]["training_seed_sensitivity"]) == 2


def test_tpe_startup_candidates_and_validity():
    cfg = small_cfg(horizon=5)
    seeds = list(range(5000, 5006))
    prof = SMOKE
    res = nested_cv_tpe(DTCOATunable, ROVA_SEARCH_SPACE, cfg, seeds, "unit_tpe", prof)
    assert len(res) == prof.K_out
    sampler = TPESampler(ROVA_SEARCH_SPACE, np.random.default_rng(5006))
    from dtcoa_val.experiments.search_space import sample_config
    rng = np.random.default_rng(5007)
    hist = [(sample_config(ROVA_SEARCH_SPACE, rng), float(rng.random())) for _ in range(12)]
    for _ in range(20):
        c = sampler.propose(hist)
        assert 1 <= c["k"] <= 10 and 0.05 <= c["gamma"] <= 0.99
        assert np.all(np.diff(c["bands"]) >= 0) and 0.02 <= c["bands"][0] and c["bands"][-1] <= 0.98
        assert c["prop_scale"] == 0.0 or 1e-3 <= c["prop_scale"] <= 5.0
        assert 1e-3 <= c["alpha"] <= 0.999
        DTCOATunable(**c)


def test_dp_uncoupled_equals_myopic():
    cfg = small_cfg(horizon=15, n_agents=3, coupling=0.0, graph_density=0.0)
    for s in TEST_SEEDS[:3]:
        dp = FullGridDP(world(cfg, s, "unit_dp"), cfg["H"], cfg["U"])
        v_opt, v_my = dp.solve()
        assert abs(v_opt - v_my) < 1e-9


def test_dp_exhaustive_solver_matches_naive_enumeration():
    """The agent-by-agent contraction over all 6^N joint levels equals explicit
    enumeration of every joint action (N = 1, 2, 3; both coupled environments)."""
    for env in ("favorable_hmm_coupled", "collapse_moderate_coupled"):
        for n in (1, 2, 3):
            for c in (0.5, 3.5, 20.0):
                cfg = small_cfg(env, c=c, horizon=8, n_agents=n, coupling=0.4, graph_density=1.0)
                for s in TEST_SEEDS[:2]:
                    sim = world(cfg, s, "unit_dp")
                    v_full, v_my = FullGridDP(sim, cfg["H"], cfg["U"]).solve()
                    v_naive = solve_naive_all_levels(sim, cfg["H"], cfg["U"])
                    assert abs(v_full - v_naive) < 1e-9 * max(1.0, abs(v_naive)), (env, n, c, s, v_full, v_naive)
                    assert v_full <= v_my + 1e-12


def test_dp_q_table_matches_explicit_evaluation():
    """Every entry of the contracted Q table equals the explicit expected loss of the
    decoded joint levels (checks the contraction order and the index decoding)."""
    rng = np.random.default_rng(5100)
    for n in (1, 2, 3):
        cfg = small_cfg("collapse_moderate_coupled", c=3.5, horizon=6, n_agents=n, coupling=0.4, graph_density=1.0)
        dp = FullGridDP(world(cfg, 5001, "unit_dp"), cfg["H"], cfg["U"])
        _h, _r, q, cost = dp._level_arrays(2)
        W = rng.random((dp.S, dp.F))
        Q = dp._q_table(q, cost, W)
        for _ in range(40):
            s, j = int(rng.integers(dp.S)), int(rng.integers(6 ** n))
            lv = dp._decode(np.array(j))
            qs = q[s, np.arange(n), lv]
            probs = np.prod(np.where(dp.fbits == 1, qs, 1.0 - qs), axis=1)
            expected = cost[s, np.arange(n), lv].sum() + probs @ W[s]
            assert abs(Q[s, j] - expected) < 1e-12, (n, s, j)


def test_dp_occupancy_uncoupled_is_myopic():
    cfg = small_cfg(horizon=12, n_agents=3, coupling=0.0, graph_density=0.0)
    dp = FullGridDP(world(cfg, 5002, "unit_dp"), cfg["H"], cfg["U"])
    dp.solve()
    dis, mid = dp.occupancy_stats()
    assert dis == 0.0 and mid == 0.0, (dis, mid)


def test_dp_vertex_check_on_frozen_model():
    """Under the frozen simulator's structure, the {0,5}^N solution equals the
    exhaustive optimum on these instances (an empirical check, reported per world)."""
    for env in ("favorable_hmm_coupled", "collapse_moderate_coupled"):
        for n in (2, 3, 4):
            cfg = small_cfg(env, c=3.5, horizon=10, n_agents=n, coupling=0.4, graph_density=1.0)
            for s in TEST_SEEDS[:2]:
                sim = world(cfg, s, "unit_dp")
                v_full, _ = FullGridDP(sim, cfg["H"], cfg["U"]).solve()
                v_vertex = CoupledDP(sim, cfg["H"], cfg["U"]).solve()
                assert abs(v_vertex - v_full) < 1e-9 * max(1.0, abs(v_full)), (env, n, s, v_vertex, v_full)


def test_dp_vertex_check_detects_non_vertex_optimum():
    """Falsification: with a convex harm weight and failure factor (pi^2), interior
    levels become optimal; the exhaustive solver finds them, the vertex solution
    is strictly worse, and naive enumeration agrees with the exhaustive solver."""
    sq = lambda p: np.asarray(p) ** 2
    gaps = []
    for c in (1.5, 3.5):
        cfg = small_cfg("collapse_moderate_coupled", c=c, horizon=8, n_agents=2, coupling=0.4, graph_density=1.0)
        for s in TEST_SEEDS[:3]:
            sim = world(cfg, s, "unit_dp")
            dp = FullGridDP(sim, cfg["H"], cfg["U"], fail_frac=sq, harm_frac=sq)
            v_full, _ = dp.solve()
            v_naive = solve_naive_all_levels(sim, cfg["H"], cfg["U"], fail_frac=sq, harm_frac=sq)
            v_vertex = CoupledDP(sim, cfg["H"], cfg["U"], fail_frac=sq, harm_frac=sq).solve()
            assert abs(v_full - v_naive) < 1e-9 * max(1.0, abs(v_naive))
            gaps.append(v_vertex - v_full)
    assert min(gaps) >= -1e-12 and max(gaps) > 1e-3, gaps


class _MyopicOracle:
    """Test-only policy acting as the myopic oracle (reads the latent state)."""

    def __init__(self, H, U):
        self.H, self.U = H, U

    def reset(self, N):
        self.N = N

    def step(self, O, truth=None):
        A = np.where(truth["h"] * self.H < truth["r"] * self.U, 5, 0)
        return A, (A == 0)


def test_dp_transition_model_matches_frozen_simulator():
    """Monte Carlo: realized loss of the myopic oracle in the frozen runner is an
    unbiased estimate of the DP's exact V_myopic for the same world."""
    cfg = small_cfg(c=8.0, horizon=20, n_agents=3, coupling=0.4, graph_density=0.8)
    diffs = []
    for s in range(5000, 5300):
        dp = FullGridDP(world(cfg, s, "unit_mc"), cfg["H"], cfg["U"])
        _, v_my = dp.solve()
        rec = evaluate_trial(cfg, _MyopicOracle(cfg["H"], cfg["U"]), s, "unit_mc")
        diffs.append(rec["sum_loss"] - v_my)
    diffs = np.array(diffs)
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    assert abs(diffs.mean()) < 4 * se + 1e-9, (diffs.mean(), se)


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    run_all()
