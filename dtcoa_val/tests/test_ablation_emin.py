"""FIX-1 (ABLATION-001) regression tests. Run: python3 -m dtcoa_val.tests.test_ablation_emin

The observation-failure threshold `e_min` must remain tunable in the no_memory
and no_decay ablations.

Rationale (EMIN_ANALYSIS): e_min feeds two independent code paths in
dtcoa_tunable.py -- a memory/decay path (the `clow` accumulator, consumed only
inside `if self.use_trust:`) and a propagation path (`obs_fail = E < e_min`
-> prop_pen, gated by use_prop / use_ma). Disabling memory or decay kills only
the first path; the second stays active in COUPLED (multi-agent) environments,
so e_min still affects regret there and must be re-tuned. In the single-agent
UNCOUPLED environment both paths are dead (FAF == 1 with n_agents == 1), so
e_min is exactly inert and keeping it tunable is harmless.
"""
import numpy as np

from dtcoa_val.experiments.environments import cfg_for
from dtcoa_val.experiments.search_space import (
    ROVA_ABLATIONS, ROVA_SEARCH_SPACE, sample_config, make_ablation_factory,
)
from dtcoa_val.experiments.nested_hpo import _eval_config
from dtcoa_val.dtcoa_tunable import DTCOATunable

E_MIN_GRID = [0.05, 0.25, 0.45, 0.60]
_ABLATIONS = ["no_memory", "no_decay"]


def _sweep_regret(env_name, ablation_name, cfg_overrides, seeds):
    """Mean regret vs e_min, holding EVERYTHING else (incl. the run tag, hence
    the Monte-Carlo draw) fixed. Only e_min varies, so any change is causal."""
    factory = make_ablation_factory(
        DTCOATunable, ROVA_ABLATIONS[ablation_name]["fixed_kwargs"]
    )
    cfg = cfg_for(env_name, 8.0, **cfg_overrides)  # high cost amplifies prop penalty
    base = sample_config(ROVA_SEARCH_SPACE, np.random.default_rng(12345))
    fixed_tag = f"emin_test_{env_name}_{ablation_name}"
    out = []
    for e in E_MIN_GRID:
        conf = dict(base, e_min=e)
        out.append(_eval_config(factory, conf, cfg, seeds, tag=fixed_tag)["mean_regret"])
    return np.asarray(out)


def test_emin_is_tunable_in_memory_and_decay_ablations():
    """Structural: FIX-1 requires e_min in these ablations' search spaces."""
    for ablation in _ABLATIONS:
        assert "e_min" in ROVA_ABLATIONS[ablation]["search_space"], (
            f"{ablation} must keep e_min tunable (FIX-1); it feeds the "
            f"propagation path independent of the memory/decay switch."
        )


def test_emin_active_in_coupled_environment():
    """Behavioral: e_min must move regret in a coupled env under these
    ablations -- proving it is NOT inert, so freezing it would confound."""
    seeds = list(range(8))
    for ablation in _ABLATIONS:
        reg = _sweep_regret("collapse_moderate_coupled", ablation,
                            dict(horizon=80), seeds)
        spread = float(reg.max() - reg.min())
        assert spread > 1e-3, (
            f"e_min appears inert under {ablation} in the coupled env "
            f"(spread={spread:.2e}); FIX-1's premise would be false. regrets={reg}"
        )


def test_emin_exactly_inert_in_single_agent_uncoupled():
    """Behavioral: e_min must be EXACTLY inert (bit-identical regret) in the
    single-agent uncoupled env, confirming FIX-1 is harmless there."""
    seeds = list(range(8))
    for ablation in _ABLATIONS:
        reg = _sweep_regret("unfavorable_oscillation_uncoupled", ablation,
                            dict(horizon=80), seeds)
        spread = float(reg.max() - reg.min())
        assert spread == 0.0, (
            f"e_min should be exactly inert under {ablation} in the single-agent "
            f"uncoupled env, but regret varied (spread={spread:.2e}). regrets={reg}"
        )


if __name__ == "__main__":
    test_emin_is_tunable_in_memory_and_decay_ablations()
    print("PASS test_emin_is_tunable_in_memory_and_decay_ablations")
    test_emin_active_in_coupled_environment()
    print("PASS test_emin_active_in_coupled_environment")
    test_emin_exactly_inert_in_single_agent_uncoupled()
    print("PASS test_emin_exactly_inert_in_single_agent_uncoupled")
    print("ALL FIX-1 TESTS PASSED")
