"""Frozen environment and cost-ratio grid definitions.

Per PRE_REGISTERED_PROTOCOL.md §5: minimum 3 environments, structurally
diverse along temporal persistence, propagation coupling, noise, and hazard
base rate, with at least one deliberately unfavorable to the governance
inductive bias (low persistence, low coupling).

Per the PI scope freeze: exactly 3 environments (not more), a sparse
cost-ratio grid (not dense). This module is the single source of truth for
both, so every experiment driver imports from here rather than
re-defining configs ad hoc.

Once any result is observed, these definitions are frozen (protocol §11
stopping criteria) -- do not add, remove, or reparameterize an environment
or grid point after seeing output.
"""
import copy

# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

_BASE = dict(
    horizon=250, obs_map="A", theta_h=0.5, r_fail=0.25,
)


def _env(**overrides):
    c = copy.deepcopy(_BASE)
    c.update(overrides)
    return c


ENVIRONMENTS = {
    # ENV_FAVORABLE: high temporal persistence (hmm regime, mean-reverting
    # regime-switching reliability) + meaningful multi-agent propagation
    # coupling. This is the setting the governance architecture's memory and
    # network-awareness components are explicitly designed for.
    "favorable_hmm_coupled": _env(
        regime="hmm", n_agents=10, graph_density=0.3, coupling=0.3,
        cascade_tau=2, noise=0.04, h_noise=0.03, p_shock=0.01,
    ),

    # ENV_UNFAVORABLE: deliberately unfavorable to the governance inductive
    # bias, per §5's explicit requirement. "oscillation" with a short period
    # is close to memoryless noise from the policy's perspective (today's
    # value is a poor predictor of tomorrow's beyond ~one period), and
    # coupling=0 / n_agents=1 removes any propagation-awareness benefit
    # entirely. If ROVA's memory/network machinery is worth anything, this
    # environment should show it winning least (or losing).
    "unfavorable_oscillation_uncoupled": _env(
        regime="oscillation", n_agents=1, coupling=0.0, graph_density=0.0,
        cascade_tau=1, noise=0.08, h_noise=0.05, p_shock=0.02,
        regime_params={"period": 8},
    ),

    # ENV_STRUCTURAL_CONTRAST: different hazard base rate and noise regime
    # from both of the above -- "collapse" (a single sharp reliability
    # breakpoint) with moderate coupling, to check whether conclusions from
    # the smooth/cyclical environments above generalize to a regime-shift
    # structure with different noise/hazard-rate characteristics.
    "collapse_moderate_coupled": _env(
        regime="collapse", n_agents=6, graph_density=0.2, coupling=0.15,
        cascade_tau=2, noise=0.05, h_noise=0.04, p_shock=0.015,
    ),
}

# ---------------------------------------------------------------------------
# Cost-ratio grid: c = H / U. U fixed at 1.0; H swept to realize each c.
# Re-derived (2026-07-04, rev. per CLAIM_GRANULARITY_AUDIT.md) from the
# actual intended claim -- "the verdict changes across the operating-cost
# spectrum," NOT a located crossover point c*. Design basis:
# - 5 points is the minimum for which "the pattern across the grid is
#   non-random" is even a testable claim: a permutation-style test over n
#   point-orderings has best-case exact p-value 2/n!; at n=4 that floor is
#   2/24=0.083 (cannot reach significance even in the best case), at n=5 it
#   is 2/120=0.017 (clears p<0.05). This is a combinatorial floor, not a
#   judgment call.
# - Per-point statistical power does NOT require more points: PILOT-001
#   found effect sizes (0.02-0.09 mean-regret differences) large relative
#   to per-point noise at n=60 seeds (BCa half-widths ~0.02-0.03) -- each
#   point is already well-resolved individually.
# - Placement is geometric (log-uniform) across the full range rather than
#   clustered near any presumed transition location, since clustering would
#   smuggle back in a c*-location goal the claim explicitly does not make.
# - PILOT-001's own non-monotonicity finding (0.5/3.33/5.0 -> 0.008/0.094/
#   0.057) is the concrete evidence that fewer than ~5 points risks
#   misreporting the qualitative pattern as noise or vice versa.
# ---------------------------------------------------------------------------

COST_RATIO_GRID = [0.5, 1.5, 3.5, 8.0, 20.0]


def cfg_for(env_name, cost_ratio, **extra_overrides):
    """Build a full run config for a given environment and cost ratio."""
    c = copy.deepcopy(ENVIRONMENTS[env_name])
    c["U"] = 1.0
    c["H"] = float(cost_ratio)
    c.update(extra_overrides)
    return c
