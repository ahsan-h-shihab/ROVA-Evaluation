"""Run profiles, experiment constants and the tag registry (plan section 3).

Tags seed the simulated worlds (``run_trial`` seeds each world with
``trial_seed(seed, 0, tag)``), so every tag used by the revision is built here
and nowhere else.
"""
from dataclasses import dataclass, field

from dtcoa_val.experiments.environments import ENVIRONMENTS, COST_RATIO_GRID, cfg_for

ENV_NAMES = list(ENVIRONMENTS)
GRID = list(COST_RATIO_GRID)
ARCHIVED_METHODS = ["static", "ewma", "rova"]
SUB_BUDGETS_FULL = (20, 40, 80)

# Scripted adversaries exactly as defined in dtcoa_val/experiments/campaign.py (V5).
ADVERSARY_PROFILES = {
    "ramp_defect": dict(type="ramp_defect", agents=[0], defect_t=150, spike=0.85, fake_obs=0.9),
    "collusion": dict(type="collusion", agents=[0, 1, 2], defect_t=150, spike=0.85, fake_obs=0.9),
    "poisoning": dict(type="poisoning", agents=[0, 1], defect_t=0, spike=0.6, fake_obs=0.9),
}

# Single-factor observation perturbations (R9), disabled in every frozen environment.
PERTURBATIONS = {
    "K1_mnar": dict(p_miss=0.10, mnar=True),
    "K2_delay": dict(delay=3),
    "K3_conflict": dict(conflict_frac=1.0 / 6.0),
}

# Sequential-oracle instance sizes (R7). Collapse N=6 is its full evaluated size.
DP_REDUCED_SIZES = {
    "favorable_hmm_coupled": [2, 3, 4, 5, 6],
    "collapse_moderate_coupled": [2, 3, 4, 5, 6],
}
DP_FULL_SIZE_ENV = "collapse_moderate_coupled"

OVERHEAD_N = [1, 6, 10, 100, 1000]


@dataclass(frozen=True)
class Profile:
    """A run profile. Only ``full`` is a scientific run; ``smoke`` is a plumbing
    check on non-experimental seeds and shrunken worlds, and ``pilot`` measures
    run time on non-experimental seeds at the full budget."""
    name: str
    seeds: tuple
    K_out: int
    K_in: int
    n_trials: int
    sub_budgets: tuple
    tpe_startup: int
    tag_prefix: str
    scientific: bool
    grid: tuple = tuple(GRID)
    cfg_overrides: dict = field(default_factory=dict)
    dp_sizes: dict = field(default_factory=dict)
    overhead_n: tuple = tuple(OVERHEAD_N)
    overhead_reps: int = 7
    n_boot: int = 5000


FULL = Profile(
    name="full", seeds=tuple(range(60)), K_out=5, K_in=3, n_trials=80,
    sub_budgets=SUB_BUDGETS_FULL, tpe_startup=20, tag_prefix="REV1", scientific=True,
    dp_sizes=DP_REDUCED_SIZES,
)

SMOKE = Profile(
    name="smoke", seeds=tuple(range(1000, 1006)), K_out=2, K_in=2, n_trials=4,
    sub_budgets=(2, 4), tpe_startup=2, tag_prefix="REV1_SMOKE", scientific=False,
    grid=(0.5, 20.0),
    cfg_overrides=dict(horizon=20, n_agents=3, graph_density=0.5, coupling=0.2),
    dp_sizes={"favorable_hmm_coupled": [2], "collapse_moderate_coupled": [2]},
    overhead_n=(1, 10), overhead_reps=2, n_boot=200,
)

PILOT = Profile(
    name="pilot", seeds=tuple(range(1000, 1060)), K_out=5, K_in=3, n_trials=80,
    sub_budgets=SUB_BUDGETS_FULL, tpe_startup=20, tag_prefix="REV1_PILOT", scientific=False,
    dp_sizes=DP_REDUCED_SIZES,
)

PROFILES = {p.name: p for p in (FULL, SMOKE, PILOT)}


def cell_cfg(profile, env, c, **extra):
    """Frozen environment config for (env, c), plus profile overrides and extras."""
    return cfg_for(env, c, **{**profile.cfg_overrides, **extra})


def adversary_valid(profile_name, env, profile):
    """A scripted profile runs only where every attacking agent exists."""
    n = cell_cfg(profile, env, 1.0).get("n_agents", 1)
    return max(ADVERSARY_PROFILES[profile_name]["agents"]) < n


# --------------------------------------------------------------------------- #
# Tag registry (plan section 3). Archived tags are reproduced byte-for-byte.
# --------------------------------------------------------------------------- #
def archived_tag(env, c, method):
    return f"{env}__c{c}__{method}"


def campaign_tag(profile, env, c, method):
    return f"{profile.tag_prefix}__{env}__c{c}__{method}"


def own_world_tag(profile, env, c, method, k):
    return f"{campaign_tag(profile, env, c, method)}_out{k}_test"


def cew_tag(profile, env, c, k):
    return f"{profile.tag_prefix}_CEW__{env}__c{c}_out{k}_test"


def adv_tag(profile, adv, env, c, k):
    return f"{profile.tag_prefix}_ADV__{adv}__{env}__c{c}_out{k}_test"


def advt_campaign_tag(profile, adv, env, c, method):
    return f"{profile.tag_prefix}_ADVT__{adv}__{env}__c{c}__{method}"


def advt_cew_tag(profile, adv, env, c, k):
    return f"{profile.tag_prefix}_ADVT_CEW__{adv}__{env}__c{c}_out{k}_test"


def sens_campaign_tag(profile, pert, env, c, method):
    return f"{profile.tag_prefix}_SENS__{pert}__{env}__c{c}__{method}"


def sens_cew_tag(profile, pert, env, c, k):
    return f"{profile.tag_prefix}_SENS_CEW__{pert}__{env}__c{c}_out{k}_test"


def dp_tag(profile, env, n, c):
    return f"{profile.tag_prefix}_DP__{env}__N{n}__c{c}"
