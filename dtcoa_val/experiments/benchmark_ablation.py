"""ABLATION-001 runtime benchmark (pre-launch gate).

Times one FULL-budget ROVA campaign (nested_cv_hpo at K_out=5, K_in=3,
n_trials=80, n_seeds=60) per environment and extrapolates to the whole
experiment so the CPU-bound wall-clock can be confirmed on the target hardware
BEFORE committing to the full run (LAUNCH_PLAN step 8).

Extrapolation model
-------------------
The experiment is 3 envs x 5 cost ratios x 5 branches (full + 4 ablations).
Cost ratio changes H only (not the simulation size), so per-trial cost is
~constant across cost ratios. Ablation branches drop parameters and disable one
mechanism, so each is <= the full branch in cost; using the measured full-branch
time for all 5 branches is therefore a conservative (slight over-)estimate.

  total_core_seconds ~= sum_env( t_env_full * 5 branches * 5 cost_ratios )
                     =  25 * sum_env( t_env_full )

Run single-threaded BLAS (OMP/MKL/OPENBLAS = 1) exactly as the real launch will,
so the measured time reflects one core per campaign.
"""
import argparse
import time

from dtcoa_val.experiments.environments import ENVIRONMENTS, cfg_for
from dtcoa_val.experiments.search_space import ROVA_SEARCH_SPACE
from dtcoa_val.experiments.nested_hpo import nested_cv_hpo
from dtcoa_val.dtcoa_tunable import DTCOATunable

N_BRANCHES = 5      # full + 4 ablations
N_COST = 5          # frozen cost-ratio grid size
FULL_BUDGET = dict(K_out=5, K_in=3, n_trials=80)
N_SEEDS = 60


def time_full_branch(env, cost_ratio=8.0):
    cfg = cfg_for(env, cost_ratio)  # full scale, no overrides
    seeds = list(range(N_SEEDS))
    t0 = time.time()
    nested_cv_hpo(DTCOATunable, ROVA_SEARCH_SPACE, cfg, seeds,
                  tag=f"bench__{env}__full", **FULL_BUDGET)
    return time.time() - t0


def extrapolate(per_env_seconds, parallelisms=(15, 30)):
    total_core_s = N_BRANCHES * N_COST * sum(per_env_seconds.values())
    out = {
        "per_env_full_branch_seconds": per_env_seconds,
        "sum_env_seconds": sum(per_env_seconds.values()),
        "total_core_seconds": total_core_s,
        "total_core_hours": total_core_s / 3600.0,
        "wall_clock_hours_by_parallelism": {
            str(p): total_core_s / p / 3600.0 for p in parallelisms
        },
        "model": "25 * sum_env(t_env_full); conservative (ablations <= full)",
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="*", default=list(ENVIRONMENTS))
    ap.add_argument("--cost", type=float, default=8.0)
    args = ap.parse_args()
    per_env = {}
    for env in args.envs:
        dt = time_full_branch(env, args.cost)
        per_env[env] = dt
        print(f"[bench] {env} full-budget FULL branch: {dt:.1f} s")
    import json
    print(json.dumps(extrapolate(per_env), indent=2))
