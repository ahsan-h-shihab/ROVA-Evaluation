"""R6: measured per-decision cost of every policy (plan R6).

Times ``policy.step`` over a fixed synthetic observation stream, per agent-step,
single-threaded. Must run with no other workload on the machine; the
orchestrator runs it alone, after all other units.
"""
import time

import numpy as np

from dtcoa_val.baselines.baselines import StaticThreshold
from dtcoa_val.experiments.runner import run_trial

T_STEPS = 250
WARMUP = 2


def time_policy(factory, N, reps, O, truth_r, truth_h, needs_truth):
    per = []
    for rep in range(WARMUP + reps):
        p = factory()
        p.reset(N)
        t0 = time.perf_counter_ns()
        for t in range(T_STEPS):
            truth = {"r": truth_r[t], "h": truth_h[t]} if needs_truth else None
            p.step(O[t], truth=truth)
        dt = time.perf_counter_ns() - t0
        if rep >= WARMUP:
            per.append(dt / (T_STEPS * N))
    per = np.array(per)
    return {"median_ns_per_agent_step": float(np.median(per)),
            "q25": float(np.percentile(per, 25)), "q75": float(np.percentile(per, 75)),
            "reps": int(reps)}


def simulator_context(cfg_factory, N, reps):
    """Per-agent-step wall time of the frozen simulator + runner with a Static policy."""
    per = []
    for rep in range(WARMUP + reps):
        cfg = cfg_factory(N)
        t0 = time.perf_counter_ns()
        run_trial(cfg, StaticThreshold(), seed=424242, tag="REV1_overhead_context")
        dt = time.perf_counter_ns() - t0
        if rep >= WARMUP:
            per.append(dt / (cfg["horizon"] * N))
    return {"median_ns_per_agent_step": float(np.median(per)), "reps": int(reps)}


def benchmark(policies, n_list, reps, cfg_factory, needs_truth=(), context_n=(1, 6, 10)):
    rng = np.random.default_rng(20260929)
    out = {"T_steps": T_STEPS, "warmup": WARMUP, "reps": reps, "per_N": {}}
    for N in n_list:
        O = rng.uniform(0.0, 1.0, size=(T_STEPS, N, 6))
        tr = rng.uniform(0.0, 1.0, size=(T_STEPS, N))
        th = rng.uniform(0.0, 1.0, size=(T_STEPS, N))
        row = {name: time_policy(f, N, reps, O, tr, th, name in needs_truth) for name, f in policies.items()}
        static = row.get("static", {}).get("median_ns_per_agent_step")
        for name, rec in row.items():
            if static:
                rec["ratio_to_static"] = rec["median_ns_per_agent_step"] / static
                rec["compute_conditional_flag_gt3x"] = bool(rec["ratio_to_static"] > 3.0)
        if N in context_n:
            row["_simulator_and_runner_static"] = simulator_context(cfg_factory, N, max(1, reps // 2))
        out["per_N"][str(N)] = row
    return out
