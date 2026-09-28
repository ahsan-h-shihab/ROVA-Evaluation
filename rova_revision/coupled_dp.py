"""Exact sequential (dynamic-programming) oracle for coupled environments (plan R7).

Given one simulated world, the base reliability and hazard trajectories and
the graph are exogenous (drawn when ``SimCore`` is constructed); the only
state that actions influence is, for each agent, the time since its last
failure, which matters only inside the cascade window ``tau``. Each agent's
state therefore takes tau + 1 values (0 = failed at the previous step, ...,
tau = no failure within the window).

The expected cost-to-go is multilinear in the agents' permission fractions,
so its minimum over the box [0, 1]^N is attained at a vertex; optimising over
levels {0, 5} per agent is exact. ``solve_full_action_set`` checks this
numerically on small instances.
"""
import itertools

import numpy as np

from dtcoa_val.sim_core.core import SimCore
from dtcoa_val.utils import make_rng, trial_seed


def world(cfg, seed, tag):
    """The simulated world exactly as ``run_trial`` constructs it."""
    if cfg.get("adversary") is not None:
        raise ValueError("coupled DP is defined for adversary-free worlds")
    sim = SimCore(cfg, make_rng(trial_seed(seed, 0, tag)))
    if sim.tau < 1:
        raise ValueError("coupled DP requires cascade_tau >= 1")
    return sim


class CoupledDP:
    def __init__(self, sim, H, U, levels=(0, 5)):
        self.sim = sim
        self.H, self.U = float(H), float(U)
        self.N, self.T, self.tau = sim.N, sim.T, sim.tau
        K = self.tau + 1
        self.states = np.array(list(itertools.product(range(K), repeat=self.N)), int)
        self.radix = K ** np.arange(self.N - 1, -1, -1)
        self.fbits = np.array(list(itertools.product((0, 1), repeat=self.N)), int)
        self.pi = np.array(list(itertools.product([a / 5.0 for a in levels], repeat=self.N)), float)
        nxt = np.where(self.fbits[None, :, :] == 1, 0, np.minimum(self.states[:, None, :] + 1, self.tau))
        self.next_idx = (nxt * self.radix).sum(-1)
        parents = np.zeros((self.N, self.N))
        for i, ps in enumerate(sim.parents):
            for j in ps:
                parents[i, j] += 1.0
        self.nfp = (self.states < self.tau).astype(float) @ parents.T
        self.s0 = int((np.full(self.N, self.tau) * self.radix).sum())
        self.S, self.M, self.F = len(self.states), len(self.pi), len(self.fbits)

    def _step_arrays(self, t):
        sim = self.sim
        h = np.clip(sim.h_base[:, t][None, :] + sim.coupling * self.nfp, 0.0, 1.0)
        r = np.clip(sim.r[:, t][None, :] * (1.0 - sim.coupling) ** self.nfp, 0.0, 1.0)
        det = r < sim.r_fail
        imm = (h * self.H - r * self.U) @ self.pi.T + (r * self.U).sum(axis=1, keepdims=True)
        p = np.where(det[:, None, :], 1.0, self.pi[None, :, :] * h[:, None, :])
        P = np.ones((self.S, self.M, self.F))
        for i in range(self.N):
            P *= np.where(self.fbits[None, None, :, i] == 1, p[:, :, i, None], 1.0 - p[:, :, i, None])
        return h, r, imm, P

    def _myopic_index(self, h, r):
        """Index into ``pi`` of the myopic oracle's joint action (level 0 when h*H >= r*U)."""
        bits = (h * self.H < r * self.U).astype(int)
        return (bits * (2 ** np.arange(self.N - 1, -1, -1))).sum(axis=1)

    def solve(self):
        """Backward induction. Returns total expected losses from the initial
        state for the optimal policy and for the myopic oracle, plus the
        per-step decision tables."""
        rows = np.arange(self.S)
        V_opt = np.zeros(self.S)
        V_my = np.zeros(self.S)
        self.a_opt = np.zeros((self.T, self.S), int)
        self.a_my = np.zeros((self.T, self.S), int)
        for t in range(self.T - 1, -1, -1):
            h, r, imm, P = self._step_arrays(t)
            Q = imm + np.einsum("smf,sf->sm", P, V_opt[self.next_idx])
            a = Q.argmin(axis=1)
            m = self._myopic_index(h, r)
            V_my_new = imm[rows, m] + np.einsum("sf,sf->s", P[rows, m], V_my[self.next_idx])
            V_opt, V_my = Q[rows, a], V_my_new
            self.a_opt[t], self.a_my[t] = a, m
        self.V_opt, self.V_my = float(V_opt[self.s0]), float(V_my[self.s0])
        return self.V_opt, self.V_my

    def disagreement(self):
        """Expected fraction of agent-steps, along the optimal policy's state
        distribution, at which the optimal and myopic actions differ."""
        rows = np.arange(self.S)
        mu = np.zeros(self.S)
        mu[self.s0] = 1.0
        total = 0.0
        for t in range(self.T):
            _h, _r, _imm, P = self._step_arrays(t)
            a, m = self.a_opt[t], self.a_my[t]
            diff = (self.pi[a] != self.pi[m]).sum(axis=1) / self.N
            total += float(mu @ diff)
            nxt = np.zeros(self.S)
            np.add.at(nxt, self.next_idx, mu[:, None] * P[rows, a])
            mu = nxt
        return total / self.T


def solve_world(cfg, seed, tag):
    sim = world(cfg, seed, tag)
    dp = CoupledDP(sim, cfg.get("H", 1.0), cfg.get("U", 0.3))
    v_opt, v_my = dp.solve()
    steps = dp.T * dp.N
    return {
        "seed": int(seed), "tag": tag, "N": dp.N, "T": dp.T, "tau": dp.tau,
        "V_opt": v_opt, "V_myopic": v_my,
        "gap_per_agent_step": (v_my - v_opt) / steps,
        "disagreement_rate": dp.disagreement(),
    }


def solve_full_action_set(sim, H, U):
    """Optimal expected loss over all 6^N joint levels (verification only)."""
    dp = CoupledDP(sim, H, U, levels=(0, 1, 2, 3, 4, 5))
    rows = np.arange(dp.S)
    V = np.zeros(dp.S)
    for t in range(dp.T - 1, -1, -1):
        _h, _r, imm, P = dp._step_arrays(t)
        Q = imm + np.einsum("smf,sf->sm", P, V[dp.next_idx])
        V = Q[rows, Q.argmin(axis=1)]
    return float(V[dp.s0])
