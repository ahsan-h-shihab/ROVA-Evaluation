"""Exact sequential (dynamic-programming) oracle for coupled environments (plan R7).

Two properties are kept apart:

(a) One-step structure. The per-step oracle loss L(a; r, h) = pi(a) h H +
    (1 - pi(a)) r U is linear in pi(a) = a/5, so the MYOPIC oracle's per-step
    optimum is a level in {0, 5}. This says nothing about sequential optimality.

(b) Sequential optimum. Given one simulated world, the base reliability and
    hazard trajectories and the graph are exogenous (drawn when ``SimCore`` is
    constructed); actions influence the future only through each agent's time
    since its last failure, which matters inside the cascade window ``tau``
    (tau + 1 values per agent). ``FullGridDP`` computes the optimal expected
    total loss by backward induction over ALL 6^N joint authorization levels
    (exhaustive; no restriction of the action set), contracting the
    failure-pattern expectation agent by agent.

The {0, 5}^N ("vertex") solution is computed separately by ``CoupledDP`` and
reported next to the exhaustive one, per world, as an empirical check. The
transition model is validated against the frozen simulator by a Monte Carlo
unit test, and the exhaustive solver against naive enumeration.
"""
import itertools

import numpy as np

from dtcoa_val.sim_core.core import SimCore
from dtcoa_val.utils import make_rng, trial_seed

LEVELS = np.arange(6)
PI = LEVELS / 5.0


def world(cfg, seed, tag):
    """The simulated world exactly as ``run_trial`` constructs it."""
    if cfg.get("adversary") is not None:
        raise ValueError("coupled DP is defined for adversary-free worlds")
    sim = SimCore(cfg, make_rng(trial_seed(seed, 0, tag)))
    if sim.tau < 1:
        raise ValueError("coupled DP requires cascade_tau >= 1")
    return sim


def _linear(p):
    return p


class _Structure:
    """State space, transitions and per-step effective (h, r) of one world."""

    def __init__(self, sim, H, U):
        self.sim = sim
        self.H, self.U = float(H), float(U)
        self.N, self.T, self.tau = sim.N, sim.T, sim.tau
        K = self.tau + 1
        self.states = np.array(list(itertools.product(range(K), repeat=self.N)), int)
        self.radix = K ** np.arange(self.N - 1, -1, -1)
        self.fbits = np.array(list(itertools.product((0, 1), repeat=self.N)), int)
        nxt = np.where(self.fbits[None, :, :] == 1, 0, np.minimum(self.states[:, None, :] + 1, self.tau))
        self.next_idx = (nxt * self.radix).sum(-1)
        parents = np.zeros((self.N, self.N))
        for i, ps in enumerate(sim.parents):
            for j in ps:
                parents[i, j] += 1.0
        self.nfp = (self.states < self.tau).astype(float) @ parents.T
        self.s0 = int((np.full(self.N, self.tau) * self.radix).sum())
        self.S, self.F = len(self.states), len(self.fbits)

    def effective(self, t):
        sim = self.sim
        h = np.clip(sim.h_base[:, t][None, :] + sim.coupling * self.nfp, 0.0, 1.0)
        r = np.clip(sim.r[:, t][None, :] * (1.0 - sim.coupling) ** self.nfp, 0.0, 1.0)
        return h, r, r < sim.r_fail

    def pattern_probs(self, q):
        """P[s, f] for per-state, per-agent failure probabilities q[s, i]."""
        P = np.ones((self.S, self.F))
        for i in range(self.N):
            P *= np.where(self.fbits[None, :, i] == 1, q[:, i, None], 1.0 - q[:, i, None])
        return P


class FullGridDP(_Structure):
    """Backward induction over all 6^N joint levels (exhaustive).

    ``fail_frac`` and ``harm_frac`` map the permission fraction to the failure
    probability factor and to the harm-cost weight; both are the identity in the
    frozen simulator and are overridden only by falsification tests."""

    def __init__(self, sim, H, U, fail_frac=_linear, harm_frac=_linear):
        super().__init__(sim, H, U)
        self.fail_frac, self.harm_frac = fail_frac, harm_frac
        self.L = len(LEVELS)

    def _level_arrays(self, t):
        h, r, det = self.effective(t)
        q = np.where(det[:, :, None], 1.0, self.fail_frac(PI)[None, None, :] * h[:, :, None])  # (S, N, L)
        cost = self.harm_frac(PI)[None, None, :] * (h * self.H)[:, :, None] + \
            (1.0 - PI)[None, None, :] * (r * self.U)[:, :, None]  # (S, N, L)
        return h, r, q, cost

    def _q_table(self, q, cost, W):
        """Q[s, j] = immediate expected loss + expected continuation for every joint
        level j (agent 0's level is the least significant base-6 digit of j).

        Agents are contracted from the last to the first, each by a batched matrix
        product over its failure bit. Agent i's immediate cost is added right after
        its contraction; this is exact because the remaining agents' failure
        probabilities sum to one."""
        S, N, L = self.S, self.N, self.L
        X = W.reshape(S, 2 ** (N - 1), 2)
        lev = 1
        for i in range(N - 1, -1, -1):
            A = np.stack([1.0 - q[:, i, :], q[:, i, :]], axis=1)  # (S, 2, L)
            Y = X @ A
            Y += cost[:, i, None, :]
            lev *= L
            if i > 0:
                Y = Y.reshape(S, 2 ** (i - 1), 2, lev).transpose(0, 1, 3, 2).reshape(S, 2 ** (i - 1) * lev, 2)
            X = Y
        return X.reshape(S, L ** N)

    def _decode(self, joint):
        digits = np.empty(joint.shape + (self.N,), int)
        rem = joint.copy()
        for i in range(self.N):
            digits[..., i] = rem % self.L
            rem //= self.L
        return digits

    def _joint_probs_cost(self, levels, q, cost):
        """Pattern probabilities and immediate cost of one joint level per state."""
        rows = np.arange(self.S)[:, None]
        cols = np.arange(self.N)[None, :]
        qs = q[rows, cols, levels]
        return self.pattern_probs(qs), cost[rows, cols, levels].sum(axis=1)

    def solve(self):
        V = np.zeros(self.S)
        V_my = np.zeros(self.S)
        self.a_opt = np.zeros((self.T, self.S), np.int64)
        self.l_my = np.zeros((self.T, self.S, self.N), int)
        for t in range(self.T - 1, -1, -1):
            h, r, q, cost = self._level_arrays(t)
            Q = self._q_table(q, cost, V[self.next_idx])
            a = Q.argmin(axis=1)
            V_new = Q[np.arange(self.S), a]
            lm = np.where(h * self.H >= r * self.U, 0, 5)
            Pm, cm = self._joint_probs_cost(lm, q, cost)
            V_my = cm + (Pm * V_my[self.next_idx]).sum(axis=1)
            V = V_new
            self.a_opt[t], self.l_my[t] = a, lm
        self.V_opt, self.V_my = float(V[self.s0]), float(V_my[self.s0])
        return self.V_opt, self.V_my

    def occupancy_stats(self):
        """Along the optimal policy's state distribution: expected fraction of
        agent-steps at which the optimal level differs from the myopic level, and
        at which the optimal level is intermediate (1-4)."""
        mu = np.zeros(self.S)
        mu[self.s0] = 1.0
        dis = mid = 0.0
        for t in range(self.T):
            _h, _r, q, cost = self._level_arrays(t)
            lo = self._decode(self.a_opt[t])
            dis += float(mu @ (lo != self.l_my[t]).mean(axis=1))
            mid += float(mu @ ((lo > 0) & (lo < 5)).mean(axis=1))
            P, _c = self._joint_probs_cost(lo, q, cost)
            nxt = np.zeros(self.S)
            np.add.at(nxt, self.next_idx, mu[:, None] * P)
            mu = nxt
        return dis / self.T, mid / self.T


class CoupledDP(_Structure):
    """Backward induction restricted to ``levels`` per agent (default {0, 5});
    used as the vertex check and, with all six levels, as naive enumeration."""

    def __init__(self, sim, H, U, levels=(0, 5), fail_frac=_linear, harm_frac=_linear):
        super().__init__(sim, H, U)
        self.fail_frac, self.harm_frac = fail_frac, harm_frac
        self.pi = np.array(list(itertools.product([a / 5.0 for a in levels], repeat=self.N)), float)
        self.M = len(self.pi)

    def _step_arrays(self, t):
        h, r, det = self.effective(t)
        imm = (self.harm_frac(self.pi)[None, :, :] * (h * self.H)[:, None, :]).sum(-1) + \
            ((1.0 - self.pi)[None, :, :] * (r * self.U)[:, None, :]).sum(-1)
        p = np.where(det[:, None, :], 1.0, self.fail_frac(self.pi)[None, :, :] * h[:, None, :])
        P = np.ones((self.S, self.M, self.F))
        for i in range(self.N):
            P *= np.where(self.fbits[None, None, :, i] == 1, p[:, :, i, None], 1.0 - p[:, :, i, None])
        return h, r, imm, P

    def solve(self):
        V = np.zeros(self.S)
        for t in range(self.T - 1, -1, -1):
            _h, _r, imm, P = self._step_arrays(t)
            Q = imm + np.einsum("smf,sf->sm", P, V[self.next_idx])
            V = Q.min(axis=1)
        self.V_opt = float(V[self.s0])
        return self.V_opt


def solve_world(cfg, seed, tag):
    sim = world(cfg, seed, tag)
    H, U = cfg.get("H", 1.0), cfg.get("U", 0.3)
    dp = FullGridDP(sim, H, U)
    v_opt, v_my = dp.solve()
    dis, mid = dp.occupancy_stats()
    v_vertex = CoupledDP(sim, H, U).solve()
    steps = dp.T * dp.N
    return {
        "seed": int(seed), "tag": tag, "N": dp.N, "T": dp.T, "tau": dp.tau,
        "V_opt": v_opt, "V_myopic": v_my, "V_opt_vertex": v_vertex,
        "vertex_minus_full": v_vertex - v_opt,
        "gap_per_agent_step": (v_my - v_opt) / steps,
        "disagreement_rate": dis, "intermediate_level_rate": mid,
    }


def solve_naive_all_levels(sim, H, U, fail_frac=_linear, harm_frac=_linear):
    """Optimal expected loss by explicit enumeration of all 6^N joint levels (tests only)."""
    return CoupledDP(sim, H, U, levels=tuple(LEVELS), fail_frac=fail_frac, harm_frac=harm_frac).solve()
