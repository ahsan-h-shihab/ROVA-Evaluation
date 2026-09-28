"""Policies added by the revision (plan R1, R3, R4, R8, R12).

Every deployed policy implements the frozen interface ``reset(N)`` and
``step(O, truth=None) -> (A, ESC)`` and uses only the observation matrix ``O``.
The frozen runner also passes the latent state as ``truth``; deployed policies
ignore it. Only the training wrappers at the end of this module read it, to
compute training targets and rewards.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from dtcoa_val.baselines.baselines import StaticThreshold, EWMA
from dtcoa_val.dtcoa_tunable import DTCOATunable

N_LEVELS = 6
OBS_INIT = 0.6   # carry-forward initial value used by the frozen baselines


# --------------------------------------------------------------------------- #
# R1: constant reference policies
# --------------------------------------------------------------------------- #
class ConstantLevel:
    """Grants the same authorization level to every agent at every step."""

    def __init__(self, level):
        self.level = int(level)

    def reset(self, N):
        self.N = N

    def step(self, O, truth=None):
        A = np.full(O.shape[0], self.level, int)
        return A, (A == 0)


# --------------------------------------------------------------------------- #
# R8: baselines whose search space can reach full restriction
# --------------------------------------------------------------------------- #
def _cap(cap_code):
    return min(int(cap_code), 5)


class CappedStaticThreshold(StaticThreshold):
    """Frozen StaticThreshold with a maximum permitted level min(cap_code, 5)."""

    def __init__(self, thr=(0.2, 0.36, 0.52, 0.68, 0.88), cap_code=24):
        super().__init__(thr)
        self.cap = _cap(cap_code)

    def step(self, O, truth=None):
        A, _ = super().step(O, truth)
        A = np.minimum(A, self.cap)
        return A, (A == 0)


class CappedEWMA(EWMA):
    """Frozen EWMA with a maximum permitted level min(cap_code, 5)."""

    def __init__(self, beta=0.7, thr=(0.2, 0.36, 0.52, 0.68, 0.88), cap_code=24):
        super().__init__(beta, thr)
        self.cap = _cap(cap_code)

    def step(self, O, truth=None):
        A, _ = super().step(O, truth)
        A = np.minimum(A, self.cap)
        return A, (A == 0)


# --------------------------------------------------------------------------- #
# R12: ROVA with a single threshold instead of a 5-band vector
# --------------------------------------------------------------------------- #
def rova_single(b, **config):
    """DTCOATunable with all five bands equal to ``b``: authorization is 0 or 5."""
    return DTCOATunable(bands=(float(b),) * 5, **config)


# --------------------------------------------------------------------------- #
# R3: learned cost-sensitive policy
# --------------------------------------------------------------------------- #
class ObsFeatures:
    """Per-agent observation features for the learned policy (plan R3)."""
    N_BASE = 13

    def __init__(self, beta, degree, window=5):
        self.beta = float(beta)
        self.degree = int(degree)
        self.window = int(window)
        self._iu = np.triu_indices(self.N_BASE)

    @property
    def dim(self):
        return self.N_BASE if self.degree == 1 else self.N_BASE + len(self._iu[0])

    def reset(self, N):
        self.last = np.full((N, 6), OBS_INIT)
        self.e = np.full((N, 6), OBS_INIT)
        self.hist = []

    def update(self, O):
        z = np.array(O, dtype=float, copy=True)
        missing = np.isnan(z)
        z[missing] = self.last[missing]
        self.last = z
        self.e = self.beta * self.e + (1.0 - self.beta) * z
        self.hist.append(z.mean(axis=1))
        self.hist = self.hist[-self.window:]
        if len(self.hist) > 1:
            v = np.std(np.array(self.hist), axis=0)
        else:
            v = np.zeros(z.shape[0])
        base = np.hstack([z, self.e, v[:, None]])
        if self.degree == 1:
            return base
        return np.hstack([base, base[:, self._iu[0]] * base[:, self._iu[1]]])


def featurize(trajectories, beta, degree, H, U):
    """Stack features and cost differences Delta = h*H - r*U from recorded rollouts."""
    Xs, Ds = [], []
    for tr in trajectories:
        f = ObsFeatures(beta, degree)
        f.reset(tr["O"].shape[1])
        for t in range(tr["O"].shape[0]):
            Xs.append(f.update(tr["O"][t]))
        Ds.append((tr["h"] * H - tr["r"] * U).reshape(-1))
    return np.vstack(Xs), np.concatenate(Ds)


def fit_cpwlr(X, delta, l2):
    """Cost-proportionate weighted logistic regression (plan R3).

    Label: restrict iff Delta > 0; weight |Delta|; objective normalised by the
    total weight, plus l2 * ||w||^2 on non-intercept weights. Returns
    (mu, sd, theta) with theta[-1] the intercept. Degenerate data give a
    constant policy (intercept only).
    """
    X = np.asarray(X, float)
    delta = np.asarray(delta, float)
    y = np.where(delta > 0, 1.0, -1.0)
    w = np.abs(delta)
    W = w.sum()
    d = X.shape[1]
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    if W <= 0 or len(np.unique(y[w > 0])) < 2:
        theta = np.zeros(d + 1)
        theta[-1] = 1.0 if (w * (y > 0)).sum() >= (w * (y < 0)).sum() else -1.0
        return mu, sd, theta, {"degenerate": True, "iterations": 0, "converged": True}
    Xa = np.hstack([(X - mu) / sd, np.ones((X.shape[0], 1))])

    def objective(theta):
        z = y * (Xa @ theta)
        loss = float((w * np.logaddexp(0.0, -z)).sum() / W + l2 * np.dot(theta[:-1], theta[:-1]))
        g = Xa.T @ (w * (-y) * expit(-z)) / W
        g[:-1] += 2.0 * l2 * theta[:-1]
        return loss, g

    res = minimize(objective, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
    return mu, sd, res.x, {"degenerate": False, "iterations": int(res.nit),
                           "converged": bool(res.success), "objective": float(res.fun)}


class LearnedCSPolicy:
    """Deployed learned policy: level 0 if the fitted score is positive, else 5."""

    def __init__(self, model):
        self.beta = float(model["beta"])
        self.degree = int(model["degree"])
        self.mu = np.asarray(model["mu"], float)
        self.sd = np.asarray(model["sd"], float)
        theta = np.asarray(model["theta"], float)
        self.w, self.b = theta[:-1], float(theta[-1])

    def reset(self, N):
        self.N = N
        self.f = ObsFeatures(self.beta, self.degree)
        self.f.reset(N)

    def step(self, O, truth=None):
        X = self.f.update(O)
        score = ((X - self.mu) / self.sd) @ self.w + self.b
        A = np.where(score > 0, 0, 5).astype(int)
        return A, (A == 0)


# --------------------------------------------------------------------------- #
# R4: tabular Q-learning policy
# --------------------------------------------------------------------------- #
class QLearnPolicy:
    """Deployed greedy tabular policy; ties resolve to the lowest level."""

    def __init__(self, model):
        self.n_bins = int(model["n_bins"])
        self.beta = float(model["beta"])
        self.Q = np.asarray(model["Q"], float).reshape(self.n_bins * self.n_bins, N_LEVELS)

    def reset(self, N):
        self.N = N
        self.last = np.full((N, 6), OBS_INIT)
        self.ce = np.full(N, OBS_INIT)

    def encode(self, O):
        z = np.array(O, dtype=float, copy=True)
        missing = np.isnan(z)
        z[missing] = self.last[missing]
        self.last = z
        c = z.mean(axis=1)
        self.ce = self.beta * self.ce + (1.0 - self.beta) * c
        nb = self.n_bins
        b1 = np.minimum((np.clip(c, 0.0, 1.0) * nb).astype(int), nb - 1)
        b2 = np.minimum((np.clip(self.ce, 0.0, 1.0) * nb).astype(int), nb - 1)
        return b1 * nb + b2

    def step(self, O, truth=None):
        s = self.encode(O)
        A = np.argmax(self.Q[s], axis=1).astype(int)
        return A, (A == 0)


# --------------------------------------------------------------------------- #
# Training wrappers (the only code that reads ``truth``)
# --------------------------------------------------------------------------- #
class RandomLevels:
    """Round-0 behaviour policy for R3: uniform random level per agent-step."""

    def __init__(self, rng):
        self.rng = rng

    def reset(self, N):
        self.N = N

    def step(self, O, truth=None):
        A = self.rng.integers(0, N_LEVELS, size=O.shape[0])
        return A, (A == 0)


class RecordingPolicy:
    """Records observations and latent state along a rollout of ``inner``.
    ``inner`` itself never receives the latent state."""

    def __init__(self, inner):
        self.inner = inner

    def reset(self, N):
        self.inner.reset(N)
        self._O, self._r, self._h = [], [], []

    def step(self, O, truth=None):
        self._O.append(np.array(O, dtype=float, copy=True))
        self._r.append(np.array(truth["r"], dtype=float, copy=True))
        self._h.append(np.array(truth["h"], dtype=float, copy=True))
        return self.inner.step(O, truth=None)

    def trajectory(self):
        return {"O": np.array(self._O), "r": np.array(self._r), "h": np.array(self._h)}


class QTrainer:
    """Epsilon-greedy one-step Q-learning over a rollout of the frozen runner.

    Reward is -L(a; r, h) at the realized latent state (training only). The
    update for step t is applied once the step-(t+1) observation is known; the
    final step of the horizon is terminal.
    """

    def __init__(self, policy, lr, gamma, H, U, horizon, eps0, eps_min, total_steps, counter, rng):
        self.p = policy
        self.lr, self.gamma = float(lr), float(gamma)
        self.H, self.U = float(H), float(U)
        self.horizon = int(horizon)
        self.eps0, self.eps_min = float(eps0), float(eps_min)
        self.total = max(1, int(total_steps))
        self.counter = counter          # one-element list shared across rollouts
        self.rng = rng

    def reset(self, N):
        self.p.reset(N)
        self.N = N
        self.t = 0
        self.pending = None

    def _update(self, s, a, r, s_next):
        Q = self.p.Q
        for i in range(self.N):
            target = r[i] if s_next is None else r[i] + self.gamma * Q[s_next[i]].max()
            Q[s[i], a[i]] += self.lr * (target - Q[s[i], a[i]])

    def step(self, O, truth=None):
        s = self.p.encode(O)
        if self.pending is not None:
            self._update(*self.pending, s)
            self.pending = None
        progress = self.counter[0] / self.total
        eps = max(self.eps_min, self.eps0 - (self.eps0 - self.eps_min) * progress)
        explore = self.rng.random(self.N) < eps
        random_a = self.rng.integers(0, N_LEVELS, size=self.N)
        greedy = np.argmax(self.p.Q[s], axis=1)
        A = np.where(explore, random_a, greedy).astype(int)
        p = A / 5.0
        reward = -(p * truth["h"] * self.H + (1.0 - p) * truth["r"] * self.U)
        self.t += 1
        self.counter[0] += 1
        if self.t >= self.horizon:
            self._update(s, A, reward, None)
        else:
            self.pending = (s, A, reward)
        return A, (A == 0)
