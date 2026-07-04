"""Tunable ROVA/DTCOA — parameterized version for the pre-registered HPO protocol.

Distinguished from dtcoa.py (the FROZEN legacy reference, kept unmodified for
reproducibility of prior-phase results) by two changes required before any
HPO/ablation work in PRE_REGISTERED_PROTOCOL.md is possible:

1. Propagation and stability penalties are continuous magnitudes (prop_scale,
   stab_scale), not integer ceil()/binary switches. This is required by:
   - §3.3: penalty magnitudes must be sampled log-uniform over a range wide
     enough to include values that effectively disable the mechanism.
   - §7 penalty-magnitude sweep / DR4 containment test: needs a continuous
     dial from 0 to full magnitude, not an on/off switch.
2. Every parameter in CURRENT_FRAMEWORK_SPECIFICATION.md §9 marked "tuned
   under final protocol? Yes" is exposed as a constructor argument:
   alpha, gamma, e_min, k, bands (B), context_risk (Rc), prop_scale,
   stab_scale, s_min. Evidence weights W remain fixed (architecture's front
   end, per spec §4) — not exposed as tunable.

The on/off structural switches (use_propagation, use_stability, use_decay,
use_trust) are retained for the factorial ablation (§7) — they are NOT part
of the main HPO search space (see search_space.py), consistent with §3.2:
"Binary structural on/off switches ... are not part of the main HPO search;
they are handled exclusively through the factorial ablation."
"""
import numpy as np

# Fixed evidence weights — architecture's front end, never tuned (spec §4).
# Identical to the frozen dtcoa.py so both implementations measure the same thing.
W = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.10])
W = W / W.sum()


def band_level(tp, bands):
    """Quantize a scalar trust statistic into a discrete level 0..len(bands)."""
    return int(np.sum(tp >= np.asarray(bands)))


class DTCOATunable:
    def __init__(
        self,
        alpha=0.70,
        gamma=0.60,
        e_min=0.35,
        k=3,
        s_min=0.50,
        bands=(0.20, 0.36, 0.52, 0.68, 0.88),
        context_risk=0.2,
        prop_scale=1.0,
        stab_scale=1.0,
        window=5,
        use_propagation=True,
        use_stability=True,
        use_decay=True,
        use_trust=True,
        use_escalation=True,
        use_multiagent=True,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.e_min = float(e_min)
        self.k = int(k)
        self.s_min = float(s_min)
        self.bands = np.sort(np.asarray(bands, dtype=float))
        self.Rc = float(context_risk)
        self.prop_scale = float(prop_scale)
        self.stab_scale = float(stab_scale)
        self.window = int(window)
        self.use_prop = use_propagation
        self.use_stab = use_stability
        self.use_decay = use_decay
        self.use_trust = use_trust
        self.use_esc = use_escalation
        self.use_ma = use_multiagent

    def reset(self, N):
        self.N = N
        self.T = np.full(N, 0.7)
        self.clow = np.zeros(N, int)
        self.last = np.full((N, 6), 0.7)
        self.Ehist = [[] for _ in range(N)]
        self._initial_fail = None

    def step(self, O, truth=None):
        N = self.N
        A = np.zeros(N, int)
        ESC = np.zeros(N, bool)
        E = np.zeros(N)
        for i in range(N):
            oi = O[i].copy()
            nanm = np.isnan(oi)
            oi[nanm] = self.last[i][nanm]
            self.last[i] = oi
            E[i] = float(np.clip(np.dot(W, oi), 0, 1))
            self.Ehist[i].append(E[i])
            self.clow[i] = self.clow[i] + 1 if E[i] < self.e_min else 0
            if self.use_trust:
                self.T[i] = self.alpha * self.T[i] + (1 - self.alpha) * E[i]
                if self.use_decay and self.clow[i] >= self.k:
                    self.T[i] = self.gamma * self.T[i]
            else:
                self.T[i] = E[i]

        # System-level propagation-amplification factor (unchanged detection logic;
        # only how it becomes a penalty is now continuous, see below).
        obs_fail = E < self.e_min
        nfail = int(obs_fail.sum())
        if self.use_ma:
            if nfail > 0 and self._initial_fail is None:
                self._initial_fail = max(1, nfail)
            if nfail == 0:
                self._initial_fail = None
            FAF = (nfail / self._initial_fail) if self._initial_fail else 1.0
        else:
            FAF = 1.0
        # Continuous overshoot magnitude, scaled by a tunable penalty magnitude
        # instead of ceil()'d to an integer. prop_scale -> 0 disables the
        # mechanism smoothly (containment test, DR4); prop_scale is swept
        # log-uniform including near-zero values per §3.3.
        raw_overshoot = max(0.0, FAF - 1.0)
        prop_pen = self.prop_scale * raw_overshoot if self.use_prop else 0.0

        for i in range(N):
            Tp = self.T[i] * (1 - self.Rc)
            L = band_level(Tp, self.bands)
            w = self.Ehist[i][-self.window:]
            S = 1 - min(1.0, (np.std(w) / 0.25 if len(w) > 1 else 0.0))
            # Continuous stability-violation magnitude (was a binary s_pen=1
            # if S < s_min). stab_scale -> 0 disables the mechanism smoothly.
            stab_deficit = max(0.0, self.s_min - S)
            s_pen = self.stab_scale * stab_deficit if self.use_stab else 0.0

            a_raw = L - prop_pen - s_pen
            a = int(np.clip(round(a_raw), 0, 5))
            A[i] = a
            ESC[i] = (a == 0) if self.use_esc else False
        return A, ESC
