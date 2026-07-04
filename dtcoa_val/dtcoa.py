"""DTCOA reference policy — FROZEN formulation, implemented verbatim. System-level controller
that governs N agents. Its internal scores are NEVER used as evaluation metrics (circularity ban)."""
import numpy as np

# frozen constants
ALPHA = 0.70; GAMMA = 0.60; E_MIN = 0.35; K = 3; S_MIN = 0.50
BANDS = np.array([0.20, 0.36, 0.52, 0.68, 0.88])      # T0..T5 boundaries -> level 0..5
# EQ1 weights — deliberately DIFFERENT from the simulator observation map (circularity break)
W = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.10]); W = W/W.sum()

def band_level(tp):
    return int(np.sum(tp >= BANDS))

class DTCOA:
    def __init__(self, context_risk=0.2, use_propagation=True, use_escalation=True,
                 use_decay=True, use_authorization=True, use_multiagent=True, use_trust=True,
                 window=5, alpha=ALPHA, gamma=GAMMA, e_min=E_MIN, k=K, s_min=S_MIN):
        self.Rc = context_risk; self.use_prop = use_propagation; self.use_esc = use_escalation
        self.use_decay = use_decay; self.use_auth = use_authorization; self.use_ma = use_multiagent
        self.use_trust = use_trust; self.window = window
        self.alpha=alpha; self.gamma=gamma; self.e_min=e_min; self.k=k; self.s_min=s_min
    def reset(self, N):
        self.N = N; self.T = np.full(N, 0.7); self.clow = np.zeros(N, int)
        self.last = np.full((N, 6), 0.7); self.Ehist = [[] for _ in range(N)]
        self._initial_fail = None
    def _impute(self, o):
        o = o.copy()
        nan = np.isnan(o)
        if nan.any(): o[nan] = self.last[nan] if hasattr(self,'last') else 0.5
        return o
    def step(self, O, truth=None):
        N = self.N; A = np.zeros(N, int); ESC = np.zeros(N, bool)
        E = np.zeros(N)
        for i in range(N):
            oi = O[i].copy(); nanm = np.isnan(oi)
            oi[nanm] = self.last[i][nanm]                      # last-value carry imputation
            self.last[i] = oi
            E[i] = float(np.clip(np.dot(W, oi), 0, 1))         # EQ1
            self.Ehist[i].append(E[i])
            self.clow[i] = self.clow[i]+1 if E[i] < self.e_min else 0
            if self.use_trust:
                self.T[i] = self.alpha*self.T[i] + (1-self.alpha)*E[i]   # EQ2
                if self.use_decay and self.clow[i] >= self.k:           # EQ7
                    self.T[i] = self.gamma*self.T[i]
            else:
                self.T[i] = E[i]                                        # ablation: no calibration (raw evidence)
        # system-level FAF (propagation amplification estimate from DTCOA-observed failures)
        obs_fail = E < self.e_min
        nfail = int(obs_fail.sum())
        if self.use_prop and self.use_ma:
            if nfail > 0 and self._initial_fail is None: self._initial_fail = max(1, nfail)
            if nfail == 0: self._initial_fail = None
            FAF = (nfail / self._initial_fail) if self._initial_fail else 1.0
        else:
            FAF = 1.0
        prop_pen = int(np.ceil(max(0.0, FAF - 1.0))) if self.use_prop else 0
        for i in range(N):
            Tp = self.T[i]*(1-self.Rc)                         # EQ8
            L = band_level(Tp)
            # trajectory stability S
            w = self.Ehist[i][-self.window:]
            S = 1 - min(1.0, (np.std(w)/0.25 if len(w) > 1 else 0.0))
            s_pen = 1 if (S < self.s_min) else 0
            if self.use_auth:
                a = int(np.clip(L - prop_pen - s_pen, 0, 5))   # EQ11
            else:
                a = L                                          # ablation: no adaptive authorization adjustment
            A[i] = a
            ESC[i] = (a == 0) if self.use_esc else False
        return A, ESC
