"""SIM CORE — independent world. NO DTCOA constructs (no EQ1/2/7/8/11, no bands, no E_min).
Latent (r,h) -> mismatched observation maps -> corruption. Multi-agent graph + cascade.
Adversary + recovery. The world is free to be non-contractive; DTCOA only sees observations."""
import numpy as np

D = 6  # observation channels

# ---------- reliability regimes (NOT EWMA / NOT a contraction) ----------
def gen_reliability(regime, T, rng, params=None):
    p = params or {}
    if regime == "stationary":
        r0 = p.get("r0", rng.uniform(0.55, 0.9)); s = p.get("sigma", 0.04)
        return np.clip(r0 + rng.normal(0, s, T), 0, 1)
    if regime == "step":
        r = np.empty(T); cur = rng.uniform(0.5, 0.9); lam = p.get("lam_cp", 0.02)
        for t in range(T):
            if rng.random() < lam: cur = np.clip(cur + rng.uniform(-0.5, 0.5), 0.05, 0.98)
            r[t] = cur
        return np.clip(r + rng.normal(0, 0.02, T), 0, 1)
    if regime == "drift":
        slope = p.get("slope", rng.choice([-1, 1]) * rng.uniform(0.0005, 0.0025))
        base = np.clip(0.8 + slope*np.arange(T) + np.cumsum(rng.normal(0, 0.01, T)), 0.05, 0.98)
        return base
    if regime == "oscillation":
        P = p.get("period", rng.choice([10, 20, 8]))  # incl. fast (attacks tracking)
        m = p.get("mean", 0.6); a = p.get("amp", 0.3)
        return np.clip(m + a*np.sin(2*np.pi*np.arange(T)/P + rng.uniform(0, 2*np.pi)) + rng.normal(0, 0.02, T), 0, 1)
    if regime == "collapse":
        cp = p.get("cp", rng.integers(T//4, 3*T//4)); hi = rng.uniform(0.8, 0.95); lo = rng.uniform(0.1, 0.3)
        r = np.where(np.arange(T) < cp, hi, lo)
        return np.clip(r + rng.normal(0, 0.02, T), 0, 1)
    if regime == "hmm":
        # modes: good, degrading, bad, recovering ; targets + mean reversion (rate != gamma)
        targets = {0: 0.9, 1: 0.55, 2: 0.2, 3: 0.6}
        Pi = np.array([[0.92,0.06,0.0,0.02],[0.05,0.8,0.13,0.02],[0.0,0.05,0.8,0.15],[0.2,0.1,0.0,0.7]])
        m = 0; r = np.empty(T); cur = targets[0]
        for t in range(T):
            cur = cur + 0.3*(targets[m]-cur) + rng.normal(0, 0.02)  # reversion rate 0.3 (independent of gamma=0.6)
            r[t] = np.clip(cur, 0, 1); m = rng.choice(4, p=Pi[m])
        return r
    raise ValueError(regime)

# ---------- hazard engine: h decoupled from r ----------
def gen_hazard(r, rng, g=0.5, bias=None, p_shock=0.01, shock_mag=(0.3,0.7), noise=0.03):
    T = len(r)
    b = rng.uniform(0.0, 0.30) if bias is None else bias      # persistent hazard bias (capable-but-unsafe)
    shocks = (rng.random(T) < p_shock) * rng.uniform(*shock_mag, T)
    h = g*(1-r) + b + shocks + rng.normal(0, noise, T)
    return np.clip(h, 0, 1), b

# ---------- observation maps (MISMATCHED from DTCOA EQ1 weights) ----------
def obs_map_A(r, h):
    return np.array([r, 0.5*r+0.5*(1-h), 1-h, 0.7*r+0.3*(1-h),
                     np.clip(1-0.5*h-0.2*(1-r),0,1), 0.6*(1-h)+0.4*r])
def obs_map_B(r, h):
    return np.array([r**1.5, (1-h)**0.8, 0.5*(r+(1-h)), np.sqrt(max(0.0,r*(1-h))),
                     np.clip(1-h**1.2,0,1), 0.3+0.4*r+0.3*(1-h)])

class SimCore:
    """Holds latent trajectories + observation/corruption + graph. The action-dependent
    outcome/cascade loop is driven by the runner (outcomes depend on granted autonomy)."""
    def __init__(self, cfg, rng):
        self.cfg = cfg; self.rng = rng
        self.N = cfg.get("n_agents", 1); self.T = cfg["horizon"]
        self.regime = cfg.get("regime", "hmm"); self.obs_map = obs_map_B if cfg.get("obs_map","A")=="B" else obs_map_A
        self.noise = cfg.get("noise", 0.04); self.delay = cfg.get("delay", 0); self.p_miss = cfg.get("p_miss", 0.0)
        self.mnar = cfg.get("mnar", False); self.conflict_frac = cfg.get("conflict_frac", 0.0)
        self.Rc = cfg.get("context_risk", 0.2); self.r_fail = cfg.get("r_fail", 0.25)
        # latent
        self.r = np.stack([gen_reliability(self.regime, self.T, rng, cfg.get("regime_params")) for _ in range(self.N)])
        hb = [gen_hazard(self.r[i], rng, g=cfg.get("g",0.5), p_shock=cfg.get("p_shock",0.01), noise=cfg.get("h_noise",0.03)) for i in range(self.N)]
        self.h_base = np.stack([x[0] for x in hb]); self.bias = np.array([x[1] for x in hb])
        # multi-agent graph (DAG-ish): each node has up to k parents among lower indices
        self.coupling = cfg.get("coupling", 0.0); self.tau = cfg.get("cascade_tau", 1)
        self.parents = [[] for _ in range(self.N)]
        dens = cfg.get("graph_density", 0.0)
        for i in range(1, self.N):
            for j in range(i):
                if rng.random() < dens: self.parents[i].append(j)
        # corruption state
        self._delay_buf = [[] for _ in range(self.N)]
        self.faulty_ch = set(int(c) for c in rng.choice(D, size=int(round(self.conflict_frac*D)), replace=False)) if self.conflict_frac>0 else set()
        # adversary
        self.adv = cfg.get("adversary", None)  # dict: {type, agents:[...], defect_t}

    # effective hazard at step t given which agents have failed (cascade)
    def effective_h(self, t, failed_recent):
        h = self.h_base[:, t].copy()
        if self.coupling > 0:
            add = np.zeros(self.N)
            for i in range(self.N):
                for pj in self.parents[i]:
                    if failed_recent[pj]: add[i] += self.coupling
            h = np.clip(h + add, 0, 1)
        # adversary hazard override (defect spike)
        if self.adv and t >= self.adv.get("defect_t", 10**9):
            for ai in self.adv.get("agents", []):
                if self.adv["type"] in ("ramp_defect","collusion"): h[ai] = max(h[ai], self.adv.get("spike",0.8))
        return h

    def effective_r(self, t, failed_recent):
        r = self.r[:, t].copy()
        if self.coupling > 0:
            for i in range(self.N):
                for pj in self.parents[i]:
                    if failed_recent[pj]: r[i] *= (1-self.coupling)
        return np.clip(r, 0, 1)

    # build the (corrupted) observation DTCOA/baselines see, from TRUE (r,h)
    def make_obs(self, t, r_eff, h_eff):
        O = np.zeros((self.N, D))
        for i in range(self.N):
            o = self.obs_map(r_eff[i], h_eff[i]).astype(float)
            # adversary observation manipulation (looks good regardless of true h)
            if self.adv and i in self.adv.get("agents", []):
                ty = self.adv["type"]
                if ty in ("ramp_defect","collusion","poisoning"):
                    o = np.maximum(o, self.adv.get("fake_obs", 0.9))   # emit high signals
            o = o + self.rng.normal(0, self.noise, D)
            # heavy-tail contamination
            if self.rng.random() < 0.02: o += self.rng.standard_t(3, D)*0.1
            # conflicting: invert faulty channels (sensor fault)
            for c in self.faulty_ch: o[c] = 1 - o[c]
            o = np.clip(o, 0, 1)
            # missingness
            pm = self.p_miss*(1.5 if (self.mnar and h_eff[i]>0.5) else 1.0)
            mask = self.rng.random(D) < pm
            o[mask] = np.nan
            # delay (per-agent FIFO on whole vector)
            if self.delay > 0:
                self._delay_buf[i].append(o.copy())
                if len(self._delay_buf[i]) > self.delay:
                    O[i] = self._delay_buf[i].pop(0)
                else:
                    O[i] = self.obs_map(0.7,0.2)  # neutral warmup
            else:
                O[i] = o
        return O
