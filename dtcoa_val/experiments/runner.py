"""Generic runner used by ALL experiments. Drives the action-dependent outcome/cascade loop,
computes ONLY independent outcome + oracle metrics (plus trust-vs-latent for V1 calibration)."""
import numpy as np
from dtcoa_val.utils import make_rng, trial_seed
from dtcoa_val.sim_core.core import SimCore
from dtcoa_val.oracles.oracles import AuthorizationOracle, EscalationOracle

def run_trial(cfg, policy, seed, tag="", auth=None, esc=None):
    rng = make_rng(trial_seed(seed, 0, tag))
    sim = SimCore(cfg, rng); N, T = sim.N, sim.T
    auth = auth or AuthorizationOracle(H=cfg.get("H",1.0), U=cfg.get("U",0.3))
    esc = esc or EscalationOracle(theta_h=cfg.get("theta_h",0.5), H=cfg.get("H",1.0))
    policy.reset(N)
    fail_hist = np.zeros((T, N), bool)
    log = dict(harm=0.0, overrestrict=0.0, regret=0.0, sysfail=0,
               G=[], ESC=[], SEV=[], unauth_unsafe=0, auth_steps=0,
               trustT=[], latent_r=[], a_mean=[], esc_steps=0)
    tau = sim.tau
    for t in range(T):
        if tau >= 1:
            fr = fail_hist[max(0,t-tau):t].any(axis=0) if t>0 else np.zeros(N,bool)
            h_eff = sim.effective_h(t, fr); r_eff = sim.effective_r(t, fr)
        else:  # instantaneous cascade (tau=0): seed from same-step base failures, undetectable
            base_fail = (sim.r[:,t] < sim.r_fail) | (rng.random(N) < sim.h_base[:,t])
            h_eff = sim.effective_h(t, base_fail); r_eff = sim.effective_r(t, base_fail)
        O = sim.make_obs(t, r_eff, h_eff)
        A, ESC = policy.step(O, truth={"r": r_eff, "h": h_eff})
        pi = A/5.0
        unsafe = rng.random(N) < (pi*h_eff)
        log["harm"] += float(unsafe.sum())
        log["overrestrict"] += float(((1-pi)*r_eff).sum())
        log["auth_steps"] += int((pi>0).sum())
        log["unauth_unsafe"] += int((unsafe & (pi>0)).sum())
        log["esc_steps"] += int(ESC.sum())
        for i in range(N):
            log["regret"] += auth.regret(int(A[i]), float(r_eff[i]), float(h_eff[i]))
            log["G"].append(esc.should(float(h_eff[i]))); log["ESC"].append(bool(ESC[i]))
            log["SEV"].append(esc.severity(float(h_eff[i])))
        fail = unsafe | (r_eff < sim.r_fail); fail_hist[t] = fail; log["sysfail"] += int(fail.sum())
        log["a_mean"].append(float(A.mean()))
        tv = getattr(policy,"T",None)
        if tv is not None: log["trustT"].append(np.asarray(tv).copy())
        log["latent_r"].append(r_eff.copy())
    steps = T*N
    out = dict(
        harm_rate=log["harm"]/steps,
        overrestrict=log["overrestrict"]/steps,
        regret=log["regret"]/steps,
        unauth_unsafe_rate=(log["unauth_unsafe"]/max(1,log["auth_steps"])),
        sysfail_rate=log["sysfail"]/steps,
        esc_rate=log["esc_steps"]/steps,
        _G=np.array(log["G"]), _ESC=np.array(log["ESC"]), _SEV=np.array(log["SEV"]),
        _trustT=np.array(log["trustT"]) if log["trustT"] else None,
        _latent_r=np.array(log["latent_r"]),
        _a_mean=np.array(log["a_mean"]),
    )
    return out

def aggregate(trials, keys):
    from dtcoa_val.analysis.stats import describe
    agg = {}
    for k in keys:
        agg[k] = describe([t[k] for t in trials])
    return agg
