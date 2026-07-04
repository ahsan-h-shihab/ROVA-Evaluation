"""Validation Package v2 — C1 REPAIR ONLY. Configs identical to v1; the ONLY change is that every condition
within a comparison is run on the SAME scenario seed (tag identifies the SCENARIO, not the policy/ablation),
giving genuine paired pairing. Re-runs V1, A1, B1, V5, V7 and recomputes all paired statistics.
No new experiments, no mechanism/SIM-CORE/DTCOA changes. Metrics kept identical to v1 to isolate the effect of C1."""
import os, copy, numpy as np
from dtcoa_val.experiments.runner import run_trial
from dtcoa_val.dtcoa import DTCOA
from dtcoa_val.baselines.baselines import StaticThreshold, EWMA, RuleBased, OracleLite
from dtcoa_val.analysis.stats import describe, wilcoxon_paired, holm_bonferroni, cliffs_delta
from dtcoa_val.utils import save_json, stamp

OUT="/mnt/user-data/outputs/dtcoa_validation/results_v2"; ARR="/mnt/user-data/outputs/dtcoa_validation/arrays_v2"
os.makedirs(OUT,exist_ok=True); os.makedirs(ARR,exist_ok=True)
BASE=dict(horizon=300,n_agents=1,regime="hmm",obs_map="A",noise=0.04,context_risk=0.2,
          H=1.0,U=0.3,theta_h=0.5,r_fail=0.25,p_shock=0.01,h_noise=0.03)
def cfg(**kw): c=copy.deepcopy(BASE); c.update(kw); return c
# SHARED-SEED runner: scenario tag is identical across compared conditions -> same world per seed index
def run_cond(c, policy, seeds, scenario):
    return [run_trial(c, policy() if callable(policy) else policy, s, tag=scenario) for s in seeds]

# ---------------- V1 (paired DTCOA vs naive on SAME worlds) ----------------
def V1():
    regimes=["stationary","step","drift","oscillation","collapse","hmm"]; seeds=list(range(20)); res={}; arr={}
    for rg in regimes:
        c=cfg(regime=rg); sc=f"V1_{rg}"                    # SAME scenario for both policies
        dt=run_cond(c, lambda: DTCOA(), seeds, sc)
        nv=run_cond(c, lambda: EWMA(beta=0.7), seeds, sc)
        def trk(trials):
            errs=[]; corrs=[]
            for t in trials:
                T=t["_trustT"][:,0]; r=t["_latent_r"][:,0]; errs.append(float(np.mean(np.abs(T-r))))
                if np.std(T)>1e-6 and np.std(r)>1e-6: corrs.append(float(np.corrcoef(T,r)[0,1]))
            return errs,corrs
        de,dc=trk(dt); ne,_=trk(nv)
        res[rg]=dict(dtcoa_tracking_err=describe(de), naive_tracking_err=describe(ne),
                     dtcoa_corr=describe(dc), wilcoxon_vs_naive=wilcoxon_paired(de,ne),
                     cliffs_delta=cliffs_delta(de,ne))
        if rg in ("collapse","hmm"): arr[rg]=dict(T=dt[0]["_trustT"][:,0].tolist(), r=dt[0]["_latent_r"][:,0].tolist())
    save_json(f"{OUT}/V1.json", dict(stamp=stamp(), c1_fixed=True, per_regime=res)); save_json(f"{ARR}/V1_traces.json", arr); return res

# ---------------- A1 (paired ablations vs full on SAME worlds) ----------------
def A1():
    c=cfg(n_agents=12,regime="hmm",graph_density=0.3,coupling=0.3,cascade_tau=2,horizon=300)
    seeds=list(range(20)); sc="A1"                          # SAME scenario for all variants
    variants={"full":dict(),"no_trust":dict(use_trust=False),"no_propagation":dict(use_propagation=False),
              "no_escalation":dict(use_escalation=False),"no_authorization":dict(use_authorization=False),
              "no_multiagent":dict(use_multiagent=False)}
    runs={k:run_cond(c,(lambda kw=v: DTCOA(**kw)),seeds,sc) for k,v in variants.items()}
    metrics=["harm_rate","regret","unauth_unsafe_rate","sysfail_rate"]; full=runs["full"]; res={}; pv=[]; lab=[]
    for k in variants:
        if k=="full": continue
        d={}
        for m in metrics:
            a=[t[m] for t in runs[k]]; b=[t[m] for t in full]; w=wilcoxon_paired(a,b)
            d[m]=dict(ablated=describe(a),full=describe(b),delta=float(np.mean(a)-np.mean(b)),
                      wilcoxon_p=w[1],cliffs=cliffs_delta(a,b)); pv.append(w[1]); lab.append((k,m))
        res[k]=d
    hb=holm_bonferroni(pv); res["_holm"]=[dict(ablation=lab[i][0],metric=lab[i][1],p=hb[i][1],reject=hb[i][3]) for i in range(len(lab))]
    save_json(f"{OUT}/A1.json", dict(stamp=stamp(), c1_fixed=True, ablations=res)); return res

# ---------------- B1 (paired DTCOA vs baselines on SAME worlds; tuning unchanged) ----------------
def B1():
    c=cfg(n_agents=6,regime="hmm",graph_density=0.2,coupling=0.2,cascade_tau=2,horizon=300)
    TUNE=list(range(100,112)); TEST=list(range(24)); sc="B1"
    def tune(fac,grid):
        best=None
        for thr in grid:
            r=np.mean([t["regret"] for t in run_cond(c, lambda: fac(thr), TUNE, "B1tune")])
            if best is None or r<best[0]: best=(r,thr)
        return best[1]
    grids=[(0.15+0.1*i,0.30+0.1*i,0.46+0.1*i,0.62+0.1*i,0.80+0.05*i) for i in range(4)]
    st=tune(lambda thr: StaticThreshold(thr),grids); ew=tune(lambda thr: EWMA(beta=0.7,thr=thr),grids)
    pols={"DTCOA":lambda: DTCOA(),"Static":lambda: StaticThreshold(st),"EWMA":lambda: EWMA(beta=0.7,thr=ew),
          "Rule":lambda: RuleBased(),"OracleLite":lambda: OracleLite()}
    runs={k:run_cond(c,f,TEST,sc) for k,f in pols.items()}
    res={}; pv=[]; lab=[]
    for m in ["regret","harm_rate","overrestrict","unauth_unsafe_rate"]:
        res[m]={k:describe([t[m] for t in runs[k]]) for k in pols}
        for k in ["Static","EWMA","Rule"]:
            w=wilcoxon_paired([t[m] for t in runs["DTCOA"]],[t[m] for t in runs[k]])
            res[m][f"DTCOA_vs_{k}_wilcoxon_p"]=w[1]; res[m][f"DTCOA_vs_{k}_cliffs"]=cliffs_delta([t[m] for t in runs["DTCOA"]],[t[m] for t in runs[k]])
            pv.append(w[1]); lab.append((m,k))
    hb=holm_bonferroni(pv); res["_holm"]=[dict(metric=lab[i][0],baseline=lab[i][1],p=hb[i][1],reject=hb[i][3]) for i in range(len(lab))]
    res["_tuned_thresholds"]=dict(static=list(st),ewma=list(ew))
    save_json(f"{OUT}/B1.json", dict(stamp=stamp(), c1_fixed=True, comparison=res)); return res

# ---------------- V5 (paired DTCOA vs no-gov on SAME worlds; metric unchanged from v1) ----------------
def V5():
    advs={"ramp_defect":dict(type="ramp_defect",agents=[0],defect_t=150,spike=0.85,fake_obs=0.9),
          "poisoning":dict(type="poisoning",agents=[0,1],defect_t=0,spike=0.6,fake_obs=0.9),
          "collusion":dict(type="collusion",agents=[0,1,2],defect_t=150,spike=0.85,fake_obs=0.9)}
    seeds=list(range(16)); res={}; pv=[]; lab=[]
    for nm,adv in advs.items():
        c=cfg(n_agents=8,regime="stationary",adversary=adv,horizon=300); sc=f"V5_{nm}"
        dt=run_cond(c, lambda: DTCOA(), seeds, sc)
        ng=run_cond(c, lambda: RuleBased(low=-1,mid=-1), seeds, sc)
        ds=[t["unauth_unsafe_rate"] for t in dt]; ns=[t["unauth_unsafe_rate"] for t in ng]
        res[nm]=dict(dtcoa_attack_success=describe(ds),nogov_attack_success=describe(ns),
                     wilcoxon=wilcoxon_paired(ds,ns),cliffs=cliffs_delta(ds,ns),
                     dtcoa_harm=describe([t["harm_rate"] for t in dt]))
        pv.append(wilcoxon_paired(ds,ns)[1]); lab.append(nm)
    hb=holm_bonferroni(pv); res["_holm"]=[dict(attack=lab[i],p=hb[i][1],reject=hb[i][3]) for i in range(len(lab))]
    save_json(f"{OUT}/V5.json", dict(stamp=stamp(), c1_fixed=True,
              metric_note="unchanged from v1: system-wide unauth-unsafe rate (M2 NOT repaired here)", attacks=res)); return res

# ---------------- V7 (paired prop-ON vs prop-OFF on SAME worlds) ----------------
def V7():
    seeds=list(range(16)); res={}; curve=[]; pv=[]; lab=[]
    for tau in [0,1,3,6]:
        c=cfg(n_agents=12,regime="hmm",graph_density=0.35,coupling=0.4,cascade_tau=tau,horizon=300); sc=f"V7_{tau}"
        on=run_cond(c, lambda: DTCOA(use_propagation=True), seeds, sc)
        off=run_cond(c, lambda: DTCOA(use_propagation=False), seeds, sc)
        a=[t["sysfail_rate"] for t in on]; b=[t["sysfail_rate"] for t in off]; w=wilcoxon_paired(a,b)
        # autonomy budgets (to expose M3 confound: is benefit just lower autonomy?)
        ona=float(np.mean([np.mean(t["_a_mean"]) for t in on])); offa=float(np.mean([np.mean(t["_a_mean"]) for t in off]))
        res[f"tau_{tau}"]=dict(prop_on_sysfail=describe(a),prop_off_sysfail=describe(b),delta=float(np.mean(a)-np.mean(b)),
                               wilcoxon_p=w[1],cliffs=cliffs_delta(a,b),
                               prop_on_mean_autonomy=ona,prop_off_mean_autonomy=offa)
        curve.append(dict(tau=tau,on=float(np.mean(a)),off=float(np.mean(b)))); pv.append(w[1]); lab.append(tau)
    hb=holm_bonferroni(pv); res["_holm"]=[dict(tau=lab[i],p=hb[i][1],reject=hb[i][3]) for i in range(len(lab))]
    save_json(f"{OUT}/V7.json", dict(stamp=stamp(), c1_fixed=True, by_tau=res)); save_json(f"{ARR}/V7_curve.json", curve); return res

if __name__=="__main__":
    import sys, time
    order=[("V1",V1),("A1",A1),("B1",B1),("V5",V5),("V7",V7)]; only=sys.argv[1:] or None
    for nm,fn in order:
        if only and nm not in only: continue
        t0=time.time(); fn(); print(f"[v2 {nm}] done in {time.time()-t0:.1f}s", flush=True)
