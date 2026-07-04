"""DTCOA validation campaign — executes V1..S1 with fixed protocol. Honest negatives reported.
No seed/metric/hypothesis switching. Results + figure-arrays saved to results dir."""
import os, time, copy, numpy as np
from scipy import stats as sps
from dtcoa_val.experiments.runner import run_trial
from dtcoa_val.dtcoa import DTCOA
from dtcoa_val.baselines.baselines import StaticThreshold, EWMA, RuleBased, OracleLite
from dtcoa_val.oracles.oracles import AuthorizationOracle, EscalationOracle
from dtcoa_val.analysis.stats import describe, bootstrap_ci, wilcoxon_paired, holm_bonferroni, cliffs_delta
from dtcoa_val.utils import save_json, stamp

OUT = "/mnt/user-data/outputs/dtcoa_validation/results"
ARR = "/mnt/user-data/outputs/dtcoa_validation/arrays"
os.makedirs(OUT, exist_ok=True); os.makedirs(ARR, exist_ok=True)
BASE = dict(horizon=300, n_agents=1, regime="hmm", obs_map="A", noise=0.04,
            context_risk=0.2, H=1.0, U=0.3, theta_h=0.5, r_fail=0.25, p_shock=0.01, h_noise=0.03)
SEEDS = list(range(20))
def cfg(**kw): c=copy.deepcopy(BASE); c.update(kw); return c
def run_set(c, factory, seeds=SEEDS, tag=""):
    return [run_trial(c, factory(), s, tag=tag) for s in seeds]

# ---------------- V1 Trust Calibration (validates the simulator) ----------------
def V1():
    regimes=["stationary","step","drift","oscillation","collapse","hmm"]; res={}; arr={}
    for rg in regimes:
        c=cfg(regime=rg)
        dt=run_set(c, lambda: DTCOA(), tag="V1d_"+rg)
        nv=run_set(c, lambda: EWMA(beta=0.7), tag="V1n_"+rg)
        def trk(trials):
            errs=[]; corrs=[]
            for t in trials:
                T=t["_trustT"][:,0]; r=t["_latent_r"][:,0]
                errs.append(float(np.mean(np.abs(T-r))))
                if np.std(T)>1e-6 and np.std(r)>1e-6: corrs.append(float(np.corrcoef(T,r)[0,1]))
            return errs, corrs
        de,dc=trk(dt); ne,_=trk(nv)
        # detection lag (collapse only)
        lags=[]
        if rg=="collapse":
            for t in dt:
                r=t["_latent_r"][:,0]; T=t["_trustT"][:,0]
                hi=r>0.6; 
                cps=np.where((~hi[1:])&(hi[:-1]))[0]
                if len(cps):
                    cp=cps[0]+1; dl=np.where(T[cp:]<0.45)[0]
                    if len(dl): lags.append(int(dl[0]))
        res[rg]=dict(dtcoa_tracking_err=describe(de), naive_tracking_err=describe(ne),
                     dtcoa_corr=describe(dc), wilcoxon_vs_naive=wilcoxon_paired(de,ne),
                     cliffs_delta=cliffs_delta(de,ne), detection_lag=describe(lags) if lags else None)
        if rg in ("collapse","hmm"):
            arr[rg]=dict(T=dt[0]["_trustT"][:,0].tolist(), r=dt[0]["_latent_r"][:,0].tolist())
    save_json(f"{OUT}/V1.json", dict(stamp=stamp(), per_regime=res))
    save_json(f"{ARR}/V1_traces.json", arr)
    return res

# ---------------- V2 Sensitivity ----------------
def V2():
    seeds=list(range(12)); base_re=cfg(regime="hmm")
    def regret_for(**over):
        tr=run_set(base_re, lambda: DTCOA(**over), seeds=seeds, tag="V2"); 
        return [t["regret"] for t in tr], [t["harm_rate"] for t in tr]
    oat={}
    for name,vals,key in [("alpha",[0.5,0.6,0.7,0.8,0.9],"alpha"),
                          ("gamma",[0.4,0.6,0.8],"gamma"),
                          ("e_min",[0.25,0.35,0.45],"e_min"),
                          ("k",[2,3,5],"k")]:
        oat[name]={}
        for v in vals:
            rg,hr=regret_for(**{key:v}); oat[name][str(v)]=dict(regret=describe(rg), harm=describe(hr))
    # joint heatmap alpha x e_min
    A=[0.5,0.6,0.7,0.8,0.9]; Em=[0.25,0.35,0.45]; heat=np.zeros((len(Em),len(A)))
    flips=np.zeros((len(Em),len(A)))
    ref=run_set(base_re, lambda: DTCOA(), seeds=seeds, tag="V2ref")
    ref_a=np.concatenate([t["_a_mean"] for t in ref])
    for ii,em in enumerate(Em):
        for jj,al in enumerate(A):
            tr=run_set(base_re, lambda: DTCOA(alpha=al,e_min=em), seeds=seeds, tag=f"V2_{al}_{em}")
            heat[ii,jj]=float(np.mean([t["regret"] for t in tr]))
            cur=np.concatenate([t["_a_mean"] for t in tr])
            flips[ii,jj]=float(np.mean(np.abs(np.round(cur)-np.round(ref_a))>0))
    save_json(f"{OUT}/V2.json", dict(stamp=stamp(), one_at_a_time=oat,
              heat_alpha=A, heat_emin=Em, regret_heat=heat.tolist(), flip_heat=flips.tolist()))
    return oat

# ---------------- V3 Stress ----------------
def V3():
    conds={"mild":cfg(), "high_noise":cfg(noise=0.15), "missing_mnar":cfg(p_miss=0.3,mnar=True),
           "conflict":cfg(conflict_frac=0.34), "delay":cfg(delay=3), "collapse":cfg(regime="collapse")}
    res={}
    for nm,c in conds.items():
        tr=run_set(c, lambda: DTCOA(), tag="V3_"+nm)
        res[nm]=dict(regret=describe([t["regret"] for t in tr]),
                     harm_rate=describe([t["harm_rate"] for t in tr]),
                     unauth_unsafe=describe([t["unauth_unsafe_rate"] for t in tr]),
                     esc_rate=describe([t["esc_rate"] for t in tr]))
    save_json(f"{OUT}/V3.json", dict(stamp=stamp(), conditions=res)); return res

# ---------------- V4 Escalation ----------------
def V4():
    c=cfg(regime="hmm"); dt=run_set(c, lambda: DTCOA(), tag="V4d"); rb=run_set(c, lambda: RuleBased(), tag="V4r")
    def detect(trials):
        G=np.concatenate([t["_G"] for t in trials]).astype(bool)
        E=np.concatenate([t["_ESC"] for t in trials]).astype(bool)
        miss=float((G&~E).sum()/max(1,G.sum())); false=float((~G&E).sum()/max(1,(~G).sum()))
        return miss,false,G,E
    dm,df,G,E=detect(dt); rm,rf,_,_=detect(rb)
    # theta sweep PR/ROC for DTCOA (vary oracle severity -> regenerate G from severity)
    SEV=np.concatenate([t["_SEV"] for t in dt]); ESC=np.concatenate([t["_ESC"] for t in dt]).astype(bool)
    pts=[]
    for theta in np.linspace(0.25,0.75,11):
        Gt=SEV>=theta
        tp=(Gt&ESC).sum(); fp=(~Gt&ESC).sum(); fn=(Gt&~ESC).sum(); tn=(~Gt&~ESC).sum()
        recall=tp/max(1,tp+fn); prec=tp/max(1,tp+fp); fpr=fp/max(1,fp+tn)
        pts.append(dict(theta=float(theta),recall=float(recall),precision=float(prec),fpr=float(fpr)))
    # delay
    delays=[]
    for t in dt:
        g=t["_G"].astype(bool); e=t["_ESC"].astype(bool)
        onsets=np.where(g[1:]&~g[:-1])[0]+1
        for o in onsets:
            nx=np.where(e[o:])[0]
            if len(nx): delays.append(int(nx[0]))
    res=dict(stamp=stamp(), dtcoa=dict(miss_rate=dm,false_rate=df),
             rulebased=dict(miss_rate=rm,false_rate=rf),
             pr_roc=pts, delay=describe(delays) if delays else None)
    save_json(f"{OUT}/V4.json", res); save_json(f"{ARR}/V4_pr.json", pts); return res

# ---------------- V5 Adversarial ----------------
def V5():
    advs={
      "ramp_defect":dict(type="ramp_defect",agents=[0],defect_t=150,spike=0.85,fake_obs=0.9),
      "poisoning":dict(type="poisoning",agents=[0,1],defect_t=0,spike=0.6,fake_obs=0.9),
      "collusion":dict(type="collusion",agents=[0,1,2],defect_t=150,spike=0.85,fake_obs=0.9),
    }
    res={}; arr={}
    for nm,adv in advs.items():
        c=cfg(n_agents=8, regime="stationary", adversary=adv, horizon=300)
        dt=run_set(c, lambda: DTCOA(), seeds=list(range(16)), tag="V5d_"+nm)
        ng=run_set(c, lambda: RuleBased(low=-1,mid=-1), seeds=list(range(16)), tag="V5n_"+nm)  # always a=5 (no-gov)
        def succ(trials):  # unauthorized unsafe rate (independent outcome)
            return [t["unauth_unsafe_rate"] for t in trials]
        ds=succ(dt); ns=succ(ng)
        res[nm]=dict(dtcoa_attack_success=describe(ds), nogov_attack_success=describe(ns),
                     wilcoxon=wilcoxon_paired(ds,ns), cliffs=cliffs_delta(ds,ns),
                     dtcoa_harm=describe([t["harm_rate"] for t in dt]))
    save_json(f"{OUT}/V5.json", dict(stamp=stamp(), attacks=res)); return res

# ---------------- V6 Long-Horizon ----------------
def V6():
    res={}
    for rg in ["stationary","hmm"]:
        c=cfg(regime=rg, horizon=4000); tr=run_set(c, lambda: DTCOA(), seeds=list(range(8)), tag="V6_"+rg)
        slopes=[]; osc=[]; fh=[]; sh=[]
        for t in tr:
            a=t["_a_mean"]; x=np.arange(len(a)); sl=sps.linregress(x,a).slope; slopes.append(sl*len(a))
            osc.append(float(np.std(a))); fh.append(float(np.mean(a[:len(a)//2]))); sh.append(float(np.mean(a[len(a)//2:])))
        statp=wilcoxon_paired(fh,sh)
        res[rg]=dict(drift_total_over_run=describe(slopes), oscillation_std=describe(osc),
                     firsthalf=describe(fh), secondhalf=describe(sh), stationarity_wilcoxon=statp,
                     harm_run_variance=float(np.var([t["harm_rate"] for t in tr])))
    save_json(f"{OUT}/V6.json", dict(stamp=stamp(), regimes=res)); return res

# ---------------- A1 Ablation ----------------
def A1():
    c=cfg(n_agents=12, regime="hmm", graph_density=0.3, coupling=0.3, cascade_tau=2, horizon=300)
    variants={"full":dict(), "no_trust":dict(use_trust=False), "no_propagation":dict(use_propagation=False),
              "no_escalation":dict(use_escalation=False), "no_authorization":dict(use_authorization=False),
              "no_multiagent":dict(use_multiagent=False)}
    runs={k:run_set(c, (lambda kw=v: DTCOA(**kw)), tag="A1_"+k) for k,v in variants.items()}
    metrics=["harm_rate","regret","unauth_unsafe_rate","sysfail_rate"]
    full=runs["full"]; res={}; pvals=[]; labels=[]
    for k in variants:
        if k=="full": continue
        d={}
        for m in metrics:
            a=[t[m] for t in runs[k]]; b=[t[m] for t in full]
            w=wilcoxon_paired(a,b); d[m]=dict(ablated=describe(a), full=describe(b), delta=float(np.mean(a)-np.mean(b)),
                                              wilcoxon_p=w[1], cliffs=cliffs_delta(a,b))
            pvals.append(w[1]); labels.append((k,m))
        res[k]=d
    hb=holm_bonferroni(pvals); 
    for (lab,(k,m)),h in zip(zip(labels,labels),hb): pass
    res["_holm"]=[dict(ablation=labels[i][0],metric=labels[i][1],p=hb[i][1],reject=hb[i][3]) for i in range(len(labels))]
    save_json(f"{OUT}/A1.json", dict(stamp=stamp(), ablations=res)); return res

# ---------------- B1 Baselines (tuned on disjoint seeds; honest null) ----------------
def B1():
    c=cfg(n_agents=6, regime="hmm", graph_density=0.2, coupling=0.2, cascade_tau=2, horizon=300)
    TUNE=list(range(100,112)); TEST=list(range(24))
    def tune(factory_for_thr, grid):
        best=None
        for thr in grid:
            tr=run_set(c, lambda: factory_for_thr(thr), seeds=TUNE, tag="tune")
            r=np.mean([t["regret"] for t in tr])
            if best is None or r<best[0]: best=(r,thr)
        return best[1]
    grids=[(0.15+0.1*i,0.30+0.1*i,0.46+0.1*i,0.62+0.1*i,0.80+0.05*i) for i in range(4)]
    st_thr=tune(lambda thr: StaticThreshold(thr), grids)
    ew_thr=tune(lambda thr: EWMA(beta=0.7,thr=thr), grids)
    pols={"DTCOA":lambda: DTCOA(), "Static":lambda: StaticThreshold(st_thr),
          "EWMA":lambda: EWMA(beta=0.7,thr=ew_thr), "Rule":lambda: RuleBased(),
          "OracleLite":lambda: OracleLite()}
    runs={k:run_set(c, f, seeds=TEST, tag="B1_"+k) for k,f in pols.items()}
    res={}; pv=[]; lab=[]
    for m in ["regret","harm_rate","overrestrict","unauth_unsafe_rate"]:
        res[m]={k:describe([t[m] for t in runs[k]]) for k in pols}
        for k in ["Static","EWMA","Rule"]:
            w=wilcoxon_paired([t[m] for t in runs["DTCOA"]],[t[m] for t in runs[k]])
            res[m][f"DTCOA_vs_{k}_wilcoxon_p"]=w[1]
            res[m][f"DTCOA_vs_{k}_cliffs"]=cliffs_delta([t[m] for t in runs["DTCOA"]],[t[m] for t in runs[k]])
            pv.append(w[1]); lab.append((m,k))
    hb=holm_bonferroni(pv)
    res["_holm"]=[dict(metric=lab[i][0],baseline=lab[i][1],p=hb[i][1],reject=hb[i][3]) for i in range(len(lab))]
    res["_tuned_thresholds"]=dict(static=list(st_thr),ewma=list(ew_thr))
    save_json(f"{OUT}/B1.json", dict(stamp=stamp(), comparison=res)); return res

# ---------------- V7 Propagation (incl. instantaneous tau=0) ----------------
def V7():
    res={}; curve=[]
    for tau in [0,1,3,6]:
        c=cfg(n_agents=12, regime="hmm", graph_density=0.35, coupling=0.4, cascade_tau=tau, horizon=300)
        on=run_set(c, lambda: DTCOA(use_propagation=True), seeds=list(range(16)), tag=f"V7on_{tau}")
        off=run_set(c, lambda: DTCOA(use_propagation=False), seeds=list(range(16)), tag=f"V7off_{tau}")
        a=[t["sysfail_rate"] for t in on]; b=[t["sysfail_rate"] for t in off]
        w=wilcoxon_paired(a,b)
        res[f"tau_{tau}"]=dict(prop_on_sysfail=describe(a), prop_off_sysfail=describe(b),
                               delta=float(np.mean(a)-np.mean(b)), wilcoxon_p=w[1], cliffs=cliffs_delta(a,b),
                               prop_on_harm=describe([t["harm_rate"] for t in on]),
                               prop_off_harm=describe([t["harm_rate"] for t in off]))
        curve.append(dict(tau=tau, on=float(np.mean(a)), off=float(np.mean(b))))
    save_json(f"{OUT}/V7.json", dict(stamp=stamp(), by_tau=res)); save_json(f"{ARR}/V7_curve.json", curve); return res

# ---------------- S1 Scalability ----------------
def S1():
    curve=[]
    for N in [5,20,50,100,200]:
        c=cfg(n_agents=N, regime="stationary", graph_density=0.1, coupling=0.1, cascade_tau=1, horizon=100)
        t0=time.time(); _=run_trial(c, DTCOA(), seed=0, tag="S1"); dt=time.time()-t0
        per_step=dt/c["horizon"]; throughput=c["horizon"]*N/dt
        curve.append(dict(N=N, total_s=dt, per_step_s=per_step, throughput_agentsteps_per_s=throughput))
    save_json(f"{OUT}/S1.json", dict(stamp=stamp(), scaling=curve)); save_json(f"{ARR}/S1_curve.json", curve); return curve

if __name__=="__main__":
    import sys
    order=[("V1",V1),("V2",V2),("V3",V3),("V4",V4),("V5",V5),("V6",V6),("A1",A1),("B1",B1),("V7",V7),("S1",S1)]
    only=sys.argv[1:] if len(sys.argv)>1 else None
    for nm,fn in order:
        if only and nm not in only: continue
        t0=time.time(); fn(); print(f"[{nm}] done in {time.time()-t0:.1f}s", flush=True)
