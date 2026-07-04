"""v2 figures from C1-fixed results_v2 (paired stats, CIs, both-side error bars, autonomy annotation)."""
import json,os,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
N="/mnt/user-data/outputs/dtcoa_validation/results_v2"; A="/mnt/user-data/outputs/dtcoa_validation/arrays_v2"
F="/mnt/user-data/outputs/dtcoa_validation/figures_v2"; os.makedirs(F,exist_ok=True)
g=lambda p: json.load(open(p)); plt.rcParams.update({"figure.dpi":130,"font.size":10,"axes.grid":True,"grid.alpha":0.3})
def er(d): return [d['mean']-d['ci_lo'], d['ci_hi']-d['mean']]

# F2 tracking error (both error bars) — corrected
v1=g(f"{N}/V1.json")["per_regime"]; regs=list(v1)
d=[v1[r]["dtcoa_tracking_err"]["mean"] for r in regs]; de=np.array([er(v1[r]["dtcoa_tracking_err"]) for r in regs]).T
nv=[v1[r]["naive_tracking_err"]["mean"] for r in regs]; ne=np.array([er(v1[r]["naive_tracking_err"]) for r in regs]).T
x=np.arange(len(regs)); w=0.38; fig,ax=plt.subplots(figsize=(8,3.4))
ax.bar(x-w/2,d,w,yerr=de,capsize=3,label="DTCOA"); ax.bar(x+w/2,nv,w,yerr=ne,capsize=3,label="naive EWMA")
ax.set_xticks(x); ax.set_xticklabels(regs,rotation=20); ax.set_ylabel("mean |estimate − latent r| (95% CI)")
ax.set_title("V1 tracking error — C1-fixed (paired)"); ax.legend(); fig.tight_layout(); fig.savefig(f"{F}/F2_tracking_error.png"); plt.close()

# F5 attack success (system-wide metric; M2 caveat in title)
v5=g(f"{N}/V5.json")["attacks"]; ks=[k for k in v5 if not k.startswith("_")]
dd=[v5[k]["dtcoa_attack_success"]["mean"] for k in ks]; dde=np.array([er(v5[k]["dtcoa_attack_success"]) for k in ks]).T
nn=[v5[k]["nogov_attack_success"]["mean"] for k in ks]; nne=np.array([er(v5[k]["nogov_attack_success"]) for k in ks]).T
x=np.arange(len(ks)); fig,ax=plt.subplots(figsize=(7,3.6))
ax.bar(x-0.2,dd,0.4,yerr=dde,capsize=3,label="DTCOA"); ax.bar(x+0.2,nn,0.4,yerr=nne,capsize=3,label="no governance")
ax.set_xticks(x); ax.set_xticklabels(ks); ax.set_ylabel("system-wide unauth-unsafe rate (95% CI)")
ax.set_title("V5 — system-wide (NOT adversary-specific: M2 unrepaired)"); ax.legend(); fig.tight_layout(); fig.savefig(f"{F}/F5_attack_success.png"); plt.close()

# F6 propagation vs tau + autonomy annotation (exposes M3)
v7=g(f"{N}/V7.json")["by_tau"]; cur=g(f"{A}/V7_curve.json")
taus=[c["tau"] for c in cur]; on=[c["on"] for c in cur]; off=[c["off"] for c in cur]
fig,ax=plt.subplots(figsize=(6.4,3.8))
ax.plot(taus,on,"o-",label="prop ON"); ax.plot(taus,off,"s-",label="prop OFF")
for t in taus:
    k=f"tau_{t}"; ax.annotate(f"aut {v7[k]['prop_on_mean_autonomy']:.2f}/{v7[k]['prop_off_mean_autonomy']:.2f}",
        (t,on[taus.index(t)]),textcoords="offset points",xytext=(4,-12),fontsize=7,color="#444")
ax.set_xlabel("cascade lead-time tau"); ax.set_ylabel("systemic failure rate")
ax.set_title("V7 — prop ON lowers sysfail AND autonomy (M3 confound)"); ax.legend(); fig.tight_layout(); fig.savefig(f"{F}/F6_propagation_tau.png"); plt.close()

# F7 ablation Δharm with Holm significance — corrected
ab=g(f"{N}/A1.json")["ablations"]; ks=[k for k in ab if not k.startswith("_")]
holm={x['ablation']:x['reject'] for x in ab['_holm'] if x['metric']=='harm_rate'}
dh=[ab[k]["harm_rate"]["delta"] for k in ks]
fig,ax=plt.subplots(figsize=(7.2,3.6))
ax.bar(range(len(ks)),dh,color=["#c0392b" if holm.get(k) else "#95a5a6" for k in ks])
ax.axhline(0,color="k",lw=0.8); ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks,rotation=20)
ax.set_ylabel("Δ harm vs full (+ = helps)"); ax.set_title("A1 ablation — C1-fixed (red=Holm-significant)")
fig.tight_layout(); fig.savefig(f"{F}/F7_ablation.png"); plt.close()

# F8 baseline regret with CIs — corrected
b1=g(f"{N}/B1.json")["comparison"]["regret"]; pol=["DTCOA","Static","EWMA","Rule","OracleLite"]
mn=[b1[p]["mean"] for p in pol]; err=np.array([er(b1[p]) for p in pol]).T
fig,ax=plt.subplots(figsize=(6.6,3.6))
ax.bar(pol,mn,yerr=err,capsize=3,color="#2980b9")
ax.set_ylabel("authorization regret (95% CI, lower=better)")
ax.set_title("B1 — DTCOA worse than tuned Static/EWMA (valid paired, p=1.2e-7)")
fig.tight_layout(); fig.savefig(f"{F}/F8_baseline_regret.png"); plt.close()
print("v2 figures ->",F)
