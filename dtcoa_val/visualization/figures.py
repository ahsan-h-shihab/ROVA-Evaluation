"""Publication-quality figures from saved results/arrays. Honest reporting (incl. negatives)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R="/mnt/user-data/outputs/dtcoa_validation/results"; A="/mnt/user-data/outputs/dtcoa_validation/arrays"
F="/mnt/user-data/outputs/dtcoa_validation/figures"; os.makedirs(F, exist_ok=True)
def g(p): return json.load(open(p))
plt.rcParams.update({"figure.dpi":130,"font.size":10,"axes.grid":True,"grid.alpha":0.3})

# F1 calibration traces
tr=g(f"{A}/V1_traces.json")
fig,ax=plt.subplots(1,2,figsize=(10,3.4))
for k,a in zip(["collapse","hmm"],ax):
    a.plot(tr[k]["r"],label="latent reliability r (truth)",lw=1.6)
    a.plot(tr[k]["T"],label="DTCOA trust T (estimate)",lw=1.2,alpha=0.85)
    a.set_title(f"V1 calibration — {k}"); a.set_xlabel("step"); a.set_ylim(0,1); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{F}/F1_calibration_traces.png"); plt.close(fig)

# F2 tracking error per regime
v1=g(f"{R}/V1.json")["per_regime"]; regs=list(v1.keys())
d=[v1[r]["dtcoa_tracking_err"]["mean"] for r in regs]; n=[v1[r]["naive_tracking_err"]["mean"] for r in regs]
de=[[v1[r]["dtcoa_tracking_err"]["mean"]-v1[r]["dtcoa_tracking_err"]["ci_lo"] for r in regs],
    [v1[r]["dtcoa_tracking_err"]["ci_hi"]-v1[r]["dtcoa_tracking_err"]["mean"] for r in regs]]
x=np.arange(len(regs)); w=0.38
fig,ax=plt.subplots(figsize=(8,3.4))
ax.bar(x-w/2,d,w,yerr=de,capsize=3,label="DTCOA")
ax.bar(x+w/2,n,w,label="naive EWMA")
ax.set_xticks(x); ax.set_xticklabels(regs,rotation=20); ax.set_ylabel("mean |estimate − latent r|")
ax.set_title("V1 tracking error (lower=better)"); ax.legend()
fig.tight_layout(); fig.savefig(f"{F}/F2_tracking_error.png"); plt.close(fig)

# F3 sensitivity heatmap
v2=g(f"{R}/V2.json"); H=np.array(v2["regret_heat"])
fig,ax=plt.subplots(figsize=(5,3.4))
im=ax.imshow(H,aspect="auto",cmap="viridis")
ax.set_xticks(range(len(v2["heat_alpha"]))); ax.set_xticklabels(v2["heat_alpha"])
ax.set_yticks(range(len(v2["heat_emin"]))); ax.set_yticklabels(v2["heat_emin"])
ax.set_xlabel("alpha"); ax.set_ylabel("E_min"); ax.set_title("V2 regret sensitivity")
fig.colorbar(im,label="mean regret"); fig.tight_layout(); fig.savefig(f"{F}/F3_sensitivity_heatmap.png"); plt.close(fig)

# F4 PR + ROC
pr=g(f"{A}/V4_pr.json")
rec=[p["recall"] for p in pr]; prec=[p["precision"] for p in pr]; fpr=[p["fpr"] for p in pr]
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
ax[0].plot(rec,prec,"o-"); ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_title("V4 escalation PR")
ax[0].set_xlim(0,1); ax[0].set_ylim(0,1.05)
ax[1].plot(fpr,rec,"o-"); ax[1].plot([0,1],[0,1],"k--",alpha=0.4); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR(recall)")
ax[1].set_title("V4 escalation ROC"); ax[1].set_xlim(0,1); ax[1].set_ylim(0,1.05)
fig.tight_layout(); fig.savefig(f"{F}/F4_escalation_pr_roc.png"); plt.close(fig)

# F5 attack success
v5=g(f"{R}/V5.json")["attacks"]; ks=list(v5.keys())
dd=[v5[k]["dtcoa_attack_success"]["mean"] for k in ks]; nn=[v5[k]["nogov_attack_success"]["mean"] for k in ks]
x=np.arange(len(ks))
fig,ax=plt.subplots(figsize=(7,3.4))
ax.bar(x-0.2,dd,0.4,label="DTCOA"); ax.bar(x+0.2,nn,0.4,label="no governance")
ax.set_xticks(x); ax.set_xticklabels(ks); ax.set_ylabel("attack success (unauth-unsafe rate)")
ax.set_title("V5 adversarial — DTCOA reduces but does not prevent"); ax.legend()
fig.tight_layout(); fig.savefig(f"{F}/F5_attack_success.png"); plt.close(fig)

# F6 propagation vs tau
cur=g(f"{A}/V7_curve.json")
taus=[c["tau"] for c in cur]; on=[c["on"] for c in cur]; off=[c["off"] for c in cur]
fig,ax=plt.subplots(figsize=(6,3.4))
ax.plot(taus,on,"o-",label="propagation ON"); ax.plot(taus,off,"s-",label="propagation OFF")
ax.set_xlabel("cascade lead-time tau (0=instantaneous)"); ax.set_ylabel("systemic failure rate")
ax.set_title("V7 propagation containment"); ax.legend()
fig.tight_layout(); fig.savefig(f"{F}/F6_propagation_tau.png"); plt.close(fig)

# F7 ablation delta harm
ab=g(f"{R}/A1.json")["ablations"]; ks=[k for k in ab if not k.startswith("_")]
dh=[ab[k]["harm_rate"]["delta"] for k in ks]; sig=[ab[k]["harm_rate"]["wilcoxon_p"]<0.05 for k in ks]
fig,ax=plt.subplots(figsize=(7,3.4))
ax.bar(range(len(ks)),dh,color=["#c0392b" if s else "#95a5a6" for s in sig])
ax.axhline(0,color="k",lw=0.8); ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks,rotation=20)
ax.set_ylabel("Δ harm vs full (+ = component helps)"); ax.set_title("A1 ablation (red=significant p<0.05)")
fig.tight_layout(); fig.savefig(f"{F}/F7_ablation.png"); plt.close(fig)

# F8 baseline regret
b1=g(f"{R}/B1.json")["comparison"]["regret"]; pol=["DTCOA","Static","EWMA","Rule","OracleLite"]
mn=[b1[p]["mean"] for p in pol]
err=[[b1[p]["mean"]-b1[p]["ci_lo"] for p in pol],[b1[p]["ci_hi"]-b1[p]["mean"] for p in pol]]
fig,ax=plt.subplots(figsize=(6.5,3.4))
ax.bar(pol,mn,yerr=err,capsize=3,color=["#2980b9","#27ae60","#27ae60","#7f8c8d","#bdc3c7"])
ax.set_ylabel("authorization regret (lower=better)")
ax.set_title("B1 baselines — DTCOA worse than tuned Static/EWMA (honest null)")
fig.tight_layout(); fig.savefig(f"{F}/F8_baseline_regret.png"); plt.close(fig)

# F9 scalability
sc=g(f"{A}/S1_curve.json")
fig,ax=plt.subplots(figsize=(6,3.4))
ax.plot([s["N"] for s in sc],[s["per_step_s"]*1000 for s in sc],"o-")
ax.set_xlabel("number of agents N"); ax.set_ylabel("per-step latency (ms)")
ax.set_title("S1 scalability — near-linear overhead")
fig.tight_layout(); fig.savefig(f"{F}/F9_scalability.png"); plt.close(fig)
print("figures written to", F)
