import json,os,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/mnt/user-data/outputs/dtcoa_validation/results_v3"; F="/mnt/user-data/outputs/dtcoa_validation/figures_v3"; os.makedirs(F,exist_ok=True)
g=lambda p: json.load(open(f"{R}/{p}")); plt.rcParams.update({"figure.dpi":130,"font.size":9,"axes.grid":True,"grid.alpha":0.3})
def er(d): return [d['mean']-d['ci_lo'],d['ci_hi']-d['mean']]

# V4: W-spec recall curve + DTCOA vs rule recall + proportionality
v=g("V4.json"); W=v["W_specification_curve"]; ws=sorted(int(k) for k in W); rec=[W[str(w)]['mean'] for w in ws]
fig,ax=plt.subplots(1,3,figsize=(12,3.4))
ax[0].plot(ws,rec,"o-"); ax[0].set_xlabel("response window W"); ax[0].set_ylabel("episode recall"); ax[0].set_title("V4 W-specification curve"); ax[0].set_ylim(0,1)
ax[1].bar(["DTCOA\n(A==0)","Rule"],[v["dtcoa_recall"]["mean"],v["rule_recall"]["mean"]],
          yerr=np.array([er(v["dtcoa_recall"]),er(v["rule_recall"])]).T,capsize=3,color=["#c0392b","#7f8c8d"])
ax[1].set_ylabel("episode recall (W=3)"); ax[1].set_title("V4 escalation coverage")
ax[2].bar(["proportionality\nρ(h,5−A)","precision","false-alarm"],
          [v["dtcoa_proportionality_rho"]["mean"],v["dtcoa_precision"]["mean"],v["dtcoa_false_alarm"]["mean"]],color="#2980b9")
ax[2].set_ylim(0,1); ax[2].set_title("V4 graded response (mitigating)")
fig.tight_layout(); fig.savefig(f"{F}/V4_escalation.png"); plt.close()

# V5: comparator bars + strength worst-case
v=g("V5.json")["attacks"]; sc=["ramp_defect","collusion"]; comp=["dtcoa","nogov","tuned_static","adversary_disabled_background"]
lab=["DTCOA","no-gov","tuned static","bg(adv off)"]; x=np.arange(len(sc)); w=0.2
fig,ax=plt.subplots(1,2,figsize=(11,3.6))
for i,(ck,cl) in enumerate(zip(comp,lab)):
    ax[0].bar(x+(i-1.5)*w,[v[s][ck]["mean"] for s in sc],w,
              yerr=np.array([er(v[s][ck]) for s in sc]).T,capsize=2,label=cl)
ax[0].set_xticks(x); ax[0].set_xticklabels(sc); ax[0].set_ylabel("adversary-specific attack success"); ax[0].set_title("V5 attack success (defect window)"); ax[0].legend(fontsize=7)
cur=v["strength_curve"]
sp=sorted(set(c["spike"] for c in cur))
for s in sp:
    pts=[c for c in cur if c["spike"]==s]; pts=sorted(pts,key=lambda d:d["defect_t"])
    ax[1].plot([p["defect_t"] for p in pts],[p["dtcoa"] for p in pts],"o-",label=f"spike {s}")
ax[1].set_xlabel("defect_t"); ax[1].set_ylabel("DTCOA attack success"); ax[1].set_title("V5 pre-registered strength curve"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{F}/V5_attack.png"); plt.close()

# V7: matched Δ by tau + targeting rho
v=g("V7.json")["by_tau"]; taus=[0,1,3,6]
on=[v[f"tau_{t}"]["prop_on_sysfail"]["mean"] for t in taus]; mb=[v[f"tau_{t}"]["matched_blind_sysfail"]["mean"] for t in taus]
tg=[v[f"tau_{t}"]["targeting_rho"]["mean"] for t in taus]
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
ax[0].plot(taus,on,"o-",label="prop-ON"); ax[0].plot(taus,mb,"s-",label="autonomy-matched blind")
ax[0].set_xlabel("tau"); ax[0].set_ylabel("systemic failure"); ax[0].set_title("V7 vs autonomy-matched control"); ax[0].legend()
ax[1].bar([str(t) for t in taus],tg,color="#c0392b"); ax[1].axhline(0,color="k",lw=0.8)
ax[1].set_xlabel("tau"); ax[1].set_ylabel("targeting ρ(restriction, true influence)"); ax[1].set_title("V7 targeting (negative = anti-targeted)")
fig.tight_layout(); fig.savefig(f"{F}/V7_attribution.png"); plt.close()
print("v3 figures ->",F)
