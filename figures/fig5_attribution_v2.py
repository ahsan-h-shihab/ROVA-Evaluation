"""
Figure 5 - fig_attribution  (ROVA figure system)  --  v2
Base: fig5_attribution.py (current NEW). Publication-quality refinement only (NO redesign).

v2 changes (evidence-based, per the approved review; nothing else touched):
  1. Restored the negligible-effect band |delta| <= 0.147 (matches Figure 4, and
     makes decay-low's near-zero effect read as negligible at a glance).
  2. Mechanism order aligned to Table 8 / the figure caption / Section 6.3 text,
     which all read memory -> propagation -> stability -> decay (the three detected
     mechanisms first, the single not-detected mechanism last). Each mechanism keeps
     its own canonical color + marker; only left-to-right order changes.
  3. Decay non-detection is expressed for BOTH regimes with ONE concise annotation
     ("not detected / both regimes") instead of a per-point label, reinforced by the
     restored band and the zero-crossing CIs. Consistent with Table 8 (decay low
     p=0.250, CI[-0.08,+0.35]; decay high p=0.125, CI[-0.09,+0.43], flagged not
     detected) and the caption ("decay is not detected in either regime").

Everything else is IDENTICAL to the NEW version: forest-plot design (point + 95% CI),
single legend, mechanism palette, y-range [-1.05, +1.05] matched to Figure 4, and
minimalist IEEE styling. Environments remain un-encoded (they enter only via the CIs).

Editable source. Data: figure_data/fig5_attribution.csv (frozen; Table 9 / dr3_regime_inference.json)
Writes fig5_attribution_v2.pdf / .svg / .png  (does NOT overwrite the NEW files).
"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=os.path.dirname(os.path.abspath(__file__)); mm=1/25.4
plt.rcParams.update({"font.family":"Arial","pdf.fonttype":42,"svg.fonttype":"none","axes.linewidth":0.8})
INK="#1A1A1A"; GRID="#DDDDDD"; BAND="#EBEBEB"
MECH={ "memory":("#0072B2","o"), "decay":("#E69F00","s"),
       "propagation":("#009E73","^"), "stability":("#CC79A7","D") }
# CANONICAL order (2026-07-22 mechanism-ordering audit): memory, decay, propagation,
# stability -- the paper's definitional order (Sec 4 spec, Table 10, Appendix B, Fig 3
# badge). Table 8 + the Fig 5 caption are the outliers and are conformed at integration.
ORDER=["memory","decay","propagation","stability"]
# 2x2 legend fills column-major; this handle order makes the visual ROWS read in
# ORDER (memory, decay / propagation, stability) = canonical M,D,P,S.
LEGORDER=["memory","propagation","decay","stability"]
OFF={m:o for m,o in zip(ORDER,[-0.30,-0.10,0.10,0.30])}
GC={"low":0.0,"high":1.0}

rows=list(csv.DictReader(open(os.path.join(HERE,"figure_data","fig5_attribution.csv"))))
D={(r["mechanism"],r["regime"]):(float(r["delta"]),float(r["ci_low"]),float(r["ci_high"]),r["flag"]) for r in rows}

fig,ax=plt.subplots(figsize=(88.9*mm,70*mm))
fig.subplots_adjust(left=0.155, right=0.985, top=0.965, bottom=0.135)
# negligible-effect band |delta|<=0.147 (restored; identical treatment to Figure 4)
ax.axhspan(-0.147, 0.147, color=BAND, zorder=0)
ax.axhline(0, color=INK, lw=0.6, zorder=1)
# faint divider between the two regime groups
ax.axvline(0.5, color=GRID, lw=0.5, zorder=0)

for m in ORDER:
    c,mk=MECH[m]
    for reg in ("low","high"):
        d,lo,hi,flag=D[(m,reg)]
        x=GC[reg]+OFF[m]
        ax.errorbar(x, d, yerr=[[d-lo],[hi-d]], fmt=mk, color=c, markersize=4.8,
                    markerfacecolor=c, markeredgecolor=c, ecolor=c, elinewidth=1.0,
                    capsize=3, capthick=1.0, zorder=3, clip_on=False)

# Decay is not detected in EITHER regime (Table 8 / Sec 6.3 / caption).
# ONE concise cue anchored at the high-cost decay point (the one most at risk of being
# read as a positive effect); decay-low reads as null via the band + its zero-crossing CI.
dx=GC["high"]+OFF["decay"]; dhi=D[("decay","high")][2]
ax.annotate("not detected\nboth regimes", (dx,dhi), textcoords="offset points",
            xytext=(0,5), fontsize=6.5, color=INK, ha="center", va="bottom", linespacing=1.15)

ax.set_xlim(-0.6,1.6); ax.set_ylim(-1.05,1.05)
ax.set_xticks([0,1]); ax.set_xticklabels(["low-cost regime","high-cost regime"], fontsize=7)
ax.set_yticks([-1,-0.5,0,0.5,1]); ax.set_yticklabels(["-1.0","-0.5","0","+0.5","+1.0"], fontsize=7)
ax.set_ylabel("Cliff's $\\delta$ (ablated vs full)", fontsize=8)
ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
ax.tick_params(length=2.5, color=INK, labelcolor=INK)
for s in ax.spines.values(): s.set_color(INK)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax.text(0.5,-0.80,"positive = removing the component worsens regret",
        fontsize=7,color=INK,ha="center",va="center",
        bbox=dict(facecolor="white",edgecolor="none",alpha=0.85,pad=1.0))

handles=[Line2D([0],[0],color=MECH[m][0],marker=MECH[m][1],linestyle="none",
                markersize=4.8,label=m) for m in LEGORDER]
ax.legend(handles=handles,fontsize=7,loc="upper center",ncol=2,frameon=True,
          borderpad=0.4,columnspacing=1.2,handletextpad=0.4,labelspacing=0.3,edgecolor=INK)

for ext in ("pdf","svg","png"):
    fig.savefig(os.path.join(HERE,f"fig5_attribution_v2.{ext}"), dpi=(300 if ext=="png" else None))
print("saved fig5_attribution_v2.pdf / .svg / .png")
