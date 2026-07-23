"""
Figure 4 - fig_transition  (ROVA figure system)  --  v3
Base: fig4_transition_v2.py. The ONLY change from v2 is the annotation refinement
approved as "Mock B" (no other stylistic changes):
  - The "delta = -1" gloss is moved from the upper (positive-delta) region to sit
    just ABOVE the flat delta=-1 segment it explains, as two lines, with a hairline
    leader to the floor. This fixes the proximity/semantic-congruence issue (a
    "ROVA better" note no longer sits in the "ROVA worse" half).
  - The now-redundant "ROVA better down-arrow" directional cue is removed: the
    floor-anchored gloss carries the "better" meaning, and "ROVA worse up-arrow"
    remains for the opposite direction.
Everything else is IDENTICAL to v2: Okabe-Ito palette, linestyle+marker redundancy,
negligible band, axes/ticks/labels, grid, spines, legend contents/placement, layout,
figure size, and the x-label "cost ratio c = H/U".

Editable source. Data: figure_data/fig4_transition.csv (frozen; Table 8 / stats_summary.json)
Writes fig4_transition_v3.pdf / .svg / .png  (does NOT overwrite v2 or earlier files).
"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
mm = 1/25.4

# ---- style guide (Part 1) ----
plt.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,        # embed TrueType in PDF
    "svg.fonttype": "none",    # keep text editable in SVG
    "axes.linewidth": 0.8,
})
INK="#1A1A1A"; GRID="#DDDDDD"; BAND="#EBEBEB"; CUE="#6E6E6E"
ENV = {  # canonical color, linestyle, marker, legend order (Part 3)
    "Favorable":  ("#D55E00","solid",   "o"),
    "Collapse":   ("#56B4E9","dashed",  "s"),
    "Unfavorable":("#4D4D4D","dotted",  "^"),
}
ORDER = ["Favorable","Collapse","Unfavorable"]
TICKLBL = ["0.5","1.5","3.5","8.0","20.0"]

# ---- data ----
rows = list(csv.DictReader(open(os.path.join(HERE,"figure_data","fig4_transition.csv"))))
series = {e:[] for e in ORDER}
for r in rows:
    series[r["environment"]].append((r["c"], float(r["delta_mean"])))
xpos = {t:i for i,t in enumerate(TICKLBL)}

# ---- figure: 88.9 x 70 mm ----
fig, ax = plt.subplots(figsize=(88.9*mm, 70*mm))
fig.subplots_adjust(left=0.155, right=0.985, top=0.965, bottom=0.145)

# negligible-effect band |d|<=0.147 and zero reference
ax.axhspan(-0.147, 0.147, color=BAND, zorder=0)
ax.axhline(0, color=INK, lw=0.6, zorder=1)

for e in ORDER:
    c,ls,mk = ENV[e]
    xs=[xpos[t] for t,_ in series[e]]; ys=[v for _,v in series[e]]
    ax.plot(xs, ys, color=c, linestyle=ls, marker=mk, markersize=4.5,
            linewidth=1.5, markerfacecolor=c, markeredgecolor=c, zorder=3, clip_on=False)

ax.set_xlim(-0.15, 4.15); ax.set_ylim(-1.05, 1.05)
ax.set_xticks(range(5)); ax.set_xticklabels(TICKLBL, fontsize=7)
ax.set_yticks([-1,-0.5,0,0.5,1]); ax.set_yticklabels(["-1.0","-0.5","0","+0.5","+1.0"], fontsize=7)
ax.set_xlabel("cost ratio $c = H/U$", fontsize=8)
ax.set_ylabel("Cliff's $\\delta$ (ROVA vs baselines)", fontsize=8)
ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
ax.tick_params(length=2.5, color=INK, labelcolor=INK)
for s in ax.spines.values(): s.set_color(INK)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# in-plot gloss (v3): delta=-1 explained AT the floor it describes, two lines above the
# flat segment, with a hairline leader to the -1 line.
ax.annotate(r"$\delta=-1$: ROVA better" + "\n" + "on every paired seed",
            xy=(3.5, -1.0), xytext=(3.5, -0.72),
            fontsize=6.5, color=INK, ha="center", va="center", linespacing=1.25,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.0),
            arrowprops=dict(arrowstyle="-", color=CUE, lw=0.7, shrinkA=2, shrinkB=2))

# directional cue: up-direction only ("better" is anchored to the -1 floor above).
ax.text(4.08, 0.86, r"ROVA worse $\uparrow$", fontsize=6.5, color=CUE,
        style="italic", ha="right", va="center")

# legend (order fixed) -- unchanged
handles=[Line2D([0],[0], color=ENV[e][0], linestyle=ENV[e][1], marker=ENV[e][2],
                markersize=4.5, linewidth=1.5, label=e) for e in ORDER]
ax.legend(handles=handles, fontsize=7, loc="lower left", frameon=True,
          borderpad=0.4, handlelength=2.2, labelspacing=0.3, edgecolor=INK)

for ext in ("pdf","svg","png"):
    fig.savefig(os.path.join(HERE,f"fig4_transition_v3.{ext}"),
                dpi=(300 if ext=="png" else None))
print("saved fig4_transition_v3.pdf / .svg / .png")
