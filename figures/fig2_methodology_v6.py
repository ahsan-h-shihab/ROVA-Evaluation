"""
Figure 2 - fig_methodology (Section 3).  ITERATION v6 (four targeted fixes over frozen v5).
v5 structure/workflow/wording/concepts are otherwise unchanged. Exactly four fixes:
  (1) tighten the header->diagram whitespace: headers 93->88 and canvas 98->93 mm (top margin unchanged);
  (2) move label 'a' clear of the Governance-mechanism -> Authorization-regret edge;
  (3) drop the redundant '(5 points)' (the five labelled cells already show it);
  (4) remove inline section refs '(§3.4)' and '(§3.7)' from the artwork (they belong in the caption).
No box moved, no arrow added/removed, no concept or manuscript term changed.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
HERE=os.path.dirname(os.path.abspath(__file__)); mm=1/25.4
plt.rcParams.update({"font.family":"Arial","pdf.fonttype":42,"svg.fonttype":"none"})
INK="#1A1A1A"; TEAL="#1B7F79"; MUTED="#6B6B6B"; LOSS="#8C8C8C"; FAINT="#DBDBDB"
W,H=181.0,93.0                                     # (1) canvas trimmed 98 -> 93 mm
fig,ax=plt.subplots(figsize=(W*mm,H*mm)); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
ax.set_aspect("equal")

def box(x,y,w,h,ec=INK,fc="white",lw=1.1):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0,rounding_size=2.5",
                 linewidth=lw,edgecolor=ec,facecolor=fc,zorder=4))
def pill(x,y,w,h,label,fs=6.7,ec=INK,fc="white",lw=1.1):
    box(x,y,w,h,ec,fc,lw); ax.text(x,y,label,fontsize=fs,color=INK,ha="center",va="center",zorder=5)
def arr(p0,p1,color=INK,lw=1.15,rad=0.0,dashed=False,ms=8):
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color,
                 linestyle=((0,(4,2)) if dashed else "-"),connectionstyle=f"arc3,rad={rad}",
                 zorder=3,shrinkA=2,shrinkB=2))
def pillar(x,y,num,title,subtitle):
    ax.add_patch(Circle((x,y),2.35,facecolor=INK,edgecolor="none",zorder=6))
    ax.text(x,y-0.05,str(num),fontsize=7.2,color="white",ha="center",va="center",fontweight="bold",zorder=7)
    ax.text(x+4.4,y,title,fontsize=8.0,fontweight="bold",color=INK,ha="left",va="center",zorder=6)
    ax.text(x-2.35,y-5.0,subtitle,fontsize=5.4,color=MUTED,ha="left",va="center",style="italic",zorder=6)

# ---------- three co-equal pillar headers ----------  (1) top headers 93 -> 88
pillar(5.8,88.0,1,"CIRCULARITY CONTROL","score the governor on latent-state ground truth it never observes")
pillar(106.0,88.0,2,"COST-RATIO RESOLUTION","repeat the whole comparison across the cost-ratio grid")
pillar(5.8,25.0,3,"SELF-AUDIT GUARDS","independent checks that the advantage is real, not an artifact")
ax.plot([101.0,101.0],[39,83],color=FAINT,lw=0.9,zorder=0)

# ---------- pillar 1: circularity control ----------
pill(16,54,19,13,"Latent state\n$(r,h)$")
pill(52,72,35,13,"Latent-state oracle\n(DP-certified)",ec=TEAL)
box(52,36,35,15)
ax.text(52,38.2,"Governance mechanism\n+ tuned baselines",fontsize=6.7,color=INK,ha="center",va="center",zorder=5)
ax.text(52,31.3,"matched-budget nested-CV tuning",fontsize=5.4,color=MUTED,ha="center",va="center",style="italic",zorder=5)
arr((25,58),(41,68),color=TEAL); ax.text(34.5,60.8,"direct",fontsize=5.6,color=TEAL,ha="center",va="center",rotation=32,rotation_mode="anchor")
arr((25,50),(41,40));            ax.text(34.5,47.3,"corrupted",fontsize=5.6,color=MUTED,ha="center",va="center",rotation=-32,rotation_mode="anchor")
ax.plot([36,69],[54,54],color=MUTED,lw=1.0,ls=(0,(5,3)),zorder=2)
ax.text(52.5,56.5,"access separation",fontsize=6.4,color=INK,ha="center",va="bottom",fontweight="bold")
ax.text(52.5,51.7,"neither channel reads the other",fontsize=5.4,color=MUTED,ha="center",va="top",style="italic")
pill(88,54,18,15,"Authorization\nregret",fs=6.9,lw=1.8)
arr((68,67),(82,58),color=TEAL); ax.text(76,63.5,"$a^\\star$",fontsize=6.6,color=TEAL)
arr((68,41),(82,50));            ax.text(76.5,43.0,"$a$",fontsize=6.6)   # (2) 'a' moved clear of the edge

# ---------- pillar 2: cost-ratio resolution ----------
arr((97.2,54),(109.5,54))
cvals=[0.5,1.5,3.5,8.0,20.0]
shades=[LOSS,"#AFAEAD","#CFCFCE","#7FB2AD",TEAL]
gx0=112.0; cw=9.0; gap=3.0; y0=50.0; ch=8.0
xc=[gx0+i*(cw+gap)+cw/2 for i in range(5)]
for i,(cv,sh) in enumerate(zip(cvals,shades)):
    ax.add_patch(Rectangle((gx0+i*(cw+gap),y0),cw,ch,facecolor=sh,edgecolor=INK,lw=0.7,zorder=3))
    ax.text(xc[i],y0-1.7,("%g"%cv),fontsize=5.4,color=MUTED,ha="center",va="top")
gmid=(xc[0]+xc[-1])/2
ax.text(gmid,61.6,"cost-ratio grid   $c=H/U$",fontsize=6.3,color=INK,ha="center")   # (3) '(5 points)' removed
ax.text(xc[0]-1.0,y0+ch+1.4,"loss",fontsize=5.8,color=LOSS,ha="left",va="bottom",fontweight="bold")
ax.text(xc[-1]+1.0,y0+ch+1.4,"win",fontsize=5.8,color=TEAL,ha="right",va="bottom",fontweight="bold")
ax.text(gmid,43.9,"per-cell verdict — symmetric rule (self-audited)",fontsize=5.8,color=INK,ha="center",style="italic")
ax.text(gmid,40.5,"shows the reversal, not a located crossover",fontsize=5.4,color=MUTED,ha="center",style="italic")  # (4) '(§3.4)' removed

# ---------- pillar 3: self-audit guards ----------
box(54,12,34,12); ax.text(54,14.0,"Containment check",fontsize=6.3,color=INK,ha="center",va="center",zorder=5)
ax.text(54,9.0,"penalty magnitudes stay non-zero",fontsize=5.4,color=MUTED,ha="center",va="center",style="italic",zorder=5)
pill(128,12,44,12,"Re-tuned attribution\n+ consistency validator",fs=6.2)
arr((58,18.0),(56,28.4),dashed=True,ms=7); ax.text(67,21.6,"reads tuned magnitudes",fontsize=5.4,color=MUTED,ha="left")
arr((71.2,12),(105.6,12)); ax.text(88.4,13.9,"complemented by",fontsize=5.4,color=MUTED,ha="center")   # (4) '(§3.7)' removed

fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
for ext in ("pdf","svg","png"):
    fig.savefig(os.path.join(HERE,f"fig2_methodology_v6.{ext}"),dpi=(300 if ext=="png" else None))
print("saved fig2_methodology_v6.pdf / .svg / .png")
