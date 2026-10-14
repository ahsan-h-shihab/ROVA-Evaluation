# Figure provenance and render notes

Sources for the paper's data figures. This directory is a provenance layer; the
shipped figure assets live in the manuscript, not here.

| Paper figure | Source here | Regenerates shipped asset? |
|---|---|---|
| Figure 1 (evolution) | `fig_evolution_v4_source.zip` (HTML + `render.py`, 600 dpi Playwright pipeline, bundled Inter fonts, SIL Open Font License 1.1: `LICENSE-Inter-OFL.txt`) | Yes, fully |
| Figure 2 (methodology) | `fig2_methodology_v6.py` | Yes |
| Figure 4 (transition) | `fig4_transition_v3.py` + `figure_data/fig4_transition.csv` | Yes, **with a dpi override** (see below) |
| Figure 5 (attribution) | `fig5_attribution_v2.py` + `figure_data/fig5_attribution.csv` | Yes, **with a dpi override** (see below) |

## 600-dpi note (Figures 4 and 5)

The checked-in scripts specify `dpi=300`; the assets shipped with the
manuscript were exported at **600 dpi** (2100×1654 px). To reproduce the
shipped files byte-compatibly in size, change the `dpi=300` argument in the
`savefig` call to `dpi=600` (or pass it through if running interactively). The
scripts are archived exactly as last run at 300 dpi and are deliberately not
modified here; this note is the record of the production override.
