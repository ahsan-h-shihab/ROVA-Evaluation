# ROVA-Evaluation

| | |
|---|---|
| **Repository** | ROVA-Evaluation — evaluation artifacts and reproducibility package |
| **Paper** | IEEE Access (submission under review, 2026) |
| **Release** | v1.0 (frozen release) |
| **License** | MIT |
| **DOI** | To be assigned upon archival release |

Evaluation artifacts, frozen reference implementation, raw experiment outputs, and
reproducibility materials for:

> **The Verdict Is in the Protocol: A Circularity-Controlled, Cost-Ratio-Resolved
> Evaluation of Runtime Governance for Agentic AI**
> A. H. Shihab, M. Mansurova, A. Kozierkiewicz, A. Mussa, and T. Sarsembayeva.
> IEEE Access (under review, 2026).

This repository is the artifact referenced by the paper's Reproducibility
Statement (Appendix D). It contains the frozen code, the raw per-cell
checkpoints, both verdict tables (archived and corrected), the analysis
scripts, and the seed sets for all three experiments.

## Frozen experiment tags

The two experiments were executed against tagged, frozen commits of the code in
this repository. The tags below are annotated and correspond byte-for-byte to
the commits cited in the paper (Appendix D.1):

| Experiment | Tag | Commit |
|---|---|---|
| MAIN-001 core cost-ratio comparison | `main-001-frozen` | `7f624ca27a86fbbbe67afc68ebb886a4edbb67a2` |
| ABLATION-001 mechanism attribution | `ablation-001-frozen` | `4533c61aa2e252698fd0c31d5bc1f9b9d04c18f0` |

`git checkout main-001-frozen` (or `ablation-001-frozen`) reproduces the exact
code state used for the corresponding experiment.

## Repository layout

```
dtcoa_val/                frozen reference implementation (simulator, framework,
                          baselines, oracles, tuning, analysis)
experiments/
  MAIN-001_core_comparison/           config, seeds, raw checkpoints, archived
                                      stats_summary.json (v1 verdicts), reports
  ABLATION-001_mechanism_attribution/ config, seeds, raw checkpoints, analysis,
                                      SHA256SUMS, immutable archive
  PILOT-001_budget_sensitivity/       budget-sensitivity pilot (config, seeds,
                                      raw outputs, report)
  SHA256SUMS-MAIN-001                 integrity manifest for MAIN-001 (88 files)
reproduce/                self-contained replay package (see below)
abl001_analysis/          regime-level attribution inference (DR3) outputs
verdicts/                 verdict tables: archived (v1) vs corrected symmetric
                          (v2) rule, plus the script deriving v2 from archives
figures/                  sources for the paper's data figures (see RENDER_NOTE)
docs/                     experiment index, manifest, execution runbook,
                          changelog, provenance audits, release notes
```

Layout note: the archived replay scripts in `reproduce/` were written inside a
tree where the code lived at `code/dtcoa_val`. The repository provides a
committed symlink `code/dtcoa_val -> ../dtcoa_val` for compatibility. On
Windows checkouts without symlink support, set `PYTHONPATH` to the repository
root and run the replay scripts directly.

## Environment

* Python **3.13.14** (archived interpreter; see
  `experiments/MAIN-001_core_comparison/raw/environment.txt` for the complete
  frozen freeze, including the host kernel).
* `pip install -r requirements.txt` installs the exact pinned versions
  (numpy 2.5.1, scipy 1.18.0, matplotlib 3.11.0, and transitive pins).
* The replay package needs **numpy only**; no network, no GPU.
* Determinism: all randomness derives from integer-seeded PCG64
  (`numpy.random.default_rng`); seeds are `0..59` per condition
  (`experiments/*/seeds.json`). Trial seeds derive from sha256 of
  `base|index|tag` (`dtcoa_val/utils.py`).

## Verifying the paper's claims

Three levels, fastest first:

1. **Integrity** — checksums over the raw artifacts:
   ```bash
   cd experiments && sha256sum -c SHA256SUMS-MAIN-001          # 88 files
   cd ABLATION-001_mechanism_attribution && sha256sum -c SHA256SUMS  # 132 files
   ```
2. **Replay** — recompute the execution-derived claims from the archived
   configs and seeds without re-running the campaigns:
   ```bash
   bash reproduce/reproduce.sh
   ```
   The replay self-validates against archived regret at `1e-9` tolerance and
   prints a claim-by-claim table; expected numbers are in
   `reproduce/expected_outputs.md`.
3. **Full re-run** — the complete nested cross-validation campaigns:
   see `docs/MAIN-001_EXECUTION_RUNBOOK.md` for the exact launch, resume, and
   archival commands. The core comparison (45 tuning units, 15 shards) took
   about one hour of CPU wall-clock on a 16-core AMD Ryzen 9 9950X
   (59 GiB RAM, Ubuntu), single-threaded shards, no GPU.

## Verdict tables (v1 archived vs v2 corrected)

`experiments/MAIN-001_core_comparison/stats_summary.json` carries the verdict
labels computed at freeze time by `analysis.py` under the **v1 (asymmetric)**
rule. The paper reports verdicts under the **v2 (symmetric)** rule: a definite
verdict requires a Holm-significant Wilcoxon result *and* a non-negligible
effect size (|Cliff's δ| > 0.147) against **both** baselines in the same
direction. Re-derive v2 from the archives at any time:

```bash
python verdicts/verdicts_v2.py
```

Exactly one of the 15 cell labels differs between the rules
(`collapse_moderate_coupled`, c = 0.5: `ROVA_loses` → `mixed`); every effect
size, p-value, and the overall outcome are identical. See `verdicts/README.md`.

## Experiments at a glance

* Protocol (all methods, identical budgets): nested cross-validation with
  K_out = 5, K_in = 3, 80 random-search trials per inner loop, 60 seeds,
  cost-ratio grid {0.5, 1.5, 3.5, 8.0, 20.0}, three environments.
* `PILOT-001` is a budget-sensitivity pilot that predates the freeze tooling of
  the two main experiments; it ships its config, seeds, raw outputs, and report
  but has no per-run environment freeze or checksum manifest of its own.
* Candid self-audits of provenance and limitations are in
  `docs/PROVENANCE_AUDIT.md` and `docs/HOSTILE_SELF_REVIEW.md`.

## License and citation

MIT — see [LICENSE](LICENSE). To cite the software or the paper, see
[CITATION.cff](CITATION.cff). Questions and issues: please open a GitHub issue.
