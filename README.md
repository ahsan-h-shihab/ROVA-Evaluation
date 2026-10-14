# ROVA-Evaluation

| | |
|---|---|
| **Repository** | ROVA-Evaluation — evaluation artifacts and reproducibility package |
| **Paper** | IEEE Access (submission under review, 2026) |
| **Release** | v2.0 (revision release; v1.0 accompanied the original submission) |
| **License** | MIT |
| **DOI** | To be assigned on Zenodo at this release |

Evaluation artifacts, frozen reference implementation, raw experiment outputs, and
reproducibility materials for:

> **The Verdict Is in the Protocol: A Circularity-Controlled, Cost-Ratio-Resolved
> Evaluation of Runtime Governance for Agentic AI**
> A. H. Shihab, M. Mansurova, A. Kozierkiewicz, A. Mussa, and T. Sarsembayeva.
> IEEE Access (under review, 2026).

This repository is the artifact referenced by the paper's Reproducibility
Statement (Appendix D). It contains the frozen code, the raw per-cell
checkpoints, both verdict tables (archived and corrected), the analysis
scripts, and the seed sets for all three original experiments; and, for the
revision, the original internal evaluation protocol, the deviation record,
two dated analysis plans, the revision code and its checksummed outputs (see
[Revision analyses](#revision-analyses-rev-001-rev-002)).

## Frozen experiment tags

Every experiment was executed against a tagged, frozen commit of the code in
this repository, and each revision plan was tagged when it was fixed. The tags
below are annotated and are the ones cited in the paper (Appendix D.1):

| Stage | Tag | Commit |
|---|---|---|
| MAIN-001 core cost-ratio comparison | `main-001-frozen` | `7f624ca27a86fbbbe67afc68ebb886a4edbb67a2` |
| ABLATION-001 mechanism attribution | `ablation-001-frozen` | `4533c61aa2e252698fd0c31d5bc1f9b9d04c18f0` |
| REV-001 analysis plan (dated 2026-09-29) | `revision-001-analysis-plan` | `8dbb054d4586e44fddd85435feaa760b4b6a9e97` |
| REV-001 revision analyses | `revision-001-frozen` | `053a1fa84b91644338f025154b3c4aa42da9fe9d` |
| REV-002 analysis plan (dated 2026-10-01) | `revision-002-analysis-plan` | `58affad5d44826d03720f30ddc72e37e6e2bb3d3` |
| REV-002 tuning convergence (A1) and cluster re-analysis (A2) | `revision-002-frozen` | `226186c29c9a4a151f14ea12b4fd8e6a4bdd4032` |

`git checkout <tag>` gives the exact code state used for the corresponding
stage.

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
  REV-001_revision_analyses/          REV-001 output archive (+ .sha256)
  REV-002_A1_tuning_convergence/      A1 output archive (+ .sha256)
  REV-002_A2_cluster_reanalysis/      A2 outputs, SHA256SUMS
  REV-002_P1_behaviour_a0/            P1 outputs (exploratory), SHA256SUMS
  REVISION_OUTPUTS.md                 index, digests and verification of the
                                      revision outputs
rova_revision/            revision code (REV-001, REV-002); imports dtcoa_val/
                          and never modifies it
reproduce/                self-contained replay package (see below)
abl001_analysis/          regime-level attribution inference (DR3) outputs
verdicts/                 verdict tables: archived (v1) vs corrected symmetric
                          (v2) rule, plus the script deriving v2 from archives
figures/                  sources for the paper's data figures (see RENDER_NOTE)
docs/                     experiment index, manifest, execution runbook,
                          changelog, provenance audits, release notes;
                          original internal protocol, deviation record,
                          revision analysis plans
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

Four levels, fastest first:

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
3. **Revision outputs** — archive digests, stage verifiers and checksum
   manifests for REV-001, A1, A2 and P1, with the commands in
   [`experiments/REVISION_OUTPUTS.md`](experiments/REVISION_OUTPUTS.md). The
   revision code's tests use non-experimental seeds and synthetic data:
   ```bash
   for t in rova_revision/tests/test_*.py; do PYTHONPATH=. python -m "$(echo "${t%.py}" | tr / .)"; done
   ```
4. **Full re-run** — the complete nested cross-validation campaigns:
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

These archived comparisons paired methods by seed index on method-specific
worlds. The revision re-runs the published analysis code on matched worlds
([`docs/PREREGISTRATION_DEVIATIONS.md`](docs/PREREGISTRATION_DEVIATIONS.md),
section 3); the published and matched-world labels are listed side by side in
the REV-001 output, `analysis/tables/verdicts.csv`.

## Experiments at a glance

* Original protocol (all methods, identical budgets): nested cross-validation with
  K_out = 5, K_in = 3, 80 random-search trials per inner loop, 60 seeds,
  cost-ratio grid {0.5, 1.5, 3.5, 8.0, 20.0}, three environments.
* `PILOT-001` is a budget-sensitivity pilot that predates the freeze tooling of
  the two main experiments; it ships its config, seeds, raw outputs, and report
  but has no per-run environment freeze or checksum manifest of its own.
* Candid self-audits of provenance and limitations are in
  `docs/PROVENANCE_AUDIT.md` and `docs/HOSTILE_SELF_REVIEW.md`.

## Revision analyses (REV-001, REV-002)

* [`docs/PRE_REGISTERED_PROTOCOL.md`](docs/PRE_REGISTERED_PROTOCOL.md) — the
  internal evaluation protocol written before the original runs, published
  verbatim (SHA-256 `7d87b08f1e15bdcea9997918d75658718ce3316708f45507816ee19e3af23712`).
  Despite its file name it was never registered externally.
* [`docs/PREREGISTRATION_DEVIATIONS.md`](docs/PREREGISTRATION_DEVIATIONS.md) —
  each commitment of that protocol, what the original study executed, and how
  the revision restores it; the matched-world correction; the referenced
  documents that were not retained.
* [`docs/REVISION_ANALYSIS_PLAN.md`](docs/REVISION_ANALYSIS_PLAN.md) (REV-001)
  and [`docs/REVISION_002_ANALYSIS_PLAN.md`](docs/REVISION_002_ANALYSIS_PLAN.md)
  (REV-002: A1, A2) — each dated, written after the previous results were known
  and before its own execution. Neither is a pre-registration.
* [`docs/REVISION_002_POSTRUN_ANALYSES.md`](docs/REVISION_002_POSTRUN_ANALYSES.md)
  — one exploratory analysis (P1), specified after the A1 and A2 results.
* [`rova_revision/`](rova_revision/README.md) — the revision code, and
  [`experiments/REVISION_OUTPUTS.md`](experiments/REVISION_OUTPUTS.md) — its
  outputs.

The documents written for v1.0 (the experiment index entries and reports of
MAIN-001, PILOT-001 and ABLATION-001, `docs/HOSTILE_SELF_REVIEW.md` and
`docs/RELEASE_NOTES_v1.0.md`) keep their content; `docs/CHANGELOG.md`,
`docs/EXPERIMENT_MANIFEST_FINAL.md`, `docs/HOSTILE_SELF_REVIEW.md` and
`docs/PROVENANCE_AUDIT.md` were condensed on 2026-10-15 (each names its original
text in tag `v1.0`). Where the revision analyses reach a different conclusion,
those documents are superseded.

## License and citation

MIT — see [LICENSE](LICENSE). To cite the software or the paper, see
[CITATION.cff](CITATION.cff). Questions and issues: please open a GitHub issue.
