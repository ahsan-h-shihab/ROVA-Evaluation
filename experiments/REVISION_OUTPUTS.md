# Revision outputs (REV-001 and REV-002)

The outputs of the revision analyses, exactly as produced; nothing in these directories was
modified after its run. The two archives are the ones the paper identifies by SHA-256 in
Appendix D.1. The plans, the original internal protocol and the deviation record are in
[`docs/`](../docs/); the code is in [`rova_revision/`](../rova_revision/).

| Directory | Analysis | Plan | Code | Run |
|---|---|---|---|---|
| `REV-001_revision_analyses/` | REV-001 revision analyses | [`REVISION_ANALYSIS_PLAN.md`](../docs/REVISION_ANALYSIS_PLAN.md), tag `revision-001-analysis-plan` | tag `revision-001-frozen` (`053a1fa`) | lab workstation, 2026-09-30 19:57 to 2026-10-01 08:54 (UTC+05:00) |
| `REV-002_A1_tuning_convergence/` | A1 tuning convergence | [`REVISION_002_ANALYSIS_PLAN.md`](../docs/REVISION_002_ANALYSIS_PLAN.md), tag `revision-002-analysis-plan` | tag `revision-002-frozen` (`226186c`) | lab workstation, 2026-10-01 16:39 to 19:24 (UTC+05:00) |
| `REV-002_A2_cluster_reanalysis/` | A2 fold-clustered re-analysis | the same plan, section 3 | tag `revision-002-frozen` (`226186c`) | separate computer, 2026-10-01; no simulation |
| `REV-002_P1_behaviour_a0/` | P1 behavioural comparison with always-full-supervision (exploratory) | [`REVISION_002_POSTRUN_ANALYSES.md`](../docs/REVISION_002_POSTRUN_ANALYSES.md), written after the A1 and A2 results | commit `f094e20` | separate computer, 2026-10-02; no simulation |

The lab workstation is the machine of the original runs (AMD Ryzen 9 9950X, Ubuntu, Python
3.13.14); each archive's `logs/environment_gate.txt` records the environment check of every
invocation against the original run's records. A2 and P1 record their environment in
`run_metadata.json`.

## Identity

| Item | SHA-256 |
|---|---|
| `REV-001_revision_analyses/revision_out.tar.gz` | `63dae057d63c4917d90391fd9da72279f5e909ffb1a1b7e192999f55d28a6b10` |
| `revision_out/SHA256SUMS` inside it (1,883 entries) | `12a72b84d635aee108950808fd8d0f7d901c56be0289b63ddf05c0f325577cc4` |
| `REV-002_A1_tuning_convergence/a1_out.tar.gz` | `35459e1e096135531849a3a96d3f7501392c8d4f10474f00bc83913baa908e03` |
| `a1_out/SHA256SUMS` inside it (182 entries) | `36a54f1644c23c429429d4c8e479cf4f8b72b848839c1f4a46a02a8ee15d1899` |
| `REV-002_A2_cluster_reanalysis/SHA256SUMS` (5 entries) | `9a142ad0e3bdb1f2fe47bc46c19ab0e2e14ea212fbdaf60f4951238b8dca2fb9` |
| `REV-002_P1_behaviour_a0/SHA256SUMS` (4 entries) | `c992e2b5bcac4e8ddc60aa0cf535d2c731020cb4eca024c47f0bab17ced4b93c` |

## Verifying

From the repository root, in the Python environment of [`requirements.txt`](../requirements.txt).
No simulation is run.

```bash
# archive digests
(cd experiments/REV-001_revision_analyses && sha256sum -c revision_out.tar.gz.sha256)
(cd experiments/REV-002_A1_tuning_convergence && sha256sum -c a1_out.tar.gz.sha256)

# extract outside the repository and run the stage verifiers
OUT=/tmp/rova_outputs; mkdir -p "$OUT"
tar xzf experiments/REV-001_revision_analyses/revision_out.tar.gz -C "$OUT"
tar xzf experiments/REV-002_A1_tuning_convergence/a1_out.tar.gz -C "$OUT"
PYTHONPATH=. python -m rova_revision.verify_outputs --profile full --out "$OUT/revision_out"
PYTHONPATH=. python -m rova_revision.verify_a1 --profile full --out "$OUT/a1_out"

# A2 and P1 checksum manifests
(cd experiments/REV-002_A2_cluster_reanalysis && sha256sum -c SHA256SUMS)
(cd experiments/REV-002_P1_behaviour_a0 && sha256sum -c SHA256SUMS)
```

Each stage verifier exits with status 0 only if every check passes: the checksum of every
file, completeness of all work units and their recorded digests, a passed reproduction gate,
the digest of the analysis summary, and a passed environment gate at every invocation. The A1
verifier also requires the recorded reproduction checks (among them the exact reproduction of
the REV-001 TPE run) to have passed and the convergence constants to equal the plan's.

A2 and P1 read only the stored outputs above and verify the SHA-256 fingerprints of their
inputs before use. They can be recomputed with

```bash
git checkout revision-002-frozen   # A2 runs only from a clean checkout of this tag
PYTHONPATH=. python -m rova_revision.clustered \
    --rev001-archive experiments/REV-001_revision_analyses/revision_out.tar.gz --work WORK_DIR --out A2_OUT
PYTHONPATH=. python -m rova_revision.behaviour_a0 \
    --rev001-dir "$OUT/revision_out" --a1-dir "$OUT/a1_out" --out P1_OUT
```

On the computer that produced them, a second run of each reproduced every file byte for
byte. `run_metadata.json` records the software environment and therefore differs on another
computer.

## Where to find what

In `revision_out/analysis/`: `REPORT.md` (the complete auto-generated report), `summary.json`,
`tables/*.csv` (among them the adversarial per-profile tables `R5a_*` and `R5b_*`, the
simulator-perturbation verdicts `R9a_*` and `R9b_*`, the decision rules `decision_rules_*` and
the factorial removal tests), `figures/` (among them the critical-difference diagrams for the
low-cost, transition and high-cost regimes, `R2g_cd_*`) and `matched_world/` (the published
analysis code re-run unchanged on matched worlds). Per-unit results are in
`revision_out/units/<unit id>/`.

In `a1_out/analysis/`: `REPORT.md`, `summary.json`, the convergence, verdict and family tables
and the convergence curves.

A2: `A2_REPORT.md`, `A2_clustered.csv` (the cluster-level table of all 300 comparisons),
`A2_G.csv` and `A2_summary.json`. P1: `BEHAVIOUR_A0_REPORT.md`, `behaviour_a0.csv` and
`behaviour_a0.json`.

## Not included

The second (determinism) runs of A2 and P1, which are byte-identical to the directories here,
and the console transcripts of the workstation runs, whose environment tables and run logs are
contained in each archive's `logs/`.
