# `rova_revision/` — code for the REV-001 revision analyses

This package executes [`docs/REVISION_ANALYSIS_PLAN.md`](../docs/REVISION_ANALYSIS_PLAN.md).
It is additive: it imports the frozen implementation `dtcoa_val/` and never modifies it.
The companion record of protocol commitments and deviations is
[`docs/PREREGISTRATION_DEVIATIONS.md`](../docs/PREREGISTRATION_DEVIATIONS.md).

The code that executes the plan is fixed at the git tag `revision-001-frozen`; the run refuses
to start from any other commit or from a modified working tree.

## Modules

| Module | Role (plan section) |
|---|---|
| `config.py` | run profiles, experiment constants, and the world-tag registry (§3) |
| `archive.py` | read-only access to archived MAIN-001 results; the frozen fold splits |
| `instrument.py` | evaluation through the frozen runner, recording per-step oracle decisions |
| `policies.py` | constant references (R1), capped baselines (R8), single-threshold ROVA (R12), learned cost-sensitive policy (R3), tabular Q-learning (R4) |
| `training.py` | nested cross-validation for the trainable policies (R3, R4) and the Q-learning stability gate |
| `tpe.py` | Tree-structured Parzen Estimator re-run of the tuning protocol (R13) |
| `coupled_dp.py` | exact sequential oracle for coupled environments (R7) |
| `overhead.py` | per-decision cost benchmark (R6) |
| `gate.py` | gate R0: reproduction of archived results (§2, E-2) |
| `env_guard.py` | execution-environment guard (§2, E-1) |
| `units.py` | enumeration and execution of work units |
| `run_revision.py` | orchestrator: stages, parallel execution, resume, checksums, archive |
| `pilot.py` | timing pilot on non-experimental seeds (§2, E-3) |
| `analysis.py` | statistical analysis, tables, figures and report (§5) |

## Profiles

| Profile | Purpose | Seeds |
|---|---|---|
| `full` | the scientific run | 0–59 (the frozen study seeds) |
| `pilot` | run-time estimate only; results are not analysed | 1000–1059 |
| `smoke` | plumbing check with shrunken worlds; no result | 1000–1005 |

## Running

Requirements: the MAIN-001 environment (Python 3.13.14; exact package pins in
[`requirements.txt`](../requirements.txt)), CPU only, one BLAS/OpenMP thread per process.
From a clean checkout of `revision-001-frozen`, with the output root outside the repository:

```bash
bash rova_revision/run_revision.sh check  /path/to/runs   # tests + environment guard
bash rova_revision/run_revision.sh pilot  /path/to/runs   # timing pilot
bash rova_revision/run_revision.sh full   /path/to/runs   # all tiers (A,B); resumable
```

`full` writes `/path/to/runs/revision_out/` and `/path/to/runs/revision_out.tar.gz` with its
SHA-256. Re-issuing the same command resumes an interrupted run. A soft environment mismatch
(hardware or package versions) aborts the run unless a written approval is supplied in the
environment variable `ROVA_ENV_APPROVAL`; the approval is recorded verbatim in
`run_metadata.json`. The commit, clean-tree and thread checks cannot be overridden.

Tests alone (non-experimental seeds and synthetic data):

```bash
PYTHONPATH=. python -m rova_revision.tests.test_core_modules
PYTHONPATH=. python -m rova_revision.tests.test_orchestration_analysis
PYTHONPATH=. python -m rova_revision.tests.test_scientific_paths
```

The analysis can be recomputed from a finished output directory without running any
simulation:

```bash
PYTHONPATH=. python -m rova_revision.analysis --profile full --out /path/to/runs/revision_out
```

## Output layout

```
revision_out/
  run_metadata.json          environment guard, versions, commit, start/end, per-stage summary
  logs/orchestrator.log
  units/<unit id>/           result.json, done.json (SHA-256 of result.json), checkpoint.json
  analysis/                  summary.json, REPORT.md, tables/*.csv, figures/*.{pdf,png}
  SHA256SUMS
```

Unit identifiers are `S<stage>__<kind>__<parameters in sorted-key order>`.
