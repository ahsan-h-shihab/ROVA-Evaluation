# `rova_revision/` — code for the REV-001 revision analyses

This package executes [`docs/REVISION_ANALYSIS_PLAN.md`](../docs/REVISION_ANALYSIS_PLAN.md).
It is additive: it imports the frozen implementation `dtcoa_val/` and never modifies it.
Companion records: [`docs/PREREGISTRATION_DEVIATIONS.md`](../docs/PREREGISTRATION_DEVIATIONS.md)
and the original internal protocol, [`docs/PRE_REGISTERED_PROTOCOL.md`](../docs/PRE_REGISTERED_PROTOCOL.md).

The code that executes the plan is fixed at the git tag `revision-001-frozen`; the run refuses
to start from any other commit or from a modified working tree.

Every between-method comparison uses matched worlds: a seed generates one world, through a
method-independent tag, and every compared method is evaluated on it (plan §3).

## Modules

| Module | Role (plan section) |
|---|---|
| `config.py` | run profiles, experiment constants, the factorial registry and the world-tag registry (§3) |
| `archive.py` | read-only access to archived MAIN-001 and ABLATION-001 results; the frozen fold splits |
| `instrument.py` | evaluation through the frozen runner, recording per-step oracle decisions |
| `policies.py` | constant references (R1), capped baselines (R8), single-threshold ROVA (R12), learned cost-sensitive policy (R3), tabular Q-learning (R4) |
| `training.py` | nested cross-validation for the trainable policies (R3, R4) and the Q-learning stability gate |
| `tpe.py` | Tree-structured Parzen Estimator re-run of the tuning protocol (R13) |
| `hpo_record.py` | random-search nested cross-validation identical to the frozen one, recording every candidate (R14) |
| `coupled_dp.py` | exhaustive sequential oracle for coupled environments and the {0, 5} check (R7) |
| `overhead.py` | per-decision cost benchmark (R6) |
| `gate.py` | gate R0: reproduction of archived results (§2, E-2) |
| `env_guard.py` | execution-environment gate (§2, E-1) |
| `units.py` | enumeration and execution of work units |
| `run_revision.py` | orchestrator: stages, parallel execution, resume, checksums, archive |
| `pilot.py` | timing pilot on non-experimental seeds (§2, E-3) |
| `stathelp.py` | paired comparisons, Holm families, descriptive statistics, Bayesian ROPE |
| `decision_rules.py` | protocol-family Holm, DR1–DR8, crossover location, factorial analysis (R15, R17) |
| `matched_pipeline.py` | the published analysis code re-run unchanged on matched worlds (R2(d)) |
| `analysis.py` | statistical analysis, tables, figures and report (§5) |
| `verify_outputs.py` | verification of a finished output directory (checksums, unit digests, completeness, gate, environment record) |

## Profiles and tiers

| Profile | Purpose | Seeds |
|---|---|---|
| `full` | the scientific run | 0–59 (the frozen study seeds) |
| `pilot` | run-time estimate only; results are not analysed | 1000–1059 |
| `smoke` | plumbing check with shrunken worlds; no result | 1000–1005 |

Tiers: A (core revision analyses and the matched-world correction), B (R5b, R11–R13),
C (R14–R16). The default runs all three; a later invocation with more tiers resumes and adds.

## Running

Requirements: the MAIN-001 environment (Python 3.13.14; exact package pins in
[`requirements.txt`](../requirements.txt)), CPU only, one BLAS/OpenMP thread per process.
From a clean checkout of `revision-001-frozen`, with the output root outside the repository:

```bash
bash rova_revision/run_revision.sh check  /path/to/runs   # tests + environment gate
bash rova_revision/run_revision.sh pilot  /path/to/runs   # timing pilot
bash rova_revision/run_revision.sh full   /path/to/runs   # tiers A,B,C; resumable
bash rova_revision/run_revision.sh verify /path/to/runs   # verify the finished output
```

`full` writes `/path/to/runs/revision_out/` and `/path/to/runs/revision_out.tar.gz` with its
SHA-256. Re-issuing the same command resumes an interrupted run. Every invocation first prints
and records the environment table (observed versus the archived reference). A soft mismatch
(hardware, OS, kernel, libraries, BLAS or worker count) aborts the run unless a written approval
is supplied in the environment variable `ROVA_ENV_APPROVAL`; the approval is recorded verbatim
in `run_metadata.json`. The commit, clean-tree, thread and no-GPU checks cannot be overridden.

Tests alone (non-experimental seeds and synthetic data):

```bash
PYTHONPATH=. python -m rova_revision.tests.test_core_modules
PYTHONPATH=. python -m rova_revision.tests.test_matched_worlds
PYTHONPATH=. python -m rova_revision.tests.test_env_gate
PYTHONPATH=. python -m rova_revision.tests.test_decision_rules
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
  run_metadata.json          environment gate record, versions, commit, start/end, per-stage summary
  logs/orchestrator.log
  logs/environment_gate.txt  the environment table of every invocation
  units/<unit id>/           result.json, done.json (SHA-256 of result.json), checkpoint.json
  analysis/                  summary.json, REPORT.md, tables/*.csv, figures/*.{pdf,png}
  analysis/matched_world/    the published analysis code re-run on matched-world inputs
  SHA256SUMS
```

Unit identifiers are `S<stage>__<kind>__<parameters in sorted-key order>`.
