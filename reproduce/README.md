# Replay package for the execution-derived monotonicity claims

This package lets an independent third party **verify or falsify** every
execution-based claim about ROVA's authorization monotonicity, working only
from the archived repository and the frozen implementation actually used in
**MAIN-001** and **ABLATION-001**. No source file and no manuscript file is
modified; the scripts here *import* the archived code and drive it.

The single technical fact under test:

> Authorization is computed as `a = clip(round(L − prop_pen − stab_pen), 0, 5)`
> where `L = band_level(T·(1−Rc))` is monotone in the trust state `T`, but
> `prop_pen` (propagation) and `stab_pen` (stability) are functions of
> **evidence**, not trust, and are subtracted *after* the band map. When those
> penalties **release** (decrease) from one step to the next while trust is
> still falling, authorization can rise as trust falls.
> (`code/dtcoa_val/dtcoa.py:57-68`, `code/dtcoa_val/dtcoa_tunable.py:122-135`.)

---

## Contents

| File | Purpose |
|------|---------|
| `replay_monotonicity.py` | Reproduces the **minimal synthetic counterexamples** (A1 propagation-release, A2 oversight-removal, A3 stability-release) on both archived implementations, and regenerates the narrow/formal property, boundedness, and contraction numbers. |
| `replay_archived_runs.py` | **Replays the archived MAIN-001 and ABLATION-001 runs** from the archived tuned configs + seeds, self-validates against archived regret, and reports every real per-agent-step monotonicity violation with full local variables. |
| `reproduce.sh` | Runs both scripts, tees logs, prints artifact `sha256`. |
| `expected_outputs.md` | Expected numbers + the final claim-by-claim reproduction table. |

Generated when you run the package: `out_monotonicity.json`, `out_archived.json`,
`_monotonicity.log`, `_archived.log`.

---

## Environment

- **Python 3** (archived runs used CPython 3.13; see
  `experiments/MAIN-001_core_comparison/raw/environment.txt`).
- **numpy** — the only third-party dependency. This package was regenerated with
  **numpy 2.4.2**. `numpy.random.default_rng` uses PCG64, whose integer-seeded
  stream is stable across numpy versions, so world reconstruction is
  version-independent; the last digits of floats and therefore file hashes may
  differ across numpy/BLAS builds. Regret self-validation uses a `1e-9`
  tolerance, well above that noise.
- Thread pinning (`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`) is set by
  `reproduce.sh` for cross-machine determinism, matching the archived launch.

No network access, no GPU, no other packages required.

## How to run

```bash
cd <repo root that contains code/ and experiments/>
bash reproduce/reproduce.sh
```

Or individually:

```bash
PYTHONPATH=code python3 reproduce/replay_monotonicity.py
PYTHONPATH=code python3 reproduce/replay_archived_runs.py --experiment both
```

`replay_archived_runs.py` options: `--experiment {main,ablation,both}`,
`--branches full,no_decay,...`, `--max-seeds N` (fast partial run; disables the
regret self-check), `--print-events N`, `--out PATH`.

> Note: `--max-seeds` writes to the same default `--out` (`out_archived.json`),
> so a partial run overwrites the canonical file. For a fast partial run pass an
> explicit `--out` (e.g. `--out /tmp/partial.json`). `reproduce.sh` always runs
> the full, self-validated replay.

---

## How the archived replay is reconstructed (and why it is exact)

For each cell `<env>__c<cost>__<branch>` and outer fold `i`, the archived
checkpoint `raw/hpo_checkpoints/<cell>__<branch>.json` stores the tuned
`results[i]["best_config"]`. The scored outer-test trial is rebuilt from the
archived pieces exactly as the original driver produced it
(`run_pipeline.py`, `run_ablation.py`, `nested_hpo.py`):

- **environment** `cfg_for(env, cost)` — `environments.py` (frozen).
- **policy** `DTCOATunable(**best_config)`; ablation branches additionally apply
  the archived `fixed_kwargs` (`use_trust/use_decay/use_propagation/use_stability`
  = False) via `ROVA_ABLATIONS`.
- **outer-test seeds** `kfold_split(range(60), 5, np.random.default_rng(0))[i][1]`
  — `seed0 = 0` (nested_cv_hpo default; neither driver overrides it).
- **per-trial world seed** `run_trial(cfg, policy, seed=s, tag=f"{cell}_out{i}_test")`
  — `tag` matches `nested_hpo.py:129` exactly.

**Two independent self-checks validate the replay:**

1. **Regret match.** The replayed mean outer-test regret is compared to the
   archived `test_mean_regret`. Verified match to `< 1e-9` on all 75 MAIN-001
   folds and all 375 ABLATION-001 folds. Any fold that does not match is printed
   as `NOT REPRODUCIBLE` and excluded from the violation counts.
2. **Penalty reconstruction.** At every agent-step the propagation/stability
   penalties are reconstructed from public post-step state and used to rebuild
   `clip(round(L − prop_pen − stab_pen))`; the script asserts this equals the
   policy's own returned authorization. Any mismatch flags the run
   `NOT REPRODUCIBLE`. (Observed mismatches: 0.)

The instrumentation is a **non-invasive wrapper** (`Recorder`) that delegates
`reset`/`step` to a real policy object and only *reads* its outputs and public
state — it never reimplements the policy, so it cannot silently diverge from the
archived behavior.

---

## Violation definition (used identically in both scripts)

For agent `i`, at timestep `t ≥ 1`, a **violation** is:

```
trust_state[i, t]  <  trust_state[i, t-1]        (strictly, tol 1e-12)
      AND
authorization[i, t] >  authorization[i, t-1]
```

where `trust_state` is the **post-update** trust `policy.T` (the quantity the
band map consumes). `oversight_removed` additionally requires
`authorization[i, t-1] == 0 and authorization[i, t] > 0` (escalation/A0 lifted).

---

## Scope notes

- The three **synthetic** scenarios (A1/A2/A3) are fixed input *test vectors*,
  not results; the scripts compute whether they violate. A3's sequence is
  additionally re-derived by a deterministic search (rng seed 0): the pinned
  sequence is the first one that this search finds.
- **A3 on the tuned class:** the exact A3 sequence triggers a violation on the
  frozen `DTCOA` (binary stability penalty) but not on the default
  `DTCOATunable` (its continuous penalty does not cross the rounding threshold
  for that one sequence). Stability-driven violations in the *tuned* class are
  instead demonstrated directly by the archived **single-agent**
  (`unfavorable_oscillation_uncoupled`, `n_agents=1`) cells in
  `replay_archived_runs.py`, where propagation is impossible and the only
  penalty is stability.
- These scripts demonstrate a **policy-behavior** property only. They make no
  claim about whether the reported regret verdict depends on it — the archived
  regret is scored by an external latent-state oracle on the authorization
  decisions, independent of monotonicity. See `expected_outputs.md`.
