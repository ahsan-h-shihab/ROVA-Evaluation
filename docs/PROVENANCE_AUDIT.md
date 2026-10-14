# Provenance audit — frozen implementation

A read-only audit made during the July 2026 verification of the package, from git, repository and execution
evidence; it modified no source, manuscript or experiment artifact. Findings and confidence levels unchanged; wording
condensed on 2026-10-15. Original text: tag `v1.0` (`docs/PROVENANCE_AUDIT.md`, git blob
`19984351012c87b7a3f96a6d4500ae6ba5ce84dd`).

**Two frozen commits, not one.** MAIN-001 was produced by `7f624ca`; ABLATION-001 by `4533c61`. Treating `4533c61`
as the implementation behind both experiments is incorrect. The audit establishes what can be shown for each.

## 1. Git provenance

- Two annotated tags resolve to the full hashes recorded in the artifacts: `main-001-frozen` →
  `7f624ca27a86fbbbe67afc68ebb886a4edbb67a2`; `ablation-001-frozen` → `4533c61aa2e252698fd0c31d5bc1f9b9d04c18f0`.
- Linear history: `7f624ca` → five ablation commits → `4533c61` (branch `ablation-001`). Branches `master` and
  `ablation-001` only; no stashes; one worktree; `git fsck` finds no dangling commits.
- Hashes recorded in the artifacts: MAIN-001 `config.json` has `git_commit = git_commit_expected = 7f624ca…` and
  `git_commit_match: true`; ABLATION-001 `config.json` has `git_commit = 4533c61…`, `match: true`, `git_branch:
  ablation-001`, `git_tag: ablation-001-frozen`, `git_anchor: branched from main-001-frozen (7f624ca)`.
- The policy files `dtcoa_val/dtcoa.py` and `dtcoa_val/dtcoa_tunable.py` were introduced in `7f624ca` and never
  modified afterwards (`git log` on them returns only `7f624ca`).
- Blob identity: `dtcoa_tunable.py` = `593bbc67…` and `dtcoa.py` = `bb68c0b8…` at `7f624ca`, at `4533c61` and at the
  audited HEAD. `git diff 7f624ca 4533c61` shows 0 changed lines in `dtcoa.py`, `dtcoa_tunable.py`, `runner.py`,
  `sim_core/core.py`, `environments.py` and `nested_hpo.py`; only `search_space.py` (30 lines, all in
  `ROVA_ABLATIONS` for FIX-1; `ROVA_SEARCH_SPACE`, MAIN-001's space, unchanged) and new ablation driver, analysis and
  test files differ.
- Tracked files were clean (only untracked `__pycache__/`).

## 2. Execution provenance

- MAIN-001 driver: `python3 -m dtcoa_val.experiments.run_pipeline --budget full` (`MAIN-001_EXECUTION_RUNBOOK.md`
  lines 63 and 68); `REPORT.md` refers to `run_pipeline._mark_done`; `run_pipeline.py:38` imports `DTCOATunable`.
- ABLATION-001 driver: `run_ablation.py`. `config.json:127` names `validate_consistency`, `dr3_attribution` and
  `dr3_regime_inference` as "committed modules at git 4533c61"; `raw/manifest.shard*.json` are its shard manifests;
  `run_ablation.py:55` and `ablation_study.py:10` import `DTCOATunable`.
- The artifact schema matches the tunable class: ROVA's `best_config` keys are exactly `ROVA_SEARCH_SPACE`
  (`alpha, bands, context_risk, e_min, gamma, k, prop_scale, s_min, stab_scale`), including the continuous
  `prop_scale`/`stab_scale` that only `DTCOATunable` accepts (the frozen `DTCOA` uses integer `ceil` penalties).
- Determinism recorded: `ABLATION-001/config.json:128` — "nested_cv_hpo deterministic given (seed0=0, run tag);
  same-tag reruns reproduce checkpoints bit-for-bit" — consistent with `run_pipeline.py:105`, which builds
  `hpo_kwargs` without `seed0` (default 0).
- Run environment captured: both `environment.txt` record Python 3.13.14 and numpy 2.5.1.

## 3. Import provenance

- The drivers (`run_pipeline.py`, `run_ablation.py`, `ablation_study.py`, `benchmark_ablation.py`) import only
  `from dtcoa_val.dtcoa_tunable import DTCOATunable`. `campaign.py` and `campaign_v2.py` import `DTCOA` but are not
  drivers: their outputs (`V1/A1/B1/V5/V7.json`) appear nowhere in the archives.
- No dynamic imports or path manipulation (`importlib`, `__import__`, `exec(`, `eval(`, `sys.path`, `PYTHONPATH`,
  `SourceFileLoader`: none), and no second copy of `dtcoa_tunable.py` anywhere on the filesystem.

## 4. Artifact provenance

- Replay: running the archived tuned configurations and seeds through the code (blob `593bbc6`, identical in both
  frozen commits) reproduces archived `test_mean_regret` to < 1e-9 on all 450 folds (75 MAIN-001 + 375
  ABLATION-001), with 0 penalty-reconstruction mismatches.
- An internal transfer snapshot of 2026-07-08 (not published; `sha256 -c` OK) contains `dtcoa_tunable.py` =
  `593bbc6…` and `dtcoa.py` = `bb68c0b…` and the embedded `code/.git` history.
- Integrity gap: `ABLATION-001/SHA256SUMS` covers results, report and configuration but no code. Code integrity
  rests on git blob identity, `git_commit_match`, the transfer snapshot and the exact replay — not on a code
  checksum manifest.

## 5. Alternative implementations

- Only two DTCOA definitions exist in the tree (`dtcoa.py`, `dtcoa_tunable.py`; the `reproduce/` scripts import
  them). No notebooks, pickles, backups or serialised policies.
- The local MAIN-001 and ABLATION-001 output directories hold only analysis scripts (`analysis.py`,
  `rebuild_and_validate.py`).
- A single git repository in the project; no rival repository and no unreachable commit with a different policy.

## 6. Remaining uncertainty

- Unknown: the exact bytecode executed on 2026-07-08/09 (no `.pyc` hashes were recorded). Mitigated, not
  eliminated, by the exact numeric replay.
- Unknown: whether numpy 2.5.1 produced the archived numbers bit for bit; the replay used numpy 2.4.2 and matched to
  1e-9. Code provenance is unaffected (deterministic PCG64, identical source).

## Evidence chain

```
MAIN-001 config.git_commit = 7f624ca (match)        ┐
ABLATION-001 config.git_commit = 4533c61 (match)    ├─ recorded hashes resolve to the tags
main-001-frozen → 7f624ca, ablation-001-frozen → 4533c61 ┘
policy blobs 593bbc6 / bb68c0b identical at 7f624ca, 4533c61 and HEAD; 0-line diff between frozen commits
drivers import only DTCOATunable (no dynamic or shadow imports)
artifact schema = ROVA_SEARCH_SPACE incl. continuous prop_scale/stab_scale
replay of blob 593bbc6 reproduces archived regret < 1e-9 on 450/450 folds
transfer snapshot (sha256 OK) carries 593bbc6, bb68c0b and the git history
```

## Confidence

| Claim | Level | Basis |
|---|---|---|
| The policy implementation (`dtcoa_tunable.py` 593bbc6, `dtcoa.py` bb68c0b) is the code behind both experiments | Verified | identical blobs in both producing commits; drivers import only this class; tunable-only schema; replay < 1e-9 |
| MAIN-001 produced by `7f624ca` | Verified | `git_commit_match=true`, tag, runbook driver |
| ABLATION-001 produced by `4533c61` | Verified | `git_commit_match=true`, tag, shard manifests |
| The archived numbers were generated by exactly this code | Strongly supported | exact replay and schema; bytecode and environment not byte-attested |
| No alternative or hidden implementation exists | Strongly supported | whole-tree search, single repository, no dynamic imports (within what the filesystem retains) |
| Exact executed bytecode; numpy 2.5.1 bit identity | Unknown | not captured at run time |

Alternatives considered and rejected: `4533c61` produced everything (MAIN-001 configuration and tag point to
`7f624ca`); the older `campaign.py`/`campaign_v2.py` produced the results (their output names are absent and they
cannot emit continuous penalties); a shadowed `dtcoa_tunable.py` was used (no second copy, no path manipulation, and
the frozen blob reproduces the numbers); the frozen commits differ in the policy (identical blobs); post-freeze
working-tree edits (tracked files clean).

## Verdict

**Verified for the policy implementation**, with two qualifications. (1) "The frozen implementation" is two commits:
MAIN-001 = `7f624ca`, ABLATION-001 = `4533c61`; the policy is identical in both, so the distinction does not affect
behaviour but must not be misstated. (2) That the archived numbers came from exactly this code is strongly supported
rather than attested: no run-time bytecode hash or bitwise output was archived and `SHA256SUMS` does not cover code;
the gap is closed by exact replay, not cryptographically.
