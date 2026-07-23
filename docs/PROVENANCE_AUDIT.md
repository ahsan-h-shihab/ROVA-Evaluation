# Read-Only Provenance Audit — ROVA_CCE Frozen Implementation

> Preserved verbatim from the provenance audit produced during the verification
> effort. Read-only; no source, manuscript, or experiment artifact was modified
> to produce it. All evidence is git/repository/execution evidence.

**Framing correction up front (do not skip):** There are **two** frozen commits, not one. Treating `4533c61` as "the implementation that produced MAIN-001 and ABLATION-001" is **incorrect**: `4533c61` produced ABLATION-001 only; MAIN-001 was produced by a **different** commit, `7f624ca`. The audit below establishes what *can* be proven about each.

## 1. Git provenance

- **Two annotated tags**, resolving to the exact full hashes recorded in the artifacts:
  - `main-001-frozen → 7f624ca27a86fbbbe67afc68ebb886a4edbb67a2`
  - `ablation-001-frozen → 4533c61aa2e252698fd0c31d5bc1f9b9d04c18f0`
- **Linear history:** `7f624ca` (master) → 5 ablation commits → `4533c61` (HEAD, branch `ablation-001`). `git branch`: only `master`, `ablation-001`. No stashes, one worktree, `git fsck` shows **no dangling commits**.
- **Recorded hashes inside artifacts:**
  - `MAIN-001/config.json`: `git_commit = git_commit_expected = 7f624ca…`, `git_commit_match: true`.
  - `ABLATION-001/config.json`: `git_commit = 4533c61…`, `match: true`, `git_branch: ablation-001`, `git_tag: ablation-001-frozen`, `git_anchor: branched from main-001-frozen (7f624ca)`.
- **Policy-file history:** `git log -- dtcoa_val/dtcoa.py dtcoa_val/dtcoa_tunable.py` returns **only `7f624ca`** — the policy files were introduced there and **never modified afterward**.
- **Blob identity (decisive):**
  - `dtcoa_tunable.py` = `593bbc67…` at `7f624ca`, `4533c61`, and `HEAD` — **identical**.
  - `dtcoa.py` = `bb68c0b8…` at all three — **identical**.
  - `git diff 7f624ca 4533c61`: `dtcoa.py`, `dtcoa_tunable.py`, `runner.py`, `sim_core/core.py`, `environments.py`, `nested_hpo.py` all **0 diff lines**. Only `search_space.py` (30 lines) plus new ablation driver/analysis/test files differ. The `search_space.py` diff touches **only `ROVA_ABLATIONS`** (FIX-1 e_min); `ROVA_SEARCH_SPACE` (MAIN-001's space) is unchanged.
- Working tree is clean for all tracked files (only untracked `__pycache__/`).

## 2. Execution provenance

- **MAIN-001 driver:** `MAIN-001_EXECUTION_RUNBOOK.md:63,68` — `python3 -m dtcoa_val.experiments.run_pipeline --budget full`. `REPORT.md` references `run_pipeline._mark_done`. `run_pipeline.py:38` imports `DTCOATunable`.
- **ABLATION-001 driver:** `run_ablation.py` (`config.json:127` names `validate_consistency, dr3_attribution, dr3_regime_inference` "committed modules at git 4533c61"; `raw/manifest.shard*.json` are `run_ablation`'s shard manifests). `run_ablation.py:55` and `ablation_study.py:10` import `DTCOATunable`.
- **Artifact schema = tunable signature:** rova `best_config` keys = `{alpha, bands, context_risk, e_min, gamma, k, prop_scale, s_min, stab_scale}` — exactly `ROVA_SEARCH_SPACE`, including **continuous `prop_scale`/`stab_scale`** that only `DTCOATunable` consumes (the frozen `DTCOA` uses integer `ceil` penalties and no such args).
- **Determinism recorded:** `ABLATION-001/config.json:128` — "nested_cv_hpo deterministic given (seed0=0, run tag); same-tag reruns reproduce checkpoints bit-for-bit." Matches source (`run_pipeline.py:105` builds `hpo_kwargs` without `seed0` → default 0).
- **Runtime env captured:** both `environment.txt` = Python 3.13.14, numpy 2.5.1; ABLATION interpreter `/…/miniconda3/envs/rova/bin/python3`.

## 3. Import provenance

- All policy imports in the package: `run_pipeline.py`, `run_ablation.py`, `ablation_study.py`, `benchmark_ablation.py` import **only `from dtcoa_val.dtcoa_tunable import DTCOATunable`**; `campaign.py`/`campaign_v2.py` import `DTCOA` but are **not the drivers** (their outputs are named `V1/A1/B1/V5/V7.json` — a pattern that appears **nowhere** in the archives).
- **No dynamic imports / path manipulation:** grep for `importlib|__import__|exec(|eval(|sys.path|PYTHONPATH|SourceFileLoader` across the package → **none**.
- **No module shadowing:** `find / -name dtcoa_tunable.py` → only the one canonical file.

## 4. Artifact provenance

- **Replay consistency (execution-level tie):** replaying the archived tuned configs + seeds through the working-tree code (blob `593bbc6`, identical to both frozen commits) reproduces the archived `test_mean_regret` to **< 1e-9 on all 450 folds** (75 MAIN-001 + 375 ABLATION-001), with 0 penalty-reconstruction mismatches. This links archived numbers → frozen policy blob.
- **Transfer snapshot corroboration:** `ROVA_CCE_transfer_20260708.tar.gz` (integrity `sha256 -c` **OK**) contains `dtcoa_tunable.py` = `593bbc6…` and `dtcoa.py` = `bb68c0b…` (byte-identical to frozen blobs) **and embeds the `code/.git` history**.
- **Integrity gap (stated honestly):** `ABLATION-001/SHA256SUMS` checksums results/REPORT/config but **0 lines cover `.py`/code**. Code integrity therefore rests on git blob identity + `git_commit_match` + the transfer-tarball snapshot + replay, **not** on a code checksum manifest.

## 5. Alternative-implementation search

- Only two DTCOA definitions exist under the entire tree: `dtcoa.py`, `dtcoa_tunable.py` (plus the `reproduce/` scripts, which *import* them). No notebooks, `.pkl/.pickle/.joblib/.bak`, or serialized policies anywhere.
- `main001_out/`, `abl001_out/` contain only `analysis.py`/`rebuild_and_validate.py` (analysis, not policy).
- **Exactly one `.git`** under all of `Projects/`; no rival repository; no dangling/unreachable commit carrying a different policy.

## 6. Remaining uncertainty

- **UNKNOWN (unprovable from repo):** the *exact bytecode* executed on 2026-07-08/09 (no run captured `.pyc` hashes; `__pycache__` is regenerable). Mitigated — not eliminated — by exact numeric replay.
- **UNKNOWN:** that the runtime numpy (2.5.1) produced the archived numbers *bit-for-bit*; the replay used numpy 2.4.2 and matched to 1e-9, not to the last bit. Provenance of *code* is unaffected (deterministic PCG64 + identical source).
- **NOT a gap, but a correction:** the premise "4533c61 produced MAIN-001" is disproven — MAIN-001 = `7f624ca`.

---

## A. Evidence chain

```
Artifact (MAIN-001)      config.git_commit=7f624ca, match=true ─┐
Artifact (ABLATION-001)  config.git_commit=4533c61, match=true ─┤
                                                                 ├─ recorded commit hashes
git tags  main-001-frozen→7f624ca,  ablation-001-frozen→4533c61 ─┘   resolve exactly

Policy code   dtcoa_tunable.py blob 593bbc6  ≡ at 7f624ca ≡ 4533c61 ≡ HEAD
              dtcoa.py        blob bb68c0b   ≡ at all three
              (git log: policy files touched ONLY by 7f624ca; 0-line diff between frozen commits)
                    │
Drivers       run_pipeline.py / run_ablation.py  ──import──▶ ONLY DTCOATunable  (no dynamic/alt imports, no shadow module)
                    │
Artifact schema  rova best_config = ROVA_SEARCH_SPACE incl. continuous prop_scale/stab_scale (tunable-only)
                    │
Replay        frozen blob 593bbc6 → reproduces archived regret <1e-9 on 450/450 folds, 0 recon mismatches
                    │
Snapshot      transfer tarball (sha256 OK) carries 593bbc6 + bb68c0b + embedded .git
```

## B. Provenance confidence assessment

| Claim | Level | Basis |
|---|---|---|
| The frozen **policy implementation** (`dtcoa_tunable.py`=593bbc6, `dtcoa.py`=bb68c0b) is the canonical code behind **both** experiments | **VERIFIED** | Identical blob across both producing commits; only ablation search-space/driver/tests differ; drivers import only this class; artifact schema is tunable-only; replay matches to <1e-9 |
| MAIN-001 produced by `7f624ca` | **VERIFIED** | `config.git_commit_match=true`, tag resolves, runbook driver |
| ABLATION-001 produced by `4533c61` | **VERIFIED** | `config.git_commit_match=true`, tag, shard manifests |
| Archived numbers were generated by exactly this policy code | **STRONGLY SUPPORTED** | Exact numeric replay (<1e-9) + schema signature; falls short of VERIFIED only because runtime bytecode/env is not byte-attested |
| No alternative/hidden DTCOA implementation exists | **STRONGLY SUPPORTED** | Whole-tree search, single `.git`, no dynamic imports/shadowing (bounded by what the filesystem retains) |
| Exact executed bytecode / numpy-2.5.1 bit-identity | **UNKNOWN** | No `.pyc` hash or bitwise output captured at run time |

## C. Alternative explanations considered

1. **`4533c61` produced everything** — *rejected*: MAIN-001 config/tag point to `7f624ca`; `4533c61` post-dates and only adds ablation code.
2. **The older `campaign.py`/`campaign_v2.py` (which import the integer-penalty `DTCOA`) produced the results** — *rejected*: their output filenames (`V1/A1/B1.json`) are absent from the archives; archived checkpoints carry continuous `prop_scale`/`stab_scale` that `DTCOA` cannot emit.
3. **A shadow/edited `dtcoa_tunable.py` on `PYTHONPATH` at run time** — *rejected*: no second copy on the filesystem, no `sys.path`/`PYTHONPATH`/`importlib` manipulation in the package, and the frozen blob reproduces the numbers.
4. **The frozen commits differ in the policy** — *rejected*: byte-identical blobs; 0-line diff on all policy/simulator/runner/env files.
5. **Post-freeze working-tree edits** — *rejected*: `git status` clean for tracked files; working `dtcoa_tunable.py` = committed blob.

## D. Final verdict

**VERIFIED — for the policy implementation.** The frozen implementation is established as the canonical **policy code** behind both experiments: `dtcoa_tunable.py` (blob `593bbc6`) and `dtcoa.py` (blob `bb68c0b`) are byte-identical at MAIN-001's commit (`7f624ca`), ABLATION-001's commit (`4533c61`), and HEAD; the two frozen commits differ **only** in ablation search-space/driver/analysis/test files, never in the policy, simulator, runner, or environments; the drivers import only this class; the artifact schema is the tunable-class signature; and the frozen blob reproduces every archived regret to <1e-9.

**Two mandatory qualifications:**
1. **"The frozen implementation" is not a single commit.** MAIN-001 = `7f624ca`, ABLATION-001 = `4533c61`. The specific claim *"`4533c61` produced MAIN-001"* is **false**. The reason provenance still holds is that the policy blob is identical across both — so the distinction is immaterial to policy behavior but must not be misstated.
2. **That the archived numbers were generated by this exact code is STRONGLY SUPPORTED, not VERIFIED-by-attestation**, because no run-time bytecode hash or bitwise output was archived and `SHA256SUMS` does not checksum the code. The gap is closed empirically (exact replay), not cryptographically.
