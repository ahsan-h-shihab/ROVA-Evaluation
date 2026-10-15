# MAIN-001 Execution Runbook

Scope: operational only. Executes the frozen rev. 5 manifest (45-unit core
campaign; distribution shift deferred to the revision stage; all other
deferred items remain future work). No experimental redesign.
(Scope line reworded on 2026-10-16; original text in tag `v1.0`.)

## 0. Pre-flight checklist
```
cd <compute-host>/rova_project/code
python3 -c "from dtcoa_val.experiments.environments import COST_RATIO_GRID; assert COST_RATIO_GRID == [0.5, 1.5, 3.5, 8.0, 20.0]"
python3 -c "from dtcoa_val.experiments.run_pipeline import SHIFT_COST_RATIO; assert SHIFT_COST_RATIO == 3.5"
python3 -c "import numpy, scipy"
df -h .
nproc
```
If any assertion fails: stop. Code doesn't match frozen rev. 5 manifest.

### 0.1 Code version identifier
Code is under version control at commit `7f624ca27a86fbbbe67afc68ebb886a4edbb67a2`
(frozen rev. 5 implementation). On the compute host:
```
git clone <path-to-rova_project/code> <compute-host-path>   # clone, not plain-copy
cd <compute-host-path> && git rev-parse HEAD   # must print 7f624ca27a86fbbbe67afc68ebb886a4edbb67a2
```
If the hash doesn't match: stop. Record this hash in config.json (section 5)
regardless.

### 0.2 Execution environment capture
On the actual compute host, immediately before launch:
```
python3 --version > $OUT/environment.txt
pip freeze >> $OUT/environment.txt
uname -a >> $OUT/environment.txt
```
Capture the host's real environment -- do not substitute a pre-recorded file.

### 0.3 Seed-set verification
```
python3 -c "
seeds = list(range(60))
assert len(seeds) == 60 and len(set(seeds)) == 60 and seeds == list(range(60))
import json; json.dump(seeds, open('$OUT/seeds_verified.json', 'w'))
print('Seed verification PASSED:', len(seeds), 'unique seeds, 0..59')
"
```
If this fails: stop. Do not launch with an unverified seed set.

## 1. Execution order
15 work units (3 environments x 5 cost ratios, fixed iteration order in
environments.py). Per unit: static -> ewma -> rova -> containment (free).
After all 15 units across all shards finish: DP-oracle validation
(unfavorable_oscillation_uncoupled only), then distribution-shift
(revision-stage, optional, not required for completion).

## 2. Parallel sharding strategy
N = min(15, available cores). Freeze N before launch; do not change across
resumes (shard assignment is unit_index % N).
```
N=<min(15, $(nproc))>
OUT=<absolute-path-to-results-dir>
for i in $(seq 0 $((N-1))); do
  mkdir -p $OUT/shard_logs
  nohup python3 -m dtcoa_val.experiments.run_pipeline \
      --budget full --out $OUT --shard-index $i --num-shards $N \
      > $OUT/shard_logs/shard_$i.log 2>&1 &
done
wait
python3 -m dtcoa_val.experiments.run_pipeline --budget full --out $OUT \
    --shard-index 0 --num-shards $N > $OUT/shard_logs/finalize.log 2>&1
```
Expected wall-clock: ~2h/unit x ceil(15/N) units/shard.

## 3. Checkpointing and resume
Checkpoints after every completed outer fold, per (method,env,cost-ratio)
tag, in $OUT/hpo_checkpoints/. $OUT/manifest.json tracks unit completion.
Resume: re-run the identical shard command; it auto-skips completed work
and resumes in-progress HPO from its last completed outer fold.

## 4. Expected directory structure
```
$OUT/
  manifest.json
  hpo_checkpoints/            (45 files: 15 units x 3 methods)
  <env>__c<ratio>__containment_report.json   (15 files)
  unfavorable_oscillation_uncoupled__dp_oracle_validation.json
  distribution_shift_report__rova.json       (optional, revision-stage)
  distribution_shift_report__ewma.json       (optional, revision-stage)
  shard_logs/
```

## 5. Artifact preservation
MAIN-001 = one experiment entry (like PILOT-001's 7 sub-runs). After final
validation (section 9) passes:
```
mkdir -p rova_project/experiments/MAIN-001_core_comparison/raw
cp -r $OUT/manifest.json $OUT/hpo_checkpoints $OUT/*.json $OUT/shard_logs \
      rova_project/experiments/MAIN-001_core_comparison/raw/
```
Then write config.json, seeds.json, analysis.py (reproduces all summary
stats from raw/ alone, per PILOT-001's pattern), REPORT.md. Register in
EXPERIMENT_INDEX.md.

config.json must include (reproducibility audit additions):
- git_commit: the hash verified in section 0.1 (7f624ca27a86fbbbe67afc68ebb886a4edbb67a2,
  or whatever hash was actually verified on the compute host if it differs --
  record the ACTUAL value used, flag any mismatch from the expected hash)
- environment_file: reference to the archived environment.txt from section 0.2
- seed_verification: reference to the archived seeds_verified.json from
  section 0.3, plus the verification result (PASSED)

Copy environment.txt and seeds_verified.json from $OUT into raw/ alongside
the other artifacts.

## 6. Logging requirements
Shard stdout/stderr to $OUT/shard_logs/shard_<i>.log. Lines are
self-describing ([run]/[skip]/[FAIL]/[wait]). Progress check:
```
python3 -c "import json; m=json.load(open('$OUT/manifest.json')); print(f'{len(m[\"completed\"])}/61 required units complete')"
```

## 7. Failure recovery
1. grep -l "\[FAIL\]" $OUT/shard_logs/*.log
2. Diagnose from traceback (resource/environment issue expected; a code
   bug found here is a genuine reproducibility problem, the one condition
   allowed to reopen scope).
3. Fix root cause.
4. Re-launch only the failed shard(s), identical flags -- auto-resumes.
5. Suspected corrupted checkpoint: delete only that tag's checkpoint file
   and its manifest entry, re-launch responsible shard. Never delete
   manifest.json wholesale.

## 8. Intermediate validation checkpoints
~25%: spot-check 2-3 checkpoint files (5 folds present, no NaNs).
~50%: run analysis.py in partial mode, confirm directionally sane values.
~100% of core+containment: manifest has exactly 60 entries, zero [FAIL].

## 9. Final validation before declaring MAIN-001 complete
1. manifest.json: all 61 required entries, zero gaps.
2. All 45 checkpoint files: exactly K_out=5 folds, non-null regret/CVaR.
3. Zero [FAIL] across all shard logs.
4. DP-oracle validation: certified_equivalent = true.
5. Full statistics pass computed and saved (BCa, Wilcoxon+Cliff's delta+Holm
   per cost ratio/environment, CVaR summary, Friedman+Holm, containment
   summary).
6. Archival complete per section 5.
7. EXPERIMENT_INDEX.md updated.
8. Reproducibility check: analysis.py run fresh from archived artifacts
   alone reproduces the same numbers (same check as PILOT-001).
9. Code version verified: archived git_commit in config.json matches the
   hash checked at launch (section 0.1); any mismatch investigated and
   resolved before declaring completion.
10. Environment captured: environment.txt present and non-empty in raw/.
11. Seed set verified: seeds_verified.json present, confirms exactly 60
    unique seeds (0..59), matching the frozen protocol.

## 10. Exact completion criteria
- [ ] All 45 method-HPO units valid in manifest.json and hpo_checkpoints/
- [ ] All 15 containment reports present
- [ ] DP-oracle validation present, certified_equivalent: true
- [ ] Zero unresolved failures across all shard logs
- [ ] Full statistics pass computed and saved
- [ ] Archival complete in experiments/MAIN-001_core_comparison/
- [ ] EXPERIMENT_INDEX.md registered
- [ ] Reproducibility verified from archived artifacts alone, fresh process
- [ ] Code version (git commit hash) recorded and verified against section 0.1
- [ ] Execution environment (environment.txt) archived
- [ ] Seed set verified (60 unique, 0..59) and seeds_verified.json archived

Distribution-shift output, if produced, is NOT part of these criteria --
remains revision-stage; presence or absence does not affect completion.
