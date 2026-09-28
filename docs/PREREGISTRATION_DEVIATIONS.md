# Protocol commitments, deviations, and revision additions

**Date of this record:** 2026-09-29.
**Scope:** the evaluation protocol that governed MAIN-001 (core cost-ratio comparison) and
ABLATION-001 (mechanism attribution), what the executed study actually did, and what the
2026 revision restores or adds. The revision's own analysis plan is
[`REVISION_ANALYSIS_PLAN.md`](REVISION_ANALYSIS_PLAN.md).

This record separates three things that must not be confused:

- **(a) Original commitments** — what the internal evaluation protocol specified before the
  experiments were run.
- **(b) Deviations already made** — where the executed study (MAIN-001, ABLATION-001) differs
  from (a), with the dated evidence for when and why.
- **(c) Revision additions** — analyses restored or added in the 2026 revision. **All of (c)
  were specified after the MAIN-001 and ABLATION-001 results were known.** They are *not*
  pre-registered; they are pre-specified in a dated plan before their own execution.

---

## 1. Status of the original protocol document

| Property | Fact |
|---|---|
| Document | `PRE_REGISTERED_PROTOCOL.md` ("Pre-Registered Evaluation Protocol", status line "FINAL") |
| SHA-256 | `7d87b08f1e15bdcea9997918d75658718ce3316708f45507816ee19e3af23712` |
| Registration | **None.** The document was an internal protocol. It was never deposited in a registry (e.g. OSF, AsPredicted) and carries no external timestamp. |
| Earliest independent evidence of existence | Referenced by name in the source code of commit `7f624ca` (git-recorded date 2026-07-04 21:28 UTC). Git dates are author-recorded; the first *external* timestamp for that commit is its public release on 2026-07-24. |
| Earlier draft | An earlier internal draft exists (`PREREGISTRATION_PROTOCOL.md`, SHA-256 `22a20b4bd8132cd2409ed7b19c9f27361af67dbb8961688182bc7ceecaff57f5`), superseded by the document above. |
| Terminology | Where the frozen source code says "pre-registered", read **"internally pre-specified"**. The published manuscript does not describe the study as pre-registered. |

The SHA-256 values let a reader verify any copy of these documents that is later released.
Whether the protocol document itself is published is recorded in the release notes of the
repository version that contains this file.

## 2. Referenced documents that were not retained

The frozen source code refers to three items that are **not in this repository and were not
retained anywhere**. They have not been reconstructed, and they will not be.

| Reference in frozen code | Where | Status |
|---|---|---|
| `PROJECT_SCOPE_FREEZE.md` | comments in `nested_hpo.py`, `run_pipeline.py`, `search_space.py`, `environments.py`, `ablation_study.py`, `distribution_shift.py`, `dp_oracle.py` | Not retained. The scope decisions it recorded are documented, with dates, in the public [`CHANGELOG.md`](CHANGELOG.md) (entries of 2026-07-04) and in [`EXPERIMENT_MANIFEST_FINAL.md`](EXPERIMENT_MANIFEST_FINAL.md) (frozen 2026-07-04), both already in this repository. |
| `experiments/baseline_tuning` | module docstring of `baselines/baselines.py` ("Thresholds tuned on DISJOINT seeds") | Not retained. The docstring describes an earlier development phase. In MAIN-001 and ABLATION-001 **both baselines were tuned by the same nested cross-validation as ROVA** (`nested_hpo.py`), not by a separate disjoint-seed procedure. The docstring is stale. |
| `PRE_REGISTERED_PROTOCOL.md` | several module docstrings | Retained internally; see §1. |

Known inaccuracies in frozen code comments (the code itself is correct and unchanged):

- `nested_hpo.py` module docstring says "K_in=5 inner folds"; the constant it uses, and the
  value used in every experiment, is `DEFAULT_K_IN = 3`. The same file already stated at the
  frozen commit: "Frozen PI-scoped budget … K_out=5, K_in=3, n_trials=80. Full-spec values
  (K_in=5, n_trials=200) are future work."
- `run_pipeline.py` module docstring lists OracleLite and a 7-point grid; the executed
  pipeline ran Static, EWMA and ROVA on the 5-point grid (see §3, items A2 and A7).

The frozen commits are immutable and are not edited; these corrections are recorded here.

## 3. Commitment-by-commitment record

Sources: **P** = the protocol document (§ numbers refer to it); **C** = public
`CHANGELOG.md`; **M** = public `EXPERIMENT_MANIFEST_FINAL.md`; **G** = frozen commit
`7f624ca` source code. "Revision" refers to experiment IDs in `REVISION_ANALYSIS_PLAN.md`.

### A. Design and tuning

| # | (a) Original commitment | (b) Executed study | (c) Revision |
|---|---|---|---|
| A1 | Budget: K_out=5, K_in=5, B=200 random-search trials (P §3.4, §4) | K_out=5, **K_in=3, 80 trials** (G, dated 2026-07-04, before MAIN-001 ran on 2026-07-08; reductions *below* this budget were tested in PILOT-001 and rejected, C 2026-07-04) | Not restored (the frozen budget is kept so that all revision comparisons use the same protocol as the published results). Disclosed in the manuscript. |
| A2 | Cost-ratio grid "pre-registered" | 7 points reduced to **5 points** {0.5, 1.5, 3.5, 8.0, 20.0} (C, M, 2026-07-04, "round 5", before MAIN-001) | Kept. |
| A3 | Three structurally diverse environments incl. one unfavorable (P §5) | Done | Kept. |
| A4 | Unit of cross-validation and resampling = scenario instance (P §4, §10) | Done at the level of seed indices, **but each method's evaluation worlds were seeded from a method-specific tag** (`run_trial` seeds each world with `trial_seed(seed, 0, tag)` and the tag contains the method name). For the same seed index, different methods were therefore evaluated on *different* independently drawn worlds. Comparisons described as paired pair methods by seed index and fold, not by a shared world. | **Disclosed. New matched-world re-evaluation** in which every method is evaluated on common evaluation worlds (Revision R2). |
| A5 | Random search primary; **Bayesian-optimization robustness re-run** at the same budget (P §3.4, §12 step 5) | Not run (M exclusion list; C 2026-07-04 round 1) | **Restored** as a Tree-structured Parzen Estimator re-run (Revision R13). |
| A6 | Tuning curves reported at sub-budgets (P §3.6) | Recorded at 20/40/80 trials in every checkpoint; **not reported** | **Restored** from archived data (Revision R2). |
| A7 | No early stopping differing by method (P §3.5) | Complied (none used) | — |

### B. Baselines and reference policies (P §6)

| # | (a) Original commitment | (b) Executed study | (c) Revision |
|---|---|---|---|
| B1 | Memoryless threshold, smoothed (EWMA) threshold — mandatory | Done | Kept. |
| B2 | **Trivial reference: always-permit and always-restrict — mandatory** | **Not implemented, and not listed in any exclusion list.** This omission was silent. | **Restored** (Revision R1). |
| B3 | Myopic oracle | Done | Kept. |
| B4 | Sequential (dynamic-programming) oracle as the authoritative regret reference, validated against the myopic oracle in every environment (P §6, §11, §12 step 2) | Validated only in the single-agent uncoupled environment, where the two are provably identical. Multi-agent DP listed as future work (M). | **Restored in part**: exact DP on the full-size collapse environment and on reduced instances (N = 2–6) of both coupled environments; the favorable environment at its full size (N = 10) is beyond exact solution (Revision R7). |
| B5 | Clairvoyant cost-sensitive threshold, tuned per cost ratio (P §6) | Implemented (`OracleLite` + `CLAIRVOYANT_SEARCH_SPACE`, present at G) but removed from the pipeline (C 2026-07-04, round 3) | **Restored exactly as designed** (Revision R10). |
| B6 | **Learned baseline trained by direct optimization of the same loss — required** (P §6) | Not run ("RL baseline … moved to future work", C 2026-07-04, round 3; M) | **Restored**: cost-sensitive learned policy (Revision R3) and tabular Q-learning (Revision R4). |
| B7 | Richer learned policies — optional (P §6) | Not run | Not added (optional in the protocol). |

### C. Ablations (P §7)

| # | (a) Original commitment | (b) Executed study | (c) Revision |
|---|---|---|---|
| C1 | Full 2⁴ factorial ablation, each re-tuned | One-at-a-time re-tuned ablation (ABLATION-001); factorial listed as future work (M) | **Not restored.** Reason: see `REVISION_ANALYSIS_PLAN.md` §9. |
| C2 | Threshold/band-vector ablation (single global threshold) | Not run | **Restored** (Revision R12). |
| C3 | Continuous penalty-magnitude sweep for the containment question | Not run; the containment check reads the tuned penalty magnitudes instead | **Restored** as a conditional sweep around each tuned configuration (Revision R11). |

### D. Statistics and decision rules (P §8, §9)

| # | (a) Original commitment | (b) Executed study | (c) Revision |
|---|---|---|---|
| D1 | Paired Wilcoxon signed-rank + Cliff's δ with negligibility band 0.147 | Done (pairing by seed index; see A4) | Matched-world re-analysis (R2); independent-samples Mann–Whitney sensitivity (R2). |
| D2 | Friedman + Holm post-hoc; critical-difference diagrams | Friedman computed (secondary); CD diagrams not produced | CD diagrams per cost-ratio regime (R2). |
| D3 | BCa bootstrap CIs for every reported number and gap | BCa CIs for mean per-seed regret computed and archived; not reported; no CIs for gaps | **Restored** (R2). |
| D4 | TOST equivalence whenever equivalence is claimed; margin "pre-registered" | No equivalence claims made — rule not triggered. **No numeric margin was ever specified.** | No equivalence claims will be made in the revision (R-plan §6). |
| D5 | Secondary Bayesian ROPE layer | Not run. **No ROPE width was ever specified.** | **Not restored**: choosing a ROPE width now, after the results are known, would introduce an unconstrained analytic choice. |
| D6 | Holm correction across the entire family of reported analyses | Holm across the 3 method pairs within each cell | Family-wide sensitivity analyses (R2). |
| D7 | CVaR-95 co-primary | Done (fold-level) | Kept; pooled-seed CVaR added descriptively. |
| D8 | DR1 superiority rule (2 of 3 environments, CI separation, no reversal) | Per-cell symmetric verdict rule, adopted after an internal audit found the first (asymmetric) implementation over-declared losses; disclosed in the manuscript | Kept; archived v1 and corrected v2 tables both published (`verdicts/`). |
| D9 | DR2: report crossover location with CI when advantage is condition-dependent | Crossover explicitly not located (claim granularity decision, C 2026-07-04 round 5) | Kept. |
| D10 | DR7: compute-conditional claims if inference compute > 3× | Not assessed | **Restored**: measured per-decision cost (R6). |

### E. Robustness and reporting

| # | (a) Original commitment | (b) Executed study | (c) Revision |
|---|---|---|---|
| E1 | Distribution-shift stress test (P §5) | Done (archived; manuscript Appendix D.6) | Kept. |
| E2 | Variance decomposition over seeds/splits (P §5) | Not reported | **Restored** from archived data (R2). |
| E3 | RQ4: per-decision inference cost and total tuning cost | Wall-clock of the whole campaign reported only | **Restored** (R6; tuning cost in trial-evaluations per method). |
| E4 | Primary deliverable: full cost-ratio curve with CIs per environment (P §12 step 11) | Curves shown as effect sizes; regret CIs not shown | **Restored** (R2 figure). |
| E5 | Changes after data collection require a new dated pre-registration (P §11) | Scope changes of 2026-07-04 were made after PILOT-001 data and before MAIN-001 data, and were recorded as dated changelog and manifest entries rather than as a new protocol document | This record, and the dated revision plan, are that documentation. |

## 4. What this record does not claim

It does not claim that the executed study followed the protocol where it did not, and it
does not describe any revision analysis as pre-registered. Deviations A1, A2 and the
round-3 scoping were decided before MAIN-001 data existed; everything under (c) was decided
after MAIN-001 and ABLATION-001 results were known.
