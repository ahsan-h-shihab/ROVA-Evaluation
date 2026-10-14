# Adversarial self-review of the reproduction package

An adversarial review made during the July 2026 verification of the reproduction package; it modified no source,
manuscript or experiment artifact. Technical content unchanged; wording condensed on 2026-10-15, and references to
an unpublished pre-submission draft are described rather than quoted. Original text: tag `v1.0`
(`docs/HOSTILE_SELF_REVIEW.md`, git blob `c79d088fb113ee1d07b12fc5fe49c055ab54b478`).

**Outcome.** The mechanical replay withstood every attack. One overreach was found — not in the package but in an
earlier audit's labelling of draft statements — and is corrected below.

## Correction to the earlier audit

An earlier audit labelled four statements of a pre-submission draft about authorisation monotonicity (three in the
text, one in a figure) as falsified. That is too strong:

- Each statement has a narrow, map-theoretic reading — "holding the other inputs fixed, authorization is
  nondecreasing in the trust argument" — under which it is true; this is the narrow property the package verifies
  (0 violations).
- The observed "trust down, authorisation up" events contradict the statements only under an operational, temporal
  reading ("over time, a drop in trust can only tighten permissions").
- The statement that attributed the guarantee to monotonicity names the map itself and is the most
  interpretation-dependent; the two phrased in operational terms are on firmer ground.

The accurate characterisation: the statements are false under the operational reading their context invites
(failure leading to escalation and human control) and true under the narrow ceteris-paribus reading; the draft used
map-monotonicity to guarantee an operational property that does not follow from it. The correction is in
`reproduce/expected_outputs.md` §D.3. The package itself never made the falsification claim — it lists such verdicts
under "Claims deliberately NOT in this table" and reports only the mechanical fact — so the correction applies to the
earlier audit, not to the package.

## Attacks that did not break the package

| Attack | Result |
|---|---|
| Penalty reconstruction is wrong | The self-check `clip(round(L−ppen−spen))==A` holds at all ~1.9M agent-steps, 0 mismatches. The `Recorder` only reads policy outputs and never reimplements `step`. |
| Violations are a reconstruction artifact | An event with ~zero penalties at both steps would contradict band monotonicity; 0 of 11,752 events are of that kind — every violation comes from penalty release. |
| Counts are fragile to floating point or BLAS (round-half flips) | In a busy cell (90,000 steps) the closest `a_raw` came to a rounding boundary was 0.038, 13 orders of magnitude above floating-point noise. |
| `seed0=0` was assumed | Shown twice: `run_pipeline.py:105` builds `hpo_kwargs` without `seed0`, so `nested_cv_hpo` uses its default 0; and replayed regret matches archived `test_mean_regret` to < 1e-9 on all 450 folds, which a wrong split could not do. |
| Replay perturbs the scored world | The wrapper returns the policy's exact `A, ESC`; the < 1e-9 match on 450/450 folds shows the worlds are identical. |
| Double counting across folds or seeds | Outer test folds are a verified disjoint partition of seeds 0–59 (sizes 12 × 5); each agent-timestep is counted once. |
| Off-by-one between trust and authorisation | `trust_after` is the post-update `policy.T`, the value the band map consumes in the same step. |

## Additional bounds (now in the package)

- Regime localisation: the deployed `rova` policy shows violations only at low cost ratios — none at c = 8.0 and
  c = 20.0, where ROVA was claimed effective. The total of 11,752 is inflated by re-tuned ablation branches; the
  deployed policy accounts for 2,366, about 45 % of them in one cell (favorable, c = 1.5).
- In the `no_memory` ablation the "trust" is raw evidence (`use_trust=False`, so `T=E`); its violations mean
  "evidence down, authorisation up" — mechanically valid but not a calibrated trust state.

## Verdict

The package's execution claims hold and are better supported after the review (four new internal-consistency checks
pass). Its rule of reporting only the mechanical "trust ↓ / authorisation ↑" fact, and not labelling draft
statements, is what protects it from the overreach above. The earlier audit's "falsified" becomes "false only under
the operational reading; true under the narrow map reading", and the 11,752 total is always to be reported with the
deployed-policy figure of 2,366.

### Reproducible checks added by this review

- Zero-penalty-violation impossibility scan → 0 of 11,752 (from `reproduce/out_archived.json`).
- Round-half boundary probe → minimum gap 0.038 over 90,000 agent-steps.
- Outer-fold partition check → sizes [12, 12, 12, 12, 12], union = seeds 0–59, no overlap.
- Per-cell deployed-`rova` violations → zero at c ∈ {8.0, 20.0} in all three environments.
