# Hostile Self-Review of the Reproduction Package

> Preserved verbatim from the adversarial self-review conducted during the
> verification effort. Technical conclusions are unedited. Read-only; no source,
> manuscript, or experiment artifact was modified.

The package was attacked on every axis. **The mechanical replay survives; one genuine overreach was found — not in the package, but in the earlier audit's manuscript-falsification labels, which the package already scopes out. It is reported below as a correction.**

## Mistakes found (reported immediately)

**M1 — Interpretation overreach in the *previous audit's* "FALSIFIED" verdicts (not in the package).**
The earlier report flatly labeled four manuscript sentences **FALSIFIED** (§3, §4.8, §4.9-clause-2, Fig 2). Under hostile reading, that is stronger than warranted:

- Every one of those sentences has a defensible **narrow map-theoretic reading** — "holding the other inputs fixed, authorization is nondecreasing in the trust argument" — under which it is **TRUE**, and which equals the narrow property separately verified (0 violations).
- The "trust down + auth up" evidence only contradicts them under the **operational/temporal reading** ("over time, a drop in trust can only tighten permissions").
- §4.8 in particular ("*monotonicity guarantees that a loss of trust can only restrict…*") is the weakest falsification: "monotonicity" there explicitly names the *map*, so a charitable reader treats the sentence as a restatement of map-monotonicity, which holds.

The honest characterization is therefore **not** "these sentences are false," but: *"they are false under the operational/safety reading the surrounding text invites (‘failure mode is escalation to oversight’, ‘hand control back to a human’), and true under a narrow ceteris-paribus reading; the manuscript conflates the two by using map-monotonicity to ‘guarantee’ an operational property that does not follow from it."* §3 ("*a drop in trust… never expand*") and Fig 2 ("*as evidence worsens*") lean operational and are on firmer ground; §4.8 is the most interpretation-dependent. This correction is written into `reproduce/expected_outputs.md` §D.3.

Crucially, **the reproduction package itself does not make the falsification claim** — it lists those verdicts under "Claims deliberately NOT in this table" and reports only the mechanical fact. So M1 corrects the prior *audit*, not the deliverable.

## Attacks that FAILED to break the package (why it survives)

| Attack | Result |
|---|---|
| **Penalty reconstruction wrong** | Self-check `clip(round(L−ppen−spen))==A` holds at **all ~1.9M agent-steps, 0 mismatches**. The `Recorder` only *reads* policy outputs — it never reimplements `step`, so it cannot diverge. |
| **Violations are a reconstruction artifact** | If any "trust down, auth up" event had ~zero penalties at both steps it would contradict band monotonicity and expose a bug. **0 of 11,752** such events exist — every violation is genuinely penalty-release-driven. |
| **Counts are float/BLAS-fragile (round-half flips)** | In a busy cell (90,000 steps) the closest `a_raw` ever came to a rounding boundary was **0.038** — 13 orders of magnitude above fp noise. Counts are portable, not knife-edge. |
| **`seed0=0` was assumed** | Proven twice: (a) `run_pipeline.py:105` builds `hpo_kwargs` without `seed0`, so `nested_cv_hpo` uses its default 0; (b) replayed regret matches archived `test_mean_regret` to `<1e-9` on all 450 folds — a wrong split could not match. |
| **Replay perturbs the scored world** | The wrapper returns the underlying policy's exact `A, ESC`; the `<1e-9` regret match on 450/450 folds proves the worlds are identical. |
| **Double-counting across folds/seeds** | Outer-test folds are a verified **disjoint partition** of seeds 0–59 (sizes 12×5, no overlap); each (agent, timestep) counted once. |
| **Off-by-one in trust/authorization pairing** | `trust_after` = post-update `policy.T`, which is exactly the value the band map consumes in that same step; authorization is paired with the trust that produced it. |

## Additional honest bounds surfaced (now in the package)

- **Regime localization:** the *deployed* `rova` policy violates only at low cost ratios — **zero at c=8.0 and c=20.0**, the regime where ROVA is claimed effective. The combined 11,752 is inflated by re-tuned **ablation** branches; the clean deployed number is **2,366**, ~45% of it from one cell (favorable, c=1.5). Without this, the aggregate over-impresses.
- **`no_memory` ablation:** its "trust" is raw evidence (`use_trust=False` ⇒ `T=E`), so its violations mean "evidence down, auth up" — legitimate mechanically but not a calibrated trust state.

## Verdict

The package's **execution claims hold** and are, if anything, better supported after the attack (four new internal-consistency checks pass). The package's central discipline — reporting only the mechanical "trust ↓ / authorization ↑" fact and explicitly refusing to label manuscript sentences — is what makes it survive M1: the interpretation risk lives entirely in the prose verdicts, which the package excludes. The one correction is to **downgrade the earlier audit's flat "FALSIFIED" to "false only under the operational reading; true under the narrow map reading,"** and to always pair the 11,752 figure with the regime-localized deployed-policy figure of 2,366.

### Reproducible checks introduced by this review

- Zero-penalty-violation impossibility scan → 0 of 11,752 (regenerated from `reproduce/out_archived.json`).
- Round-half boundary gap probe → min gap 0.038 over 90,000 agent-steps.
- Outer-fold disjoint-partition check → sizes [12,12,12,12,12], union = seeds 0–59, no overlap.
- Per-cell deployed-`rova` violation breakdown → zero at c∈{8.0, 20.0} in all three environments.
