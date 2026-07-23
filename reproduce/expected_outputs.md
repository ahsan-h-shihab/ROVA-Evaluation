# Expected Outputs & Claim-by-Claim Reproduction Table

Generated with **numpy 2.4.2**, thread-pinned (`*_NUM_THREADS=1`).
All numbers below are **produced by the scripts**, not hardcoded. Counts are the
stable verification targets; float low-order digits and file hashes may differ
across numpy/BLAS builds (regret self-validation tolerates this at `1e-9`).

Reference artifact hashes on the generating machine (sha256):

```
out_monotonicity.json  f40c019542deed3f9620d7c0dae44ac0a15b8059b2f872bb4120ad179f3f9904
out_archived.json      (environment-dependent; ~9.3 MB, contains full per-event locals)
```

`out_archived.json` embeds floats, so its hash is environment-specific by
design; verify it by the **counts** below, not its hash.

---

## A. `replay_monotonicity.py` — expected

Minimal synthetic counterexamples (violating timesteps), penalty-reconstruction
self-check, and the narrow-property / boundedness / contraction regeneration:

| Item | frozen `DTCOA` | tuned `DTCOATunable` |
|---|---|---|
| A1 propagation-release violations | **1** (t=3: trust 0.67372→0.66960, auth 2→3; prop_pen 1.0→0.0) | **1** (identical trust/auth) |
| A2 oversight-removal violations | **2** (t=7 and t=11: trust down, ESC True→False, auth 0→1) | **2** (identical) |
| A3 stability-release violations | **1** (t=6: evidence 0.429→0.359, trust 0.51351→0.46716, auth 1→2; stab_pen 1.0→0.0) | **0** (see note) |
| penalty-reconstruction mismatches | **0** | **0** |

- **A3 note (documented, not an error):** the exact A3 sequence crosses the
  binary stability threshold on frozen `DTCOA` but not the continuous rounding
  threshold on the default `DTCOATunable`. Stability-driven violations in the
  *tuned* class are demonstrated directly in the archived **single-agent**
  `unfavorable_oscillation_uncoupled` cells (Section B), where `n_agents=1`
  makes propagation impossible, so stability is the only possible driver.

Regenerated property numbers (implementation-independent):

- **Narrow / formal monotone conservatism:** `violations = 0` holding
  `(context_risk, prop_pen, stab_pen)` fixed → authorization **is** nondecreasing
  in the trust argument. (`holds = True`.)
- **Boundedness:** trust stays in `[0,1]` (observed range ≈ `[0.065, 0.79]`),
  authorization in `{0..5}` → `bounded = True`.
- **Trust contraction:** worst `|f(T)-f(T')|/|T-T'| = 0.7000`, equal to the bound
  `max(α,γ)=0.70`.
- **A3 independent re-derivation** (deterministic search, rng seed 0):
  `[0.593, 0.818, 0.45, 0.334, 0.563, 0.429, 0.359, 0.677]`.

---

## B. `replay_archived_runs.py --experiment both` — expected

**Self-validation:** `folds replayed: 450  reproducible: 450  NOT REPRODUCIBLE: 0`
(75 MAIN-001 rova + 375 ABLATION-001 across 5 branches; every fold's replayed
mean test-regret matches the archived `test_mean_regret` to `<1e-9`, and every
agent-step's reconstructed authorization equals the policy's own output).

**Archived-run monotonicity violations** (trust strictly down & authorization
strictly up), counted only over self-validated folds:

| Metric | Combined | MAIN-001 | ABLATION-001 |
|---|---|---|---|
| folds reproducible | 450/450 | 75/75 | 375/375 |
| violating agent-steps | **11 752** | **2 366** | **9 386** |
| of which oversight removed (A0 lifted) | **5 786** | 1 478 | 4 308 |
| violating (env, cost, branch, fold) configs | **59** | 10 | 49 |
| affected cells (env × cost) | **15 / 15** | 8 | 15 |
| affected environments | all 3 | 3 | 3 |
| affected cost ratios | {0.5, 1.5, 3.5, 8.0, 20.0} | {0.5, 1.5, 3.5} | all 5 |
| affected folds | {0,1,2,3,4} | {0..4} | {0..4} |
| affected seeds | all 60 | 12 distinct | all 60 |
| affected branches | full, no_decay, no_memory, no_propagation, no_stability, rova | rova | 5 ablation branches |

**Regenerated tuned-penalty magnitudes (MAIN-001 rova, 75 folds):**
`prop_scale` frac>0.1 = **0.27**, max = **4.675**, n>0.5 = 10;
`stab_scale` frac>0.1 = **0.28**, max = **4.672**, n>0.5 = 15; both>0.1 = **6**.

**Representative full violation record** (first printed; all locals present):

```json
{"experiment":"MAIN-001","environment":"collapse_moderate_coupled","cost_ratio":0.5,
 "branch":"rova","fold":3,"seed":8,"agent_id":0,"timestep":156,
 "trust_before":0.6921083630313263,"trust_after":0.6656378812851972,
 "authorization_before":0,"authorization_after":2,"evidence":0.26928256797343225,
 "propagation_penalty":3.0764372782684015,"propagation_penalty_before":6.152874556536803,
 "stability_penalty":0.00013532609337243734,"stability_penalty_before":0.00036002891454993267,
 "context_risk":0.16191427275266435,"esc_flag":false,"oversight_removed":true}
```

The propagation penalty releasing from 6.15 → 3.08 while trust falls is what
lifts authorization from 0 (human oversight) to 2.

---

## C. Reproduction table — previous audit's execution-derived claims

Each row is exactly one of: **VERIFIED**, **PARTIALLY VERIFIED**,
**NOT REPRODUCIBLE**, **REJECTED**. Evidence = the script + the field that
produces it.

| Previous audit claim | Reproduced? | Evidence |
|----------------------|-------------|----------|
| Narrow/formal monotone conservatism holds: authorization nondecreasing in trust with penalties fixed (0 violations) | **VERIFIED** | `replay_monotonicity.py` → `narrow_property.violations = 0` |
| Boundedness: trust ∈ [0,1], authorization ∈ {0..5} | **VERIFIED** | `replay_monotonicity.py` → `boundedness_contraction.bounded = True` |
| Trust contraction: worst Lipschitz ratio = 0.70 = max(α,γ) | **VERIFIED** | `replay_monotonicity.py` → `contraction_worst_ratio = 0.7000` |
| A1 counterexample: 3-agent propagation release → trust 0.67372→0.66960, auth 2→3 | **VERIFIED** | `replay_monotonicity.py` → A1 event (both implementations) |
| A2 counterexample: human oversight removed while trust falls (ESC True→False, auth 0→1) | **VERIFIED** | `replay_monotonicity.py` → A2, 2 events, `oversight_removed=True` (both implementations) |
| A3 counterexample: single agent, own evidence worsens 0.429→0.359, trust down, auth 1→2 | **PARTIALLY VERIFIED** | `replay_monotonicity.py` → A3 = 1 on frozen `DTCOA`; = 0 on default `DTCOATunable` for that exact sequence (continuous penalty). Tuned-class stability violations shown instead in §B single-agent cells |
| A3 sequence is not cherry-picked (independently re-derivable) | **VERIFIED** | `replay_monotonicity.py` → `A3_regenerated_sequence` (deterministic search, seed 0) |
| Penalties are evidence-driven and, when they release, drive the violation | **VERIFIED** | `replay_*` → every event shows `prop/stab_penalty_before > prop/stab_penalty` at the step authorization rises |
| ≈27–28% of 75 tuned MAIN rova folds have penalty scale > 0.1 (prop 0.27, stab 0.28, both>0.1 = 6) | **VERIFIED** | `replay_archived_runs.py` → `tuned_penalty_stats_MAIN_rova` |
| Prior report's stated limitation: *per-step authorization traces of the archived runs cannot be verified from the archived artifacts* | **REJECTED** | `replay_archived_runs.py` replays all 450 folds and finds **11 752** real violating agent-steps; the limitation no longer holds |
| The evaluated policies actually exhibit the non-monotonicity in archived MAIN-001 runs | **VERIFIED** | `replay_archived_runs.py` → MAIN-001: 2 366 agent-steps across 8 cells |
| … and in archived ABLATION-001 runs | **VERIFIED** | `replay_archived_runs.py` → ABLATION-001: 9 386 agent-steps across 15 cells, all 5 branches |
| Replay reproduces the archived regret (world reconstruction is exact) | **VERIFIED** | `replay_archived_runs.py` → 450/450 folds `regret_match=True` (<1e-9); 0 recon mismatches |

## D. Hostile-review caveats (bounds on what the numbers mean)

These do not change any count above; they bound its interpretation.

1. **Deployed-policy violations are confined to low cost ratios.** For the
   actually-deployed `rova` policy (MAIN-001), per-cell violating agent-steps are:

   | env | c=0.5 | c=1.5 | c=3.5 | c=8.0 | c=20.0 |
   |---|---|---|---|---|---|
   | favorable_hmm_coupled | 7 | 1067 | 434 | **0** | **0** |
   | unfavorable_oscillation_uncoupled | 0 | 36 | 1 | **0** | **0** |
   | collapse_moderate_coupled | 594 | 114 | 113 | **0** | **0** |

   At c=8.0 and c=20.0 — the high-cost regime where ROVA is claimed to win —
   the deployed policy shows **zero** violations, because tuning drives its
   penalty scales toward zero there (calibrated conservatism). The 2 366 total
   is real but regime-localized, and ~45% of it is a single cell
   (favorable, c=1.5). The combined 11 752 is inflated by re-tuned **ablation**
   branches, which are not the deployed policy.

2. **`no_memory` "trust" is raw evidence.** In the `no_memory` ablation
   (`use_trust=False`) the band map consumes `T=E` directly, so a "trust
   decreases" event there is literally "evidence decreases." Legitimate for the
   mechanical definition, but not a *calibrated* trust state. The clean
   deployed-policy number is MAIN-001 `rova` = 2 366.

3. **These are policy-behavior facts, not manuscript falsifications.** "Trust
   down & authorization up" contradicts a manuscript sentence only under an
   **operational/temporal** reading of that sentence ("over time, a drop in
   trust can only tighten permissions"). Under a narrow **map-theoretic** reading
   ("holding other inputs fixed, authorization is nondecreasing in the trust
   argument") the same sentences are TRUE — and that narrow property is
   separately VERIFIED here (`narrow_property.violations = 0`). This package
   deliberately reports only the mechanical fact and leaves the
   sentence-by-sentence verdict, which is interpretation-dependent, to the reader.

### Claims deliberately NOT in this table

- *"No reported empirical result depends on the false property"* and the
  manuscript claim-strength verdicts (VERIFIED/FALSIFIED labels on specific
  sentences) are **interpretive**, not execution-derived, so they are out of
  scope for this reproduction package, which only demonstrates policy behavior.
- The headline regret verdict itself (baseline vs ROVA across cost ratios) is
  read from archived `stats_summary.json`; this package independently confirms
  only that the **replayed per-fold ROVA regret matches the archive** (the
  `regret_match` self-check), not the full baseline comparison.
