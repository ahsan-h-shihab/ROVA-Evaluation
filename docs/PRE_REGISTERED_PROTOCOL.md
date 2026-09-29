# Pre-Registered Evaluation Protocol

**Status: FINAL. This is the complete protocol as it currently stands, incorporating every correction identified by adversarial review. It is the protocol that would actually be executed. No experiment results currently exist for this protocol in its final form — see `PROJECT_STATUS_AND_HANDOVER.md` for what has and has not yet been run.**

This document is self-contained. It assumes a comparison between a richer, history-and-network-aware runtime-governance policy (referred to generically as the **governance policy**; see `CURRENT_FRAMEWORK_SPECIFICATION.md` for its full specification) and simpler baseline policies, evaluated in a simulated environment under an asymmetric cost structure (the cost of failing to restrict a harmful action vs. the cost of unnecessarily restricting a benign one).

Epistemic tags: **[FACT]** established methodological result; **[METHOD]** literature-grounded recommendation; **[ASSUMPTION]** a stipulation this protocol adopts; **[JUDGMENT]** a choice made where the literature underdetermines the decision.

---

## 1. Research Questions

**RQ1 — Best-found-under-budget performance.** *(Reworded from an earlier, overclaiming "best-achievable" framing — fixed-budget hyperparameter optimization estimates the best configuration *found* under a stated budget, not the architecture's true optimum, and these are different quantities that can diverge across methods of different search-space dimensionality.)* Under matched-budget hyperparameter optimization with nested cross-validation, what is each method's best-found held-out performance at each operating condition (cost ratio)?
*Answered by:* nested-CV test performance per method across the pre-registered cost-ratio grid, with confidence intervals and reported tuning curves. *Not answered by:* a single operating point, default-hyperparameter performance, or non-nested (in-sample) hyperparameter selection.

**RQ2 — Architectural/mechanism contribution.** *(Reworded similarly.)* Does the governance policy's distinctive machinery contribute performance beyond a fully-tuned simple baseline, and which specific component(s), if any, can the evidence attribute it to?
*Answered by:* factorial component ablations (see §6), each re-tuned after modification, with effect sizes, replicated across environments. *Not answered by:* aggregate governance-vs-baseline performance alone (this cannot attribute an effect to a component), or by ablations that are not re-tuned (which confound removing a component with simply de-tuning the method).

**RQ3 — Operating-condition (cost-ratio) dependence.** How does relative performance between methods vary with the cost ratio, and is there a crossover (rank change) in the ranking of methods as the cost ratio varies?
*Answered by:* the full performance curve over the cost-ratio grid, per environment. *Not answered by:* a single cost ratio, or by any aggregation across cost ratios that would average away a crossover.

**RQ4 — Computational cost.** What inference and tuning compute does each method require, and how does performance trade off against compute?
*Answered by:* per-decision inference cost (FLOPs or wall-clock) and total tuning cost (trial-evaluations) per method, reported alongside a performance-vs-compute view. *Not answered by:* performance numbers alone.

**RQ5 — Robustness and generalization.** Do the conclusions above hold across environments with different structural properties, across random seeds and data splits, and under distributional shift?
*Answered by:* replication across at least three structurally different environments (§5), a variance decomposition over seeds/splits, and at least one distribution-shift stress test. *Not answered by:* a single environment, a single seed, or in-distribution evaluation only.

---

## 2. Evaluation Philosophy

**Primary paradigm: tune every hyperparameter of every method at a matched budget [METHOD].** The literature on fair model comparison (e.g., black-box hyperparameter optimization used to control for confounding hyperparameter effects when comparing architectures; the CASH formalism in AutoML, which treats even algorithm/structural choice as part of a single tunable search space; tunability characterizations showing the performance gain from tuning is large and method-dependent) converges on this as the methodological standard. A protocol that *freezes* the richer method's memory, decay, or penalty-magnitude parameters while only tuning its output thresholds is **not** the field's standard practice and was rejected after a dedicated literature audit (see `THEORY_AND_METHODOLOGY_RECORD.md`, Part B).

**Secondary paradigm: an attribution-only analysis, reported alongside the primary.** After the primary (tune-everything) analysis, a secondary view fixes each method's non-output-layer parameters at their published/default values and tunes only the output threshold/band vector. This secondary view is *not* used to license superiority claims; it is reported only to help interpret *why* any observed advantage exists (is it carried by the operating-point/calibration layer alone, or does it require the richer dynamics?).

**Tertiary lens: factorial component ablation (§7).** Used exclusively to answer RQ2 (mechanism attribution), never to license a general superiority claim.

**What is fixed regardless of paradigm:** the problem definition itself — the environment/simulator, the observation channels available to every method, the loss function, and the cost-ratio grid swept. These are not hyperparameters of any method; they define the comparison.

---

## 3. Matched-Budget Hyperparameter Optimization

### 3.1 What is tuned

*All* performance-affecting hyperparameters of *every* method, in the primary analysis:
- Governance policy: output band vector, memory/trust constant, decay parameters (decay factor, dwell/patience length, internal evidence gate), contextual-risk discount scaling, and **penalty magnitudes** (network-propagation penalty scale, trajectory-stability penalty scale).
- Memoryless threshold baseline: full monotone threshold vector.
- Smoothed-threshold baseline: full monotone threshold vector plus its smoothing constant.

### 3.2 What is fixed (as part of the problem definition, not as a per-method hyperparameter)

The environment/simulator and its parameters, the observation channels, the loss function, the cost-ratio grid, and the HPO algorithm/budget itself (identical across methods — see §3.4). Binary structural on/off switches for the governance policy's components (propagation-awareness, stability-awareness) are **not** part of the main HPO search; they are handled exclusively through the factorial ablation in §7, because toggling them changes which method is being evaluated rather than tuning a fixed method.

### 3.3 Search ranges

- Scale-type parameters (memory/trust constant, decay factor, penalty magnitudes, contextual-discount scaling, smoothing constant): sampled log-uniform over a pre-registered range wide enough to include values that effectively **disable** the corresponding mechanism (e.g., penalty magnitude → 0, memory constant → a value producing near-memoryless behavior). This is essential and intentional: it allows the search itself to discover whether the richer method's optimal configuration collapses toward the simpler method's behavior, which is direct evidence relevant to RQ2 and to the containment question in DR4 (§9).
- Threshold/band vectors: sampled under a monotonicity constraint (the sampled vector is sorted) across its full meaningful range, identically in dimensionality and range across every method that has such a vector.

### 3.4 HPO algorithm and budget

**Identical HPO algorithm and identical trial budget for every method: random search [METHOD].** Random search is reproducible, requires no surrogate-model assumptions that could implicitly favor one method's search landscape over another's, is embarrassingly parallel, and has an unambiguous budget definition (number of trials). A **Bayesian-optimization re-run** is performed as a secondary robustness check, using the same budget, to confirm conclusions are not an artifact of the chosen optimizer.

**Budget-definition comparison (resolved explicitly) [METHOD/JUDGMENT]:**
- *Equal trials under an identical algorithm* — hardware-independent and reproducible, but a higher-dimensional search space may need more trials to cover equally well. **Selected as primary**, paired with the tuning-curve reporting below to make any resulting asymmetry visible rather than hidden.
- *Equal wall-clock time* — rejected as a primary criterion: it conflates the comparison with implementation/hardware details and is not reproducible across re-runs.
- *Equal FLOPs* — rejected as a primary criterion: in this simulation setting FLOPs are typically dominated by the simulator itself rather than by search effort, so equalizing FLOPs does not cleanly equalize *search effort*. Compute is instead reported separately, addressing RQ4.

A fixed trial budget *B_HPO* (e.g., 200 trials) is pre-registered identically for every method in every environment and every outer fold.

### 3.5 Early stopping

No early-stopping rule that differs by method. If any multi-fidelity scheduling (e.g., successive halving) is used to manage compute, the identical schedule is applied to every method without exception.

### 3.6 Budget-sensitivity analysis

For every method/environment/fold, the **tuning curve** (best-so-far performance as a function of trials used so far) is recorded and reported at sub-budgets (e.g., 25, 50, 100, 200 trials), not only at the final budget. This makes any budget-dependence of the conclusions visible and auditable rather than hidden inside a single reported number, and is the chosen mitigation for the irreducible search-space-dimensionality asymmetry noted in §10.

---

## 4. Nested Cross-Validation

**Required [FACT]:** selection-bias over-fitting from choosing hyperparameters and then reporting performance on the same data can be of a magnitude comparable to the performance differences between the methods under comparison, and must not be ignored.

- **Outer folds** (*K_out* = 5): estimate generalization performance. Each outer test fold is touched exactly once, after all tuning is complete for that fold.
- **Inner folds** (*K_in* = 5, or an equivalent fixed inner holdout): used exclusively to select each method's hyperparameters within a given outer training set.
- The unit of cross-validation is the **scenario instance** (a full simulated trajectory), not an individual timestep within an instance, because timesteps within an instance are not exchangeable (see §10, statistical assumptions).

---

## 5. Evaluation Environments

**Minimum: three environments, frozen before any results are observed [JUDGMENT].** One environment cannot distinguish a genuine effect from an artifact of that environment's specific generative structure; two environments yield only a binary consistency check; three allows a consistent pattern to be distinguished from an idiosyncratic one.

**Required structural diversity.** The chosen environments must differ along at least: (a) strength of temporal persistence in the hazard process (does the past predict the future, which is what rewards memory); (b) strength of network-propagation/coupling (do failures cascade across agents, which is what rewards network-awareness); (c) observation noise level; (d) hazard base rate. **At least one environment must be deliberately constructed to be unfavorable to the governance inductive bias** (low temporal persistence, low propagation coupling), so that the test is not stacked in favor of the richer architecture.

**Robustness measurement:** per-environment cost-ratio curves are reported individually (not pooled across environments by default); consistency is assessed by whether the decision-rule outcomes (§9) agree across environments; a seed/split-level variance decomposition is performed; and at least one distribution-shift stress test is run (train/tune under one regime, evaluate under a shifted one) for RQ5.

**Disclosed limitation, not removed by this protocol:** the environments are chosen by the researchers, not sampled from any external or principled distribution over possible governance environments. This is an irreducible, disclosed limitation of any internally-constructed benchmark (see §10).

---

## 6. Baselines and Reference Policies

**Mandatory baselines:**
- A **memoryless threshold** policy with a fully tunable monotone threshold vector.
- A **smoothed-threshold** policy (exponentially-weighted moving average of the observation, thresholded), with a tunable smoothing constant and tunable threshold vector — the minimal memory-using baseline.
- A **trivial reference** (always-permit and always-restrict) to anchor the metric's range.

**Oracle / reference policies (both required):**
- A **myopic, per-step Bayes-optimal oracle**, computed from the true (simulator-internal) state at each step. This was the regret reference used in earlier, pre-final phases of this project.
- A **validated sequential (dynamic-programming) optimum**, computed via value/policy iteration on the known generative model, which accounts for the possibility that governance actions affect *future* states (an endogenous-hazard setting). **This is the corrected, authoritative regret reference for the final protocol.** The myopic and sequential oracles must be explicitly compared before any regret numbers are trusted: if they agree (within a pre-registered tolerance) across the environments used, either may be reported, with the sequential version as the reference of record; if they disagree, **only the sequential oracle's regret is reported as the primary metric**, and the myopic-oracle regret is reported separately, labeled as a legacy/comparison quantity, not as evidence.
- A **clairvoyant cost-sensitive threshold**, tuned separately *per cost ratio* with full knowledge of the true state distribution, used to express how much headroom exists at each operating condition.

**Learned baseline (required):** a policy trained by direct optimization (e.g., reinforcement learning) of the same loss function, given the same observation channels as the memoryless/smoothed baselines. This guards against the criticism that the baseline set consists only of weak, hand-designed policies.

**Optional, strengthening but not required:** richer learned (e.g., recurrent) policies; additional governance-policy variants.

---

## 7. Ablation Studies (Factorial Design)

**Corrected from an earlier one-at-a-time design [fix applied].** Removing components one at a time cannot attribute an effect that depends on the *interaction* of two or more components — if two components only help jointly, neither passes a one-at-a-time test, and a real joint mechanism would be mislabeled as a pure containment/calibration effect. The final design is **factorial**.

**Binary components subjected to factorial ablation:** memory (on / effectively off via the memory constant), network-propagation awareness (on/off), trajectory-stability awareness (on/off), and the decay/forgetting submechanism (on/off). This yields 2⁴ = 16 configurations of the governance policy's architecture, **each independently re-tuned** under the full matched-budget HPO protocol of §3 (not merely evaluated with the full method's tuned hyperparameters re-used) — re-tuning is required, or the comparison conflates "removing a component" with "removing a component and de-tuning everything else."

**Two additional, non-binary ablations:**
- **Threshold/band vector ablation:** replace the tuned monotone vector with a single global threshold, isolating how much performance is attributable to output-calibration richness alone.
- **Penalty-magnitude sweep:** evaluate performance as the penalty-magnitude hyperparameters are swept continuously from 0 to their full pre-registered range (rather than only on/off), used specifically to test the *containment* question: does the governance policy's optimal configuration, under full tuning, collapse toward zero penalty magnitude (i.e., toward the simple baseline's behavior) or remain bounded away from zero?

**What each ablation answers:** memory ablation → does temporal integration contribute; propagation ablation → does network-failure awareness contribute; stability ablation → does trajectory regularization contribute; decay ablation → does the forgetting mechanism contribute beyond plain memory; threshold-vector ablation → how much is attributable to calibration richness alone; penalty-magnitude sweep → is any surviving advantage a genuine mechanism effect or a containment/expressiveness effect (directly informs DR4).

---

## 8. Statistical Analysis

- **Hypothesis tests:** paired **Wilcoxon signed-rank test** across seeds for any two-method comparison at a fixed cost ratio; **Friedman test** with **Holm post-hoc** correction when comparing more than two methods across the cost-ratio grid; **critical-difference diagrams** for the overall ranking. *(Caveat, disclosed rather than hidden: the Friedman procedure formally assumes independent comparison blocks, and the cost-ratio conditions here are evaluated on the same policies and the same scenario instances, so blocks are correlated; the per-cost-ratio Wilcoxon tests, effect sizes, and confidence intervals — not the Friedman omnibus alone — carry the primary evidentiary weight for this reason.)*
- **Effect sizes:** **Cliff's δ** reported alongside every p-value, with a pre-registered negligibility band (|δ| < 0.147) below which no practically meaningful difference is claimed regardless of statistical significance.
- **Confidence intervals:** 95% **BCa bootstrap** confidence intervals over seeds for every reported performance number and every reported gap between methods.
- **Equivalence testing:** **TOST** (two one-sided tests) at a pre-registered margin, used whenever an "equivalent" or "no difference" claim is made. A non-significant difference *alone* is never reported as evidence of equivalence — only a passed TOST licenses that claim (DR5).
- **Secondary Bayesian layer:** a Bayesian signed-rank / region-of-practical-equivalence analysis, reported as a robustness complement to the frequentist tests above, not as a replacement for them.
- **Multiple-comparison correction — corrected scope [fix applied]:** Holm correction is applied across the **entire family of analyses actually reported**, not only the primary tune-everything comparison. This explicitly includes the ablation comparisons, the Bayesian-optimization robustness re-run, the multiple-budget tuning-curve comparisons, and the distribution-shift stress test — every comparison a reader will see contributes to the corrected family, because each one inflates the chance of an apparently-significant result if left outside the correction.
- **Tail-risk co-primary metric [fix applied]:** in addition to mean regret, **CVaR at the 95th percentile** of per-instance loss is computed and reported as a co-primary metric for every comparison above. Mean regret is not, by itself, an adequate functional for evaluating a system whose purpose is plausibly tail-risk reduction rather than average-case optimization; conclusions that hold for mean regret but not for the tail-risk metric (or vice versa) must be reported as such, not collapsed into a single verdict.

---

## 9. Decision Rules (Pre-Registered; Determine What May Be Claimed)

All rules are symmetric — they apply identically regardless of which method is "A" and which is "B." "Significant" means Holm-corrected p < 0.05 under the corrected family scope of §8.

- **DR1 — Pointwise superiority [amended with a no-reversal clause — fix applied].** Claim "method A is superior to method B at cost ratio c" **if and only if**: A's nested-CV test performance is significantly better than B's, with Cliff's δ ≥ 0.147, with confidence intervals separating, in at least 2 of 3 environments, **and there is no environment in which B significantly beats A at that same c.** *(The earlier, uncorrected version of this rule allowed a majority-environment win to license a superiority claim even if the remaining environment showed a significant reversal in the opposite direction; this has been fixed. If a significant reversal exists anywhere, no superiority claim may be made at that c — report it instead as environment-specific and directionally inconsistent.)*
- **DR2 — General superiority.** Claim "A is generally superior to B" only if DR1's conditions hold at *every* cost ratio in *every* environment. If A is superior at some cost ratios and not others, claim only an *operating-condition-dependent advantage*, and report the crossover location with its confidence interval.
- **DR3 — Mechanism attribution.** Claim "component X contributes to performance" only if, under the factorial ablation of §7, removing X (holding the *combination* of other components fixed at each of its 8 factorial settings, then re-tuned) produces a significant degradation with δ ≥ 0.147 in at least 2 environments. A component that contributes only jointly with another (i.e., passes this test only in *some* factorial configurations) must be reported as a joint/interacting effect, not attributed to either component alone.
- **DR4 — Containment guard.** If the full governance policy weakly dominates the baseline but no component passes DR3, **and** the penalty-magnitude sweep (§7) shows the tuned-optimal magnitude is statistically indistinguishable from zero, report the result as an **expressiveness/containment effect** (the richer method's hypothesis class happens to contain a good policy, not that its distinctive machinery is doing anything) — report only the margin and the compute cost, and explicitly do not claim a mechanism.
- **DR5 — Equivalence.** Claim "A and B are equivalent at c" only if a TOST at the pre-registered margin rejects both one-sided null hypotheses. Non-significance of the ordinary difference test, by itself, never licenses this claim.
- **DR6 — Single-environment caveat.** If an effect (superiority, attribution, or otherwise) appears in only one of the tested environments, "general" claims of that effect are prohibited; report it explicitly as environment-specific with a stated hypothesis for why.
- **DR7 — Compute-conditional claims.** If method A is superior on performance but requires more than 3× the inference compute (FLOPs or wall-clock, pre-registered threshold) of method B, the superiority claim must be reported as compute-conditional, accompanied by the compute-normalized performance view.
- **DR8 — Practical-significance floor.** If |Cliff's δ| is below the negligibility band at every cost ratio in every environment, the paper must report "no practically significant difference" and frame the relevant section as a null/characterization result rather than implying any practical advantage exists.

---

## 10. Protocol Assumptions and Disclosed (Irreducible) Limitations

**Stated assumptions:**
- Scenario instances are treated as exchangeable for resampling purposes (bootstrap, Wilcoxon); timesteps *within* an instance are not assumed exchangeable, which is why the scenario instance, not the timestep, is the unit of cross-validation and resampling throughout.
- The sequential (DP-optimal) oracle is computed on the simulator's known generative model; its correctness is conditional on that model being an accurate representation of the environment being studied. This is validated by direct comparison against the myopic oracle (§6), but cannot be validated against any "true" external ground truth, since the simulator *is* the environment of study.
- The HPO algorithm (random search, with a Bayesian-optimization robustness check) is assumed to be a reasonable, non-favoring search procedure for all methods; this is mitigated, not eliminated, by using the identical algorithm and budget for every method.

**Disclosed limitations that this protocol does not, and cannot, remove [FACT — irreducible]:**
- **Search-space-dimensionality asymmetry.** Methods with more tunable parameters generically need a larger search budget to reach a comparably-converged optimum; at any *finite*, *matched-trial-count* budget, this can advantage the lower-dimensional method. No finite-budget protocol removes this for methods of genuinely unequal capacity — it can only be made visible (via the tuning-curve and multi-budget reporting of §3.6) and disclosed, never eliminated. This is treated as a property of the underlying scientific problem (comparing heterogeneous-capacity architectures), not a flaw specific to this protocol.
- **Environment-selection bias.** The environments in §5 are chosen by the researchers along axes the researchers identified as relevant; this is not equivalent to sampling from a principled or externally-validated distribution over possible governance environments. Including a deliberately unfavorable environment mitigates but does not eliminate this.

These two items must be stated explicitly as limitations in any resulting manuscript; this protocol does not claim to have solved them, only to have controlled and disclosed them as carefully as a finite-budget, internally-constructed evaluation can.

---

## 11. Stopping and Failure Criteria

**Stopping criteria:** every outer cross-validation fold is run to completion; there is no interim analysis, no optional stopping based on observed significance, and no decision to add or remove an environment, baseline, or ablation after any result has been observed. Any change to this protocol after data collection begins requires a new, dated pre-registration documenting the change and the reason for it.

**Failure / null-result criteria (pre-specified, not decided after seeing results):**
- If DR8 fires everywhere (no practically significant difference at any cost ratio, in any environment): the result is reported as a clean **null/characterization** finding — "under matched-budget tuning, the governance policy's distinctive machinery does not produce a practically significant advantage over the tuned simple baselines, across the tested range of operating conditions and environments." This is treated as a fully valid, fully reportable scientific outcome, not as a failed experiment.
- If results are inconsistent in direction across the three environments with no resolution under DR1's no-reversal clause: the result is reported as **environment-specific and inconclusive at the general level**, with the per-environment results reported individually rather than forced into a single headline claim.
- If the myopic and sequential oracles disagree substantially: this is reported explicitly, the sequential oracle's regret is used as authoritative going forward, and any prior-phase conclusions based on the myopic oracle are flagged as provisional/superseded in light of the corrected reference.

---

## 12. Evaluation Workflow (Execution Order)

1. Freeze this document: environments and their generators/parameters/seeds, search spaces, HPO budget, decision rules, and statistical plan. No further changes without a new dated pre-registration.
2. Validate the regret reference: compute both the myopic and the sequential (DP-optimal) oracle on each environment; report their agreement or disagreement; adopt the sequential oracle as authoritative per §6 and §11.
3. For each environment × each method: run the matched-budget random-search HPO (§3) with nested cross-validation (§4); record the inner-selected configuration per outer fold and the resulting outer-fold test performance; record the tuning curve at the pre-registered sub-budgets.
4. Evaluate every method's tuned configuration (per outer fold) across the full pre-registered cost-ratio grid, recording both mean regret and the CVaR tail-risk metric.
5. Run the Bayesian-optimization robustness re-run (§3.4) at the same budget, for comparison against the random-search result.
6. Run the factorial ablation study (§7), with full re-tuning at every one of the 16 architectural configurations, plus the threshold-vector ablation and the continuous penalty-magnitude sweep.
7. Run the distribution-shift stress test (train/tune under one regime, evaluate under a shifted one) for RQ5.
8. Compute the full statistical analysis of §8: per-cost-ratio paired tests, the Friedman/Holm omnibus, Cliff's δ, BCa bootstrap confidence intervals, TOST equivalence tests where relevant, and the secondary Bayesian ROPE layer — with the multiple-comparison correction applied across the entire reported family of analyses from steps 3–7.
9. Apply decision rules DR1–DR8 mechanically to the results of step 8 to determine exactly which claims are licensed; do not assert any claim not licensed by a decision rule.
10. Report compute (§3, RQ4): inference cost per decision and total tuning trial-evaluations per method, alongside the performance results.
11. Produce the primary deliverable: the full cost-ratio curve (mean regret and CVaR, with confidence intervals) per environment, never collapsed to a single scalar "winner."
12. Write the limitations section explicitly disclosing the two irreducible items in §10; do not present the protocol as having eliminated them.
