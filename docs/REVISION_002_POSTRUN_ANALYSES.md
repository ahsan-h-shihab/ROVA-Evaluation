# REV-002 post-run analyses (exploratory)

**Date:** 2026-10-02. These analyses were specified **after** the A1 and A2 results were known.
They are exploratory, read-only re-analyses of stored outputs: they run no simulation, import no
simulator code, and do not modify any REV-001, A1 or A2 output. They are reported as exploratory
and never as pre-specified.

## P1 — Behavioural comparison with always-full-supervision

**Question.** How closely does the tuned ROVA policy reproduce always-full-supervision (A0), and
how does this compare with the tuned threshold baselines?

**Inputs.** REV-001 `revision_out/` (`SHA256SUMS` SHA-256
`12a72b84d635aee108950808fd8d0f7d901c56be0289b63ddf05c0f325577cc4`, 1,883 entries) and the A1
output `a1_out/` (archive SHA-256 `35459e1e096135531849a3a96d3f7501392c8d4f10474f00bc83913baa908e03`;
`SHA256SUMS` SHA-256 `36a54f1644c23c429429d4c8e479cf4f8b72b848839c1f4a46a02a8ee15d1899`, 182
entries); both verified before use.

**Quantities** (every cell; methods: ROVA, static, EWMA with the published random-search
configurations, and each at TPE 80, 160 and 320 candidates), on the matched evaluation worlds:
number of worlds in which every agent-step is at level 0 (the method's regret then equals A0's;
checked); fractions of agent-steps at level 0 and 5; the oracle's level-0 fraction; mean regret
minus A0's; number of outer folds in which every world is full supervision.

**Code.** `rova_revision/behaviour_a0.py`.

    python -m rova_revision.behaviour_a0 --rev001-dir REV001/revision_out --a1-dir A1/a1_out --out OUT
