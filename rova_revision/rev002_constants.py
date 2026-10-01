"""Constants fixed by docs/REVISION_002_ANALYSIS_PLAN.md (dependency-free, so the A2
re-analysis can import them without importing any simulator code)."""

FROZEN_TAG_REV002 = "revision-002-frozen"

# Plan section 2.3 (A1 convergence criterion). Not to be changed after any A1 result is seen.
CONV_REL = 0.05
CONV_ABS = 0.0002

# Plan section 2.2, G-IN: the REV-001 output used as input.
REV001_ARCHIVE_SHA256 = "63dae057d63c4917d90391fd9da72279f5e909ffb1a1b7e192999f55d28a6b10"
REV001_SUMS_SHA256 = "12a72b84d635aee108950808fd8d0f7d901c56be0289b63ddf05c0f325577cc4"
REV001_SUMS_ENTRIES = 1883

# Plan section 3.4 (A2).
A2_N_BOOT = 5000
A2_SEED = 0


def fold_converged(b_half, b_final, rel=CONV_REL, abs_=CONV_ABS):
    """Plan section 2.3 rule 1: B(half) - B(final) <= max(rel * B(half), abs)."""
    return bool((b_half - b_final) <= max(rel * b_half, abs_))
