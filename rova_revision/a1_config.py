"""REV-002 constants: A1 profiles, the fixed convergence criterion and the
REV-001 input fingerprint (docs/REVISION_002_ANALYSIS_PLAN.md, sections 2 and 3).

Every constant that the plan fixes is defined here and recorded in the A1
output, so the verifier can check that the executed values equal the plan's.
"""
from dataclasses import dataclass

from .config import FULL, SMOKE, Profile
from .rev002_constants import (A2_N_BOOT, A2_SEED, CONV_ABS, CONV_REL, FROZEN_TAG_REV002,  # noqa: F401
                               REV001_ARCHIVE_SHA256, REV001_SUMS_ENTRIES, REV001_SUMS_SHA256,
                               fold_converged)

A1_METHODS = ("static", "ewma", "rova")


@dataclass(frozen=True)
class A1Profile:
    """``base`` is the REV-001 profile whose folds, seeds, world tags and frozen
    startup draw (``base.n_trials`` candidates per fold) are reused unchanged."""
    name: str
    base: Profile
    n_total: int
    select_budgets: tuple   # (reproduction budget, half budget, final budget)
    curve_budgets: tuple
    scientific: bool
    n_boot: int

    @property
    def b_repro(self):
        return self.select_budgets[0]

    @property
    def b_half(self):
        return self.select_budgets[1]

    @property
    def b_final(self):
        return self.select_budgets[2]


A1_FULL = A1Profile(name="full", base=FULL, n_total=320, select_budgets=(80, 160, 320),
                    curve_budgets=(20, 40, 80, 120, 160, 240, 320), scientific=True, n_boot=5000)

# Plumbing check: non-experimental seeds and shrunken worlds (inherited from SMOKE).
A1_SMOKE = A1Profile(name="smoke", base=SMOKE, n_total=16, select_budgets=(4, 8, 16),
                     curve_budgets=(2, 4, 8, 12, 16), scientific=False, n_boot=200)

A1_PROFILES = {p.name: p for p in (A1_FULL, A1_SMOKE)}


def method_name(method, budget):
    """Name of a method tuned by the extended TPE run and selected at ``budget``."""
    return f"{method}_tpe{budget}"
