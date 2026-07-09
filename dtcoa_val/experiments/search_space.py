"""Hyperparameter search spaces — one per method — for the matched-budget
random-search HPO required by PRE_REGISTERED_PROTOCOL.md §3.

Design rules encoded here directly from the protocol:
- §3.3: scale-type parameters are sampled log-uniform over a range wide
  enough to include values that effectively disable the mechanism.
- §3.3: threshold/band vectors are sampled under a monotonicity constraint
  (sample then sort), identical dimensionality/range across every method
  that has one — this is what "matched calibration freedom" means in
  practice, and is why StaticThreshold/EWMA and ROVA's bands all use the
  same 5-dimensional, same-range band sampler below.
- §3.2: W (fixed evidence weights) and binary structural switches are NOT
  part of this search space; W is architecture, switches are handled via
  the factorial ablation (search_space is for the *continuous* HPO only).

A search space is a dict: param_name -> spec, where spec is one of:
  ("loguniform", low, high)        -- scalar, sampled log-uniform
  ("uniform", low, high)           -- scalar, sampled uniform
  ("int_uniform", low, high)       -- integer, sampled uniform inclusive
  ("band_vector", n, low, high)    -- n-vector sampled uniform in [low,high], sorted
"""
import numpy as np

# Shared band-vector range: identical across every method with a threshold/band
# vector, per §3.3's matched-dimensionality-and-range requirement.
BAND_DIM = 5
BAND_LOW, BAND_HIGH = 0.02, 0.98

ROVA_SEARCH_SPACE = {
    "alpha": ("loguniform_complement", 1e-3, 0.999),   # near 0 -> memoryless; near 1 -> long memory
    "gamma": ("uniform", 0.05, 0.99),
    "e_min": ("uniform", 0.05, 0.60),
    "k": ("int_uniform", 1, 10),
    "context_risk": ("uniform", 0.0, 0.5),
    "bands": ("band_vector", BAND_DIM, BAND_LOW, BAND_HIGH),
    "prop_scale": ("loguniform_with_zero", 1e-3, 5.0),   # includes near-zero -> disables mechanism
    "stab_scale": ("loguniform_with_zero", 1e-3, 5.0),
    "s_min": ("uniform", 0.0, 0.95),
}

STATIC_THRESHOLD_SEARCH_SPACE = {
    "thr": ("band_vector", BAND_DIM, BAND_LOW, BAND_HIGH),
}

EWMA_SEARCH_SPACE = {
    "beta": ("loguniform_complement", 1e-3, 0.999),
    "thr": ("band_vector", BAND_DIM, BAND_LOW, BAND_HIGH),
}

# Clairvoyant cost-sensitive threshold (protocol §6): OracleLite with a
# tunable threshold vector, re-tuned PER COST RATIO with full knowledge of
# true reliability r (still hazard-blind by design -- it exists to express
# available headroom, not to compete as a policy). Same band range as every
# other threshold vector, per the matched-dimensionality requirement.
CLAIRVOYANT_SEARCH_SPACE = {
    "thr": ("band_vector", BAND_DIM, BAND_LOW, BAND_HIGH),
}


def sample_param(spec, rng):
    kind = spec[0]
    if kind == "uniform":
        _, lo, hi = spec
        return float(rng.uniform(lo, hi))
    if kind == "loguniform":
        _, lo, hi = spec
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if kind == "loguniform_with_zero":
        # 20% of trials sample exactly the floor (effectively disabled) so the
        # search can find/confirm a containment optimum at the boundary rather
        # than only approaching it asymptotically; remainder log-uniform.
        _, lo, hi = spec
        if rng.random() < 0.2:
            return 0.0
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if kind == "loguniform_complement":
        # For parameters in (0,1) where "near 1" is the meaningful extreme
        # (e.g. memory/smoothing constants): sample (1 - x) log-uniform in
        # [1-hi, 1-lo] so both the near-memoryless (x near lo) and
        # near-full-memory (x near hi) ends get adequate density, instead of
        # a plain uniform draw which would barely ever sample x > 0.99.
        _, lo, hi = spec
        comp = np.exp(rng.uniform(np.log(1 - hi), np.log(1 - lo)))
        return float(np.clip(1 - comp, lo, hi))
    if kind == "int_uniform":
        _, lo, hi = spec
        return int(rng.integers(lo, hi + 1))
    if kind == "band_vector":
        _, n, lo, hi = spec
        v = np.sort(rng.uniform(lo, hi, size=n))
        return v
    raise ValueError(f"unknown search-space kind: {kind}")


def sample_config(search_space, rng):
    return {name: sample_param(spec, rng) for name, spec in search_space.items()}


# ---------------------------------------------------------------------------
# One-at-a-time re-tuned ablations (frozen scope: NOT the full 2^4 factorial,
# see PROJECT_SCOPE_FREEZE.md -- full factorial is future work). Each
# ablation fixes one structural switch off and drops that component's
# now-irrelevant hyperparameters from the search space, so the remaining
# budget isn't wasted tuning parameters that have no effect.
# ---------------------------------------------------------------------------

def _restricted(exclude):
    return {k: v for k, v in ROVA_SEARCH_SPACE.items() if k not in exclude}


ROVA_ABLATIONS = {
    # memory ablation: does temporal integration (trust/EWMA) contribute?
    # NOTE (FIX-1, ABLATION-001): e_min is NOT dropped here. Disabling memory
    # (use_trust=False) kills only e_min's memory/decay path (the clow
    # accumulator, consumed solely inside `if self.use_trust:` in
    # dtcoa_tunable.py); e_min stays behaviorally active via the independent
    # propagation path (`obs_fail = E < e_min` -> prop_pen, gated by use_prop),
    # so it must be re-tuned in coupled environments. See EMIN_ANALYSIS.
    "no_memory": {
        "fixed_kwargs": {"use_trust": False},
        "search_space": _restricted({"alpha", "gamma", "k"}),
    },
    # decay ablation: does the forgetting submechanism contribute beyond plain memory?
    # NOTE (FIX-1, ABLATION-001): e_min is NOT dropped here either. Disabling
    # decay (use_decay=False) removes only the clow>=k decay trigger; e_min
    # remains active via the same propagation path, so it stays tunable.
    "no_decay": {
        "fixed_kwargs": {"use_decay": False},
        "search_space": _restricted({"gamma", "k"}),
    },
    # propagation ablation: does network-failure awareness contribute?
    "no_propagation": {
        "fixed_kwargs": {"use_propagation": False},
        "search_space": _restricted({"prop_scale"}),
    },
    # stability ablation: does trajectory regularization contribute?
    "no_stability": {
        "fixed_kwargs": {"use_stability": False},
        "search_space": _restricted({"stab_scale", "s_min"}),
    },
}


def make_ablation_factory(base_cls, fixed_kwargs):
    """Wrap a policy class so nested_cv_hpo's `policy_class(**config)` calls
    also receive the ablation's fixed structural kwargs. Duck-typed: nested_hpo
    only ever calls this as policy_class(**config), so a plain function works."""
    def factory(**config):
        merged = dict(fixed_kwargs)
        merged.update(config)
        return base_cls(**merged)
    return factory
