"""Mechanical application of the protocol's decision rules DR1-DR8 (protocol
section 9) and the factorial ablation analysis (protocol section 7), on the
common evaluation worlds. Operationalizations are fixed in the plan (R17, R15).

"Significant" means Holm-rejected across the entire family of tests reported by
the revision (``protocol_family_holm``), as protocol section 8 requires; the
revision's per-cell labels (within-family Holm, plan section 1) are reported
separately and are not used here.
"""
from itertools import product

import numpy as np

from dtcoa_val.analysis.stats import bca_bootstrap_ci, holm_bonferroni

from . import config as C
from .stathelp import NEGLIGIBLE, compare, holm_rows


# --------------------------------------------------------------------------- #
# Protocol-family Holm
# --------------------------------------------------------------------------- #
def collect_rows(obj, out=None):
    """Every test row (a dict carrying 'wilcoxon_p' and 'code') inside ``obj``."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        if "wilcoxon_p" in obj and "code" in obj:
            out.append(obj)
        for k, v in obj.items():
            if not str(k).startswith("_"):
                collect_rows(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_rows(v, out)
    return out


def protocol_family_holm(sections):
    rows = []
    for s in sections:
        collect_rows(s, rows)
    if not rows:
        return {"n_tests": 0}
    for r, (_i, _p, thr, rej) in zip(rows, holm_bonferroni([r["wilcoxon_p"] for r in rows])):
        r["protocol_family_threshold"], r["protocol_family_reject"] = thr, bool(rej)
    return {"n_tests": len(rows)}


# --------------------------------------------------------------------------- #
# DR1, DR2 (with crossover), DR6, DR8
# --------------------------------------------------------------------------- #
def _pair_row(cell, A, B):
    for fam in (cell.get("families") or {}).values():
        for r in fam:
            if "wilcoxon_p" not in r or "protocol_family_reject" not in r:
                continue
            if (r["a"], r["b"]) == (A, B):
                return r, 1.0
            if (r["a"], r["b"]) == (B, A):
                return r, -1.0
    return None, None


def _env_outcome(cell, A, B):
    r, sgn = _pair_row(cell, A, B)
    if r is None or A not in cell["methods"] or B not in cell["methods"]:
        return None
    d = sgn * r["cliffs_delta"]  # Cliff's delta(A, B); < 0 means A tends to lower regret
    sig = r["protocol_family_reject"]
    ca, cb = cell["methods"][A]["mean_ci"], cell["methods"][B]["mean_ci"]
    return {"delta_AB": d, "significant": sig,
            "A_better": bool(sig and d <= -NEGLIGIBLE and ca[1] < cb[0]),
            "B_better": bool(sig and d >= NEGLIGIBLE and cb[1] < ca[0]),
            "A_sig_direction": bool(sig and d < 0), "B_sig_direction": bool(sig and d > 0),
            "non_negligible_sig": bool(sig and abs(d) >= NEGLIGIBLE)}


def crossover(diffs_by_c, grid, n_boot):
    """Zero crossings of the mean paired gap A-B between adjacent grid points
    (linear in log c), with percentile intervals over world resamples drawn
    independently per cost ratio (the worlds differ across cost ratios)."""
    grid = [c for c in grid if c in diffs_by_c]
    if len(grid) < 2:
        return []
    x = np.log(np.array(grid, float))
    gaps = np.array([np.mean(diffs_by_c[c]) for c in grid])
    rng = np.random.default_rng(0)
    boot = np.stack([np.asarray(diffs_by_c[c])[rng.integers(0, len(diffs_by_c[c]), (n_boot, len(diffs_by_c[c])))].mean(axis=1)
                     for c in grid], axis=1)
    out = []
    for j in range(len(grid) - 1):
        g0, g1 = gaps[j], gaps[j + 1]
        if np.sign(g0) == np.sign(g1) or g0 == g1:
            continue
        xs = x[j] + g0 / (g0 - g1) * (x[j + 1] - x[j])
        b0, b1 = boot[:, j], boot[:, j + 1]
        cross = np.sign(b0) != np.sign(b1)
        xb = x[j] + b0[cross] / (b0[cross] - b1[cross]) * (x[j + 1] - x[j])
        ci = [float(np.exp(np.percentile(xb, 2.5))), float(np.exp(np.percentile(xb, 97.5)))] if cross.any() else None
        out.append({"bracket": [grid[j], grid[j + 1]], "crossing_c": float(np.exp(xs)), "ci_percentile": ci,
                    "fraction_resamples_crossing_in_bracket": float(cross.mean()),
                    "note": "resolution limited by the 5-point grid; interpolation linear in log c"})
    return out


def pairwise_rules(cells, raw, pairs, grid, envs, n_boot):
    out = {}
    for A, B in pairs:
        key = f"{A}_vs_{B}"
        per_c, deltas_all, any_data = {}, [], False
        for c in grid:
            env_out = {e: _env_outcome(cells.get(f"{e}|{c}", {}), A, B) if cells.get(f"{e}|{c}", {}).get("status") == "ok"
                       else None for e in envs}
            present = {e: v for e, v in env_out.items() if v is not None}
            if not present:
                continue
            any_data = True
            deltas_all += [v["delta_AB"] for v in present.values()]
            a_wins = sum(v["A_better"] for v in present.values())
            b_wins = sum(v["B_better"] for v in present.values())
            a_rev = any(v["B_sig_direction"] for v in present.values())
            b_rev = any(v["A_sig_direction"] for v in present.values())
            dr1 = f"{A} superior" if (a_wins >= 2 and not a_rev) else (
                f"{B} superior" if (b_wins >= 2 and not b_rev) else "no superiority claim")
            n_eff = sum(v["non_negligible_sig"] for v in present.values())
            per_c[str(c)] = {"per_env": env_out, "DR1": dr1, "envs_with_data": len(present),
                             "DR6_environment_specific": n_eff == 1}
        if not any_data:
            out[key] = {"status": "no data"}
            continue
        full_grid = len(per_c) == len(grid)
        all_env_A = full_grid and all(all(v and v["A_better"] for v in pc["per_env"].values()) for pc in per_c.values())
        all_env_B = full_grid and all(all(v and v["B_better"] for v in pc["per_env"].values()) for pc in per_c.values())
        some_A = any(pc["DR1"] == f"{A} superior" for pc in per_c.values())
        some_B = any(pc["DR1"] == f"{B} superior" for pc in per_c.values())
        dr2 = (f"{A} generally superior" if all_env_A else f"{B} generally superior" if all_env_B else
               "operating-condition-dependent advantage" if (some_A or some_B) else "no superiority claim")
        cross = {}
        if dr2 == "operating-condition-dependent advantage":
            for e in envs:
                diffs = {c: raw[(e, c)]["data"][A] - raw[(e, c)]["data"][B] for c in grid
                         if (e, c) in raw and A in raw[(e, c)]["data"] and B in raw[(e, c)]["data"]}
                cross[e] = crossover(diffs, grid, n_boot)
        dr8 = bool(deltas_all) and all(abs(d) < NEGLIGIBLE for d in deltas_all)
        out[key] = {"status": "ok", "per_cost_ratio": per_c, "DR2": dr2, "DR2_crossovers": cross,
                    "DR8_no_practically_significant_difference": dr8}
    return out


# --------------------------------------------------------------------------- #
# Factorial ablation (protocol section 7) and DR3
# --------------------------------------------------------------------------- #
def factorial_cell(data, n_boot, n_post):
    rows, identical = [], []
    for X in C.COMPONENTS:
        others = [y for y in C.COMPONENTS if y != X]
        for bits in product((0, 1), repeat=3):
            off_others = {y for y, b in zip(others, bits) if b}
            on_arch = C.architecture_name(off_others)
            off_arch = C.architecture_name(off_others | {X})
            setting = "+".join(y for y in others if y in off_others) or "none"
            base = {"component": X, "others_off": setting, "a": off_arch, "b": on_arch}
            if on_arch == off_arch:
                identical.append(dict(base, note="same architecture: decay acts only through memory"))
            elif on_arch in data and off_arch in data:
                rows.append({**base, **compare(data[off_arch], data[on_arch], n_boot, n_post)})
            else:
                rows.append({**base, "missing": [m for m in (off_arch, on_arch) if m not in data]})
    holm_rows(rows)
    effects = factorial_effects(data, n_boot)
    return {"removal_tests": rows, "structurally_identical": identical, "effects": effects}


def _arch_vec(data, off):
    name = C.architecture_name(off)
    return data.get(name)


def factorial_effects(data, n_boot):
    """Per-world main effects and two-way interactions of switching components off
    (positive = regret increases when switched off), BCa over worlds. Descriptive."""
    names = [C.architecture_name(s) for s in C.ARCHITECTURES]
    if any(n not in data for n in names):
        return {"status": "missing architectures", "missing": [n for n in names if n not in data]}
    comps = list(C.COMPONENTS)
    out = {"main": {}, "two_way": {}}
    for X in comps:
        others = [y for y in comps if y != X]
        terms = [_arch_vec(data, {y for y, b in zip(others, bits) if b} | {X}) -
                 _arch_vec(data, {y for y, b in zip(others, bits) if b}) for bits in product((0, 1), repeat=3)]
        v = np.mean(terms, axis=0)
        m, lo, hi = bca_bootstrap_ci(v, n_boot=n_boot)
        out["main"][X] = {"mean": m, "ci": [lo, hi]}
    for i, X in enumerate(comps):
        for Y in comps[i + 1:]:
            rest = [z for z in comps if z not in (X, Y)]
            terms = []
            for bits in product((0, 1), repeat=2):
                z = {w for w, b in zip(rest, bits) if b}
                terms.append(_arch_vec(data, z | {X, Y}) - _arch_vec(data, z | {X}) - _arch_vec(data, z | {Y})
                             + _arch_vec(data, z))
            v = np.mean(terms, axis=0)
            m, lo, hi = bca_bootstrap_ci(v, n_boot=n_boot)
            out["two_way"][f"{X}x{Y}"] = {"mean": m, "ci": [lo, hi]}
    return out


def dr3(factorial_by_cell, grid, envs):
    """Protocol DR3 per component and cost ratio: removing X degrades regret
    significantly (protocol family) with delta >= 0.147 in at least two environments,
    at every one of X's 8 factorial settings -> contributes; at some -> joint/interacting."""
    out = {}
    for X in C.COMPONENTS:
        out[X] = {}
        for c in grid:
            settings = {}
            for e in envs:
                cell = factorial_by_cell.get(f"{e}|{c}") or {}
                for r in cell.get("removal_tests", []):
                    if r["component"] != X or "protocol_family_reject" not in r:
                        continue
                    ok = bool(r["protocol_family_reject"] and r["cliffs_delta"] >= NEGLIGIBLE)
                    settings.setdefault(r["others_off"], {})[e] = ok
                for r in cell.get("structurally_identical", []):
                    if r["component"] == X:
                        settings.setdefault(r["others_off"], {})[e] = False
            if not settings:
                out[X][str(c)] = {"status": "no data"}
                continue
            passing = {s: sum(v.values()) >= 2 for s, v in settings.items()}
            n_pass = sum(passing.values())
            outcome = ("contributes" if n_pass == 8 else "joint/interacting effect" if n_pass > 0 else
                       "not detected at this budget")
            out[X][str(c)] = {"settings_passing": n_pass, "settings_evaluated": len(settings),
                              "per_setting_env_passes": settings, "outcome": outcome}
    return out


def dr4(pairwise, dr3_out, sweep):
    """Containment guard: weak dominance over a baseline, no component passing DR3,
    and tuned penalty magnitudes indistinguishable from zero in the sweep."""
    no_component = all(v.get("outcome") != "contributes" for X in dr3_out.values() for v in X.values())
    sweep_rows = [r for cell in sweep.values() if cell.get("status") == "ok" for r in cell["tuned_vs_zero"]
                  if r.get("b") in ("prop_scale=0", "stab_scale=0") and "protocol_family_reject" in r]
    indistinct = bool(sweep_rows) and all(not (r["protocol_family_reject"] and abs(r["cliffs_delta"]) >= NEGLIGIBLE)
                                          for r in sweep_rows)
    out = {"no_component_contributes": no_component, "penalties_indistinguishable_from_zero": indistinct}
    for base in ("static", "ewma"):
        pr = pairwise.get(f"rova_vs_{base}", {})
        cells = [pe for pc in pr.get("per_cost_ratio", {}).values() for pe in pc["per_env"].values() if pe]
        weak = bool(cells) and not any(pe["B_sig_direction"] and abs(pe["delta_AB"]) >= NEGLIGIBLE for pe in cells) \
            and any(pe["A_better"] for pe in cells)
        out[f"rova_weakly_dominates_{base}"] = weak
        out[f"DR4_fires_vs_{base}"] = bool(weak and no_component and indistinct)
    return out


BASE_CLASS = {"static_tpe": "static", "ewma_tpe": "ewma", "rova_tpe": "rova", "static_pb": "static",
              "ewma_pb": "ewma", "rova_pb": "rova", "rova_single": "rova", "rova_out": "rova", "ewma_out": "ewma"}


def dr7(pairwise, overhead):
    """Compute-conditional flag: DR1 superiority of A over B while A's measured
    per-decision time exceeds 3x B's (at N in {1, 6, 10})."""
    if not overhead or "per_N" not in overhead:
        return {"status": "overhead unavailable"}
    def t(m, N):
        name = BASE_CLASS.get(m, "rova" if m.startswith(("abl_", "fx_")) else m)
        return (overhead["per_N"].get(str(N)) or {}).get(name, {}).get("median_ns_per_agent_step")
    out = {}
    for key, pr in pairwise.items():
        if pr.get("status") != "ok":
            continue
        A, B = key.split("_vs_")
        winners = {pc["DR1"].replace(" superior", "") for pc in pr["per_cost_ratio"].values()
                   if pc["DR1"].endswith("superior")}
        for W in winners:
            L = B if W == A else A
            ratios = {N: (t(W, N) / t(L, N)) if t(W, N) and t(L, N) else None for N in (1, 6, 10)}
            out[f"{W}_over_{L}"] = {"time_ratio": ratios,
                                    "compute_conditional": any(r is not None and r > 3.0 for r in ratios.values()),
                                    "timing_note": "tuned variants timed through their policy class"}
    return out


DR5_STATUS = {"applicable": False,
              "reason": "protocol section 8 requires TOST at a pre-registered margin, but no numeric margin was "
                        "specified; the revision makes no equivalence claims"}
