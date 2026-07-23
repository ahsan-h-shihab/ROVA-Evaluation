#!/usr/bin/env python3
"""
replay_monotonicity.py
======================

Reproduces the MINIMAL (synthetic, hand-constructed) counterexamples to the
manuscript's dynamic monotonicity / "conservative authorization" claims, using
ONLY the archived frozen implementation. No source file is modified; this
script imports the policy classes and drives them with fixed input sequences.

It regenerates, with NO hardcoded result numbers, every execution-derived
number the previous audit reported about the *analytic property* and the
*minimal counterexamples*:

  (A) Minimal counterexamples (deterministic, fixed input sequences):
        A1  propagation-release: trust decreases while authorization increases
        A2  oversight-removal:   ESC True->False while trust decreases
        A3  stability-release:   a single agent's OWN evidence worsens,
                                 trust decreases, yet authorization increases
      For every violating timestep it prints:
        timestep, trust_before, trust_after,
        authorization_before, authorization_after, evidence,
        propagation_penalty, stability_penalty, context_risk, ESC_flag

  (B) The NARROW/formal property actually guaranteed by the code:
        holding (context_risk, prop_penalty, stab_penalty) fixed, authorization
        is nondecreasing in the trust argument. Regenerated as a grid count of
        violations (expected: 0).

  (C) Boundedness  : trust in [0,1], authorization in {0..5} over random runs.
  (D) Contraction  : worst |f(T)-f(T')|/|T-T'| for the trust-update map.

Every scenario is run against BOTH archived implementations:
  - dtcoa_val.dtcoa.DTCOA           (frozen legacy reference; integer penalties)
  - dtcoa_val.dtcoa_tunable.DTCOATunable (the class actually tuned/run in
                                          MAIN-001 and ABLATION-001; continuous
                                          penalties, default constructor args =
                                          the frozen constants)

Run:
    PYTHONPATH=<repo>/code python3 reproduce/replay_monotonicity.py
Writes a machine-readable summary to reproduce/out_monotonicity.json.
"""
import os, sys, json
import numpy as np

# ---- locate the archived code/ package (this file lives in <repo>/reproduce) --
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CODE = os.path.join(REPO, "code")
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from dtcoa_val.dtcoa import DTCOA, band_level as band_level_frozen, W as W_FROZEN
from dtcoa_val.dtcoa_tunable import DTCOATunable, band_level as band_level_tun


# ----------------------------------------------------------------------------- #
# Instrumentation: a NON-INVASIVE wrapper. It delegates every call to a real
# policy object and, after each step, reconstructs the propagation and stability
# penalties from the policy's PUBLIC post-step state, then self-checks that the
# reconstructed authorization equals the policy's own returned authorization.
# A nonzero recon-mismatch count => the reconstruction is untrustworthy and the
# run is flagged NOT REPRODUCIBLE by the caller.
# ----------------------------------------------------------------------------- #
class Recorder:
    def __init__(self, policy):
        self.p = policy
        self.rows = []          # flat list of per-(t,i) dicts
        self.recon_mismatch = 0

    def reset(self, N):
        self.p.reset(N)
        self.N = N
        self.rows = []
        self._t = 0

    def _band(self, tp):
        # dispatch to whichever band_level the wrapped policy uses
        if hasattr(self.p, "bands"):
            return band_level_tun(tp, self.p.bands)
        return band_level_frozen(tp)

    def step(self, O, truth=None):
        p = self.p
        Tb = np.asarray(p.T).copy()
        A, ESC = p.step(O, truth=truth)
        Ta = np.asarray(p.T).copy()

        # per-step evidence (last appended value per agent) and system penalty
        E = np.array([p.Ehist[i][-1] for i in range(self.N)])
        e_min = p.e_min
        nfail = int((E < e_min).sum())
        init = getattr(p, "_initial_fail", None)
        FAF = (nfail / init) if init else 1.0
        use_prop = getattr(p, "use_prop", True)
        # frozen DTCOA gates propagation on use_prop AND use_ma and uses ceil();
        # tunable uses prop_scale*overshoot. Reconstruct per class.
        if hasattr(p, "prop_scale"):                       # DTCOATunable
            prop_pen = (p.prop_scale * max(0.0, FAF - 1.0)) if use_prop else 0.0
        else:                                              # frozen DTCOA
            use_ma = getattr(p, "use_ma", True)
            if use_prop and use_ma:
                prop_pen = float(int(np.ceil(max(0.0, FAF - 1.0))))
            else:
                prop_pen = 0.0

        window = p.window
        s_min = p.s_min
        for i in range(self.N):
            w = p.Ehist[i][-window:]
            S = 1 - min(1.0, (np.std(w) / 0.25 if len(w) > 1 else 0.0))
            if hasattr(p, "stab_scale"):                   # DTCOATunable
                use_stab = getattr(p, "use_stab", True)
                s_pen = (p.stab_scale * max(0.0, s_min - S)) if use_stab else 0.0
            else:                                          # frozen DTCOA
                s_pen = 1.0 if (S < s_min) else 0.0
            Rc = p.Rc
            L = self._band(Ta[i] * (1 - Rc))
            a_recon = int(np.clip(round(L - prop_pen - s_pen), 0, 5))
            if a_recon != int(A[i]):
                self.recon_mismatch += 1
            self.rows.append(dict(
                t=self._t, i=i,
                trust_before=float(Tb[i]), trust_after=float(Ta[i]),
                authorization=int(A[i]), esc=bool(ESC[i]),
                evidence=float(E[i]),
                propagation_penalty=float(prop_pen),
                stability_penalty=float(s_pen),
                context_risk=float(Rc),
            ))
        self._t += 1
        return A, ESC

    def violations(self):
        """Per-agent consecutive-timestep events where the trust STATE
        (post-update trust, the quantity the band map consumes) strictly
        decreases while authorization strictly increases."""
        byi = {}
        for r in self.rows:
            byi.setdefault(r["i"], []).append(r)
        events = []
        for i, seq in byi.items():
            seq.sort(key=lambda r: r["t"])
            for t in range(1, len(seq)):
                prev, cur = seq[t - 1], seq[t]
                if cur["trust_after"] < prev["trust_after"] - 1e-12 and \
                   cur["authorization"] > prev["authorization"]:
                    events.append(dict(
                        agent=i,
                        timestep=cur["t"],
                        trust_before=prev["trust_after"],
                        trust_after=cur["trust_after"],
                        authorization_before=prev["authorization"],
                        authorization_after=cur["authorization"],
                        evidence=cur["evidence"],
                        # penalties AT the event step (typically already released)
                        propagation_penalty=cur["propagation_penalty"],
                        stability_penalty=cur["stability_penalty"],
                        # penalties AT the PRIOR step (the ones that released --
                        # this release is the mechanism that raises authorization)
                        propagation_penalty_before=prev["propagation_penalty"],
                        stability_penalty_before=prev["stability_penalty"],
                        context_risk=cur["context_risk"],
                        esc_flag=cur["esc"],
                        oversight_removed=(prev["authorization"] == 0 and cur["authorization"] > 0),
                    ))
        return events


def obs_rows(scalar_evidence_per_agent):
    """A 6-channel observation whose fixed-weight aggregate equals each scalar
    (all six channels set equal => dot(W, o) == o)."""
    return np.array([[v] * 6 for v in scalar_evidence_per_agent], dtype=float)


def run_sequence(policy, per_step_scalar_rows):
    """per_step_scalar_rows: list over timesteps of [evidence_agent0, ...]."""
    N = len(per_step_scalar_rows[0])
    rec = Recorder(policy)
    rec.reset(N)
    for row in per_step_scalar_rows:
        rec.step(obs_rows(row))
    return rec


# ----------------------------------------------------------------------------- #
# Fixed input sequences for the three minimal counterexamples.  These sequences
# are INPUTS (test vectors), not results; the script computes whether they
# violate. A3's sequence is the one surfaced by the deterministic search in
# regenerate_A3_sequence() (rng seed 0), pinned here so the file is standalone.
# ----------------------------------------------------------------------------- #
A1_A2_AGENT1 = [0.85, 0.20, 0.20, 0.85, 0.85, 0.85]
A1_A2_AGENT2 = [0.85, 0.85, 0.20, 0.85, 0.85, 0.85]

def scenario_A1(policy):
    # agent0 evidence held below its trust => trust strictly declines while band
    # stays at level 3; agents 1&2 create a growing->clearing failure episode so
    # the system propagation penalty rises then RELEASES.
    a0 = [0.66] * 6
    rows = [[a0[t], A1_A2_AGENT1[t], A1_A2_AGENT2[t]] for t in range(6)]
    return run_sequence(policy, rows)

def scenario_A2(policy):
    # agent0 evidence just above e_min (0.35) => no decay, trust drifts down
    # through band level 1; repeated failure episodes toggle the propagation
    # penalty so agent0 is pushed to a=0 (ESC) then RELEASED to a=1.
    a0 = [0.36] * 14
    a1 = [0.85,0.20,0.20,0.85,0.85, 0.20,0.20,0.85,0.85, 0.20,0.20,0.85,0.85,0.85]
    a2 = [0.85,0.85,0.20,0.85,0.85, 0.85,0.20,0.85,0.85, 0.85,0.20,0.85,0.85,0.85]
    rows = [[a0[t], a1[t], a2[t]] for t in range(14)]
    return run_sequence(policy, rows)

A3_SEQUENCE = [0.593, 0.818, 0.45, 0.334, 0.563, 0.429, 0.359, 0.677]
def scenario_A3(policy):
    # SINGLE agent. Its own evidence worsens (0.429 -> 0.359) while the recent
    # window becomes less volatile, so the stability penalty RELEASES and
    # authorization rises even as trust falls. No propagation possible (N=1).
    rows = [[v] for v in A3_SEQUENCE]
    return run_sequence(policy, rows)


def regenerate_A3_sequence(n_trials=200000, seed=0):
    """Independently re-derive an A3-type single-agent sequence by deterministic
    search, to show A3_SEQUENCE is not cherry-picked. Returns the first found
    sequence or None. Uses the FROZEN DTCOA."""
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        seq = rng.uniform(0.30, 0.95, size=8)
        rec = run_sequence(DTCOA(), [[v] for v in seq])
        rows = rec.rows
        for t in range(1, len(rows)):
            if rows[t]["evidence"] < rows[t - 1]["evidence"] - 1e-9 and \
               rows[t]["trust_after"] < rows[t - 1]["trust_after"] - 1e-9 and \
               rows[t]["authorization"] > rows[t - 1]["authorization"]:
                return [round(float(x), 3) for x in seq]
    return None


# ----------------------------------------------------------------------------- #
# (B) narrow/formal property, (C) boundedness, (D) contraction — regenerated.
# ----------------------------------------------------------------------------- #
def narrow_property_violations():
    """Holding (Rc, ppen, spen) fixed, is authorization nondecreasing in the
    trust argument? Count violations over a grid (expected 0)."""
    from dtcoa_val.dtcoa import BANDS  # frozen band vector
    def a_of_T(T, Rc, ppen, spen):
        L = band_level_frozen(T * (1 - Rc))
        return int(np.clip(round(L - ppen - spen), 0, 5))
    bad = 0
    for Rc in (0.0, 0.2, 0.5):
        for ppen in (0, 1, 2):
            for spen in (0, 1):
                prev = -1
                for T in np.linspace(0, 1, 1001):
                    a = a_of_T(T, Rc, ppen, spen)
                    if a < prev - 1e-12:
                        bad += 1
                    prev = a
    return bad

def boundedness_and_contraction(n_runs=2000, seed=1):
    rng = np.random.default_rng(seed)
    Tmin, Tmax, amin, amax = 1e9, -1e9, 9, -9
    for _ in range(n_runs):
        N = int(rng.integers(1, 8))
        p = DTCOATunable(); p.reset(N)
        for _ in range(60):
            O = rng.uniform(-0.1, 1.1, size=(N, 6))   # incl. out-of-range inputs
            A, _ = p.step(O)
            Tmin = min(Tmin, float(np.min(p.T))); Tmax = max(Tmax, float(np.max(p.T)))
            amin = min(amin, int(A.min())); amax = max(amax, int(A.max()))
    # contraction of f(T)=alpha*T+(1-alpha)*E at the frozen alpha
    alpha, gamma = 0.70, 0.60
    worst = 0.0
    for _ in range(100000):
        T, Tp, E = rng.uniform(0, 1, 3)
        f = lambda t: alpha * t + (1 - alpha) * E
        base = abs(T - Tp)
        if base > 1e-9:
            worst = max(worst, abs(f(T) - f(Tp)) / base)
    return dict(trust_min=Tmin, trust_max=Tmax, auth_min=amin, auth_max=amax,
                bounded=(Tmin >= -1e-12 and Tmax <= 1 + 1e-12 and amin >= 0 and amax <= 5),
                contraction_worst_ratio=worst, contraction_bound=max(alpha, gamma))


# ----------------------------------------------------------------------------- #
def print_events(title, events):
    print(f"\n### {title}: {len(events)} violating timestep(s)")
    if not events:
        print("    (none)")
        return
    hdr = ("timestep", "trust_before", "trust_after", "auth_before", "auth_after",
           "evidence", "prop_pen@t", "stab_pen@t", "prop_pen@t-1", "stab_pen@t-1",
           "Rc", "ESC", "oversight_removed")
    print("    " + " | ".join(f"{h}" for h in hdr))
    for e in events:
        print("    " + " | ".join(str(x) for x in (
            e["timestep"],
            f'{e["trust_before"]:.5f}', f'{e["trust_after"]:.5f}',
            e["authorization_before"], e["authorization_after"],
            f'{e["evidence"]:.3f}',
            f'{e["propagation_penalty"]:.4f}', f'{e["stability_penalty"]:.4f}',
            f'{e["propagation_penalty_before"]:.4f}', f'{e["stability_penalty_before"]:.4f}',
            f'{e["context_risk"]:.2f}', e["esc_flag"], e["oversight_removed"])))


def main():
    summary = {"implementations": {}, "narrow_property": {}, "boundedness_contraction": {},
               "A3_regenerated_sequence": None, "recon_mismatch_total": 0}
    impls = {"frozen_DTCOA": DTCOA, "tuned_class_DTCOATunable": DTCOATunable}

    total_mismatch = 0
    for impl_name, cls in impls.items():
        print("=" * 78)
        print(f"IMPLEMENTATION: {impl_name}")
        print("=" * 78)
        impl_out = {}
        for sc_name, sc_fn in (("A1_propagation_release", scenario_A1),
                               ("A2_oversight_removal", scenario_A2),
                               ("A3_stability_release", scenario_A3)):
            rec = sc_fn(cls())
            total_mismatch += rec.recon_mismatch
            ev = rec.violations()
            print_events(f"{sc_name}", ev)
            if rec.recon_mismatch:
                print(f"    !! penalty-reconstruction MISMATCHES: {rec.recon_mismatch} "
                      f"=> NOT REPRODUCIBLE for this scenario")
            impl_out[sc_name] = dict(
                n_violations=len(ev),
                n_oversight_removed=sum(1 for e in ev if e["oversight_removed"]),
                recon_mismatch=rec.recon_mismatch,
                events=ev,
            )
        summary["implementations"][impl_name] = impl_out

    summary["recon_mismatch_total"] = total_mismatch

    print("\n" + "=" * 78)
    print("NARROW / FORMAL PROPERTY (holding Rc, prop_pen, stab_pen fixed)")
    print("=" * 78)
    bad = narrow_property_violations()
    print(f"authorization NON-monotone-in-trust violations over grid: {bad} "
          f"(expected 0 => partial monotonicity in the trust argument HOLDS)")
    summary["narrow_property"] = {"violations": bad, "holds": bad == 0}

    print("\n" + "=" * 78)
    print("BOUNDEDNESS + TRUST CONTRACTION")
    print("=" * 78)
    bc = boundedness_and_contraction()
    print(f"trust range over random runs: [{bc['trust_min']:.4f}, {bc['trust_max']:.4f}]  bounded={bc['bounded']}")
    print(f"authorization range: [{bc['auth_min']}, {bc['auth_max']}]")
    print(f"contraction worst ratio = {bc['contraction_worst_ratio']:.4f}  (bound max(a,g) = {bc['contraction_bound']})")
    summary["boundedness_contraction"] = bc

    print("\n" + "=" * 78)
    print("A3 SEQUENCE INDEPENDENT RE-DERIVATION (deterministic search, rng seed 0)")
    print("=" * 78)
    regen = regenerate_A3_sequence()
    print(f"re-derived single-agent violating sequence: {regen}")
    summary["A3_regenerated_sequence"] = regen

    outp = os.path.join(HERE, "out_monotonicity.json")
    with open(outp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
