"""Reconstruct MAIN-001 manifest.json from the authoritative per-unit
artifacts (per-unit HPO checkpoints, containment reports, DP-oracle JSON) and
validate it against the runbook's completion criteria (sections 9.1, 9.2, 9.4).

Rationale: the live manifest.json produced during the parallel run is
unreliable due to a lost-update race (15 shards read-modify-writing one shared
file; utils.save_json is non-atomic/unlocked). The per-unit checkpoint files
are each written by a single shard (units never overlap across shards), so they
are the trustworthy source of truth. This script derives the manifest from them
-- it changes no scientific data, hyperparameter, or protocol.

Run:  python3 rebuild_and_validate.py <OUT_DIR>
Exit 0 iff all 61 required entries validate.
"""
import sys, os, json, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from dtcoa_val.experiments.environments import ENVIRONMENTS, COST_RATIO_GRID

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
CK = os.path.join(OUT, "hpo_checkpoints")
METHODS = ["static", "ewma", "rova"]
DP_ENV = "unfavorable_oscillation_uncoupled"

problems = []
completed = {}


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def check_method(env, c, method):
    tag = f"{env}__c{c}__{method}"
    path = os.path.join(CK, f"{tag}.json")
    if not os.path.exists(path):
        problems.append(f"MISSING checkpoint: {tag}")
        return
    d = json.load(open(path))
    res = d.get("results", [])
    if len(res) != 5:
        problems.append(f"{tag}: expected 5 folds, got {len(res)}")
        return
    for r in res:
        if not finite(r.get("test_mean_regret")):
            problems.append(f"{tag} fold {r.get('outer_fold')}: non-finite regret")
        if not finite(r.get("test_cvar95_regret")):
            problems.append(f"{tag} fold {r.get('outer_fold')}: non-finite CVaR")
    if d.get("K_out") != 5 or d.get("K_in") != 3 or d.get("n_trials") != 80:
        problems.append(f"{tag}: budget mismatch K_out={d.get('K_out')} "
                        f"K_in={d.get('K_in')} n_trials={d.get('n_trials')} (want 5/3/80)")
    mean = sum(r["test_mean_regret"] for r in res) / len(res)
    completed[tag] = {"summary": {"mean_test_regret": mean}, "source": "checkpoint"}


def check_containment(env, c):
    tag = f"{env}__c{c}__containment"
    path = os.path.join(OUT, f"{env}__c{c}__containment_report.json")
    if not os.path.exists(path):
        problems.append(f"MISSING containment report: {env}__c{c}")
        return
    d = json.load(open(path))
    for k in ("prop_collapsed_to_zero", "stab_collapsed_to_zero"):
        if k not in d:
            problems.append(f"{tag}: missing field {k}")
    completed[tag] = {"summary": {k: d.get(k) for k in
                      ("prop_collapsed_to_zero", "stab_collapsed_to_zero")},
                      "source": "containment_report"}


def check_dp():
    tag = f"{DP_ENV}__dp_oracle_validation"
    path = os.path.join(OUT, f"{tag}.json")
    if not os.path.exists(path):
        problems.append(f"MISSING DP-oracle validation: {tag}")
        return
    d = json.load(open(path))
    if d.get("certified_equivalent") is not True:
        problems.append(f"{tag}: certified_equivalent != true ({d.get('certified_equivalent')})")
    completed[tag] = {"summary": {"certified_equivalent": d.get("certified_equivalent")},
                      "source": "dp_oracle_validation"}


# Enumerate the expected 61 required entries from the FROZEN grid.
for env in ENVIRONMENTS:
    for c in COST_RATIO_GRID:
        for m in METHODS:
            check_method(env, c, m)
        check_containment(env, c)
check_dp()

n_method = len(ENVIRONMENTS) * len(COST_RATIO_GRID) * len(METHODS)   # 45
n_cont = len(ENVIRONMENTS) * len(COST_RATIO_GRID)                    # 15
expected = n_method + n_cont + 1                                     # 61

print(f"expected required entries: {expected} (= {n_method} method + {n_cont} containment + 1 dp)")
print(f"validated & assembled:     {len(completed)}")
print(f"problems:                  {len(problems)}")
for p in problems:
    print("  - " + p)

if problems or len(completed) != expected:
    print("VALIDATION FAILED — manifest NOT written.")
    sys.exit(1)

manifest = {"completed": completed,
            "note": "Reconstructed from authoritative per-unit artifacts after a "
                    "15-shard lost-update race on the live manifest.json; see "
                    "manifest.live_raced.json for the raw raced file."}
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"VALIDATION PASSED — manifest.json written with {len(completed)}/61 required entries.")
