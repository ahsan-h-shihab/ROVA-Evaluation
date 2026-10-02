"""Post-run behavioural comparison with always-full-supervision (A0).

Exploratory, read-only re-analysis specified after the A1 results
(docs/REVISION_002_POSTRUN_ANALYSES.md). It runs no simulation and imports no
simulator code. For every cell and method it reports, on the matched evaluation
worlds, how often the method's whole trajectory is full supervision (every
agent-step at level 0, so its regret equals A0's on that world), the fractions of
agent-steps at level 0 and level 5, the oracle's full-supervision fraction and the
mean regret difference from A0.

    python -m rova_revision.behaviour_a0 --rev001-dir REV001/revision_out --a1-dir A1/a1_out --out OUT
"""
import argparse
import json
import os
import sys

import numpy as np

from .clustered import (CEW_UNIT_PREFIXES, ENVS, GRID, environment_record, sha256_file, verify_rev001_dir,
                        write_csv, write_json, write_sums)

A1_SUMS_SHA256 = "36a54f1644c23c429429d4c8e479cf4f8b72b848839c1f4a46a02a8ee15d1899"
A1_SUMS_ENTRIES = 182
REV001_METHODS = ("rova", "static", "ewma")
A1_METHODS = tuple(f"{m}_tpe{b}" for m in ("rova", "static", "ewma") for b in (80, 160, 320))


def _key(r):
    return int(r["outer_fold"]), int(r["seed"])


def load_cell(rev001_dir, a1_dir, env, c):
    recs = {}
    for pre in CEW_UNIT_PREFIXES:
        with open(os.path.join(rev001_dir, "units", f"{pre}{c}__{env}", "result.json")) as f:
            recs.update(json.load(f)["records"])
    with open(os.path.join(a1_dir, "units", f"A1__S3__a1_eval__{c}__{env}", "result.json")) as f:
        recs.update(json.load(f)["records"])
    return recs


def cell_rows(recs, env, c):
    a0 = {_key(r): r for r in recs["A0"]}
    rows = []
    for m in REV001_METHODS + A1_METHODS:
        rs = sorted(recs[m], key=_key)
        if [_key(r) for r in rs] != sorted(a0) or any(r.get("tag") != a0[_key(r)].get("tag") for r in rs):
            raise ValueError(f"{env} c={c} {m}: worlds differ from A0's")
        steps = np.array([r["n_agent_steps"] for r in rs], float)
        l0 = np.array([r["a_counts"][0] for r in rs], float)
        l5 = np.array([r["a_counts"][5] for r in rs], float)
        ast0 = np.array([r["astar_counts"][0] for r in rs], float)
        all_full = l0 == steps
        d = np.array([r["regret"] - a0[_key(r)]["regret"] for r in rs])
        rows.append({
            "env": env, "c": c, "method": m, "n_worlds": len(rs),
            "worlds_all_full_supervision": int(all_full.sum()),
            "fraction_worlds_all_full_supervision": float(all_full.mean()),
            "regret_equals_A0_on_those_worlds": bool(np.all(np.abs(d[all_full]) == 0.0)),
            "fraction_agent_steps_level0": float(l0.sum() / steps.sum()),
            "fraction_agent_steps_level5": float(l5.sum() / steps.sum()),
            "oracle_fraction_level0": float(ast0.sum() / steps.sum()),
            "mean_regret": float(np.mean([r["regret"] for r in rs])),
            "mean_regret_minus_A0": float(d.mean()),
            "folds_with_all_worlds_full_supervision": int(sum(
                all(all_full[i] for i, r in enumerate(rs) if int(r["outer_fold"]) == k)
                for k in sorted({int(r["outer_fold"]) for r in rs}))),
        })
    return rows


def run(rev001_dir, a1_dir):
    return [row for env in ENVS for c in GRID for row in cell_rows(load_cell(rev001_dir, a1_dir, env, c), env, c)]


def verify_a1_dir(a1_dir):
    res = verify_rev001_dir(a1_dir, expected_sums=A1_SUMS_SHA256, expected_entries=A1_SUMS_ENTRIES)
    return res


def report(rows):
    L = ["# Behavioural comparison with always-full-supervision (A0)", "",
         "Exploratory, post-run, read-only (docs/REVISION_002_POSTRUN_ANALYSES.md). Matched evaluation worlds, "
         "n = 60 per cell. 'All full supervision' = every agent-step of the world at level 0 (regret then equals A0's).", "",
         "| env | c | method | worlds all full supervision | agent-steps level 0 | level 5 | oracle level 0 | mean regret − A0 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['env']} | {r['c']} | {r['method']} | {r['worlds_all_full_supervision']}/{r['n_worlds']} | "
                 f"{r['fraction_agent_steps_level0']:.3f} | {r['fraction_agent_steps_level5']:.3f} | "
                 f"{r['oracle_fraction_level0']:.3f} | {r['mean_regret_minus_A0']:+.5f} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rev001-dir", required=True)
    ap.add_argument("--a1-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-unverified", action="store_true", help="code verification only")
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    if os.path.exists(out) and os.listdir(out):
        ap.error(f"output directory {out} is not empty")
    rv, av = verify_rev001_dir(os.path.abspath(args.rev001_dir)), verify_a1_dir(os.path.abspath(args.a1_dir))
    if not (rv["passed"] and av["passed"]) and not args.allow_unverified:
        raise SystemExit("input verification FAILED; nothing was analysed")
    rows = run(os.path.abspath(args.rev001_dir), os.path.abspath(args.a1_dir))
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "behaviour_a0.csv"), rows)
    write_json(os.path.join(out, "behaviour_a0.json"), {"rows": rows})
    with open(os.path.join(out, "BEHAVIOUR_A0_REPORT.md"), "w", newline="\n", encoding="utf-8") as f:
        f.write(report(rows))
    env = environment_record()
    write_json(os.path.join(out, "run_metadata.json"), {
        "analysis": "behaviour_a0 (docs/REVISION_002_POSTRUN_ANALYSES.md)",
        "label": "exploratory post-run read-only re-analysis; no simulation",
        "environment": {k: v for k, v in env.items() if k != "git_frozen_tag_commit"},
        "rev001_inputs": {k: rv[k] for k in ("sha256sums_sha256", "entries", "passed")},
        "a1_inputs": {k: av[k] for k in ("sha256sums_sha256", "entries", "passed")}})
    n = write_sums(out)
    print(f"behaviour_a0 done: {len(rows)} rows; SHA256SUMS {n} files; digest {sha256_file(os.path.join(out, 'SHA256SUMS'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
