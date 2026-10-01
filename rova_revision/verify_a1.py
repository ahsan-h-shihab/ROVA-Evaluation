"""Verify a finished A1 output directory (docs/REVISION_002_ANALYSIS_PLAN.md, section 5).
Runs no simulation.

    python -m rova_revision.verify_a1 --profile full --out a1_out
Exit status 0 only if every check passes.
"""
import argparse
import json
import os
import sys

from .a1_config import A1_PROFILES, CONV_ABS, CONV_REL
from .a1_units import a1_uid, enumerate_a1_units
from .verify_outputs import _sha256


def verify(a1prof, out_dir):
    checks, problems = {}, []
    listed = {}
    sums = os.path.join(out_dir, "SHA256SUMS")
    if os.path.exists(sums):
        with open(sums) as f:
            for line in f:
                if line.strip():
                    d, rel = line.rstrip("\r\n").split("  ", 1)
                    listed[rel] = d
    bad = [r for r, d in listed.items()
           if not os.path.exists(os.path.join(out_dir, r)) or _sha256(os.path.join(out_dir, r)) != d]
    present = {os.path.relpath(os.path.join(root, n), out_dir).replace(os.sep, "/")
               for root, _d, files in os.walk(out_dir) for n in files}
    unlisted = sorted(present - set(listed) - {"SHA256SUMS"})
    checks["sha256sums"] = bool(listed) and not bad and not unlisted
    problems += [f"checksum mismatch or missing: {r}" for r in bad] + [f"not in SHA256SUMS: {r}" for r in unlisted]

    not_done, digest_bad = [], []
    for u in enumerate_a1_units(a1prof):
        d = os.path.join(out_dir, "units", u["uid"])
        done, res = os.path.join(d, "done.json"), os.path.join(d, "result.json")
        if not (os.path.exists(done) and os.path.exists(res)):
            not_done.append(u["uid"])
            continue
        with open(done) as f:
            rec = json.load(f)
        if rec.get("status") != "ok" or rec.get("result_sha256") != _sha256(res):
            digest_bad.append(u["uid"])
    checks["all_units_done"] = not not_done
    checks["unit_digests"] = not digest_bad and not not_done
    problems += [f"unit not done: {x}" for x in not_done] + [f"unit digest mismatch: {x}" for x in digest_bad]

    def load(u):
        p = os.path.join(out_dir, "units", u, "result.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    gate = load(a1_uid(1, "a1_gate"))
    checks["gate_passed"] = bool(gate.get("passed")) and bool(gate.get("r0_exact_zero"))
    ana = load(a1_uid(4, "a1_analysis"))
    spath = os.path.join(out_dir, ana.get("summary_path", "analysis/summary.json"))
    checks["analysis_summary_digest"] = bool(ana) and os.path.exists(spath) and _sha256(spath) == ana.get("summary_sha256")
    checks["analysis_section_errors"] = bool(ana) and not ana.get("section_errors")
    S = json.load(open(spath)) if os.path.exists(spath) else {}
    rep = S.get("reproduction", {})
    checks["reproduction_checks"] = bool(rep.get("gate_passed") and rep.get("campaign_checks_ok") and rep.get("eval_checks_ok"))
    const = S.get("constants", {})
    checks["convergence_constants_equal_plan"] = (const.get("CONV_REL") == CONV_REL and const.get("CONV_ABS") == CONV_ABS
                                                  and const.get("n_total") == a1prof.n_total
                                                  and const.get("select_budgets") == list(a1prof.select_budgets)
                                                  and const.get("frozen_startup_draw_per_fold") == a1prof.base.n_trials)
    meta = json.load(open(os.path.join(out_dir, "run_metadata.json"))) if os.path.exists(
        os.path.join(out_dir, "run_metadata.json")) else {}
    runs = meta.get("runs", [])
    checks["env_guard_passed_every_invocation"] = bool(runs) and all(r["env_guard"]["passed"] for r in runs)
    heads = sorted({r["env_guard"]["observed"]["git_head"] for r in runs})
    tags = sorted({r["env_guard"]["observed"].get("git_frozen_tag_name") for r in runs})
    checks["single_commit_at_rev002_frozen"] = len(heads) == 1 and tags == ["revision-002-frozen"]
    if not a1prof.scientific:  # plumbing check: the environment gate is informational
        checks.pop("env_guard_passed_every_invocation")
    return {"passed": all(checks.values()), "checks": checks, "problems": problems[:200], "git_heads": heads,
            "hosts": sorted({r["env_guard"]["observed"]["hostname"] for r in runs}),
            "convergence_counts": ana.get("convergence_counts")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(A1_PROFILES), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    res = verify(A1_PROFILES[args.profile], os.path.abspath(args.out))
    print(json.dumps(res, indent=1))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
