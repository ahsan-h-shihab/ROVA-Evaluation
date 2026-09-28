"""Verify a finished REV-001 output directory (plan section 7). Runs no simulation.

    python -m rova_revision.verify_outputs --profile full --out revision_out
Exit status 0 only if every check passes.
"""
import argparse
import hashlib
import json
import os
import sys

from .config import PROFILES
from .units import enumerate_units, uid


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(profile, out_dir):
    checks, problems = {}, []
    sums_path = os.path.join(out_dir, "SHA256SUMS")
    listed = {}
    if os.path.exists(sums_path):
        with open(sums_path) as f:
            for line in f:
                if line.strip():
                    digest, rel = line.rstrip("\r\n").split("  ", 1)
                    listed[rel] = digest
    bad = [r for r, d in listed.items()
           if not os.path.exists(os.path.join(out_dir, r)) or _sha256(os.path.join(out_dir, r)) != d]
    present = set()
    for root, _dirs, files in os.walk(out_dir):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), out_dir).replace(os.sep, "/")
            if rel != "SHA256SUMS":
                present.add(rel)
    unlisted = sorted(present - set(listed))
    checks["sha256sums"] = bool(listed) and not bad and not unlisted
    problems += [f"checksum mismatch or missing: {r}" for r in bad] + [f"not in SHA256SUMS: {r}" for r in unlisted]

    not_done, digest_bad = [], []
    for u in enumerate_units(profile):
        d = os.path.join(out_dir, "units", u["uid"])
        if not (os.path.exists(os.path.join(d, "done.json")) and os.path.exists(os.path.join(d, "result.json"))):
            not_done.append(u["uid"])
            continue
        with open(os.path.join(d, "done.json")) as f:
            rec = json.load(f)
        if rec.get("status") != "ok" or rec.get("result_sha256") != _sha256(os.path.join(d, "result.json")):
            digest_bad.append(u["uid"])
    checks["all_units_done"] = not not_done
    checks["unit_digests"] = not digest_bad
    problems += [f"unit not done: {x}" for x in not_done] + [f"unit digest mismatch: {x}" for x in digest_bad]

    gate_path = os.path.join(out_dir, "units", uid(1, "gate"), "result.json")
    gate = json.load(open(gate_path)) if os.path.exists(gate_path) else {}
    checks["gate_passed"] = bool(gate.get("passed"))

    ana_path = os.path.join(out_dir, "units", uid(5, "analysis"), "result.json")
    ana = json.load(open(ana_path)) if os.path.exists(ana_path) else {}
    spath = os.path.join(out_dir, ana.get("summary_path", "analysis/summary.json"))
    checks["analysis_summary_digest"] = bool(ana) and os.path.exists(spath) and _sha256(spath) == ana.get("summary_sha256")
    checks["analysis_section_errors"] = bool(ana) and not ana.get("section_errors")

    meta = {}
    if os.path.exists(os.path.join(out_dir, "run_metadata.json")):
        with open(os.path.join(out_dir, "run_metadata.json")) as f:
            meta = json.load(f)
    runs = meta.get("runs", [])
    checks["env_guard_passed_every_invocation"] = bool(runs) and all(r["env_guard"]["passed"] for r in runs)
    heads = sorted({r["env_guard"]["observed"]["git_head"] for r in runs})
    checks["single_commit"] = len(heads) == 1
    approvals = [r["env_guard"]["approval_text"] for r in runs if r["env_guard"].get("overridden_by_approval")]
    return {"passed": all(checks.values()), "checks": checks, "problems": problems[:200],
            "git_heads": heads, "approvals_recorded": approvals,
            "hosts": sorted({r["env_guard"]["observed"]["hostname"] for r in runs}),
            "profiles": sorted({r["profile"] for r in runs})}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    res = verify(PROFILES[args.profile], os.path.abspath(args.out))
    print(json.dumps(res, indent=1))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
