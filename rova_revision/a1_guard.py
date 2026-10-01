"""A1 execution-environment gate (docs/REVISION_002_ANALYSIS_PLAN.md, section 2.2, G-ENV).

The REV-001 gate (``env_guard``) unchanged — same archived reference, same hard
and soft checks — except that the commit must be the one tagged
``revision-002-frozen`` instead of ``revision-001-frozen``.

    python -m rova_revision.a1_guard --workers 15 --json env_guard_a1.json
"""
import json
import sys

from . import env_guard as EG
from .rev002_constants import FROZEN_TAG_REV002


def collect(workers=None):
    info = EG.collect(workers)
    info["git_frozen_tag_name"] = FROZEN_TAG_REV002
    info["git_frozen_tag_commit"] = EG._git("rev-parse", f"{FROZEN_TAG_REV002}^{{commit}}")
    return info


def check(approval=None, workers=None):
    return EG.evaluate(collect(workers), EG.reference(), approval)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approval", default=None)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    res = check(approval=args.approval, workers=args.workers)
    print(EG.format_table(res))
    if args.json:
        with open(args.json, "w", newline="\n") as f:
            json.dump(res, f, indent=2)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
