"""ABLATION-001 driver (mechanism attribution / DR3).

Standalone entry point for the one-at-a-time re-tuned ablation experiment. It
reuses the FROZEN MAIN-001 evaluation harness, protocol, environments,
cost-ratio grid, seeds and HPO budget without touching the core-comparison
driver (run_pipeline.py) or any MAIN-001 artifact.

For each (environment, cost-ratio) cell it runs 5 independently re-tuned
campaigns -- the full ROVA model plus one per component ablation
(no_memory / no_decay / no_propagation / no_stability) -- each via the same
nested_cv_hpo at the same budget as MAIN-001. It then emits a per-cell DR3
attribution report and the containment read-out.

Design notes
------------
- Unit of parallelism is the CELL (env, cost_ratio); cells are independent
  (separate checkpoints, separate manifest keys) so shards never contend.
- Per-fold checkpointing/resumability is inherited from nested_cv_hpo: a
  re-run skips completed folds and resumes in-progress campaigns.
- FIX-5 (manifest race hardening): each shard writes ONLY its own manifest
  shard file (manifest.shard{idx}.json) using atomic write-and-rename, so the
  15-shard lost-update race that hit MAIN-001's single shared manifest.json
  cannot occur. The consolidated manifest.json is assembled by merging the
  shard files (idempotent, race-free). Per-cell checkpoints remain the
  authoritative source of truth.
- The ablation "full" branch carries a different run tag than MAIN-001's ROVA
  (`__full` vs `__rova`), so per-trial simulation seeding differs and the two
  are independent Monte-Carlo draws of the same estimand -- distributional
  consistency, not bit-identity (see validate_consistency.py).

Usage
-----
    python -m dtcoa_val.experiments.run_ablation --budget pilot --out <dir>
    python -m dtcoa_val.experiments.run_ablation --budget full  --out <dir> \
        --shard-index K --num-shards N
    python -m dtcoa_val.experiments.run_ablation --consolidate --out <dir>
"""
import argparse
import os
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from dtcoa_val.utils import save_json, load_json, stamp
from dtcoa_val.experiments.environments import ENVIRONMENTS, COST_RATIO_GRID, cfg_for
from dtcoa_val.experiments.search_space import (
    ROVA_SEARCH_SPACE, ROVA_ABLATIONS, make_ablation_factory,
)
from dtcoa_val.experiments.nested_hpo import nested_cv_hpo
from dtcoa_val.experiments.ablation_study import ablation_report, containment_report
from dtcoa_val.dtcoa_tunable import DTCOATunable
# Reuse the SAME frozen budget object as the core-comparison driver so the
# ablation cannot silently drift from MAIN-001's protocol (single source of
# truth). Importing run_pipeline is side-effect-free (its work is __main__-guarded).
from dtcoa_val.experiments.run_pipeline import BUDGETS

# The five re-tuned branches per cell. "full" uses the full ROVA search space;
# each ablation fixes one structural switch off and drops only the parameters
# provably inert under it (e_min stays tunable in no_memory/no_decay -- FIX-1).
BRANCHES = ["full", "no_memory", "no_decay", "no_propagation", "no_stability"]


def _branch_spec(branch):
    """Return (policy_class_or_factory, search_space) for a branch name."""
    if branch == "full":
        return DTCOATunable, ROVA_SEARCH_SPACE
    spec = ROVA_ABLATIONS[branch]
    return make_ablation_factory(DTCOATunable, spec["fixed_kwargs"]), spec["search_space"]


def _work_units():
    """Every (env, cost_ratio) cell in a fixed deterministic order -- the unit
    of sharding. Mirrors run_pipeline._work_units so the frozen grid is shared."""
    return [(env, c) for env in ENVIRONMENTS for c in COST_RATIO_GRID]


# --------------------------------------------------------------------------- #
# FIX-5: race-free per-shard manifests with atomic write-and-rename.
# --------------------------------------------------------------------------- #
def _shard_manifest_path(out_dir, shard_index):
    return os.path.join(out_dir, f"manifest.shard{shard_index}.json")


def _atomic_save_json(path, obj):
    """Write to a unique temp file then os.replace() -- atomic on POSIX, so a
    reader never sees a torn file and concurrent writers to DIFFERENT paths
    never interfere."""
    tmp = f"{path}.tmp.{os.getpid()}"
    save_json(tmp, obj)
    os.replace(tmp, path)


def _load_shard_manifest(out_dir, shard_index):
    p = _shard_manifest_path(out_dir, shard_index)
    if os.path.exists(p):
        return load_json(p)
    return {"shard": shard_index, "started_at": stamp(), "completed": {}}


def _mark_done(out_dir, manifest, shard_index, key, summary):
    manifest["completed"][key] = {"finished_at": stamp(), "summary": summary}
    _atomic_save_json(_shard_manifest_path(out_dir, shard_index), manifest)


def consolidate_manifest(out_dir):
    """Merge every manifest.shard*.json into a single manifest.json (atomic).
    Idempotent and race-free: only reads the single-writer shard files."""
    completed = {}
    shard_files = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("manifest.shard") and fn.endswith(".json"):
            shard_files.append(fn)
            d = load_json(os.path.join(out_dir, fn))
            completed.update(d.get("completed", {}))
    manifest = {
        "consolidated_at": stamp(),
        "source_shard_files": shard_files,
        "note": ("Assembled by merging single-writer per-shard manifests "
                 "(FIX-5). Per-cell checkpoints remain the authoritative source "
                 "of truth; see rebuild-from-checkpoints in validate_consistency.py."),
        "completed": completed,
    }
    _atomic_save_json(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest


# --------------------------------------------------------------------------- #
# Environment / seed capture (mirrors MAIN-001 raw/environment.txt + seeds).
# --------------------------------------------------------------------------- #
def _capture_environment(out_dir, seeds):
    env_path = os.path.join(out_dir, "environment.txt")
    if not os.path.exists(env_path):
        lines = [f"Python {platform.python_version()}"]
        try:
            pf = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=60)
            lines.append(pf.stdout.strip())
        except Exception as e:  # pragma: no cover - best-effort capture
            lines.append(f"(pip freeze unavailable: {e})")
        lines.append(platform.platform())
        save_json_text(env_path, "\n".join(lines) + "\n")
    seeds_path = os.path.join(out_dir, "seeds_verified.json")
    if not os.path.exists(seeds_path):
        save_json(seeds_path, list(seeds))


def save_json_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def run_ablation(out_dir, budget_name="pilot", shard_index=0, num_shards=1):
    budget = BUDGETS[budget_name]
    ckpt_dir = os.path.join(out_dir, "hpo_checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    seeds = list(range(budget["n_seeds"]))
    hpo_kwargs = dict(K_out=budget["K_out"], K_in=budget["K_in"], n_trials=budget["n_trials"])

    manifest = _load_shard_manifest(out_dir, shard_index)
    if shard_index == 0:
        _capture_environment(out_dir, seeds)

    all_units = _work_units()
    my_units = [(env, c) for i, (env, c) in enumerate(all_units) if i % num_shards == shard_index]
    print(f"[shard {shard_index}/{num_shards}] assigned {len(my_units)} of {len(all_units)} cells "
          f"(budget={budget_name}, seeds={len(seeds)}, {hpo_kwargs})")

    for env_name, c in my_units:
        cfg = cfg_for(env_name, c, **budget.get("cfg_overrides", {}))
        base_tag = f"{env_name}__c{c}"

        # ---- 5 re-tuned branches per cell -------------------------------- #
        for branch in BRANCHES:
            key = f"{base_tag}__{branch}"
            if key in manifest["completed"]:
                print(f"[skip] {key} (already done)")
                continue
            print(f"[run]  {key}")
            try:
                cls_or_factory, space = _branch_spec(branch)
                res = nested_cv_hpo(cls_or_factory, space, cfg, seeds,
                                    tag=key, checkpoint_dir=ckpt_dir, **hpo_kwargs)
                summary = {"mean_test_regret": sum(r["test_mean_regret"] for r in res) / len(res)}
                _mark_done(out_dir, manifest, shard_index, key, summary)
            except Exception as e:
                print(f"[FAIL] {key}: {e}")
                traceback.print_exc()
                raise

        # ---- per-cell DR3 report + containment (recomputed from checkpoints) #
        report_key = f"{base_tag}__ablation_report"
        if report_key not in manifest["completed"]:
            ablation_results = {}
            for branch in BRANCHES:
                ck = os.path.join(ckpt_dir, f"{base_tag}__{branch}.json")
                ablation_results[branch] = load_json(ck)["results"]
            dr3 = ablation_report(ablation_results)
            cont = containment_report(ablation_results["full"])
            save_json(os.path.join(out_dir, f"{base_tag}__ablation_report.json"), dr3)
            save_json(os.path.join(out_dir, f"{base_tag}__containment_report.json"), cont)
            attributed = [n for n, v in dr3.items() if v["significant_degradation"]]
            _mark_done(out_dir, manifest, shard_index, report_key,
                       {"attributed_components": attributed,
                        "prop_collapsed_to_zero": cont["prop_collapsed_to_zero"],
                        "stab_collapsed_to_zero": cont["stab_collapsed_to_zero"]})
            print(f"[done] {report_key}: attributed={attributed}")

    print(f"[shard {shard_index}/{num_shards}] assigned cells complete.")
    if shard_index == 0:
        consolidate_manifest(out_dir)
        print("[shard 0] consolidated manifest.json written.")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", choices=list(BUDGETS.keys()), default="pilot")
    ap.add_argument("--out", default="/mnt/user-data/outputs/rova_ablation")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--consolidate", action="store_true",
                    help="Only merge existing per-shard manifests into manifest.json and exit.")
    args = ap.parse_args()
    if args.consolidate:
        m = consolidate_manifest(args.out)
        print(f"consolidated {len(m['completed'])} entries into manifest.json")
    else:
        run_ablation(args.out, args.budget, shard_index=args.shard_index, num_shards=args.num_shards)
