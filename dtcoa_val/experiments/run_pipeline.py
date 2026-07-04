"""Top-level experiment pipeline. Single entry point that runs the complete
frozen-scope essential experimental suite:

  for each environment (3, frozen in environments.py)
    for each cost ratio (7, frozen in environments.py)
      - StaticThreshold: nested-CV-HPO
      - EWMA:            nested-CV-HPO
      - OracleLite (clairvoyant): nested-CV-HPO
      - ROVA:            nested-CV-HPO + 4 one-at-a-time re-tuned ablations
                          + containment report
    - distribution-shift: (only once, for the designated train->shift pair)
    - DP-oracle validation: (only for environments in single-agent-uncoupled
      scope, i.e. the "unfavorable" environment)

Every unit of work is independently checkpointed (via nested_cv_hpo's
checkpoint_dir) AND tracked in a top-level manifest, so re-running this
script after an interruption skips everything already completed rather than
restarting from zero. This is what "reproducible pipeline" and "complete
logging and resumability" mean concretely in the frozen scope.

Usage:
    python -m dtcoa_val.experiments.run_pipeline --budget pilot
    python -m dtcoa_val.experiments.run_pipeline --budget full
"""
import argparse
import os
import time
import traceback

from dtcoa_val.utils import save_json, load_json
from dtcoa_val.experiments.environments import ENVIRONMENTS, COST_RATIO_GRID, cfg_for
from dtcoa_val.experiments.search_space import (
    ROVA_SEARCH_SPACE, STATIC_THRESHOLD_SEARCH_SPACE, EWMA_SEARCH_SPACE,
)
from dtcoa_val.experiments.nested_hpo import nested_cv_hpo
from dtcoa_val.experiments.ablation_study import containment_report
from dtcoa_val.experiments.distribution_shift import evaluate_under_shift, shift_report
from dtcoa_val.dtcoa_tunable import DTCOATunable
from dtcoa_val.baselines.baselines import StaticThreshold, EWMA
from dtcoa_val.oracles.dp_oracle import verify_sequential_equals_myopic

BUDGETS = {
    # Reduced budget for fast end-to-end verification (Milestone 9 pilot).
    # cfg_overrides shrinks horizon/n_agents too -- this is purely a
    # plumbing/connectivity check, NOT a scientific run, so it does not need
    # to respect the frozen environment definitions' actual scale.
    "pilot": dict(K_out=2, K_in=2, n_trials=5, n_seeds=10,
                  cfg_overrides=dict(horizon=25, n_agents=3, graph_density=0.2, coupling=0.2)),
    # Frozen PI-scoped budget (see PROJECT_SCOPE_FREEZE.md). No overrides --
    # uses the frozen environment definitions exactly as specified.
    "full": dict(K_out=5, K_in=3, n_trials=80, n_seeds=60, cfg_overrides={}),
}

# Distribution-shift pair (frozen minimal scope: exactly one pair; reserved
# for revision as of manifest rev. 4, code kept correct regardless).
SHIFT_TRAIN_ENV = "favorable_hmm_coupled"
SHIFT_EVAL_ENV = "unfavorable_oscillation_uncoupled"
SHIFT_COST_RATIO = 3.5  # closest grid point to the old 3.33 after the rev-5 grid re-derivation

# DP-oracle validation applies only to environments actually in single-agent,
# uncoupled scope.
DP_VALID_ENVS = ["unfavorable_oscillation_uncoupled"]


def _manifest_path(out_dir):
    return os.path.join(out_dir, "manifest.json")


def _load_manifest(out_dir):
    p = _manifest_path(out_dir)
    if os.path.exists(p):
        return load_json(p)
    return {"completed": {}, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def _mark_done(out_dir, manifest, key, summary):
    manifest["completed"][key] = {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": summary}
    save_json(_manifest_path(out_dir), manifest)


def _work_units():
    """Every (env, cost_ratio) pair, in a fixed, deterministic order -- the
    unit of sharding for parallel execution. Each unit internally runs 3
    baselines + ROVA(full+4 ablations) = 8 nested_cv_hpo campaigns, and is
    completely independent of every other unit (separate checkpoint files,
    separate manifest keys), so shards never contend or double-write."""
    return [(env, c) for env in ENVIRONMENTS for c in COST_RATIO_GRID]


def run_pipeline(out_dir, budget_name="pilot", shard_index=0, num_shards=1):
    """If num_shards > 1, only the (env, cost_ratio) work units whose index
    mod num_shards == shard_index are processed by this process. Run
    num_shards processes (one per shard_index, e.g. via a job array or
    separate terminals/machines) against the SAME --out directory to
    parallelize the full campaign; the manifest and per-unit checkpoints
    make this safe since units never overlap across shards. DP-oracle
    validation and the distribution-shift step are cheap and always run by
    shard 0 only, after its own units complete, to avoid duplication.
    """
    budget = BUDGETS[budget_name]
    ckpt_dir = os.path.join(out_dir, "hpo_checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    manifest = _load_manifest(out_dir)
    seeds = list(range(budget["n_seeds"]))
    hpo_kwargs = dict(K_out=budget["K_out"], K_in=budget["K_in"], n_trials=budget["n_trials"])

    methods = {
        "static": (StaticThreshold, STATIC_THRESHOLD_SEARCH_SPACE),
        "ewma": (EWMA, EWMA_SEARCH_SPACE),
        "rova": (DTCOATunable, ROVA_SEARCH_SPACE),
    }

    my_units = [(env, c) for i, (env, c) in enumerate(_work_units()) if i % num_shards == shard_index]
    print(f"[shard {shard_index}/{num_shards}] assigned {len(my_units)} of {len(_work_units())} work units")

    for env_name, c in my_units:
        cfg = cfg_for(env_name, c, **budget.get("cfg_overrides", {}))
        base_tag = f"{env_name}__c{c}"

        for method_name, (cls, space) in methods.items():
            key = f"{base_tag}__{method_name}"
            if key in manifest["completed"]:
                print(f"[skip] {key} (already done)")
                continue
            print(f"[run]  {key}")
            try:
                res = nested_cv_hpo(cls, space, cfg, seeds, tag=key, checkpoint_dir=ckpt_dir, **hpo_kwargs)
                summary = {"mean_test_regret": sum(r["test_mean_regret"] for r in res) / len(res)}
                _mark_done(out_dir, manifest, key, summary)
            except Exception as e:
                print(f"[FAIL] {key}: {e}")
                traceback.print_exc()
                raise

        # ROVA is now just another entry in `methods` (mechanism attribution /
        # ablations are future work per round-3 PI scoping, see
        # IMPLEMENTATION_STATUS.md). Containment check is FREE: it only reads
        # prop_scale/stab_scale out of ROVA's own already-tuned configs above.
        rova_key = f"{base_tag}__rova"
        if rova_key in manifest["completed"] and f"{base_tag}__containment" not in manifest["completed"]:
            rova_ckpt = os.path.join(ckpt_dir, f"{rova_key}.json")
            from dtcoa_val.utils import load_json as _lj2
            rova_results = _lj2(rova_ckpt)["results"]
            cont_rep = containment_report(rova_results)
            save_json(os.path.join(out_dir, f"{base_tag}__containment_report.json"), cont_rep)
            _mark_done(out_dir, manifest, f"{base_tag}__containment",
                       {"prop_collapsed_to_zero": cont_rep["prop_collapsed_to_zero"],
                        "stab_collapsed_to_zero": cont_rep["stab_collapsed_to_zero"]})

    print(f"[shard {shard_index}/{num_shards}] assigned units complete.")

    # DP-oracle validation and distribution-shift are cheap, environment-level
    # (not per-unit) steps -- run exactly once, by shard 0, after checking
    # its prerequisite units are actually done (in a multi-shard run these
    # may belong to a different shard, so wait/skip rather than assume).
    if shard_index != 0:
        return manifest

    for env_name in DP_VALID_ENVS:
        key = f"{env_name}__dp_oracle_validation"
        if key in manifest["completed"]:
            print(f"[skip] {key} (already done)")
            continue
        print(f"[run]  {key}")
        dp_overrides = dict(budget.get("cfg_overrides", {}))
        dp_overrides.pop("n_agents", None)  # DP oracle requires n_agents=1 exactly
        dp_overrides.pop("coupling", None)  # and coupling=0 exactly
        dp_overrides.pop("graph_density", None)
        cfg = cfg_for(env_name, SHIFT_COST_RATIO, **dp_overrides)
        res = verify_sequential_equals_myopic(cfg, seeds, n_perturbations_per_seed=8)
        save_json(os.path.join(out_dir, f"{env_name}__dp_oracle_validation.json"), res)
        _mark_done(out_dir, manifest, key, {"certified_equivalent": res["certified_equivalent"]})

    # Distribution shift: reuse the already-completed tuning from the
    # training environment at the shift cost ratio, for ROVA AND at least
    # one baseline (EWMA) -- a shift delta only means something set against
    # a comparison point; evaluating ROVA alone was an implementation gap
    # (see IMPLEMENTATION_STATUS.md), fixed here.
    shift_targets = {
        "rova": (DTCOATunable, f"{SHIFT_TRAIN_ENV}__c{SHIFT_COST_RATIO}__rova.json"),
        "ewma": (EWMA, f"{SHIFT_TRAIN_ENV}__c{SHIFT_COST_RATIO}__ewma.json"),
    }
    for target_name, (cls, ckpt_name) in shift_targets.items():
        key = f"shift__{target_name}__{SHIFT_TRAIN_ENV}_to_{SHIFT_EVAL_ENV}"
        if key in manifest["completed"]:
            print(f"[skip] {key} (already done)")
            continue
        train_key = f"{SHIFT_TRAIN_ENV}__c{SHIFT_COST_RATIO}__{target_name}"
        if train_key not in manifest["completed"]:
            print(f"[wait] {key}: prerequisite '{train_key}' not yet completed by its shard.")
            continue
        print(f"[run]  {key}")
        from dtcoa_val.utils import load_json as _lj
        train_ckpt = os.path.join(ckpt_dir, ckpt_name)
        trained = _lj(train_ckpt)["results"]
        shift_cfg = cfg_for(SHIFT_EVAL_ENV, SHIFT_COST_RATIO, **budget.get("cfg_overrides", {}))
        shift_res = evaluate_under_shift(cls, trained, shift_cfg, seeds, tag=key)
        rep = shift_report(shift_res)
        save_json(os.path.join(out_dir, f"distribution_shift_report__{target_name}.json"),
                  {"per_fold": shift_res, "summary": rep})
        _mark_done(out_dir, manifest, key, rep)

    print("Pipeline complete (shard 0 finalization done).")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", choices=list(BUDGETS.keys()), default="pilot")
    ap.add_argument("--out", default="/mnt/user-data/outputs/rova_pipeline")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    run_pipeline(args.out, args.budget, shard_index=args.shard_index, num_shards=args.num_shards)
