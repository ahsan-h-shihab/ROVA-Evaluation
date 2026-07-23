#!/usr/bin/env python3
"""
replay_archived_runs.py
=======================

Replays the archived MAIN-001 and ABLATION-001 experiments using the ARCHIVED
tuned per-outer-fold configurations, and detects, in the real evaluated worlds,
every per-agent timestep where the trust state strictly decreases while
authorization strictly increases.

Nothing here modifies source. It imports the frozen simulator/runner/policy and
reconstructs each scored outer-test trial exactly, using the archived pieces:

  * environment cfg      : dtcoa_val.experiments.environments.cfg_for(env, c)
  * policy class         : dtcoa_val.dtcoa_tunable.DTCOATunable (MAIN 'rova' and
                           ABLATION 'full'); ablation variants wrap it with the
                           archived fixed_kwargs via make_ablation_factory
  * tuned config         : raw/hpo_checkpoints/<cell>__<branch>.json -> results[fold]['best_config']
  * outer-test seeds     : nested_hpo.kfold_split(range(60), 5, default_rng(0))[fold][1]
  * per-trial world seed : runner.run_trial(..., tag=f"{cell}_out{fold}_test")

SELF-VALIDATION (this is what makes the replay trustworthy):
  1. For every fold, the replayed mean test regret is compared to the archived
     'test_mean_regret'. If it does not match to < 1e-9, that fold is reported
     NOT REPRODUCIBLE and excluded from violation counts.
  2. At every agent-step the propagation/stability penalties are reconstructed
     from public post-step state and used to rebuild authorization; if the
     rebuilt authorization ever disagrees with the policy's own output the run
     is flagged (recon_mismatch > 0 => NOT REPRODUCIBLE).

All reported numbers are computed here; none are hardcoded.

Usage:
    PYTHONPATH=<repo>/code python3 reproduce/replay_archived_runs.py [options]
Options:
    --experiment {main,ablation,both}   (default: both)
    --branches full,no_decay,...        ablation branches (default: all 5)
    --max-seeds N                       cap test seeds per fold (default: all)
    --print-events N                    print the first N violation records in full
    --out PATH                          summary JSON (default reproduce/out_archived.json)
"""
import os, sys, json, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CODE = os.path.join(REPO, "code")
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from dtcoa_val.experiments.nested_hpo import kfold_split
from dtcoa_val.experiments.runner import run_trial
from dtcoa_val.experiments.environments import cfg_for
from dtcoa_val.experiments.search_space import ROVA_ABLATIONS
from dtcoa_val.dtcoa_tunable import DTCOATunable, band_level

N_SEEDS = 60           # frozen protocol: seeds 0..59 (seeds_verified.json)
K_OUT = 5              # frozen protocol
SEED0 = 0              # nested_cv_hpo default; run_pipeline/run_ablation do not override

MAIN_ROOT = os.path.join(REPO, "experiments", "MAIN-001_core_comparison")
ABL_ROOT = os.path.join(REPO, "experiments", "ABLATION-001_mechanism_attribution")

# fixed structural kwargs per ablation branch (archived spec)
BRANCH_FIXED = {"full": {}, **{k: v["fixed_kwargs"] for k, v in ROVA_ABLATIONS.items()}}


class Recorder:
    """Non-invasive per-step logger + penalty reconstruction + self-check.
    Wraps a DTCOATunable built with (fixed_kwargs + tuned_config)."""
    def __init__(self, fixed_kwargs, tuned_config):
        self.p = DTCOATunable(**{**fixed_kwargs, **tuned_config})
        self.recon_mismatch = 0

    def reset(self, N):
        self.p.reset(N); self.N = N; self._t = 0
        self.T_state = [[] for _ in range(N)]   # post-update trust per agent
        self.A = [[] for _ in range(N)]
        self.ESC = [[] for _ in range(N)]
        self.E = [[] for _ in range(N)]
        self.PP = [[] for _ in range(N)]        # propagation penalty (system, repeated per agent)
        self.SP = [[] for _ in range(N)]        # stability penalty per agent

    def step(self, O, truth=None):
        p = self.p
        A, ESC = p.step(O, truth=truth)
        Ta = np.asarray(p.T).copy()
        E = np.array([p.Ehist[i][-1] for i in range(self.N)])
        nfail = int((E < p.e_min).sum())
        init = getattr(p, "_initial_fail", None)
        FAF = (nfail / init) if init else 1.0
        prop_pen = (p.prop_scale * max(0.0, FAF - 1.0)) if p.use_prop else 0.0
        for i in range(self.N):
            w = p.Ehist[i][-p.window:]
            S = 1 - min(1.0, (np.std(w) / 0.25 if len(w) > 1 else 0.0))
            s_pen = (p.stab_scale * max(0.0, p.s_min - S)) if p.use_stab else 0.0
            L = band_level(Ta[i] * (1 - p.Rc), p.bands)
            a_recon = int(np.clip(round(L - prop_pen - s_pen), 0, 5))
            if a_recon != int(A[i]):
                self.recon_mismatch += 1
            self.T_state[i].append(float(Ta[i]))
            self.A[i].append(int(A[i])); self.ESC[i].append(bool(ESC[i]))
            self.E[i].append(float(E[i]))
            self.PP[i].append(float(prop_pen)); self.SP[i].append(float(s_pen))
        self._t += 1
        return A, ESC

    def violations(self, meta):
        out = []
        for i in range(self.N):
            Ts, As = self.T_state[i], self.A[i]
            for t in range(1, len(Ts)):
                if Ts[t] < Ts[t - 1] - 1e-12 and As[t] > As[t - 1]:
                    rec = dict(meta)
                    rec.update(dict(
                        agent_id=i, timestep=t,
                        trust_before=Ts[t - 1], trust_after=Ts[t],
                        authorization_before=As[t - 1], authorization_after=As[t],
                        evidence=self.E[i][t],
                        propagation_penalty=self.PP[i][t],
                        stability_penalty=self.SP[i][t],
                        propagation_penalty_before=self.PP[i][t - 1],
                        stability_penalty_before=self.SP[i][t - 1],
                        context_risk=float(self.p.Rc),
                        esc_flag=self.ESC[i][t],
                        oversight_removed=(As[t - 1] == 0 and As[t] > 0),
                    ))
                    out.append(rec)
        return out


def parse_cell_tag(celltag):
    # "<env>__c<cost>__<branch>" -> (env, cost, branch)
    env, cpart, branch = celltag.rsplit("__", 2)
    return env, float(cpart[1:]), branch


def replay_checkpoint(ckpt_path, experiment, fixed_kwargs, outer_folds, max_seeds, tol=1e-9):
    d = json.load(open(ckpt_path))
    celltag = d["tag"]
    env, cost, branch = parse_cell_tag(celltag)
    cfg = cfg_for(env, cost)
    fold_results = []
    for fold in d["results"]:
        i = fold["outer_fold"]
        bc = fold["best_config"]
        test_seeds = list(outer_folds[i][1])
        if max_seeds:
            test_seeds = test_seeds[:max_seeds]
        regs = []
        events = []
        recon_mismatch = 0
        for s in test_seeds:
            rec = Recorder(fixed_kwargs, bc)
            out = run_trial(cfg, rec, seed=s, tag=f"{celltag}_out{i}_test")
            regs.append(out["regret"])
            recon_mismatch += rec.recon_mismatch
            meta = dict(experiment=experiment, environment=env, cost_ratio=cost,
                        branch=branch, fold=i, seed=s)
            events.extend(rec.violations(meta))
        replay_mean = float(np.mean(regs))
        archived_mean = float(fold["test_mean_regret"]) if max_seeds is None else None
        regret_match = (archived_mean is not None) and (abs(replay_mean - archived_mean) < tol)
        fold_results.append(dict(
            experiment=experiment,
            environment=env, cost_ratio=cost, branch=branch, fold=i,
            n_test_seeds=len(test_seeds), test_seeds=test_seeds,
            replay_mean_regret=replay_mean, archived_mean_regret=archived_mean,
            regret_match=regret_match, recon_mismatch=recon_mismatch,
            reproducible=(recon_mismatch == 0 and (max_seeds is not None or regret_match)),
            n_violations=len(events), events=events,
        ))
    return fold_results


def gather_checkpoints(root, branches):
    out = []
    for ck in sorted(glob.glob(os.path.join(root, "raw", "hpo_checkpoints", "*.json"))):
        base = os.path.basename(ck)[:-5]
        _, _, branch = parse_cell_tag(base)
        if branch in branches:
            out.append(ck)
    return out


def _counts(repro, not_repro):
    events = [e for f in repro for e in f["events"]]
    def uniq(key_fn):
        return sorted({key_fn(e) for e in events})
    cells_with_viol = sorted({(e["environment"], e["cost_ratio"]) for e in events})
    configs_with_viol = sorted({(e["environment"], e["cost_ratio"], e["branch"], e["fold"])
                                for e in events})
    return dict(
        n_folds_total=len(repro) + len(not_repro),
        n_folds_reproducible=len(repro),
        n_folds_not_reproducible=len(not_repro),
        n_violating_agent_steps=len(events),
        n_oversight_removed_agent_steps=sum(1 for e in events if e["oversight_removed"]),
        n_violating_configs=len(configs_with_viol),
        violating_configs=configs_with_viol,
        n_affected_cells=len(cells_with_viol),
        affected_cells=cells_with_viol,
        affected_environments=uniq(lambda e: e["environment"]),
        affected_cost_ratios=uniq(lambda e: e["cost_ratio"]),
        affected_folds=uniq(lambda e: e["fold"]),
        affected_seeds=uniq(lambda e: e["seed"]),
        affected_branches=uniq(lambda e: e["branch"]),
    )


def summarize(all_folds):
    """Aggregate every previous-audit-style count from the fold records. No
    hardcoded numbers. Only counts folds that self-validated as reproducible."""
    repro = [f for f in all_folds if f["reproducible"]]
    not_repro = [f for f in all_folds if not f["reproducible"]]
    events = [e for f in repro for e in f["events"]]
    def uniq(key_fn):
        return sorted({key_fn(e) for e in events})
    cells_with_viol = sorted({(e["environment"], e["cost_ratio"]) for e in events})
    configs_with_viol = sorted({(e["environment"], e["cost_ratio"], e["branch"], e["fold"])
                                for e in events})
    return dict(
        n_folds_total=len(all_folds),
        n_folds_reproducible=len(repro),
        n_folds_not_reproducible=len(not_repro),
        not_reproducible=[{k: f[k] for k in ("environment", "cost_ratio", "branch", "fold",
                                             "regret_match", "recon_mismatch")} for f in not_repro],
        n_violating_agent_steps=len(events),
        n_oversight_removed_agent_steps=sum(1 for e in events if e["oversight_removed"]),
        n_violating_configs=len(configs_with_viol),      # (env, cost, branch, fold) tuples
        violating_configs=configs_with_viol,
        n_affected_cells=len(cells_with_viol),           # (env, cost) tuples
        affected_cells=cells_with_viol,
        affected_environments=uniq(lambda e: e["environment"]),
        affected_cost_ratios=uniq(lambda e: e["cost_ratio"]),
        affected_folds=uniq(lambda e: e["fold"]),
        affected_seeds=uniq(lambda e: e["seed"]),
        affected_branches=uniq(lambda e: e["branch"]),
        by_experiment={
            exp: _counts([f for f in repro if _fold_exp(f) == exp],
                         [f for f in not_repro if _fold_exp(f) == exp])
            for exp in sorted({_fold_exp(f) for f in all_folds})
        },
    )


def _fold_exp(f):
    # fold records carry experiment via their events; recover from meta-free field
    return f.get("experiment", "?")


def tuned_penalty_stats():
    """Regenerate the 'fraction of tuned folds with substantial penalties'
    numbers the previous audit reported (MAIN-001 rova checkpoints)."""
    ps, ss = [], []
    for ck in glob.glob(os.path.join(MAIN_ROOT, "raw", "hpo_checkpoints", "*__rova.json")):
        d = json.load(open(ck))
        for fold in d["results"]:
            c = fold["best_config"]
            ps.append(c["prop_scale"]); ss.append(c["stab_scale"])
    ps, ss = np.array(ps), np.array(ss)
    return dict(
        n_tuned_folds=int(len(ps)),
        prop_scale=dict(min=float(ps.min()), median=float(np.median(ps)), max=float(ps.max()),
                        frac_gt_0p1=float((ps > 0.1).mean()), n_gt_0p5=int((ps > 0.5).sum())),
        stab_scale=dict(min=float(ss.min()), median=float(np.median(ss)), max=float(ss.max()),
                        frac_gt_0p1=float((ss > 0.1).mean()), n_gt_0p5=int((ss > 0.5).sum())),
        n_both_gt_0p1=int(((ps > 0.1) & (ss > 0.1)).sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["main", "ablation", "both"], default="both")
    ap.add_argument("--branches", default="full,no_memory,no_decay,no_propagation,no_stability")
    ap.add_argument("--max-seeds", type=int, default=None)
    ap.add_argument("--print-events", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "out_archived.json"))
    args = ap.parse_args()

    outer_folds = kfold_split(list(range(N_SEEDS)), K_OUT, np.random.default_rng(SEED0))
    abl_branches = set(args.branches.split(","))

    all_folds = []
    if args.experiment in ("main", "both"):
        for ck in gather_checkpoints(MAIN_ROOT, branches={"rova"}):
            all_folds += replay_checkpoint(ck, "MAIN-001", {}, outer_folds, args.max_seeds)
            print(f"[main] replayed {os.path.basename(ck)}")
    if args.experiment in ("ablation", "both"):
        for ck in gather_checkpoints(ABL_ROOT, branches=abl_branches):
            _, _, branch = parse_cell_tag(os.path.basename(ck)[:-5])
            fixed = BRANCH_FIXED[branch]
            all_folds += replay_checkpoint(ck, "ABLATION-001", fixed, outer_folds, args.max_seeds)
            print(f"[ablation] replayed {os.path.basename(ck)}")

    summary = summarize(all_folds)
    summary["folds"] = all_folds          # full per-fold records incl. events
    summary["tuned_penalty_stats_MAIN_rova"] = tuned_penalty_stats()
    summary["params"] = dict(experiment=args.experiment, max_seeds=args.max_seeds,
                             branches=sorted(abl_branches), seed0=SEED0, k_out=K_OUT, n_seeds=N_SEEDS)

    # ---- console report ----
    print("\n" + "=" * 78)
    print("SELF-VALIDATION")
    print("=" * 78)
    print(f"folds replayed: {summary['n_folds_total']}  "
          f"reproducible: {summary['n_folds_reproducible']}  "
          f"NOT REPRODUCIBLE: {summary['n_folds_not_reproducible']}")
    for nr in summary["not_reproducible"]:
        print(f"   NOT REPRODUCIBLE: {nr}")

    print("\n" + "=" * 78)
    print("ARCHIVED-RUN VIOLATIONS (trust strictly down & authorization strictly up)")
    print("=" * 78)
    print(f"violating agent-steps          : {summary['n_violating_agent_steps']}")
    print(f"  of which oversight removed    : {summary['n_oversight_removed_agent_steps']}")
    print(f"violating (env,cost,branch,fold): {summary['n_violating_configs']}")
    print(f"affected cells (env,cost)       : {summary['n_affected_cells']} -> {summary['affected_cells']}")
    print(f"affected environments           : {summary['affected_environments']}")
    print(f"affected cost ratios            : {summary['affected_cost_ratios']}")
    print(f"affected folds                  : {summary['affected_folds']}")
    print(f"affected seeds                  : {summary['affected_seeds']}")
    print(f"affected branches               : {summary['affected_branches']}")
    for exp, c in summary["by_experiment"].items():
        print(f"  [{exp}] folds={c['n_folds_reproducible']}/{c['n_folds_total']} repro, "
              f"agent_steps={c['n_violating_agent_steps']}, oversight_removed={c['n_oversight_removed_agent_steps']}, "
              f"configs={c['n_violating_configs']}, cells={c['n_affected_cells']}")

    events = [e for f in all_folds if f["reproducible"] for e in f["events"]]
    print(f"\nfirst {min(args.print_events, len(events))} violation record(s) in full:")
    for e in events[:args.print_events]:
        print("   " + json.dumps(e))

    ps = summary["tuned_penalty_stats_MAIN_rova"]
    print("\n" + "=" * 78)
    print("TUNED PENALTY MAGNITUDES (MAIN-001 rova, regenerated)")
    print("=" * 78)
    print(f"  {ps['n_tuned_folds']} tuned folds; "
          f"prop_scale frac>0.1={ps['prop_scale']['frac_gt_0p1']:.2f} max={ps['prop_scale']['max']:.3f}; "
          f"stab_scale frac>0.1={ps['stab_scale']['frac_gt_0p1']:.2f} max={ps['stab_scale']['max']:.3f}; "
          f"both>0.1={ps['n_both_gt_0p1']}")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
