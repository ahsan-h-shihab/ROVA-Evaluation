"""Nested cross-validation for trainable policies (plan R3, R4).

Mirrors ``dtcoa_val.experiments.nested_hpo.nested_cv_hpo`` exactly: the same
outer and inner splits, the same 80 candidates per outer fold drawn by the
frozen ``sample_config`` from ``default_rng(0)`` in fold order, the same
validation and test tags, the same selection rule and tuning-curve
sub-budgets. The only addition is a fitting step on training seeds, whose
worlds carry their own tags (``..._infold{j}_train``, ``..._final_train``).

All candidates are drawn before the first fold is processed, so a resumed run
proposes exactly the candidates an uninterrupted run would.
"""
import json
import os

import numpy as np

from dtcoa_val.experiments.nested_hpo import _eval_config, _cvar
from dtcoa_val.experiments.runner import run_trial
from dtcoa_val.experiments.search_space import sample_config
from dtcoa_val.utils import make_rng, trial_seed

from .archive import inner_folds, outer_folds
from .policies import (LearnedCSPolicy, QLearnPolicy, QTrainer, RandomLevels,
                       RecordingPolicy, featurize, fit_cpwlr)

LEARNED_SPACE = {
    "beta": ("loguniform_complement", 1e-3, 0.999),
    "degree": ("int_uniform", 1, 2),
    "l2": ("loguniform", 1e-6, 1.0),
    "onpolicy_rounds": ("int_uniform", 0, 1),
}

QLEARN_SPACE = {
    "n_bins": ("int_uniform", 3, 10),
    "beta": ("loguniform_complement", 1e-3, 0.999),
    "lr": ("loguniform", 1e-3, 0.5),
    "gamma": ("uniform", 0.0, 0.95),
    "eps0": ("uniform", 0.05, 0.5),
    "epochs": ("int_uniform", 1, 3),
}

EPS_MIN = 0.01


def _hu(cfg):
    return float(cfg.get("H", 1.0)), float(cfg.get("U", 0.3))


def record_rollout(cfg, inner_policy, seed, tag):
    rec = RecordingPolicy(inner_policy)
    run_trial(cfg, rec, seed=seed, tag=tag)
    return rec.trajectory()


# --------------------------------------------------------------------------- #
# Fitters: (config, cfg, train_seeds, train_tag, cache) -> model dict
# --------------------------------------------------------------------------- #
def fit_learned(conf, cfg, train_seeds, train_tag, cache):
    H, U = _hu(cfg)
    if train_tag not in cache:
        cache[train_tag] = [
            record_rollout(cfg, RandomLevels(make_rng(trial_seed(s, 1, train_tag))), s, train_tag)
            for s in train_seeds
        ]
    beta, degree, l2 = float(conf["beta"]), int(conf["degree"]), float(conf["l2"])
    X, D = featurize(cache[train_tag], beta, degree, H, U)
    mu, sd, theta, diag = fit_cpwlr(X, D, l2)
    diags = [diag]
    model = {"beta": beta, "degree": degree, "mu": mu.tolist(), "sd": sd.tolist(), "theta": theta.tolist()}
    if int(conf["onpolicy_rounds"]) >= 1:
        onpol = [record_rollout(cfg, LearnedCSPolicy(model), s, train_tag) for s in train_seeds]
        X1, D1 = featurize(onpol, beta, degree, H, U)
        mu, sd, theta, diag = fit_cpwlr(np.vstack([X, X1]), np.concatenate([D, D1]), l2)
        diags.append(diag)
        model = {"beta": beta, "degree": degree, "mu": mu.tolist(), "sd": sd.tolist(), "theta": theta.tolist()}
    model["fit_diagnostics"] = diags
    return model


def fit_qlearn(conf, cfg, train_seeds, train_tag, cache, seed_offset=0):
    """Tabular Q-learning; ``seed_offset`` selects an alternative exploration
    stream (0 = primary; used only for the training-seed sensitivity report)."""
    H, U = _hu(cfg)
    nb = int(conf["n_bins"])
    epochs = int(conf["epochs"])
    horizon = int(cfg["horizon"])
    model = {"n_bins": nb, "beta": float(conf["beta"]), "Q": np.zeros((nb * nb, 6))}
    pol = QLearnPolicy(model)
    pol.Q = model["Q"]
    total = epochs * len(train_seeds) * horizon
    counter = [0]
    for e in range(epochs):
        for s in train_seeds:
            rng = make_rng(trial_seed(s, 2 + e + seed_offset, train_tag))
            trainer = QTrainer(pol, conf["lr"], conf["gamma"], H, U, horizon,
                               conf["eps0"], EPS_MIN, total, counter, rng)
            run_trial(cfg, trainer, seed=s, tag=train_tag)
    return {"n_bins": nb, "beta": float(conf["beta"]), "Q": pol.Q.tolist(),
            "lr": float(conf["lr"]), "gamma": float(conf["gamma"])}


def qlearn_gate(model, cfg):
    """Plan R4 gate checks G1 (finite) and G2 (theoretical bounds)."""
    H, U = _hu(cfg)
    Q = np.asarray(model["Q"], float)
    lmax = max(H, U)
    lower = -lmax / (1.0 - float(model["gamma"])) - 1e-9
    finite = bool(np.all(np.isfinite(Q)))
    bounded = bool(finite and Q.min() >= lower and Q.max() <= 1e-12)
    return {"G1_finite": finite, "G2_bounded": bounded,
            "q_min": float(Q.min()) if finite else None, "q_max": float(Q.max()) if finite else None,
            "lower_bound": lower}


METHODS = {
    "learned": {"space": LEARNED_SPACE, "fit": fit_learned, "policy": LearnedCSPolicy},
    "qlearn": {"space": QLEARN_SPACE, "fit": fit_qlearn, "policy": QLearnPolicy},
}


def _factory(policy_cls, model):
    return lambda: policy_cls(model)


def nested_cv_trainable(method, cfg, seeds, tag, profile, checkpoint_path=None):
    spec = METHODS[method]
    space, fit, policy_cls = spec["space"], spec["fit"], spec["policy"]
    master_rng = np.random.default_rng(0)
    folds = outer_folds(seeds, profile.K_out)
    candidates = [[sample_config(space, master_rng) for _ in range(profile.n_trials)]
                  for _ in range(profile.K_out)]

    results = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            results = json.load(f)["results"]
    done = {r["outer_fold"] for r in results}

    for out_i, (outer_train, outer_test) in enumerate(folds):
        if out_i in done:
            continue
        ifolds = inner_folds(outer_train, profile.K_in, out_i)
        cache = {}
        scores = np.full(profile.n_trials, np.nan)
        curve, best = {}, np.inf
        for i, conf in enumerate(candidates[out_i]):
            vals = []
            for j, (itrain, ival) in enumerate(ifolds):
                model = fit(conf, cfg, list(itrain), f"{tag}_out{out_i}_infold{j}_train", cache)
                r = _eval_config(_factory(policy_cls, model), {}, cfg, ival,
                                 tag=f"{tag}_out{out_i}_hpo{i}_infold{j}")
                vals.append(r["mean_regret"])
            scores[i] = float(np.mean(vals))
            best = min(best, scores[i])
            if (i + 1) in profile.sub_budgets:
                curve[i + 1] = best
        best_idx = int(np.nanargmin(scores))
        best_conf = candidates[out_i][best_idx]
        final_tag = f"{tag}_out{out_i}_final_train"
        model = fit(best_conf, cfg, list(outer_train), final_tag, cache)
        test = _eval_config(_factory(policy_cls, model), {}, cfg, outer_test, tag=f"{tag}_out{out_i}_test")
        entry = {
            "outer_fold": out_i,
            "outer_train_n": len(outer_train),
            "outer_test_n": len(outer_test),
            "best_config": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best_conf.items()},
            "best_inner_score": float(scores[best_idx]),
            "inner_scores": scores.tolist(),
            "tuning_curve": curve,
            "test_mean_regret": test["mean_regret"],
            "test_cvar95_regret": test["cvar95_regret"],
            "test_regrets": test["regrets"],
            "model": model,
        }
        if method == "qlearn":
            gate = qlearn_gate(model, cfg)
            if out_i == 0:
                again = fit(best_conf, cfg, list(outer_train), final_tag, {})
                gate["G3_deterministic"] = bool(np.array_equal(np.asarray(again["Q"]), np.asarray(model["Q"])))
            entry["gate"] = gate
            sens = []
            for off in (100, 200):
                alt = fit_qlearn(best_conf, cfg, list(outer_train), final_tag, {}, seed_offset=off)
                ev = _eval_config(_factory(policy_cls, alt), {}, cfg, outer_test, tag=f"{tag}_out{out_i}_test")
                sens.append({"seed_offset": off, "test_mean_regret": ev["mean_regret"],
                             "test_regrets": ev["regrets"]})
            entry["training_seed_sensitivity"] = sens
        results.append(entry)
        results.sort(key=lambda r: r["outer_fold"])
        if checkpoint_path:
            _atomic_json(checkpoint_path, {"tag": tag, "method": method, "K_out": profile.K_out,
                                           "K_in": profile.K_in, "n_trials": profile.n_trials,
                                           "results": results})
    return sorted(results, key=lambda r: r["outer_fold"])


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
