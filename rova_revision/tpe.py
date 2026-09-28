"""Tree-structured Parzen Estimator re-run of the tuning protocol (plan R13).

The first ``tpe_startup`` candidates of every outer fold are the frozen
random-search candidates (drawn from ``default_rng(0)`` exactly as
``nested_cv_hpo`` draws them) and are evaluated on the frozen validation
tags, so the tuning curve at ``tpe_startup`` must reproduce the archived
value. Later candidates are proposed by TPE (Bergstra et al., 2011).
"""
import json
import os

import numpy as np
from scipy.stats import norm

from dtcoa_val.experiments.nested_hpo import _eval_config
from dtcoa_val.experiments.search_space import sample_config
from dtcoa_val.utils import make_rng, trial_seed

from .archive import inner_folds, outer_folds

GAMMA = 0.25
N_EI_CANDIDATES = 24


class _Continuous:
    """Truncated Gaussian Parzen estimator with a uniform-width prior component."""

    def __init__(self, obs, low, high):
        obs = np.asarray(obs, float)
        width = high - low
        mus = np.concatenate([obs, [(low + high) / 2.0]])
        order = np.argsort(mus, kind="stable")
        sm = mus[order]
        left = np.diff(np.concatenate([[low], sm]))
        right = np.diff(np.concatenate([sm, [high]]))
        sig = np.clip(np.maximum(left, right), width / min(100.0, 1.0 + len(mus)), width)
        sigmas = np.empty_like(sig)
        sigmas[order] = sig
        sigmas[-1] = width
        self.mus, self.sig = mus, sigmas
        self.low, self.high = low, high
        self.mass = norm.cdf((high - mus) / sigmas) - norm.cdf((low - mus) / sigmas)
        self.w = np.full(len(mus), 1.0 / len(mus))

    def sample(self, rng):
        j = int(rng.choice(len(self.mus), p=self.w))
        for _ in range(100):
            x = rng.normal(self.mus[j], self.sig[j])
            if self.low <= x <= self.high:
                return float(x)
        return float(np.clip(rng.normal(self.mus[j], self.sig[j]), self.low, self.high))

    def logpdf(self, x):
        dens = self.w * norm.pdf((x - self.mus) / self.sig) / self.sig / np.maximum(self.mass, 1e-300)
        return float(np.log(max(dens.sum(), 1e-300)))


class _Categorical:
    def __init__(self, obs, categories):
        self.cats = list(categories)
        counts = np.array([sum(1 for o in obs if o == c) for c in self.cats], float) + 1.0
        self.p = counts / counts.sum()

    def sample(self, rng):
        return self.cats[int(rng.choice(len(self.cats), p=self.p))]

    def logpdf(self, x):
        return float(np.log(self.p[self.cats.index(x)]))


def _param_models(spec, values):
    """Parzen models for one parameter; returns (sample_fn, logpdf_fn)."""
    kind = spec[0]
    if kind == "uniform":
        m = _Continuous(values, spec[1], spec[2])
        return (lambda rng: m.sample(rng)), m.logpdf
    if kind == "loguniform":
        lo, hi = np.log(spec[1]), np.log(spec[2])
        m = _Continuous(np.log(values), lo, hi)
        return (lambda rng: float(np.exp(m.sample(rng)))), (lambda x: m.logpdf(np.log(x)))
    if kind == "loguniform_complement":
        lo, hi = spec[1], spec[2]
        u = np.log(1.0 - np.clip(np.asarray(values, float), lo, hi))
        m = _Continuous(u, np.log(1.0 - hi), np.log(1.0 - lo))
        return (lambda rng: float(np.clip(1.0 - np.exp(m.sample(rng)), lo, hi))), \
               (lambda x: m.logpdf(np.log(1.0 - np.clip(x, lo, hi))))
    if kind == "loguniform_with_zero":
        lo, hi = np.log(spec[1]), np.log(spec[2])
        zero = _Categorical([v == 0.0 for v in values], [True, False])
        nz = [np.log(v) for v in values if v != 0.0]
        m = _Continuous(nz, lo, hi)

        def sample(rng):
            return 0.0 if zero.sample(rng) else float(np.exp(m.sample(rng)))

        def logpdf(x):
            return zero.logpdf(True) if x == 0.0 else zero.logpdf(False) + m.logpdf(np.log(x))
        return sample, logpdf
    if kind == "int_uniform":
        cats = list(range(spec[1], spec[2] + 1))
        m = _Categorical([int(v) for v in values], cats)
        return (lambda rng: int(m.sample(rng))), (lambda x: m.logpdf(int(x)))
    if kind == "band_vector":
        n, lo, hi = spec[1], spec[2], spec[3]
        arr = np.sort(np.asarray(values, float), axis=1)
        ms = [_Continuous(arr[:, j], lo, hi) for j in range(n)]

        def sample(rng):
            return np.sort(np.array([ms[j].sample(rng) for j in range(n)]))

        def logpdf(x):
            x = np.sort(np.asarray(x, float))
            return sum(ms[j].logpdf(x[j]) for j in range(n))
        return sample, logpdf
    raise ValueError(f"unsupported search-space kind for TPE: {kind}")


class TPESampler:
    def __init__(self, space, rng, gamma=GAMMA, n_candidates=N_EI_CANDIDATES):
        self.space, self.rng = space, rng
        self.gamma, self.n_candidates = gamma, n_candidates

    def propose(self, history):
        scores = np.array([np.inf if not np.isfinite(s) else s for _, s in history])
        order = np.argsort(scores, kind="stable")
        n_good = max(1, int(np.ceil(self.gamma * len(history))))
        good = [history[i][0] for i in order[:n_good]]
        bad = [history[i][0] for i in order[n_good:]] or good
        models = {}
        for name, spec in self.space.items():
            models[name] = (_param_models(spec, [c[name] for c in good]),
                            _param_models(spec, [c[name] for c in bad]))
        best, best_score = None, -np.inf
        for _ in range(self.n_candidates):
            cand = {name: models[name][0][0](self.rng) for name in self.space}
            score = sum(models[n][0][1](cand[n]) - models[n][1][1](cand[n]) for n in self.space)
            if score > best_score:
                best, best_score = cand, score
        return best


def nested_cv_tpe(policy_class, space, cfg, seeds, tag, profile, checkpoint_path=None):
    master_rng = np.random.default_rng(0)
    folds = outer_folds(seeds, profile.K_out)
    frozen = [[sample_config(space, master_rng) for _ in range(profile.n_trials)]
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
        sampler = TPESampler(space, make_rng(trial_seed(out_i, 0, f"{tag}__tpe")))
        history, curve, best = [], {}, np.inf
        for i in range(profile.n_trials):
            conf = frozen[out_i][i] if i < profile.tpe_startup else sampler.propose(history)
            vals = [_eval_config(policy_class, conf, cfg, ival,
                                 tag=f"{tag}_out{out_i}_hpo{i}_infold{j}")["mean_regret"]
                    for j, (_it, ival) in enumerate(ifolds)]
            score = float(np.mean(vals))
            history.append((conf, score))
            best = min(best, score)
            if (i + 1) in profile.sub_budgets or (i + 1) == profile.tpe_startup:
                curve[i + 1] = best
        scores = np.array([s for _, s in history])
        best_idx = int(np.nanargmin(scores))
        best_conf = history[best_idx][0]
        test = _eval_config(policy_class, best_conf, cfg, outer_test, tag=f"{tag}_out{out_i}_test")
        results.append({
            "outer_fold": out_i,
            "best_config": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best_conf.items()},
            "best_index": best_idx,
            "best_inner_score": float(scores[best_idx]),
            "inner_scores": scores.tolist(),
            "tuning_curve": curve,
            "test_mean_regret": test["mean_regret"],
            "test_cvar95_regret": test["cvar95_regret"],
            "test_regrets": test["regrets"],
        })
        results.sort(key=lambda r: r["outer_fold"])
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            tmp = checkpoint_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"tag": tag, "results": results}, f)
            os.replace(tmp, checkpoint_path)
    return sorted(results, key=lambda r: r["outer_fold"])
