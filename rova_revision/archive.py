"""Read-only access to the archived MAIN-001 artifacts shipped in this repository."""
import json
import os

import numpy as np

from dtcoa_val.experiments.nested_hpo import kfold_split

from .config import archived_tag

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN001 = os.path.join(REPO, "experiments", "MAIN-001_core_comparison")
CKPT_DIR = os.path.join(MAIN001, "raw", "hpo_checkpoints")
STATS_SUMMARY = os.path.join(MAIN001, "stats_summary.json")
ABL001 = os.path.join(REPO, "experiments", "ABLATION-001_mechanism_attribution")
ABL_CKPT_DIR = os.path.join(ABL001, "raw", "hpo_checkpoints")
ABL_REGIME_PUBLISHED = os.path.join(REPO, "abl001_analysis", "dr3_regime", "dr3_regime_inference.json")
VERDICTS_V2 = os.path.join(REPO, "verdicts", "verdicts_v2.json")


def load_archived(env, c, method):
    """Per-outer-fold results of an archived MAIN-001 tuning unit, fold order."""
    path = os.path.join(CKPT_DIR, archived_tag(env, c, method) + ".json")
    with open(path) as f:
        results = json.load(f)["results"]
    return sorted(results, key=lambda r: r["outer_fold"])


def load_archived_ablation(env, c, variant):
    """Per-outer-fold results of an archived ABLATION-001 variant (full, no_memory, ...)."""
    path = os.path.join(ABL_CKPT_DIR, archived_tag(env, c, variant) + ".json")
    with open(path) as f:
        results = json.load(f)["results"]
    return sorted(results, key=lambda r: r["outer_fold"])


def load_stats_summary():
    with open(STATS_SUMMARY) as f:
        return json.load(f)


def outer_folds(seeds, K_out, seed0=0):
    """The frozen outer split: ``kfold_split(seeds, K_out, default_rng(seed0))``."""
    return kfold_split(list(seeds), K_out, np.random.default_rng(seed0))


def inner_folds(outer_train, K_in, out_i, seed0=0):
    """The frozen inner split used by ``nested_cv_hpo`` for outer fold ``out_i``."""
    return kfold_split(outer_train, K_in, np.random.default_rng(seed0 * 7 + out_i))
