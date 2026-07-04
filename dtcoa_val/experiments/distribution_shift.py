"""Minimal distribution-shift stress test (RQ5, frozen minimal scope --
one train-regime -> shifted-regime pair, per PROJECT_SCOPE_FREEZE.md).

Protocol requirement: "train/tune under one regime, evaluate under a shifted
one." This reuses the ALREADY-TUNED per-outer-fold configs from a
nested_cv_hpo run on the training environment and evaluates them, unchanged,
on the shifted environment -- no new HPO. This is deliberately cheap: its
purpose is a single generalization check, not a second full experiment.
"""
import numpy as np
from dtcoa_val.experiments.nested_hpo import _eval_config


def evaluate_under_shift(policy_class, trained_results, shifted_cfg, shifted_seeds, tag=""):
    """For each outer fold's tuned config (from a nested_cv_hpo run on the
    TRAINING environment), evaluate it -- without re-tuning -- on
    shifted_seeds drawn under shifted_cfg (the SHIFTED environment).

    Returns per-fold shifted performance plus the delta vs. that fold's
    original (in-distribution) test performance, so degradation under shift
    is directly visible fold-by-fold.
    """
    out = []
    for fold in trained_results:
        config = dict(fold["best_config"])
        # band_vector params come back as plain lists from JSON/dict storage;
        # policy constructors expect them as-is (np.sort applied inside).
        shifted_eval = _eval_config(
            policy_class, config, shifted_cfg, shifted_seeds,
            tag=f"{tag}_shift_fold{fold['outer_fold']}",
        )
        out.append({
            "outer_fold": fold["outer_fold"],
            "config": config,
            "in_distribution_mean_regret": fold["test_mean_regret"],
            "in_distribution_cvar95": fold["test_cvar95_regret"],
            "shifted_mean_regret": shifted_eval["mean_regret"],
            "shifted_cvar95": shifted_eval["cvar95_regret"],
            "regret_delta_under_shift": shifted_eval["mean_regret"] - fold["test_mean_regret"],
        })
    return out


def shift_report(shift_results):
    deltas = [r["regret_delta_under_shift"] for r in shift_results]
    return {
        "mean_regret_delta_under_shift": float(np.mean(deltas)),
        "max_regret_delta_under_shift": float(np.max(deltas)),
        "per_fold_deltas": deltas,
        "note": (
            "Positive delta = performance degrades under distribution shift, "
            "as expected for any tuned method; large disparities between "
            "methods' deltas (not just their absolute levels) are the "
            "generalization-relevant signal here."
        ),
    }
