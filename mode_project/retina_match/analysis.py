# -*- coding: utf-8 -*-
"""analysis.py"""

import numpy as np
import pandas as pd

from .features import (
    build_state_features,
    joint_zscore,
    columnwise_permutation_null,
    joint_robust_scale,
)

from .clustering import (
    cluster_states,
    cluster_centroids_present,
    match_centroids,
    total_assignment_cost,
)


def cluster_match_for_seed(fpA, fpB, k, seed,noise=0.02, null_mode=None, use_log_occ=False):
    rng = np.random.default_rng(seed)

    XA = build_state_features(fpA, use_log_occ=use_log_occ)
    XB = build_state_features(fpB, use_log_occ=use_log_occ)

    if null_mode == "colperm":
        XB = columnwise_permutation_null(XB, rng)

    # XA, XB = joint_zscore(XA, XB)
    XA, XB = joint_robust_scale(XA, XB)

    if noise and noise > 0:
        XA = XA + noise * rng.normal(size=XA.shape)
        XB = XB + noise * rng.normal(size=XB.shape)

    labA = cluster_states(XA, k)
    labB = cluster_states(XB, k)

    CA, uniqA = cluster_centroids_present(XA, labA, method="mean")
    CB, uniqB = cluster_centroids_present(XB, labB, method="mean")

    pairs, D = match_centroids(CA, CB)

    return {
        "XA": XA,
        "XB": XB,
        "labA": labA,
        "labB": labB,
        "CA": CA,
        "CB": CB,
        "uniqA": uniqA,
        "uniqB": uniqB,
        "pairs": pairs,
        "D": D,
    }


def evaluate_k_standard(
    fpA,
    fpB,
    k,
    n_seeds=30,
    noise=0.02,
    agg="mean",
    null_mode="colperm",
    use_log_occ=False,
):

    real_costs = []
    null_costs = []

    for sd in range(n_seeds):

        out_real = cluster_match_for_seed(
            fpA,
            fpB,
            k,
            seed=sd,
            noise=noise,
            null_mode=None,
            use_log_occ=use_log_occ,
        )

        out_null = cluster_match_for_seed(
            fpA,
            fpB,
            k,
            seed=10000 + sd,
            noise=noise,
            null_mode=null_mode,
            use_log_occ=use_log_occ,
        )

        real_costs.append(total_assignment_cost(out_real["pairs"], agg=agg))
        null_costs.append(total_assignment_cost(out_null["pairs"], agg=agg))

    real_costs = np.array(real_costs, float)
    null_costs = np.array(null_costs, float)

    improvement = null_costs - real_costs

    return {
        "k": int(k),
        "real_mean": float(real_costs.mean()),
        "real_std": float(real_costs.std(ddof=1)),
        "null_mean": float(null_costs.mean()),
        "null_std": float(null_costs.std(ddof=1)),
        "impr_mean": float(improvement.mean()),
        "impr_std": float(improvement.std(ddof=1)),
        "impr_z": float(improvement.mean() / (improvement.std(ddof=1) + 1e-12)),
    }


def mode_contrast_table(modes, contrast, K=None):

    modes = np.asarray(modes)
    contrast = np.asarray(contrast)

    assert len(modes) == len(contrast)

    # 去掉 NaN/无效 contrast
    valid = np.isfinite(contrast)
    m = modes[valid]
    c = contrast[valid].astype(int)

    if K is None:
        K = int(m.max()) + 1

    # 交叉表
    tab = pd.crosstab(m, c)

    # 行归一化
    frac = tab.div(tab.sum(axis=1), axis=0)

    cover_n = (frac > 0.01).sum(axis=1)
    purity = frac.max(axis=1)
    main_contrast = frac.idxmax(axis=1)

    summary = pd.DataFrame(
        {
            "total_bins": tab.sum(axis=1),
            "n_contrasts(>1%)": cover_n,
            "purity(max frac)": purity,
            "main_contrast": main_contrast,
        }
    ).sort_values(
        ["purity(max frac)", "total_bins"],
        ascending=[False, False],
    )

    return tab, frac, summary


def full_mode_contrast_table(modes, contrast):

    modes = np.asarray(modes)
    contrast = np.asarray(contrast)

    assert len(modes) == len(contrast)

    # NaN contrast 标记为 -1
    contrast_clean = contrast.copy()
    contrast_clean = np.nan_to_num(contrast_clean, nan=-1)

    tab = pd.crosstab(modes, contrast_clean)

    total_bins = tab.sum(axis=1)

    frac = tab.div(total_bins, axis=0)

    return tab, frac