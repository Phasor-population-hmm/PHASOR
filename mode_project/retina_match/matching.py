import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .plotting import remap_states_keep, plot_viterbi_bands_reps


MAIN_SCORE_COL = "dist_sim_global"

SUMMARY_COLS = [
    "A_state", "B_state",
    "match_dist", "dist_norm_global", "dist_sim_global", "pair_score",
    "A_occ", "B_occ", "pair_weight",
]


# ============================================================
# Helpers
# ============================================================

def _get_occ(fp, state=None):
    """
    Prefer empirical occupancy when available.
    """
    arr = fp["occupancy_emp"] if "occupancy_emp" in fp else fp["occupancy"]
    arr = np.asarray(arr, float)
    if state is None:
        return arr
    return float(arr[state])


def build_global_distance_similarity(D, global_dmin, global_dmax, eps=1e-12):
    """
    Normalize distances using pooled global min/max,
    not pair-specific min/max.
    """
    D = np.asarray(D, float)
    Dnorm = (D - global_dmin) / (global_dmax - global_dmin + eps)
    Dnorm = np.clip(Dnorm, 0.0, 1.0)
    Sdist = np.exp(-Dnorm)
    return Dnorm, Sdist


def hungarian_pairs_from_distance(D):
    D = np.asarray(D, float)
    row_ind, col_ind = linear_sum_assignment(D)
    pairs = [(int(i), int(j), float(D[i, j])) for i, j in zip(row_ind, col_ind)]
    return pairs


# ============================================================
# Diagnose whether 1-to-1 is trustworthy
# ============================================================

def assess_one_to_one_from_pairs(
    D, pairs,
    ratio_thr=0.85,
    forced_rate_thr=0.25,
    gap_ratio_thr=3.0,
    strong_mismatch_min=2
):
    D = np.asarray(D, float)
    k_row, _ = D.shape

    amap = {int(i): int(j) for i, j, _ in pairs}

    ranks = np.full(k_row, -1, dtype=int)
    strong_ratio_but_not_nn = 0
    assigned_rows = 0

    for i in range(k_row):
        if i not in amap:
            continue

        assigned_rows += 1
        j_assigned = amap[i]

        order = np.argsort(D[i])
        rank = int(np.where(order == j_assigned)[0][0]) + 1
        ranks[i] = rank

        if len(order) >= 2:
            j1, j2 = int(order[0]), int(order[1])
            d1, d2 = float(D[i, j1]), float(D[i, j2])
            ratio = d1 / (d2 + 1e-12)

            if (ratio < ratio_thr) and (j_assigned != j1):
                strong_ratio_but_not_nn += 1

    valid_ranks = ranks[ranks >= 1]
    forced_rate = float(np.mean(valid_ranks > 1)) if len(valid_ranks) else 0.0

    costs = np.array(sorted([float(c) for _, _, c in pairs]), float)
    gaps = costs[1:] - costs[:-1]
    med_gap = float(np.median(gaps)) if len(gaps) else 0.0
    max_gap = float(np.max(gaps)) if len(gaps) else 0.0
    gap_ratio = (max_gap / (med_gap + 1e-12)) if len(gaps) else 0.0

    ok = True
    reasons = []

    if assigned_rows < k_row:
        ok = False
        reasons.append(f"only_assigned_rows={assigned_rows}/{k_row}")

    if forced_rate > forced_rate_thr:
        ok = False
        reasons.append(f"forced_rate={forced_rate:.2f} > {forced_rate_thr}")

    if strong_ratio_but_not_nn >= strong_mismatch_min:
        ok = False
        reasons.append(
            f"strong_ratio_mismatches={strong_ratio_but_not_nn} >= {strong_mismatch_min}"
        )

    if gap_ratio > gap_ratio_thr:
        ok = False
        reasons.append(f"gap_ratio={gap_ratio:.2f} > {gap_ratio_thr}")

    return {
        "ok_1to1": ok,
        "reasons": reasons,
        "assigned_rows": int(assigned_rows),
        "n_rows": int(k_row),
        "forced_rate": forced_rate,
        "strong_ratio_mismatches": int(strong_ratio_but_not_nn),
        "gap_ratio": gap_ratio,
        "ranks": ranks,
        "costs_sorted": costs,
    }


# ============================================================
# Matching
# ============================================================

def greedy_threshold_match(D, tau, allow_many_to_one=True):
    D = np.asarray(D, float)
    k_src, _ = D.shape

    matches = []
    used_tgt = set()
    unmatched_src = []

    for i in range(k_src):
        j = int(np.argmin(D[i]))
        d = float(D[i, j])

        if d > tau:
            unmatched_src.append(i)
            continue

        if (not allow_many_to_one) and (j in used_tgt):
            unmatched_src.append(i)
            continue

        matches.append((i, j, d))
        used_tgt.add(j)

    matches.sort(key=lambda x: x[2])
    return matches, unmatched_src

def topk_global_threshold_match(D, tau, k=3):
    D = np.asarray(D, float)
    n_src, n_tgt = D.shape

    matches = []
    unmatched_src = []

    for i in range(n_src):
        order = np.argsort(D[i])

        keep = []
        for j in order:
            d = float(D[i, j])
            if d <= tau:
                keep.append((i, int(j), d))
            if len(keep) >= k:
                break

        if len(keep) == 0:
            unmatched_src.append(i)
        else:
            matches.extend(keep)

    return matches, unmatched_src


def decide_and_match(D, pairs, tau, allow_many_to_one=True, diag=None):
    if diag is None:
        diag = assess_one_to_one_from_pairs(D, pairs)

    if diag["ok_1to1"]:
        return {
            "method": "1to1_ok",
            "diag": diag,
            "matches": pairs,
            "unmatched_A": [],
            "tau": tau,
        }

    m, unA = greedy_threshold_match(
        D, tau=tau, allow_many_to_one=allow_many_to_one
    )
    return {
        "method": "fallback_greedy",
        "diag": diag,
        "matches": m,
        "unmatched_A": unA,
        "tau": tau,
    }


def build_match_out(D, pairs, tau, allow_many_to_one=True):
    diag = assess_one_to_one_from_pairs(D, pairs)

    print("1-1 diagnostic:")
    print(diag)

    out = decide_and_match(
        D,
        pairs,
        tau=tau,
        allow_many_to_one=allow_many_to_one,
        diag=diag,
    )

    print("matching method:", out["method"])
    print("diagnostic:", out["diag"])
    print("unmatched rows of current source side:", out["unmatched_A"])
    print("tau:", out["tau"])
    return out


# ============================================================
# Row builders
# ============================================================

def build_one_match_row(
    a_state, b_state, dist,
    fpA, fpB,
    Dnorm_global, Sdist_global,
    z_repsA=None, z_repsB=None,
    do_plot=True,
):
    print("\nMATCH:")
    print(f"A_state {a_state} -> B_state {b_state}, dist={dist:.3f}")

    if do_plot and (z_repsA is not None):
        Z_keepA = remap_states_keep(z_repsA, keep_states=[a_state], other_state=-1)
        plot_viterbi_bands_reps(
            Z_keepA,
            bin_ms=10,
            title=f"A state {a_state}",
            legend_top_n=None,
            other_state=-1,
        )

    if do_plot and (z_repsB is not None):
        Z_keepB = remap_states_keep(z_repsB, keep_states=[b_state], other_state=-1)
        plot_viterbi_bands_reps(
            Z_keepB,
            bin_ms=10,
            title=f"B state {b_state}",
            legend_top_n=None,
            other_state=-1,
        )

    occA = _get_occ(fpA, a_state)
    occB = _get_occ(fpB, b_state)
    pair_weight = 0.5 * (occA + occB)

    dist_norm_global = float(Dnorm_global[a_state, b_state])
    dist_sim_global = float(Sdist_global[a_state, b_state])

    return {
        "A_state": int(a_state),
        "B_state": int(b_state),
        "match_dist": float(dist),
        "dist_norm_global": dist_norm_global,
        "dist_sim_global": dist_sim_global,
        "pair_score": float(dist_sim_global),
        "A_occ": occA,
        "B_occ": occB,
        "pair_weight": pair_weight,
    }


def build_unmatched_row_A(a_state, fpA, z_repsA=None, do_plot=True):
    print("\nUNMATCHED A STATE:")
    print(f"A_state {a_state}")

    if do_plot and (z_repsA is not None):
        Z_keepA = remap_states_keep(z_repsA, keep_states=[a_state], other_state=-1)
        plot_viterbi_bands_reps(
            Z_keepA,
            bin_ms=10,
            title=f"A state {a_state} (UNMATCHED)",
            legend_top_n=None,
            other_state=-1,
        )

    occA = _get_occ(fpA, a_state)

    return {
        "A_state": int(a_state),
        "B_state": None,
        "match_dist": None,
        "dist_norm_global": None,
        "dist_sim_global": None,
        "pair_score": 0.0,
        "A_occ": occA,
        "B_occ": None,
        "pair_weight": occA,
    }


def build_unmatched_row_B(b_state, fpB, z_repsB=None, do_plot=True):
    print("\nUNMATCHED B STATE:")
    print(f"B_state {b_state}")

    if do_plot and (z_repsB is not None):
        Z_keepB = remap_states_keep(z_repsB, keep_states=[b_state], other_state=-1)
        plot_viterbi_bands_reps(
            Z_keepB,
            bin_ms=10,
            title=f"B state {b_state} (UNMATCHED)",
            legend_top_n=None,
            other_state=-1,
        )

    occB = _get_occ(fpB, b_state)

    return {
        "A_state": None,
        "B_state": int(b_state),
        "match_dist": None,
        "dist_norm_global": None,
        "dist_sim_global": None,
        "pair_score": 0.0,
        "A_occ": None,
        "B_occ": occB,
        "pair_weight": occB,
    }


# ============================================================
# Similarity summaries
# ============================================================

def compute_directional_similarity(df, weight_col, total_weight=None):
    if df.empty:
        return {
            "coverage": 0.0,
            "matched_pair_score": 0.0,
            "overall_similarity": 0.0,
        }

    valid = df["A_state"].notna() & df["B_state"].notna()

    weights = df[weight_col].fillna(0.0).to_numpy(float)
    scores = df["pair_score"].fillna(0.0).to_numpy(float)
    valid_mask = valid.to_numpy(bool)

    if total_weight is None:
        total_weight = float(weights.sum())
    else:
        total_weight = float(total_weight)

    if total_weight == 0:
        return {
            "coverage": 0.0,
            "matched_pair_score": 0.0,
            "overall_similarity": 0.0,
        }

    matched_weight = float(weights[valid_mask].sum())
    numerator = float(np.sum(weights * scores))

    coverage = matched_weight / total_weight
    matched_pair_score = numerator / matched_weight if matched_weight > 0 else 0.0
    overall_similarity = numerator / total_weight

    return {
        "coverage": float(coverage),
        "matched_pair_score": float(matched_pair_score),
        "overall_similarity": float(overall_similarity),
    }


def summarize_retina_matching_directional_from_out(
    out,
    source_side,
    fpA, fpB,
    Dnorm_global, Sdist_global,
    z_repsA=None, z_repsB=None,
    do_plot=True,
):
    rows = []

    if source_side == "A":
        for a_state, b_state, dist in out["matches"]:
            row = build_one_match_row(
                a_state, b_state, dist,
                fpA, fpB,
                Dnorm_global=Dnorm_global,
                Sdist_global=Sdist_global,
                z_repsA=z_repsA,
                z_repsB=z_repsB,
                do_plot=do_plot,
            )
            rows.append(row)

        for a_state in out["unmatched_A"]:
            rows.append(
                build_unmatched_row_A(
                    a_state,
                    fpA,
                    z_repsA=z_repsA,
                    do_plot=do_plot,
                )
            )

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=SUMMARY_COLS)

        sim = compute_directional_similarity(
            df,
            weight_col="A_occ",
            total_weight=float(np.sum(_get_occ(fpA))),
        )
        return df, sim

    elif source_side == "B":
        for b_state, a_state, dist in out["matches"]:
            row = build_one_match_row(
                a_state, b_state, dist,
                fpA, fpB,
                Dnorm_global=Dnorm_global,
                Sdist_global=Sdist_global,
                z_repsA=z_repsA,
                z_repsB=z_repsB,
                do_plot=do_plot,
            )
            rows.append(row)

        for b_state in out["unmatched_A"]:
            rows.append(
                build_unmatched_row_B(
                    b_state,
                    fpB,
                    z_repsB=z_repsB,
                    do_plot=do_plot,
                )
            )

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=SUMMARY_COLS)

        sim = compute_directional_similarity(
            df,
            weight_col="B_occ",
            total_weight=float(np.sum(_get_occ(fpB))),
        )
        return df, sim

    else:
        raise ValueError("source_side must be 'A' or 'B'")


def summarize_retina_matching_true_bidirectional(
    D,
    fpA, fpB,
    tau,
    global_dmin,
    global_dmax,
    pairs_AB=None,
    pairs_BA=None,
    z_repsA=None, z_repsB=None,
    allow_many_to_one=True,
    do_plot=True,
):
    D = np.asarray(D, float)

    if pairs_AB is None:
        pairs_AB = hungarian_pairs_from_distance(D)

    if pairs_BA is None:
        pairs_BA = hungarian_pairs_from_distance(D.T)

    Dnorm_global, Sdist_global = build_global_distance_similarity(
        D, global_dmin=global_dmin, global_dmax=global_dmax
    )

    print("\n=== Global distance normalization ===")
    print("global_dmin =", global_dmin)
    print("global_dmax =", global_dmax)
    print("tau =", tau)

    print("\n==============================")
    print("=== Forward matching: A -> B ===")
    print("==============================")
    out_AB = build_match_out(
        D,
        pairs_AB,
        tau=tau,
        allow_many_to_one=allow_many_to_one,
    )

    df_AB, sim_AB = summarize_retina_matching_directional_from_out(
        out_AB,
        source_side="A",
        fpA=fpA, fpB=fpB,
        Dnorm_global=Dnorm_global,
        Sdist_global=Sdist_global,
        z_repsA=z_repsA,
        z_repsB=z_repsB,
        do_plot=do_plot,
    )

    print("\n=== Directional summary table: A -> B ===")
    print(df_AB)

    print("\n==============================")
    print("=== Reverse matching: B -> A ===")
    print("==============================")
    out_BA = build_match_out(
        D.T,
        pairs_BA,
        tau=tau,
        allow_many_to_one=allow_many_to_one,
    )

    df_BA, sim_BA = summarize_retina_matching_directional_from_out(
        out_BA,
        source_side="B",
        fpA=fpA, fpB=fpB,
        Dnorm_global=Dnorm_global,
        Sdist_global=Sdist_global,
        z_repsA=z_repsA,
        z_repsB=z_repsB,
        do_plot=do_plot,
    )

    print("\n=== Directional summary table: B -> A ===")
    print(df_BA)

    sim_sym = 0.5 * (
        sim_AB["overall_similarity"] + sim_BA["overall_similarity"]
    )

    sim_quality_sym = 0.5 * (
        sim_AB["matched_pair_score"] + sim_BA["matched_pair_score"]
    )

    print("\n=== A -> B ===")
    print("matched_pair_score =", sim_AB["matched_pair_score"])
    print("coverage =", sim_AB["coverage"])
    print("overall_similarity =", sim_AB["overall_similarity"])
    print("overall_similarity_pct =", 100 * sim_AB["overall_similarity"])

    print("\n=== B -> A ===")
    print("matched_pair_score =", sim_BA["matched_pair_score"])
    print("coverage =", sim_BA["coverage"])
    print("overall_similarity =", sim_BA["overall_similarity"])
    print("overall_similarity_pct =", 100 * sim_BA["overall_similarity"])

    print("\n=== Symmetric ===")
    print("symmetric_similarity =", sim_sym)
    print("symmetric_similarity_pct =", 100 * sim_sym)
    print("symmetric_matched_quality =", sim_quality_sym)
    print("symmetric_matched_quality_pct =", 100 * sim_quality_sym)

    outs = {
        "AB": out_AB,
        "BA": out_BA,
    }

    return df_AB, df_BA, sim_AB, sim_BA, sim_sym, outs


# ============================================================
# Global calibration
# ============================================================

def collect_all_nn_distances(distance_dict):
    """
    distance_dict:
        {(nameA, nameB): D_ab, ...}
        only unique pairs, e.g. i < j
    """
    all_nn = []

    for (_, _), D in distance_dict.items():
        D = np.asarray(D, float)
        all_nn.extend(D.min(axis=1).tolist())  # A -> B nearest
        all_nn.extend(D.min(axis=0).tolist())  # B -> A nearest

    return np.asarray(all_nn, float)



def choose_global_calibration(distance_dict, tau_quantile=0.15):
    if len(distance_dict) == 0:
        raise ValueError("distance_dict is empty")

    all_entries = []
    for D in distance_dict.values():
        D = np.asarray(D, float)
        if D.size > 0:
            all_entries.append(D.ravel())

    if len(all_entries) == 0:
        raise ValueError("No distance entries found")

    all_entries = np.concatenate(all_entries)

    tau = float(np.quantile(all_entries, tau_quantile))
    global_dmin = float(np.min(all_entries))
    global_dmax = float(np.max(all_entries))

    return {
        "tau": tau,
        "global_dmin": global_dmin,
        "global_dmax": global_dmax,
        "all_entries": all_entries,
    }