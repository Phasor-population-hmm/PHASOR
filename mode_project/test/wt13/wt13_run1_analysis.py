import sys
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

# =========================
# User settings
# =========================
PROJECT_ROOT = Path("/Users/sichenchen/Desktop/mode_project")
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "wt13" / "run1"

STIM_JSON_PATH = DATA_DIR / "FFsine_2Hz_stims.json"
SYNCTONES_CSV_PATH = DATA_DIR / "2026-03-11T12-17-28wt13_Green=255_OD=4_2Hz_run1_B-00068_synctones.csv"
SPIKE_TXT_PATH = DATA_DIR / "2026-03-11T12-17-28wt13_Green=255_OD=4_2Hz_run1_B-00068.txt"

MODEL_PATH = DATA_DIR / "chowliu_K12_eta0.005_alpha0.5.npz"
OUTPUT_DIR = DATA_DIR / "final_analysis"

BIN_SIZE_S = 0.01
TARGET_HZ = 2
N_TRAIN_REP = 10
N_MODES = 12
CONTRAST_LEVELS = [50, 60, 70, 80, 90]
N_REP_TOTAL = 15

# =========================
# Imports from project
# =========================
from data_processing import *
from hmm_pkg import *
from hmm_pkg.utils import load_chowliu_hmm

from analysis.modes import (
    summarize_modes,
    compute_Smax_combinatorial,
    normalized_entropy_ratio,
    chowliu_entropy_rate_all_modes,
)

from analysis.transitions import off_diagonal_transition, row_entropy
from analysis.dwell import compute_dwell_times, check_dwell_consistency

from viz.mode_plot import (
    plot_mode_probabilities,
    plot_entropy_vs_kbar,
    plot_normalized_entropy,
)

from viz.dwell_plots import plot_dwell_histograms
from viz.transitions_plots import (
    plot_transition_matrices,
    plot_transition_graph,
    plot_transition_entropy_vs_kbar,
)

from retina_match.features import extract_fingerprint, split_fingerprint_by_state
from retina_match.io import save_state_fps

# =========================
# Color palette
# =========================
COLORBLIND_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    "#000000", "#999999", "#882255", "#AA4499", "#117733", "#44AA99", "#DDCC77",
    "#88CCEE", "#332288", "#661100", "#6699CC", "#AA4455", "#228833", "#CCBB44",
    "#EE6677", "#AA7744", "#774411", "#114477", "#771155", "#555555", "#99DDFF",
    "#77AADD", "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02",
    "#A6761D", "#666666", "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854",
    "#FFD92F", "#E5C494", "#B3B3B3",
]

def segments_from_states(z):
    z = np.asarray(z).astype(int)
    T = len(z)
    if T == 0:
        return []
    segs = []
    start = 0
    cur = z[0]
    for t in range(1, T):
        if z[t] != cur:
            segs.append((start, t, cur))
            start = t
            cur = z[t]
    segs.append((start, T, cur))
    return segs

def plot_viterbi_bands_reps(
    z_reps,
    save_path,
    bin_ms=None,
    title="All contrasts: Viterbi segmentation across reps",
    rep_labels=None,
    figsize=None,
    show_legend=True,
    legend_top_n=30
):
    Z = np.asarray(z_reps).astype(int)
    assert Z.ndim == 2, "z_reps must be 2D: (n_reps, T_rep)"
    n_reps, T = Z.shape

    if bin_ms is None:
        to_x = lambda t: t
        xlabel = "time bin"
        x_end = T
    else:
        to_x = lambda t: t * (bin_ms / 1000.0)
        xlabel = "time (s)"
        x_end = to_x(T)

    all_states = np.unique(Z)
    state_counts = {s: np.sum(Z == s) for s in all_states}
    sorted_states = sorted(all_states, key=lambda s: state_counts[s], reverse=True)

    if legend_top_n is not None:
        sorted_states = sorted_states[:legend_top_n]

    state_to_color = {
        s: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        for i, s in enumerate(sorted_states)
    }

    if rep_labels is None:
        rep_labels = [f"rep {i+1}" for i in range(n_reps)]

    if figsize is None:
        figsize = (14, max(2.5, 0.55 * n_reps))

    fig, ax = plt.subplots(figsize=figsize)
    band_h = 0.8
    gap = 0.25

    for r in range(n_reps):
        z = Z[r]
        segs = segments_from_states(z)
        y0 = (n_reps - 1 - r) * (band_h + gap)

        for (a, b, s) in segs:
            if s not in state_to_color:
                continue
            xa, xb = to_x(a), to_x(b)
            ax.add_patch(Rectangle(
                (xa, y0), xb - xa, band_h,
                linewidth=0,
                facecolor=state_to_color[s]
            ))

    ax.set_xlim(0, x_end)
    ax.set_ylim(-gap, n_reps * (band_h + gap))
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    yticks = [(n_reps - 1 - r) * (band_h + gap) + band_h / 2 for r in range(n_reps)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(rep_labels)
    ax.grid(True, axis="x", alpha=0.3)

    if show_legend:
        handles = [Patch(facecolor=state_to_color[s], label=f"State {s}") for s in sorted_states]
        max_rows = 16
        n_items = len(handles)
        ncol = max(1, math.ceil(n_items / max_rows))
        fig.subplots_adjust(right=0.78)
        ax.legend(
            handles=handles,
            ncol=ncol,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            columnspacing=1.0,
            handlelength=1.2,
            handletextpad=0.4,
            fontsize=9,
            frameon=True
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def build_bin_labels_from_plan(df_train, plan_train, col="contrast"):
    if col not in df_train.columns:
        raise KeyError(f"df_train has no column '{col}'")
    if "n_bins" not in plan_train.columns:
        raise KeyError("plan_train must contain column 'n_bins'")
    if len(df_train) != len(plan_train):
        raise ValueError(f"len mismatch: df_train={len(df_train)} plan_train={len(plan_train)}")

    labels = []
    for i in range(len(df_train)):
        n_bins = int(plan_train.loc[i, "n_bins"])
        labels.extend([df_train.loc[i, col]] * n_bins)
    return np.array(labels)

def plot_fig3d_readable(
    w_sorted, m_centroid, k_bar,
    topK=15,
    metric="euclidean",
    grid_n=240,
    pad=0.55,
    spread=2.4,
    s0=0.035,
    s1=0.10,
    sigma_min=0.03,
    sigma_max=0.13,
    sigma_growth="sqrt",
    height_transform="log",
    beta=0.8,
    z_gamma=0.85,
    z_floor=0.02,
    elev=28,
    azim=-135,
    save_path=None,
    show=False,
):
    w_sorted = np.asarray(w_sorted, float)
    m_centroid = np.asarray(m_centroid, float)
    k_bar = np.asarray(k_bar, float)
    eps = 1e-12

    K = min(topK, len(w_sorted))
    w = w_sorted[:K]
    m = m_centroid[:K]
    kb = k_bar[:K]

    D = pairwise_distances(m, metric=metric)
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=0).fit_transform(D)

    k0 = int(np.argmax(w))
    coords = coords - coords[k0]

    angles = np.arctan2(coords[:, 1], coords[:, 0])
    r = (kb - kb.min()) / (kb.max() - kb.min() + eps)

    cx = spread * r * np.cos(angles)
    cy = spread * r * np.sin(angles)

    if height_transform == "log":
        A = np.log(w + eps)
        A = A - A.min()
    elif height_transform == "sqrt":
        A = np.sqrt(w + eps)
    else:
        A = w.copy()

    A = A / (A.max() + eps)
    A = A * np.exp(-beta * r)

    if sigma_growth == "sqrt":
        rg = np.sqrt(r)
    elif sigma_growth == "linear":
        rg = r
    else:
        raise ValueError("sigma_growth must be 'sqrt' or 'linear'")

    sigma = s0 + s1 * rg
    sigma = np.clip(sigma, sigma_min, sigma_max)

    x_min, x_max = cx.min() - pad, cx.max() + pad
    y_min, y_max = cy.min() - pad, cy.max() + pad
    gx = np.linspace(x_min, x_max, grid_n)
    gy = np.linspace(y_min, y_max, grid_n)
    Xg, Yg = np.meshgrid(gx, gy)

    Z = np.zeros_like(Xg)
    for i in range(K):
        Z += A[i] * np.exp(-0.5 * (((Xg - cx[i])**2 + (Yg - cy[i])**2) / (sigma[i]**2 + eps)))

    Z = (Z - Z.min()) / (Z.max() - Z.min() + eps)
    Z = Z**z_gamma
    Z = z_floor + (1 - z_floor) * Z

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xg, Yg, Z, cmap="hot", linewidth=0, antialiased=True, shade=True)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("m")
    ax.set_ylabel("")
    ax.set_zlabel("P")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    dist = np.sqrt((cx[:, None] - cx[None, :])**2 + (cy[:, None] - cy[None, :])**2)
    min_nonzero = dist[dist > 0].min() if K > 1 else np.nan
    print(f"[debug] sigma min/max = {sigma.min():.3f}/{sigma.max():.3f} | min center dist = {min_nonzero:.3f}")

    return {
        "coords": np.c_[cx, cy],
        "A": A,
        "sigma": sigma,
        "r": r,
        "silent_idx": k0,
    }

def save_fingerprint_summary(fp, output_path):
    with open(output_path, "w") as f:
        f.write("Sticky states (Aii top 5):\n")
        f.write(str(np.argsort(fp["Aii"])[-5:]) + "\n\n")

        f.write("Most occupied states:\n")
        f.write(str(np.argsort(fp["occupancy"])[-5:]) + "\n\n")

        f.write("Most structured states (deg_std high):\n")
        f.write(str(np.argsort(fp["deg_std"])[-5:]) + "\n\n")

        f.write(f"'cv_dwell_emp' in fp: {'cv_dwell_emp' in fp}\n")
        f.write(f"'cv_gap_emp' in fp: {'cv_gap_emp' in fp}\n")
        f.write(f"'time_profile_emp' in fp: {'time_profile_emp' in fp}\n")
        f.write(f"'time_profile_shape_emp' in fp: {'time_profile_shape_emp' in fp}\n\n")

        if "gap_quantiles_emp" in fp:
            f.write("gap_quantiles_emp[:, [2, 4]] (p50, p90):\n")
            f.write(str(fp["gap_quantiles_emp"][:, [2, 4]]) + "\n\n")

        if "time_profile_emp" in fp:
            f.write(f"time_profile_emp shape: {fp['time_profile_emp'].shape}\n")

        if "time_profile_shape_emp" in fp:
            f.write(f"time_profile_shape_emp shape: {fp['time_profile_shape_emp'].shape}\n")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df_cond = load_stim_json(str(STIM_JSON_PATH))
    df_time = load_synctone_csv(str(SYNCTONES_CSV_PATH))
    df_spk = load_spike_txt(str(SPIKE_TXT_PATH))

    print("Building analysis windows...")
    df_cond = build_analysis_windows(
        df_cond,
        df_time=df_time,
        t0=0.0,
        synctone_col="onset_sec"
    )

    print("Selecting stimulus segments...")
    df_sel = select_stimulus_segments(
        df_cond,
        selection_strategy="sequential_repetition",
        target_hz=TARGET_HZ,
        contrast_levels=CONTRAST_LEVELS,
        n_rep=N_REP_TOTAL
    )

    print("Splitting train/test...")
    df_train, df_rep = split_train_test_by_rep(df_sel, n_train_rep=N_TRAIN_REP)

    print("Making neuron index...")
    df_spk2, neuron_list, neuron_to_idx = make_neuron_index(df_spk)
    N = len(neuron_list)
    print("N neurons =", N)

    print("Assigning training spikes...")
    df_spk_labeled_train, df_spk_sel_train = assign_spikes_to_condition(df_spk2, df_train)
    plan_train = compute_binning_plan(df_train, bin_size_s=BIN_SIZE_S)
    X_train = build_binary_matrix(
        df_spk_sel_train,
        df_train,
        plan_train,
        bin_size_s=BIN_SIZE_S,
        n_neurons=N
    )
    print("X_train shape:", X_train.shape)

    print("Assigning test spikes...")
    df_spk_labeled_test, df_spk_sel_test = assign_spikes_to_condition(df_spk2, df_rep)
    plan_test = compute_binning_plan(df_rep, bin_size_s=BIN_SIZE_S)
    X_test = build_binary_matrix(
        df_spk_sel_test,
        df_rep,
        plan_test,
        bin_size_s=BIN_SIZE_S,
        n_neurons=N
    )
    print("X_test shape:", X_test.shape)

    print("Loading model...")
    model = load_chowliu_hmm(str(MODEL_PATH))
    pi = model.pi
    A = model.A
    E = model.emission.p1_all
    print("pi, A, E shapes:", pi.shape, A.shape, E.shape)

    print("Running Viterbi...")
    modes_train = model.viterbi(X_train)
    modes_test = model.viterbi(X_test)
    print("modes_train shape:", modes_train.shape)
    print("modes_test shape:", modes_test.shape)

    contrast_id = build_bin_labels_from_plan(df_train, plan_train, col="contrast")
    np.savez_compressed(
        OUTPUT_DIR / "bundle.npz",
        modes_train=modes_train.astype(np.int16),
        modes_test=modes_test.astype(np.int16),
        contrast=contrast_id
    )

    print("Running mode summary...")
    p1_all = model.emission.p1_all
    edges_all = model.emission.edges_all
    pair_probs_all = model.emission.pair_probs_all

    w_sorted, order, m_centroid, k_bar, _ = summarize_modes(
        X_train, modes_train, K=N_MODES, dt=BIN_SIZE_S
    )

    S_bits_s = chowliu_entropy_rate_all_modes(
        p1_all,
        edges_all,
        pair_probs_all,
        dt=BIN_SIZE_S
    )
    Smax_bits_s = compute_Smax_combinatorial(k_bar, N=X_train.shape[1], dt=BIN_SIZE_S)
    ratio = normalized_entropy_ratio(Smax_bits_s, S_bits_s)

    fig1, _ = plot_mode_probabilities(w_sorted, title="Mode probabilities")
    fig1.savefig(OUTPUT_DIR / "mode_probabilities.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, _ = plot_entropy_vs_kbar(k_bar, S_bits_s, title="Entropy vs k_bar")
    fig2.savefig(OUTPUT_DIR / "entropy_vs_kbar.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    fig3, _ = plot_normalized_entropy(k_bar, ratio, title="Normalized entropy")
    fig3.savefig(OUTPUT_DIR / "normalized_entropy.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)

    print("Saving 3D mode landscape...")
    out_fig3d = plot_fig3d_readable(
        w_sorted,
        m_centroid,
        k_bar,
        topK=25,
        azim=-225,
        save_path=OUTPUT_DIR / "fig3d_mode_landscape.png",
        show=False,
    )
    print("silent idx in plotted modes:", out_fig3d["silent_idx"])

    print("Running transition analysis...")
    A_off = off_diagonal_transition(A)
    H_all = row_entropy(A)
    H_off = row_entropy(A_off)

    fig4, _ = plot_transition_matrices(A, A_off)
    fig4.savefig(OUTPUT_DIR / "transition_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig4)

    fig5, _ = plot_transition_entropy_vs_kbar(k_bar, H_all, H_off, K=A.shape[0])
    fig5.savefig(OUTPUT_DIR / "transition_entropy_vs_kbar.png", dpi=200, bbox_inches="tight")
    plt.close(fig5)

    fig6, ax = plot_transition_graph(
        A, modes_train, N_MODES,
        label_fontsize=26, edge_scale=30, th=0.05, seed=0
    )
    fig6.savefig(OUTPUT_DIR / "transition_graph.png", dpi=200, bbox_inches="tight")
    plt.close(fig6)

    print("Running dwell analysis...")
    dwell_times = compute_dwell_times(modes_train, BIN_SIZE_S)
    diff = check_dwell_consistency(
        dwell_times,
        bin_size_s=BIN_SIZE_S,
        T_original=len(modes_train),
    )
    print("Consistency diff:", diff)

    fig7 = plot_dwell_histograms(dwell_times, max_cols=5)
    fig7.savefig(OUTPUT_DIR / "dwell_histograms.png", dpi=200, bbox_inches="tight")
    plt.close(fig7)

    print("Running fingerprint analysis...")
    fp = extract_fingerprint(model, X_train, modes=modes_train, bin_size_s=BIN_SIZE_S)

    print("Sticky states (Aii top 5):")
    print(np.argsort(fp["Aii"])[-5:])

    print("Most occupied states:")
    print(np.argsort(fp["occupancy"])[-5:])

    print("Most structured states (deg_std high):")
    print(np.argsort(fp["deg_std"])[-5:])

    if "gap_quantiles_emp" in fp:
        print("gap_quantiles_emp p50 and p90:")
        print(fp["gap_quantiles_emp"][:, [2, 4]])

    print("cv_dwell_emp" in fp, "cv_gap_emp" in fp)
    print("time_profile_emp" in fp, "time_profile_shape_emp" in fp)

    if "time_profile_emp" in fp:
        print("time_profile_emp shape:", fp["time_profile_emp"].shape)

    if "time_profile_shape_emp" in fp:
        print("time_profile_shape_emp shape:", fp["time_profile_shape_emp"].shape)

    state_fps = split_fingerprint_by_state(fp)
    save_state_fps(state_fps, str(OUTPUT_DIR / "state_fingerprints.npz"))
    save_fingerprint_summary(fp, OUTPUT_DIR / "fingerprint_summary.txt")

    print("Saving Viterbi bands...")
    z_vit_train = np.asarray(modes_train)
    z_vit_test = np.asarray(modes_test)

    z_reps_train = z_vit_train.reshape(10, 3000)
    z_reps_test = z_vit_test.reshape(5, 3000)
    z_reps = np.concatenate((z_reps_train, z_reps_test), axis=0)

    plot_viterbi_bands_reps(
        z_reps,
        save_path=OUTPUT_DIR / "viterbi_bands.png",
        title="All contrasts: Viterbi segmentation across reps",
        legend_top_n=30
    )

    print(f"All analysis finished. Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()