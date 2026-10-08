# -*- coding: utf-8 -*-

import numpy as np


# ============================================================
# Scaling helpers
# ============================================================

def joint_zscore(XA, XB, eps=1e-12):
    Xall = np.vstack([XA, XB])
    mu = Xall.mean(axis=0, keepdims=True)
    sd = Xall.std(axis=0, keepdims=True) + eps
    return (XA - mu) / sd, (XB - mu) / sd


def joint_robust_scale(XA, XB, eps=1e-12):
    Xall = np.vstack([XA, XB])
    med = np.median(Xall, axis=0, keepdims=True)
    mad = np.median(np.abs(Xall - med), axis=0, keepdims=True)
    scale = 1.4826 * mad + eps
    return (XA - med) / scale, (XB - med) / scale


def columnwise_permutation_null(X, rng):
    X = np.asarray(X, float).copy()
    for j in range(X.shape[1]):
        X[:, j] = X[rng.permutation(X.shape[0]), j]
    return X


# ============================================================
# Time-profile / empirical temporal features
# ============================================================

def _get_run_starts_ends(mask_1d):
    """
    mask_1d : 1D bool / {0,1}
    return starts, ends of contiguous True runs
    """
    x = np.asarray(mask_1d)

    if x.ndim != 1:
        raise ValueError(f"mask_1d must be 1D, got shape={x.shape}")

    x = x.astype(bool).astype(np.int8)
    padded = np.concatenate(([0], x, [0]))
    dx = np.diff(padded)

    starts = np.flatnonzero(dx == 1)
    ends = np.flatnonzero(dx == -1) - 1

    if starts.size != ends.size:
        raise ValueError(
            f"Run parsing failed: starts.shape={starts.shape}, ends.shape={ends.shape}"
        )

    return starts, ends


def compute_dwell_and_gap_times(modes, K=None, bin_size_s=1.0):
    """
    Compute empirical dwell times and gap times for each mode
    from a 1D mode sequence.

    Parameters
    ----------
    modes : array-like, shape (T,)
        1D state sequence
    K : int or None
        Number of modes. If None, inferred from modes.max()+1
    bin_size_s : float
        Bin size in seconds

    Returns
    -------
    dwell_times_per_mode : list of np.ndarray
        dwell times (seconds) for each mode
    gap_times_per_mode : list of np.ndarray
        gap times (seconds) for each mode
    n_visits_per_mode : np.ndarray, shape (K,)
        total number of visits
    occ_per_mode : np.ndarray, shape (K,)
        occupancy fraction across all bins
    """
    Z = np.asarray(modes)

    if Z.ndim != 1:
        raise ValueError(f"modes must be 1D, got shape={Z.shape}")

    if not np.issubdtype(Z.dtype, np.integer):
        if np.all(np.isfinite(Z)) and np.allclose(Z, np.round(Z)):
            Z = np.round(Z).astype(int)
        else:
            raise ValueError(f"modes must contain integer labels, got dtype={Z.dtype}")

    if K is None:
        K = int(np.nanmax(Z)) + 1

    dwell_times_per_mode = []
    gap_times_per_mode = []
    n_visits_per_mode = np.zeros(K, dtype=float)
    occ_per_mode = np.zeros(K, dtype=float)

    for k in range(K):
        mask = (Z == k)
        occ_per_mode[k] = mask.mean()

        starts, ends = _get_run_starts_ends(mask)

        if starts.size == 0:
            dwell_times_per_mode.append(np.array([], dtype=float))
            gap_times_per_mode.append(np.array([], dtype=float))
            continue

        dwells = (ends - starts + 1) * bin_size_s
        dwell_times_per_mode.append(dwells)
        n_visits_per_mode[k] = len(dwells)

        if starts.size > 1:
            gaps = (starts[1:] - ends[:-1] - 1) * bin_size_s
        else:
            gaps = np.array([], dtype=float)

        gap_times_per_mode.append(gaps)

    return (
        dwell_times_per_mode,
        gap_times_per_mode,
        n_visits_per_mode,
        occ_per_mode,
    )


def summarize_event_times(
    times_per_mode,
    prefix,
    qs=(0.10, 0.25, 0.50, 0.75, 0.90),
):
    K = len(times_per_mode)

    mean_emp = np.full(K, np.nan, dtype=float)
    median_emp = np.full(K, np.nan, dtype=float)
    std_emp = np.full(K, np.nan, dtype=float)
    cv_emp = np.full(K, np.nan, dtype=float)
    n_emp = np.zeros(K, dtype=float)
    quantiles_emp = np.full((K, len(qs)), np.nan, dtype=float)

    for k, arr in enumerate(times_per_mode):
        arr = np.asarray(arr, dtype=float)

        if arr.size == 0:
            continue

        mean_emp[k] = arr.mean()
        median_emp[k] = np.median(arr)
        std_emp[k] = arr.std()
        cv_emp[k] = std_emp[k] / (mean_emp[k] + 1e-12)
        n_emp[k] = arr.size
        quantiles_emp[k] = np.quantile(arr, qs)

    return {
        f"mean_{prefix}_emp": mean_emp,
        f"median_{prefix}_emp": median_emp,
        f"std_{prefix}_emp": std_emp,
        f"cv_{prefix}_emp": cv_emp,
        f"n_{prefix}_emp": n_emp,
        f"{prefix}_quantiles_emp": quantiles_emp,
    }

def compute_time_profile(
    modes,
    K=None,
    n_bins_profile=30,
    n_reps=10,
    rep_len=600,
    normalize_shape=True,
    eps=1e-12,
):
    """
    Compute state usage profile across trial time.

    Parameters
    ----------
    modes : 1D array of length n_reps * rep_len
    K : int or None
    n_bins_profile : int
        number of coarse time bins across one repeat
    n_reps : int
    rep_len : int
    normalize_shape : bool
        If True, normalize each state's profile to sum to 1 across bins.
        This emphasizes *when* the state occurs within a trial, rather than
        total occupancy magnitude.
    eps : float

    Returns
    -------
    prof : np.ndarray, shape (K, n_bins_profile)
        raw profile (mean occupancy per coarse bin across reps)
    prof_shape : np.ndarray, shape (K, n_bins_profile)
        row-normalized profile shape
    """
    Z = np.asarray(modes)

    if Z.ndim != 1:
        raise ValueError(f"modes must be 1D, got shape={Z.shape}")

    if not np.issubdtype(Z.dtype, np.integer):
        if np.all(np.isfinite(Z)) and np.allclose(Z, np.round(Z)):
            Z = np.round(Z).astype(int)
        else:
            raise ValueError(f"modes must contain integer labels, got dtype={Z.dtype}")

    if K is None:
        K = int(np.nanmax(Z)) + 1

    expected = n_reps * rep_len
    if Z.size != expected:
        raise ValueError(f"Expected size {expected}, got {Z.size}")

    Zr = Z.reshape(n_reps, rep_len)

    edges = np.linspace(0, rep_len, n_bins_profile + 1, dtype=int)
    prof = np.zeros((K, n_bins_profile), dtype=float)

    for k in range(K):
        mask = (Zr == k).astype(float)
        for b in range(n_bins_profile):
            s, e = edges[b], edges[b + 1]
            prof[k, b] = mask[:, s:e].mean()

    if normalize_shape:
        row_sum = prof.sum(axis=1, keepdims=True)
        prof_shape = np.divide(
            prof,
            row_sum + eps,
            out=np.zeros_like(prof),
            where=row_sum > 0,
        )
    else:
        prof_shape = prof.copy()

    return prof, prof_shape


def extract_empirical_temporal_features(
    modes,
    bin_size_s,
    K=None,
    qs=(0.10, 0.25, 0.50, 0.75, 0.90),
    n_bins_profile=30,
    n_reps=10,
    rep_len=600,
):
    """
    1D modes -> dwell/gap/visit/occupancy/time-profile summaries
    """
    (
        dwell_times_per_mode,
        gap_times_per_mode,
        n_visits_per_mode,
        occ_per_mode,
    ) = compute_dwell_and_gap_times(modes, K=K, bin_size_s=bin_size_s)

    dwell_summary = summarize_event_times(
        dwell_times_per_mode, prefix="dwell", qs=qs
    )
    gap_summary = summarize_event_times(
        gap_times_per_mode, prefix="gap", qs=qs
    )

    Z = np.asarray(modes)
    total_time_s = Z.shape[0] * bin_size_s
    visit_rate_emp = n_visits_per_mode / (total_time_s + 1e-12)

    time_profile_emp, time_profile_shape_emp = compute_time_profile(
        modes,
        K=K,
        n_bins_profile=n_bins_profile,
        n_reps=n_reps,
        rep_len=rep_len,
        normalize_shape=True,
    )

    return {
        "dwell_times_emp": dwell_times_per_mode,
        "gap_times_emp": gap_times_per_mode,
        "n_visits_emp": n_visits_per_mode,
        "visit_rate_emp": visit_rate_emp,
        "occupancy_emp": occ_per_mode,
        "time_profile_emp": time_profile_emp,
        "time_profile_shape_emp": time_profile_shape_emp,
        **dwell_summary,
        **gap_summary,
    }


# ============================================================
# Graph helpers
# ============================================================

def _canonical_edge_set(edges):
    """Undirected canonical edges: {(min(u,v), max(u,v)), ...}"""
    return set((int(min(u, v)), int(max(u, v))) for (u, v) in edges)


def _edges_to_adj(edges_set, N):
    """Build NxN adjacency (0/1) from canonical edge set."""
    adj = np.zeros((N, N), dtype=np.uint8)
    for u, v in edges_set:
        if 0 <= u < N and 0 <= v < N:
            adj[u, v] = 1
            adj[v, u] = 1
    return adj


# ============================================================
# Fingerprint extraction
# ============================================================

def extract_fingerprint(
    model,
    X=None,
    modes=None,
    bin_size_s=None,
    n_bins_profile=30,
    n_reps=10,
    rep_len=600,
):
    """
    Create fingerprint for each hidden state.
    Returns dict with structured features.
    """
    p1 = model.emission.p1_all
    edges_all = model.emission.edges_all
    A = model.A
    pi = model.pi

    K, N = p1.shape

    # -------------------------
    # theoretical dynamics
    # -------------------------
    Aii = np.diag(A)
    mean_dwell = 1.0 / (1.0 - Aii + 1e-12)

    A_off = A.copy()
    np.fill_diagonal(A_off, 0)
    row_sum = A_off.sum(axis=1, keepdims=True)
    A_off = np.divide(A_off, row_sum, out=np.zeros_like(A_off), where=row_sum > 0)
    H_off = -(A_off * np.log(A_off + 1e-12)).sum(axis=1)

    # -------------------------
    # structure
    # -------------------------
    deg_mean = np.zeros(K)
    deg_std = np.zeros(K)

    edge_sets = [None] * K
    adj_all = [None] * K
    n_edges = np.zeros(K, dtype=int)

    for k in range(K):
        edges = edges_all[k]
        Eset = _canonical_edge_set(edges)
        edge_sets[k] = Eset
        n_edges[k] = len(Eset)

        deg = np.zeros(N, dtype=float)
        for u, v in Eset:
            if 0 <= u < N and 0 <= v < N:
                deg[u] += 1
                deg[v] += 1

        deg_mean[k] = deg.mean()
        deg_std[k] = deg.std()
        adj_all[k] = _edges_to_adj(Eset, N)

    # -------------------------
    # occupancy from decoded z or fallback
    # -------------------------
    if X is not None and hasattr(model, "viterbi"):
        z = model.viterbi(X)
        counts = np.bincount(z.reshape(-1), minlength=K)
        occ = counts / counts.sum()
    else:
        occ = pi / (pi.sum() + 1e-12)

    fingerprint = {
        "p1": p1,
        "edges_all": edges_all,
        "edge_set": edge_sets,
        "adj": adj_all,
        "n_edges": n_edges,
        "deg_mean": deg_mean,
        "deg_std": deg_std,
        "Aii": Aii,
        "mean_dwell": mean_dwell,
        "H_off": H_off,
        "occupancy": occ,
        "A": A,
        "pi": pi,
    }

    if modes is not None and bin_size_s is not None:
        temporal_feat = extract_empirical_temporal_features(
            modes,
            bin_size_s,
            K=K,
            n_bins_profile=n_bins_profile,
            n_reps=n_reps,
            rep_len=rep_len,
        )
        fingerprint.update(temporal_feat)

    return fingerprint


# ============================================================
# Per-state split / merge helpers
# ============================================================

def split_fingerprint_by_state(fp):
    """
    Split full fingerprint dict into a list of per-state dicts.
    """
    K = fp["p1"].shape[0]
    state_fps = []

    for i in range(K):
        state_dict = {
            "state_id": i,
            "p1": np.asarray(fp["p1"][i], dtype=float),

            "deg_mean": float(fp["deg_mean"][i]),
            "deg_std": float(fp["deg_std"][i]),

            "Aii": float(fp["Aii"][i]),
            "mean_dwell": float(fp["mean_dwell"][i]),
            "H_off": float(fp["H_off"][i]),
            "occupancy": float(fp["occupancy"][i]),
        }

        # empirical occupancy / visits
        if "occupancy_emp" in fp:
            state_dict["occupancy_emp"] = float(fp["occupancy_emp"][i])

        if "n_visits_emp" in fp:
            state_dict["n_visits_emp"] = float(fp["n_visits_emp"][i])

        if "visit_rate_emp" in fp:
            state_dict["visit_rate_emp"] = float(fp["visit_rate_emp"][i])

        # empirical dwell
        if "mean_dwell_emp" in fp:
            state_dict["mean_dwell_emp"] = float(fp["mean_dwell_emp"][i])

        if "median_dwell_emp" in fp:
            state_dict["median_dwell_emp"] = float(fp["median_dwell_emp"][i])

        if "std_dwell_emp" in fp:
            state_dict["std_dwell_emp"] = float(fp["std_dwell_emp"][i])

        if "cv_dwell_emp" in fp:
            state_dict["cv_dwell_emp"] = float(fp["cv_dwell_emp"][i])

        if "n_dwell_emp" in fp:
            state_dict["n_dwell_emp"] = float(fp["n_dwell_emp"][i])

        if "dwell_quantiles_emp" in fp:
            state_dict["dwell_quantiles_emp"] = np.asarray(
                fp["dwell_quantiles_emp"][i], dtype=float
            )

        if "dwell_times_emp" in fp:
            state_dict["dwell_times_emp"] = np.asarray(
                fp["dwell_times_emp"][i], dtype=float
            )

        # empirical gap
        if "mean_gap_emp" in fp:
            state_dict["mean_gap_emp"] = float(fp["mean_gap_emp"][i])

        if "median_gap_emp" in fp:
            state_dict["median_gap_emp"] = float(fp["median_gap_emp"][i])

        if "std_gap_emp" in fp:
            state_dict["std_gap_emp"] = float(fp["std_gap_emp"][i])

        if "cv_gap_emp" in fp:
            state_dict["cv_gap_emp"] = float(fp["cv_gap_emp"][i])

        if "n_gap_emp" in fp:
            state_dict["n_gap_emp"] = float(fp["n_gap_emp"][i])

        if "gap_quantiles_emp" in fp:
            state_dict["gap_quantiles_emp"] = np.asarray(
                fp["gap_quantiles_emp"][i], dtype=float
            )

        if "gap_times_emp" in fp:
            state_dict["gap_times_emp"] = np.asarray(
                fp["gap_times_emp"][i], dtype=float
            )

        # time profile
        if "time_profile_emp" in fp:
            state_dict["time_profile_emp"] = np.asarray(
                fp["time_profile_emp"][i], dtype=float
            )

        if "time_profile_shape_emp" in fp:
            state_dict["time_profile_shape_emp"] = np.asarray(
                fp["time_profile_shape_emp"][i], dtype=float
            )

        # graph
        if "n_edges" in fp:
            state_dict["n_edges"] = int(fp["n_edges"][i])

        if "edge_set" in fp:
            state_dict["edge_set"] = fp["edge_set"][i]

        if "adj" in fp:
            state_dict["adj"] = fp["adj"][i]

        # transition row / pi
        if "A" in fp:
            state_dict["A_row"] = np.asarray(fp["A"][i], dtype=float)

        if "pi" in fp:
            state_dict["pi"] = float(fp["pi"][i])

        state_fps.append(state_dict)

    return state_fps






def _moving_average_1d(x, win=9):
    x = np.asarray(x, float)
    if win <= 1:
        return x.copy()
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(xp, kernel, mode="valid")


def _run_lengths_1d(x):
    x = np.asarray(x, dtype=int)
    if x.size == 0:
        return np.array([], dtype=int)
    d = np.diff(np.r_[0, x, 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return ends - starts


def _coarse_block_means(x, edges):
    x = np.asarray(x, float)
    vals = []
    for s, e in zip(edges[:-1], edges[1:]):
        vals.append(0.0 if e <= s else x[s:e].mean())
    return np.asarray(vals, float)


def _top_k_peak_positions(x, k=3, min_distance=8):
    x = np.asarray(x, float)
    T = len(x)
    peaks = []

    for i in range(1, T - 1):
        if x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            peaks.append((x[i], i))

    peaks.sort(reverse=True, key=lambda z: z[0])

    chosen = []
    for val, idx in peaks:
        if all(abs(idx - j) >= min_distance for j in chosen):
            chosen.append(idx)
        if len(chosen) == k:
            break

    chosen = sorted(chosen)
    out = np.full(k, -1.0, dtype=float)
    for i, idx in enumerate(chosen):
        out[i] = idx / max(T - 1, 1)
    return out

def _dilate_binary_1d(x, radius=1):
    x = np.asarray(x, dtype=bool)
    out = x.copy()
    for r in range(1, radius + 1):
        y = x.copy()
        y[:-r] |= x[r:]
        y[r:] |= x[:-r]
        out |= y
    return out.astype(float)


def _state_block_counts(Z, state_k, edges):
    """
    Z: shape (n_reps, T)
    返回:
        counts: shape (n_reps, B), 每个rep、每个block里 state_k 出现的次数
    """
    Z = np.asarray(Z)
    n_reps, T = Z.shape
    B = len(edges) - 1
    counts = np.zeros((n_reps, B), dtype=float)

    for r in range(n_reps):
        for b, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
            counts[r, b] = np.sum(Z[r, s:e] == state_k)

    return counts







# def build_state_features(
#     fp,
#     z_reps=None,
#     qs_p1=(0.1,0.25, 0.5, 0.75, 0.9),
#     eps=1e-12,
#     w_emission=1,
#     w_temporal=1,
#     w_profile=1,
#     w_overlap=1,
#     w_soft_seg=1,
#     n_coarse_bins=12,
#     smooth_win=11,
#     anchor_quantile=0.70,
#     # 新增 PCA 参数
#     w_emission_pca=0,
#     n_pca_components=5,
# ):
#     p1 = np.asarray(fp["p1"], float)
#     K = p1.shape[0]

#     blocks = []
#     names = []

#     def _append_block(X_block, block_names, weight):
#         if weight == 0 or X_block is None or X_block.shape[1] == 0:
#             return
#         X_block = np.asarray(X_block, float)
#         X_block = X_block / np.sqrt(X_block.shape[1])
#         blocks.append(weight * X_block)
#         names.extend(block_names)

#     # --------------------------------------------------
#     # 1) emission
#     # --------------------------------------------------
#     qfeat = np.stack([np.quantile(p1[k], qs_p1) for k in range(K)], axis=0)
#     mean_p1 = p1.mean(axis=1, keepdims=True)
#     sd_p1 = p1.std(axis=1, ddof=0, keepdims=True)

#     X_emission = np.concatenate([qfeat, mean_p1, sd_p1], axis=1)
#     emission_names = [f"p1_q{int(q*100)}" for q in qs_p1] + ["p1_mean", "p1_sd"]
#     _append_block(X_emission, emission_names, w_emission)

#     # --------------------------------------------------
#     # 1b) emission PCA (新增)
#     # --------------------------------------------------
#     if w_emission_pca != 0 and n_pca_components > 0:
#         # p1 形状 (K, N)，N 为每个状态观测点数
#         p1_centered = p1 - p1.mean(axis=0, keepdims=True)
#         U, s, Vt = np.linalg.svd(p1_centered, full_matrices=False)
#         comps = Vt[:n_pca_components]                     # (n_comp, N)
#         pca_scores = p1_centered @ comps.T                # (K, n_comp)
#         pca_names = [f"p1_pca_{i+1}" for i in range(n_pca_components)]
#         _append_block(pca_scores, pca_names, w_emission_pca)

#     # --------------------------------------------------
#     # 2) temporal
#     # add log_gap_q50_emp
#     # --------------------------------------------------
#     temporal_blocks = []
#     temporal_names = []

#     if "dwell_quantiles_emp" in fp:
#         dq = np.asarray(fp["dwell_quantiles_emp"], float)
#         temporal_blocks.append(np.log(dq[:, [2, 4]] + eps))
#         temporal_names.extend(["log_dwell_q50_emp", "log_dwell_q90_emp"])

#     if "gap_quantiles_emp" in fp:
#         gq = np.asarray(fp["gap_quantiles_emp"], float)
#         temporal_blocks.append(np.log(gq[:, [2, 4]] + eps))
#         temporal_names.extend(["log_gap_q50_emp", "log_gap_q90_emp"])

#     if "visit_rate_emp" in fp:
#         vr = np.asarray(fp["visit_rate_emp"], float)[:, None]
#         temporal_blocks.append(np.log(vr + eps))
#         temporal_names.append("log_visit_rate_emp")

#     if temporal_blocks:
#         X_temporal = np.concatenate(temporal_blocks, axis=1)
#         _append_block(X_temporal, temporal_names, w_temporal)

#     # --------------------------------------------------
#     # 3) time profile
#     # --------------------------------------------------
#     if "time_profile_shape_emp" in fp:
#         tp = np.asarray(fp["time_profile_shape_emp"], float)
#         profile_names = [f"time_profile_shape_bin_{i}" for i in range(tp.shape[1])]
#         _append_block(tp, profile_names, w_profile)
#     elif "time_profile_emp" in fp:
#         tp = np.asarray(fp["time_profile_emp"], float)
#         profile_names = [f"time_profile_bin_{i}" for i in range(tp.shape[1])]
#         _append_block(tp, profile_names, w_profile)

#     # --------------------------------------------------
#     # 4) overlap: only anchor_bin + anchor_center
#     # --------------------------------------------------
#     if z_reps is not None and w_overlap != 0:
#         Z = np.asarray(z_reps)
#         if Z.ndim != 2:
#             raise ValueError(f"z_reps must be 2D, got shape {Z.shape}")

#         n_reps, T = Z.shape
#         edges = np.linspace(0, T, n_coarse_bins + 1).astype(int)

#         overlap_feats = []
#         overlap_names = [f"anchor_bin_{b}" for b in range(n_coarse_bins)] + ["anchor_center"]

#         for k in range(K):
#             mask = (Z == k).astype(float)
#             avg_profile = mask.mean(axis=0)
#             smooth_profile = _moving_average_1d(avg_profile, win=smooth_win)

#             coarse_occ = []
#             for b in range(n_coarse_bins):
#                 s, e = edges[b], edges[b + 1]
#                 coarse_occ.append(0.0 if e <= s else smooth_profile[s:e].mean())
#             coarse_occ = np.asarray(coarse_occ, float)

#             if np.all(coarse_occ <= eps):
#                 anchor = np.zeros(n_coarse_bins, dtype=float)
#             else:
#                 thr = np.quantile(coarse_occ, anchor_quantile)
#                 anchor = (coarse_occ >= max(thr, eps)).astype(float)
#                 if anchor.sum() == 0:
#                     anchor[np.argmax(coarse_occ)] = 1.0

#             bin_centers = (np.arange(n_coarse_bins) + 0.5) / n_coarse_bins
#             if anchor.sum() <= 0:
#                 anchor_center = 0.0
#             else:
#                 p = anchor / anchor.sum()
#                 anchor_center = (p * bin_centers).sum()

#             overlap_feats.append(np.concatenate([anchor, [anchor_center]]))

#         X_overlap = np.asarray(overlap_feats, float)
#         _append_block(X_overlap, overlap_names, w_overlap)

#     # --------------------------------------------------
#     # 5) soft segmentation: only seg_time_center
#     # --------------------------------------------------
#     if z_reps is not None and w_soft_seg != 0:
#         Z = np.asarray(z_reps)
#         n_reps, T = Z.shape
#         tgrid = np.arange(T, dtype=float)

#         seg_feats = []
#         seg_names = ["seg_time_center"]

#         for k in range(K):
#             mask = (Z == k).astype(float)
#             avg_profile = mask.mean(axis=0)
#             prof_sum = avg_profile.sum()

#             if prof_sum <= eps:
#                 time_center = 0.0
#             else:
#                 p = avg_profile / (prof_sum + eps)
#                 mean_t = (p * tgrid).sum()
#                 time_center = mean_t / max(T - 1, 1)

#             seg_feats.append([time_center])

#         X_seg = np.asarray(seg_feats, float)
#         _append_block(X_seg, seg_names, w_soft_seg)

#     if len(blocks) == 0:
#         raise ValueError("No feature blocks were added.")

#     X = np.concatenate(blocks, axis=1)
#     return X, names
def build_state_features(
    fp,
    z_reps=None,
    qs_p1=(0.1, 0.25, 0.5, 0.75, 0.9),
    eps=1e-12,
    w_emission=1.0,
    w_temporal=1.0,
    w_overlap=1.0,
    w_emission_pca=0.0,
    n_pca_components=5,
    stim_block_len=None,   # 例如 300
    block_mean_thr=0.25,
    absent_code=2.0,
):
    p1 = np.asarray(fp["p1"], float)
    K = p1.shape[0]

    blocks = []
    names = []

    def _append_block(X_block, block_names, weight):
        if weight == 0 or X_block is None or X_block.shape[1] == 0:
            return
        X_block = np.asarray(X_block, float)
        X_block = X_block / np.sqrt(X_block.shape[1])
        blocks.append(weight * X_block)
        names.extend(block_names)

    # --------------------------------------------------
    # 1) emission
    # --------------------------------------------------
    qfeat = np.stack([np.quantile(p1[k], qs_p1) for k in range(K)], axis=0)
    mean_p1 = p1.mean(axis=1, keepdims=True)
    sd_p1 = p1.std(axis=1, ddof=0, keepdims=True)

    X_emission = np.concatenate([qfeat, mean_p1, sd_p1], axis=1)
    emission_names = [f"p1_q{int(q * 100)}" for q in qs_p1] + ["p1_mean", "p1_sd"]
    _append_block(X_emission, emission_names, w_emission)

    # --------------------------------------------------
    # 1b) emission PCA
    # --------------------------------------------------
    if w_emission_pca != 0 and n_pca_components > 0:
        p1_centered = p1 - p1.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(p1_centered, full_matrices=False)
        n_comp = min(n_pca_components, Vt.shape[0])
        comps = Vt[:n_comp]
        pca_scores = p1_centered @ comps.T
        pca_names = [f"p1_pca_{i+1}" for i in range(n_comp)]
        _append_block(pca_scores, pca_names, w_emission_pca)

    # --------------------------------------------------
    # 2) temporal
    # 保留你现在真正用的几个核心 temporal features
    # --------------------------------------------------
    temporal_blocks = []
    temporal_names = []

    if "dwell_quantiles_emp" in fp:
        dq = np.asarray(fp["dwell_quantiles_emp"], float)

        # q50
        temporal_blocks.append(np.log(dq[:, [2]] + eps))
        temporal_names.append("log_dwell_q50_emp")

        # IQR = q75 - q25
        if dq.shape[1] >= 4:
            dwell_iqr = np.log((dq[:, 3] - dq[:, 1]) + eps)[:, None]
            temporal_blocks.append(dwell_iqr)
            temporal_names.append("log_dwell_iqr_emp")

    if "A" in fp:
        A = np.asarray(fp["A"], float)

        # self-transition
        Aii = np.clip(np.diag(A), eps, 1 - eps)[:, None]
        temporal_blocks.append(np.log(Aii / (1 - Aii)))
        temporal_names.append("logit_Aii")

        # off-diagonal transition entropy
        A_off = A.copy()
        np.fill_diagonal(A_off, 0.0)
        row_sum = A_off.sum(axis=1, keepdims=True)
        P_off = np.divide(A_off, row_sum, out=np.zeros_like(A_off), where=row_sum > 0)

        H_off = -(P_off * np.log(P_off + eps)).sum(axis=1, keepdims=True)
        temporal_blocks.append(H_off)
        temporal_names.append("H_off")

    if temporal_blocks:
        X_temporal = np.concatenate(temporal_blocks, axis=1)
        _append_block(X_temporal, temporal_names, w_temporal)

    # --------------------------------------------------
    # 3) overlap
    # thresholded stimulus-block position code
    #
    # 出现 -> 0
    # 不出现 -> absent_code
    # --------------------------------------------------
    if z_reps is not None and w_overlap != 0:
        Z = np.asarray(z_reps)
        if Z.ndim != 2:
            raise ValueError(f"z_reps must be 2D, got shape {Z.shape}")

        n_reps, T = Z.shape

        if stim_block_len is None or stim_block_len <= 0:
            raise ValueError("stim_block_len must be a positive integer")

        if T % stim_block_len != 0:
            raise ValueError(
                f"T={T} is not divisible by stim_block_len={stim_block_len}"
            )

        n_blocks = T // stim_block_len

        overlap_feats = []
        overlap_names = [f"stim{b+1}_poscode" for b in range(n_blocks)]

        for k in range(K):
            props = np.zeros((n_reps, n_blocks), dtype=float)

            for r in range(n_reps):
                for b in range(n_blocks):
                    s = b * stim_block_len
                    e = (b + 1) * stim_block_len
                    props[r, b] = np.mean(Z[r, s:e] == k)

            block_mean = props.mean(axis=0)

            mx = block_mean.max()
            if mx > eps:
                rel_mean = block_mean / (mx + eps)
            else:
                rel_mean = np.zeros_like(block_mean)

            present = rel_mean >= block_mean_thr
            poscode = np.where(present, 0.0, absent_code)

            overlap_feats.append(poscode)

        X_overlap = np.asarray(overlap_feats, float)
        _append_block(X_overlap, overlap_names, w_overlap)

    if len(blocks) == 0:
        raise ValueError("No feature blocks were added.")

    X = np.concatenate(blocks, axis=1)
    return X, names


