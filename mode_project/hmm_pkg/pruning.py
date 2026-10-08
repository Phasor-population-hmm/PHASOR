
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import copy
import numpy as np


def viterbi_occupancy(z: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute hard occupancy from a Viterbi path.

    Parameters
    ----------
    z : array, shape (T,)
        Viterbi state sequence (int labels in [0, K-1]).
    K : int
        Number of states.

    Returns
    -------
    occ : array, shape (K,)
        Fraction of time assigned to each state.
    counts : array, shape (K,)
        Raw counts per state.
    """
    z = np.asarray(z)
    counts = np.bincount(z, minlength=K)
    occ = counts / len(z)
    return occ, counts


def select_states_by_occupancy(
    occ: np.ndarray,
    counts: np.ndarray,
    min_occ: float = 0.005,
    min_count: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decide which states to keep/drop based on occupancy thresholds.

    Keep state k iff:
        occ[k] >= min_occ AND counts[k] >= min_count

    Returns
    -------
    keep_idx : array of kept state indices
    drop_idx : array of dropped state indices
    """
    occ = np.asarray(occ, dtype=float)
    counts = np.asarray(counts, dtype=int)

    keep = (occ >= float(min_occ)) & (counts >= int(min_count))
    keep_idx = np.where(keep)[0]
    drop_idx = np.where(~keep)[0]
    return keep_idx, drop_idx


def prune_chowliu_model(model, keep_idx: Iterable[int]):
    """
    Prune a trained Chow–Liu HMM by keeping only a subset of states.

    Notes
    -----
    - This function does NOT retrain the model; it only slices parameters.
    - Use the output as an initialization for a final EM retraining run.

    Requirements on `model`:
    - model.pi: (K,)
    - model.A: (K,K)
    - model.emission.p1_all: (K,N)
    - model.emission.edges_all: list/array length K
    - model.emission.pair_probs_all: list/array length K
    """
    keep_idx = np.asarray(list(keep_idx), dtype=int)
    newK = len(keep_idx)
    if newK <= 0:
        raise ValueError("keep_idx is empty; cannot prune to K=0.")

    model_new = copy.deepcopy(model)

    # --- pi ---
    pi_new = model.pi[keep_idx].copy()
    pi_sum = pi_new.sum()
    if pi_sum <= 0:
        raise ValueError("Sum of kept pi is non-positive; invalid keep_idx.")
    pi_new /= pi_sum

    # --- A ---
    A_new = model.A[np.ix_(keep_idx, keep_idx)].copy()
    row_sums = A_new.sum(axis=1, keepdims=True)
    # Avoid division by zero (should not happen in normal training)
    row_sums[row_sums == 0] = 1.0
    A_new /= row_sums

    # --- emissions ---
    model_new.emission.p1_all = model.emission.p1_all[keep_idx].copy()

    edges_all = model.emission.edges_all
    pair_probs_all = model.emission.pair_probs_all

    if isinstance(edges_all, list):
        model_new.emission.edges_all = [edges_all[k] for k in keep_idx]
    else:
        model_new.emission.edges_all = edges_all[keep_idx].copy()

    if isinstance(pair_probs_all, list):
        model_new.emission.pair_probs_all = [pair_probs_all[k] for k in keep_idx]
    else:
        model_new.emission.pair_probs_all = pair_probs_all[keep_idx].copy()

    # --- write back ---
    model_new.K = newK
    model_new.pi = pi_new
    model_new.A = A_new
    model_new.emission.K = newK

    return model_new
