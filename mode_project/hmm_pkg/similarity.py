
from __future__ import annotations

from typing import Optional, Iterable, List, Tuple
import numpy as np


def edges_to_set_safe(edges) -> Optional[set]:
    """
    Convert an edge list to a set of undirected edges.

    Parameters
    ----------
    edges : (D-1, 2) array-like, list of pairs, or None

    Returns
    -------
    edge_set : set of (min(i,j), max(i,j)) or None if edges is None
    """
    if edges is None:
        return None

    e = np.asarray(edges)
    s = set()
    for a, b in e:
        a = int(a); b = int(b)
        if a == b:
            continue
        s.add((a, b) if a < b else (b, a))
    return s


def edge_share_matrix_safe(edges_all) -> np.ndarray:
    """
    Compute KxK matrix of edge overlap fractions for Chow–Liu trees.

    M[i,j] = |E_i ∩ E_j| / |E_i|
    Uses NaN when either state has no valid tree (edges=None or empty).

    Note: This definition can be non-symmetric if |E_i| differs across i,
    but for valid trees |E_i| is typically D-1 and M is ~symmetric.
    """
    K = len(edges_all)
    edge_sets = [edges_to_set_safe(edges_all[k]) for k in range(K)]

    M = np.full((K, K), np.nan, dtype=float)

    for i in range(K):
        Ei = edge_sets[i]
        if Ei is None or len(Ei) == 0:
            continue
        denom = float(len(Ei))

        for j in range(K):
            Ej = edge_sets[j]
            if Ej is None or len(Ej) == 0:
                continue
            inter = len(Ei & Ej)
            M[i, j] = inter / denom

    return M


def corr_matrix_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Correlation matrix between rows of X.
    X: (K, N) -> returns (K, K)
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)
    Xc = Xc / (Xc.std(axis=1, keepdims=True) + eps)
    return (Xc @ Xc.T) / X.shape[1]


def nearest_neighbor_stats(S: np.ndarray) -> list[tuple[int, int, float]]:
    """
    For each i, find j != i that maximizes S[i,j].
    Returns a list of (i, j, S[i,j]) sorted descending by similarity.
    """
    K = S.shape[0]
    nn = []
    for i in range(K):
        s = np.array(S[i], copy=True)
        s[i] = -np.inf  # exclude self
        j = int(np.nanargmax(s))  # nan-safe
        nn.append((i, j, float(S[i, j])))
    nn_sorted = sorted(nn, key=lambda x: x[2], reverse=True)
    return nn_sorted
