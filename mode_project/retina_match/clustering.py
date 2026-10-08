import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment



def cluster_states(X, k):
    Z = linkage(X, method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust") - 1
    return labels


def cluster_centroids_present(X, labels, method="medoid"):
    """
    Return a representative vector for each cluster.

    method:
        "medoid"  -> pick the real point minimizing sum of distances (recommended)
        "mean"    -> original behavior
        "median"  -> coordinate-wise median
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    reps = []

    for u in uniq:
        Xu = X[labels == u]

        if method == "mean":
            rep = Xu.mean(axis=0)

        elif method == "median":
            rep = np.median(Xu, axis=0)

        elif method == "medoid":
            if Xu.shape[0] == 1:
                rep = Xu[0]
            else:
                D = np.linalg.norm(Xu[:, None, :] - Xu[None, :, :], axis=2)
                idx = np.argmin(D.sum(axis=1))
                rep = Xu[idx]

        else:
            raise ValueError("method must be 'mean', 'median', or 'medoid'")

        reps.append(rep)

    C = np.vstack(reps)
    return C, uniq


def match_centroids(CA, CB):
    D = np.linalg.norm(CA[:, None, :] - CB[None, :, :], axis=2)

    r, c = linear_sum_assignment(D)

    pairs = [(int(i), int(j), float(D[i, j])) for i, j in zip(r, c)]
    pairs.sort(key=lambda x: x[2])

    return pairs, D


def total_assignment_cost(pairs, agg="mean"):
    costs = np.array([c for _, _, c in pairs], dtype=float)

    if agg == "mean":
        return float(costs.mean())
    if agg == "sum":
        return float(costs.sum())
    if agg == "median":
        return float(np.median(costs))

    raise ValueError("agg must be one of: 'mean', 'sum', 'median'")