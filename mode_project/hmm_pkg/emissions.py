import numpy as np


class BernoulliEmission:
    def __init__(self, K, N, random_state=0, init_from_mean=None, eps=1e-8):
        self.K = K
        self.N = N
        self.eps = eps
        rng = np.random.default_rng(random_state)
        if init_from_mean is None:
            E = rng.uniform(0.25, 0.75, size=(K, N))
        else:
            mean_rate = np.asarray(init_from_mean, dtype=float)
            E = np.tile(mean_rate[None, :], (K, 1))
            E += 0.1 * rng.standard_normal((K, N))
        self.E = np.clip(E, eps, 1.0 - eps)

    def log_B(self, X):
        X = np.asarray(X, dtype=float)
        E = np.clip(self.E, self.eps, 1.0 - self.eps)
        return X @ np.log(E).T + (1.0 - X) @ np.log(1.0 - E).T

    def m_step(self, X, gamma):
        X = np.asarray(X, dtype=float)
        gamma = np.asarray(gamma, dtype=float)
        denom = gamma.sum(axis=0)[:, None] + self.eps
        E = (gamma.T @ X) / denom
        self.E = np.clip(E, self.eps, 1.0 - self.eps)


def _joint_from_marginals_and_covariance(mi, mj, covariance, eps):
    c11 = mi * mj + covariance
    lower = max(eps, mi + mj - 1.0 + eps)
    upper = min(mi, mj) - eps
    if lower > upper:
        c11 = min(max(mi * mj, max(0.0, mi + mj - 1.0)), min(mi, mj))
    else:
        c11 = float(np.clip(c11, lower, upper))
    return np.array(
        [
            [1.0 - mi - mj + c11, mj - c11],
            [mi - c11, c11],
        ],
        dtype=float,
    )


def _mutual_information(table, eps):
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    return float(np.sum(table * np.log((table + eps) / (row @ col + eps))))


def fit_chow_liu_weighted(X, weights, eps=1e-12, eta=0.002, alpha=0.5):
    X = np.asarray(X, dtype=int)
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if X.ndim != 2:
        raise ValueError("X must have shape (T, N)")
    if w.ndim != 1 or w.shape[0] != X.shape[0]:
        raise ValueError("weights must have shape (T,)")
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")

    _, N = X.shape
    W = float(w.sum())
    if W <= 0.0:
        p1 = np.clip(X.mean(axis=0), eps, 1.0 - eps)
        return p1, [], {}

    p1 = (w @ X + alpha) / (W + 2.0 * alpha)
    p1 = np.clip(p1, eps * 10.0, 1.0 - eps * 10.0)

    mi_matrix = np.zeros((N, N), dtype=float)
    candidate_tables = {}
    joint_denom = W + 4.0 * alpha

    for i in range(N):
        xi = X[:, i]
        mi = float(p1[i])
        for j in range(i + 1, N):
            xj = X[:, j]
            mj = float(p1[j])
            n11 = float(w @ (xi * xj))
            raw_cov = (n11 + alpha) / joint_denom - mi * mj
            if abs(raw_cov) <= eta:
                regularized_cov = 0.0
            else:
                regularized_cov = np.sign(raw_cov) * (abs(raw_cov) - eta)
            table = _joint_from_marginals_and_covariance(mi, mj, regularized_cov, eps)
            candidate_tables[(i, j)] = table
            mi_value = _mutual_information(table, eps)
            mi_matrix[i, j] = mi_matrix[j, i] = mi_value

    if N <= 1:
        return p1, [], {}

    selected = np.zeros(N, dtype=bool)
    selected[0] = True
    edges = []
    for _ in range(N - 1):
        best_i = best_j = -1
        best_weight = -np.inf
        for i in range(N):
            if not selected[i]:
                continue
            for j in range(N):
                if selected[j]:
                    continue
                if mi_matrix[i, j] > best_weight:
                    best_i, best_j = i, j
                    best_weight = mi_matrix[i, j]
        if best_i < 0:
            raise RuntimeError("failed to construct a spanning tree")
        selected[best_j] = True
        edge = (min(best_i, best_j), max(best_i, best_j))
        edges.append(edge)

    pair_probs = {edge: candidate_tables[edge] for edge in edges}
    return p1, edges, pair_probs


def compute_log_emission_all_modes(X, p1_all, edges_all, pair_probs_all, eps=1e-12, eta=None):
    X = np.asarray(X, dtype=int)
    T, _ = X.shape
    K = p1_all.shape[0]
    log_B = np.zeros((T, K), dtype=float)
    for k in range(K):
        p1 = np.clip(np.asarray(p1_all[k], dtype=float), eps, 1.0 - eps)
        log_p1 = np.log(p1)
        log_p0 = np.log(1.0 - p1)
        score = X @ log_p1 + (1 - X) @ log_p0
        for i, j in edges_all[k]:
            table = np.asarray(pair_probs_all[k][(i, j)], dtype=float)
            xi = X[:, i]
            xj = X[:, j]
            score += (
                np.log(table[xi, xj])
                - np.where(xi == 1, log_p1[i], log_p0[i])
                - np.where(xj == 1, log_p1[j], log_p0[j])
            )
        log_B[:, k] = score
    return log_B


class ChowLiuEmission:
    def __init__(self, K, N, random_state=0, init_from_mean=None, eps=1e-8, eta=0.002, alpha=0.5):
        self.K = K
        self.N = N
        self.eps = eps
        self.eta = float(eta)
        self.alpha = float(alpha)
        if self.eta < 0.0:
            raise ValueError("eta must be non-negative")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        rng = np.random.default_rng(random_state)
        if init_from_mean is None:
            p1_all = rng.uniform(0.1, 0.9, size=(K, N))
        else:
            mean_rate = np.asarray(init_from_mean, dtype=float)
            p1_all = np.tile(mean_rate[None, :], (K, 1))
            p1_all += 0.1 * rng.standard_normal((K, N))
        self.p1_all = np.clip(p1_all, 1e-3, 1.0 - 1e-3)
        self.edges_all = [[] for _ in range(K)]
        self.pair_probs_all = [{} for _ in range(K)]

    def log_B(self, X):
        return compute_log_emission_all_modes(
            X, self.p1_all, self.edges_all, self.pair_probs_all, eps=self.eps
        )

    def m_step(self, X, gamma):
        X = np.asarray(X, dtype=int)
        gamma = np.asarray(gamma, dtype=float)
        if gamma.shape != (X.shape[0], self.K):
            raise ValueError("gamma must have shape (T, K)")
        new_p1_all = np.empty_like(self.p1_all)
        new_edges_all = []
        new_pair_probs_all = []
        for k in range(self.K):
            p1_k, edges_k, pair_probs_k = fit_chow_liu_weighted(
                X, gamma[:, k], eps=self.eps, eta=self.eta, alpha=self.alpha
            )
            new_p1_all[k] = p1_k
            new_edges_all.append(edges_k)
            new_pair_probs_all.append(pair_probs_k)
        self.p1_all = new_p1_all
        self.edges_all = new_edges_all
        self.pair_probs_all = new_pair_probs_all
