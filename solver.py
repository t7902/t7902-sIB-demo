"""Agglomerative Information Bottleneck (aIB), Sequential Information Bottleneck (sIB),
and X->T->Y mapping solvers.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from data import compute_cluster_distribution
from evaluator import entropy, js_divergence, mutual_information


def agglomerative_ib(
    p_x: np.ndarray,
    p_y_given_x: np.ndarray,
    n_clusters: int = 10,
    record_full_hierarchy: bool = True,
) -> Dict[str, Any]:
    """Run Agglomerative Information Bottleneck (aIB) bottom-up hard clustering.

    Starts with each x in its own singleton cluster and greedily merges the pair
    of clusters (t_i, t_j) that minimizes the information loss:
        Delta I(t_i, t_j) = (p(t_i) + p(t_j)) * JSD_{pi_i, pi_j}(p(y|t_i), p(y|t_j))

    Args:
        p_x: Marginal probability P(X) of shape (n_x,).
        p_y_given_x: Conditional probability P(Y|X) of shape (n_x, n_y).
        n_clusters: Target number of clusters K.
        record_full_hierarchy: If True, continue merging down to K=1 to record the full
            information curve, while returning the partition at K=n_clusters.

    Returns:
        Dictionary with 'labels' (at K=n_clusters), 'i_ty', and 'history'.
    """
    n_x = len(p_x)
    if not (1 <= n_clusters <= n_x):
        raise ValueError(f"n_clusters must be in [1, {n_x}], got {n_clusters}.")

    cluster_probs = p_x.astype(np.float64).copy()
    cluster_py = p_y_given_x.astype(np.float64).copy()
    active = np.ones(n_x, dtype=bool)
    labels = np.arange(n_x, dtype=np.int64)

    # Precompute upper-triangular pairwise merger costs.
    cost_matrix = np.full((n_x, n_x), np.inf, dtype=np.float64)
    for i in range(n_x - 1):
        p_sum = cluster_probs[i] + cluster_probs[i + 1 :]
        pi1 = cluster_probs[i] / p_sum
        pi2 = cluster_probs[i + 1 :] / p_sum
        jsd = js_divergence(cluster_py[i], cluster_py[i + 1 :], pi1, pi2)
        cost_matrix[i, i + 1 :] = p_sum * jsd

    stop_k = 1 if record_full_hierarchy else n_clusters
    target_labels = None
    target_i_ty = None
    history: List[Dict[str, float]] = []

    current_i_ty = mutual_information(p_x[:, None] * p_y_given_x)
    history.append(
        {"k": n_x, "I(T;Y)": current_i_ty, "H(T)": entropy(cluster_probs), "delta_I": 0.0}
    )
    if n_clusters == n_x:
        target_labels = labels.copy()
        target_i_ty = current_i_ty

    for k in range(n_x, stop_k, -1):
        flat_idx = int(np.argmin(cost_matrix))
        i, j = divmod(flat_idx, n_x)
        merge_cost = float(cost_matrix[i, j])

        # Merge cluster j into cluster i.
        p_new = cluster_probs[i] + cluster_probs[j]
        cluster_py[i] = (
            cluster_probs[i] * cluster_py[i] + cluster_probs[j] * cluster_py[j]
        ) / p_new
        cluster_probs[i] = p_new
        active[j] = False
        labels[labels == j] = i

        current_i_ty = max(0.0, current_i_ty - merge_cost)
        next_k = k - 1
        history.append(
            {
                "k": next_k,
                "I(T;Y)": current_i_ty,
                "H(T)": entropy(cluster_probs[active]),
                "delta_I": merge_cost,
            }
        )

        if next_k == n_clusters:
            _, target_labels = np.unique(labels, return_inverse=True)
            # Recompute exact I(T;Y) at target K to avoid cumulative rounding drift.
            _, _, p_ty = compute_cluster_distribution(
                target_labels, p_x, p_y_given_x, n_clusters=n_clusters
            )
            target_i_ty = mutual_information(p_ty)

        # Invalidate j and update all pairs involving i.
        cost_matrix[j, :] = np.inf
        cost_matrix[:, j] = np.inf
        cost_matrix[i, :] = np.inf
        cost_matrix[:, i] = np.inf

        other_idx = np.where(active)[0]
        other_idx = other_idx[other_idx != i]
        if len(other_idx) > 0:
            p_sum = cluster_probs[i] + cluster_probs[other_idx]
            pi1 = cluster_probs[i] / p_sum
            pi2 = cluster_probs[other_idx] / p_sum
            costs = p_sum * js_divergence(
                cluster_py[i], cluster_py[other_idx], pi1, pi2
            )
            lo = other_idx < i
            hi = ~lo
            cost_matrix[other_idx[lo], i] = costs[lo]
            cost_matrix[i, other_idx[hi]] = costs[hi]

    return {
        "labels": target_labels,
        "i_ty": target_i_ty,
        "history": history,
    }


def _random_partition(n_x: int, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Create a random deterministic partition of n_x items into n_clusters non-empty clusters."""
    perm = rng.permutation(n_x)
    labels = np.empty(n_x, dtype=np.int64)
    labels[perm[:n_clusters]] = np.arange(n_clusters)
    if n_x > n_clusters:
        labels[perm[n_clusters:]] = rng.integers(0, n_clusters, size=n_x - n_clusters)
    return labels


def sequential_ib(
    p_x: np.ndarray,
    p_y_given_x: np.ndarray,
    n_clusters: int = 10,
    max_iter: int = 100,
    n_init: int = 10,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Run Sequential Information Bottleneck (sIB) hard clustering.

    Randomly initializes a deterministic K-partition, then iteratively draws each
    sample x out of its current cluster and reassigns it to the cluster t* that
    maximizes I(T; Y) (equivalently minimizes the merger cost d_M(x, t)), until
    no sample changes its cluster assignment.

    Args:
        p_x: Marginal probability P(X) of shape (n_x,).
        p_y_given_x: Conditional probability P(Y|X) of shape (n_x, n_y).
        n_clusters: Target number of clusters K.
        max_iter: Maximum number of full passes over X per initialization.
        n_init: Number of random restarts; the best I(T; Y) solution is returned.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'labels', 'i_ty', 'mi_history', and 'all_init_scores'.
    """
    n_x = len(p_x)
    if not (1 <= n_clusters <= n_x):
        raise ValueError(f"n_clusters must be in [1, {n_x}], got {n_clusters}.")

    rng = np.random.default_rng(seed)
    best_i_ty = -np.inf
    best_labels = None
    best_history: List[float] = []
    all_init_scores: List[float] = []

    for _ in range(n_init):
        labels = _random_partition(n_x, n_clusters, rng)
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        p_t, p_y_given_t, p_ty = compute_cluster_distribution(
            labels, p_x, p_y_given_x, n_clusters=n_clusters
        )
        mi_history = [mutual_information(p_ty)]

        for _it in range(max_iter):
            changed = False
            for x_idx in rng.permutation(n_x):
                old_t = int(labels[x_idx])
                # Keep singleton clusters intact so the number of clusters stays K.
                if cluster_sizes[old_t] <= 1:
                    continue

                # 1. Draw x out of its current cluster.
                px = p_x[x_idx]
                pyx = p_y_given_x[x_idx]
                p_old_rest = p_t[old_t] - px
                p_y_given_t[old_t] = (p_t[old_t] * p_y_given_t[old_t] - px * pyx) / max(
                    p_old_rest, 1e-12
                )
                p_t[old_t] = p_old_rest
                cluster_sizes[old_t] -= 1

                # 2. Evaluate merger cost d_M(x, t) for every candidate cluster t.
                # Minimizing d_M(x, t) is equivalent to maximizing I(T; Y).
                p_sum = px + p_t
                pi_x = px / p_sum
                pi_t = p_t / p_sum
                costs = p_sum * js_divergence(pyx, p_y_given_t, pi_x, pi_t)

                # 3. Assign x deterministically to the highest-scoring cluster.
                new_t = int(np.argmin(costs))
                p_new_total = p_t[new_t] + px
                p_y_given_t[new_t] = (
                    p_t[new_t] * p_y_given_t[new_t] + px * pyx
                ) / p_new_total
                p_t[new_t] = p_new_total
                cluster_sizes[new_t] += 1
                labels[x_idx] = new_t

                if new_t != old_t:
                    changed = True

            _, _, p_ty = compute_cluster_distribution(
                labels, p_x, p_y_given_x, n_clusters=n_clusters
            )
            mi_history.append(mutual_information(p_ty))
            if not changed:
                break

        final_i_ty = mi_history[-1]
        all_init_scores.append(final_i_ty)
        if final_i_ty > best_i_ty:
            best_i_ty = final_i_ty
            best_labels = labels.copy()
            best_history = mi_history

    return {
        "labels": best_labels,
        "i_ty": best_i_ty,
        "mi_history": best_history,
        "all_init_scores": all_init_scores,
    }


def build_cluster_decode_table(
    x_train: np.ndarray,
    prob_train: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cluster centroids T and posterior P(Y|T) decoding table from training clusters."""
    x_mean = x_train.mean(axis=0)
    p_mean = prob_train.mean(axis=0)

    centroids = np.empty((n_clusters, x_train.shape[1]), dtype=np.float64)
    p_y_given_t = np.empty((n_clusters, prob_train.shape[1]), dtype=np.float64)
    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            centroids[k] = x_train[mask].mean(axis=0)
            p_y_given_t[k] = prob_train[mask].mean(axis=0)
        else:
            centroids[k] = x_mean
            p_y_given_t[k] = p_mean

    y_decode_table = (p_y_given_t > threshold).astype(np.int64)
    return centroids, y_decode_table, p_y_given_t


def nearest_centroid_predict(
    x_test: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Assign each test sample to its nearest cluster centroid in Euclidean space."""
    dists = cdist(np.asarray(x_test, dtype=np.float64), centroids, metric="euclidean")
    return np.argmin(dists, axis=1)


def knn_predict(
    x_train: np.ndarray,
    labels: np.ndarray,
    x_test: np.ndarray,
    n_clusters: int,
    n_neighbors: int = 5,
    weighted: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict cluster assignments via K-Nearest Neighbors (standardized Euclidean).

    Returns:
        Tuple of (train_pred, test_pred) cluster indices.
    """
    x_tr = np.asarray(x_train, dtype=np.float64)
    x_te = np.asarray(x_test, dtype=np.float64)
    mu = x_tr.mean(axis=0)
    sd = x_tr.std(axis=0) + 1e-8
    x_tr_s = (x_tr - mu) / sd
    x_te_s = (x_te - mu) / sd

    k_eff = min(n_neighbors, len(x_tr_s))

    def _vote(x_query: np.ndarray) -> np.ndarray:
        dists = cdist(x_query, x_tr_s, metric="euclidean")
        idx = np.argpartition(dists, k_eff - 1, axis=1)[:, :k_eff]
        rows = np.arange(len(x_query))[:, None]
        d_k = dists[rows, idx]
        w_k = 1.0 / (d_k + 1e-8) if weighted else np.ones_like(d_k)
        lbl_k = labels[idx]
        scores = np.zeros((len(x_query), n_clusters), dtype=np.float64)
        for c in range(n_clusters):
            scores[:, c] = np.sum(w_k * (lbl_k == c), axis=1)
        return np.argmax(scores, axis=1)

    return _vote(x_tr_s), _vote(x_te_s)


def svm_fit_predict(
    x_train: np.ndarray,
    labels: np.ndarray,
    x_test: np.ndarray,
    n_clusters: int,
    C: float = 10.0,
    gamma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit One-vs-Rest RBF Kernel L2-SVM via L-BFGS-B and predict cluster assignments.

    Returns:
        Tuple of (train_pred, test_pred) cluster indices.
    """
    x_tr = np.asarray(x_train, dtype=np.float64)
    x_te = np.asarray(x_test, dtype=np.float64)
    mu = x_tr.mean(axis=0)
    sd = x_tr.std(axis=0) + 1e-8
    x_tr_s = (x_tr - mu) / sd
    x_te_s = (x_te - mu) / sd

    if gamma is None:
        gamma = 1.0 / x_tr_s.shape[1]

    n = len(x_tr_s)
    k_tr = np.exp(-gamma * cdist(x_tr_s, x_tr_s, metric="sqeuclidean"))
    k_reg = k_tr + 1e-8 * np.eye(n)
    k_te = np.exp(-gamma * cdist(x_te_s, x_tr_s, metric="sqeuclidean"))

    scores_tr = np.full((n, n_clusters), -np.inf, dtype=np.float64)
    scores_te = np.full((len(x_te_s), n_clusters), -np.inf, dtype=np.float64)

    for c in range(n_clusters):
        pos_mask = labels == c
        if not np.any(pos_mask):
            continue
        yc = np.where(pos_mask, 1.0, -1.0)

        def _obj(params: np.ndarray) -> Tuple[float, np.ndarray]:
            beta = params[:n]
            b = params[n]
            f = k_tr @ beta + b
            m = np.maximum(0.0, 1.0 - yc * f)
            loss = float(0.5 * (beta @ (k_reg @ beta)) + C * np.sum(m ** 2))
            g = -2.0 * C * yc * m
            grad_beta = k_reg @ beta + k_tr @ g
            grad_b = float(np.sum(g))
            return loss, np.append(grad_beta, grad_b)

        res = minimize(_obj, np.zeros(n + 1, dtype=np.float64), method="L-BFGS-B", jac=True)
        beta, b = res.x[:n], res.x[n]
        scores_tr[:, c] = k_tr @ beta + b
        scores_te[:, c] = k_te @ beta + b

    return np.argmax(scores_tr, axis=1), np.argmax(scores_te, axis=1)
