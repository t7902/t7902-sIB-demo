"""Data generation and probability distribution utilities for Information Bottleneck."""

from typing import Dict, Optional, Tuple
import numpy as np


def generate_data(
    n_x: int = 100,
    n_clusters: int = 10,
    n_features: int = 25,
    n_cluster_features: int = 8,
    p_in: float = 0.5,
    p_out: float = 0.1,
    n_draws: int = 100,
    seed: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic clustering data for Information Bottleneck experiments.

    X consists of integer labels 1..n_x randomly partitioned into n_clusters clusters.
    Each cluster is associated with n_cluster_features randomly chosen features out of
    n_features. For each sample x in a cluster, each associated feature is generated
    with probability p_in, and each non-associated feature with probability p_out.

    Args:
        n_x: Number of X labels (1 to n_x).
        n_clusters: Number of ground-truth clusters.
        n_features: Dimensionality of feature vector Y.
        n_cluster_features: Number of active features associated with each cluster.
        p_in: Probability of generating an associated feature.
        p_out: Probability of generating a non-associated feature.
        n_draws: Number of 25-dim Bernoulli feature vectors sampled per x.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing X, true_clusters, cluster_features, prob_matrix, and counts.
    """
    rng = np.random.default_rng(seed)
    x_labels = np.arange(1, n_x + 1)

    # Randomly assign each x to one of n_clusters, ensuring every cluster is non-empty.
    while True:
        true_clusters = rng.integers(0, n_clusters, size=n_x)
        if len(np.unique(true_clusters)) == n_clusters:
            break

    # Each cluster corresponds to n_cluster_features randomly selected features.
    cluster_features = np.zeros((n_clusters, n_features), dtype=bool)
    for c in range(n_clusters):
        chosen = rng.choice(n_features, size=n_cluster_features, replace=False)
        cluster_features[c, chosen] = True

    # Conditional generation probabilities P(Y_j = 1 | X = x).
    prob_matrix = np.where(cluster_features[true_clusters], p_in, p_out)

    # Sample n_draws binary feature vectors per x and aggregate feature counts.
    counts = rng.binomial(n_draws, prob_matrix).astype(np.float64)

    return {
        "X": x_labels,
        "true_clusters": true_clusters,
        "cluster_features": cluster_features,
        "prob_matrix": prob_matrix,
        "counts": counts,
    }


def compute_joint_distribution(
    counts: np.ndarray, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute empirical joint and conditional probability distributions from counts.

    Args:
        counts: Non-negative matrix of shape (n_x, n_y).
        eps: Small constant to prevent division by zero.

    Returns:
        Tuple of (p_xy, p_x, p_y, p_y_given_x).
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Total count must be positive.")

    p_xy = counts / total
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    p_y_given_x = p_xy / np.maximum(p_x[:, None], eps)
    return p_xy, p_x, p_y, p_y_given_x


def compute_cluster_distribution(
    labels: np.ndarray,
    p_x: np.ndarray,
    p_y_given_x: np.ndarray,
    n_clusters: Optional[int] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cluster probabilities P(T), P(Y|T), and P(T, Y) for hard clustering.

    Args:
        labels: Integer cluster assignments of shape (n_x,).
        p_x: Marginal distribution P(X) of shape (n_x,).
        p_y_given_x: Conditional distribution P(Y|X) of shape (n_x, n_y).
        n_clusters: Number of clusters; inferred from labels if None.
        eps: Small constant to prevent division by zero.

    Returns:
        Tuple of (p_t, p_y_given_t, p_ty).
    """
    labels = np.asarray(labels, dtype=np.int64)
    if n_clusters is None:
        n_clusters = int(labels.max()) + 1

    n_y = p_y_given_x.shape[1]
    p_ty = np.zeros((n_clusters, n_y), dtype=np.float64)
    p_xy = p_x[:, None] * p_y_given_x
    np.add.at(p_ty, labels, p_xy)

    p_t = p_ty.sum(axis=1)
    p_y_given_t = p_ty / np.maximum(p_t[:, None], eps)
    return p_t, p_y_given_t, p_ty
