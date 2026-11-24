"""Data generation and probability distribution utilities for Information Bottleneck."""

from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Tuple
import numpy as np


def _polynomial_features(
    x: np.ndarray, max_degree: int = 3
) -> Tuple[List[np.ndarray], List[Tuple[int, ...]]]:
    """Build multivariate polynomial basis matrices up to max_degree (excluding bias)."""
    n_dims = x.shape[1]
    matrices_by_degree: List[np.ndarray] = []
    all_terms: List[Tuple[int, ...]] = []

    for degree in range(1, max_degree + 1):
        combos = list(combinations_with_replacement(range(n_dims), degree))
        idx = np.asarray(combos, dtype=np.int64)
        mat = np.prod(x[:, idx], axis=2)
        matrices_by_degree.append(mat)
        all_terms.extend(combos)

    return matrices_by_degree, all_terms


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid function."""
    z_clipped = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def generate_data(
    n_x: int = 200,
    x_dim: int = 20,
    x_low: int = 0,
    x_high: int = 10,
    y_dim: int = 10,
    max_degree: int = 3,
    logit_scale: float = 1.5,
    coeff_std: Optional[float] = None,
    seed: Optional[int] = 42,
) -> Dict[str, object]:
    """Generate synthetic edge-communication data for task-oriented compression.

    X consists of n_x samples, each an x_dim-dimensional vector of independent discrete
    integers uniformly distributed on [x_low, x_high]. For each sample, a degree-<=max_degree
    multivariate polynomial with normally distributed coefficients followed by a sigmoid
    computes the posterior probability matrix P(Y_j = 1 | X_i).

    Returns:
        Dictionary containing X, prob_matrix, poly_logits, poly_bias, poly_weights,
        and input/polynomial metadata.
    """
    if max_degree < 1:
        raise ValueError(f"max_degree must be >= 1, got {max_degree}.")

    rng = np.random.default_rng(seed)

    # 1. Sample X: (n_x, x_dim), independent uniform integers on [x_low, x_high].
    x_samples = rng.integers(x_low, x_high + 1, size=(n_x, x_dim), dtype=np.int64)

    mean_x = 0.5 * (x_low + x_high)
    n_vals = x_high - x_low + 1
    std_x = np.sqrt((n_vals ** 2 - 1.0) / 12.0) if n_vals > 1 else 1.0
    x_norm = (x_samples.astype(np.float64) - mean_x) / std_x

    # 2. Construct polynomial basis up to max_degree and sample normal coefficients.
    basis_mats, _ = _polynomial_features(x_norm, max_degree=max_degree)

    bias_std = coeff_std if coeff_std is not None else 0.25 * logit_scale
    poly_bias = rng.normal(0.0, bias_std, size=(y_dim,))
    poly_logits = np.broadcast_to(poly_bias, (n_x, y_dim)).copy()

    poly_weights: List[np.ndarray] = []
    per_degree_std = logit_scale / np.sqrt(max_degree)
    for mat in basis_mats:
        n_terms = mat.shape[1]
        if coeff_std is not None:
            w_std = coeff_std
        else:
            term_rms = float(np.sqrt(np.mean(mat ** 2)))
            w_std = per_degree_std / (np.sqrt(n_terms) * max(term_rms, 1e-12))
        w = rng.normal(0.0, w_std, size=(n_terms, y_dim))
        poly_weights.append(w)
        poly_logits += mat @ w

    # 3. Map polynomial values to posterior probabilities P(Y | X).
    prob_matrix = _sigmoid(poly_logits)

    return {
        "X": x_samples,
        "prob_matrix": prob_matrix,
        "poly_logits": poly_logits,
        "poly_bias": poly_bias,
        "poly_weights": poly_weights,
        "x_low": x_low,
        "x_high": x_high,
        "x_dim": x_dim,
        "max_degree": max_degree,
    }


def compute_y_marginal_stats(
    prob_matrix: np.ndarray, threshold: float = 0.5
) -> Dict[str, object]:
    """Compute marginal P(Y_i = 1) and threshold balance statistics across samples.

    Args:
        prob_matrix: Posterior probability matrix P(Y_i = 1 | X) of shape (n_x, y_dim).
        threshold: Decision threshold used for binary label evaluation.

    Returns:
        Dictionary with per-dimension and total P(Y_i = 1), positive ratios above
        threshold, and majority-class baseline accuracies.
    """
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    p_y_i = prob_matrix.mean(axis=0)
    pos_ratio_i = (prob_matrix > threshold).mean(axis=0)
    majority_baseline_i = np.maximum(pos_ratio_i, 1.0 - pos_ratio_i)
    return {
        "p_y_i": p_y_i,
        "p_y_total": float(prob_matrix.mean()),
        "pos_ratio_i": pos_ratio_i,
        "pos_ratio_total": float((prob_matrix > threshold).mean()),
        "majority_baseline_i": majority_baseline_i,
        "majority_baseline_total": float(majority_baseline_i.mean()),
    }


def sample_from_mapping(
    data: Dict[str, object],
    n_samples: int = 500,
    seed: Optional[int] = 123,
) -> Dict[str, np.ndarray]:
    """Generate new X test samples and their posterior P(Y | X) from the stored mapping."""
    rng = np.random.default_rng(seed)
    x_low = int(data["x_low"])
    x_high = int(data["x_high"])
    x_dim = int(data["x_dim"])
    max_degree = int(data["max_degree"])
    poly_bias = np.asarray(data["poly_bias"], dtype=np.float64)
    poly_weights = data["poly_weights"]

    x_samples = rng.integers(x_low, x_high + 1, size=(n_samples, x_dim), dtype=np.int64)
    mean_x = 0.5 * (x_low + x_high)
    n_vals = x_high - x_low + 1
    std_x = np.sqrt((n_vals ** 2 - 1.0) / 12.0) if n_vals > 1 else 1.0
    x_norm = (x_samples.astype(np.float64) - mean_x) / std_x

    basis_mats, _ = _polynomial_features(x_norm, max_degree=max_degree)
    poly_logits = np.broadcast_to(poly_bias, (n_samples, len(poly_bias))).copy()
    for mat, w in zip(basis_mats, poly_weights):
        poly_logits += mat @ w

    prob_matrix = _sigmoid(poly_logits)

    return {
        "X": x_samples,
        "prob_matrix": prob_matrix,
        "poly_logits": poly_logits,
    }


def compute_posterior_distribution(
    prob_matrix: np.ndarray, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build P(X, Y_tilde), P(X), P(Y_tilde), and P(Y_tilde | X) directly from P(Y | X).

    For each sample x_i (uniform prior P(X=x_i) = 1/n_x) and each binary coordinate j,
    encodes both P(Y_j=1 | x_i) and 1 - P(Y_j=1 | x_i) scaled by 1/y_dim.
    """
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    n_x, y_dim = prob_matrix.shape
    if n_x == 0 or y_dim == 0:
        raise ValueError("prob_matrix must be non-empty.")

    p_x = np.full(n_x, 1.0 / n_x, dtype=np.float64)
    p_succ = prob_matrix / float(y_dim)
    p_fail = (1.0 - prob_matrix) / float(y_dim)
    p_y_given_x = np.hstack([p_succ, p_fail])
    p_y_given_x = p_y_given_x / np.maximum(p_y_given_x.sum(axis=1, keepdims=True), eps)
    p_xy = p_x[:, None] * p_y_given_x
    p_y = p_xy.sum(axis=0)
    return p_xy, p_x, p_y, p_y_given_x


def compute_cluster_distribution(
    labels: np.ndarray,
    p_x: np.ndarray,
    p_y_given_x: np.ndarray,
    n_clusters: Optional[int] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cluster probabilities P(T), P(Y|T), and P(T, Y) for hard clustering."""
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
