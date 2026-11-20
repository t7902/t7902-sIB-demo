"""Information-theoretic functions and clustering evaluation metrics."""

from typing import Dict, Optional
import numpy as np

from data import compute_cluster_distribution


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Shannon entropy H(P) in bits."""
    p = np.asarray(p, dtype=np.float64)
    mask = p > eps
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def mutual_information(p_ab: np.ndarray, eps: float = 1e-12) -> float:
    """Compute mutual information I(A; B) in bits from joint distribution P(A, B)."""
    p_ab = np.asarray(p_ab, dtype=np.float64)
    p_a = p_ab.sum(axis=1, keepdims=True)
    p_b = p_ab.sum(axis=0, keepdims=True)
    outer = p_a * p_b
    mask = (p_ab > eps) & (outer > eps)
    return float(np.sum(p_ab[mask] * np.log2(p_ab[mask] / outer[mask])))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute KL divergence D_KL(p || q) in bits (supports batched last axis)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    safe_p = np.where(p > eps, p, 1.0)
    safe_q = np.where(q > eps, q, 1.0)
    term = np.where(p > eps, p * np.log2(safe_p / safe_q), 0.0)
    return np.sum(term, axis=-1)


def js_divergence(
    p1: np.ndarray, p2: np.ndarray, pi1: np.ndarray, pi2: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Compute weighted Jensen-Shannon divergence in bits.

    Args:
        p1: Distribution(s) of shape (..., n_y).
        p2: Distribution(s) of shape (..., n_y).
        pi1: Weight(s) for p1, broadcastable to p1[..., 0].
        pi2: Weight(s) for p2, broadcastable to p2[..., 0].
        eps: Numerical stability threshold.

    Returns:
        JSD value(s) in bits.
    """
    p_bar = pi1[..., None] * p1 + pi2[..., None] * p2
    return pi1 * kl_divergence(p1, p_bar, eps=eps) + pi2 * kl_divergence(
        p2, p_bar, eps=eps
    )


def _contingency_matrix(
    true_labels: np.ndarray, pred_labels: np.ndarray
) -> np.ndarray:
    """Build contingency matrix between ground-truth and predicted labels."""
    _, true_idx = np.unique(true_labels, return_inverse=True)
    _, pred_idx = np.unique(pred_labels, return_inverse=True)
    n_true = int(true_idx.max()) + 1
    n_pred = int(pred_idx.max()) + 1
    mat = np.zeros((n_true, n_pred), dtype=np.int64)
    np.add.at(mat, (true_idx, pred_idx), 1)
    return mat


def purity_score(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Compute clustering purity in [0, 1]."""
    mat = _contingency_matrix(true_labels, pred_labels)
    return float(np.sum(np.max(mat, axis=0)) / len(true_labels))


def normalized_mutual_info(
    true_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """Compute Normalized Mutual Information (NMI) using arithmetic mean."""
    mat = _contingency_matrix(true_labels, pred_labels).astype(np.float64)
    total = mat.sum()
    if total == 0:
        return 0.0
    p_ab = mat / total
    mi = mutual_information(p_ab)
    h_true = entropy(p_ab.sum(axis=1))
    h_pred = entropy(p_ab.sum(axis=0))
    denom = 0.5 * (h_true + h_pred)
    return float(mi / denom) if denom > 1e-12 else 0.0


def adjusted_rand_index(
    true_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """Compute Adjusted Rand Index (ARI)."""
    mat = _contingency_matrix(true_labels, pred_labels).astype(np.float64)
    n = mat.sum()
    if n < 2:
        return 1.0

    def comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1.0) / 2.0

    sum_comb_c = float(np.sum(comb2(mat)))
    sum_comb_a = float(np.sum(comb2(mat.sum(axis=1))))
    sum_comb_b = float(np.sum(comb2(mat.sum(axis=0))))
    total_comb = float(comb2(n))

    expected = (sum_comb_a * sum_comb_b) / total_comb
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 1.0 if abs(sum_comb_c - expected) < 1e-12 else 0.0
    return float((sum_comb_c - expected) / denom)


def evaluate_clustering(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    p_x: np.ndarray,
    p_y_given_x: np.ndarray,
    i_xy: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate a hard clustering using both IB objective and external metrics."""
    if i_xy is None:
        i_xy = mutual_information(p_x[:, None] * p_y_given_x)
    _, _, p_ty = compute_cluster_distribution(pred_labels, p_x, p_y_given_x)
    i_ty = mutual_information(p_ty)
    return {
        "I(T;Y)": i_ty,
        "I(T;Y)/I(X;Y)": i_ty / i_xy if i_xy > 1e-12 else 0.0,
        "NMI": normalized_mutual_info(true_labels, pred_labels),
        "ARI": adjusted_rand_index(true_labels, pred_labels),
        "Purity": purity_score(true_labels, pred_labels),
    }
