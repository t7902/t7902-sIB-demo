"""Information-theoretic functions and transmission accuracy evaluation."""

from typing import Dict
import numpy as np


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
    """Compute weighted Jensen-Shannon divergence in bits."""
    p_bar = pi1[..., None] * p1 + pi2[..., None] * p2
    return pi1 * kl_divergence(p1, p_bar, eps=eps) + pi2 * kl_divergence(
        p2, p_bar, eps=eps
    )


def evaluate_transmission(
    prob_y_given_t_pred: np.ndarray,
    prob_y_given_x_true: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Evaluate overall Y reconstruction rate by checking whether P(Y|T) and P(Y|X)
    lie on the same side of threshold (> threshold -> 1, else 0).
    """
    y_pred = (np.asarray(prob_y_given_t_pred, dtype=np.float64) > threshold).astype(
        np.int64
    )
    y_true = (np.asarray(prob_y_given_x_true, dtype=np.float64) > threshold).astype(
        np.int64
    )
    matches = y_pred == y_true
    correct_count = int(matches.sum())
    total_count = int(matches.size)
    return {
        "correct_count": correct_count,
        "total_count": total_count,
        "overall_acc": correct_count / total_count,
    }
