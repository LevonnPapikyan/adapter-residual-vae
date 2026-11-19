"""Matching the topics of two LDAs, and how much they agree.

What belongs here
-----------------
- The one-to-one matching on parallel documents and the agreement counts.

What does NOT belong here
-------------------------
- Fitting the LDAs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TopicMatch:
    """``to_a[j]`` is the column of ``a`` assigned to column ``j`` of ``b``, with that pair's correlation."""

    to_a: np.ndarray
    correlation: np.ndarray


def match_topics(*, theta_a: np.ndarray, theta_b: np.ndarray) -> TopicMatch:
    """Correlate topic shares over parallel documents and solve the assignment (Hungarian algorithm)."""

    if theta_a.shape != theta_b.shape:
        raise ValueError(f"theta_a {theta_a.shape} and theta_b {theta_b.shape} must have the same shape.")
    num_topics = theta_a.shape[1]
    centered_a, centered_b = theta_a - theta_a.mean(0), theta_b - theta_b.mean(0)
    correlation = (centered_a.T @ centered_b) / (
        np.linalg.norm(centered_a, axis=0)[:, None] * np.linalg.norm(centered_b, axis=0)[None] + 1e-12
    )
    rows, columns = linear_sum_assignment(-correlation)
    to_a = np.empty(num_topics, int)
    to_a[columns] = rows
    return TopicMatch(to_a=to_a, correlation=correlation[to_a, np.arange(num_topics)])


def map_onto(*, theta_b: np.ndarray, to_a: np.ndarray) -> np.ndarray:
    """Write ``b``'s mixtures over ``a``'s topics."""

    mapped = np.zeros_like(theta_b)
    mapped[:, to_a] = theta_b
    return mapped


def agreement_counts(*, theta_a: np.ndarray, theta_b_mapped: np.ndarray) -> dict[str, Any]:
    """Top topic, top-2 pair, and top-k agreement of two mixtures over the same topics."""

    order_a, order_b = np.argsort(-theta_a, 1), np.argsort(-theta_b_mapped, 1)
    shared = np.array([len(set(x) & set(y)) for x, y in zip(order_a[:, :2], order_b[:, :2])])
    topk = [int((order_b[:, :k] == order_a[:, :1]).any(1).sum()) for k in range(1, theta_a.shape[1] + 1)]
    return {
        "top_topic": topk[0],
        "same_pair": int((shared == 2).sum()),
        "one_shared": int((shared >= 1).sum()),
        "topk": topk,
    }
