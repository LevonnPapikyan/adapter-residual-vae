"""Score 1: is the top topic right?

What belongs here
-----------------
- Counting agreement with the teacher's top topic on the English original, at
  top 1 and top k, and the floors to compare against.

What does NOT belong here
-------------------------
- Producing the mixtures.
"""

from __future__ import annotations

import numpy as np


def topic_score(*, theta_pred: np.ndarray, target: np.ndarray) -> int:
    """Documents whose predicted top topic is the teacher's top topic on the English text."""

    if theta_pred.shape != target.shape:
        raise ValueError(f"theta_pred {theta_pred.shape} must have the shape of target {target.shape}.")
    return int((theta_pred.argmax(1) == target.argmax(1)).sum())


def topk_counts(*, theta_pred: np.ndarray, target: np.ndarray) -> list[int]:
    """For k = 1..K: documents with the teacher's top topic within the predictor's top k."""

    rank = (np.argsort(-theta_pred, 1) == target.argmax(1)[:, None]).argmax(1)
    return [int((rank < k).sum()) for k in range(1, target.shape[1] + 1)]


def majority_count(*, target: np.ndarray) -> int:
    """Always answering the most common topic: the floor of score 1."""

    return int(np.bincount(target.argmax(1)).max())


def most_common_floor(*, target: np.ndarray) -> list[int]:
    """For k = 1..K: documents won by always answering the k most common topics."""

    top = target.argmax(1)
    common = np.argsort(-np.bincount(top, minlength=target.shape[1]))
    return [int(np.isin(top, common[:k]).sum()) for k in range(1, target.shape[1] + 1)]
