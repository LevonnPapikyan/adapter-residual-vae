"""A linear map from features to topic mixtures, fit to the teacher's ``theta``.

What belongs here
-----------------
- The soft-target logistic regression used by the frozen baseline and the layer scan.

What does NOT belong here
-------------------------
- Extracting the features, or seeding. The caller seeds torch before fitting, since
  the map starts from random weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LinearTopicMap:
    """A fitted ``softmax(W x + b)`` on standardized features."""

    linear: torch.nn.Linear
    mean: np.ndarray
    scale: np.ndarray
    device: str

    def _standardized(self, features: np.ndarray) -> torch.Tensor:
        return torch.tensor((features - self.mean) / self.scale, dtype=torch.float32, device=self.device)

    def predict(self, *, features: np.ndarray) -> np.ndarray:
        """Topic mixtures for new features."""

        return F.softmax(self.linear(self._standardized(features)), -1).detach().cpu().numpy()


def fit_linear_map(*, features: np.ndarray, theta: np.ndarray, device: str, steps: int = 500) -> LinearTopicMap:
    """Fit with Adam on the full batch. Standardizes first: Qwen has a few huge coordinates."""

    if features.ndim != 2 or theta.ndim != 2 or len(features) != len(theta):
        raise ValueError(f"features {features.shape} and theta {theta.shape} must be matrices with the same rows.")
    linear = torch.nn.Linear(features.shape[1], theta.shape[1]).to(device)
    fitted = LinearTopicMap(linear=linear, mean=features.mean(0), scale=features.std(0) + 1e-6, device=device)
    inputs = fitted._standardized(features)
    targets = torch.tensor(theta, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        optimizer.zero_grad()
        (-(targets * F.log_softmax(linear(inputs), -1)).sum(-1).mean()).backward()
        optimizer.step()
    return fitted
