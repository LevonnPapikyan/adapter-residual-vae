"""Seeding a run.

What belongs here
-----------------
- ``seed_run``: exactly what the final runs seeded, and nothing more.

What does NOT belong here
-------------------------
- Deterministic-backend flags or seeding Python's and numpy's global generators.
  The final runs set neither, and setting them could change the results.
"""

from __future__ import annotations

import numpy as np
import torch


def seed_run(*, seed: int) -> np.random.Generator:
    """Seed torch's global generators and return the numpy generator that shuffles the data."""

    torch.manual_seed(seed)
    return np.random.default_rng(seed)
