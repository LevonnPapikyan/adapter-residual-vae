"""The learning-rate schedule: linear warm-up over 5% of the steps, then cosine decay to 0.

What belongs here
-----------------
- The multiplier as a pure function, and the scheduler built from it.

What does NOT belong here
-------------------------
- Other schedules. The final runs used this one, so it is the only one offered.
"""

from __future__ import annotations

import math

import torch

WARMUP_FRACTION = 0.05


def warmup_steps_for(*, total_steps: int) -> int:
    """Steps of linear warm-up, at least one."""

    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    return max(1, int(WARMUP_FRACTION * total_steps))


def warmup_cosine_multiplier(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Multiplier of the base learning rate at 0-indexed optimizer step ``step``."""

    warmup = min(1.0, (step + 1) / warmup_steps)
    return warmup * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))


def build_warmup_cosine_scheduler(
    *, optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """A per-step scheduler; it advances only when the optimizer took a step."""

    warmup_steps = warmup_steps_for(total_steps=total_steps)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_multiplier(step, warmup_steps=warmup_steps, total_steps=total_steps),
    )
