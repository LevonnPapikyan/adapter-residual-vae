"""One training step.

What belongs here
-----------------
- Computing the batch's loss and applying the guarded update.

What does NOT belong here
-------------------------
- Batch order, schedules, epochs, and files (``train.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from core.factorization.model import TopicVAE
from core.train.loop import guarded_update


@dataclass(frozen=True)
class TrainStepResult:
    """Whether the update was taken, and the batch's loss parts."""

    took_step: bool
    stats: dict[str, float]


def train_one_step(
    *,
    model: TopicVAE,
    texts: Sequence[str],
    theta: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> TrainStepResult:
    """Compute the objective and update, unless the loss or gradient is not finite."""

    objective = model.training_objective(texts=texts, theta=theta)
    took_step = guarded_update(loss=objective.loss, parameters=parameters, optimizer=optimizer, scheduler=scheduler)
    return TrainStepResult(took_step=took_step, stats=objective.stats())
