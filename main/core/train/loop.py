"""Small pieces of the training loop.

What belongs here
-----------------
- The optimizer, the batch order, the held-out split, the guarded parameter update,
  and which steps get a checkpoint.

What does NOT belong here
-------------------------
- The loop itself (``experiments/factorization/train.py``), or anything about
  documents and teachers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch

MAX_GRAD_NORM = 2.0


@dataclass(frozen=True)
class HeldOutSplit:
    """Indices of the texts held out to pick the epoch and of the texts trained on."""

    held_out: np.ndarray
    kept: np.ndarray


def resolve_device(device: str | None) -> str:
    """The requested device, or the GPU when there is one."""

    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def split_held_out(*, rng: np.random.Generator, num_texts: int, val_frac: float) -> HeldOutSplit:
    """Hold out ``int(val_frac * num_texts)`` texts. Draws one permutation from ``rng``."""

    held_out = rng.permutation(num_texts)[: int(val_frac * num_texts)]
    if len(held_out) == 0:
        raise ValueError(f"val_frac={val_frac} of {num_texts} texts holds out nothing; the epoch could not be picked.")
    return HeldOutSplit(held_out=held_out, kept=np.setdiff1d(np.arange(num_texts), held_out))


def count_steps_per_epoch(*, num_texts: int, batch_size: int) -> int:
    """Steps that cover ``num_texts`` once."""

    return -(-num_texts // batch_size)


def epoch_batches(*, rng: np.random.Generator, num_texts: int, num_steps: int) -> list[np.ndarray]:
    """One epoch's batches of text indices. Draws one permutation from ``rng``."""

    return np.array_split(rng.permutation(num_texts), num_steps)


def planned_checkpoint_steps(*, requested: Iterable[int], last_step: int) -> frozenset[int]:
    """The steps that get a checkpoint: none if none were asked for, else those plus step 0 and the last step."""

    requested = set(requested)
    if not requested:
        return frozenset()
    return frozenset(step for step in requested if step <= last_step) | {0, last_step}


def build_adamw(
    *,
    parameter_groups: Mapping[str, list[torch.nn.Parameter]],
    learning_rates: Mapping[str, float],
) -> torch.optim.AdamW:
    """AdamW with one learning rate per non-empty group, with PyTorch's other defaults."""

    return torch.optim.AdamW(
        [{"params": group, "lr": learning_rates[name]} for name, group in parameter_groups.items() if group]
    )


def guarded_update(
    *,
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> bool:
    """Backpropagate and step, unless the loss or the gradient is not finite.

    Returns whether a step was taken. A skipped step moves neither the weights nor
    the schedule. The gradient norm is clipped at ``MAX_GRAD_NORM``.
    """

    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        return False
    loss.backward()
    if not torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)):
        return False
    optimizer.step()
    scheduler.step()
    return True
