"""Pieces of the diagonal-Gaussian posterior ``q(z | y)``.

What belongs here
-----------------
- The posterior's output type, the last-token pooling, and the reparameterized
  sample.

What does NOT belong here
-------------------------
- The backbone and the linear maps themselves; ``TopicVAE`` owns them, because
  the order in which it builds them fixes the order of the random draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PosteriorOutput:
    """Moments of ``q(z | y)``, each with shape ``[batch, num_topics]``."""

    mu: torch.Tensor
    logvar: torch.Tensor


def last_token_state(*, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return ``[batch, hidden]``: the state at the last real token of each right-padded text."""

    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must have shape [batch, seq, hidden], got {tuple(hidden_states.shape)}.")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"attention_mask {tuple(attention_mask.shape)} must match hidden_states {tuple(hidden_states.shape[:2])}."
        )
    batch = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch, attention_mask.sum(1) - 1]


def sample_z(*, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """One reparameterized sample ``mu + sigma * eps`` per row. Draws from torch's global generator."""

    return mu + torch.randn_like(mu) * (0.5 * logvar).exp()
