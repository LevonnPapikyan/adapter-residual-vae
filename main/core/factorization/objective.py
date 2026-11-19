"""The training objective of the topic-aligned VAE.

    loss = (rec + beta * rate) / T + lam * align

``rec`` is the generator's negative log-likelihood summed over each text's tokens,
``T`` the mean number of tokens in the batch, ``rate`` is ``KL(q(z | y) || N(0, I))``
(the router is this fixed ``N(0, I)``), and ``align`` is
``-sum_k theta_k log softmax(mu)_k``, which is ``KL(theta || theta_hat)`` plus a
constant. Reconstruction and rate are per token, alignment is per document.

What belongs here
-----------------
- The arithmetic of the loss and its reported parts.

What does NOT belong here
-------------------------
- Sampling ``z`` or running the networks; ``TopicVAE`` does that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from core.factorization.config import ObjectiveConfig


@dataclass(frozen=True)
class ObjectiveOutput:
    """The loss of one batch and its parts as plain numbers.

    Attributes
    ----------
    loss:
        The scalar to backpropagate.
    rec:
        Reconstruction per token, nats.
    align:
        Alignment per document, nats.
    rate:
        KL to the prior per document, nats.
    """

    loss: torch.Tensor
    rec: float
    align: float
    rate: float

    def stats(self) -> dict[str, float]:
        """The row ``history.json`` keeps for this step."""

        return {"rec": self.rec, "align": self.align, "rate": self.rate}


def standard_normal_rate(*, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``KL(N(mu, sigma^2) || N(0, I))`` per row."""

    return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1)


def alignment_to_teacher(*, theta: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of ``softmax(mu)`` under the teacher's ``theta``, per row."""

    if theta.shape != mu.shape:
        raise ValueError(f"theta {tuple(theta.shape)} must have the shape of mu {tuple(mu.shape)}.")
    return -(theta * F.log_softmax(mu, -1)).sum(-1)


def compute_objective(
    *,
    nll: torch.Tensor,
    num_tokens: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    theta: torch.Tensor,
    config: ObjectiveConfig,
) -> ObjectiveOutput:
    """Combine per-text ``nll`` and ``num_tokens`` with the posterior moments and the teacher's ``theta``."""

    rate = standard_normal_rate(mu=mu, logvar=logvar)
    align = alignment_to_teacher(theta=theta, mu=mu)
    mean_tokens = num_tokens.mean()
    loss = (nll + config.beta * rate).mean() / mean_tokens + config.lam * align.mean()
    return ObjectiveOutput(
        loss=loss,
        rec=nll.mean().item() / mean_tokens.item(),
        align=align.mean().item(),
        rate=rate.mean().item(),
    )
