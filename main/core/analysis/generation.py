"""Score 3: does the generator write about the topic?

What belongs here
-----------------
- Writing from a topic's average latent, and counting what the teacher says about it.

What does NOT belong here
-------------------------
- Seeding (the caller seeds torch before sampling).
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np
import torch

from core.factorization.posterior import sample_z


class SupportsTheta(Protocol):
    """A teacher: its topic count and its topic mixture for texts."""

    num_topics: int

    def theta(self, texts: Sequence[str]) -> np.ndarray:
        """``[texts, num_topics]`` mixtures."""


@torch.no_grad()
def generate_by_topic(
    *,
    model: torch.nn.Module,
    teacher: SupportsTheta,
    texts: Sequence[str],
    per_topic: int = 10,
    max_new_tokens: int = 96,
    batch_size: int = 32,
) -> dict[int, list[str]]:
    """For each topic: sample ``z`` for every text the teacher puts there, average, and write ``per_topic`` texts."""

    top = teacher.theta(texts).argmax(1)
    samples = []
    for i in range(0, len(texts), batch_size):
        posterior = model.posterior(texts=texts[i : i + batch_size])
        samples.append(sample_z(mu=posterior.mu, logvar=posterior.logvar).cpu())
    z = torch.cat(samples)
    return {
        topic: model.write_from_z(
            z=z[top == topic].mean(0, keepdim=True).repeat(per_topic, 1).to(model.device),
            max_new_tokens=max_new_tokens,
        )
        for topic in range(teacher.num_topics)
        if (top == topic).any()
    }


def generation_score(*, teacher: SupportsTheta, generated: Mapping[int, list[str]]) -> dict[int, int]:
    """For each topic: how many of the written texts the teacher puts back in it."""

    return {topic: int((teacher.theta(texts).argmax(1) == topic).sum()) for topic, texts in generated.items()}
