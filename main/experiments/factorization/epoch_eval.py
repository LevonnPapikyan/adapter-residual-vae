"""The two evaluations run after every training epoch.

What belongs here
-----------------
- ``held_out_loss``, the number that picks the epoch.
- ``watch_test_scores``, the numbers in ``test.json``. They never pick anything.

What does NOT belong here
-------------------------
- The full evaluation of a finished run (``evaluate.py``).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from core.factorization.model import TopicVAE


@torch.no_grad()
def held_out_loss(*, model: TopicVAE, texts: Sequence[str], theta: torch.Tensor, batch_size: int = 32) -> float:
    """Reconstruction per token plus ``lam`` times alignment, at ``z = mu``, on the held-out training texts."""

    model.eval()
    total = 0.0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        mu = model.posterior(texts=batch).mu
        align = -(theta[i : i + batch_size].to(model.device) * F.log_softmax(mu, -1)).sum(1)
        total += (model.config.objective.lam * align).sum().item()
        if model.config.use_generator:  # without a generator the held-out loss is the alignment alone
            reconstruction = model.reconstruction_nll(z=mu, texts=batch)
            total += (reconstruction.nll / reconstruction.num_tokens).sum().item()
    model.train()
    return total / len(texts)


@torch.no_grad()
def watch_test_scores(
    *,
    model: TopicVAE,
    docs: Mapping[str, Sequence[str]],
    theta_target: np.ndarray,
    batch_size: int = 32,
) -> dict[str, float]:
    """Reconstruction and alignment at ``z = mu`` on the English text, and the top topic right in English and German."""

    model.eval()
    rec = align = 0.0
    top: dict[str, list[torch.Tensor]] = {"en": [], "de": []}
    for i in range(0, len(theta_target), batch_size):
        theta = torch.tensor(theta_target[i : i + batch_size], dtype=torch.float32, device=model.device)
        for lang in top:
            batch = docs[lang][i : i + batch_size]
            mu = model.posterior(texts=batch).mu
            top[lang].append(mu.argmax(1).cpu())
            if lang == "en":
                if model.config.use_generator:
                    reconstruction = model.reconstruction_nll(z=mu, texts=batch)
                    rec += (reconstruction.nll / reconstruction.num_tokens).sum().item()
                align += -(theta * F.log_softmax(mu, -1)).sum().item()
    model.train()

    def right(lang: str) -> int:
        return int((torch.cat(top[lang]).numpy() == theta_target.argmax(1)).sum())

    return {"rec": rec / len(theta_target), "align": align / len(theta_target), "en": right("en"), "de": right("de")}
