"""Where is the language inside the plain language model?

What belongs here
-----------------
- Reading the last-token state after every block, and the three scores per block.

What does NOT belong here
-------------------------
- Seeding (the caller seeds torch once before the scan) or plotting.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from core.analysis.language_probe import language_score
from core.analysis.linear_map import fit_linear_map
from core.analysis.representation import TextIndex, finds_original
from core.analysis.topic_score import topic_score


@torch.no_grad()
def layer_states(*, model: torch.nn.Module, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
    """``[texts, blocks + 1, hidden]``: index 0 is the embeddings, the last is the final block."""

    out = []
    for i in range(0, len(texts), batch_size):
        input_ids, attention_mask = model.tokenize(texts=texts[i : i + batch_size], max_tokens=model.config.encoder_max_tokens)
        hidden = model.posterior_backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
        last = attention_mask.sum(1) - 1
        out.append(torch.stack([state[torch.arange(len(last)), last] for state in hidden], 1).float().cpu())
    return torch.cat(out).numpy()


def score_layers(
    *,
    train_states: np.ndarray,
    test_states: np.ndarray,
    theta_train: np.ndarray,
    target: np.ndarray,
    langs: Sequence[str],
    index: TextIndex,
    device: str,
    linear_map_steps: int = 500,
) -> list[dict[str, Any]]:
    """For every block: the topic score per language, the language score, and finds-original (mean over translations)."""

    text_lang = index.text_lang
    rows = []
    for block in range(train_states.shape[1]):
        fitted = fit_linear_map(features=train_states[:, block], theta=theta_train, device=device, steps=linear_map_steps)
        thetas = {lang: fitted.predict(features=test_states[text_lang == j, block]) for j, lang in enumerate(langs)}
        row = {
            "topic": {lang: topic_score(theta_pred=thetas[lang], target=target) for lang in langs},
            "language": language_score(thetas=thetas),
            "finds_original": float(
                np.mean([finds_original(features=test_states[:, block], lang_index=j, index=index) for j in range(1, len(langs))])
            ),
        }
        rows.append(row)
        print(
            f"block {block:2d}  topic " + " ".join(f"{row['topic'][lang]:4d}" for lang in langs)
            + f"   language {row['language']:.2f}   finds original {row['finds_original']:4.0f}",
            flush=True,
        )
    return rows
