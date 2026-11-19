"""Running a model over many texts, in fixed batches, into numpy arrays.

What belongs here
-----------------
- ``readout_features``, ``topic_mixtures``, ``backbone_states``, ``encoded_states``:
  batched reads of a model on texts.
- ``run_encoder_in_float32``: the same weights read without the bfloat16 rounding.

What does NOT belong here
-------------------------
- Scoring what they return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BackboneStates:
    """Last-token states with the LoRA on and off (plain language model), ``[texts, hidden]``."""

    on: np.ndarray
    off: np.ndarray


@dataclass(frozen=True)
class EncodedStates:
    """Last-token states with the LoRA on and off, and ``mu``, ``[texts, ...]``."""

    on: np.ndarray
    off: np.ndarray
    mu: np.ndarray


def run_encoder_in_float32(model: torch.nn.Module) -> torch.nn.Module:
    """Move the posterior backbone to float32, in place. The weights were trained in bfloat16.

    There is no way back: bfloat16 would also round the LoRA, so build a new model for that.
    """

    model.posterior_backbone.float()
    return model


@torch.no_grad()
def encoded_states(*, model: torch.nn.Module, texts: Sequence[str], batch_size: int = 32) -> EncodedStates:
    """``h_on``, ``h_off`` and ``mu`` of every text, on the same arithmetic as ``model.posterior``."""

    on, off, mu = [], [], []
    for i in range(0, len(texts), batch_size):
        input_ids, attention_mask = model.tokenize(texts=texts[i : i + batch_size], max_tokens=model.config.encoder_max_tokens)
        state_on = model.backbone_state(input_ids=input_ids, attention_mask=attention_mask)
        with model.posterior_backbone.disable_adapter():
            state_off = model.backbone_state(input_ids=input_ids, attention_mask=attention_mask)
        readout = state_on if model.config.mu_from == "h" else state_on - state_off
        on.append(state_on.cpu())
        off.append(state_off.cpu())
        mu.append(model.mu_head(readout).cpu())
    return EncodedStates(on=torch.cat(on).numpy(), off=torch.cat(off).numpy(), mu=torch.cat(mu).numpy())


@torch.no_grad()
def readout_features(*, model: torch.nn.Module, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """What ``mu`` is read from, for each text."""

    return torch.cat(
        [model.readout(texts=texts[i : i + batch_size]).cpu() for i in range(0, len(texts), batch_size)]
    ).numpy()


@torch.no_grad()
def topic_mixtures(*, model: torch.nn.Module, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """The model's ``theta_hat = softmax(mu)`` for each text, ``[texts, num_topics]``."""

    return torch.cat(
        [
            F.softmax(model.posterior(texts=texts[i : i + batch_size]).mu, -1).cpu()
            for i in range(0, len(texts), batch_size)
        ]
    ).numpy()


@torch.no_grad()
def backbone_states(*, model: torch.nn.Module, texts: Sequence[str], batch_size: int = 32) -> BackboneStates:
    """The last-token state of every text with the LoRA on, and off."""

    on, off = [], []
    for i in range(0, len(texts), batch_size):
        input_ids, attention_mask = model.tokenize(texts=texts[i : i + batch_size], max_tokens=model.config.encoder_max_tokens)
        on.append(model.backbone_state(input_ids=input_ids, attention_mask=attention_mask).cpu())
        with model.posterior_backbone.disable_adapter():
            off.append(model.backbone_state(input_ids=input_ids, attention_mask=attention_mask).cpu())
    return BackboneStates(on=torch.cat(on).numpy(), off=torch.cat(off).numpy())
