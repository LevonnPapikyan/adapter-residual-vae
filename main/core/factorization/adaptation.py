"""LoRA on a frozen language model.

What belongs here
-----------------
- Turning a ``LoraConfig`` into a PEFT-wrapped model.

What does NOT belong here
-------------------------
- Which parameters get which learning rate (``TopicVAE``), or anything about
  saved files (``checkpointing``).
"""

from __future__ import annotations

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model

from core.factorization.config import LoraConfig


def apply_lora(model: torch.nn.Module, *, lora: LoraConfig, task_type: str | None = None) -> torch.nn.Module:
    """Wrap a frozen model in a LoRA on its attention projections.

    Every projection ``W`` becomes ``W + (alpha / rank) B A`` with ``B = 0`` at
    the start, so at step 0 the wrapped model is exactly the plain one. PEFT
    draws ``A`` at random here, so where this is called fixes the order of the
    random draws.
    """

    extra = {} if task_type is None else {"task_type": task_type}
    peft_config = PeftLoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(lora.target_modules),
        **extra,
    )
    return get_peft_model(model, peft_config)
