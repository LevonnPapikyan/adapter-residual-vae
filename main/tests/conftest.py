"""Shared test settings.

What belongs here
-----------------
- Skips for what a machine does not have, and the paths of the read-only inputs.

What does NOT belong here
-------------------------
- Any test. Nothing in the tests writes outside a temporary folder.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

MAIN_ROOT = Path(__file__).resolve().parents[1]
VAE_ROOT = MAIN_ROOT.parent
FLAT_ROOT = Path(os.environ.get("VAE_FLAT_ROOT", VAE_ROOT)).resolve()  # the deleted flat code: a read-only copy can be named here
DATA_DIR = VAE_ROOT / "data"
RUNS_DIR = VAE_ROOT / "models"
FINAL_DELTA = RUNS_DIR / "final_0.5b" / "delta"
FINAL_H = RUNS_DIR / "final_0.5b" / "h"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _qwen_is_cached() -> bool:
    """The files the model loads. Not ``snapshot_download``: it also wants the README and LICENSE, which no run needs."""

    try:
        from huggingface_hub import try_to_load_from_cache

        return all(isinstance(try_to_load_from_cache("Qwen/Qwen2.5-0.5B", name), str) for name in ("config.json", "model.safetensors", "tokenizer.json"))
    except Exception:
        return False


needs_weights = pytest.mark.skipif(not FINAL_DELTA.exists(), reason="the saved weights are not on this machine")
needs_data = pytest.mark.skipif(not DATA_DIR.exists(), reason="the data folder is not on this machine")
needs_qwen = pytest.mark.skipif(not _qwen_is_cached(), reason="Qwen2.5-0.5B is not cached on this machine")
