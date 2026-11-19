"""The ``model.pt`` file format: ``{"config": dict, "weights": {name: tensor}}``.

What belongs here
-----------------
- Writing and reading that outer format, and refusing to overwrite a file.

What does NOT belong here
-------------------------
- What the names inside mean (``core/factorization/checkpointing.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class SavedModelFile:
    """The two parts of a ``model.pt``."""

    config: dict[str, Any]
    weights: dict[str, torch.Tensor]


def save_model_file(path: str | Path, *, config: dict[str, Any], weights: dict[str, torch.Tensor]) -> Path:
    """Write a ``model.pt``. Never overwrites an existing file."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; a trained model is never overwritten.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "weights": weights}, path)
    return path


def load_model_file(path: str | Path, *, map_location: str | torch.device = "cpu") -> SavedModelFile:
    """Read and validate a ``model.pt``."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No trained model at {path}.")
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict) or not isinstance(payload.get("weights"), dict):
        raise ValueError(f"{path} must hold a dictionary with 'config' and 'weights' dictionaries.")
    return SavedModelFile(config=payload["config"], weights=payload["weights"])
