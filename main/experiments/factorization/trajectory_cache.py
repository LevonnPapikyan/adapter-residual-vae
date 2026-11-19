"""Where the trajectory analysis keeps things, so figures can be rebuilt without a GPU.

    output_dir/
      results.json                       every number
      saved_weights_check.json           the saved weights through the same code, if asked for
      features/labels.npz, labels.json   who each row is, and the teacher's answers
      features/{model}/run.json          steps, read vector, kept step
      features/{model}/weights.json      W and BA against the final checkpoint
      features/{model}/{float}_{step}.npz, .json   states of one checkpoint, and its scores
      features/saved/{model}_{float}.npz the saved weights' states
      features/maps/{model}_{vector}.npz PCA and t-SNE coordinates of the float32 states
      figures/                           PDF and PNG

What belongs here
-----------------
- ``TrajectoryPaths``, the file names, and writing a file so a killed job leaves no half of it.

What does NOT belong here
-------------------------
- Computing anything.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from core.runs.io import write_json

MODELS: tuple[str, ...] = ("h", "delta")
PRECISIONS: tuple[str, ...] = ("float32", "bfloat16")  # float32 is the primary format, of every figure


@dataclass(frozen=True)
class TrajectoryPaths:
    """The files under one ``output_dir``."""

    output_dir: Path

    @property
    def results(self) -> Path:
        return self.output_dir / "results.json"

    @property
    def saved_check(self) -> Path:
        return self.output_dir / "saved_weights_check.json"

    @property
    def features(self) -> Path:
        return self.output_dir / "features"

    @property
    def figures(self) -> Path:
        return self.output_dir / "figures"

    @property
    def labels_npz(self) -> Path:
        return self.features / "labels.npz"

    @property
    def labels_json(self) -> Path:
        return self.features / "labels.json"

    def run_json(self, model: str) -> Path:
        return self.features / model / "run.json"

    def weights_json(self, model: str) -> Path:
        return self.features / model / "weights.json"

    def states(self, model: str, precision: str, step: int) -> Path:
        return self.features / model / f"{precision}_{step}.npz"

    def row(self, model: str, precision: str, step: int) -> Path:
        return self.features / model / f"{precision}_{step}.json"

    def saved_states(self, model: str, precision: str) -> Path:
        return self.features / "saved" / f"{model}_{precision}.npz"

    def maps(self, model: str, vector: str) -> Path:
        return self.features / "maps" / f"{model}_{vector}.npz"

    @property
    def maps_h_off(self) -> Path:
        return self.features / "maps" / "h_off.npz"


def file_md5(path: str | Path) -> str:
    """The md5 of a file; ties cached states to the checkpoint they came from."""

    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def write_npz(path: str | Path, **arrays: Any) -> None:
    """Write an ``.npz`` under a temporary name first, so an existing file is always whole."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def write_json_whole(path: str | Path, payload: Any) -> None:
    """``write_json`` under a temporary name first, so an existing file is always whole."""

    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    write_json(temporary, payload, indent=1)
    os.replace(temporary, path)
