"""What a training or evaluation run prints and writes.

What belongs here
-----------------
- Writing ``test.json`` and ``results.json``, the comparison against a reference,
  ``environment.json`` (the GPU and versions behind a run), and small printed tables.
- Refusing to replace a ``results.json`` nobody asked to replace.
- The names ``results.json`` uses for predictors and representations.

What does NOT belong here
-------------------------
- Computing any of the numbers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from core.analysis.comparison import compare_results
from core.runs.io import read_json, write_json
from experiments.data.translations import LANG_NAMES

LABEL_H = "μ from h"
LABEL_DELTA = "μ from Δh"
LABEL_FROZEN = "frozen Qwen + linear map"
LABEL_LDA = "LDA on this text"
LABEL_STATE_OFF = "h_off (plain Qwen)"
LABEL_STATE_H = "h (h model)"
LABEL_STATE_DELTA = "Δh = h_on − h_off"


def print_row(label: str, values: Sequence[Any]) -> None:
    """One line of a printed table."""

    print(f"{label:32s}" + "".join(f"{value:>10}" for value in values))


def gpu_name(device: str) -> str | None:
    """The GPU's name. Rounding differs by GPU type, so it belongs with every number."""

    return torch.cuda.get_device_name(0) if device == "cuda" else None


def write_environment(*, output_dir: str | Path, device: str) -> None:
    """Write ``environment.json``, next to the files it explains; ``results.json`` keeps the shipped file's keys."""

    import transformers  # slow to import, and only a run that used it gets here

    environment = {
        "device": device,
        "gpu": gpu_name(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(Path(output_dir) / "environment.json", environment, indent=1)
    print(f"environment: {environment}")


def refuse_existing_results(*, output_dir: str | Path, overwrite: bool) -> None:
    """Stop before anything is written when a run would replace the scores already in ``output_dir``."""

    path = Path(output_dir) / "results.json"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists, and this run would replace it and the figures beside it. "
            "Pick another output_dir, or ask for it: --overwrite on the command line, VAE_OVERWRITE_OUTPUT=1 or OVERWRITE = True in the notebook."
        )


def write_test_rows(*, run_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the per-epoch scores of a run."""

    write_json(Path(run_dir) / "test.json", rows, indent=1)


def write_results(*, output_dir: str | Path, results: Mapping[str, Any]) -> Path:
    """Write ``results.json`` with the keys and order of the shipped file."""

    path = Path(output_dir) / "results.json"
    write_json(path, results, indent=1)
    print(f"wrote {path}")
    return path


def write_comparison(
    *,
    output_dir: str | Path,
    results: Mapping[str, Any],
    reference_path: str | Path | None,
    is_partial: bool,
) -> dict[str, Any] | None:
    """Compare with a reference ``results.json``, print the table, and write ``comparison.json``."""

    if reference_path is None:
        print("no reference results given: nothing to compare against")
        return None
    reference = read_json(reference_path)
    if is_partial:
        print("small-data options are set, so these differences say nothing about reproduction")
    comparison = compare_results(results=results, reference=reference)
    langs = list(LANG_NAMES)
    print(f"\nthis run minus {reference_path}:")
    print_row("", [LANG_NAMES[lang] for lang in langs] + ["language", "written"])
    for name, scores in comparison["topic"].items():
        language = comparison["language"].get(name)
        written = comparison["generation_total"].get(name)
        print_row(
            name,
            [f"{scores[lang]:+d}" for lang in langs]
            + ["" if language is None else f"{language:+.2f}", "" if written is None else f"{written:+d}"],
        )
    path = Path(output_dir) / "comparison.json"
    write_json(path, {"reference": str(reference_path), "partial": is_partial, "difference": comparison}, indent=1)
    print(f"wrote {path}")
    return comparison
