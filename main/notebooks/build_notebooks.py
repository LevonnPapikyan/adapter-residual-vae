"""Build the two notebooks from their percent-cell sources.

    python notebooks/build_notebooks.py              # train.py -> train.ipynb, inference.py -> inference.ipynb
    python notebooks/build_notebooks.py train.py     # just one
    python notebooks/build_notebooks.py inference.py --output copy.ipynb    # one, written elsewhere

The ``.py`` files are the source: edit them, not the notebooks. A cell starts at a
line ``# %%`` (code) or ``# %% [markdown]`` (markdown, every line prefixed ``# ``).

A build has no outputs, so it would erase the ones a run saved: a notebook that has them
is refused unless ``--force``. With ``--eval-config PATH`` the notebook is a run's copy, and
its first cell names the config and weights its outputs will come from.

What belongs here
-----------------
- Turning one source file into one notebook.

What does NOT belong here
-------------------------
- Anything about the model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import nbformat

KERNEL = {"display_name": "python3", "language": "python", "name": "python3"}
SOURCES = ("train.py", "inference.py")
MARKER = re.compile(r"^# %%(?:\s*\[(\w+)\])?\s*$")


def split_cells(source: str) -> list[tuple[str, str]]:
    """``[(kind, text)]`` for a percent-cell source. Empty cells are dropped."""

    cells: list[tuple[str, str]] = []
    kind, lines = "code", []

    def flush() -> None:
        text = "\n".join(lines).strip("\n")
        if kind == "markdown":
            text = "\n".join(re.sub(r"^# ?", "", line) for line in text.split("\n"))
        if text.strip():
            cells.append((kind, text))

    for line in source.split("\n"):
        match = MARKER.match(line)
        if match:
            flush()
            kind, lines = ("markdown" if match.group(1) == "markdown" else "code"), []
        else:
            lines.append(line)
    flush()
    return cells


def _has_outputs(path: Path) -> bool:
    return any(cell.get("outputs") for cell in nbformat.read(path, as_version=4).cells)


def _run_note(eval_config: str) -> str:
    """The first cell of a run's copy: which weights its saved outputs come from."""

    config = json.loads(Path(eval_config).read_text(encoding="utf-8"))
    job = os.environ.get("SLURM_JOB_ID", "not a batch job")
    return (
        f"**The saved outputs in this copy score the weights in `{config['runs_dir']}`** (config `{eval_config}`, written to "
        f"`{config['output_dir']}`, job {job}; `environment.json` there names the GPU). This cell is added when the notebook is "
        "built for a run; it is not in `notebooks/inference.py`.\n\n"
        "Run All in an editor does not repeat them unless `CONFIG` in section 0 is this config: its default is "
        "`configs/paper/evaluation.json`, the saved weights in `../models/final_0.5b`. To repeat them, from `main/`: "
        f"`VAE_NOTEBOOK_OUTPUT=outputs/checkpointed/notebooks/inference.ipynb VAE_EVAL_CONFIG={eval_config} VAE_OVERWRITE_OUTPUT=1 "
        "VAE_NOTEBOOK_FORCE=1 sbatch slurm/notebook.sbatch inference`."
    )


def build_notebook(*, source_path: Path, output_path: Path | None = None, force: bool = False, eval_config: str | None = None) -> Path:
    """Write the ``.ipynb`` next to its source, or to ``output_path``. See the module docstring for ``force`` and ``eval_config``."""

    output_path = source_path.with_suffix(".ipynb") if output_path is None else output_path
    if output_path.exists() and not force and _has_outputs(output_path):
        raise FileExistsError(f"{output_path} holds saved outputs, which a build would erase. Copy it elsewhere first, or pass --force.")
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = KERNEL
    notebook.metadata["language_info"] = {"name": "python"}
    if eval_config is not None:
        notebook.cells.append(nbformat.v4.new_markdown_cell(_run_note(eval_config)))
    for kind, text in split_cells(source_path.read_text(encoding="utf-8")):
        cell = nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text)
        notebook.cells.append(cell)
    nbformat.validate(notebook)
    nbformat.write(notebook, output_path)
    num_code = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"{source_path.name} -> {output_path.name}: {len(notebook.cells)} cells ({num_code} code, {len(notebook.cells) - num_code} markdown)")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the notebooks from their sources.")
    parser.add_argument("names", nargs="*", default=list(SOURCES))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--eval-config", default=None)
    args = parser.parse_args(argv)
    here = Path(__file__).resolve().parent
    if args.output is not None and len(args.names) != 1:
        parser.error("--output builds one notebook; name it.")
    for name in args.names:
        build_notebook(source_path=here / name, output_path=args.output, force=args.force, eval_config=args.eval_config)


if __name__ == "__main__":
    main()
