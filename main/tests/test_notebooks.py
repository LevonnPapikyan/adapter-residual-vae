"""The notebook sources compile and build into valid notebooks."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import nbformat
import pytest

from conftest import MAIN_ROOT

sys.path.insert(0, str(MAIN_ROOT / "notebooks"))
from build_notebooks import build_notebook, split_cells  # noqa: E402


def test_split_cells() -> None:
    source = "# %% [markdown]\n# # Title\n# text\n\n# %%\nx = 1\n\n# %%\n\n# %%\ny = 2\n"
    assert split_cells(source) == [("markdown", "# Title\ntext"), ("code", "x = 1"), ("code", "y = 2")]


@pytest.mark.parametrize("name", ["train", "inference"])
def test_source_builds_and_compiles(name: str, tmp_path: Path) -> None:
    source_path = tmp_path / f"{name}.py"
    shutil.copy(MAIN_ROOT / "notebooks" / f"{name}.py", source_path)
    notebook = nbformat.read(build_notebook(source_path=source_path), as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) > 5 and notebook.metadata["kernelspec"]["name"] == "python3"
    for cell in code_cells:
        compile(cell.source, f"{name} cell", "exec")


def _source(tmp_path: Path) -> Path:
    source_path = tmp_path / "small.py"
    source_path.write_text("# %%\nx = 1\n", encoding="utf-8")
    return source_path


def test_a_notebook_with_outputs_is_not_overwritten(tmp_path: Path) -> None:
    source_path = _source(tmp_path)
    path = build_notebook(source_path=source_path)
    build_notebook(source_path=source_path)  # no outputs yet: nothing to lose
    notebook = nbformat.read(path, as_version=4)
    notebook.cells[0].outputs = [nbformat.v4.new_output("stream", text="saved")]
    nbformat.write(notebook, path)
    with pytest.raises(FileExistsError, match="--force"):
        build_notebook(source_path=source_path)
    assert nbformat.read(path, as_version=4).cells[0].outputs
    build_notebook(source_path=source_path, force=True)
    assert not nbformat.read(path, as_version=4).cells[0].outputs


def test_a_run_copy_names_its_weights_in_its_first_cell(tmp_path: Path) -> None:
    config = tmp_path / "evaluation.json"
    config.write_text(json.dumps({"runs_dir": "outputs/somewhere", "output_dir": "outputs/somewhere/evaluation"}), encoding="utf-8")
    path = build_notebook(source_path=_source(tmp_path), output_path=tmp_path / "copy.ipynb", eval_config=str(config))
    first = nbformat.read(path, as_version=4).cells[0]
    assert first.cell_type == "markdown" and "outputs/somewhere" in first.source and str(config) in first.source
