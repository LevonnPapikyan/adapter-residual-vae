#!/bin/bash
# Runs a notebook top to bottom on one GPU and saves its outputs into it. Submit from main/, after `mkdir -p outputs/logs`:
#     sbatch slurm/notebook.sbatch inference      # every score and figure from the saved weights
#     sbatch slurm/notebook.sbatch train          # corpus, teacher, both models
# The notebook is rebuilt from its .py source first, so edit notebooks/train.py or notebooks/inference.py. Rebuilding drops
# the saved outputs, so a notebook that has them is refused unless VAE_NOTEBOOK_FORCE=1.
# With VAE_NOTEBOOK_OUTPUT=path.ipynb a copy is built and run there instead, and notebooks/ is left alone. With VAE_EVAL_CONFIG
# the notebook scores that config's weights, and its first cell says so. An output folder that already holds a results.json
# is refused unless VAE_OVERWRITE_OUTPUT=1. The retrained pair:
#     VAE_NOTEBOOK_OUTPUT=outputs/checkpointed/notebooks/inference.ipynb VAE_EVAL_CONFIG=configs/paper/evaluation_checkpointed.json VAE_OVERWRITE_OUTPUT=1 VAE_NOTEBOOK_FORCE=1 sbatch slurm/notebook.sbatch inference
# An A100 unless --gres says otherwise on the sbatch line; rounding differs by GPU type.
#SBATCH --job-name=main-notebook
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/logs/notebook.%j.out
set -e
# Set VAE_ENV_ACTIVATE to a script that activates the Python environment (for example a conda activate line).
source "${VAE_ENV_ACTIVATE:?set VAE_ENV_ACTIVATE to a script that activates the Python environment}"
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
NOTEBOOK="${1:-inference}"
BUILD_OPTIONS=()
[ "${VAE_NOTEBOOK_FORCE:-}" = "1" ] && BUILD_OPTIONS+=(--force)
[ -n "${VAE_EVAL_CONFIG:-}" ] && [ "$NOTEBOOK" = "inference" ] && BUILD_OPTIONS+=(--eval-config "$VAE_EVAL_CONFIG")
if [ -n "${VAE_NOTEBOOK_OUTPUT:-}" ]; then
    mkdir -p "$(dirname "$VAE_NOTEBOOK_OUTPUT")"
    python notebooks/build_notebooks.py "$NOTEBOOK.py" --output "$VAE_NOTEBOOK_OUTPUT" "${BUILD_OPTIONS[@]}"
    python -c '
import sys
import nbclient, nbformat
path = sys.argv[1]
notebook = nbformat.read(path, as_version=4)
try:  # run from main/, whatever folder the copy is in. What ran is saved even when a cell fails.
    nbclient.NotebookClient(notebook, kernel_name="python3", timeout=None, resources={"metadata": {"path": "."}}).execute()
finally:
    nbformat.write(notebook, path)
' "$VAE_NOTEBOOK_OUTPUT"
    exit 0
fi
python notebooks/build_notebooks.py "$NOTEBOOK.py" "${BUILD_OPTIONS[@]}"
jupyter nbconvert --to notebook --execute --inplace "notebooks/$NOTEBOOK.ipynb" --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1
