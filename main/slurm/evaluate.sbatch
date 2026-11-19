#!/bin/bash
# Every score and figure from the saved weights, on one GPU. Submit from main/, after `mkdir -p outputs/logs`:
#     sbatch slurm/evaluate.sbatch configs/paper/evaluation.json
# An A100 unless --gres says otherwise on the sbatch line; rounding differs by GPU type. Options after the config go to
# the command, and a folder that already holds a results.json needs --overwrite.
#SBATCH --job-name=main-evaluate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/logs/evaluate.%j.out
# Set VAE_ENV_ACTIVATE to a script that activates the Python environment (for example a conda activate line).
source "${VAE_ENV_ACTIVATE:?set VAE_ENV_ACTIVATE to a script that activates the Python environment}"
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
python -u -m experiments.factorization.evaluate --config "${1:-configs/paper/evaluation.json}" "${@:2}"
