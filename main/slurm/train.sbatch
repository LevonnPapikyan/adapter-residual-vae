#!/bin/bash
# Trains one model on one GPU. Submit from main/, after `mkdir -p outputs/logs`:
#     sbatch slurm/train.sbatch configs/paper/factorization_delta.json
# A folder that already holds a model.pt is refused. An A100 unless --gres says otherwise on the sbatch line, as the originals.
#SBATCH --job-name=main-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/logs/train.%j.out
# Set VAE_ENV_ACTIVATE to a script that activates the Python environment (for example a conda activate line).
source "${VAE_ENV_ACTIVATE:?set VAE_ENV_ACTIVATE to a script that activates the Python environment}"
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
python -u -m experiments.factorization.train --config "${1:?usage: sbatch slurm/train.sbatch CONFIG}"
