#!/bin/bash
# Every checkpoint of a pair: states, scores, maps, figures. Submit from main/, after `mkdir -p outputs/checkpointed/logs`:
#     sbatch --gres=gpu:a100:1 slurm/trajectory.sbatch configs/paper/trajectory.json
# Stopped jobs can be submitted again: what the cache holds is reused. To rebuild everything but the states without a GPU:
#     sbatch --partition=day --gres=none slurm/trajectory.sbatch configs/paper/trajectory.json --figures-only
#SBATCH --job-name=main-trajectory
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=outputs/checkpointed/logs/trajectory.%j.out
set -e
# Set VAE_ENV_ACTIVATE to a script that activates the Python environment (for example a conda activate line).
source "${VAE_ENV_ACTIVATE:?set VAE_ENV_ACTIVATE to a script that activates the Python environment}"
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
python -u -m experiments.factorization.trajectory --config "${1:?usage: sbatch slurm/trajectory.sbatch CONFIG [OPTIONS]}" "${@:2}"
