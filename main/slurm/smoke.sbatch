#!/bin/bash
# The tests and the small example on a GPU node, under an hour. Submit from main/, after `mkdir -p outputs/logs`:
#     sbatch slurm/smoke.sbatch
#SBATCH --job-name=main-smoke
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:55:00
#SBATCH --output=outputs/logs/smoke.%j.out
# Set VAE_ENV_ACTIVATE to a script that activates the Python environment (for example a conda activate line).
source "${VAE_ENV_ACTIVATE:?set VAE_ENV_ACTIVATE to a script that activates the Python environment}"
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
python -m pytest tests -v
RUN="outputs/smoke/$SLURM_JOB_ID"
python -u -m experiments.factorization.train --config configs/small_cpu_example/factorization_delta.json --output-dir "$RUN/factorization_delta"
python -u -m experiments.factorization.evaluate --config configs/small_cpu_example/evaluation.json --output-dir "$RUN/evaluation"
