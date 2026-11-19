#!/bin/bash
# All of tests/ on one CPU thread, no GPU. Submit from main/, after `mkdir -p outputs/checkpointed/logs`:
#     sbatch slurm/tests.sbatch
#SBATCH --job-name=main-tests
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=outputs/checkpointed/logs/tests.%j.out
# Set VAE_ENV_ACTIVATE to a script that activates the Python environment (for example a conda activate line).
source "${VAE_ENV_ACTIVATE:?set VAE_ENV_ACTIVATE to a script that activates the Python environment}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
python -m pytest tests -v -rs -p no:cacheprovider
