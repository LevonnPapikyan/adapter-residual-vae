#!/bin/bash
# The latent aligned to the language label. Submit from the repository root:
#     sbatch extensions/language_latent/run.sbatch --config ../extensions/language_latent/delta.json
# Edit the #SBATCH lines for your cluster, and set VAE_ENV_ACTIVATE to a script that activates the Python environment.
#SBATCH --job-name=language_latent
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%x.%j.out
ROOT="$SLURM_SUBMIT_DIR"
source "${VAE_ENV_ACTIVATE:?set VAE_ENV_ACTIVATE to a script that activates the Python environment}"
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$ROOT/main"
cd "$ROOT/main"  # the code's paths are relative to main/
python -u "$ROOT/extensions/language_latent/train_language_guide.py" "$@"
