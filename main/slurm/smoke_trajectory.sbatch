#!/bin/bash
# The whole checkpoint chain on a few steps and a few documents. Submit from main/, after `mkdir -p outputs/checkpointed/logs`:
#     sbatch slurm/smoke_trajectory.sbatch
#SBATCH --job-name=main-smoke-trajectory
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=30:00
#SBATCH --output=outputs/checkpointed/logs/smoke_trajectory.%j.out
set -e
# Set VAE_ENV_ACTIVATE to a script that activates the Python environment (for example a conda activate line).
source "${VAE_ENV_ACTIVATE:?set VAE_ENV_ACTIVATE to a script that activates the Python environment}"
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
RUN="outputs/checkpointed/smoke/$SLURM_JOB_ID"
for name in h delta; do
  python -u -m experiments.factorization.train --config "configs/small_cpu_example/checkpointed_$name.json" --output-dir "$RUN/$name"
done
python -u -m experiments.factorization.trajectory --config configs/small_cpu_example/trajectory.json --runs-dir "$RUN" --output-dir "$RUN"
# everything but the states again, with no GPU in sight
rm -rf "$RUN/figures" "$RUN/results.json" "$RUN/features/maps" "$RUN"/features/*/float32_*.json "$RUN"/features/*/bfloat16_*.json
CUDA_VISIBLE_DEVICES="" python -u -m experiments.factorization.trajectory --config configs/small_cpu_example/trajectory.json --runs-dir "$RUN" --output-dir "$RUN" --figures-only
echo "figures: $(ls "$RUN/figures" | wc -l) files in $RUN/figures"
# the paper's evaluation on the retrained pair
python -u -m experiments.factorization.evaluate --config configs/small_cpu_example/evaluation.json --runs-dir "$RUN" --output-dir "$RUN/evaluation"
