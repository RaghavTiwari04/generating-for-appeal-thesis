#!/bin/bash
#SBATCH --job-name=gc-syseval
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%j-syseval.out
#SBATCH --error=logs/slurm-%j-syseval.err

# System evaluation — generates cards under all 4 conditions (A/B/C/D)
# Long job: ~2hr on A100, ~4hr on A40

set -euo pipefail
module load anaconda3/2024 2>/dev/null || module load anaconda3
conda activate gc
cd "$SLURM_SUBMIT_DIR"

echo "=== System evaluation ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

python -m eval.system_eval --study-id system_eval_v1

echo "=== Done: $(date) ==="
