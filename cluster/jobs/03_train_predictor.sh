#!/bin/bash
#SBATCH --job-name=gc-predictor
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j-predictor.out
#SBATCH --error=logs/slurm-%j-predictor.err

# Train 5-head saleability predictor — small model, any GPU fine

set -euo pipefail
module load anaconda3/2024 2>/dev/null || module load anaconda3
conda activate gc
cd "$SLURM_SUBMIT_DIR"

echo "=== Predictor training ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python -m models.predictor.train --epochs 30 --batch-size 64
python -m eval.predictor_eval_standalone
python -m data.features.predictor_scores

echo "=== Done: $(date) ==="
