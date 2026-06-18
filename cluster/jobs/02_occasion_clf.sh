#!/bin/bash
#SBATCH --job-name=gc-occasion-clf
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j-occasion-clf.out
#SBATCH --error=logs/slurm-%j-occasion-clf.err

# Occasion classifier: train DistilBERT then infer on all listings

set -euo pipefail
module load anaconda3/2024 2>/dev/null || module load anaconda3
conda activate gc
cd "$SLURM_SUBMIT_DIR"

echo "=== Occasion classifier ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python -m data.features.occasion_classifier train --epochs 5
python -m data.features.occasion_classifier infer --limit 10000
python -m data.features.dedup

echo "=== Done: $(date) ==="
