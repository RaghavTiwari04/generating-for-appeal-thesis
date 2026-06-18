#!/bin/bash
#SBATCH --job-name=gc-clip-embed
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j-clip-embed.out
#SBATCH --error=logs/slurm-%j-clip-embed.err

# CLIP embedding (SigLIP) — any GPU works, ~2GB VRAM
# Also runs OCR, palette, complexity (CPU-bound, but saves a separate job)

set -euo pipefail
module load anaconda3/2024 2>/dev/null || module load anaconda3
conda activate gc
cd "$SLURM_SUBMIT_DIR"

echo "=== CLIP embed + feature extraction ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python -m data.features.clip_embed
python -m data.features.ocr
python -m data.features.palette
python -m data.features.image_complexity

echo "=== Done: $(date) ==="
