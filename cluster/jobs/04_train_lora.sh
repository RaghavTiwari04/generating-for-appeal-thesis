#!/bin/bash
#SBATCH --job-name=gc-lora
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j-lora.out
#SBATCH --error=logs/slurm-%j-lora.err

# LoRA fine-tuning per occasion — needs 24GB+ VRAM (A40/A100)
# Trains top-5 occasions sequentially

set -euo pipefail
module load anaconda3/2024 2>/dev/null || module load anaconda3
conda activate gc
cd "$SLURM_SUBMIT_DIR"

echo "=== LoRA training ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

OCCASIONS=(
    "birthday/general"
    "christmas/general"
    "mothers_day"
    "valentines_day"
    "sympathy/bereavement"
)

for occ in "${OCCASIONS[@]}"; do
    echo "--- Training LoRA for: $occ ($(date)) ---"
    python -m generation.image.loras.train_lora --occasion "$occ" --rank 8 --steps 1000
done

echo "=== Done: $(date) ==="
