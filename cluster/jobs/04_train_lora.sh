#!/bin/bash
#SBATCH --job-name=gc-lora
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j-lora.out
#SBATCH --error=logs/slurm-%j-lora.err
#SBATCH --mail-type=END,FAIL

# LoRA fine-tuning per occasion — needs 24GB+ VRAM
# Use a40 (48GB) or a100 (80GB). a30 (24GB) might work tight.

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

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
