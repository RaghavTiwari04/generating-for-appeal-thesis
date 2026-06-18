#!/bin/bash
#SBATCH --job-name=gc-generate
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j-generate.out
#SBATCH --error=logs/slurm-%j-generate.err

# Card generation with Flux — needs 22GB+ VRAM
# To use SDXL instead (7GB), set DIFFUSION_BACKEND=sdxl in .env
# and remove the --constraint above

set -euo pipefail
module load anaconda3/2024 2>/dev/null || module load anaconda3
conda activate gc
cd "$SLURM_SUBMIT_DIR"

echo "=== Card generation ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Backend: $DIFFUSION_BACKEND"
echo "Start: $(date)"

OCCASIONS=(
    "birthday/general"
    "christmas/general"
    "mothers_day"
    "valentines_day"
    "sympathy/bereavement"
)

for occ in "${OCCASIONS[@]}"; do
    echo "--- Generating for: $occ ($(date)) ---"
    python -m pipeline.orchestrator \
        --occasion "$occ" \
        --tone warm-humorous \
        --n 8 --top-k 3
done

echo "=== Done: $(date) ==="
