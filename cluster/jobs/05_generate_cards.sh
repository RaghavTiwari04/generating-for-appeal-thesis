#!/bin/bash
#SBATCH --job-name=gc-generate
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j-generate.out
#SBATCH --error=logs/slurm-%j-generate.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}

# Card generation with Flux — needs 22GB+ VRAM (a40/a100)
# For SDXL fallback (7GB), change partition to t4/a16 and set
# DIFFUSION_BACKEND=sdxl in .env

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

echo "=== Card generation ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Backend: ${DIFFUSION_BACKEND:-flux}"
echo "Start: $(date)"

OCCASIONS=(
    "birthday/general"
    "birthday/milestone"
    "birthday/kids"
    "birthday/relationship"
)

for occ in "${OCCASIONS[@]}"; do
    echo "--- Generating for: $occ ($(date)) ---"
    python -m pipeline.orchestrator \
        "$occ" \
        --tone warm-humorous \
        --n 8 --top-k 3
done

echo "=== Done: $(date) ==="
