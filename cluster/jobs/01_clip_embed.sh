#!/bin/bash
#SBATCH --job-name=gc-clip-embed
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j-clip-embed.out
#SBATCH --error=logs/slurm-%j-clip-embed.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

echo "=== CLIP embed + feature extraction ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python -m data.features.clip_embed
python -m data.features.ocr
python -m data.features.palette
python -m data.features.image_complexity

echo "=== Done: $(date) ==="
