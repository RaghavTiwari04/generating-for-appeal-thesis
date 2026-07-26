#!/bin/bash
#SBATCH --job-name=gc-clip-embed
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%j-clip-embed.out
#SBATCH --error=logs/slurm-%j-clip-embed.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

PG_DIR="/vol/bitbucket/$USER/pgdata"
trap 'pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true' EXIT

echo "=== CLIP embed + feature extraction ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python -u -m data.features.clip_embed
python -u -m data.features.ocr
python -u -m data.features.palette
python -u -m data.features.image_complexity

# Dedup needs the embeddings and pHashes above. Print-on-demand catalogues
# carry many near-identical designs; duplicates overfit the LoRA and leak
# across the seller-based train/test split.
echo ""
echo "--- Deduplicating listings ---"
python -u -m data.features.dedup

echo "=== Done: $(date) ==="
