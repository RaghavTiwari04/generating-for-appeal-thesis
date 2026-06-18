#!/bin/bash
#SBATCH --job-name=gc-occasion-clf
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j-occasion-clf.out
#SBATCH --error=logs/slurm-%j-occasion-clf.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

echo "=== Occasion classifier ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python -m data.features.occasion_classifier train --epochs 5
python -m data.features.occasion_classifier infer --limit 10000
python -m data.features.dedup

echo "=== Done: $(date) ==="
