#!/bin/bash
#SBATCH --job-name=gc-predictor
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j-predictor.out
#SBATCH --error=logs/slurm-%j-predictor.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

echo "=== Predictor training ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python -m models.predictor.train --epochs 30 --batch-size 64
python -m eval.predictor_eval_standalone
python -m data.features.predictor_scores

echo "=== Done: $(date) ==="
