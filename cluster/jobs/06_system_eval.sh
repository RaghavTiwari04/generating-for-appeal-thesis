#!/bin/bash
#SBATCH --job-name=gc-syseval
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%j-syseval.out
#SBATCH --error=logs/slurm-%j-syseval.err
#SBATCH --mail-type=END,FAIL

# System evaluation — generates cards under all 4 conditions (A/B/C/D)

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

echo "=== System evaluation ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

python -m eval.system_eval --study-id system_eval_v1

echo "=== Done: $(date) ==="
