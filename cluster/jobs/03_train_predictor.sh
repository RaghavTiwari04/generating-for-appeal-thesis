#!/bin/bash
#SBATCH --job-name=gc-predictor
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
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

# Seeds to average over. One run is not a measurement here: identical configs
# have differed by 0.12 on a head, so a change cannot be told from noise
# without the spread.
SEEDS="${SEEDS:-5}"
TRUNK="${TRUNK:-512}"
HEAD_HIDDEN="${HEAD_HIDDEN:-128}"
DROPOUT="${DROPOUT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
echo "seeds=$SEEDS trunk=$TRUNK head=$HEAD_HIDDEN dropout=$DROPOUT wd=$WEIGHT_DECAY"

python -m models.predictor.train --epochs 30 --batch-size 64     --seeds "$SEEDS" --trunk-hidden "$TRUNK" --head-hidden "$HEAD_HIDDEN"     --dropout "$DROPOUT" --weight-decay "$WEIGHT_DECAY"
python -m eval.predictor_eval_standalone
python -m data.features.predictor_scores

echo "=== Done: $(date) ==="
