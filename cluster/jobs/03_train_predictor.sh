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

# Seeds to average over. One run is not a measurement: seed-to-seed sd on
# purchase intent is about 0.013, so anything under roughly 0.03 is noise.
SEEDS="${SEEDS:-5}"
TRUNK="${TRUNK:-512}"
HEAD_HIDDEN="${HEAD_HIDDEN:-128}"
DROPOUT="${DROPOUT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
# 1,792 cards at batch 64 is 28 steps an epoch, so the epoch count is really a
# step budget. These defaults used to be 30 epochs at lr 1e-4 — about 840 steps
# — which is the configuration measured at 0.510 on purchase intent against
# 0.586 for the values below. Every comparison the job ran against the ridge
# baseline while those defaults stood was therefore comparing ridge to a
# deliberately under-trained MLP, which is exactly the confound to avoid: ridge
# solves its objective exactly, so an under-trained MLP loses to it for reasons
# that look like capacity but are not.
EPOCHS="${EPOCHS:-1500}"
LR="${LR:-1e-2}"
PATIENCE="${PATIENCE:-150}"
# Both measured harmful, so off by default. SKIP=--skip-connection /
# NORM=--input-norm to reproduce the ablation.
SKIP="${SKIP:---no-skip-connection}"
NORM="${NORM:---no-input-norm}"
# Per-dimension z-scoring from the training split. Ridge picks its penalty per
# head by CV; the MLP applies one weight decay to raw embeddings, so this is
# the closest thing to the advantage ridge gets for free.
STANDARDISE="${STANDARDISE:---no-standardise}"
echo "seeds=$SEEDS trunk=$TRUNK head=$HEAD_HIDDEN dropout=$DROPOUT wd=$WEIGHT_DECAY"
echo "epochs=$EPOCHS lr=$LR patience=$PATIENCE standardise=$STANDARDISE"

python -m models.predictor.train --batch-size 64     --epochs "$EPOCHS" --lr "$LR" --early-stop-patience "$PATIENCE"     --seeds "$SEEDS" --trunk-hidden "$TRUNK" --head-hidden "$HEAD_HIDDEN"     --dropout "$DROPOUT" --weight-decay "$WEIGHT_DECAY" "$SKIP" "$NORM" "$STANDARDISE"

# The linear control, fitted on the same split. It is what reranking uses by
# default, and it is the bar the MLP has to clear to earn its place.
python -m models.predictor.ridge

python -m eval.predictor_eval_standalone
python -m data.features.predictor_scores

echo "=== Done: $(date) ==="
