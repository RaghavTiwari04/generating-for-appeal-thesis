#!/bin/bash
#SBATCH --job-name=gc-nli
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%j-nli.out
#SBATCH --error=logs/slurm-%j-nli.err

# Zero-shot birthday subtype classification from titles via NLI entailment.
#
# Classifies every listing title. Needs a GPU for deberta-v3-large-zeroshot.
#
# Generous time limit: importing torch/transformers from the NFS venv on a
# cold node took over an hour, before the ~1.7GB model download. Inference
# itself is minutes. Subsequent runs are far faster — the model is cached
# under HF_HOME on /vol/bitbucket.
#
# DRY_RUN=1 (default) reports what would change without writing.
# DRY_RUN=0 applies the labels.

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

PG_DIR="/vol/bitbucket/$USER/pgdata"
trap 'pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true' EXIT

DRY_RUN="${DRY_RUN:-1}"
THRESHOLD="${THRESHOLD:-0.55}"
MODEL="${MODEL:-MoritzLaurer/deberta-v3-large-zeroshot-v2.0}"

echo "=== NLI subtype classification ==="
echo "Node: $(hostname)  Start: $(date)  dry_run=$DRY_RUN threshold=$THRESHOLD model=$MODEL"
nvidia-smi --query-gpu=name --format=csv,noheader || true

echo ""
echo "--- Distribution before ---"
python -u -m scripts.audit_labels

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "--- NLI (dry run) ---"
    python -u -m data.features.occasion_nli --dry-run --threshold "$THRESHOLD" --model-id "$MODEL"
else
    echo "--- NLI (writing labels) ---"
    python -u -m data.features.occasion_nli --threshold "$THRESHOLD" --model-id "$MODEL"
    echo ""
    echo "--- Distribution after NLI ---"
    python -u -m scripts.audit_labels
fi

echo "=== Done: $(date) ==="
