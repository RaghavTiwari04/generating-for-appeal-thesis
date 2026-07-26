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
# Classifies every listing title. Needs a GPU.
#
# MODEL= and TEMPLATE= override the checkpoint and hypothesis template.
# deberta-v3-large-zeroshot-v2.0 was tried and collapsed the taxonomy
# (2434/3905 into milestone); retest it with TEMPLATE='This example is {}.'
# before concluding the model is unsuitable.
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


DRY_RUN="${DRY_RUN:-1}"
THRESHOLD="${THRESHOLD:-0.55}"
MODEL="${MODEL:-facebook/bart-large-mnli}"
# Set with a conditional, not ${VAR:-default}: the "}" inside "{}" closes the
# parameter expansion early and appends the remainder as literal text.
if [ -z "${TEMPLATE:-}" ]; then TEMPLATE='This is {}.'; fi

echo "=== NLI subtype classification ==="
echo "Node: $(hostname)  Start: $(date)  dry_run=$DRY_RUN threshold=$THRESHOLD model=$MODEL"
echo "Template: $TEMPLATE"
nvidia-smi --query-gpu=name --format=csv,noheader || true

echo ""
echo "--- Distribution before ---"
python -u -m scripts.audit_labels

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "--- NLI (dry run) ---"
    python -u -m data.features.occasion_nli --dry-run --threshold "$THRESHOLD" --model-id "$MODEL" --hypothesis-template "$TEMPLATE"
else
    echo "--- NLI (writing labels) ---"
    python -u -m data.features.occasion_nli --threshold "$THRESHOLD" --model-id "$MODEL" --hypothesis-template "$TEMPLATE"
    echo ""
    echo "--- Distribution after NLI ---"
    python -u -m scripts.audit_labels
fi

echo "=== Done: $(date) ==="
