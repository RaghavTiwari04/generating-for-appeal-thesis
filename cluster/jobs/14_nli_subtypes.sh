#!/bin/bash
#SBATCH --job-name=gc-nli
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j-nli.out
#SBATCH --error=logs/slurm-%j-nli.err

# Zero-shot birthday subtype classification from titles via NLI entailment.
#
# Runs the keyword pass first so explicit cases are settled by rules, then NLI
# only where the rules had no evidence. Needs a GPU for bart-large-mnli.
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

echo "=== NLI subtype classification ==="
echo "Node: $(hostname)  Start: $(date)  dry_run=$DRY_RUN threshold=$THRESHOLD"
nvidia-smi --query-gpu=name --format=csv,noheader || true

echo ""
echo "--- Keyword pass (settles explicit cases) ---"
python -u -m data.features.occasion_classifier infer

echo ""
echo "--- Distribution after keyword rules ---"
python -u -m scripts.audit_labels --stored

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "--- NLI (dry run) ---"
    python -u -m data.features.occasion_nli --dry-run --threshold "$THRESHOLD"
else
    echo "--- NLI (writing labels) ---"
    python -u -m data.features.occasion_nli --threshold "$THRESHOLD"
    echo ""
    echo "--- Distribution after NLI ---"
    python -u -m scripts.audit_labels --stored
fi

echo "=== Done: $(date) ==="
