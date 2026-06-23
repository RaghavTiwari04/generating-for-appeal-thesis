#!/bin/bash
#SBATCH --job-name=gc-vlm-label
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j-vlm-label.out
#SBATCH --error=logs/slurm-%j-vlm-label.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}

# VLM labelling — scores scraped cards on 4 dimensions via Claude Sonnet 4.
# CPU-only (API calls). Needs ANTHROPIC_API_KEY in .env.
# Runs AFTER occasion classifier (needs occasion labels to filter).

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

echo "=== VLM labelling ==="
echo "Provider: anthropic (claude-sonnet-4-6)"
echo "Start: $(date)"

python -m data.labels.vlm_labels label \
    --provider anthropic \
    --five-heads

echo ""
echo "--- Label stats ---"
python -m data.labels.vlm_labels stats

echo "=== Done: $(date) ==="
