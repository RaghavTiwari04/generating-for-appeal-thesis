#!/bin/bash
#SBATCH --job-name=gc-vlm-label
#SBATCH --partition=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j-vlm-label.out
#SBATCH --error=logs/slurm-%j-vlm-label.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}

# LLM labelling — scores scraped cards on 5 dimensions.
# SSR (Maier et al. 2025) for purchase intent, rubric judge (Zheng et al. 2023)
# for the quality dims. 10 API calls per card, so this is the step that costs
# money: check the call count it logs before letting a full run proceed.
#
# LIMIT=20 scores only the first 20 cards — a ~200-call smoke test that proves
# SSR, the judge and persistence all work before committing to the full run.
# Resumable: cards already scored are skipped, so a limited run is not wasted.
#
# API calls only, no GPU. Needs ANTHROPIC_API_KEY (judge + personas) and
# OPENAI_API_KEY (SSR embeddings) in .env, plus occasion labels from job 14
# since the pool filters on them.

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

PG_DIR="/vol/bitbucket/$USER/pgdata"
trap 'pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true' EXIT

PROVIDER="${PROVIDER:-anthropic}"
LIMIT="${LIMIT:-}"

echo "=== LLM labelling (SSR + rubric judge) ==="
echo "Node: $(hostname)  Start: $(date)  provider=$PROVIDER limit=${LIMIT:-all}"

if [ -n "$LIMIT" ]; then
    python -u -m data.labels.vlm_labels label --provider "$PROVIDER" --limit "$LIMIT"
else
    python -u -m data.labels.vlm_labels label --provider "$PROVIDER"
fi

echo ""
echo "--- Label stats ---"
python -u -m data.labels.vlm_labels stats

echo "=== Done: $(date) ==="
