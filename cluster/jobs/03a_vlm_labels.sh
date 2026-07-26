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
# Keep a comparison run out of the canonical source: the predictor, LoRA
# selection and condition D all read the default one, so a second provider's
# scores must not land in it.
LABEL_SOURCE="${LABEL_SOURCE:-}"
# Re-score cards that already have labels. Needed when overwriting a run made
# with a different provider or scoring config.
FORCE="${FORCE:-}"

echo "=== LLM labelling (SSR + rubric judge) ==="
echo "Node: $(hostname)  Start: $(date)"
echo "provider=$PROVIDER limit=${LIMIT:-all} source=${LABEL_SOURCE:-default} force=${FORCE:-no}"

ARGS=(--provider "$PROVIDER")
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
[ -n "$LABEL_SOURCE" ] && ARGS+=(--label-source "$LABEL_SOURCE")
[ -n "$FORCE" ] && ARGS+=(--force)

python -u -m data.labels.vlm_labels label "${ARGS[@]}"

STATS_ARGS=()
[ -n "$LABEL_SOURCE" ] && STATS_ARGS+=(--label-source "$LABEL_SOURCE")

echo ""
echo "--- Label stats ---"
python -u -m data.labels.vlm_labels stats "${STATS_ARGS[@]}"

echo "=== Done: $(date) ==="
