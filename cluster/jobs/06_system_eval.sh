#!/bin/bash
#SBATCH --job-name=gc-llm-eval
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j-llm-eval.out
#SBATCH --error=logs/slurm-%j-llm-eval.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}

# LLM system evaluation — scores generated cards via SSR + rubric judge.
# CPU-only (API calls + sentence-transformers embeddings).
# Needs: OPENAI_API_KEY in .env, sentence-transformers installed.
# Runs AFTER card generation (step 05).

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

echo "=== LLM system evaluation ==="
echo "Start: $(date)"

OCCASIONS="birthday/general,birthday/milestone,birthday/kids,birthday/relationship"

python -m eval.llm_system_eval \
    --occasions "$OCCASIONS" \
    --provider openai \
    --out-dir ./artifacts/llm_system_eval

echo "=== Done: $(date) ==="
