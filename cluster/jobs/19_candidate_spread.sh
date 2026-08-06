#!/bin/bash
#SBATCH --job-name=gc-spread
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j-spread.out
#SBATCH --error=logs/slurm-%j-spread.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}

# Within-batch candidate spread against the scraped test set.
#
# Settles the mechanism behind the reranking null: best-of-N gain is bounded by
# how far apart the candidates are, not by how well the scorer ranks them, and
# the within-batch spread has never been measured because a normal run persists
# only the returned card.
#
# GPU required. This loads FLUX and generates BATCHES * N cards, so it cannot
# run on a login or cloud VM; attempting it there is killed while loading the
# diffusion weights.
#
# Needs: LoRA weights (step 04), ridge predictor (step 03),
#        ANTHROPIC_API_KEY for the briefs.
#
# The probe writes its cards under condition_tag='probe_candidate_spread' so
# they can never be mistaken for the evaluated set.

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

N="${N:-8}"
BATCHES="${BATCHES:-12}"
SEED="${SEED:-9000}"

echo "=== Candidate spread probe ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "n=$N batches=$BATCHES seed=$SEED"
echo "Start: $(date)"

python -u -m eval.reports.candidate_spread \
    --n "$N" --batches "$BATCHES" --seed "$SEED"

echo "Done: $(date)"
echo
echo "Wrote artifacts/candidate_spread.json and report/figures/candidate_spread.pdf"
echo "Probe cards are tagged probe_candidate_spread; to remove them:"
echo "  psql -h localhost -p 5433 -d greeting_cards -c \\"
echo "    \"DELETE FROM generated_cards WHERE condition_tag='probe_candidate_spread';\""
