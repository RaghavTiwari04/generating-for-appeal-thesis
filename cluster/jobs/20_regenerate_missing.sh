#!/bin/bash
#SBATCH --job-name=gc-regen
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j-regen.out
#SBATCH --error=logs/slurm-%j-regen.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}

# Recover the one condition B card a generation failure cost, taking the
# evaluation set from 159 to the designed 160.
#
# The seed is derived, not chosen. The dry run confirmed:
#   birthday/kids seeds 22100..22109, missing 22103
#   22103 = 20000 + 2*1000 + 1*100 + 3, the design's own formula
# so this regenerates a pre-specified design point rather than adding a card.
#
# Generated once. No second attempt, no selection among attempts. Whatever it
# produces goes into the reported set, including if it scores badly.
#
# GPU required: loads FLUX. Needs the LoRA (step 04), the ridge predictor
# (step 03) and ANTHROPIC_API_KEY for the brief and message.
#
# Afterwards the card still has to be scored, by the same judge as the rest of
# the set and by both robustness judges, and every figure and derived number
# regenerated. See step 06 and eval.judge_robustness.

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

RUN_TAG="${RUN_TAG:-run_20260805_0236}"

echo "=== Recovering the missing condition B card ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "run_tag=$RUN_TAG"
echo "Start: $(date)"

# Dry run first, in the job log, so the derived seed is recorded next to the
# generation that used it.
python -u -m scripts.regenerate_missing_card --run-tag "$RUN_TAG"

echo
echo "--- applying ---"
python -u -m scripts.regenerate_missing_card --run-tag "$RUN_TAG" --apply

echo "Done: $(date)"
echo
echo "Next: score the new card with the reported judge, then re-run"
echo "  eval.judge_robustness for gpt-4o and claude-sonnet-4.6,"
echo "  eval.reports.thesis_figures and eval.reports.judge_agreement."
