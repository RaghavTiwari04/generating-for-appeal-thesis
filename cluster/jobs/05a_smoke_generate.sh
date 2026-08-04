#!/bin/bash
#SBATCH --job-name=gc-smoke
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j-smoke.out
#SBATCH --error=logs/slurm-%j-smoke.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}

# A few cards from one occasion, to check a LoRA before spending a full A/B/C
# run on it. Flux needs more than 24GB, and the cluster refuses interactive
# jobs on those GPUs, so this exists as a batch script rather than an srun.
#
# a100 to match 05_generate_cards.sh, not merely for speed: DiffusionConfig
# sets free_between_passes from available VRAM at a 60GB threshold, so on a 48GB
# a40 the gen pipeline is torn down before every Fill pass and the LoRA reloaded
# for the next card. A smoke test on a40 would exercise a different branch from
# the run it is meant to derisk, and would misreport per-card timing.
#
# What to read in the log:
#   "Loaded LoRA: <name>"                    the LoRA resolved and applied.
#                                            Absent means base Flux, so
#                                            condition B would equal condition A.
#   "Headline rendered into artwork for N/M" how often Flux lettered the card
#                                            itself instead of falling back to
#                                            the typographic overlay.

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

OCCASION="${OCCASION:-birthday/general}"
N="${N:-4}"
# Unset by default, so each brief picks its own register from the occasion's
# bestsellers — the path the evaluation takes. Set TONE to pin one, which is
# what the website does when a customer chooses.
TONE="${TONE:-}"
# LORA_SCALE=0 generates from base Flux, which is the control for "is the LoRA
# the reason the headline is not being lettered".
export LORA_SCALE="${LORA_SCALE:-0.4}"
# Keep the covers that failed headline verification. The finished card always
# carries overlay text, so it cannot show what the model drew on its own.
export REJECTED_DIR="${REJECTED_DIR:-./artifacts/rejected_covers}"

echo "=== Smoke generation: $OCCASION ($N candidates, lora_scale=$LORA_SCALE, tone=${TONE:-brief chooses}) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

TONE_ARG=()
if [ -n "$TONE" ]; then TONE_ARG=(--tone "$TONE"); fi
python -u -m pipeline.orchestrator "$OCCASION" "${TONE_ARG[@]}" --n "$N" --top-k "$N"

echo ""
echo "--- Lettering outcome for the cards just generated ---"
psql -h localhost -p 5433 -d greeting_cards -t -c "
SELECT cover_path,
       predicted_scores->>'text_in_image' AS lettered,
       round((predicted_scores->>'headline_match')::numeric, 2) AS ocr_match
FROM generated_cards
ORDER BY generated_at DESC
LIMIT $N;"

echo "=== Done: $(date) ==="
