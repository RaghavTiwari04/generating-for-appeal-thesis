#!/bin/bash
#SBATCH --job-name=gc-lora
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j-lora.out
#SBATCH --error=logs/slurm-%j-lora.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}

# LoRA fine-tuning per occasion — needs 24GB+ VRAM
# Use a40 (48GB) or a100 (80GB). a30 (24GB) might work tight.

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

echo "=== LoRA training ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

# One LoRA over all four birthday subtypes, not one each.
#
# The text encoder is not trained, so the LoRA carries style only and the
# occasion's semantics come from the prompt. The four subtypes share one visual
# idiom, so training them separately fits nearly the same distribution four
# times from a quarter of the data each. Passing the group name selects every
# `birthday/*` subtype, and generation resolves `birthday/kids` to this LoRA
# when no subtype-specific one exists.
#
# To go back to per-subtype LoRAs, loop over the four occasion strings instead;
# the trainer still accepts them and generation prefers a subtype match.
OCCASION="${OCCASION:-birthday}"
N_IMAGES="${N_IMAGES:-150}"

echo "--- Training LoRA for: $OCCASION ($(date)) ---"
python -m generation.image.loras.train_lora     --occasion "$OCCASION" --rank 32 --steps 1000 --lr 1e-4 --n-images "$N_IMAGES"

echo "=== Done: $(date) ==="
