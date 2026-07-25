#!/usr/bin/env bash
# Master pipeline runner — submits all SLURM jobs in dependency order.
#
# Usage:
#   cd /vol/bitbucket/$USER/masters_thesis
#   bash cluster/jobs/run_all.sh          # full pipeline (includes clean slate)
#   bash cluster/jobs/run_all.sh --no-clean  # skip DB wipe, re-run from features
#
# Each step waits for its dependencies via --dependency=afterok:<jobid>.
# If any step fails, downstream jobs are cancelled automatically.

set -euo pipefail

SKIP_CLEAN=false
if [[ "${1:-}" == "--no-clean" ]]; then
    SKIP_CLEAN=true
    shift
fi

mkdir -p logs

echo "============================================="
echo "  Greeting Card Pipeline — Full Run"
echo "  $(date)"
echo "============================================="

# --- Preflight checks ---
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Create it with API keys." >&2
    exit 1
fi

source .env
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY not set in .env" >&2
    exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set in .env" >&2
    exit 1
fi

echo "API keys: OK"
echo ""

# --- Step 0: Clean slate (optional) ---
if [[ "$SKIP_CLEAN" == false ]]; then
    echo "Step 0: Clean slate..."
    JOB0=$(sbatch --parsable cluster/jobs/00_clean_slate.sh)
    echo "  Submitted: $JOB0"
    AFTER_CLEAN="--dependency=afterok:$JOB0"
else
    echo "Step 0: Skipped (--no-clean)"
    AFTER_CLEAN=""
fi

# --- Step 1: Feature extraction (CLIP + OCR + palette + complexity) ---
echo "Step 1: Feature extraction..."
JOB1=$(sbatch --parsable $AFTER_CLEAN cluster/jobs/01_clip_embed.sh)
echo "  Submitted: $JOB1"

# --- Step 2: Occasion classification (NLI zero-shot over titles) ---
echo "Step 2: Occasion classification..."
JOB2=$(DRY_RUN=0 sbatch --parsable --dependency=afterok:$JOB1 cluster/jobs/14_nli_subtypes.sh)
echo "  Submitted: $JOB2"

# --- Step 3: VLM labelling (needs occasion from step 2) ---
echo "Step 3: VLM labelling..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 cluster/jobs/03a_vlm_labels.sh)
echo "  Submitted: $JOB3"

# --- Step 3b: Predictor training (needs CLIP + VLM labels) ---
echo "Step 3b: Predictor training..."
JOB3B=$(sbatch --parsable --dependency=afterok:$JOB3 cluster/jobs/03_train_predictor.sh)
echo "  Submitted: $JOB3B"

# --- Step 4: LoRA training (needs saleability labels from step 3) ---
echo "Step 4: LoRA training..."
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 cluster/jobs/04_train_lora.sh)
echo "  Submitted: $JOB4"

# --- Step 5: Card generation (needs LoRA + predictor) ---
echo "Step 5: Card generation (all conditions)..."
JOB5=$(sbatch --parsable --dependency=afterok:$JOB3B:$JOB4 cluster/jobs/05_generate_cards.sh)
echo "  Submitted: $JOB5"

# --- Step 6: LLM system eval (needs generated cards) ---
echo "Step 6: LLM system evaluation..."
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 cluster/jobs/06_system_eval.sh)
echo "  Submitted: $JOB6"

echo ""
echo "============================================="
echo "  All jobs submitted!"
echo "============================================="
echo ""
echo "  Pipeline:"
echo "    00 clean     : ${JOB0:-skipped}"
echo "    01 features  : $JOB1"
echo "    02 occasion  : $JOB2"
echo "    03 vlm-label : $JOB3"
echo "    03b predictor: $JOB3B"
echo "    04 lora      : $JOB4"
echo "    05 generate  : $JOB5"
echo "    06 llm-eval  : $JOB6"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Logs:    tail -f logs/slurm-<jobid>-*.out"
echo ""
echo "  Estimated total wall time: ~12-16 hours"
echo "  (steps 3b + 4 run in parallel)"
