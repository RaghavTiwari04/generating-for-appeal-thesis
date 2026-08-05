#!/bin/bash
#SBATCH --job-name=gc-generate
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j-generate.out
#SBATCH --error=logs/slurm-%j-generate.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}

# Generate cards for all 3 evaluation conditions (A/B/C) across birthday occasions.
# Needs: LoRA weights (step 04), predictor checkpoint (step 03).
# Needs: ANTHROPIC_API_KEY (brief/message gen), OPENAI_API_KEY (LLM reranking).

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

echo "=== Card generation (all conditions) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

# Postgres watchdog — restart if it crashes during long generation
PG_DIR="/vol/bitbucket/$USER/pgdata"
(while true; do
    sleep 60
    if ! pg_isready -h localhost -p 5433 -d greeting_cards -q 2>/dev/null; then
        echo "[watchdog] Postgres down, restarting... $(date)"
        pg_ctl -D "$PG_DIR" -l "$PG_DIR/postgres.log" start 2>/dev/null || true
        sleep 3
    fi
done) &
WATCHDOG_PID=$!
# Kill the watchdog first, then shut Postgres down cleanly — an unclean exit
# forces crash recovery on the next job's startup.
cleanup() {
    kill $WATCHDOG_PID 2>/dev/null || true
    # SLURM allows only ~30s between SIGTERM and SIGKILL on scancel, so a long
    # timeout here just gets killed mid-shutdown and forces crash recovery on
    # the next start — the very thing this is trying to avoid.
    pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true
}
trap cleanup EXIT

OCCASIONS="birthday/general,birthday/milestone,birthday/kids,birthday/relationship"
# Cards per condition per occasion, so 4 occasions x N_PER x 3 conditions.
#
# At N_PER=5 the TOST equivalence test between B and human bestsellers came out
# at p=0.054 with the means 0.0006 apart — the point estimate could hardly be
# closer, and 20 cards a condition still cannot squeeze the interval inside the
# 0.02 margin. Raising this is the only thing that resolves it.
#
# Condition C renders 8 candidates per card, so cost is roughly
# 4 x N_PER x 10 renders at ~13s each, plus a pipeline reload between the
# no-LoRA A cards and the rest of each occasion.
N_PER="${N_PER:-5}"
SEED="${SEED:-20000}"
# Labels every card this run produces, so the analysis scores one run instead
# of pooling every card ever generated — including smoke tests made under an
# earlier prompt and a different reranking objective.
RUN_TAG="${RUN_TAG:-run_$(date +%Y%m%d_%H%M)}"
echo "run_tag=$RUN_TAG n_per=$N_PER seed=$SEED (expect $((4 * N_PER * 3)) cards)"

export PYTHONUNBUFFERED=1
python -u -m pipeline.conditions \
    --occasions "$OCCASIONS" \
    --conditions "A,B,C" \
    --n "$N_PER" \
    --seed "$SEED" \
    --run-tag "$RUN_TAG"

echo "=== Done: $(date) ==="
