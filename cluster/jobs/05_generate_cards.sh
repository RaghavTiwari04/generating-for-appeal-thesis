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
    if ! pg_isready -h localhost -p 5433 -q 2>/dev/null; then
        echo "[watchdog] Postgres down, restarting... $(date)"
        pg_ctl -D "$PG_DIR" -l "$PG_DIR/postgres.log" start 2>/dev/null || true
        sleep 3
    fi
done) &
WATCHDOG_PID=$!
trap "kill $WATCHDOG_PID 2>/dev/null" EXIT

OCCASIONS="birthday/general,birthday/milestone,birthday/kids,birthday/relationship"
N_PER=5

python -m pipeline.conditions \
    --occasions "$OCCASIONS" \
    --conditions "A,B,C" \
    --n "$N_PER" \
    --seed 20000

echo "=== Done: $(date) ==="
