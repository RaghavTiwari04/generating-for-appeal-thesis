#!/bin/bash
#SBATCH --job-name=gc-app
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j-app.out
#SBATCH --error=logs/slurm-%j-app.err

# Serve the card generator web app.
#
# a100 because the app calls the pipeline in-process, so it holds Flux and
# Flux-Fill resident — the same reason 05_generate_cards.sh runs there. On a
# card under 60GB the pipeline tears itself down between passes and every
# request pays the reload.
#
# Compute nodes are not reachable from outside the cluster, so this prints an
# SSH tunnel command with the node it landed on. Run that on your laptop, then
# open http://localhost:8000. The tunnel is the whole deployment story: a
# public site would need a rented GPU host, or a queue that submits SLURM jobs
# from a frontend that lives elsewhere.

set -euo pipefail
. /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

PORT="${PORT:-8000}"
NODE=$(hostname -s)

echo "=== Card generator app ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Node: $NODE  Port: $PORT"
echo ""
echo "Run this on your laptop, then open http://localhost:$PORT"
echo ""
echo "  ssh -N -J $USER@shell1.doc.ic.ac.uk -L $PORT:$NODE:$PORT $USER@gpucluster2.doc.ic.ac.uk"
echo ""
echo "Start: $(date)"

# The first request would otherwise pay ~15 minutes of model loading while the
# browser waits. Loading now means the app is slow to start and fast to use.
python -u -c "
from generation.image.diffusion import get_runner
r = get_runner()
r._load_pipeline()
print('Flux loaded and resident', flush=True)
"

exec uvicorn app.api:app --host 0.0.0.0 --port "$PORT" --log-level info
