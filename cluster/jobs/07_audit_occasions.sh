#!/bin/bash
#SBATCH --job-name=gc-audit-occ
#SBATCH --partition=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j-audit-occ.out
#SBATCH --error=logs/slurm-%j-audit-occ.err

# Read-only audit: do stored occasion labels match the current keyword rules?
# CPU only, no GPU. Runs on a compute node so Postgres is not on the shared
# login node, and shuts the DB down cleanly so the next job does not pay for
# crash recovery.

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

PG_DIR="/vol/bitbucket/$USER/pgdata"
trap 'pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true' EXIT

echo "=== Occasion label audit ==="
echo "Node: $(hostname)  Start: $(date)"

python -u -m scripts.audit_occasions

echo "=== Done: $(date) ==="
