#!/bin/bash
#SBATCH --job-name=gc-relabel
#SBATCH --partition=a16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j-relabel.out
#SBATCH --error=logs/slurm-%j-relabel.err

# Audit occasion labels, re-classify with the current rules, audit again.
#
# The first audit shows what is wrong in the database now; the second shows
# what survives the fixed rules. Any remaining violations are rule gaps worth
# another pass — the target is zero.

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

PG_DIR="/vol/bitbucket/$USER/pgdata"
trap 'pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true' EXIT

echo "=== Occasion label audit and relabel ==="
echo "Node: $(hostname)  Start: $(date)"

echo ""
echo "########## BEFORE — labels currently in the database ##########"
python -u -m scripts.audit_labels --stored

echo ""
echo "########## Re-classifying with current rules ##########"
python -u -m data.features.occasion_classifier infer

echo ""
echo "########## AFTER — labels now in the database ##########"
python -u -m scripts.audit_labels --stored

echo "=== Done: $(date) ==="
