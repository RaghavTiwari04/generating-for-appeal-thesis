#!/bin/bash
#SBATCH --job-name=gc-reclass-occ
#SBATCH --partition=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j-reclass-occ.out
#SBATCH --error=logs/slurm-%j-reclass-occ.err

# Clear all occasion labels and re-classify from scratch using the current
# keyword rules on TITLES ONLY.
#
# This rewrites listing_features.occasion for every listing, which changes:
#   - which images train each per-occasion LoRA
#   - the bestseller titles injected into the brief prompt
#   - which listings condition D samples
#   - the occasion feature fed to the predictor
# Existing labels are snapshotted to a backup table first (see RESTORE below).

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

PG_DIR="/vol/bitbucket/$USER/pgdata"
trap 'pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true' EXIT

PSQL="psql -h localhost -p 5433 -d greeting_cards -v ON_ERROR_STOP=1"
BACKUP="occasion_backup_$(date +%Y%m%d_%H%M%S)"

echo "=== Occasion re-classification ==="
echo "Node: $(hostname)  Start: $(date)"

echo ""
echo "--- Before ---"
$PSQL -c "SELECT occasion, COUNT(*) AS n FROM listing_features GROUP BY 1 ORDER BY 2 DESC;"

echo ""
echo "--- Snapshotting current labels to $BACKUP ---"
$PSQL -c "CREATE TABLE $BACKUP AS
          SELECT listing_id, occasion, occasion_confidence, occasion_multilabel
          FROM listing_features;"
$PSQL -c "SELECT COUNT(*) AS rows_backed_up FROM $BACKUP;"

echo ""
echo "--- Clearing all occasion labels ---"
$PSQL -c "UPDATE listing_features
          SET occasion = NULL,
              occasion_confidence = NULL,
              occasion_multilabel = NULL,
              computed_at = NOW();"

echo ""
echo "--- Re-classifying (titles only) ---"
python -u -m data.features.occasion_classifier infer

echo ""
echo "--- After ---"
$PSQL -c "SELECT occasion, COUNT(*) AS n FROM listing_features GROUP BY 1 ORDER BY 2 DESC;"

echo ""
echo "RESTORE if this went wrong:"
echo "  UPDATE listing_features lf"
echo "  SET occasion = b.occasion,"
echo "      occasion_confidence = b.occasion_confidence,"
echo "      occasion_multilabel = b.occasion_multilabel"
echo "  FROM $BACKUP b WHERE b.listing_id = lf.listing_id;"

echo "=== Done: $(date) ==="
