#!/bin/bash
#SBATCH --job-name=gc-clus-insp
#SBATCH --partition=a16
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j-clus-insp.out
#SBATCH --error=logs/slurm-%j-clus-insp.err

# Explain why the large duplicate clusters were merged.
#
# Union-find takes a transitive closure, so a cluster can hold visibly
# different cards together through a chain of merely-similar pairs. This
# recomputes pairwise similarity within each cluster and reports whether the
# cluster is dense (genuine duplicates) or chained (a few links holding many
# members).

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

PG_DIR="/vol/bitbucket/$USER/pgdata"
trap 'pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true' EXIT

CLUSTERS="${CLUSTERS:-6}"

echo "=== Cluster merge inspection ==="
echo "Node: $(hostname)  Start: $(date)  clusters=$CLUSTERS"

python -u -m scripts.inspect_clusters --clusters "$CLUSTERS"

echo "=== Done: $(date) ==="
