#!/bin/bash
#SBATCH --job-name=gc-clusters
#SBATCH --partition=a16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j-clusters.out
#SBATCH --error=logs/slurm-%j-clusters.err

# Export duplicate clusters as an HTML gallery so the clustering can be checked
# by eye before anything downstream relies on it.
#
# Output: ./cluster_gallery/  (copy off the cluster and open index.html)
# CLUSTERS= and PER_CLUSTER= control how much is shown.

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh


CLUSTERS="${CLUSTERS:-40}"
PER_CLUSTER="${PER_CLUSTER:-12}"

echo "=== Duplicate cluster gallery ==="
echo "Node: $(hostname)  Start: $(date)  clusters=$CLUSTERS per_cluster=$PER_CLUSTER"

python -u -m scripts.export_cluster_gallery \
    --out ./cluster_gallery \
    --clusters "$CLUSTERS" \
    --per-cluster "$PER_CLUSTER"

echo ""
echo "Copy to your laptop with:"
echo "  scp -r rt325@shell5.doc.ic.ac.uk:/vol/bitbucket/$USER/masters_thesis/cluster_gallery ~/Desktop/"
echo "=== Done: $(date) ==="
