#!/bin/bash
#SBATCH --job-name=gc-gallery
#SBATCH --partition=a16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j-gallery.out
#SBATCH --error=logs/slurm-%j-gallery.err

# Export scraped listings as an HTML gallery grouped by assigned occasion, so
# the classifier's labels and the stored cover images can both be eyeballed.
#
# Output: ./listing_gallery/  (copy off the cluster and open index.html)
# Override sample size with PER_OCCASION=n

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh


PER_OCCASION="${PER_OCCASION:-40}"

echo "=== Listing gallery ==="
echo "Node: $(hostname)  Start: $(date)  per_occasion=$PER_OCCASION"

python -u -m scripts.export_listing_gallery \
    --out ./listing_gallery \
    --per-occasion "$PER_OCCASION"

echo ""
echo "Copy to your laptop with:"
echo "  scp -r rt325@shell5.doc.ic.ac.uk:/vol/bitbucket/$USER/masters_thesis/listing_gallery ~/Desktop/"
echo "=== Done: $(date) ==="
