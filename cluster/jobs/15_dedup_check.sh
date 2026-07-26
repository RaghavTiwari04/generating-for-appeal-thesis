#!/bin/bash
#SBATCH --job-name=gc-dedup-chk
#SBATCH --partition=a16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%j-dedup-chk.out
#SBATCH --error=logs/slurm-%j-dedup-chk.err

# Read-only check on what dedup wrote.
#
# The run reported 4385 listings in 800 clusters against a catalogue of 3906,
# which is impossible — the union-find only holds ids drawn from these tables.
# Establish the database's own counts before trusting or fixing the reported
# figures.

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

PG_DIR="/vol/bitbucket/$USER/pgdata"
trap 'pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true' EXIT

PSQL="psql -h localhost -p 5433 -d greeting_cards -v ON_ERROR_STOP=1"

echo "=== Dedup check ==="
echo "Node: $(hostname)  Start: $(date)"

echo ""
echo "--- Table sizes ---"
$PSQL -c "SELECT
   (SELECT COUNT(*) FROM listings)                                    AS listings,
   (SELECT COUNT(*) FROM listing_features)                            AS features,
   (SELECT COUNT(*) FROM listing_images)                              AS images,
   (SELECT COUNT(*) FROM listing_images WHERE is_primary)             AS primary_images,
   (SELECT COUNT(DISTINCT listing_id) FROM listing_images WHERE is_primary) AS listings_with_primary;"

echo ""
echo "--- Listings with more than one primary image (would double-count) ---"
$PSQL -c "SELECT COUNT(*) AS listings_with_multiple_primaries FROM (
   SELECT listing_id FROM listing_images WHERE is_primary
   GROUP BY listing_id HAVING COUNT(*) > 1) x;"

echo ""
echo "--- Cluster assignment as stored ---"
$PSQL -c "SELECT
   COUNT(*) FILTER (WHERE duplicate_cluster_id IS NOT NULL) AS listings_in_a_cluster,
   COUNT(DISTINCT duplicate_cluster_id)                     AS distinct_clusters,
   COUNT(*) FILTER (WHERE duplicate_cluster_size > 1)       AS in_multi_member_cluster
 FROM listing_features;"

echo ""
echo "--- Cluster size distribution ---"
$PSQL -c "SELECT duplicate_cluster_size AS size, COUNT(*) AS listings
 FROM listing_features
 WHERE duplicate_cluster_id IS NOT NULL
 GROUP BY 1 ORDER BY 1 DESC LIMIT 15;"

echo ""
echo "--- Distinct cards remaining if one kept per cluster ---"
$PSQL -c "SELECT
   (SELECT COUNT(DISTINCT COALESCE(duplicate_cluster_id::text, listing_id::text))
    FROM listing_features) AS distinct_after_dedup,
   (SELECT COUNT(*) FROM listing_features) AS total_rows;"

echo ""
echo "--- Distinct designs per occasion (LoRA trains on 150) ---"
$PSQL -c "SELECT lf.occasion,
                 COUNT(*) AS listings,
                 COUNT(DISTINCT COALESCE(lf.duplicate_cluster_id::text, lf.listing_id::text)) AS distinct_designs
          FROM listing_features lf
          WHERE lf.occasion IS NOT NULL
          GROUP BY 1 ORDER BY 3 DESC;"

echo ""
echo "--- Largest clusters, with a sample title ---"
$PSQL -c "SELECT lf.duplicate_cluster_size AS size,
                 COUNT(*) AS rows,
                 MIN(LEFT(l.title, 58)) AS example
          FROM listing_features lf JOIN listings l USING (listing_id)
          WHERE lf.duplicate_cluster_size > 20
          GROUP BY 1 ORDER BY 1 DESC LIMIT 8;"

echo "=== Done: $(date) ==="
