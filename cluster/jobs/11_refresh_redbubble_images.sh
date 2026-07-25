#!/bin/bash
#SBATCH --job-name=gc-rb-images
#SBATCH --partition=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%j-rb-images.out
#SBATCH --error=logs/slurm-%j-rb-images.err

# Replace Redbubble card mockups with flat artwork.
#
# Stored Redbubble primaries are the tilted 3D card render (papergc), not the
# artwork, because the old URL rewrite targeted a transform Redbubble does not
# serve. That fed paper edges and perspective into LoRA training and VLM
# scoring across most of the dataset.
#
# Order matters:
#   1. re-parse cached pages  -> raw_metadata.image_urls gets flat URLs
#   2. drop Redbubble listing_images rows
#   3. re-download           -> downloader only fetches listings with no image
#
# listing_images rows are snapshotted first; restore SQL is printed at the end.
# Old blobs are left in place (orphaned but harmless).

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

PG_DIR="/vol/bitbucket/$USER/pgdata"
trap 'pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true' EXIT

PSQL="psql -h localhost -p 5433 -d greeting_cards -v ON_ERROR_STOP=1"
BACKUP="rb_images_backup_$(date +%Y%m%d_%H%M%S)"

echo "=== Redbubble image refresh ==="
echo "Node: $(hostname)  Start: $(date)"

echo ""
echo "--- Before: primary image variant mix ---"
$PSQL -c "SELECT CASE
                   WHEN (l.raw_metadata->'image_urls'->>0) LIKE '%/flat,%'    THEN 'flat (artwork)'
                   WHEN (l.raw_metadata->'image_urls'->>0) LIKE '%/papergc,%' THEN 'papergc (mockup)'
                   ELSE 'other'
                 END AS variant,
                 COUNT(*) AS n
          FROM listings l
          WHERE l.source = 'redbubble' AND l.raw_metadata ? 'image_urls'
          GROUP BY 1 ORDER BY 2 DESC;"

echo ""
echo "--- Snapshotting Redbubble listing_images to $BACKUP ---"
$PSQL -c "CREATE TABLE $BACKUP AS
          SELECT li.* FROM listing_images li
          JOIN listings l USING (listing_id)
          WHERE l.source = 'redbubble';"
$PSQL -c "SELECT COUNT(*) AS rows_backed_up FROM $BACKUP;"

echo ""
echo "--- Re-parsing cached pages (flat URLs into raw_metadata) ---"
python -u -m scripts.reparse_redbubble

echo ""
echo "--- After re-parse: primary image variant mix ---"
$PSQL -c "SELECT CASE
                   WHEN (l.raw_metadata->'image_urls'->>0) LIKE '%/flat,%'    THEN 'flat (artwork)'
                   WHEN (l.raw_metadata->'image_urls'->>0) LIKE '%/papergc,%' THEN 'papergc (mockup)'
                   ELSE 'other'
                 END AS variant,
                 COUNT(*) AS n
          FROM listings l
          WHERE l.source = 'redbubble' AND l.raw_metadata ? 'image_urls'
          GROUP BY 1 ORDER BY 2 DESC;"

echo ""
echo "--- Dropping Redbubble listing_images so they are re-fetched ---"
$PSQL -c "DELETE FROM listing_images li
          USING listings l
          WHERE li.listing_id = l.listing_id AND l.source = 'redbubble';"

echo ""
echo "--- Re-downloading images ---"
python -u -m data.scrapers.image_downloader --limit 20000

echo ""
echo "--- After ---"
$PSQL -c "SELECT l.source, COUNT(DISTINCT li.listing_id) AS listings_with_images
          FROM listings l JOIN listing_images li USING (listing_id)
          WHERE li.storage_path IS NOT NULL
          GROUP BY 1 ORDER BY 2 DESC;"

echo ""
echo "RESTORE if this went wrong:"
echo "  DELETE FROM listing_images li USING listings l"
echo "   WHERE li.listing_id = l.listing_id AND l.source = 'redbubble';"
echo "  INSERT INTO listing_images SELECT * FROM $BACKUP;"

echo "=== Done: $(date) ==="
