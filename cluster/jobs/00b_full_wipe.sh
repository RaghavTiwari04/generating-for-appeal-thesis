#!/bin/bash
#SBATCH --job-name=gc-wipe
#SBATCH --partition=a16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j-wipe.out
#SBATCH --error=logs/slurm-%j-wipe.err

# FULL WIPE — deletes scraped listings as well as everything derived from them.
#
# Unlike 00_clean_slate.sh (which preserves listings and images), this returns
# the database to empty so the catalogue can be rebuilt with the birthday gate
# and flat-image parsing applied uniformly.
#
# DESTRUCTIVE AND NOT AUTOMATICALLY REVERSIBLE:
#   - listings, listing_images, listing_features, listing_snapshots
#   - saleability_labels  (VLM labels — regenerating costs API credits)
#   - generated_cards
#   - stored image blobs in MinIO and $GC_BLOB_ROOT
#   - trained LoRA weights, predictor checkpoints, generated card artifacts
#
# DELIBERATELY PRESERVED:
#   - .cache/raw_html  — 30-day scraper cache. Keeping it means the re-scrape
#     re-parses from disk instead of refetching every page, and it is the only
#     copy of pages whose listings we are about to delete.
#   - survey_* saleability_labels — human Bradley-Terry scores from Prolific,
#     which cannot be regenerated at all.
#
# Requires CONFIRM=yes to run:
#     CONFIRM=yes sbatch cluster/jobs/00b_full_wipe.sh

set -euo pipefail

if [ "${CONFIRM:-}" != "yes" ]; then
    echo "Refusing to run without CONFIRM=yes." >&2
    echo "This deletes all scraped listings, images and VLM labels." >&2
    exit 1
fi

source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

PG_DIR="/vol/bitbucket/$USER/pgdata"
trap 'pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true' EXIT

PSQL="psql -h localhost -p 5433 -d greeting_cards -v ON_ERROR_STOP=1"

echo "=== FULL WIPE ==="
echo "Node: $(hostname)  Start: $(date)"

echo ""
echo "--- Before ---"
$PSQL -c "SELECT 'listings' AS tbl, COUNT(*) FROM listings
          UNION ALL SELECT 'listing_images', COUNT(*) FROM listing_images
          UNION ALL SELECT 'listing_features', COUNT(*) FROM listing_features
          UNION ALL SELECT 'saleability_labels', COUNT(*) FROM saleability_labels
          UNION ALL SELECT 'generated_cards', COUNT(*) FROM generated_cards;"

echo ""
echo "--- Preserving human survey labels ---"
$PSQL -c "CREATE TABLE IF NOT EXISTS survey_labels_preserved AS
          SELECT * FROM saleability_labels WHERE label_source LIKE 'survey_%';"
$PSQL -c "SELECT COUNT(*) AS survey_rows_preserved FROM survey_labels_preserved;"

echo ""
echo "--- Deleting DB rows ---"
# listing_images / listing_features / listing_snapshots cascade from listings,
# but delete explicitly so the counts below are meaningful.
$PSQL <<'SQL'
BEGIN;
DELETE FROM generated_cards;
DELETE FROM saleability_labels;
DELETE FROM listing_images;
DELETE FROM listing_features;
DELETE FROM listing_snapshots;
DELETE FROM listings;
COMMIT;
SQL

echo ""
echo "--- Deleting stored image blobs ---"
BLOB_ROOT="${GC_BLOB_ROOT:-/vol/bitbucket/$USER/blobstore}"
for d in "/vol/bitbucket/$USER/minio-data/greeting-cards" \
         "/vol/bitbucket/$USER/minio-data/greeting-cards-raw" \
         "$BLOB_ROOT"; do
    if [ -d "$d" ]; then
        echo "  removing contents of $d"
        find "$d" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
    fi
done

echo ""
echo "--- Deleting trained artifacts ---"
rm -rf artifacts/predictor/ artifacts/occasion_classifier.pt \
       artifacts/generated_cards/ artifacts/lora_train/ artifacts/llm_system_eval/
find generation/image/loras/ -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "--- Raw HTML cache (preserved) ---"
du -sh .cache/raw_html 2>/dev/null || echo "  (no cache)"

echo ""
echo "--- After ---"
$PSQL -c "SELECT 'listings' AS tbl, COUNT(*) FROM listings
          UNION ALL SELECT 'listing_images', COUNT(*) FROM listing_images
          UNION ALL SELECT 'listing_features', COUNT(*) FROM listing_features
          UNION ALL SELECT 'saleability_labels', COUNT(*) FROM saleability_labels
          UNION ALL SELECT 'generated_cards', COUNT(*) FROM generated_cards;"

echo ""
echo "Next: sbatch cluster/jobs/09_scrape_birthday.sh"
echo "=== Done: $(date) ==="
