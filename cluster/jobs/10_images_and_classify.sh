#!/bin/bash
#SBATCH --job-name=gc-img-clf
#SBATCH --partition=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j-img-clf.out
#SBATCH --error=logs/slurm-%j-img-clf.err

# Download images for any listings missing them, then classify occasions.
#
# Split out of 09_scrape_birthday.sh because scraping is rate-limited and can
# consume a whole 8h allocation on its own, leaving these phases unreached.
# Both are resumable: the downloader only fetches listings with no stored
# image, and classification is idempotent.

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

PG_DIR="/vol/bitbucket/$USER/pgdata"
trap 'pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true' EXIT

PSQL="psql -h localhost -p 5433 -d greeting_cards -v ON_ERROR_STOP=1"

echo "=== Images + occasion classification ==="
echo "Node: $(hostname)  Start: $(date)"

echo ""
echo "--- Before ---"
$PSQL -c "SELECT source, COUNT(*) AS n FROM listings GROUP BY 1 ORDER BY 2 DESC;"
$PSQL -c "SELECT COUNT(*) AS listings_without_images
          FROM listings l
          WHERE NOT EXISTS (SELECT 1 FROM listing_images li
                            WHERE li.listing_id = l.listing_id
                              AND li.storage_path IS NOT NULL);"

echo ""
echo "--- Downloading images ---"
python -u -m data.scrapers.image_downloader --limit 20000

echo ""
echo "--- Classifying occasions (titles only) ---"
python -u -m data.features.occasion_classifier infer

echo ""
echo "--- After ---"
$PSQL -c "SELECT lf.occasion, COUNT(*) AS n FROM listing_features lf GROUP BY 1 ORDER BY 2 DESC;"
$PSQL -c "SELECT COUNT(*) AS listings_without_images
          FROM listings l
          WHERE NOT EXISTS (SELECT 1 FROM listing_images li
                            WHERE li.listing_id = l.listing_id
                              AND li.storage_path IS NOT NULL);"

echo "=== Done: $(date) ==="
