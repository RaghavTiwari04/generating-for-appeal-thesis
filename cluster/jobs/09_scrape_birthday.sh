#!/bin/bash
#SBATCH --job-name=gc-scrape
#SBATCH --partition=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j-scrape.out
#SBATCH --error=logs/slurm-%j-scrape.err

# Sweep the birthday catalogue, download images, then classify subtypes.
#
# Order matters: scrape only stores image URLs in listings.raw_metadata, so
# listing_images stays empty until the downloader runs. Classification is a
# separate pass so the search query never decides the subtype label.
#
# NOT run here (expensive, run separately afterwards):
#   01_clip_embed.sh   — embeddings for new listings
#   03a_vlm_labels.sh  — VLM scoring, costs API credits per new card

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh


PSQL="psql -h localhost -p 5433 -d greeting_cards -v ON_ERROR_STOP=1"
LIMIT="${LIMIT:-1000}"

echo "=== Birthday catalogue sweep ==="
echo "Node: $(hostname)  Start: $(date)  limit=$LIMIT per query"

echo ""
echo "--- Before ---"
$PSQL -c "SELECT source, COUNT(*) AS n FROM listings GROUP BY 1 ORDER BY 2 DESC;"

echo ""
echo "--- Scraping redbubble ---"
python -u -m data.scrapers.run_scraper --source redbubble --limit "$LIMIT"

echo ""
echo "--- Scraping greetings_island ---"
python -u -m data.scrapers.run_scraper --source greetings_island --limit "$LIMIT"


echo ""
echo "--- Downloading images ---"
python -u -m data.scrapers.image_downloader --limit 20000


echo ""
echo "--- After ---"
$PSQL -c "SELECT source, COUNT(*) AS n FROM listings GROUP BY 1 ORDER BY 2 DESC;"
$PSQL -c "SELECT COUNT(*) AS listings_without_images
          FROM listings l
          WHERE NOT EXISTS (SELECT 1 FROM listing_images li
                            WHERE li.listing_id = l.listing_id
                              AND li.storage_path IS NOT NULL);"

echo ""
echo "Next: sbatch cluster/jobs/14_nli_subtypes.sh   (occasion classification, needs a GPU)"
echo "=== Done: $(date) ==="
