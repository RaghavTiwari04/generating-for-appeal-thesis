#!/bin/bash
#SBATCH --job-name=gc-clean
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm-%j-clean.out
#SBATCH --error=logs/slurm-%j-clean.err

# Clean slate: wipe all computed data, keep scraped listings + images.
# Run this ONCE before re-running the full pipeline.

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

echo "=== Clean slate reset ==="
echo "Start: $(date)"

# --- DB cleanup ---
psql -h localhost -p 5433 -U gc -d greeting_cards <<'SQL'
BEGIN;

-- Delete all VLM / pseudo labels
DELETE FROM saleability_labels;

-- Delete all generated cards
DELETE FROM generated_cards;

-- Null out computed columns in listing_features (keep listing_id FK intact)
UPDATE listing_features SET
    clip_embedding = NULL,
    occasion = NULL,
    occasion_confidence = NULL,
    occasion_multilabel = NULL,
    extracted_text = NULL,
    palette_lab = NULL,
    image_complexity = NULL,
    duplicate_cluster_id = NULL,
    duplicate_cluster_size = NULL,
    predictor_scores = NULL;

COMMIT;

-- Show what's left
SELECT 'listings' AS tbl, COUNT(*) FROM listings
UNION ALL
SELECT 'listing_images', COUNT(*) FROM listing_images
UNION ALL
SELECT 'listing_features', COUNT(*) FROM listing_features
UNION ALL
SELECT 'saleability_labels', COUNT(*) FROM saleability_labels
UNION ALL
SELECT 'generated_cards', COUNT(*) FROM generated_cards;
SQL

# --- Artifact cleanup ---
echo "Cleaning artifacts..."
rm -rf artifacts/predictor/
rm -rf artifacts/occasion_classifier.pt
rm -rf artifacts/generated_cards/
rm -rf artifacts/lora_train/
rm -rf artifacts/llm_system_eval/

# Clean LoRA weights (keep directory structure)
find generation/image/loras/ -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

echo "=== Clean slate done: $(date) ==="
echo "Scraped data preserved. All computed data wiped."
