#!/bin/bash
#SBATCH --job-name=gc-gallery
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm-%j-gallery.out
#SBATCH --error=logs/slurm-%j-gallery.err

# Everything read-only that the writeup still needs, in one job so the services
# start once. No GPU, no API calls, nothing written to the database.
#
#   sbatch cluster/jobs/22_eval_gallery.sh
#
# Do not run these with `sbatch --wrap`: that executes under /bin/sh, where
# `source` does not exist and `set -o pipefail` is rejected, so the services
# script dies on its first line.
#
# Produces:
#   eval_gallery/            the current run's cards, with scores, as HTML
#   headline repetition      distinct headlines per condition, and the
#                            unselected-candidate control that separates
#                            selection from generation
#   corpus reconciliation    every corpus number the text quotes
#   split leakage            how many duplicate pairs straddle the split

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

echo "=== Read-only reporting sweep ==="
echo "Node: $(hostname)  Start: $(date)"

echo
echo "########## 1. Card gallery for the current run ##########"
python -u -m eval.export_gallery --out ./eval_gallery

echo
echo "########## 2. Headline repetition by condition ##########"
python -u -m scripts.headline_diversity

echo
echo "########## 3. Corpus numbers ##########"
python -u -m scripts.verify_corpus_numbers

echo
echo "########## 4. Duplicate leakage across the split ##########"
python -u -m scripts.check_split_leakage

echo
echo "Done: $(date)"
echo "Fetch the gallery with:"
echo "  cd $SLURM_SUBMIT_DIR && zip -qr eval_gallery.zip eval_gallery"
echo "  scp gpucluster:$SLURM_SUBMIT_DIR/eval_gallery.zip ~/Desktop/"
