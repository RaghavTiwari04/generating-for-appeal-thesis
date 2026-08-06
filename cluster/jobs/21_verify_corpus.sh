#!/bin/bash
#SBATCH --job-name=gc-verify-corpus
#SBATCH --partition=a16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%j-verify-corpus.out
#SBATCH --error=logs/slurm-%j-verify-corpus.err

# Recompute every corpus number the writeup quotes and reconcile them, then
# print both corpus tables as LaTeX with the measured values.
#
# Read-only: no writes, no API calls, no GPU. Needs Postgres, which only runs
# on a compute node, so this cannot be run from the login node.
#
#   sbatch cluster/jobs/21_verify_corpus.sh
#
# Read the "reconciliation" block and the two LaTeX tables from the log.

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

echo "=== Corpus number verification ==="
echo "Node: $(hostname)  Start: $(date)"

python -u -m scripts.verify_corpus_numbers

echo "Done: $(date)"
