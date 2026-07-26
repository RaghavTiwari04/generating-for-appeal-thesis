#!/bin/bash
#SBATCH --job-name=gc-cmp-labels
#SBATCH --partition=a16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%j-cmp-labels.out
#SBATCH --error=logs/slurm-%j-cmp-labels.err

# Compare judge models that have scored the same cards under different label
# sources. Read-only, no API calls.
#
# Needs Postgres, which only runs on a compute node, so this cannot be run from
# the login node.
#
#   SOURCES=llm_ssr_rubric_v2,llm_ssr_rubric_v2_qwen sbatch cluster/jobs/18_compare_labels.sh
#
# The first source is the reference every other is compared against.

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

SOURCES="${SOURCES:-}"
if [ -z "$SOURCES" ]; then
    echo "SOURCES is required, e.g. SOURCES=a,b sbatch $0" >&2
    exit 1
fi
TOP="${TOP:-5}"
CHARS="${CHARS:-300}"

echo "=== Label source comparison ==="
echo "Node: $(hostname)  Start: $(date)"
echo "sources=$SOURCES"

python -u -m scripts.compare_label_sources \
    --sources "$SOURCES" --top "$TOP" --chars "$CHARS"

echo "=== Done: $(date) ==="
