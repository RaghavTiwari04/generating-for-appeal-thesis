#!/usr/bin/env bash
# One-time setup on Imperial HPC cluster.
# Usage: bash cluster/setup_env.sh

set -euo pipefail

echo "=== Setting up greeting-cards environment ==="

# Load modules (adjust names to match Imperial's module system)
module load anaconda3/2024 2>/dev/null || module load anaconda3 2>/dev/null || {
    echo "WARNING: Could not load anaconda3 module. Trying conda directly..."
}

# Check for Tesseract
if ! command -v tesseract &>/dev/null; then
    echo "WARNING: Tesseract not found. OCR step will fail."
    echo "Try: module load tesseract  OR  conda install -c conda-forge tesseract"
fi

# Create conda environment
if conda info --envs | grep -q "^gc "; then
    echo "Conda env 'gc' already exists. Updating..."
    conda activate gc
    pip install -e ".[dev]"
else
    echo "Creating conda env 'gc' with Python 3.11..."
    conda create -n gc python=3.11 -y
    conda activate gc

    # Install PyTorch with CUDA (Imperial typically has CUDA 12.x)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # Install project
    pip install -e ".[dev]"

    # Playwright for scraping (optional on cluster)
    # playwright install chromium
fi

# Create directories
mkdir -p logs artifacts/predictor artifacts/lora artifacts/generation

# Download fonts (CPU, quick)
python -m generation.layout.download_fonts 2>/dev/null || echo "Font download skipped (run manually if needed)"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate gc"
echo "Then copy and edit .env: cp cluster/.env.cluster .env && nano .env"
