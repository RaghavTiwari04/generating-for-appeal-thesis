#!/usr/bin/env bash
# One-time setup on Imperial DoC GPU cluster.
#
# IMPORTANT: Run this on a SHELL SERVER or LAB PC, NOT on gpucluster2/3.
#   ssh SHORTCODE@shell3.doc.ic.ac.uk
#   cd /vol/bitbucket/$USER/masters_thesis
#   bash cluster/setup_env.sh
#
# Or use an interactive session:
#   salloc --gres=gpu:1 -p t4
#   bash cluster/setup_env.sh

set -euo pipefail

WORK="/vol/bitbucket/$USER"
export HF_HOME="$WORK/.cache/huggingface"
mkdir -p "$HF_HOME"
echo "=== Setting up in $WORK/masters_thesis ==="

# CUDA
if [ -f /vol/cuda/12.0.0/setup.sh ]; then
    source /vol/cuda/12.0.0/setup.sh
    echo "CUDA 12.0 loaded"
fi

# Use virtualenv (cluster standard) — NOT conda
VENV="$WORK/venvs/gc"
if [ -d "$VENV" ]; then
    echo "Virtualenv $VENV already exists. Activating..."
    source "$VENV/bin/activate"
else
    echo "Creating virtualenv at $VENV ..."
    python3 -m virtualenv "$VENV"
    source "$VENV/bin/activate"

    # Install PyTorch with CUDA 11.8 (compatible with cluster's CUDA 12.0)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # Install project
    pip install -e ".[dev]"
fi

# Create directories on bitbucket (NOT home — home is only 12GB)
mkdir -p "$WORK/masters_thesis/logs"
mkdir -p "$WORK/masters_thesis/artifacts/predictor"
mkdir -p "$WORK/masters_thesis/artifacts/lora"
mkdir -p "$WORK/pgdata"
mkdir -p "$WORK/minio-data"

# Download fonts
python -m generation.layout.download_fonts 2>/dev/null || echo "Font download skipped"

# Pre-download HuggingFace models (compute nodes may have no internet!)
echo ""
echo "=== Pre-downloading model weights ==="
echo "This may take a while on first run..."
python -c "
from transformers import AutoModel, AutoTokenizer
print('Downloading SigLIP...')
AutoModel.from_pretrained('google/siglip-base-patch16-384')
print('SigLIP cached.')
" 2>/dev/null || echo "SigLIP download failed — try manually"

python -c "
from diffusers import StableDiffusionXLInpaintPipeline
import torch
print('Downloading SDXL...')
StableDiffusionXLInpaintPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,
)
print('SDXL cached.')
" 2>/dev/null || echo "SDXL download failed — try manually"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV/bin/activate"
echo "Copy and edit .env: cp cluster/.env.cluster .env && nano .env"
echo ""
echo "IMPORTANT: If using Flux instead of SDXL, also pre-download:"
echo "  python -c \"from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev')\""
