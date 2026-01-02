#!/bin/bash

# molecore_foundry Environment Setup Script
# This script sets up the molecore_foundry environment on macOS or Linux

set -e

echo "🚀 Setting up molecore_foundry environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install Python 3.12 if not available
echo "📦 Installing Python 3.12..."
uv python install 3.12

# Create virtual environment
echo "🏗️  Creating virtual environment..."
uv venv molecore_foundry --python 3.12

# Activate environment
echo "🔄 Activating environment..."
source molecore_foundry/bin/activate

# Install packages
echo "📦 Installing packages..."
pip install -r requirements.txt

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import torch, atomworks, biotite, numpy as np, pandas as pd, scipy
import requests, rcsbsearchapi, openai, modal
print('✅ All packages imported successfully!')
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "To use the environment:"
echo "source molecore_foundry/bin/activate"
echo ""
echo "For CUDA support on Linux (optional):"
echo "pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
