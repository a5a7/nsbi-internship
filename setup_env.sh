#!/usr/bin/env bash
set -e

REPO="/mnt/c/Users/Unnat/nsbi-internship"

# ── 0. Recreate venv with Python 3.12 (3.14 is incompatible with pandas/numpy pins) ──
# If python3.12 isn't found, install it first:
#   sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-dev
PYBIN=$(which python3.12 2>/dev/null || which python3.11 2>/dev/null)
if [ -z "$PYBIN" ]; then
    echo "ERROR: python3.12 or python3.11 not found. Run:"
    echo "  sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-dev"
    exit 1
fi
echo "Using: $($PYBIN --version)"
rm -rf ~/nsbi-venv
$PYBIN -m venv ~/nsbi-venv
source ~/nsbi-venv/bin/activate

# ── 1. PyTorch with CUDA 12.8 ────────────────────────────────────────────
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify GPU is visible:
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# ── 2. Core dependencies ──────────────────────────────────────────────────
pip install \
    pytorch-lightning==2.6.1 \
    pandas==2.1.4 \
    numpy==1.26.4 \
    scikit-learn==1.8.0 \
    scipy==1.17.1 \
    matplotlib \
    pyyaml \
    wandb \
    huggingface-hub \
    tqdm \
    torchmetrics

# ── 3. Install EveNet-Lite (editable from local checkout) ────────────────
cd "$REPO/EveNet-Lite-main"
pip install -e .

# ── 4. Install the NSBI tutorial package (editable) ──────────────────────
cd "$REPO/sessions/day2/nsbi-tutorial"
pip install -e .

# ── 5. Install remaining pinned packages, skipping what's already installed ──
cd "$REPO"
# Filter out the git+https evenet_lite line (we already installed it locally above)
grep -v 'git+https' requirements.txt > /tmp/requirements_filtered.txt
pip install -r /tmp/requirements_filtered.txt \
    --ignore-installed torch torchvision torchaudio evenet_lite evenet-core

echo ""
echo "Done! Verify with:"
echo "  python -c \"import torch, pandas, evenet_lite; print('torch', torch.__version__, 'pandas', pandas.__version__)\""
