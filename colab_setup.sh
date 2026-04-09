#!/usr/bin/env bash
# colab_setup.sh
# One-shot Google Colab environment setup.
#
# Paste this entire block into a Colab cell and run it ONCE per session:
#
#   !git clone https://github.com/Saket-Maganti/gnn-fraud.git
#   %cd gnn-fraud
#   !bash colab_setup.sh
#
# After setup completes:
#   !bash run_upgrades.sh --cuda          # full run (~55 min on T4)
#   !bash run_upgrades.sh --cuda --fast   # reduced epochs (~30 min on T4)

set -euo pipefail

echo "══════════════════════════════════════════════════"
echo "  GNN Fraud — Colab Setup"
echo "══════════════════════════════════════════════════"

# ── Step 1: GPU check ─────────────────────────────────────────────────────────
echo ""
echo "Step 1: GPU"
if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; then
  echo "  CUDA available."
else
  echo "  WARNING: No GPU detected."
  echo "  Go to Runtime > Change runtime type > T4 GPU, then reconnect."
  echo "  Continuing with CPU (much slower)."
fi
python3 -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')" 2>/dev/null || true

# ── Step 2: Install PyG ───────────────────────────────────────────────────────
echo ""
echo "Step 2: Install torch-geometric and dependencies"

TORCH_VER=$(python3 -c "import torch; v=torch.__version__; print(v.split('+')[0])" 2>/dev/null || echo "2.2.0")
CUDA_VER=$(python3  -c "import torch; print(torch.version.cuda or 'cpu')"           2>/dev/null || echo "12.1")
CUDA_TAG="cu$(echo "$CUDA_VER" | tr -d '.')"
echo "  Torch=$TORCH_VER  CUDA=$CUDA_VER  Tag=$CUDA_TAG"

pip install -q \
  torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html" \
  2>&1 | grep -E "^(Successfully|ERROR|WARNING)" || true

pip install -q torch-geometric 2>&1 | grep -E "^(Successfully|ERROR)" || true

pip install -q \
  scikit-learn \
  pandas \
  networkx \
  xgboost \
  scipy \
  matplotlib \
  2>&1 | grep -E "^(Successfully|ERROR)" || true

echo "  PyG + dependencies installed."

# ── Step 3: Verify imports ────────────────────────────────────────────────────
echo ""
echo "Step 3: Verify imports"
python3 -c "
import torch, torch_geometric, sklearn, pandas, networkx, scipy, xgboost
print(f'  torch          {torch.__version__}   cuda={torch.cuda.is_available()}')
print(f'  torch-geometric {torch_geometric.__version__}')
print(f'  sklearn        {sklearn.__version__}')
print(f'  xgboost        {xgboost.__version__}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'  GPU: {p.name}  ({p.total_memory/1e9:.1f} GB VRAM)')
    import torch.backends.cudnn as cudnn
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print('  TF32 + cudnn.benchmark enabled.')
"

# ── Step 4: Dataset download ──────────────────────────────────────────────────
echo ""
echo "Step 4: Elliptic dataset"

FEATURES="data/raw/elliptic_txs_features.csv"
mkdir -p data/raw

if [[ -f "$FEATURES" ]]; then
  echo "  Already present: $(du -sh "$FEATURES" | cut -f1)"
else
  echo "  Downloading from Kaggle …"
  echo ""
  echo "  You need a Kaggle API token. Steps:"
  echo "    1. Go to https://www.kaggle.com/settings"
  echo "    2. API section > Create New Token  → downloads kaggle.json"
  echo "    3. Upload kaggle.json using the cell below, then re-run this step"
  echo ""

  # Check if token is already in place
  if [[ -f "/root/.kaggle/kaggle.json" || -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo "  kaggle.json found — downloading dataset …"
    pip install -q kaggle
    kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw/ --unzip \
      2>&1 | grep -v "^Downloading" || true

    # Kaggle may unzip into a subdirectory
    if [[ -d "data/raw/elliptic_bitcoin_dataset" ]]; then
      mv data/raw/elliptic_bitcoin_dataset/*.csv data/raw/ 2>/dev/null || true
      rmdir data/raw/elliptic_bitcoin_dataset 2>/dev/null || true
    fi
    echo "  Download complete: $(du -sh "$FEATURES" | cut -f1)"
  else
    echo ""
    echo "  ── To upload your kaggle.json token, run this in a NEW cell ──"
    echo ""
    echo "     from google.colab import files"
    echo "     import os, json"
    echo "     uploaded = files.upload()   # select your kaggle.json"
    echo "     os.makedirs('/root/.kaggle', exist_ok=True)"
    echo "     with open('/root/.kaggle/kaggle.json', 'wb') as f:"
    echo "         f.write(list(uploaded.values())[0])"
    echo "     os.chmod('/root/.kaggle/kaggle.json', 0o600)"
    echo "     !pip install -q kaggle"
    echo "     !kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw/ --unzip"
    echo "     !mv data/raw/elliptic_bitcoin_dataset/*.csv data/raw/ 2>/dev/null; echo done"
    echo ""
    echo "  Then re-run: !bash run_upgrades.sh --cuda"
    echo ""
    echo "  Alternatively, if the files are on your Google Drive:"
    echo "     from google.colab import drive"
    echo "     drive.mount('/content/drive')"
    echo "     !cp /content/drive/MyDrive/elliptic_bitcoin_dataset/*.csv data/raw/"
    echo ""
    echo "  Setup complete (dataset step skipped — provide data to run experiments)."
    exit 0
  fi
fi

echo ""
echo "══════════════════════════════════════════════════"
echo "  Setup complete!"
echo "══════════════════════════════════════════════════"
echo ""
echo "Run experiments:"
echo "  !bash run_upgrades.sh --cuda           # full run  (~55 min on T4)"
echo "  !bash run_upgrades.sh --cuda --fast    # fast mode (~30 min on T4)"
echo "  !bash run_upgrades.sh --cuda --part=9  # t-SNE embeddings only"
echo "  !bash run_upgrades.sh --cuda --part=8  # distribution shift only"
echo ""
echo "Estimated runtimes on T4 GPU:"
printf "  %-35s %s\n" "Pre-req (temporal_analysis.py)" "~12 min"
printf "  %-35s %s\n" "Part 2 (graph ablation, 3×3 runs)" "~18 min"
printf "  %-35s %s\n" "Part 3 (significance, 10 runs)" "~15 min"
printf "  %-35s %s\n" "Part 4 (3 SAGE variants × 3 seeds)" "~12 min"
printf "  %-35s %s\n" "Part 5 (calibration, 4 models)" "~8 min"
printf "  %-35s %s\n" "Part 6 (business cost, no training)" "~5 sec"
printf "  %-35s %s\n" "Part 7 (hybrid ensemble, 2 models)" "~5 min"
printf "  %-35s %s\n" "Part 8 (distribution shift, no train)" "~1 min"
printf "  %-35s %s\n" "Part 9 (embedding t-SNE)" "~5 min"
printf "  %-35s %s\n" "Part 10 (PR curves, 3 models)" "~8 min"
echo "  ─────────────────────────────────────────────────"
printf "  %-35s %s\n" "TOTAL" "~85 min  (~45 min with --fast)"
echo ""
