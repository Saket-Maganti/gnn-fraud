#!/usr/bin/env bash
# colab_setup.sh
# Run this ONCE at the start of a Google Colab session (Runtime: T4/A100 GPU).
# Then run: bash run_upgrades.sh --cuda
#
# Usage in Colab cell:
#   !git clone <your-repo-url> gnn-fraud && cd gnn-fraud && bash colab_setup.sh

set -euo pipefail

echo "══════════════════════════════════════════════"
echo "  GNN Fraud — Colab GPU Setup"
echo "══════════════════════════════════════════════"

# ── 1. Check GPU ──────────────────────────────────
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
python3 -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || true

# ── 2. Install PyG + dependencies ─────────────────
echo ""
echo "Installing dependencies …"

# Detect torch version for correct PyG wheel
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "2.2.0")
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "12.1")
CUDA_TAG="cu$(echo $CUDA_VER | tr -d '.')"

echo "  Torch=$TORCH_VER  CUDA=$CUDA_VER  Tag=$CUDA_TAG"

pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html" 2>&1 | tail -3

pip install -q torch-geometric 2>&1 | tail -2

pip install -q \
  scikit-learn>=1.3.0 \
  pandas>=2.0 \
  networkx>=3.0 \
  xgboost>=1.7.0 \
  scipy>=1.10 \
  matplotlib>=3.7 \
  2>&1 | tail -3

# ── 3. Verify imports ─────────────────────────────
echo ""
echo "Verifying imports …"
python3 -c "
import torch, torch_geometric, sklearn, pandas, networkx, xgboost, scipy
print(f'  torch         {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'  torch_geometric {torch_geometric.__version__}')
print(f'  sklearn       {sklearn.__version__}')
print(f'  xgboost       {xgboost.__version__}')
print(f'  networkx      {networkx.__version__}')
print()
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'  GPU: {props.name}  ({props.total_memory/1e9:.1f} GB)')
    print(f'  CUDA ENABLED — expect 6-8x speedup vs CPU')
else:
    print('  WARNING: CUDA not available — check runtime type (Runtime > Change runtime type > T4 GPU)')
"

# ── 4. Enable CUDA optimizations ─────────────────
python3 -c "
import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print('CUDA optimizations enabled (TF32 + cudnn benchmark)')
"

echo ""
echo "Setup complete. To run all experiments:"
echo "  bash run_upgrades.sh --cuda"
echo ""
echo "For individual parts:"
echo "  bash run_upgrades.sh --cuda --part=8    # distribution shift (fast, no training)"
echo "  bash run_upgrades.sh --cuda --part=9    # t-SNE embeddings"
echo "  bash run_upgrades.sh --cuda --part=2    # graph ablation"
echo "  bash run_upgrades.sh --cuda --fast      # all parts, reduced epochs"
echo ""
echo "Estimated runtimes on T4 GPU:"
echo "  Full run (all 10 parts):  ~50-70 min"
echo "  Fast mode (--fast):       ~25-35 min"
echo "  Parts 8+10 only (no train needed): ~5 min"
