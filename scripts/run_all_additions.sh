#!/bin/bash
# Run all 5 additions in order.
# Total runtime: ~75 min
# Run from project root: bash scripts/run_all_additions.sh

set -e
echo "======================================================"
echo "  Running all 5 additions"
echo "======================================================"

# Step 0: Save checkpoints for all 4 main configs (~15 min)
echo ""
echo "Step 0: Saving checkpoints (~15 min)..."
python scripts/save_checkpoints.py

# Addition 1: MLP baseline (~30 min, 5 configs × 3 seeds)
echo ""
echo "Addition 1: MLP baseline comparison (~30 min)..."
python scripts/addition1_mlp_baseline.py

# Addition 2: Temporal analysis (~5 min, loads checkpoints)
echo ""
echo "Addition 2: Temporal drift analysis (~5 min)..."
python scripts/addition2_temporal_analysis.py

# Addition 3: PR curves (~3 min)
echo ""
echo "Addition 3: Precision-recall curves (~3 min)..."
python scripts/addition3_pr_curves.py

# Addition 4: Confusion + business cost (~3 min)
echo ""
echo "Addition 4: Confusion matrix + business cost (~3 min)..."
python scripts/addition4_confusion_analysis.py

# Addition 5: Ablation (~20 min, trains MLP 3 seeds)
echo ""
echo "Addition 5: Ablation study (~20 min)..."
python scripts/addition5_ablation.py

echo ""
echo "======================================================"
echo "  All additions complete!"
echo "  Results in: results/additions/"
echo "======================================================"
ls -lh results/additions/
