# GNN Fraud Detection Under Temporal Drift

A cleaned, paper-aligned repository for studying Graph Neural Networks, temporal distribution shift, graph ablation, calibration, and hybrid tabular-graph modeling on the Elliptic Bitcoin Dataset.

## At A Glance

- Under the stricter inductive temporal protocol used in the paper, a plain MLP reaches F1 = 0.744 and beats every tested standalone GNN.
- A Random Forest on raw node features reaches F1 = 0.820, establishing a strong non-graph baseline that exceeds the tested GNN configurations.
- GraphSAGE is the strongest standalone GNN in the paper, but its best reported F1 is 0.697, well below the strongest tabular baselines.
- Temporal distribution shift is the central failure mode: the illicit rate collapses by roughly 39x between the training regime and the hardest test steps.
- The graph-structure ablation is one of the most important results: shuffled edges improve F1 from 0.268 to 0.383, which means the original graph can be actively harmful under this protocol.
- Hybrid modeling changes the story: concatenating GNN embeddings with raw tabular features and fitting a downstream Random Forest recovers F1 = 0.807.
- Calibration alone is not enough to solve the failure mode. Temperature scaling leaves F1 effectively unchanged in the paper’s reported analysis.
- The repo therefore supports a nuanced conclusion: graph-derived representations still matter, but direct message passing on the raw transaction graph is not reliably beneficial under temporal drift.

## Why This Repository Exists

This repository is organized around a single research question: what really happens to graph-based fraud detection when evaluation is made more deployment-relevant, more temporal, and more honest about test-time graph exposure?

The paper argues that prior Elliptic-style evaluations can overstate the value of direct message passing by mixing training and test-time structure in ways a live fraud system would not have. The code here is therefore built to support a stricter story: inductive evaluation, temporal diagnostics, baseline comparisons, and a paper-ready figure pipeline that isolates exactly what the manuscript uses.

This cleaned release has four guiding goals:

- keep experiment code intact
- make the paper self-contained
- make figure dependencies explicit
- make reproduction less fragile for academic readers

## Repository Status

- `paper/` is now the canonical home of the LaTeX manuscript.
- `figures/` contains only the image assets actually referenced by the paper sources.
- `scripts/sync_paper_figures.py` regenerates the paper figure bundle and rewrites figure paths.
- `results/` remains the local working area for generated outputs, but the paper no longer depends on it directly.
- Legacy `scripts/addition*.py` files are retained for completeness and traceability, but the preferred paper-facing entry points live in `experiments/`.

## Cleaned Layout

```text
gnn-fraud/
├── data/                     # dataset loaders, preprocessing, graph construction helpers
├── experiments/              # primary experiment entry points used to reproduce paper results
├── figures/                  # paper-only image assets, synced from experiment outputs
├── logs/                     # optional runtime logs
├── models/                   # GNN, MLP, and hybrid model definitions
├── paper/                    # cleaned LaTeX paper sources and bibliography
├── results/                  # local experiment outputs, checkpoints, plots (ignored)
├── scripts/                  # legacy/additional analysis scripts and paper figure sync pipeline
├── utils/                    # training loops, metrics, imbalance utilities, visualization helpers
├── README.md                 # long-form project guide
├── requirements.txt          # Python dependencies
├── train.py                  # minibatch training entry point
├── train_proper.py           # alternate training flow with validation split
├── config.py                 # shared experiment defaults
└── colab_setup.sh            # Colab-oriented bootstrap helper
```

## Quick Start

If you want the shortest path from a clean clone to a paper build, follow this sequence:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python train.py --model sage --strategy weighted --epochs 150
python experiments/run_insights.py --fast
python scripts/sync_paper_figures.py
cd paper
TEXMFVAR=/tmp/texmfvar pdflatex -interaction=nonstopmode main.tex
TEXMFVAR=/tmp/texmfvar bibtex main
TEXMFVAR=/tmp/texmfvar pdflatex -interaction=nonstopmode main.tex
TEXMFVAR=/tmp/texmfvar pdflatex -interaction=nonstopmode main.tex
```

## Paper-Repo Alignment

The repository is intentionally aligned with the paper in the following way:

- The manuscript lives in `paper/main.tex` and uses `paper/paper_shared_body.tex` as the shared core text.
- The bibliography lives in `paper/references.bib`.
- Every image included by the paper is copied into `figures/` and tracked in `paper/figure_manifest.json`.
- The paper no longer hardcodes references into `results/`, `results2/`, or any other exploratory output tree.
- The manuscript author field is normalized to `Saket Maganti` across the paper variants that remain in the repo.
- External repository links were removed from the LaTeX paper sources so the manuscript is cleaner for academic submission contexts.

## Key Findings From The Paper

### Finding 1

Under the stricter inductive temporal protocol used in the paper, a plain MLP reaches F1 = 0.744 and beats every tested standalone GNN.

Why it matters:

The result changes how this repository should be interpreted. The code is not a generic “GNNs win on fraud” benchmark; it is a controlled argument about where graph structure helps, where it hurts, and how those conclusions depend on protocol design.

### Finding 2

A Random Forest on raw node features reaches F1 = 0.820, establishing a strong non-graph baseline that exceeds the tested GNN configurations.

Why it matters:

The result changes how this repository should be interpreted. The code is not a generic “GNNs win on fraud” benchmark; it is a controlled argument about where graph structure helps, where it hurts, and how those conclusions depend on protocol design.

### Finding 3

GraphSAGE is the strongest standalone GNN in the paper, but its best reported F1 is 0.697, well below the strongest tabular baselines.

Why it matters:

The result changes how this repository should be interpreted. The code is not a generic “GNNs win on fraud” benchmark; it is a controlled argument about where graph structure helps, where it hurts, and how those conclusions depend on protocol design.

### Finding 4

Temporal distribution shift is the central failure mode: the illicit rate collapses by roughly 39x between the training regime and the hardest test steps.

Why it matters:

The result changes how this repository should be interpreted. The code is not a generic “GNNs win on fraud” benchmark; it is a controlled argument about where graph structure helps, where it hurts, and how those conclusions depend on protocol design.

### Finding 5

The graph-structure ablation is one of the most important results: shuffled edges improve F1 from 0.268 to 0.383, which means the original graph can be actively harmful under this protocol.

Why it matters:

The result changes how this repository should be interpreted. The code is not a generic “GNNs win on fraud” benchmark; it is a controlled argument about where graph structure helps, where it hurts, and how those conclusions depend on protocol design.

### Finding 6

Hybrid modeling changes the story: concatenating GNN embeddings with raw tabular features and fitting a downstream Random Forest recovers F1 = 0.807.

Why it matters:

The result changes how this repository should be interpreted. The code is not a generic “GNNs win on fraud” benchmark; it is a controlled argument about where graph structure helps, where it hurts, and how those conclusions depend on protocol design.

### Finding 7

Calibration alone is not enough to solve the failure mode. Temperature scaling leaves F1 effectively unchanged in the paper’s reported analysis.

Why it matters:

The result changes how this repository should be interpreted. The code is not a generic “GNNs win on fraud” benchmark; it is a controlled argument about where graph structure helps, where it hurts, and how those conclusions depend on protocol design.

### Finding 8

The repo therefore supports a nuanced conclusion: graph-derived representations still matter, but direct message passing on the raw transaction graph is not reliably beneficial under temporal drift.

Why it matters:

The result changes how this repository should be interpreted. The code is not a generic “GNNs win on fraud” benchmark; it is a controlled argument about where graph structure helps, where it hurts, and how those conclusions depend on protocol design.

## Dataset Assumptions

The repository expects the Elliptic Bitcoin Dataset, but the raw files are not distributed here. This keeps the public release light and avoids bundling large data artifacts.

Expected raw files:

- `data/raw/elliptic_txs_features.csv`
- `data/raw/elliptic_txs_classes.csv`
- `data/raw/elliptic_txs_edgelist.csv`

Important dataset notes:

- The paper focuses on the standard temporal split with training on steps 1 through 34 and evaluation on steps 35 through 49.
- Unknown labels are excluded from evaluation in the standard way used throughout the codebase.
- The repo’s argument depends on temporal ordering, so avoid random reshuffling that destroys the chronological split.

## Environment Setup

Recommended local workflow:

1. Create a fresh virtual environment.
2. Install the pinned requirements from `requirements.txt`.
3. Verify that PyTorch and PyTorch Geometric import cleanly.
4. Place the dataset under `data/raw/`.
5. Run a small training command before attempting the long pipelines.

Recommended installation commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python - <<'PY'
import torch
import torch_geometric
print(torch.__version__)
print(torch_geometric.__version__)
PY
```

## Recommended Reproduction Path

A practical reproduction path is more valuable than a maximal one. The order below is the best default for a new reader coming from the paper.

1. Run a single quick GNN training command to confirm the environment is healthy.
2. Run the classical baselines to establish a non-graph reference point.
3. Run the temporal analysis because it encodes the paper’s central failure mode.
4. Run the hybrid ensemble to test the “graph representations survive in hybrids” claim.
5. Run the graph ablation and significance analysis if you are validating the graph-harm argument in depth.
6. Sync figures and build the paper only after the earlier steps look plausible.

## Command Cookbook

### 1. Quick sanity-check training

```bash
python train.py --model sage --strategy weighted --epochs 150
python train.py --model gat --strategy baseline --epochs 100
```

### 2. Classical baselines

```bash
python experiments/baselines.py
```

### 3. Multi-seed benchmark

```bash
python experiments/run_seeds.py
```

### 4. Temporal drift analysis

```bash
python experiments/temporal_analysis.py
python experiments/temporal_analysis.py --eval_only
```

### 5. Hybrid ensemble

```bash
python experiments/hybrid_ensemble.py
```

### 6. Master insight pipeline

```bash
python experiments/run_insights.py
python experiments/run_insights.py --fast
```

### 7. Business cost analysis

```bash
python experiments/business_cost.py
```

### 8. Calibration study

```bash
python experiments/calibration_study.py
```

### 9. Distribution shift analysis

```bash
python experiments/distribution_shift.py
```

### 10. Embedding visualization

```bash
python experiments/embedding_viz.py
```

### 11. Significance testing

```bash
python experiments/significance_tests.py
```

### 12. Figure sync and paper build

```bash
python scripts/sync_paper_figures.py
cd paper
TEXMFVAR=/tmp/texmfvar pdflatex -interaction=nonstopmode main.tex
TEXMFVAR=/tmp/texmfvar bibtex main
TEXMFVAR=/tmp/texmfvar pdflatex -interaction=nonstopmode main.tex
TEXMFVAR=/tmp/texmfvar pdflatex -interaction=nonstopmode main.tex
```

## How To Use This Repo From Google Colab

The Colab story matters because many readers will encounter the repo through the paper but will not have a configured local environment.

Suggested Colab flow:

1. Open a fresh Colab notebook.
2. Clone the repository into the notebook session.
3. Install the requirements from `requirements.txt`.
4. Upload or mount the Elliptic dataset into `data/raw/`.
5. Start with the baseline or temporal scripts before attempting the long pipelines.

Example Colab cells:

```python
!git clone /content/gnn-fraud-local-copy gnn-fraud  # replace with your own source as needed
%cd gnn-fraud
!pip install -r requirements.txt
```

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
!mkdir -p data/raw
!cp /content/drive/MyDrive/elliptic_txs_features.csv data/raw/
!cp /content/drive/MyDrive/elliptic_txs_classes.csv data/raw/
!cp /content/drive/MyDrive/elliptic_txs_edgelist.csv data/raw/
```

```python
!python train.py --model sage --strategy weighted --epochs 50
!python experiments/run_insights.py --fast
```

Colab caveats:

- Expect long-running scripts to reset if the notebook disconnects.
- Persist outputs you care about to Drive.
- Prefer `--fast` or smaller epoch counts for exploratory runs.
- Rebuild paper figures only after the relevant outputs exist.

## Figure Pipeline

The paper figure pipeline is one of the most important improvements in this cleaned release.

What it does:

- scans every LaTeX file in `paper/`
- extracts every `\includegraphics{...}` reference
- resolves the referenced image from known source directories
- copies only the used assets into `figures/`
- normalizes the filenames into snake_case
- rewrites the LaTeX figure paths to use `figures/...`
- emits `paper/figure_manifest.json` as an audit trail

Why this matters:

- the manuscript is no longer coupled to bulky experiment-output folders
- the paper can be shared independently from the full exploratory result history
- public readers can quickly inspect the exact paper figures without searching through nested directories

Current figure inventory:

- `figures/calibration_curves.png`: Calibration curves sourced from experiment outputs.
- `figures/distribution_shift.png`: Distribution shift figure used to support the drift interpretation.
- `figures/embedding_tsne_label.png`: Embedding visualization colored by label.
- `figures/embedding_tsne_time.png`: Embedding visualization colored by time.
- `figures/fig10_confusion_matrices.png`: Confusion matrix comparison figure.
- `figures/fig11_heatmap.png`: Heatmap summarizing broad experiment outcomes.
- `figures/fig12_training_curves.png`: Training-curve summary figure.
- `figures/fig13_temperature_scaling.png`: Temperature scaling or calibration figure.
- `figures/fig14_strategy_deep_dive.png`: Strategy-level deep dive figure.
- `figures/fig15_variance_analysis.png`: Variance analysis figure across seeds or settings.
- `figures/fig16_ablation_advantage.png`: Ablation advantage figure used in the discussion of graph harm.
- `figures/fig17_dataset_analysis.png`: Dataset characterization figure used early in the paper.
- `figures/fig1_main_results.png`: Headline benchmark figure comparing major model families and outcomes.
- `figures/fig2_temporal_drift.png`: Temporal drift plot showing performance collapse across test timesteps.
- `figures/fig3_perclass_temporal.png`: Per-class temporal behavior for the test horizon.
- `figures/fig4_feature_importance.png`: Feature importance analysis supporting the role of raw node features.
- `figures/fig5_community_structure.png`: Graph/community structure visualization.
- `figures/fig6_pr_curves.png`: Precision-recall curves for selected models and strategies.
- `figures/fig7_business_cost.png`: Business-cost summary under asymmetric penalties.
- `figures/fig8_prior_work.png`: Comparison against prior reported work.
- `figures/fig9_evolvegcn.png`: EvolveGCN-related comparison figure.
- `figures/fig_graph_construction.png`: Graph construction comparison figure.
- `figures/fig_model_comparison.png`: Model comparison figure used in the shared manuscript body.
- `figures/graph_ablation_extended.png`: Extended graph ablation figure.
- `figures/hybrid_ensemble.png`: Hybrid ensemble figure showing recovery from pure GNN degradation.

## Directory Walkthrough

### data/

Dataset access and preprocessing.

What belongs here:

This directory exists because it groups one layer of the reproducibility story into a predictable place. The README, the paper, and the experiment scripts all refer back to this structure.

What not to put here:

Avoid mixing generated outputs with source files unless the directory is explicitly an output area such as `results/` or `logs/`.

### experiments/

Preferred experiment entry points.

What belongs here:

This directory exists because it groups one layer of the reproducibility story into a predictable place. The README, the paper, and the experiment scripts all refer back to this structure.

What not to put here:

Avoid mixing generated outputs with source files unless the directory is explicitly an output area such as `results/` or `logs/`.

### figures/

Paper-only figures.

What belongs here:

This directory exists because it groups one layer of the reproducibility story into a predictable place. The README, the paper, and the experiment scripts all refer back to this structure.

What not to put here:

Avoid mixing generated outputs with source files unless the directory is explicitly an output area such as `results/` or `logs/`.

### logs/

Optional runtime logs.

What belongs here:

This directory exists because it groups one layer of the reproducibility story into a predictable place. The README, the paper, and the experiment scripts all refer back to this structure.

What not to put here:

Avoid mixing generated outputs with source files unless the directory is explicitly an output area such as `results/` or `logs/`.

### models/

Architectures and embedding utilities.

What belongs here:

This directory exists because it groups one layer of the reproducibility story into a predictable place. The README, the paper, and the experiment scripts all refer back to this structure.

What not to put here:

Avoid mixing generated outputs with source files unless the directory is explicitly an output area such as `results/` or `logs/`.

### paper/

Cleaned LaTeX sources.

What belongs here:

This directory exists because it groups one layer of the reproducibility story into a predictable place. The README, the paper, and the experiment scripts all refer back to this structure.

What not to put here:

Avoid mixing generated outputs with source files unless the directory is explicitly an output area such as `results/` or `logs/`.

### results/

Local outputs and checkpoints.

What belongs here:

This directory exists because it groups one layer of the reproducibility story into a predictable place. The README, the paper, and the experiment scripts all refer back to this structure.

What not to put here:

Avoid mixing generated outputs with source files unless the directory is explicitly an output area such as `results/` or `logs/`.

### scripts/

Legacy helper scripts and figure sync pipeline.

What belongs here:

This directory exists because it groups one layer of the reproducibility story into a predictable place. The README, the paper, and the experiment scripts all refer back to this structure.

What not to put here:

Avoid mixing generated outputs with source files unless the directory is explicitly an output area such as `results/` or `logs/`.

### utils/

Shared training, metric, and visualization helpers.

What belongs here:

This directory exists because it groups one layer of the reproducibility story into a predictable place. The README, the paper, and the experiment scripts all refer back to this structure.

What not to put here:

Avoid mixing generated outputs with source files unless the directory is explicitly an output area such as `results/` or `logs/`.

## Experiment Guide By Theme

### Core Training

- `train.py`
  Purpose: Main minibatch training entry point for single-run GNN experiments. Good for sanity checks and quick comparisons.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `train_proper.py`
  Purpose: Alternate training path with a validation-aware loop. Useful when you want a simpler or older-style training script.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

### Paper Reproduction

- `experiments/run_insights.py`
  Purpose: Top-level insight pipeline that produces aggregate metrics, ablations, and graph statistics.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/run_seeds.py`
  Purpose: Multi-seed benchmark used to quantify variance across models and imbalance strategies.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/temporal_analysis.py`
  Purpose: Per-timestep evaluation pipeline that exposes the temporal collapse central to the paper.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/significance_tests.py`
  Purpose: Statistical comparison utility for seed-level significance testing.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/baselines.py`
  Purpose: Classical tabular baselines such as logistic regression, random forest, and XGBoost.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

### Additional Analyses

- `experiments/hybrid_ensemble.py`
  Purpose: Builds and evaluates the hybrid ensemble that combines GNN embeddings with raw features.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/graph_analysis.py`
  Purpose: Computes graph topology summaries for graph construction variants.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/graph_ablation_extended.py`
  Purpose: Extended graph ablation analysis for original, shuffled, and altered graph structures.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/calibration_study.py`
  Purpose: Calibration-focused experiments and summary outputs.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/business_cost.py`
  Purpose: Evaluates model behavior under asymmetric false-negative and false-positive costs.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/pr_analysis.py`
  Purpose: Precision-recall analysis and threshold sweep outputs.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/distribution_shift.py`
  Purpose: Distribution shift diagnostics that support the temporal drift narrative.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/embedding_viz.py`
  Purpose: Embedding-space visualizations such as t-SNE and PCA.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/temporal_deep.py`
  Purpose: Deeper temporal breakdowns beyond the headline drift plot.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `experiments/run_l4.py`
  Purpose: Higher-effort “Level 4” experiment wrapper for extended model families and adaptive strategies.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

### Legacy Supplement Scripts

- `scripts/addition1_mlp_baseline.py`
  Purpose: Legacy one-off script for MLP baseline comparisons.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/addition2_temporal_analysis.py`
  Purpose: Legacy temporal analysis script that predates the cleaner experiment entry points.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/addition3_pr_curves.py`
  Purpose: Legacy PR-curve generation helper.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/addition4_confusion_analysis.py`
  Purpose: Legacy confusion matrix and business-cost analysis helper.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/addition5_ablation.py`
  Purpose: Legacy ablation pipeline with checkpoint reuse.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/addition6_seed_table.py`
  Purpose: Legacy seed-summary table builder.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/addition7_temporal_calibration.py`
  Purpose: Legacy temporal calibration script.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/addition8_temperature_scaling.py`
  Purpose: Legacy temperature-scaling experiment script.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/addition9_subgraph_viz.py`
  Purpose: Legacy community and subgraph visualization helper.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/addition10_feature_importance.py`
  Purpose: Legacy feature-importance analysis helper.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/addition11_evolvegcn_run.py`
  Purpose: Legacy EvolveGCN comparison script.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/addition12_prior_work_comparison.py`
  Purpose: Legacy prior-work comparison visualization script.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/addition13_perclass_temporal.py`
  Purpose: Legacy per-class temporal analysis helper.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/save_checkpoints.py`
  Purpose: Saves named checkpoints for downstream legacy scripts.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/tune_threshold.py`
  Purpose: Threshold tuning helper.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

- `scripts/sync_paper_figures.py`
  Purpose: Current figure pipeline that scans LaTeX, copies used assets, rewrites paths, and writes a manifest.
  Recommended use: Run this when you are answering the specific research question that the file encodes, rather than treating every script as part of the default reproduction path.

## File-By-File Guide

This section is intentionally granular. It is here so a reader can map the paper to actual code quickly.

### `config.py`

Role: Shared experiment defaults

Description: Centralizes reusable configuration values so scripts agree on hidden sizes, seeds, and other defaults.

Why it matters: Read this early if you want to understand the repo-wide assumptions before changing hyperparameters.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `colab_setup.sh`

Role: Colab bootstrap helper

Description: Bootstraps a Google Colab environment for this repo.

Why it matters: Useful when reproducing paper figures without a local workstation setup.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `train.py`

Role: Primary training entry point

Description: Runs minibatch GNN training with the selectable model and imbalance strategy.

Why it matters: Best first script for a new user who wants a quick sanity check.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `train_proper.py`

Role: Alternate training flow

Description: Older or alternate supervised training path with validation handling.

Why it matters: Useful if you want a simpler baseline training loop or to compare behaviors.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `data/__init__.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `data/README.md`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `data/dataset.py`

Role: Dataset loading and preprocessing

Description: Loads the Elliptic dataset, preprocesses features, and defines the temporal split used throughout the repo.

Why it matters: Critical for understanding how inductive evaluation is enforced.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `data/graph_builder.py`

Role: Graph construction helpers

Description: Builds graph variants such as original, similarity, feature-kNN, temporal, and augmented constructions.

Why it matters: Important when reproducing graph-construction and graph-ablation results.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/__init__.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/baselines.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/business_cost.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/calibration_study.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/distribution_shift.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/embedding_viz.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/graph_ablation_extended.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/graph_analysis.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/hybrid_ensemble.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/pr_analysis.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/run_insights.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/run_l4.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/run_seeds.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/significance_tests.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/temporal_analysis.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `experiments/temporal_deep.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `models/__init__.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `models/gnn.py`

Role: Core GNN model definitions

Description: Defines GCN, GraphSAGE, GAT, and related model-building utilities.

Why it matters: Start here when changing architecture details.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `models/hybrid.py`

Role: Hybrid model utilities

Description: Handles embedding extraction and downstream hybrid training.

Why it matters: This file is central to the repo’s “graph representations survive in hybrids” claim.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `models/mlp_baseline.py`

Role: Feature-only baseline model

Description: Defines the MLP control used to test whether graph structure is helping at all.

Why it matters: Essential for the central paper comparison.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `paper/main.tex`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `paper/paper_arxiv.tex`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `paper/paper_elsevier.tex`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `paper/paper_ieee.tex`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `paper/paper_shared_body.tex`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `paper/paper_springer.tex`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `paper/paper_acl.tex`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `paper/references.bib`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `paper/figure_manifest.json`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition1_mlp_baseline.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition2_temporal_analysis.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition3_pr_curves.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition4_confusion_analysis.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition5_ablation.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition6_seed_table.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition7_temporal_calibration.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition8_temperature_scaling.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition9_subgraph_viz.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition10_feature_importance.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition11_evolvegcn_run.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition12_prior_work_comparison.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/addition13_perclass_temporal.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/run_all_additions.sh`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/save_checkpoints.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/sync_paper_figures.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `scripts/tune_threshold.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `utils/__init__.py`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `utils/imbalance.py`

Role: Imbalance handling

Description: Contains baseline, weighted, focal, oversampling, graph augmentation, and adaptive imbalance strategies.

Why it matters: Read this when auditing what each strategy really changes.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `utils/metrics.py`

Role: Metrics and summaries

Description: Computes F1, precision, recall, AUC, and aggregate statistics used throughout the repo.

Why it matters: Helpful for verifying that paper tables and local reruns are aligned.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `utils/temporal.py`

Role: Temporal evaluation helpers

Description: Builds temporal views and per-step summaries for drift analysis.

Why it matters: Useful when extending or debugging the timestep-level evaluation.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `utils/trainer.py`

Role: Full-graph training utilities

Description: Contains training loops and checkpoint logic for standard and EvolveGCN-style workflows.

Why it matters: Relevant when comparing training behaviors across scripts.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `utils/trainer_minibatch.py`

Role: Minibatch training utilities

Description: Implements the RandomNodeLoader-compatible training path used by the main entry script.

Why it matters: Important for CPU-friendly experimentation.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `utils/visualise.py`

Role: Visualization helpers

Description: Stores reusable plotting helpers used across experiments.

Why it matters: Handy when you want to change plot styling or output conventions.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `requirements.txt`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

### `README.md`

Role: Project file

Description: Supports some part of the training, analysis, paper, or release workflow.

Why it matters: Open the file directly when the path name matches the part of the pipeline you want to inspect.

Notes:

- Keep source files under version control.
- Prefer changing the preferred experiment entry points before editing legacy supplement scripts.
- If a paper claim depends on this file, mention the exact command used when you publish derivative results.

## Reproducibility Checklist

- [ ] Confirm that raw data files exist and are readable.
- [ ] Record the exact commit hash used for the run.
- [ ] Record Python, PyTorch, and PyG versions.
- [ ] State whether the run was CPU-only, GPU-enabled, or Colab-based.
- [ ] State whether the temporal split was altered.
- [ ] Record seed values for all multi-seed experiments.
- [ ] Record any modified epoch counts or early-stopping patience values.
- [ ] Keep output metrics with timestamps in `results/` or a tracked experiment log.
- [ ] Run the figure sync script after regenerating paper figures.
- [ ] Rebuild the paper from `paper/main.tex` only after figures are synced.

## FAQ

### What is the main claim of the repository?

The main claim is not that graphs are useless. The paper argues that direct GNN message passing on the raw Elliptic transaction graph degrades under an inductive temporal protocol, while graph-derived embeddings still add value in a hybrid model.

### Why is the README so long?

Because the target audience is someone who may discover the code through the paper, the documentation is intentionally exhaustive. It is meant to reduce ambiguity during reproduction rather than optimize for brevity.

### Why keep both experiments/ and scripts/?

The repo now treats experiments/ as the preferred interface and scripts/ as legacy or supplement-oriented helpers. The legacy scripts are retained because they still encode useful analysis paths and paper-adjacent plots.

### Why does the repo keep results/ ignored?

Experiment outputs, checkpoints, and generated plots can become large and are usually derivable. The paper-facing assets now live in figures/ so the manuscript no longer depends on bulky results folders.

### Why is there a separate figures/ directory?

The figures/ directory is a release-oriented paper asset bundle. It isolates the exact images used by the manuscript from exploratory outputs and duplicate experiment artifacts.

### What does the figure sync script do?

It scans LaTeX sources, extracts every includegraphics reference, resolves the underlying image, copies only the used assets into figures/, rewrites LaTeX paths, and emits a manifest for traceability.

### Do I need the raw dataset to build the paper?

Not if the figures are already synced and the bibliography is present. You need the raw dataset only if you want to regenerate metrics, checkpoints, or plots from scratch.

### What is the safest first reproduction step?

Start with a small sanity-check training run, then run the faster version of the insight pipeline, then build the paper. This gives you a staged path from code health to paper alignment.

### Why does the paper emphasize per-timestep analysis?

Because a single aggregate test metric can hide catastrophic late-period failure. The temporal drift plot is one of the strongest reasons the paper reaches a different conclusion from prior transductive studies.

### Is the graph always harmful?

No. The stronger claim is conditional: the raw transaction graph can be harmful under the paper’s inductive temporal evaluation, yet graph-derived embeddings still contribute in hybrid settings.

### Why compare against MLP and Random Forest so heavily?

Because the question is not whether GNNs can fit the data, but whether they actually beat simpler alternatives under a deployment-relevant evaluation regime.

### What should I cite if I use this repository?

Cite the accompanying paper and mention the repository release that matches the experiment code you used. The citation block at the end of this README provides a starting BibTeX entry.

## Troubleshooting

### The dataset loader cannot find files.

Confirm that the three Elliptic CSV files are under data/raw/ and that their names match the expected filenames exactly.

### PyTorch Geometric installation fails.

Install the pinned PyTorch version first, then install torch-geometric, then install the remaining requirements. On unusual platforms you may need to follow PyG wheel guidance manually.

### Training is too slow on CPU.

Lower epochs, reduce seed count, start with --fast options when available, and use train.py for quick sanity checks before running full paper pipelines.

### The paper build fails on font generation.

Set TEXMFVAR to a writable temporary directory such as /tmp/texmfvar before running pdflatex or bibtex.

### The paper build cannot find figures.

Run python scripts/sync_paper_figures.py from the repository root. Then build paper/main.tex from the paper/ directory.

### Why am I seeing repeated figure or table destination warnings?

The current manuscript contains some duplicate floating-object identifiers. They do not block PDF generation, but they are worth cleaning if you want a warning-free final manuscript.

### Why am I seeing “Label(s) may have changed” warnings?

Run pdflatex at least twice after bibtex. This is standard for LaTeX documents with many cross-references.

### Why do some legacy scripts expect checkpoints?

Several supplement scripts were designed to reuse earlier trained models instead of retraining every time. Use scripts/save_checkpoints.py or the newer experiment entry points before running them.

### Why is there both a hybrid pipeline and a baseline pipeline?

Because they answer different research questions. The baseline pipeline asks whether graphs help at all; the hybrid pipeline asks whether graph-derived embeddings help once tabular features remain available.

### What should I delete before publishing a fresh clone?

In most cases nothing beyond local results/, logs, and environment folders. The repo now ignores those paths and isolates paper-ready figures in figures/.

## Interpretation Guide

If you are coming from the paper, it helps to treat the repository as answering three separate but related questions:

1. Do standalone GNNs beat strong feature-only baselines under an inductive temporal split?
2. If not, is the problem calibration, temporal drift, or graph construction?
3. Even if standalone GNNs underperform, do graph-derived embeddings still carry complementary signal?

The codebase is organized around these questions more than around any single model family.

## Visual Reference Guide

Readers often want to know which figures to inspect first. The sequence below mirrors the narrative arc of the paper.

1. `figures/calibration_curves.png`
   Interpretation: Calibration curves sourced from experiment outputs.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

2. `figures/distribution_shift.png`
   Interpretation: Distribution shift figure used to support the drift interpretation.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

3. `figures/embedding_tsne_label.png`
   Interpretation: Embedding visualization colored by label.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

4. `figures/embedding_tsne_time.png`
   Interpretation: Embedding visualization colored by time.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

5. `figures/fig10_confusion_matrices.png`
   Interpretation: Confusion matrix comparison figure.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

6. `figures/fig11_heatmap.png`
   Interpretation: Heatmap summarizing broad experiment outcomes.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

7. `figures/fig12_training_curves.png`
   Interpretation: Training-curve summary figure.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

8. `figures/fig13_temperature_scaling.png`
   Interpretation: Temperature scaling or calibration figure.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

9. `figures/fig14_strategy_deep_dive.png`
   Interpretation: Strategy-level deep dive figure.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

10. `figures/fig15_variance_analysis.png`
   Interpretation: Variance analysis figure across seeds or settings.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

11. `figures/fig16_ablation_advantage.png`
   Interpretation: Ablation advantage figure used in the discussion of graph harm.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

12. `figures/fig17_dataset_analysis.png`
   Interpretation: Dataset characterization figure used early in the paper.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

13. `figures/fig1_main_results.png`
   Interpretation: Headline benchmark figure comparing major model families and outcomes.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

14. `figures/fig2_temporal_drift.png`
   Interpretation: Temporal drift plot showing performance collapse across test timesteps.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

15. `figures/fig3_perclass_temporal.png`
   Interpretation: Per-class temporal behavior for the test horizon.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

16. `figures/fig4_feature_importance.png`
   Interpretation: Feature importance analysis supporting the role of raw node features.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

17. `figures/fig5_community_structure.png`
   Interpretation: Graph/community structure visualization.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

18. `figures/fig6_pr_curves.png`
   Interpretation: Precision-recall curves for selected models and strategies.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

19. `figures/fig7_business_cost.png`
   Interpretation: Business-cost summary under asymmetric penalties.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

20. `figures/fig8_prior_work.png`
   Interpretation: Comparison against prior reported work.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

21. `figures/fig9_evolvegcn.png`
   Interpretation: EvolveGCN-related comparison figure.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

22. `figures/fig_graph_construction.png`
   Interpretation: Graph construction comparison figure.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

23. `figures/fig_model_comparison.png`
   Interpretation: Model comparison figure used in the shared manuscript body.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

24. `figures/graph_ablation_extended.png`
   Interpretation: Extended graph ablation figure.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

25. `figures/hybrid_ensemble.png`
   Interpretation: Hybrid ensemble figure showing recovery from pure GNN degradation.
   Use it when: you want to verify the visual evidence behind the corresponding manuscript argument.

## What Changed In This Cleaned Release

- The paper was moved into a dedicated `paper/` directory.
- A top-level `figures/` directory now contains only paper-used images.
- Figure references in the LaTeX sources were rewritten to point at the dedicated figure bundle.
- The old duplicate paper directory was removed.
- Ignore rules were expanded to cover environments, local outputs, checkpoints, and paper build artifacts.
- The manuscript author name was normalized to `Saket Maganti` in the retained paper variants.
- Repository links were removed from the LaTeX paper sources.
- A figure manifest is now written automatically for auditability.

## Recommended Reading Order For New Contributors

1. Read this README from the top until the command cookbook.
2. Open `paper/main.tex` and `paper/paper_shared_body.tex` to understand the paper narrative.
3. Read `data/dataset.py` to understand the temporal split and preprocessing behavior.
4. Read `models/gnn.py`, `models/mlp_baseline.py`, and `models/hybrid.py` to understand the model families being compared.
5. Read `experiments/temporal_analysis.py` and `experiments/hybrid_ensemble.py` because they encode the two most important interpretive claims.
6. Only then branch into the legacy supplement scripts if you need older plots or alternate analyses.

## Citation

If you use this repository or build on the accompanying paper, cite the manuscript and identify the exact repository snapshot you used.

Example BibTeX entry:

```bibtex
@misc{maganti_gnn_fraud_temporal_shift,
  author       = {Saket Maganti},
  title        = {When Graph Structure Becomes a Liability: Temporal Distribution Shift Degrades GNN Fraud Detection but Graph-Derived Representations Survive in Hybrid Ensembles},
  year         = {2026},
  note         = {Repository-aligned research release},
}
```

## Closing Notes

This repository is strongest when it is used transparently: report the split, report the seeds, report the baselines, and be explicit about whether your conclusion concerns standalone GNNs, graph construction, or hybrid models.

If you are extending the work, the most valuable contributions are the ones that preserve this transparency: better temporal robustness, stronger hybrid designs, cleaner ablations, or sharper diagnostics for when relational structure helps and when it misleads.

## Glossary

### Inductive evaluation

Training excludes test-period nodes and edges, so the model must generalize beyond the training graph.

### Transductive evaluation

Training can exploit the full graph structure, including relationships that would not be available at deployment time.

### Temporal drift

Performance degradation associated with the test horizon moving into a different class-prior regime.

### Graph ablation

A controlled comparison where graph structure is altered or removed to measure how much it contributes.

### Hybrid ensemble

A model that combines graph-derived embeddings with raw features before a downstream classifier.

### Calibration

The relationship between predicted confidence and observed correctness.

### Business cost analysis

Evaluation under asymmetric penalties where false negatives and false positives are not equally costly.

### Feature-only baseline

A model such as MLP or Random Forest trained directly on node features without message passing.

### Paper figure manifest

A machine-readable record of the source and destination for every paper-used figure.

### Legacy supplement script

A script retained for traceability or extra analysis even if a cleaner experiment entry point now exists.

## Extended Appendix: Reproduction Reminders

### Reproduction Reminder 1

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 2

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 3

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 4

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 5

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 6

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 7

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 8

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 9

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 10

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 11

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 12

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 13

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 14

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 15

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 16

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 17

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 18

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 19

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 20

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 21

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 22

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 23

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 24

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 25

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 26

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 27

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 28

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 29

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 30

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 31

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 32

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 33

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 34

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 35

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 36

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 37

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 38

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 39

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 40

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 41

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 42

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 43

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 44

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 45

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 46

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 47

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 48

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 49

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 50

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 51

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 52

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 53

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 54

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 55

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 56

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 57

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 58

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 59

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 60

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 61

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 62

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 63

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 64

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 65

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 66

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 67

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 68

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 69

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 70

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 71

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 72

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 73

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 74

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 75

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 76

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 77

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 78

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 79

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 80

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 81

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 82

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 83

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 84

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 85

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 86

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 87

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 88

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 89

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 90

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 91

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 92

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 93

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 94

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 95

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 96

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 97

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 98

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 99

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 100

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 101

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 102

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 103

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 104

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 105

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 106

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 107

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 108

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 109

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 110

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 111

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 112

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 113

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 114

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 115

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 116

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 117

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 118

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 119

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

### Reproduction Reminder 120

Record what you changed before running an experiment. Even small deviations such as epoch count, seed count, class-weight choices, or checkpoint reuse can alter the interpretation of the result.

When relevant, note which paper figure or table the run is supposed to support. This makes later figure sync and manuscript updates much easier to audit.

