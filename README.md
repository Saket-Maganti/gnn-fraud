# When Graph Structure Becomes a Liability

**A Critical Re-Evaluation of Graph Neural Networks for Bitcoin Fraud
Detection under Temporal Distribution Shift**

*Saket Maganti*

This repository contains the code, configuration, and figure pipeline
behind the arXiv preprint of the same title.  The paper re-examines the
widely cited claim that Graph Neural Networks (GNNs) are the
state-of-the-art model family for fraud detection on the Elliptic
Bitcoin Dataset, and shows that this claim does not survive an
inductive, per-timestep evaluation protocol.

- 📄 Paper source: [`gnnpaper/main.tex`](gnnpaper/main.tex)
- 🖼️ Figures: [`figures/`](figures/)
- 🧪 Experiments: [`experiments/`](experiments/)

---

## 1. Overview

The Elliptic Bitcoin Dataset has become the canonical benchmark for
graph-based fraud detection.  Published results consistently rank
GCN (F1 = 0.70), augmented GCN (0.74), GraphSAGE + self-supervision
(0.75), and EvolveGCN (0.77) above feature-only baselines.  Every one
of those results is produced under a **transductive** protocol that
leaks test-period structure into training and reports a **single
aggregate F1** over a 15-step test window.

We re-run GCN, GraphSAGE, and GAT under **strictly inductive** batching
with **multi-seed validation** and **per-timestep** reporting.  Under
this corrected protocol the ranking inverts:

| Model                             | Protocol          | Graph | F1         |
| --------------------------------- | ----------------- | ----- | ---------- |
| Random Forest (raw features)      | Inductive (ours)  | No    | **0.820**  |
| **Concat hybrid** (SAGE emb + RF) | Inductive (ours)  | Yes   | **0.807**  |
| XGBoost (raw features)            | Inductive (ours)  | No    | 0.775      |
| MLP, 3 layers                     | Inductive (ours)  | No    | 0.744      |
| GraphSAGE, weighted CE            | Inductive (ours)  | Yes   | 0.697      |
| GAT, graph-aug                    | Inductive (ours)  | Yes   | 0.576      |
| GCN, weighted CE                  | Inductive (ours)  | Yes   | 0.473      |
| GCN (Weber et al. 2019)           | Transductive      | Yes   | 0.700      |
| EvolveGCN (Pareja et al. 2020)    | Transductive      | Yes   | 0.770      |

All tested GNNs collapse from F1 ≈ 0.78 at step 35 to F1 < 0.03 after
step 43.  The fraud base rate drops 39× between training and the worst
test steps, and temperature scaling yields ΔF1 ≈ 0, so the failure is
distributional, not calibration.

A multi-seed edge-shuffle ablation pushes this further: **randomly
shuffling the Bitcoin transaction edges improves GraphSAGE F1 from
0.268 to 0.383**.  The real graph is not just uninformative, it is
actively harmful to an end-to-end GNN under this protocol.

The constructive half of the paper recovers graph information inside a
hybrid: **GraphSAGE embeddings concatenated with raw features and fed
into a downstream Random Forest** reach F1 = 0.807, exceeding every
prior published result on Elliptic while respecting inductive
evaluation.  Graph structure is therefore a **conditional asset**, not
an unconditional improvement.

---

## 2. Key Contributions

1. **Corrected evaluation protocol.**  Strictly inductive batching for
   GCN, GraphSAGE, and GAT on Elliptic, with multi-seed validation
   (3–5 seeds) and per-timestep reporting across the full 15-step test
   window.
2. **Inversion of the published ranking.**  Under the corrected
   protocol, a plain three-layer MLP (0.744) and a Random Forest
   (0.820) beat every tested GNN.
3. **Multi-seed graph-structure ablation.**  Original vs. randomly
   shuffled vs. no edges; shuffled edges outperform original edges by
   11.6 F1 points (Cohen's *d* > 2), identifying the transaction graph
   itself as a source of degradation under shift.
4. **Graph construction sweep.**  Five variants (original, similarity,
   feature k-NN, temporal, augmented) under identical training
   conditions; the temporal cross-timestep graph lifts GNN F1 by 89%
   relative to the original.
5. **Calibration diagnosis.**  Temperature scaling of all four GNN
   configurations; ΔF1 ≈ 0, ΔECE < 0.002, ruling out post-hoc
   recalibration as a remedy for temporal collapse.
6. **Concatenation hybrid.**  256-dim GNN embeddings concatenated with
   165 raw features into a downstream Random Forest; F1 = 0.807,
   exceeding all prior published results while remaining inductive.
7. **Business cost analysis.**  Asymmetric FN/FP penalties with
   cost-ratio sweeps from 1:1 to 100:1, showing that F1-optimal models
   are not cost-optimal.
8. **Reproducibility package.**  Full experiment scripts, figure
   pipeline, and frozen checkpoints; the paper’s figure manifest is
   checked into [`gnnpaper/figure_manifest.json`](gnnpaper/figure_manifest.json).

---

## 3. Methodology

### Data

The Elliptic Bitcoin Dataset provides 203,769 transaction nodes across
49 temporal steps, 234,355 directed BTC-flow edges, and 165 node
features (94 local + 71 manually engineered neighbourhood aggregates).
46,564 nodes are labelled (9.8% illicit).  We follow the standard
temporal split: training on steps 1–34, test on steps 35–49.

### Models

- **MLP** — three fully connected layers (128 dim), BatchNorm, ReLU,
  dropout 0.5.  Accepts `edge_index` for API parity but ignores it.
- **GCN** — two-layer symmetric-normalised GCN (Kipf & Welling, 2017),
  hidden dim 128.
- **GraphSAGE** — three-layer mean-aggregation SAGE (Hamilton et al.,
  2017), designed for inductive learning.  Main GNN under study.
- **GAT** — three-layer GAT (Veličković et al., 2018), 4 heads, mean
  pooling across heads, edge dropout 0.2.
- **SAGE-Deep / SAGE-MaxPool** — depth and aggregator ablations.
- **EvolveGCN** — hardware-characterisation run only; CPU-constrained.
- **Classical baselines** — Logistic Regression, Random Forest, XGBoost
  on the raw 165-dim feature vector.
- **Hybrid** — GraphSAGE penultimate-layer embeddings (256 dim)
  concatenated with the 165 raw features, classified by a downstream
  Random Forest.

### Imbalance strategies

- *Baseline* — plain cross-entropy.
- *Weighted CE* — square-root inverse-frequency weighting with 20-epoch
  linear warmup.
- *Graph augmentation* — 30 cloned 1-hop ego-graphs around fraud seed
  nodes with σ = 0.02 feature perturbation, attached as isolated
  components (Algorithm 1 in the paper).

### Evaluation protocol

- **Train/test split.** Steps 1–34 / steps 35–49.
- **Inductive batching.** `RandomNodeLoader` with batch size 512.
  Each batch only sees edges among its sampled nodes; test-period
  neighbourhoods never enter training.
- **Seeds.** 3 seeds for the main benchmark and the graph-ablation;
  5-seed secondary runs confirm ranking stability.
- **Metrics.** F1 on the illicit class (primary), precision, recall,
  AUC-ROC, ECE, Brier score, business cost under asymmetric penalties,
  and per-timestep breakdowns for every configuration.

---

## 4. Repository Structure

```text
gnn-fraud/
├── gnnpaper/                    # arXiv preprint source (LaTeX)
│   ├── main.tex                 # arXiv wrapper with clean preamble
│   ├── paper_shared_body.tex    # all sections, equations, tables, figures
│   ├── references.bib           # bibliography
│   └── figure_manifest.json     # frozen figure SHA-256 index
├── figures/                     # every figure used by the paper (25 PNGs)
├── data/                        # dataset loaders + graph construction
│   ├── dataset.py
│   └── graph_builder.py
├── models/                      # architecture definitions
│   ├── gnn.py                   # GCN / SAGE / GAT / SAGE-Deep / SAGE-MaxPool
│   ├── mlp_baseline.py
│   └── hybrid.py
├── experiments/                 # primary experiment entry points
│   ├── baselines.py             # LR / RF / XGBoost baselines
│   ├── run_seeds.py             # multi-seed GNN benchmark
│   ├── run_transductive.py      # GraphSAGE, full-graph (leaky) training
│   ├── run_inductive.py         # GraphSAGE, strict inductive training
│   ├── compare_leakage_gap.py   # transductive vs inductive F1 table
│   ├── temporal_analysis.py     # per-timestep F1 / precision / recall
│   ├── graph_ablation_extended.py  # original vs shuffled vs no edges
│   ├── hybrid_ensemble.py       # concat + prob-level fusion
│   ├── calibration_study.py     # temperature scaling + reliability
│   ├── pr_analysis.py           # PR curves, optimal thresholds
│   ├── business_cost.py         # FN/FP cost sweeps
│   ├── distribution_shift.py    # MMD, L2 drift diagnostics
│   ├── embedding_viz.py         # t-SNE of GNN embeddings
│   └── significance_tests.py    # Welch t-test, Cohen's d
├── notebooks/
│   └── colab_transductive_vs_inductive.ipynb  # Colab-ready leakage-gap run
├── scripts/                     # auxiliary figure and add-on runners
│   ├── addition*.py             # individual figure-generating scripts
│   └── sync_paper_figures.py    # copies figures into the paper tree
├── utils/
│   ├── trainer.py               # full-graph trainer (legacy)
│   ├── trainer_minibatch.py     # inductive minibatch trainer
│   ├── imbalance.py             # weighted loss + graph augmentation
│   ├── metrics.py
│   └── visualise.py
├── train.py                     # minibatch training entry point
├── train_proper.py              # alt. training flow with validation split
├── config.py                    # Config dataclass
├── requirements.txt
└── README.md
```

---

## 5. Installation

We recommend Python 3.10+.  The code has been validated on Apple
Silicon (M4 CPU) and on Linux with CUDA.

```bash
git clone https://github.com/<your-user>/gnn-fraud.git
cd gnn-fraud

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset

The Elliptic Bitcoin Dataset is publicly available on Kaggle
(search: `elliptic-data-set`).  Download the three CSVs and place them
under `data/raw/elliptic_bitcoin_dataset/`:

```text
data/raw/elliptic_bitcoin_dataset/
├── elliptic_txs_classes.csv
├── elliptic_txs_edgelist.csv
└── elliptic_txs_features.csv
```

`data/dataset.py` handles parsing, feature normalisation, the temporal
split, and conversion to a PyG `Data` object.

---

## 6. How to Run the Experiments

Every experiment reads a `Config` dataclass from `config.py`.  The
commands below reproduce the paper’s main tables and figures.

### 6.1 Main GNN benchmark (Table 2, Figure 1)

```bash
# Multi-seed GraphSAGE / GCN / GAT / MLP benchmark
python experiments/run_seeds.py \
    --models mlp gcn sage gat \
    --strategies baseline weighted graph_aug \
    --seeds 0 1 2
```

### 6.2 Classical baselines (Table 3)

```bash
python experiments/baselines.py --models lr rf xgboost
```

### 6.3 Graph construction sweep (Figure, Table 4)

```bash
for g in original similarity feature_knn temporal augmented; do
  python train.py --model sage --strategy weighted --graph_type $g --epochs 300
done
```

### 6.4 Graph-structure ablation (Table 6, Figure)

```bash
python experiments/graph_ablation_extended.py --seeds 0 1 2
```

This runs GraphSAGE under original / shuffled / no-edge conditions and
emits `figures/graph_ablation_extended.png`.

### 6.5 Per-timestep temporal analysis (Figures 2, 3)

```bash
python experiments/temporal_analysis.py
python experiments/temporal_deep.py
```

### 6.6 Calibration study (Figures, Table 7)

```bash
python experiments/calibration_study.py
```

### 6.7 Hybrid ensembles (Table 8, Figure)

```bash
python experiments/hybrid_ensemble.py
```

### 6.8 Business cost analysis (Figure 7, Table 9)

```bash
python experiments/business_cost.py --fn_cost 50000 --fp_cost 2000
```

### 6.9 Transductive vs. Inductive Evaluation Gap

This lightweight experiment quantifies how much GraphSAGE's Elliptic
performance comes from **structural leakage** — test-period edges being
visible to message passing during training — rather than from true
inductive generalisation.  Both runs use **identical** hyperparameters,
seed, loss, and the same `GraphSAGE` class from `models/gnn.py`; they
differ **only** in the adjacency used at training time.

- **Transductive:** message passing during training uses the full graph
  (edges that touch test-period nodes are visible).  This is the
  protocol behind most published Elliptic results.
- **Inductive:** the training adjacency is restricted to edges whose
  **both** endpoints lie in time steps 1–34.  At inference time the
  full adjacency is restored so that test-period nodes can still
  receive messages from their train-period neighbourhoods.

#### Local execution

```bash
# 1. Train GraphSAGE with the full (leaky) adjacency
python experiments/run_transductive.py --epochs 200 --seed 42

# 2. Train GraphSAGE with the inductive adjacency mask
python experiments/run_inductive.py   --epochs 200 --seed 42

# 3. Print the comparison table
python experiments/compare_leakage_gap.py
```

The runners write JSON results to:

```text
results/transductive_results.json
results/inductive_results.json
```

Each JSON contains the best test metrics, the per-timestep F1 breakdown
for steps 35–49, and the full hyperparameter snapshot.  The comparison
utility prints a small table of the form:

```text
Transductive vs Inductive Results

Model       Setting       F1
--------------------------------
GraphSAGE   Transductive  X.XXX
GraphSAGE   Inductive     X.XXX

Leakage gap (transductive − inductive F1): +X.XXX
```

#### Google Colab

A ready-to-run notebook lives at
[`notebooks/colab_transductive_vs_inductive.ipynb`](notebooks/colab_transductive_vs_inductive.ipynb).

Steps inside the notebook:

1. `pip install` `torch-geometric` and dependencies.
2. `git clone` this repository.
3. Pull the Elliptic CSVs from Kaggle (prompts for your `kaggle.json`
   API token).
4. Run `experiments/run_transductive.py`.
5. Run `experiments/run_inductive.py`.
6. Run `experiments/compare_leakage_gap.py`.
7. Print the per-timestep F1 table for both runs.

#### Expected outputs

- `results/transductive_results.json` — leaky full-graph training.
- `results/inductive_results.json`    — strictly inductive training.
- Console table from `compare_leakage_gap.py` showing the F1 gap.

#### Estimated runtimes

On Apple M4 CPU with the default configuration (200 epochs,
`hidden=256`, `layers=3`):

| Runner                                  | Wall time  |
| --------------------------------------- | ---------- |
| `experiments/run_transductive.py`       | ~6–10 min  |
| `experiments/run_inductive.py`          | ~5–9 min (slightly fewer edges) |
| `experiments/baselines.py` (MLP)        | ~3 min     |

The inductive run is typically a touch faster because the masked
adjacency contains fewer edges, which reduces the cost of message
passing at each training step.  On a Colab T4 GPU both runners drop to
~1–2 minutes each.

### 6.10 Build the paper

```bash
cd gnnpaper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## 7. Results Summary

### Main benchmark (3 seeds, inductive)

| Model | Strategy      | F1                | Prec.             | Recall            | AUC               |
| ----- | ------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| MLP   | Baseline      | **0.744 ± 0.005** | **0.825 ± 0.004** | 0.678 ± 0.006     | 0.891 ± 0.001     |
| MLP   | Weighted      | 0.735 ± 0.001     | 0.796 ± 0.008     | 0.682 ± 0.005     | **0.893 ± 0.006** |
| GCN   | Weighted      | 0.473 ± 0.001     | 0.889 ± 0.005     | 0.322 ± 0.001     | 0.872 ± 0.004     |
| SAGE  | Weighted      | 0.697 ± 0.009     | 0.736 ± 0.015     | 0.662 ± 0.004     | 0.868 ± 0.003     |
| SAGE  | Graph Aug.    | 0.686 ± 0.008     | 0.719 ± 0.042     | 0.659 ± 0.019     | 0.861 ± 0.007     |
| GAT   | Weighted      | 0.555 ± 0.024     | 0.533 ± 0.048     | **0.581 ± 0.013** | 0.855 ± 0.004     |
| GAT   | Graph Aug.    | 0.576 ± 0.013     | 0.581 ± 0.033     | 0.573 ± 0.010     | 0.842 ± 0.002     |

### Graph-structure ablation (3 seeds)

| Condition       | F1                | Precision        | Recall           | AUC              |
| --------------- | ----------------- | ---------------- | ---------------- | ---------------- |
| Original graph  | 0.268 ± 0.030     | 0.163 ± 0.020    | 0.745 ± 0.074    | 0.825 ± 0.047    |
| Shuffled edges  | **0.383 ± 0.023** | **0.258 ± 0.022**| **0.754 ± 0.007**| **0.884 ± 0.003**|
| No edges (MLP)  | 0.332 ± 0.018     | 0.210 ± 0.015    | 0.799 ± 0.007    | 0.878 ± 0.003    |

### Hybrid ensembles

| Strategy                          | F1        | Precision | Recall | AUC       |
| --------------------------------- | --------- | --------- | ------ | --------- |
| GNN alone (SAGE+weighted)         | 0.311     | 0.459     | 0.236  | 0.760     |
| MLP alone                         | 0.314     | 0.195     | 0.807  | 0.879     |
| Weighted avg (α = 0.65)           | 0.453     | 0.330     | 0.722  | 0.864     |
| **Concat (GNN emb + raw → RF)**   | **0.807** | **0.976** | 0.687  | **0.900** |

### Temporal collapse (SAGE + weighted)

| Region            | Mean F1 | Mean Precision | Mean Recall |
| ----------------- | ------- | -------------- | ----------- |
| Stable (35–42)    | 0.381   | 0.326          | 0.542       |
| Collapsed (43–49) | 0.028   | 0.022          | 0.087       |

---

## 8. Figures

All 25 paper figures live in [`figures/`](figures/).  Each is
referenced by `figures/<name>.png` in the LaTeX source.  The most
load-bearing ones:

| Figure                            | What it shows                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| `fig1_main_results.png`           | Main F1 / precision / recall benchmark across 10 configurations, 3 seeds.                   |
| `fig2_temporal_drift.png`         | Per-step F1 vs. true illicit rate; all models collapse together after step 42.              |
| `fig3_perclass_temporal.png`      | Per-class temporal breakdown showing the collapse is a recall failure.                     |
| `graph_ablation_extended.png`     | Original vs. shuffled vs. no-edges GraphSAGE ablation, per time step.                       |
| `hybrid_ensemble.png`             | Hybrid fusion comparison; concat hybrid dramatically outperforms components.               |
| `calibration_curves.png`          | Reliability diagrams for all four GNN configurations.                                       |
| `fig13_temperature_scaling.png`   | Temperature scaling ΔF1 ≈ 0 diagnosis.                                                      |
| `fig7_business_cost.png`          | Asymmetric-cost ranking reshuffles F1-based ranking.                                        |
| `fig16_ablation_advantage.png`    | Per-step GraphSAGE advantage over MLP; brief early gain, persistent later deficit.         |
| `fig17_dataset_analysis.png`      | Dataset characterisation: temporal fraud rate, class composition, feature correlations.    |

`gnnpaper/figure_manifest.json` contains SHA-256 hashes for every
paper figure so that regenerated figures can be diffed against the
frozen set used in the preprint.

---

## 9. Reproducibility Notes

- **Determinism.**  All randomness flows through a single `--seed`
  argument.  Seeds 0, 1, 2 reproduce the main benchmark; seeds 0–4
  reproduce the 5-seed secondary validation.  Shuffled-edges
  experiments draw edge permutations from the same seed stream.
- **Hardware.**  Primary results were produced on Apple M4 CPU
  (16 GB RAM) using PyTorch 2.2.2, torch-geometric 2.5.3, and
  `RandomNodeLoader`.  `NeighborLoader`, which provides cleaner k-hop
  inductive batching, is unavailable on M4 CPU without `torch-sparse`;
  the `limitations` section of the paper discusses this.  The code
  runs unchanged on CUDA.
- **Frozen checkpoints.**  `best_model.pt` pins the GraphSAGE weights
  used for the hybrid ensemble and embedding visualisations.
- **Figure regeneration.**  `scripts/sync_paper_figures.py` copies
  regenerated figures from `results/` into `figures/` and rewrites the
  manifest.  The manifest guarantees the paper always compiles against
  the exact image set listed in `figure_manifest.json`.
- **Unknowns.**  The 157,205 unknown-label nodes in Elliptic are
  excluded from training and evaluation, matching all prior work.

---

## 10. Future Work

1. **GPU-class inductive evaluation of temporal GNNs.**  EvolveGCN,
   DySAT, and ROLAND all require full temporal snapshot sequences and
   GPU memory.  The open question is whether they retain their
   published advantage when the evaluation is made inductive and
   per-timestep.
2. **Sliding-window retraining.**  The 39× base-rate drop between
   training and deployment argues for online retraining.  Finding the
   optimal window W and cadence Δt is a hyperparameter problem in its
   own right.
3. **Deeper hybrids.**  What happens when the downstream classifier is
   XGBoost or a small neural network instead of a Random Forest?  Does
   the hybrid collapse under the same temporal shift, or does the
   downstream model's arbitration survive?
4. **Adversarial robustness.**  If fraud actors know GNNs look for
   dense clusters, Nettack-style attacks should be tested.
5. **Cross-chain generalisation.**  Bitcoin → Ethereum, and from
   blockchains into traditional payment-network graphs.
6. **Federated GNN training.**  Multi-institution fraud detection
   without sharing raw transaction data.

---

## Citation

```bibtex
@article{maganti2026graphliability,
  title   = {When Graph Structure Becomes a Liability:
             A Critical Re-Evaluation of Graph Neural Networks for
             Bitcoin Fraud Detection under Temporal Distribution Shift},
  author  = {Maganti, Saket},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgements

The author thanks VIT-AP University for computational resources and
the Elliptic team for publicly releasing the Bitcoin fraud dataset.

## License

Code in this repository is released for academic research use.  The
Elliptic Bitcoin Dataset is subject to its own redistribution terms;
see the Kaggle dataset page for details.
