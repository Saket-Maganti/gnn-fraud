# Dataset Setup

The Elliptic Bitcoin Dataset is **not included in this repository** (657 MB raw files).

## Download

1. Go to: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
2. Download and unzip — you will get a folder called `elliptic_bitcoin_dataset/` containing:
   - `elliptic_txs_features.csv`
   - `elliptic_txs_classes.csv`
   - `elliptic_txs_edgelist.csv`

3. Place the three CSV files in `data/raw/`:

```
gnn-fraud/
└── data/
    └── raw/
        ├── elliptic_txs_features.csv    (657 MB)
        ├── elliptic_txs_classes.csv     (3.2 MB)
        └── elliptic_txs_edgelist.csv    (4.3 MB)
```

## On Google Colab

```python
# Option A: Upload via Kaggle API
!pip install -q kaggle
# Upload your kaggle.json API key first, then:
!kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw/ --unzip
!mv data/raw/elliptic_bitcoin_dataset/*.csv data/raw/ 2>/dev/null || true

# Option B: Mount Google Drive (if you have the files there)
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p data/raw
!cp "/content/drive/MyDrive/elliptic_bitcoin_dataset/"*.csv data/raw/
```
