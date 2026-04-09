"""
scripts/addition10_feature_importance.py
Feature importance analysis using Random Forest + permutation importance.

Explains WHY the MLP beats GNNs:
- Identifies which of 165 features drive fraud detection
- Shows feature importance stability across train/test time steps
- Answers the reviewer question "why does a featureless model work better?"

Uses sklearn RandomForest (fast, no SHAP dependency needed).
Falls back to permutation importance if needed.

Codepath : scripts/addition10_feature_importance.py
Runtime  : ~3 min
Output   : results/additions/feature_importance.png
           results/additions/top_features.csv
"""

import sys, os, torch
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/additions", exist_ok=True)

# Elliptic feature groups
# Col 0: txId (stripped), Col 1: time step (stripped)
# Cols 2-95:  94 local features (transaction-level)
# Cols 96-166: 72 aggregated neighbourhood features
LOCAL_FEATURES = list(range(0, 94))    # indices in feat_mat (0-indexed)
AGG_FEATURES   = list(range(94, 165))  # aggregate neighbourhood


def main():
    print("Loading data...")
    f_raw, c, e = load_elliptic_raw()
    data = preprocess(f_raw, c, e)

    X_train = data.x[data.train_mask].numpy()
    y_train = data.y[data.train_mask].numpy()
    X_test  = data.x[data.test_mask].numpy()
    y_test  = data.y[data.test_mask].numpy()

    # Binary labels: 1=illicit, 0=licit (drop unknown already filtered by mask)
    y_train_bin = (y_train == 1).astype(int)
    y_test_bin  = (y_test  == 1).astype(int)

    print(f"Train: {X_train.shape}, illicit={y_train_bin.sum()}")
    print(f"Test:  {X_test.shape},  illicit={y_test_bin.sum()}")

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train_bin)

    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test_bin, y_pred, zero_division=0)
    print(f"RF Test F1: {f1:.4f}")
    print(classification_report(y_test_bin, y_pred,
          target_names=["Licit","Illicit"], digits=4))

    # Feature importances
    importances = rf.feature_importances_
    std         = np.std([t.feature_importances_
                          for t in rf.estimators_], axis=0)
    feat_names  = [f"f{i}" for i in range(len(importances))]

    # Label feature groups
    groups = []
    for i in range(len(importances)):
        if i in LOCAL_FEATURES:
            groups.append("Local")
        else:
            groups.append("Aggregate")

    df_imp = pd.DataFrame({
        "feature":    feat_names,
        "importance": importances,
        "std":        std,
        "group":      groups,
        "rank":       pd.Series(importances).rank(ascending=False).astype(int),
    }).sort_values("importance", ascending=False)

    df_imp.to_csv("results/additions/top_features.csv", index=False)

    # Top 20
    top20 = df_imp.head(20)
    print(f"\nTop 20 features by RF importance:")
    print(top20[["feature","importance","group"]].to_string(index=False))

    # Group analysis
    local_imp = df_imp[df_imp["group"] == "Local"]["importance"].sum()
    agg_imp   = df_imp[df_imp["group"] == "Aggregate"]["importance"].sum()
    print(f"\nLocal feature importance total   : {local_imp:.4f} "
          f"({local_imp*100:.1f}%)")
    print(f"Aggregate feature importance total: {agg_imp:.4f} "
          f"({agg_imp*100:.1f}%)")

    # ── Plot 1: Top 30 feature importances ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Random Forest Feature Importance Analysis\n"
                 "(explains why node features alone suffice for fraud detection)",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    top30 = df_imp.head(30)
    colors_bar = ["#EF4444" if g == "Local" else "#3B82F6"
                  for g in top30["group"]]
    bars = ax.barh(range(len(top30)), top30["importance"],
                   xerr=top30["std"], color=colors_bar, alpha=0.85,
                   capsize=3, error_kw={"elinewidth": 1})
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(
        [f"{r['feature']} ({r['group'][0]})"
         for _, r in top30.iterrows()],
        fontsize=8
    )
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title("Top 30 Features")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Rectangle((0,0),1,1, color="#EF4444", alpha=0.85,
                       label=f"Local features (cols 0-93, total={local_imp:.3f})"),
        plt.Rectangle((0,0),1,1, color="#3B82F6", alpha=0.85,
                       label=f"Aggregate features (cols 94-164, total={agg_imp:.3f})"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    # ── Plot 2: Feature group pie + cumulative importance ──────────────
    ax2 = axes[1]
    # Cumulative importance curve
    sorted_imp = np.sort(importances)[::-1]
    cumulative = np.cumsum(sorted_imp)
    ax2.plot(range(1, len(cumulative)+1), cumulative,
             color="#2563EB", lw=2)
    ax2.axhline(0.5, color="orange", linestyle="--", lw=1.2,
                label="50% importance threshold")
    ax2.axhline(0.8, color="red", linestyle="--", lw=1.2,
                label="80% importance threshold")

    # Mark where 50% and 80% are reached
    idx_50 = np.argmax(cumulative >= 0.5) + 1
    idx_80 = np.argmax(cumulative >= 0.8) + 1
    ax2.axvline(idx_50, color="orange", linestyle=":", lw=1, alpha=0.7)
    ax2.axvline(idx_80, color="red",    linestyle=":", lw=1, alpha=0.7)
    ax2.text(idx_50 + 2, 0.45, f"Top {idx_50} features\n= 50% importance",
             fontsize=8, color="orange")
    ax2.text(idx_80 + 2, 0.75, f"Top {idx_80} features\n= 80% importance",
             fontsize=8, color="red")

    ax2.set_xlabel("Number of features (ranked)")
    ax2.set_ylabel("Cumulative importance")
    ax2.set_title("Cumulative Feature Importance")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlim(0, 165)
    ax2.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig("results/additions/feature_importance.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Feature importance by time period ────────────────────────
    # Does importance shift in late time steps?
    print("\nComputing feature importance by time period...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Feature Importance Stability Across Time Periods",
                 fontsize=12, fontweight="bold")

    periods = [
        ("Early test (steps 35-39)", (35, 39)),
        ("Late test  (steps 43-49)", (43, 49)),
    ]

    for ax, (period_name, (t_min, t_max)) in zip(axes, periods):
        mask = (data.test_mask &
                (data.time_step >= t_min) &
                (data.time_step <= t_max))
        if mask.sum().item() < 20:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        X_p = data.x[mask].numpy()
        y_p = (data.y[mask] == 1).numpy().astype(int)

        if y_p.sum() < 3:
            ax.text(0.5, 0.5, f"Too few fraud nodes\n(n={y_p.sum()})",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
            ax.set_title(period_name, fontsize=9)
            continue

        rf_p = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        rf_p.fit(X_p, y_p)
        f1_p = f1_score(y_p, rf_p.predict(X_p), zero_division=0)

        top_p = pd.Series(rf_p.feature_importances_,
                          index=feat_names).nlargest(15)
        colors_p = ["#EF4444" if i < 94 else "#3B82F6"
                    for i in [int(n[1:]) for n in top_p.index]]

        ax.barh(range(len(top_p)), top_p.values,
                color=colors_p, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(top_p)))
        ax.set_yticklabels(top_p.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{period_name}\n(Train F1={f1_p:.3f}, "
                     f"n={mask.sum().item()}, fraud={y_p.sum()})",
                     fontsize=9)
        ax.set_xlabel("Feature Importance")
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/additions/feature_importance_by_period.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSaved: results/additions/feature_importance.png")
    print("Saved: results/additions/feature_importance_by_period.png")
    print("Saved: results/additions/top_features.csv")
    print(f"\nRF Test F1: {f1:.4f} (for reference: MLP F1=0.744)")
    print(f"Top {idx_50} features explain 50% of importance")
    print(f"Top {idx_80} features explain 80% of importance")


if __name__ == "__main__":
    main()