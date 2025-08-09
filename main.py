#!/usr/bin/env python3
"""
Fraud Detection with PyOD AutoEncoder — Evaluation-Only
--------------------------------------------------------
- Trains AutoEncoder on NORMAL transactions only (unsupervised anomaly detection).
- Tests on a mixed set (held-out normals + all frauds).
- Outputs:
    * Precision–Recall and ROC curves
    * Classification metrics
    * Score histogram with chosen threshold
    * (Optional) score histogram for top-k% threshold

Compatible with PyOD 1.0.7 and Kaggle creditcard.csv dataset.
"""

import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from pyod.models.auto_encoder import AutoEncoder

# -----------------------
# Global configuration
# -----------------------
RANDOM_STATE = 42  # reproducibility


def set_seeds(seed: int = RANDOM_STATE):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_out_dir(path: str):
    """Create output directory if it doesn’t exist."""
    os.makedirs(path, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset and check for Class column."""
    df = pd.read_csv(csv_path)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column in the dataset.")
    return df


def run_eval_style(df: pd.DataFrame, out_dir: str, top_k_percent: float | None = None):
    """Run evaluation-only fraud detection using PyOD AutoEncoder."""
    # Features and labels
    X = df.drop(columns=["Class"]).values
    y = df["Class"].astype(int).values

    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Split: normal transactions only for training
    normal_mask = (y == 0)
    X_normal = X_scaled[normal_mask]
    X_fraud = X_scaled[~normal_mask]

    X_train, X_hold_normal = train_test_split(
        X_normal, test_size=0.30, random_state=RANDOM_STATE, shuffle=True
    )

    # Mixed test set: held-out normals + all frauds
    X_test = np.vstack([X_hold_normal, X_fraud])
    y_test = np.hstack([
        np.zeros(len(X_hold_normal), dtype=int),
        np.ones(len(X_fraud), dtype=int)
    ])

    # Contamination based on dataset fraud rate
    true_contam = float((y == 1).sum()) / float(len(y))

    # Train AutoEncoder (PyOD 1.0.7-compatible)
    clf = AutoEncoder(
        epoch_num=30,
        contamination=true_contam,
        hidden_neuron_list=[64, 30, 30, 64],
        preprocessing=False,   # already scaled
        batch_size=32,
        dropout_rate=0.2,
        verbose=1,
    )
    clf.fit(X_train)

    # Scores & predictions
    scores = clf.decision_function(X_test)
    thr_default = clf.threshold_
    y_pred = (scores > thr_default).astype(int)

    # Metrics
    roc = roc_auc_score(y_test, scores)
    ap = average_precision_score(y_test, scores)
    report = classification_report(y_test, y_pred, target_names=["Normal", "Fraud"], digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics
    metrics_path = os.path.join(out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=== Evaluation Metrics ===\n")
        f.write(f"Total: {len(y)}, Normal: {(y==0).sum()}, Fraud: {(y==1).sum()}\n")
        f.write(f"Train normals: {len(X_train)}, Test size: {len(X_test)}\n")
        f.write(f"Contamination: {true_contam:.6f}\n")
        f.write(f"Threshold (default): {thr_default:.6f}\n\n")
        f.write(f"ROC AUC: {roc:.6f}\n")
        f.write(f"PR AUC: {ap:.6f}\n\n")
        f.write("Classification Report:\n" + report + "\n")
        f.write("Confusion Matrix [ [TN FP]\n [FN TP] ]:\n")
        f.write(str(cm) + "\n")

    # PR curve
    plt.figure()
    PrecisionRecallDisplay.from_predictions(y_test, scores)
    plt.title("AutoEncoder — Precision-Recall Curve")
    plt.savefig(os.path.join(out_dir, "ae_pr_curve.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # ROC curve
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, scores)
    plt.title("AutoEncoder — ROC Curve")
    plt.savefig(os.path.join(out_dir, "ae_roc_curve.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # Score histogram
    plt.figure()
    plt.hist(scores, bins=60)
    plt.axvline(thr_default, linestyle="dotted")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title("Score Distribution with Default Threshold")
    plt.savefig(os.path.join(out_dir, "ae_score_hist.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # Optional: top-k% threshold
    if top_k_percent:
        q = 1 - (top_k_percent / 100.0)
        thr_topk = np.quantile(scores, q)
        y_pred_topk = (scores >= thr_topk).astype(int)
        report_topk = classification_report(
            y_test, y_pred_topk, target_names=["Normal", "Fraud"], digits=4
        )
        cm_topk = confusion_matrix(y_test, y_pred_topk)

        with open(metrics_path, "a") as f:
            f.write("\n=== Top-k% Threshold Evaluation ===\n")
            f.write(f"Top-k%: {top_k_percent}% | Threshold: {thr_topk:.6f}\n")
            f.write("Classification Report:\n" + report_topk + "\n")
            f.write("Confusion Matrix:\n" + str(cm_topk) + "\n")

        plt.figure()
        plt.hist(scores, bins=60)
        plt.axvline(thr_topk, linestyle="dotted")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Count")
        plt.title(f"Score Distribution with Top-{top_k_percent}% Threshold")
        plt.savefig(os.path.join(out_dir, "ae_score_hist_topk.png"), dpi=160, bbox_inches="tight")
        plt.close()


def append_environment_info(out_dir: str):
    """Append environment info to metrics.txt for reproducibility."""
    try:
        import pyod, sklearn, matplotlib, numpy, pandas
        env_text = [
            "\n=== Environment ===",
            f"Python: {sys.version.split()[0]}",
            f"pyod: {pyod.__version__}",
            f"scikit-learn: {sklearn.__version__}",
            f"numpy: {numpy.__version__}",
            f"pandas: {pandas.__version__}",
            f"matplotlib: {matplotlib.__version__}",
        ]
        with open(os.path.join(out_dir, "metrics.txt"), "a") as f:
            f.write("\n".join(env_text) + "\n")
    except Exception as e:
        print("Warning: could not append environment info:", e)


def parse_args():
    parser = argparse.ArgumentParser(description="PyOD AutoEncoder Fraud Detection — Eval Only")
    parser.add_argument("--data_path", type=str, required=True, help="Path to creditcard.csv")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--top_k_percent", type=float, default=None,
                        help="Also evaluate top-k%% threshold (e.g., 0.2 = top 0.2%% flagged)")
    return parser.parse_args()


def main():
    set_seeds(RANDOM_STATE)
    args = parse_args()
    ensure_out_dir(args.out_dir)
    df = load_data(args.data_path)
    run_eval_style(df, args.out_dir, top_k_percent=args.top_k_percent)
    append_environment_info(args.out_dir)
    print("Done. Outputs saved to:", args.out_dir)


if __name__ == "__main__":
    main()
