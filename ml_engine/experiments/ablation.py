"""AI-NIDS Feature Ablation Experiment.

Investigates the contribution of different feature subsets:
1. All features (baseline 24 features)
2. Top-10 features (by Random Forest importance)
3. Inter-arrival time & rate features only
4. Packet length features only
5. Flags & Window features only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from ml_engine.evaluate import compute_false_positive_rate, measure_inference_latency

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_ablation_experiment(
    data_dir: Path,
    results_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate feature subsets to measure performance vs latency tradeoffs."""
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    # Train full model to extract feature importances
    full_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)
    full_rf.fit(X_train, y_train_enc)
    importances = pd.Series(full_rf.feature_importances_, index=X_train.columns)
    top_10_features = importances.nlargest(10).index.tolist()

    # Define ablation subsets
    subsets: Dict[str, List[str]] = {
        "All Features (24)": list(X_train.columns),
        "Top-10 Features": top_10_features,
        "IAT & Rates Only": [c for c in X_train.columns if "iat" in c or "sec" in c or "duration" in c],
        "Packet Lengths Only": [c for c in X_train.columns if "len" in c or "bytes" in c],
        "Flags & Windows Only": [c for c in X_train.columns if "tcp" in c or "win" in c or "port" in c],
    }

    records: List[Dict[str, Any]] = []

    for name, cols in subsets.items():
        if not cols:
            continue
        logger.info(f"Evaluating ablation group: [{name}] ({len(cols)} features)...")
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)
        clf.fit(X_train[cols], y_train_enc)
        preds = clf.predict(X_test[cols])
        preds_str = pd.Series(encoder.inverse_transform(preds), index=y_test.index)

        acc = float(accuracy_score(y_test_enc, preds))
        macro_f1 = float(f1_score(y_test_enc, preds, average="macro"))
        fpr = compute_false_positive_rate(y_test, preds_str)
        latency = measure_inference_latency(clf, X_test[cols].iloc[:20], n_iterations=100)

        records.append({
            "subset_name": name,
            "feature_count": len(cols),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "false_positive_rate": fpr,
            "latency_mean_ms": latency["latency_mean_ms"],
        })

    results_df = pd.DataFrame(records)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "ablation_study.csv"
    results_df.to_csv(out_csv, index=False)
    logger.info(f"Saved ablation study to: {out_csv}")

    # Generate LaTeX table
    tex_dir = results_dir / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / "ablation_study.tex"

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Feature ablation study results\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Feature Subset & Features & Accuracy & Macro-F1 & Latency (ms) \\\\\n\\hline\n")
        for _, row in results_df.iterrows():
            f.write(
                f"{row['subset_name']} & {int(row['feature_count'])} & {row['accuracy']*100:.2f}\\% & "
                f"{row['macro_f1']:.4f} & {row['latency_mean_ms']:.3f} ms \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n")
    logger.info(f"Saved LaTeX table to: {tex_path}")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature ablation evaluation.")
    parser.add_argument("--data-dir", type=str, default="dataset/processed")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    data_d = (REPO_ROOT / args.data_dir).resolve()
    res_d = (REPO_ROOT / args.results_dir).resolve()

    df = run_ablation_experiment(data_d, res_d)

    print("\n" + "=" * 70)
    print(" FEATURE ABLATION EXPERIMENTAL RESULTS")
    print("=" * 70)
    for _, row in df.iterrows():
        print(
            f"{row['subset_name']:22s} ({row['feature_count']:2d} feats) | "
            f"Macro-F1: {row['macro_f1']:.4f} | Latency: {row['latency_mean_ms']:.2f} ms"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
