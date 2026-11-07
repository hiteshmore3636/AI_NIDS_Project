"""AI-NIDS Multi-Seed Statistical Validation Experiment.

Trains and evaluates the primary model across >= 5 randomized seeds to guarantee
scientific reproducibility and report true empirical Mean +/- Standard Deviation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from ml_engine.evaluate import compute_false_positive_rate

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_multi_seed_experiment(
    data_dir: Path,
    seeds: List[int],
    results_dir: Path,
) -> pd.DataFrame:
    """Train and evaluate models across multiple seeds."""
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    records: List[Dict[str, Any]] = []

    for seed in seeds:
        logger.info(f"Running experiment with random_seed={seed}...")
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_split=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X_train, y_train_enc)
        preds_enc = clf.predict(X_test)
        preds_str = pd.Series(encoder.inverse_transform(preds_enc), index=y_test.index)

        acc = float(accuracy_score(y_test_enc, preds_enc))
        macro_f1 = float(f1_score(y_test_enc, preds_enc, average="macro"))
        weighted_f1 = float(f1_score(y_test_enc, preds_enc, average="weighted"))
        macro_prec = float(precision_score(y_test_enc, preds_enc, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_test_enc, preds_enc, average="macro", zero_division=0))
        fpr = compute_false_positive_rate(y_test, preds_str, benign_label="BENIGN")

        records.append({
            "seed": seed,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "false_positive_rate": fpr,
        })

    results_df = pd.DataFrame(records)

    # Calculate summary statistics row
    summary_mean = results_df.mean(axis=0)
    summary_std = results_df.std(axis=0)

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "multi_seed.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved multi-seed experiment results to: {csv_path}")

    # Generate LaTeX table
    tex_dir = results_dir / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / "multi_seed.tex"

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Multi-seed stability results\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Seed & Accuracy & Macro-F1 & Precision & FPR \\\\\n\\hline\n")
        for _, row in results_df.iterrows():
            f.write(
                f"{int(row['seed'])} & {row['accuracy']*100:.2f}\\% & {row['macro_f1']:.4f} & "
                f"{row['macro_precision']:.4f} & {row['false_positive_rate']*100:.2f}\\% \\\\\n"
            )
        f.write("\\hline\n")
        f.write(
            f"Mean $\\pm$ Std & {summary_mean['accuracy']*100:.2f}\\pm{summary_std['accuracy']*100:.2f}\\% & "
            f"{summary_mean['macro_f1']:.4f}\\pm{summary_std['macro_f1']:.4f} & "
            f"{summary_mean['macro_precision']:.4f}\\pm{summary_std['macro_precision']:.4f} & "
            f"{summary_mean['false_positive_rate']*100:.2f}\\pm{summary_std['false_positive_rate']*100:.2f}\\% \\\\\n"
        )
        f.write("\\hline\n\\end{tabular}\n")
    logger.info(f"Saved LaTeX table to: {tex_path}")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed empirical evaluation.")
    parser.add_argument("--data-dir", type=str, default="dataset/processed")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    data_d = (REPO_ROOT / args.data_dir).resolve()
    res_d = (REPO_ROOT / args.results_dir).resolve()

    seeds = [42, 101, 777, 1337, 2024]
    df = run_multi_seed_experiment(data_d, seeds, res_d)

    print("\n" + "=" * 65)
    print(" MULTI-SEED EXPERIMENT SUMMARY (5 SEEDS)")
    print("=" * 65)
    print(f"Accuracy  : {df['accuracy'].mean()*100:.2f}% +/- {df['accuracy'].std()*100:.2f}%")
    print(f"Macro-F1  : {df['macro_f1'].mean():.4f} +/- {df['macro_f1'].std():.4f}")
    print(f"Benign FPR: {df['false_positive_rate'].mean()*100:.2f}% +/- {df['false_positive_rate'].std()*100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
