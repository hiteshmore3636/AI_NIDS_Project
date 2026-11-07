"""AI-NIDS Class Imbalance Handling Experiment.

Compares class weighting, random undersampling, and unweighted baselines
to assess recall preservation on minority attack classes (Botnet, WebAttack).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder

from ml_engine.evaluate import compute_false_positive_rate

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_imbalance_experiment(
    data_dir: Path,
    results_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare imbalance handling strategies on training set only."""
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

    # 1. Strategy A: No Weighting (Baseline)
    logger.info("Evaluating Strategy: No Imbalance Mitigation...")
    clf_none = RandomForestClassifier(n_estimators=100, class_weight=None, random_state=seed, n_jobs=-1)
    clf_none.fit(X_train, y_train_enc)
    preds_none = clf_none.predict(X_test)
    preds_none_str = pd.Series(encoder.inverse_transform(preds_none), index=y_test.index)

    rec_bot_none = recall_score(y_test == "Botnet", preds_none_str == "Botnet", zero_division=0)
    rec_web_none = recall_score(y_test == "WebAttack", preds_none_str == "WebAttack", zero_division=0)

    records.append({
        "strategy": "No Weighting",
        "accuracy": float(accuracy_score(y_test_enc, preds_none)),
        "macro_f1": float(f1_score(y_test_enc, preds_none, average="macro")),
        "botnet_recall": float(rec_bot_none),
        "webattack_recall": float(rec_web_none),
        "false_positive_rate": compute_false_positive_rate(y_test, preds_none_str),
    })

    # 2. Strategy B: Cost-Sensitive Balanced Class Weights
    logger.info("Evaluating Strategy: Balanced Class Weights...")
    clf_bal = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)
    clf_bal.fit(X_train, y_train_enc)
    preds_bal = clf_bal.predict(X_test)
    preds_bal_str = pd.Series(encoder.inverse_transform(preds_bal), index=y_test.index)

    rec_bot_bal = recall_score(y_test == "Botnet", preds_bal_str == "Botnet", zero_division=0)
    rec_web_bal = recall_score(y_test == "WebAttack", preds_bal_str == "WebAttack", zero_division=0)

    records.append({
        "strategy": "Balanced Weights",
        "accuracy": float(accuracy_score(y_test_enc, preds_bal)),
        "macro_f1": float(f1_score(y_test_enc, preds_bal, average="macro")),
        "botnet_recall": float(rec_bot_bal),
        "webattack_recall": float(rec_web_bal),
        "false_positive_rate": compute_false_positive_rate(y_test, preds_bal_str),
    })

    # 3. Strategy C: Random Undersampling of Majority Benign class on Train
    logger.info("Evaluating Strategy: Controlled Undersampling on Train...")
    train_benign = train_df[train_df["label"] == "BENIGN"]
    train_attack = train_df[train_df["label"] != "BENIGN"]
    # Downsample benign to 1.5x total attacks
    n_sample = min(len(train_benign), int(len(train_attack) * 1.5))
    benign_down = train_benign.sample(n=n_sample, random_state=seed)
    train_down = pd.concat([benign_down, train_attack]).sample(frac=1.0, random_state=seed)

    X_down = train_down.drop(columns=["label"])
    y_down_enc = encoder.transform(train_down["label"])

    clf_down = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf_down.fit(X_down, y_down_enc)
    preds_down = clf_down.predict(X_test)
    preds_down_str = pd.Series(encoder.inverse_transform(preds_down), index=y_test.index)

    rec_bot_down = recall_score(y_test == "Botnet", preds_down_str == "Botnet", zero_division=0)
    rec_web_down = recall_score(y_test == "WebAttack", preds_down_str == "WebAttack", zero_division=0)

    records.append({
        "strategy": "Controlled Undersampling",
        "accuracy": float(accuracy_score(y_test_enc, preds_down)),
        "macro_f1": float(f1_score(y_test_enc, preds_down, average="macro")),
        "botnet_recall": float(rec_bot_down),
        "webattack_recall": float(rec_web_down),
        "false_positive_rate": compute_false_positive_rate(y_test, preds_down_str),
    })

    results_df = pd.DataFrame(records)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "imbalance_comparison.csv"
    results_df.to_csv(out_csv, index=False)
    logger.info(f"Saved imbalance comparison to: {out_csv}")

    # Generate LaTeX table
    tex_dir = results_dir / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / "imbalance_comparison.tex"

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Class imbalance strategy comparison\n")
        f.write("\\begin{tabular}{lccccc}\n\\hline\n")
        f.write("Strategy & Accuracy & Macro-F1 & Botnet Recall & WebAttack Recall & FPR \\\\\n\\hline\n")
        for _, row in results_df.iterrows():
            f.write(
                f"{row['strategy']} & {row['accuracy']*100:.2f}\\% & {row['macro_f1']:.4f} & "
                f"{row['botnet_recall']*100:.2f}\\% & {row['webattack_recall']*100:.2f}\\% & "
                f"{row['false_positive_rate']*100:.2f}\\% \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n")
    logger.info(f"Saved LaTeX table to: {tex_path}")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Class imbalance evaluation.")
    parser.add_argument("--data-dir", type=str, default="dataset/processed")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    data_d = (REPO_ROOT / args.data_dir).resolve()
    res_d = (REPO_ROOT / args.results_dir).resolve()

    df = run_imbalance_experiment(data_d, res_d)

    print("\n" + "=" * 65)
    print(" CLASS IMBALANCE MITIGATION COMPARISON")
    print("=" * 65)
    for _, row in df.iterrows():
        print(
            f"{row['strategy']:23s} | Macro-F1: {row['macro_f1']:.4f} | "
            f"Botnet: {row['botnet_recall']*100:.1f}% | FPR: {row['false_positive_rate']*100:.2f}%"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
