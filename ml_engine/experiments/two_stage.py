"""AI-NIDS Two-Stage vs Single-Stage Architecture Experiment.

Research Question 4: Does a hierarchical pipeline (Stage 1 Binary: Benign vs Attack;
Stage 2 Multiclass: Attack Type) improve detection accuracy, false positive rate,
or inference throughput compared to a single-stage end-to-end classifier?
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from ml_engine.evaluate import compute_false_positive_rate

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TwoStageClassifier:
    """Hierarchical Two-Stage NIDS Classifier."""

    def __init__(
        self,
        binary_clf: RandomForestClassifier,
        multi_clf: RandomForestClassifier,
        multi_encoder: LabelEncoder,
    ) -> None:
        self.binary_clf = binary_clf
        self.multi_clf = multi_clf
        self.multi_encoder = multi_encoder

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict labels using two-stage decision logic."""
        # Stage 1: Binary prediction (0 = BENIGN, 1 = ATTACK)
        bin_preds = self.binary_clf.predict(X)
        final_preds = np.array(["BENIGN"] * len(X), dtype=object)

        attack_idx = np.where(bin_preds == 1)[0]
        if len(attack_idx) > 0:
            X_attack = X.iloc[attack_idx]
            multi_preds_enc = self.multi_clf.predict(X_attack)
            multi_preds_str = self.multi_encoder.inverse_transform(multi_preds_enc)
            final_preds[attack_idx] = multi_preds_str

        return final_preds


def train_two_stage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
) -> TwoStageClassifier:
    """Train binary stage 1 and multiclass stage 2 classifiers."""
    # Stage 1: Binary (0 = BENIGN, 1 = ATTACK)
    y_train_binary = (y_train != "BENIGN").astype(int)
    binary_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    binary_clf.fit(X_train, y_train_binary)

    # Stage 2: Multiclass (Trained strictly on attack samples)
    attack_mask = y_train != "BENIGN"
    X_train_attack = X_train[attack_mask]
    y_train_attack = y_train[attack_mask]

    multi_encoder = LabelEncoder()
    y_train_attack_enc = multi_encoder.fit_transform(y_train_attack)

    multi_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    multi_clf.fit(X_train_attack, y_train_attack_enc)

    return TwoStageClassifier(
        binary_clf=binary_clf,
        multi_clf=multi_clf,
        multi_encoder=multi_encoder,
    )


def measure_stage_latency(
    model: Any,
    X_sample: pd.DataFrame,
    n_iterations: int = 50,
) -> float:
    """Measure average per-sample latency in milliseconds."""
    latencies = []
    for _ in range(n_iterations):
        for idx in range(len(X_sample)):
            row = X_sample.iloc[[idx]]
            t0 = time.perf_counter()
            _ = model.predict(row)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
    return float(np.mean(latencies))


def run_two_stage_experiment(
    data_dir: Path,
    results_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare single-stage vs two-stage architecture."""
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

    # 1. Single-Stage Architecture
    logger.info("Evaluating Single-Stage Classifier...")
    single_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    single_clf.fit(X_train, y_train_enc)
    single_preds_enc = single_clf.predict(X_test)
    single_preds_str = pd.Series(encoder.inverse_transform(single_preds_enc), index=y_test.index)

    single_acc = float(accuracy_score(y_test_enc, single_preds_enc))
    single_macro_f1 = float(f1_score(y_test_enc, single_preds_enc, average="macro"))
    single_fpr = compute_false_positive_rate(y_test, single_preds_str)
    single_attack_rec = float(recall_score(y_test != "BENIGN", single_preds_str != "BENIGN"))
    single_latency = measure_stage_latency(single_clf, X_test.iloc[:10])

    records.append({
        "architecture": "Single-Stage (Direct Multiclass)",
        "accuracy": single_acc,
        "macro_f1": single_macro_f1,
        "false_positive_rate": single_fpr,
        "attack_recall": single_attack_rec,
        "latency_ms": single_latency,
    })

    # 2. Two-Stage Architecture
    logger.info("Evaluating Two-Stage (Binary -> Multiclass) Classifier...")
    two_stage_clf = train_two_stage(X_train, y_train, seed=seed)
    two_stage_preds_str = pd.Series(two_stage_clf.predict(X_test), index=y_test.index)
    two_stage_preds_enc = encoder.transform(two_stage_preds_str)

    two_acc = float(accuracy_score(y_test_enc, two_stage_preds_enc))
    two_macro_f1 = float(f1_score(y_test_enc, two_stage_preds_enc, average="macro"))
    two_fpr = compute_false_positive_rate(y_test, two_stage_preds_str)
    two_attack_rec = float(recall_score(y_test != "BENIGN", two_stage_preds_str != "BENIGN"))
    two_latency = measure_stage_latency(two_stage_clf, X_test.iloc[:10])

    records.append({
        "architecture": "Two-Stage (Binary -> Multiclass)",
        "accuracy": two_acc,
        "macro_f1": two_macro_f1,
        "false_positive_rate": two_fpr,
        "attack_recall": two_attack_rec,
        "latency_ms": two_latency,
    })

    results_df = pd.DataFrame(records)

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "two_stage_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved two-stage comparison to: {csv_path}")

    # Generate LaTeX table
    tex_dir = results_dir / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / "two_stage.tex"

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Two-stage vs Single-stage comparison\n")
        f.write("\\begin{tabular}{lccccc}\n\\hline\n")
        f.write("Architecture & Accuracy & Macro-F1 & FPR & Attack Recall & Latency (ms) \\\\\n\\hline\n")
        for _, row in results_df.iterrows():
            f.write(
                f"{row['architecture']} & {row['accuracy']*100:.2f}\\% & "
                f"{row['macro_f1']:.4f} & {row['false_positive_rate']*100:.2f}\\% & "
                f"{row['attack_recall']*100:.2f}\\% & {row['latency_ms']:.2f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n")
    logger.info(f"Saved two-stage LaTeX table to: {tex_path}")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-Stage vs Single-Stage Evaluation")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "dataset" / "processed")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_two_stage_experiment(args.data_dir, args.results_dir, seed=args.seed)
    print("\n=== Two-Stage Experiment Summary ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
