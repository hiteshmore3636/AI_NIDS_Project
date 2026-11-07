"""AI-NIDS Unsupervised Anomaly Detection Module.

Trains an Isolation Forest anomaly detector on BENIGN network flows only.
Tunes the anomaly threshold on the validation split so normal FPR <= 1.0%,
allowing the system to detect novel zero-day or out-of-distribution attacks.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_partitions_for_anomaly(
    data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load train benign partition and full validation partition."""
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Parquet files not found in {data_dir}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Filter training data to BENIGN only (Unsupervised novelty detection principle)
    train_benign = train_df[train_df["label"] == "BENIGN"].drop(columns=["label"])
    X_val = val_df.drop(columns=["label"])
    y_val = val_df["label"]

    logger.info(f"Loaded {len(train_benign):,} BENIGN training flows for anomaly modeling.")
    logger.info(f"Loaded {len(X_val):,} validation flows for threshold tuning.")
    return train_benign, X_val, y_val


def compute_normalized_anomaly_scores(
    raw_scores: np.ndarray,
    min_score: float,
    max_score: float,
) -> np.ndarray:
    """Normalize raw decision_function scores into [0, 1] range.

    Higher normalized score -> more anomalous.
    """
    denom = max(max_score - min_score, 1e-6)
    # raw_score is lower for anomalies, higher for inliers
    norm = 1.0 - (raw_scores - min_score) / denom
    return np.clip(norm, 0.0, 1.0)


def train_anomaly_model(
    train_benign: pd.DataFrame,
    contamination: float = 0.01,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Tuple[IsolationForest, float, float]:
    """Train Isolation Forest on benign flows and establish score range."""
    logger.info("Training Isolation Forest on benign baseline flows...")
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(train_benign)

    train_raw = iso.decision_function(train_benign)
    min_score = float(np.min(train_raw))
    max_score = float(np.max(train_raw))
    return iso, min_score, max_score


def tune_anomaly_threshold(
    model: IsolationForest,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_score: float,
    max_score: float,
    target_benign_fpr: float = 0.01,
) -> Tuple[float, Dict[str, float]]:
    """Determine the optimal anomaly threshold on validation benign flows.

    Enforces target FPR <= 1.0% on normal traffic while measuring attack recall.
    """
    val_raw = model.decision_function(X_val)
    val_scores = compute_normalized_anomaly_scores(val_raw, min_score, max_score)

    benign_mask = y_val == "BENIGN"
    attack_mask = ~benign_mask

    benign_scores = val_scores[benign_mask]
    attack_scores = val_scores[attack_mask]

    # Pick threshold at the (1 - target_benign_fpr) percentile of benign scores
    # e.g., for 1% FPR, threshold is 99th percentile of benign validation scores
    percentile = (1.0 - target_benign_fpr) * 100.0
    threshold = float(np.percentile(benign_scores, percentile))

    # Evaluate validation metrics at this threshold
    benign_fp = int((benign_scores > threshold).sum())
    actual_fpr = float(benign_fp / len(benign_scores)) if len(benign_scores) > 0 else 0.0

    attack_detected = int((attack_scores > threshold).sum())
    attack_recall = float(attack_detected / len(attack_scores)) if len(attack_scores) > 0 else 0.0

    metrics = {
        "chosen_threshold": threshold,
        "target_fpr": target_benign_fpr,
        "empirical_val_fpr": actual_fpr,
        "val_attack_recall": attack_recall,
        "benign_val_count": int(len(benign_scores)),
        "attack_val_count": int(len(attack_scores)),
    }
    logger.info(
        f"Tuned Anomaly Threshold = {threshold:.4f} -> Empirical Benign FPR: {actual_fpr * 100:.2f}%, "
        f"Attack Anomaly Recall: {attack_recall * 100:.2f}%"
    )
    return threshold, metrics


def run_anomaly_pipeline(
    data_dir: Path,
    models_dir: Path,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end anomaly training, tuning, and artifact serialization."""
    models_dir.mkdir(parents=True, exist_ok=True)

    train_benign, X_val, y_val = load_partitions_for_anomaly(data_dir)
    iso, min_s, max_s = train_anomaly_model(train_benign, random_state=random_state)
    threshold, tuning_metrics = tune_anomaly_threshold(
        model=iso,
        X_val=X_val,
        y_val=y_val,
        min_score=min_s,
        max_score=max_s,
        target_benign_fpr=0.01,
    )

    # Persist model artifact
    model_path = models_dir / "anomaly_model.joblib"
    joblib.dump(iso, model_path)
    logger.info(f"Saved anomaly model to: {model_path}")

    # Persist metadata
    metadata: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_type": "IsolationForest",
        "random_state": random_state,
        "min_score": min_s,
        "max_score": max_s,
        "threshold": threshold,
        "tuning_metrics": tuning_metrics,
        "features": list(train_benign.columns),
    }

    meta_path = models_dir / "anomaly_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved anomaly metadata to: {meta_path}")

    return metadata


def main() -> None:
    """CLI entrypoint for anomaly detection training."""
    parser = argparse.ArgumentParser(description="AI-NIDS Anomaly Detection Pipeline.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/processed",
        help="Path to processed parquet directory",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="ml_engine/models",
        help="Path to models artifact directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    data_d = (REPO_ROOT / args.data_dir).resolve()
    models_d = (REPO_ROOT / args.models_dir).resolve()

    summary = run_anomaly_pipeline(data_d, models_d, random_state=args.seed)
    tm = summary["tuning_metrics"]

    print("\n" + "=" * 65)
    print(" ANOMALY DETECTOR TRAINING & THRESHOLD TUNING COMPLETED")
    print("=" * 65)
    print(f"Algorithm           : {summary['model_type']}")
    print(f"Optimal Threshold   : {summary['threshold']:.4f}")
    print(f"Empirical Benign FPR: {tm['empirical_val_fpr'] * 100:.2f}% (Target <= {tm['target_fpr'] * 100:.1f}%)")
    print(f"Attack Recall       : {tm['val_attack_recall'] * 100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
