"""AI-NIDS Model Training Pipeline.

Trains baseline classifiers (Decision Tree, Random Forest, XGBoost)
under strict leakage-free protocol with balanced class weighting,
stratified 5-fold cross-validation, and comprehensive artifact serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_git_commit() -> str:
    """Retrieve current Git commit hash for provenance tracking."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return commit
    except Exception:
        return "uncommitted_development"


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for experiment reproducibility."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def load_processed_data(
    processed_dir: Path,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train and validation Parquet partitions."""
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_dir}. Run ml_engine.preprocessing first."
        )

    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_parquet(train_path)
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    logger.info(f"Loading validation data from {val_path}")
    val_df = pd.read_parquet(val_path)
    X_val = val_df.drop(columns=["label"])
    y_val = val_df["label"]

    return X_train, y_train, X_val, y_val


def get_model_instances(random_state: int = 42) -> Dict[str, Any]:
    """Instantiate baseline classification models with balanced weights."""
    models: Dict[str, Any] = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_split=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=random_state,
        ),
    }
    return models


def train_and_cross_validate(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train_enc: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Perform 5-fold stratified cross validation on the training partition only."""
    cv_results: Dict[str, Dict[str, float]] = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for name, model in models.items():
        logger.info(f"Running {n_splits}-fold Stratified CV for [{name}]...")
        t0 = time.time()
        scores = cross_val_score(
            model,
            X_train,
            y_train_enc,
            cv=skf,
            scoring="f1_macro",
            n_jobs=-1,
        )
        elapsed = time.time() - t0
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        cv_results[name] = {
            "cv_f1_macro_mean": mean_score,
            "cv_f1_macro_std": std_score,
            "cv_time_sec": elapsed,
        }
        logger.info(f"  -> {name} Macro-F1: {mean_score:.4f} (+/- {std_score:.4f}) in {elapsed:.2f}s")

    return cv_results


def train_models_pipeline(
    processed_dir: Path,
    models_dir: Path,
    results_dir: Path,
    random_state: int = 42,
    selected_model_name: str = "random_forest",
) -> Dict[str, Any]:
    """Execute end-to-end model training, CV benchmark, and artifact persistence."""
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Parquet partitions
    X_train, y_train, X_val, y_val = load_processed_data(processed_dir)
    train_hash = compute_file_sha256(processed_dir / "train.parquet")

    # 2. Fit LabelEncoder strictly on labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    # 3. Instantiate and benchmark candidate models via Stratified CV
    models = get_model_instances(random_state=random_state)
    cv_summary = train_and_cross_validate(
        models=models,
        X_train=X_train,
        y_train_enc=y_train_enc,
        n_splits=5,
        random_state=random_state,
    )

    # 4. Train the primary model on the full training set
    if selected_model_name not in models:
        raise ValueError(f"Unknown model name '{selected_model_name}'. Choose from: {list(models.keys())}")

    primary_model = models[selected_model_name]
    logger.info(f"Training primary selected model: [{selected_model_name}]...")
    t0 = time.time()
    primary_model.fit(X_train, y_train_enc)
    train_duration = time.time() - t0

    # Validation accuracy & macro F1
    val_preds = primary_model.predict(X_val)
    from sklearn.metrics import accuracy_score, f1_score
    val_acc = float(accuracy_score(y_val_enc, val_preds))
    val_macro_f1 = float(f1_score(y_val_enc, val_preds, average="macro"))
    logger.info(f"Validation performance -> Accuracy: {val_acc:.4f}, Macro-F1: {val_macro_f1:.4f}")

    # 5. Persist artifacts
    model_path = models_dir / "model.joblib"
    encoder_path = models_dir / "label_encoder.joblib"

    joblib.dump(primary_model, model_path)
    joblib.dump(label_encoder, encoder_path)
    logger.info(f"Saved primary model to: {model_path}")
    logger.info(f"Saved label encoder to: {encoder_path}")

    # 6. Save detailed metadata and CV benchmark
    git_commit = get_git_commit()
    metadata: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit,
        "random_seed": random_state,
        "dataset_hash": train_hash,
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "features": list(X_train.columns),
        "classes": label_encoder.classes_.tolist(),
        "primary_model": {
            "name": selected_model_name,
            "params": {k: str(v) for k, v in primary_model.get_params().items()},
            "fit_time_sec": train_duration,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
        },
        "model_comparison_cv": cv_summary,
    }

    metadata_path = models_dir / "training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote training metadata to: {metadata_path}")

    return metadata


def main() -> None:
    """CLI entrypoint for model training."""
    parser = argparse.ArgumentParser(description="AI-NIDS Baseline Model Training Pipeline.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/processed",
        help="Path to processed parquet directory (default: dataset/processed)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="ml_engine/models",
        help="Directory to persist model artifacts (default: ml_engine/models)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["decision_tree", "random_forest", "xgboost"],
        help="Primary model architecture to train (default: random_forest)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    proc_d = (REPO_ROOT / args.data_dir).resolve()
    models_d = (REPO_ROOT / args.models_dir).resolve()
    results_d = (REPO_ROOT / "results").resolve()

    summary = train_models_pipeline(
        processed_dir=proc_d,
        models_dir=models_d,
        results_dir=results_d,
        random_state=args.seed,
        selected_model_name=args.model,
    )

    print("\n" + "=" * 65)
    print(" MODEL TRAINING AND CROSS-VALIDATION COMPLETED")
    print("=" * 65)
    print(f"Primary Model     : {summary['primary_model']['name']}")
    print(f"Val Accuracy      : {summary['primary_model']['val_accuracy'] * 100:.2f}%")
    print(f"Val Macro-F1      : {summary['primary_model']['val_macro_f1']:.4f}")
    print("\n5-Fold Stratified CV Comparison (Macro-F1):")
    for m_name, stats in summary["model_comparison_cv"].items():
        print(f"  - {m_name:15s}: {stats['cv_f1_macro_mean']:.4f} (+/- {stats['cv_f1_macro_std']:.4f})")
    print("=" * 65)


if __name__ == "__main__":
    main()
