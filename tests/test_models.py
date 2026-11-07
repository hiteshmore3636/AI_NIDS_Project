"""AI-NIDS Model Verification Unit Tests.

Enforces:
1. Model Artifact Integrity: trained classifier and LabelEncoder load properly.
2. Prediction Sanity & Determinism: probabilities sum to 1.0, repeatable with seed.
3. Metric Computation Sanity: FPR calculation and latency profiling validity.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from ml_engine.evaluate import compute_false_positive_rate, measure_inference_latency

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_model_and_encoder_artifacts_exist():
    """Verify that model.joblib and label_encoder.joblib exist and load correctly."""
    model_path = REPO_ROOT / "ml_engine" / "models" / "model.joblib"
    encoder_path = REPO_ROOT / "ml_engine" / "models" / "label_encoder.joblib"
    metadata_path = REPO_ROOT / "ml_engine" / "models" / "training_metadata.json"

    assert model_path.exists(), "Trained model.joblib not found in ml_engine/models/"
    assert encoder_path.exists(), "label_encoder.joblib not found in ml_engine/models/"
    assert metadata_path.exists(), "training_metadata.json not found in ml_engine/models/"

    model = joblib.load(model_path)
    encoder: LabelEncoder = joblib.load(encoder_path)

    assert hasattr(model, "predict"), "Loaded model does not have a predict method"
    assert hasattr(model, "predict_proba"), "Loaded model does not have predict_proba method"
    assert len(encoder.classes_) >= 5, "Expected at least 5 classes in LabelEncoder"
    assert "BENIGN" in encoder.classes_, "BENIGN class must be present in encoder"


def test_model_prediction_probabilities():
    """Verify that predicted probabilities sum to 1.0 and match class count."""
    model_path = REPO_ROOT / "ml_engine" / "models" / "model.joblib"
    encoder_path = REPO_ROOT / "ml_engine" / "models" / "label_encoder.joblib"
    test_path = REPO_ROOT / "dataset" / "processed" / "test.parquet"

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    test_df = pd.read_parquet(test_path)
    X_sample = test_df.drop(columns=["label"]).iloc[:10]

    probs = model.predict_proba(X_sample)

    assert probs.shape == (10, len(encoder.classes_))
    # Every row probability vector must sum to 1.0
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities must sum to 1.0"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities must be bounded in [0, 1]"


def test_determinism_with_fixed_seed():
    """Verify that training with a fixed seed produces identical predictions."""
    X = np.random.RandomState(42).randn(50, 5)
    y = np.random.RandomState(42).choice([0, 1, 2], size=50)

    rf1 = RandomForestClassifier(n_estimators=10, random_state=42)
    rf1.fit(X, y)
    preds1 = rf1.predict(X)

    rf2 = RandomForestClassifier(n_estimators=10, random_state=42)
    rf2.fit(X, y)
    preds2 = rf2.predict(X)

    np.testing.assert_array_equal(preds1, preds2, err_msg="Model training must be deterministic with seed")


def test_compute_false_positive_rate():
    """Verify FPR calculation logic with synthetic edge cases."""
    # Case 1: Zero false positives
    y_true_1 = pd.Series(["BENIGN", "BENIGN", "DoS", "DDoS"])
    y_pred_1 = pd.Series(["BENIGN", "BENIGN", "DoS", "DoS"])
    assert compute_false_positive_rate(y_true_1, y_pred_1, benign_label="BENIGN") == 0.0

    # Case 2: 1 false positive out of 2 benigns -> FPR = 50%
    y_true_2 = pd.Series(["BENIGN", "BENIGN", "DoS"])
    y_pred_2 = pd.Series(["BENIGN", "DoS", "DoS"])
    assert compute_false_positive_rate(y_true_2, y_pred_2, benign_label="BENIGN") == 0.5


def test_measure_inference_latency():
    """Verify that latency profiler returns valid positive timings."""
    model_path = REPO_ROOT / "ml_engine" / "models" / "model.joblib"
    test_path = REPO_ROOT / "dataset" / "processed" / "test.parquet"

    model = joblib.load(model_path)
    test_df = pd.read_parquet(test_path)
    X_sample = test_df.drop(columns=["label"]).iloc[:20]

    profile = measure_inference_latency(model, X_sample, n_iterations=30)

    assert profile["latency_mean_ms"] > 0.0
    assert profile["latency_p50_ms"] > 0.0
    assert profile["latency_p95_ms"] >= profile["latency_p50_ms"]
    assert profile["latency_p99_ms"] >= profile["latency_p95_ms"]
    assert profile["throughput_flows_per_sec"] > 0.0
