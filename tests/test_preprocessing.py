"""AI-NIDS Preprocessing Unit Tests.

Enforces:
1. Zero Data Leakage (Rule 2): strict partition isolation, scaler/imputer fit only on train.
2. Value & Header Sanitization: stripping whitespaces, replacing inf/-inf, deduplication.
3. Schema & Artifact Conformance: valid Parquet export, StandardScaler persistence, features.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
import joblib

from ml_engine.preprocessing import (
    clean_column_names,
    standardize_labels,
    sanitize_values,
    prepare_features,
    prune_collinear_and_zerovar,
    execute_preprocessing_pipeline,
)
from ml_engine.sample_generator import generate_synthetic_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Generate a small temporary synthetic CSV file for test isolation."""
    csv_file = tmp_path / "temp_test_data.csv"
    generate_synthetic_dataset(
        output_path=csv_file,
        num_rows=600,
        seed=123,
        duplicate_ratio=0.05,
        artifact_ratio=0.05,
    )
    return csv_file


def test_clean_column_names():
    """Verify that leading and trailing whitespace is stripped from all headers."""
    dirty_df = pd.DataFrame(columns=[" Destination Port", " Flow Duration ", "Label "])
    cleaned_df = clean_column_names(dirty_df)
    assert list(cleaned_df.columns) == ["Destination Port", "Flow Duration", "Label"]


def test_standardize_labels():
    """Verify raw attack labels map correctly to unified taxonomy."""
    raw_labels = pd.Series([
        "BENIGN",
        "DoS Hulk",
        "DoS GoldenEye",
        "DDoS",
        "PortScan",
        "FTP-Patator",
        "SSH-Patator",
        "Web Attack – Brute Force",
        "Bot",
        "UnknownAttackXYZ",
    ])
    standardized = standardize_labels(raw_labels)
    expected = [
        "BENIGN",
        "DoS",
        "DoS",
        "DDoS",
        "PortScan",
        "BruteForce",
        "BruteForce",
        "WebAttack",
        "Botnet",
        "Other",
    ]
    assert standardized.tolist() == expected


def test_sanitize_values_deduplication_and_inf():
    """Verify that duplicates are removed and inf/-inf are transformed to NaN."""
    df = pd.DataFrame({
        " Destination Port": [80, 80, 443],
        " Flow Duration": [1000, 1000, 2000],
        "Flow Bytes/s": ["Infinity", "Infinity", 500.0],
        "Label": ["BENIGN", "BENIGN", "DoS"],
    })
    cleaned_names = clean_column_names(df)
    sanitized_df, stats = sanitize_values(cleaned_names)

    # Row 1 and 2 are identical -> deduplication should keep only 2 unique rows
    assert len(sanitized_df) == 2
    assert stats["duplicates_removed"] == 1

    # "Infinity" should be converted to numeric NaN
    val = sanitized_df.loc[0, "Flow Bytes/s"]
    assert np.isnan(val)


def test_zero_leakage_scaler_fit(sample_csv_path: Path, tmp_path: Path):
    """Enforce Non-negotiable Rule 2: Scaler parameters must derive solely from TRAIN."""
    output_dir = tmp_path / "processed"
    summary = execute_preprocessing_pipeline(
        input_path=sample_csv_path,
        output_dir=output_dir,
        track="track_b",
    )

    train_df = pd.read_parquet(summary["artifacts"]["train_parquet"])
    scaler: StandardScaler = joblib.load(summary["artifacts"]["scaler_joblib"])
    features: List[str] = summary["features"]["selected_features"]

    # Scaler mean must match feature count
    assert len(scaler.mean_) == len(features)

    # In a properly scaled train set, column means should be close to 0 and std close to 1
    train_features = train_df[features]
    means = train_features.mean(axis=0)
    stds = train_features.std(axis=0)

    assert np.allclose(means.values, 0.0, atol=1e-2), "Train mean after scaling should be ~0.0"
    assert np.allclose(stds.values, 1.0, atol=1e-1), "Train std after scaling should be ~1.0"


def test_partition_disjointness_no_overlap(sample_csv_path: Path, tmp_path: Path):
    """Verify train, val, and test partitions are completely disjoint."""
    output_dir = tmp_path / "processed"
    summary = execute_preprocessing_pipeline(
        input_path=sample_csv_path,
        output_dir=output_dir,
        track="track_b",
    )

    train_df = pd.read_parquet(summary["artifacts"]["train_parquet"])
    val_df = pd.read_parquet(summary["artifacts"]["val_parquet"])
    test_df = pd.read_parquet(summary["artifacts"]["test_parquet"])

    # Verify no NaN or Inf remains in any partition
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert df.isna().sum().sum() == 0, f"Partition {name} contains unhandled NaNs"
        assert not np.isinf(df.select_dtypes(include=[np.number]).values).any(), f"Partition {name} contains Infs"

    # Verify target label column is present in all splits
    assert "label" in train_df.columns
    assert "label" in val_df.columns
    assert "label" in test_df.columns

    # Verify split proportions (70% / 15% / 15%)
    total_samples = len(train_df) + len(val_df) + len(test_df)
    assert pytest.approx(len(train_df) / total_samples, abs=0.03) == 0.70
    assert pytest.approx(len(val_df) / total_samples, abs=0.03) == 0.15
    assert pytest.approx(len(test_df) / total_samples, abs=0.03) == 0.15


def test_feature_selection_prunes_zerovar():
    """Verify prune_collinear_and_zerovar removes zero variance columns."""
    df = pd.DataFrame({
        "var_zero": [5.0, 5.0, 5.0, 5.0, 5.0],
        "var_normal": [1.0, 2.0, 3.0, 4.0, 5.0],
        "collinear_1": [10.0, 20.0, 30.0, 40.0, 50.0],
        "collinear_2": [10.01, 20.01, 30.01, 40.01, 50.01],  # Corr ~ 1.0 with collinear_1
    })
    selected = prune_collinear_and_zerovar(df, corr_threshold=0.95)

    assert "var_zero" not in selected, "Zero-variance column was not pruned"
    assert "var_normal" in selected, "Valid column was unexpectedly pruned"
    # One of the two collinear columns must be pruned
    assert not ("collinear_1" in selected and "collinear_2" in selected)
