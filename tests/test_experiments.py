"""Unit and Integration Tests for Phase 3: Research Experiments & Anomaly Detection."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "ml_engine" / "models"
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "dataset" / "processed"


def test_anomaly_model_artifacts_exist():
    """Verify serialized anomaly model and metadata files exist."""
    model_path = MODELS_DIR / "anomaly_model.joblib"
    meta_path = MODELS_DIR / "anomaly_metadata.json"

    assert model_path.exists(), "anomaly_model.joblib missing"
    assert meta_path.exists(), "anomaly_metadata.json missing"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "threshold" in meta
    assert "tuning_metrics" in meta
    assert "empirical_val_fpr" in meta["tuning_metrics"]
    assert "val_attack_recall" in meta["tuning_metrics"]
    assert meta["tuning_metrics"]["empirical_val_fpr"] <= 0.05, f"Anomaly FPR too high: {meta['tuning_metrics']['empirical_val_fpr']}"


def test_anomaly_inference_scoring():
    """Verify anomaly detector scores test set and produces expected ranges."""
    model = joblib.load(MODELS_DIR / "anomaly_model.joblib")
    with open(MODELS_DIR / "anomaly_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    X_test = test_df.drop(columns=["label"])
    sample = X_test.iloc[:20]

    # Raw anomaly score: IsolationForest decision_function
    scores = model.decision_function(sample)
    assert len(scores) == 20
    assert not np.isnan(scores).any()

    # Inverted score [0, 1]
    norm_scores = 0.5 - (scores / 2.0)
    assert ((norm_scores >= -1.0) & (norm_scores <= 2.0)).all()


def test_calibrated_model_artifacts_and_inference():
    """Verify calibrated classifier exists and provides valid calibrated probabilities."""
    cal_path = MODELS_DIR / "calibrated_model.joblib"
    rq3_path = RESULTS_DIR / "rq3_calibration.json"

    assert cal_path.exists(), "calibrated_model.joblib missing"
    assert rq3_path.exists(), "rq3_calibration.json missing"

    with open(rq3_path, "r", encoding="utf-8") as f:
        rq3 = json.load(f)

    assert "raw_log_loss" in rq3
    assert "calibrated_log_loss" in rq3
    assert rq3["calibrated_log_loss"] <= rq3["raw_log_loss"]

    cal_model = joblib.load(cal_path)
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    X_test = test_df.drop(columns=["label"]).iloc[:15]

    probs = cal_model.predict_proba(X_test)
    assert probs.shape[0] == 15
    # Verify rows sum to 1.0 within floating point precision
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_research_experiment_csv_results():
    """Verify all 4 research experiment CSV result files exist and contain non-empty data."""
    csv_files = [
        RESULTS_DIR / "multi_seed.csv",
        RESULTS_DIR / "imbalance_comparison.csv",
        RESULTS_DIR / "ablation_study.csv",
        RESULTS_DIR / "two_stage_comparison.csv",
    ]

    for p in csv_files:
        assert p.exists(), f"Experiment result file {p.name} missing"
        df = pd.read_csv(p)
        assert not df.empty, f"Experiment result {p.name} is empty"


def test_research_latex_tables_exist():
    """Verify LaTeX tables exist in results/tables/ for academic paper inclusion."""
    tex_dir = RESULTS_DIR / "tables"
    expected_tables = [
        "multi_seed.tex",
        "imbalance_comparison.tex",
        "ablation_study.tex",
        "two_stage.tex",
    ]

    for tbl in expected_tables:
        tex_path = tex_dir / tbl
        assert tex_path.exists(), f"LaTeX table {tbl} missing"
        content = tex_path.read_text(encoding="utf-8")
        assert "\\begin{tabular}" in content
        assert "\\end{tabular}" in content
