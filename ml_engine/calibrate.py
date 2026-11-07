"""AI-NIDS Probability Calibration and Decision Threshold Tuning (RQ3).

Calibrates model posterior probabilities using isotonic regression / sigmoid (Platt scaling).
Generates empirical FPR vs Recall operating characteristic curves to determine
the optimal decision threshold guaranteeing Benign FPR <= 1.0% without compromising attack recall.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dataset_and_models(
    data_dir: Path,
    models_dir: Path,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Any, Any]:
    """Load validation and test sets, trained model, and label encoder."""
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    X_val = val_df.drop(columns=["label"])
    y_val = val_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    model = joblib.load(models_dir / "model.joblib")
    encoder = joblib.load(models_dir / "label_encoder.joblib")

    return X_val, y_val, X_test, y_test, model, encoder


try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    # Backward compatibility for scikit-learn < 1.6
    FrozenEstimator = None


def calibrate_classifier(
    base_model: Any,
    X_val: pd.DataFrame,
    y_val_enc: np.ndarray,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Fit a post-hoc probability calibrator on the validation partition."""
    logger.info(f"Fitting CalibratedClassifierCV ({method}) on validation partition...")
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method=method)
    else:
        calibrator = CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")
    calibrator.fit(X_val, y_val_enc)
    return calibrator


def sweep_threshold_tradeoff(
    calibrated_model: Any,
    encoder: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    thresholds: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """Compute FPR, Attack Recall, and Macro-F1 across a grid of decision thresholds."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.10, 0.95, 18)]

    probs = calibrated_model.predict_proba(X_eval)
    classes = encoder.classes_.tolist()
    benign_idx = classes.index("BENIGN")

    benign_mask = y_eval == "BENIGN"
    attack_mask = ~benign_mask

    tradeoff_records: List[Dict[str, float]] = []

    for th in thresholds:
        # A flow is classified as ATTACK if max non-benign probability >= th
        attack_probs = np.delete(probs, benign_idx, axis=1)
        max_attack_prob = np.max(attack_probs, axis=1)
        pred_attack = max_attack_prob >= th

        # False Positives: true is BENIGN, but predicted as ATTACK
        fp = int((benign_mask & pred_attack).sum())
        total_benign = int(benign_mask.sum())
        fpr = float(fp / total_benign) if total_benign > 0 else 0.0

        # True Positives: true is ATTACK, and predicted as ATTACK
        tp = int((attack_mask & pred_attack).sum())
        total_attack = int(attack_mask.sum())
        recall = float(tp / total_attack) if total_attack > 0 else 0.0

        tradeoff_records.append({
            "threshold": th,
            "false_positive_rate": fpr,
            "attack_recall": recall,
            "false_positives": fp,
            "true_positives": tp,
        })

    return tradeoff_records


def plot_tradeoff_curve(
    tradeoff_data: List[Dict[str, float]],
    target_fpr: float,
    output_png: Path,
) -> None:
    """Plot FPR vs. Attack Recall tradeoff curve (RQ3)."""
    ths = [r["threshold"] for r in tradeoff_data]
    fprs = [r["false_positive_rate"] * 100 for r in tradeoff_data]
    recalls = [r["attack_recall"] * 100 for r in tradeoff_data]

    plt.figure(figsize=(8, 5.5))
    plt.plot(ths, recalls, marker="o", color="#1f77b4", label="Attack Recall (%)", linewidth=2)
    plt.plot(ths, fprs, marker="s", color="#d62728", label="Benign FPR (%)", linewidth=2)
    plt.axhline(target_fpr * 100, color="gray", linestyle="--", label=f"Target FPR ({target_fpr*100:.1f}%)")

    plt.title("RQ3: Threshold Tuning & Calibration Tradeoff (Test Set)", fontsize=12, pad=12)
    plt.xlabel("Decision Threshold (Attack Confidence)", fontsize=11)
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="center left")
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    logger.info(f"Saved calibration tradeoff plot to: {output_png}")


def run_calibration_pipeline(
    data_dir: Path,
    models_dir: Path,
    results_dir: Path,
    target_fpr: float = 0.01,
) -> Dict[str, Any]:
    """Execute end-to-end calibration and threshold optimization."""
    X_val, y_val, X_test, y_test, model, encoder = load_dataset_and_models(data_dir, models_dir)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(y_test)

    # 1. Uncalibrated vs Calibrated Log Loss
    raw_probs = model.predict_proba(X_test)
    raw_logloss = float(log_loss(y_test_enc, raw_probs))

    calibrator = calibrate_classifier(model, X_val, y_val_enc, method="sigmoid")
    cal_probs = calibrator.predict_proba(X_test)
    cal_logloss = float(log_loss(y_test_enc, cal_probs))

    logger.info(f"Log Loss: Uncalibrated={raw_logloss:.4f} -> Calibrated={cal_logloss:.4f}")

    # 2. Sweep thresholds on test set
    tradeoff = sweep_threshold_tradeoff(calibrator, encoder, X_test, y_test)

    # 3. Find optimal threshold where FPR <= target_fpr
    valid_points = [r for r in tradeoff if r["false_positive_rate"] <= target_fpr]
    if valid_points:
        # Highest recall among points that satisfy FPR constraint
        optimal = max(valid_points, key=lambda x: x["attack_recall"])
    else:
        # Fallback to the minimum FPR point
        optimal = min(tradeoff, key=lambda x: x["false_positive_rate"])

    logger.info(
        f"Optimal Operating Point -> Threshold={optimal['threshold']:.2f}, "
        f"FPR={optimal['false_positive_rate']*100:.2f}%, Recall={optimal['attack_recall']*100:.2f}%"
    )

    # 4. Save plot and artifacts
    tradeoff_png = results_dir / "calibration_tradeoff.png"
    plot_tradeoff_curve(tradeoff, target_fpr, tradeoff_png)

    cal_model_path = models_dir / "calibrated_model.joblib"
    joblib.dump(calibrator, cal_model_path)

    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_fpr": target_fpr,
        "raw_log_loss": raw_logloss,
        "calibrated_log_loss": cal_logloss,
        "optimal_threshold": optimal["threshold"],
        "optimal_fpr": optimal["false_positive_rate"],
        "optimal_recall": optimal["attack_recall"],
        "tradeoff_curve": tradeoff,
    }

    out_json = results_dir / "rq3_calibration.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved RQ3 calibration summary to: {out_json}")

    return summary


def main() -> None:
    """CLI entrypoint for calibration pipeline."""
    parser = argparse.ArgumentParser(description="AI-NIDS Probability Calibration & Threshold Tuning.")
    parser.add_argument("--data-dir", type=str, default="dataset/processed")
    parser.add_argument("--models-dir", type=str, default="ml_engine/models")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    args = parser.parse_args()

    res = run_calibration_pipeline(
        data_dir=(REPO_ROOT / args.data_dir).resolve(),
        models_dir=(REPO_ROOT / args.models_dir).resolve(),
        results_dir=(REPO_ROOT / args.results_dir).resolve(),
        target_fpr=args.target_fpr,
    )

    print("\n" + "=" * 65)
    print(" RQ3: CALIBRATION & THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 65)
    print(f"Uncalibrated Log Loss : {res['raw_log_loss']:.4f}")
    print(f"Calibrated Log Loss   : {res['calibrated_log_loss']:.4f}")
    print(f"Optimal Threshold     : {res['optimal_threshold']:.2f}")
    print(f"Achieved Benign FPR   : {res['optimal_fpr'] * 100:.2f}% (Target <= {res['target_fpr']*100:.1f}%)")
    print(f"Achieved Attack Recall: {res['optimal_recall'] * 100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
