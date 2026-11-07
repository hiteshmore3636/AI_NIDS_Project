"""AI-NIDS Model Evaluation Pipeline.

Computes comprehensive hold-out test metrics on `test.parquet`:
- Accuracy, Macro-F1, Weighted-F1, Macro-Precision, Macro-Recall
- Per-class Precision, Recall, F1, and Support
- False Positive Rate (FPR) on normal/benign traffic
- Confusion matrix (JSON & PNG heatmap visualization)
- Sub-millisecond latency profiling (mean, p50, p95, p99 ms) & throughput (flows/sec)
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_test_dataset(test_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test partition."""
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    logger.info(f"Loading held-out test data from: {test_path}")
    test_df = pd.read_parquet(test_path)
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]
    return X_test, y_test


def compute_false_positive_rate(
    y_true_str: pd.Series,
    y_pred_str: pd.Series,
    benign_label: str = "BENIGN",
) -> float:
    """Calculate the empirical False Positive Rate on benign connections."""
    benign_mask = y_true_str == benign_label
    total_benign = int(benign_mask.sum())
    if total_benign == 0:
        return 0.0

    # False Positive: true label is BENIGN, but model predicted any Attack class
    false_positives = int((benign_mask & (y_pred_str != benign_label)).sum())
    fpr = float(false_positives / total_benign)
    return fpr


def measure_inference_latency(
    model: Any,
    X_sample: pd.DataFrame,
    n_iterations: int = 500,
) -> Dict[str, float]:
    """Measure single-flow inference latency distribution and throughput."""
    # Warm up model prediction engine
    for _ in range(20):
        _ = model.predict(X_sample.iloc[[0]])

    latencies_ms: List[float] = []
    num_samples = len(X_sample)

    for i in range(n_iterations):
        sample_row = X_sample.iloc[[i % num_samples]]
        t0 = time.perf_counter()
        _ = model.predict(sample_row)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

    latencies_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies_arr))
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    throughput = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "latency_mean_ms": mean_lat,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "throughput_flows_per_sec": throughput,
    }


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    output_png: Path,
) -> None:
    """Plot high-contrast normalized confusion matrix heatmap and save to disk."""
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("AI-NIDS Multi-Class Confusion Matrix (Test Set)", fontsize=13, pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, labels, fontsize=10)

    # Print numbers in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix heatmap to: {output_png}")


def generate_results_markdown(
    metrics: Dict[str, Any],
    output_md: Path,
) -> None:
    """Generate Markdown report for documentation and research paper."""
    overall = metrics["overall_metrics"]
    latency = metrics["latency_profile"]
    per_class = metrics["per_class_metrics"]

    md_lines = [
        "# AI-NIDS Empirical Model Evaluation Report",
        "",
        f"**Evaluation Timestamp**: {metrics['timestamp']}  ",
        f"**Test Set Size**: {metrics['test_samples']:,} flows  ",
        f"**Evaluated Model**: `{metrics['model_name']}`  ",
        "",
        "## 1. Overall Performance Metrics",
        "",
        "| Metric | Target | Measured Result | Status |",
        "|---|---|---|---|",
        f"| **Overall Accuracy** | $> 98.0\\%$ | **{overall['accuracy'] * 100:.2f}%** | {'PASS' if overall['accuracy'] >= 0.98 else 'FLAG'} |",
        f"| **Macro-F1 Score** | $> 0.90$ | **{overall['macro_f1']:.4f}** | {'PASS' if overall['macro_f1'] >= 0.90 else 'FLAG'} |",
        f"| **Weighted-F1 Score** | - | **{overall['weighted_f1']:.4f}** | - |",
        f"| **Macro-Precision** | - | **{overall['macro_precision']:.4f}** | - |",
        f"| **Macro-Recall** | - | **{overall['macro_recall']:.4f}** | - |",
        f"| **False Positive Rate (FPR)** | $< 1.0\\%$ | **{overall['false_positive_rate'] * 100:.2f}%** | {'PASS' if overall['false_positive_rate'] <= 0.01 else 'FLAG'} |",
        "",
        "## 2. Real-Time Latency & Throughput Profile",
        "",
        "| Attribute | Target | Measured Result |",
        "|---|---|---|",
        f"| **Mean Latency per Flow** | $< 100\\text{{ ms}}$ | **{latency['latency_mean_ms']:.3f} ms** |",
        f"| **Median (p50) Latency** | - | **{latency['latency_p50_ms']:.3f} ms** |",
        f"| **95th Percentile (p95)** | - | **{latency['latency_p95_ms']:.3f} ms** |",
        f"| **99th Percentile (p99)** | - | **{latency['latency_p99_ms']:.3f} ms** |",
        f"| **Throughput Capacity** | $> 1,000\\text{{ flows/s}}$ | **{latency['throughput_flows_per_sec']:,.0f} flows/sec** |",
        "",
        "## 3. Per-Class Granular Breakdown",
        "",
        "| Class | Precision | Recall | F1-Score | Support |",
        "|---|---|---|---|---|",
    ]

    for cls_name, vals in per_class.items():
        md_lines.append(
            f"| `{cls_name}` | {vals['precision']:.4f} | {vals['recall']:.4f} | {vals['f1-score']:.4f} | {int(vals['support'])} |"
        )

    md_lines.extend([
        "",
        "## 4. Confusion Matrix Visualization",
        "",
        "![Confusion Matrix](results/confusion_matrix.png)",
        "",
    ])

    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Wrote model results report to: {output_md}")


def evaluate_models_pipeline(
    test_path: Path,
    model_path: Path,
    encoder_path: Path,
    results_dir: Path,
    docs_dir: Path,
) -> Dict[str, Any]:
    """Execute full evaluation pipeline on held-out test data."""
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load test data and artifacts
    X_test, y_test = load_test_dataset(test_path)
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    # 2. Run predictions
    y_test_enc = label_encoder.transform(y_test)
    y_pred_enc = model.predict(X_test)
    y_pred_str = pd.Series(label_encoder.inverse_transform(y_pred_enc), index=y_test.index)

    # 3. Overall Metrics
    acc = float(accuracy_score(y_test_enc, y_pred_enc))
    macro_f1 = float(f1_score(y_test_enc, y_pred_enc, average="macro"))
    weighted_f1 = float(f1_score(y_test_enc, y_pred_enc, average="weighted"))
    macro_prec = float(precision_score(y_test_enc, y_pred_enc, average="macro"))
    macro_rec = float(recall_score(y_test_enc, y_pred_enc, average="macro"))
    fpr = compute_false_positive_rate(y_test, y_pred_str, benign_label="BENIGN")

    # 4. Classification Report (per-class)
    report_dict = classification_report(
        y_test,
        y_pred_str,
        output_dict=True,
        zero_division=0,
    )
    # Extract only individual classes
    per_class_stats = {
        k: v for k, v in report_dict.items() if k not in ("accuracy", "macro avg", "weighted avg")
    }

    # 5. Confusion Matrix
    classes_list = label_encoder.classes_.tolist()
    cm = confusion_matrix(y_test, y_pred_str, labels=classes_list)
    cm_png = results_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(cm, classes_list, cm_png)

    # 6. Latency & Throughput Profile
    latency_profile = measure_inference_latency(model, X_test, n_iterations=300)

    # 7. Aggregate Metrics JSON
    metrics_summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": type(model).__name__,
        "test_samples": len(X_test),
        "overall_metrics": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "false_positive_rate": fpr,
        },
        "per_class_metrics": per_class_stats,
        "confusion_matrix": {
            "labels": classes_list,
            "matrix": cm.tolist(),
        },
        "latency_profile": latency_profile,
    }

    metrics_json = results_dir / "model_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved evaluation metrics to: {metrics_json}")

    # 8. Generate Documentation Report
    report_md = docs_dir / "model_results.md"
    generate_results_markdown(metrics_summary, report_md)

    return metrics_summary


def main() -> None:
    """CLI entrypoint for model evaluation."""
    parser = argparse.ArgumentParser(description="AI-NIDS Model Evaluation Pipeline.")
    parser.add_argument(
        "--test-path",
        type=str,
        default="dataset/processed/test.parquet",
        help="Path to test partition (default: dataset/processed/test.parquet)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ml_engine/models/model.joblib",
        help="Path to trained model artifact (default: ml_engine/models/model.joblib)",
    )
    parser.add_argument(
        "--encoder-path",
        type=str,
        default="ml_engine/models/label_encoder.joblib",
        help="Path to label encoder artifact (default: ml_engine/models/label_encoder.joblib)",
    )
    args = parser.parse_args()

    test_p = (REPO_ROOT / args.test_path).resolve()
    model_p = (REPO_ROOT / args.model_path).resolve()
    enc_p = (REPO_ROOT / args.encoder_path).resolve()
    res_d = (REPO_ROOT / "results").resolve()
    docs_d = (REPO_ROOT / "documentation").resolve()

    summary = evaluate_models_pipeline(
        test_path=test_p,
        model_path=model_p,
        encoder_path=enc_p,
        results_dir=res_d,
        docs_dir=docs_d,
    )

    ov = summary["overall_metrics"]
    lat = summary["latency_profile"]

    print("\n" + "=" * 65)
    print(" HELD-OUT TEST EVALUATION COMPLETED")
    print("=" * 65)
    print(f"Model Evaluated   : {summary['model_name']}")
    print(f"Accuracy          : {ov['accuracy'] * 100:.2f}%")
    print(f"Macro-F1          : {ov['macro_f1']:.4f}")
    print(f"False Pos. Rate   : {ov['false_positive_rate'] * 100:.2f}%")
    print(f"Mean Latency      : {lat['latency_mean_ms']:.3f} ms / flow")
    print(f"Throughput        : {lat['throughput_flows_per_sec']:,.0f} flows / sec")
    print("=" * 65)


if __name__ == "__main__":
    main()
