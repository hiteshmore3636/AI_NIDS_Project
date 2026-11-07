# AI-NIDS Empirical Model Evaluation Report

**Evaluation Timestamp**: 2026-11-07T18:45:20Z  
**Test Set Size**: 735 flows  
**Evaluated Model**: `RandomForestClassifier`  

## 1. Overall Performance Metrics

| Metric | Target | Measured Result | Status |
|---|---|---|---|
| **Overall Accuracy** | $> 98.0\%$ | **98.37%** | PASS |
| **Macro-F1 Score** | $> 0.90$ | **0.9531** | PASS |
| **Weighted-F1 Score** | - | **0.9838** | - |
| **Macro-Precision** | - | **0.9481** | - |
| **Macro-Recall** | - | **0.9584** | - |
| **False Positive Rate (FPR)** | $< 1.0\%$ | **1.49%** | FLAG |

## 2. Real-Time Latency & Throughput Profile

| Attribute | Target | Measured Result |
|---|---|---|
| **Mean Latency per Flow** | $< 100\text{ ms}$ | **38.283 ms** |
| **Median (p50) Latency** | - | **35.893 ms** |
| **95th Percentile (p95)** | - | **49.535 ms** |
| **99th Percentile (p99)** | - | **64.092 ms** |
| **Throughput Capacity** | $> 1,000\text{ flows/s}$ | **26 flows/sec** |

## 3. Per-Class Granular Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `BENIGN` | 0.9978 | 0.9851 | 0.9915 | 471 |
| `Botnet` | 0.8750 | 0.8750 | 0.8750 | 8 |
| `BruteForce` | 0.9677 | 1.0000 | 0.9836 | 30 |
| `DDoS` | 0.9821 | 1.0000 | 0.9910 | 55 |
| `DoS` | 0.9474 | 0.9818 | 0.9643 | 110 |
| `PortScan` | 1.0000 | 1.0000 | 1.0000 | 46 |
| `WebAttack` | 0.8667 | 0.8667 | 0.8667 | 15 |

## 4. Confusion Matrix Visualization

![Confusion Matrix](results/confusion_matrix.png)
