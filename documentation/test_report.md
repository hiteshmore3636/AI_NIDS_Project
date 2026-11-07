# AI-NIDS Test & Verification Report

## 1. Test Methodology
The project enforces test-driven development and empirical evaluation across four distinct verification levels:

1. **Unit Testing (`pytest`)**:
   * Preprocessing sanity, zero-leakage splits, column mapping.
   * Flow feature schema parity against `feature_schema.json`.
   * Security constraints: JWT generation/validation, password hashing, SQL injection prevention.
2. **Model Evaluation & Empirical Verification**:
   * Multi-seed evaluation ($\ge 5$ seeds) reporting mean and standard deviation.
   * Hold-out evaluation metrics: Accuracy, Precision, Recall, Macro-F1, Weighted-F1, FPR, ROC-AUC, PR-AUC.
   * Cross-dataset generalization metrics between CIC-IDS2017 and UNSW-NB15.
3. **Pipeline & Integration Testing**:
   * Replay of standardized attack PCAPs through the capture engine to the database and WebSocket broadcast.
   * End-to-end latency measurements recorded in `results/latency.json`.
4. **Lab Attack Simulation**:
   * Isolated internal testing with standard security auditing tools (e.g., Nmap SYN scans, Slowloris, Hydra brute-force) in a sandboxed virtual lab environment.

---

## 2. Test Execution Log

| Phase | Test Suite | Target Component | Status | Last Run Date |
|---|---|---|---|---|
| **Phase 0** | `tests/test_environment.py` | Configs, schemas, environment imports | PASSED (4/4) | 2026-11-05 |
| **Phase 1** | `tests/test_preprocessing.py`| Data cleaning, zero-leakage partitions | PASSED (6/6) | 2026-11-07 |
| **Phase 2** | `tests/test_models.py` | Supervised baseline models, schema checks| PASSED (5/5) | 2026-11-08 |
| **Phase 3** | `tests/test_experiments.py` | Anomaly baselines, calibration, ablation | PASSED (5/5) | 2026-11-08 |

---

## 3. Phase 3 Empirical Experiment Findings

### RQ1: Multi-Seed Statistical Stability (5 Randomized Seeds)
* **Objective**: Measure reproducibility across seeds `[42, 101, 777, 1337, 2024]`.
* **Empirical Results**:
  * Accuracy: $98.34\% \pm 0.20\%$
  * Macro-F1: $0.9564 \pm 0.0076$
  * Benign False Positive Rate (FPR): $1.61\% \pm 0.12\%$
* **Artifacts**: `results/multi_seed.csv`, `results/tables/multi_seed.tex`.

### RQ2: Class Imbalance Mitigation Strategies
* **Objective**: Compare baseline unweighted vs cost-sensitive balanced weights vs training-only controlled undersampling.
* **Empirical Results**:
  * *No Weighting*: Macro-F1: $0.8790$, Botnet Recall: $62.5\%$, FPR: $1.06\%$
  * *Balanced Class Weights*: Macro-F1: $\mathbf{0.9469}$, Botnet Recall: $\mathbf{87.5\%}$, FPR: $1.49\%$
  * *Controlled Undersampling*: Macro-F1: $0.8890$, Botnet Recall: $75.0\%$, FPR: $1.06\%$
* **Conclusion**: Balanced class weighting dramatically improves recall on low-volume attack classes with minimal degradation in false alarms.
* **Artifacts**: `results/imbalance_comparison.csv`, `results/tables/imbalance_comparison.tex`.

### RQ3: Probability Calibration & Low-FPR Threshold Tuning
* **Objective**: Post-hoc probability calibration (Sigmoid) on the validation set to minimize overconfident misclassifications and tune threshold to $\le 1.0\%$ FPR.
* **Empirical Results**:
  * Uncalibrated Log Loss: $0.0754$ $\rightarrow$ Calibrated Log Loss: $\mathbf{0.0503}$
  * Optimal Decision Threshold: $\tau = 0.55$
  * Tuned Benign FPR: $\mathbf{0.85\%}$ (meets strict $< 1.0\%$ engineering constraint)
  * Attack Recall at $\tau = 0.55$: $\mathbf{98.11\%}$
* **Artifacts**: `ml_engine/models/calibrated_model.joblib`, `results/calibration_tradeoff.png`, `results/rq3_calibration.json`.

### RQ4: Architecture - Two-Stage (Binary $\rightarrow$ Multiclass) vs Single-Stage
* **Objective**: Test if separating benign filtering from attack classification improves throughput or detection fidelity.
* **Empirical Results**:
  * *Single-Stage (Direct Multiclass)*: Accuracy: $\mathbf{98.10\%}$, Macro-F1: $\mathbf{0.9429}$, FPR: $\mathbf{1.70\%}$, Attack Recall: $\mathbf{99.62\%}$, Latency: $\mathbf{36.86\text{ ms}}$
  * *Two-Stage (Binary $\rightarrow$ Multiclass)*: Accuracy: $97.14\%$, Macro-F1: $0.8885$, FPR: $1.91\%$, Attack Recall: $97.35\%$, Latency: $48.77\text{ ms}$
* **Conclusion**: Single-stage direct multiclass outperforms two-stage in both accuracy and latency because errors in Stage 1 cascade into Stage 2.
* **Artifacts**: `results/two_stage_comparison.csv`, `results/tables/two_stage.tex`.

### RQ5: Feature Ablation Study
* **Objective**: Assess model performance across distinct feature functional groupings.
* **Empirical Results**:
  * *All 24 Features*: Macro-F1: $0.9469$, Latency: $49.49\text{ ms}$
  * *Top-10 Features*: Macro-F1: $\mathbf{0.9540}$, Latency: $\mathbf{41.02\text{ ms}}$
  * *Packet Lengths Only (8 feats)*: Macro-F1: $0.8725$, Latency: $52.11\text{ ms}$
  * *IAT & Rates Only (7 feats)*: Macro-F1: $0.7972$, Latency: $45.21\text{ ms}$
  * *Flags & Windows Only (8 feats)*: Macro-F1: $0.6124$, Latency: $47.79\text{ ms}$
* **Artifacts**: `results/ablation_study.csv`, `results/tables/ablation_study.tex`.

### Anomaly Detection Layer (Zero-Day Behaviour)
* **Objective**: Train Isolation Forest strictly on benign training flows to flag deviations without attack signatures.
* **Empirical Results**:
  * Tuned threshold: $0.7749$
  * Validation Benign FPR: $1.06\%$
  * Zero-Day Attack Anomaly Recall: $64.39\%$
* **Artifacts**: `ml_engine/models/anomaly_model.joblib`, `ml_engine/models/anomaly_metadata.json`.
| **Phase 4** | `tests/test_explainability.py`| SHAP attribution consistency | Pending | - |
| **Phase 5** | `tests/test_pipeline.py` | Flow extractor, feature parity, latency | Pending | - |
| **Phase 6** | `tests/test_api.py` | FastAPI endpoints, auth, WebSockets | Pending | - |
| **Phase 8** | `tests/test_integration.py` | End-to-end PCAP replay to dashboard | Pending | - |
