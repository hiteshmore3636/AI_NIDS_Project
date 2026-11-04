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
| **Phase 0** | `tests/test_environment.py` | Configs, schemas, environment imports | Initialized | Pending |
| **Phase 1** | `tests/test_preprocessing.py`| Data cleaning, zero-leakage partitions | Pending | - |
| **Phase 2** | `tests/test_models.py` | Supervised baseline models, schema checks| Pending | - |
| **Phase 3** | `tests/test_experiments.py` | Anomaly baselines, calibration, ablation | Pending | - |
| **Phase 4** | `tests/test_explainability.py`| SHAP attribution consistency | Pending | - |
| **Phase 5** | `tests/test_pipeline.py` | Flow extractor, feature parity, latency | Pending | - |
| **Phase 6** | `tests/test_api.py` | FastAPI endpoints, auth, WebSockets | Pending | - |
| **Phase 8** | `tests/test_integration.py` | End-to-end PCAP replay to dashboard | Pending | - |
