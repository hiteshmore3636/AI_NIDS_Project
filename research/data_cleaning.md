# Data Cleaning & Preprocessing Protocol

## 1. Known Dataset Anomalies in CIC-IDS2017
The original CIC-IDS2017 `MachineLearningCVE` CSV files contain documented artifacts that require systematic remediation:

1. **Header Whitespace**: Column headers contain leading and trailing whitespace (e.g., `' Destination Port'`, `' Flow Duration'`).
2. **Infinite and Missing Values**: Rate features such as `Flow Bytes/s` and `Flow Packets/s` produce division-by-zero artifacts resulting in `Infinity`, `-Infinity`, and `NaN`.
3. **Duplicate Flows**: Re-transmitted packets or multi-interface logs introduce identical duplicate rows.
4. **Label Inconsistencies**: Typos in labels across days (e.g., `Web Attack  Brute Force`, `Web Attack  XSS`, `Web Attack  Sql Injection`).
5. **Session Leakage Columns**: Flow ID, Source IP, Destination IP, and Timestamp must not be exposed to the model as features to prevent learning specific subnet topology.

---

## 2. Standardized Cleaning Pipeline

```
Raw CSV Files (8 Files)
         │
         ▼
[1. Strip Header Whitespaces]
         │
         ▼
[2. Remove Exact Duplicate Records]
         │
         ▼
[3. Map & Standardize Attack Labels]
    (e.g., DoS Slowloris / Hulk -> DoS; Brute Force FTP/SSH -> BruteForce)
         │
         ▼
[4. Sanitize Numeric Columns: inf/-inf -> NaN -> median imputation or drop]
         │
         ▼
[5. Drop Identifiers: Flow ID, Src IP, Dst IP, Timestamp]
         │
         ▼
[6. Remove Zero-Variance & Multicollinear Features (>0.95 Pearson)]
         │
         ▼
[7. Leakage-Free Partitioning (Train: 70%, Val: 15%, Test: 15%)]
         │
         ▼
[8. Fit Scaler & Encoders on Train ONLY -> Transform Val & Test]
         │
         ▼
Save Processed Parquet Files & Scaler Artifact
```

---

## 3. Label Mapping Standardization

| Raw Label | Standardized Class | Description |
|---|---|---|
| `BENIGN` | `BENIGN` | Normal legitimate traffic |
| `DoS Hulk`, `DoS GoldenEye`, `DoS slowloris`, `DoS Slowhttptest` | `DoS` | Denial of Service attacks |
| `DDoS` | `DDoS` | Distributed Denial of Service (LOIC) |
| `PortScan` | `PortScan` | Horizontal and vertical port probes |
| `FTP-Patator`, `SSH-Patator` | `BruteForce` | Authentication credential brute-forcing |
| `Web Attack – Brute Force`, `Web Attack – XSS`, `Web Attack – Sql Injection` | `WebAttack` | Application-layer web exploit attempts |
| `Bot` | `Botnet` | ARES botnet command and control traffic |
| `Heartbleed`, `Infiltration` | `Other` (or isolated evaluation) | Low-sample / edge exploit attempts |


---

## 4. Empirical Cleaning & Preprocessing Results

The following metrics reflect the empirical execution of `ml_engine/preprocessing.py` on the benchmark dataset sample (saved in `results/cleaning_summary.json`):

### 4.1 Data Volume & Sanitization
* **Initial Records Ingested**: 5,000
* **Exact Duplicate Rows Removed**: 100
* **Cleaned Unique Records**: 4,900
* **Infinite / Division-by-Zero Values Sanitized**: 58
* **Missing Value Imputation**: Median computed strictly on the training partition (`X_train`) to avoid data leakage.

### 4.2 Stratified Partitioning (70% / 15% / 15%)
* **Training Partition**: 3,430 rows (70.0%)
* **Validation Partition**: 735 rows (15.0%)
* **Testing Partition**: 735 rows (15.0%)

### 4.3 Class Distributions Across Partitions
| Class | Train Count (Ratio) | Val Count (Ratio) | Test Count (Ratio) |
|---|---|---|---|
| `BENIGN` | 2,199 (64.11%) | 471 (64.08%) | 471 (64.08%) |
| `DoS` | 513 (14.96%) | 110 (14.97%) | 110 (14.97%) |
| `DDoS` | 254 (7.41%) | 54 (7.35%) | 55 (7.48%) |
| `PortScan` | 218 (6.36%) | 47 (6.39%) | 46 (6.26%) |
| `BruteForce` | 139 (4.05%) | 30 (4.08%) | 30 (4.08%) |
| `WebAttack` | 70 (2.04%) | 15 (2.04%) | 15 (2.04%) |
| `Botnet` | 37 (1.08%) | 8 (1.09%) | 8 (1.09%) |
| **Total** | **3,430 (100%)** | **735 (100%)** | **735 (100%)** |

### 4.4 Feature Selection & Pruning (Evaluated on Train Only)
* **Initial Feature Set**: 29 features
* **Zero-Variance Features Removed**: `tcp_urg_count`, `protocol` (2 features)
* **High Multicollinearity ($r > 0.95$) Features Removed**: `pkt_len_std`, `iat_max_ms`, `fwd_iat_total_ms` (3 features)
* **Final Selected Features**: 24 features
* **Normalization**: `StandardScaler` fitted exclusively on `X_train` ($\mu \approx 0, \sigma \approx 1$), serialized to `ml_engine/models/scaler.joblib`.
