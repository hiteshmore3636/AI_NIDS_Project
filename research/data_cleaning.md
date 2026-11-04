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
