# Software Requirements Specification (SRS)

## AI-Based Real-Time Network Intrusion Detection System (AI-NIDS)

---

## 1. Functional Requirements

| ID | Requirement Name | Description |
|---|---|---|
| **FR1** | Traffic Acquisition | The system shall capture live packets from a designated network adapter or replay standardized PCAP/PCAPNG capture files. |
| **FR2** | Bidirectional Flow Assembly | The system shall aggregate packets into bidirectional flows keyed by 5-tuple (`src_ip`, `dst_ip`, `src_port`, `dst_port`, `proto`) with configurable timeouts (active 120s, idle 5s). |
| **FR3** | Standardized Feature Extraction | The system shall compute statistical flow features strictly conforming to `feature_schema.json` upon flow closure. |
| **FR4** | Multi-Class Supervised Classification | The system shall classify each flow into `BENIGN` or a specific attack category (`DoS`, `DDoS`, `PortScan`, `BruteForce`, `WebAttack`, `Botnet`) with an associated confidence score. |
| **FR5** | Unsupervised Anomaly Detection | The system shall compute an anomaly score on completed flows to detect zero-day or out-of-distribution traffic. |
| **FR6** | Severity Assignment & Decision Logic | The decision engine shall combine supervised and anomaly outputs, applying severity classification (`Low`, `Medium`, `High`, `Critical`) and deduplication within a 10s window. |
| **FR7** | Persistent Event Storage | The system shall persist flow records and alert events with full metadata, timestamps, IPs, ports, and SHAP top features in SQLite/PostgreSQL. |
| **FR8** | Real-Time Dashboard Broadcasting | The system shall broadcast live detection events over WebSocket to a connected browser dashboard within 100 ms of flow completion. |
| **FR9** | Automated Incident Notifications | The alert manager shall trigger SMTP email and Telegram bot alerts for `High` and `Critical` events with rate-limiting. |
| **FR10**| Administration & Export | The system shall provide secure JWT-based admin access, alert acknowledgement endpoints, and audit-ready CSV/PDF export capabilities. |

---

## 2. Non-Functional Requirements

| Category | Metric / Target | Verification Method |
|---|---|---|
| **Latency** | End-to-end latency $< 100\text{ ms}$ per flow (flow termination $\to$ dashboard push) | Automated timestamp tracing |
| **Throughput** | $\ge 1,000\text{ flows/sec}$ on standard x86 quad-core hardware with 8 GB RAM | Benchmark replay scripts |
| **Detection Quality**| Overall Accuracy $> 98\%$, Macro-F1 $> 0.90$, False Positive Rate (FPR) $< 1.0\%$ | Held-out test set & cross-validation |
| **Explainability** | Top-5 SHAP feature contributions computed for every flagged alert | TreeExplainer unit test |
| **Security** | JWT authentication (HS256), bcrypt password hashing, input validation via Pydantic | Security unit tests |
| **Reliability** | Decoupled capture, inference, and API processes with queue isolation and auto-restart | Stress testing with simulated drops |
| **Portability** | Runs natively on Windows 10/11 (Npcap) and Linux (Ubuntu 22.04 LTS / libpcap) | Multi-OS CI pipelines |
