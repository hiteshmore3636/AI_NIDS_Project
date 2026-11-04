# Literature Review: Network Intrusion Detection Systems (NIDS)

## 1. Introduction & Taxonomy
Network Intrusion Detection Systems (NIDS) inspect network traffic to identify unauthorized access, policy violations, and malicious operations.

### 1.1 IDS vs. IPS
* **Intrusion Detection System (IDS)**: Operates passively out-of-band via port mirroring (SPAN) or TAP devices. It analyzes traffic, flags malicious patterns, logs incidents, and sends real-time alerts without impeding line-rate transmission.
* **Intrusion Prevention System (IPS)**: Placed inline within the network path. It inspects packets in real-time and actively drops, resets, or blocks malicious connections. However, inline blocking introduces latency and risks blocking legitimate business traffic upon false positives.

### 1.2 Signature-Based vs. Anomaly-Based Detection
* **Signature-Based (e.g., Snort, Suricata)**: Relies on deterministic rule-matching against known byte sequences or patterns. While achieving near-zero false alarms on documented threats, it fails against zero-day exploits, polymorphism, and obfuscation.
* **Anomaly-Based**: Learns statistical or structural baselines of benign traffic. Deviations beyond a threshold trigger alerts. This detects novel attacks, but traditionally suffered from higher false-positive rates (FPR).
* **Hybrid Approach (This Work)**: Combines supervised multi-class classifiers for high-confidence known attack identification with unsupervised anomaly detection (e.g., Autoencoders, Isolation Forests) for out-of-distribution traffic.

### 1.3 Packet-Level vs. Flow-Level Inspection
* **Deep Packet Inspection (DPI)**: Examines packet headers and payloads. Heavily impaired by widespread modern transport layer encryption (TLS 1.3, HTTPS, QUIC).
* **Flow-Based Analysis**: Aggregates packets sharing a common 5-tuple (`src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`) within an active/idle time window. Features capture behavioral timing, packet sizes, inter-arrival times (IAT), and TCP flags, remaining robust even with encrypted payloads.

---

## 2. Benchmark Datasets Review
1. **KDD Cup 99 & NSL-KDD**: Historically significant but outdated; redundant records and unrealistic modern traffic patterns.
2. **UNSW-NB15**: Synthesized modern attacks with contemporary normal traffic; 49 features.
3. **CIC-IDS2017**: Produced by the Canadian Institute for Cybersecurity. Features 80 flow metrics across 5 days, simulating benign traffic alongside diverse modern attack vectors (DoS, DDoS, Web Attacks, Infiltration, Botnets, Brute Force).
