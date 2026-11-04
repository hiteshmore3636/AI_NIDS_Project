# System Architecture

## 1. High-Level Pipeline Architecture

The AI-NIDS architecture follows a multi-stage streaming pipeline designed for sub-100ms end-to-end processing latency:

```
  ┌────────────────────────────────────────────────────────┐
  │                 Traffic Acquisition                    │
  │     Live NIC (Scapy / Npcap)  OR  PCAP Replay          │
  └───────────────────────────┬────────────────────────────┘
                              │ Raw Packets
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │            Flow Assembly & Feature Engine              │
  │   - Bidirectional 5-Tuple Keying                       │
  │   - Active Timeout (120s) / Idle Timeout (5s)          │
  │   - Standardized Schema Extraction (feature_schema)    │
  └───────────────────────────┬────────────────────────────┘
                              │ Completed Flow Records
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │              Inference & Decision Engine               │
  │   - Stage 1: Preprocessing & Scaling (StandardScaler)  │
  │   - Stage 2: Supervised Multi-Class (XGBoost/RF)       │
  │   - Stage 3: Anomaly Detection (Isolation Forest)      │
  │   - Stage 4: SHAP Explainer (Top-5 Features)           │
  │   - Stage 5: Hybrid Decision & Alert Deduplication     │
  └───────────────────────────┬────────────────────────────┘
                              │ Classified Events & Alerts
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │                     FastAPI Backend                    │
  │   - REST API (Auth, Filters, History, Model Stats)     │
  │   - SQLite / PostgreSQL Database Persistence           │
  │   - Alert Manager (Email / Telegram Notifications)     │
  │   - WebSocket Live Broadcast                           │
  └───────────────────────────┬────────────────────────────┘
                              │ WebSocket Stream
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │                 React + Vite Dashboard                 │
  │   - Real-time Threat Monitor & Gauge                   │
  │   - Attack Breakdown & Temporal Analytics              │
  │   - Alerts Table with SHAP Contribution Drawer         │
  └────────────────────────────────────────────────────────┘
```

---

## 2. Real-Time Data Flow Sequence

1. **Packet Capture**: Packets are intercepted from the wire or read from a PCAP stream.
2. **Flow Tracking**: Packets update an in-memory bidirectional flow table keyed by `(min(src,dst), max(src,dst), min(sport,dport), max(sport,dport), proto)`.
3. **Flow Termination**: A flow is dispatched when a TCP FIN/RST is observed, when idle duration exceeds 5 seconds, or when active duration exceeds 120 seconds.
4. **Feature Extraction**: Statistical features are computed to strictly match `feature_schema.json`.
5. **Inference**:
   * Preprocessor aligns and scales features using the offline fitted scaler.
   * Supervised model computes class probabilities $[P_{\text{benign}}, P_{\text{DoS}}, \dots]$.
   * Unsupervised model computes anomaly score $S_{\text{anomaly}} \in [0, 1]$.
   * TreeExplainer computes SHAP attributions for top-5 influential features.
6. **Decision & Alert Engine**:
   * If supervised $P_{\text{attack}} \ge 0.80$, an alert is raised with the predicted attack label.
   * If classified as `BENIGN` but $S_{\text{anomaly}} > \theta_{\text{anomaly}}$, a `Suspicious Anomaly` (Medium) alert is generated.
   * If both supervised and anomaly engines trigger, severity is boosted (e.g., High $\to$ Critical).
   * Identical alerts from the same source within 10 seconds are deduplicated.
7. **Storage & Broadcast**: The flow event and alert are written to the database; the WebSocket manager broadcasts the payload to all connected clients.
