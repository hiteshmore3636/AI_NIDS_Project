# AI-NIDS: AI-Based Real-Time Network Intrusion Detection System

An enterprise-grade, research-backed Network Intrusion Detection System (NIDS) combining supervised multi-class attack classification, unsupervised anomaly detection, SHAP explainability, and a sub-100ms real-time monitoring dashboard.

---

## 1. Project Overview

Modern network environments encounter continuous, multi-vector attacks including DoS/DDoS, brute-force credential stuffing, botnets, and web application exploits. Signature-based NIDS (e.g., Snort, Suricata) struggle against novel or encrypted threats and require continuous manual rule updates.

**AI-NIDS** addresses these challenges via:
* **Bidirectional Flow Assembly**: Reconstructs packet streams into transport-layer bidirectional flows keyed by 5-tuple.
* **Dual-Track ML Pipeline**:
  * **Supervised Multi-Class Classifier**: Random Forest, XGBoost, and LightGBM for known attack taxonomy (`DoS`, `DDoS`, `PortScan`, `BruteForce`, `WebAttack`, `Botnet`).
  * **Unsupervised Anomaly Detector**: Isolation Forest / Autoencoder for zero-day and out-of-distribution traffic.
* **Explainable AI (XAI)**: Fast TreeExplainer SHAP attribution per detection.
* **Real-Time Streaming Engine**: Sub-100ms end-to-end processing pipeline pushing events to a FastAPI backend and interactive React dashboard via WebSockets.

---

## 2. Ethics & Responsible Use Statement

> **IMPORTANT ETHICAL & LEGAL NOTICE**:
> This software is developed strictly for defensive monitoring, academic research, and security education.
>
> 1. All testing and evaluations are conducted using authorized public datasets (CIC-IDS2017, UNSW-NB15) or isolated virtual labs.
> 2. This project does **NOT** contain, deploy, or automate offensive attack capabilities against third-party systems.
> 3. Packet sniffing must only be run on networks and interfaces where you have explicit administrative authorization.

---

## 3. Repository Structure

```
AI-NIDS/
├── configs/            # YAML configs: data, models, capture parameters
├── dataset/            # Raw & processed datasets (gitignored)
├── research/           # Literature review, research gaps, cleaning protocols, bibtex
├── ml_engine/          # Feature schema, preprocessing, training, calibration, SHAP
├── traffic_monitor/    # Live sniffer, flow builder, PCAP replay, feature mapper
├── backend/            # FastAPI server, SQLAlchemy DB models, auth, WebSockets
├── frontend/           # React + Vite + Tailwind + Recharts monitoring UI
├── results/            # Empirical experiment outputs (JSON, CSV, plots)
├── tests/              # Comprehensive pytest test suite
├── documentation/      # SRS, architecture, test reports, user manuals
├── scripts/            # Admin seeding, data download guides, benchmarks
└── requirements.txt    # Pinned Python package dependencies
```

---

## 4. Git Branching Strategy

This project follows a structured GitFlow branching model:
* `main`: Production-ready, verified releases.
* `develop`: Integration branch for tested features.
* `feature/<phase-or-module>`: Isolated work branches for specific milestones.

---

## 5. Quickstart & Installation

### Prerequisites
* **Python**: 3.10+ (Recommended: 3.11)
* **Node.js**: 18+ or 20 LTS
* **Packet Capture Driver**:
  * **Windows**: Install [Npcap](https://npcap.com/) in *WinPcap API-compatible Mode*.
  * **Linux (Ubuntu)**: `sudo apt-get install libpcap-dev build-essential`

### Setup Instructions
```bash
# 1. Clone repository
git clone https://github.com/hiteshmore3636/AI_NIDS_Project.git
cd AI_NIDS_Project

# 2. Virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux:
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env

# 5. Run test suite
pytest -v tests/
```

---

## 6. Remote Git Repository Setup

To link your local repository to your GitHub remote, run:
```bash
git remote add origin https://github.com/hiteshmore3636/AI_NIDS_Project.git
git push -u origin main
```
