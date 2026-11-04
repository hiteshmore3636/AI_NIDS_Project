# AI-NIDS User & Operation Manual

## 1. System Requirements

* **Operating System**: Windows 10/11 (with Npcap installed) OR Ubuntu 20.04/22.04 LTS (with libpcap-dev).
* **Python Runtime**: Python 3.10+ (Recommended: Python 3.11).
* **Node.js**: Node 18+ or Node 20 LTS (for the frontend dashboard).
* **Hardware**: Minimum 8 GB RAM, Quad-Core CPU (Intel Core i5 / AMD Ryzen 5 or higher).

---

## 2. Installation & Setup

### 2.1 Clone and Environment Setup
```bash
git clone https://github.com/hiteshmore3636/AI-NIDS.git
cd AI-NIDS

# Create and activate Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 Environment Configuration
Copy `.env.example` to `.env` and configure appropriate variables:
```bash
cp .env.example .env
```

### 2.3 Windows Packet Sniffing Prerequisite
1. Download and install **Npcap** from [https://npcap.com/](https://npcap.com/).
2. During installation, check the option: **"Install Npcap in WinPcap API-compatible Mode"**.

---

## 3. Running the System

### 3.1 Backend & WebSocket Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3.2 Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
Open your browser at `http://localhost:5173`.

### 3.3 Running Flow Monitoring / PCAP Replay
```bash
# Replay a sample capture file
python -m traffic_monitor.pcap_replay --pcap data/sample.pcap

# Or run live interface capture (run terminal as Administrator on Windows)
python -m traffic_monitor.capture --interface "Ethernet"
```
