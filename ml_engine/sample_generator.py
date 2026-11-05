"""Synthetic CIC-IDS2017 Dataset Generator.

Generates schema-compliant raw CSV data closely simulating the Canadian
Institute for Cybersecurity (CIC-IDS2017) MachineLearningCVE files.
Enables offline pipeline validation, automated testing, and CI runs
without requiring the 3+ GB raw dataset download upfront.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import math
from pathlib import Path
import random
from typing import Any, Dict, List

# Exact column headers as found in the raw CIC-IDS2017 CSV files (preserving whitespace)
CICIDS2017_RAW_HEADERS: List[str] = [
    " Destination Port",
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    " Fwd Packet Length Max",
    " Fwd Packet Length Min",
    " Fwd Packet Length Mean",
    " Fwd Packet Length Std",
    "Bwd Packet Length Max",
    " Bwd Packet Length Min",
    " Bwd Packet Length Mean",
    " Bwd Packet Length Std",
    "Flow Bytes/s",
    " Flow Packets/s",
    " Flow IAT Mean",
    " Flow IAT Std",
    " Flow IAT Max",
    " Flow IAT Min",
    "Fwd IAT Total",
    " Fwd IAT Mean",
    " Fwd IAT Std",
    " Fwd IAT Max",
    " Fwd IAT Min",
    "Bwd IAT Total",
    " Bwd IAT Mean",
    " Bwd IAT Std",
    " Bwd IAT Max",
    " Bwd IAT Min",
    "Fwd PSH Flags",
    " Bwd PSH Flags",
    " Fwd URG Flags",
    " Bwd URG Flags",
    " Fwd Header Length",
    " Bwd Header Length",
    "Fwd Packets/s",
    " Bwd Packets/s",
    " Min Packet Length",
    " Max Packet Length",
    " Packet Length Mean",
    " Packet Length Std",
    " Packet Length Variance",
    "FIN Flag Count",
    " SYN Flag Count",
    " RST Flag Count",
    " PSH Flag Count",
    " ACK Flag Count",
    " URG Flag Count",
    " CWE Flag Count",
    " ECE Flag Count",
    " Down/Up Ratio",
    " Average Packet Size",
    " Avg Fwd Segment Size",
    " Avg Bwd Segment Size",
    " Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    " Fwd Avg Packets/Bulk",
    " Fwd Avg Bulk Rate",
    " Bwd Avg Bytes/Bulk",
    " Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    " Subflow Fwd Bytes",
    " Subflow Bwd Packets",
    " Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    " Init_Win_bytes_backward",
    " act_data_pkt_fwd",
    " min_seg_size_forward",
    "Active Mean",
    " Active Std",
    " Active Max",
    " Active Min",
    "Idle Mean",
    " Idle Std",
    " Idle Max",
    " Idle Min",
    " Label",
]

# Attack profiles and realistic proportions in synthetic data
ATTACK_PROFILES: Dict[str, Dict[str, Any]] = {
    "BENIGN": {
        "weight": 0.65,
        "ports": [80, 443, 53, 8080, 25, 110, 445],
        "duration_range": (5000, 10000000),  # microseconds
        "fwd_pkts_range": (1, 60),
        "bwd_pkts_range": (1, 80),
        "fwd_len_mean": (40.0, 600.0),
        "bwd_len_mean": (100.0, 1400.0),
        "syn_prob": 0.15,
        "fin_prob": 0.85,
        "rst_prob": 0.05,
    },
    "DoS Hulk": {
        "weight": 0.08,
        "ports": [80, 443, 8080],
        "duration_range": (500, 500000),
        "fwd_pkts_range": (10, 150),
        "bwd_pkts_range": (0, 10),
        "fwd_len_mean": (200.0, 800.0),
        "bwd_len_mean": (0.0, 100.0),
        "syn_prob": 0.80,
        "fin_prob": 0.10,
        "rst_prob": 0.30,
    },
    "DoS GoldenEye": {
        "weight": 0.04,
        "ports": [80, 443],
        "duration_range": (1000, 800000),
        "fwd_pkts_range": (8, 90),
        "bwd_pkts_range": (1, 15),
        "fwd_len_mean": (150.0, 600.0),
        "bwd_len_mean": (20.0, 200.0),
        "syn_prob": 0.70,
        "fin_prob": 0.20,
        "rst_prob": 0.20,
    },
    "DoS slowloris": {
        "weight": 0.03,
        "ports": [80, 8080],
        "duration_range": (30000000, 120000000),  # Very long duration
        "fwd_pkts_range": (3, 15),
        "bwd_pkts_range": (0, 3),
        "fwd_len_mean": (20.0, 100.0),
        "bwd_len_mean": (0.0, 50.0),
        "syn_prob": 0.30,
        "fin_prob": 0.05,
        "rst_prob": 0.05,
    },
    "DDoS": {
        "weight": 0.07,
        "ports": [80, 443, 53],
        "duration_range": (100, 100000),  # Very short, flood
        "fwd_pkts_range": (20, 200),
        "bwd_pkts_range": (0, 5),
        "fwd_len_mean": (40.0, 200.0),
        "bwd_len_mean": (0.0, 40.0),
        "syn_prob": 0.95,
        "fin_prob": 0.02,
        "rst_prob": 0.10,
    },
    "PortScan": {
        "weight": 0.06,
        "ports": list(range(20, 1024, 23)),  # Probing wide range of ports
        "duration_range": (10, 50000),
        "fwd_pkts_range": (1, 4),
        "bwd_pkts_range": (0, 2),
        "fwd_len_mean": (0.0, 60.0),
        "bwd_len_mean": (0.0, 40.0),
        "syn_prob": 0.90,
        "fin_prob": 0.01,
        "rst_prob": 0.75,
    },
    "FTP-Patator": {
        "weight": 0.02,
        "ports": [21],
        "duration_range": (5000, 200000),
        "fwd_pkts_range": (5, 25),
        "bwd_pkts_range": (4, 20),
        "fwd_len_mean": (40.0, 150.0),
        "bwd_len_mean": (50.0, 200.0),
        "syn_prob": 0.20,
        "fin_prob": 0.90,
        "rst_prob": 0.10,
    },
    "SSH-Patator": {
        "weight": 0.02,
        "ports": [22],
        "duration_range": (10000, 500000),
        "fwd_pkts_range": (8, 40),
        "bwd_pkts_range": (6, 35),
        "fwd_len_mean": (80.0, 300.0),
        "bwd_len_mean": (80.0, 400.0),
        "syn_prob": 0.20,
        "fin_prob": 0.90,
        "rst_prob": 0.10,
    },
    "Web Attack – Brute Force": {
        "weight": 0.02,
        "ports": [80, 443, 8080],
        "duration_range": (20000, 1000000),
        "fwd_pkts_range": (6, 30),
        "bwd_pkts_range": (4, 25),
        "fwd_len_mean": (150.0, 800.0),
        "bwd_len_mean": (100.0, 1200.0),
        "syn_prob": 0.15,
        "fin_prob": 0.80,
        "rst_prob": 0.10,
    },
    "Bot": {
        "weight": 0.01,
        "ports": [8080, 6667, 4444],
        "duration_range": (100000, 5000000),
        "fwd_pkts_range": (4, 20),
        "bwd_pkts_range": (3, 15),
        "fwd_len_mean": (60.0, 350.0),
        "bwd_len_mean": (50.0, 300.0),
        "syn_prob": 0.25,
        "fin_prob": 0.70,
        "rst_prob": 0.15,
    },
}


def generate_flow_record(
    label: str,
    profile: Dict[str, Any],
    inject_nan_or_inf: bool = False,
) -> Dict[str, Any]:
    """Generate a single realistic network flow record matching CIC-IDS2017 schema."""
    dst_port = random.choice(profile["ports"])
    duration = random.randint(*profile["duration_range"])
    fwd_pkts = random.randint(*profile["fwd_pkts_range"])
    bwd_pkts = random.randint(*profile["bwd_pkts_range"])
    total_pkts = fwd_pkts + bwd_pkts

    fwd_mean = random.uniform(*profile["fwd_len_mean"])
    bwd_mean = random.uniform(*profile["bwd_len_mean"]) if bwd_pkts > 0 else 0.0

    tot_fwd_len = int(fwd_pkts * fwd_mean)
    tot_bwd_len = int(bwd_pkts * bwd_mean)
    tot_len = tot_fwd_len + tot_bwd_len

    # Packet length stats
    fwd_min = max(0.0, fwd_mean - random.uniform(0, fwd_mean * 0.5))
    fwd_max = fwd_mean + random.uniform(0, fwd_mean * 1.5)
    fwd_std = (fwd_max - fwd_min) / 4.0 if fwd_pkts > 1 else 0.0

    bwd_min = max(0.0, bwd_mean - random.uniform(0, bwd_mean * 0.5)) if bwd_pkts > 0 else 0.0
    bwd_max = bwd_mean + random.uniform(0, bwd_mean * 1.5) if bwd_pkts > 0 else 0.0
    bwd_std = (bwd_max - bwd_min) / 4.0 if bwd_pkts > 1 else 0.0

    min_len = min(fwd_min, bwd_min if bwd_pkts > 0 else fwd_min)
    max_len = max(fwd_max, bwd_max)
    pkt_mean = tot_len / total_pkts if total_pkts > 0 else 0.0
    pkt_std = (max_len - min_len) / 4.0 if total_pkts > 1 else 0.0
    pkt_var = pkt_std**2

    # Flow rates
    dur_sec = max(duration / 1e6, 1e-6)
    if inject_nan_or_inf:
        # Deliberately inject edge cases found in raw CIC-IDS2017 for preprocessing tests
        choice = random.choice(["inf", "-inf", "nan"])
        if choice == "inf":
            flow_bytes_s = "Infinity"
            flow_pkts_s = "Infinity"
        elif choice == "-inf":
            flow_bytes_s = "-Infinity"
            flow_pkts_s = "-Infinity"
        else:
            flow_bytes_s = "NaN"
            flow_pkts_s = "NaN"
    else:
        flow_bytes_s = round(tot_len / dur_sec, 4)
        flow_pkts_s = round(total_pkts / dur_sec, 4)

    # Inter-arrival times
    iat_mean = duration / (total_pkts - 1) if total_pkts > 1 else duration
    iat_std = iat_mean * random.uniform(0.1, 0.8)
    iat_max = iat_mean * random.uniform(1.2, 3.0)
    iat_min = max(0.0, iat_mean * random.uniform(0.0, 0.5))

    fwd_iat_total = duration * random.uniform(0.7, 1.0)
    fwd_iat_mean = fwd_iat_total / (fwd_pkts - 1) if fwd_pkts > 1 else fwd_iat_total
    fwd_iat_std = fwd_iat_mean * random.uniform(0.1, 0.6)
    fwd_iat_max = fwd_iat_mean * random.uniform(1.2, 2.5)
    fwd_iat_min = max(0.0, fwd_iat_mean * random.uniform(0.0, 0.4))

    bwd_iat_total = duration * random.uniform(0.5, 0.9) if bwd_pkts > 1 else 0.0
    bwd_iat_mean = bwd_iat_total / (bwd_pkts - 1) if bwd_pkts > 1 else bwd_iat_total
    bwd_iat_std = bwd_iat_mean * random.uniform(0.1, 0.6)
    bwd_iat_max = bwd_iat_mean * random.uniform(1.2, 2.5)
    bwd_iat_min = max(0.0, bwd_iat_mean * random.uniform(0.0, 0.4))

    # Flags
    syn = 1 if random.random() < profile["syn_prob"] else 0
    fin = 1 if random.random() < profile["fin_prob"] else 0
    rst = 1 if random.random() < profile["rst_prob"] else 0
    psh = 1 if random.random() < 0.3 else 0
    ack = 1 if total_pkts > 2 else 0
    urg = 0

    init_win_fwd = random.choice([8192, 14600, 29200, 65535])
    init_win_bwd = random.choice([0, 8192, 14600, 29200, 65535]) if bwd_pkts > 0 else -1

    header_len_fwd = fwd_pkts * 20
    header_len_bwd = bwd_pkts * 20

    active_mean = random.uniform(100.0, 5000.0) if duration > 50000 else 0.0
    idle_mean = random.uniform(5000.0, 30000.0) if duration > 100000 else 0.0

    return {
        " Destination Port": dst_port,
        " Flow Duration": duration,
        " Total Fwd Packets": fwd_pkts,
        " Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": tot_fwd_len,
        " Total Length of Bwd Packets": tot_bwd_len,
        " Fwd Packet Length Max": round(fwd_max, 2),
        " Fwd Packet Length Min": round(fwd_min, 2),
        " Fwd Packet Length Mean": round(fwd_mean, 2),
        " Fwd Packet Length Std": round(fwd_std, 2),
        "Bwd Packet Length Max": round(bwd_max, 2),
        " Bwd Packet Length Min": round(bwd_min, 2),
        " Bwd Packet Length Mean": round(bwd_mean, 2),
        " Bwd Packet Length Std": round(bwd_std, 2),
        "Flow Bytes/s": flow_bytes_s,
        " Flow Packets/s": flow_pkts_s,
        " Flow IAT Mean": round(iat_mean, 2),
        " Flow IAT Std": round(iat_std, 2),
        " Flow IAT Max": round(iat_max, 2),
        " Flow IAT Min": round(iat_min, 2),
        "Fwd IAT Total": round(fwd_iat_total, 2),
        " Fwd IAT Mean": round(fwd_iat_mean, 2),
        " Fwd IAT Std": round(fwd_iat_std, 2),
        " Fwd IAT Max": round(fwd_iat_max, 2),
        " Fwd IAT Min": round(fwd_iat_min, 2),
        "Bwd IAT Total": round(bwd_iat_total, 2),
        " Bwd IAT Mean": round(bwd_iat_mean, 2),
        " Bwd IAT Std": round(bwd_iat_std, 2),
        " Bwd IAT Max": round(bwd_iat_max, 2),
        " Bwd IAT Min": round(bwd_iat_min, 2),
        "Fwd PSH Flags": psh,
        " Bwd PSH Flags": 0,
        " Fwd URG Flags": 0,
        " Bwd URG Flags": 0,
        " Fwd Header Length": header_len_fwd,
        " Bwd Header Length": header_len_bwd,
        "Fwd Packets/s": round(fwd_pkts / dur_sec, 4),
        " Bwd Packets/s": round(bwd_pkts / dur_sec, 4),
        " Min Packet Length": round(min_len, 2),
        " Max Packet Length": round(max_len, 2),
        " Packet Length Mean": round(pkt_mean, 2),
        " Packet Length Std": round(pkt_std, 2),
        " Packet Length Variance": round(pkt_var, 2),
        "FIN Flag Count": fin,
        " SYN Flag Count": syn,
        " RST Flag Count": rst,
        " PSH Flag Count": psh,
        " ACK Flag Count": ack,
        " URG Flag Count": urg,
        " CWE Flag Count": 0,
        " ECE Flag Count": 0,
        " Down/Up Ratio": round(bwd_pkts / fwd_pkts, 2) if fwd_pkts > 0 else 0,
        " Average Packet Size": round(pkt_mean, 2),
        " Avg Fwd Segment Size": round(fwd_mean, 2),
        " Avg Bwd Segment Size": round(bwd_mean, 2),
        " Fwd Header Length.1": header_len_fwd,
        "Fwd Avg Bytes/Bulk": 0,
        " Fwd Avg Packets/Bulk": 0,
        " Fwd Avg Bulk Rate": 0,
        " Bwd Avg Bytes/Bulk": 0,
        " Bwd Avg Packets/Bulk": 0,
        "Bwd Avg Bulk Rate": 0,
        "Subflow Fwd Packets": fwd_pkts,
        " Subflow Fwd Bytes": tot_fwd_len,
        " Subflow Bwd Packets": bwd_pkts,
        " Subflow Bwd Bytes": tot_bwd_len,
        "Init_Win_bytes_forward": init_win_fwd,
        " Init_Win_bytes_backward": init_win_bwd,
        " act_data_pkt_fwd": max(0, fwd_pkts - 1),
        " min_seg_size_forward": 20,
        "Active Mean": round(active_mean, 2),
        " Active Std": 0.0,
        " Active Max": round(active_mean, 2),
        " Active Min": round(active_mean, 2),
        "Idle Mean": round(idle_mean, 2),
        " Idle Std": 0.0,
        " Idle Max": round(idle_mean, 2),
        " Idle Min": round(idle_mean, 2),
        " Label": label,
    }


def generate_synthetic_dataset(
    output_path: Path,
    num_rows: int = 5000,
    seed: int = 42,
    duplicate_ratio: float = 0.02,
    artifact_ratio: float = 0.01,
) -> Dict[str, Any]:
    """Generate a synthetic CSV dataset matching CIC-IDS2017 structure and write to disk.

    Args:
        output_path: Destination file path for the CSV.
        num_rows: Total rows to generate.
        seed: Random seed for exact reproducibility.
        duplicate_ratio: Fraction of rows to duplicate (tests deduplication).
        artifact_ratio: Fraction of rows to inject inf/NaN into (tests cleaning).

    Returns:
        Summary statistics dictionary of generated records.
    """
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(ATTACK_PROFILES.keys())
    weights = [ATTACK_PROFILES[lbl]["weight"] for lbl in labels]

    rows: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {lbl: 0 for lbl in labels}
    artifact_count = 0

    target_unique = int(num_rows * (1.0 - duplicate_ratio))

    for i in range(target_unique):
        chosen_label = random.choices(labels, weights=weights, k=1)[0]
        inject_artifact = random.random() < artifact_ratio
        if inject_artifact:
            artifact_count += 1

        profile = ATTACK_PROFILES[chosen_label]
        row = generate_flow_record(
            label=chosen_label,
            profile=profile,
            inject_nan_or_inf=inject_artifact,
        )
        rows.append(row)
        class_counts[chosen_label] += 1

    # Inject duplicate rows to test deduplication step in preprocessing
    num_duplicates = num_rows - target_unique
    if num_duplicates > 0 and rows:
        dup_indices = random.choices(range(len(rows)), k=num_duplicates)
        for idx in dup_indices:
            dup_row = dict(rows[idx])
            rows.append(dup_row)
            class_counts[dup_row[" Label"]] += 1

    # Shuffle to ensure natural distribution
    random.shuffle(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CICIDS2017_RAW_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output_path": str(output_path),
        "total_rows": len(rows),
        "unique_rows": target_unique,
        "duplicate_rows": num_duplicates,
        "artifact_rows_injected": artifact_count,
        "class_counts": class_counts,
    }
    return summary


def main() -> None:
    """CLI entrypoint for synthetic sample generator."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic CIC-IDS2017 dataset for offline testing."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5000,
        help="Number of flow records to generate (default: 5000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/raw/sample_synthetic.csv",
        help="Output CSV path (default: dataset/raw/sample_synthetic.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    out_file = Path(args.output).resolve()
    summary = generate_synthetic_dataset(
        output_path=out_file,
        num_rows=args.rows,
        seed=args.seed,
    )

    print(f"[+] Successfully generated synthetic dataset at: {summary['output_path']}")
    print(f"    Total rows: {summary['total_rows']} (Duplicates: {summary['duplicate_rows']}, Artifacts: {summary['artifact_rows_injected']})")
    print("    Class distribution:")
    for lbl, count in summary["class_counts"].items():
        pct = (count / summary["total_rows"]) * 100
        print(f"      - {lbl:26s}: {count:5d} ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
