"""Dataset Download and Verification Helper.

Provides CLI utilities to inspect, verify, and document datasets in `dataset/raw/`.
Assists researchers and students in verifying CIC-IDS2017 and UNSW-NB15 files.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

CICIDS2017_EXPECTED_FILES: List[str] = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

DATASET_SOURCES: Dict[str, str] = {
    "CIC-IDS2017 (Official UNB)": "https://www.unb.ca/cic/datasets/ids-2017.html",
    "CIC-IDS2017 (Kaggle Mirror)": "https://www.kaggle.com/datasets/cicdataset/cicids2017",
    "UNSW-NB15 (Official UNSW)": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
}


def compute_file_hash(filepath: Path, chunk_size: int = 1048576) -> str:
    """Compute SHA-256 hash of a file efficiently in chunks."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_raw_datasets(raw_dir: Path, compute_hashes: bool = False) -> Dict[str, bool]:
    """Check existence and status of benchmark raw CSV files."""
    print("=" * 70)
    print(" AI-NIDS Raw Dataset Verification")
    print("=" * 70)
    print(f"Target Directory: {raw_dir}")
    print("-" * 70)

    results: Dict[str, bool] = {}
    present_count = 0

    for filename in CICIDS2017_EXPECTED_FILES:
        target_file = raw_dir / filename
        exists = target_file.exists()
        results[filename] = exists

        if exists:
            present_count += 1
            size_mb = target_file.stat().st_size / (1024 * 1024)
            print(f"[FOUND] {filename:55s} ({size_mb:6.1f} MB)")
            if compute_hashes:
                file_hash = compute_file_hash(target_file)
                print(f"        SHA256: {file_hash}")
        else:
            print(f"[MISSING] {filename}")

    # Also check for synthetic sample
    sample_file = raw_dir / "sample_synthetic.csv"
    if sample_file.exists():
        size_mb = sample_file.stat().st_size / (1024 * 1024)
        print(f"[SAMPLE] sample_synthetic.csv is available ({size_mb:6.2f} MB)")

    print("-" * 70)
    print(f"Summary: {present_count}/{len(CICIDS2017_EXPECTED_FILES)} official files present in {raw_dir}")

    if present_count == 0 and not sample_file.exists():
        print("\n[!] No dataset files found. To download the official data:")
        for name, url in DATASET_SOURCES.items():
            print(f"    - {name}: {url}")
        print("\nAlternatively, generate an instant synthetic testing sample with:")
        print("    python -m ml_engine.sample_generator --rows 10000\n")

    return results


def main() -> None:
    """CLI entrypoint for dataset verification tool."""
    parser = argparse.ArgumentParser(description="Check and verify AI-NIDS raw datasets.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="dataset/raw",
        help="Path to raw dataset folder (default: dataset/raw)",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Calculate SHA-256 hash for present files (slow for multi-GB files)",
    )
    args = parser.parse_args()

    target_dir = (REPO_ROOT / args.raw_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    check_raw_datasets(target_dir, compute_hashes=args.hash)


if __name__ == "__main__":
    main()
