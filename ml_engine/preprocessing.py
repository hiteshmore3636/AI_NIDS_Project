"""AI-NIDS Preprocessing Pipeline.

Implements zero-leakage data ingestion, column cleaning, duplicate removal,
infinite/missing value sanitization, label standardization, correlation/variance
pruning, stratified splitting, and StandardScaler normalization.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Standardized label taxonomy mapping for CIC-IDS2017
LABEL_MAPPING: Dict[str, str] = {
    "BENIGN": "BENIGN",
    # DoS variations
    "DOS HULK": "DoS",
    "DOS GOLDENEYE": "DoS",
    "DOS SLOWLORIS": "DoS",
    "DOS SLOWHTTPTEST": "DoS",
    # DDoS
    "DDOS": "DDoS",
    # PortScan
    "PORTSCAN": "PortScan",
    # Brute Force Patator
    "FTP-PATATOR": "BruteForce",
    "SSH-PATATOR": "BruteForce",
    # Web attacks
    "WEB ATTACK - BRUTE FORCE": "WebAttack",
    "WEB ATTACK - XSS": "WebAttack",
    "WEB ATTACK - SQL INJECTION": "WebAttack",
    "WEB ATTACK  BRUTE FORCE": "WebAttack",
    "WEB ATTACK  XSS": "WebAttack",
    "WEB ATTACK  SQL INJECTION": "WebAttack",
    # Botnet
    "BOT": "Botnet",
    # Rare / other attacks
    "INFILTRATION": "Other",
    "HEARTBLEED": "Other",
}

# Raw CIC-IDS2017 headers to schema-aligned feature names
RAW_TO_TRACK_B_MAP: Dict[str, str] = {
    "Destination Port": "destination_port",
    "Flow Duration": "flow_duration_ms",
    "Total Fwd Packets": "fwd_packets",
    "Total Backward Packets": "bwd_packets",
    "Total Length of Fwd Packets": "fwd_bytes",
    "Total Length of Bwd Packets": "bwd_bytes",
    "Min Packet Length": "pkt_len_min",
    "Max Packet Length": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Flow Bytes/s": "flow_bytes_per_sec",
    "Flow Packets/s": "flow_packets_per_sec",
    "Flow IAT Mean": "iat_mean_ms",
    "Flow IAT Std": "iat_std_ms",
    "Flow IAT Min": "iat_min_ms",
    "Flow IAT Max": "iat_max_ms",
    "Fwd IAT Total": "fwd_iat_total_ms",
    "Bwd IAT Total": "bwd_iat_total_ms",
    "SYN Flag Count": "tcp_syn_count",
    "FIN Flag Count": "tcp_fin_count",
    "RST Flag Count": "tcp_rst_count",
    "PSH Flag Count": "tcp_psh_count",
    "ACK Flag Count": "tcp_ack_count",
    "URG Flag Count": "tcp_urg_count",
    "Init_Win_bytes_forward": "init_win_bytes_fwd",
    "Init_Win_bytes_backward": "init_win_bytes_bwd",
    "Active Mean": "active_mean_ms",
    "Idle Mean": "idle_mean_ms",
}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespaces and standardise column naming."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def standardize_labels(series: pd.Series) -> pd.Series:
    """Map diverse raw labels into the unified NIDS multi-class taxonomy."""
    cleaned = series.astype(str).str.strip().str.upper()
    # Normalize dash / hyphen variations
    cleaned = cleaned.str.replace("–", "-", regex=False).str.replace("—", "-", regex=False)
    mapped = cleaned.map(LABEL_MAPPING).fillna("Other")
    return mapped


def load_raw_dataset(input_path: Path) -> pd.DataFrame:
    """Load raw dataset file or merge multiple CSVs if a directory is given."""
    if input_path.is_file():
        logger.info(f"Loading single dataset file: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)
        return clean_column_names(df)

    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {input_path}")

        logger.info(f"Merging {len(csv_files)} CSV files from {input_path}")
        dfs = []
        for f in csv_files:
            logger.info(f"  Reading: {f.name}")
            sub_df = pd.read_csv(f, low_memory=False)
            sub_df = clean_column_names(sub_df)
            dfs.append(sub_df)
        merged = pd.concat(dfs, ignore_index=True)
        return merged

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def sanitize_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean duplicate rows and replace inf/-inf with NaN."""
    initial_rows = len(df)
    stats: Dict[str, Any] = {"initial_rows": initial_rows}

    # 1. Deduplication
    df = df.drop_duplicates().copy()
    stats["deduplicated_rows"] = len(df)
    stats["duplicates_removed"] = initial_rows - len(df)

    # 2. Convert string/object columns that might contain numbers, "Infinity", or "NaN" to numeric
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col.lower() != "label":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0.5 * len(df):
                df[col] = converted

    # 3. Infinite values handling
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[num_cols])
    stats["inf_values_found"] = int(inf_mask.values.sum())

    # Replace +/- inf with NaN in all numeric columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    return df, stats


def prepare_features(
    df: pd.DataFrame,
    track: str = "track_b",
    drop_identifiers: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target label and select feature columns according to the specified track."""
    df = df.copy()

    # Find the Label column (case-insensitive)
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if not label_col:
        raise KeyError("Could not find 'Label' column in dataset.")

    y = standardize_labels(df[label_col])
    X = df.drop(columns=[label_col])

    # Drop explicit identifier columns
    default_identifiers = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "External IP",
        "SimillarHTTP",
    ]
    to_drop = list(set((drop_identifiers or []) + default_identifiers))
    existing_drops = [c for c in to_drop if c in X.columns]
    if existing_drops:
        X = X.drop(columns=existing_drops)

    if track.lower() == "track_b":
        # Align with Track B Deployable Schema
        aligned_cols = {}
        for raw_col, track_b_col in RAW_TO_TRACK_B_MAP.items():
            if raw_col in X.columns:
                val = X[raw_col].copy()
                # Convert microseconds to milliseconds for duration & IAT
                if "ms" in track_b_col and "duration" in track_b_col:
                    val = val / 1000.0
                elif "ms" in track_b_col and "iat" in track_b_col:
                    val = val / 1000.0
                elif "ms" in track_b_col and ("active" in track_b_col or "idle" in track_b_col):
                    val = val / 1000.0
                aligned_cols[track_b_col] = val

        # Handle transport protocol
        if "Protocol" in X.columns:
            aligned_cols["protocol"] = X["Protocol"]
        else:
            # Default to TCP (6) or infer
            aligned_cols["protocol"] = 6

        X_processed = pd.DataFrame(aligned_cols, index=X.index)
        # Reorder to ensure deterministic schema
        return X_processed, y

    # Track A: Keep all available numeric columns
    X_numeric = X.select_dtypes(include=[np.number]).copy()
    return X_numeric, y


def prune_collinear_and_zerovar(
    X_train: pd.DataFrame,
    corr_threshold: float = 0.95,
) -> List[str]:
    """Identify and prune zero-variance and multicollinear features using TRAIN partition ONLY.

    Non-negotiable Rule 2: Prevents data leakage by fitting feature selection on train split only.
    """
    # 1. Zero variance check on train
    stds = X_train.std(axis=0)
    zero_var_cols = stds[stds == 0].index.tolist()
    logger.info(f"Zero-variance columns found on train ({len(zero_var_cols)}): {zero_var_cols}")

    active_cols = [c for c in X_train.columns if c not in zero_var_cols]
    X_active = X_train[active_cols]

    # 2. Pearson correlation pruning on train
    corr_matrix = X_active.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    collinear_drops = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
    logger.info(
        f"Collinear columns exceeding threshold {corr_threshold} ({len(collinear_drops)}): {collinear_drops}"
    )

    selected_features = [c for c in active_cols if c not in collinear_drops]
    return selected_features


def execute_preprocessing_pipeline(
    input_path: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    track: str = "track_b",
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute end-to-end zero-leakage preprocessing pipeline and export artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = REPO_ROOT / "ml_engine" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load config
    cfg: Dict[str, Any] = {}
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    split_cfg = cfg.get("split_strategy", {})
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio = split_cfg.get("val_ratio", 0.15)
    test_ratio = split_cfg.get("test_ratio", 0.15)
    random_seed = split_cfg.get("random_seed", 42)
    prep_cfg = cfg.get("preprocessing", {})
    corr_threshold = prep_cfg.get("correlation_threshold", 0.95)

    # 2. Ingestion & Sanitization
    df_raw = load_raw_dataset(input_path)
    if sample_limit and len(df_raw) > sample_limit:
        logger.info(f"Subsampling raw dataset to {sample_limit} rows.")
        df_raw = df_raw.sample(n=sample_limit, random_state=random_seed)

    df_clean, sanitization_stats = sanitize_values(df_raw)

    # 3. Feature & Target Separation
    X, y = prepare_features(
        df_clean,
        track=track,
        drop_identifiers=prep_cfg.get("drop_identifiers", []),
    )

    # 4. Zero-Leakage Partitioning (Split BEFORE fitting scaler / imputers)
    # Stratified split: Train (70%), Val (15%), Test (15%)
    val_test_ratio = val_ratio + test_ratio
    can_stratify_1 = y.value_counts().min() >= 2
    stratify_1 = y if can_stratify_1 else None

    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=val_test_ratio,
        random_state=random_seed,
        stratify=stratify_1,
    )

    relative_test_ratio = test_ratio / val_test_ratio
    can_stratify_2 = y_temp.value_counts().min() >= 2
    stratify_2 = y_temp if can_stratify_2 else None

    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_ratio,
        random_state=random_seed,
        stratify=stratify_2,
    )

    logger.info(f"Split sizes: Train={len(X_train_raw)}, Val={len(X_val_raw)}, Test={len(X_test_raw)}")

    # 5. Median Imputation fitted on TRAIN only (Zero Leakage)
    train_medians = X_train_raw.median(axis=0)
    # Fallback to 0 if a column is entirely NaN
    train_medians = train_medians.fillna(0.0)

    X_train_imp = X_train_raw.fillna(train_medians)
    X_val_imp = X_val_raw.fillna(train_medians)
    X_test_imp = X_test_raw.fillna(train_medians)

    # 6. Feature Selection (pruning collinear/zero-var on TRAIN only)
    selected_features = prune_collinear_and_zerovar(
        X_train_imp,
        corr_threshold=corr_threshold,
    )
    logger.info(f"Selected {len(selected_features)} features out of {X.shape[1]}")

    X_train_sel = X_train_imp[selected_features].copy()
    X_val_sel = X_val_imp[selected_features].copy()
    X_test_sel = X_test_imp[selected_features].copy()

    # 7. Scaler Fitting strictly on TRAIN only
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_sel),
        columns=selected_features,
        index=X_train_sel.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_sel),
        columns=selected_features,
        index=X_val_sel.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_sel),
        columns=selected_features,
        index=X_test_sel.index,
    )

    # 8. Export Processed Parquets with Target Label
    train_df = X_train_scaled.copy()
    train_df["label"] = y_train

    val_df = X_val_scaled.copy()
    val_df["label"] = y_val

    test_df = X_test_scaled.copy()
    test_df["label"] = y_test

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    logger.info(f"Saved processed partitions to {output_dir}")

    # 9. Save Scaler & Selected Features JSON Artifacts
    scaler_path = models_dir / "scaler.joblib"
    features_path = models_dir / "features.json"

    joblib.dump(scaler, scaler_path)

    feature_metadata = {
        "track": track,
        "n_features": len(selected_features),
        "features": selected_features,
        "labels": sorted(y.unique().tolist()),
    }
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_metadata, f, indent=2)

    # 10. Compile Empirical Cleaning & Split Report
    summary: Dict[str, Any] = {
        "sanitization": sanitization_stats,
        "split": {
            "strategy": "Stratified Zero-Leakage Split",
            "random_seed": random_seed,
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
        },
        "class_distributions": {
            "train": y_train.value_counts().to_dict(),
            "val": y_val.value_counts().to_dict(),
            "test": y_test.value_counts().to_dict(),
        },
        "features": {
            "initial_feature_count": X.shape[1],
            "selected_feature_count": len(selected_features),
            "pruned_features": [c for c in X.columns if c not in selected_features],
            "selected_features": selected_features,
        },
        "artifacts": {
            "train_parquet": str(train_path),
            "val_parquet": str(val_path),
            "test_parquet": str(test_path),
            "scaler_joblib": str(scaler_path),
            "features_json": str(features_path),
        },
    }

    summary_file = results_dir / "cleaning_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote empirical cleaning summary to: {summary_file}")

    return summary


def main() -> None:
    """CLI entrypoint for data preprocessing."""
    parser = argparse.ArgumentParser(description="AI-NIDS Zero-Leakage Preprocessing Pipeline.")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/raw/sample_synthetic.csv",
        help="Input raw CSV path or raw directory (default: dataset/raw/sample_synthetic.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/processed",
        help="Directory to export parquet partitions (default: dataset/processed)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Data configuration YAML path (default: configs/data_config.yaml)",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="track_b",
        choices=["track_a", "track_b"],
        help="Feature track: track_b (deployable NFStream) or track_a (benchmark full)",
    )
    args = parser.parse_args()

    input_p = (REPO_ROOT / args.input).resolve()
    output_d = (REPO_ROOT / args.output_dir).resolve()
    config_p = (REPO_ROOT / args.config).resolve()

    summary = execute_preprocessing_pipeline(
        input_path=input_p,
        output_dir=output_d,
        config_path=config_p,
        track=args.track,
    )

    print("\n" + "=" * 60)
    print(" PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Train Partition : {summary['split']['train_count']:,} rows")
    print(f"Val Partition   : {summary['split']['val_count']:,} rows")
    print(f"Test Partition  : {summary['split']['test_count']:,} rows")
    print(f"Features Kept   : {summary['features']['selected_feature_count']} features")
    print("=" * 60)


if __name__ == "__main__":
    main()
