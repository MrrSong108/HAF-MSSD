import os
import re
import json
import joblib
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =========================
# 1. Path and basic configuration
# =========================
#
# Expected directory format:
#   class_0_challenging/
#       0cluster_0_train.csv
#       0cluster_0_validation.csv
#       0cluster_0_test.csv
#       ...
#
#   class_1_friendly/
#       1cluster_0_train.csv
#       1cluster_0_validation.csv
#       1cluster_0_test.csv
#       ...
#
# Example:
#   CLASS_0_DIR=/path/to/class_0_challenging \
#   CLASS_1_DIR=/path/to/class_1_friendly \
#   CLASS_0_SCALER_PATH=/path/to/class_0_scaler.pkl \
#   CLASS_1_SCALER_PATH=/path/to/class_1_scaler.pkl \
#   OUTPUT_DIR=/path/to/outputs \
#   python calculate_cluster_centers.py

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(os.getenv("BASE_DIR", "data/processed/clusters"))

CLASS_DIRS = {
    0: Path(os.getenv("CLASS_0_DIR", BASE_DIR / "class_0_challenging")),
    1: Path(os.getenv("CLASS_1_DIR", BASE_DIR / "class_1_friendly")),
}

# Optional explicit scaler paths. If not provided, the script will try to find scalers automatically.
SCALER_PATHS = {
    0: os.getenv("CLASS_0_SCALER_PATH", ""),
    1: os.getenv("CLASS_1_SCALER_PATH", ""),
}

# Choose which split to use for calculating cluster centers.
# Options: "train", "validation", "test", "all"
CENTER_SPLIT = os.getenv("CENTER_SPLIT", "train").lower()

if CENTER_SPLIT not in {"train", "validation", "test", "all"}:
    raise ValueError("CENTER_SPLIT must be one of: train, validation, test, all.")

TARGET_COL = os.getenv("TARGET_COL", "predict_T2_0.5")
TIME_COL = os.getenv("TIME_COL", "start_time")

# Columns that should not be used as clustering-center features.
DROP_COLUMNS = [
    "cluster",
    TARGET_COL,
    TIME_COL,
]

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/cluster_centers"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CENTER_PATH = OUTPUT_DIR / "cluster_centers.pkl"
OUTPUT_META_PATH = OUTPUT_DIR / "cluster_centers_meta.json"

# Detailed outputs may contain local file paths and feature names.
# Keep disabled for public repositories.
SAVE_DETAILED_OUTPUTS = os.getenv("SAVE_DETAILED_OUTPUTS", "0") == "1"


# =========================
# 2. Scaler loading
# =========================
def find_scaler(class_id: int, class_dir: Path):
    """
    Find the scaler corresponding to the current class.

    Priority:
    1. Explicit environment variable:
       CLASS_0_SCALER_PATH or CLASS_1_SCALER_PATH
    2. Files containing "scaler" under the class directory.
    3. Files containing "scaler" under BASE_DIR.

    For reproducibility, explicit scaler paths are recommended.
    """

    explicit_scaler_path = SCALER_PATHS.get(class_id, "")

    if explicit_scaler_path:
        scaler_path = Path(explicit_scaler_path)

        if not scaler_path.exists():
            raise FileNotFoundError(f"Explicit scaler path does not exist: {scaler_path}")

        scaler = joblib.load(scaler_path)
        return scaler, scaler_path

    candidate_paths = []

    candidate_paths.extend(class_dir.glob("*scaler*.pkl"))
    candidate_paths.extend(BASE_DIR.glob(f"*class_{class_id}*scaler*.pkl"))
    candidate_paths.extend(BASE_DIR.glob(f"*cluster{class_id}*scaler*.pkl"))
    candidate_paths.extend(BASE_DIR.glob(f"*{class_id}*scaler*.pkl"))

    candidate_paths = list(dict.fromkeys(candidate_paths))

    if len(candidate_paths) == 0:
        raise FileNotFoundError(
            f"No scaler was found for class {class_id}. "
            "Please provide CLASS_0_SCALER_PATH or CLASS_1_SCALER_PATH."
        )

    if len(candidate_paths) > 1:
        print(f"\nWarning: multiple scaler files were found for class {class_id}.")
        print("The first one will be used. Explicit scaler paths are recommended.")
        for path in candidate_paths:
            print(f"  - {path}")

    scaler_path = candidate_paths[0]
    scaler = joblib.load(scaler_path)

    return scaler, scaler_path


# =========================
# 3. Parse cluster file names
# =========================
def parse_cluster_file(file_path: Path):
    """
    Parse file names such as:
        0cluster_0_train.csv
        0cluster_0_validation.csv
        0cluster_0_test.csv
        1cluster_3_train.csv
        1cluster_3_validation.csv
        1cluster_3_test.csv

    Returns:
        file_class_id, cluster_id, split_name
    """

    stem = file_path.stem

    pattern = r"^(\d+)cluster[_-]?(\d+)_(train|validation|valid|val|test)$"
    match = re.match(pattern, stem, flags=re.IGNORECASE)

    if match is None:
        return None, None, None

    file_class_id = int(match.group(1))
    cluster_id = int(match.group(2))
    split_name = match.group(3).lower()

    if split_name in {"valid", "val"}:
        split_name = "validation"

    return file_class_id, cluster_id, split_name


def get_cluster_csv_files(class_id: int, class_dir: Path):
    """
    Get cluster CSV files under the current class directory.

    CENTER_SPLIT controls which split is used:
    - "train": only *_train.csv
    - "validation": only *_validation.csv
    - "test": only *_test.csv
    - "all": train, validation, and test files
    """

    cluster_files = []

    for file_path in class_dir.glob("*.csv"):
        file_class_id, cluster_id, split_name = parse_cluster_file(file_path)

        if file_class_id is None:
            continue

        if file_class_id != class_id:
            print(f"Warning: file class ID does not match current class. Skipped: {file_path.name}")
            continue

        if CENTER_SPLIT != "all" and split_name != CENTER_SPLIT:
            continue

        cluster_files.append(
            {
                "cluster_id": int(cluster_id),
                "split": split_name,
                "path": file_path,
            }
        )

    cluster_files = sorted(
        cluster_files,
        key=lambda x: (x["cluster_id"], x["split"]),
    )

    return cluster_files


# =========================
# 4. Feature preparation
# =========================
def prepare_features(df: pd.DataFrame, scaler):
    """
    Remove non-feature columns and keep numeric features only.

    If the scaler has feature_names_in_, the feature matrix will be aligned
    strictly according to the scaler's original training feature order.
    """

    df = df.copy()

    existing_drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=existing_drop_cols, errors="ignore")

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(0)

    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)

        missing_cols = [col for col in feature_names if col not in numeric_df.columns]
        extra_cols = [col for col in numeric_df.columns if col not in feature_names]

        if missing_cols:
            raise ValueError(
                "The current cluster data is missing feature columns used during scaler fitting. "
                f"First 10 missing columns: {missing_cols[:10]}. "
                f"Total missing columns: {len(missing_cols)}."
            )

        if extra_cols:
            print(
                "Notice: extra numeric columns not used by the scaler will be ignored. "
                f"First 10 extra columns: {extra_cols[:10]}."
            )

        numeric_df = numeric_df[feature_names]

    return numeric_df


# =========================
# 5. Cluster center calculation
# =========================
def calculate_cluster_centers():
    cluster_centers = {}
    cluster_meta = {}

    for class_id, class_dir in CLASS_DIRS.items():

        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory does not exist: {class_dir}")

        print(f"\nProcessing class {class_id}: {class_dir}")

        scaler, scaler_path = find_scaler(class_id, class_dir)
        print(f"Using scaler: {scaler_path}")

        cluster_files = get_cluster_csv_files(class_id, class_dir)

        if len(cluster_files) == 0:
            raise FileNotFoundError(
                f"No cluster CSV files matching CENTER_SPLIT='{CENTER_SPLIT}' were found in {class_dir}."
            )

        print(f"Detected {len(cluster_files)} cluster CSV files for CENTER_SPLIT='{CENTER_SPLIT}'.")

        for item in cluster_files:

            cluster_id = item["cluster_id"]
            split_name = item["split"]
            csv_path = item["path"]

            df = pd.read_csv(csv_path)

            if df.empty:
                print(f"Warning: empty file skipped: {csv_path.name}")
                continue

            features = prepare_features(df, scaler)

            if features.empty:
                print(f"Warning: no valid numeric features found. Skipped: {csv_path.name}")
                continue

            scaled_data = scaler.transform(features)

            center = scaled_data.mean(axis=0)

            # Example: 0cluster0, 1cluster7
            center_key = f"{class_id}cluster{cluster_id}"

            if center_key in cluster_centers and CENTER_SPLIT == "all":
                print(
                    f"Warning: duplicate center key detected: {center_key}. "
                    "This may happen when CENTER_SPLIT='all'. "
                    "The later file will overwrite the earlier one."
                )

            cluster_centers[center_key] = center

            meta_record = {
                "class_id": int(class_id),
                "cluster_id": int(cluster_id),
                "split": split_name,
                "sample_count": int(len(df)),
                "feature_count": int(features.shape[1]),
            }

            if SAVE_DETAILED_OUTPUTS:
                meta_record.update(
                    {
                        "csv_file": str(csv_path),
                        "scaler_path": str(scaler_path),
                        "feature_names": list(features.columns),
                    }
                )

            cluster_meta[center_key] = meta_record

            print(
                f"Completed: {center_key} | split={split_name} | "
                f"samples={len(df)} | features={features.shape[1]} | file={csv_path.name}"
            )

    return cluster_centers, cluster_meta


# =========================
# 6. Save results
# =========================
def main():
    cluster_centers, cluster_meta = calculate_cluster_centers()

    joblib.dump(cluster_centers, OUTPUT_CENTER_PATH)

    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(cluster_meta, f, ensure_ascii=True, indent=4)

    print("\nCluster centers saved successfully.")
    print(f"Center file: {OUTPUT_CENTER_PATH}")
    print(f"Metadata file: {OUTPUT_META_PATH}")

    print("\nCluster center keys:")
    for key in cluster_centers.keys():
        print(key)


if __name__ == "__main__":
    main()