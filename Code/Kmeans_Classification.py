import os
import json
import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")


# =========================
# 1. Path and basic configuration
# =========================
# Use relative paths or environment variables for public GitHub repositories.
#
# Example:
#   CLASSIFIER_MODEL_PATH=/path/to/lstm_classifier.keras \
#   CLASSIFIER_SCALER_PATH=/path/to/scaler_X.pkl \
#   TRAIN_PATH=/path/to/train.csv \
#   VALIDATION_PATH=/path/to/validation.csv \
#   TEST_PATH=/path/to/test.csv \
#   OUTPUT_DIR=/path/to/outputs \
#   python kmeans_classification_public.py

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DATA_DIR = Path(os.getenv("DATA_DIR", "data/processed"))

MODEL_PATH = Path(
    os.getenv(
        "CLASSIFIER_MODEL_PATH",
        "outputs/classifier/classifier_model.keras",
    )
)

SCALER_PATH = Path(
    os.getenv(
        "CLASSIFIER_SCALER_PATH",
        "outputs/classifier/scaler_X.pkl",
    )
)

TRAIN_PATH = Path(os.getenv("TRAIN_PATH", DATA_DIR / "train.csv"))
VALIDATION_PATH = Path(os.getenv("VALIDATION_PATH", DATA_DIR / "validation.csv"))
TEST_PATH = Path(os.getenv("TEST_PATH", DATA_DIR / "test.csv"))

OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        "outputs/kmeans_clustered_data",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = os.getenv("TARGET_COL", "predict_T2_0.5")
TIME_COL = os.getenv("TIME_COL", "start_time")

CLASSIFIER_THRESHOLD = float(os.getenv("CLASSIFIER_THRESHOLD", "0.5"))
CLASSIFIER_BATCH_SIZE = int(os.getenv("CLASSIFIER_BATCH_SIZE", "128"))

# Model input format:
# - "sequence": reshape to [num_samples, 1, num_features], used by LSTM/GRU/BiLSTM/CNN classifiers.
# - "flat": keep [num_samples, num_features], used by RF/XGBoost classifiers.
CLASSIFIER_INPUT_MODE = os.getenv("CLASSIFIER_INPUT_MODE", "sequence").lower()

if CLASSIFIER_INPUT_MODE not in {"sequence", "flat"}:
    raise ValueError("CLASSIFIER_INPUT_MODE must be either 'sequence' or 'flat'.")

# Class-specific KMeans configuration.
CLASS_INFO = {
    0: {
        "name": "challenging",
        "n_clusters": int(os.getenv("CLASS_0_N_CLUSTERS", "10")),
    },
    1: {
        "name": "friendly",
        "n_clusters": int(os.getenv("CLASS_1_N_CLUSTERS", "16")),
    },
}

# Detailed outputs may contain complete labeled data, feature names, and local paths.
# Keep disabled for public repositories.
SAVE_DETAILED_OUTPUTS = os.getenv("SAVE_DETAILED_OUTPUTS", "0") == "1"

# Aggregate outputs contain full train/validation/test data with labels and clusters.
# They are useful locally, but should not be uploaded to GitHub.
SAVE_AGGREGATE_OUTPUTS = os.getenv("SAVE_AGGREGATE_OUTPUTS", "0") == "1"


# =========================
# 2. File validation
# =========================
for file_path in [MODEL_PATH, SCALER_PATH, TRAIN_PATH, VALIDATION_PATH, TEST_PATH]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

if VALIDATION_PATH.resolve() == TEST_PATH.resolve():
    raise ValueError(
        "Validation and test files must be different. "
        "Using the test set as the validation set causes data leakage."
    )


# =========================
# 3. Feature utilities
# =========================
def get_feature_columns(df: pd.DataFrame, scaler) -> list:
    """
    Get feature columns used by the classifier.

    If the scaler has feature_names_in_, use it first.
    Otherwise, remove leakage-prone columns and keep the remaining columns.
    """

    if hasattr(scaler, "feature_names_in_"):
        feature_cols = list(scaler.feature_names_in_)

        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                "The input data is missing feature columns used during classifier-scaler fitting. "
                f"First 10 missing columns: {missing_cols[:10]}. "
                f"Total missing columns: {len(missing_cols)}."
            )

        return feature_cols

    leakage_prefixes = ("predict_",)

    drop_cols = {
        TIME_COL,
        TARGET_COL,
        "label",
        "true_label",
        "pred_label",
        "class_label",
        "cluster",
        "cluster_global",
        "y_true",
        "y_pred",
        "true_value",
        "pred_value",
        "prediction",
        "error",
        "abs_error",
        "relative_error",
        "difficulty_label",
    }

    feature_cols = []

    for col in df.columns:
        if col in drop_cols:
            continue

        if any(col.startswith(prefix) for prefix in leakage_prefixes):
            continue

        feature_cols.append(col)

    return feature_cols


def validate_feature_columns(df: pd.DataFrame, feature_cols: list, dataset_name: str):
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"The {dataset_name} dataset is missing required feature columns. "
            f"First 10 missing columns: {missing_cols[:10]}. "
            f"Total missing columns: {len(missing_cols)}."
        )


def build_feature_matrix(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Build a numeric feature matrix.
    Non-numeric values are coerced to NaN and then filled with 0.
    """

    X = df[feature_cols].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0)

    return X


def transform_with_scaler(X_raw: pd.DataFrame, scaler):
    """
    Transform features using the classifier scaler.
    """

    if hasattr(scaler, "feature_names_in_"):
        X_raw = X_raw[list(scaler.feature_names_in_)]
        X_scaled = scaler.transform(X_raw)
    else:
        X_scaled = scaler.transform(X_raw.values)

    return X_scaled


# =========================
# 4. Classifier prediction
# =========================
def predict_class(df: pd.DataFrame, feature_cols: list, scaler, model) -> np.ndarray:
    """
    Predict difficulty labels using the trained classifier.

    Label meaning:
    - 0: challenging samples
    - 1: friendly samples
    """

    X_raw = build_feature_matrix(df, feature_cols)
    X_scaled = transform_with_scaler(X_raw, scaler)

    if CLASSIFIER_INPUT_MODE == "sequence":
        X_input = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    else:
        X_input = X_scaled

    raw_pred = model.predict(
        X_input,
        batch_size=CLASSIFIER_BATCH_SIZE,
        verbose=0,
    )

    raw_pred = np.asarray(raw_pred)

    if raw_pred.ndim == 2 and raw_pred.shape[1] == 1:
        proba = raw_pred.reshape(-1)
        pred_label = (proba >= CLASSIFIER_THRESHOLD).astype(int)
    elif raw_pred.ndim == 2 and raw_pred.shape[1] > 1:
        pred_label = np.argmax(raw_pred, axis=1).astype(int)
    else:
        proba = raw_pred.reshape(-1)
        pred_label = (proba >= CLASSIFIER_THRESHOLD).astype(int)

    return pred_label.astype(int)


# =========================
# 5. Save cluster-level CSV files
# =========================
def save_cluster_csv(data: pd.DataFrame, feature_cols: list, class_label: int, split_name: str, class_dir: Path):
    """
    Save each cluster as an independent CSV file.

    Downstream cluster-specific prediction models can read these files and
    drop the cluster column and target column before training.
    """

    if len(data) == 0:
        print(f"{split_name} | class {class_label} has no samples. No cluster CSV will be saved.")
        return

    save_cols = feature_cols + ["cluster", TARGET_COL]

    for cluster_id in sorted(data["cluster"].unique()):
        cluster_data = data[data["cluster"] == cluster_id].copy()

        save_path = class_dir / f"{class_label}cluster_{cluster_id}_{split_name}.csv"

        cluster_data[save_cols].to_csv(save_path, index=False)

        print(
            f"{split_name} | class {class_label} | cluster {cluster_id} | "
            f"samples: {len(cluster_data)} | saved to: {save_path}"
        )


def save_class_all_csv(data: pd.DataFrame, feature_cols: list, class_label: int, class_name: str, split_name: str, class_dir: Path):
    """
    Optionally save all samples of a class in one CSV file.
    This is disabled by default because it contains full local data.
    """

    if len(data) == 0:
        return

    save_cols = feature_cols + ["label", "cluster", "cluster_global", TARGET_COL]

    save_path = class_dir / f"class_{class_label}_{class_name}_{split_name}_all.csv"
    data[save_cols].to_csv(save_path, index=False)


# =========================
# 6. Main process
# =========================
def main():
    print("Loading classifier model and scaler...")

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if hasattr(scaler, "set_output"):
        scaler.set_output(transform="default")

    print("Loading train, validation, and test datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for dataset_name, df in {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }.items():
        if TARGET_COL not in df.columns:
            raise ValueError(f"The target column '{TARGET_COL}' is missing in the {dataset_name} dataset.")

    feature_cols = get_feature_columns(train_df, scaler)

    validate_feature_columns(validation_df, feature_cols, "validation")
    validate_feature_columns(test_df, feature_cols, "test")

    print(f"Number of features: {len(feature_cols)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Test samples: {len(test_df)}")

    if SAVE_DETAILED_OUTPUTS:
        print("First 10 feature columns:", feature_cols[:10])

    # =========================
    # Predict class labels
    # =========================
    print("\nPredicting train labels...")
    train_df["label"] = predict_class(train_df, feature_cols, scaler, model)

    print("Predicting validation labels...")
    validation_df["label"] = predict_class(validation_df, feature_cols, scaler, model)

    print("Predicting test labels...")
    test_df["label"] = predict_class(test_df, feature_cols, scaler, model)

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts().sort_index())

    print("\nValidation label distribution:")
    print(validation_df["label"].value_counts().sort_index())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts().sort_index())

    all_train_clustered = []
    all_validation_clustered = []
    all_test_clustered = []

    summary = {}

    # =========================
    # Fit KMeans on train only, then assign validation/test
    # =========================
    for class_label, info in CLASS_INFO.items():
        class_name = info["name"]
        n_clusters = info["n_clusters"]

        print("\n" + "=" * 80)
        print(f"Processing class {class_label}: {class_name} | K={n_clusters}")
        print("=" * 80)

        class_dir = OUTPUT_DIR / f"class_{class_label}_{class_name}"
        class_dir.mkdir(parents=True, exist_ok=True)

        train_sub = train_df[train_df["label"] == class_label].copy()
        validation_sub = validation_df[validation_df["label"] == class_label].copy()
        test_sub = test_df[test_df["label"] == class_label].copy()

        if len(train_sub) < n_clusters:
            raise ValueError(
                f"Class {class_label} has fewer training samples ({len(train_sub)}) "
                f"than the configured number of clusters ({n_clusters}). "
                "Please reduce CLASS_0_N_CLUSTERS or CLASS_1_N_CLUSTERS."
            )

        X_train_raw = build_feature_matrix(train_sub, feature_cols)

        cluster_scaler = StandardScaler()
        X_train_scaled = cluster_scaler.fit_transform(X_train_raw)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        train_sub["cluster"] = kmeans.fit_predict(X_train_scaled)
        train_sub["cluster_global"] = train_sub["label"] * 1000 + train_sub["cluster"]

        if len(validation_sub) > 0:
            X_validation_raw = build_feature_matrix(validation_sub, feature_cols)
            X_validation_scaled = cluster_scaler.transform(X_validation_raw)

            validation_sub["cluster"] = kmeans.predict(X_validation_scaled)
            validation_sub["cluster_global"] = validation_sub["label"] * 1000 + validation_sub["cluster"]
        else:
            validation_sub["cluster"] = pd.Series(dtype=int)
            validation_sub["cluster_global"] = pd.Series(dtype=int)

        if len(test_sub) > 0:
            X_test_raw = build_feature_matrix(test_sub, feature_cols)
            X_test_scaled = cluster_scaler.transform(X_test_raw)

            test_sub["cluster"] = kmeans.predict(X_test_scaled)
            test_sub["cluster_global"] = test_sub["label"] * 1000 + test_sub["cluster"]
        else:
            test_sub["cluster"] = pd.Series(dtype=int)
            test_sub["cluster_global"] = pd.Series(dtype=int)

        dbi = davies_bouldin_score(X_train_scaled, train_sub["cluster"])

        print(f"\nClass {class_label} train cluster distribution:")
        print(train_sub["cluster"].value_counts().sort_index())

        print(f"\nClass {class_label} validation cluster distribution:")
        if len(validation_sub) > 0:
            print(validation_sub["cluster"].value_counts().sort_index())
        else:
            print("No validation samples in this class.")

        print(f"\nClass {class_label} test cluster distribution:")
        if len(test_sub) > 0:
            print(test_sub["cluster"].value_counts().sort_index())
        else:
            print("No test samples in this class.")

        print(f"\nClass {class_label} train DBI: {dbi:.6f}")
        print(f"Class {class_label} KMeans inertia: {kmeans.inertia_:.6f}")

        # Save KMeans scaler and model for the current class.
        joblib.dump(cluster_scaler, class_dir / f"cluster_scaler_label{class_label}.pkl")
        joblib.dump(kmeans, class_dir / f"kmeans_label{class_label}.pkl")

        # Save cluster-level train/validation/test CSV files.
        save_cluster_csv(train_sub, feature_cols, class_label, "train", class_dir)
        save_cluster_csv(validation_sub, feature_cols, class_label, "validation", class_dir)
        save_cluster_csv(test_sub, feature_cols, class_label, "test", class_dir)

        if SAVE_AGGREGATE_OUTPUTS:
            save_class_all_csv(train_sub, feature_cols, class_label, class_name, "train", class_dir)
            save_class_all_csv(validation_sub, feature_cols, class_label, class_name, "validation", class_dir)
            save_class_all_csv(test_sub, feature_cols, class_label, class_name, "test", class_dir)

        train_save_cols = feature_cols + ["label", "cluster", "cluster_global", TARGET_COL]
        all_train_clustered.append(train_sub[train_save_cols])

        if len(validation_sub) > 0:
            all_validation_clustered.append(validation_sub[train_save_cols])

        if len(test_sub) > 0:
            all_test_clustered.append(test_sub[train_save_cols])

        summary[str(class_label)] = {
            "class_name": class_name,
            "n_clusters": int(n_clusters),
            "train_samples": int(len(train_sub)),
            "validation_samples": int(len(validation_sub)),
            "test_samples": int(len(test_sub)),
            "dbi_train_scaled": float(dbi),
            "kmeans_inertia": float(kmeans.inertia_),
            "train_cluster_counts": {
                str(k): int(v) for k, v in train_sub["cluster"].value_counts().sort_index().items()
            },
            "validation_cluster_counts": {
                str(k): int(v) for k, v in validation_sub["cluster"].value_counts().sort_index().items()
            } if len(validation_sub) > 0 else {},
            "test_cluster_counts": {
                str(k): int(v) for k, v in test_sub["cluster"].value_counts().sort_index().items()
            } if len(test_sub) > 0 else {},
        }

    # =========================
    # Save optional aggregate outputs
    # =========================
    if SAVE_AGGREGATE_OUTPUTS:
        train_clustered_all = pd.concat(all_train_clustered, axis=0).sort_index()
        train_clustered_all.to_csv(OUTPUT_DIR / "train_labeled_clustered_all.csv", index=False)

        if len(all_validation_clustered) > 0:
            validation_clustered_all = pd.concat(all_validation_clustered, axis=0).sort_index()
            validation_clustered_all.to_csv(OUTPUT_DIR / "validation_labeled_clustered_all.csv", index=False)

        if len(all_test_clustered) > 0:
            test_clustered_all = pd.concat(all_test_clustered, axis=0).sort_index()
            test_clustered_all.to_csv(OUTPUT_DIR / "test_labeled_clustered_all.csv", index=False)

    feature_config = {
        "num_features": len(feature_cols),
        "target_col": TARGET_COL,
        "classifier_input_mode": CLASSIFIER_INPUT_MODE,
        "classifier_threshold": CLASSIFIER_THRESHOLD,
        "class_info": CLASS_INFO,
        "save_detailed_outputs": SAVE_DETAILED_OUTPUTS,
        "save_aggregate_outputs": SAVE_AGGREGATE_OUTPUTS,
    }

    if SAVE_DETAILED_OUTPUTS:
        feature_config.update(
            {
                "model_path": str(MODEL_PATH),
                "scaler_path": str(SCALER_PATH),
                "train_path": str(TRAIN_PATH),
                "validation_path": str(VALIDATION_PATH),
                "test_path": str(TEST_PATH),
                "output_dir": str(OUTPUT_DIR),
                "feature_cols": feature_cols,
            }
        )

    with open(OUTPUT_DIR / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=True, indent=2)

    with open(OUTPUT_DIR / "kmeans_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print("\n" + "=" * 80)
    print("KMeans clustering completed.")
    print(f"Result directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()