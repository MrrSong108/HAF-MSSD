import os
import re
import json
import joblib
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# =========================
# 1. Path and basic configuration
# =========================
#
# Example:
#   PREDICT_PATH=/path/to/test.csv \
#   CLASSIFIER_MODEL_PATH=/path/to/rf_classifier.pkl \
#   CLASSIFIER_SCALER_PATH=/path/to/scaler_X.pkl \
#   CLUSTER_BASE_DIR=/path/to/kmeans_clustered_data \
#   RESULT_MODEL_DIR=/path/to/cluster_models \
#   OUTPUT_DIR=/path/to/outputs \
#   python final_prediction_public.py

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DATA_DIR = Path(os.getenv("DATA_DIR", "data/processed"))

PREDICT_PATH = Path(
    os.getenv(
        "PREDICT_PATH",
        DATA_DIR / "test.csv",
    )
)

CLASSIFIER_MODEL_PATH = Path(
    os.getenv(
        "CLASSIFIER_MODEL_PATH",
        "outputs/classifier/classifier_model.pkl",
    )
)

CLASSIFIER_SCALER_PATH = Path(
    os.getenv(
        "CLASSIFIER_SCALER_PATH",
        "outputs/classifier/scaler_X.pkl",
    )
)

CLUSTER_BASE_DIR = Path(
    os.getenv(
        "CLUSTER_BASE_DIR",
        "outputs/kmeans_clustered_data",
    )
)

CLASS_DIRS = {
    0: CLUSTER_BASE_DIR / "class_0_challenging",
    1: CLUSTER_BASE_DIR / "class_1_friendly",
}

CLUSTER_SCALER_PATHS = {
    0: CLASS_DIRS[0] / "cluster_scaler_label0.pkl",
    1: CLASS_DIRS[1] / "cluster_scaler_label1.pkl",
}

KMEANS_PATHS = {
    0: CLASS_DIRS[0] / "kmeans_label0.pkl",
    1: CLASS_DIRS[1] / "kmeans_label1.pkl",
}

# This directory stores cluster-specific submodels.
# Both layouts are supported:
#   RESULT_MODEL_DIR/0cluster_0/best_rf_model.pkl
#   RESULT_MODEL_DIR/class_0_challenging/0cluster_0/best_rf_model.pkl
RESULT_MODEL_DIR = Path(
    os.getenv(
        "RESULT_MODEL_DIR",
        "outputs/cluster_models",
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        "outputs/final_prediction",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREDICTION_PATH = Path(
    os.getenv(
        "OUTPUT_PREDICTION_PATH",
        OUTPUT_DIR / "final_prediction_result.csv",
    )
)

OUTPUT_METRICS_PATH = Path(
    os.getenv(
        "OUTPUT_METRICS_PATH",
        OUTPUT_DIR / "final_prediction_metrics.json",
    )
)

TARGET_COL = os.getenv("TARGET_COL", "predict_T2_0.5")
TIME_COL = os.getenv("TIME_COL", "start_time")

CLASS_INFO = {
    0: {
        "name": "challenging",
        "dir_name": "class_0_challenging",
    },
    1: {
        "name": "friendly",
        "dir_name": "class_1_friendly",
    },
}

CLASSIFIER_THRESHOLD = float(os.getenv("CLASSIFIER_THRESHOLD", "0.5"))
CLASSIFIER_BATCH_SIZE = int(os.getenv("CLASSIFIER_BATCH_SIZE", "128"))

# Classifier input format:
# - "flat": for RF/XGBoost/sklearn classifiers.
# - "sequence": for LSTM/GRU/BiLSTM/CNN classifiers.
CLASSIFIER_INPUT_MODE = os.getenv("CLASSIFIER_INPUT_MODE", "flat").lower()

if CLASSIFIER_INPUT_MODE not in {"flat", "sequence"}:
    raise ValueError("CLASSIFIER_INPUT_MODE must be either 'flat' or 'sequence'.")

# If TARGET_COL exists in PREDICT_PATH, metrics will be calculated.
EVALUATE_IF_TARGET_EXISTS = os.getenv("EVALUATE_IF_TARGET_EXISTS", "1") == "1"

# Prediction outputs may contain true values, errors, class labels, cluster labels,
# and timestamps. Keep sensitive outputs out of public repositories.
SAVE_PREDICTIONS = os.getenv("SAVE_PREDICTIONS", "1") == "1"
INCLUDE_TARGET_IN_OUTPUT = os.getenv("INCLUDE_TARGET_IN_OUTPUT", "1") == "1"
INCLUDE_TIME_IN_OUTPUT = os.getenv("INCLUDE_TIME_IN_OUTPUT", "1") == "1"

# Detailed metadata may expose local paths and model structure.
SAVE_DETAILED_OUTPUTS = os.getenv("SAVE_DETAILED_OUTPUTS", "0") == "1"


# =========================
# 2. Basic utilities
# =========================
def check_path(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def load_model_auto(model_path: Path):
    """
    Automatically load sklearn/joblib models, Keras models, or XGBoost JSON models.
    """

    suffix = model_path.suffix.lower()

    if suffix in [".pkl", ".joblib"]:
        return joblib.load(model_path)

    if suffix in [".keras", ".h5"]:
        from tensorflow.keras.models import load_model
        return load_model(model_path)

    if suffix == ".json":
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        return model

    raise ValueError(f"Unsupported model format: {model_path}")


def is_keras_model_path(model_path: Path):
    return model_path.suffix.lower() in [".keras", ".h5"]


def get_default_feature_columns(df: pd.DataFrame):
    """
    Generate feature columns when the scaler does not have feature_names_in_.
    Leakage-prone columns are removed automatically.
    """

    leakage_columns = {
        TIME_COL,
        TARGET_COL,
        "label",
        "true_label",
        "pred_label",
        "class_label",
        "cluster",
        "cluster_id",
        "cluster_global",
        "y_true",
        "y_pred",
        "true",
        "predicted",
        "prediction",
        "error",
        "abs_error",
        "relative_error",
        "difficulty_label",
    }

    feature_cols = []

    for col in df.columns:
        if col in leakage_columns:
            continue

        if col.startswith("predict_"):
            continue

        feature_cols.append(col)

    return feature_cols


def align_features(df: pd.DataFrame, scaler, name: str):
    """
    Align input features according to the scaler's training feature names.
    This avoids feature-order mismatches between training and inference.
    """

    df = df.copy()

    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    else:
        feature_names = get_default_feature_columns(df)

    missing_cols = [col for col in feature_names if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"{name} is missing feature columns used during scaler fitting. "
            f"First 10 missing columns: {missing_cols[:10]}. "
            f"Total missing columns: {len(missing_cols)}."
        )

    X = df[feature_names].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0)

    return X


def normalize_classifier_output(pred):
    """
    Normalize classifier outputs to integer labels.

    Supported outputs:
    - sklearn predict: 0/1 labels
    - neural network predict: probability values
    - multi-class probability matrix: argmax labels
    """

    pred = np.asarray(pred)

    if pred.ndim == 2:
        if pred.shape[1] == 1:
            pred = pred.reshape(-1)
            pred = (pred >= CLASSIFIER_THRESHOLD).astype(int)
        else:
            pred = np.argmax(pred, axis=1).astype(int)
    else:
        pred = pred.reshape(-1)

        unique_values = np.unique(pred)

        if not set(unique_values).issubset({0, 1}):
            pred = (pred >= CLASSIFIER_THRESHOLD).astype(int)

    return pred.astype(int)


def find_one_file(folder: Path, patterns, name: str):
    """
    Find one file in a folder by multiple glob patterns.
    """

    matched_files = []

    for pattern in patterns:
        matched_files.extend(list(folder.glob(pattern)))

    matched_files = list(dict.fromkeys(matched_files))

    if len(matched_files) == 0:
        raise FileNotFoundError(
            f"No {name} was found in {folder}. Patterns: {patterns}"
        )

    if len(matched_files) > 1:
        print(f"Warning: multiple {name} files were found in {folder}. The first one will be used:")
        for file_path in matched_files:
            print(f"  - {file_path}")

    return matched_files[0]


def get_model_type_from_path(model_path: Path):
    """
    Infer model type from file name.
    """

    name = model_path.name.lower()

    known_types = ["bilstm", "lstm", "gru", "cnn", "rf", "randomforest", "xgb", "xgboost"]

    for model_type in known_types:
        if model_type in name:
            if model_type == "xgboost":
                return "xgb"
            if model_type == "randomforest":
                return "rf"
            return model_type

    match = re.search(r"best_(.*?)_model", name)

    if match:
        model_type = match.group(1)
        if model_type in ["xgboost", "xgb"]:
            return "xgb"
        if model_type in ["randomforest", "rf"]:
            return "rf"
        return model_type

    raise ValueError(f"Cannot identify model type from file name: {model_path.name}")


def directional_symmetry(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) <= 1:
        return 0.0

    true_direction = np.sign(y_true[1:] - y_true[:-1])
    pred_direction = np.sign(y_pred[1:] - y_pred[:-1])

    return float(np.mean(true_direction == pred_direction))


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1))))
    ds = directional_symmetry(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "DS": ds,
    }


# =========================
# 3. Classification and clustering
# =========================
def predict_class_labels(df: pd.DataFrame, classifier_model, classifier_scaler):
    """
    Use the trained classifier to predict class labels.
    """

    X_cls = align_features(df, classifier_scaler, "classifier input features")
    X_cls_scaled = classifier_scaler.transform(X_cls)

    if CLASSIFIER_INPUT_MODE == "sequence":
        X_input = X_cls_scaled.reshape(X_cls_scaled.shape[0], 1, X_cls_scaled.shape[1])
    else:
        X_input = X_cls_scaled

    pred_raw = classifier_model.predict(X_input)

    return normalize_classifier_output(pred_raw)


def assign_clusters(df: pd.DataFrame, label_pred: np.ndarray):
    """
    Assign each sample to a KMeans cluster according to its predicted class.
    """

    cluster_pred = np.full(len(df), fill_value=-1, dtype=int)

    for class_id in [0, 1]:
        idx = np.where(label_pred == class_id)[0]

        if len(idx) == 0:
            print(f"class {class_id} has no samples. Clustering is skipped.")
            continue

        cluster_scaler = joblib.load(CLUSTER_SCALER_PATHS[class_id])
        kmeans = joblib.load(KMEANS_PATHS[class_id])

        X_class_raw = df.iloc[idx].copy()
        X_cluster = align_features(X_class_raw, cluster_scaler, f"class {class_id} clustering input features")
        X_cluster_scaled = cluster_scaler.transform(X_cluster)

        cluster_labels = kmeans.predict(X_cluster_scaled).astype(int)

        cluster_pred[idx] = cluster_labels

        print(f"\nclass {class_id} cluster assignment distribution:")
        print(pd.Series(cluster_labels).value_counts().sort_index())

    return cluster_pred


# =========================
# 4. Cluster-specific prediction
# =========================
def get_cluster_model_dir(class_id: int, cluster_id: int):
    """
    Support both model directory layouts:

    1. Flat layout:
       RESULT_MODEL_DIR/0cluster_0/

    2. Class-subfolder layout:
       RESULT_MODEL_DIR/class_0_challenging/0cluster_0/
    """

    cluster_folder_name = f"{class_id}cluster_{cluster_id}"
    class_dir_name = CLASS_INFO[class_id]["dir_name"]

    candidate_dirs = [
        RESULT_MODEL_DIR / cluster_folder_name,
        RESULT_MODEL_DIR / class_dir_name / cluster_folder_name,
    ]

    for candidate_dir in candidate_dirs:
        if candidate_dir.exists():
            return candidate_dir

    raise FileNotFoundError(
        f"No model directory was found for class={class_id}, cluster={cluster_id}. "
        f"Checked: {candidate_dirs}"
    )


def predict_one_cluster(class_id: int, cluster_id: int, X_df: pd.DataFrame):
    """
    Locate and run the cluster-specific prediction model.
    """

    cluster_model_dir = get_cluster_model_dir(class_id, cluster_id)
    cluster_folder_name = f"{class_id}cluster_{cluster_id}"

    print(
        f"\nPredicting: class={class_id}, cluster={cluster_id}, "
        f"samples={len(X_df)}, model_dir={cluster_model_dir}"
    )

    model_path = find_one_file(
        cluster_model_dir,
        [
            "best_*_model.pkl",
            "best_*_model.joblib",
            "best_*_model.keras",
            "best_*_model.h5",
            "best_*_model.json",
            "*xgboost*.json",
            "*xgb*.json",
        ],
        name="best prediction model",
    )

    model_type = get_model_type_from_path(model_path)

    scaler_X_path = find_one_file(
        cluster_model_dir,
        [
            "scaler_X.pkl",
            "scaler_x.pkl",
            "*scaler_X*.pkl",
            "*scaler_x*.pkl",
        ],
        name="feature scaler",
    )

    scaler_y_path = find_one_file(
        cluster_model_dir,
        [
            "scaler_y.pkl",
            "scaler_Y.pkl",
            "*scaler_y*.pkl",
            "*scaler_Y*.pkl",
        ],
        name="target scaler",
    )

    print(f"Model file: {model_path.name}")
    print(f"Model type: {model_type}")
    print(f"Feature scaler: {scaler_X_path.name}")
    print(f"Target scaler: {scaler_y_path.name}")

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    model = load_model_auto(model_path)

    X_aligned = align_features(X_df, scaler_X, f"{cluster_folder_name} model input features")
    X_scaled = scaler_X.transform(X_aligned)

    if model_type in ["lstm", "gru", "bilstm", "cnn"]:
        X_model = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    else:
        X_model = X_scaled

    if is_keras_model_path(model_path):
        pred_scaled = model.predict(X_model, verbose=0)
    else:
        pred_scaled = model.predict(X_model)

    pred_scaled = np.asarray(pred_scaled).reshape(-1, 1)
    pred = scaler_y.inverse_transform(pred_scaled).reshape(-1)
    pred = np.maximum(pred, 0)

    return pred


def run_cluster_specific_prediction(df: pd.DataFrame, label_pred: np.ndarray, cluster_pred: np.ndarray):
    """
    Run prediction for all class-cluster groups.
    """

    final_pred = np.full(len(df), fill_value=np.nan, dtype=float)

    for class_id in [0, 1]:
        used_clusters = sorted(np.unique(cluster_pred[label_pred == class_id]))

        for cluster_id in used_clusters:
            if cluster_id == -1:
                continue

            sample_idx = np.where(
                (label_pred == class_id) & (cluster_pred == cluster_id)
            )[0]

            if len(sample_idx) == 0:
                continue

            X_sub = df.iloc[sample_idx].copy()

            pred_sub = predict_one_cluster(
                class_id=class_id,
                cluster_id=int(cluster_id),
                X_df=X_sub,
            )

            final_pred[sample_idx] = pred_sub

    nan_count = int(np.isnan(final_pred).sum())

    if nan_count > 0:
        bad_idx = np.where(np.isnan(final_pred))[0][:20]
        raise ValueError(
            f"{nan_count} samples did not receive predictions. "
            f"First 20 indexes: {bad_idx}"
        )

    return final_pred


# =========================
# 5. Main process
# =========================
def main():
    check_path(PREDICT_PATH, "prediction input data")
    check_path(CLASSIFIER_MODEL_PATH, "classifier model")
    check_path(CLASSIFIER_SCALER_PATH, "classifier scaler")

    for class_id in [0, 1]:
        check_path(CLUSTER_SCALER_PATHS[class_id], f"class {class_id} clustering scaler")
        check_path(KMEANS_PATHS[class_id], f"class {class_id} KMeans model")

    print("Loading prediction input data...")
    predict_df = pd.read_csv(PREDICT_PATH)

    print(f"Prediction input shape: {predict_df.shape}")

    has_target = TARGET_COL in predict_df.columns

    if not has_target:
        print(
            f"Notice: target column '{TARGET_COL}' was not found. "
            "Predictions will be generated without evaluation metrics."
        )

    print("Loading classifier model and scaler...")
    classifier_scaler = joblib.load(CLASSIFIER_SCALER_PATH)
    classifier_model = load_model_auto(CLASSIFIER_MODEL_PATH)

    if hasattr(classifier_scaler, "set_output"):
        classifier_scaler.set_output(transform="default")

    print("\nPredicting class labels...")
    label_pred = predict_class_labels(
        predict_df,
        classifier_model,
        classifier_scaler,
    )

    print("\nClass label distribution:")
    print(pd.Series(label_pred).value_counts().sort_index())

    print("\nAssigning KMeans clusters...")
    cluster_pred = assign_clusters(
        predict_df,
        label_pred,
    )

    print("\nRunning cluster-specific prediction...")
    final_pred = run_cluster_specific_prediction(
        predict_df,
        label_pred,
        cluster_pred,
    )

    metrics = None

    if has_target and EVALUATE_IF_TARGET_EXISTS:
        y_true = predict_df[TARGET_COL].values
        metrics = calc_metrics(y_true, final_pred)

        print("\nFinal prediction metrics:")
        print(f"RMSE: {metrics['RMSE']:.6f}")
        print(f"MAE : {metrics['MAE']:.6f}")
        print(f"MAPE: {metrics['MAPE']:.6f}")
        print(f"DS  : {metrics['DS']:.6f}")

        with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=True, indent=4)

        print(f"\nMetrics saved to: {OUTPUT_METRICS_PATH}")

    if SAVE_PREDICTIONS:
        result_df = pd.DataFrame(
            {
                "predicted": final_pred,
                "class_label": label_pred,
                "cluster_id": cluster_pred,
                "cluster_model_key": [
                    f"{label_pred[i]}cluster_{cluster_pred[i]}" for i in range(len(predict_df))
                ],
            }
        )

        if INCLUDE_TIME_IN_OUTPUT and TIME_COL in predict_df.columns:
            result_df.insert(0, TIME_COL, predict_df[TIME_COL].values)

        if has_target and INCLUDE_TARGET_IN_OUTPUT:
            y_true = predict_df[TARGET_COL].values

            result_df.insert(
                0,
                "true",
                y_true,
            )

            result_df["abs_error"] = np.abs(y_true - final_pred)
            result_df["relative_error"] = np.abs(y_true - final_pred) / (y_true + 1)

        OUTPUT_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(OUTPUT_PREDICTION_PATH, index=False)

        print(f"\nPrediction results saved to: {OUTPUT_PREDICTION_PATH}")

    run_metadata = {
        "predict_path": str(PREDICT_PATH) if SAVE_DETAILED_OUTPUTS else PREDICT_PATH.name,
        "target_col": TARGET_COL,
        "has_target": bool(has_target),
        "classifier_input_mode": CLASSIFIER_INPUT_MODE,
        "classifier_threshold": CLASSIFIER_THRESHOLD,
        "num_samples": int(len(predict_df)),
        "class_distribution": {
            str(k): int(v) for k, v in pd.Series(label_pred).value_counts().sort_index().items()
        },
        "cluster_distribution": {
            str(k): int(v) for k, v in pd.Series(cluster_pred).value_counts().sort_index().items()
        },
        "metrics": metrics,
        "save_predictions": SAVE_PREDICTIONS,
        "include_target_in_output": INCLUDE_TARGET_IN_OUTPUT,
        "include_time_in_output": INCLUDE_TIME_IN_OUTPUT,
        "save_detailed_outputs": SAVE_DETAILED_OUTPUTS,
    }

    if SAVE_DETAILED_OUTPUTS:
        run_metadata.update(
            {
                "classifier_model_path": str(CLASSIFIER_MODEL_PATH),
                "classifier_scaler_path": str(CLASSIFIER_SCALER_PATH),
                "cluster_base_dir": str(CLUSTER_BASE_DIR),
                "result_model_dir": str(RESULT_MODEL_DIR),
                "output_prediction_path": str(OUTPUT_PREDICTION_PATH),
                "output_metrics_path": str(OUTPUT_METRICS_PATH),
            }
        )

    metadata_path = OUTPUT_DIR / "final_prediction.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, ensure_ascii=True, indent=4)

    print(f"Metadata saved to: {metadata_path}")
    print("\nFinal prediction completed.")


if __name__ == "__main__":
    main()