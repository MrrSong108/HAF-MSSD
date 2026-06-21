import os
import json
import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")


# =========================
# 1. Path and basic configuration
# =========================
#
# Example:
#   CLASSIFIER_MODEL_PATH=/path/to/lstm_classifier.keras \
#   CLASSIFIER_SCALER_PATH=/path/to/scaler_X.pkl \
#   TRAIN_PATH=/path/to/train.csv \
#   OUTPUT_DIR=/path/to/outputs \
#   python search_dbi_public.py

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

TRAIN_PATH = Path(
    os.getenv(
        "TRAIN_PATH",
        DATA_DIR / "train.csv",
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        "outputs/kmeans_dbi_search",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = os.getenv("TARGET_COL", "predict_T2_0.5")
TIME_COL = os.getenv("TIME_COL", "start_time")

K_MIN = int(os.getenv("K_MIN", "2"))
K_MAX = int(os.getenv("K_MAX", "20"))

CLASSIFIER_THRESHOLD = float(os.getenv("CLASSIFIER_THRESHOLD", "0.5"))
CLASSIFIER_BATCH_SIZE = int(os.getenv("CLASSIFIER_BATCH_SIZE", "128"))

# Model input format:
# - "sequence": reshape to [num_samples, 1, num_features], used by LSTM/GRU/BiLSTM classifiers.
# - "flat": keep [num_samples, num_features], used by RF/XGBoost classifiers.
CLASSIFIER_INPUT_MODE = os.getenv("CLASSIFIER_INPUT_MODE", "sequence").lower()

if CLASSIFIER_INPUT_MODE not in {"sequence", "flat"}:
    raise ValueError("CLASSIFIER_INPUT_MODE must be either 'sequence' or 'flat'.")

# Detailed outputs may contain complete labeled data, feature names, and local paths.
# Keep disabled for public repositories.
SAVE_DETAILED_OUTPUTS = os.getenv("SAVE_DETAILED_OUTPUTS", "0") == "1"

CLASS_NAME_MAP = {
    0: "challenging",
    1: "friendly",
}


# =========================
# 2. File validation
# =========================
for file_path in [MODEL_PATH, SCALER_PATH, TRAIN_PATH]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


# =========================
# 3. Feature utilities
# =========================
def get_feature_columns(df, scaler):
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


def build_feature_matrix(df, feature_cols):
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


def transform_with_scaler(X_raw, scaler):
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
def predict_labels(df, feature_cols, scaler, model):
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
# 5. DBI search
# =========================
def search_best_kmeans_dbi(data, feature_cols, class_label, k_min=2, k_max=20):
    """
    Search the best number of KMeans clusters for one predicted class.

    A new StandardScaler is used only for KMeans clustering.
    Do not reuse the classifier scaler for KMeans clustering because it
    represents the classifier preprocessing pipeline.
    """

    class_name = CLASS_NAME_MAP.get(class_label, f"class_{class_label}")

    X_raw = build_feature_matrix(data, feature_cols)

    cluster_scaler = StandardScaler()
    X_scaled = cluster_scaler.fit_transform(X_raw)

    dbi_scores = {}

    max_k_allowed = min(k_max, len(data) - 1)

    if max_k_allowed < k_min:
        raise ValueError(
            f"Class {class_label} has too few samples for KMeans search. "
            f"Sample count: {len(data)}."
        )

    print("\n" + "=" * 80)
    print(f"Searching best K for class {class_label} ({class_name})")
    print(f"Sample count: {len(data)}")
    print("=" * 80)

    for k in range(k_min, max_k_allowed + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        clusters = kmeans.fit_predict(X_scaled)
        dbi = davies_bouldin_score(X_scaled, clusters)

        dbi_scores[int(k)] = float(dbi)

        print(f"class={class_label}, K={k}, DBI={dbi:.6f}")

    best_k = min(dbi_scores, key=dbi_scores.get)
    best_dbi = dbi_scores[best_k]

    print(f"\nClass {class_label} ({class_name}) best K: {best_k}")
    print(f"Class {class_label} ({class_name}) minimum DBI: {best_dbi:.6f}")

    return dbi_scores, best_k, best_dbi


def plot_dbi_curve(dbi_scores, class_label, save_path):
    """
    Save the DBI curve.
    In Linux or AutoDL environments, figures should be saved directly.
    """

    x = list(dbi_scores.keys())
    y = list(dbi_scores.values())

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Davies-Bouldin Index")
    plt.title(f"DBI Curve for Class {class_label}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 6. Main process
# =========================
def main():
    print("Loading classifier model and scaler...")

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if hasattr(scaler, "set_output"):
        scaler.set_output(transform="default")

    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH)

    if TARGET_COL not in train_df.columns:
        print(
            f"Notice: target column '{TARGET_COL}' was not found in the training data. "
            "DBI search can continue because the target column is not required."
        )

    feature_cols = get_feature_columns(train_df, scaler)

    print(f"\nTraining sample count: {len(train_df)}")
    print(f"Number of features used for classification and KMeans: {len(feature_cols)}")

    if SAVE_DETAILED_OUTPUTS:
        print("First 10 feature columns:", feature_cols[:10])

    print("\nPredicting labels for the training set...")
    train_df["label"] = predict_labels(train_df, feature_cols, scaler, model)

    print("\nPredicted label distribution:")
    print(train_df["label"].value_counts().sort_index())

    if SAVE_DETAILED_OUTPUTS:
        labeled_train_path = OUTPUT_DIR / "train_label.csv"
        train_df.to_csv(labeled_train_path, index=False)
        print(f"\nLabeled training data saved to: {labeled_train_path}")

    summary = {}

    for class_label in [0, 1]:
        class_data = train_df[train_df["label"] == class_label].copy()

        if len(class_data) == 0:
            print(f"Class {class_label} has no samples. Skipping.")
            continue

        dbi_scores, best_k, best_dbi = search_best_kmeans_dbi(
            data=class_data,
            feature_cols=feature_cols,
            class_label=class_label,
            k_min=K_MIN,
            k_max=K_MAX,
        )

        class_name = CLASS_NAME_MAP.get(class_label, f"class_{class_label}")

        fig_path = OUTPUT_DIR / f"dbi_curve_class_{class_label}_{class_name}.png"
        plot_dbi_curve(dbi_scores, class_label, fig_path)

        dbi_result_path = OUTPUT_DIR / f"dbi_scores_class_{class_label}_{class_name}.json"

        dbi_result = {
            "class_label": int(class_label),
            "class_name": class_name,
            "sample_count": int(len(class_data)),
            "dbi_scores": {str(k): float(v) for k, v in dbi_scores.items()},
            "best_k": int(best_k),
            "best_dbi": float(best_dbi),
        }

        if SAVE_DETAILED_OUTPUTS:
            dbi_result.update(
                {
                    "figure_path": str(fig_path),
                    "json_path": str(dbi_result_path),
                    "feature_cols": feature_cols,
                }
            )

        with open(dbi_result_path, "w", encoding="utf-8") as f:
            json.dump(
                dbi_result,
                f,
                ensure_ascii=True,
                indent=2,
            )

        summary[str(class_label)] = {
            "class_name": class_name,
            "sample_count": int(len(class_data)),
            "best_k": int(best_k),
            "best_dbi": float(best_dbi),
            "dbi_scores": {str(k): float(v) for k, v in dbi_scores.items()},
        }

        if SAVE_DETAILED_OUTPUTS:
            summary[str(class_label)].update(
                {
                    "figure_path": str(fig_path),
                    "json_path": str(dbi_result_path),
                }
            )

    summary_metadata = {
        "classifier_input_mode": CLASSIFIER_INPUT_MODE,
        "classifier_threshold": CLASSIFIER_THRESHOLD,
        "k_search_range": [K_MIN, K_MAX],
        "random_state": RANDOM_STATE,
        "save_detailed_outputs": SAVE_DETAILED_OUTPUTS,
    }

    if SAVE_DETAILED_OUTPUTS:
        summary_metadata.update(
            {
                "model_path": str(MODEL_PATH),
                "scaler_path": str(SCALER_PATH),
                "train_path": str(TRAIN_PATH),
                "output_dir": str(OUTPUT_DIR),
                "num_features": len(feature_cols),
                "feature_cols": feature_cols,
            }
        )
    else:
        summary_metadata.update(
            {
                "num_features": len(feature_cols),
            }
        )

    final_summary = {
        "metadata": summary_metadata,
        "results": summary,
    }

    summary_path = OUTPUT_DIR / "dbi_search_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            final_summary,
            f,
            ensure_ascii=True,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("DBI search completed.")
    print(f"Result directory: {OUTPUT_DIR}")
    print(f"Summary file: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()