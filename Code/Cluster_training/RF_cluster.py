import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import re
import time
import json
import joblib
import optuna
import random
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# ==================================================
# 1. Path and basic configuration
# ==================================================
# Use relative paths or environment variables for public GitHub repositories.
#
# Expected file format:
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

CLASS_DIRS = {
    "class_0_challenging": Path(
        os.getenv(
            "CLASS_0_DIR",
            "data/processed/clusters/class_0_challenging",
        )
    ),
    "class_1_friendly": Path(
        os.getenv(
            "CLASS_1_DIR",
            "data/processed/clusters/class_1_friendly",
        )
    ),
}

SAVE_ROOT = Path(
    os.getenv(
        "SAVE_ROOT",
        "outputs/rf_cluster_models",
    )
)
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_COL = os.getenv("TARGET_COL", "predict_T2_0.5")
TIME_COL = os.getenv("TIME_COL", "start_time")

N_TRIALS = int(os.getenv("N_TRIALS", "50"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Random Forest parallel jobs.
RF_N_JOBS = int(os.getenv("RF_N_JOBS", "-1"))

# Keep Optuna single-threaded by default to avoid excessive CPU usage.
OPTUNA_N_JOBS = int(os.getenv("OPTUNA_N_JOBS", "1"))

# Detailed outputs may contain true values, predictions, file paths,
# and feature names. Keep disabled for public repositories.
SAVE_DETAILED_OUTPUTS = os.getenv("SAVE_DETAILED_OUTPUTS", "0") == "1"


# ==================================================
# 2. Random seed
# ==================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


set_seed(RANDOM_STATE)


# ==================================================
# 3. Metric functions
# ==================================================
def directional_symmetry(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) <= 1:
        return 0.0

    true_diff = np.sign(y_true[1:] - y_true[:-1])
    pred_diff = np.sign(y_pred[1:] - y_pred[:-1])

    return float(np.mean(true_diff == pred_diff))


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    y_pred = np.maximum(y_pred, 0)

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


# ==================================================
# 4. Find valid cluster training files
# ==================================================
def get_cluster_train_files(class_dir, class_prefix):
    """
    Match only true cluster training files, such as:
    0cluster_1_train.csv
    1cluster_3_train.csv

    This avoids matching aggregate files such as:
    class_0_challenging_train_all.csv
    class_1_friendly_train_all.csv
    """

    pattern = re.compile(rf"^{class_prefix}cluster_(\d+)_train\.csv$")

    matched_files = []

    for file_path in class_dir.iterdir():
        match = pattern.match(file_path.name)
        if match:
            cluster_id = int(match.group(1))
            matched_files.append((cluster_id, file_path))

    matched_files = sorted(matched_files, key=lambda x: x[0])

    return [x[1] for x in matched_files]


def get_related_paths(train_path):
    """
    For a cluster training file, locate its validation and test files.

    Example:
        0cluster_5_train.csv
        0cluster_5_validation.csv
        0cluster_5_test.csv
    """

    validation_path = Path(str(train_path).replace("_train.csv", "_validation.csv"))
    test_path = Path(str(train_path).replace("_train.csv", "_test.csv"))

    return validation_path, test_path


# ==================================================
# 5. Feature preparation
# ==================================================
def get_feature_columns(train_df, validation_df, test_df):
    drop_cols = [TARGET_COL]

    if "cluster" in train_df.columns:
        drop_cols.append("cluster")

    if TIME_COL in train_df.columns:
        drop_cols.append(TIME_COL)

    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    missing_in_validation = [c for c in feature_cols if c not in validation_df.columns]
    missing_in_test = [c for c in feature_cols if c not in test_df.columns]

    if missing_in_validation:
        raise ValueError(f"The validation file is missing feature columns: {missing_in_validation}")

    if missing_in_test:
        raise ValueError(f"The test file is missing feature columns: {missing_in_test}")

    return feature_cols


def build_feature_matrix(df, feature_cols):
    """
    Convert selected feature columns to numeric values.
    Non-numeric values are coerced to NaN and then filled with 0.
    """

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    return X


def prepare_train_validation_test(train_df, validation_df, test_df):
    """
    Use train, validation, and test as three independent datasets.

    - train: used for model fitting.
    - validation: used for Optuna tuning.
    - test: used only for final evaluation.
    """

    for name, df in {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }.items():
        if TARGET_COL not in df.columns:
            raise ValueError(f"The target column '{TARGET_COL}' is missing in the {name} dataset.")

    feature_cols = get_feature_columns(train_df, validation_df, test_df)

    X_train_raw = build_feature_matrix(train_df, feature_cols)
    X_validation_raw = build_feature_matrix(validation_df, feature_cols)
    X_test_raw = build_feature_matrix(test_df, feature_cols)

    y_train_raw = train_df[TARGET_COL].values.reshape(-1, 1)
    y_validation_raw = validation_df[TARGET_COL].values.reshape(-1, 1)
    y_test_raw = test_df[TARGET_COL].values.reshape(-1, 1)

    # Scalers must be fitted only on the training set.
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(X_train_raw)
    X_validation = scaler_X.transform(X_validation_raw)
    X_test = scaler_X.transform(X_test_raw)

    y_train = scaler_y.fit_transform(y_train_raw).reshape(-1)
    y_validation = scaler_y.transform(y_validation_raw).reshape(-1)
    y_test = scaler_y.transform(y_test_raw).reshape(-1)

    return {
        "feature_cols": feature_cols,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "y_train_raw": y_train_raw.reshape(-1),
        "y_validation_raw": y_validation_raw.reshape(-1),
        "y_test_raw": y_test_raw.reshape(-1),
    }


# ==================================================
# 6. Model helper
# ==================================================
def create_rf_model(params):
    return RandomForestRegressor(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=RF_N_JOBS,
    )


def inverse_predict(model, X, scaler_y):
    y_pred_scaled = model.predict(X).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
    y_pred = np.maximum(y_pred, 0)
    return y_pred


# ==================================================
# 7. Train one cluster
# ==================================================
def train_one_cluster(class_name, train_path, validation_path, test_path):
    cluster_name = train_path.name.replace("_train.csv", "")

    print("\n" + "=" * 100)
    print(f"Current class: {class_name}")
    print(f"Current cluster: {cluster_name}")
    print(f"Training file: {train_path}")
    print(f"Validation file: {validation_path}")
    print(f"Test file: {test_path}")

    class_save_dir = SAVE_ROOT / class_name
    cluster_save_dir = class_save_dir / cluster_name
    cluster_save_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = cluster_save_dir / "best_rf_model.pkl"
    scaler_x_save_path = cluster_save_dir / "scaler_X.pkl"
    scaler_y_save_path = cluster_save_dir / "scaler_y.pkl"
    params_save_path = cluster_save_dir / "best_params.json"
    metrics_save_path = cluster_save_dir / "metrics.csv"
    test_pred_save_path = cluster_save_dir / "test_predictions.csv"
    data_usage_info_save_path = cluster_save_dir / "data_usage_info.json"

    # Resume training: if the model and metrics already exist, skip this cluster.
    if model_save_path.exists() and metrics_save_path.exists():
        print(f"{class_name} - {cluster_name} has already been trained. Skipping.")
        old_metrics = pd.read_csv(metrics_save_path)
        test_row = old_metrics[old_metrics["dataset"] == "test"].iloc[0].to_dict()

        return {
            "class_name": class_name,
            "cluster": cluster_name,
            "model": "RandomForest",
            "n_train": int(old_metrics["n_samples"].iloc[0]) if "n_samples" in old_metrics.columns else None,
            "test_RMSE": test_row.get("RMSE"),
            "test_MAE": test_row.get("MAE"),
            "test_MAPE": test_row.get("MAPE"),
            "test_DS": test_row.get("DS"),
        }

    start_time = time.time()

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    data = prepare_train_validation_test(train_df, validation_df, test_df)

    feature_cols = data["feature_cols"]
    scaler_X = data["scaler_X"]
    scaler_y = data["scaler_y"]

    X_train = data["X_train"]
    X_validation = data["X_validation"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_validation = data["y_validation"]
    y_test = data["y_test"]

    y_train_raw = data["y_train_raw"]
    y_validation_raw = data["y_validation_raw"]
    y_test_raw = data["y_test_raw"]

    # ==================================================
    # Optuna objective function
    # ==================================================
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 30, step=1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, step=1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10, step=1),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5],
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

        model = create_rf_model(params)

        model.fit(X_train, y_train)

        y_validation_pred = inverse_predict(model, X_validation, scaler_y)

        validation_rmse = np.sqrt(mean_squared_error(y_validation_raw, y_validation_pred))

        return validation_rmse

    # ==================================================
    # Hyperparameter optimization
    # ==================================================
    study = optuna.create_study(direction="minimize")

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=OPTUNA_N_JOBS,
    )

    best_params = study.best_params

    final_params = {
        **best_params,
        "random_state": RANDOM_STATE,
        "n_jobs": RF_N_JOBS,
    }

    # ==================================================
    # Retrain with the best hyperparameters
    # ==================================================
    best_model = RandomForestRegressor(**final_params)

    best_model.fit(X_train, y_train)

    # ==================================================
    # Final prediction and evaluation
    # ==================================================
    y_train_pred = inverse_predict(best_model, X_train, scaler_y)
    y_validation_pred = inverse_predict(best_model, X_validation, scaler_y)
    y_test_pred = inverse_predict(best_model, X_test, scaler_y)

    train_metrics = calc_metrics(y_train_raw, y_train_pred)
    validation_metrics = calc_metrics(y_validation_raw, y_validation_pred)
    test_metrics = calc_metrics(y_test_raw, y_test_pred)

    run_time = time.time() - start_time

    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "train",
                "class_name": class_name,
                "cluster": cluster_name,
                "model": "RandomForest",
                "n_samples": len(train_df),
                **train_metrics,
                "run_time_seconds": run_time,
            },
            {
                "dataset": "validation",
                "class_name": class_name,
                "cluster": cluster_name,
                "model": "RandomForest",
                "n_samples": len(validation_df),
                **validation_metrics,
                "run_time_seconds": run_time,
            },
            {
                "dataset": "test",
                "class_name": class_name,
                "cluster": cluster_name,
                "model": "RandomForest",
                "n_samples": len(test_df),
                **test_metrics,
                "run_time_seconds": run_time,
            },
        ]
    )

    # ==================================================
    # Save model, scalers, parameters, metrics, and optional details
    # ==================================================
    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler_X, scaler_x_save_path)
    joblib.dump(scaler_y, scaler_y_save_path)

    with open(params_save_path, "w", encoding="utf-8") as f:
        json.dump(final_params, f, ensure_ascii=True, indent=4, default=str)

    data_usage_info = {
        "data_usage": "cluster_training",
        "n_train": len(train_df),
        "n_validation": len(validation_df),
        "n_test": len(test_df),
        "target_col": TARGET_COL,
        "num_features": len(feature_cols),
        "run_time_seconds": run_time,
        "save_detailed_outputs": SAVE_DETAILED_OUTPUTS,
    }

    if SAVE_DETAILED_OUTPUTS:
        data_usage_info.update(
            {
                "source_train_file": str(train_path),
                "source_validation_file": str(validation_path),
                "source_test_file": str(test_path),
                "feature_cols": feature_cols,
            }
        )

    with open(data_usage_info_save_path, "w", encoding="utf-8") as f:
        json.dump(data_usage_info, f, ensure_ascii=True, indent=4)

    metrics_df.to_csv(
        metrics_save_path,
        index=False,
    )

    if SAVE_DETAILED_OUTPUTS:
        test_pred_df = pd.DataFrame(
            {
                "y_true": y_test_raw,
                "y_pred": y_test_pred,
                "abs_error": np.abs(y_test_raw - y_test_pred),
            }
        )

        test_pred_df.to_csv(
            test_pred_save_path,
            index=False,
        )

    print(f"{class_name} - {cluster_name} training completed.")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Validation RMSE: {validation_metrics['RMSE']:.6f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.6f}")
    print(f"Test MAE : {test_metrics['MAE']:.6f}")
    print(f"Test MAPE: {test_metrics['MAPE']:.6f}")
    print(f"Test DS  : {test_metrics['DS']:.6f}")
    print(f"Runtime: {run_time:.2f} seconds")
    print(f"Saved to: {cluster_save_dir}")

    summary_record = {
        "class_name": class_name,
        "cluster": cluster_name,
        "model": "RandomForest",
        "n_train": len(train_df),
        "n_validation": len(validation_df),
        "n_test": len(test_df),
        "validation_RMSE": validation_metrics["RMSE"],
        "validation_MAE": validation_metrics["MAE"],
        "validation_MAPE": validation_metrics["MAPE"],
        "validation_DS": validation_metrics["DS"],
        "test_RMSE": test_metrics["RMSE"],
        "test_MAE": test_metrics["MAE"],
        "test_MAPE": test_metrics["MAPE"],
        "test_DS": test_metrics["DS"],
        "run_time_seconds": run_time,
    }

    return summary_record


# ==================================================
# 8. Main process: train class 0 and class 1 cluster models
# ==================================================
def main():
    all_metrics = []

    summary_csv_path = SAVE_ROOT / "cluster_rf.csv"
    excel_txt_path = SAVE_ROOT / "excel_copy_metrics.txt"
    excel_four_only_path = SAVE_ROOT / "excel_copy_four_metrics_only.txt"
    error_log_path = SAVE_ROOT / "error_log.txt"

    for class_name, class_dir in CLASS_DIRS.items():

        if "class_0" in class_name:
            class_prefix = "0"
        elif "class_1" in class_name:
            class_prefix = "1"
        else:
            raise ValueError(f"Unable to identify class prefix: {class_name}")

        print("\n" + "#" * 100)
        print(f"Processing class: {class_name}")
        print(f"Data directory: {class_dir}")

        if not class_dir.exists():
            print(f"Directory does not exist. Skipping: {class_dir}")

            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"[CLASS_DIR_NOT_EXIST]\t{class_name}\t{class_dir}\n")

            continue

        train_files = get_cluster_train_files(class_dir, class_prefix)

        print(f"Number of cluster training files found for {class_name}: {len(train_files)}")

        if len(train_files) == 0:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"[NO_CLUSTER_TRAIN_FILE]\t{class_name}\t{class_dir}\n")
            continue

        for train_path in train_files:
            validation_path, test_path = get_related_paths(train_path)

            if not validation_path.exists():
                print(f"Validation file does not exist. Skipping: {validation_path}")

                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"[VALIDATION_FILE_NOT_EXIST]\t{class_name}\t{train_path.name}\t{validation_path.name}\n"
                    )
                continue

            if not test_path.exists():
                print(f"Test file does not exist. Skipping: {test_path}")

                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"[TEST_FILE_NOT_EXIST]\t{class_name}\t{train_path.name}\t{test_path.name}\n"
                    )
                continue

            try:
                metrics = train_one_cluster(
                    class_name=class_name,
                    train_path=train_path,
                    validation_path=validation_path,
                    test_path=test_path,
                )

                all_metrics.append(metrics)

                # Save the summary table immediately after each cluster is completed.
                summary_df = pd.DataFrame(all_metrics)

                summary_df.to_csv(
                    summary_csv_path,
                    index=False,
                )

                # Tab-separated format for copying to Excel.
                excel_columns = [
                    "class_name",
                    "cluster",
                    "model",
                    "n_train",
                    "n_validation",
                    "n_test",
                    "validation_RMSE",
                    "validation_MAE",
                    "validation_MAPE",
                    "validation_DS",
                    "test_RMSE",
                    "test_MAE",
                    "test_MAPE",
                    "test_DS",
                ]

                existing_excel_columns = [c for c in excel_columns if c in summary_df.columns]
                excel_df = summary_df[existing_excel_columns]

                excel_df.to_csv(
                    excel_txt_path,
                    index=False,
                    sep="\t",
                )

                # Save only the four test metrics for easy copying.
                four_df = summary_df[
                    [
                        "test_RMSE",
                        "test_MAE",
                        "test_MAPE",
                        "test_DS",
                    ]
                ]

                four_df.to_csv(
                    excel_four_only_path,
                    index=False,
                    sep="\t",
                )

            except Exception as e:
                print(f"Training failed: {class_name} - {train_path.name}")
                print(f"Error message: {e}")

                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"[TRAIN_ERROR]\t{class_name}\t{train_path.name}\t{str(e)}\n"
                    )

    print("\n" + "=" * 100)
    print("Training completed for all classes and all clusters.")
    print(f"Summary metrics file: {summary_csv_path}")
    print(f"Excel copy file: {excel_txt_path}")
    print(f"Four test-metric copy file: {excel_four_only_path}")
    print(f"Error log: {error_log_path}")


if __name__ == "__main__":
    main()