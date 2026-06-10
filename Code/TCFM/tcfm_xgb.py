import os
import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =========================================================
# 1. Path and basic configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "data/processed"))

TRAIN_PATH = Path(os.getenv("TRAIN_PATH", DATA_DIR / "train.csv"))
VALIDATION_PATH = Path(os.getenv("VALIDATION_PATH", DATA_DIR / "validation.csv"))
TEST_PATH = Path(os.getenv("TEST_PATH", DATA_DIR / "test.csv"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/tcfm_euclidean_xgboost_optuna"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = os.getenv("TARGET_COL", "predict_T2_0.5")
TIME_COL = os.getenv("TIME_COL", "start_time")

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Search range for the number of KMeans clusters.
K_MIN = int(os.getenv("K_MIN", "2"))
K_MAX = int(os.getenv("K_MAX", "20"))

MIN_CLUSTER_SAMPLES = int(os.getenv("MIN_CLUSTER_SAMPLES", "100"))

N_TRIALS = int(os.getenv("N_TRIALS", "50"))

# Time-series clustering configuration.
CLUSTER_METRIC = os.getenv("CLUSTER_METRIC", "euclidean")
CLUSTER_FEATURE_PREFIX = os.getenv("CLUSTER_FEATURE_PREFIX", "queue_countpassed")
KMEANS_N_JOBS = int(os.getenv("KMEANS_N_JOBS", "4"))

# XGBoost parallel jobs.
XGB_N_JOBS = int(os.getenv("XGB_N_JOBS", "8"))

XGBOOST_DEVICE = os.getenv("XGBOOST_DEVICE", "cpu")

SAVE_DETAILED_OUTPUTS = os.getenv("SAVE_DETAILED_OUTPUTS", "0") == "1"


# =========================================================
# 2. File validation
# =========================================================
for file_path in [TRAIN_PATH, VALIDATION_PATH, TEST_PATH]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required data file not found: {file_path}")

if VALIDATION_PATH.resolve() == TEST_PATH.resolve():
    raise ValueError(
        "Validation and test files must be different. "
        "Using the test set as the validation set causes data leakage."
    )


# =========================================================
# 3. Evaluation metrics
# =========================================================
def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1))

    if len(y_true) > 1:
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        ds = np.mean(np.sign(true_diff) == np.sign(pred_diff))
    else:
        ds = np.nan

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "DS": float(ds),
    }


# =========================================================
# 4. Data loading
# =========================================================
def read_data(path):
    df = pd.read_csv(path)

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.sort_values(TIME_COL).reset_index(drop=True)

    return df


print("Loading data...")

train_df = read_data(TRAIN_PATH)
valid_df = read_data(VALIDATION_PATH)
test_df = read_data(TEST_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {valid_df.shape}")
print(f"Test shape: {test_df.shape}")


# =========================================================
# 5. Basic checks
# =========================================================
for name, df in [("train", train_df), ("validation", valid_df), ("test", test_df)]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"The target column '{TARGET_COL}' is missing in the {name} dataset.")

if TIME_COL not in train_df.columns:
    print(f"Warning: time column '{TIME_COL}' was not found. Data will be processed in the original order.")


# =========================================================
# 6. Remove invalid columns and target-leakage columns
# =========================================================
def build_feature_data(train_df, valid_df, test_df, target_col):
    """
    Remove:
    1. The time column.
    2. All predict_* target columns to avoid target leakage.
    3. Columns that are entirely missing in the training set.
    4. Constant columns in the training set.

    Keep valid multi-source features for prediction.
    """

    predict_cols = [c for c in train_df.columns if c.startswith("predict_")]

    drop_cols = set(predict_cols)
    drop_cols.add(target_col)

    if TIME_COL in train_df.columns:
        drop_cols.add(TIME_COL)

    drop_cols = [c for c in drop_cols if c in train_df.columns]

    y_train = train_df[target_col].copy()
    y_valid = valid_df[target_col].copy()
    y_test = test_df[target_col].copy()

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
    X_test = test_df.drop(columns=drop_cols, errors="ignore")

    common_cols = sorted(list(set(X_train.columns) & set(X_valid.columns) & set(X_test.columns)))

    X_train = X_train[common_cols]
    X_valid = X_valid[common_cols]
    X_test = X_test[common_cols]

    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        print(f"Number of all-empty columns removed from training data: {len(all_nan_cols)}")
        X_train = X_train.drop(columns=all_nan_cols)
        X_valid = X_valid.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)

    nunique = X_train.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"Number of constant columns removed from training data: {len(constant_cols)}")
        X_train = X_train.drop(columns=constant_cols)
        X_valid = X_valid.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)

    fill_values = X_train.mean(numeric_only=True)

    X_train = X_train.fillna(fill_values).fillna(0)
    X_valid = X_valid.fillna(fill_values).fillna(0)
    X_test = X_test.fillna(fill_values).fillna(0)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


X_train, X_valid, X_test, y_train, y_valid, y_test = build_feature_data(
    train_df,
    valid_df,
    test_df,
    TARGET_COL,
)

print(f"Processed X_train shape: {X_train.shape}")
print(f"Processed X_valid shape: {X_valid.shape}")
print(f"Processed X_test shape: {X_test.shape}")


# =========================================================
# 7. Select clustering features
# =========================================================
def get_cluster_columns(columns):
    """
    Use historical queue-related features as the clustering basis.
    """

    cluster_cols = [
        c for c in columns
        if c.startswith(CLUSTER_FEATURE_PREFIX) and c != CLUSTER_FEATURE_PREFIX
    ]

    return cluster_cols


cluster_cols = get_cluster_columns(X_train.columns)

if len(cluster_cols) == 0:
    raise ValueError(
        f"No clustering features were found with prefix '{CLUSTER_FEATURE_PREFIX}'. "
        "Please check the feature names or set CLUSTER_FEATURE_PREFIX."
    )

print(f"Number of clustering features: {len(cluster_cols)}")

X_train_cluster = X_train[cluster_cols].copy()
X_valid_cluster = X_valid[cluster_cols].copy()
X_test_cluster = X_test[cluster_cols].copy()


# =========================================================
# 8. Convert clustering input to time-series format
# =========================================================
print(f"\nTime-series clustering metric: {CLUSTER_METRIC}")

X_train_cluster_ts = X_train_cluster.values.reshape(
    X_train_cluster.shape[0],
    X_train_cluster.shape[1],
    1,
)

X_valid_cluster_ts = X_valid_cluster.values.reshape(
    X_valid_cluster.shape[0],
    X_valid_cluster.shape[1],
    1,
)

X_test_cluster_ts = X_test_cluster.values.reshape(
    X_test_cluster.shape[0],
    X_test_cluster.shape[1],
    1,
)

ts_scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)

X_train_cluster_scaled = ts_scaler.fit_transform(X_train_cluster_ts)
X_valid_cluster_scaled = ts_scaler.transform(X_valid_cluster_ts)
X_test_cluster_scaled = ts_scaler.transform(X_test_cluster_ts)


# =========================================================
# 9. Search for the optimal number of clusters using DBI
# =========================================================
print("\nSearching for the optimal number of clusters with TimeSeriesKMeans...")

dbi_records = []

X_train_flat = X_train_cluster_scaled.reshape(
    X_train_cluster_scaled.shape[0],
    -1,
)


def format_cluster_counts(labels):
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    return {int(k): int(v) for k, v in counts.items()}


for k in range(K_MIN, K_MAX + 1):
    print(f"Evaluating K={k}, metric={CLUSTER_METRIC} ...")

    ts_kmeans = TimeSeriesKMeans(
        n_clusters=k,
        metric=CLUSTER_METRIC,
        max_iter=50,
        n_init=3,
        random_state=RANDOM_STATE,
        verbose=False,
        n_jobs=KMEANS_N_JOBS,
    )

    train_labels = ts_kmeans.fit_predict(X_train_cluster_scaled)

    dbi = davies_bouldin_score(X_train_flat, train_labels)
    cluster_counts = format_cluster_counts(train_labels)

    dbi_records.append(
        {
            "K": int(k),
            "DBI": float(dbi),
            "cluster_counts": cluster_counts,
        }
    )

    print(f"K={k}, DBI={dbi:.6f}, cluster_counts={cluster_counts}")


dbi_df = pd.DataFrame(
    [
        {
            "K": r["K"],
            "DBI": r["DBI"],
            "cluster_counts": json.dumps(r["cluster_counts"], ensure_ascii=True),
        }
        for r in dbi_records
    ]
)

dbi_path = OUTPUT_DIR / f"dbi_search_results_{CLUSTER_METRIC}.csv"
dbi_df.to_csv(dbi_path, index=False)

best_record = min(dbi_records, key=lambda x: x["DBI"])
best_k = best_record["K"]

print("\nCluster number search completed.")
print(f"Best metric = {CLUSTER_METRIC}")
print(f"Best K = {best_k}")
print(f"Best DBI = {best_record['DBI']:.6f}")
print(f"Best cluster counts = {best_record['cluster_counts']}")


# =========================================================
# 10. Train the final TimeSeriesKMeans model
# =========================================================
print("\nTraining the final TimeSeriesKMeans clustering model...")

final_kmeans = TimeSeriesKMeans(
    n_clusters=best_k,
    metric=CLUSTER_METRIC,
    max_iter=50,
    n_init=5,
    random_state=RANDOM_STATE,
    verbose=False,
    n_jobs=KMEANS_N_JOBS,
)

train_cluster_labels = final_kmeans.fit_predict(X_train_cluster_scaled)
valid_cluster_labels = final_kmeans.predict(X_valid_cluster_scaled)
test_cluster_labels = final_kmeans.predict(X_test_cluster_scaled)

print("Training cluster distribution:")
print(pd.Series(train_cluster_labels).value_counts().sort_index())

print("Validation cluster distribution:")
print(pd.Series(valid_cluster_labels).value_counts().sort_index())

print("Test cluster distribution:")
print(pd.Series(test_cluster_labels).value_counts().sort_index())


# =========================================================
# 11. Optional: save cluster labels
# =========================================================
if SAVE_DETAILED_OUTPUTS:
    cluster_label_df_train = pd.DataFrame(
        {
            "dataset": "train",
            "cluster": train_cluster_labels,
            "y_true": y_train.values,
        }
    )

    cluster_label_df_valid = pd.DataFrame(
        {
            "dataset": "validation",
            "cluster": valid_cluster_labels,
            "y_true": y_valid.values,
        }
    )

    cluster_label_df_test = pd.DataFrame(
        {
            "dataset": "test",
            "cluster": test_cluster_labels,
            "y_true": y_test.values,
        }
    )

    if TIME_COL in train_df.columns:
        cluster_label_df_train[TIME_COL] = train_df[TIME_COL].values
        cluster_label_df_valid[TIME_COL] = valid_df[TIME_COL].values
        cluster_label_df_test[TIME_COL] = test_df[TIME_COL].values

    cluster_label_all = pd.concat(
        [cluster_label_df_train, cluster_label_df_valid, cluster_label_df_test],
        axis=0,
        ignore_index=True,
    )

    cluster_label_all.to_csv(
        OUTPUT_DIR / "cluster_labels.csv",
        index=False,
    )


# =========================================================
# 12. XGBoost configuration
# =========================================================
BASE_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "gamma": 1e-5,
    "reg_alpha": 1e-6,
    "reg_lambda": 1e-6,
}


def suggest_xgb_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def create_xgb_model(params):
    model = XGBRegressor(
        **params,
        objective="reg:squarederror",
        tree_method="hist",
        device=XGBOOST_DEVICE,
        random_state=RANDOM_STATE,
        n_jobs=XGB_N_JOBS,
    )

    return model


def optimize_xgb_for_cluster(
    X_c_train,
    y_c_train,
    X_c_valid,
    y_c_valid,
    cluster_id,
    n_trials=N_TRIALS,
):
    """
    Optimize XGBoost hyperparameters within a single cluster.
    The objective is to minimize validation RMSE.
    """

    def objective(trial):
        params = suggest_xgb_params(trial)

        model = create_xgb_model(params)
        model.fit(X_c_train, y_c_train)

        valid_pred = model.predict(X_c_valid)
        rmse = np.sqrt(mean_squared_error(y_c_valid, valid_pred))

        return rmse

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"xgb_cluster_{cluster_id}",
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_rmse = study.best_value

    print(f"Cluster {cluster_id} Optuna best RMSE: {best_rmse:.6f}")
    print(f"Cluster {cluster_id} Optuna best params: {best_params}")

    return best_params, best_rmse, study


# =========================================================
# 13. Train cluster-specific XGBoost models
# =========================================================
print("\nTraining cluster-specific XGBoost models...")

cluster_models = {}
cluster_best_params = {}
cluster_optuna_records = []
cluster_metrics_records = []

# Global fallback model used when a cluster has too few samples
# or has no validation samples.
print("Training the global fallback XGBoost model...")

global_model = create_xgb_model(BASE_XGB_PARAMS)
global_model.fit(X_train, y_train)

for cluster_id in range(best_k):
    print(f"\n========== Training Cluster {cluster_id} ==========")

    train_idx = np.where(train_cluster_labels == cluster_id)[0]
    valid_idx = np.where(valid_cluster_labels == cluster_id)[0]

    n_train_cluster = len(train_idx)
    n_valid_cluster = len(valid_idx)

    print(f"Cluster {cluster_id} training samples: {n_train_cluster}")
    print(f"Cluster {cluster_id} validation samples: {n_valid_cluster}")

    if n_train_cluster < MIN_CLUSTER_SAMPLES:
        print(
            f"Cluster {cluster_id} has fewer than {MIN_CLUSTER_SAMPLES} training samples. "
            "The global fallback model will be used."
        )

        cluster_models[cluster_id] = global_model
        cluster_best_params[cluster_id] = BASE_XGB_PARAMS.copy()

        cluster_optuna_records.append(
            {
                "cluster": int(cluster_id),
                "train_samples": int(n_train_cluster),
                "validation_samples": int(n_valid_cluster),
                "status": "fallback_small_train_cluster",
                "best_rmse": None,
                "best_params": json.dumps(BASE_XGB_PARAMS, ensure_ascii=True),
            }
        )

        continue

    X_c_train = X_train.iloc[train_idx]
    y_c_train = y_train.iloc[train_idx]

    if n_valid_cluster == 0:
        print(
            f"Cluster {cluster_id} has no validation samples. "
            "The base XGBoost parameters will be used."
        )

        model = create_xgb_model(BASE_XGB_PARAMS)
        model.fit(X_c_train, y_c_train)

        cluster_models[cluster_id] = model
        cluster_best_params[cluster_id] = BASE_XGB_PARAMS.copy()

        cluster_optuna_records.append(
            {
                "cluster": int(cluster_id),
                "train_samples": int(n_train_cluster),
                "validation_samples": int(n_valid_cluster),
                "status": "base_params_no_validation_samples",
                "best_rmse": None,
                "best_params": json.dumps(BASE_XGB_PARAMS, ensure_ascii=True),
            }
        )

        continue

    X_c_valid = X_valid.iloc[valid_idx]
    y_c_valid = y_valid.iloc[valid_idx]

    best_params, best_rmse, study = optimize_xgb_for_cluster(
        X_c_train=X_c_train,
        y_c_train=y_c_train,
        X_c_valid=X_c_valid,
        y_c_valid=y_c_valid,
        cluster_id=cluster_id,
        n_trials=N_TRIALS,
    )

    cluster_best_params[cluster_id] = best_params

    cluster_optuna_records.append(
        {
            "cluster": int(cluster_id),
            "train_samples": int(n_train_cluster),
            "validation_samples": int(n_valid_cluster),
            "status": "optuna_success",
            "best_rmse": float(best_rmse),
            "best_params": json.dumps(best_params, ensure_ascii=True),
        }
    )

    if SAVE_DETAILED_OUTPUTS:
        trial_df = study.trials_dataframe()
        trial_df.to_csv(
            OUTPUT_DIR / f"optuna_trials_cluster_{cluster_id}.csv",
            index=False,
        )

    model = create_xgb_model(best_params)
    model.fit(X_c_train, y_c_train)

    cluster_models[cluster_id] = model

    valid_pred = model.predict(X_c_valid)
    metrics = calc_metrics(y_c_valid, valid_pred)

    record = {
        "cluster": int(cluster_id),
        "train_samples": int(n_train_cluster),
        "validation_samples": int(n_valid_cluster),
        **metrics,
    }

    cluster_metrics_records.append(record)

    print(
        f"Cluster {cluster_id} Validation - "
        f"RMSE: {metrics['RMSE']:.4f}, "
        f"MAE: {metrics['MAE']:.4f}, "
        f"MAPE: {metrics['MAPE']:.4f}, "
        f"DS: {metrics['DS']:.4f}"
    )


cluster_metrics_df = pd.DataFrame(cluster_metrics_records)
cluster_metrics_df.to_csv(
    OUTPUT_DIR / "cluster_validation_metrics.csv",
    index=False,
)

cluster_optuna_df = pd.DataFrame(cluster_optuna_records)
cluster_optuna_df.to_csv(
    OUTPUT_DIR / "cluster_optuna_best_params.csv",
    index=False,
)


# =========================================================
# 14. Cluster-based fusion prediction
# =========================================================
def predict_by_cluster_models(X, cluster_labels, cluster_models):
    """
    Predict each sample using the XGBoost model corresponding to its assigned cluster.
    """
    y_pred = np.zeros(len(X), dtype=float)

    for cluster_id in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cluster_id)[0]

        model = cluster_models.get(cluster_id, global_model)

        X_part = X.iloc[idx]
        y_pred[idx] = model.predict(X_part)

    return y_pred


# =========================================================
# 15. Prediction and evaluation
# =========================================================
print("\nRunning TCFM-XGBoost fusion prediction...")

train_pred = predict_by_cluster_models(X_train, train_cluster_labels, cluster_models)
valid_pred = predict_by_cluster_models(X_valid, valid_cluster_labels, cluster_models)
test_pred = predict_by_cluster_models(X_test, test_cluster_labels, cluster_models)

train_metrics = calc_metrics(y_train, train_pred)
valid_metrics = calc_metrics(y_valid, valid_pred)
test_metrics = calc_metrics(y_test, test_pred)

print("\n========== Final TCFM-XGBoost-Optuna Results ==========")
print("Train:", train_metrics)
print("Validation:", valid_metrics)
print("Test:", test_metrics)


# =========================================================
# 16. Save overall metrics
# =========================================================
overall_metrics_df = pd.DataFrame(
    [
        {"dataset": "train", **train_metrics},
        {"dataset": "validation", **valid_metrics},
        {"dataset": "test", **test_metrics},
    ]
)

overall_metrics_df.to_csv(
    OUTPUT_DIR / "overall_metrics.csv",
    index=False,
)


# =========================================================
# 17. Optional: save test predictions
# =========================================================
if SAVE_DETAILED_OUTPUTS:
    test_result_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": test_pred,
            "cluster": test_cluster_labels,
        }
    )

    if TIME_COL in test_df.columns:
        test_result_df.insert(0, TIME_COL, test_df[TIME_COL].values)

    test_result_df["abs_error"] = np.abs(test_result_df["y_true"] - test_result_df["y_pred"])
    test_result_df["ape"] = test_result_df["abs_error"] / (np.abs(test_result_df["y_true"]) + 1)

    test_result_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )


# =========================================================
# 18. Save models and configuration
# =========================================================
joblib.dump(ts_scaler, OUTPUT_DIR / f"timeseries_scaler_{CLUSTER_METRIC}.pkl")
joblib.dump(final_kmeans, OUTPUT_DIR / f"timeseries_kmeans_{CLUSTER_METRIC}.pkl")
joblib.dump(cluster_models, OUTPUT_DIR / "cluster_xgboost_models.pkl")
joblib.dump(global_model, OUTPUT_DIR / "global_xgboost_model.pkl")
joblib.dump(cluster_best_params, OUTPUT_DIR / "cluster_xgboost_best_params.pkl")

config = {
    "target_col": TARGET_COL,
    "time_col": TIME_COL,
    "best_k": int(best_k),
    "best_dbi": float(best_record["DBI"]),
    "cluster_metric": CLUSTER_METRIC,
    "cluster_model": "TimeSeriesKMeans",
    "cluster_feature_prefix": CLUSTER_FEATURE_PREFIX,
    "num_cluster_features": int(len(cluster_cols)),
    "num_prediction_features": int(X_train.shape[1]),
    "k_search_range": [int(K_MIN), int(K_MAX)],
    "min_cluster_samples": int(MIN_CLUSTER_SAMPLES),
    "prediction_model": "XGBRegressor",
    "optuna_trials": int(N_TRIALS),
    "save_detailed_outputs": bool(SAVE_DETAILED_OUTPUTS),
    "base_xgb_params": BASE_XGB_PARAMS
}

with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=True, indent=4)


print("\nAll results have been saved to:")
print(OUTPUT_DIR)