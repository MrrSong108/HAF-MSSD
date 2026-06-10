import os
import json
import joblib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================
# 1. Basic configuration
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("XGBoost version:", xgb.__version__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data/processed/traff"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/xgboost"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# For public code, CPU is safer as the default.
# If GPU is available, run:
#   XGBOOST_DEVICE=cuda python train_xgboost_public.py
XGBOOST_DEVICE = os.getenv("XGBOOST_DEVICE", "cpu")
XGBOOST_N_JOBS = int(os.getenv("XGBOOST_N_JOBS", "8"))

ERROR_THRESHOLD = float(os.getenv("ERROR_THRESHOLD", "0.1"))


# =========================
# 2. Data loading
# =========================
predict_columns = [
    "predict_T2_0.5", "predict_T2_1", "predict_T2_1.5", "predict_T2_2", "predict_T2_3",
    "predict_T2_4", "predict_T2_5", "predict_T2_6", "predict_T2_7", "predict_T2_8",
    "predict_T2_9", "predict_T2_10", "predict_T2_11", "predict_T2_12", "predict_T2_13",
    "predict_T2_14", "predict_T2_15", "predict_T2_16", "predict_T2_17", "predict_T2_18",
    "predict_T2_19", "predict_T2_20", "predict_T2_21", "predict_T2_22", "predict_T2_23",
    "predict_T2_24",
    "predict_T3_0.5", "predict_T3_1", "predict_T3_1.5", "predict_T3_2", "predict_T3_3",
    "predict_T3_4", "predict_T3_5", "predict_T3_6", "predict_T3_7", "predict_T3_8",
    "predict_T3_9", "predict_T3_10", "predict_T3_11", "predict_T3_12", "predict_T3_13",
    "predict_T3_14", "predict_T3_15", "predict_T3_16", "predict_T3_17", "predict_T3_18",
    "predict_T3_19", "predict_T3_20", "predict_T3_21", "predict_T3_22", "predict_T3_23",
    "predict_T3_24",
]

train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
validation_path = DATA_DIR / "validation.csv"

for file_path in [train_path, test_path, validation_path]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required data file not found: {file_path}")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
validation = pd.read_csv(validation_path)

target_col = "predict_T2_0.5"

drop_cols = predict_columns + ["start_time"]

features_train = train.drop(columns=drop_cols, errors="ignore").fillna(0)
features_test = test.drop(columns=drop_cols, errors="ignore").fillna(0)
features_validation = validation.drop(columns=drop_cols, errors="ignore").fillna(0)

train_Y = train[target_col]
test_Y = test[target_col]
validation_Y = validation[target_col]

print("Data loaded successfully.")
print("Training feature shape:", features_train.shape)
print("Test feature shape:", features_test.shape)
print("Validation feature shape:", features_validation.shape)


# =========================
# 3. Normalization
# =========================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(features_train)
X_test = scaler_X.transform(features_test)
X_validation = scaler_X.transform(features_validation)

y_train = scaler_y.fit_transform(train_Y.values.reshape(-1, 1))
y_test = scaler_y.transform(test_Y.values.reshape(-1, 1))
y_validation = scaler_y.transform(validation_Y.values.reshape(-1, 1))

print("Normalization completed.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("X_validation shape:", X_validation.shape)


# =========================
# 4. Optuna objective function
# =========================
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": SEED,
        "verbosity": 0,

        # XGBoost 2.x recommended setting.
        # Use XGBOOST_DEVICE=cuda to enable GPU training if available.
        "tree_method": "hist",
        "device": XGBOOST_DEVICE,

        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 9, step=1),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),

        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),

        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),

        # Avoid excessive resource competition when Optuna also uses parallelism.
        "n_jobs": XGBOOST_N_JOBS,
    }

    model = xgb.XGBRegressor(**params)

    model.fit(
        X_train,
        y_train.ravel(),
        eval_set=[(X_validation, y_validation.ravel())],
        verbose=False,
    )

    y_val_scaled = model.predict(X_validation)
    y_val_pred = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1))
    y_val_true = scaler_y.inverse_transform(y_validation.reshape(-1, 1))

    y_val_pred = np.maximum(y_val_pred, 0)

    rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))

    return rmse


# =========================
# 5. Hyperparameter optimization with Optuna
# =========================
study_xgb = optuna.create_study(direction="minimize")

study_xgb.optimize(
    objective,
    n_trials=50,
    n_jobs=1,
)

print("Best validation RMSE:", study_xgb.best_value)
print("Best hyperparameters:", study_xgb.best_params)

best_params = study_xgb.best_params

with open(OUTPUT_DIR / "best_xgboost_params.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, ensure_ascii=True, indent=4)


# =========================
# 6. Train the final XGBoost model with the best hyperparameters
# =========================
final_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": SEED,
    "verbosity": 0,
    "tree_method": "hist",
    "device": XGBOOST_DEVICE,
    "n_jobs": XGBOOST_N_JOBS,
    **best_params,
}

xgb_model = xgb.XGBRegressor(**final_params)

xgb_model.fit(
    X_train,
    y_train.ravel(),
    eval_set=[(X_validation, y_validation.ravel())],
    verbose=True,
)


# =========================
# 7. Prediction and evaluation
# =========================
def inverse_predict(model, X):
    pred_scaled = model.predict(X)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
    return np.maximum(pred, 0)


y_train_pred = inverse_predict(xgb_model, X_train)
y_validation_pred = inverse_predict(xgb_model, X_validation)
y_test_pred = inverse_predict(xgb_model, X_test)

y_train_true = scaler_y.inverse_transform(y_train.reshape(-1, 1))
y_validation_true = scaler_y.inverse_transform(y_validation.reshape(-1, 1))
y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
validation_rmse = np.sqrt(mean_squared_error(y_validation_true, y_validation_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))

train_mae = mean_absolute_error(y_train_true, y_train_pred)
validation_mae = mean_absolute_error(y_validation_true, y_validation_pred)
test_mae = mean_absolute_error(y_test_true, y_test_pred)

print("Train RMSE:", train_rmse)
print("Validation RMSE:", validation_rmse)
print("Test RMSE:", test_rmse)

print("Train MAE:", train_mae)
print("Validation MAE:", validation_mae)
print("Test MAE:", test_mae)


# =========================
# 8. Error-based classification label generation
# =========================
def compare_values(true_values, predicted_values, threshold=0.1):
    result = []

    true_values = np.asarray(true_values).reshape(-1)
    predicted_values = np.asarray(predicted_values).reshape(-1)

    for true, pred in zip(true_values, predicted_values):
        if 0 <= true <= 10:
            if 0 <= pred <= 20:
                result.append(1)
            else:
                result.append(0)
        else:
            ratio = abs((pred - true) / (true + 1e-8))

            if ratio > threshold:
                result.append(0)
            else:
                result.append(1)

    return result


# Use 0.1 for 10% error-based classification.
# Use 0.2 for a 20% sensitivity experiment.
train_compared = compare_values(y_train_true, y_train_pred, threshold=ERROR_THRESHOLD)
validation_compared = compare_values(y_validation_true, y_validation_pred, threshold=ERROR_THRESHOLD)
test_compared = compare_values(y_test_true, y_test_pred, threshold=ERROR_THRESHOLD)

unique, counts = np.unique(train_compared, return_counts=True)
distribution = {int(k): int(v) for k, v in zip(unique, counts)}
print("Training label distribution:", distribution)

unique, counts = np.unique(validation_compared, return_counts=True)
distribution = {int(k): int(v) for k, v in zip(unique, counts)}
print("Validation label distribution:", distribution)

unique, counts = np.unique(test_compared, return_counts=True)
distribution = {int(k): int(v) for k, v in zip(unique, counts)}
print("Test label distribution:", distribution)


# =========================
# 9. Save model, scalers, and generated results
# =========================
joblib.dump(xgb_model, OUTPUT_DIR / "best_xgboost_model.pkl")
xgb_model.save_model(OUTPUT_DIR / "best_xgboost_model.json")

joblib.dump(scaler_X, OUTPUT_DIR / "scaler_X.pkl")
joblib.dump(scaler_y, OUTPUT_DIR / "scaler_y.pkl")

X_train_data = pd.DataFrame(X_train, columns=features_train.columns)
X_train_data["target"] = train_compared

X_validation_data = pd.DataFrame(X_validation, columns=features_validation.columns)
X_validation_data["target"] = validation_compared

X_test_data = pd.DataFrame(X_test, columns=features_test.columns)
X_test_data["target"] = test_compared

X_train_data.to_csv(OUTPUT_DIR / "classification_train.csv", index=False)
X_validation_data.to_csv(OUTPUT_DIR / "classification_validation.csv", index=False)
X_test_data.to_csv(OUTPUT_DIR / "classification_test.csv", index=False)

pred_result = pd.DataFrame(
    {
        "test_true": y_test_true.reshape(-1),
        "test_pred": y_test_pred.reshape(-1),
        "test_label": test_compared,
    }
)
pred_result.to_csv(OUTPUT_DIR / "test_prediction_result.csv", index=False)

print("XGBoost model, scalers, classification data, and prediction results saved to:", OUTPUT_DIR)
