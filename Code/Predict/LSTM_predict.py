import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc
import json
import joblib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# =========================
# 1. GPU configuration
# =========================
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Number of GPUs detected: {len(gpus)}")
        print("GPU devices:", gpus)
    except RuntimeError as e:
        print("GPU configuration failed:", e)
else:
    print("No GPU detected. Running on CPU.")


# =========================
# 2. Basic configuration
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path(os.getenv("DATA_DIR", "data/processed"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/lstm"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_THRESHOLD = float(os.getenv("ERROR_THRESHOLD", "0.1"))


# =========================
# 3. Data loading
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
# 4. Normalization
# =========================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(features_train)
X_test = scaler_X.transform(features_test)
X_validation = scaler_X.transform(features_validation)

y_train = scaler_y.fit_transform(train_Y.values.reshape(-1, 1))
y_test = scaler_y.transform(test_Y.values.reshape(-1, 1))
y_validation = scaler_y.transform(validation_Y.values.reshape(-1, 1))

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
X_validation = X_validation.reshape(X_validation.shape[0], 1, X_validation.shape[1])

print("Data normalization and reshaping completed.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("X_validation shape:", X_validation.shape)


# =========================
# 5. LSTM model construction function
# =========================
def build_lstm_model(
    input_shape,
    lstm_units_1,
    lstm_units_2,
    dropout_rate,
    learning_rate,
    l2_reg,
):
    model = Sequential()

    model.add(
        LSTM(
            lstm_units_1,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg),
            input_shape=input_shape,
        )
    )
    model.add(Dropout(dropout_rate))

    model.add(
        LSTM(
            lstm_units_2,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg),
        )
    )
    model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
    )

    return model


# =========================
# 6. Optuna objective function
# =========================
def objective(trial):
    K.clear_session()
    gc.collect()

    lstm_units_1 = trial.suggest_int("lstm_units_1", 64, 192, step=32)
    lstm_units_2 = trial.suggest_int("lstm_units_2", 32, 96, step=32)

    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.35)

    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    learning_rate = trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True)

    l2_reg = trial.suggest_float("l2_reg", 1e-6, 5e-4, log=True)

    patience = trial.suggest_int("patience", 8, 15)

    max_epochs = 150

    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units_1=lstm_units_1,
        lstm_units_2=lstm_units_2,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        l2_reg=l2_reg,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )

    model.fit(
        X_train,
        y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(X_validation, y_validation),
        verbose=0,
        callbacks=[early_stopping],
    )

    y_val_scaled = model.predict(X_validation, batch_size=batch_size, verbose=0)
    y_val_pred = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1))
    y_val_true = scaler_y.inverse_transform(y_validation.reshape(-1, 1))

    y_val_pred = np.maximum(y_val_pred, 0)

    rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))

    K.clear_session()
    del model
    gc.collect()

    return rmse


# =========================
# 7. Hyperparameter optimization with Optuna
# =========================
study_lstm = optuna.create_study(direction="minimize")

study_lstm.optimize(
    objective,
    n_trials=50,
    n_jobs=1,
)

print("Best validation RMSE:", study_lstm.best_value)
print("Best hyperparameters:", study_lstm.best_params)

best_params = study_lstm.best_params

with open(OUTPUT_DIR / "best_lstm_params.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, ensure_ascii=True, indent=4)


# =========================
# 8. Retrain LSTM model with the best hyperparameters
# =========================
K.clear_session()
gc.collect()

best_model = build_lstm_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    lstm_units_1=best_params["lstm_units_1"],
    lstm_units_2=best_params["lstm_units_2"],
    dropout_rate=best_params["dropout_rate"],
    learning_rate=best_params["learning_rate"],
    l2_reg=best_params["l2_reg"],
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=best_params["patience"],
    restore_best_weights=True,
    verbose=1,
)

history = best_model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=best_params["batch_size"],
    validation_data=(X_validation, y_validation),
    verbose=1,
    callbacks=[early_stopping],
)


# =========================
# 9. Prediction and evaluation
# =========================
def inverse_predict(model, X, batch_size):
    pred_scaled = model.predict(X, batch_size=batch_size, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
    return np.maximum(pred, 0)


y_train_pred = inverse_predict(best_model, X_train, best_params["batch_size"])
y_test_pred = inverse_predict(best_model, X_test, best_params["batch_size"])
y_validation_pred = inverse_predict(best_model, X_validation, best_params["batch_size"])

y_train_true = scaler_y.inverse_transform(y_train.reshape(-1, 1))
y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
y_validation_true = scaler_y.inverse_transform(y_validation.reshape(-1, 1))

train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
validation_rmse = np.sqrt(mean_squared_error(y_validation_true, y_validation_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))

print("Train RMSE:", train_rmse)
print("Validation RMSE:", validation_rmse)
print("Test RMSE:", test_rmse)


# =========================
# 10. Error-based classification label generation
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


train_compared = compare_values(y_train_true, y_train_pred, threshold=ERROR_THRESHOLD)
test_compared = compare_values(y_test_true, y_test_pred, threshold=ERROR_THRESHOLD)
validation_compared = compare_values(y_validation_true, y_validation_pred, threshold=ERROR_THRESHOLD)

unique, counts = np.unique(train_compared, return_counts=True)
distribution = {int(k): int(v) for k, v in zip(unique, counts)}
print("Training label distribution:", distribution)

unique, counts = np.unique(test_compared, return_counts=True)
distribution = {int(k): int(v) for k, v in zip(unique, counts)}
print("Test label distribution:", distribution)

unique, counts = np.unique(validation_compared, return_counts=True)
distribution = {int(k): int(v) for k, v in zip(unique, counts)}
print("Validation label distribution:", distribution)


# =========================
# 11. Save model, scalers, and generated classification data
# =========================
best_model.save(OUTPUT_DIR / "best_lstm_model.keras")
joblib.dump(scaler_X, OUTPUT_DIR / "scaler_X.pkl")
joblib.dump(scaler_y, OUTPUT_DIR / "scaler_y.pkl")

X_train_data = pd.DataFrame(
    X_train.reshape(X_train.shape[0], X_train.shape[2]),
    columns=features_train.columns,
)
X_train_data["target"] = train_compared

X_validation_data = pd.DataFrame(
    X_validation.reshape(X_validation.shape[0], X_validation.shape[2]),
    columns=features_validation.columns,
)
X_validation_data["target"] = validation_compared

X_test_data = pd.DataFrame(
    X_test.reshape(X_test.shape[0], X_test.shape[2]),
    columns=features_test.columns,
)
X_test_data["target"] = test_compared

X_train_data.to_csv(OUTPUT_DIR / "classification_train.csv", index=False)
X_test_data.to_csv(OUTPUT_DIR / "classification_test.csv", index=False)
X_validation_data.to_csv(OUTPUT_DIR / "classification_validation.csv", index=False)

pred_result = pd.DataFrame(
    {
        "test_true": y_test_true.reshape(-1),
        "test_pred": y_test_pred.reshape(-1),
        "test_label": test_compared,
    }
)
pred_result.to_csv(OUTPUT_DIR / "test_prediction_result.csv", index=False)

print("Model, scalers, classification data, and prediction results saved to:", OUTPUT_DIR)