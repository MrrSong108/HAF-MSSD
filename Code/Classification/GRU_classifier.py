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
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)


# =========================
# 1. Basic configuration
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASSIFICATION_DATA_DIR = Path(
    os.getenv("CLASSIFICATION_DATA_DIR", "data/processed/classification")
)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/gru_classifier"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. GPU configuration
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
# 3. Path configuration
# =========================
train_path = CLASSIFICATION_DATA_DIR / "classification_train.csv"
test_path = CLASSIFICATION_DATA_DIR / "classification_test.csv"
validation_path = CLASSIFICATION_DATA_DIR / "classification_validation.csv"

for file_path in [train_path, test_path, validation_path]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required data file not found: {file_path}")


# =========================
# 4. Data loading
# =========================
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
validation = pd.read_csv(validation_path)

required_column = "target"
for name, dataset in {
    "train": train,
    "validation": validation,
    "test": test,
}.items():
    if required_column not in dataset.columns:
        raise ValueError(f"The '{required_column}' column is missing in the {name} dataset.")

scaler_X = MinMaxScaler()

X_train = scaler_X.fit_transform(
    train.drop(columns="target").fillna(0)
).astype(np.float32)

X_test = scaler_X.transform(
    test.drop(columns="target").fillna(0)
).astype(np.float32)

X_validation = scaler_X.transform(
    validation.drop(columns="target").fillna(0)
).astype(np.float32)

y_train = train["target"].astype(int).values
y_test = test["target"].astype(int).values
y_validation = validation["target"].astype(int).values

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
X_validation = X_validation.reshape(X_validation.shape[0], 1, X_validation.shape[1])

print("Data loaded successfully.")
print("X_train shape:", X_train.shape)
print("X_validation shape:", X_validation.shape)
print("X_test shape:", X_test.shape)

print("\nTraining label distribution:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nValidation label distribution:")
print(pd.Series(y_validation).value_counts().sort_index())

print("\nTest label distribution:")
print(pd.Series(y_test).value_counts().sort_index())


# =========================
# 5. GRU model construction function
# =========================
def build_gru_model(
    input_shape,
    gru_units_1,
    gru_units_2,
    dense_units,
    dropout_rate,
    learning_rate,
    l2_reg,
):
    model = Sequential()

    model.add(
        GRU(
            gru_units_1,
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
        GRU(
            gru_units_2,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg),
        )
    )
    model.add(Dropout(dropout_rate))

    model.add(
        Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=l2(l2_reg),
        )
    )
    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =========================
# 6. Optuna objective function
# =========================
def objective(trial):
    K.clear_session()
    gc.collect()

    gru_units_1 = trial.suggest_int("gru_units_1", 64, 192, step=32)
    gru_units_2 = trial.suggest_int("gru_units_2", 32, 96, step=32)

    dense_units = trial.suggest_categorical("dense_units", [32, 64, 128])

    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.35)

    learning_rate = trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True)

    l2_reg = trial.suggest_float("l2_reg", 1e-6, 5e-4, log=True)

    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    patience = trial.suggest_int("patience", 8, 15)

    max_epochs = 150

    model = build_gru_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        gru_units_1=gru_units_1,
        gru_units_2=gru_units_2,
        dense_units=dense_units,
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

    y_val_prob = model.predict(
        X_validation,
        batch_size=batch_size,
        verbose=0,
    ).reshape(-1)

    y_val_pred = (y_val_prob >= 0.5).astype(int)

    balanced_acc = balanced_accuracy_score(y_validation, y_val_pred)

    K.clear_session()
    del model
    gc.collect()

    return 1 - balanced_acc


# =========================
# 7. Hyperparameter optimization with Optuna
# =========================
study_gru_cls = optuna.create_study(direction="minimize")

study_gru_cls.optimize(
    objective,
    n_trials=50,
    n_jobs=1,
)

print(f"\nBest balanced accuracy: {1 - study_gru_cls.best_value:.4f}")
print("Best hyperparameters:")
print(study_gru_cls.best_params)


# =========================
# 8. Save best hyperparameters
# =========================
best_params = study_gru_cls.best_params

with open(
    OUTPUT_DIR / "best_gru_classifier_params.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(best_params, f, ensure_ascii=True, indent=4)


# =========================
# 9. Train the final GRU classifier with the best hyperparameters
# =========================
K.clear_session()
gc.collect()

best_model = build_gru_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    gru_units_1=best_params["gru_units_1"],
    gru_units_2=best_params["gru_units_2"],
    dense_units=best_params["dense_units"],
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
# 10. Prediction
# =========================
def predict_label(model, X, batch_size):
    prob = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
    pred = (prob >= 0.5).astype(int)
    return prob, pred


y_train_prob, y_train_pred = predict_label(
    best_model,
    X_train,
    best_params["batch_size"],
)

y_validation_prob, y_validation_pred = predict_label(
    best_model,
    X_validation,
    best_params["batch_size"],
)

y_test_prob, y_test_pred = predict_label(
    best_model,
    X_test,
    best_params["batch_size"],
)


# =========================
# 11. Evaluation function
# =========================
def evaluate_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "dataset": name.lower(),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
    }


# =========================
# 12. Evaluation output
# =========================
metrics = [
    evaluate_model("TRAIN", y_train, y_train_pred),
    evaluate_model("VALIDATION", y_validation, y_validation_pred),
    evaluate_model("TEST", y_test, y_test_pred),
]


# =========================
# 13. Save model, scaler, metrics, and predictions
# =========================
best_model.save(OUTPUT_DIR / "gru_classifier.keras")

joblib.dump(
    scaler_X,
    OUTPUT_DIR / "scaler_X.pkl",
)

metrics_result = pd.DataFrame(metrics)
metrics_result.to_csv(OUTPUT_DIR / "classification_metrics.csv", index=False)


def save_prediction_result(file_name, y_true, y_prob, y_pred):
    result = pd.DataFrame(
        {
            "true_label": y_true,
            "pred_prob": y_prob,
            "pred_label": y_pred,
        }
    )
    result.to_csv(OUTPUT_DIR / file_name, index=False)


save_prediction_result(
    "train_prediction_result.csv",
    y_train,
    y_train_prob,
    y_train_pred,
)

save_prediction_result(
    "validation_prediction_result.csv",
    y_validation,
    y_validation_prob,
    y_validation_pred,
)

save_prediction_result(
    "test_prediction_result.csv",
    y_test,
    y_test_prob,
    y_test_pred,
)

print("\nModel, scaler, metrics, and prediction results saved to:", OUTPUT_DIR)