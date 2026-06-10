import os
import json
import joblib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)


# =========================
# 1. Basic configuration
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CLASSIFICATION_DATA_DIR = Path(
    os.getenv("CLASSIFICATION_DATA_DIR", "data/processed/classification")
)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/rf_classifier"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RF_N_JOBS = int(os.getenv("RF_N_JOBS", "-1"))


# =========================
# 2. Path configuration
# =========================
train_path = CLASSIFICATION_DATA_DIR / "classification_train.csv"
test_path = CLASSIFICATION_DATA_DIR / "classification_test.csv"
validation_path = CLASSIFICATION_DATA_DIR / "classification_validation.csv"

for file_path in [train_path, test_path, validation_path]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required data file not found: {file_path}")


# =========================
# 3. Data loading
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
# 4. Optuna objective function
# =========================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            800,
            step=100,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            5,
            30,
            step=5,
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            2,
            20,
            step=2,
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            1,
            10,
            step=1,
        ),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", 0.2, 0.3, 0.4, 0.5, None],
        ),
        "bootstrap": trial.suggest_categorical(
            "bootstrap",
            [True, False],
        ),
        "criterion": trial.suggest_categorical(
            "criterion",
            ["gini", "entropy"],
        ),
        "n_jobs": RF_N_JOBS,
        "random_state": SEED,
    }

    model = RandomForestClassifier(**params)

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_validation)

    balanced_acc = balanced_accuracy_score(y_validation, y_val_pred)

    return 1 - balanced_acc


# =========================
# 5. Hyperparameter optimization with Optuna
# =========================
study_rf_cls = optuna.create_study(direction="minimize")

study_rf_cls.optimize(
    objective,
    n_trials=50,
    n_jobs=1,
)

print(f"\nBest balanced accuracy: {1 - study_rf_cls.best_value:.4f}")
print("Best hyperparameters:")
print(study_rf_cls.best_params)


# =========================
# 6. Save best hyperparameters
# =========================
best_params = study_rf_cls.best_params

final_params = {
    **best_params,
    "n_jobs": RF_N_JOBS,
    "random_state": SEED,
}

with open(
    OUTPUT_DIR / "best_rf_classifier_params.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(final_params, f, ensure_ascii=True, indent=4)


# =========================
# 7. Train the final Random Forest classifier
# =========================
rf_model = RandomForestClassifier(**final_params)

rf_model.fit(X_train, y_train)


# =========================
# 8. Prediction
# =========================
y_train_pred = rf_model.predict(X_train)
y_validation_pred = rf_model.predict(X_validation)
y_test_pred = rf_model.predict(X_test)

y_train_prob = rf_model.predict_proba(X_train)[:, 1]
y_validation_prob = rf_model.predict_proba(X_validation)[:, 1]
y_test_prob = rf_model.predict_proba(X_test)[:, 1]


# =========================
# 9. Evaluation function
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
# 10. Evaluation output
# =========================
metrics = [
    evaluate_model("TRAIN", y_train, y_train_pred),
    evaluate_model("VALIDATION", y_validation, y_validation_pred),
    evaluate_model("TEST", y_test, y_test_pred),
]


# =========================
# 11. Save model, scaler, metrics, and predictions
# =========================
joblib.dump(
    rf_model,
    OUTPUT_DIR / "rf_classifier.pkl",
)

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