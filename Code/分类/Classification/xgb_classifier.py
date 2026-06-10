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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)


# =========================
# 1. Basic configuration
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("XGBoost version:", xgb.__version__)

CLASSIFICATION_DATA_DIR = Path(
    os.getenv("CLASSIFICATION_DATA_DIR", "data/processed/classification")
)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/xgboost_classifier"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

XGBOOST_DEVICE = os.getenv("XGBOOST_DEVICE", "cpu")
XGBOOST_N_JOBS = int(os.getenv("XGBOOST_N_JOBS", "8"))


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
print("X_test shape:", X_test.shape)
print("X_validation shape:", X_validation.shape)

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
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": SEED,
        "verbosity": 0,

        "tree_method": "hist",
        "device": XGBOOST_DEVICE,

        "n_jobs": XGBOOST_N_JOBS,

        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            1000,
            step=100,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.1,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            1.0,
            20.0,
            log=True,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            1e-8,
            1.0,
            log=True,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-6,
            10.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-6,
            10.0,
            log=True,
        ),
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )

    y_val_pred = model.predict(X_validation)

    balanced_acc = balanced_accuracy_score(y_validation, y_val_pred)

    return 1 - balanced_acc


# =========================
# 5. Hyperparameter optimization with Optuna
# =========================
study_xgb_cls = optuna.create_study(direction="minimize")

study_xgb_cls.optimize(
    objective,
    n_trials=50,
    n_jobs=1,
)

print(f"\nBest balanced accuracy: {1 - study_xgb_cls.best_value:.4f}")
print("Best hyperparameters:")
print(study_xgb_cls.best_params)


# =========================
# 6. Save best hyperparameters
# =========================
best_params = study_xgb_cls.best_params

final_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": SEED,
    "verbosity": 0,
    "tree_method": "hist",
    "device": XGBOOST_DEVICE,
    "n_jobs": XGBOOST_N_JOBS,
    **best_params,
}

with open(
    OUTPUT_DIR / "best_xgboost_classifier_params.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(final_params, f, ensure_ascii=True, indent=4)


# =========================
# 7. Train the final XGBoost classifier
# =========================
xgb_model = xgb.XGBClassifier(**final_params)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation, y_validation)],
    verbose=True,
)


# =========================
# 8. Prediction
# =========================
def get_positive_class_probability(model, X):
    probabilities = model.predict_proba(X)

    if probabilities.shape[1] == 1:
        if model.classes_[0] == 1:
            return probabilities[:, 0]
        return np.zeros(probabilities.shape[0])

    positive_class_index = list(model.classes_).index(1)
    return probabilities[:, positive_class_index]


y_train_pred = xgb_model.predict(X_train)
y_validation_pred = xgb_model.predict(X_validation)
y_test_pred = xgb_model.predict(X_test)

y_train_prob = get_positive_class_probability(xgb_model, X_train)
y_validation_prob = get_positive_class_probability(xgb_model, X_validation)
y_test_prob = get_positive_class_probability(xgb_model, X_test)


# =========================
# 9. Evaluation function
# =========================
def evaluate_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"F1 score: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "dataset": name.lower(),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "f1_score": float(f1),
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
    xgb_model,
    OUTPUT_DIR / "xgboost_classifier.pkl",
)

xgb_model.save_model(
    OUTPUT_DIR / "xgboost_classifier.json",
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