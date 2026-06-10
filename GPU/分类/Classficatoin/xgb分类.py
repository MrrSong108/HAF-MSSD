import os
import json
import joblib
import random
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb

from optuna.pruners import MedianPruner
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    balanced_accuracy_score
)

# =========================
# 1. 固定随机种子
# =========================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

print("XGBoost version:", xgb.__version__)


# =========================
# 2. 数据读取
# =========================
train = pd.read_csv(
    "/root/autodl-tmp/airport_project/data/xiao/traff/predict/lstm/classification_train.csv"
)

test = pd.read_csv(
    "/root/autodl-tmp/airport_project/data/xiao/traff/predict/lstm/classification_test.csv"
)

validation = pd.read_csv(
    "/root/autodl-tmp/airport_project/data/xiao/traff/predict/lstm/classification_validation.csv"
)
save_dir = "/root/autodl-tmp/airport_project/data/xiao/traff/class/xgb"
os.makedirs(save_dir, exist_ok=True)
scaler_X = MinMaxScaler()

X_train = scaler_X.fit_transform(
    train.drop(columns='target').fillna(0)
)

X_test = scaler_X.transform(
    test.drop(columns='target').fillna(0)
)

X_validation = scaler_X.transform(
    validation.drop(columns='target').fillna(0)
)

y_train = train['target']
y_test = test['target']
y_validation = validation['target']

print("数据已输入")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("X_validation shape:", X_validation.shape)

# =========================
# 4. Optuna 目标函数
# =========================
def objective(trial):

    params = {

        # 基础参数
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": SEED,
        "verbosity": 0,

        # GPU
        "tree_method": "hist",
        "device": "cuda",

        # 并行
        "n_jobs": 8,

        # Boosting 参数
        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            1000,
            step=100
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.1,
            log=True
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            1.0,
            20.0,
            log=True
        ),

        "gamma": trial.suggest_float(
            "gamma",
            1e-8,
            1.0,
            log=True
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-6,
            10.0,
            log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-6,
            10.0,
            log=True
        )
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_validation, y_validation)
        ],

        verbose=False
    )

    y_pred = model.predict(X_validation)

    acc = balanced_accuracy_score(y_validation, y_pred)

    return 1 - acc


# =========================
# 5. Optuna
# =========================
pruner = MedianPruner(
    n_startup_trials=10,
    n_warmup_steps=5
)

study = optuna.create_study(
    direction='minimize',
    pruner=pruner
)

study.optimize(
    objective,
    n_trials=50,
    n_jobs=1
)

print(f"\n最优Accuracy: {1 - study.best_value:.4f}")
print(f"最佳超参数:\n{study.best_params}")


# =========================
# 6. 最佳参数
# =========================
best_params = study.best_params

final_params = {

    "objective": "binary:logistic",
    "eval_metric": "logloss",

    "random_state": SEED,
    "verbosity": 0,

    "tree_method": "hist",
    "device": "cuda",

    "n_jobs": 8,

    **best_params
}

# 保存参数


with open(
    os.path.join(save_dir, "best_xgb_classifier_params.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(final_params, f, ensure_ascii=False, indent=4)


# =========================
# 7. 最终模型
# =========================
xgb_model = xgb.XGBClassifier(**final_params)

xgb_model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_validation, y_validation)
    ],

    verbose=True
)


# =========================
# 8. 预测
# =========================
y_train_pred = xgb_model.predict(X_train)

y_validation_pred = xgb_model.predict(X_validation)

y_test_pred = xgb_model.predict(X_test)


# =========================
# 9. 输出结果函数
# =========================
def evaluate_model(name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


# =========================
# 10. 评估
# =========================
evaluate_model(
    "TRAIN",
    y_train,
    y_train_pred
)

evaluate_model(
    "VALIDATION",
    y_validation,
    y_validation_pred
)

evaluate_model(
    "TEST",
    y_test,
    y_test_pred
)


# =========================
# 11. 保存模型
# =========================
joblib.dump(
    xgb_model,
    os.path.join(save_dir, "xgb_classifier.pkl")
)

xgb_model.save_model(
    os.path.join(save_dir, "xgb_classifier.json")
)

joblib.dump(
    scaler_X,
    os.path.join(save_dir, "scaler_X.pkl")
)

print("\n模型已保存:", save_dir)