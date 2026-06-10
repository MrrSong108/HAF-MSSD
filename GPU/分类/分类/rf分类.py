import os
import json
import joblib
import random
import numpy as np
import pandas as pd
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# =========================
# 1. 固定随机种子
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =========================
# 2. 路径设置
# =========================
data_dir = "/root/autodl-tmp/airport_project/data/xiao/air/predict/lstm"

train_path = os.path.join(data_dir, "classification_train.csv")
test_path = os.path.join(data_dir, "classification_test.csv")
validation_path = os.path.join(data_dir, "classification_validation.csv")

save_dir = "/root/autodl-tmp/airport_project/data/xiao/traff/class/rf"
os.makedirs(save_dir, exist_ok=True)


# =========================
# 3. 数据读取
# =========================
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
validation = pd.read_csv(validation_path)

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

y_train = train["target"].astype(int)
y_test = test["target"].astype(int)
y_validation = validation["target"].astype(int)

print("数据已输入")
print("X_train shape:", X_train.shape)
print("X_validation shape:", X_validation.shape)
print("X_test shape:", X_test.shape)

print("\n训练集类别分布:")
print(y_train.value_counts().sort_index())

print("\n验证集类别分布:")
print(y_validation.value_counts().sort_index())

print("\n测试集类别分布:")
print(y_test.value_counts().sort_index())

# =========================
# 5. Optuna 目标函数
# =========================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            800,
            step=100
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            5,
            30,
            step=5
        ),

        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            2,
            20,
            step=2
        ),

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            1,
            10,
            step=1
        ),

        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", 0.2, 0.3, 0.4, 0.5, None]
        ),

        "bootstrap": trial.suggest_categorical(
            "bootstrap",
            [True, False]
        ),

        "criterion": trial.suggest_categorical(
            "criterion",
            ["gini", "entropy"]
        ),

        "n_jobs": -1,
        "random_state": SEED
    }

    model = RandomForestClassifier(**params)

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_validation)

    acc = accuracy_score(y_validation, y_val_pred)

    return 1 - acc


# =========================
# 6. Optuna 调参
# =========================
study_rf_cls = optuna.create_study(direction="minimize")

study_rf_cls.optimize(
    objective,
    n_trials=50,
    n_jobs=1
)

print(f"\n最优Accuracy: {1 - study_rf_cls.best_value:.4f}")
print("最佳超参数:")
print(study_rf_cls.best_params)


# =========================
# 7. 保存最佳参数
# =========================
best_params = study_rf_cls.best_params

final_params = {
    **best_params,
    "n_jobs": 8,
    "random_state": SEED
}

with open(
    os.path.join(save_dir, "best_rf_classifier_params.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(final_params, f, ensure_ascii=False, indent=4)

# best_params = {'n_estimators': 600, 'max_depth': 25, 'min_samples_split': 20, 'min_samples_leaf': 1, 'max_features': 0.4, 'bootstrap': False, 'criterion': 'gini'}
final_params = {
    **best_params,
    "n_jobs": 8,
    "random_state": SEED
}
# =========================
# 8. 训练最终模型
# =========================
rf_model = RandomForestClassifier(**final_params)

rf_model.fit(X_train, y_train)


# =========================
# 9. 预测
# =========================
y_train_pred = rf_model.predict(X_train)
y_validation_pred = rf_model.predict(X_validation)
y_test_pred = rf_model.predict(X_test)


# =========================
# 10. 评估函数
# =========================
def evaluate_model(name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


# =========================
# 11. 输出评估结果
# =========================
evaluate_model("TRAIN", y_train, y_train_pred )

evaluate_model("VALIDATION", y_validation, y_validation_pred)

evaluate_model("TEST", y_test, y_test_pred)


# =========================
# 12. 保存模型、scaler、预测结果
# =========================
joblib.dump(
    rf_model,
    os.path.join(save_dir, "rf_classifier.pkl")
)

joblib.dump(
    scaler_X,
    os.path.join(save_dir, "scaler_X.pkl")
)

print("\n模型已保存:", save_dir)