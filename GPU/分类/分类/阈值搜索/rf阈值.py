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
    classification_report,
    balanced_accuracy_score,
    f1_score
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
data_dir = "/root/autodl-tmp/airport_project/results/result_predict/result_xgboost_gpu"

train_path = os.path.join(data_dir, "classification_train.csv")
test_path = os.path.join(data_dir, "classification_test.csv")
validation_path = os.path.join(data_dir, "classification_validation.csv")

save_dir = "/root/autodl-tmp/airport_project/results/result_classifier/result_rf_classifier"
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

y_train = train["target"].astype(int).values
y_test = test["target"].astype(int).values
y_validation = validation["target"].astype(int).values

print("数据已输入")
print("X_train shape:", X_train.shape)
print("X_validation shape:", X_validation.shape)
print("X_test shape:", X_test.shape)

print("\n训练集类别分布:")
print(pd.Series(y_train).value_counts().sort_index())

print("\n验证集类别分布:")
print(pd.Series(y_validation).value_counts().sort_index())

print("\n测试集类别分布:")
print(pd.Series(y_test).value_counts().sort_index())


# =========================
# 4. 获取1类预测概率
# =========================
def get_positive_prob(model, X):
    """
    获取类别1的预测概率。
    这样写比直接 [:, 1] 更稳妥，避免类别顺序异常。
    """
    prob_all = model.predict_proba(X)

    class_index = np.where(model.classes_ == 1)[0]

    if len(class_index) == 0:
        raise ValueError("模型类别中未找到类别 1，请检查 y_train 是否包含 1 类。")

    prob = prob_all[:, class_index[0]]

    return prob


# =========================
# 5. 搜索最优阈值函数
# =========================
def find_best_threshold(y_true, prob, metric="balanced_accuracy"):
    """
    在验证集上搜索最优分类阈值。

    metric 可选：
    - "balanced_accuracy"：推荐，用于类别不均衡情况
    - "accuracy"：普通准确率，容易偏向多数类
    - "f1"：默认计算1类F1
    """

    thresholds = np.linspace(0.01, 0.99, 199)

    best_threshold = 0.5
    best_score = -1

    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)

        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred)
        elif metric == "accuracy":
            score = accuracy_score(y_true, pred)
        elif metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            raise ValueError("metric 只能是 balanced_accuracy、accuracy 或 f1")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


# =========================
# 6. 概率分布检测函数
# =========================
def check_prob_distribution(name, y_true, prob):
    print(f"\n========== {name} 概率分布检测 ==========")

    print("\n整体预测概率分布:")
    print(
        pd.Series(prob).describe(
            percentiles=[
                0.01, 0.05, 0.10, 0.25,
                0.50, 0.75, 0.90, 0.95, 0.99
            ]
        )
    )

    print("\n不同真实类别下的预测概率分布:")
    df_prob = pd.DataFrame({
        "y_true": y_true,
        "prob": prob
    })
    print(df_prob.groupby("y_true")["prob"].describe())

    print("\n固定阈值0.5时，预测为1的比例:")
    print((prob >= 0.5).mean())

    print("固定阈值0.5时，预测类别分布:")
    print(pd.Series((prob >= 0.5).astype(int)).value_counts().sort_index())


# =========================
# 7. 评估函数
# =========================
def evaluate_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"\n========== {name} ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    print("\n预测类别分布:")
    print(pd.Series(y_pred).value_counts().sort_index())

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


# =========================
# 8. Optuna 目标函数
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

    # 预测验证集属于1类的概率
    y_val_prob = get_positive_prob(model, X_validation)

    # 在验证集上搜索最优阈值
    best_threshold, best_score = find_best_threshold(
        y_validation,
        y_val_prob,
        metric="balanced_accuracy"
    )

    # 保存每个 trial 对应的最优阈值，方便查看
    trial.set_user_attr("best_threshold", float(best_threshold))
    trial.set_user_attr("best_balanced_accuracy", float(best_score))

    # 这里直接最大化 balanced accuracy
    return best_score


# =========================
# 9. Optuna 调参
# =========================
study_rf_cls = optuna.create_study(direction="maximize")

study_rf_cls.optimize(
    objective,
    n_trials=50,
    n_jobs=1
)

print(f"\nOptuna 最优验证集 Balanced Accuracy: {study_rf_cls.best_value:.4f}")
print("最佳超参数:")
print(study_rf_cls.best_params)

print("\nOptuna 最优 trial 对应阈值:")
print(study_rf_cls.best_trial.user_attrs)


# =========================
# 10. 保存最佳参数
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


# =========================
# 11. 训练最终模型
# =========================
rf_model = RandomForestClassifier(**final_params)

rf_model.fit(X_train, y_train)


# =========================
# 12. 预测概率
# =========================
y_train_prob = get_positive_prob(rf_model, X_train)

y_validation_prob = get_positive_prob(rf_model, X_validation)

y_test_prob = get_positive_prob(rf_model, X_test)


# =========================
# 13. 概率分布检测
# =========================
check_prob_distribution(
    "TRAIN",
    y_train,
    y_train_prob
)

check_prob_distribution(
    "VALIDATION",
    y_validation,
    y_validation_prob
)

check_prob_distribution(
    "TEST",
    y_test,
    y_test_prob
)


# =========================
# 14. 在验证集上搜索最终最优阈值
# =========================
best_threshold, best_threshold_score = find_best_threshold(
    y_validation,
    y_validation_prob,
    metric="balanced_accuracy"
)

print("\n========== 最优阈值搜索结果 ==========")
print(f"Best Threshold on Validation: {best_threshold:.4f}")
print(f"Best Validation Balanced Accuracy: {best_threshold_score:.4f}")

# 保存最优阈值
threshold_info = {
    "best_threshold": float(best_threshold),
    "best_validation_balanced_accuracy": float(best_threshold_score),
    "threshold_metric": "balanced_accuracy",
    "optuna_best_value": float(study_rf_cls.best_value),
    "optuna_best_trial_threshold": float(
        study_rf_cls.best_trial.user_attrs.get("best_threshold", -1)
    )
}

with open(
    os.path.join(save_dir, "best_threshold.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(threshold_info, f, ensure_ascii=False, indent=4)


# =========================
# 15. 使用最优阈值生成分类结果
# =========================
y_train_pred = (y_train_prob >= best_threshold).astype(int)
y_validation_pred = (y_validation_prob >= best_threshold).astype(int)
y_test_pred = (y_test_prob >= best_threshold).astype(int)


# =========================
# 16. 输出最终评估结果
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
# 17. 同时输出固定0.5阈值下的结果，便于对比
# =========================
print("\n\n==============================")
print("固定阈值 0.5 的结果，仅用于对比")
print("==============================")

evaluate_model(
    "TRAIN - threshold 0.5",
    y_train,
    (y_train_prob >= 0.5).astype(int)
)

evaluate_model(
    "VALIDATION - threshold 0.5",
    y_validation,
    (y_validation_prob >= 0.5).astype(int)
)

evaluate_model(
    "TEST - threshold 0.5",
    y_test,
    (y_test_prob >= 0.5).astype(int)
)


# =========================
# 18. 保存模型、scaler、预测结果
# =========================
joblib.dump(
    rf_model,
    os.path.join(save_dir, "rf_classifier.pkl")
)

joblib.dump(
    scaler_X,
    os.path.join(save_dir, "scaler_X.pkl")
)

# 保存预测结果，方便后续检查
train_result = pd.DataFrame({
    "y_true": y_train,
    "y_prob": y_train_prob,
    "y_pred": y_train_pred
})

validation_result = pd.DataFrame({
    "y_true": y_validation,
    "y_prob": y_validation_prob,
    "y_pred": y_validation_pred
})

test_result = pd.DataFrame({
    "y_true": y_test,
    "y_prob": y_test_prob,
    "y_pred": y_test_pred
})

train_result.to_csv(
    os.path.join(save_dir, "train_prediction_result.csv"),
    index=False,
    encoding="utf-8-sig"
)

validation_result.to_csv(
    os.path.join(save_dir, "validation_prediction_result.csv"),
    index=False,
    encoding="utf-8-sig"
)

test_result.to_csv(
    os.path.join(save_dir, "test_prediction_result.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n模型、scaler、最优阈值和预测结果已保存到:")
print(save_dir)