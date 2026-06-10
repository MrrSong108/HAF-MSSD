import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc
import json
import joblib
import random
import numpy as np
import pandas as pd
import optuna
import tensorflow as tf

from optuna.pruners import MedianPruner

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
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
    f1_score
)

# =========================
# 1. 固定随机种子
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# 2. GPU 设置
# =========================
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"检测到 GPU 数量: {len(gpus)}")
        print("GPU 设备:", gpus)
    except RuntimeError as e:
        print("GPU 设置失败:", e)
else:
    print("未检测到 GPU，将使用 CPU 运行。")

# =========================
# 3. 路径设置
# =========================
data_dir = "/root/autodl-tmp/airport_project/results/result_predict/result_xgboost_gpu"

train_path = os.path.join(data_dir, "classification_train.csv")
test_path = os.path.join(data_dir, "classification_test.csv")
validation_path = os.path.join(data_dir, "classification_validation.csv")

save_dir = "/root/autodl-tmp/airport_project/results/result_classifier/result_gru_classifier"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 4. 数据读取
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

# GRU 输入格式：[样本数, 时间步长, 特征数]
# 当前沿用单时间步输入形式
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
X_validation = X_validation.reshape(X_validation.shape[0], 1, X_validation.shape[1])

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
# 5. GRU 建模函数
# =========================
def build_gru_model(
    input_shape,
    gru_units_1,
    gru_units_2,
    dense_units,
    dropout_rate,
    learning_rate,
    l2_reg
):
    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(
        GRU(
            gru_units_1,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg)
        )
    )
    model.add(Dropout(dropout_rate))

    model.add(
        GRU(
            gru_units_2,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg)
        )
    )
    model.add(Dropout(dropout_rate))

    model.add(
        Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=l2(l2_reg)
        )
    )
    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================
# 6. 搜索最优阈值函数
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
# 7. 概率分布检测函数
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
# 8. 评估函数
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
# 9. Optuna 目标函数
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
        l2_reg=l2_reg
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    model.fit(
        X_train,
        y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(X_validation, y_validation),
        verbose=0,
        callbacks=[early_stopping]
    )

    y_val_prob = model.predict(
        X_validation,
        batch_size=batch_size,
        verbose=0
    ).reshape(-1)

    # 在验证集上搜索最优阈值
    best_threshold, best_score = find_best_threshold(
        y_validation,
        y_val_prob,
        metric="balanced_accuracy"
    )

    # 保存每个 trial 的最优阈值，方便后续查看
    trial.set_user_attr("best_threshold", float(best_threshold))
    trial.set_user_attr("best_balanced_accuracy", float(best_score))

    K.clear_session()
    del model
    gc.collect()

    # 这里直接最大化 balanced accuracy
    return best_score


# =========================
# 10. Optuna 调参
# =========================
pruner = MedianPruner(
    n_startup_trials=10,
    n_warmup_steps=5
)

study_gru_cls = optuna.create_study(
    direction="maximize",
    pruner=pruner
)

study_gru_cls.optimize(
    objective,
    n_trials=50,
    n_jobs=1
)

print(f"\nOptuna 最优验证集 Balanced Accuracy: {study_gru_cls.best_value:.4f}")
print("最佳超参数:")
print(study_gru_cls.best_params)

print("\nOptuna 最优 trial 对应阈值:")
print(study_gru_cls.best_trial.user_attrs)


# =========================
# 11. 保存最佳参数
# =========================
best_params = study_gru_cls.best_params

with open(
    os.path.join(save_dir, "best_gru_classifier_params.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(best_params, f, ensure_ascii=False, indent=4)


# =========================
# 12. 使用最佳参数训练最终 GRU 模型
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
    l2_reg=best_params["l2_reg"]
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=best_params["patience"],
    restore_best_weights=True,
    verbose=1
)

history = best_model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=best_params["batch_size"],
    validation_data=(X_validation, y_validation),
    verbose=1,
    callbacks=[early_stopping]
)


# =========================
# 13. 预测概率
# =========================
def predict_prob(model, X, batch_size):
    prob = model.predict(
        X,
        batch_size=batch_size,
        verbose=0
    ).reshape(-1)

    return prob


y_train_prob = predict_prob(
    best_model,
    X_train,
    best_params["batch_size"]
)

y_validation_prob = predict_prob(
    best_model,
    X_validation,
    best_params["batch_size"]
)

y_test_prob = predict_prob(
    best_model,
    X_test,
    best_params["batch_size"]
)


# =========================
# 14. 概率分布检测
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
# 15. 在验证集上搜索最终最优阈值
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
    "optuna_best_value": float(study_gru_cls.best_value),
    "optuna_best_trial_threshold": float(
        study_gru_cls.best_trial.user_attrs.get("best_threshold", -1)
    )
}

with open(
    os.path.join(save_dir, "best_threshold.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(threshold_info, f, ensure_ascii=False, indent=4)


# =========================
# 16. 使用最优阈值生成分类结果
# =========================
y_train_pred = (y_train_prob >= best_threshold).astype(int)
y_validation_pred = (y_validation_prob >= best_threshold).astype(int)
y_test_pred = (y_test_prob >= best_threshold).astype(int)


# =========================
# 17. 输出最终评估结果
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
# 18. 同时输出固定0.5阈值下的结果，便于对比
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
# 19. 保存模型、scaler、预测结果
# =========================
best_model.save(
    os.path.join(save_dir, "gru_classifier.keras")
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