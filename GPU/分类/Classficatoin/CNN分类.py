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

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
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
data_dir = "/root/autodl-tmp/airport_project/data/xiao/traff/predict/lstm"

train_path = os.path.join(data_dir, "classification_train.csv")
test_path = os.path.join(data_dir, "classification_test.csv")
validation_path = os.path.join(data_dir, "classification_validation.csv")

save_dir = "/root/autodl-tmp/airport_project/data/xiao/traff/class/cnn"
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

# CNN 输入格式：[样本数, 特征数, 1]
# 这样 Conv1D 才能沿着特征维度卷积
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)

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
# 7. CNN 建模函数
# =========================
def build_model(
    input_shape,
    filters_1,
    filters_2,
    kernel_size,
    dense_units,
    dropout_rate,
    learning_rate,
    l2_reg
):
    model = Sequential()

    model.add(
        Conv1D(
            filters=filters_1,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=l2(l2_reg),
            input_shape=input_shape
        )
    )
    model.add(Dropout(dropout_rate))

    model.add(
        Conv1D(
            filters=filters_2,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=l2(l2_reg)
        )
    )
    model.add(Dropout(dropout_rate))

    model.add(GlobalAveragePooling1D())

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
# 8. Optuna 目标函数
# =========================
def objective(trial):
    K.clear_session()
    gc.collect()

    filters_1 = trial.suggest_int("filters_1", 32, 128, step=32)
    filters_2 = trial.suggest_int("filters_2", 16, 96, step=16)

    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])

    dense_units = trial.suggest_categorical("dense_units", [32, 64, 128])

    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.35)

    learning_rate = trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True)

    l2_reg = trial.suggest_float("l2_reg", 1e-6, 5e-4, log=True)

    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    patience = trial.suggest_int("patience", 8, 15)

    max_epochs = 150

    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        filters_1=filters_1,
        filters_2=filters_2,
        kernel_size=kernel_size,
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
        callbacks=[early_stopping],
    )

    y_val_prob = model.predict(
        X_validation,
        batch_size=batch_size,
        verbose=0
    ).reshape(-1)

    y_val_pred = (y_val_prob >= 0.5).astype(int)

    score = balanced_accuracy_score(y_validation, y_val_pred)

    K.clear_session()
    del model
    gc.collect()

    return 1 - score


# =========================
# 9. Optuna 调参
# =========================
study_cnn_cls = optuna.create_study(direction="minimize")

study_cnn_cls.optimize(
    objective,
    n_trials=50,
    n_jobs=1
)

print(f"\n最优Accuracy: {1 - study_cnn_cls.best_value:.4f}")
print("最佳超参数:")
print(study_cnn_cls.best_params)


# =========================
# 10. 保存最佳参数
# =========================
best_params = study_cnn_cls.best_params

with open(
    os.path.join(save_dir, "best_cnn_classifier_params.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(best_params, f, ensure_ascii=False, indent=4)

# =========================
# 11. 使用最佳参数训练最终 CNN 模型
# =========================
K.clear_session()
gc.collect()

best_model = build_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    filters_1=best_params["filters_1"],
    filters_2=best_params["filters_2"],
    kernel_size=best_params["kernel_size"],
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
# 12. 预测
# =========================
def predict_label(model, X, batch_size):
    prob = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
    pred = (prob >= 0.5).astype(int)
    return prob, pred


y_train_prob, y_train_pred = predict_label(
    best_model,
    X_train,
    best_params["batch_size"]
)

y_validation_prob, y_validation_pred = predict_label(
    best_model,
    X_validation,
    best_params["batch_size"]
)

y_test_prob, y_test_pred = predict_label(
    best_model,
    X_test,
    best_params["batch_size"]
)


# =========================
# 13. 评估函数
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
# 14. 输出评估结果
# =========================
evaluate_model("TRAIN", y_train, y_train_pred )

evaluate_model("VALIDATION", y_validation, y_validation_pred)

evaluate_model("TEST", y_test, y_test_pred)

# =========================
# 15. 保存模型、scaler、预测结果
# =========================
best_model.save(
    os.path.join(save_dir, "cnn_classifier.keras")
)

joblib.dump(
    scaler_X,
    os.path.join(save_dir, "scaler_X.pkl")
)

print("\n模型已保存:", save_dir)