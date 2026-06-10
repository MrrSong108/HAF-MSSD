import os
import json
import joblib
import random
import numpy as np
import pandas as pd
import optuna

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================
# 1. 固定随机种子
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =========================
# 2. 数据读取
# =========================
predict_columns = [
    'predict_T2_0.5', 'predict_T2_1', 'predict_T2_1.5', 'predict_T2_2', 'predict_T2_3',
    'predict_T2_4', 'predict_T2_5', 'predict_T2_6', 'predict_T2_7', 'predict_T2_8',
    'predict_T2_9', 'predict_T2_10', 'predict_T2_11', 'predict_T2_12', 'predict_T2_13',
    'predict_T2_14', 'predict_T2_15', 'predict_T2_16', 'predict_T2_17', 'predict_T2_18',
    'predict_T2_19', 'predict_T2_20', 'predict_T2_21', 'predict_T2_22', 'predict_T2_23',
    'predict_T2_24',
    'predict_T3_0.5', 'predict_T3_1', 'predict_T3_1.5', 'predict_T3_2', 'predict_T3_3',
    'predict_T3_4', 'predict_T3_5', 'predict_T3_6', 'predict_T3_7', 'predict_T3_8',
    'predict_T3_9', 'predict_T3_10', 'predict_T3_11', 'predict_T3_12', 'predict_T3_13',
    'predict_T3_14', 'predict_T3_15', 'predict_T3_16', 'predict_T3_17', 'predict_T3_18',
    'predict_T3_19', 'predict_T3_20', 'predict_T3_21', 'predict_T3_22', 'predict_T3_23',
    'predict_T3_24'
]

train_path = "/root/autodl-tmp/airport_project/data/xiao/traff/train.csv"
test_path = "/root/autodl-tmp/airport_project/data/xiao/traff/test.csv"
validation_path = "/root/autodl-tmp/airport_project/data/xiao/traff/test.csv"

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

print("数据已读取")
print("训练集特征维度:", features_train.shape)
print("测试集特征维度:", features_test.shape)
print("验证集特征维度:", features_validation.shape)


# =========================
# 3. 归一化
# =========================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(features_train)
X_test = scaler_X.transform(features_test)
X_validation = scaler_X.transform(features_validation)

y_train = scaler_y.fit_transform(train_Y.values.reshape(-1, 1))
y_test = scaler_y.transform(test_Y.values.reshape(-1, 1))
y_validation = scaler_y.transform(validation_Y.values.reshape(-1, 1))

print("归一化完成")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("X_validation shape:", X_validation.shape)


# =========================
# 4. Optuna 目标函数
# =========================
def objective(trial):
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 8, 30, step=2),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, step=2),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10, step=1),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", 0.2, 0.3, 0.4, 0.5]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),

        # RF 内部并行可以保留，但不要让 Optuna 也并行
        "n_jobs": -1,
        "random_state": SEED,
    }

    model = RandomForestRegressor(**param_grid)

    model.fit(X_train, y_train.ravel())

    y_val_scaled = model.predict(X_validation)
    y_val_pred = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1))
    y_val_true = scaler_y.inverse_transform(y_validation.reshape(-1, 1))

    y_val_pred = np.maximum(y_val_pred, 0)

    rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))

    return rmse


# =========================
# 5. Optuna 调参
# =========================
study_rf = optuna.create_study(direction="minimize")

study_rf.optimize(
    objective,
    n_trials=50,
    n_jobs=1
)

print("最优验证集 RMSE:", study_rf.best_value)
print("最佳超参数:", study_rf.best_params)

best_params = study_rf.best_params

with open("/root/autodl-tmp/airport_project/data/xiao/air/predict/best_rf_params.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, ensure_ascii=False, indent=4)


# =========================
# 6. 使用最佳参数训练最终 RF 模型
# =========================
rf_model = RandomForestRegressor(
    **best_params,
    n_jobs=-1,
    random_state=SEED
)

rf_model.fit(X_train, y_train.ravel())


# =========================
# 7. 预测与评估
# =========================
def inverse_predict(model, X):
    pred_scaled = model.predict(X)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
    return np.maximum(pred, 0)


y_train_pred = inverse_predict(rf_model, X_train)
y_validation_pred = inverse_predict(rf_model, X_validation)
y_test_pred = inverse_predict(rf_model, X_test)

y_train_true = scaler_y.inverse_transform(y_train.reshape(-1, 1))
y_validation_true = scaler_y.inverse_transform(y_validation.reshape(-1, 1))
y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
validation_rmse = np.sqrt(mean_squared_error(y_validation_true, y_validation_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))

train_mae = mean_absolute_error(y_train_true, y_train_pred)
validation_mae = mean_absolute_error(y_validation_true, y_validation_pred)
test_mae = mean_absolute_error(y_test_true, y_test_pred)

print("train rmse:", train_rmse)
print("validation rmse:", validation_rmse)
print("test rmse:", test_rmse)

print("train mae:", train_mae)
print("validation mae:", validation_mae)
print("test mae:", test_mae)


# =========================
# 8. 误差分类标签生成
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


# 如果你做 10% 误差分类，用 threshold=0.1
# 如果你做 20% 敏感性实验，改成 threshold=0.2
error_threshold = 0.1

train_compared = compare_values(y_train_true, y_train_pred, threshold=error_threshold)
validation_compared = compare_values(y_validation_true, y_validation_pred, threshold=error_threshold)
test_compared = compare_values(y_test_true, y_test_pred, threshold=error_threshold)

unique, counts = np.unique(train_compared, return_counts=True)
distribution = {int(k): int(v) for k, v in zip(unique, counts)}
print("训练集类别分布:", distribution)

unique, counts = np.unique(validation_compared, return_counts=True)
distribution = {int(k): int(v) for k, v in zip(unique, counts)}
print("验证集类别分布:", distribution)

unique, counts = np.unique(test_compared, return_counts=True)
distribution = {int(k): int(v) for k, v in zip(unique, counts)}
print("测试集类别分布:", distribution)


# =========================
# 9. 保存模型、归一化器和结果
# =========================
save_dir = "/root/autodl-tmp/airport_project/data/xiao/traff/predict/rf"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(rf_model, os.path.join(save_dir, "best_rf_model.pkl"))
joblib.dump(scaler_X, os.path.join(save_dir, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(save_dir, "scaler_y.pkl"))
X_train_data = pd.DataFrame(X_train, columns=features_train.columns)
X_train_data["target"] = train_compared

X_validation_data = pd.DataFrame(X_validation, columns=features_validation.columns)

X_validation_data["target"] = validation_compared

X_test_data = pd.DataFrame(X_test, columns=features_test.columns)
X_test_data["target"] = test_compared

X_train_data.to_csv(os.path.join(save_dir, "classification_train.csv"), index=False)
X_validation_data.to_csv(os.path.join(save_dir, "classification_validation.csv"), index=False)
X_test_data.to_csv(os.path.join(save_dir, "classification_test.csv"), index=False)

pred_result = pd.DataFrame({
    "test_true": y_test_true.reshape(-1),
    "test_pred": y_test_pred.reshape(-1),
    "test_label": test_compared
})
pred_result.to_csv(os.path.join(save_dir, "test_prediction_result.csv"), index=False)

metrics_result = pd.DataFrame({
    "dataset": ["train", "validation", "test"],
    "rmse": [train_rmse, validation_rmse, test_rmse],
    "mae": [train_mae, validation_mae, test_mae]
})
metrics_result.to_csv(os.path.join(save_dir, "metrics_result.csv"), index=False)

print("RF模型、归一化器、分类数据和预测结果已保存到:", save_dir)