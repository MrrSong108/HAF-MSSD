import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import re
import gc
import time
import json
import joblib
import optuna
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ==================================================
# 1. 路径配置：同时训练 0 类和 1 类
# ==================================================

CLASS_DIRS = {
    "class_0_challenging": Path("/root/autodl-tmp/airport_project/data/xiao/check/cluster/kmeans_dbi_search/class_0_challenging"),
    "class_1_friendly": Path("/root/autodl-tmp/airport_project/data/xiao/check/cluster/kmeans_dbi_search/class_1_friendly"),
}

SAVE_ROOT = Path("/root/autodl-tmp/airport_project/data/xiao/check/cluster/lstm")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_COL = "predict_T2_0.5"

N_TRIALS = 50
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 单 GPU 下不建议 Optuna 并行 trial
OPTUNA_N_JOBS = 1


# ==================================================
# 2. 固定随机种子 + GPU 按需增长
# ==================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"检测到 GPU 数量：{len(gpus)}，已设置显存按需增长")
        except Exception as e:
            print(f"设置 GPU 显存按需增长失败：{e}")
    else:
        print("未检测到 GPU，将使用 CPU 运行")


set_seed(RANDOM_STATE)
set_gpu_memory_growth()


# ==================================================
# 3. 指标函数
# ==================================================

def directional_symmetry(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) <= 1:
        return 0.0

    true_diff = np.sign(y_true[1:] - y_true[:-1])
    pred_diff = np.sign(y_pred[1:] - y_pred[:-1])

    return float(np.mean(true_diff == pred_diff))


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    y_pred = np.maximum(y_pred, 0)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1))))
    ds = directional_symmetry(y_true, y_pred)

    return rmse, mae, mape, ds


# ==================================================
# 4. 只匹配真正的簇文件
# ==================================================

def get_cluster_train_files(class_dir, class_prefix):
    """
    只匹配：
    0cluster_数字_train.csv
    或
    1cluster_数字_train.csv

    不会匹配：
    class_0_challenging_train_all.csv
    class_1_friendly_train_all.csv
    """

    pattern = re.compile(rf"^{class_prefix}cluster_(\d+)_train\.csv$")

    matched_files = []

    for file_path in class_dir.iterdir():
        match = pattern.match(file_path.name)
        if match:
            cluster_id = int(match.group(1))
            matched_files.append((cluster_id, file_path))

    matched_files = sorted(matched_files, key=lambda x: x[0])

    return [x[1] for x in matched_files]


# ==================================================
# 5. 构建 LSTM 模型
# ==================================================

def build_lstm_model(input_shape, lstm_units_1, lstm_units_2, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(LSTM(
        units=lstm_units_1,
        activation="tanh",
        return_sequences=True
    ))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(
        units=lstm_units_2,
        activation="tanh",
        return_sequences=False
    ))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error"
    )

    return model


# ==================================================
# 6. 单个簇训练
# ==================================================

def train_one_cluster(class_name, train_path, valid_path):
    cluster_name = train_path.name.replace("_train.csv", "")

    print("\n" + "=" * 100)
    print(f"当前类别：{class_name}")
    print(f"当前簇：{cluster_name}")
    print(f"训练集文件：{train_path}")
    print(f"外部验证集文件：{valid_path if valid_path is not None else '无'}")

    has_external_validation = valid_path is not None and Path(valid_path).exists()

    if has_external_validation:
        print("划分方式：train 文件内部 8:2 划分，其中 8 用于训练，2 与外部 validation 合并后共同验证")
    else:
        print("划分方式：train 文件内部 8:2 划分，其中 8 用于训练，2 用于验证")

    class_save_dir = SAVE_ROOT / class_name
    cluster_save_dir = class_save_dir / cluster_name
    cluster_save_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = cluster_save_dir / "best_lstm_model.keras"
    scaler_x_save_path = cluster_save_dir / "scaler_X.pkl"
    scaler_y_save_path = cluster_save_dir / "scaler_y.pkl"
    params_save_path = cluster_save_dir / "best_params.json"
    metrics_save_path = cluster_save_dir / "metrics.csv"
    pred_save_path = cluster_save_dir / "predictions.csv"
    split_info_save_path = cluster_save_dir / "split_info.json"

    # 断点续跑：如果已经训练完成，则读取旧结果并跳过
    if model_save_path.exists() and metrics_save_path.exists():
        print(f"{class_name} - {cluster_name} 已训练完成，跳过。")
        old_metrics = pd.read_csv(metrics_save_path)
        return old_metrics.iloc[0].to_dict()

    start_time = time.time()

    train_df = pd.read_csv(train_path)

    if has_external_validation:
        external_valid_df = pd.read_csv(valid_path)
        print(f"检测到外部验证集，将使用：{valid_path}")
    else:
        external_valid_df = None
        print("未检测到外部验证集，仅使用 train 文件内部 8:2 中的 2 成作为验证集。")

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"{train_path} 中缺少目标列：{TARGET_COL}")

    if has_external_validation and TARGET_COL not in external_valid_df.columns:
        raise ValueError(f"{valid_path} 中缺少目标列：{TARGET_COL}")

    if len(train_df) < 5:
        raise ValueError(f"{train_path} 样本量过少，仅 {len(train_df)} 条，不适合 8:2 划分。")

    drop_cols = [TARGET_COL]

    if "cluster" in train_df.columns:
        drop_cols.append("cluster")

    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    if has_external_validation:
        missing_cols = [c for c in feature_cols if c not in external_valid_df.columns]
        if missing_cols:
            raise ValueError(f"{valid_path} 中缺少以下特征列：{missing_cols}")

    X_train_all_raw = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train_all_raw = train_df[TARGET_COL].values.reshape(-1, 1)

    if has_external_validation:
        X_external_valid_raw = external_valid_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_external_valid_raw = external_valid_df[TARGET_COL].values.reshape(-1, 1)
    else:
        X_external_valid_raw = None
        y_external_valid_raw = np.empty((0, 1))

    # ==================================================
    # 核心逻辑：
    # train 文件内部先做 8:2
    # 8 = 真正训练集
    # 2 = 内部验证集
    # 最终验证集 = 内部验证集 + 外部 validation 文件
    # 如果没有外部 validation，则最终验证集 = 内部验证集
    # ==================================================

    X_train_raw, X_inner_valid_raw, y_train_raw, y_inner_valid_raw = train_test_split(
        X_train_all_raw,
        y_train_all_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    if has_external_validation:
        X_valid_raw = pd.concat(
            [
                X_inner_valid_raw.reset_index(drop=True),
                X_external_valid_raw.reset_index(drop=True)
            ],
            axis=0,
            ignore_index=True
        )

        y_valid_raw = np.vstack(
            [
                y_inner_valid_raw,
                y_external_valid_raw
            ]
        )

        validation_source_list = (
            ["inner_validation_from_train"] * len(y_inner_valid_raw)
            + ["external_validation"] * len(y_external_valid_raw)
        )

        split_method = "train_file_8_2_plus_external_validation"

    else:
        X_valid_raw = X_inner_valid_raw.reset_index(drop=True)
        y_valid_raw = y_inner_valid_raw

        validation_source_list = ["inner_validation_from_train"] * len(y_inner_valid_raw)

        split_method = "train_file_8_2_only_inner_validation"

    # scaler 只能在真正训练集上 fit
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(X_train_raw)
    X_valid = scaler_X.transform(X_valid_raw)

    y_train = scaler_y.fit_transform(y_train_raw).reshape(-1)
    y_valid = scaler_y.transform(y_valid_raw).reshape(-1)

    # LSTM 输入格式：[样本数, 时间步长, 特征数]
    # 当前每一行已经是一个样本，所以时间步长设为 1
    X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_valid_lstm = X_valid.reshape(X_valid.shape[0], 1, X_valid.shape[1])

    input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])

    # ==================================================
    # Optuna 目标函数
    # ==================================================

    def objective(trial):
        K.clear_session()
        gc.collect()
        set_seed(RANDOM_STATE)

        params = {
            "lstm_units_1": trial.suggest_int("lstm_units_1", 32, 512, step=32),
            "lstm_units_2": trial.suggest_int("lstm_units_2", 16, 256, step=16),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
            "epochs": trial.suggest_int("epochs", 50, 200, step=10),
            "patience": trial.suggest_int("patience", 5, 20),
        }

        model = build_lstm_model(
            input_shape=input_shape,
            lstm_units_1=params["lstm_units_1"],
            lstm_units_2=params["lstm_units_2"],
            dropout_rate=params["dropout_rate"],
            learning_rate=params["learning_rate"]
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=params["patience"],
            restore_best_weights=True
        )

        model.fit(
            X_train_lstm,
            y_train,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            validation_data=(X_valid_lstm, y_valid),
            verbose=0,
            callbacks=[early_stopping]
        )

        y_pred_scaled = model.predict(X_valid_lstm, verbose=0).reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
        y_pred = np.maximum(y_pred, 0)

        y_true = y_valid_raw.reshape(-1)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        del model
        K.clear_session()
        gc.collect()

        return rmse

    # ==================================================
    # 开始搜索
    # ==================================================

    study = optuna.create_study(direction="minimize")

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=OPTUNA_N_JOBS
    )

    best_params = study.best_params

    # ==================================================
    # 使用最优参数重新训练
    # ==================================================

    K.clear_session()
    gc.collect()
    set_seed(RANDOM_STATE)

    best_model = build_lstm_model(
        input_shape=input_shape,
        lstm_units_1=best_params["lstm_units_1"],
        lstm_units_2=best_params["lstm_units_2"],
        dropout_rate=best_params["dropout_rate"],
        learning_rate=best_params["learning_rate"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=best_params["patience"],
        restore_best_weights=True
    )

    history = best_model.fit(
        X_train_lstm,
        y_train,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        validation_data=(X_valid_lstm, y_valid),
        verbose=1,
        callbacks=[early_stopping]
    )

    y_pred_scaled = best_model.predict(X_valid_lstm, verbose=0).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
    y_pred = np.maximum(y_pred, 0)

    y_true = y_valid_raw.reshape(-1)

    rmse, mae, mape, ds = calc_metrics(y_true, y_pred)

    run_time = time.time() - start_time

    actual_epochs = len(history.history["loss"])

    metrics = {
        "class_name": class_name,
        "cluster": cluster_name,
        "model": "LSTM",

        "n_train_file_total": len(train_df),
        "n_train_used": len(X_train_raw),
        "n_inner_validation": len(X_inner_valid_raw),
        "n_external_validation": len(external_valid_df) if has_external_validation else 0,
        "n_validation_total": len(X_valid_raw),

        "split_method": split_method,

        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "DS": ds,
        "actual_epochs": actual_epochs,
        "run_time_seconds": run_time
    }

    # ==================================================
    # 保存模型、scaler、参数、指标、预测值
    # ==================================================

    best_model.save(model_save_path)
    joblib.dump(scaler_X, scaler_x_save_path)
    joblib.dump(scaler_y, scaler_y_save_path)

    final_params = {
        "random_state": RANDOM_STATE,
        **best_params
    }

    with open(params_save_path, "w", encoding="utf-8") as f:
        json.dump(final_params, f, ensure_ascii=False, indent=4, default=str)

    split_info = {
        "source_train_file": str(train_path),
        "source_external_validation_file": str(valid_path) if has_external_validation else None,
        "split_method": split_method,
        "train_file_total": len(train_df),
        "train_used": len(X_train_raw),
        "inner_validation_from_train_file": len(X_inner_valid_raw),
        "external_validation": len(external_valid_df) if has_external_validation else 0,
        "validation_total": len(X_valid_raw),
        "train_ratio_inside_train_file": 0.8,
        "inner_validation_ratio_inside_train_file": 0.2,
        "random_state": RANDOM_STATE,
        "shuffle": True,
        "target_col": TARGET_COL,
        "feature_cols": feature_cols
    }

    with open(split_info_save_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=4)

    pd.DataFrame([metrics]).to_csv(
        metrics_save_path,
        index=False,
        encoding="utf-8-sig"
    )

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "abs_error": np.abs(y_true - y_pred),
        "data_source": validation_source_list
    })

    pred_df.to_csv(
        pred_save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"{class_name} - {cluster_name} 训练完成")
    print(f"train文件总样本量: {len(train_df)}")
    print(f"实际训练样本量: {len(X_train_raw)}")
    print(f"内部验证样本量: {len(X_inner_valid_raw)}")
    print(f"外部验证样本量: {len(external_valid_df) if has_external_validation else 0}")
    print(f"最终验证样本量: {len(X_valid_raw)}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"MAPE: {mape:.6f}")
    print(f"DS  : {ds:.6f}")
    print(f"实际训练轮数: {actual_epochs}")
    print(f"运行时间: {run_time:.2f} 秒")
    print(f"保存目录: {cluster_save_dir}")

    del best_model
    K.clear_session()
    gc.collect()

    return metrics


# ==================================================
# 7. 主程序：依次训练 0 类和 1 类
# ==================================================

def main():
    all_metrics = []

    summary_csv_path = SAVE_ROOT / "lstm_all_classes_all_clusters_metrics.csv"
    excel_txt_path = SAVE_ROOT / "excel_copy_metrics.txt"
    excel_four_only_path = SAVE_ROOT / "excel_copy_four_metrics_only.txt"
    error_log_path = SAVE_ROOT / "error_log.txt"

    for class_name, class_dir in CLASS_DIRS.items():

        if "class_0" in class_name:
            class_prefix = "0"
        elif "class_1" in class_name:
            class_prefix = "1"
        else:
            raise ValueError(f"无法识别类别前缀：{class_name}")

        print("\n" + "#" * 100)
        print(f"开始处理类别：{class_name}")
        print(f"数据路径：{class_dir}")

        if not class_dir.exists():
            print(f"路径不存在，跳过：{class_dir}")

            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"[CLASS_DIR_NOT_EXIST]\t{class_name}\t{class_dir}\n")

            continue

        train_files = get_cluster_train_files(class_dir, class_prefix)

        print(f"{class_name} 共找到 {len(train_files)} 个簇训练文件")

        if len(train_files) == 0:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"[NO_CLUSTER_TRAIN_FILE]\t{class_name}\t{class_dir}\n")
            continue

        for train_path in train_files:
            valid_path = Path(str(train_path).replace("_train.csv", "_validation.csv"))

            if not valid_path.exists():
                print(f"提示：找不到对应外部验证集，将只使用 train 文件内部 8:2 划分出来的 2 成作为验证集：{valid_path}")

                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"[NO_EXTERNAL_VALIDATION_USE_INNER_VALIDATION]\t{class_name}\t{train_path.name}\t{valid_path.name}\n"
                    )

                valid_path = None

            try:
                metrics = train_one_cluster(class_name, train_path, valid_path)
                all_metrics.append(metrics)

                # 每训练完一个簇，立即保存总表
                summary_df = pd.DataFrame(all_metrics)

                summary_df.to_csv(
                    summary_csv_path,
                    index=False,
                    encoding="utf-8-sig"
                )

                # 可以直接复制到 Excel
                excel_df = summary_df[
                    [
                        "class_name",
                        "cluster",
                        "model",
                        "n_train_file_total",
                        "n_train_used",
                        "n_inner_validation",
                        "n_external_validation",
                        "n_validation_total",
                        "RMSE",
                        "MAE",
                        "MAPE",
                        "DS"
                    ]
                ]

                excel_df.to_csv(
                    excel_txt_path,
                    index=False,
                    sep="\t",
                    encoding="utf-8-sig"
                )

                # 只保存四个指标，方便直接复制
                four_df = summary_df[
                    [
                        "RMSE",
                        "MAE",
                        "MAPE",
                        "DS"
                    ]
                ]

                four_df.to_csv(
                    excel_four_only_path,
                    index=False,
                    sep="\t",
                    encoding="utf-8-sig"
                )

            except Exception as e:
                print(f"训练失败：{class_name} - {train_path.name}")
                print(f"错误信息：{e}")

                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"[TRAIN_ERROR]\t{class_name}\t{train_path.name}\t{str(e)}\n"
                    )

                K.clear_session()
                gc.collect()

    print("\n" + "=" * 100)
    print("全部类别、全部簇训练结束")
    print(f"总指标文件：{summary_csv_path}")
    print(f"Excel 复制版：{excel_txt_path}")
    print(f"仅四个指标复制版：{excel_four_only_path}")
    print(f"错误日志：{error_log_path}")


if __name__ == "__main__":
    main()