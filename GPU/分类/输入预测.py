import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================
# 1. 路径配置
# =========================
PROJECT_DIR = Path("/root/autodl-tmp/airport_project")

# 测试集路径：根据你的实际路径修改
TEST_PATH = PROJECT_DIR / "data/validation.csv"

# 分类模型和分类 scaler：根据你的实际路径修改
CLASSIFIER_MODEL_PATH = PROJECT_DIR / "results/result_classifier/result_rf_classifier/rf_classifier.pkl"
CLASSIFIER_SCALER_PATH = PROJECT_DIR / "results/result_classifier/result_rf_classifier/scaler_X.pkl"

# 聚类模型路径
CLUSTER_BASE_DIR = PROJECT_DIR / "results/result_cluster/result_rf_kmeans"

CLASS_DIRS = {
    0: CLUSTER_BASE_DIR / "class_0_challenging",
    1: CLUSTER_BASE_DIR / "class_1_friendly",
}

CLUSTER_SCALER_PATHS = {
    0: CLASS_DIRS[0] / "cluster_scaler_label0.pkl",
    1: CLASS_DIRS[1] / "cluster_scaler_label1.pkl",
}

KMEANS_PATHS = {
    0: CLASS_DIRS[0] / "kmeans_label0.pkl",
    1: CLASS_DIRS[1] / "kmeans_label1.pkl",
}

# 子模型统一目录
RESULT_MODEL_DIR = PROJECT_DIR / "results/result_model"

# 输出结果路径
OUTPUT_PATH = PROJECT_DIR / "results/result_model/final_validation_result.csv"


# =========================
# 2. 目标列配置
# =========================
TARGET_COL = "predict_T2_0.5"

predict_columns = [
    "predict_T2_0.5", "predict_T2_1", "predict_T2_1.5", "predict_T2_2", "predict_T2_3",
    "predict_T2_4", "predict_T2_5", "predict_T2_6", "predict_T2_7", "predict_T2_8",
    "predict_T2_9", "predict_T2_10", "predict_T2_11", "predict_T2_12", "predict_T2_13",
    "predict_T2_14", "predict_T2_15", "predict_T2_16", "predict_T2_17", "predict_T2_18",
    "predict_T2_19", "predict_T2_20", "predict_T2_21", "predict_T2_22", "predict_T2_23",
    "predict_T2_24",
    "predict_T3_0.5", "predict_T3_1", "predict_T3_1.5", "predict_T3_2", "predict_T3_3",
    "predict_T3_4", "predict_T3_5", "predict_T3_6", "predict_T3_7", "predict_T3_8",
    "predict_T3_9", "predict_T3_10", "predict_T3_11", "predict_T3_12", "predict_T3_13",
    "predict_T3_14", "predict_T3_15", "predict_T3_16", "predict_T3_17", "predict_T3_18",
    "predict_T3_19", "predict_T3_20", "predict_T3_21", "predict_T3_22", "predict_T3_23",
    "predict_T3_24"
]


# =========================
# 3. 基础工具函数
# =========================
def check_path(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} 不存在：{path}")


def align_features(df: pd.DataFrame, scaler, name: str):
    """
    按 scaler 训练时的 feature_names_in_ 对齐特征。
    避免训练和预测时列顺序不一致。
    """
    df = df.copy()
    df = df.select_dtypes(include=[np.number])

    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)

        missing_cols = [col for col in feature_names if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in feature_names]

        if missing_cols:
            raise ValueError(
                f"{name} 缺少 scaler 训练时使用的特征列，"
                f"前 10 个缺失列为：{missing_cols[:10]}，"
                f"共缺少 {len(missing_cols)} 个。"
            )

        if extra_cols:
            print(f"{name} 存在额外数值列，将自动忽略：{extra_cols[:10]}")

        df = df[feature_names]

    return df


def normalize_classifier_output(pred):
    """
    兼容分类模型输出：
    - RF/XGB: predict 直接输出 0/1
    - 神经网络: 输出概率，需要按 0.5 转 0/1
    - predict_proba: 二维概率，取最大概率类别
    """
    pred = np.asarray(pred)

    if pred.ndim == 2:
        if pred.shape[1] == 1:
            pred = pred.ravel()
            pred = (pred >= 0.5).astype(int)
        else:
            pred = np.argmax(pred, axis=1)
    else:
        pred = pred.ravel()
        unique_values = np.unique(pred)

        if not set(unique_values).issubset({0, 1}):
            pred = (pred >= 0.5).astype(int)

    return pred.astype(int)


def load_model_auto(model_path: Path):
    """
    自动加载 sklearn/joblib 模型或 keras 模型。
    """
    suffix = model_path.suffix.lower()

    if suffix in [".pkl", ".joblib"]:
        return joblib.load(model_path)

    if suffix in [".keras", ".h5"]:
        from tensorflow.keras.models import load_model
        return load_model(model_path)

    raise ValueError(f"暂不支持的模型格式：{model_path}")


def find_one_file(folder: Path, patterns, name: str):
    """
    在指定文件夹中按多个规则查找文件。
    """
    matched_files = []

    for pattern in patterns:
        matched_files.extend(list(folder.glob(pattern)))

    matched_files = list(dict.fromkeys(matched_files))

    if len(matched_files) == 0:
        raise FileNotFoundError(
            f"在 {folder} 中没有找到 {name}，匹配规则：{patterns}"
        )

    if len(matched_files) > 1:
        print(f"警告：在 {folder} 中找到多个 {name}，默认使用第一个：")
        for p in matched_files:
            print(f"  - {p}")

    return matched_files[0]


def get_model_type_from_path(model_path: Path):
    """
    从 best_gru_model.keras、best_rf_model.pkl 等文件名中提取模型类型。
    """
    name = model_path.name.lower()

    match = re.search(r"best_(.*?)_model", name)

    if match is None:
        raise ValueError(f"无法从模型文件名中识别模型类型：{model_path.name}")

    model_type = match.group(1)

    # 统一命名
    if model_type in ["xgb", "xgboost"]:
        model_type = "xgb"

    return model_type


def directional_symmetry(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) <= 1:
        return 0

    true_direction = np.sign(y_true[1:] - y_true[:-1])
    pred_direction = np.sign(y_pred[1:] - y_pred[:-1])

    return np.mean(true_direction == pred_direction)


# =========================
# 4. 单簇预测函数
# =========================
def predict_one_cluster(class_id: int, cluster_id: int, X_df: pd.DataFrame):
    """
    根据 class_id 和 cluster_id 自动定位：
    /results/result_model/0cluster_1/
        best_gru_model.keras
        scaler_X.pkl
        scaler_y.pkl

    或：
    /results/result_model/0cluster_2/
        best_rf_model.pkl
        scaler_X.pkl
        scaler_y.pkl
    """

    cluster_folder_name = f"{class_id}cluster_{cluster_id}"
    cluster_model_dir = RESULT_MODEL_DIR / cluster_folder_name

    check_path(cluster_model_dir, f"{cluster_folder_name} 子模型目录")

    print(
        f"\n正在预测：class={class_id}, cluster={cluster_id}, "
        f"samples={len(X_df)}, model_dir={cluster_model_dir}"
    )

    model_path = find_one_file(
        cluster_model_dir,
        [
            "best_*_model.pkl",
            "best_*_model.joblib",
            "best_*_model.keras",
            "best_*_model.h5",
        ],
        name="最优预测模型"
    )

    model_type = get_model_type_from_path(model_path)

    scaler_X_path = find_one_file(
        cluster_model_dir,
        [
            "scaler_X.pkl",
            "scaler_x.pkl",
            "*scaler_X*.pkl",
            "*scaler_x*.pkl",
        ],
        name="特征 scaler"
    )

    scaler_y_path = find_one_file(
        cluster_model_dir,
        [
            "scaler_y.pkl",
            "scaler_Y.pkl",
            "*scaler_y*.pkl",
            "*scaler_Y*.pkl",
        ],
        name="目标 scaler"
    )

    print(f"使用模型：{model_path.name}")
    print(f"模型类型：{model_type}")
    print(f"特征 scaler：{scaler_X_path.name}")
    print(f"目标 scaler：{scaler_y_path.name}")

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    model = load_model_auto(model_path)

    X_aligned = align_features(X_df, scaler_X, f"{cluster_folder_name} 子模型输入特征")
    X_scaled = scaler_X.transform(X_aligned)

    if model_type in ["lstm", "gru", "bilstm", "cnn"]:
        X_model = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    else:
        X_model = X_scaled

    if model_path.suffix.lower() in [".keras", ".h5"]:
        pred_scaled = model.predict(X_model, verbose=0)
    else:
        pred_scaled = model.predict(X_model)

    pred_scaled = np.asarray(pred_scaled).reshape(-1, 1)
    pred = scaler_y.inverse_transform(pred_scaled).reshape(-1)

    pred = np.maximum(pred, 0)

    return pred


# =========================
# 5. 加载测试集和分类模型
# =========================
check_path(TEST_PATH, "测试集")
check_path(CLASSIFIER_MODEL_PATH, "分类模型")
check_path(CLASSIFIER_SCALER_PATH, "分类 scaler")

for class_id in [0, 1]:
    check_path(CLUSTER_SCALER_PATHS[class_id], f"class {class_id} 聚类 scaler")
    check_path(KMEANS_PATHS[class_id], f"class {class_id} KMeans 模型")

test = pd.read_csv(TEST_PATH)
print(f"测试集读取完成：{test.shape}")

if TARGET_COL not in test.columns:
    raise ValueError(f"测试集中缺少目标列：{TARGET_COL}")

y_true = test[TARGET_COL].values

drop_cols = predict_columns + ["start_time"]
X_raw = test.drop(columns=drop_cols, errors="ignore").fillna(0)

classifier_scaler = joblib.load(CLASSIFIER_SCALER_PATH)
classifier_model = joblib.load(CLASSIFIER_MODEL_PATH)

X_cls = align_features(X_raw, classifier_scaler, "分类模型输入特征")
X_cls_scaled = classifier_scaler.transform(X_cls)

label_pred_raw = classifier_model.predict(X_cls_scaled)
label_pred = normalize_classifier_output(label_pred_raw)

print("\n分类结果分布：")
print(pd.Series(label_pred).value_counts().sort_index())


# =========================
# 6. 使用对应类别 KMeans 分配簇
# =========================
cluster_pred = np.full(len(test), fill_value=-1, dtype=int)

for class_id in [0, 1]:
    idx = np.where(label_pred == class_id)[0]

    if len(idx) == 0:
        print(f"class {class_id} 没有样本，跳过聚类。")
        continue

    cluster_scaler = joblib.load(CLUSTER_SCALER_PATHS[class_id])
    kmeans = joblib.load(KMEANS_PATHS[class_id])

    X_class_raw = X_raw.iloc[idx].copy()
    X_cluster = align_features(X_class_raw, cluster_scaler, f"class {class_id} 聚类输入特征")
    X_cluster_scaled = cluster_scaler.transform(X_cluster)

    cluster_labels = kmeans.predict(X_cluster_scaled).astype(int)

    cluster_pred[idx] = cluster_labels

    print(f"\nclass {class_id} 聚类分配结果：")
    print(pd.Series(cluster_labels).value_counts().sort_index())


# =========================
# 7. 按 class + cluster 调用对应子模型
# =========================
final_pred = np.full(len(test), fill_value=np.nan, dtype=float)

for class_id in [0, 1]:
    used_clusters = sorted(np.unique(cluster_pred[label_pred == class_id]))

    for cluster_id in used_clusters:
        if cluster_id == -1:
            continue

        sample_idx = np.where(
            (label_pred == class_id) & (cluster_pred == cluster_id)
        )[0]

        if len(sample_idx) == 0:
            continue

        X_sub = X_raw.iloc[sample_idx].copy()

        pred_sub = predict_one_cluster(
            class_id=class_id,
            cluster_id=cluster_id,
            X_df=X_sub
        )

        final_pred[sample_idx] = pred_sub


# =========================
# 8. 检查是否所有样本都有预测值
# =========================
nan_count = np.isnan(final_pred).sum()

if nan_count > 0:
    bad_idx = np.where(np.isnan(final_pred))[0][:20]
    raise ValueError(
        f"存在 {nan_count} 个样本没有得到预测结果，前 20 个索引为：{bad_idx}"
    )


# =========================
# 9. 评价指标
# =========================
rmse = np.sqrt(mean_squared_error(y_true, final_pred))
mae = mean_absolute_error(y_true, final_pred)
mape = np.mean(np.abs((y_true - final_pred) / (y_true + 1)))
ds = directional_symmetry(y_true, final_pred)

print("\n最终测试集结果：")
print(f"Test RMSE: {rmse:.6f}")
print(f"Test MAE : {mae:.6f}")
print(f"Test MAPE: {mape:.6f}")
print(f"Test DS  : {ds:.6f}")


# =========================
# 10. 保存预测结果
# =========================
result_df = pd.DataFrame({
    "true": y_true,
    "predicted": final_pred,
    "class_label": label_pred,
    "cluster_id": cluster_pred,
    "cluster_model_dir": [
        f"{label_pred[i]}cluster_{cluster_pred[i]}" for i in range(len(test))
    ],
    "abs_error": np.abs(y_true - final_pred),
    "relative_error": np.abs(y_true - final_pred) / (y_true + 1),
})

if "start_time" in test.columns:
    result_df.insert(0, "start_time", test["start_time"])

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"\n预测结果已保存：{OUTPUT_PATH}")