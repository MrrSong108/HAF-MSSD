import json
import pdb
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# =========================
# 1. 路径配置
# =========================


PROJECT_DIR = Path("/root/autodl-tmp/airport_project")

MODEL_PATH = PROJECT_DIR / "data/xiao/check/class/lstm/lstm_classifier.keras"
SCALER_PATH = PROJECT_DIR / "data/xiao/check/class/lstm/scaler_X.pkl"

TRAIN_PATH = PROJECT_DIR / "data/xiao/check/train.csv"
VALIDATION_PATH = PROJECT_DIR / "data/xiao/check/test.csv"

OUTPUT_DIR = PROJECT_DIR / "data/xiao/check/cluster/kmeans_dbi_search"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 0类：困难样本；1类：友好样本
CLASS_INFO = {
    0: {
        "name": "challenging",
        "n_clusters": 10
    },
    1: {
        "name": "friendly",
        "n_clusters": 16
    }
}

TARGET_COL = "predict_T2_0.5"
RANDOM_STATE = 42


# =========================
# 2. 需要排除的预测目标列
# =========================
predict_columns = [
    'predict_T2_0.5', 'predict_T2_1', 'predict_T2_1.5', 'predict_T2_2', 'predict_T2_3',
    'predict_T2_4', 'predict_T2_5', 'predict_T2_6', 'predict_T2_7', 'predict_T2_8',
    'predict_T2_9', 'predict_T2_10', 'predict_T2_11', 'predict_T2_12', 'predict_T2_13',
    'predict_T2_14', 'predict_T2_15', 'predict_T2_16', 'predict_T2_17', 'predict_T2_18',
    'predict_T2_19', 'predict_T2_20', 'predict_T2_21', 'predict_T2_22', 'predict_T2_23', 'predict_T2_24',
    'predict_T3_0.5', 'predict_T3_1', 'predict_T3_1.5', 'predict_T3_2', 'predict_T3_3',
    'predict_T3_4', 'predict_T3_5', 'predict_T3_6', 'predict_T3_7', 'predict_T3_8',
    'predict_T3_9', 'predict_T3_10', 'predict_T3_11', 'predict_T3_12', 'predict_T3_13',
    'predict_T3_14', 'predict_T3_15', 'predict_T3_16', 'predict_T3_17', 'predict_T3_18',
    'predict_T3_19', 'predict_T3_20', 'predict_T3_21', 'predict_T3_22', 'predict_T3_23', 'predict_T3_24'
]


# =========================
# 3. 工具函数
# =========================
def get_feature_columns(df: pd.DataFrame, scaler) -> list:
    """
    优先使用RF分类模型训练时scaler记录的特征名。
    如果scaler没有feature_names_in_，则按排除法生成特征列。
    """
    if hasattr(scaler, "feature_names_in_"):
        feature_cols = list(scaler.feature_names_in_)
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"当前数据缺少scaler训练时使用的特征列: {missing_cols[:10]}")
        return feature_cols

    drop_cols = set(predict_columns + [
        "start_time",
        "label",
        "true_label",
        "pred_label",
        "class_label",
        "cluster",
        "cluster_global"
    ])

    feature_cols = [c for c in df.columns if c not in drop_cols]

    if TARGET_COL in feature_cols:
        feature_cols.remove(TARGET_COL)

    return feature_cols


def build_feature_matrix(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    构造模型输入特征。
    非数值列会尝试转成数值，无法转换的置为NaN，最后统一填充为0。
    """
    X = df[feature_cols].copy()

    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def predict_class(df: pd.DataFrame, feature_cols: list, scaler, model) -> np.ndarray:
    """
    使用RF分类模型预测样本类别。
    0类：困难样本
    1类：友好样本
    """
    X_raw = build_feature_matrix(df, feature_cols)

    if hasattr(scaler, "feature_names_in_"):
        X_scaled = scaler.transform(X_raw)
    else:
        X_scaled = scaler.transform(X_raw.values)

    # pred_label = model.predict(X_scaled)
    X_input = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    proba = model.predict(
        X_input,
        batch_size=128,
        verbose=0
    ).reshape(-1)
    pred_label = (proba >= 0.5).astype(int)
    return pred_label.astype(int)


def save_cluster_csv(data: pd.DataFrame, feature_cols: list, class_label: int, split_name: str, class_dir: Path):
    """
    保存每个类别下每个簇的数据。
    下游预测模型读取时，只需要 drop cluster 和 TARGET_COL 即可。
    """
    save_cols = feature_cols + ["cluster", TARGET_COL]

    for cluster_id in sorted(data["cluster"].unique()):
        cluster_data = data[data["cluster"] == cluster_id].copy()
        save_path = class_dir / f"{class_label}cluster_{cluster_id}_{split_name}.csv"
        cluster_data[save_cols].to_csv(save_path, index=False)

        print(f"{split_name} | 类别 {class_label} | 簇 {cluster_id} | 样本数: {len(cluster_data)} | 已保存: {save_path}")


# =========================
# 4. 读取模型和数据
# =========================
print("正在读取分类模型和scaler...")
# model = joblib.load(MODEL_PATH)
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

if hasattr(scaler, "set_output"):
    scaler.set_output(transform="default")

print("正在读取train和validation数据...")
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VALIDATION_PATH)

if TARGET_COL not in train_df.columns:
    raise ValueError(f"train数据中缺少目标列: {TARGET_COL}")

if TARGET_COL not in val_df.columns:
    raise ValueError(f"validation数据中缺少目标列: {TARGET_COL}")

feature_cols = get_feature_columns(train_df, scaler)

print(f"特征数量: {len(feature_cols)}")
print(f"train样本数: {len(train_df)}")
print(f"validation样本数: {len(val_df)}")


# =========================
# 5. 对train和validation分类
# =========================
print("\n正在使用分类模型预测train类别...")
train_df["label"] = predict_class(train_df, feature_cols, scaler, model)

print("正在使用分类模型预测validation类别...")
val_df["label"] = predict_class(val_df, feature_cols, scaler, model)

print("\ntrain分类结果:")
print(train_df["label"].value_counts().sort_index())

print("\nvalidation分类结果:")
print(val_df["label"].value_counts().sort_index())


# =========================
# 6. 对0类和1类分别在train上拟合KMeans
# =========================
all_train_clustered = []
all_val_clustered = []
summary = {}

for class_label, info in CLASS_INFO.items():
    class_name = info["name"]
    n_clusters = info["n_clusters"]

    print("\n" + "=" * 80)
    print(f"开始处理类别 {class_label}: {class_name} | K={n_clusters}")
    print("=" * 80)

    class_dir = OUTPUT_DIR / f"class_{class_label}_{class_name}"
    class_dir.mkdir(parents=True, exist_ok=True)

    train_sub = train_df[train_df["label"] == class_label].copy()
    val_sub = val_df[val_df["label"] == class_label].copy()

    if len(train_sub) < n_clusters:
        raise ValueError(
            f"类别 {class_label} 的train样本数为 {len(train_sub)}，小于聚类数 {n_clusters}，无法KMeans聚类。"
        )

    # 聚类特征
    X_train_raw = build_feature_matrix(train_sub, feature_cols)
    X_val_raw = build_feature_matrix(val_sub, feature_cols) if len(val_sub) > 0 else None

    # 只在train上fit聚类scaler
    cluster_scaler = StandardScaler()
    X_train_scaled = cluster_scaler.fit_transform(X_train_raw)

    # 只在train上fit KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10
    )

    train_sub["cluster"] = kmeans.fit_predict(X_train_scaled)
    train_sub["cluster_global"] = train_sub["label"] * 1000 + train_sub["cluster"]

    # validation只用已有scaler和KMeans分配簇
    if len(val_sub) > 0:
        X_val_scaled = cluster_scaler.transform(X_val_raw)
        val_sub["cluster"] = kmeans.predict(X_val_scaled)
        val_sub["cluster_global"] = val_sub["label"] * 1000 + val_sub["cluster"]
    else:
        val_sub["cluster"] = []
        val_sub["cluster_global"] = []

    # 计算DBI：使用标准化后的train聚类空间
    dbi = davies_bouldin_score(X_train_scaled, train_sub["cluster"])

    print(f"\n类别 {class_label} train聚类分布:")
    print(train_sub["cluster"].value_counts().sort_index())

    print(f"\n类别 {class_label} validation聚类分布:")
    if len(val_sub) > 0:
        print(val_sub["cluster"].value_counts().sort_index())
    else:
        print("validation中没有该类别样本")

    print(f"\n类别 {class_label} DBI: {dbi:.6f}")
    print(f"类别 {class_label} KMeans inertia: {kmeans.inertia_:.6f}")

    # 保存模型和scaler
    joblib.dump(cluster_scaler, class_dir / f"cluster_scaler_label{class_label}.pkl")
    joblib.dump(kmeans, class_dir / f"kmeans_label{class_label}.pkl")

    # 保存每个簇的数据
    save_cluster_csv(train_sub, feature_cols, class_label, "train", class_dir)

    if len(val_sub) > 0:
        save_cluster_csv(val_sub, feature_cols, class_label, "validation", class_dir)

    # 保存类别整体数据
    train_save_cols = feature_cols + ["label", "cluster", "cluster_global", TARGET_COL]
    val_save_cols = feature_cols + ["label", "cluster", "cluster_global", TARGET_COL]

    train_sub[train_save_cols].to_csv(class_dir / f"class_{class_label}_{class_name}_train_all.csv", index=False)

    if len(val_sub) > 0:
        val_sub[val_save_cols].to_csv(class_dir / f"class_{class_label}_{class_name}_validation_all.csv", index=False)

    all_train_clustered.append(train_sub[train_save_cols])
    if len(val_sub) > 0:
        all_val_clustered.append(val_sub[val_save_cols])

    summary[str(class_label)] = {
        "class_name": class_name,
        "n_clusters": n_clusters,
        "train_samples": int(len(train_sub)),
        "validation_samples": int(len(val_sub)),
        "dbi_train_scaled": float(dbi),
        "kmeans_inertia": float(kmeans.inertia_),
        "train_cluster_counts": {
            str(k): int(v) for k, v in train_sub["cluster"].value_counts().sort_index().items()
        },
        "validation_cluster_counts": {
            str(k): int(v) for k, v in val_sub["cluster"].value_counts().sort_index().items()
        } if len(val_sub) > 0 else {}
    }


# =========================
# 7. 保存总体聚类结果
# =========================
train_clustered_all = pd.concat(all_train_clustered, axis=0).sort_index()
train_clustered_all.to_csv(OUTPUT_DIR / "train_labeled_clustered_all.csv", index=False)

if len(all_val_clustered) > 0:
    val_clustered_all = pd.concat(all_val_clustered, axis=0).sort_index()
    val_clustered_all.to_csv(OUTPUT_DIR / "validation_labeled_clustered_all.csv", index=False)

# 保存特征列，后续推理必须保持一致
with open(OUTPUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "kmeans_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 80)
print("KMeans聚类完成")
print(f"结果保存路径: {OUTPUT_DIR}")
print("=" * 80)