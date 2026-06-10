import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # AutoDL/Linux无图形界面时必须使用Agg
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
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

OUTPUT_DIR = PROJECT_DIR / "data/xiao/check/cluster/kmeans_dbi_search"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "predict_T2_0.5"
RANDOM_STATE = 42

# 0类：困难样本；1类：友好样本
CLASS_NAME_MAP = {
    0: "challenging",
    1: "friendly"
}


# =========================
# 2. 预测目标列，不参与分类和聚类
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
def get_feature_columns(df, scaler):
    """
    获取RF分类模型使用的特征列。
    如果RF的scaler保存了feature_names_in_，优先使用；
    否则通过排除目标列、时间列、标签列、预测结果列来构造特征列。
    """

    if hasattr(scaler, "feature_names_in_"):
        feature_cols = list(scaler.feature_names_in_)
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"数据中缺少RF训练时使用的特征列: {missing_cols[:10]}")
        return feature_cols

    drop_cols = set(predict_columns + [
        "start_time",
        "label",
        "true_label",
        "pred_label",
        "class_label",
        "cluster",
        "cluster_global",
        "y_true",
        "y_pred",
        "true_value",
        "pred_value",
        "prediction",
        "error",
        "abs_error",
        "relative_error",
        "difficulty_label"
    ])

    feature_cols = [c for c in df.columns if c not in drop_cols]

    return feature_cols


def build_feature_matrix(df, feature_cols):
    """
    构造特征矩阵，非数值列转为数值，无法转换的填0。
    """
    X = df[feature_cols].copy()

    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def predict_label(df, feature_cols, scaler, model):
    """
    使用已经训练好的RF分类模型预测样本类别。
    0类：困难样本
    1类：友好样本
    """
    X_raw = build_feature_matrix(df, feature_cols)

    if hasattr(scaler, "feature_names_in_"):
        X_scaled = scaler.transform(X_raw)
    else:
        X_scaled = scaler.transform(X_raw.values)
    X_input = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    # pred_label = model.predict(X_scaled)
    proba = model.predict(
        X_input,
        batch_size=128,
        verbose=0
    ).reshape(-1)
    pred_label = (proba >= 0.5).astype(int)
    return pred_label.astype(int)


def search_best_kmeans_dbi(data, feature_cols, class_label, k_min=2, k_max=20):
    """
    对某一类样本搜索最佳K。
    注意：
    这里使用新的StandardScaler进行聚类标准化；
    不要使用RF分类模型的scaler_X.fit_transform，否则会覆盖分类scaler的含义。
    """
    class_name = CLASS_NAME_MAP.get(class_label, f"class_{class_label}")

    X_raw = build_feature_matrix(data, feature_cols)

    cluster_scaler = StandardScaler()
    X_scaled = cluster_scaler.fit_transform(X_raw)

    dbi_scores = {}

    max_k_allowed = min(k_max, len(data) - 1)

    if max_k_allowed < k_min:
        raise ValueError(f"类别 {class_label} 样本数过少，无法搜索KMeans。样本数: {len(data)}")

    print("\n" + "=" * 80)
    print(f"开始搜索类别 {class_label} ({class_name}) 的最佳K")
    print(f"样本数: {len(data)}")
    print("=" * 80)

    for k in range(k_min, max_k_allowed + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        clusters = kmeans.fit_predict(X_scaled)
        dbi = davies_bouldin_score(X_scaled, clusters)
        dbi_scores[k] = float(dbi)

        print(f"class={class_label}, K={k}, DBI={dbi:.6f}")

    best_k = min(dbi_scores, key=dbi_scores.get)
    best_dbi = dbi_scores[best_k]

    print(f"\n类别 {class_label} ({class_name}) 最佳K: {best_k}")
    print(f"类别 {class_label} ({class_name}) 最小DBI: {best_dbi:.6f}")

    return dbi_scores, best_k, best_dbi


def plot_dbi_curve(dbi_scores, class_label, save_path):
    """
    保存DBI曲线图。
    AutoDL中不建议plt.show()，直接保存图片即可。
    """
    x = list(dbi_scores.keys())
    y = list(dbi_scores.values())

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Davies-Bouldin Index")
    plt.title(f"DBI Curve for Class {class_label}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 4. 主程序
# =========================
if __name__ == "__main__":

    print("正在加载分类模型和scaler...")
    model = load_model(MODEL_PATH)
    # rf_model = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if hasattr(scaler, "set_output"):
        scaler.set_output(transform="default")

    print("正在读取训练数据...")
    train_df = pd.read_csv(TRAIN_PATH)

    print("\n当前训练数据列名如下：")
    print(train_df.columns.tolist())

    if TARGET_COL not in train_df.columns:
        print(f"\n注意：训练数据中没有目标列 {TARGET_COL}，本次DBI搜索不需要目标列，因此继续运行。")

    feature_cols = get_feature_columns(train_df, scaler)

    print(f"\n训练样本数: {len(train_df)}")
    print(f"用于分类和KMeans聚类的特征数量: {len(feature_cols)}")
    print("前10个特征列:", feature_cols[:10])

    print("\n正在使用分类模型预测train类别...")
    train_df["label"] = predict_label(train_df, feature_cols, scaler, model)

    print("\n分类后的train样本分布:")
    print(train_df["label"].value_counts().sort_index())

    labeled_train_path = OUTPUT_DIR / "train_label.csv"
    train_df.to_csv(labeled_train_path, index=False)
    print(f"\n带分类标签的train已保存: {labeled_train_path}")

    summary = {}

    for class_label in [0, 1]:
        class_data = train_df[train_df["label"] == class_label].copy()

        if len(class_data) == 0:
            print(f"类别 {class_label} 没有样本，跳过。")
            continue

        dbi_scores, best_k, best_dbi = search_best_kmeans_dbi(
            data=class_data,
            feature_cols=feature_cols,
            class_label=class_label,
            k_min=2,
            k_max=20
        )

        class_name = CLASS_NAME_MAP.get(class_label, f"class_{class_label}")

        fig_path = OUTPUT_DIR / f"dbi_curve_class_{class_label}_{class_name}.png"
        plot_dbi_curve(dbi_scores, class_label, fig_path)

        dbi_result_path = OUTPUT_DIR / f"dbi_scores_class_{class_label}_{class_name}.json"
        with open(dbi_result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "class_label": class_label,
                    "class_name": class_name,
                    "sample_count": int(len(class_data)),
                    "dbi_scores": {str(k): v for k, v in dbi_scores.items()},
                    "best_k": int(best_k),
                    "best_dbi": float(best_dbi)
                },
                f,
                ensure_ascii=False,
                indent=2
            )

        summary[str(class_label)] = {
            "class_name": class_name,
            "sample_count": int(len(class_data)),
            "best_k": int(best_k),
            "best_dbi": float(best_dbi),
            "dbi_scores": {str(k): v for k, v in dbi_scores.items()},
            "figure_path": str(fig_path),
            "json_path": str(dbi_result_path)
        }

    summary_path = OUTPUT_DIR / "dbi_search_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("DBI搜索完成")
    print(f"结果保存目录: {OUTPUT_DIR}")
    print(f"总结果文件: {summary_path}")
    print("=" * 80)