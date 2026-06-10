import os
import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =========================================================
# 1. 路径配置
# =========================================================
PROJECT_DIR = Path("/root/autodl-tmp/airport_project")

TRAIN_PATH = PROJECT_DIR / "data/xiao/train.csv"
VALIDATION_PATH = PROJECT_DIR / "data/xiao/test.csv"
TEST_PATH = PROJECT_DIR / "data/xiao/test.csv"

OUTPUT_DIR = PROJECT_DIR / "data/xiao/TCFM/tcfm_xgb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "predict_T2_0.5"
TIME_COL = "start_time"

RANDOM_STATE = 42

# KMeans 聚类数搜索范围
K_MIN = 2
K_MAX = 20

# 每个簇最少样本数，过小的簇不单独训练模型
MIN_CLUSTER_SAMPLES = 100


# =========================================================
# 2. 评价指标
# =========================================================
def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # 与你之前论文中常用的平滑 MAPE 保持一致，避免 y_true=0
    mape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1)) * 100

    # Directional Symmetry，方向一致性
    if len(y_true) > 1:
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        ds = np.mean(np.sign(true_diff) == np.sign(pred_diff))
    else:
        ds = np.nan

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "DS": float(ds)
    }


# =========================================================
# 3. 读取数据
# =========================================================
def read_data(path):
    df = pd.read_csv(path)

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.sort_values(TIME_COL).reset_index(drop=True)

    return df


print("正在读取数据...")
train_df = read_data(TRAIN_PATH)
valid_df = read_data(VALIDATION_PATH)
test_df = read_data(TEST_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {valid_df.shape}")
print(f"Test shape: {test_df.shape}")


# =========================================================
# 4. 基础检查
# =========================================================
for name, df in [("train", train_df), ("validation", valid_df), ("test", test_df)]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"{name} 数据中缺少目标列: {TARGET_COL}")

if TIME_COL not in train_df.columns:
    print(f"警告：未找到时间列 {TIME_COL}，将按原始顺序处理。")


# =========================================================
# 5. 删除无效列与目标泄露列
# =========================================================
def build_feature_data(train_df, valid_df, test_df, target_col):
    """
    删除：
    1. start_time 时间列
    2. 所有 predict_* 目标列，防止目标泄露
    3. 训练集中全空列
    4. 训练集中常数列
    """

    # 所有预测目标列都删除，避免使用其他未来预测目标造成泄露
    predict_cols = [c for c in train_df.columns if c.startswith("predict_")]

    drop_cols = set(predict_cols)
    drop_cols.add(target_col)

    if TIME_COL in train_df.columns:
        drop_cols.add(TIME_COL)

    drop_cols = [c for c in drop_cols if c in train_df.columns]

    y_train = train_df[target_col].copy()
    y_valid = valid_df[target_col].copy()
    y_test = test_df[target_col].copy()

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
    X_test = test_df.drop(columns=drop_cols, errors="ignore")

    # 只保留三份数据共有的列
    common_cols = list(set(X_train.columns) & set(X_valid.columns) & set(X_test.columns))
    common_cols = sorted(common_cols)

    X_train = X_train[common_cols]
    X_valid = X_valid[common_cols]
    X_test = X_test[common_cols]

    # 删除训练集中全空列
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        print(f"删除训练集中全空列数量: {len(all_nan_cols)}")
        X_train = X_train.drop(columns=all_nan_cols)
        X_valid = X_valid.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)

    # 删除训练集中常数列
    nunique = X_train.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"删除训练集中常数列数量: {len(constant_cols)}")
        X_train = X_train.drop(columns=constant_cols)
        X_valid = X_valid.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)

    # 缺失值使用训练集均值填充
    fill_values = X_train.mean(numeric_only=True)

    X_train = X_train.fillna(fill_values)
    X_valid = X_valid.fillna(fill_values)
    X_test = X_test.fillna(fill_values)

    # 如果还有缺失，统一填 0
    X_train = X_train.fillna(0)
    X_valid = X_valid.fillna(0)
    X_test = X_test.fillna(0)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


X_train, X_valid, X_test, y_train, y_valid, y_test = build_feature_data(
    train_df, valid_df, test_df, TARGET_COL
)

print(f"处理后 X_train shape: {X_train.shape}")
print(f"处理后 X_valid shape: {X_valid.shape}")
print(f"处理后 X_test shape: {X_test.shape}")


# =========================================================
# 6. 选择聚类特征：目标历史窗口特征
# =========================================================
def get_cluster_columns(columns):
    """
    优先选择 queue_countpassed* 作为目标历史时间序列片段特征。
    这些特征最接近单变量目标序列的历史窗口。
    """
    cluster_cols = []

    for c in columns:
        if c.startswith("queue_countpassed"):
            cluster_cols.append(c)

    # 排除可能表示当前总量、非窗口统计的字段，可根据实际情况调整
    # 如果你希望保留 queue_countpassed 本身，可以删除下面这一行
    cluster_cols = [c for c in cluster_cols if c != "queue_countpassed"]

    return cluster_cols


cluster_cols = get_cluster_columns(X_train.columns)

if len(cluster_cols) == 0:
    raise ValueError("未找到 queue_countpassed* 聚类特征，请检查数据列名。")

print(f"用于聚类的目标历史窗口特征数量: {len(cluster_cols)}")
print("聚类特征示例:", cluster_cols[:20])

X_train_cluster = X_train[cluster_cols].copy()
X_valid_cluster = X_valid[cluster_cols].copy()
X_test_cluster = X_test[cluster_cols].copy()


# =========================================================
# 7. 标准化聚类特征
# =========================================================
cluster_scaler = StandardScaler()

X_train_cluster_scaled = cluster_scaler.fit_transform(X_train_cluster)
X_valid_cluster_scaled = cluster_scaler.transform(X_valid_cluster)
X_test_cluster_scaled = cluster_scaler.transform(X_test_cluster)


# =========================================================
# 8. 使用 DBI 搜索最优聚类数 K
# =========================================================
print("\n开始搜索最优聚类数 K...")

dbi_records = []

for k in range(K_MIN, K_MAX + 1):
    print(f"正在计算 K={k} ...")

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    train_labels = kmeans.fit_predict(X_train_cluster_scaled)

    dbi = davies_bouldin_score(X_train_cluster_scaled, train_labels)

    cluster_counts = pd.Series(train_labels).value_counts().sort_index().to_dict()

    dbi_records.append({
        "K": k,
        "DBI": dbi,
        "cluster_counts": cluster_counts
    })

    print(f"K={k}, DBI={dbi:.6f}, cluster_counts={cluster_counts}")

dbi_df = pd.DataFrame([
    {
        "K": r["K"],
        "DBI": r["DBI"],
        "cluster_counts": json.dumps(r["cluster_counts"], ensure_ascii=False)
    }
    for r in dbi_records
])

dbi_path = OUTPUT_DIR / "dbi_search_results.csv"
dbi_df.to_csv(dbi_path, index=False, encoding="utf-8-sig")

best_record = min(dbi_records, key=lambda x: x["DBI"])
best_k = best_record["K"]

print("\n最优聚类数搜索完成")
print(f"Best K = {best_k}")
print(f"Best DBI = {best_record['DBI']:.6f}")
print(f"Best cluster counts = {best_record['cluster_counts']}")


# =========================================================
# 9. 使用最优 K 重新训练 KMeans
# =========================================================
print("\n正在训练最终 KMeans 聚类模型...")

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=10
)

train_cluster_labels = final_kmeans.fit_predict(X_train_cluster_scaled)
valid_cluster_labels = final_kmeans.predict(X_valid_cluster_scaled)
test_cluster_labels = final_kmeans.predict(X_test_cluster_scaled)

print("训练集簇分布:")
print(pd.Series(train_cluster_labels).value_counts().sort_index())

print("验证集簇分布:")
print(pd.Series(valid_cluster_labels).value_counts().sort_index())

print("测试集簇分布:")
print(pd.Series(test_cluster_labels).value_counts().sort_index())


# =========================================================
# 10. 保存聚类标签
# =========================================================
cluster_label_df_train = pd.DataFrame({
    "dataset": "train",
    "cluster": train_cluster_labels,
    "y_true": y_train.values
})

cluster_label_df_valid = pd.DataFrame({
    "dataset": "validation",
    "cluster": valid_cluster_labels,
    "y_true": y_valid.values
})

cluster_label_df_test = pd.DataFrame({
    "dataset": "test",
    "cluster": test_cluster_labels,
    "y_true": y_test.values
})

if TIME_COL in train_df.columns:
    cluster_label_df_train[TIME_COL] = train_df[TIME_COL].values
    cluster_label_df_valid[TIME_COL] = valid_df[TIME_COL].values
    cluster_label_df_test[TIME_COL] = test_df[TIME_COL].values

cluster_label_all = pd.concat(
    [cluster_label_df_train, cluster_label_df_valid, cluster_label_df_test],
    axis=0,
    ignore_index=True
)

cluster_label_all.to_csv(
    OUTPUT_DIR / "cluster_labels.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 11. 训练每个簇内部的 XGBoost 子模型
# =========================================================
def create_xgb_model():
    """
    这里使用相对稳健的默认参数。
    如果你后续需要 Optuna，可以在每个簇内部继续搜索。
    """
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=8
    )
    return model


print("\n开始训练每个簇内部的 XGBoost 子模型...")

cluster_models = {}
cluster_metrics_records = []

# 全局兜底模型：当某个簇样本过少时使用
print("正在训练全局兜底模型...")
global_model = create_xgb_model()
global_model.fit(X_train, y_train)

for cluster_id in range(best_k):
    print(f"\n========== 训练 Cluster {cluster_id} ==========")

    train_idx = np.where(train_cluster_labels == cluster_id)[0]
    valid_idx = np.where(valid_cluster_labels == cluster_id)[0]

    n_train_cluster = len(train_idx)
    n_valid_cluster = len(valid_idx)

    print(f"Cluster {cluster_id} train samples: {n_train_cluster}")
    print(f"Cluster {cluster_id} validation samples: {n_valid_cluster}")

    if n_train_cluster < MIN_CLUSTER_SAMPLES:
        print(
            f"Cluster {cluster_id} 样本数小于 {MIN_CLUSTER_SAMPLES}，"
            f"不单独训练，使用全局模型作为兜底。"
        )
        cluster_models[cluster_id] = global_model
        continue

    X_c_train = X_train.iloc[train_idx]
    y_c_train = y_train.iloc[train_idx]

    model = create_xgb_model()
    model.fit(X_c_train, y_c_train)

    cluster_models[cluster_id] = model

    # 验证集上评估该簇模型
    if n_valid_cluster > 0:
        X_c_valid = X_valid.iloc[valid_idx]
        y_c_valid = y_valid.iloc[valid_idx]

        valid_pred = model.predict(X_c_valid)
        metrics = calc_metrics(y_c_valid, valid_pred)

        record = {
            "cluster": cluster_id,
            "train_samples": n_train_cluster,
            "validation_samples": n_valid_cluster,
            **metrics
        }

        cluster_metrics_records.append(record)

        print(
            f"Cluster {cluster_id} Validation - "
            f"RMSE: {metrics['RMSE']:.4f}, "
            f"MAE: {metrics['MAE']:.4f}, "
            f"MAPE: {metrics['MAPE']:.4f}, "
            f"DS: {metrics['DS']:.4f}"
        )

cluster_metrics_df = pd.DataFrame(cluster_metrics_records)
cluster_metrics_df.to_csv(
    OUTPUT_DIR / "cluster_validation_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 12. TCFM 融合预测函数
# =========================================================
def predict_by_cluster_models(X, cluster_labels, cluster_models):
    """
    每个样本根据所属簇调用对应子模型预测。
    """
    y_pred = np.zeros(len(X), dtype=float)

    for cluster_id in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cluster_id)[0]

        model = cluster_models.get(cluster_id, global_model)

        X_part = X.iloc[idx]
        y_pred[idx] = model.predict(X_part)

    return y_pred


# =========================================================
# 13. 在训练集、验证集、测试集上预测
# =========================================================
print("\n开始进行 TCFM 融合预测...")

train_pred = predict_by_cluster_models(X_train, train_cluster_labels, cluster_models)
valid_pred = predict_by_cluster_models(X_valid, valid_cluster_labels, cluster_models)
test_pred = predict_by_cluster_models(X_test, test_cluster_labels, cluster_models)

train_metrics = calc_metrics(y_train, train_pred)
valid_metrics = calc_metrics(y_valid, valid_pred)
test_metrics = calc_metrics(y_test, test_pred)

print("\n========== TCFM-XGBoost 最终结果 ==========")
print("Train:", train_metrics)
print("Validation:", valid_metrics)
print("Test:", test_metrics)


# =========================================================
# 14. 保存整体评价结果
# =========================================================
overall_metrics_df = pd.DataFrame([
    {"dataset": "train", **train_metrics},
    {"dataset": "validation", **valid_metrics},
    {"dataset": "test", **test_metrics}
])

overall_metrics_df.to_csv(
    OUTPUT_DIR / "overall_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 15. 保存测试集预测结果
# =========================================================
test_result_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": test_pred,
    "cluster": test_cluster_labels
})

if TIME_COL in test_df.columns:
    test_result_df.insert(0, TIME_COL, test_df[TIME_COL].values)

test_result_df["abs_error"] = np.abs(test_result_df["y_true"] - test_result_df["y_pred"])
test_result_df["ape"] = test_result_df["abs_error"] / (np.abs(test_result_df["y_true"]) + 1)

test_result_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 16. 保存模型和参数
# =========================================================
joblib.dump(cluster_scaler, OUTPUT_DIR / "cluster_scaler.pkl")
joblib.dump(final_kmeans, OUTPUT_DIR / "kmeans_model.pkl")
joblib.dump(cluster_models, OUTPUT_DIR / "cluster_xgb_models.pkl")
joblib.dump(global_model, OUTPUT_DIR / "global_xgb_model.pkl")

config = {
    "target_col": TARGET_COL,
    "time_col": TIME_COL,
    "best_k": int(best_k),
    "best_dbi": float(best_record["DBI"]),
    "cluster_cols": cluster_cols,
    "feature_cols": X_train.columns.tolist(),
    "k_search_range": [K_MIN, K_MAX],
    "min_cluster_samples": MIN_CLUSTER_SAMPLES,
    "model": "TCFM-XGBoost",
    "description": "Time-series clustering-based fusion forecasting model using target historical queue features."
}

with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)


print("\n所有结果已保存到:")
print(OUTPUT_DIR)
print("\n主要输出文件包括:")
print("1. dbi_search_results.csv：不同 K 的 DBI 结果")
print("2. cluster_labels.csv：训练集、验证集、测试集的聚类标签")
print("3. cluster_validation_metrics.csv：各簇内部模型验证集结果")
print("4. overall_metrics.csv：整体 Train / Validation / Test 结果")
print("5. test_predictions.csv：测试集真实值、预测值、所属簇")
print("6. kmeans_model.pkl：KMeans 聚类模型")
print("7. cluster_xgb_models.pkl：各簇 XGBoost 子模型")