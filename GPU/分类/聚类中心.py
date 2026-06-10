import re
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# =========================
# 1. 路径配置
# =========================
BASE_DIR = Path("/root/autodl-tmp/airport_project/results/result_cluster/result_rf_kmeans")

CLASS_DIRS = {
    0: BASE_DIR / "class_0_challenging",
    1: BASE_DIR / "class_1_friendly",
}

# 只用 train 文件计算聚类中心
CENTER_SPLIT = "train"   # 可选："train"、"validation"、"all"

DROP_COLUMNS = ["cluster", "predict_T2_0.5"]

OUTPUT_CENTER_PATH = BASE_DIR / "cluster_centers.pkl"
OUTPUT_META_PATH = BASE_DIR / "cluster_centers_meta.json"


# =========================
# 2. scaler 自动查找
# =========================
def find_scaler(class_id: int, class_dir: Path):
    """
    自动查找当前类别对应的 scaler。
    优先在当前类别文件夹中查找，其次在总目录中查找。
    """

    candidate_paths = []

    candidate_paths.extend(class_dir.glob("*scaler*.pkl"))
    candidate_paths.extend(BASE_DIR.glob(f"*class_{class_id}*scaler*.pkl"))
    candidate_paths.extend(BASE_DIR.glob(f"*cluster{class_id}*scaler*.pkl"))
    candidate_paths.extend(BASE_DIR.glob(f"*{class_id}*scaler*.pkl"))

    candidate_paths = list(dict.fromkeys(candidate_paths))

    if len(candidate_paths) == 0:
        raise FileNotFoundError(
            f"没有找到 class {class_id} 对应的 scaler，请手动检查 scaler 路径。"
        )

    if len(candidate_paths) > 1:
        print(f"\n警告：class {class_id} 找到多个 scaler，默认使用第一个：")
        for p in candidate_paths:
            print(f"  - {p}")

    scaler_path = candidate_paths[0]
    scaler = joblib.load(scaler_path)

    return scaler, scaler_path


# =========================
# 3. 识别文件名中的 cluster 编号和数据集划分
# =========================
def parse_cluster_file(file_path: Path):
    """
    解析类似以下文件名：
    0cluster_0_train.csv
    0cluster_0_validation.csv
    1cluster_3_train.csv
    1cluster_3_validation.csv

    返回：
    file_class_id, cluster_id, split_name
    """

    stem = file_path.stem

    pattern = r"^(\d+)cluster[_-]?(\d+)_(train|validation|valid|val|test)$"
    match = re.match(pattern, stem, flags=re.IGNORECASE)

    if match is None:
        return None, None, None

    file_class_id = int(match.group(1))
    cluster_id = int(match.group(2))
    split_name = match.group(3).lower()

    if split_name in ["valid", "val"]:
        split_name = "validation"

    return file_class_id, cluster_id, split_name


def get_cluster_csv_files(class_id: int, class_dir: Path):
    """
    自动获取当前类别目录下符合命名规则的簇文件。
    根据 CENTER_SPLIT 控制使用 train、validation 或全部文件。
    """

    cluster_files = []

    for file_path in class_dir.glob("*.csv"):
        file_class_id, cluster_id, split_name = parse_cluster_file(file_path)

        if file_class_id is None:
            continue

        if file_class_id != class_id:
            print(f"警告：文件类别与当前目录类别不一致，跳过：{file_path.name}")
            continue

        if CENTER_SPLIT != "all" and split_name != CENTER_SPLIT:
            continue

        cluster_files.append(
            {
                "cluster_id": cluster_id,
                "split": split_name,
                "path": file_path,
            }
        )

    cluster_files = sorted(
        cluster_files,
        key=lambda x: (x["cluster_id"], x["split"])
    )

    return cluster_files


# =========================
# 4. 特征处理
# =========================
def prepare_features(df: pd.DataFrame, scaler):
    """
    删除无关列，只保留数值特征。
    如果 scaler 中保存了 feature_names_in_，则严格按照训练时特征顺序对齐。
    """

    df = df.copy()

    existing_drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    numeric_df = df.select_dtypes(include=[np.number])

    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)

        missing_cols = [col for col in feature_names if col not in numeric_df.columns]
        extra_cols = [col for col in numeric_df.columns if col not in feature_names]

        if missing_cols:
            raise ValueError(
                f"当前簇数据缺少 scaler 训练时使用的特征列，"
                f"前10个缺失列为：{missing_cols[:10]}，"
                f"共缺少 {len(missing_cols)} 个。"
            )

        if extra_cols:
            print(f"提示：存在 scaler 未使用的额外数值列，将自动忽略：{extra_cols[:10]}")

        numeric_df = numeric_df[feature_names]

    return numeric_df


# =========================
# 5. 自动计算聚类中心
# =========================
cluster_centers = {}
cluster_meta = {}

for class_id, class_dir in CLASS_DIRS.items():

    if not class_dir.exists():
        raise FileNotFoundError(f"类别目录不存在：{class_dir}")

    print(f"\n正在处理 class {class_id}: {class_dir}")

    scaler, scaler_path = find_scaler(class_id, class_dir)
    print(f"使用 scaler: {scaler_path}")

    cluster_files = get_cluster_csv_files(class_id, class_dir)

    if len(cluster_files) == 0:
        raise FileNotFoundError(
            f"在 {class_dir} 中没有找到符合规则的 {CENTER_SPLIT} 簇文件。"
        )

    print(f"检测到 {len(cluster_files)} 个 {CENTER_SPLIT} 簇文件。")

    for item in cluster_files:

        cluster_id = item["cluster_id"]
        split_name = item["split"]
        csv_path = item["path"]

        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"警告：文件为空，跳过：{csv_path.name}")
            continue

        features = prepare_features(df, scaler)

        if features.empty:
            print(f"警告：没有可用数值特征，跳过：{csv_path.name}")
            continue

        scaled_data = scaler.transform(features)

        center = scaled_data.mean(axis=0)

        # 例如：0cluster0、1cluster7
        center_key = f"{class_id}cluster{cluster_id}"

        cluster_centers[center_key] = center

        cluster_meta[center_key] = {
            "class_id": class_id,
            "cluster_id": cluster_id,
            "split": split_name,
            "csv_file": str(csv_path),
            "sample_count": int(len(df)),
            "feature_count": int(features.shape[1]),
            "scaler_path": str(scaler_path),
        }

        print(
            f"完成：{center_key} | split={split_name} | "
            f"样本数={len(df)} | 特征数={features.shape[1]} | 文件={csv_path.name}"
        )


# =========================
# 6. 保存结果
# =========================
joblib.dump(cluster_centers, OUTPUT_CENTER_PATH)

with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
    json.dump(cluster_meta, f, ensure_ascii=False, indent=4)

print("\n聚类中心保存完成：")
print(f"中心文件：{OUTPUT_CENTER_PATH}")
print(f"元信息文件：{OUTPUT_META_PATH}")

print("\n聚类中心 keys：")
for key in cluster_centers.keys():
    print(key)