from pathlib import Path
import shutil


# =========================
# 1. 基础路径配置
# =========================

BASE_DIR = Path("/root/autodl-tmp/airport_project/results/result_cluster")
TARGET_DIR = Path("/root/autodl-tmp/airport_project/results/result_model")

TARGET_DIR.mkdir(parents=True, exist_ok=True)

# 是否覆盖目标目录中已存在的同名 cluster 文件夹
# False：目标已存在则跳过
# True ：目标已存在则删除后重新复制
OVERWRITE = False


# =========================
# 2. 各模型源目录配置
# =========================

MODEL_DIRS = {
    "gru": BASE_DIR / "gru_cluster_models_split82_plus_validation",
    "lstm": BASE_DIR / "lstm_cluster_models_split82_plus_validation",
    "cnn": BASE_DIR / "cnn_cluster_models_split82_plus_validation",
    "xgb": BASE_DIR / "xgb_cluster_models_split82_plus_validation",
    "rf": BASE_DIR / "rf_cluster_models_split82_plus_validation",
}


# =========================
# 3. 指定需要复制的 cluster
# =========================

TASKS = [
    # GRU
    ("gru", "1cluster_9"),
    ("gru", "1cluster_12"),
    ("gru", "0cluster_1"),
    ("gru", "0cluster_7"),

    # LSTM
    ("lstm", "1cluster_13"),
    ("lstm", "1cluster_15"),
    ("lstm", "0cluster_4"),

    # CNN
    ("cnn", "1cluster_0"),
    ("cnn", "1cluster_6"),
    ("cnn", "1cluster_7"),

    # XGB
    ("xgb", "1cluster_2"),
    ("xgb", "1cluster_4"),
    ("xgb", "1cluster_8"),
    ("xgb", "1cluster_10"),
    ("xgb", "1cluster_14"),

    # RF
    ("rf", "1cluster_1"),
    ("rf", "1cluster_3"),
    ("rf", "1cluster_11"),
    ("rf", "0cluster_0"),
    ("rf", "0cluster_2"),
    ("rf", "0cluster_3"),
    ("rf", "0cluster_5"),
    ("rf", "0cluster_6"),
    ("rf", "0cluster_8"),
    ("rf", "0cluster_9"),
]


# =========================
# 4. 根据 cluster 名称判断类别目录
# =========================

def get_class_dir(cluster_name: str) -> str:
    if cluster_name.startswith("0cluster_"):
        return "class_0_challenging"
    elif cluster_name.startswith("1cluster_"):
        return "class_1_friendly"
    else:
        raise ValueError(f"无法识别 cluster 类别：{cluster_name}")


# =========================
# 5. 执行复制
# =========================

def copy_selected_clusters():
    success_count = 0
    skip_count = 0
    missing_count = 0
    fail_count = 0

    print(f"目标目录：{TARGET_DIR}")
    print("开始复制指定 cluster 文件夹...")
    print("=" * 80)

    for model_name, cluster_name in TASKS:
        model_dir = MODEL_DIRS[model_name]
        class_dir = get_class_dir(cluster_name)

        source_path = model_dir / class_dir / cluster_name
        target_path = TARGET_DIR / cluster_name

        print(f"模型：{model_name} | cluster：{cluster_name}")
        print(f"源路径：{source_path}")
        print(f"目标路径：{target_path}")

        if not source_path.exists():
            print("状态：源目录不存在，跳过")
            missing_count += 1
            print("-" * 80)
            continue

        if not source_path.is_dir():
            print("状态：源路径不是文件夹，跳过")
            missing_count += 1
            print("-" * 80)
            continue

        if target_path.exists():
            if OVERWRITE:
                print("状态：目标目录已存在，执行覆盖")
                shutil.rmtree(target_path)
            else:
                print("状态：目标目录已存在，跳过")
                skip_count += 1
                print("-" * 80)
                continue

        try:
            shutil.copytree(source_path, target_path)
            print("状态：复制成功")
            success_count += 1
        except Exception as e:
            print(f"状态：复制失败，错误信息：{e}")
            fail_count += 1

        print("-" * 80)

    print("复制任务完成。")
    print(f"成功复制数量：{success_count}")
    print(f"已存在跳过数量：{skip_count}")
    print(f"源目录缺失数量：{missing_count}")
    print(f"复制失败数量：{fail_count}")


if __name__ == "__main__":
    copy_selected_clusters()