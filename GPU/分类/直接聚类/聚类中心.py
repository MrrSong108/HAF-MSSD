import joblib
import numpy as np
import pandas as pd
import os

# 设置路径
path = r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\无异常划分\聚类划分8类'
scaler_path = os.path.join(path, r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\无异常划分\聚类划分8类\cluster_scaler.pkl')  # 聚类时使用的统一 scaler

# 读取标准化器
scaler = joblib.load(scaler_path)

# 初始化聚类数据集
clusters = {}
for i in range(8):  # 统一读取 cluster_0.csv 到 cluster_15.csv
    cluster_file = os.path.join(path, f'cluster_{i}.csv')
    if os.path.exists(cluster_file):
        df = pd.read_csv(cluster_file).drop(columns=['cluster', 'predict_T2_0.5'], errors='ignore')
        clusters[f'cluster_{i}'] = df
    else:
        print(f"警告：文件不存在 {cluster_file}")

# 计算每个聚类的中心
cluster_centers = {}
for name, df in clusters.items():
    numeric_data = df.select_dtypes(include=[np.number])
    scaled_data = scaler.transform(numeric_data)
    center = scaled_data.mean(axis=0)  # 归一化后中心
    cluster_centers[name] = center

# 打印聚类中心
for name, center in cluster_centers.items():
    print(f"{name} 的中心向量：\n{center}\n")

# 保存聚类中心
joblib.dump(cluster_centers, os.path.join(path, f'{path}\\cluster_centers.pkl'))
