import pdb

import matplotlib
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
predict_columns = ['predict_T2_0.5', 'predict_T2_1', 'predict_T2_1.5', 'predict_T2_2', 'predict_T2_3',
                   'predict_T2_4', 'predict_T2_5', 'predict_T2_6', 'predict_T2_7', 'predict_T2_8',
                   'predict_T2_9', 'predict_T2_10', 'predict_T2_11', 'predict_T2_12', 'predict_T2_13',
                   'predict_T2_14', 'predict_T2_15', 'predict_T2_16', 'predict_T2_17', 'predict_T2_18',
                   'predict_T2_19', 'predict_T2_20', 'predict_T2_21', 'predict_T2_22', 'predict_T2_23', 'predict_T2_24',
                   'predict_T3_0.5', 'predict_T3_1', 'predict_T3_1.5', 'predict_T3_2', 'predict_T3_3',
                   'predict_T3_4', 'predict_T3_5', 'predict_T3_6', 'predict_T3_7', 'predict_T3_8',
                   'predict_T3_9', 'predict_T3_10', 'predict_T3_11', 'predict_T3_12', 'predict_T3_13',
                   'predict_T3_14', 'predict_T3_15', 'predict_T3_16', 'predict_T3_17', 'predict_T3_18',
                   'predict_T3_19', 'predict_T3_20', 'predict_T3_21', 'predict_T3_22', 'predict_T3_23', 'predict_T3_24']
train = pd.read_csv(r"D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\拼接后数据\train.csv")
features_train = train.drop(columns=predict_columns).drop(columns=['start_time']).fillna(0)
Y = train['predict_T2_0.5']
features_train['predict_T2_0.5'] = Y
# scaler_X = StandardScaler()
# print("len",len(features_train))
# def search_best_kmeans_DBIs(data):
#     target = data['predict_T2_0.5']
#     X = data.drop(columns=['predict_T2_0.5']).copy()
#     X_scaled = scaler_X.fit_transform(X)
#
#     dbi_scores = {}
#     for n_clusters in range(2, 21):  # KMeans聚类数量从2到10
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         clusters = kmeans.fit_predict(X_scaled)
#         dbi = davies_bouldin_score(X_scaled, clusters)
#         dbi_scores[n_clusters] = dbi
#         print(f"K={n_clusters}, DBI={dbi:.4f}")
#
#     # 取 DBI 最小的聚类数量
#     best_k = min(dbi_scores, key=dbi_scores.get)
#     best_dbi = dbi_scores[best_k]
#     print(f"\n>> 最佳聚类数  : K={best_k}, 最小DBI={best_dbi:.4f}\n")
#     return dbi_scores, best_k, best_dbi
# dbi_scores_0, best_k_0, best_dbi_0 = search_best_kmeans_DBIs(features_train)
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# matplotlib.use('TkAgg')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# def plot_dbi_curve(dbi_scores, label_value):
#     ks = list(dbi_scores.keys())
#     dbi_vals = list(dbi_scores.values())
#     plt.figure(figsize=(6, 4))
#     plt.plot(ks, dbi_vals, marker='o')
#     plt.xlabel('聚类数 (K)')
#     plt.ylabel('DBI')
#     plt.title(f'label={label_value} DBI')
#     plt.grid(True)
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#
#     plt.tight_layout()
#     plt.show()
# plot_dbi_curve(dbi_scores_0, '无异常划分')
# pdb.set_trace()
#
# X_train_clustered = features_train_1.drop(columns=['label', 'predict_T2_0.5'])
# 1. 在聚类前保存目标列
X_train_0_target = features_train['predict_T2_0.5']
X_cluster_input = features_train.drop(columns=['predict_T2_0.5'])

cluster0_scaler = StandardScaler()
X_scaled = cluster0_scaler.fit_transform(X_cluster_input)

# 3. 聚类
model = KMeans(n_clusters=8, random_state=42)
cluster_labels = model.fit_predict(X_scaled)

# 4. 保存聚类标签
X_train_0_clustered = X_cluster_input.copy()
X_train_0_clustered['cluster'] = cluster_labels
print(X_train_0_clustered['cluster'].value_counts())

# 5. 合并目标列
cluster0_data = X_train_0_clustered.copy()
cluster0_data['predict_T2_0.5'] = X_train_0_target

for cluster_label in cluster0_data['cluster'].unique():  # 遍历每个聚类
    print(f"正在保存聚类 {cluster_label} 的模型")

    # 5. 针对每个聚类过滤训练数据
    cluster_data_sub = cluster0_data[cluster0_data['cluster'] == cluster_label]
    cluster_data_sub.to_csv(fr'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\无异常划分\聚类划分8类\cluster_{cluster_label}.csv', index=False)
    print(f"聚类 {cluster_label} 的数据已保存为 'cluster_{cluster_label}.csv'")
joblib.dump(cluster0_scaler,'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\无异常划分\聚类划分8类\cluster_scaler.pkl')

dbi_0 = davies_bouldin_score(X_scaled, cluster_labels)
print(f"聚类的DBI: {dbi_0}")

