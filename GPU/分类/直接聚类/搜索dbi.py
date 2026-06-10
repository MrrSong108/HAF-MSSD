import json

import joblib
import matplotlib
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import pdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

matplotlib.use('TkAgg')
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


# 定义搜索函数
def search_best_kmeans_DBIs(data):
    X = data.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbi_scores = {}
    for n_clusters in range(2, 20):  # KMeans聚类数量从2到20
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_scaled)
        dbi = davies_bouldin_score(X_scaled, clusters)
        dbi_scores[n_clusters] = dbi
        print(f"K={n_clusters}, DBI={dbi:.4f}")

    # 取 DBI 最小的聚类数量
    best_k = min(dbi_scores, key=dbi_scores.get)
    best_dbi = dbi_scores[best_k]
    print(f"\n>> 最佳聚类数: K={best_k}, 最小DBI={best_dbi:.4f}\n")
    return dbi_scores, best_k, best_dbi


dbi_scores, best_k, best_dbi = search_best_kmeans_DBIs(features_train)

def plot_dbi_curve(dbi_scores):
    plt.plot(list(dbi_scores.keys()), list(dbi_scores.values()), marker='o')
    plt.xlabel('聚类数 (K)')
    plt.ylabel('Davies-Bouldin 指数')
    plt.title(f'全体样本 DBI 曲线')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

plot_dbi_curve(dbi_scores)