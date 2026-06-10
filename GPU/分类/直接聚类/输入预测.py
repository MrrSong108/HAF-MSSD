import json
import pdb
import pandas as pd
import joblib
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
centers = joblib.load(r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\无异常划分\聚类划分8类\cluster_centers.pkl')
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

test = pd.read_csv(r"D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\拼接后数据\test.csv")
print('数据已输入')
features_test = test.drop(columns=predict_columns).drop(columns=['start_time']).fillna(0)
scaler = joblib.load(r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\无异常划分\聚类划分8类\cluster_scaler.pkl')

def assign_clusters(df, centers, scaler, prefix):
    """
    根据特征和聚类中心，计算三种相似度加权后，给每个样本分配最相似的中心，返回 one-hot 形式的分配矩阵。

    参数：
    - features_label_df: 包含特征和标签的DataFrame，包含 'predict_T2_0.5' 列作为标签列
    - centers: 聚类中心字典，key是中心名，value是中心特征向量
    - scaler: 用于对特征进行标准化的Scaler对象（必须有transform方法）
    - prefix: 用于筛选聚类中心的前缀字符串，比如 "0" 或 "1"

    返回：
    - result_one_hot: (N, K)的numpy数组，one-hot编码，表示每个样本所属聚类中心
    - max_indices: (N,)的numpy数组，每个样本最相似中心的索引
    - Y_0: 标签列，Series格式
    """

    features_array = df.values
    cluster_centers = {k: v for k, v in centers.items() if k.startswith(prefix)}
    center_names = list(cluster_centers.keys())
    centers_matrix = np.array(list(cluster_centers.values()))
    features_scaled = scaler.transform(features_array)

    # 1. 余弦相似度
    cos_sim = cosine_similarity(features_scaled, centers_matrix)  # N×K

    # 2. 欧几里得距离转相似度
    euc_dist = cdist(features_scaled, centers_matrix, metric='euclidean')
    euc_sim = 1 / (1 + euc_dist)

    # 3. 皮尔逊相关系数
    pearson_sim = np.zeros((features_scaled.shape[0], centers_matrix.shape[0]))
    for i in range(features_scaled.shape[0]):
        for j in range(centers_matrix.shape[0]):
            try:
                pearson_sim[i, j], _ = pearsonr(features_scaled[i], centers_matrix[j])
            except:
                pearson_sim[i, j] = 0

    # 4. 三种相似度求和
    total_sim = cos_sim + euc_sim + pearson_sim  # N×K

    # 5. 找最大相似度对应的聚类中心索引
    max_indices = np.argmax(total_sim, axis=1)

    # 6. 构造one-hot编码结果
    result_one_hot = np.zeros_like(total_sim)
    for i, idx in enumerate(max_indices):
        result_one_hot[i, idx] = 1

    print(f"分到每个聚类中心的数量（前缀={prefix}）：", np.sum(result_one_hot, axis=0).astype(int))

    return result_one_hot, features_array, max_indices, center_names

result, features_array, cluster_indices, cluster_names = assign_clusters(features_test, centers, scaler, prefix="cluster")
all_cluster_ids = [cluster_names[i] for i in cluster_indices]
def predict_with_model(cluster_id: str, model_type: str, features_array, base_path: str):
    """
    使用给定聚类中心编号和模型类型进行预测并还原真实值。

    参数：
    - cluster_id: int，聚类中心编号（如 0, 1, 2...）
    - model_type: str，模型类型（如 'gru', 'lstm', 'cnn', 'rf', 'xgboost'）
    - features_array: np.ndarray，输入的特征数组 (N, D)
    - base_path: str，模型所在目录路径

    返回：
    - 预测还原后的结果 (N, 1)
    """
    # 模型路径拼接
    print(f"正在进行{cluster_id}，预测模型为{model_type}")
    scaler_X_path = fr'{base_path}\scaler_X_{cluster_id}.pkl'
    scaler_y_path = fr'{base_path}\scaler_y_{cluster_id}.pkl'
    model_path = fr'{base_path}\model_{cluster_id}.pkl'

    # 加载模型和Scaler
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    model = joblib.load(model_path)

    X_scaled = scaler_X.transform(features_array)
    X_model = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]) if model_type in ['lstm', 'gru', 'bilstm',
                                                                                          'cnn'] else X_scaled
    predictions = model.predict(X_model)
    inv_predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))
    return np.maximum(inv_predictions, 0)

path = r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\无异常划分\聚类划分8类'
res_list = []
for i in range(result.shape[1]):  # 遍历所有聚类中心
    if i in [0, 1, 2, 4, 6]:
        model_type = 'rf'
    elif i in [3, 5, 7]:
        model_type = 'lstm'
    else:
        raise ValueError(f"未指定 cluster_{i} 的模型类型")

    res = predict_with_model(f'cluster_{i}', model_type, features_array, path)
    res_list.append(res)

def weighted_cluster_prediction(res_list, result):
    res_list = [r.reshape(-1, 1) for r in res_list]
    predictions = np.hstack(res_list)
    return (predictions * result).sum(axis=1)
all_predictions = weighted_cluster_prediction(res_list, result)
all_y_true = test['predict_T2_0.5']

rmse = np.sqrt(mean_squared_error(all_y_true, all_predictions))
print("Test RMSE:", rmse)
mae = mean_absolute_error(all_y_true, all_predictions)
print("Test MAE:", mae)
mape = np.mean(np.abs((all_y_true - all_predictions) / (all_y_true + 1)))
print("Test MAPE:", mape)

def directional_symmetry(y_true, y_pred):
    # 计算预测方向一致的比例
    correct_direction = np.sum(np.sign(y_pred - np.roll(y_pred, 1)) == np.sign(y_true - np.roll(y_true, 1))
                               ) - 1  # 排除第一个元素
    total = len(y_true) - 1  # 排除第一个元素
    return correct_direction / total if total > 0 else 0

ds = directional_symmetry(all_y_true, all_predictions)
print("test DS:", ds)
pdb.set_trace()
df = pd.DataFrame({'true': all_y_true, 'predicted': all_predictions, 'cluster_id_used': all_cluster_ids, 'errs':abs(all_y_true - all_predictions)/(1+all_y_true)})
df.to_csv(r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\无异常划分\聚类划分8类\8预测结果.csv',index=False)
pdb.set_trace()
x = np.arange(len(df))