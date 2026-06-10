import pandas as pd
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import pdb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Bidirectional, Conv1D, Flatten, GRU
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import optuna
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
train = pd.read_csv(r"D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\拼接后数据/train.csv")
test = pd.read_csv(r"D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\拼接后数据/test.csv")
# validation = pd.read_csv(r"D:\真能毕业吗\机场客流量\机场客流量预测投稿\正确目标\validation.csv")
features_train = train.drop(columns=predict_columns).drop(columns=['start_time']).fillna(0)
features_test = test.drop(columns=predict_columns).drop(columns=['start_time']).fillna(0)
# features_validation = validation.drop(columns=predict_columns).drop(columns=['start_time']).fillna(0)
train_Y = train['predict_T2_0.5']
test_Y = test['predict_T2_0.5']
# validation_Y = validation['predict_T2_0.5']

# 数据归一化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(features_train)
y_train = scaler_y.fit_transform(train_Y.values.reshape(-1, 1))
X_test = scaler_X.transform(features_test)
y_test = scaler_y.transform(test_Y.values.reshape(-1, 1))
# X_validation = scaler_X.transform(features_validation)
# y_validation = scaler_y.transform(validation_Y.values.reshape(-1, 1))
start_time = time.time()

def objective(trial):
    # XGBoost 的超参数空间
    param_grid = {
        'verbosity': 0,
        # 'objective': 'reg:squarederror',  # 回归问题，通常可以默认此设置
        'tree_method': 'auto',  # 你可以考虑设置为 'hist' 或 'gpu_hist' 来加速训练

        # 正则化相关
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),  # L2 正则化
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),  # L1 正则化
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),  # 树的剪枝所需的最小损失减少

        # 样本采样相关
        'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),  # 训练样本采样比例
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),  # 每棵树采样的特征比例
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0, step=0.05),  # 每一层节点的特征采样比例
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0, step=0.05),  # 每个节点的特征采样比例

        # 树的深度与结构
        'max_depth': trial.suggest_int('max_depth', 3, 12, step=1),  # 树的最大深度，考虑更细粒度的范围
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # 节点分裂的最小样本权重和
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),  # 最大增量步长，用于处理类不平衡问题

        # 学习率
        'eta': trial.suggest_float('eta', 0.001, 0.1, log=True),  # 学习率，设置更宽的范围

        # 类别权重调整
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20, step=1),  # 类别不平衡时的权重调整

        # 树的结构调整
        'max_bin': trial.suggest_int('max_bin', 128, 1024, step=128),  # 分箱数目，适用于 histogram 计算方法
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),  # 决策树生长策略

        # 训练过程中使用的其他超参数
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),  # 选择 booster 类型
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),  # 树的数量
    }

    # 初始化XGBoost模型
    model = xgb.XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        **param_grid,
        early_stopping_rounds=50
    )

    # 训练模型，并使用早期停止
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],  # 50轮无提升则停止训练
              verbose=False)

    # 使用模型预测
    y_pred_scaled = model.predict(X_test)

    # 将预测值反归一化
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_test_inv = scaler_y.inverse_transform(y_test)

    # 计算均方根误差(RMSE)
    mse = mean_squared_error(y_test_inv, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# 创建Optuna的study对象
# study = optuna.create_study(direction='minimize')
#
# # 优化超参数
# study.optimize(objective, n_trials=30, n_jobs=-1)  # 进行50次优化试验, 使用1个核心 (并行化可改为n_jobs=-1)
#
# # 输出最佳结果
# print(f"最优RMSE: {study.best_value}")
# print(f"最佳超参数: {study.best_params}")
# # 使用最佳参数训练XGBoost模型
# best_params = study.best_params
best_model = xgb.XGBRegressor(random_state=42,n_jobs=-1)
best_model.fit(X_train, y_train)

# 使用最佳模型进行预测
y_pred_scaled = best_model.predict(X_test)

# 将预测值反归一化
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_pred_original = np.maximum(y_pred, 0)
y_test_inv = scaler_y.inverse_transform(y_test)

print('运行时长：', time.time() - start_time)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_original))
print("test RMSE:", rmse)
mae = mean_absolute_error(y_test_inv, y_pred_original)
print("test MAE:", mae)
mape = np.mean(np.abs((y_test_inv - y_pred_original) / (y_test_inv+1)))
print("test MAPE:", mape)
def directional_symmetry(y_true, y_pred):
    # 计算预测方向一致的比例
    correct_direction = np.sum(np.sign(y_pred - np.roll(y_pred, 1)) == np.sign(y_true - np.roll(y_true, 1))
                               ) - 1  # 排除第一个元素
    total = len(y_true) - 1  # 排除第一个元素
    return correct_direction / total if total > 0 else 0
ds = directional_symmetry(y_test_inv, y_pred_original)
print("test DS:", ds)

##验证
# y_validation_scaled = best_model.predict(X_validation)
#
# # 将预测值反归一化
# y_validation_pred = scaler_y.inverse_transform(y_validation_scaled.reshape(-1, 1))
# y_validation_original = np.maximum(y_validation_pred, 0)
# y_validation_inv = scaler_y.inverse_transform(y_validation)
#
# rmse_validation = np.sqrt(mean_squared_error(y_validation_inv, y_validation_original))
# print("validation RMSE:", rmse_validation)
# mae_validation = mean_absolute_error(y_validation_inv, y_validation_original)
# print("validation MAE:", mae_validation)
# mape_validation = np.mean(np.abs((y_validation_inv - y_validation_original) / (y_validation_inv+1))) * 100
# print("validation MAPE:", mape_validation)
# def directional_symmetry(y_true, y_pred):
#     # 计算预测方向一致的比例
#     correct_direction = np.sum(np.sign(y_pred - np.roll(y_pred, 1)) == np.sign(y_true - np.roll(y_true, 1))
#                                ) - 1  # 排除第一个元素
#     total = len(y_true) - 1  # 排除第一个元素
#     return correct_direction / total if total > 0 else 0
#
# ds_validation = directional_symmetry(y_validation_inv, y_validation_original)
#
# print("validation DS:", ds_validation)

test_df = pd.DataFrame({
    'Actual': y_test_inv.flatten(),
    'xgb': y_pred_original.flatten()
})
test_df.to_csv(r'/root/autodl-tmp/queues/test_xgb.csv', index=False)
# pdb.set_trace()
# test_df_validation = pd.DataFrame({
#     'Actual': y_validation_inv.flatten(),
#     'xgb': y_validation_original.flatten()
# })
# test_df_validation.to_csv(r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\validation_xgb.csv', index=False)

# plt.figure(figsize=(12, 6))  # 设置图形的大小
# plt.plot(y_test, label='Actual Values', color='blue', linestyle='-', linewidth=2)  # 真实值
# plt.plot(y_pred, label='Predicted Values', color='orange', linestyle='--', linewidth=2)  # 预测值
#
# # 添加图例
# plt.legend()
# # 添加标题和标签
# plt.title('Actual vs Predicted Values')
# plt.xlabel('Samples')
# plt.ylabel('Values')
# # 显示网格
# plt.grid()
# # 显示图形
# plt.show()
# pdb.set_trace()