import pandas as pd
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林模型
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna

# 设置预测列
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

# 读取数据
train = pd.read_csv(r"D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\拼接后数据\train.csv")
test = pd.read_csv(r"D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\拼接后数据\test.csv")
# validation = pd.read_csv(r"D:\真能毕业吗\机场客流量\机场客流量预测投稿\正确目标\validation.csv")

# 数据处理
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
     # 随机森林的超参数空间
     param_grid = {
         'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),  # 树的数量，扩大范围到100-1000
         'max_depth': trial.suggest_int('max_depth', 3, 30, step=1),  # 树的最大深度，扩大范围到3-30
         'min_samples_split': trial.suggest_int('min_samples_split', 2, 20, step=1),  # 节点分裂的最小样本数，扩大范围到2-20
         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, step=1),  # 叶子节点的最小样本数，扩大范围到1-20
         'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5]),
         # 增加0.1到0.5的特征比例选择
         'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),  # 是否使用有放回采样
         'n_jobs': -1  # 使用所有可用的CPU核心
     }

     # 初始化随机森林模型
     model = RandomForestRegressor(
         random_state=42,
         **param_grid
     )

     # 训练模型
     model.fit(X_train, y_train.ravel())

     # 预测
     y_pred_scaled = model.predict(X_test)

     # 反归一化
     y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
     y_test_inv = scaler_y.inverse_transform(y_test)

     # 计算均方根误差 (RMSE)
     mse = mean_squared_error(y_test_inv, y_pred)
     rmse = np.sqrt(mse)
     return rmse

 # 创建Optuna的study对象
# study = optuna.create_study(direction='minimize')
#
#  # 优化超参数
# study.optimize(objective, n_trials=30, n_jobs=1)  # 进行50次优化试验, 使用1个核心
#
# # 输出最佳结果
# print(f"最优RMSE: {study.best_value}")
# print(f"最佳超参数: {study.best_params}")
# print('开始训练')
# 使用最佳参数训练随机森林模型
# best_params = {'n_estimators': 400, 'max_depth': 10, 'min_samples_split': 16, 'min_samples_leaf': 4, 'max_features': 0.5, 'bootstrap': False}
print("开始训练")
best_model = RandomForestRegressor(n_estimators=400, max_depth=3, bootstrap=False, n_jobs=-1)
best_model.fit(X_train, y_train.ravel())
print('训练完成')
# 使用最佳模型进行预测
y_pred_scaled = best_model.predict(X_test)

# 反归一化
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_pred_original = np.maximum(y_pred, 0)
y_test_inv = scaler_y.inverse_transform(y_test)

print('运行时长：', time.time() - start_time)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_original))
print("test RMSE:", rmse)
mae = mean_absolute_error(y_test_inv, y_pred_original)
print("test MAE:", mae)
mape = np.mean(np.abs((y_test_inv - y_pred_original) / (y_test_inv + 1)))
print("test MAPE:", mape)

# 方向一致性
def directional_symmetry(y_true, y_pred):
    correct_direction = np.sum(np.sign(y_pred - np.roll(y_pred, 1)) == np.sign(y_true - np.roll(y_true, 1))) - 1
    total = len(y_true) - 1
    return correct_direction / total if total > 0 else 0

ds = directional_symmetry(y_test_inv, y_pred_original)
print("test DS:", ds)
pdb.set_trace()
# 验证集
# y_validation_pred = best_model.predict(X_validation)
#
# # 反归一化
# y_validation_original = np.maximum(scaler_y.inverse_transform(y_validation_pred.reshape(-1, 1)), 0)
# y_validation_inv = scaler_y.inverse_transform(y_validation)
#
# rmse_validation = np.sqrt(mean_squared_error(y_validation_inv, y_validation_original))
# print("validation RMSE:", rmse_validation)
# mae_validation = mean_absolute_error(y_validation_inv, y_validation_original)
# print("validation MAE:", mae_validation)
# mape_validation = np.mean(np.abs((y_validation_inv - y_validation_original) / (y_validation_inv + 1))) * 100
# print("validation MAPE:", mape_validation)
#
# ds_validation = directional_symmetry(y_validation_inv, y_validation_original)
# print("validation DS:", ds_validation)
#
# 保存结果
test_df = pd.DataFrame({
    'Actual': y_test_inv.flatten(),
    'rf': y_pred_original.flatten()
})
test_df.to_csv(r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\对比模型返回结果\test_rf.csv', index=False)
#
# test_df_validation = pd.DataFrame({
#     'Actual': y_validation_inv.flatten(),
#     'rf': y_validation_original.flatten()
# })
# test_df_validation.to_csv(r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\validation_rf.csv', index=False)
#
# # 绘制图形
# plt.figure(figsize=(12, 6))  # 设置图形的大小
# plt.plot(y_test, label='Actual Values', color='blue', linestyle='-', linewidth=2)  # 真实值
# plt.plot(y_pred, label='Predicted Values', color='orange', linestyle='--', linewidth=2)  # 预测值
#
# plt.legend()
# plt.title('Actual vs Predicted Values')
# plt.xlabel('Samples')
# plt.ylabel('Values')
# plt.grid()
# plt.show()
# pdb.set_trace()
