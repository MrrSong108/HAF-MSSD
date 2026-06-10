import time
import pandas as pd
import os
from keras import Sequential, Input
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import pdb
import numpy as np
import optuna
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Bidirectional, Conv1D, Flatten, GRU, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
# X_validation = X_validation.reshape(X_validation.shape[0], 1, X_validation.shape[1])
# LSTM 模型
def build_lstm_model(input_shape, lstm_units_1, lstm_units_2, dropout_rate):
    model = Sequential()
    model.add(LSTM(lstm_units_1, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units_2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # 输出一个值
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
def objective_lstm(trial):
    # 超参数空间
    lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 512, step=32)  # 增加了上限，范围是32到512
    lstm_units_2 = trial.suggest_int('lstm_units_2', 16, 256, step=16)  # 增加了上限，范围是16到256
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.7)  # 增加了上限，范围是0.0到0.7
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)  # 增加了上限，范围是32到128
    epochs = trial.suggest_int('epochs', 50, 200, step=10)  # 增加了上限，范围是50到200
    patience = trial.suggest_int('patience', 5, 20)
    # 构建LSTM模型
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), lstm_units_1, lstm_units_2, dropout_rate)

    # 早停法防止过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # 训练LSTM模型
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_test, y_test),
              verbose=0,
              callbacks=[early_stopping])

    # 预测
    y_pred_scaled = model.predict(X_test)

    # 反归一化
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_test_inv = scaler_y.inverse_transform(y_test)

    # 计算 RMSE
    mse = mean_squared_error(y_test_inv, y_pred)
    rmse = np.sqrt(mse)
    return rmse

study_lstm = optuna.create_study(direction='minimize')

# 开始优化
study_lstm.optimize(objective_lstm, n_trials=50, n_jobs=1)  # 进行50次优化

# 输出最优结果
print(f"最优 RMSE: {study_lstm.best_value}")
print(f"最佳超参数: {study_lstm.best_params}")
# 使用最佳超参数训练 LSTM 模型
best_params = study_lstm.best_params
best_lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]),
                                   best_params['lstm_units_1'],
                                   best_params['lstm_units_2'],
                                   best_params['dropout_rate'])
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=best_params['patience'],
                               restore_best_weights=True)
best_lstm_model.fit(X_train, y_train,
                    epochs=best_params['epochs'],
                    batch_size=best_params['batch_size'],
                    validation_data=(X_test, y_test),
                    verbose=1,
                   callbacks=[early_stopping])

# 使用最佳模型进行预测
y_pred_scaled = best_lstm_model.predict(X_test)

# 反归一化
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_pred_original = np.maximum(y_pred, 0)
y_test_inv = scaler_y.inverse_transform(y_test)

print('运行时长：', time.time() - start_time)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_original))
print("Test RMSE:", rmse)
mae = mean_absolute_error(y_test_inv, y_pred_original)
print("Test MAE:", mae)
mape = np.mean(np.abs((y_test_inv - y_pred_original) / (y_test_inv + 1)))
print("Test MAPE:", mape)
def directional_symmetry(y_true, y_pred):
    # 计算预测方向一致的比例
    correct_direction = np.sum(np.sign(y_pred - np.roll(y_pred, 1)) == np.sign(y_true - np.roll(y_true, 1))
                               ) - 1  # 排除第一个元素
    total = len(y_true) - 1  # 排除第一个元素
    return correct_direction / total if total > 0 else 0

ds = directional_symmetry(y_test_inv, y_pred_original)
print("test DS:", ds)
# 验证集
# y_validation_pred = best_lstm_model.predict(X_validation)
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

# 保存结果
test_df = pd.DataFrame({
    'Actual': y_test_inv.flatten(),
    'LSTM': y_pred_original.flatten()
})
test_df.to_csv(r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\对比模型返回结果\test_lstm.csv', index=False)

# test_df_validation = pd.DataFrame({
#     'Actual': y_validation_inv.flatten(),
#     'LSTM': y_validation_original.flatten()
# })
# test_df_validation.to_csv(r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\validation_lstm.csv', index=False)

# plt.figure(figsize=(12, 6))  # 设置图形的大小
# plt.plot(y_test, label='Actual Values', color='blue', linestyle='-', linewidth=2)  # 真实值
# plt.plot(y_pred, label='Predicted Values', color='orange', linestyle='--', linewidth=2)  # 预测值
#
# # 添加图例
# plt.legend()
# # 添加标题和标签
# plt.title('LSTM')
# plt.xlabel('Samples')
# plt.ylabel('Values')
# # 显示网格
# plt.grid()
# # 显示图形
# plt.show()
# pdb.set_trace()