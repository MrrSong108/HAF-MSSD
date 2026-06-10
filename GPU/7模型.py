import matplotlib
matplotlib.use('TkAgg')   # 必须放在 pyplot 前面

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 1. 路径设置
# =========================
# 对比模型结果所在文件夹
compare_dir = Path(r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\对比模型返回结果')

# Proposed 模型预测结果
proposed_path = Path(r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\无特征选择GRU\预测结果.csv')


# =========================
# 2. 模型文件设置
# =========================
models = ['xgb', 'rf', 'bilstm', 'cnn', 'gru', 'lstm']

model_labels = {
    'xgb': 'XGBoost',
    'rf': 'RF',
    'bilstm': 'BiLSTM',
    'cnn': 'CNN',
    'gru': 'GRU',
    'lstm': 'LSTM'
}

colors = {
    'Actual': 'black',
    'xgb': '#E69F00',
    'rf': '#D55E00',
    'bilstm': '#56B4E9',
    'cnn': '#009E73',
    'gru': '#CC79A7',
    'lstm': '#0072B2',
    'proposed': '#F58383'
}


# =========================
# 3. 读取对比模型结果
# =========================
plot_data = None

for model in models:
    file_path = compare_dir / f'test_{model}.csv'

    if not file_path.exists():
        print(f'文件不存在，已跳过：{file_path}')
        continue

    df_model = pd.read_csv(file_path)

    # 假设文件格式为：Actual, xgb / rf / cnn ...
    actual_col = 'Actual'
    pred_col = model

    if pred_col not in df_model.columns:
        # 如果第二列就是预测结果，则自动重命名
        pred_col = df_model.columns[1]
        df_model = df_model.rename(columns={pred_col: model})

    if plot_data is None:
        plot_data = pd.DataFrame({
            'Actual': df_model[actual_col].values
        })

    plot_data[model] = df_model[model].values


# =========================
# 4. 读取 Proposed 结果
# =========================
df_proposed = pd.read_csv(proposed_path)

# 你的 Proposed 文件中一般是 true, predicted, cluster_id_used, errs
plot_data['proposed'] = df_proposed['predicted'].values


# =========================
# 5. 添加时间索引
# =========================
plot_data['Time'] = range(len(plot_data))
plot_data = plot_data.copy()[1440*4:1440*5]

# =========================
# 6. 绘图
# =========================
plt.figure(figsize=(16, 7))

# Actual
plt.plot(
    plot_data['Time'],
    plot_data['Actual'],
    color=colors['Actual'],
    label='Actual',
    linewidth=1.8,
    alpha=0.9
)

# Baseline models
for model in models:
    if model in plot_data.columns:
        plt.plot(
            plot_data['Time'],
            plot_data[model],
            color=colors[model],
            label=model_labels[model],
            linewidth=1.1,
            alpha=0.85
        )

# Proposed
plt.plot(
    plot_data['Time'],
    plot_data['proposed'],
    color=colors['proposed'],
    label='Proposed',
    linewidth=1.4,
    alpha=0.95
)


# =========================
# 7. 图形格式
# =========================
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value', fontsize=12)

# 论文图中一般不建议放内部标题
# plt.title('Model Predictions vs Actual')

plt.legend(
    loc='upper right',
    fontsize=9,
    frameon=True
)

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.35)
plt.xticks(rotation=0)

plt.tight_layout()

# =========================
# 8. 保存高质量图片
# =========================
plt.savefig('D:\真能毕业吗\机场客流量\机场客流量预测投稿\\JATM航空运输管理\论文\大修\图片\图10(xiao).png', dpi=600, bbox_inches='tight')
# plt.savefig('model_predictions_vs_actual.svg', bbox_inches='tight')

plt.show()