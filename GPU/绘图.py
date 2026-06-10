import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取数据，并截取你想画的区间
df = pd.read_csv(r'D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\无特征选择GRU\预测结果.csv')
# df = df[1440 * 4: 1440 * 5].copy()

# 重置索引，避免横轴不连续
df.reset_index(drop=True, inplace=True)


# ====== 自定义 cluster 顺序 ======
# 如果你想和示例图一样，优先 1cluster，再 0cluster，可以用这个排序函数
def sort_key(x):
    m = re.match(r'(\d+)cluster(\d+)', str(x))
    if m:
        prefix = int(m.group(1))  # 0 或 1
        suffix = int(m.group(2))  # cluster后面的编号
        return (-prefix, suffix)  # 让1开头的排前面，再按编号升序
    return (999, 999)

color_map = {
    '1cluster0': '#8e24aa',
    '1cluster1': '#1a33d6',
    '1cluster3': '#1e88e5',
    '1cluster4': '#26a69a',
    '1cluster5': '#2ca02c',
    '0cluster0': '#00cc44',
    '0cluster1': '#64dd17',
    '0cluster2': '#ffd600',
    '0cluster5': '#ff7f0e'
}
valid_cluster_ids = sorted(df['cluster_id_used'].dropna().unique(), key=sort_key)

# ====== 颜色映射 ======
cmap = plt.get_cmap('nipy_spectral')
colors = [cmap(0.1 + 0.8 * i / len(valid_cluster_ids)) for i in range(len(valid_cluster_ids))]

# ====== 开始绘图 ======
fig, ax = plt.subplots(figsize=(16, 7))

# 1. True 曲线
true_line, = ax.plot(
    df.index,
    df['true'],
    label='True',
    color='gray',
    linewidth=2,
    alpha=0.7
)

handles = [true_line]
labels = ['True']

# 2. 按 cluster_id 分组绘制 predicted 散点
for cluster_id in valid_cluster_ids:
    cluster_data = df[df['cluster_id_used'] == cluster_id]

    h = ax.scatter(
        cluster_data.index,
        cluster_data['predicted'],
        color=color_map[cluster_id],
        s=7,  # 点大小，可改成 3/5/10
        alpha=0.9,
        label=f'Cluster Center {cluster_id}'
    )

    handles.append(h)
    labels.append(f'Cluster Center {cluster_id}')

# ====== 图形设置 ======
# ax.set_title("True vs Predicted by Cluster", fontsize=14)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Value", fontsize=12)

ax.legend(handles, labels, loc='upper left', markerscale=1.5, fontsize=10)
ax.grid(False)

plt.tight_layout()
plt.savefig('D:\真能毕业吗\机场客流量\机场客流量预测投稿\JATM航空运输管理\论文\大修\图片\图9.png', dpi=600, bbox_inches='tight')
plt.show()