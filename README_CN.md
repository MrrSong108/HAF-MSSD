# HAF-MSSD：基于多源结构化数据的机场安检排队人数预测框架

本项目实现了一个面向机场安检排队人数预测的层级自适应建模框架。框架以多源结构化机场运行数据为输入，包括安检排队数据、安检过检数据、航班计划数据、航站楼交通流数据等，通过特征构建、样本难度分类、类别内聚类和簇级子模型预测，实现对未来安检排队人数的短时预测。

本仓库主要用于论文实验复现与模型流程展示。由于原始机场运行数据涉及实际业务信息，仓库中不直接提供原始数据、训练数据、模型权重和预测结果文件。用户需要根据自己的数据路径和数据格式进行配置后运行。

---

## 1. 项目版本

当前代码版本：

```text
Version: v1.0.0
```

建议运行环境：
```text
Python: 3.10
Main dependencies: numpy, pandas, scikit-learn, xgboost, tensorflow, keras, optuna, tslearn, tsfeatures
```
推荐使用 Conda 或虚拟环境管理依赖，避免不同项目之间的包版本冲突。

---

## 2. 项目主要流程

本项目整体流程如下：

```text
原始多源数据
    ↓
分钟级数据构建与统计特征提取
    ↓
初始回归模型生成预测误差标签
    ↓
样本难度分类模型训练
    ↓
根据分类结果进行类别内 KMeans 聚类
    ↓
保存类别-簇聚类中心
    ↓
为每个类别-簇训练独立预测子模型
    ↓
最终预测：分类 → 聚类中心匹配 → 子模型预测
```

其中，样本类别通常分为：

```text
0: challenging samples，表示预测难度较高的样本
1: friendly samples，表示预测难度较低的样本
```

---

## 3. 数据说明

本项目使用的原始数据按天或按若干天分表存储。每类数据通常包括：

```text
queue              安检排队数据
trafft2            T2 航站楼交通流数据
trafft3            T3 航站楼交通流数据
check              安检过检数据
plane / flight     航班计划数据
```

原始数据建议按如下形式组织：

```text
Data/
    queue/
        20240723/
            queues.xlsx
        20240724/
            queues.xlsx

    traffic/
        T2/
            20240723_20240726.xlsx
        T3/
            20240723_20240726.xlsx

    check/
        20240720_20240725.xlsx

    plane/
        20240716_20240730.xlsx
```

实际项目中，部分数据可能是每天一个表，部分数据可能是每三天、每六天或每半个月一个表。因此，代码中保留了按日期区间判断数据文件路径的写法。用户只需要将基础路径修改为自己的本地数据目录即可。

---

## 4. 路径配置方式

为避免泄露本地路径，代码中不建议直接写入真实绝对路径，例如：

```text
D:\真实项目目录\真实数据目录
```

推荐使用环境变量或统一基础路径。例如：

```python
from pathlib import Path
import os

RAW_DATA_ROOT = Path(os.getenv(
    "RAW_DATA_ROOT",
    r"D:\your_project\airport_raw_data"
))

OUTPUT_DIR = Path(os.getenv(
    "OUTPUT_DIR",
    r"D:\your_project\airport_processed_data"
))
```

运行时可以根据自己的数据位置修改：

```bash
set RAW_DATA_ROOT=D:\your_project\airport_raw_data
set OUTPUT_DIR=D:\your_project\airport_processed_data
```

Linux 或服务器环境下可以使用：

```bash
export RAW_DATA_ROOT=/root/airport_project/raw_data
export OUTPUT_DIR=/root/airport_project/processed_data
```

---

## 5. 依赖安装

建议先创建独立环境：

```bash
conda create -n haf_mssd python=3.10
conda activate haf_mssd
```

然后安装依赖：

```bash
pip install -r requirements.txt
```

如果不使用 requirements.txt，也可以手动安装主要依赖：

```bash
pip install pandas numpy scikit-learn xgboost optuna joblib tensorflow tslearn tsfeatures matplotlib openpyxl
```

如果使用 GPU，需要根据本机 CUDA 和 TensorFlow 版本单独配置 GPU 环境。

---

## 6. 运行流程

### 6.1 构建分钟级多源特征数据

首先运行数据构建脚本，将原始 queue、trafft2、trafft3、check、plane 数据转换为分钟级建模数据。

输入数据包括：

```text
queue
traffic
check
flight
```

输出数据通常为按天保存的 CSV 文件：

```text
20240723.csv
20240724.csv
...
```

运行示例：

```bash
python datacreate.py
```

该步骤会生成基础特征，包括历史排队统计特征、历史过检统计特征、交通流统计特征、未来航班计划特征、时间索引特征以及未来预测目标列。

---

### 6.2 构建时间序列统计特征

如果需要额外使用时间序列统计特征，可以运行统计特征构建脚本。

该脚本会基于过去 24 小时的安检人数序列，计算 tsfeatures 特征。

运行示例：

```bash
python datacreate_tsfeatures.py
```

输出结果为：

```text
tsfeatures/
    20241025.csv
    20241026.csv
    ...
```

---

### 6.3 划分 train、validation 和 test 数据

在模型训练前，需要将构建好的数据按时间顺序划分为：

```text
train.csv
validation.csv
test.csv
```

其中：

```text
train.csv       用于模型训练
validation.csv  用于 Optuna 调参和 early stopping
test.csv        只用于最终测试结果
```

需要注意，validation 和 test 必须是不同的数据文件，不能将 test 同时作为 validation 使用，否则会造成数据泄露。

---

### 6.4 训练初始回归模型并生成误差标签

可以先训练一个基础回归模型，例如 RF、XGBoost、CNN、LSTM、GRU 或 BiLSTM，用于生成预测误差标签。

误差标签生成规则通常为：

```text
label = 0: challenging sample
label = 1: friendly sample
```

标签由真实值和预测值之间的误差决定。例如，当相对误差超过设定阈值时，样本被标记为 challenging sample。

运行示例：

```bash
python Code/Predict/modelname_predict.py
```

输出结果通常包括：

```text
classification_train.csv
classification_validation.csv
classification_test.csv
```

这些文件用于后续训练样本难度分类器。

---

### 6.5 训练样本难度分类器

接着训练分类模型，用于判断输入样本属于 challenging 还是 friendly。

可选分类模型包括：

```text
Random Forest
XGBoost
CNN
LSTM
GRU
BiLSTM
```

运行示例：

```bash
python Code/Classification/modelname_classifier.py
```

输出结果通常包括：

```text
classifier model
scaler_X.pkl
classification metrics
```

最终预测阶段会先调用该分类器，对新样本进行难度类别判断。

---

### 6.6 搜索类别内最优聚类数

在完成样本分类后，可以对每一类样本分别搜索最佳 KMeans 聚类数。常用评价指标为 Davies-Bouldin Index，即 DBI。

运行示例：

```bash
python Code/DBI_Search.py
```

输出结果通常包括：

```text
dbi_search_summary.json
dbi curve figures
best K for challenging samples
best K for friendly samples
```

DBI 越小，通常表示聚类效果越好。

---

### 6.7 类别内 KMeans 聚类

根据上一步得到的最佳聚类数，对 challenging 和 friendly 两类样本分别进行 KMeans 聚类。

运行示例：

```bash
python Code/Kmeans_Classification.py
```

该脚本会：

```text
1. 读取 train、validation 和 test 数据
2. 使用训练好的分类器预测每条样本的类别
3. 在 train 数据上为每个类别训练 KMeans
4. 使用训练好的 KMeans 为 validation 和 test 分配簇
5. 按类别和簇保存数据
```

输出文件格式如下：

```text
class_0_challenging/
    0cluster_0_train.csv
    0cluster_0_validation.csv
    0cluster_0_test.csv

class_1_friendly/
    1cluster_0_train.csv
    1cluster_0_validation.csv
    1cluster_0_test.csv
```

---

### 6.8 保存聚类中心

在完成类别内 KMeans 聚类后，可以运行 cluster_center.py 保存各类别下不同簇的聚类中心。该步骤的作用是将已经形成的类别-簇结构固化下来，便于最终预测阶段直接调用聚类中心进行样本匹配或中心距离比对。

运行示例：

```bash
python Code/cluster_center.py
```

该脚本会读取类别内聚类后得到的簇级数据，并根据每个类别和簇计算对应的中心向量。输出结果通常包括：

```text
cluster_centers.pkl
cluster_centers_meta.json
```

其中：

cluster_centers.pkl       保存不同 class-cluster 对应的聚类中心
cluster_centers_meta.json 保存聚类中心的基本元信息，例如类别、簇编号、样本数量和特征数量

保存聚类中心后，最终预测阶段可以根据新样本所属类别，直接与该类别下的聚类中心进行距离计算，从而确定最接近的簇，并调用对应的簇级预测子模型。这一步可以避免在预测阶段重新计算聚类中心，保证训练阶段和预测阶段使用一致的簇划分依据。

需要注意，聚类中心是基于训练数据统计得到的，包含一定的数据分布信息。因此，cluster_centers.pkl 和 cluster_centers_meta.json 不建议上传到公开 GitHub 仓库。如果需要公开代码，可以只保留 cluster_center.py 脚本，而将实际生成的聚类中心文件保存在本地或服务器环境中。

---

### 6.9 训练簇级预测子模型

对每个类别-簇分别训练预测模型。每个簇都使用独立的 train、validation 和 test 文件。

数据使用方式为：

```text
*_train.csv       只用于训练
*_validation.csv  只用于调参和 early stopping
*_test.csv        只用于最终测试
```

可选簇级预测模型包括：

```text
Random Forest
XGBoost
CNN
LSTM
GRU
BiLSTM
```

运行示例：

```bash
python Code/Cluster_training/modelname_cluster.py
```

每个模型会输出：

```text
best model
scaler_X.pkl
scaler_y.pkl
best_params.json
metrics.csv
```

---


### 6.10 最终预测

最终预测流程为：

```text
输入新样本
    ↓
使用分类器预测样本类别
    ↓
根据预测类别读取对应类别下的聚类中心
    ↓
计算样本与各聚类中心的距离，并确定最接近的簇
    ↓
根据 class + cluster 调用对应簇级子模型
    ↓
输出最终预测值
```

运行示例：

```bash
python Code/Final_Predict.py
```

如果输入数据中包含真实目标列，脚本会自动计算：

```text
RMSE
MAE
MAPE
DS
```

如果输入数据中不包含真实目标列，则只输出预测结果。

---

## 7. 输出文件说明

常见输出文件包括：

```text
best_model.keras / best_model.pkl
scaler_X.pkl
scaler_y.pkl
best_params.json
metrics.csv
cluster_centers.pkl
cluster_centers_meta.json
kmeans_label0.pkl
kmeans_label1.pkl
cluster_scaler_label0.pkl
cluster_scaler_label1.pkl
cluster_rf.csv
cluster_xgboost.csv
cluster_cnn.csv
cluster_lstm.csv
cluster_gru.csv
cluster_bilstm.csv
final_prediction_result.csv
final_prediction_metrics.json
```

---

## 8. 说明

本项目代码用于学术研究和实验复现。由于不同机场、不同时间段和不同数据系统的数据格式可能存在差异，运行前需要根据自己的数据字段、日期范围和路径结构进行适配。

如果需要在其他机场或其他时间段使用本框架，需要重新构建特征、重新训练分类模型、重新进行类别内聚类，并重新训练对应的簇级预测模型。
