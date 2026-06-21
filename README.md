# HAF-MSSD: A Multi-Source Structured Data-Based Framework for Airport Security Queue Length Prediction

This project implements a hierarchical adaptive modeling framework for airport security queue length prediction. The framework takes multi-source structured airport operational data as input, including security queue data, security check records, flight schedule data, and terminal traffic flow data. Through feature construction, sample difficulty classification, intra-class clustering, and cluster-specific submodel prediction, the framework enables short-term prediction of future security queue length.

This repository is mainly intended for reproducing the experimental workflow and demonstrating the modeling pipeline used in the research study. Since the original airport operational data involve real business information, this repository does not provide raw data, training data, model weights, or prediction result files. Users should configure the data paths and data formats according to their own local environment before running the code.

---

## 1. Project Version

Current code version:

```text
Version: v1.0.0
```

Recommended environment:

```text
Python: 3.10
Main dependencies: numpy, pandas, scikit-learn, xgboost, tensorflow, keras, optuna, tslearn, tsfeatures
```

It is recommended to use Conda or another virtual environment to manage dependencies and avoid package conflicts between different projects.

---

## 2. Main Workflow

The overall workflow of this project is as follows:

```text
Raw multi-source data
    ↓
Minute-level data construction and statistical feature extraction
    ↓
Initial regression model for prediction-error label generation
    ↓
Sample difficulty classifier training
    ↓
Intra-class KMeans clustering based on classification results
    ↓
Saving class-cluster centers
    ↓
Training independent prediction submodels for each class-cluster group
    ↓
Final prediction: classification → cluster-center matching → submodel prediction
```

The sample categories are usually defined as:

```text
0: challenging samples, indicating samples with higher prediction difficulty
1: friendly samples, indicating samples with lower prediction difficulty
```

---

## 3. Data Description

The raw data used in this project are stored by day or by several-day intervals. The major data sources include:

```text
queue              security queue data
trafft2            T2 terminal traffic flow data
trafft3            T3 terminal traffic flow data
check              security check records
plane / flight     flight schedule data
```

The raw data are recommended to be organized as follows:

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

In actual use, some data sources may be stored as one table per day, while others may be stored as one table every three days, six days, or half a month. Therefore, the code retains the date-interval-based file path selection logic. Users only need to modify the base path according to their own local data directory.

---

## 4. Path Configuration

To avoid exposing local file paths, it is not recommended to directly write real absolute paths in the code, such as:

```text
D:\real_project_directory\real_data_directory
```

It is recommended to use environment variables or a unified base path. For example:

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

On Windows, you can modify the paths according to your local data location:

```bash
set RAW_DATA_ROOT=D:\your_project\airport_raw_data
set OUTPUT_DIR=D:\your_project\airport_processed_data
```

On Linux or server environments, you can use:

```bash
export RAW_DATA_ROOT=/root/airport_project/raw_data
export OUTPUT_DIR=/root/airport_project/processed_data
```

---

## 5. Dependency Installation

It is recommended to create an independent environment first:

```bash
conda create -n haf_mssd python=3.10
conda activate haf_mssd
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not used, the main dependencies can also be installed manually:

```bash
pip install pandas numpy scikit-learn xgboost optuna joblib tensorflow tslearn tsfeatures matplotlib openpyxl
```

If GPU acceleration is used, CUDA and TensorFlow should be configured according to the local GPU and CUDA environment.

---

## 6. Running Workflow

### 6.1 Construct Minute-Level Multi-Source Feature Data

First, run the data construction script to convert the raw queue, trafft2, trafft3, check, and plane data into minute-level modeling data.

The input data include:

```text
queue
traffic
check
flight
```

The output data are usually saved as daily CSV files:

```text
20240723.csv
20240724.csv
...
```

Example command:

```bash
python datacreate.py
```

This step generates basic features, including historical queue statistics, historical security check statistics, terminal traffic flow statistics, future flight schedule features, time index features, and future prediction target columns.

---

### 6.2 Construct Time-Series Statistical Features

If additional time-series statistical features are required, the statistical feature construction script can be run.

This script calculates `tsfeatures` based on the security passenger count series over the past 24 hours.

Example command:

```bash
python datacreate_tsfeatures.py
```

The output results are saved as:

```text
tsfeatures/
    20241025.csv
    20241026.csv
    ...
```

---

### 6.3 Split Train, Validation, and Test Data

Before model training, the constructed data should be split chronologically into:

```text
train.csv
validation.csv
test.csv
```

Their usage is as follows:

```text
train.csv       used for model training
validation.csv  used for Optuna hyperparameter tuning and early stopping
test.csv        used only for final testing
```

Note that the validation and test files must be different. Using the test set as the validation set will cause data leakage.

---

### 6.4 Train the Initial Regression Model and Generate Error Labels

An initial regression model, such as RF, XGBoost, CNN, LSTM, GRU, or BiLSTM, can first be trained to generate prediction-error labels.

The error-labeling rule is usually defined as:

```text
label = 0: challenging sample
label = 1: friendly sample
```

The label is determined by the error between the true value and the predicted value. For example, when the relative error exceeds a specified threshold, the sample is labeled as a challenging sample.

Example command:

```bash
python Code/Predict/modelname_predict.py
```

The output results usually include:

```text
classification_train.csv
classification_validation.csv
classification_test.csv
```

These files are used for subsequent sample difficulty classifier training.

---

### 6.5 Train the Sample Difficulty Classifier

Next, train a classifier to determine whether an input sample belongs to the challenging or friendly category.

Available classification models include:

```text
Random Forest
XGBoost
CNN
LSTM
GRU
BiLSTM
```

Example command:

```bash
python Code/Classification/modelname_classifier.py
```

The output results usually include:

```text
classifier model
scaler_X.pkl
classification metrics
```

During final prediction, this classifier is first used to determine the difficulty category of each new sample.

---

### 6.6 Search for the Optimal Number of Intra-Class Clusters

After sample classification, the optimal number of KMeans clusters can be searched separately for each sample category. The commonly used evaluation metric is the Davies-Bouldin Index, abbreviated as DBI.

Example command:

```bash
python Code/DBI_Search.py
```

The output results usually include:

```text
dbi_search_summary.json
dbi curve figures
best K for challenging samples
best K for friendly samples
```

A smaller DBI value usually indicates better clustering performance.

---

### 6.7 Intra-Class KMeans Clustering

Based on the optimal cluster numbers obtained in the previous step, KMeans clustering is performed separately for challenging and friendly samples.

Example command:

```bash
python Code/Kmeans_Classification.py
```

This script performs the following steps:

```text
1. Read the train, validation, and test data
2. Use the trained classifier to predict the category of each sample
3. Train KMeans for each category using only the train data
4. Assign clusters to validation and test samples using the trained KMeans models
5. Save the data by category and cluster
```

The output file format is as follows:

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

### 6.8 Save Cluster Centers

After intra-class KMeans clustering is completed, `cluster_center.py` can be run to save the cluster centers for different clusters under each class. This step fixes the learned class-cluster structure and allows the final prediction stage to directly use cluster centers for sample matching or distance-based comparison.

Example command:

```bash
python Code/cluster_center.py
```

This script reads the cluster-level data generated after intra-class clustering and calculates the corresponding center vector for each class and cluster. The output results usually include:

```text
cluster_centers.pkl
cluster_centers_meta.json
```

where:

```text
cluster_centers.pkl       stores the cluster center for each class-cluster pair
cluster_centers_meta.json stores basic metadata of the cluster centers, such as class label, cluster ID, sample size, and feature count
```

After saving the cluster centers, the final prediction stage can identify the closest cluster by calculating the distance between a new sample and the cluster centers under its predicted class. The corresponding cluster-specific prediction submodel is then called. This step avoids recalculating cluster centers during prediction and ensures that the training and prediction stages use consistent cluster assignment criteria.

Note that cluster centers are calculated from the training data and contain certain information about the training data distribution. Therefore, `cluster_centers.pkl` and `cluster_centers_meta.json` are not recommended to be uploaded to a public GitHub repository. If the code needs to be shared publicly, only the `cluster_center.py` script should be kept in the repository, while the actual generated cluster center files should be stored locally or on the server.

---

### 6.9 Train Cluster-Specific Prediction Submodels

For each class-cluster group, an independent prediction model is trained. Each cluster uses independent train, validation, and test files.

The data usage is as follows:

```text
*_train.csv       used only for training
*_validation.csv  used only for hyperparameter tuning and early stopping
*_test.csv        used only for final testing
```

Available cluster-specific prediction models include:

```text
Random Forest
XGBoost
CNN
LSTM
GRU
BiLSTM
```

Example command:

```bash
python Code/Cluster_training/modelname_cluster.py
```

Each model outputs:

```text
best model
scaler_X.pkl
scaler_y.pkl
best_params.json
metrics.csv
```

---

### 6.10 Final Prediction

The final prediction workflow is as follows:

```text
Input new samples
    ↓
Use the classifier to predict the sample category
    ↓
Read the cluster centers under the predicted class
    ↓
Calculate the distance between the sample and each cluster center, and identify the closest cluster
    ↓
Call the corresponding cluster-specific submodel according to class + cluster
    ↓
Output the final predicted value
```

Example command:

```bash
python Code/Final_Predict.py
```

If the input data contain the true target column, the script automatically calculates:

```text
RMSE
MAE
MAPE
DS
```

If the input data do not contain the true target column, only prediction results will be generated.

---

## 7. Output Files

Common output files include:

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

## 8. Notes

This project is intended for academic research and experimental reproducibility. Since data formats may vary across airports, time periods, and operational systems, users need to adapt the data fields, date ranges, and path structures according to their own datasets before running the code.

If this framework is applied to another airport or another time period, the features should be reconstructed, the classification model should be retrained, intra-class clustering should be performed again, and the corresponding cluster-specific prediction submodels should also be retrained.
