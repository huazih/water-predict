import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

from dataset.WaterLevelDataset import WaterLevelDataset
from load_data import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
combined_data, water_level_data = load_data()

# 数据清洗：确保没有 NaN、Inf、过大的值
combined_data = np.nan_to_num(combined_data, nan=0, posinf=1e10, neginf=-1e10)

# 缩放数据：先通过除法减少值的大小
num_samples = combined_data.shape[0]  # 122
num_features = combined_data.shape[1] * combined_data.shape[2] * combined_data.shape[3]  # 64 * 231 * 271
reshaped_data = combined_data.reshape(num_samples, num_features)

# 缩小数值的范围，避免过大的值
reshaped_data = reshaped_data / 1e5  # 调整缩放因子

# 使用 StandardScaler 进行标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(reshaped_data).astype(np.float16)  # 使用 float16 减少内存占用

# 重新组织数据形状
scaled_data = scaled_data.reshape(num_samples, 64, 231, 271)

# 创建数据集
seq_length = 7
dataset = WaterLevelDataset(scaled_data, water_level_data, seq_length)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 训练数据
train_data = []
train_labels = []
for inputs, labels in train_loader:
    inputs = inputs.view(-1, inputs.size(1) * inputs.size(2) * inputs.size(3) * inputs.size(4)).to(device).half().cpu().numpy()
    labels = labels.to(device).half().cpu().numpy()
    train_data.append(inputs)
    train_labels.append(labels)

train_data = np.vstack(train_data).astype(np.int16)
train_labels = np.hstack(train_labels).astype(np.float16)

# 使用随机森林和其他模型进行训练
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),  # 并行训练
    "SVR": SVR(kernel='rbf'),
    "Linear Regression": LinearRegression()
}

# 存储每个模型的预测结果和均方误差
predictions = {}
mse_scores = {}

# 训练每个模型并评估
for name, model in models.items():
    train_data = train_data.astype(np.int16)
    train_labels = train_labels.astype(np.float16)
    model.fit(train_data, train_labels)
    train_predictions = model.predict(train_data)
    mse = mean_squared_error(train_labels, train_predictions)
    predictions[name] = train_predictions
    mse_scores[name] = mse
    print(f"{name} - Training MSE: {mse}")
# 绘制散点图进行对比
plt.figure(figsize=(12, 8))

# 绘制每个模型的预测结果与真实标签的对比
for i, (name, preds) in enumerate(predictions.items()):
    plt.subplot(2, 2, i+1)
    plt.scatter(train_labels, preds, label=name, s=10)
    plt.plot([min(train_labels), max(train_labels)], [min(train_labels), max(train_labels)], color='red', linestyle='--')  # 45度线
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f"{name} Prediction vs True Values")
    plt.legend()
    plt.savefig(f'./plots/{name}_scatter_plot.png', dpi=300)  # 指定文件路径和分辨率
    plt.clf()  # 清空当前的绘图，准备绘制下一个
