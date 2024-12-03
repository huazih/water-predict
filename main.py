import numpy as np
from matplotlib import pyplot as plt

from load_data import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataset.WaterLevelDataset import WaterLevelDataset
from torch.utils.data import DataLoader, Dataset, random_split
from model.CNN_LSTM import CNN_LSTM
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combined_data, water_level_data = load_data()
combined_data = torch.tensor(combined_data, dtype=torch.float32)  # 根据需要选择 dtype
water_level_data = torch.tensor(water_level_data, dtype=torch.float32)  # 根据需要选择 dtype
scaler = StandardScaler()
bs,channel,x,y=combined_data.shape
combined_data=combined_data.view(bs,-1)
combined_data = scaler.fit_transform(combined_data)
combined_data=combined_data.reshape(bs,channel,x,y)
seq_length = 7
dataset = WaterLevelDataset(combined_data, water_level_data, seq_length)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))  # 80% 训练集
val_size = len(dataset) - train_size   # 20% 验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 模型训练
# model = CNN_LSTM(input_size=64, hidden_size=128).to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for inputs, labels in train_loader:
    for t in range(inputs.shape[1]):  # inputs.shape[1] 是 7，即有7个时间步
        # 选择当前时间步的数据
        inputs_time_step = inputs[0, t, :, :, :]  # 获取当前时间步的 (61, 213, 271) 数据

        # 获取输入数据的维度
        dim_x, dim_y, dim_z = inputs_time_step.shape  # (61, 213, 271)

        # 创建一个三维网格，表示每个点的 x, y, z 坐标
        x, y, z = np.indices((dim_x, dim_y, dim_z))  # 创建三维网格 (61, 213, 271)

        # 展平三维网格数据
        x = x.flatten()  # 展平为一维数组
        y = y.flatten()  # 展平为一维数组
        z = z.flatten()  # 展平为一维数组
        values = inputs_time_step.flatten().cpu().numpy()  # 获取每个点的特征值

        # 创建一个 3D 散点图
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制3D散点图，使用 values 作为颜色
        sc = ax.scatter(x, y, z, c=values, cmap='viridis', s=1)

        # 添加颜色条
        plt.colorbar(sc, label='Feature Value')

        ax.set_title(f"3D Scatter Plot of Feature Values at Time Step {t + 1}")
        ax.set_xlabel("Feature Num")
        ax.set_ylabel("X Coordinate")
        ax.set_zlabel("Y Coordinate")
        plt.savefig(f"./plots/scatter_plot_time_step_{t + 1}.png")
        plt.clf()  # 或者 plt.close() 以释放内存
# for epoch in range(100):  # 示例训练轮次
#     model.train()  # 设置为训练模式
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移动到 GPU
#         optimizer.zero_grad()
#         labels = labels.float()  # 确保标签为 Float 类型
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()80
#         optimizer.step()
#
#     # 验证
#     model.eval()  # 设置为评估模式
#     val_loss = 0.0
#     with torch.no_grad():  # 禁用梯度计算
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移动到 GPU
#             outputs = model(inputs)
#             val_loss += criterion(outputs, labels).item()
#
#     val_loss /= len(val_loader)
#     print(f"Epoch {epoch + 1}, Train Loss: {loss.item()}, Val Loss: {val_loss}")
#
# # 保存模型
# torch.save(model.state_dict(), "water_level_predictor.pth")