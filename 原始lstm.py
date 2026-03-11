import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict
from pylab import mpl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 配置类
class Config():
    data_path = "/Users/yaoyuechen/Desktop/数据集/one year lstm.csv"
    timestep = 8  # 时间步长
    batch_size = 128  # 批次大小
    feature_size = 22  # 特征数量（除去目标变量）
    hidden_size = 128  # 隐层大小
    output_size = 1  # 输出大小
    num_layers = 2  # LSTM 层数
    epochs = 228  # 迭代轮数
    learning_rate = 0.0001  # 学习率
    model_name = 'lstm'  # 模型名称


config = Config()

# 1. 加载数据
df = pd.read_csv(config.data_path, index_col=0)
mag_max_obs = df['Mag_max_obs'].values.reshape(-1, 1)
df = df.drop(['Mag_max_obs'], axis=1)
COLS = list(df.columns)

# 2. 数据标准化
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

data = scaler_features.fit_transform(df)
mag_max_obs_scaled = scaler_target.fit_transform(mag_max_obs)

# 将目标变量添加回数据
data = np.hstack((data, mag_max_obs_scaled))

# 3. 划分训练和测试数据
def split_data(data, timestep, feature_size):
    dataX, dataY = [], []
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep][:, :-1])
        dataY.append(data[index + timestep][-1])

    dataX = np.array(dataX)
    dataY = np.array(dataY)
    train_size = int(0.8 * dataX.shape[0])

    x_train = dataX[:train_size].reshape(-1, timestep, feature_size)
    y_train = dataY[:train_size].reshape(-1, 1)
    x_test = dataX[train_size:].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, 1)

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = split_data(data, config.timestep, config.feature_size)

# 转换为tensor
torch.manual_seed(42)
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
x_test_tensor = torch.from_numpy(x_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

# 4. 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, _ = self.lstm(x, (h_0, c_0))
        output = self.fc(output[:, -1, :])
        return output


model = LSTM(config.feature_size, config.hidden_size, config.num_layers, config.output_size)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# 5. 训练模型并记录损失
train_losses = []
test_losses = []

for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_function(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {train_loss:.4f}")

    # 评估测试集损失
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch)
            test_loss += loss.item()
    test_loss = test_loss / len(test_loader)
    test_losses.append(test_loss)
    print(f"Test Loss after Epoch [{epoch + 1}/{config.epochs}]: {test_loss:.4f}")

# 6. 评估模型
model.eval()
with torch.no_grad():
    train_pred = model(x_train_tensor).cpu().numpy()
    test_pred = model(x_test_tensor).cpu().numpy()
    train_real = scaler_target.inverse_transform(y_train_tensor.cpu().numpy())
    test_real = scaler_target.inverse_transform(y_test_tensor.cpu().numpy())
    train_pred = scaler_target.inverse_transform(train_pred)
    test_pred = scaler_target.inverse_transform(test_pred)

# 7. 计算误差
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mse, mae, mape, r2


train_rmse, train_mse, train_mae, train_mape, train_r2 = calculate_metrics(train_real, train_pred)
test_rmse, test_mse, test_mae, test_mape, test_r2 = calculate_metrics(test_real, test_pred)
print(f"Train RMSE: {train_rmse:.3f}, Train MSE: {train_mse:.3f}, Train MAE: {train_mae:.3f}, Train MAPE: {train_mape:.3f}%, Train R2: {train_r2:.3f}")
print(f"Test RMSE: {test_rmse:.3f}, Test MSE: {test_mse:.3f}, Test MAE: {test_mae:.3f}, Test MAPE: {test_mape:.3f}%, Test R2: {test_r2:.3f}")

# 8. 绘制训练集和测试集的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, config.epochs + 1), train_losses, label='Train Loss', color='b')
plt.plot(range(1, config.epochs + 1), test_losses, label='Test Loss', color='r')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Train and Test Loss per Epoch', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

# 9. 绘制预测结果 - 训练集和测试集
mpl.rcParams['font.sans-serif'] = ['SimHei']
fig = plt.figure()
plt.gca().set_aspect(1)  # 设置横纵坐标单位长度相同
plt.scatter(test_real, test_pred, color='k', marker='o', s=15, label='Test')
plt.scatter(train_real, train_pred, color='b', marker='x', s=15, label='Train')
plt.xlabel("REAL_MAG", size=20)
plt.ylabel("PRE_MAG", size=20)
plt.tick_params(labelsize=16)
x = np.asarray([i for i in range(20)])
plt.plot(x, x, color='r', label='y=x')
plt.plot(x, x + 0.5, color='lightslategrey', linestyle='--', label='y=x+0.5')
plt.plot(x, x - 0.5, color='lightslategrey', linestyle='--', label='y=x-0.5')
plt.xlim(4, 8.5)
plt.ylim(4, 8.5)
fig.set_figheight(10)
fig.set_figwidth(10)
plt.legend(loc='lower right', fontsize=16)
plt.show()

# 10. 计算特征重要性
baseline_mae = test_mae
feature_importances = []

for k in range(config.feature_size):
    print(f'Computing importance for feature: {COLS[k]}')
    x_train_modified = torch.cat((x_train_tensor[:, :, :k], x_train_tensor[:, :, k + 1:]), dim=2)
    x_test_modified = torch.cat((x_test_tensor[:, :, :k], x_test_tensor[:, :, k + 1:]), dim=2)

    # 重新定义 LSTM 模型，调整输入层大小
    modified_model = LSTM(config.feature_size - 1, config.hidden_size, config.num_layers, config.output_size)

    # 加载与新模型兼容的权重
    modified_state_dict = model.state_dict()
    new_state_dict = modified_model.state_dict()

    # 替换所有兼容的参数（忽略输入层不匹配的权重）
    for key in new_state_dict.keys():
        if key in modified_state_dict and modified_state_dict[key].shape == new_state_dict[key].shape:
            new_state_dict[key] = modified_state_dict[key]

    modified_model.load_state_dict(new_state_dict, strict=False)

    # 使用修改后的特征进行模型预测
    modified_model.eval()
    with torch.no_grad():
        modified_test_pred = modified_model(x_test_modified).cpu().numpy()
        modified_test_pred = scaler_target.inverse_transform(modified_test_pred)
        modified_mae = np.mean(np.abs(modified_test_pred - test_real))

    # 计算特征重要性（误差增量越大，特征越重要）
    importance = modified_mae - baseline_mae
    feature_importances.append({'feature': COLS[k], 'importance': importance})
    print(f'Feature: {COLS[k]}, MAE after removal: {modified_mae:.3f}, Importance: {importance:.3f}')

# 11. 绘制特征重要性
feature_importances_df = pd.DataFrame(feature_importances)
feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
plt.figure(figsize=(12, 8))
plt.barh(feature_importances_df['feature'], feature_importances_df['importance'], color='skyblue')
plt.xlabel('Importance (MAE Increase)', size=20)
plt.ylabel('Features', size=20)
plt.title('Feature Importance Based on MAE Increase After Removal', size=24)
plt.gca().invert_yaxis()
plt.xticks(fontsize=16)
plt.yticks(fontsize=14)
plt.show()