import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, auc
import seaborn as sns

def plot_results(
    y_true,
    anomaly_scores,
    y_pred,
    threshold,
    train_scores=None,
    output_pdf="anomaly_detection_results.pdf",
):
    """绘制异常检测可视化图（与 mra.py 完全一致）"""
    anomaly_scores = np.asarray(anomaly_scores).flatten()

    plt.figure(figsize=(6, 5))
    plt.plot(anomaly_scores, label='Anomaly Score', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.title('CNN Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/akira/codespace/mra-detection/anomaly_detection_results.png', dpi=150)
    print("\nPlot saved to: /home/akira/codespace/mra-detection/anomaly_detection_results.png")
    plt.show()

"""
读取原始数据并生成掩码矩阵
取前41个状态变量，返回原始数据，并替换NaN为1、正常数据为0生成掩码
"""
def generate_mask_matrix():
    try:
        data_path = Path(__file__).resolve().parent.parent / "TEP_3000_Block_Split.csv"
        df = pd.read_csv(data_path)
        # 提取 xmeas_1 到 xmeas_41 (丢弃后面的 xmv)
        data_df = df.filter(like='xmeas_').iloc[:, :41]
        data = data_df.astype(float).to_numpy()
        # 1 代表缺失 (NaN), 0 代表观测值
        mask = np.isnan(data).astype(int)
        return data, mask
    except FileNotFoundError:
        print("警告: 找不到数据文件。正在生成模拟数据用于演示代码运行...")
        # 生成模拟数据以确保代码可运行
        mock_data = np.random.randn(3000, 41)
        # 随机设置一些 NaN
        mock_data[np.random.rand(*mock_data.shape) < 0.1] = np.nan
        mock_mask = np.isnan(mock_data).astype(int)
        return mock_data, mock_mask

def prepare_data():
    # 1. 加载数据
    data, mask = generate_mask_matrix()
    
    if data is None:
        return None, None, None, None

    # 2. 处理缺失值 (NaN)
    # 神经网络不能输入 NaN。这里我们将 NaN 填充为 0。
    # 由于后续会做标准化 (StandardScaler)，0 通常接近均值，是一个安全的填充值。
    # 如果希望模型感知"缺失"这个特征，可以将 mask 也作为输入拼接到 data 后面，
    # 但为了保持 CNN 简单，这里仅做填充。
    data = np.nan_to_num(data, nan=0.0)

    # 3. 生成标签 (0: 正常, 1: 异常)
    # 前1500正常，后1500异常
    y_normal = np.zeros(1500)
    y_faulty = np.ones(1500)
    y = np.concatenate([y_normal, y_faulty])

    # 4. 数据分割 (训练集/测试集)
    # 前 50% 用于训练，后 50% 用于测试（保持时间顺序，不打乱）
    split_idx = len(data) // 2
    X_train, X_test = data[:split_idx], data[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 5. 标准化 (Standardization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # 使用训练集的参数转换测试集

    # 6. 重塑数据以适应 1D-CNN
    # PyTorch Conv1d 输入形状: (Batch_Size, Channels, Length)
    # 我们将 41 个特征视为长度为 41 的序列，通道数为 1
    X_train = X_train.reshape(-1, 1, 41)
    X_test = X_test.reshape(-1, 1, 41)

    # 7. 转为 PyTorch Tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1) # shape变为 (N, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# ---------------------------------------------------------
# 2. 定义 1D-CNN 模型
# ---------------------------------------------------------

class AnomalyDetectorCNN(nn.Module):
    def __init__(self):
        super(AnomalyDetectorCNN, self).__init__()

        # 1D-CNN Autoencoder: train on normal, detect anomalies by reconstruction error
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 41 -> 20
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 20 -> 10
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=2, stride=2),  # 10 -> 20
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=16,
                out_channels=1,
                kernel_size=2,
                stride=2,
                output_padding=1,  # 20 -> 41
            ),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

# ---------------------------------------------------------
# 3. 训练与评估流程
# ---------------------------------------------------------

def train_model():
    # 准备数据
    X_train, y_train, X_test, y_test = prepare_data()
    
    # 创建 DataLoader
    train_dataset = TensorDataset(X_train, X_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化模型、损失函数、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyDetectorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 30
    print(f"开始训练，共 {epochs} 个 Epoch...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            recon = model(inputs)
            loss = criterion(recon, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 评估模型
    print("\n--- 异常检测评估结果 ---")
    model.eval()
    with torch.no_grad():
        X_train_dev = X_train.to(device)
        X_test_dev = X_test.to(device)

        recon_train = model(X_train_dev)
        train_scores = (recon_train - X_train_dev).pow(2).mean(dim=[1, 2]).detach().cpu().numpy()

        train_mean = float(np.mean(train_scores))
        train_std = float(np.std(train_scores))
        thr_mean3std = train_mean + 3.0 * train_std
        threshold = float(np.quantile(train_scores, 0.99))

        recon_test = model(X_test_dev)
        test_scores = (recon_test - X_test_dev).pow(2).mean(dim=[1, 2]).detach().cpu().numpy()

        y_true = y_test.detach().cpu().numpy().flatten().astype(int)
        y_pred = (test_scores > threshold).astype(int)

        print(f"Device: {device}")
        print(f"Train recon error: mean={train_mean:.6f}, std={train_std:.6f}")
        print(f"Threshold: p99={threshold:.6f} (mean+3std={thr_mean3std:.6f})")
        print(f"Anomalies detected in test: {(y_pred == 1).sum()} / {len(y_pred)}")

        print("\n分类报告:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=["Normal", "Anomaly"],
                zero_division=0,
            )
        )

        plot_results(
            y_true,
            test_scores,
            y_pred,
            threshold,
            train_scores=train_scores,
            output_pdf="anomaly_detection_results.pdf",
        )
        
if __name__ == "__main__":
    train_model()
