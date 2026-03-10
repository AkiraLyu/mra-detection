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
    """绘制异常检测可视化图（每页一个 figure，保存为多页 PDF）"""
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)
    anomaly_scores = np.asarray(anomaly_scores).flatten()
    if train_scores is not None:
        train_scores = np.asarray(train_scores).flatten()

    figures = []

    # ── 1. 混淆矩阵 ──────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax1,
    )
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    ax1.set_title("Confusion Matrix")
    fig1.suptitle("CNN Autoencoder Anomaly Detection", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    figures.append(fig1)

    # ── 2. 样本异常分数散点图（按索引顺序） ──────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    indices = np.arange(len(y_true))
    colors = np.where(y_pred == y_true, "steelblue", "tomato")
    ax2.scatter(indices, anomaly_scores, c=colors, s=8, alpha=0.6)
    ax2.axhline(
        y=threshold,
        color="black",
        linestyle="--",
        lw=1.5,
        label=f"Threshold={threshold:.6f}",
    )
    ax2.set_xlabel("Sample Index (Test Set)")
    ax2.set_ylabel("Reconstruction Error")
    ax2.set_title("Per-Sample Anomaly Score (Blue=Correct, Red=Misclassified)")
    ax2.legend()
    fig2.suptitle("CNN Autoencoder Anomaly Detection", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    figures.append(fig2)

    # ── 3. 异常分数分布 ──────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    if train_scores is not None and len(train_scores) > 0:
        ax3.hist(train_scores, bins=30, alpha=0.6, color="steelblue", label="Train (Normal)")

    test_normal = anomaly_scores[y_true == 0]
    test_anomaly = anomaly_scores[y_true == 1]
    if len(test_normal) > 0:
        ax3.hist(test_normal, bins=30, alpha=0.6, color="royalblue", label="Test Normal")
    if len(test_anomaly) > 0:
        ax3.hist(test_anomaly, bins=30, alpha=0.6, color="tomato", label="Test Anomaly")

    ax3.axvline(x=threshold, color="black", linestyle="--", lw=1.5, label="Threshold")
    ax3.set_xlabel("Reconstruction Error")
    ax3.set_ylabel("Count")
    ax3.set_title("Anomaly Score Distribution")
    ax3.legend()
    fig3.suptitle("CNN Autoencoder Anomaly Detection", fontsize=14, fontweight="bold")
    fig3.tight_layout()
    figures.append(fig3)

    # ── 4. 准确率 / Precision / Recall / F1 / AUC ───────────
    fig4, ax4 = plt.subplots(figsize=(7, 6))
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }

    roc_title = None
    unique_test_labels = np.unique(y_true)
    if len(unique_test_labels) >= 2:
        fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        metrics["AUC"] = roc_auc
        roc_title = "ROC Curve (Test)"
    elif train_scores is not None and len(train_scores) > 0:
        y_combined = np.concatenate([np.zeros_like(train_scores, dtype=int), y_true])
        score_combined = np.concatenate([train_scores, anomaly_scores])
        fpr, tpr, _ = roc_curve(y_combined, score_combined)
        roc_auc = auc(fpr, tpr)
        metrics["AUC"] = roc_auc
        roc_title = "ROC Curve (Train Normal vs Test)"

    base_colors = ["steelblue", "mediumseagreen", "goldenrod", "coral", "mediumpurple"]
    colors = base_colors[: len(metrics)]
    bars = ax4.bar(metrics.keys(), metrics.values(), color=colors)
    ax4.set_ylim(0, 1.1)
    ax4.set_ylabel("Score")
    ax4.set_title("Overall Metrics")
    for bar, val in zip(bars, metrics.values()):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig4.suptitle("CNN Autoencoder Anomaly Detection", fontsize=14, fontweight="bold")
    fig4.tight_layout()
    figures.append(fig4)

    # ── 5. ROC 曲线（如可计算） ───────────────────────────────
    if roc_title is not None:
        fig5, ax5 = plt.subplots(figsize=(7, 6))
        ax5.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        ax5.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
        ax5.set_xlim([0.0, 1.0])
        ax5.set_ylim([0.0, 1.05])
        ax5.set_xlabel("False Positive Rate")
        ax5.set_ylabel("True Positive Rate")
        ax5.set_title(roc_title)
        ax5.legend(loc="lower right")
        fig5.suptitle("CNN Autoencoder Anomaly Detection", fontsize=14, fontweight="bold")
        fig5.tight_layout()
        figures.append(fig5)

    with PdfPages(output_pdf) as pdf:
        for fig in figures:
            pdf.savefig(fig, dpi=150, bbox_inches="tight")

    plt.show()
    for fig in figures:
        plt.close(fig)
    print(f"图表已保存至 {output_pdf}")

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
