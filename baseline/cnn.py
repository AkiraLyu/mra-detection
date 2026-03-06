import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

def plot_results(y_true, y_pred_prob, y_pred):
    """绘制分类效果可视化图"""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('1D-CNN Anomaly Detection', fontsize=16, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. 混淆矩阵 ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'], ax=ax1)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix')

    # ── 2. ROC 曲线 ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc='lower right')

    # ── 3. 预测概率分布 ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    normal_probs  = y_pred_prob[y_true.flatten() == 0]
    anomaly_probs = y_pred_prob[y_true.flatten() == 1]
    ax3.hist(normal_probs,  bins=30, alpha=0.6, color='steelblue',  label='Normal')
    ax3.hist(anomaly_probs, bins=30, alpha=0.6, color='tomato',     label='Anomaly')
    ax3.axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Threshold=0.5')
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Count')
    ax3.set_title('Prediction Probability Distribution')
    ax3.legend()

    # ── 4. 样本预测散点图（按索引顺序） ──────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    indices = np.arange(len(y_true))
    colors = np.where(y_pred.flatten() == y_true.flatten(), 'steelblue', 'tomato')
    ax4.scatter(indices, y_pred_prob, c=colors, s=8, alpha=0.6)
    ax4.axhline(y=0.5, color='black', linestyle='--', lw=1.5, label='Threshold=0.5')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Predicted Probability')
    ax4.set_title('Per-Sample Prediction  (Blue=Correct, Red=Misclassified)')
    ax4.legend()

    # ── 5. 准确率 / Precision / Recall / F1 柱状图 ──────────
    ax5 = fig.add_subplot(gs[1, 2])
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    metrics = {
        'Accuracy' : accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall'   : recall_score(y_true, y_pred),
        'F1-Score' : f1_score(y_true, y_pred),
        'AUC'      : roc_auc,
    }
    bars = ax5.bar(metrics.keys(), metrics.values(),
                   color=['steelblue', 'mediumseagreen', 'goldenrod', 'coral', 'mediumpurple'])
    ax5.set_ylim(0, 1.1)
    ax5.set_ylabel('Score')
    ax5.set_title('Overall Metrics')
    for bar, val in zip(bars, metrics.values()):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.savefig('classification_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图表已保存至 classification_results.png")

# ---------------------------------------------------------
# 1. 数据加载与预处理 (包含你的原始函数)
# ---------------------------------------------------------

def generate_mask_matrix():
    """读取之前生成的块状分布 CSV 数据"""
    try:
        # 读取 CSV，保留表头以便确认列名
        df = pd.read_csv("../TEP_3000_Block_Split.csv")
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
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 5. 标准化 (Standardization) - 对神经网络非常重要
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
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一层卷积: 输入通道1, 输出通道16, 卷积核大小3
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 41 -> 20
            
            # 第二层卷积
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 20 -> 10
        )
        
        # 全连接分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10, 64), # 32通道 * 长度10
            nn.ReLU(),
            nn.Dropout(0.5),        # 防止过拟合
            nn.Linear(64, 1),
            nn.Sigmoid()            # 二分类输出 0~1 概率
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------------------------------------------
# 3. 训练与评估流程
# ---------------------------------------------------------

def train_model():
    # 准备数据
    X_train, y_train, X_test, y_test = prepare_data()
    
    # 创建 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化模型、损失函数、优化器
    model = AnomalyDetectorCNN()
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 20
    print(f"开始训练，共 {epochs} 个 Epoch...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 评估模型
    print("\n--- 评估结果 ---")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float() # 阈值 0.5
        
        y_true = y_test.numpy()
        y_pred = predicted.numpy()

        # 计算准确率
        accuracy = (predicted.eq(y_test).sum() / float(y_test.shape[0])).item()
        print(f"测试集准确率: {accuracy * 100:.2f}%")
        
        # 详细报告
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
        y_pred_prob = outputs.numpy().flatten()
        plot_results(y_true, y_pred_prob, y_pred)
        
if __name__ == "__main__":
    train_model()