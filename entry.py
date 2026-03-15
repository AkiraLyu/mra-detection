import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from model.mstransformer import MSTransformer

import random


def seed_everything(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(40)


# ==========================================
# 1. 数据加载与预处理模块
# ==========================================
def load_csv_dir(dir_path, file_pattern="*.csv"):
    """Load CSV files matching file_pattern from a directory and concatenate."""
    csv_files = sorted(glob.glob(os.path.join(dir_path, file_pattern)))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching '{file_pattern}' in {dir_path}")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, header=None)
        dfs.append(df)
        print(f"  Loaded {f}: {len(df)} rows, {df.shape[1]} cols")
    data = pd.concat(dfs, ignore_index=True).to_numpy(dtype=np.float32)
    return data, data.shape[1]


def create_windows(data, mask, seq_len, stride=1):
    """创建滑动窗口，与 mra.py 的 create_windows 逻辑一致"""
    X, M = [], []
    n = len(data)

    if n == 0:
        return np.zeros((0, seq_len, data.shape[1])), np.zeros((0, seq_len, data.shape[1]))

    for i in range(0, n, stride):
        if i < seq_len:
            pad_len = seq_len - i - 1
            window_data = np.concatenate([
                np.tile(data[0:1], (pad_len, 1)),
                data[0 : i + 1]
            ], axis=0)
            window_mask = np.concatenate([
                np.tile(mask[0:1], (pad_len, 1)),
                mask[0 : i + 1]
            ], axis=0)
        else:
            window_data = data[i - seq_len + 1 : i + 1]
            window_mask = mask[i - seq_len + 1 : i + 1]

        X.append(window_data)
        M.append(window_mask)

    return np.stack(X), np.stack(M)


# ==========================================
# 2. 核心检测与绘图逻辑
# ==========================================
def run_full_detection():
    # --- A. 准备数据 ---
    DATA_DIR = "./data"
    TRAIN_PATTERN = "train_*.csv"
    TEST_PATTERN  = "test_*.csv"

    print("Loading training data...")
    train_data, num_features = load_csv_dir(os.path.join(DATA_DIR, "train"), TRAIN_PATTERN)
    print(f"Training data: {train_data.shape}, num_features={num_features}")

    print("\nLoading test data...")
    test_data, _ = load_csv_dir(os.path.join(DATA_DIR, "test"), TEST_PATTERN)
    print(f"Test data: {test_data.shape}")

    # 处理 NaN
    train_data = np.nan_to_num(train_data, nan=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0)

    # 标准化 (fit on training data only)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data).astype(np.float32)
    test_scaled = scaler.transform(test_data).astype(np.float32)

    # --- B. 模型参数配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = num_features
    window_size = 60
    s_rate = 3

    # --- C. 创建滑动窗口 ---
    train_mask = np.ones_like(train_scaled, dtype=np.float32)
    test_mask = np.ones_like(test_scaled, dtype=np.float32)
    X_train, M_train = create_windows(train_scaled, train_mask, seq_len=window_size, stride=1)
    X_test, M_test = create_windows(test_scaled, test_mask, seq_len=window_size, stride=1)

    # --- D. 初始化模型 ---
    model = MSTransformer(
        enc_in=feature_dim,
        dec_in=feature_dim,
        c_out=feature_dim,
        d_model=64,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        dff=256,
        query_size=16,
        value_size=16,
        sampling_rate=s_rate,
    ).to(device)

    print(f"正在开始检测... 设备: {device}")

    # 强制开启 eval 模式
    model.eval()

    # --- E. 计算训练集异常分数 (用于确定阈值) ---
    train_scores = []
    with torch.no_grad():
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(M_train)),
            batch_size=32
        )
        for x, m in train_loader:
            x, m = x.to(device), m.to(device)

            # 构造 enc_types
            enc_types = torch.arange(x.size(1), device=device).unsqueeze(0).expand(x.size(0), -1) % s_rate
            pred_types = torch.randint(0, s_rate, (x.size(0), 0), dtype=torch.long).to(device)

            # 推理
            reconstructed = model(
                x, x, m, m,
                enc_types, pred_types,
                IFALL=1,
            )

            # 计算重构误差
            sq_err = ((reconstructed - x) * m).pow(2).sum(dim=[1, 2])
            obs_cnt = m.sum(dim=[1, 2]).clamp_min(1e-8)
            train_scores.extend((sq_err / obs_cnt).cpu().numpy())

    train_scores = np.array(train_scores)
    #threshold = np.mean(train_scores) + 3 * np.std(train_scores)
    threshold = np.mean(train_scores)

    print(f"\nTraining Set Score Stats:")
    print(f"  Mean: {np.mean(train_scores):.6f}")
    print(f"  Std:  {np.std(train_scores):.6f}")
    print(f"  Threshold (mean + 3*std): {threshold:.6f}")

    # --- F. 计算测试集异常分数 ---
    test_scores = []
    with torch.no_grad():
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(M_test)),
            batch_size=32
        )
        for x, m in test_loader:
            x, m = x.to(device), m.to(device)

            # 构造 enc_types
            enc_types = torch.arange(x.size(1), device=device).unsqueeze(0).expand(x.size(0), -1) % s_rate
            pred_types = torch.randint(0, s_rate, (x.size(0), 0), dtype=torch.long).to(device)

            # 推理
            reconstructed = model(
                x, x, m, m,
                enc_types, pred_types,
                IFALL=1,
            )

            # 计算重构误差
            sq_err = ((reconstructed - x) * m).pow(2).sum(dim=[1, 2])
            obs_cnt = m.sum(dim=[1, 2]).clamp_min(1e-8)
            test_scores.extend((sq_err / obs_cnt).cpu().numpy())

    test_scores_arr = np.array(test_scores)

    # Splice train scores to front of test scores for evaluation
    all_scores = np.concatenate([train_scores, test_scores_arr])

    print(f"\nAnomaly Detection Results:")
    print(f"  Mean Score: {np.mean(all_scores):.6f}")
    print(f"  Std Score:  {np.std(all_scores):.6f}")
    print(f"  Threshold (from train): {threshold:.6f}")
    print(f"  Anomalies detected: {(all_scores > threshold).sum()} / {len(all_scores)}")

    # Classification Metrics
    # Labels: 0 for train (normal), 1 for test (anomaly)
    y_true = np.concatenate([np.zeros(len(train_scores), dtype=int), np.ones(len(test_scores_arr), dtype=int)])
    y_pred = (all_scores > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # --- G. 结果可视化 ---
    plot_results(all_scores, threshold)


def plot_results(scores, threshold):
    """可视化异常检测结果"""
    plt.figure(figsize=(6, 5))

    plt.plot(scores, label='异常分数', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.4f})')
    plt.xlabel('样本索引')
    plt.ylabel('重构误差')
    plt.title('Multirate Former 异常检测')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/akira/codespace/mra-detection/anomaly_detection_metrics.png', dpi=150)
    plt.show()
    print("图表已保存为 anomaly_detection_metrics.png")


if __name__ == "__main__":
    run_full_detection()

