#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from graph_gcn_reconstruction import (
    ObservedStandardScaler,
    build_feature_limits,
    build_windows,
    configure_chinese_font,
    load_csv_series,
    plot_series_grid,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 mra.py 中拆出频域模块，独立完成重构和可视化。"
    )
    parser.add_argument(
        "--input-glob",
        default="data/train/train_*.csv",
        help="输入 CSV 模式，默认: data/train/train_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/freq_reconstruction",
        help="输出目录，默认: outputs/freq_reconstruction",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=50,
        help="滑动窗口长度，默认: 50",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="窗口步长，默认: 1",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数，默认: 10",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批大小，默认: 64",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率，默认: 1e-3",
    )
    parser.add_argument(
        "--mask-drop-rate",
        type=float,
        default=0.1,
        help="训练时对观测值随机遮挡的比例，默认: 0.1",
    )
    parser.add_argument(
        "--plot-points",
        type=int,
        default=300,
        help="绘图展示的时间步数，默认: 300",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=40,
        help="随机种子，默认: 40",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="运行设备，默认自动选择 cuda 或 cpu",
    )
    return parser.parse_args()


class FrequencyImputer(nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.freq_len = seq_len // 2 + 1
        self.attention = nn.Sequential(
            nn.Linear(self.freq_len * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.freq_len),
            nn.Sigmoid(),
        )
        self.freq_enhance = nn.Sequential(
            nn.Linear(self.freq_len * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.freq_len * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.permute(0, 2, 1)
        xf = torch.fft.rfft(x_perm, dim=2)

        real = xf.real
        imag = xf.imag
        feat = torch.cat([real, imag], dim=-1)

        att_weights = self.attention(feat)
        feat_enhanced = self.freq_enhance(feat)
        real_enh = feat_enhanced[..., : self.freq_len]
        imag_enh = feat_enhanced[..., self.freq_len :]

        xf_enhanced = xf + torch.complex(real_enh * att_weights, imag_enh * att_weights)
        x_rec = torch.fft.irfft(xf_enhanced, n=x.size(1), dim=2)
        return x_rec.permute(0, 2, 1)


class FrequencyOnlyReconstructor(nn.Module):
    def __init__(self, num_features: int, seq_len: int) -> None:
        super().__init__()
        self.freq = FrequencyImputer(seq_len=seq_len)
        self.output_head = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Linear(num_features, num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_freq = self.freq(x)
        return x_freq + self.output_head(x_freq)


def masked_target_loss(
    x_rec: torch.Tensor,
    x_true: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    squared_error = ((x_rec - x_true) * target_mask).pow(2)
    return squared_error.sum() / target_mask.sum().clamp_min(1.0)


def train_model(
    model: nn.Module,
    train_windows: np.ndarray,
    train_masks: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    mask_drop_rate: float,
) -> nn.Module:
    loader = DataLoader(
        TensorDataset(torch.tensor(train_windows), torch.tensor(train_masks)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"开始训练频域重构模块，共 {epochs} 个 Epoch...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, mask in loader:
            x = x.to(device)
            mask = mask.to(device)

            observed = ~mask.bool()
            random_drop = (torch.rand_like(x) < mask_drop_rate) & observed
            target_mask = random_drop.float()
            if not random_drop.any():
                target_mask = observed.float()

            input_mask = mask.clone()
            input_mask[random_drop] = 1.0
            x_input = x.masked_fill(input_mask.bool(), 0.0)

            x_rec = model(x_input)
            loss = masked_target_loss(x_rec, x, target_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Freq Epoch {epoch + 1:02d}/{epochs}  Loss: {avg_loss:.6f}")

    return model


def reconstruct_sequence(
    model: nn.Module,
    windows: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.tensor(windows), torch.tensor(masks)),
        batch_size=batch_size,
        shuffle=False,
    )

    last_step_reconstruction = []
    model.eval()
    with torch.no_grad():
        for x, mask in loader:
            x = x.to(device)
            mask = mask.to(device)
            x_input = x.masked_fill(mask.bool(), 0.0)
            x_rec = model(x_input)
            last_step_reconstruction.append(x_rec[:, -1, :].cpu().numpy())

    return np.concatenate(last_step_reconstruction, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    configure_chinese_font()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data, missing_mask, feature_names = load_csv_series(args.input_glob)
    scaler = ObservedStandardScaler().fit(raw_data, missing_mask)
    scaled_data = scaler.transform(raw_data, missing_mask)

    windows, window_masks = build_windows(
        scaled_data,
        missing_mask.astype(np.float32),
        seq_len=args.seq_len,
        stride=args.stride,
    )

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = FrequencyOnlyReconstructor(
        num_features=scaled_data.shape[1],
        seq_len=args.seq_len,
    ).to(device)

    train_model(
        model=model,
        train_windows=windows,
        train_masks=window_masks,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_drop_rate=args.mask_drop_rate,
    )

    reconstructed_scaled = reconstruct_sequence(
        model=model,
        windows=windows,
        masks=window_masks,
        device=device,
        batch_size=args.batch_size,
    )
    reconstructed = scaler.inverse_transform(reconstructed_scaled)

    original_csv = output_dir / "original_data.csv"
    reconstructed_csv = output_dir / "reconstructed_data.csv"
    pd.DataFrame(raw_data, columns=feature_names).to_csv(original_csv, index=False)
    pd.DataFrame(reconstructed, columns=feature_names).to_csv(reconstructed_csv, index=False)

    limits = build_feature_limits(raw_data, reconstructed)
    plot_series_grid(
        raw_data,
        feature_names,
        output_dir / "original_data.png",
        "原始数据",
        limits,
        args.plot_points,
    )
    plot_series_grid(
        reconstructed,
        feature_names,
        output_dir / "reconstructed_data.png",
        "频域模块重构数据",
        limits,
        args.plot_points,
    )

    print("输出完成:")
    print(f"  原始数据 CSV: {original_csv}")
    print(f"  重构数据 CSV: {reconstructed_csv}")
    print(f"  原始数据图  : {output_dir / 'original_data.png'}")
    print(f"  重构数据图  : {output_dir / 'reconstructed_data.png'}")


if __name__ == "__main__":
    main()
