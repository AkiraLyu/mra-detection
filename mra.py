import argparse
import copy
import json
import os
import random
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams["font.sans-serif"] = ["SimHei"]

WINDOW_START_INDEX = 49
WINDOW_SAMPLE_COUNT = 4000
TEST_SPLIT_INDEX = 2000


@dataclass(frozen=True)
class ModelConfig:
    seq_len: int = 50
    d_model: int = 64
    sampling_rate: int = 6
    graph_hidden_dim: int = 32
    graph_static_dim: int = 8
    graph_coord_dim: int = 8
    self_loop_weight: float = 1.0
    tcn_kernel_sizes: tuple[int, ...] = field(default_factory=lambda: (3, 5, 9))
    use_graph: bool = True
    use_gcn: bool = True
    use_tcn: bool = True
    use_freq: bool = True
    fusion_mode: str = "gated"
    use_sampling_embedding: bool = True


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3
    seeds: tuple[int, ...] = field(default_factory=lambda: (40, 41, 42))
    rand_drop_prob: float = 0.1


@dataclass(frozen=True)
class LossConfig:
    freq_weight: float = 0.1
    sparsity_weight: float = 0.1


@dataclass(frozen=True)
class DetectionConfig:
    ewma_alpha: float = 0.02
    threshold_std: float = 1.5
    min_run: int = 150


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    data_dir: str = "./data"
    checkpoint_root: str = "./test/checkpoints"
    checkpoint_prefix: Optional[str] = None
    output_root: str = "./test/results"
    reuse_checkpoints: bool = True
    save_plot: bool = True

    @property
    def resolved_checkpoint_prefix(self) -> str:
        return self.checkpoint_prefix or self.name


@dataclass
class PreparedData:
    train_windows: np.ndarray
    train_masks: np.ndarray
    test_windows: np.ndarray
    test_masks: np.ndarray
    num_features: int


_DATA_CACHE: dict[tuple[str, int], PreparedData] = {}


def seed_everything(seed: int = 40) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_device(device: Optional[str] = None) -> str:
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class DatasetBuilder:
    def __init__(self, seq_len: int = 60, stride: int = 1):
        self.seq_len = seq_len
        self.stride = stride
        self.num_features = None

    def load_dir(self, dir_path: str, file_pattern: str = "*.csv"):
        import glob

        csv_files = sorted(glob.glob(os.path.join(dir_path, file_pattern)))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files matching '{file_pattern}' in {dir_path}")

        dfs = []
        masks = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, header=None)
            arr = df.to_numpy(dtype=np.float32)
            dfs.append(arr)
            masks.append(np.isnan(arr).astype(np.float32))
            print(f"  Loaded {csv_file}: {len(df)} rows, {df.shape[1]} cols")

        data = np.concatenate(dfs, axis=0)
        mask = np.concatenate(masks, axis=0)
        self.num_features = data.shape[1]
        return data, mask

    def create_windows(self, data: np.ndarray, mask: np.ndarray):
        x_windows = []
        mask_windows = []
        n_rows = len(data)
        num_feat = data.shape[1]

        if n_rows == 0:
            shape = (0, self.seq_len, num_feat)
            return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

        stop_idx = min(n_rows, WINDOW_START_INDEX + WINDOW_SAMPLE_COUNT * self.stride)
        if stop_idx <= WINDOW_START_INDEX:
            shape = (0, self.seq_len, num_feat)
            return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

        for idx in range(WINDOW_START_INDEX, stop_idx, self.stride):
            if idx < self.seq_len:
                pad_len = self.seq_len - idx - 1
                window_data = np.concatenate(
                    [np.tile(data[0:1], (pad_len, 1)), data[0 : idx + 1]],
                    axis=0,
                )
                window_mask = np.concatenate(
                    [np.tile(mask[0:1], (pad_len, 1)), mask[0 : idx + 1]],
                    axis=0,
                )
            else:
                window_data = data[idx - self.seq_len + 1 : idx + 1]
                window_mask = mask[idx - self.seq_len + 1 : idx + 1]

            x_windows.append(window_data)
            mask_windows.append(window_mask)

        return np.stack(x_windows).astype(np.float32), np.stack(mask_windows).astype(np.float32)


def prepare_datasets(
    seq_len: int,
    data_dir: str = "./data",
    train_pattern: str = "train_*.csv",
    test_pattern: str = "test_*.csv",
) -> PreparedData:
    cache_key = (os.path.abspath(data_dir), seq_len)
    if cache_key in _DATA_CACHE:
        print(f"Reusing cached windows for seq_len={seq_len}")
        return _DATA_CACHE[cache_key]

    builder = DatasetBuilder(seq_len=seq_len)

    print("Loading training data...")
    train_data, train_mask = builder.load_dir(os.path.join(data_dir, "train"), train_pattern)
    num_features = builder.num_features
    print(f"Training data: {train_data.shape}, num_features={num_features}")

    print("\nLoading test data...")
    test_data, test_mask = builder.load_dir(os.path.join(data_dir, "test"), test_pattern)
    print(f"Test data: {test_data.shape}")

    train_filled = np.nan_to_num(train_data, nan=0.0)
    test_filled = np.nan_to_num(test_data, nan=0.0)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_filled).astype(np.float32)
    test_scaled = scaler.transform(test_filled).astype(np.float32)

    train_windows, train_masks = builder.create_windows(train_scaled, train_mask)
    test_windows, test_masks = builder.create_windows(test_scaled, test_mask)

    prepared = PreparedData(
        train_windows=train_windows,
        train_masks=train_masks,
        test_windows=test_windows,
        test_masks=test_masks,
        num_features=num_features,
    )
    _DATA_CACHE[cache_key] = prepared
    return prepared


class GraphLearner(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 32,
        static_dim: int = 8,
        coord_dim: int = 8,
        self_loop_weight: float = 1.0,
        base_adj: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.self_loop_weight = self_loop_weight

        if base_adj is None:
            self.register_buffer("base_adj", None)
        else:
            self.register_buffer("base_adj", base_adj.float())

        self.static_context = nn.Parameter(torch.randn(num_nodes, static_dim))
        self.node_coords = nn.Parameter(torch.randn(num_nodes, coord_dim))

        dynamic_dim = 4
        self.node_encoder = nn.Sequential(
            nn.Linear(dynamic_dim + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _build_prior_adjacency(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        eye = torch.eye(self.num_nodes, device=device, dtype=dtype)
        if self.base_adj is not None:
            prior = self.base_adj.to(device=device, dtype=dtype).clamp_min(0.0)
            return prior * (1.0 - eye)

        dist = torch.cdist(self.node_coords, self.node_coords, p=2)
        prior = 1.0 / (1.0 + dist)
        return prior * (1.0 - eye)

    def _last_observed(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        time_index = torch.arange(x.size(1), device=x.device, dtype=x.dtype).view(1, -1, 1)
        latest_index = (time_index * observed).argmax(dim=1).long()
        return x.gather(1, latest_index.unsqueeze(1)).squeeze(1)

    def _dynamic_context(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        observed = 1.0 - mask
        count = observed.sum(dim=1).clamp_min(1.0)
        mean = (x * observed).sum(dim=1) / count
        centered = (x - mean.unsqueeze(1)) * observed
        std = torch.sqrt(centered.pow(2).sum(dim=1) / count + 1e-6)
        last = self._last_observed(x, observed)
        missing_ratio = mask.mean(dim=1)
        return torch.stack([mean, std, last, missing_ratio], dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.zeros_like(x)

        batch_size = x.size(0)
        node_dynamic = self._dynamic_context(x, mask)
        node_static = self.static_context.unsqueeze(0).expand(batch_size, -1, -1)
        node_features = torch.cat([node_dynamic, node_static], dim=-1)
        node_repr = self.node_encoder(node_features)

        h_i = node_repr.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        h_j = node_repr.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)
        pair_features = torch.cat([h_i, h_j, torch.abs(h_i - h_j), h_i * h_j], dim=-1)

        edge_modifier = F.relu(self.edge_mlp(pair_features).squeeze(-1))
        prior = self._build_prior_adjacency(x.device, x.dtype).unsqueeze(0)
        adapt_adj = edge_modifier * prior

        eye = torch.eye(self.num_nodes, device=x.device, dtype=x.dtype).unsqueeze(0)
        adapt_adj = adapt_adj + eye * self.self_loop_weight

        degree = adapt_adj.sum(dim=-1).clamp_min(1e-6)
        inv_sqrt_degree = degree.pow(-0.5)
        return inv_sqrt_degree.unsqueeze(-1) * adapt_adj * inv_sqrt_degree.unsqueeze(-2)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if adj.dim() == 2:
            return torch.einsum("nm,bsmd->bsnd", adj, x)
        return torch.einsum("bnm,bsmd->bsnd", adj, x)


class MultiScaleTCN(nn.Module):
    def __init__(self, num_nodes: int, kernel_sizes: Sequence[int], causal: bool = True):
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")

        self.causal = causal
        self.kernel_sizes = list(kernel_sizes)
        self.convs = nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(
                nn.Conv1d(
                    num_nodes,
                    num_nodes,
                    kernel_size=kernel_size,
                    padding=0,
                    groups=num_nodes,
                )
            )

        self.fusion = nn.Linear(len(self.kernel_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for conv, kernel_size in zip(self.convs, self.kernel_sizes):
            if self.causal:
                padded = F.pad(x, (kernel_size - 1, 0))
            else:
                padded = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2))

            outputs.append(conv(padded))

        out_stack = torch.stack(outputs, dim=-1)
        return self.fusion(out_stack).squeeze(-1)


class FrequencyImputer(nn.Module):
    def __init__(self, seq_len: int):
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
        real, imag = xf.real, xf.imag
        feat = torch.cat([real, imag], dim=-1)

        att_weights = self.attention(feat)
        feat_enhanced = self.freq_enhance(feat)
        real_enh = feat_enhanced[..., : self.freq_len]
        imag_enh = feat_enhanced[..., self.freq_len :]

        real_attended = real_enh * att_weights
        imag_attended = imag_enh * att_weights
        xf_enhanced = xf + torch.complex(real_attended, imag_attended)
        x_rec = torch.fft.irfft(xf_enhanced, n=x.size(1), dim=2)
        return x_rec.permute(0, 2, 1)


class GatedFusion(nn.Module):
    def __init__(self, num_nodes: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv1d(num_nodes * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, num_nodes, kernel_size=1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(num_nodes)

    def forward(self, h_time: torch.Tensor, h_freq: torch.Tensor) -> torch.Tensor:
        h_time_perm = h_time.permute(0, 2, 1)
        h_freq_perm = h_freq.permute(0, 2, 1)
        combined = torch.cat([h_time_perm, h_freq_perm], dim=1)
        gate = self.gate_net(combined).permute(0, 2, 1)
        fused = gate * h_time + (1.0 - gate) * h_freq
        return self.norm(fused)


class AGF_ADNet(nn.Module):
    def __init__(self, num_nodes: int = 18, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.num_nodes = num_nodes
        self.seq_len = self.config.seq_len
        self.sampling_rate = self.config.sampling_rate

        self.graph = None
        if self.config.use_graph:
            self.graph = GraphLearner(
                num_nodes=num_nodes,
                hidden_dim=self.config.graph_hidden_dim,
                static_dim=self.config.graph_static_dim,
                coord_dim=self.config.graph_coord_dim,
                self_loop_weight=self.config.self_loop_weight,
            )

        self.gcn = GCNLayer(1, 1) if self.config.use_gcn else None
        self.tcn = None
        if self.config.use_tcn:
            self.tcn = MultiScaleTCN(num_nodes, kernel_sizes=self.config.tcn_kernel_sizes, causal=False)

        self.freq = FrequencyImputer(self.seq_len) if self.config.use_freq else None
        self.time_norm = nn.LayerNorm(num_nodes) if (self.gcn is not None or self.tcn is not None) else None
        self.freq_norm = nn.LayerNorm(num_nodes) if self.freq is not None else None
        self.fusion = GatedFusion(num_nodes) if self.config.fusion_mode == "gated" else None

        self.input_proj = nn.Linear(num_nodes, self.config.d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, self.seq_len, self.config.d_model))
        self.sampling_rate_embedding = None
        if self.config.use_sampling_embedding:
            self.sampling_rate_embedding = nn.Embedding(self.sampling_rate, self.config.d_model)

        encoder = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        self.output_proj = nn.Linear(self.config.d_model, num_nodes)

    def _identity_adj(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        eye = torch.eye(self.num_nodes, device=device, dtype=dtype)
        return eye.unsqueeze(0).expand(batch_size, -1, -1)

    def _build_sampling_type_index(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1) % self.sampling_rate

    def _compute_time_branch(self, x: torch.Tensor, adj: torch.Tensor) -> Optional[torch.Tensor]:
        parts = []
        if self.gcn is not None:
            parts.append(self.gcn(x.unsqueeze(-1), adj).squeeze(-1))
        if self.tcn is not None:
            parts.append(self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1))
        if not parts:
            return None

        h_time = parts[0]
        for part in parts[1:]:
            h_time = h_time + part
        return self.time_norm(h_time) if self.time_norm is not None else h_time

    def _fuse_branches(
        self,
        h_time: Optional[torch.Tensor],
        h_freq: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if h_time is None and h_freq is None:
            raise ValueError("At least one branch must be enabled")
        if h_time is None:
            return h_freq
        if h_freq is None:
            return h_time

        if self.config.fusion_mode == "gated":
            return self.fusion(h_time, h_freq)
        if self.config.fusion_mode == "mean":
            return 0.5 * (h_time + h_freq)
        if self.config.fusion_mode == "time":
            return h_time
        if self.config.fusion_mode == "freq":
            return h_freq
        raise ValueError(f"Unsupported fusion mode: {self.config.fusion_mode}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        adj = self.graph(x, mask) if self.graph is not None else self._identity_adj(x.size(0), x.device, x.dtype)
        h_time = self._compute_time_branch(x, adj)

        h_freq = None
        if self.freq is not None:
            h_freq = self.freq(x)
            if self.freq_norm is not None:
                h_freq = self.freq_norm(h_freq)

        x_imp = self._fuse_branches(h_time, h_freq)

        observed_mask = 1.0 - mask
        x_filled = x * observed_mask + x_imp * mask

        seq_len = x_filled.size(1)
        if seq_len > self.pos_enc.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds configured seq_len {self.pos_enc.size(1)}"
            )

        z = self.input_proj(x_filled) + self.pos_enc[:, :seq_len]
        if self.sampling_rate_embedding is not None:
            sampling_types = self._build_sampling_type_index(seq_len, x_filled.size(0), x_filled.device)
            z = z + self.sampling_rate_embedding(sampling_types)

        z = self.transformer(z)
        x_rec = self.output_proj(z)
        return x_rec, adj, x_imp


def dual_domain_loss(
    x_rec: torch.Tensor,
    x_true: torch.Tensor,
    missing_mask: torch.Tensor,
    target_mask: torch.Tensor,
    adj: torch.Tensor,
    loss_config: Optional[LossConfig] = None,
) -> torch.Tensor:
    loss_config = loss_config or LossConfig()
    target_mask = target_mask.float()
    recon_loss = ((x_rec - x_true) * target_mask).pow(2).sum() / target_mask.sum().clamp_min(1.0)

    freq_loss = torch.tensor(0.0, device=x_rec.device)
    if loss_config.freq_weight > 0:
        observed_ratio = (1.0 - missing_mask).mean(dim=[1, 2])
        valid_samples = observed_ratio > 0.5
        freq_target = x_rec.detach() * (1.0 - target_mask) + x_true * target_mask

        if valid_samples.sum() > 0:
            x_rec_valid = x_rec[valid_samples].permute(0, 2, 1)
            x_true_valid = freq_target[valid_samples].permute(0, 2, 1)
            fft_rec = torch.fft.rfft(x_rec_valid, dim=2)
            fft_true = torch.fft.rfft(x_true_valid, dim=2)
            freq_loss = (fft_rec.abs() - fft_true.abs()).pow(2).mean()

    sparsity_loss = torch.tensor(0.0, device=x_rec.device)
    if loss_config.sparsity_weight > 0:
        sparsity_loss = -(adj * torch.log(adj + 1e-8)).sum(dim=-1).mean()

    return recon_loss + loss_config.freq_weight * freq_loss + loss_config.sparsity_weight * sparsity_loss


def apply_missing_mask(x: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(missing_mask.bool(), 0.0)


def anomaly_scores(
    model: nn.Module,
    windows: np.ndarray,
    masks: np.ndarray,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    scores = []
    loader = DataLoader(TensorDataset(torch.tensor(windows), torch.tensor(masks)), batch_size=batch_size)

    model.eval()
    with torch.no_grad():
        for x, missing_mask in loader:
            x = x.to(device)
            missing_mask = missing_mask.to(device)

            observed_mask = 1.0 - missing_mask
            x_input = apply_missing_mask(x, missing_mask)
            x_rec, _, _ = model(x_input, missing_mask)

            sq_err = ((x_rec - x) * observed_mask).pow(2).sum(dim=[1, 2])
            obs_cnt = observed_mask.sum(dim=[1, 2]).clamp_min(1e-8)
            scores.extend((sq_err / obs_cnt).cpu().numpy())

    return np.array(scores)


def ewma_smooth(scores: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    smoothed = np.empty_like(scores, dtype=np.float32)
    smoothed[0] = scores[0]
    for idx in range(1, len(scores)):
        smoothed[idx] = alpha * scores[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def enforce_min_run(predictions: np.ndarray, min_run: int = 100) -> np.ndarray:
    filtered = np.zeros_like(predictions, dtype=int)
    start = None

    for idx, value in enumerate(predictions.astype(int)):
        if value == 1 and start is None:
            start = idx
        if (value == 0 or idx == len(predictions) - 1) and start is not None:
            end = idx if value == 0 else idx + 1
            if end - start >= min_run:
                filtered[start:end] = 1
            start = None

    return filtered


def persistent_fault_detection(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    alpha: float = 0.05,
    threshold_std: float = 1.0,
    min_run: int = 100,
):
    train_smoothed = ewma_smooth(train_scores, alpha=alpha)
    test_smoothed = ewma_smooth(test_scores, alpha=alpha)
    threshold = float(np.mean(train_smoothed) + threshold_std * np.std(train_smoothed))
    point_pred = (test_smoothed > threshold).astype(int)
    persistent_pred = enforce_min_run(point_pred, min_run=min_run)
    return train_smoothed, test_smoothed, threshold, point_pred, persistent_pred


def build_test_labels(num_scores: int) -> np.ndarray:
    labels = np.zeros(num_scores, dtype=int)
    split_idx = min(TEST_SPLIT_INDEX, num_scores)
    labels[split_idx:] = 1
    return labels


def evaluate_scores(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    detection_config: DetectionConfig,
) -> dict[str, Any]:
    raw_threshold = float(np.mean(train_scores) + np.std(train_scores))
    test_labels = build_test_labels(len(test_scores))
    test_split_idx = min(TEST_SPLIT_INDEX, len(test_scores))

    raw_pred = (test_scores > raw_threshold).astype(int)
    train_smoothed, test_smoothed, threshold, point_pred, persistent_pred = persistent_fault_detection(
        train_scores,
        test_scores,
        alpha=detection_config.ewma_alpha,
        threshold_std=detection_config.threshold_std,
        min_run=detection_config.min_run,
    )

    metrics = {
        "raw_accuracy": float(accuracy_score(test_labels, raw_pred)),
        "raw_precision": float(precision_score(test_labels, raw_pred, zero_division=0)),
        "raw_recall": float(recall_score(test_labels, raw_pred, zero_division=0)),
        "raw_f1": float(f1_score(test_labels, raw_pred, zero_division=0)),
        "persistent_accuracy": float(accuracy_score(test_labels, persistent_pred)),
        "persistent_precision": float(precision_score(test_labels, persistent_pred, zero_division=0)),
        "persistent_recall": float(recall_score(test_labels, persistent_pred, zero_division=0)),
        "persistent_f1": float(f1_score(test_labels, persistent_pred, zero_division=0)),
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
        "test_score_mean": float(np.mean(test_scores)),
        "test_score_std": float(np.std(test_scores)),
        "smoothed_test_mean": float(np.mean(test_smoothed)),
        "smoothed_test_std": float(np.std(test_smoothed)),
        "raw_threshold": raw_threshold,
        "smoothed_threshold": float(threshold),
        "point_alarm_count": int((point_pred == 1).sum()),
        "persistent_alarm_count": int((persistent_pred == 1).sum()),
        "test_split_idx": int(test_split_idx),
    }

    return {
        "metrics": metrics,
        "labels": test_labels,
        "raw_pred": raw_pred,
        "point_pred": point_pred,
        "persistent_pred": persistent_pred,
        "train_smoothed": train_smoothed,
        "test_smoothed": test_smoothed,
    }


def plot_results(
    scores: np.ndarray,
    threshold: float,
    split_idx: int,
    raw_scores: Optional[np.ndarray] = None,
    save_path: str = "./anomaly_detection_results.png",
) -> None:
    plt.figure(figsize=(6, 5))
    if raw_scores is not None:
        plt.plot(raw_scores, label="原始异常分数", alpha=0.25, color="gray")
    plt.plot(scores, label="平滑异常分数", alpha=0.9)
    plt.axhline(y=threshold, color="r", linestyle="--", label=f"阈值 ({threshold:.4f})")
    plt.axvline(x=split_idx, color="g", linestyle=":", label="测试集分界")
    plt.xlabel("测试样本索引")
    plt.ylabel("异常分数")
    plt.title("AGF-ADNet持续故障检测")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def build_checkpoint_path(experiment: ExperimentConfig, seed: int) -> str:
    return os.path.join(experiment.checkpoint_root, f"{experiment.resolved_checkpoint_prefix}_seed{seed}.pth")


def train_single_agf_model(
    train_windows: np.ndarray,
    train_masks: np.ndarray,
    num_features: int,
    device: str,
    seed: int,
    model_config: Optional[ModelConfig] = None,
    train_config: Optional[TrainConfig] = None,
    loss_config: Optional[LossConfig] = None,
    checkpoint_path: Optional[str] = None,
    reuse_checkpoint: bool = True,
):
    model_config = model_config or ModelConfig()
    train_config = train_config or TrainConfig()
    loss_config = loss_config or LossConfig()

    seed_everything(seed)
    model = AGF_ADNet(num_nodes=num_features, config=model_config).to(device)

    if reuse_checkpoint and checkpoint_path is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model, None, True

    loader = DataLoader(
        TensorDataset(torch.tensor(train_windows), torch.tensor(train_masks)),
        batch_size=train_config.batch_size,
        shuffle=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=train_config.lr)
    best_state = None
    best_loss = float("inf")

    for epoch in range(train_config.epochs):
        model.train()
        total_loss = 0.0

        for x, mask in loader:
            x = x.to(device)
            mask = mask.to(device)

            observed = ~mask.bool()
            rand_drop = (torch.rand_like(x) < train_config.rand_drop_prob) & observed
            target_mask = rand_drop.float()
            if not rand_drop.any():
                target_mask = observed.float()

            input_mask = mask.clone()
            input_mask[rand_drop] = 1.0
            x_input = apply_missing_mask(x, input_mask)

            x_rec, adj, _ = model(x_input, input_mask)
            loss = dual_domain_loss(x_rec, x, mask, target_mask, adj, loss_config=loss_config)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

        print(f"  Seed {seed} Epoch {epoch + 1:02d}/{train_config.epochs}  Loss: {avg_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    return model, best_loss, False


def save_experiment_outputs(experiment: ExperimentConfig, result: dict[str, Any]) -> str:
    exp_dir = Path(experiment.output_root) / experiment.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "experiment": to_serializable(experiment),
    }
    metrics_payload = {
        "metrics": to_serializable(result["metrics"]),
        "train_losses": to_serializable(result["train_losses"]),
        "loaded_seeds": result["loaded_seeds"],
    }

    with (exp_dir / "config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(config_payload, file_obj, ensure_ascii=False, indent=2)

    with (exp_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics_payload, file_obj, ensure_ascii=False, indent=2)

    np.savez(
        exp_dir / "scores.npz",
        train_scores=result["train_scores"],
        test_scores=result["test_scores"],
        train_smoothed=result["train_smoothed"],
        test_smoothed=result["test_smoothed"],
        labels=result["labels"],
        raw_pred=result["raw_pred"],
        point_pred=result["point_pred"],
        persistent_pred=result["persistent_pred"],
    )

    if experiment.save_plot:
        plot_results(
            result["test_smoothed"],
            result["metrics"]["smoothed_threshold"],
            result["metrics"]["test_split_idx"],
            raw_scores=result["test_scores"],
            save_path=str(exp_dir / "anomaly_scores.png"),
        )

    return str(exp_dir)


def run_experiment(
    experiment: ExperimentConfig,
    device: Optional[str] = None,
    force_retrain: bool = False,
) -> dict[str, Any]:
    print(f"\n=== Running experiment: {experiment.name} ===")
    if experiment.description:
        print(experiment.description)

    prepared = prepare_datasets(seq_len=experiment.model.seq_len, data_dir=experiment.data_dir)
    if len(prepared.train_windows) == 0:
        raise RuntimeError("No training windows available")

    resolved_device = resolve_device(device)
    print(f"Device: {resolved_device}")

    if experiment.checkpoint_root:
        os.makedirs(experiment.checkpoint_root, exist_ok=True)

    ensemble_models = []
    train_losses = {}
    loaded_seeds = []

    for seed in experiment.train.seeds:
        checkpoint_path = build_checkpoint_path(experiment, seed)
        model, best_loss, loaded = train_single_agf_model(
            prepared.train_windows,
            prepared.train_masks,
            num_features=prepared.num_features,
            device=resolved_device,
            seed=seed,
            model_config=experiment.model,
            train_config=experiment.train,
            loss_config=experiment.loss,
            checkpoint_path=checkpoint_path,
            reuse_checkpoint=experiment.reuse_checkpoints and not force_retrain,
        )
        if loaded:
            print(f"  Loaded checkpoint for seed {seed}: {checkpoint_path}")
            loaded_seeds.append(seed)
        else:
            print(f"  Best loss for seed {seed}: {best_loss:.6f}")
            print(f"  Checkpoint saved to: {checkpoint_path}")
        train_losses[str(seed)] = None if best_loss is None else float(best_loss)
        ensemble_models.append(model)

    print("\nComputing anomaly scores...")
    train_score_list = []
    test_score_list = []
    for seed, model in zip(experiment.train.seeds, ensemble_models):
        train_seed_scores = anomaly_scores(
            model,
            prepared.train_windows,
            prepared.train_masks,
            device=resolved_device,
            batch_size=experiment.train.batch_size,
        )
        test_seed_scores = anomaly_scores(
            model,
            prepared.test_windows,
            prepared.test_masks,
            device=resolved_device,
            batch_size=experiment.train.batch_size,
        )
        train_score_list.append(train_seed_scores)
        test_score_list.append(test_seed_scores)
        print(
            f"  Seed {seed} score mean: "
            f"train={np.mean(train_seed_scores):.6f}, test={np.mean(test_seed_scores):.6f}"
        )

    train_scores = np.mean(np.stack(train_score_list, axis=0), axis=0)
    test_scores = np.mean(np.stack(test_score_list, axis=0), axis=0)
    evaluation = evaluate_scores(train_scores, test_scores, experiment.detection)

    metrics = dict(evaluation["metrics"])
    metrics.update(
        {
            "experiment": experiment.name,
            "description": experiment.description,
            "device": resolved_device,
            "num_features": prepared.num_features,
            "seed_count": len(experiment.train.seeds),
            "loaded_checkpoint_count": len(loaded_seeds),
            "seq_len": experiment.model.seq_len,
            "d_model": experiment.model.d_model,
            "lr": experiment.train.lr,
            "ewma_alpha": experiment.detection.ewma_alpha,
            "threshold_std": experiment.detection.threshold_std,
            "min_run": experiment.detection.min_run,
        }
    )

    result = {
        "experiment": experiment,
        "metrics": metrics,
        "train_losses": train_losses,
        "loaded_seeds": loaded_seeds,
        "models": ensemble_models,
        "train_scores": train_scores,
        "test_scores": test_scores,
        **evaluation,
    }
    output_dir = save_experiment_outputs(experiment, result)
    result["output_dir"] = output_dir

    print("\nPersistent Fault Metrics:")
    print(f"  Accuracy:  {metrics['persistent_accuracy']:.4f}")
    print(f"  Precision: {metrics['persistent_precision']:.4f}")
    print(f"  Recall:    {metrics['persistent_recall']:.4f}")
    print(f"  F1-Score:  {metrics['persistent_f1']:.4f}")
    print(f"  Results saved to: {output_dir}")

    return result


def build_default_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="full",
        description="完整 AGF-ADNet 基线实验",
        model=ModelConfig(seq_len=50, d_model=64, sampling_rate=6),
        train=TrainConfig(epochs=3, batch_size=32, lr=1e-3, seeds=(40, 41, 42)),
        loss=LossConfig(freq_weight=0.1, sparsity_weight=0.1),
        detection=DetectionConfig(ewma_alpha=0.02, threshold_std=1.5, min_run=150),
        checkpoint_root="./agf_adnet_checkpoints",
        checkpoint_prefix="agf_adnet",
        output_root="./test/results",
        reuse_checkpoints=True,
        save_plot=True,
    )


def train(force_retrain: bool = False, device: Optional[str] = None):
    experiment = build_default_experiment()
    result = run_experiment(experiment, device=device, force_retrain=force_retrain)
    plot_results(
        result["test_smoothed"],
        result["metrics"]["smoothed_threshold"],
        result["metrics"]["test_split_idx"],
        raw_scores=result["test_scores"],
        save_path="./anomaly_detection_results.png",
    )
    return result["models"], result["test_smoothed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the AGF-ADNet baseline.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--force-retrain", action="store_true", help="Ignore cached checkpoints and retrain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(force_retrain=args.force_retrain, device=args.device)


if __name__ == "__main__":
    main()
