#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mra import FrequencyImputer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the FrequencyImputer input/output on one feature from a CSV file "
            "and save a time-series visualization."
        )
    )
    parser.add_argument(
        "--csv-path",
        default="data/test/test_C5_1.csv",
        help="Input CSV path. Default: data/test/test_C5_1.csv",
    )
    parser.add_argument(
        "--feature-index",
        type=int,
        default=10,
        help="1-based feature index to visualize. Default: 10",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Window length used by FrequencyImputer. Default: 60",
    )
    parser.add_argument(
        "--window-end",
        type=int,
        default=None,
        help=(
            "0-based row index of the window end. Default: last valid row for the "
            "selected feature."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when no checkpoint is provided. Default: 7",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional checkpoint path. Supports either a FrequencyImputer state dict "
            "or an AGF_ADNet checkpoint containing freq.* keys."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/frequency_imputer_simulation",
        help="Directory to save the plot and CSV output. Default: outputs/frequency_imputer_simulation",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure DPI. Default: 180",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    candidates = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "PingFang SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in available), None)

    if selected:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False


def load_feature_series(csv_path: Path, feature_index_1based: int) -> pd.Series:
    df = pd.read_csv(csv_path, header=None)
    feature_index_0based = feature_index_1based - 1

    if feature_index_0based < 0 or feature_index_0based >= df.shape[1]:
        raise ValueError(
            f"Feature index {feature_index_1based} is out of range for {csv_path} "
            f"with {df.shape[1]} columns."
        )

    return pd.to_numeric(df.iloc[:, feature_index_0based], errors="coerce")


def resolve_window_end(series: pd.Series, requested_end: int | None) -> int:
    if requested_end is not None:
        if requested_end < 0 or requested_end >= len(series):
            raise ValueError(
                f"window-end {requested_end} is out of range for series length {len(series)}."
            )
        return requested_end

    last_valid = series.last_valid_index()
    if last_valid is None:
        raise ValueError("Selected feature contains no valid values.")
    return int(last_valid)


def build_window(values: np.ndarray, seq_len: int, end_idx: int) -> np.ndarray:
    if end_idx < seq_len:
        pad_len = seq_len - end_idx - 1
        return np.concatenate(
            [np.tile(values[0:1], pad_len), values[0 : end_idx + 1]], axis=0
        )

    return values[end_idx - seq_len + 1 : end_idx + 1]


def build_all_windows(values: np.ndarray, seq_len: int) -> np.ndarray:
    windows = [build_window(values, seq_len, end_idx) for end_idx in range(len(values))]
    return np.stack(windows, axis=0)


def load_frequency_imputer(
    seq_len: int, checkpoint: str | None, seed: int
) -> tuple[FrequencyImputer, str]:
    torch.manual_seed(seed)
    module = FrequencyImputer(seq_len=seq_len, num_nodes=1)
    module.eval()

    if not checkpoint:
        return module, f"random_init_seed_{seed}"

    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint}.")

    if any(key.startswith("freq.") for key in state):
        freq_state = {key.removeprefix("freq."): value for key, value in state.items() if key.startswith("freq.")}
    else:
        freq_state = state

    missing, unexpected = module.load_state_dict(freq_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint {checkpoint} does not match FrequencyImputer. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )

    return module, f"checkpoint_{Path(checkpoint).stem}"


def run_simulation(
    module: FrequencyImputer, window: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isnan(window)
    module_input = np.nan_to_num(window, nan=0.0).astype(np.float32)

    input_tensor = torch.from_numpy(module_input).view(1, -1, 1)
    with torch.no_grad():
        module_output = module(input_tensor).squeeze(0).squeeze(-1).cpu().numpy()

    return mask, module_input, module_output


def run_full_series_simulation(
    module: FrequencyImputer, series: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_windows = build_all_windows(series, seq_len)
    mask = np.isnan(series)
    module_input = np.nan_to_num(series, nan=0.0).astype(np.float32)

    input_tensor = torch.from_numpy(
        np.nan_to_num(all_windows, nan=0.0).astype(np.float32)
    ).unsqueeze(-1)
    with torch.no_grad():
        all_outputs = module(input_tensor).squeeze(-1).cpu().numpy()

    module_output = all_outputs[:, -1]
    return mask, module_input, module_output


def save_outputs(
    output_dir: Path,
    csv_path: Path,
    feature_index: int,
    end_idx: int,
    window: np.ndarray,
    mask: np.ndarray,
    module_input: np.ndarray,
    module_output: np.ndarray,
    run_label: str,
    dpi: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{csv_path.stem}_feature{feature_index:02d}_end{end_idx}_{run_label}"

    data_path = output_dir / f"{stem}.csv"
    plot_path = output_dir / f"{stem}.png"

    steps = np.arange(len(window))
    observed_mask = ~mask
    observed_values = np.where(observed_mask, window, np.nan)

    out_df = pd.DataFrame(
        {
            "window_step": steps,
            "raw_feature": window,
            "observed_feature": observed_values,
            "missing_mask": mask.astype(int),
            "module_input": module_input,
            "module_output": module_output,
        }
    )
    out_df.to_csv(data_path, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(steps, module_input, color="#1f77b4", linewidth=1.8, label="Module input")
    axes[0].scatter(
        steps[observed_mask],
        window[observed_mask],
        color="#111111",
        s=22,
        label="Observed points",
        zorder=3,
    )
    axes[0].scatter(
        steps[mask],
        module_input[mask],
        color="#d62728",
        marker="x",
        s=24,
        label="Missing -> 0",
        zorder=3,
    )
    axes[0].set_title("FrequencyImputer input window")
    axes[0].set_ylabel("Feature value")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(steps, module_output, color="#ff7f0e", linewidth=1.8, label="Module output")
    axes[1].plot(
        steps,
        module_input,
        color="#1f77b4",
        linewidth=1.0,
        linestyle="--",
        alpha=0.75,
        label="Input reference",
    )
    axes[1].scatter(
        steps[observed_mask],
        window[observed_mask],
        color="#111111",
        s=18,
        alpha=0.8,
        label="Observed points",
        zorder=3,
    )
    axes[1].set_title("FrequencyImputer output window")
    axes[1].set_xlabel("Window step")
    axes[1].set_ylabel("Feature value")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.suptitle(
        f"{csv_path.name} | feature {feature_index} | window end {end_idx} | {run_label}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return data_path, plot_path


def save_full_series_outputs(
    output_dir: Path,
    csv_path: Path,
    feature_index: int,
    series: np.ndarray,
    mask: np.ndarray,
    module_input: np.ndarray,
    module_output: np.ndarray,
    run_label: str,
    dpi: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{csv_path.stem}_feature{feature_index:02d}_full_series_{run_label}"

    data_path = output_dir / f"{stem}.csv"
    plot_path = output_dir / f"{stem}.png"

    steps = np.arange(len(series))
    observed_mask = ~mask
    observed_values = np.where(observed_mask, series, np.nan)

    out_df = pd.DataFrame(
        {
            "sample_index": steps,
            "raw_feature": series,
            "observed_feature": observed_values,
            "missing_mask": mask.astype(int),
            "module_input": module_input,
            "module_output": module_output,
        }
    )
    out_df.to_csv(data_path, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    axes[0].plot(steps, module_input, color="#1f77b4", linewidth=1.2, label="Module input")
    axes[0].scatter(
        steps[observed_mask],
        series[observed_mask],
        color="#111111",
        s=8,
        label="Observed points",
        zorder=3,
    )
    axes[0].scatter(
        steps[mask],
        module_input[mask],
        color="#d62728",
        marker="x",
        s=10,
        alpha=0.65,
        label="Missing -> 0",
        zorder=3,
    )
    axes[0].set_title("Full-series FrequencyImputer input")
    axes[0].set_ylabel("Feature value")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(steps, module_output, color="#ff7f0e", linewidth=1.2, label="Module output")
    axes[1].plot(
        steps,
        module_input,
        color="#1f77b4",
        linewidth=0.9,
        linestyle="--",
        alpha=0.7,
        label="Input reference",
    )
    axes[1].scatter(
        steps[observed_mask],
        series[observed_mask],
        color="#111111",
        s=7,
        alpha=0.75,
        label="Observed points",
        zorder=3,
    )
    axes[1].set_title("Full-series FrequencyImputer output")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Feature value")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.suptitle(
        f"{csv_path.name} | feature {feature_index} | full series | {run_label}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return data_path, plot_path


def save_readable_full_series_figure(
    output_dir: Path,
    csv_path: Path,
    feature_index: int,
    series: np.ndarray,
    mask: np.ndarray,
    module_output: np.ndarray,
    run_label: str,
    dpi: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = (
        output_dir
        / f"{csv_path.stem}_feature{feature_index:02d}_full_series_readable_{run_label}.png"
    )

    steps = np.arange(len(series))
    observed_mask = ~mask

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(16, 13),
        gridspec_kw={"height_ratios": [1.6, 0.22, 1.8, 1.0, 1.0, 1.0]},
    )

    axes[0].scatter(
        steps[observed_mask],
        series[observed_mask],
        color="#111111",
        s=8,
        alpha=0.85,
        label="Observed input samples",
        zorder=3,
    )
    axes[0].set_title(
        "Observed input samples (missing samples are zero-filled before FrequencyImputer)"
    )
    axes[0].set_ylabel("Feature value")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].imshow(
        mask[np.newaxis, :].astype(int),
        aspect="auto",
        cmap="coolwarm",
        interpolation="nearest",
    )
    axes[1].set_title("Missing mask overview (blue=observed, red=missing)")
    axes[1].set_yticks([])
    axes[1].set_ylabel("")

    axes[2].plot(
        steps,
        module_output,
        color="#ff7f0e",
        linewidth=1.2,
        label="Module output",
    )
    axes[2].scatter(
        steps[observed_mask],
        series[observed_mask],
        color="#111111",
        s=7,
        alpha=0.7,
        label="Observed input samples",
        zorder=3,
    )
    axes[2].set_title("Full-series FrequencyImputer output")
    axes[2].set_ylabel("Feature value")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")

    zoom_span = min(180, len(series))
    zoom_starts = [
        0,
        max(0, len(series) // 2 - zoom_span // 2),
        max(0, len(series) - zoom_span),
    ]
    zoom_titles = ["Zoom: start", "Zoom: middle", "Zoom: end"]

    for axis, start, title in zip(axes[3:], zoom_starts, zoom_titles):
        end = min(len(series), start + zoom_span)
        zoom_steps = steps[start:end]
        zoom_series = series[start:end]
        zoom_mask = mask[start:end]
        zoom_output = module_output[start:end]
        zoom_observed = ~zoom_mask

        axis.plot(
            zoom_steps,
            zoom_output,
            color="#ff7f0e",
            linewidth=1.5,
            label="Module output",
        )
        axis.scatter(
            zoom_steps[zoom_observed],
            zoom_series[zoom_observed],
            color="#111111",
            s=16,
            alpha=0.9,
            label="Observed input samples",
            zorder=3,
        )
        axis.set_title(f"{title} ({start} to {end - 1})")
        axis.set_ylabel("Value")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right")

    axes[5].set_xlabel("Sample index")

    fig.suptitle(
        f"{csv_path.name} | feature {feature_index} | readable full-series view | {run_label}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def main() -> None:
    args = parse_args()
    configure_plot_style()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)

    series = load_feature_series(csv_path, args.feature_index)
    end_idx = resolve_window_end(series, args.window_end)
    window = build_window(series.to_numpy(dtype=np.float32), args.seq_len, end_idx)

    module, run_label = load_frequency_imputer(
        seq_len=args.seq_len,
        checkpoint=args.checkpoint,
        seed=args.seed,
    )
    mask, module_input, module_output = run_simulation(module, window)
    data_path, plot_path = save_outputs(
        output_dir=output_dir,
        csv_path=csv_path,
        feature_index=args.feature_index,
        end_idx=end_idx,
        window=window,
        mask=mask,
        module_input=module_input,
        module_output=module_output,
        run_label=run_label,
        dpi=args.dpi,
    )
    full_mask, full_input, full_output = run_full_series_simulation(
        module=module,
        series=series.to_numpy(dtype=np.float32),
        seq_len=args.seq_len,
    )
    full_data_path, full_plot_path = save_full_series_outputs(
        output_dir=output_dir,
        csv_path=csv_path,
        feature_index=args.feature_index,
        series=series.to_numpy(dtype=np.float32),
        mask=full_mask,
        module_input=full_input,
        module_output=full_output,
        run_label=run_label,
        dpi=args.dpi,
    )
    readable_full_plot_path = save_readable_full_series_figure(
        output_dir=output_dir,
        csv_path=csv_path,
        feature_index=args.feature_index,
        series=series.to_numpy(dtype=np.float32),
        mask=full_mask,
        module_output=full_output,
        run_label=run_label,
        dpi=args.dpi,
    )

    print(f"CSV input        : {csv_path}")
    print(f"Feature index    : {args.feature_index}")
    print(f"Window end row   : {end_idx}")
    print(f"Sequence length  : {args.seq_len}")
    print(f"Simulation mode  : {run_label}")
    print(f"Missing points   : {int(mask.sum())} / {len(mask)}")
    print(f"Saved window data: {data_path}")
    print(f"Saved window fig : {plot_path}")
    print(f"Saved full data  : {full_data_path}")
    print(f"Saved full fig   : {full_plot_path}")
    print(f"Saved clean fig  : {readable_full_plot_path}")


if __name__ == "__main__":
    main()
