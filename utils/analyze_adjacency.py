#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPECTED_FEATURES = 18


def make_feature_names(num_features: int) -> list[str]:
    return [f"feature_{i:02d}" for i in range(1, num_features + 1)]


@dataclass
class Sample:
    path: Path
    values: np.ndarray
    valid_mask: np.ndarray

    @property
    def num_rows(self) -> int:
        return int(self.values.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load data/train/*.csv as multivariate time-series samples and analyze "
            "feature adjacency with zero-lag and lead/lag correlations."
        )
    )
    parser.add_argument(
        "--input-glob",
        default="data/train/*.csv",
        help="Glob pattern for training CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/adjacency_analysis",
        help="Directory for CSV reports and heatmaps.",
    )
    parser.add_argument(
        "--expected-features",
        type=int,
        default=EXPECTED_FEATURES,
        help="Expected number of columns in each CSV.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=10,
        help="Maximum lead/lag (in timesteps) to scan on each side.",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=30,
        help="Minimum number of overlapping observations required per pair.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="How many strongest feature relations to summarize.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for generated heatmaps.",
    )
    return parser.parse_args()


def load_samples(input_glob: str, expected_features: int) -> list[Sample]:
    csv_paths = [Path(path) for path in sorted(glob.glob(input_glob))]
    if not csv_paths:
        raise FileNotFoundError(f"No files matched: {input_glob}")

    samples: list[Sample] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] != expected_features:
            raise ValueError(
                f"{csv_path} has {df.shape[1]} columns, expected {expected_features}."
            )

        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        values = numeric_df.to_numpy(dtype=np.float64)
        valid_mask = ~np.isnan(values)
        samples.append(Sample(path=csv_path, values=values, valid_mask=valid_mask))

    return samples


def aligned_views(
    values: np.ndarray, valid_mask: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if lag == 0:
        return values, valid_mask, values, valid_mask

    if lag > 0:
        return values[:-lag], valid_mask[:-lag], values[lag:], valid_mask[lag:]

    shift = -lag
    return values[shift:], valid_mask[shift:], values[:-shift], valid_mask[:-shift]


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_scale = np.linalg.norm(x_centered)
    y_scale = np.linalg.norm(y_centered)

    if x_scale < 1e-12 or y_scale < 1e-12:
        return np.nan

    return float(np.dot(x_centered, y_centered) / (x_scale * y_scale))


def accumulate_lagged_correlations(
    samples: list[Sample], max_lag: int, min_overlap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_features = samples[0].values.shape[1]
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    corr_sum = np.zeros((len(lags), num_features, num_features), dtype=np.float64)
    overlap_sum = np.zeros((len(lags), num_features, num_features), dtype=np.float64)

    for sample in samples:
        for lag_index, lag in enumerate(lags):
            left_values, left_valid, right_values, right_valid = aligned_views(
                sample.values, sample.valid_mask, int(lag)
            )

            for i in range(num_features):
                xi = left_values[:, i]
                mi = left_valid[:, i]

                for j in range(num_features):
                    yj = right_values[:, j]
                    mj = right_valid[:, j]
                    joint = mi & mj
                    overlap = int(joint.sum())

                    if overlap < min_overlap:
                        continue

                    corr = pearson_correlation(xi[joint], yj[joint])
                    if not np.isfinite(corr):
                        continue

                    corr_sum[lag_index, i, j] += corr * overlap
                    overlap_sum[lag_index, i, j] += overlap

    mean_corr = np.full_like(corr_sum, np.nan)
    np.divide(corr_sum, overlap_sum, out=mean_corr, where=overlap_sum > 0)
    return lags, mean_corr, overlap_sum


def build_feature_coverage(samples: list[Sample], feature_names: list[str]) -> pd.DataFrame:
    total_rows = sum(sample.num_rows for sample in samples)
    observed_counts = sum(sample.valid_mask.sum(axis=0) for sample in samples)
    observed_ratio = observed_counts / max(total_rows, 1)
    return pd.DataFrame(
        {
            "feature": feature_names,
            "observed_count": observed_counts.astype(int),
            "observed_ratio": observed_ratio,
            "missing_ratio": 1.0 - observed_ratio,
        }
    )


def summarize_best_relations(
    feature_names: list[str],
    matrix: np.ndarray,
    overlap: np.ndarray,
    top_k: int,
) -> list[tuple[float, float, int, str, str]]:
    relations: list[tuple[float, float, int, str, str]] = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            corr = matrix[i, j]
            count = int(overlap[i, j])
            if not np.isfinite(corr) or count <= 0:
                continue
            relations.append((abs(corr), corr, count, feature_names[i], feature_names[j]))

    relations.sort(key=lambda item: item[0], reverse=True)
    return relations[:top_k]


def summarize_best_lagged_relations(
    feature_names: list[str],
    best_corr: np.ndarray,
    best_lag: np.ndarray,
    best_overlap: np.ndarray,
    top_k: int,
) -> list[tuple[float, float, int, int, str, str]]:
    relations: list[tuple[float, float, int, int, str, str]] = []
    for i in range(best_corr.shape[0]):
        for j in range(i + 1, best_corr.shape[1]):
            corr = best_corr[i, j]
            lag = int(best_lag[i, j])
            count = int(best_overlap[i, j])
            if not np.isfinite(corr) or count <= 0:
                continue
            relations.append(
                (abs(corr), corr, lag, count, feature_names[i], feature_names[j])
            )

    relations.sort(key=lambda item: item[0], reverse=True)
    return relations[:top_k]


def relation_text(feature_a: str, feature_b: str, lag: int) -> str:
    if lag == 0:
        return f"{feature_a} <-> {feature_b} (same timestep)"
    if lag > 0:
        return f"{feature_a} -> {feature_b} ({lag} steps)"
    return f"{feature_b} -> {feature_a} ({abs(lag)} steps)"


def save_matrix_csv(
    matrix: np.ndarray, feature_names: list[str], output_path: Path, float_format: str = "%.6f"
) -> None:
    df = pd.DataFrame(matrix, index=feature_names, columns=feature_names)
    df.to_csv(output_path, float_format=float_format)


def save_heatmap(
    matrix: np.ndarray,
    feature_names: list[str],
    title: str,
    output_path: Path,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    dpi: int,
    fmt: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "nan" if not np.isfinite(value) else format(value, fmt)
            ax.text(col, row, text, ha="center", va="center", fontsize=6, color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def relu_row_normalize(matrix: np.ndarray) -> np.ndarray:
    relu_matrix = np.maximum(matrix, 0.0).copy()
    np.fill_diagonal(relu_matrix, 0.0)
    row_sums = relu_matrix.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(relu_matrix)
    np.divide(relu_matrix, row_sums, out=normalized, where=row_sums > 0)
    return normalized


def select_best_lag(
    lags: np.ndarray, mean_corr: np.ndarray, overlap_sum: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_features = mean_corr.shape[1]
    best_corr = np.full((num_features, num_features), np.nan, dtype=np.float64)
    best_lag = np.zeros((num_features, num_features), dtype=int)
    best_overlap = np.zeros((num_features, num_features), dtype=int)

    for i in range(num_features):
        for j in range(num_features):
            lag_series = mean_corr[:, i, j]
            valid = np.isfinite(lag_series)

            if not np.any(valid):
                continue

            valid_indices = np.flatnonzero(valid)
            local_idx = valid_indices[np.argmax(np.abs(lag_series[valid]))]
            best_corr[i, j] = lag_series[local_idx]
            best_lag[i, j] = int(lags[local_idx])
            best_overlap[i, j] = int(overlap_sum[local_idx, i, j])

    return best_corr, best_lag, best_overlap


def build_summary_text(
    samples: list[Sample],
    feature_coverage: pd.DataFrame,
    zero_lag_corr: np.ndarray,
    zero_lag_overlap: np.ndarray,
    best_corr: np.ndarray,
    best_lag: np.ndarray,
    best_overlap: np.ndarray,
    max_lag: int,
    top_k: int,
) -> str:
    feature_names = feature_coverage["feature"].tolist()
    total_rows = sum(sample.num_rows for sample in samples)
    zero_relations = summarize_best_relations(
        feature_names, zero_lag_corr, zero_lag_overlap, top_k
    )
    lagged_relations = summarize_best_lagged_relations(
        feature_names, best_corr, best_lag, best_overlap, top_k
    )

    lines: list[str] = []
    lines.append("Adjacency analysis summary")
    lines.append(f"Files loaded: {len(samples)}")
    lines.append(f"Total timesteps: {total_rows}")
    lines.append(f"Features per timestep: {len(feature_names)}")
    lines.append("")
    lines.append("Feature coverage")
    for row in feature_coverage.itertuples(index=False):
        lines.append(
            f"  {row.feature}: observed={row.observed_count}, "
            f"observed_ratio={row.observed_ratio:.3f}, missing_ratio={row.missing_ratio:.3f}"
        )

    lines.append("")
    lines.append(f"Top {len(zero_relations)} zero-lag relations")
    for strength, corr, overlap, feature_a, feature_b in zero_relations:
        lines.append(
            f"  {feature_a} <-> {feature_b}: corr={corr:.4f}, "
            f"|corr|={strength:.4f}, overlap={overlap}"
        )

    lines.append("")
    lines.append(
        f"Top {len(lagged_relations)} relations within lag window [-{max_lag}, {max_lag}]"
    )
    for strength, corr, lag, overlap, feature_a, feature_b in lagged_relations:
        lines.append(
            f"  {relation_text(feature_a, feature_b, lag)}: corr={corr:.4f}, "
            f"|corr|={strength:.4f}, overlap={overlap}"
        )

    lines.append("")
    lines.append("Lag sign convention")
    lines.append(
        "  Positive lag means the row feature at time t aligns best with the column "
        "feature at time t + lag, so the row feature leads."
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input_glob, args.expected_features)
    num_features = samples[0].values.shape[1]
    feature_names = make_feature_names(num_features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_coverage = build_feature_coverage(samples, feature_names)
    lags, mean_corr, overlap_sum = accumulate_lagged_correlations(
        samples=samples,
        max_lag=args.max_lag,
        min_overlap=args.min_overlap,
    )

    zero_lag_index = int(np.where(lags == 0)[0][0])
    zero_lag_corr = mean_corr[zero_lag_index].copy()
    zero_lag_overlap = overlap_sum[zero_lag_index].copy()
    best_corr, best_lag, best_overlap = select_best_lag(lags, mean_corr, overlap_sum)
    np.fill_diagonal(zero_lag_corr, 1.0)
    np.fill_diagonal(best_corr, 0.0)
    np.fill_diagonal(best_lag, 0)
    np.fill_diagonal(best_overlap, 0)
    adjacency_strength = np.abs(best_corr)
    relu_row_normalized_adjacency = relu_row_normalize(best_corr)

    feature_coverage.to_csv(output_dir / "feature_coverage.csv", index=False)
    save_matrix_csv(zero_lag_corr, feature_names, output_dir / "zero_lag_correlation.csv")
    save_matrix_csv(best_corr, feature_names, output_dir / "best_lag_correlation.csv")
    save_matrix_csv(adjacency_strength, feature_names, output_dir / "adjacency_strength.csv")
    save_matrix_csv(
        relu_row_normalized_adjacency,
        feature_names,
        output_dir / "relu_row_normalized_adjacency.csv",
    )
    save_matrix_csv(best_lag, feature_names, output_dir / "best_lag.csv", float_format="%d")
    save_matrix_csv(best_overlap, feature_names, output_dir / "overlap_count.csv", float_format="%d")

    save_heatmap(
        zero_lag_corr,
        feature_names,
        "Zero-lag correlation",
        output_dir / "zero_lag_correlation_heatmap.png",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        dpi=args.dpi,
        fmt=".2f",
    )
    save_heatmap(
        adjacency_strength,
        feature_names,
        "Adjacency strength |best lagged corr|",
        output_dir / "adjacency_strength_heatmap.png",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        dpi=args.dpi,
        fmt=".2f",
    )
    save_heatmap(
        relu_row_normalized_adjacency,
        feature_names,
        "ReLU + row-normalized adjacency",
        output_dir / "relu_row_normalized_adjacency_heatmap.png",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        dpi=args.dpi,
        fmt=".2f",
    )
    save_heatmap(
        best_lag.astype(np.float64),
        feature_names,
        "Best lag (row feature leads when positive)",
        output_dir / "best_lag_heatmap.png",
        cmap="coolwarm",
        vmin=-float(args.max_lag),
        vmax=float(args.max_lag),
        dpi=args.dpi,
        fmt=".0f",
    )

    summary_text = build_summary_text(
        samples=samples,
        feature_coverage=feature_coverage,
        zero_lag_corr=zero_lag_corr,
        zero_lag_overlap=zero_lag_overlap,
        best_corr=best_corr,
        best_lag=best_lag,
        best_overlap=best_overlap,
        max_lag=args.max_lag,
        top_k=args.top_k,
    )
    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    print(summary_text)
    print("")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
