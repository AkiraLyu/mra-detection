#!/usr/bin/env python3
"""
Visualize each step of the GraphLearner computation in mra.py.

Default behavior:
- uses a fixed, reproducible demo with num_nodes=5 and embed_dim=4
- saves one overview figure plus per-step heatmaps
- also exports each matrix as CSV for report writing
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


def build_demo_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Return fixed demo embeddings so the visualization is deterministic."""
    e1 = np.array(
        [
            [0.90, -0.20, 0.10, 0.60],
            [-0.70, 0.80, -0.40, 0.20],
            [0.30, 0.70, 0.50, -0.90],
            [0.60, -0.10, -0.80, 0.40],
            [-0.20, -0.60, 0.90, 0.70],
        ],
        dtype=np.float32,
    )
    e2 = np.array(
        [
            [0.50, 0.10, -0.70, 0.80],
            [-0.40, 0.90, 0.30, -0.20],
            [0.80, -0.50, 0.60, 0.10],
            [0.20, 0.70, -0.90, 0.40],
            [-0.60, 0.30, 0.80, -0.70],
        ],
        dtype=np.float32,
    )
    return e1, e2


def build_random_embeddings(
    num_nodes: int, embed_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return random embeddings for custom experiments."""
    rng = np.random.default_rng(seed)
    e1 = rng.standard_normal((num_nodes, embed_dim), dtype=np.float32)
    e2 = rng.standard_normal((num_nodes, embed_dim), dtype=np.float32)
    return e1, e2


def compute_graphlearner_steps(
    e1: np.ndarray, e2: np.ndarray, alpha: float
) -> list[tuple[str, np.ndarray, str]]:
    """Reproduce GraphLearner.forward() step by step."""
    alpha_e1 = alpha * e1
    alpha_e2 = alpha * e2
    m1 = np.tanh(alpha_e1)
    m2 = np.tanh(alpha_e2)
    a_raw = m1 @ m2.T
    a_relu = np.maximum(a_raw, 0.0)
    row_sums = a_relu.sum(axis=1, keepdims=True)
    a_norm = a_relu / (row_sums + 1e-8)

    return [
        ("step_01_E1", e1, "E1"),
        ("step_02_E2", e2, "E2"),
        ("step_03_alpha_E1", alpha_e1, f"alpha * E1 (alpha={alpha:g})"),
        ("step_04_alpha_E2", alpha_e2, f"alpha * E2 (alpha={alpha:g})"),
        ("step_05_M1", m1, "M1 = tanh(alpha * E1)"),
        ("step_06_M2", m2, "M2 = tanh(alpha * E2)"),
        ("step_07_A_raw", a_raw, "A_raw = M1 @ M2.T"),
        ("step_08_A_relu", a_relu, "A_relu = ReLU(A_raw)"),
        ("step_09_row_sums", row_sums, "row sums of A_relu"),
        ("step_10_A_norm", a_norm, "A_norm = A_relu / (row_sum + 1e-8)"),
    ]


def axis_labels(matrix: np.ndarray, title: str) -> tuple[list[str], list[str]]:
    """Choose readable labels for the current matrix."""
    rows, cols = matrix.shape
    if "row sums" in title:
        x_labels = ["sum"]
        y_labels = [f"node_{i}" for i in range(rows)]
    elif rows == cols:
        x_labels = [f"node_{i}" for i in range(cols)]
        y_labels = [f"node_{i}" for i in range(rows)]
    else:
        x_labels = [f"dim_{j}" for j in range(cols)]
        y_labels = [f"node_{i}" for i in range(rows)]
    return x_labels, y_labels


def pick_style(title: str, matrix: np.ndarray) -> tuple[str, float | None, float | None]:
    """Pick a colormap and limits that suit the step semantics."""
    if "A_norm" in title or "row sums" in title or "ReLU" in title:
        return "YlOrRd", 0.0, float(np.max(matrix)) if matrix.size else 1.0

    abs_max = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    return "coolwarm", -abs_max, abs_max


def annotate_color(matrix: np.ndarray, row: int, col: int, vmin: float | None, vmax: float | None) -> str:
    """Choose annotation color so numbers remain legible."""
    value = matrix[row, col]
    if not np.isfinite(value):
        return "black"
    if vmin is None or vmax is None or vmax <= vmin:
        return "black"
    midpoint = (vmin + vmax) / 2.0
    return "white" if value > midpoint else "black"


def draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    *,
    add_colorbar: bool = True,
    title_fontsize: int = 10,
    tick_fontsize: int = 8,
    value_fontsize: int = 7,
) -> None:
    """Draw one annotated heatmap on the given axes."""
    x_labels, y_labels = axis_labels(matrix, title)
    cmap, vmin, vmax = pick_style(title, matrix)

    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(y_labels, fontsize=tick_fontsize)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "nan" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=value_fontsize,
                color=annotate_color(matrix, row, col, vmin, vmax),
            )

    if add_colorbar:
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save_matrix_csv(matrix: np.ndarray, title: str, output_dir: Path) -> None:
    """Save the matrix as CSV for later use in reports."""
    x_labels, y_labels = axis_labels(matrix, title)
    df = pd.DataFrame(matrix, index=y_labels, columns=x_labels)
    csv_name = title.lower().replace(" ", "_").replace("/", "_")
    df.to_csv(output_dir / f"{csv_name}.csv", float_format="%.6f")


def save_individual_heatmaps(steps: list[tuple[str, np.ndarray, str]], output_dir: Path) -> None:
    """Save one PNG per step."""
    for step_name, matrix, title in steps:
        fig, ax = plt.subplots(figsize=(6, 4.8))
        draw_heatmap(ax, matrix, title)
        fig.tight_layout()
        fig.savefig(output_dir / f"{step_name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        save_matrix_csv(matrix, title, output_dir)


def save_overview_figure(steps: list[tuple[str, np.ndarray, str]], output_dir: Path) -> None:
    """Save all steps in a single overview image."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 8.5))
    for ax, (_, matrix, title) in zip(axes.flat, steps):
        draw_heatmap(ax, matrix, title)
    fig.suptitle("GraphLearner step-by-step visualization", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "graphlearner_steps_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_arrow_between_axes(
    fig: plt.Figure,
    ax_from: plt.Axes,
    ax_to: plt.Axes,
    label: str,
    *,
    start_side: str = "right",
    end_side: str = "left",
    start_frac: float = 0.5,
    end_frac: float = 0.5,
    text_offset: tuple[float, float] = (0.0, 0.0),
    rad: float = 0.0,
) -> None:
    """Add a figure-level arrow and a short label between two axes."""

    def anchor(ax: plt.Axes, side: str, frac: float) -> tuple[float, float]:
        bbox = ax.get_position()
        if side == "right":
            return bbox.x1, bbox.y0 + bbox.height * frac
        if side == "left":
            return bbox.x0, bbox.y0 + bbox.height * frac
        if side == "top":
            return bbox.x0 + bbox.width * frac, bbox.y1
        if side == "bottom":
            return bbox.x0 + bbox.width * frac, bbox.y0
        raise ValueError(f"Unsupported side: {side}")

    start = anchor(ax_from, start_side, start_frac)
    end = anchor(ax_to, end_side, end_frac)
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        lw=1.8,
        color="#4a5568",
        connectionstyle=f"arc3,rad={rad}",
    )
    fig.add_artist(arrow)

    text_x = (start[0] + end[0]) / 2.0 + text_offset[0]
    text_y = (start[1] + end[1]) / 2.0 + text_offset[1]
    fig.text(
        text_x,
        text_y,
        label,
        ha="center",
        va="center",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#cbd5e0",
            "alpha": 0.95,
        },
    )


def save_paper_style_figure(
    steps: list[tuple[str, np.ndarray, str]], output_dir: Path, alpha: float
) -> None:
    """Save a paper-style summary figure with formulas and arrows."""
    step_map = {step_name: (matrix, title) for step_name, matrix, title in steps}

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.0, 1.0, 1.15, 1.05, 1.05],
        height_ratios=[1.0, 1.0],
        wspace=0.38,
        hspace=0.45,
    )

    ax_e1 = fig.add_subplot(gs[0, 0])
    ax_m1 = fig.add_subplot(gs[0, 1])
    ax_a_raw = fig.add_subplot(gs[0, 2])
    ax_a_relu = fig.add_subplot(gs[0, 3])
    ax_a_norm = fig.add_subplot(gs[0, 4])
    ax_e2 = fig.add_subplot(gs[1, 0])
    ax_m2 = fig.add_subplot(gs[1, 1])
    ax_note = fig.add_subplot(gs[1, 2:5])

    draw_heatmap(
        ax_e1,
        step_map["step_01_E1"][0],
        "E1\nlearnable node embeddings",
        add_colorbar=False,
        title_fontsize=11,
        tick_fontsize=8,
        value_fontsize=8,
    )
    draw_heatmap(
        ax_m1,
        step_map["step_05_M1"][0],
        "M1 = tanh(alpha E1)",
        add_colorbar=False,
        title_fontsize=11,
        tick_fontsize=8,
        value_fontsize=8,
    )
    draw_heatmap(
        ax_e2,
        step_map["step_02_E2"][0],
        "E2\nlearnable node embeddings",
        add_colorbar=False,
        title_fontsize=11,
        tick_fontsize=8,
        value_fontsize=8,
    )
    draw_heatmap(
        ax_m2,
        step_map["step_06_M2"][0],
        "M2 = tanh(alpha E2)",
        add_colorbar=False,
        title_fontsize=11,
        tick_fontsize=8,
        value_fontsize=8,
    )
    draw_heatmap(
        ax_a_raw,
        step_map["step_07_A_raw"][0],
        "A_raw = M1 @ M2.T",
        add_colorbar=False,
        title_fontsize=11,
        tick_fontsize=8,
        value_fontsize=8,
    )
    draw_heatmap(
        ax_a_relu,
        step_map["step_08_A_relu"][0],
        "A_relu = ReLU(A_raw)",
        add_colorbar=False,
        title_fontsize=11,
        tick_fontsize=8,
        value_fontsize=8,
    )
    draw_heatmap(
        ax_a_norm,
        step_map["step_10_A_norm"][0],
        "A = row-normalized adjacency",
        add_colorbar=False,
        title_fontsize=11,
        tick_fontsize=8,
        value_fontsize=8,
    )

    ax_note.axis("off")
    note_text = (
        "GraphLearner pipeline\n\n"
        f"1. Scale the embeddings with alpha = {alpha:g}.\n"
        "2. Apply tanh to limit each embedding dimension to (-1, 1).\n"
        "3. Use matrix multiplication M1 @ M2.T to score every node pair.\n"
        "4. Keep only non-negative edges with ReLU.\n"
        "5. Normalize each row so outgoing weights sum to 1.\n\n"
        "Interpretation:\n"
        "A[i, j] is the learned influence weight from node j to node i."
    )
    ax_note.text(
        0.02,
        0.92,
        note_text,
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.5,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f8fafc",
            "edgecolor": "#cbd5e0",
            "linewidth": 1.2,
        },
    )

    add_arrow_between_axes(
        fig,
        ax_e1,
        ax_m1,
        f"scale by alpha={alpha:g}\nthen tanh",
        text_offset=(0.0, 0.05),
    )
    add_arrow_between_axes(
        fig,
        ax_e2,
        ax_m2,
        f"scale by alpha={alpha:g}\nthen tanh",
        text_offset=(0.0, 0.05),
    )
    add_arrow_between_axes(
        fig,
        ax_m1,
        ax_a_raw,
        "matrix product\nwith M2.T",
        text_offset=(0.0, 0.055),
    )
    add_arrow_between_axes(
        fig,
        ax_m2,
        ax_a_raw,
        "transpose\nand align nodes",
        start_side="right",
        end_side="bottom",
        start_frac=0.55,
        end_frac=0.55,
        text_offset=(-0.03, 0.02),
        rad=-0.18,
    )
    add_arrow_between_axes(
        fig,
        ax_a_raw,
        ax_a_relu,
        "ReLU\nclip negatives to 0",
        text_offset=(0.0, 0.05),
    )
    add_arrow_between_axes(
        fig,
        ax_a_relu,
        ax_a_norm,
        "divide by row sum\nA[i,:] / sum_j A[i,j]",
        text_offset=(0.0, 0.055),
    )

    fig.suptitle(
        "GraphLearner paper-style view: from embeddings to adjacency matrix",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Demo setting: num_nodes=5, embed_dim=4. Values are annotated so the figure can be cited directly in a report.",
        ha="center",
        fontsize=10,
        color="#4a5568",
    )
    fig.savefig(output_dir / "graphlearner_paper_style.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_summary(steps: list[tuple[str, np.ndarray, str]]) -> None:
    """Print the matrices to stdout so the user can inspect numeric values quickly."""
    print("GraphLearner visualization steps:")
    for _, matrix, title in steps:
        print(f"\n[{title}] shape={matrix.shape}")
        print(np.array2string(matrix, precision=3, floatmode="fixed"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize each step of GraphLearner with heatmaps."
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=5,
        help="Number of nodes. Default is 5 for the fixed demo.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=4,
        help="Embedding dimension. Default is 4 for the fixed demo.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=3.0,
        help="Scaling factor used before tanh.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when --random is enabled or dimensions differ from the fixed demo.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random E1/E2 instead of the fixed educational demo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/graphlearner_steps"),
        help="Directory where PNG and CSV outputs will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_fixed_demo = (
        not args.random and args.num_nodes == 5 and args.embed_dim == 4
    )
    if use_fixed_demo:
        e1, e2 = build_demo_embeddings()
    else:
        e1, e2 = build_random_embeddings(args.num_nodes, args.embed_dim, args.seed)

    steps = compute_graphlearner_steps(e1, e2, args.alpha)
    save_individual_heatmaps(steps, args.output_dir)
    save_overview_figure(steps, args.output_dir)
    save_paper_style_figure(steps, args.output_dir, args.alpha)
    print_summary(steps)
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
