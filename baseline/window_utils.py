from __future__ import annotations

import numpy as np


TEST_SEGMENT_LENGTH = 2050
TEST_WINDOW_COUNT = 2000


def _empty_window_array(data: np.ndarray, seq_len: int) -> np.ndarray:
    return np.zeros((0, seq_len, data.shape[1]), dtype=data.dtype)


def build_prompt_test_windows(
    data: np.ndarray,
    seq_len: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror mra.py by windowing normal and fault test segments independently."""
    windows = []
    labels = []
    stop_idx = min(TEST_SEGMENT_LENGTH, seq_len + TEST_WINDOW_COUNT)

    for segment_start, label in ((0, 0), (TEST_SEGMENT_LENGTH, 1)):
        segment_data = data[segment_start : segment_start + TEST_SEGMENT_LENGTH]
        if len(segment_data) < seq_len:
            continue

        for end_idx in range(seq_len, stop_idx, stride):
            if end_idx > len(segment_data):
                break
            windows.append(segment_data[end_idx - seq_len : end_idx])
            labels.append(label)

    if not windows:
        return _empty_window_array(data, seq_len), np.zeros((0,), dtype=np.int64)

    return np.stack(windows).astype(data.dtype, copy=False), np.asarray(
        labels, dtype=np.int64
    )


def build_prompt_test_windows_with_mask(
    data: np.ndarray,
    mask: np.ndarray,
    seq_len: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror mra.py by windowing test data and masks within each test segment."""
    windows = []
    window_masks = []
    labels = []
    stop_idx = min(TEST_SEGMENT_LENGTH, seq_len + TEST_WINDOW_COUNT)

    for segment_start, label in ((0, 0), (TEST_SEGMENT_LENGTH, 1)):
        segment_data = data[segment_start : segment_start + TEST_SEGMENT_LENGTH]
        segment_mask = mask[segment_start : segment_start + TEST_SEGMENT_LENGTH]
        if len(segment_data) < seq_len:
            continue

        for end_idx in range(seq_len, stop_idx, stride):
            if end_idx > len(segment_data):
                break
            windows.append(segment_data[end_idx - seq_len : end_idx])
            window_masks.append(segment_mask[end_idx - seq_len : end_idx])
            labels.append(label)

    if not windows:
        return (
            _empty_window_array(data, seq_len),
            _empty_window_array(mask, seq_len),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(windows).astype(data.dtype, copy=False),
        np.stack(window_masks).astype(mask.dtype, copy=False),
        np.asarray(labels, dtype=np.int64),
    )


def apply_ewaf(scores: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"ewaf alpha 必须在 (0, 1] 内，收到 {alpha}")
    if scores.size == 0:
        return scores.astype(np.float32)

    smoothed = np.empty_like(scores, dtype=np.float32)
    smoothed[0] = np.float32(scores[0])
    for idx in range(1, len(scores)):
        smoothed[idx] = np.float32(
            alpha * scores[idx] + (1.0 - alpha) * smoothed[idx - 1]
        )
    return smoothed


def apply_ewaf_by_segments(
    scores: np.ndarray,
    alpha: float,
    segment_lengths: list[int] | None = None,
) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32)
    if not segment_lengths:
        return apply_ewaf(scores, alpha)

    parts = []
    cursor = 0
    for length in segment_lengths:
        if length <= 0:
            continue
        next_cursor = min(cursor + length, len(scores))
        parts.append(apply_ewaf(scores[cursor:next_cursor], alpha))
        cursor = next_cursor
        if cursor >= len(scores):
            break

    if cursor < len(scores):
        parts.append(apply_ewaf(scores[cursor:], alpha))

    if not parts:
        return scores.astype(np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)
