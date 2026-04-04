from __future__ import annotations

import numpy as np


WINDOW_START_INDEX = 99
TEST_WINDOW_COUNT = 4000
TEST_SPLIT_INDEX = 2000
# Backward-compatible alias for older callers; it now represents the split index.
TEST_SEGMENT_LENGTH = TEST_SPLIT_INDEX


def _empty_window_array(data: np.ndarray, seq_len: int) -> np.ndarray:
    return np.zeros((0, seq_len, data.shape[1]), dtype=data.dtype)


def _resolve_window_start_index(seq_len: int, start_index: int) -> int:
    return max(start_index, seq_len - 1)


def _resolve_window_end_start(seq_len: int, start_index: int) -> int:
    return max(start_index + 1, seq_len)


def _resolve_training_stop(
    length: int,
    start_index: int,
    stride: int,
    max_window_count: int | None,
) -> int:
    if max_window_count is None:
        return length
    return min(length, start_index + max_window_count * stride)


def _resolve_eval_stop(
    length: int,
    start_end_idx: int,
    stride: int,
    window_count: int | None,
) -> int:
    if window_count is None:
        return length + 1
    return min(length + 1, start_end_idx + window_count * stride)


def _resolve_split_index(
    split_index: int,
    legacy_segment_length: int | None,
) -> int:
    resolved = (
        legacy_segment_length if legacy_segment_length is not None else split_index
    )
    if resolved < 0:
        raise ValueError(f"split_index 必须 >= 0，收到 {resolved}")
    return resolved


def _build_eval_end_indices(
    length: int,
    seq_len: int,
    stride: int,
    start_index: int,
    window_count: int | None,
) -> range:
    start_end_idx = _resolve_window_end_start(seq_len, start_index)
    if length < seq_len or start_end_idx > length:
        return range(0)
    stop_idx = _resolve_eval_stop(length, start_end_idx, stride, window_count)
    return range(start_end_idx, stop_idx, stride)


def _build_split_labels(window_count: int, split_index: int) -> np.ndarray:
    labels = np.zeros((window_count,), dtype=np.int64)
    labels[min(split_index, window_count) :] = 1
    return labels


def build_windows(
    data: np.ndarray,
    mask: np.ndarray,
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    window_masks = []
    num_steps = data.shape[0]
    for end_idx in range(0, num_steps, stride):
        start_idx = end_idx - seq_len + 1
        if start_idx >= 0:
            window = data[start_idx : end_idx + 1]
            window_mask = mask[start_idx : end_idx + 1]
        else:
            pad_len = -start_idx
            window = np.concatenate(
                [np.repeat(data[0:1], pad_len, axis=0), data[: end_idx + 1]],
                axis=0,
            )
            window_mask = np.concatenate(
                [np.repeat(mask[0:1], pad_len, axis=0), mask[: end_idx + 1]],
                axis=0,
            )
        windows.append(window)
        window_masks.append(window_mask)

    return np.stack(windows).astype(np.float32), np.stack(window_masks).astype(np.float32)


def build_standard_windows(
    data: np.ndarray,
    mask: np.ndarray,
    seq_len: int,
    stride: int = 1,
    *,
    start_index: int = WINDOW_START_INDEX,
) -> tuple[np.ndarray, np.ndarray]:
    end_indices = _build_eval_end_indices(
        len(data),
        seq_len,
        stride,
        start_index,
        window_count=None,
    )
    if len(end_indices) == 0:
        shape = (0, seq_len, data.shape[1])
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    windows = [data[end_idx - seq_len : end_idx] for end_idx in end_indices]
    window_masks = [mask[end_idx - seq_len : end_idx] for end_idx in end_indices]
    return np.stack(windows).astype(np.float32), np.stack(window_masks).astype(np.float32)


def build_prompt_test_windows(
    data: np.ndarray,
    mask: np.ndarray,
    seq_len: int,
    stride: int = 1,
    *,
    start_index: int = WINDOW_START_INDEX,
    split_index: int = TEST_SPLIT_INDEX,
    window_count: int | None = TEST_WINDOW_COUNT,
    segment_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    end_indices = _build_eval_end_indices(
        len(data),
        seq_len,
        stride,
        start_index,
        window_count,
    )
    if len(end_indices) == 0:
        return (
            _empty_window_array(data, seq_len).astype(np.float32),
            _empty_window_array(mask, seq_len).astype(np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    split_index = _resolve_split_index(split_index, segment_length)
    windows = [data[end_idx - seq_len : end_idx] for end_idx in end_indices]
    window_masks = [mask[end_idx - seq_len : end_idx] for end_idx in end_indices]
    labels = _build_split_labels(len(end_indices), split_index)

    return (
        np.stack(windows).astype(np.float32),
        np.stack(window_masks).astype(np.float32),
        labels,
    )


def build_prompt_test_windows_values(
    data: np.ndarray,
    seq_len: int,
    stride: int = 1,
    *,
    start_index: int = WINDOW_START_INDEX,
    split_index: int = TEST_SPLIT_INDEX,
    window_count: int | None = TEST_WINDOW_COUNT,
    segment_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    end_indices = _build_eval_end_indices(
        len(data),
        seq_len,
        stride,
        start_index,
        window_count,
    )
    if len(end_indices) == 0:
        return _empty_window_array(data, seq_len), np.zeros((0,), dtype=np.int64)

    split_index = _resolve_split_index(split_index, segment_length)
    windows = [data[end_idx - seq_len : end_idx] for end_idx in end_indices]
    labels = _build_split_labels(len(end_indices), split_index)

    return np.stack(windows).astype(data.dtype, copy=False), labels


def build_front_padded_windows(
    data: np.ndarray,
    seq_len: int,
    stride: int = 1,
    *,
    start_index: int = WINDOW_START_INDEX,
    max_window_count: int | None = None,
) -> np.ndarray:
    n = len(data)
    if n == 0:
        return _empty_window_array(data, seq_len)

    first_idx = _resolve_window_start_index(seq_len, start_index)
    stop_idx = _resolve_training_stop(n, first_idx, stride, max_window_count)
    if stop_idx <= first_idx:
        return _empty_window_array(data, seq_len)

    windows = []
    for idx in range(first_idx, stop_idx, stride):
        window = data[idx - seq_len + 1 : idx + 1]
        windows.append(window)

    return np.stack(windows)


def build_front_padded_windows_with_mask(
    data: np.ndarray,
    mask: np.ndarray,
    seq_len: int,
    stride: int = 1,
    *,
    start_index: int = WINDOW_START_INDEX,
    max_window_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(data)
    if n == 0:
        shape = (0, seq_len, data.shape[1])
        return np.zeros(shape, dtype=data.dtype), np.zeros(shape, dtype=mask.dtype)

    first_idx = _resolve_window_start_index(seq_len, start_index)
    stop_idx = _resolve_training_stop(n, first_idx, stride, max_window_count)
    if stop_idx <= first_idx:
        shape = (0, seq_len, data.shape[1])
        return np.zeros(shape, dtype=data.dtype), np.zeros(shape, dtype=mask.dtype)

    windows = []
    window_masks = []
    for idx in range(first_idx, stop_idx, stride):
        window = data[idx - seq_len + 1 : idx + 1]
        window_mask = mask[idx - seq_len + 1 : idx + 1]
        windows.append(window)
        window_masks.append(window_mask)

    return np.stack(windows).astype(data.dtype, copy=False), np.stack(window_masks).astype(
        mask.dtype, copy=False
    )


def _build_forecast_target(
    values: np.ndarray,
    start_idx: int,
    prediction_horizon: int,
) -> np.ndarray:
    end_idx = start_idx + prediction_horizon
    if end_idx <= len(values):
        return values[start_idx:end_idx]

    available = values[start_idx:] if start_idx < len(values) else values[-1:]
    pad_needed = prediction_horizon - len(available)
    return np.concatenate([available, np.tile(values[-1:], (pad_needed, 1))], axis=0)


def build_forecasting_windows(
    data: np.ndarray,
    sequence_length: int,
    prediction_horizon: int = 1,
    stride: int = 1,
    *,
    training: bool = True,
    start_index: int = WINDOW_START_INDEX,
    max_window_count: int | None = None,
    test_split_index: int = TEST_SPLIT_INDEX,
    test_window_count: int | None = TEST_WINDOW_COUNT,
    test_segment_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = data.astype(np.float32)
    x_windows = []
    y_windows = []
    labels = []

    if training:
        first_idx = _resolve_window_start_index(sequence_length, start_index)
        stop_idx = _resolve_training_stop(
            len(values),
            first_idx,
            stride,
            max_window_count,
        )
        for idx in range(first_idx, stop_idx, stride):
            window = values[idx - sequence_length + 1 : idx + 1]
            target = _build_forecast_target(values, idx + 1, prediction_horizon)
            x_windows.append(window)
            y_windows.append(target)
            labels.append(0)
    else:
        end_indices = _build_eval_end_indices(
            len(values),
            sequence_length,
            stride,
            start_index,
            test_window_count,
        )
        test_split_index = _resolve_split_index(
            test_split_index,
            test_segment_length,
        )
        for window_idx, end_idx in enumerate(end_indices):
            window = values[end_idx - sequence_length : end_idx]
            target = _build_forecast_target(values, end_idx, prediction_horizon)
            x_windows.append(window)
            y_windows.append(target)
            labels.append(0 if window_idx < test_split_index else 1)

    if not x_windows:
        return (
            np.zeros((0, sequence_length, values.shape[1]), dtype=np.float32),
            np.zeros((0, prediction_horizon, values.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(x_windows).astype(np.float32),
        np.stack(y_windows).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
    )
