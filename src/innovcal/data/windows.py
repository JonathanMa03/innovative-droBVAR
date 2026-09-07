"""Leakage-safe scaling, splitting, and window construction."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ChronologicalData:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, minimum_scale: float = 1e-8) -> "Standardizer":
        values = _validate_series(values)
        scale = values.std(axis=0, ddof=1)
        scale = np.maximum(scale, minimum_scale)
        return cls(values.mean(axis=0), scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


def _validate_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("values must have shape (time, dimension) with at least 3 rows")
    if not np.isfinite(values).all():
        raise ValueError("values contain non-finite entries")
    return values


def chronological_split(
    values: np.ndarray, train_fraction: float = 0.6, validation_fraction: float = 0.2
) -> ChronologicalData:
    values = _validate_series(values)
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")
    train_end = int(len(values) * train_fraction)
    validation_end = train_end + int(len(values) * validation_fraction)
    if train_end < 2 or validation_end <= train_end or validation_end >= len(values):
        raise ValueError("split produces an empty partition")
    return ChronologicalData(
        values[:train_end].copy(),
        values[train_end:validation_end].copy(),
        values[validation_end:].copy(),
    )


class WindowDataset(Dataset):
    """Consecutive history windows and one-step targets."""

    def __init__(self, values: np.ndarray, history: int = 20):
        values = _validate_series(values)
        if history < 1 or history >= len(values):
            raise ValueError("history must be positive and shorter than the series")
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.history = int(history)

    def __len__(self) -> int:
        return len(self.values) - self.history

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.values[index : index + self.history], self.values[index + self.history]


def partition_window_datasets(
    split: ChronologicalData,
    standardizer: Standardizer,
    history: int = 20,
) -> tuple[WindowDataset, WindowDataset, WindowDataset]:
    """Construct leakage-safe windows while retaining every validation/test target.

    The validation context receives only the final training observations, and the
    test context receives only the final validation observations. Consequently,
    the first target in each evaluation partition has a fully historical context
    without borrowing any future values.
    """
    if history < 1 or min(len(split.train), len(split.validation)) < history:
        raise ValueError("history exceeds the available preceding partition")
    train = WindowDataset(standardizer.transform(split.train), history)
    validation_values = np.concatenate(
        [split.train[-history:], split.validation], axis=0
    )
    test_values = np.concatenate(
        [split.validation[-history:], split.test], axis=0
    )
    validation = WindowDataset(standardizer.transform(validation_values), history)
    test = WindowDataset(standardizer.transform(test_values), history)
    return train, validation, test
