import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WindowedTimeSeriesDataset(Dataset):
    """
    Windowed dataset for sequence-to-sequence forecasting.

    Given y with shape (T, k), creates samples:

        context: y[t : t + context_length]
        target:  y[t + context_length : t + context_length + prediction_length]
    """

    def __init__(
        self,
        y: np.ndarray,
        context_length: int,
        prediction_length: int,
    ):
        y = np.asarray(y, dtype=np.float32)

        if y.ndim != 2:
            raise ValueError("y must have shape (T, k).")

        self.y = y
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.n_windows = len(y) - context_length - prediction_length + 1

        if self.n_windows <= 0:
            raise ValueError("Time series is too short for requested window lengths.")

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        context_start = idx
        context_end = idx + self.context_length
        target_end = context_end + self.prediction_length

        context = self.y[context_start:context_end]
        target = self.y[context_end:target_end]

        return (
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


def make_deepar_dataloader(
    y: np.ndarray,
    context_length: int,
    prediction_length: int,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    dataset = WindowedTimeSeriesDataset(
        y=y,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )