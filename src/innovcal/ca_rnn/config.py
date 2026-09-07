"""Configuration objects for CA-RNN experiments."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CARNNConfig:
    input_dim: int
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    min_scale: float = 1e-4

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.hidden_dim < 1 or self.num_layers < 1:
            raise ValueError("model dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lambda_cal: float = 0.0
    lambda_seq: float = 0.0
    kl_weight: float = 0.01
    bayesian_samples: int = 1
    patience: int = 25
    min_delta: float = 1e-5
    calibration_grid_size: int = 19
    calibration_temperature: float = 0.03
    sequential_lags: int = 5
    gradient_clip: float = 5.0
    seed: int = 123
    selection_metric: Literal["nll", "objective"] = "nll"
    early_stopping: bool = True

    def __post_init__(self) -> None:
        positive = (self.epochs, self.batch_size, self.learning_rate, self.patience)
        if any(value <= 0 for value in positive):
            raise ValueError("epochs, batch size, learning rate, and patience must be positive")
        if min(self.lambda_cal, self.lambda_seq, self.kl_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.bayesian_samples < 1 or self.calibration_grid_size < 2:
            raise ValueError("sample and grid sizes are too small")
        if self.selection_metric not in {"nll", "objective"}:
            raise ValueError("selection_metric must be 'nll' or 'objective'")
