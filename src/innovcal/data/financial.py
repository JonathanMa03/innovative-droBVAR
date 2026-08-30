"""Financial transformations and leakage-safe chronological splitting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    """Non-overlapping training, calibration, and test partitions."""

    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


def prices_to_log_returns(
    prices: pd.DataFrame,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Convert positive, time-ordered asset prices to aligned log returns."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.empty or prices.shape[1] < 1:
        raise ValueError("prices must contain at least one asset column.")
    if prices.index.has_duplicates:
        raise ValueError("prices must not contain duplicate timestamps.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be ordered chronologically.")

    numeric = prices.astype(float)
    if (numeric <= 0).any().any():
        raise ValueError("prices must be strictly positive.")

    returns = np.log(numeric).diff()
    if drop_missing:
        returns = returns.dropna(how="any")
    return returns


def chronological_split(
    values: np.ndarray,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> ChronologicalSplit:
    """Split observations chronologically without shuffling or overlap."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape (n_observations, n_series).")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")
    if not 0 < calibration_fraction < 1:
        raise ValueError(
            "calibration_fraction must lie strictly between 0 and 1."
        )
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must sum to less than 1.")

    train_end = int(len(values) * train_fraction)
    calibration_end = train_end + int(len(values) * calibration_fraction)
    if train_end < 2 or calibration_end <= train_end or calibration_end >= len(values):
        raise ValueError("not enough observations for three non-empty partitions.")

    return ChronologicalSplit(
        train=values[:train_end].copy(),
        calibration=values[train_end:calibration_end].copy(),
        test=values[calibration_end:].copy(),
    )


def make_demo_prices(
    n_observations: int = 500,
    n_assets: int = 4,
    seed: int = 123,
) -> pd.DataFrame:
    """Create deterministic correlated prices for notebook smoke runs.

    These data validate the workflow only and must not be presented as the
    empirical financial case study.
    """
    if n_observations < 10 or n_assets < 1:
        raise ValueError("n_observations must be at least 10 and n_assets positive.")
    rng = np.random.default_rng(seed)
    correlation = 0.35 * np.ones((n_assets, n_assets)) + 0.65 * np.eye(n_assets)
    scales = np.linspace(0.008, 0.014, n_assets)
    covariance = correlation * np.outer(scales, scales)
    returns = rng.multivariate_normal(
        mean=np.full(n_assets, 0.0002),
        cov=covariance,
        size=n_observations - 1,
    )
    prices = np.vstack(
        [np.full(n_assets, 100.0), 100.0 * np.exp(np.cumsum(returns, axis=0))]
    )
    return pd.DataFrame(
        prices,
        index=pd.bdate_range("2020-01-01", periods=n_observations),
        columns=[f"Asset_{i + 1}" for i in range(n_assets)],
    )
