"""Mechanism diagnostics for sample-based interval forecasts."""

import numpy as np
import pandas as pd

from innovcal.evaluation.metrics import _validate


def interval_components(
    target: np.ndarray,
    samples: np.ndarray,
    interval_level: float = 0.9,
) -> dict[str, np.ndarray]:
    """Return origin-by-dimension interval-score components."""
    target, samples = _validate(target, samples)
    if not 0 < interval_level < 1:
        raise ValueError("interval_level must lie in (0, 1)")
    alpha = 1.0 - interval_level
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    width = upper - lower
    lower_penalty = (2.0 / alpha) * (lower - target) * (target < lower)
    upper_penalty = (2.0 / alpha) * (target - upper) * (target > upper)
    return {
        "width": width,
        "lower_penalty": lower_penalty,
        "upper_penalty": upper_penalty,
        "interval_score": width + lower_penalty + upper_penalty,
        "coverage": ((target >= lower) & (target <= upper)).astype(float),
        "lower_miss": (target < lower).astype(float),
        "upper_miss": (target > upper).astype(float),
        "predictive_sd": samples.std(axis=0, ddof=1),
        "absolute_error": np.abs(samples.mean(axis=0) - target),
    }


def summarize_interval_components(
    target: np.ndarray,
    samples: np.ndarray,
    asset_names: list[str] | None = None,
    interval_level: float = 0.9,
) -> pd.DataFrame:
    """Summarize interval mechanisms overall and separately by asset."""
    components = interval_components(target, samples, interval_level)
    dimension = target.shape[1]
    names = asset_names or [f"asset_{index}" for index in range(dimension)]
    if len(names) != dimension:
        raise ValueError("asset_names must match the target dimension")
    rows = []
    for asset, index in [("all", None), *zip(names, range(dimension), strict=True)]:
        row = {"asset": asset}
        for metric, values in components.items():
            selected = values if index is None else values[:, index]
            row[metric] = float(selected.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_components_by_state(
    target: np.ndarray,
    samples: np.ndarray,
    state: np.ndarray,
    labels: list[str] | None = None,
    interval_level: float = 0.9,
) -> pd.DataFrame:
    """Summarize origin-level components within an external state partition."""
    components = interval_components(target, samples, interval_level)
    state = np.asarray(state)
    if state.ndim != 1 or len(state) != len(target):
        raise ValueError("state must have one label per forecast origin")
    levels = labels or list(dict.fromkeys(state.tolist()))
    rows = []
    for level in levels:
        selected = state == level
        if not selected.any():
            continue
        row = {"state": level, "n_origins": int(selected.sum())}
        for metric, values in components.items():
            row[metric] = float(values[selected].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def distribution_shift_summary(
    train: np.ndarray,
    test: np.ndarray,
) -> dict[str, float]:
    """Describe scale, correlation, tail, and location changes."""
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("train and test must share their feature dimension")
    train_scale = train.std(axis=0, ddof=1)
    test_scale = test.std(axis=0, ddof=1)
    standardized = (test - train.mean(axis=0)) / np.maximum(train_scale, 1e-12)
    return {
        "mean_volatility_ratio": float(np.mean(test_scale / train_scale)),
        "max_volatility_ratio": float(np.max(test_scale / train_scale)),
        "correlation_shift_frobenius": float(
            np.linalg.norm(np.corrcoef(test, rowvar=False) - np.corrcoef(train, rowvar=False))
        ),
        "standardized_location_shift": float(np.linalg.norm(standardized.mean(axis=0))),
        "mean_standardized_fourth_moment": float(np.mean(standardized**4)),
    }
