"""Portfolio transformations for simulated asset-return paths."""

import numpy as np


def portfolio_simple_returns_from_log_returns(
    log_returns: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return periodically rebalanced portfolio simple returns.

    ``log_returns`` may have any leading dimensions and assets on the last
    axis. Weights are applied to asset simple returns, not log returns.
    """
    log_returns = np.asarray(log_returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if log_returns.shape[-1] != len(weights):
        raise ValueError("weights must match the final asset dimension.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to one.")
    if not np.isfinite(log_returns).all() or not np.isfinite(weights).all():
        raise ValueError("log_returns and weights must be finite.")
    return np.sum(np.expm1(log_returns) * weights, axis=-1)


def portfolio_wealth_paths(
    portfolio_simple_returns: np.ndarray,
    initial_wealth: float = 1.0,
) -> np.ndarray:
    """Compound portfolio simple returns along the final time axis."""
    values = np.asarray(portfolio_simple_returns, dtype=float)
    if (values <= -1.0).any():
        raise ValueError("simple returns cannot be less than or equal to -100%.")
    return initial_wealth * np.cumprod(1.0 + values, axis=-1)
