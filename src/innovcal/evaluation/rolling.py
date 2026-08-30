"""Evaluation across genuine rolling forecast origins."""

import numpy as np
import pandas as pd

from innovcal.evaluation.metrics import summarize_probabilistic_forecast
from innovcal.forecasting.portfolio import portfolio_simple_returns_from_log_returns


def evaluate_rolling_forecasts(
    forecast_tensors: dict[str, np.ndarray],
    observations: np.ndarray,
    horizons: tuple[int, ...] = (1, 5, 20),
    interval: tuple[float, float] = (0.05, 0.95),
    nominal_levels: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> pd.DataFrame:
    """Evaluate each model across origins separately at each horizon."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 3:
        raise ValueError("observations must have shape (origin, horizon, asset).")
    rows = []
    for model, tensor in forecast_tensors.items():
        tensor = np.asarray(tensor, dtype=float)
        if tensor.ndim != 4 or tensor.shape[0] != observations.shape[0]:
            raise ValueError(
                "forecast tensors must have shape (origin, path, horizon, asset)."
            )
        for horizon in horizons:
            if horizon < 1 or horizon > tensor.shape[2]:
                raise ValueError("requested horizon is outside the forecast tensor.")
            # Existing metrics interpret axis 1 as repeated forecast cases. At a
            # fixed horizon, those cases are now genuine rolling origins.
            paths = tensor[:, :, horizon - 1, :].transpose(1, 0, 2)
            truth = observations[:, horizon - 1, :]
            summary = summarize_probabilistic_forecast(
                paths,
                truth,
                interval=interval,
                nominal_levels=nominal_levels,
            )
            row = {
                "innovation_model": model,
                "horizon": horizon,
                "n_origins": len(observations),
                "avg_coverage": summary["avg_coverage"],
                "avg_width": summary["avg_width"],
                "ece": summary["ece"],
                "pit_deviation": summary["pit_deviation"],
                "crps": summary["crps"],
                "energy_score": summary["energy_score"],
                "interval_score": summary["interval_score"],
            }
            row["abs_coverage_error"] = abs(row["avg_coverage"] - 0.9)
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_rolling_portfolios(
    forecast_tensors: dict[str, np.ndarray],
    observations: np.ndarray,
    weights: np.ndarray | None = None,
    interval: tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """Evaluate exact, periodically rebalanced terminal portfolio returns."""
    observations = np.asarray(observations, dtype=float)
    k = observations.shape[-1]
    if weights is None:
        weights = np.full(k, 1.0 / k)
    realized_period = portfolio_simple_returns_from_log_returns(observations, weights)
    realized_terminal = np.prod(1.0 + realized_period, axis=1) - 1.0
    lower_q, upper_q = interval
    rows = []
    for name, tensor in forecast_tensors.items():
        period = portfolio_simple_returns_from_log_returns(tensor, weights)
        terminal = np.prod(1.0 + period, axis=2) - 1.0
        lower = np.quantile(terminal, lower_q, axis=1)
        upper = np.quantile(terminal, upper_q, axis=1)
        median = np.median(terminal, axis=1)
        rows.append(
            {
                "innovation_model": name,
                "horizon": observations.shape[1],
                "n_origins": len(observations),
                "portfolio_coverage": float(
                    np.mean((realized_terminal >= lower) & (realized_terminal <= upper))
                ),
                "portfolio_interval_width": float(np.mean(upper - lower)),
                "portfolio_mae": float(np.mean(np.abs(realized_terminal - median))),
                "portfolio_var_breach_rate": float(np.mean(realized_terminal < lower)),
            }
        )
    return pd.DataFrame(rows)
