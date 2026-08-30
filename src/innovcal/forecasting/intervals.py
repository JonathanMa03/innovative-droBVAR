import numpy as np


def forecast_quantiles(
    forecast_paths: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> dict[float, np.ndarray]:
    """
    Compute forecast quantiles over Monte Carlo paths.

    Parameters
    ----------
    forecast_paths:
        Shape (n_paths, horizon, k)

    Returns
    -------
    dict:
        Each value has shape (horizon, k).
    """
    forecast_paths = np.asarray(forecast_paths, dtype=float)

    return {
        q: np.quantile(forecast_paths, q, axis=0)
        for q in quantiles
    }


def prediction_interval(
    forecast_paths: np.ndarray,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    qs = forecast_quantiles(
        forecast_paths,
        quantiles=(lower_q, upper_q),
    )

    return qs[lower_q], qs[upper_q]


def median_forecast(
    forecast_paths: np.ndarray,
) -> np.ndarray:
    return np.quantile(
        forecast_paths,
        0.5,
        axis=0,
    )


def interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return upper - lower


def average_interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return np.mean(
        interval_width(lower, upper),
        axis=0,
    )