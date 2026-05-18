import numpy as np

from src.forecasting.recursive import recursive_forecast_path


def simulate_forecast_paths(
    y_history: np.ndarray,
    beta: np.ndarray,
    innovation_paths: np.ndarray,
    lags: int,
    include_intercept: bool = True,
) -> np.ndarray:
    """
    Generate Monte Carlo forecast paths from sampled innovations.

    Parameters
    ----------
    innovation_paths:
        Shape (n_paths, horizon, k)

    Returns
    -------
    paths:
        Shape (n_paths, horizon, k)
    """
    innovation_paths = np.asarray(innovation_paths, dtype=float)

    if innovation_paths.ndim != 3:
        raise ValueError("innovation_paths must have shape (n_paths, horizon, k).")

    paths = []

    for i in range(innovation_paths.shape[0]):
        path = recursive_forecast_path(
            y_history=y_history,
            beta=beta,
            innovations=innovation_paths[i],
            lags=lags,
            include_intercept=include_intercept,
        )

        paths.append(path)

    return np.asarray(paths)


def simulate_multiple_forecast_paths(
    model_fit: dict,
    y_history: np.ndarray,
    innovation_paths: np.ndarray,
) -> np.ndarray:
    """
    Convenience wrapper for fitted VAR-style models.
    """
    return simulate_forecast_paths(
        y_history=y_history,
        beta=model_fit["beta"],
        innovation_paths=innovation_paths,
        lags=model_fit["lags"],
        include_intercept=model_fit.get("include_intercept", True),
    )