import numpy as np


def _build_forecast_vector(
    history: list[np.ndarray],
    lags: int,
    include_intercept: bool,
) -> np.ndarray:
    x_parts = []

    if include_intercept:
        x_parts.append(1.0)

    for lag in range(1, lags + 1):
        x_parts.extend(history[-lag])

    return np.array(x_parts)


def forecast_var_mean(
    y_history: np.ndarray,
    beta: np.ndarray,
    horizon: int,
    lags: int,
    include_intercept: bool = True,
) -> np.ndarray:
    """
    Deterministic recursive mean forecasts.
    """
    history = list(y_history.copy())
    forecasts = []

    for _ in range(horizon):
        x = _build_forecast_vector(
            history=history,
            lags=lags,
            include_intercept=include_intercept,
        )

        y_next = x @ beta

        forecasts.append(y_next)
        history.append(y_next)

    return np.asarray(forecasts)


def forecast_var_path(
    y_history: np.ndarray,
    beta: np.ndarray,
    shocks: np.ndarray,
    lags: int,
    include_intercept: bool = True,
) -> np.ndarray:
    """
    Generate one stochastic forecast trajectory.

    Parameters
    ----------
    shocks : np.ndarray
        Shape (horizon, k)
    """
    history = list(y_history.copy())
    forecasts = []

    for shock in shocks:
        x = _build_forecast_vector(
            history=history,
            lags=lags,
            include_intercept=include_intercept,
        )

        y_next = x @ beta + shock

        forecasts.append(y_next)
        history.append(y_next)

    return np.asarray(forecasts)


def forecast_var_paths(
    y_history: np.ndarray,
    beta: np.ndarray,
    shock_paths: np.ndarray,
    lags: int,
    include_intercept: bool = True,
) -> np.ndarray:
    """
    Generate Monte Carlo forecast trajectories.

    Parameters
    ----------
    shock_paths : np.ndarray
        Shape (n_paths, horizon, k)

    Returns
    -------
    paths : np.ndarray
        Shape (n_paths, horizon, k)
    """
    n_paths = shock_paths.shape[0]

    paths = []

    for i in range(n_paths):
        path = forecast_var_path(
            y_history=y_history,
            beta=beta,
            shocks=shock_paths[i],
            lags=lags,
            include_intercept=include_intercept,
        )

        paths.append(path)

    return np.asarray(paths)


def forecast_quantiles(
    forecast_paths: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> dict:
    """
    Compute forecast quantiles over Monte Carlo paths.
    """
    return {
        q: np.quantile(forecast_paths, q, axis=0)
        for q in quantiles
    }