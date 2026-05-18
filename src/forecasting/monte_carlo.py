import numpy as np

from src.var.forecast import forecast_var_simulated


def simulate_var_forecast_paths(
    y_history: np.ndarray,
    beta: np.ndarray,
    shocks: np.ndarray,
    lags: int,
) -> np.ndarray:
    """
    Generate Monte Carlo forecast paths for a fitted VAR.

    Parameters
    ----------
    y_history : np.ndarray
        Historical observations with shape at least (lags, k).

    beta : np.ndarray
        Fitted VAR coefficient matrix including intercept.

    shocks : np.ndarray
        Innovation samples with shape (n_paths, horizon, k).

    lags : int
        VAR lag order.

    Returns
    -------
    paths : np.ndarray
        Forecast paths with shape (n_paths, horizon, k).
    """
    n_paths, horizon, k = shocks.shape

    paths = np.zeros((n_paths, horizon, k))

    for i in range(n_paths):
        paths[i] = forecast_var_simulated(
            y_history=y_history,
            beta=beta,
            shocks=shocks[i],
            lags=lags,
        )

    return paths


def forecast_path_quantiles(
    paths: np.ndarray,
    quantiles: list[float] | tuple[float, ...] = (0.05, 0.5, 0.95),
) -> dict[float, np.ndarray]:
    """
    Compute forecast quantiles across Monte Carlo paths.

    Parameters
    ----------
    paths : np.ndarray
        Shape (n_paths, horizon, k).

    Returns
    -------
    quantile_dict : dict
        Each value has shape (horizon, k).
    """
    return {
        q: np.quantile(paths, q, axis=0)
        for q in quantiles
    }