import numpy as np


def forecast_var_mean(
    y_history: np.ndarray,
    beta: np.ndarray,
    h: int,
    lags: int,
) -> np.ndarray:
    """
    Generate deterministic mean forecasts from fitted VAR(p).
    """
    history = list(y_history.copy())
    k = y_history.shape[1]

    forecasts = []

    for _ in range(h):
        x_parts = [1.0]

        for lag in range(1, lags + 1):
            x_parts.extend(history[-lag])

        x = np.array(x_parts)
        y_next = x @ beta

        forecasts.append(y_next)
        history.append(y_next)

    return np.array(forecasts)


def forecast_var_simulated(
    y_history: np.ndarray,
    beta: np.ndarray,
    shocks: np.ndarray,
    lags: int,
) -> np.ndarray:
    """
    Generate one simulated forecast path using supplied future shocks.

    shocks shape: (h, k)
    """
    history = list(y_history.copy())
    forecasts = []

    for shock in shocks:
        x_parts = [1.0]

        for lag in range(1, lags + 1):
            x_parts.extend(history[-lag])

        x = np.array(x_parts)
        y_next = x @ beta + shock

        forecasts.append(y_next)
        history.append(y_next)

    return np.array(forecasts)