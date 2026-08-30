import numpy as np


def build_lagged_forecast_vector(
    history: list[np.ndarray],
    lags: int,
    include_intercept: bool = True,
) -> np.ndarray:
    x_parts = []

    if include_intercept:
        x_parts.append(1.0)

    for lag in range(1, lags + 1):
        x_parts.extend(history[-lag])

    return np.asarray(x_parts, dtype=float)


def recursive_forecast_path(
    y_history: np.ndarray,
    beta: np.ndarray,
    innovations: np.ndarray,
    lags: int,
    include_intercept: bool = True,
) -> np.ndarray:
    """
    Generate one recursive forecast path.

    Parameters
    ----------
    y_history:
        Historical observations, shape (T_history, k). Must contain at least `lags` rows.

    beta:
        Fitted forecasting coefficients.

    innovations:
        Future innovation path, shape (horizon, k).

    lags:
        Forecast model lag order.

    Returns
    -------
    path:
        Forecast path, shape (horizon, k).
    """
    if len(y_history) < lags:
        raise ValueError("y_history must contain at least `lags` observations.")

    history = [row.copy() for row in y_history]
    path = []

    for shock in innovations:
        x = build_lagged_forecast_vector(
            history=history,
            lags=lags,
            include_intercept=include_intercept,
        )

        y_next = x @ beta + shock

        path.append(y_next)
        history.append(y_next)

    return np.asarray(path)