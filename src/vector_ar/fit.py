import numpy as np

from src.data.transforms import add_intercept


def create_lag_matrix(
    y: np.ndarray,
    lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct lagged design matrix for VAR(p).

    Parameters
    ----------
    y : np.ndarray
        Shape (T, k)

    lags : int
        VAR lag order

    Returns
    -------
    Y : np.ndarray
        Shape (T - lags, k)

    X : np.ndarray
        Shape (T - lags, k * lags)
    """
    if lags < 1:
        raise ValueError("lags must be at least 1.")

    T, k = y.shape

    Y = y[lags:]

    X_parts = []

    for lag in range(1, lags + 1):
        X_parts.append(
            y[lags - lag : T - lag]
        )

    X = np.concatenate(X_parts, axis=1)

    return Y, X


def fit_var_ols(
    y: np.ndarray,
    lags: int = 1,
    include_intercept: bool = True,
) -> dict:
    """
    Fit VAR(p) using ordinary least squares.
    """
    Y, X = create_lag_matrix(y, lags)

    if include_intercept:
        X = add_intercept(X)

    beta = np.linalg.lstsq(X, Y, rcond=None)[0]

    fitted = X @ beta
    residuals = Y - fitted

    n_eff = Y.shape[0]
    n_params = X.shape[1]

    Sigma_hat = residuals.T @ residuals / (n_eff - n_params)

    return {
        "beta": beta,
        "fitted": fitted,
        "residuals": residuals,
        "Sigma_hat": Sigma_hat,
        "lags": lags,
        "include_intercept": include_intercept,
        "Y": Y,
        "X": X,
    }