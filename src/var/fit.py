import numpy as np


def create_lag_matrix(y: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create response and lagged design matrix for VAR(p).

    Returns
    -------
    Y : np.ndarray
        Shape (T - lags, k)

    X : np.ndarray
        Shape (T - lags, 1 + k * lags)
    """
    T, k = y.shape

    Y = y[lags:]
    X_parts = [np.ones((T - lags, 1))]

    for lag in range(1, lags + 1):
        X_parts.append(y[lags - lag : T - lag])

    X = np.hstack(X_parts)

    return Y, X


def fit_var_ols(y: np.ndarray, lags: int = 1) -> dict:
    """
    Fit VAR(p) by OLS.
    """
    Y, X = create_lag_matrix(y, lags)

    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    fitted = X @ beta
    residuals = Y - fitted

    n_eff = Y.shape[0]
    n_params = X.shape[1]

    Sigma_hat = residuals.T @ residuals / (n_eff - n_params)

    return {
        "beta": beta,
        "intercept": beta[0],
        "coefs": beta[1:],
        "Sigma": Sigma_hat,
        "fitted": fitted,
        "residuals": residuals,
        "lags": lags,
        "Y": Y,
        "X": X,
    }