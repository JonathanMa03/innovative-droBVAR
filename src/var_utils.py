from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


ArrayLike = np.ndarray


@dataclass
class VARFit:
    """Container for fitted VAR outputs."""
    B_hat: np.ndarray
    residuals: np.ndarray
    Sigma_hat: np.ndarray
    Y_target: np.ndarray
    X: np.ndarray
    feature_names: List[str]
    intercept: Optional[np.ndarray]
    A_hat_list: List[np.ndarray]
    fitted_values: np.ndarray


def companion_matrix(A_list: Sequence[np.ndarray]) -> np.ndarray:
    """
    Construct the companion matrix for a VAR(p).

    Parameters
    ----------
    A_list : sequence of np.ndarray
        List of lag coefficient matrices [A1, ..., Ap], each of shape (k, k).

    Returns
    -------
    np.ndarray
        Companion matrix of shape (k*p, k*p).
    """
    if len(A_list) == 0:
        raise ValueError("A_list must contain at least one lag matrix.")

    p = len(A_list)
    k = A_list[0].shape[0]

    for i, A in enumerate(A_list):
        if A.shape != (k, k):
            raise ValueError(
                f"All A matrices must have shape ({k}, {k}); "
                f"got {A.shape} at lag {i+1}."
            )

    top_block = np.hstack(A_list)

    if p == 1:
        return top_block

    lower_block = np.hstack([
        np.eye(k * (p - 1)),
        np.zeros((k * (p - 1), k))
    ])

    return np.vstack([top_block, lower_block])


def is_var_stable(
    A_list: Sequence[np.ndarray],
    tol: float = 0.999
) -> Tuple[bool, float, np.ndarray]:
    """
    Check whether a VAR(p) is stable via companion-matrix eigenvalues.

    Parameters
    ----------
    A_list : sequence of np.ndarray
        VAR lag coefficient matrices.
    tol : float, default=0.999
        Stability threshold. Stable if max modulus < tol.

    Returns
    -------
    stable : bool
    max_modulus : float
    eigvals : np.ndarray
    """
    F = companion_matrix(A_list)
    eigvals = np.linalg.eigvals(F)
    max_modulus = float(np.max(np.abs(eigvals)))
    stable = max_modulus < tol
    return stable, max_modulus, eigvals


def simulate_var(
    A_list: Sequence[np.ndarray],
    Sigma: np.ndarray,
    T: int,
    burnin: int = 200,
    seed: int = 123,
    intercept: Optional[np.ndarray] = None,
    return_shocks: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a VAR(p) process.

    Model:
        y_t = c + A1 y_{t-1} + ... + Ap y_{t-p} + u_t
        u_t ~ N(0, Sigma)

    Parameters
    ----------
    A_list : sequence of np.ndarray
        VAR coefficient matrices.
    Sigma : np.ndarray
        Innovation covariance matrix of shape (k, k).
    T : int
        Number of retained observations.
    burnin : int, default=200
        Number of burn-in observations.
    seed : int, default=123
        Random seed.
    intercept : np.ndarray, optional
        Intercept vector of shape (k,).
    return_shocks : bool, default=False
        Whether to also return retained shocks.

    Returns
    -------
    Y_retained : np.ndarray
        Simulated series of shape (T, k).
    shocks_retained : np.ndarray, optional
        Retained innovation sequence of shape (T, k).
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if burnin < 0:
        raise ValueError("burnin must be nonnegative.")

    local_rng = np.random.default_rng(seed)

    p = len(A_list)
    if p == 0:
        raise ValueError("A_list must contain at least one lag matrix.")

    k = A_list[0].shape[0]
    Sigma = np.asarray(Sigma, dtype=float)

    if Sigma.shape != (k, k):
        raise ValueError(f"Sigma must have shape ({k}, {k}), got {Sigma.shape}.")

    if intercept is None:
        intercept = np.zeros(k, dtype=float)
    else:
        intercept = np.asarray(intercept, dtype=float)
        if intercept.shape != (k,):
            raise ValueError(f"intercept must have shape ({k},), got {intercept.shape}.")

    total_T = T + burnin
    Y = np.zeros((total_T + p, k), dtype=float)

    shocks = local_rng.multivariate_normal(
        mean=np.zeros(k),
        cov=Sigma,
        size=total_T
    )

    for t in range(p, total_T + p):
        y_t = intercept.copy()

        for lag in range(1, p + 1):
            y_t += A_list[lag - 1] @ Y[t - lag]

        y_t += shocks[t - p]
        Y[t] = y_t

    Y_retained = Y[p + burnin:]
    shocks_retained = shocks[burnin:]

    if return_shocks:
        return Y_retained, shocks_retained
    return Y_retained


def build_var_regression_matrices(
    Y: np.ndarray,
    p: int,
    include_intercept: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build regression matrices for VAR estimation.

    Parameters
    ----------
    Y : np.ndarray
        Time series array of shape (T, k).
    p : int
        Lag order.
    include_intercept : bool, default=True
        Whether to include a constant.

    Returns
    -------
    Y_target : np.ndarray
        Response matrix of shape (T-p, k).
    X : np.ndarray
        Regressor matrix of shape (T-p, m).
    feature_names : list of str
        Names of columns in X.
    """
    Y = np.asarray(Y, dtype=float)

    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D, got shape {Y.shape}.")

    T, k = Y.shape

    if p <= 0:
        raise ValueError("p must be positive.")
    if T <= p:
        raise ValueError(f"Need T > p, but got T={T}, p={p}.")

    X_rows = []
    Y_rows = []
    feature_names: List[str] = []

    if include_intercept:
        feature_names.append("const")

    for lag in range(1, p + 1):
        for j in range(k):
            feature_names.append(f"y{j+1}_lag{lag}")

    for t in range(p, T):
        row = []

        if include_intercept:
            row.append(1.0)

        for lag in range(1, p + 1):
            row.extend(Y[t - lag, :])

        X_rows.append(row)
        Y_rows.append(Y[t, :])

    X = np.asarray(X_rows, dtype=float)
    Y_target = np.asarray(Y_rows, dtype=float)

    return Y_target, X, feature_names


def extract_var_coefficients(
    B: np.ndarray,
    k: int,
    p: int,
    include_intercept: bool = True,
) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
    """
    Extract intercept and lag matrices from stacked coefficient matrix B.

    Parameters
    ----------
    B : np.ndarray
        Coefficient matrix of shape (m, k).
    k : int
        Number of variables.
    p : int
        Lag order.
    include_intercept : bool, default=True
        Whether first row of B is an intercept.

    Returns
    -------
    intercept : np.ndarray or None
    A_list : list of np.ndarray
        Each A_i has shape (k, k).
    """
    B = np.asarray(B, dtype=float)
    row_idx = 0
    intercept: Optional[np.ndarray] = None

    if include_intercept:
        intercept = B[0, :]
        row_idx = 1

    A_list: List[np.ndarray] = []
    for lag in range(p):
        block = B[row_idx + lag * k: row_idx + (lag + 1) * k, :]
        A_list.append(block.T)

    return intercept, A_list


def fit_var_ols(
    Y: np.ndarray,
    p: int,
    include_intercept: bool = True,
) -> VARFit:
    """
    Fit a VAR(p) by multivariate OLS.

    Parameters
    ----------
    Y : np.ndarray
        Series array of shape (T, k).
    p : int
        Lag order.
    include_intercept : bool, default=True
        Whether to include a constant.

    Returns
    -------
    VARFit
        Dataclass containing fitted coefficients, residuals, covariance, etc.
    """
    Y_target, X, feature_names = build_var_regression_matrices(
        Y=Y,
        p=p,
        include_intercept=include_intercept
    )

    XtX = X.T @ X
    XtY = X.T @ Y_target
    B_hat = np.linalg.solve(XtX, XtY)

    fitted_values = X @ B_hat
    residuals = Y_target - fitted_values

    T_eff, k = Y_target.shape
    Sigma_hat = (residuals.T @ residuals) / T_eff

    intercept_hat, A_hat_list = extract_var_coefficients(
        B_hat,
        k=k,
        p=p,
        include_intercept=include_intercept
    )

    return VARFit(
        B_hat=B_hat,
        residuals=residuals,
        Sigma_hat=Sigma_hat,
        Y_target=Y_target,
        X=X,
        feature_names=feature_names,
        intercept=intercept_hat,
        A_hat_list=A_hat_list,
        fitted_values=fitted_values,
    )


def var_forecast(
    A_list: Sequence[np.ndarray],
    intercept: Optional[np.ndarray],
    y_history: np.ndarray,
    h: int,
) -> np.ndarray:
    """
    Deterministic h-step-ahead forecast with zero shocks.

    Parameters
    ----------
    A_list : sequence of np.ndarray
        VAR lag matrices.
    intercept : np.ndarray or None
        Intercept vector.
    y_history : np.ndarray
        Most recent p observations, shape (p, k), ordered oldest -> newest.
    h : int
        Forecast horizon.

    Returns
    -------
    np.ndarray
        Forecast path of shape (h, k).
    """
    if h <= 0:
        raise ValueError("h must be positive.")

    y_history = np.asarray(y_history, dtype=float)
    p = len(A_list)
    k = A_list[0].shape[0]

    if y_history.shape != (p, k):
        raise ValueError(
            f"y_history must have shape ({p}, {k}), got {y_history.shape}."
        )

    history = [y_history[i].copy() for i in range(p)]
    forecasts = []

    for _ in range(h):
        y_next = intercept.copy() if intercept is not None else np.zeros(k, dtype=float)

        for lag in range(1, p + 1):
            y_next += A_list[lag - 1] @ history[-lag]

        forecasts.append(y_next.copy())
        history.append(y_next.copy())

    return np.asarray(forecasts, dtype=float)


def simulate_var_forecast_with_shocks(
    A_list: Sequence[np.ndarray],
    intercept: Optional[np.ndarray],
    y_history: np.ndarray,
    shock_path: np.ndarray,
) -> np.ndarray:
    """
    Simulate a forecast path with a supplied shock path.

    Parameters
    ----------
    A_list : sequence of np.ndarray
        VAR lag matrices.
    intercept : np.ndarray or None
        Intercept vector.
    y_history : np.ndarray
        Most recent p observations, shape (p, k), ordered oldest -> newest.
    shock_path : np.ndarray
        Forecast-period shocks of shape (h, k).

    Returns
    -------
    np.ndarray
        Simulated forecast path of shape (h, k).
    """
    y_history = np.asarray(y_history, dtype=float)
    shock_path = np.asarray(shock_path, dtype=float)

    p = len(A_list)
    k = A_list[0].shape[0]
    h = shock_path.shape[0]

    if y_history.shape != (p, k):
        raise ValueError(
            f"y_history must have shape ({p}, {k}), got {y_history.shape}."
        )
    if shock_path.ndim != 2 or shock_path.shape[1] != k:
        raise ValueError(
            f"shock_path must have shape (h, {k}), got {shock_path.shape}."
        )

    history = [y_history[i].copy() for i in range(p)]
    forecasts = []

    for t in range(h):
        y_next = intercept.copy() if intercept is not None else np.zeros(k, dtype=float)

        for lag in range(1, p + 1):
            y_next += A_list[lag - 1] @ history[-lag]

        y_next += shock_path[t]
        forecasts.append(y_next.copy())
        history.append(y_next.copy())

    return np.asarray(forecasts, dtype=float)