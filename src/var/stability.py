import numpy as np


def companion_matrix(beta: np.ndarray, k: int, lags: int) -> np.ndarray:
    """
    Construct companion matrix from VAR coefficient matrix.

    beta excludes intercept and has shape (k * lags, k).
    """
    A_blocks = []

    for lag in range(lags):
        block = beta[lag * k : (lag + 1) * k].T
        A_blocks.append(block)

    top = np.hstack(A_blocks)

    if lags == 1:
        return top

    bottom_left = np.eye(k * (lags - 1))
    bottom_right = np.zeros((k * (lags - 1), k))
    bottom = np.hstack([bottom_left, bottom_right])

    return np.vstack([top, bottom])


def var_eigenvalues(beta_no_intercept: np.ndarray, k: int, lags: int) -> np.ndarray:
    C = companion_matrix(beta_no_intercept, k, lags)
    return np.linalg.eigvals(C)


def is_var_stable(beta_no_intercept: np.ndarray, k: int, lags: int) -> bool:
    eigvals = var_eigenvalues(beta_no_intercept, k, lags)
    return np.max(np.abs(eigvals)) < 1