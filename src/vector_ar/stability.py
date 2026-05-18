import numpy as np


def companion_matrix(
    beta_no_intercept: np.ndarray,
    k: int,
    lags: int,
) -> np.ndarray:
    """
    Construct VAR companion matrix.
    """
    blocks = []

    for lag in range(lags):
        block = beta_no_intercept[
            lag * k : (lag + 1) * k
        ].T

        blocks.append(block)

    top = np.hstack(blocks)

    if lags == 1:
        return top

    lower_left = np.eye(k * (lags - 1))
    lower_right = np.zeros((k * (lags - 1), k))

    bottom = np.hstack([lower_left, lower_right])

    return np.vstack([top, bottom])


def var_eigenvalues(
    beta_no_intercept: np.ndarray,
    k: int,
    lags: int,
) -> np.ndarray:
    C = companion_matrix(
        beta_no_intercept=beta_no_intercept,
        k=k,
        lags=lags,
    )

    return np.linalg.eigvals(C)


def is_var_stable(
    beta_no_intercept: np.ndarray,
    k: int,
    lags: int,
) -> bool:
    eigvals = var_eigenvalues(
        beta_no_intercept=beta_no_intercept,
        k=k,
        lags=lags,
    )

    return np.max(np.abs(eigvals)) < 1


def stability_summary(
    beta_no_intercept: np.ndarray,
    k: int,
    lags: int,
) -> dict:
    eigvals = var_eigenvalues(
        beta_no_intercept=beta_no_intercept,
        k=k,
        lags=lags,
    )

    return {
        "stable": bool(np.max(np.abs(eigvals)) < 1),
        "eigenvalues": eigvals,
        "max_modulus": float(np.max(np.abs(eigvals))),
    }