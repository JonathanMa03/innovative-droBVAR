import numpy as np


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def make_psd(
    cov: np.ndarray,
    jitter: float = 1e-8,
    max_tries: int = 8,
) -> np.ndarray:
    """
    Make a covariance matrix numerically positive semidefinite.
    """
    cov = symmetrize(cov)

    if not np.all(np.isfinite(cov)):
        raise ValueError("Matrix contains NaN or Inf.")

    for i in range(max_tries):
        cov_try = cov + (10**i) * jitter * np.eye(cov.shape[0])

        try:
            np.linalg.cholesky(cov_try)
            return cov_try
        except np.linalg.LinAlgError:
            continue

    eigvals = np.linalg.eigvalsh(cov)
    min_eig = eigvals.min()

    if min_eig < 0:
        cov = cov + (-min_eig + jitter) * np.eye(cov.shape[0])

    return symmetrize(cov)


def condition_number(matrix: np.ndarray) -> float:
    return float(np.linalg.cond(matrix))


def eigen_summary(matrix: np.ndarray) -> dict:
    eigvals = np.linalg.eigvals(matrix)

    return {
        "eigvals": eigvals,
        "max_abs_eig": float(np.max(np.abs(eigvals))),
        "min_real_eig": float(np.min(np.real(eigvals))),
        "max_real_eig": float(np.max(np.real(eigvals))),
    }