import numpy as np


def make_psd(
    cov: np.ndarray,
    jitter: float = 1e-8,
    max_tries: int = 8,
) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)

    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance matrix contains NaN or Inf.")

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

    return cov