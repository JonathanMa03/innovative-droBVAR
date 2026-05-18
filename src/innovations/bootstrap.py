import numpy as np


def sample_bootstrap_innovations(
    residuals: np.ndarray,
    n_paths: int,
    horizon: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample innovations by bootstrapping residual vectors.

    Returns
    -------
    shocks : np.ndarray
        Shape (n_paths, horizon, k).
    """
    rng = np.random.default_rng(seed)

    n_resid, k = residuals.shape
    idx = rng.integers(0, n_resid, size=(n_paths, horizon))

    shocks = residuals[idx]

    return shocks.reshape(n_paths, horizon, k)