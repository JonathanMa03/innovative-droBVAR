import numpy as np

from src.utils.linalg import make_psd


def sample_gaussian_innovations(
    mean: np.ndarray,
    cov: np.ndarray,
    n_paths: int,
    horizon: int,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    mean = np.asarray(mean, dtype=float)
    cov = make_psd(cov)

    shocks = rng.multivariate_normal(
        mean=mean,
        cov=cov,
        size=(n_paths, horizon),
        check_valid="raise",
    )

    return shocks