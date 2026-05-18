import numpy as np

from src.utils.linalg import make_psd


def sample_student_t_innovations(
    mean: np.ndarray,
    cov: np.ndarray,
    df: float,
    n_paths: int,
    horizon: int,
    seed: int | None = None,
) -> np.ndarray:
    if df <= 2:
        raise ValueError("df must be greater than 2 for finite covariance.")

    rng = np.random.default_rng(seed)

    mean = np.asarray(mean, dtype=float)
    cov = make_psd(cov)
    k = mean.shape[0]

    z = rng.multivariate_normal(
        mean=np.zeros(k),
        cov=cov,
        size=(n_paths, horizon),
        check_valid="raise",
    )

    g = rng.chisquare(df=df, size=(n_paths, horizon, 1))

    return mean + z / np.sqrt(g / df)