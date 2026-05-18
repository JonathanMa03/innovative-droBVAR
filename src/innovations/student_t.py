import numpy as np

from src.utils.linalg import make_psd


def fit_student_t_innovation_model(
    residuals: np.ndarray,
    df: float = 5.0,
) -> dict:
    if df <= 2:
        raise ValueError("df must be greater than 2 for finite covariance.")

    residuals = np.asarray(residuals, dtype=float)

    mean = residuals.mean(axis=0)
    empirical_cov = np.cov(residuals.T)

    # If z / sqrt(g / df) has covariance df / (df - 2) * scale,
    # then choose scale so the resulting Student-t covariance matches empirical covariance.
    scale = empirical_cov * (df - 2.0) / df
    scale = make_psd(scale)

    return {
        "name": "student_t",
        "mean": mean,
        "scale": scale,
        "df": df,
        "empirical_cov": empirical_cov,
    }


def sample_student_t_innovations(
    mean: np.ndarray,
    scale: np.ndarray,
    df: float,
    n_paths: int,
    horizon: int,
    seed: int | None = None,
) -> np.ndarray:
    if df <= 2:
        raise ValueError("df must be greater than 2 for finite covariance.")

    rng = np.random.default_rng(seed)

    mean = np.asarray(mean, dtype=float)
    scale = make_psd(scale)
    k = mean.shape[0]

    z = rng.multivariate_normal(
        mean=np.zeros(k),
        cov=scale,
        size=(n_paths, horizon),
        check_valid="raise",
    )

    g = rng.chisquare(
        df=df,
        size=(n_paths, horizon, 1),
    )

    return mean + z / np.sqrt(g / df)


def sample_from_student_t_model(
    model: dict,
    n_paths: int,
    horizon: int,
    seed: int | None = None,
) -> np.ndarray:
    return sample_student_t_innovations(
        mean=model["mean"],
        scale=model["scale"],
        df=model["df"],
        n_paths=n_paths,
        horizon=horizon,
        seed=seed,
    )