import numpy as np

from innovcal.utils.linalg import make_psd


def fit_gaussian_innovation_model(
    residuals: np.ndarray,
) -> dict:
    residuals = np.asarray(residuals, dtype=float)

    mean = residuals.mean(axis=0)
    cov = np.cov(residuals.T)
    cov = make_psd(cov)

    return {
        "name": "gaussian",
        "mean": mean,
        "cov": cov,
    }


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


def sample_from_gaussian_model(
    model: dict,
    n_paths: int,
    horizon: int,
    seed: int | None = None,
) -> np.ndarray:
    return sample_gaussian_innovations(
        mean=model["mean"],
        cov=model["cov"],
        n_paths=n_paths,
        horizon=horizon,
        seed=seed,
    )