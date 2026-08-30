import numpy as np


def fit_bootstrap_innovation_model(
    residuals: np.ndarray,
) -> dict:
    residuals = np.asarray(residuals, dtype=float)

    return {
        "name": "bootstrap",
        "residuals": residuals,
        "n_residuals": residuals.shape[0],
    }


def sample_bootstrap_innovations(
    residuals: np.ndarray,
    n_paths: int,
    horizon: int,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    residuals = np.asarray(residuals, dtype=float)
    n_resid = residuals.shape[0]

    idx = rng.integers(
        low=0,
        high=n_resid,
        size=(n_paths, horizon),
    )

    return residuals[idx]


def sample_from_bootstrap_model(
    model: dict,
    n_paths: int,
    horizon: int,
    seed: int | None = None,
) -> np.ndarray:
    return sample_bootstrap_innovations(
        residuals=model["residuals"],
        n_paths=n_paths,
        horizon=horizon,
        seed=seed,
    )