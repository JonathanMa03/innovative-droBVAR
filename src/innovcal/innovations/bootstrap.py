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


def sample_block_bootstrap_innovations(
    residuals: np.ndarray,
    n_paths: int,
    horizon: int,
    block_length: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    """Sample circular blocks of joint residual vectors."""
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2 or not np.isfinite(residuals).all():
        raise ValueError("residuals must be a finite two-dimensional array.")
    if not 1 <= block_length <= len(residuals):
        raise ValueError("block_length must lie between 1 and n_residuals.")

    rng = np.random.default_rng(seed)
    n_residuals, k = residuals.shape
    n_blocks = int(np.ceil(horizon / block_length))
    output = np.empty((n_paths, n_blocks * block_length, k))

    offsets = np.arange(block_length)
    for path in range(n_paths):
        starts = rng.integers(0, n_residuals, size=n_blocks)
        indices = (starts[:, None] + offsets[None, :]) % n_residuals
        output[path] = residuals[indices.reshape(-1)]
    return output[:, :horizon]


def fit_block_bootstrap_innovation_model(
    residuals: np.ndarray,
    block_length: int = 10,
) -> dict:
    residuals = np.asarray(residuals, dtype=float)
    if block_length < 1:
        raise ValueError("block_length must be positive.")
    return {
        "name": "block_bootstrap",
        "residuals": residuals,
        "block_length": block_length,
    }


def sample_from_block_bootstrap_model(
    model: dict,
    n_paths: int,
    horizon: int,
    seed: int | None = None,
) -> np.ndarray:
    return sample_block_bootstrap_innovations(
        residuals=model["residuals"],
        n_paths=n_paths,
        horizon=horizon,
        block_length=model["block_length"],
        seed=seed,
    )
