import numpy as np


def flatten_forecast_paths(
    paths: np.ndarray,
) -> np.ndarray:
    """
    Flatten forecast paths from (n_paths, horizon, k)
    to (n_paths * horizon, k).
    """
    paths = np.asarray(paths, dtype=float)

    if paths.ndim != 3:
        raise ValueError("paths must have shape (n_paths, horizon, k).")

    n_paths, horizon, k = paths.shape

    return paths.reshape(n_paths * horizon, k)


def path_mean(
    paths: np.ndarray,
) -> np.ndarray:
    """
    Mean forecast path across Monte Carlo samples.
    """
    return np.mean(paths, axis=0)


def path_std(
    paths: np.ndarray,
) -> np.ndarray:
    """
    Standard deviation across Monte Carlo samples.
    """
    return np.std(paths, axis=0, ddof=1)


def terminal_values(
    paths: np.ndarray,
) -> np.ndarray:
    """
    Extract terminal forecast values.

    Returns shape (n_paths, k).
    """
    return paths[:, -1, :]


def pathwise_max(
    paths: np.ndarray,
) -> np.ndarray:
    """
    Max value over horizon for each path and variable.

    Returns shape (n_paths, k).
    """
    return np.max(paths, axis=1)


def pathwise_min(
    paths: np.ndarray,
) -> np.ndarray:
    """
    Min value over horizon for each path and variable.

    Returns shape (n_paths, k).
    """
    return np.min(paths, axis=1)


def pathwise_range(
    paths: np.ndarray,
) -> np.ndarray:
    """
    Range over horizon for each path and variable.

    Returns shape (n_paths, k).
    """
    return pathwise_max(paths) - pathwise_min(paths)