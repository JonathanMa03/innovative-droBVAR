import numpy as np


def empirical_wasserstein_1d(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Empirical 1D Wasserstein distance between two samples.
    """
    x = np.sort(np.asarray(x, dtype=float).ravel())
    y = np.sort(np.asarray(y, dtype=float).ravel())

    n = min(len(x), len(y))

    return float(
        np.mean(
            np.abs(x[:n] - y[:n])
        )
    )


def marginal_wasserstein_distance(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Compute marginal 1D Wasserstein distance by variable.

    Parameters
    ----------
    x, y:
        Shape (n_samples, k)

    Returns
    -------
    distances:
        Shape (k,)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same number of variables.")

    k = x.shape[1]

    return np.asarray([
        empirical_wasserstein_1d(
            x[:, j],
            y[:, j],
        )
        for j in range(k)
    ])


def average_marginal_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    return float(
        marginal_wasserstein_distance(x, y).mean()
    )


def wasserstein_radius_grid(
    max_radius: float,
    n_grid: int = 10,
    include_zero: bool = True,
) -> np.ndarray:
    if n_grid < 1:
        raise ValueError("n_grid must be at least 1.")

    if include_zero:
        return np.linspace(0.0, max_radius, n_grid)

    return np.linspace(max_radius / n_grid, max_radius, n_grid)