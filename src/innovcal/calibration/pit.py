import numpy as np


def pit_values_univariate(
    samples: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """
    Compute PIT values for one variable over forecast horizon.

    Parameters
    ----------
    samples:
        Shape (n_samples, horizon)

    observations:
        Shape (horizon,)

    Returns
    -------
    pit:
        Shape (horizon,)
    """
    samples = np.asarray(samples, dtype=float)
    observations = np.asarray(observations, dtype=float)

    return np.mean(samples <= observations[None, :], axis=0)


def pit_values_multivariate_marginal(
    forecast_paths: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """
    Compute marginal PIT values for each variable.

    Parameters
    ----------
    forecast_paths:
        Shape (n_paths, horizon, k)

    y_true:
        Shape (horizon, k)

    Returns
    -------
    pit:
        Shape (horizon, k)
    """
    forecast_paths = np.asarray(forecast_paths, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    n_paths, horizon, k = forecast_paths.shape

    pit = np.zeros((horizon, k))

    for j in range(k):
        pit[:, j] = pit_values_univariate(
            samples=forecast_paths[:, :, j],
            observations=y_true[:, j],
        )

    return pit


def pit_histogram(
    pit_values: np.ndarray,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return PIT histogram counts and bin edges.
    """
    pit_values = np.asarray(pit_values, dtype=float).ravel()

    counts, edges = np.histogram(
        pit_values,
        bins=bins,
        range=(0.0, 1.0),
    )

    return counts, edges


def pit_deviation_from_uniform(
    pit_values: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Simple scalar PIT nonuniformity diagnostic.

    Returns mean absolute deviation of normalized PIT bin counts
    from the uniform bin probability.
    """
    counts, _ = pit_histogram(pit_values, bins=bins)

    probs = counts / counts.sum()
    uniform = np.ones(bins) / bins

    return float(np.mean(np.abs(probs - uniform)))