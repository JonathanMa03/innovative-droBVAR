import numpy as np


def energy_score(
    samples: np.ndarray,
    observation: np.ndarray,
) -> float:
    """
    Compute multivariate energy score for one observation.

    Parameters
    ----------
    samples : np.ndarray
        Forecast samples with shape (n_samples, k).

    observation : np.ndarray
        Observed value with shape (k,).

    Returns
    -------
    score : float
    """
    samples = np.asarray(samples)
    observation = np.asarray(observation)

    term_1 = np.mean(np.linalg.norm(samples - observation, axis=1))

    diffs = samples[:, None, :] - samples[None, :, :]
    term_2 = 0.5 * np.mean(np.linalg.norm(diffs, axis=2))

    return float(term_1 - term_2)


def mean_energy_score(
    paths: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Compute average energy score over forecast horizon.

    Parameters
    ----------
    paths : np.ndarray
        Shape (n_paths, horizon, k)

    y_true : np.ndarray
        Shape (horizon, k)

    Returns
    -------
    mean_score : float
    """
    horizon = y_true.shape[0]

    scores = [
        energy_score(paths[:, h, :], y_true[h])
        for h in range(horizon)
    ]

    return float(np.mean(scores))