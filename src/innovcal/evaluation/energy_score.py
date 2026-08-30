import numpy as np


def energy_score(
    samples: np.ndarray,
    observation: np.ndarray,
) -> float:
    """
    Multivariate energy score for one forecast distribution and one observation.

    Parameters
    ----------
    samples:
        Shape (n_samples, k)

    observation:
        Shape (k,)
    """
    samples = np.asarray(samples, dtype=float)
    observation = np.asarray(observation, dtype=float)

    term_1 = np.mean(
        np.linalg.norm(samples - observation, axis=1)
    )

    diffs = samples[:, None, :] - samples[None, :, :]

    term_2 = 0.5 * np.mean(
        np.linalg.norm(diffs, axis=2)
    )

    return float(term_1 - term_2)


def mean_energy_score(
    forecast_paths: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Average energy score across forecast horizon.

    Parameters
    ----------
    forecast_paths:
        Shape (n_paths, horizon, k)

    y_true:
        Shape (horizon, k)
    """
    forecast_paths = np.asarray(forecast_paths, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    horizon = y_true.shape[0]

    scores = [
        energy_score(
            samples=forecast_paths[:, h, :],
            observation=y_true[h],
        )
        for h in range(horizon)
    ]

    return float(np.mean(scores))