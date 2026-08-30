import numpy as np


def crps_ensemble_univariate(
    samples: np.ndarray,
    observation: float,
) -> float:
    """
    Ensemble CRPS for one univariate forecast distribution.

    CRPS = E|X - y| - 0.5 E|X - X'|
    """
    samples = np.asarray(samples, dtype=float)

    term_1 = np.mean(np.abs(samples - observation))

    diffs = np.abs(samples[:, None] - samples[None, :])
    term_2 = 0.5 * np.mean(diffs)

    return float(term_1 - term_2)


def mean_crps_marginal(
    forecast_paths: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Average marginal CRPS across horizon and variables.

    Parameters
    ----------
    forecast_paths:
        Shape (n_paths, horizon, k)

    y_true:
        Shape (horizon, k)
    """
    forecast_paths = np.asarray(forecast_paths, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    _, horizon, k = forecast_paths.shape

    scores = []

    for h in range(horizon):
        for j in range(k):
            scores.append(
                crps_ensemble_univariate(
                    samples=forecast_paths[:, h, j],
                    observation=y_true[h, j],
                )
            )

    return float(np.mean(scores))


def crps_by_series(
    forecast_paths: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """
    Average CRPS for each variable.
    """
    forecast_paths = np.asarray(forecast_paths, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    _, horizon, k = forecast_paths.shape

    out = np.zeros(k)

    for j in range(k):
        scores_j = []

        for h in range(horizon):
            scores_j.append(
                crps_ensemble_univariate(
                    samples=forecast_paths[:, h, j],
                    observation=y_true[h, j],
                )
            )

        out[j] = np.mean(scores_j)

    return out