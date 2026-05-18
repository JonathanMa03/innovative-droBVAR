import numpy as np


def exponential_tilt_weights(
    losses: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    Exponential tilting weights.

    Larger eta puts more mass on high-loss samples.
    """
    losses = np.asarray(losses, dtype=float)

    shifted = losses - np.max(losses)

    raw = np.exp(eta * shifted)
    weights = raw / raw.sum()

    return weights


def weighted_expectation(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    return float(
        np.sum(weights * values)
    )


def effective_sample_size(
    weights: np.ndarray,
) -> float:
    weights = np.asarray(weights, dtype=float)

    return float(
        1.0 / np.sum(weights**2)
    )


def reweight_forecast_paths(
    losses: np.ndarray,
    eta: float,
) -> dict:
    """
    Produce robust/sample-stress weights from path losses.
    """
    weights = exponential_tilt_weights(
        losses=losses,
        eta=eta,
    )

    return {
        "weights": weights,
        "eta": eta,
        "effective_sample_size": effective_sample_size(weights),
        "weighted_loss": weighted_expectation(losses, weights),
        "unweighted_loss": float(np.mean(losses)),
    }