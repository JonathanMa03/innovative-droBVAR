import numpy as np


def interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Compute empirical prediction interval coverage by variable.

    Parameters
    ----------
    y_true:
        Shape (horizon, k)

    lower:
        Shape (horizon, k)

    upper:
        Shape (horizon, k)

    Returns
    -------
    coverage:
        Shape (k,)
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    inside = (y_true >= lower) & (y_true <= upper)

    return inside.mean(axis=0)


def average_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    return float(
        interval_coverage(y_true, lower, upper).mean()
    )


def coverage_error(
    empirical_coverage: float,
    nominal_coverage: float,
) -> float:
    return float(empirical_coverage - nominal_coverage)


def absolute_coverage_error(
    empirical_coverage: float,
    nominal_coverage: float,
) -> float:
    return float(abs(empirical_coverage - nominal_coverage))


def coverage_by_horizon(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Compute coverage at each horizon averaged over variables.

    Returns
    -------
    coverage:
        Shape (horizon,)
    """
    inside = (y_true >= lower) & (y_true <= upper)
    return inside.mean(axis=1)