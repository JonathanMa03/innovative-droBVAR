import numpy as np


def interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Compute empirical interval coverage by variable.

    Parameters
    ----------
    y_true : np.ndarray
        Shape (n_obs, k) or (horizon, k).

    lower : np.ndarray
        Lower interval bound with same shape.

    upper : np.ndarray
        Upper interval bound with same shape.

    Returns
    -------
    coverage : np.ndarray
        Coverage rate for each variable, shape (k,).
    """
    inside = (y_true >= lower) & (y_true <= upper)
    return inside.mean(axis=0)


def average_interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Compute average interval width by variable.
    """
    return (upper - lower).mean(axis=0)


def rolling_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Scalar average coverage across all time points and variables.
    """
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())