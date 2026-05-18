import numpy as np


def standardize_residuals(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize residuals columnwise.
    """
    mean = residuals.mean(axis=0)
    std = residuals.std(axis=0, ddof=1)

    standardized = (residuals - mean) / std

    return standardized, mean, std


def unstandardize_residuals(
    residuals_standardized: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return residuals_standardized * std + mean


def residual_summary(residuals: np.ndarray) -> dict:
    return {
        "mean": residuals.mean(axis=0),
        "std": residuals.std(axis=0, ddof=1),
        "cov": np.cov(residuals.T),
        "min": residuals.min(axis=0),
        "max": residuals.max(axis=0),
    }