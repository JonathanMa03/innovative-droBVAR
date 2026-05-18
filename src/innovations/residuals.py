import numpy as np


def extract_residuals(fit_result: dict) -> np.ndarray:
    if "residuals" not in fit_result:
        raise KeyError("fit_result must contain a 'residuals' key.")

    return np.asarray(fit_result["residuals"], dtype=float)


def residual_mean_cov(
    residuals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.asarray(residuals, dtype=float)

    mean = residuals.mean(axis=0)
    cov = np.cov(residuals.T)

    return mean, cov


def standardize_residuals(
    residuals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residuals = np.asarray(residuals, dtype=float)

    mean = residuals.mean(axis=0)
    std = residuals.std(axis=0, ddof=1)
    std_safe = np.where(std == 0, 1.0, std)

    standardized = (residuals - mean) / std_safe

    return standardized, mean, std_safe


def unstandardize_residuals(
    residuals_standardized: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return residuals_standardized * std + mean


def check_residuals_finite(
    residuals: np.ndarray,
) -> bool:
    return bool(np.all(np.isfinite(residuals)))