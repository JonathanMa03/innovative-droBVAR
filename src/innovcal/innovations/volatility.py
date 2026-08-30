"""Simple volatility-standardized innovation models."""

import numpy as np
import pandas as pd

from innovcal.innovations.bootstrap import sample_bootstrap_innovations


def ewma_volatility_scales(
    residuals: np.ndarray,
    span: int = 60,
    minimum_scale: float = 1e-8,
) -> np.ndarray:
    """Estimate causal exponentially weighted marginal residual scales."""
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2 or len(residuals) < 2:
        raise ValueError("residuals must have shape (n>=2, k).")
    if span < 2:
        raise ValueError("span must be at least 2.")
    filtered_variance = (
        pd.DataFrame(residuals)
        .pow(2)
        .ewm(span=span, adjust=False)
        .mean()
        .to_numpy()
    )
    # Standardize each residual only with information available beforehand.
    causal_variance = np.vstack([filtered_variance[0], filtered_variance[:-1]])
    return np.maximum(np.sqrt(causal_variance), minimum_scale)


def fit_volatility_bootstrap_model(
    residuals: np.ndarray,
    span: int = 60,
) -> dict:
    """Fit bootstrap shocks after causal EWMA volatility standardization."""
    residuals = np.asarray(residuals, dtype=float)
    scales = ewma_volatility_scales(residuals, span=span)
    standardized = residuals[1:] / scales[1:]
    standardized = standardized - standardized.mean(axis=0, keepdims=True)
    forecast_variance = (
        pd.DataFrame(residuals)
        .pow(2)
        .ewm(span=span, adjust=False)
        .mean()
        .to_numpy()[-1]
    )
    return {
        "name": "volatility_bootstrap",
        "standardized_residuals": standardized,
        "scales": scales,
        "forecast_scale": np.maximum(np.sqrt(forecast_variance), 1e-8),
        "span": span,
    }


def sample_from_volatility_bootstrap_model(
    model: dict,
    n_paths: int,
    horizon: int,
    seed: int | None = None,
) -> np.ndarray:
    standardized = sample_bootstrap_innovations(
        residuals=model["standardized_residuals"],
        n_paths=n_paths,
        horizon=horizon,
        seed=seed,
    )
    return standardized * np.asarray(model["forecast_scale"])[None, None, :]
