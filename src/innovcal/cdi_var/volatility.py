"""Causal EWMA state and conditional-diffusion design construction."""

from __future__ import annotations

import numpy as np


def ewma_decay(span: int) -> float:
    if span < 2:
        raise ValueError("span must be at least 2.")
    return 1.0 - 2.0 / (span + 1.0)


def causal_standardize(residuals: np.ndarray, span: int = 60,
                       floor: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize each residual using variance known before it arrived."""
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2 or len(residuals) < 3:
        raise ValueError("residuals must have shape (n>=3, k).")
    decay = ewma_decay(span)
    variance = np.maximum(residuals[0] ** 2, floor ** 2)
    scales = []
    standardized = []
    for residual in residuals:
        scale = np.sqrt(np.maximum(variance, floor ** 2))
        scales.append(scale)
        standardized.append(residual / scale)
        variance = decay * variance + (1.0 - decay) * residual ** 2
    return np.asarray(standardized), np.asarray(scales), variance


def make_conditional_design(standardized: np.ndarray, scales: np.ndarray,
                            context_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Create targets and contexts of lagged shocks plus normalized log scale."""
    standardized = np.asarray(standardized, dtype=float)
    log_scale = np.log(np.maximum(scales, 1e-8))
    log_mean = log_scale.mean(axis=0)
    log_std = np.maximum(log_scale.std(axis=0), 1e-8)
    normalized_scale = (log_scale - log_mean) / log_std
    contexts, targets = [], []
    for t in range(context_lags, len(standardized)):
        contexts.append(np.concatenate([
            standardized[t - context_lags:t].reshape(-1),
            normalized_scale[t],
        ]))
        targets.append(standardized[t])
    return np.asarray(targets), np.asarray(contexts)
