"""Fit and sample volatility-aware conditional diffusion innovations."""

from __future__ import annotations

import numpy as np
import torch

from innovcal.cdi_var.diffusion import (
    ConditionalResidualDenoiser,
    sample_conditional_ddpm,
    train_conditional_diffusion,
)
from innovcal.cdi_var.volatility import (
    causal_standardize,
    ewma_decay,
    make_conditional_design,
)
from innovcal.diffusion.schedules import make_ddpm_schedule


def _context(history: np.ndarray, variance: np.ndarray, log_mean: np.ndarray,
             log_std: np.ndarray) -> np.ndarray:
    scale_state = (np.log(np.sqrt(np.maximum(variance, 1e-16))) - log_mean) / log_std
    return np.concatenate([history.reshape(history.shape[0], -1), scale_state], axis=1)


def _sample_recursive(model: torch.nn.Module, schedule: dict, n_paths: int,
                      horizon: int, history: np.ndarray, variance: np.ndarray,
                      log_mean: np.ndarray, log_std: np.ndarray, decay: float,
                      device: str | torch.device, seed: int | None,
                      multipliers: np.ndarray | None = None) -> np.ndarray:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    histories = np.repeat(history[None, :, :], n_paths, axis=0)
    variances = np.repeat(variance[None, :], n_paths, axis=0)
    output = []
    for lead in range(horizon):
        contexts = _context(histories, variances, log_mean, log_std)
        standardized = sample_conditional_ddpm(model, contexts, schedule, device)
        scale = np.sqrt(np.maximum(variances, 1e-16))
        latent_shocks = standardized * scale
        emitted_shocks = latent_shocks
        if multipliers is not None:
            multiplier = multipliers[min(lead, len(multipliers) - 1)]
            center = np.median(latent_shocks, axis=0, keepdims=True)
            emitted_shocks = center + multiplier * (latent_shocks - center)
        output.append(emitted_shocks)
        # Calibration is an output-layer adjustment. The observable state
        # evolves under the uncalibrated conditional process so an interval
        # multiplier cannot recursively inflate all later volatility states.
        variances = decay * variances + (1.0 - decay) * latent_shocks ** 2
        histories = np.concatenate([histories[:, 1:], standardized[:, None, :]], axis=1)
    return np.stack(output, axis=1)


def fit_cdi_innovation_model(
    residuals: np.ndarray,
    context_lags: int = 5,
    volatility_span: int = 60,
    validation_fraction: float = 0.2,
    calibration_fraction: float = 0.15,
    calibration_horizons: tuple[int, ...] = (1, 5, 20),
    calibration_paths: int = 32,
    calibration_shrinkage: float = 0.5,
    calibration_bounds: tuple[float, float] = (0.8, 1.25),
    adaptive_calibration_window: int = 12,
    timesteps: int = 100,
    epochs: int = 300,
    lr: float = 5e-4,
    hidden_dim: int = 128,
    time_embedding_dim: int = 32,
    batch_size: int = 64,
    early_stopping_patience: int = 40,
    device: str | torch.device = "cpu",
    seed: int = 123,
    verbose: bool = False,
) -> dict:
    """Fit CDI innovations and calibrate lead-specific dispersion."""
    residuals = np.asarray(residuals, dtype=float)
    standardized, scales, final_variance = causal_standardize(residuals, volatility_span)
    targets, contexts = make_conditional_design(standardized, scales, context_lags)
    if validation_fraction <= 0 or calibration_fraction <= 0 or validation_fraction + calibration_fraction >= 0.5:
        raise ValueError("validation and calibration fractions must be positive and sum below 0.5.")
    train_end = int(len(targets) * (1.0 - validation_fraction - calibration_fraction))
    validation_end = int(len(targets) * (1.0 - calibration_fraction))
    if train_end < max(10, context_lags + 1) or validation_end >= len(targets):
        raise ValueError("Not enough conditional residual cases for validation and calibration.")
    log_scales = np.log(np.maximum(scales[: train_end + context_lags], 1e-8))
    log_mean = log_scales.mean(axis=0)
    log_std = np.maximum(log_scales.std(axis=0), 1e-8)
    # Rebuild contexts with training-only scale normalization.
    normalized = (np.log(np.maximum(scales, 1e-8)) - log_mean) / log_std
    contexts = np.asarray([
        np.concatenate([standardized[t-context_lags:t].reshape(-1), normalized[t]])
        for t in range(context_lags, len(standardized))
    ])

    device = torch.device(device)
    schedule = make_ddpm_schedule(timesteps, 1e-4, 2e-2, "linear", device)
    model = ConditionalResidualDenoiser(
        residuals.shape[1], contexts.shape[1], hidden_dim, time_embedding_dim
    )
    history = train_conditional_diffusion(
        model, targets[:train_end], contexts[:train_end], schedule,
        validation_targets=targets[train_end:validation_end],
        validation_contexts=contexts[train_end:validation_end],
        epochs=epochs, batch_size=batch_size, lr=lr,
        patience=early_stopping_patience, device=device, seed=seed, verbose=verbose,
    )

    # The final chronological block is reserved for full VAR forecast
    # calibration. Multipliers remain neutral until that separate step, which
    # requires the original time series and cannot be fitted from residuals
    # alone.
    max_h = min(max(calibration_horizons), len(targets) - validation_end)
    multipliers = np.ones(max_h)

    decay = ewma_decay(volatility_span)
    initial_history = standardized[-context_lags:]
    def sample_fn(n_paths: int, horizon: int, seed: int | None = None) -> np.ndarray:
        return _sample_recursive(
            model, schedule, n_paths, horizon, initial_history, final_variance,
            log_mean, log_std, decay, device, seed, None,
        )

    def sample_with_history_fn(residual_history: np.ndarray, n_paths: int,
                               horizon: int, seed: int | None = None) -> np.ndarray:
        state_z, _, state_variance = causal_standardize(
            residual_history, volatility_span
        )
        if len(state_z) < context_lags:
            raise ValueError("residual_history is shorter than context_lags.")
        return _sample_recursive(
            model, schedule, n_paths, horizon, state_z[-context_lags:],
            state_variance, log_mean, log_std, decay, device, seed, None,
        )

    return {
        "name": "cdi_var", "method": "cdi_var", "model": model,
        "schedule": schedule, "history": history, "sample_fn": sample_fn,
        "sample_with_history_fn": sample_with_history_fn,
        "context_lags": context_lags, "volatility_span": volatility_span,
        "validation_fraction": validation_fraction,
        "calibration_fraction": calibration_fraction,
        "calibration_multipliers": multipliers,
        "calibration_horizons": calibration_horizons,
        "calibration_start_residual": validation_end + context_lags,
        "calibration_paths": calibration_paths,
        "calibration_shrinkage": calibration_shrinkage,
        "calibration_bounds": calibration_bounds,
        "adaptive_calibration_window": adaptive_calibration_window,
        "log_scale_mean": log_mean, "log_scale_std": log_std,
    }
