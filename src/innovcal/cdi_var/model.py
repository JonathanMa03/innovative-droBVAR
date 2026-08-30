"""High-level Conditional Diffusion-Innovation VAR estimator."""

from dataclasses import dataclass, field

import numpy as np

from innovcal.api.innovations import fit_innovations, sample_innovations
from innovcal.di_var.residuals import rolling_var_residuals
from innovcal.forecasting.monte_carlo import simulate_forecast_paths
from innovcal.vector_ar.fit import fit_var_ols
from innovcal.cdi_var.calibration import (
    apply_forecast_calibration,
    fit_rolling_var_forecast_calibration,
)


@dataclass(frozen=True)
class CDIVARConfig:
    lags: int = 1
    context_lags: int = 5
    volatility_span: int = 60
    validation_fraction: float = 0.2
    calibration_fraction: float = 0.15
    calibration_horizons: tuple[int, ...] = (1, 5, 20)
    calibration_paths: int = 32
    calibration_shrinkage: float = 0.5
    calibration_bounds: tuple[float, float] = (0.8, 1.25)
    adaptive_calibration_window: int = 12
    diffusion_timesteps: int = 100
    diffusion_epochs: int = 300
    diffusion_lr: float = 5e-4
    diffusion_hidden_dim: int = 128
    diffusion_time_embedding_dim: int = 32
    diffusion_batch_size: int = 64
    early_stopping_patience: int = 40
    include_intercept: bool = True
    device: str = "cpu"
    seed: int = 123
    verbose: bool = False


@dataclass(frozen=True)
class CDIVARForecast:
    forecast_paths: np.ndarray
    innovation_paths: np.ndarray
    horizon: int
    n_paths: int
    metadata: dict[str, object] = field(default_factory=dict)

    def quantiles(self, levels=(0.05, 0.5, 0.95)):
        return {q: np.quantile(self.forecast_paths, q, axis=0) for q in levels}


class CDIVAR:
    """VAR mean plus conditional, volatility-aware, calibrated innovations."""

    def __init__(self, config: CDIVARConfig | None = None):
        self.config = config or CDIVARConfig()
        self.var_model_ = None
        self.innovation_model_ = None
        self.residuals_ = None
        self.fit_data_ = None

    def fit(self, y: np.ndarray, calibration_data: np.ndarray | None = None):
        y = self._validate(y, "y")
        cfg = self.config
        combined = y if calibration_data is None else np.vstack([
            y, self._validate(calibration_data, "calibration_data")
        ])
        initial_window = max(cfg.lags + 2, int(0.7 * len(y))) if calibration_data is None else len(y)
        self.residuals_ = rolling_var_residuals(
            combined, initial_window=initial_window, lags=cfg.lags,
            expanding=True, include_intercept=cfg.include_intercept,
        )
        self.var_model_ = fit_var_ols(
            combined, lags=cfg.lags, include_intercept=cfg.include_intercept
        )
        self.fit_data_ = combined.copy()
        self.innovation_model_ = fit_innovations(
            self.residuals_, method="cdi_var", context_lags=cfg.context_lags,
            volatility_span=cfg.volatility_span,
            validation_fraction=cfg.validation_fraction,
            calibration_fraction=cfg.calibration_fraction,
            calibration_horizons=cfg.calibration_horizons,
            calibration_paths=cfg.calibration_paths,
            calibration_shrinkage=cfg.calibration_shrinkage,
            calibration_bounds=cfg.calibration_bounds,
            adaptive_calibration_window=cfg.adaptive_calibration_window,
            timesteps=cfg.diffusion_timesteps, epochs=cfg.diffusion_epochs,
            lr=cfg.diffusion_lr, hidden_dim=cfg.diffusion_hidden_dim,
            time_embedding_dim=cfg.diffusion_time_embedding_dim,
            batch_size=cfg.diffusion_batch_size,
            early_stopping_patience=cfg.early_stopping_patience,
            device=cfg.device, seed=cfg.seed, verbose=cfg.verbose,
        )
        fit_rolling_var_forecast_calibration(
            self.innovation_model_, combined, self.residuals_,
            initial_window=initial_window, lags=cfg.lags,
            include_intercept=cfg.include_intercept,
            horizons=cfg.calibration_horizons,
            n_paths=cfg.calibration_paths, seed=cfg.seed,
            shrinkage=cfg.calibration_shrinkage,
            bounds=cfg.calibration_bounds,
        )
        return self

    def forecast(self, horizon: int, n_paths: int = 1000,
                 y_history: np.ndarray | None = None,
                 seed: int | None = None) -> CDIVARForecast:
        if self.var_model_ is None:
            raise RuntimeError("fit must be called before forecast.")
        history = self.fit_data_ if y_history is None else self._validate(y_history, "y_history")
        innovations = sample_innovations(
            self.innovation_model_, n_paths, horizon,
            self.config.seed if seed is None else seed,
        )
        paths = simulate_forecast_paths(
            history[-self.config.lags:], self.var_model_["beta"], innovations,
            self.config.lags, self.config.include_intercept,
        )
        paths = apply_forecast_calibration(
            paths, self.innovation_model_["calibration_multipliers"]
        )
        return CDIVARForecast(
            paths, innovations, horizon, n_paths,
            metadata={
                "model": "CDI-VAR", "seed": seed,
                "calibration_multipliers": self.innovation_model_["calibration_multipliers"],
            },
        )

    @staticmethod
    def _validate(value, name):
        value = np.asarray(value, dtype=float)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite two-dimensional array.")
        return value
