"""High-level Diffusion-Innovation VAR workflow.

This module keeps orchestration out of notebooks while delegating numerical
operations to the existing VAR, innovation, and forecasting modules.
"""

from dataclasses import dataclass, field

import numpy as np

from innovcal.api.innovations import fit_innovations, sample_innovations
from innovcal.di_var.residuals import rolling_var_residuals
from innovcal.forecasting.monte_carlo import simulate_forecast_paths
from innovcal.vector_ar.fit import fit_var_ols


@dataclass(frozen=True)
class DIVARConfig:
    """Configuration for fitting and sampling a DI-VAR model."""

    lags: int = 1
    include_intercept: bool = True
    residual_window: int | None = None
    expanding_window: bool = True
    refit_every: int = 1
    diffusion_timesteps: int = 200
    diffusion_epochs: int = 1500
    diffusion_lr: float = 5e-4
    diffusion_hidden_dim: int = 256
    diffusion_time_embedding_dim: int = 64
    diffusion_batch_size: int = 64
    device: str = "cpu"
    verbose: bool = False


@dataclass(frozen=True)
class DIVARForecast:
    """Probabilistic forecast paths and the innovations that generated them."""

    forecast_paths: np.ndarray
    innovation_paths: np.ndarray
    horizon: int
    n_paths: int
    metadata: dict[str, object] = field(default_factory=dict)

    def quantiles(
        self,
        levels: tuple[float, ...] = (0.05, 0.5, 0.95),
    ) -> dict[float, np.ndarray]:
        return {
            level: np.quantile(self.forecast_paths, level, axis=0)
            for level in levels
        }


class DIVAR:
    """VAR conditional mean with a diffusion model for joint innovations."""

    def __init__(self, config: DIVARConfig | None = None) -> None:
        self.config = config or DIVARConfig()
        self.var_model_: dict | None = None
        self.innovation_model_: dict | None = None
        self.residuals_: np.ndarray | None = None
        self.fit_data_: np.ndarray | None = None

    def fit(
        self,
        y: np.ndarray,
        calibration_data: np.ndarray | None = None,
    ) -> "DIVAR":
        """Fit VAR dynamics and a diffusion law for joint forecast errors.

        When ``calibration_data`` is supplied, the VAR is fitted on ``y`` and
        rolling residuals are generated over the combined chronological sample,
        beginning at the boundary between the two arrays. Otherwise a trailing
        portion of ``y`` is used to construct rolling residuals.
        """
        y = self._validate_series(y, "y")
        cfg = self.config

        self.var_model_ = fit_var_ols(
            y,
            lags=cfg.lags,
            include_intercept=cfg.include_intercept,
        )
        self.fit_data_ = y.copy()

        if calibration_data is not None:
            calibration_data = self._validate_series(
                calibration_data,
                "calibration_data",
            )
            if calibration_data.shape[1] != y.shape[1]:
                raise ValueError("y and calibration_data must have equal width.")
            combined = np.vstack([y, calibration_data])
            residuals = rolling_var_residuals(
                combined,
                initial_window=len(y),
                lags=cfg.lags,
                expanding=cfg.expanding_window,
                refit_every=cfg.refit_every,
                include_intercept=cfg.include_intercept,
            )
        else:
            initial_window = cfg.residual_window
            if initial_window is None:
                initial_window = max(cfg.lags + 2, int(0.7 * len(y)))
            residuals = rolling_var_residuals(
                y,
                initial_window=initial_window,
                lags=cfg.lags,
                expanding=cfg.expanding_window,
                refit_every=cfg.refit_every,
                include_intercept=cfg.include_intercept,
            )

        self.residuals_ = residuals
        self.innovation_model_ = fit_innovations(
            residuals=residuals,
            method="diffusion",
            timesteps=cfg.diffusion_timesteps,
            epochs=cfg.diffusion_epochs,
            lr=cfg.diffusion_lr,
            hidden_dim=cfg.diffusion_hidden_dim,
            time_embedding_dim=cfg.diffusion_time_embedding_dim,
            batch_size=cfg.diffusion_batch_size,
            device=cfg.device,
            verbose=cfg.verbose,
        )
        return self

    def forecast(
        self,
        horizon: int,
        n_paths: int = 1000,
        y_history: np.ndarray | None = None,
        seed: int | None = None,
    ) -> DIVARForecast:
        """Generate recursive probabilistic forecasts from diffusion shocks."""
        if self.var_model_ is None or self.innovation_model_ is None:
            raise RuntimeError("fit must be called before forecast.")
        if horizon < 1 or n_paths < 1:
            raise ValueError("horizon and n_paths must be positive.")

        history = self.fit_data_ if y_history is None else y_history
        history = self._validate_series(history, "y_history")
        if len(history) < self.config.lags:
            raise ValueError("y_history is shorter than the configured lag order.")

        innovations = sample_innovations(
            self.innovation_model_,
            n_paths=n_paths,
            horizon=horizon,
            seed=seed,
        )
        paths = simulate_forecast_paths(
            y_history=history[-self.config.lags :],
            beta=self.var_model_["beta"],
            innovation_paths=innovations,
            lags=self.config.lags,
            include_intercept=self.config.include_intercept,
        )
        return DIVARForecast(
            forecast_paths=paths,
            innovation_paths=innovations,
            horizon=horizon,
            n_paths=n_paths,
            metadata={"model": "DI-VAR", "seed": seed},
        )

    @staticmethod
    def _validate_series(value: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape (n_observations, n_series).")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values.")
        return value
