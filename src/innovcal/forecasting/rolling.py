"""Rolling-origin probabilistic VAR forecasting."""

from dataclasses import dataclass

import numpy as np

from innovcal.api.innovations import sample_innovations
from innovcal.forecasting.monte_carlo import simulate_forecast_paths
from innovcal.vector_ar.fit import fit_var_ols
from innovcal.cdi_var.calibration import (
    apply_forecast_calibration,
    fit_adaptive_multipliers,
)


@dataclass(frozen=True)
class RollingForecasts:
    """Forecast tensor with shape (origin, path, horizon, asset)."""

    forecasts: dict[str, np.ndarray]
    observations: np.ndarray
    origins: np.ndarray
    horizon: int
    calibration_history: dict[str, np.ndarray] | None = None


def generate_rolling_var_forecasts(
    returns: np.ndarray,
    test_start: int,
    innovation_models: dict[str, dict],
    horizon: int = 20,
    n_paths: int = 250,
    lags: int = 1,
    origin_step: int = 20,
    window_length: int | None = None,
    seed: int = 123,
) -> RollingForecasts:
    """Generate expanding- or rolling-window forecasts at many origins."""
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or not np.isfinite(returns).all():
        raise ValueError("returns must be a finite two-dimensional array.")
    if not lags < test_start < len(returns):
        raise ValueError("test_start must leave fitting and test observations.")
    if horizon < 1 or origin_step < 1 or n_paths < 1:
        raise ValueError("horizon, origin_step, and n_paths must be positive.")

    origins = np.arange(test_start, len(returns) - horizon + 1, origin_step)
    if not len(origins):
        raise ValueError("No complete rolling forecast origins are available.")
    observations = np.stack([returns[o : o + horizon] for o in origins])
    stores = {name: [] for name in innovation_models}
    calibration_records = {name: [] for name in innovation_models}
    calibration_used = {name: [] for name in innovation_models}

    for origin_index, origin in enumerate(origins):
        start = 0 if window_length is None else max(0, origin - window_length)
        fit_sample = returns[start:origin]
        fitted = fit_var_ols(fit_sample, lags=lags, include_intercept=True)
        history = returns[origin - lags : origin]

        for method_index, (name, model) in enumerate(innovation_models.items()):
            sample_seed = seed + 10_000 * origin_index + method_index
            if "sample_with_history_fn" in model:
                innovations = model["sample_with_history_fn"](
                    fitted["residuals"], n_paths, horizon, sample_seed
                )
            else:
                innovations = sample_innovations(
                    model,
                    n_paths=n_paths,
                    horizon=horizon,
                    seed=sample_seed,
                )
            raw_paths = simulate_forecast_paths(
                y_history=history,
                beta=fitted["beta"],
                innovation_paths=innovations,
                lags=lags,
                include_intercept=True,
            )
            paths = raw_paths
            if model.get("method") == "cdi_var":
                calibration_horizons = tuple(
                    h for h in model.get("calibration_horizons", (1, 5, 20))
                    if h <= horizon
                )
                multipliers = fit_adaptive_multipliers(
                    calibration_records[name],
                    calibration_horizons,
                    fallback=model["calibration_multipliers"][:horizon],
                    window=int(model.get("adaptive_calibration_window", 12)),
                    shrinkage=float(model.get("calibration_shrinkage", 0.5)),
                    bounds=tuple(model.get("calibration_bounds", (0.8, 1.25))),
                    current_origin=int(origin),
                )
                paths = apply_forecast_calibration(
                    raw_paths, multipliers
                )
                calibration_used[name].append(multipliers)
                calibration_records[name].append({
                    "draws": {h: raw_paths[:, h - 1].copy() for h in calibration_horizons},
                    "truths": {h: returns[origin + h - 1].copy() for h in calibration_horizons},
                    "available_at": {h: int(origin + h) for h in calibration_horizons},
                })
            if not np.isfinite(paths).all():
                raise FloatingPointError(f"Non-finite rolling forecasts for {name}.")
            stores[name].append(paths)

    return RollingForecasts(
        forecasts={name: np.stack(paths) for name, paths in stores.items()},
        observations=observations,
        origins=origins,
        horizon=horizon,
        calibration_history={
            name: np.stack(values) for name, values in calibration_used.items() if values
        },
    )
