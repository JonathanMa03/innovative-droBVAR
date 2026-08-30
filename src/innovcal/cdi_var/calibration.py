"""Transparent scale calibration for conditional innovation intervals."""

import numpy as np

from innovcal.forecasting.monte_carlo import simulate_forecast_paths
from innovcal.vector_ar.fit import fit_var_ols


def fit_scale_multiplier(samples: np.ndarray, truth: np.ndarray,
                         nominal_coverage: float = 0.9,
                         grid: np.ndarray | None = None) -> float:
    """Choose the smallest-error symmetric scale multiplier on held-out cases.

    Parameters
    ----------
    samples : array, shape (case, draw, asset)
    truth : array, shape (case, asset)
    """
    if grid is None:
        grid = np.linspace(0.5, 2.0, 151)
    samples = np.asarray(samples, dtype=float)
    truth = np.asarray(truth, dtype=float)
    median = np.median(samples, axis=1, keepdims=True)
    centered = samples - median
    alpha = (1.0 - nominal_coverage) / 2.0
    best = None
    for multiplier in grid:
        adjusted = median + multiplier * centered
        lower = np.quantile(adjusted, alpha, axis=1)
        upper = np.quantile(adjusted, 1.0 - alpha, axis=1)
        coverage = np.mean((truth >= lower) & (truth <= upper))
        candidate = (
            abs(coverage - nominal_coverage),
            abs(float(multiplier) - 1.0),
            float(multiplier),
        )
        if best is None or candidate < best:
            best = candidate
    return best[2]


def apply_scale_multiplier(samples: np.ndarray, multiplier: float) -> np.ndarray:
    """Scale draws around their median without changing the point forecast."""
    median = np.median(samples, axis=0, keepdims=True)
    return median + multiplier * (samples - median)


def apply_forecast_calibration(
    forecast_paths: np.ndarray,
    multipliers: np.ndarray,
) -> np.ndarray:
    """Scale full VAR forecast deviations around the lead-specific median."""
    paths = np.asarray(forecast_paths, dtype=float)
    multipliers = np.asarray(multipliers, dtype=float)
    if paths.ndim != 3 or not len(multipliers):
        raise ValueError("paths must be (path, horizon, asset) and multipliers non-empty.")
    center = np.median(paths, axis=0, keepdims=True)
    lead_scale = np.asarray([
        multipliers[min(lead, len(multipliers) - 1)]
        for lead in range(paths.shape[1])
    ])[None, :, None]
    return center + lead_scale * (paths - center)


def regularize_anchor_multipliers(
    anchors: dict[int, float],
    shrinkage: float = 0.5,
    bounds: tuple[float, float] = (0.8, 1.25),
) -> dict[int, float]:
    """Shrink raw calibration estimates toward one and impose safe bounds."""
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1].")
    lower, upper = bounds
    if not 0 < lower <= 1.0 <= upper:
        raise ValueError("bounds must be positive and contain one.")
    return {
        int(horizon): float(np.clip(1.0 + shrinkage * (value - 1.0), lower, upper))
        for horizon, value in anchors.items()
    }


def expand_anchor_multipliers(
    anchors: dict[int, float],
    max_horizon: int,
) -> np.ndarray:
    """Carry each calibration anchor forward until the next anchor horizon."""
    ordered = sorted(anchors)
    if not ordered:
        return np.ones(max_horizon)
    output = np.ones(max_horizon)
    active = anchors[ordered[0]]
    for lead in range(1, max_horizon + 1):
        eligible = [h for h in ordered if h <= lead]
        if eligible:
            active = anchors[max(eligible)]
        output[lead - 1] = active
    return output


def fit_adaptive_multipliers(
    forecast_records: list[dict],
    horizons: tuple[int, ...],
    fallback: np.ndarray,
    window: int = 12,
    shrinkage: float = 0.5,
    bounds: tuple[float, float] = (0.8, 1.25),
    current_origin: int | None = None,
) -> np.ndarray:
    """Fit multipliers using only previously realized rolling forecasts."""
    if not forecast_records:
        return np.asarray(fallback, dtype=float).copy()
    anchors = {}
    for horizon in horizons:
        usable = [
            record for record in forecast_records
            if horizon in record["draws"]
            and (
                current_origin is None
                or record.get("available_at", {}).get(horizon, -1) <= current_origin
            )
        ]
        usable = usable[-window:]
        if not usable:
            continue
        anchors[horizon] = fit_scale_multiplier(
            np.stack([record["draws"][horizon] for record in usable]),
            np.stack([record["truths"][horizon] for record in usable]),
        )
    if not anchors:
        return np.asarray(fallback, dtype=float).copy()
    regularized = regularize_anchor_multipliers(anchors, shrinkage, bounds)
    return expand_anchor_multipliers(regularized, len(fallback))


def fit_rolling_var_forecast_calibration(
    innovation_model: dict,
    series: np.ndarray,
    residuals: np.ndarray,
    initial_window: int,
    lags: int = 1,
    include_intercept: bool = True,
    horizons: tuple[int, ...] = (1, 5, 20),
    n_paths: int | None = None,
    max_origins: int = 12,
    seed: int = 123,
    shrinkage: float = 0.5,
    bounds: tuple[float, float] = (0.8, 1.25),
) -> np.ndarray:
    """Fit lead multipliers against genuine rolling VAR forecast outcomes.

    ``residuals[j]`` must be the one-step error for ``series[initial_window+j]``.
    Only origins in the innovation model's reserved calibration block are used.
    The model dictionary is updated in place and the multipliers are returned.
    """
    series = np.asarray(series, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if innovation_model.get("method") != "cdi_var":
        raise ValueError("rolling VAR calibration requires a CDI-VAR innovation model.")
    if len(series) != initial_window + len(residuals):
        raise ValueError("series and residuals do not match the stated initial_window.")
    if n_paths is None:
        n_paths = int(innovation_model.get("calibration_paths", 32))

    usable_horizons = tuple(sorted(h for h in horizons if h >= 1))
    max_horizon = max(usable_horizons)
    first_residual = int(innovation_model["calibration_start_residual"])
    residual_origins = np.arange(first_residual, len(residuals) - max_horizon + 1)
    if len(residual_origins) > max_origins:
        residual_origins = residual_origins[-max_origins:]
    if not len(residual_origins):
        raise ValueError("The reserved calibration block has no complete origins.")

    draws = {h: [] for h in usable_horizons}
    truths = {h: [] for h in usable_horizons}
    for offset, residual_origin in enumerate(residual_origins):
        series_origin = initial_window + residual_origin
        fitted_var = fit_var_ols(
            series[:series_origin], lags=lags,
            include_intercept=include_intercept,
        )
        innovations = innovation_model["sample_with_history_fn"](
            residuals[:residual_origin], n_paths, max_horizon,
            seed + 10_000 * offset,
        )
        paths = simulate_forecast_paths(
            series[series_origin-lags:series_origin], fitted_var["beta"],
            innovations, lags, include_intercept,
        )
        for horizon in usable_horizons:
            draws[horizon].append(paths[:, horizon - 1])
            truths[horizon].append(series[series_origin + horizon - 1])

    anchors = {
        horizon: fit_scale_multiplier(
            np.stack(draws[horizon]), np.stack(truths[horizon])
        )
        for horizon in usable_horizons
    }
    regularized_anchors = regularize_anchor_multipliers(
        anchors, shrinkage=shrinkage, bounds=bounds
    )
    multipliers = expand_anchor_multipliers(regularized_anchors, max_horizon)
    innovation_model["calibration_multipliers"] = multipliers
    innovation_model["raw_calibration_anchor_multipliers"] = anchors
    innovation_model["calibration_anchor_multipliers"] = regularized_anchors
    innovation_model["calibration_origin_count"] = len(residual_origins)
    innovation_model["calibration_shrinkage"] = shrinkage
    innovation_model["calibration_bounds"] = bounds
    return multipliers
