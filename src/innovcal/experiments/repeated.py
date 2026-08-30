"""Repeated rolling-origin experiments across seeds and estimation windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from innovcal.api.innovations import fit_innovations
from innovcal.data.financial import chronological_split
from innovcal.di_var.residuals import rolling_var_residuals
from innovcal.evaluation.rolling import evaluate_rolling_forecasts
from innovcal.forecasting.rolling import generate_rolling_var_forecasts
from innovcal.cdi_var.calibration import fit_rolling_var_forecast_calibration


def run_repeated_rolling_experiment(
    returns: np.ndarray,
    seeds: tuple[int, ...] = (101, 202, 303),
    methods: tuple[str, ...] = (
        "gaussian",
        "student_t",
        "bootstrap",
        "block_bootstrap",
        "volatility_bootstrap",
        "diffusion",
    ),
    window_lengths: tuple[int | None, ...] = (None,),
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
    lags: int = 1,
    horizon: int = 20,
    origin_step: int = 20,
    n_paths: int = 250,
    block_length: int = 10,
    volatility_span: int = 60,
    diffusion_options: dict | None = None,
    cdi_var_options: dict | None = None,
) -> pd.DataFrame:
    """Return tidy rolling metrics for every seed and window specification."""
    returns = np.asarray(returns, dtype=float)
    split = chronological_split(returns, train_fraction, calibration_fraction)
    test_start = len(split.train) + len(split.calibration)
    calibration_sample = returns[:test_start]
    residuals = rolling_var_residuals(
        calibration_sample,
        initial_window=len(split.train),
        lags=lags,
        expanding=True,
        include_intercept=True,
    )

    tables = []
    for seed in seeds:
        models = {}
        for method in methods:
            options = {}
            if method == "block_bootstrap":
                options["block_length"] = block_length
            elif method == "volatility_bootstrap":
                options["volatility_span"] = volatility_span
            elif method == "diffusion":
                options.update(diffusion_options or {})
                options["seed"] = seed
            elif method == "cdi_var":
                options.update(cdi_var_options or {})
                options["seed"] = seed
            models[method] = fit_innovations(residuals, method=method, **options)
            if method == "cdi_var":
                fit_rolling_var_forecast_calibration(
                    models[method], calibration_sample, residuals,
                    initial_window=len(split.train), lags=lags,
                    horizons=tuple(options.get("calibration_horizons", (1, 5, 20))),
                    n_paths=options.get("calibration_paths", 32), seed=seed,
                    shrinkage=options.get("calibration_shrinkage", 0.5),
                    bounds=tuple(options.get("calibration_bounds", (0.8, 1.25))),
                )

        for window_length in window_lengths:
            rolling = generate_rolling_var_forecasts(
                returns,
                test_start=test_start,
                innovation_models=models,
                horizon=horizon,
                n_paths=n_paths,
                lags=lags,
                origin_step=origin_step,
                window_length=window_length,
                seed=seed,
            )
            table = evaluate_rolling_forecasts(
                rolling.forecasts,
                rolling.observations,
                horizons=tuple(h for h in (1, 5, 20) if h <= horizon),
            )
            table["seed"] = seed
            table["window"] = "expanding" if window_length is None else str(window_length)
            tables.append(table)

    return pd.concat(tables, ignore_index=True)


def summarize_repeated_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated runs without hiding seed-to-seed dispersion."""
    metrics = [
        "avg_coverage",
        "abs_coverage_error",
        "avg_width",
        "ece",
        "pit_deviation",
        "crps",
        "energy_score",
        "interval_score",
    ]
    return (
        results.groupby(["innovation_model", "horizon", "window"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
