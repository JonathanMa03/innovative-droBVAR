"""End-to-end residual-innovation VAR experiment orchestration."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from innovcal.api.evaluation import compare_innovation_models
from innovcal.api.innovations import fit_innovations, sample_innovations
from innovcal.data.financial import ChronologicalSplit, chronological_split
from innovcal.di_var.residuals import rolling_var_residuals
from innovcal.forecasting.monte_carlo import simulate_forecast_paths
from innovcal.vector_ar.fit import fit_var_ols


@dataclass(frozen=True)
class FinancialExperimentResult:
    """Artifacts produced by one chronological innovation-model comparison."""

    split: ChronologicalSplit
    fitted_var: dict
    residuals: np.ndarray
    innovation_models: dict[str, dict]
    forecasts: dict[str, dict]
    evaluation: pd.DataFrame


def run_financial_experiment(
    returns: np.ndarray,
    methods: tuple[str, ...] = ("gaussian", "student_t", "bootstrap"),
    lags: int = 1,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
    n_paths: int = 1000,
    seed: int = 123,
    student_t_df: float = 5.0,
    diffusion_options: dict | None = None,
) -> FinancialExperimentResult:
    """Run one leakage-aware VAR innovation comparison.

    Rolling residuals are generated across the calibration period. The final
    VAR is then refitted on training plus calibration observations, while the
    innovation laws are fitted only to the recorded pseudo-out-of-sample errors.
    """
    split = chronological_split(
        returns,
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
    )
    calibration_sample = np.vstack([split.train, split.calibration])
    residuals = rolling_var_residuals(
        calibration_sample,
        initial_window=len(split.train),
        lags=lags,
        expanding=True,
        include_intercept=True,
    )
    fitted_var = fit_var_ols(
        calibration_sample,
        lags=lags,
        include_intercept=True,
    )

    innovation_models: dict[str, dict] = {}
    forecasts: dict[str, dict] = {}

    for offset, method in enumerate(methods):
        options: dict = {}
        if method == "student_t":
            options["df"] = student_t_df
        elif method == "diffusion":
            options.update(diffusion_options or {})

        innovation_model = fit_innovations(
            residuals=residuals,
            method=method,
            **options,
        )
        innovations = sample_innovations(
            innovation_model,
            n_paths=n_paths,
            horizon=len(split.test),
            seed=seed + offset,
        )
        paths = simulate_forecast_paths(
            y_history=calibration_sample[-lags:],
            beta=fitted_var["beta"],
            innovation_paths=innovations,
            lags=lags,
            include_intercept=True,
        )
        innovation_models[method] = innovation_model
        forecasts[method] = {
            "forecast_paths": paths,
            "innovation_paths": innovations,
            "innovation_model": method,
            "y_true": split.test,
        }

    evaluation = compare_innovation_models(
        forecasts,
        y_true=split.test,
        dgp_name="financial",
        forecast_model="VAR",
    )
    return FinancialExperimentResult(
        split=split,
        fitted_var=fitted_var,
        residuals=residuals,
        innovation_models=innovation_models,
        forecasts=forecasts,
        evaluation=evaluation,
    )
