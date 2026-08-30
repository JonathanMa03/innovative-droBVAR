"""Deployment-shift experiments with stressed realized innovations.

The forecast distribution is held fixed while the data-generating innovations
are perturbed.  This measures loss of forecast credibility under shift rather
than merely describing a wider forecast distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from innovcal.api.innovations import sample_innovations
from innovcal.dro.perturbations import perturb_innovation_paths
from innovcal.evaluation.metrics import summarize_probabilistic_forecast
from innovcal.forecasting.monte_carlo import simulate_forecast_paths
from innovcal.forecasting.portfolio import portfolio_simple_returns_from_log_returns


def generate_shifted_realizations(
    fitted_var: dict,
    truth_innovation_model: dict,
    y_history: np.ndarray,
    horizon: int,
    n_realizations: int,
    method: str,
    epsilon: float,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate common realized paths from nominal or shifted innovations."""
    innovations = sample_innovations(
        truth_innovation_model,
        n_paths=n_realizations,
        horizon=horizon,
        seed=seed,
    )
    shifted = perturb_innovation_paths(
        innovations,
        method=method,
        epsilon=epsilon,
        seed=seed + 1,
    )
    realized = simulate_forecast_paths(
        y_history=np.asarray(y_history),
        beta=fitted_var["beta"],
        innovation_paths=shifted,
        lags=fitted_var["lags"],
        include_intercept=fitted_var.get("include_intercept", True),
    )
    return realized, shifted


def evaluate_shifted_realizations(
    forecasts: dict[str, dict | np.ndarray],
    realized_paths: np.ndarray,
    method: str,
    epsilon: float,
    interval: tuple[float, float] = (0.05, 0.95),
    nominal_levels: tuple[float, ...] = (0.5, 0.8, 0.9),
    portfolio_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Score fixed forecast distributions against many shifted realizations."""
    realized_paths = np.asarray(realized_paths, dtype=float)
    if realized_paths.ndim != 3:
        raise ValueError("realized_paths must have shape (realization, horizon, asset).")

    k = realized_paths.shape[-1]
    if portfolio_weights is None:
        portfolio_weights = np.full(k, 1.0 / k)

    rows = []
    for name, result in forecasts.items():
        forecast_paths = np.asarray(
            result["forecast_paths"] if isinstance(result, dict) else result,
            dtype=float,
        )
        summaries = [
            summarize_probabilistic_forecast(
                forecast_paths,
                truth,
                interval=interval,
                nominal_levels=nominal_levels,
            )
            for truth in realized_paths
        ]

        forecast_portfolio = portfolio_simple_returns_from_log_returns(
            forecast_paths,
            portfolio_weights,
        )
        realized_portfolio = portfolio_simple_returns_from_log_returns(
            realized_paths,
            portfolio_weights,
        )
        forecast_terminal = np.prod(1.0 + forecast_portfolio, axis=1) - 1.0
        realized_terminal = np.prod(1.0 + realized_portfolio, axis=1) - 1.0
        forecast_var_05 = float(np.quantile(forecast_terminal, 0.05))

        rows.append(
            {
                "innovation_model": name,
                "shift_method": method,
                "epsilon": float(epsilon),
                "n_realizations": len(realized_paths),
                "avg_coverage": float(np.mean([x["avg_coverage"] for x in summaries])),
                "avg_width": float(np.mean([x["avg_width"] for x in summaries])),
                "ece": float(np.mean([x["ece"] for x in summaries])),
                "pit_deviation": float(np.mean([x["pit_deviation"] for x in summaries])),
                "crps": float(np.mean([x["crps"] for x in summaries])),
                "energy_score": float(np.mean([x["energy_score"] for x in summaries])),
                "interval_score": float(np.mean([x["interval_score"] for x in summaries])),
                "portfolio_var_05": forecast_var_05,
                "portfolio_breach_rate": float(np.mean(realized_terminal < forecast_var_05)),
                "realized_portfolio_q05": float(np.quantile(realized_terminal, 0.05)),
            }
        )

    return pd.DataFrame(rows)


def add_stress_degradation(
    results: pd.DataFrame,
    baseline_epsilon: float = 0.0,
) -> pd.DataFrame:
    """Add changes relative to the matching unshifted result."""
    output = results.copy()
    metrics = [
        "avg_coverage",
        "ece",
        "pit_deviation",
        "crps",
        "energy_score",
        "interval_score",
        "portfolio_breach_rate",
    ]
    for metric in metrics:
        baseline = output.loc[
            output["epsilon"] == baseline_epsilon,
            ["innovation_model", "shift_method", metric],
        ].drop_duplicates(["innovation_model", "shift_method"])
        baseline_lookup = {
            (row.innovation_model, row.shift_method): getattr(row, metric)
            for row in baseline.itertuples(index=False)
        }
        baseline_values = [
            baseline_lookup.get((model, method), np.nan)
            for model, method in zip(output["innovation_model"], output["shift_method"])
        ]
        output[f"{metric}_change"] = output[metric] - baseline_values
    return output
