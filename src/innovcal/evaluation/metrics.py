import numpy as np

from innovcal.forecasting.intervals import forecast_quantiles, prediction_interval, average_interval_width
from innovcal.calibration.coverage import interval_coverage, average_coverage
from innovcal.calibration.ece import expected_calibration_error_intervals
from innovcal.calibration.pit import pit_values_multivariate_marginal, pit_deviation_from_uniform
from innovcal.evaluation.scoring_rules import summarize_scoring_rules


def summarize_probabilistic_forecast(
    forecast_paths: np.ndarray,
    y_true: np.ndarray,
    interval: tuple[float, float] = (0.05, 0.95),
    nominal_levels: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict:
    """
    Unified forecast summary.

    Parameters
    ----------
    forecast_paths:
        Shape (n_paths, horizon, k)

    y_true:
        Shape (horizon, k)
    """
    lower_q, upper_q = interval
    alpha = lower_q + (1.0 - upper_q)

    qs = forecast_quantiles(
        forecast_paths,
        quantiles=(lower_q, 0.5, upper_q),
    )

    lower = qs[lower_q]
    median = qs[0.5]
    upper = qs[upper_q]

    coverage = interval_coverage(
        y_true=y_true,
        lower=lower,
        upper=upper,
    )

    avg_cov = average_coverage(
        y_true=y_true,
        lower=lower,
        upper=upper,
    )

    width = average_interval_width(
        lower=lower,
        upper=upper,
    )

    scoring = summarize_scoring_rules(
        forecast_paths=forecast_paths,
        y_true=y_true,
        lower=lower,
        upper=upper,
        alpha=alpha,
    )

    ece_result = expected_calibration_error_intervals(
        forecast_paths=forecast_paths,
        y_true=y_true,
        nominal_levels=nominal_levels,
    )

    pit = pit_values_multivariate_marginal(
        forecast_paths=forecast_paths,
        y_true=y_true,
    )

    return {
        "avg_coverage": float(avg_cov),
        "coverage_by_series": coverage,
        "avg_width": float(width.mean()),
        "width_by_series": width,
        "energy_score": float(scoring["energy_score"]),
        "crps": float(scoring["crps"]),
        "interval_score": float(scoring["interval_score"]),
        "ece": float(ece_result["ece"]),
        "pit_deviation": float(pit_deviation_from_uniform(pit)),
        "pit_values": pit,
        "lower": lower,
        "median": median,
        "upper": upper,
    }


def make_summary_row(
    dgp_name: str,
    forecast_model: str,
    innovation_model: str,
    summary: dict,
) -> dict:
    coverage = summary["coverage_by_series"]
    width = summary["width_by_series"]

    row = {
        "dgp": dgp_name,
        "forecast_model": forecast_model,
        "innovation_model": innovation_model,
        "avg_coverage": summary["avg_coverage"],
        "avg_width": summary["avg_width"],
        "energy_score": summary["energy_score"],
        "crps": summary["crps"],
        "interval_score": summary["interval_score"],
        "ece": summary["ece"],
        "pit_deviation": summary["pit_deviation"],
    }

    for j in range(len(coverage)):
        row[f"coverage_{j+1}"] = float(coverage[j])
        row[f"width_{j+1}"] = float(width[j])

    return row


def summarize_many_forecasts(
    forecast_dict: dict,
    y_true: np.ndarray,
    dgp_name: str,
    forecast_model: str,
) -> tuple[list[dict], dict]:
    """
    Summarize multiple innovation forecast outputs.

    Parameters
    ----------
    forecast_dict:
        Dictionary where keys are innovation model names and values are
        forecast paths with shape (n_paths, horizon, k).
    """
    rows = []
    summaries = {}

    for innovation_model, paths in forecast_dict.items():
        summary = summarize_probabilistic_forecast(
            forecast_paths=paths,
            y_true=y_true,
        )

        rows.append(
            make_summary_row(
                dgp_name=dgp_name,
                forecast_model=forecast_model,
                innovation_model=innovation_model,
                summary=summary,
            )
        )

        summaries[innovation_model] = summary

    return rows, summaries