from pathlib import Path

import numpy as np
import pandas as pd

from innovcal.evaluation.metrics import (
    summarize_probabilistic_forecast,
    make_summary_row,
)

from innovcal.experiments.artifacts import save_table


def evaluate_calibration(
    forecast_paths: np.ndarray,
    y_true: np.ndarray,
    interval: tuple[float, float] = (0.05, 0.95),
    nominal_levels: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict:
    """
    Evaluate calibration-focused metrics.
    """
    summary = summarize_probabilistic_forecast(
        forecast_paths=forecast_paths,
        y_true=y_true,
        interval=interval,
        nominal_levels=nominal_levels,
    )

    return {
        "avg_coverage": summary["avg_coverage"],
        "coverage_by_series": summary["coverage_by_series"],
        "avg_width": summary["avg_width"],
        "width_by_series": summary["width_by_series"],
        "ece": summary["ece"],
        "pit_deviation": summary["pit_deviation"],
        "pit_values": summary["pit_values"],
        "lower": summary["lower"],
        "median": summary["median"],
        "upper": summary["upper"],
    }


def evaluate_probabilistic_scores(
    forecast_paths: np.ndarray,
    y_true: np.ndarray,
    interval: tuple[float, float] = (0.05, 0.95),
    nominal_levels: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict:
    """
    Evaluate probabilistic scoring-rule metrics.
    """
    summary = summarize_probabilistic_forecast(
        forecast_paths=forecast_paths,
        y_true=y_true,
        interval=interval,
        nominal_levels=nominal_levels,
    )

    return {
        "energy_score": summary["energy_score"],
        "crps": summary["crps"],
        "interval_score": summary["interval_score"],
    }


def evaluate_forecast(
    forecast_result: dict,
    y_true: np.ndarray | None = None,
    interval: tuple[float, float] = (0.05, 0.95),
    nominal_levels: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict:
    """
    Evaluate one forecast result dictionary.

    forecast_result must contain:
        forecast_paths

    y_true can be provided directly or stored inside forecast_result.
    """
    forecast_paths = forecast_result["forecast_paths"]

    if y_true is None:
        if "y_true" not in forecast_result:
            raise ValueError("y_true must be provided or included in forecast_result.")
        y_true = forecast_result["y_true"]

    return summarize_probabilistic_forecast(
        forecast_paths=forecast_paths,
        y_true=y_true,
        interval=interval,
        nominal_levels=nominal_levels,
    )


def evaluate_forecast_row(
    forecast_result: dict,
    y_true: np.ndarray | None = None,
    dgp_name: str = "unknown",
    forecast_model: str = "VAR",
    innovation_model: str | None = None,
    interval: tuple[float, float] = (0.05, 0.95),
    nominal_levels: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict:
    """
    Evaluate one forecast and return a flat table row.
    """
    if innovation_model is None:
        innovation_model = forecast_result.get("innovation_model", "unknown")

    summary = evaluate_forecast(
        forecast_result=forecast_result,
        y_true=y_true,
        interval=interval,
        nominal_levels=nominal_levels,
    )

    row = make_summary_row(
        dgp_name=dgp_name,
        forecast_model=forecast_model,
        innovation_model=innovation_model,
        summary=summary,
    )

    nominal_coverage = interval[1] - interval[0]
    row["nominal_coverage"] = nominal_coverage
    row["coverage_error"] = row["avg_coverage"] - nominal_coverage
    row["abs_coverage_error"] = abs(row["coverage_error"])

    return row


def compare_innovation_models(
    forecasts: dict,
    y_true: np.ndarray | None = None,
    dgp_name: str = "unknown",
    forecast_model: str = "VAR",
    interval: tuple[float, float] = (0.05, 0.95),
    nominal_levels: tuple[float, ...] = (0.5, 0.8, 0.9),
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str = "innovation_model_comparison.csv",
) -> pd.DataFrame:
    """
    Compare forecast outputs across innovation models.

    Parameters
    ----------
    forecasts:
        Dictionary like:
        {
            "gaussian": {"forecast_paths": ...},
            "bootstrap": {"forecast_paths": ...},
            "student_t": {"forecast_paths": ...},
            "diffusion": {"forecast_paths": ...},
        }
    """
    rows = []

    for innovation_model, forecast_result in forecasts.items():
        row = evaluate_forecast_row(
            forecast_result=forecast_result,
            y_true=y_true,
            dgp_name=dgp_name,
            forecast_model=forecast_model,
            innovation_model=innovation_model,
            interval=interval,
            nominal_levels=nominal_levels,
        )

        rows.append(row)

    results_df = pd.DataFrame(rows)

    if save and output_dir is not None:
        save_table(
            results_df,
            Path(output_dir) / filename,
        )

    return results_df


def relative_improvement_vs_baseline(
    results_df: pd.DataFrame,
    baseline_model: str = "gaussian",
    group_col: str = "dgp",
    model_col: str = "innovation_model",
    metrics: list[str] | None = None,
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str = "relative_improvement_vs_baseline.csv",
) -> pd.DataFrame:
    """
    Compute relative improvement against a baseline innovation model.

    Positive values mean improvement for lower-is-better metrics.
    """
    if metrics is None:
        metrics = [
            "ece",
            "pit_deviation",
            "crps",
            "energy_score",
            "interval_score",
            "abs_coverage_error",
        ]

    rows = []

    for group_value in results_df[group_col].unique():
        group_df = results_df[results_df[group_col] == group_value]

        baseline = group_df[
            group_df[model_col] == baseline_model
        ].iloc[0]

        for _, row in group_df.iterrows():
            out = {
                group_col: group_value,
                model_col: row[model_col],
            }

            for metric in metrics:
                baseline_value = baseline[metric]
                model_value = row[metric]

                if baseline_value == 0:
                    improvement = np.nan
                else:
                    improvement = (
                        baseline_value - model_value
                    ) / baseline_value

                out[f"{metric}_relative_improvement"] = improvement

            rows.append(out)

    improvement_df = pd.DataFrame(rows)

    if save and output_dir is not None:
        save_table(
            improvement_df,
            Path(output_dir) / filename,
        )

    return improvement_df


def headline_metric_rankings(
    results_df: pd.DataFrame,
    group_col: str = "dgp",
    model_col: str = "innovation_model",
    metrics: list[str] | None = None,
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str = "headline_metric_rankings.csv",
) -> pd.DataFrame:
    """
    Return best innovation model by metric within each DGP/group.
    """
    if metrics is None:
        metrics = [
            "ece",
            "pit_deviation",
            "crps",
            "energy_score",
            "interval_score",
            "abs_coverage_error",
        ]

    rows = []

    for group_value in results_df[group_col].unique():
        group_df = results_df[results_df[group_col] == group_value]

        row = {
            group_col: group_value,
        }

        for metric in metrics:
            best_idx = group_df[metric].idxmin()
            best_row = group_df.loc[best_idx]

            row[f"best_{metric}_model"] = best_row[model_col]
            row[f"best_{metric}_value"] = best_row[metric]

        rows.append(row)

    ranking_df = pd.DataFrame(rows)

    if save and output_dir is not None:
        save_table(
            ranking_df,
            Path(output_dir) / filename,
        )

    return ranking_df


def metric_win_counts(
    ranking_df: pd.DataFrame,
    metrics: list[str] | None = None,
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str = "metric_win_counts.csv",
) -> pd.DataFrame:
    """
    Count how often each innovation model wins each metric.
    """
    if metrics is None:
        metrics = [
            "ece",
            "crps",
            "energy_score",
            "interval_score",
        ]

    rows = []

    for metric in metrics:
        col = f"best_{metric}_model"

        counts = (
            ranking_df[col]
            .value_counts()
            .reset_index()
        )

        counts.columns = ["innovation_model", "wins"]
        counts["metric"] = metric

        rows.append(counts)

    win_df = pd.concat(rows, ignore_index=True)
    win_df = win_df[
        ["metric", "innovation_model", "wins"]
    ].sort_values(
        ["metric", "wins"],
        ascending=[True, False],
    )

    if save and output_dir is not None:
        save_table(
            win_df,
            Path(output_dir) / filename,
        )

    return win_df