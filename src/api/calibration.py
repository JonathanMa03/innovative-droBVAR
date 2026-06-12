import pandas as pd

from src.evaluation.metrics import (

    summarize_probabilistic_forecast,

    make_summary_row,

)

def calibration_results_table(

    forecast_store,

    forecast_models,

    dgp_names,

    innovation_model_names,

    interval,

    nominal_levels,

):

    summary_rows = []

    summary_objects = {}

    for forecast_model in forecast_models:

        summary_objects[forecast_model] = {}

        for dgp_name in dgp_names:

            summary_objects[forecast_model][dgp_name] = {}

            for innovation_model in innovation_model_names:

                obj = forecast_store[forecast_model][dgp_name][innovation_model]

                summary = summarize_probabilistic_forecast(

                    forecast_paths=obj["forecast_paths"],

                    y_true=obj["y_true"],

                    interval=interval,

                    nominal_levels=nominal_levels,

                )

                summary_objects[forecast_model][dgp_name][innovation_model] = summary

                summary_rows.append(

                    make_summary_row(

                        dgp_name=dgp_name,

                        forecast_model=forecast_model,

                        innovation_model=innovation_model,

                        summary=summary,

                    )

                )

    results_df = pd.DataFrame(summary_rows)

    nominal_coverage = interval[1] - interval[0]

    results_df["nominal_coverage"] = nominal_coverage

    results_df["coverage_error"] = (

        results_df["avg_coverage"] - nominal_coverage

    )

    results_df["abs_coverage_error"] = (

        results_df["coverage_error"].abs()

    )

    results_df = results_df.sort_values(

        ["dgp", "forecast_model", "ece"]

    )

    return results_df, summary_objects

def relative_improvement_table(
    main_results_df,
    forecast_models,
    dgp_names,
    innovation_model_names,
    baseline_model="gaussian",
    metrics=None,
):
    import numpy as np
    import pandas as pd

    if metrics is None:
        metrics = [
            "ece",
            "pit_deviation",
            "crps",
            "energy_score",
            "interval_score",
            "abs_coverage_error",
        ]

    relative_rows = []

    for forecast_model in forecast_models:
        for dgp_name in dgp_names:

            baseline = main_results_df[
                (main_results_df["forecast_model"] == forecast_model)
                & (main_results_df["dgp"] == dgp_name)
                & (main_results_df["innovation_model"] == baseline_model)
            ].iloc[0]

            for innovation_model in innovation_model_names:

                row = main_results_df[
                    (main_results_df["forecast_model"] == forecast_model)
                    & (main_results_df["dgp"] == dgp_name)
                    & (main_results_df["innovation_model"] == innovation_model)
                ].iloc[0]

                out = {
                    "forecast_model": forecast_model,
                    "dgp": dgp_name,
                    "innovation_model": innovation_model,
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

                    out[
                        f"{metric}_relative_improvement"
                    ] = improvement

                relative_rows.append(out)

    return pd.DataFrame(relative_rows)

def headline_ranking_table(
    main_results_df,
    forecast_models,
    dgp_names,
    ranking_metrics=None,
):
    import pandas as pd

    if ranking_metrics is None:
        ranking_metrics = [
            "ece",
            "pit_deviation",
            "crps",
            "energy_score",
            "interval_score",
            "abs_coverage_error",
        ]

    ranking_rows = []

    for forecast_model in forecast_models:
        for dgp_name in dgp_names:

            dgp_df = main_results_df[
                (main_results_df["forecast_model"] == forecast_model)
                & (main_results_df["dgp"] == dgp_name)
            ]

            row = {
                "forecast_model": forecast_model,
                "dgp": dgp_name,
            }

            for metric in ranking_metrics:

                best_idx = dgp_df[metric].idxmin()
                best_row = dgp_df.loc[best_idx]

                row[f"best_{metric}_innovation"] = (
                    best_row["innovation_model"]
                )

                row[f"best_{metric}_value"] = (
                    best_row[metric]
                )

            ranking_rows.append(row)

    return pd.DataFrame(
        ranking_rows
    )