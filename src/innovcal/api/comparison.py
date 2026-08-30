from pathlib import Path

import pandas as pd

from innovcal.experiments.artifacts import save_table


def best_model_table(
    results_df: pd.DataFrame,
    group_cols: list[str] | None = None,
    model_col: str = "innovation_model",
    metrics: list[str] | None = None,
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str = "best_model_table.csv",
) -> pd.DataFrame:
    """
    Return best model by metric within each group.

    Lower metric values are assumed better.
    """
    if group_cols is None:
        group_cols = ["dgp"]

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

    for group_values, group_df in results_df.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row = {
            col: value
            for col, value in zip(group_cols, group_values)
        }

        for metric in metrics:
            best_idx = group_df[metric].idxmin()
            best = group_df.loc[best_idx]

            row[f"best_{metric}_model"] = best[model_col]
            row[f"best_{metric}_value"] = best[metric]

        rows.append(row)

    out = pd.DataFrame(rows)

    if save and output_dir is not None:
        save_table(out, Path(output_dir) / filename)

    return out


def win_counts(
    ranking_df: pd.DataFrame,
    metrics: list[str] | None = None,
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str = "win_counts.csv",
) -> pd.DataFrame:
    """
    Count model wins from a best-model ranking table.
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

    out = pd.concat(rows, ignore_index=True)
    out = out[
        ["metric", "innovation_model", "wins"]
    ].sort_values(["metric", "wins"], ascending=[True, False])

    if save and output_dir is not None:
        save_table(out, Path(output_dir) / filename)

    return out


def relative_improvement_table(
    results_df: pd.DataFrame,
    baseline_model: str = "gaussian",
    group_col: str = "dgp",
    model_col: str = "innovation_model",
    metrics: list[str] | None = None,
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str = "relative_improvement.csv",
) -> pd.DataFrame:
    """
    Compute relative improvement against a baseline model.

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
        group_df = results_df[
            results_df[group_col] == group_value
        ]

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
                    improvement = None
                else:
                    improvement = (
                        baseline_value - model_value
                    ) / baseline_value

                out[f"{metric}_relative_improvement"] = improvement

            rows.append(out)

    out = pd.DataFrame(rows)

    if save and output_dir is not None:
        save_table(out, Path(output_dir) / filename)

    return out


def robustness_best_model_summary(
    summary_dfs: dict[str, pd.DataFrame],
    metric_col: str = "mean_degradation",
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str = "robustness_best_model_summary.csv",
) -> pd.DataFrame:
    """
    Combine robustness summaries from multiple stress experiments.

    Expected input:
        {
            "scale_perturbation": scale_summary_df,
            "tail_contamination": tail_summary_df,
            "outlier_contamination": outlier_summary_df,
        }

    Each summary_df should contain:
        dgp
        innovation_model
        metric_col
    """
    rows = []

    for experiment_name, df in summary_dfs.items():
        for dgp_name in df["dgp"].unique():
            subset = df[df["dgp"] == dgp_name]

            best = subset.sort_values(metric_col).iloc[0]

            row = {
                "dgp": dgp_name,
                "experiment": experiment_name,
                "best_innovation_model": best["innovation_model"],
                metric_col: best[metric_col],
            }

            if "max_degradation" in best:
                row["max_degradation"] = best["max_degradation"]

            rows.append(row)

    out = pd.DataFrame(rows).sort_values(
        ["dgp", "experiment"]
    )

    if save and output_dir is not None:
        save_table(out, Path(output_dir) / filename)

    return out


def compact_results_table(
    results_df: pd.DataFrame,
    cols: list[str] | None = None,
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str = "compact_results.csv",
) -> pd.DataFrame:
    """
    Return a compact display version of the main results table.
    """
    if cols is None:
        cols = [
            "dgp",
            "innovation_model",
            "avg_coverage",
            "abs_coverage_error",
            "avg_width",
            "ece",
            "pit_deviation",
            "crps",
            "energy_score",
            "interval_score",
        ]

    out = results_df[cols].copy()

    if save and output_dir is not None:
        save_table(out, Path(output_dir) / filename)

    return out