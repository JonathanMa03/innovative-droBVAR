from pathlib import Path

import numpy as np
import pandas as pd

from innovcal.api.evaluation import evaluate_forecast_row
from innovcal.forecasting.monte_carlo import simulate_forecast_paths
from innovcal.experiments.artifacts import save_table


def perturb_distribution(
    innovation_paths: np.ndarray,
    method: str = "scale",
    epsilon=None,
    random_state: int = 123,
) -> np.ndarray:
    """
    Unified perturbation interface for innovation paths.

    innovation_paths shape:
        (n_paths, horizon, k)
    """
    method = method.lower()

    if method == "scale":
        if epsilon is None:
            epsilon = 0.10

        return scale_contamination(
            innovation_paths=innovation_paths,
            epsilon=float(epsilon),
        )

    if method == "tail":
        if epsilon is None:
            epsilon = {
                "prob": 0.05,
                "multiplier": 5.0,
            }

        return tail_contamination(
            innovation_paths=innovation_paths,
            contamination_prob=epsilon["prob"],
            multiplier=epsilon["multiplier"],
            random_state=random_state,
        )

    if method == "outlier":
        if epsilon is None:
            epsilon = {
                "prob": 0.05,
                "scale": 10.0,
            }

        return outlier_contamination(
            innovation_paths=innovation_paths,
            contamination_prob=epsilon["prob"],
            outlier_scale=epsilon["scale"],
            random_state=random_state,
        )

    raise ValueError(
        "method must be one of: scale, tail, outlier."
    )


def scale_contamination(
    innovation_paths: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Inflate innovation paths around their empirical center.

    epsilon=0.10 means approximately 10% scale inflation.
    """
    innovation_paths = np.asarray(innovation_paths, dtype=float)

    center = innovation_paths.mean(
        axis=(0, 1),
        keepdims=True,
    )

    return center + (1.0 + epsilon) * (
        innovation_paths - center
    )


def tail_contamination(
    innovation_paths: np.ndarray,
    contamination_prob: float,
    multiplier: float,
    random_state: int = 123,
) -> np.ndarray:
    """
    Rare tail amplification.

    With probability p:
        epsilon -> multiplier * epsilon
    """
    rng = np.random.default_rng(random_state)

    contaminated = np.asarray(
        innovation_paths,
        dtype=float,
    ).copy()

    mask = (
        rng.random(size=contaminated.shape)
        < contamination_prob
    )

    contaminated[mask] *= multiplier

    return contaminated


def outlier_contamination(
    innovation_paths: np.ndarray,
    contamination_prob: float,
    outlier_scale: float,
    random_state: int = 123,
) -> np.ndarray:
    """
    Replace a fraction of innovation entries with external outliers.

    Unlike tail contamination, this creates new shocks not necessarily
    aligned with observed innovation directions.
    """
    rng = np.random.default_rng(random_state)

    contaminated = np.asarray(
        innovation_paths,
        dtype=float,
    ).copy()

    scale = contaminated.std(
        axis=(0, 1),
        ddof=1,
    )

    scale = np.where(
        scale < 1e-8,
        1.0,
        scale,
    )

    mask = (
        rng.random(size=contaminated.shape)
        < contamination_prob
    )

    outliers = rng.normal(
        loc=0.0,
        scale=outlier_scale,
        size=contaminated.shape,
    )

    outliers = outliers * scale.reshape(
        1,
        1,
        -1,
    )

    contaminated[mask] = outliers[mask]

    return contaminated


def stress_test_forecast(
    fitted_model: dict,
    forecast_result: dict,
    y_history: np.ndarray,
    y_true: np.ndarray,
    method: str,
    epsilon,
    dgp_name: str = "unknown",
    forecast_model: str = "VAR",
    innovation_model: str | None = None,
    interval: tuple[float, float] = (0.05, 0.95),
    nominal_levels: tuple[float, ...] = (0.5, 0.8, 0.9),
    random_state: int = 123,
) -> dict:
    """
    Stress test one forecast result.

    forecast_result must contain:
        forecast_paths
        innovation_paths
    """
    if innovation_model is None:
        innovation_model = forecast_result.get(
            "innovation_model",
            "unknown",
        )

    perturbed_innovations = perturb_distribution(
        innovation_paths=forecast_result["innovation_paths"],
        method=method,
        epsilon=epsilon,
        random_state=random_state,
    )

    perturbed_paths = simulate_forecast_paths(
        y_history=y_history,
        beta=fitted_model["beta"],
        innovation_paths=perturbed_innovations,
        lags=fitted_model["lags"],
        include_intercept=fitted_model.get(
            "include_intercept",
            True,
        ),
    )

    stressed_result = {
        "forecast_paths": perturbed_paths,
        "innovation_paths": perturbed_innovations,
        "innovation_model": innovation_model,
    }

    row = evaluate_forecast_row(
        forecast_result=stressed_result,
        y_true=y_true,
        dgp_name=dgp_name,
        forecast_model=forecast_model,
        innovation_model=innovation_model,
        interval=interval,
        nominal_levels=nominal_levels,
    )

    row["stress_method"] = method
    row["epsilon"] = str(epsilon)

    if method == "scale":
        row["epsilon_value"] = float(epsilon)

    elif method == "tail":
        row["contamination_prob"] = float(epsilon["prob"])
        row["tail_multiplier"] = float(epsilon["multiplier"])

    elif method == "outlier":
        row["contamination_prob"] = float(epsilon["prob"])
        row["outlier_scale"] = float(epsilon["scale"])

    return row


def stress_test(
    fitted_model: dict,
    forecasts: dict,
    y_history: np.ndarray,
    y_true: np.ndarray,
    method: str,
    epsilon_grid: list,
    dgp_name: str = "unknown",
    forecast_model: str = "VAR",
    interval: tuple[float, float] = (0.05, 0.95),
    nominal_levels: tuple[float, ...] = (0.5, 0.8, 0.9),
    random_state: int = 123,
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> pd.DataFrame:
    """
    Stress test all innovation models for one DGP.

    forecasts should be:
        {
            "gaussian": forecast_result,
            "bootstrap": forecast_result,
            ...
        }
    """
    rows = []

    for innovation_model, forecast_result in forecasts.items():
        for epsilon in epsilon_grid:
            row = stress_test_forecast(
                fitted_model=fitted_model,
                forecast_result=forecast_result,
                y_history=y_history,
                y_true=y_true,
                method=method,
                epsilon=epsilon,
                dgp_name=dgp_name,
                forecast_model=forecast_model,
                innovation_model=innovation_model,
                interval=interval,
                nominal_levels=nominal_levels,
                random_state=random_state,
            )

            rows.append(row)

    results_df = pd.DataFrame(rows)

    results_df = add_stress_degradation_columns(
        results_df=results_df,
        method=method,
    )

    if save and output_dir is not None:
        if filename is None:
            filename = f"{method}_stress_test.csv"

        save_table(
            results_df,
            Path(output_dir) / filename,
        )

    return results_df


def add_stress_degradation_columns(
    results_df: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """
    Add degradation columns relative to the least-stressed case.

    For scale:
        baseline epsilon_value == 0 if present,
        otherwise smallest epsilon_value.

    For tail/outlier:
        baseline contamination_prob == 0 if present,
        otherwise smallest contamination_prob.
    """
    df = results_df.copy()

    metrics = [
        "ece",
        "pit_deviation",
        "crps",
        "energy_score",
        "interval_score",
        "avg_width",
        "avg_coverage",
        "abs_coverage_error",
    ]

    group_cols = [
        "dgp",
        "innovation_model",
        "stress_method",
    ]

    if method == "scale":
        baseline_col = "epsilon_value"
    else:
        baseline_col = "contamination_prob"

    degraded_rows = []

    for _, group in df.groupby(group_cols):
        baseline_value = group[baseline_col].min()
        baseline = group[group[baseline_col] == baseline_value].iloc[0]

        for _, row in group.iterrows():
            out = row.to_dict()

            for metric in metrics:
                out[f"{metric}_degradation"] = (
                    row[metric] - baseline[metric]
                )

            degraded_rows.append(out)

    return pd.DataFrame(degraded_rows)


def scale_grid(
    epsilons: tuple[float, ...] = (
        0.0,
        0.05,
        0.10,
        0.20,
        0.30,
    ),
) -> list[float]:
    return list(epsilons)


def tail_grid(
    probs: tuple[float, ...] = (
        0.0,
        0.02,
        0.05,
        0.10,
    ),
    multipliers: tuple[float, ...] = (
        3.0,
        5.0,
        8.0,
    ),
) -> list[dict]:
    return [
        {
            "prob": p,
            "multiplier": m,
        }
        for m in multipliers
        for p in probs
    ]


def outlier_grid(
    probs: tuple[float, ...] = (
        0.0,
        0.02,
        0.05,
        0.10,
    ),
    scales: tuple[float, ...] = (
        5.0,
        10.0,
        20.0,
    ),
) -> list[dict]:
    return [
        {
            "prob": p,
            "scale": s,
        }
        for s in scales
        for p in probs
    ]


def stress_summary(
    stress_df: pd.DataFrame,
    method: str,
    metric: str = "energy_score_degradation",
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> pd.DataFrame:
    """
    Summarize mean/max degradation by DGP and innovation model.
    """
    if method == "scale":
        filtered = stress_df[
            stress_df["epsilon_value"] > 0
        ]
    else:
        filtered = stress_df[
            stress_df["contamination_prob"] > 0
        ]

    summary = (
        filtered
        .groupby(["dgp", "innovation_model"])
        .agg(
            mean_degradation=(metric, "mean"),
            max_degradation=(metric, "max"),
        )
        .reset_index()
        .sort_values(["dgp", "mean_degradation"])
    )

    if save and output_dir is not None:
        if filename is None:
            filename = f"{method}_stress_summary.csv"

        save_table(
            summary,
            Path(output_dir) / filename,
        )

    return summary


def best_model_by_stress_experiment(
    summaries: dict[str, pd.DataFrame],
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str = "robustness_best_model_summary.csv",
) -> pd.DataFrame:
    """
    Combine scale/tail/outlier summaries into one best-model table.

    summaries:
        {
            "scale_perturbation": scale_summary_df,
            "tail_contamination": tail_summary_df,
            "outlier_contamination": outlier_summary_df,
        }
    """
    rows = []

    for experiment_name, summary_df in summaries.items():
        for dgp_name in summary_df["dgp"].unique():
            subset = summary_df[
                summary_df["dgp"] == dgp_name
            ]

            best = subset.sort_values(
                "mean_degradation"
            ).iloc[0]

            rows.append({
                "dgp": dgp_name,
                "experiment": experiment_name,
                "best_innovation_model": best["innovation_model"],
                "mean_degradation": best["mean_degradation"],
                "max_degradation": best["max_degradation"],
            })

    out = pd.DataFrame(rows).sort_values(
        ["dgp", "experiment"]
    )

    if save and output_dir is not None:
        save_table(
            out,
            Path(output_dir) / filename,
        )

    return out