from pathlib import Path

import numpy as np
import pandas as pd

from src.api.modeling import fit_model, extract_residuals
from src.api.innovations import fit_innovations
from src.api.forecasts import generate_forecasts
from src.api.evaluation import compare_innovation_models
from src.api.comparison import (
    best_model_table,
    win_counts,
    relative_improvement_table,
    compact_results_table,
)
from src.api.stress import (
    stress_test,
    scale_grid,
    tail_grid,
    outlier_grid,
    stress_summary,
    best_model_by_stress_experiment,
)

from src.experiments.artifacts import save_table, save_json, save_array_npz
from src.experiments.paths import result_dirs


def train_test_split_time_series(
    y: np.ndarray,
    n_train: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a time series into train and fixed-horizon test set.
    """
    y = np.asarray(y, dtype=float)

    y_train = y[:n_train]
    y_test = y[n_train:n_train + horizon]

    if len(y_test) < horizon:
        raise ValueError(
            f"Requested horizon={horizon}, but only got {len(y_test)} test observations."
        )

    return y_train, y_test


def run_single_dgp_experiment(
    y: np.ndarray,
    dgp_name: str,
    config,
    test: bool = False,
    save: bool = True,
) -> dict:
    """
    Run the full nominal numerical experiment for one DGP.

    Steps:
    1. split data
    2. fit VAR
    3. extract residuals
    4. fit innovation models
    5. generate forecasts
    6. evaluate calibration/scores
    """
    dirs = result_dirs(
        experiment_name=dgp_name,
        test=test,
    )

    y_train, y_test = train_test_split_time_series(
        y=y,
        n_train=config.n_train,
        horizon=config.horizon,
    )

    fitted_model = fit_model(
        y_train,
        model="var",
        lags=config.lags,
        include_intercept=True,
        save=save,
        output_dir=dirs["models"],
    )

    residuals = extract_residuals(
        fitted_model,
        model="var",
        save=save,
        output_dir=dirs["residuals"],
    )

    innovation_models = {}

    for method in config.innovation_models:
        kwargs = {}

        if method == "student_t":
            kwargs["df"] = config.student_t_df

        if method == "diffusion":
            kwargs.update(
                {
                    "timesteps": config.diffusion_timesteps,
                    "epochs": config.diffusion_epochs,
                    "lr": config.diffusion_lr,
                    "hidden_dim": config.diffusion_hidden_dim,
                    "time_embedding_dim": config.diffusion_time_embedding_dim,
                    "device": "cpu",
                    "verbose": True,
                }
            )

        innovation_models[method] = fit_innovations(
            residuals=residuals,
            method=method,
            save=save,
            output_dir=dirs["innovations"],
            **kwargs,
        )

    forecasts = {}

    for method, innovation_model in innovation_models.items():
        forecasts[method] = generate_forecasts(
            fitted_model=fitted_model,
            innovation_model=innovation_model,
            y_history=y_train[-config.lags:],
            h=config.horizon,
            n_paths=config.n_paths,
            model="var",
            seed=config.seed,
            save=save,
            output_dir=dirs["forecasts"],
            filename=f"{dgp_name}_{method}_forecast_paths.npz",
        )

        forecasts[method]["y_true"] = y_test

    results_df = compare_innovation_models(
        forecasts=forecasts,
        y_true=y_test,
        dgp_name=dgp_name,
        forecast_model=config.forecast_model,
        interval=config.interval,
        nominal_levels=config.nominal_levels,
        save=save,
        output_dir=dirs["tables"],
        filename=f"{dgp_name}_main_results.csv",
    )

    compact_df = compact_results_table(
        results_df,
        save=save,
        output_dir=dirs["tables"],
        filename=f"{dgp_name}_compact_results.csv",
    )

    rankings_df = best_model_table(
        results_df,
        save=save,
        output_dir=dirs["tables"],
        filename=f"{dgp_name}_best_model_table.csv",
    )

    wins_df = win_counts(
        rankings_df,
        save=save,
        output_dir=dirs["tables"],
        filename=f"{dgp_name}_win_counts.csv",
    )

    if "gaussian" in results_df["innovation_model"].values:
        relative_df = relative_improvement_table(
            results_df,
            baseline_model="gaussian",
            save=save,
            output_dir=dirs["tables"],
            filename=f"{dgp_name}_relative_improvement.csv",
        )
    else:
        relative_df = pd.DataFrame()

    if save:
        save_array_npz(
            dirs["results"] / f"{dgp_name}_train_test_split.npz",
            y_train=y_train,
            y_test=y_test,
        )

        save_json(
            {
                "dgp": dgp_name,
                "n_train": config.n_train,
                "horizon": config.horizon,
                "n_paths": config.n_paths,
                "lags": config.lags,
            },
            dirs["logs"] / f"{dgp_name}_experiment_config.json",
        )

    return {
        "dgp": dgp_name,
        "dirs": dirs,
        "y_train": y_train,
        "y_test": y_test,
        "fitted_model": fitted_model,
        "residuals": residuals,
        "innovation_models": innovation_models,
        "forecasts": forecasts,
        "results_df": results_df,
        "compact_df": compact_df,
        "rankings_df": rankings_df,
        "wins_df": wins_df,
        "relative_df": relative_df,
    }


def run_numerical_experiment(
    datasets: dict[str, np.ndarray],
    config,
    test: bool = False,
    save: bool = True,
) -> dict:
    """
    Run nominal numerical experiments for all DGPs.
    """
    outputs = {}

    all_results = []
    all_rankings = []
    all_relative = []

    for dgp_name, y in datasets.items():
        print(f"\nRunning numerical experiment for: {dgp_name}")

        out = run_single_dgp_experiment(
            y=y,
            dgp_name=dgp_name,
            config=config,
            test=test,
            save=save,
        )

        outputs[dgp_name] = out

        all_results.append(out["results_df"])
        all_rankings.append(out["rankings_df"])
        all_relative.append(out["relative_df"])

    combined_results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined_rankings_df = best_model_table(
        combined_results_df,
    )

    combined_win_counts_df = win_counts(
        combined_rankings_df,
    )

    combined_relative_df = pd.concat(
        all_relative,
        ignore_index=True,
    )

    dirs = result_dirs(
        experiment_name="numerical_experiment",
        test=test,
    )

    if save:
        save_table(
            combined_results_df,
            dirs["tables"] / "combined_main_results.csv",
        )

        save_table(
            combined_rankings_df,
            dirs["tables"] / "combined_headline_rankings.csv",
        )

        save_table(
            combined_win_counts_df,
            dirs["tables"] / "combined_win_counts.csv",
        )

        save_table(
            combined_relative_df,
            dirs["tables"] / "combined_relative_improvement.csv",
        )

    outputs["combined"] = {
        "results_df": combined_results_df,
        "rankings_df": combined_rankings_df,
        "win_counts_df": combined_win_counts_df,
        "relative_df": combined_relative_df,
        "dirs": dirs,
    }

    return outputs


def run_stress_experiments_for_dgp(
    experiment_output: dict,
    config,
    dgp_name: str,
    test: bool = False,
    save: bool = True,
) -> dict:
    """
    Run scale, tail, and outlier stress tests for one fitted DGP experiment.
    """
    dirs = result_dirs(
        experiment_name=f"{dgp_name}_stress_tests",
        test=test,
    )

    fitted_model = experiment_output["fitted_model"]
    forecasts = experiment_output["forecasts"]
    y_train = experiment_output["y_train"]
    y_test = experiment_output["y_test"]

    y_history = y_train[-config.lags:]

    scale_df = stress_test(
        fitted_model=fitted_model,
        forecasts=forecasts,
        y_history=y_history,
        y_true=y_test,
        method="scale",
        epsilon_grid=scale_grid(),
        dgp_name=dgp_name,
        forecast_model=config.forecast_model,
        interval=config.interval,
        nominal_levels=config.nominal_levels,
        save=save,
        output_dir=dirs["tables"],
        filename=f"{dgp_name}_scale_stress.csv",
    )

    tail_df = stress_test(
        fitted_model=fitted_model,
        forecasts=forecasts,
        y_history=y_history,
        y_true=y_test,
        method="tail",
        epsilon_grid=tail_grid(),
        dgp_name=dgp_name,
        forecast_model=config.forecast_model,
        interval=config.interval,
        nominal_levels=config.nominal_levels,
        save=save,
        output_dir=dirs["tables"],
        filename=f"{dgp_name}_tail_stress.csv",
    )

    outlier_df = stress_test(
        fitted_model=fitted_model,
        forecasts=forecasts,
        y_history=y_history,
        y_true=y_test,
        method="outlier",
        epsilon_grid=outlier_grid(),
        dgp_name=dgp_name,
        forecast_model=config.forecast_model,
        interval=config.interval,
        nominal_levels=config.nominal_levels,
        save=save,
        output_dir=dirs["tables"],
        filename=f"{dgp_name}_outlier_stress.csv",
    )

    scale_summary = stress_summary(
        scale_df,
        method="scale",
        save=save,
        output_dir=dirs["tables"],
        filename=f"{dgp_name}_scale_summary.csv",
    )

    tail_summary = stress_summary(
        tail_df,
        method="tail",
        save=save,
        output_dir=dirs["tables"],
        filename=f"{dgp_name}_tail_summary.csv",
    )

    outlier_summary = stress_summary(
        outlier_df,
        method="outlier",
        save=save,
        output_dir=dirs["tables"],
        filename=f"{dgp_name}_outlier_summary.csv",
    )

    robustness_summary = best_model_by_stress_experiment(
        summaries={
            "scale_perturbation": scale_summary,
            "tail_contamination": tail_summary,
            "outlier_contamination": outlier_summary,
        },
        save=save,
        output_dir=dirs["tables"],
        filename=f"{dgp_name}_robustness_best_model_summary.csv",
    )

    return {
        "dirs": dirs,
        "scale_df": scale_df,
        "tail_df": tail_df,
        "outlier_df": outlier_df,
        "scale_summary": scale_summary,
        "tail_summary": tail_summary,
        "outlier_summary": outlier_summary,
        "robustness_summary": robustness_summary,
    }


def run_all_stress_experiments(
    numerical_outputs: dict,
    config,
    test: bool = False,
    save: bool = True,
) -> dict:
    """
    Run robustness experiments for every DGP output from run_numerical_experiment().
    """
    stress_outputs = {}

    all_robustness = []

    for dgp_name in config.dgp_names:
        print(f"\nRunning stress tests for: {dgp_name}")

        out = run_stress_experiments_for_dgp(
            experiment_output=numerical_outputs[dgp_name],
            config=config,
            dgp_name=dgp_name,
            test=test,
            save=save,
        )

        stress_outputs[dgp_name] = out
        all_robustness.append(out["robustness_summary"])

    combined_robustness_df = pd.concat(
        all_robustness,
        ignore_index=True,
    )

    dirs = result_dirs(
        experiment_name="combined_stress_tests",
        test=test,
    )

    if save:
        save_table(
            combined_robustness_df,
            dirs["tables"] / "combined_robustness_best_model_summary.csv",
        )

    stress_outputs["combined"] = {
        "robustness_summary": combined_robustness_df,
        "dirs": dirs,
    }

    return stress_outputs