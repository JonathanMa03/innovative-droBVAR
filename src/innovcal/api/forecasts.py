from pathlib import Path

import numpy as np

from innovcal.api.innovations import sample_innovations
from innovcal.forecasting.monte_carlo import simulate_forecast_paths
from innovcal.deep_ar.forecast import forecast_rnn_paths
from innovcal.experiments.artifacts import save_array_npz, save_json


def generate_forecasts(
    fitted_model: dict,
    innovation_model: dict | None,
    y_history: np.ndarray,
    h: int,
    n_paths: int = 250,
    model: str = "var",
    seed: int | None = None,
    save: bool = False,
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> dict:
    """
    Generate probabilistic forecast trajectories.

    VAR:
        Uses sampled innovation paths propagated through VAR recursion.

    RNN:
        Uses the model's Gaussian predictive distribution directly.

    Returns
    -------
    dict with:
        forecast_paths: shape (n_paths, h, k)
        innovation_paths: shape (n_paths, h, k)
    """
    model = model.lower()

    if model == "var":
        if innovation_model is None:
            raise ValueError(
                "VAR forecast generation requires an innovation_model."
            )

        result = _generate_var_forecasts(
            fitted_model=fitted_model,
            innovation_model=innovation_model,
            y_history=y_history,
            h=h,
            n_paths=n_paths,
            seed=seed,
        )

    elif model in {"rnn", "deepar"}:
        result = _generate_rnn_forecasts(
            fitted_model=fitted_model,
            y_history=y_history,
            h=h,
            n_paths=n_paths,
            seed=seed,
        )

    else:
        raise ValueError(
            "model must be one of: var, rnn, deepar."
        )

    if save and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            innovation_name = result.get(
                "innovation_model",
                "predictive_distribution",
            )
            filename = f"{model}_{innovation_name}_forecast_paths.npz"

        save_array_npz(
            output_dir / filename,
            forecast_paths=result["forecast_paths"],
            innovation_paths=result["innovation_paths"],
        )

        save_json(
            {
                "forecast_model": model,
                "innovation_model": result.get(
                    "innovation_model",
                    "predictive_distribution",
                ),
                "horizon": h,
                "n_paths": n_paths,
                "seed": seed,
            },
            output_dir / filename.replace(".npz", "_metadata.json"),
        )

    return result


def _generate_var_forecasts(
    fitted_model: dict,
    innovation_model: dict,
    y_history: np.ndarray,
    h: int,
    n_paths: int,
    seed: int | None = None,
) -> dict:
    innovation_paths = sample_innovations(
        innovation_model=innovation_model,
        n_paths=n_paths,
        horizon=h,
        seed=seed,
    )

    forecast_paths = simulate_forecast_paths(
        y_history=y_history,
        beta=fitted_model["beta"],
        innovation_paths=innovation_paths,
        lags=fitted_model["lags"],
        include_intercept=fitted_model.get("include_intercept", True),
    )

    return {
        "forecast_paths": forecast_paths,
        "innovation_paths": innovation_paths,
        "horizon": h,
        "n_paths": n_paths,
        "forecast_model": "var",
        "innovation_model": innovation_model["method"],
    }


def _generate_rnn_forecasts(
    fitted_model: dict,
    y_history: np.ndarray,
    h: int,
    n_paths: int,
    seed: int | None = None,
) -> dict:
    if h != fitted_model["prediction_length"]:
        raise ValueError(
            "For RNN forecasts, h must match fitted_model['prediction_length']. "
            f"Got h={h}, prediction_length={fitted_model['prediction_length']}."
        )

    context_length = fitted_model["context_length"]

    if len(y_history) < context_length:
        raise ValueError(
            f"RNN forecast requires at least context_length={context_length} "
            "observations in y_history."
        )

    context = y_history[-context_length:]

    rnn_result = forecast_rnn_paths(
        model=fitted_model["model"],
        context=context,
        n_paths=n_paths,
        device=fitted_model.get("device", "cpu"),
        seed=seed,
    )

    forecast_paths = rnn_result["forecast_paths"]
    mean = rnn_result["mean"]

    predictive_deviations = forecast_paths - mean[None, :, :]

    return {
        "forecast_paths": forecast_paths,
        "innovation_paths": predictive_deviations,
        "predictive_mean": mean,
        "predictive_scale": rnn_result["scale"],
        "horizon": h,
        "n_paths": n_paths,
        "forecast_model": "rnn",
        "innovation_model": "rnn_gaussian_predictive",
    }

def generate_rnn_residual_injection_forecasts(
    fitted_model: dict,
    y_history: np.ndarray,
    residual_shocks: np.ndarray,
) -> dict:
    """
    Generate RNN forecast paths by injecting pre-sampled residual shocks around
    the RNN predictive mean.

    Parameters
    ----------
    fitted_model:
        Fitted RNN model dictionary.

    y_history:
        Training/history series used to form the RNN context.

    residual_shocks:
        Shape (n_paths, horizon, k).

    Returns
    -------
    dict with forecast_paths, innovation_paths, predictive_mean, predictive_scale.
    """
    n_paths, horizon, k = residual_shocks.shape

    if horizon != fitted_model["prediction_length"]:
        raise ValueError(
            "Residual shock horizon must match fitted_model['prediction_length']. "
            f"Got horizon={horizon}, prediction_length={fitted_model['prediction_length']}."
        )

    context_length = fitted_model["context_length"]

    if len(y_history) < context_length:
        raise ValueError(
            f"RNN residual injection requires at least context_length={context_length} "
            "observations in y_history."
        )

    context = y_history[-context_length:]

    base_result = forecast_rnn_paths(
        model=fitted_model["model"],
        context=context,
        n_paths=1,
        device=fitted_model.get("device", "cpu"),
        seed=None,
    )

    predictive_mean = base_result["mean"]
    predictive_scale = base_result["scale"]

    forecast_paths = (
        predictive_mean[None, :, :]
        + residual_shocks
    )

    return {
        "forecast_paths": forecast_paths,
        "innovation_paths": residual_shocks,
        "predictive_mean": predictive_mean,
        "predictive_scale": predictive_scale,
    }

def generate_rnn_forecast_experiments(
    datasets: dict,
    rnn_fits: dict,
    prediction_length: int,
    n_paths: int,
    seed: int,
    output_dir,
) -> dict:
    """
    Generate probabilistic RNN forecasts across all DGP datasets.
    """
    from pathlib import Path

    output_dir = Path(output_dir)

    rnn_forecasts = {}

    for name, data in datasets.items():
        print(f"\nGenerating RNN forecasts for: {name}")

        forecast = generate_forecasts(
            fitted_model=rnn_fits[name],
            innovation_model=None,
            y_history=data["y_train"],
            h=prediction_length,
            n_paths=n_paths,
            model="rnn",
            seed=seed,
            save=True,
            output_dir=output_dir / name / "rnn",
            filename=f"{name}_rnn_forecast_paths.npz",
        )

        rnn_forecasts[name] = forecast

        print(
            name,
            forecast["forecast_paths"].shape,
            forecast["predictive_mean"].shape,
        )

    return rnn_forecasts

def prepare_innovation_paths_for_forecasting(
    forecast_models,
    dgp_names,
    n_paths,
    horizon,
    seed,
    student_t_df,
    residual_dir,
    diffusion_dir,
) -> tuple[dict, dict]:
    """
    Fit/sample classical innovation models and load diffusion samples,
    reshaping all innovation draws to (n_paths, horizon, k).
    """
    from pathlib import Path

    from innovcal.experiments.artifacts import load_array_npz
    from innovcal.api.innovations import fit_innovations, sample_innovations

    residual_dir = Path(residual_dir)
    diffusion_dir = Path(diffusion_dir)

    innovation_models = {}
    innovation_paths = {}

    for forecast_model in forecast_models:
        innovation_models[forecast_model] = {}
        innovation_paths[forecast_model] = {}

        for dgp_name in dgp_names:
            innovation_models[forecast_model][dgp_name] = {}
            innovation_paths[forecast_model][dgp_name] = {}

            residual_path = (
                residual_dir
                / f"{dgp_name}_{forecast_model.lower()}_residuals.npz"
            )

            residual_data = load_array_npz(residual_path)
            residuals = residual_data["residuals"]

            for method in [
                "gaussian",
                "bootstrap",
                "student_t",
            ]:
                kwargs = {}

                if method == "student_t":
                    kwargs["df"] = student_t_df

                innov_model = fit_innovations(
                    residuals=residuals,
                    method=method,
                    **kwargs,
                )

                shocks = sample_innovations(
                    innovation_model=innov_model,
                    n_paths=n_paths,
                    horizon=horizon,
                    seed=seed,
                )

                innovation_models[forecast_model][dgp_name][method] = innov_model
                innovation_paths[forecast_model][dgp_name][method] = shocks

                print(
                    forecast_model,
                    dgp_name,
                    method,
                    shocks.shape,
                )

            diffusion_path = (
                diffusion_dir
                / forecast_model.lower()
                / dgp_name
                / "diffusion_samples.npz"
            )

            diffusion_data = load_array_npz(diffusion_path)
            diffusion_samples = diffusion_data["innovations"]

            k = diffusion_samples.shape[1]
            needed = n_paths * horizon

            if diffusion_samples.shape[0] < needed:
                raise ValueError(
                    f"Not enough diffusion samples for {forecast_model} | {dgp_name}. "
                    f"Need {needed}, got {diffusion_samples.shape[0]}."
                )

            diffusion_shocks = diffusion_samples[:needed].reshape(
                n_paths,
                horizon,
                k,
            )

            innovation_paths[forecast_model][dgp_name]["diffusion"] = (
                diffusion_shocks
            )

            print(
                forecast_model,
                dgp_name,
                "diffusion",
                diffusion_shocks.shape,
            )

    return innovation_models, innovation_paths

def generate_forecast_experiments(
    datasets,
    fitted_models,
    innovation_paths,
    forecast_models,
    dgp_names,
    innovation_model_names,
    lags,
    horizon,
    n_paths,
    output_dir,
):
    """
    Generate forecast paths for VAR and RNN using pre-sampled innovation paths.
    """
    from pathlib import Path

    from innovcal.experiments.artifacts import save_array_npz
    from innovcal.forecasting.monte_carlo import simulate_forecast_paths

    output_dir = Path(output_dir)

    forecast_store = {}

    for forecast_model in forecast_models:
        forecast_store[forecast_model] = {}

        for dgp_name in dgp_names:
            forecast_store[forecast_model][dgp_name] = {}

            y_history = datasets[dgp_name]["y_train"]

            for innovation_model in innovation_model_names:

                shocks = innovation_paths[
                    forecast_model
                ][
                    dgp_name
                ][
                    innovation_model
                ]

                if forecast_model == "VAR":

                    paths = simulate_forecast_paths(
                        y_history=y_history[-lags:],
                        beta=fitted_models[forecast_model][dgp_name]["beta"],
                        innovation_paths=shocks,
                        lags=lags,
                        include_intercept=True,
                    )

                    forecast = {
                        "forecast_paths": paths,
                        "innovation_paths": shocks,
                        "horizon": horizon,
                        "n_paths": n_paths,
                        "forecast_model": "VAR",
                        "innovation_model": innovation_model,
                        "y_true": datasets[dgp_name]["y_test"],
                    }

                elif forecast_model == "RNN":

                    rnn_result = generate_rnn_residual_injection_forecasts(
                        fitted_model=fitted_models[forecast_model][dgp_name],
                        y_history=y_history,
                        residual_shocks=shocks,
                    )

                    forecast = {
                        "forecast_paths": rnn_result["forecast_paths"],
                        "innovation_paths": rnn_result["innovation_paths"],
                        "predictive_mean": rnn_result["predictive_mean"],
                        "predictive_scale": rnn_result["predictive_scale"],
                        "horizon": horizon,
                        "n_paths": n_paths,
                        "forecast_model": "RNN",
                        "innovation_model": innovation_model,
                        "y_true": datasets[dgp_name]["y_test"],
                    }

                else:
                    raise ValueError(
                        "forecast_model must be one of: 'VAR', 'RNN'."
                    )

                forecast_store[
                    forecast_model
                ][
                    dgp_name
                ][
                    innovation_model
                ] = forecast

                save_array_npz(
                    (
                        output_dir
                        / forecast_model.lower()
                        / dgp_name
                        / f"{innovation_model}_forecast_paths.npz"
                    ),
                    forecast_paths=forecast["forecast_paths"],
                    innovation_paths=forecast["innovation_paths"],
                    y_true=forecast["y_true"],
                )

                print(
                    forecast_model,
                    dgp_name,
                    innovation_model,
                    forecast["forecast_paths"].shape,
                )

    return forecast_store

def forecast_summary_table(
    forecast_store,
    datasets,
    forecast_models,
    dgp_names,
    innovation_model_names,
    interval=(0.05, 0.95),
    nominal_levels=(0.5, 0.8, 0.9),
):
    import pandas as pd

    from innovcal.evaluation.metrics import (
        summarize_probabilistic_forecast,
        make_summary_row,
    )

    summary_rows = []

    for forecast_model in forecast_models:
        for dgp_name in dgp_names:

            y_true = datasets[dgp_name]["y_test"]

            for innovation_model in innovation_model_names:

                paths = forecast_store[
                    forecast_model
                ][
                    dgp_name
                ][
                    innovation_model
                ]["forecast_paths"]

                summary = summarize_probabilistic_forecast(
                    forecast_paths=paths,
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

                summary_rows.append(row)

    return pd.DataFrame(summary_rows)

def forecast_interval_width_table(
    forecast_store,
    forecast_models,
    dgp_names,
    innovation_model_names,
):
    import pandas as pd

    from innovcal.forecasting.intervals import (
        prediction_interval,
        average_interval_width,
    )

    width_rows = []

    for forecast_model in forecast_models:
        for dgp_name in dgp_names:
            for innovation_model in innovation_model_names:

                paths = forecast_store[
                    forecast_model
                ][
                    dgp_name
                ][
                    innovation_model
                ]["forecast_paths"]

                lower, upper = prediction_interval(
                    paths,
                    lower_q=0.05,
                    upper_q=0.95,
                )

                width = average_interval_width(
                    lower=lower,
                    upper=upper,
                )

                width_rows.append(
                    {
                        "forecast_model": forecast_model,
                        "dgp": dgp_name,
                        "innovation_model": innovation_model,
                        "avg_width": float(width.mean()),
                        "width_1": float(width[0]),
                        "width_2": float(width[1]),
                        "width_3": float(width[2]),
                    }
                )

    return pd.DataFrame(width_rows)

def best_forecast_models_table(
    forecast_summary_df,
    metrics=None,
):
    import pandas as pd

    if metrics is None:
        metrics = [
            "energy_score",
            "crps",
            "interval_score",
            "ece",
            "pit_deviation",
        ]

    best_rows = []

    for dgp_name in forecast_summary_df["dgp"].unique():

        subset = forecast_summary_df[
            forecast_summary_df["dgp"] == dgp_name
        ]

        for metric in metrics:

            best = subset.loc[
                subset[metric].idxmin()
            ]

            best_rows.append(
                {
                    "dgp": dgp_name,
                    "metric": metric,
                    "best_forecast_model": best["forecast_model"],
                    "best_innovation_model": best["innovation_model"],
                    "best_value": best[metric],
                }
            )

    return pd.DataFrame(
        best_rows
    )

def load_forecast_store(
    forecast_dir,
    forecast_models,
    dgp_names,
    innovation_model_names,
):
    from pathlib import Path

    from innovcal.experiments.artifacts import load_array_npz

    forecast_dir = Path(forecast_dir)

    forecast_store = {}

    for forecast_model in forecast_models:
        forecast_store[forecast_model] = {}

        for dgp_name in dgp_names:
            forecast_store[forecast_model][dgp_name] = {}

            for innovation_model in innovation_model_names:

                path = (
                    forecast_dir
                    / forecast_model.lower()
                    / dgp_name
                    / f"{innovation_model}_forecast_paths.npz"
                )

                data = load_array_npz(path)

                forecast_store[forecast_model][dgp_name][innovation_model] = {
                    "forecast_paths": data["forecast_paths"],
                    "innovation_paths": data["innovation_paths"],
                    "y_true": data["y_true"],
                }

                print(
                    forecast_model,
                    dgp_name,
                    innovation_model,
                    data["forecast_paths"].shape,
                    data["y_true"].shape,
                )

    return forecast_store