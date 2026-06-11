from pathlib import Path

import numpy as np

from src.api.innovations import sample_innovations
from src.forecasting.monte_carlo import simulate_forecast_paths
from src.deep_ar.forecast import forecast_rnn_paths
from src.experiments.artifacts import save_array_npz, save_json


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