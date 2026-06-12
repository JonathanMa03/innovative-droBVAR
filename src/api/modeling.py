from pathlib import Path
import pandas as pd
import numpy as np
import torch

from src.vector_ar.fit import fit_var_ols
from src.innovations.residuals import extract_residuals as _extract_residuals

from src.deep_ar.dataset import make_rnn_dataloader
from src.deep_ar.model import ProbabilisticRNN
from src.deep_ar.trainer import train_rnn_forecaster, save_rnn_model
from src.deep_ar.forecast import rnn_predict_distribution

from src.experiments.artifacts import (
    save_json,
    save_array_npz,
)


def fit_model(
    data: np.ndarray,
    model: str = "var",
    save: bool = False,
    output_dir: str | Path | None = None,
    **kwargs,
):
    """
    High-level model fitting interface.

    Supported models:
        "var"
        "rnn"
        "deepar"  # alias for rnn
    """
    data = np.asarray(data, dtype=float)
    model = model.lower()

    if model == "var":
        fitted_model = fit_var_ols(
            y=data,
            **kwargs,
        )

    elif model in {"rnn", "deepar"}:
        fitted_model = _fit_rnn_model(
            data=data,
            save=save,
            output_dir=output_dir,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown model: {model}"
        )

    if save and output_dir is not None and model == "var":
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model": model,
            "n_obs": int(data.shape[0]),
            "n_series": int(data.shape[1]),
            "kwargs": kwargs,
        }

        save_json(
            metadata,
            output_dir / f"{model}_fit_metadata.json",
        )

        save_array_npz(
            output_dir / f"{model}_fit.npz",
            beta=fitted_model["beta"],
            fitted=fitted_model["fitted"],
            residuals=fitted_model["residuals"],
            Sigma_hat=fitted_model["Sigma_hat"],
            Y=fitted_model["Y"],
            X=fitted_model["X"],
        )

    return fitted_model

def fit_var_experiments(
    datasets: dict,
    lags: int,
    output_dir,
) -> tuple[dict, pd.DataFrame]:
    """
    Fit VAR models across all DGP datasets and summarize fit diagnostics.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd

    from src.vector_ar.stability import stability_summary

    output_dir = Path(output_dir)

    var_fits = {}
    fit_rows = []

    for name, data in datasets.items():
        y_train = data["y_train"]

        fit = fit_model(
            data=y_train,
            model="var",
            lags=lags,
            include_intercept=True,
            save=True,
            output_dir=output_dir / name,
        )

        k = y_train.shape[1]

        beta_no_intercept = (
            fit["beta"][1:]
            if fit["include_intercept"]
            else fit["beta"]
        )

        stability = stability_summary(
            beta_no_intercept=beta_no_intercept,
            k=k,
            lags=lags,
        )

        var_fits[name] = {
            "fit": fit,
            "stability": stability,
        }

        fit_rows.append(
            {
                "dgp": name,
                "model": "VAR",
                "lags": lags,
                "n_train": len(y_train),
                "n_test": len(data["y_test"]),
                "stable": stability["stable"],
                "max_modulus": stability["max_modulus"],
                "resid_mean_abs": float(
                    np.mean(
                        np.abs(
                            fit["residuals"].mean(axis=0)
                        )
                    )
                ),
                "resid_avg_std": float(
                    fit["residuals"]
                    .std(axis=0, ddof=1)
                    .mean()
                ),
            }
        )

    fit_summary_df = pd.DataFrame(fit_rows)

    return var_fits, fit_summary_df

def _fit_rnn_model(
    data: np.ndarray,
    context_length: int = 40,
    prediction_length: int = 40,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    verbose: bool = True,
    save: bool = False,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Fit a probabilistic GRU/RNN forecaster.
    """
    device = torch.device(device)

    dataloader = make_rnn_dataloader(
        y=data,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        shuffle=True,
    )

    rnn = ProbabilisticRNN(
        input_dim=data.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        prediction_length=prediction_length,
        dropout=dropout,
    )

    history = train_rnn_forecaster(
        model=rnn,
        dataloader=dataloader,
        epochs=epochs,
        lr=lr,
        device=device,
        verbose=verbose,
    )

    fitted_model = {
        "model_type": "rnn",
        "model": rnn,
        "history": history,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "input_dim": data.shape[1],
        "device": str(device),
        "fit_data": data,
        "kwargs": {
            "context_length": context_length,
            "prediction_length": prediction_length,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
        },
    }

    if save and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_rnn_model(
            rnn,
            output_dir / "rnn_model.pt",
        )

        save_json(
            {
                "model": "rnn",
                "n_obs": int(data.shape[0]),
                "n_series": int(data.shape[1]),
                "context_length": context_length,
                "prediction_length": prediction_length,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "device": str(device),
            },
            output_dir / "rnn_fit_metadata.json",
        )

        save_array_npz(
            output_dir / "rnn_training_history.npz",
            loss=np.asarray(history["loss"]),
        )

    return fitted_model


def extract_residuals(
    fitted_model: dict,
    model: str = "var",
    data: np.ndarray | None = None,
    save: bool = False,
    output_dir: str | Path | None = None,
) -> np.ndarray:
    """
    Extract model residuals.

    For VAR:
        Returns in-sample one-step residuals from OLS fit.

    For RNN:
        Returns rolling one-step residuals using the trained probabilistic RNN.
    """
    model = model.lower()

    if model == "var":
        residuals = _extract_residuals(
            fitted_model
        )

    elif model in {"rnn", "deepar"}:
        if data is None:
            data = fitted_model.get("fit_data")

        if data is None:
            raise ValueError(
                "For RNN residual extraction, provide data or fit with fit_data stored."
            )

        residuals = _extract_rnn_residuals(
            fitted_model=fitted_model,
            data=data,
        )

    else:
        raise ValueError(
            f"Unknown model: {model}"
        )

    if save and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_array_npz(
            output_dir / f"{model}_residuals.npz",
            residuals=residuals,
        )

    return residuals


def _extract_rnn_residuals(
    fitted_model: dict,
    data: np.ndarray,
) -> np.ndarray:
    """
    Rolling one-step RNN residuals.

    For each t >= context_length:
        context = y[t-context_length:t]
        prediction = mean forecast at horizon 1
        residual = y[t] - prediction
    """
    data = np.asarray(data, dtype=float)

    model = fitted_model["model"]
    context_length = fitted_model["context_length"]
    device = fitted_model.get("device", "cpu")

    residuals = []

    for t in range(context_length, len(data)):
        context = data[t - context_length:t]

        mean, _ = rnn_predict_distribution(
            model=model,
            context=context,
            device=device,
        )

        pred = mean[0]
        resid = data[t] - pred

        residuals.append(resid)

    return np.asarray(residuals)

def generate_var_mean_forecasts(
    datasets: dict,
    var_fits: dict,
    lags: int,
    horizon: int,
    output_dir,
) -> dict:
    """
    Generate mean VAR forecasts for each fitted DGP.
    """
    from pathlib import Path

    from src.vector_ar.forecast import forecast_var_mean
    from src.experiments.artifacts import save_array_npz

    output_dir = Path(output_dir)

    mean_forecasts = {}

    for name, data in datasets.items():
        fit = var_fits[name]["fit"]

        y_history = data["y_train"][-lags:]

        mean_forecast = forecast_var_mean(
            y_history=y_history,
            beta=fit["beta"],
            horizon=horizon,
            lags=lags,
            include_intercept=fit["include_intercept"],
        )

        mean_forecasts[name] = mean_forecast

        save_array_npz(
            output_dir / name / "mean_forecast.npz",
            forecast=mean_forecast,
        )

    return mean_forecasts

def fit_rnn_experiments(
    datasets: dict,
    output_dir,
    context_length: int,
    prediction_length: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    device,
):
    import pandas as pd

    rnn_fits = {}
    rnn_rows = []

    for name, data in datasets.items():
        print(f"\nFitting RNN for: {name}")

        fit = fit_model(
            data=data["y_train"],
            model="rnn",
            context_length=context_length,
            prediction_length=prediction_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            device=device,
            verbose=True,
            save=True,
            output_dir=output_dir / name / "rnn",
        )

        rnn_fits[name] = fit

        rnn_rows.append(
            {
                "dgp": name,
                "model": "RNN",
                "context_length": context_length,
                "prediction_length": prediction_length,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "epochs": epochs,
                "final_loss": fit["history"]["loss"][-1],
            }
        )

    return rnn_fits, pd.DataFrame(rnn_rows)

def compare_mean_forecasts(
    datasets,
    mean_forecasts,
    rnn_forecasts,
    horizon,
):
    import numpy as np
    import pandas as pd

    rows = []

    for name in datasets:

        y_test_h = datasets[name]["y_test"][:horizon]

        var_mean = mean_forecasts[name]
        rnn_mean = rnn_forecasts[name]["predictive_mean"]

        rows.append(
            {
                "dgp": name,
                "var_mean_mse": float(
                    np.mean((y_test_h - var_mean) ** 2)
                ),
                "rnn_mean_mse": float(
                    np.mean((y_test_h - rnn_mean) ** 2)
                ),
                "var_mean_mae": float(
                    np.mean(np.abs(y_test_h - var_mean))
                ),
                "rnn_mean_mae": float(
                    np.mean(np.abs(y_test_h - rnn_mean))
                ),
            }
        )

    return pd.DataFrame(rows)

def fit_forecasting_models(
    datasets,
    forecast_models,
    lags,
    rnn_context_length,
    rnn_prediction_length,
    rnn_hidden_dim,
    rnn_num_layers,
    rnn_dropout,
    rnn_batch_size,
    rnn_epochs,
    rnn_lr,
    rnn_device,
    output_dir,
):
    fitted_models = {}

    for forecast_model in forecast_models:
        fitted_models[forecast_model] = {}

        for dgp_name, data in datasets.items():

            y_train = data["y_train"]

            if forecast_model == "VAR":

                fit = fit_model(
                    data=y_train,
                    model="var",
                    lags=lags,
                    include_intercept=True,
                    save=True,
                    output_dir=output_dir / "var" / dgp_name,
                )

            elif forecast_model == "RNN":

                fit = fit_model(
                    data=y_train,
                    model="rnn",
                    context_length=rnn_context_length,
                    prediction_length=rnn_prediction_length,
                    hidden_dim=rnn_hidden_dim,
                    num_layers=rnn_num_layers,
                    dropout=rnn_dropout,
                    batch_size=rnn_batch_size,
                    epochs=rnn_epochs,
                    lr=rnn_lr,
                    device=rnn_device,
                    verbose=False,
                    save=True,
                    output_dir=output_dir / "rnn" / dgp_name,
                )

            fitted_models[forecast_model][dgp_name] = fit

            print(
                forecast_model,
                dgp_name,
                "fit complete",
            )

    return fitted_models