from pathlib import Path

import numpy as np
import torch

from src.innovations.gaussian import (
    fit_gaussian_innovation_model,
    sample_from_gaussian_model,
)

from src.innovations.bootstrap import (
    fit_bootstrap_innovation_model,
    sample_from_bootstrap_model,
)

from src.innovations.student_t import (
    fit_student_t_innovation_model,
    sample_from_student_t_model,
)

from src.innovations.residuals import standardize_residuals

from src.diffusion.networks import ResidualDenoisingMLP
from src.diffusion.schedules import make_ddpm_schedule
from src.diffusion.trainer import train_diffusion_model
from src.diffusion.sampler import sample_diffusion_innovations

from src.experiments.artifacts import save_json, save_array_npz


def _json_safe(obj):
    if isinstance(obj, torch.device):
        return str(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    if isinstance(obj, dict):
        return {
            key: _json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            _json_safe(value)
            for value in obj
        ]

    return obj


def fit_innovations(
    residuals: np.ndarray,
    method: str = "gaussian",
    save: bool = False,
    output_dir: str | Path | None = None,
    **kwargs,
) -> dict:
    """
    Fit a forecast innovation model to residuals.

    Supported methods:
        gaussian
        bootstrap
        student_t
        diffusion
    """
    residuals = np.asarray(residuals, dtype=float)
    method = method.lower()

    if method == "gaussian":
        result = _fit_gaussian(residuals)

    elif method == "bootstrap":
        result = _fit_bootstrap(residuals)

    elif method == "student_t":
        result = _fit_student_t(
            residuals,
            df=kwargs.get("df", 5.0),
        )

    elif method == "diffusion":
        result = _fit_diffusion(
            residuals,
            timesteps=kwargs.get("timesteps", 200),
            beta_start=kwargs.get("beta_start", 1e-4),
            beta_end=kwargs.get("beta_end", 2e-2),
            epochs=kwargs.get("epochs", 1500),
            lr=kwargs.get("lr", 5e-4),
            hidden_dim=kwargs.get("hidden_dim", 256),
            time_embedding_dim=kwargs.get("time_embedding_dim", 64),
            batch_size=kwargs.get("batch_size", 64),
            device=kwargs.get("device", "cpu"),
            verbose=kwargs.get("verbose", True),
        )

    else:
        raise ValueError(
            "method must be one of: gaussian, bootstrap, student_t, diffusion."
        )

    if save and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_json(
            {
                "method": method,
                "n_residuals": int(residuals.shape[0]),
                "n_series": int(residuals.shape[1]),
                "kwargs": _json_safe(kwargs),
            },
            output_dir / f"{method}_innovation_metadata.json",
        )

    return result


def sample_innovations(
    innovation_model: dict,
    n_paths: int,
    horizon: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample forecast innovation paths.

    Returns
    -------
    shocks:
        Shape (n_paths, horizon, k)
    """
    return innovation_model["sample_fn"](
        n_paths=n_paths,
        horizon=horizon,
        seed=seed,
    )


def _fit_gaussian(
    residuals: np.ndarray,
) -> dict:
    model = fit_gaussian_innovation_model(
        residuals
    )

    def sample_fn(
        n_paths: int,
        horizon: int,
        seed: int | None = None,
    ) -> np.ndarray:
        return sample_from_gaussian_model(
            model=model,
            n_paths=n_paths,
            horizon=horizon,
            seed=seed,
        )

    return {
        "method": "gaussian",
        "model": model,
        "sample_fn": sample_fn,
    }


def _fit_bootstrap(
    residuals: np.ndarray,
) -> dict:
    model = fit_bootstrap_innovation_model(
        residuals
    )

    def sample_fn(
        n_paths: int,
        horizon: int,
        seed: int | None = None,
    ) -> np.ndarray:
        return sample_from_bootstrap_model(
            model=model,
            n_paths=n_paths,
            horizon=horizon,
            seed=seed,
        )

    return {
        "method": "bootstrap",
        "model": model,
        "sample_fn": sample_fn,
    }


def _fit_student_t(
    residuals: np.ndarray,
    df: float = 5.0,
) -> dict:
    model = fit_student_t_innovation_model(
        residuals=residuals,
        df=df,
    )

    def sample_fn(
        n_paths: int,
        horizon: int,
        seed: int | None = None,
    ) -> np.ndarray:
        return sample_from_student_t_model(
            model=model,
            n_paths=n_paths,
            horizon=horizon,
            seed=seed,
        )

    return {
        "method": "student_t",
        "model": model,
        "df": df,
        "sample_fn": sample_fn,
    }


def _fit_diffusion(
    residuals: np.ndarray,
    timesteps: int = 200,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    epochs: int = 1500,
    lr: float = 5e-4,
    hidden_dim: int = 256,
    time_embedding_dim: int = 64,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict:
    residuals_std, mean, std = standardize_residuals(
        residuals
    )

    device = torch.device(device)

    schedule = make_ddpm_schedule(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule_type="linear",
        device=device,
    )

    model = ResidualDenoisingMLP(
        input_dim=residuals.shape[1],
        hidden_dim=hidden_dim,
        time_embedding_dim=time_embedding_dim,
        num_hidden_layers=2,
        dropout=0.0,
    ).to(device)

    history = train_diffusion_model(
        model=model,
        residuals=residuals_std,
        schedule=schedule,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        verbose=verbose,
    )

    def sample_fn(
        n_paths: int,
        horizon: int,
        seed: int | None = None,
    ) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        return sample_diffusion_innovations(
            model=model,
            n_paths=n_paths,
            horizon=horizon,
            input_dim=residuals.shape[1],
            schedule=schedule,
            device=device,
            mean=mean,
            std=std,
        )

    return {
        "method": "diffusion",
        "model": model,
        "schedule": schedule,
        "history": history,
        "mean": mean,
        "std": std,
        "timesteps": timesteps,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "sample_fn": sample_fn,
    }


def save_sampled_innovations(
    innovation_model: dict,
    n_paths: int,
    horizon: int,
    output_dir: str | Path,
    filename: str | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample and save innovation paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shocks = sample_innovations(
        innovation_model=innovation_model,
        n_paths=n_paths,
        horizon=horizon,
        seed=seed,
    )

    if filename is None:
        filename = f"{innovation_model['method']}_innovations.npz"

    save_array_npz(
        output_dir / filename,
        innovations=shocks,
    )

    return shocks

def fit_and_sample_classical_innovations(
    residual_store: dict,
    forecast_models: list[str],
    dgp_names: list[str],
    innovation_model_names: list[str],
    n_paths: int,
    horizon: int,
    seed: int,
    output_dir,
    student_t_df: float = 5.0,
) -> tuple[dict, dict]:
    """
    Fit and sample classical innovation models across forecast models and DGPs.
    """
    from pathlib import Path

    from src.experiments.artifacts import save_array_npz

    output_dir = Path(output_dir)

    innovation_models = {}
    sampled_innovations = {}

    for forecast_model in forecast_models:
        innovation_models[forecast_model] = {}
        sampled_innovations[forecast_model] = {}

        for dgp_name in dgp_names:
            residuals = residual_store[forecast_model][dgp_name]

            innovation_models[forecast_model][dgp_name] = {}
            sampled_innovations[forecast_model][dgp_name] = {}

            for method in innovation_model_names:

                kwargs = {}

                if method == "student_t":
                    kwargs["df"] = student_t_df

                fitted_innovation = fit_innovations(
                    residuals=residuals,
                    method=method,
                    save=True,
                    output_dir=(
                        output_dir
                        / forecast_model.lower()
                        / dgp_name
                        / method
                    ),
                    **kwargs,
                )

                samples = sample_innovations(
                    innovation_model=fitted_innovation,
                    n_paths=n_paths,
                    horizon=horizon,
                    seed=seed,
                )

                innovation_models[forecast_model][dgp_name][method] = (
                    fitted_innovation
                )

                sampled_innovations[forecast_model][dgp_name][method] = (
                    samples
                )

                save_array_npz(
                    (
                        output_dir
                        / forecast_model.lower()
                        / dgp_name
                        / f"{method}_samples.npz"
                    ),
                    innovations=samples,
                )

                print(
                    forecast_model,
                    dgp_name,
                    method,
                    samples.shape,
                )

    return innovation_models, sampled_innovations

def innovation_diagnostics_table(
    residual_store,
    sampled_innovations,
    forecast_models,
    dgp_names,
    innovation_model_names,
):
    import pandas as pd

    from src.innovations.diagnostics import summarize_innovations

    diagnostic_rows = []

    for forecast_model in forecast_models:

        for dgp_name in dgp_names:

            residuals = residual_store[
                forecast_model
            ][
                dgp_name
            ]

            residual_diag = summarize_innovations(
                residuals
            )

            for j in range(residuals.shape[1]):
                diagnostic_rows.append(
                    {
                        "forecast_model": forecast_model,
                        "dgp": dgp_name,
                        "innovation_model": "empirical_residuals",
                        "series": j + 1,
                        "mean": residual_diag["mean"][j],
                        "std": residual_diag["std"][j],
                        "skewness": residual_diag["skewness"][j],
                        "kurtosis": residual_diag["kurtosis"][j],
                        "excess_kurtosis": residual_diag["excess_kurtosis"][j],
                        "jb_pvalue": residual_diag["jarque_bera_pvalue"][j],
                    }
                )

            for method in innovation_model_names:

                samples = sampled_innovations[
                    forecast_model
                ][
                    dgp_name
                ][
                    method
                ]

                flat_samples = samples.reshape(
                    -1,
                    samples.shape[-1],
                )

                diag = summarize_innovations(
                    flat_samples
                )

                for j in range(flat_samples.shape[1]):
                    diagnostic_rows.append(
                        {
                            "forecast_model": forecast_model,
                            "dgp": dgp_name,
                            "innovation_model": method,
                            "series": j + 1,
                            "mean": diag["mean"][j],
                            "std": diag["std"][j],
                            "skewness": diag["skewness"][j],
                            "kurtosis": diag["kurtosis"][j],
                            "excess_kurtosis": diag["excess_kurtosis"][j],
                            "jb_pvalue": diag["jarque_bera_pvalue"][j],
                        }
                    )

    return pd.DataFrame(
        diagnostic_rows
    )

def innovation_diagnostics_summary(
    innovation_diagnostics_df,
):
    return (
        innovation_diagnostics_df
        .groupby(
            [
                "forecast_model",
                "dgp",
                "innovation_model",
            ]
        )[
            [
                "mean",
                "std",
                "skewness",
                "kurtosis",
                "excess_kurtosis",
                "jb_pvalue",
            ]
        ]
        .mean()
        .reset_index()
    )

def innovation_recovery_summary(
    innovation_diagnostics_df,
    forecast_models,
    dgp_names,
    innovation_model_names,
):
    import numpy as np
    import pandas as pd

    recovery_rows = []

    for forecast_model in forecast_models:
        for dgp_name in dgp_names:

            empirical = (
                innovation_diagnostics_df[
                    (innovation_diagnostics_df["forecast_model"] == forecast_model)
                    &
                    (innovation_diagnostics_df["dgp"] == dgp_name)
                    &
                    (
                        innovation_diagnostics_df["innovation_model"]
                        == "empirical_residuals"
                    )
                ]
            )

            for method in innovation_model_names:

                fitted = (
                    innovation_diagnostics_df[
                        (innovation_diagnostics_df["forecast_model"] == forecast_model)
                        &
                        (innovation_diagnostics_df["dgp"] == dgp_name)
                        &
                        (
                            innovation_diagnostics_df["innovation_model"]
                            == method
                        )
                    ]
                )

                recovery_rows.append(
                    {
                        "forecast_model": forecast_model,
                        "dgp": dgp_name,
                        "innovation_model": method,
                        "kurtosis_error":
                            np.mean(
                                np.abs(
                                    fitted["kurtosis"].values
                                    -
                                    empirical["kurtosis"].values
                                )
                            ),
                        "skewness_error":
                            np.mean(
                                np.abs(
                                    fitted["skewness"].values
                                    -
                                    empirical["skewness"].values
                                )
                            ),
                    }
                )

    return pd.DataFrame(recovery_rows)