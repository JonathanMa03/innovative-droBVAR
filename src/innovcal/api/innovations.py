from pathlib import Path

import numpy as np
import torch

from innovcal.innovations.gaussian import (
    fit_gaussian_innovation_model,
    sample_from_gaussian_model,
)

from innovcal.innovations.bootstrap import (
    fit_bootstrap_innovation_model,
    fit_block_bootstrap_innovation_model,
    sample_from_bootstrap_model,
    sample_from_block_bootstrap_model,
)
from innovcal.innovations.volatility import (
    fit_volatility_bootstrap_model,
    sample_from_volatility_bootstrap_model,
)

from innovcal.innovations.student_t import (
    fit_student_t_innovation_model,
    sample_from_student_t_model,
)

from innovcal.innovations.residuals import standardize_residuals
from innovcal.cdi_var.innovation import fit_cdi_innovation_model

from innovcal.diffusion.networks import ResidualDenoisingMLP
from innovcal.diffusion.schedules import make_ddpm_schedule
from innovcal.diffusion.trainer import train_diffusion_model
from innovcal.diffusion.sampler import sample_diffusion_innovations

from innovcal.experiments.artifacts import save_json, save_array_npz


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

    elif method == "block_bootstrap":
        result = _fit_block_bootstrap(
            residuals,
            block_length=kwargs.get("block_length", 10),
        )

    elif method == "volatility_bootstrap":
        result = _fit_volatility_bootstrap(
            residuals,
            span=kwargs.get("volatility_span", 60),
        )

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
            validation_fraction=kwargs.get("validation_fraction", 0.2),
            early_stopping_patience=kwargs.get("early_stopping_patience", 50),
            seed=kwargs.get("seed", 123),
        )

    elif method == "cdi_var":
        result = fit_cdi_innovation_model(
            residuals,
            context_lags=kwargs.get("context_lags", 5),
            volatility_span=kwargs.get("volatility_span", 60),
            validation_fraction=kwargs.get("validation_fraction", 0.2),
            calibration_fraction=kwargs.get("calibration_fraction", 0.15),
            calibration_horizons=kwargs.get("calibration_horizons", (1, 5, 20)),
            calibration_paths=kwargs.get("calibration_paths", 32),
            calibration_shrinkage=kwargs.get("calibration_shrinkage", 0.5),
            calibration_bounds=kwargs.get("calibration_bounds", (0.8, 1.25)),
            adaptive_calibration_window=kwargs.get("adaptive_calibration_window", 12),
            timesteps=kwargs.get("timesteps", 100),
            epochs=kwargs.get("epochs", 300),
            lr=kwargs.get("lr", 5e-4),
            hidden_dim=kwargs.get("hidden_dim", 128),
            time_embedding_dim=kwargs.get("time_embedding_dim", 32),
            batch_size=kwargs.get("batch_size", 64),
            early_stopping_patience=kwargs.get("early_stopping_patience", 40),
            device=kwargs.get("device", "cpu"),
            seed=kwargs.get("seed", 123),
            verbose=kwargs.get("verbose", False),
        )

    else:
        raise ValueError(
            "method must be one of: gaussian, bootstrap, block_bootstrap, "
            "volatility_bootstrap, student_t, diffusion, cdi_var."
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


def _fit_block_bootstrap(
    residuals: np.ndarray,
    block_length: int,
) -> dict:
    model = fit_block_bootstrap_innovation_model(residuals, block_length)

    def sample_fn(n_paths: int, horizon: int, seed: int | None = None) -> np.ndarray:
        return sample_from_block_bootstrap_model(model, n_paths, horizon, seed)

    return {
        "method": "block_bootstrap",
        "model": model,
        "block_length": block_length,
        "sample_fn": sample_fn,
    }


def _fit_volatility_bootstrap(
    residuals: np.ndarray,
    span: int,
) -> dict:
    model = fit_volatility_bootstrap_model(residuals, span=span)

    def sample_fn(n_paths: int, horizon: int, seed: int | None = None) -> np.ndarray:
        return sample_from_volatility_bootstrap_model(model, n_paths, horizon, seed)

    return {
        "method": "volatility_bootstrap",
        "model": model,
        "volatility_span": span,
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
    validation_fraction: float = 0.2,
    early_stopping_patience: int | None = 50,
    seed: int = 123,
) -> dict:
    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in [0, 0.5).")
    split_index = len(residuals)
    if validation_fraction > 0:
        split_index = int(len(residuals) * (1.0 - validation_fraction))
        if split_index < 2 or split_index >= len(residuals):
            raise ValueError("Not enough residuals for diffusion validation.")
    training_residuals = residuals[:split_index]
    validation_residuals = residuals[split_index:]
    residuals_std, mean, std = standardize_residuals(training_residuals)
    validation_std = None
    if len(validation_residuals):
        validation_std = (validation_residuals - mean) / std

    torch.manual_seed(seed)
    np.random.seed(seed)
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
        validation_residuals=validation_std,
        early_stopping_patience=early_stopping_patience,
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
        "validation_fraction": validation_fraction,
        "seed": seed,
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

    from innovcal.experiments.artifacts import save_array_npz

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

    from innovcal.innovations.diagnostics import summarize_innovations

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

def fit_and_sample_diffusion_innovations(
    residual_store: dict,
    forecast_models: list[str],
    dgp_names: list[str],
    output_dir,
    seed: int,
    **diffusion_kwargs,
) -> tuple[dict, dict]:
    """
    Fit diffusion innovation models across forecast models and DGPs,
    then sample flat one-step innovation draws.

    Residual standardization is handled inside _fit_diffusion().
    """
    from pathlib import Path

    from innovcal.experiments.artifacts import save_array_npz

    output_dir = Path(output_dir)

    n_samples = diffusion_kwargs.pop("n_samples")

    diffusion_models = {}
    diffusion_samples = {}

    for forecast_model in forecast_models:
        diffusion_models[forecast_model] = {}
        diffusion_samples[forecast_model] = {}

        for dgp_name in dgp_names:

            residuals = residual_store[forecast_model][dgp_name]

            print(
                f"\nTraining diffusion model:"
                f" {forecast_model} | {dgp_name}"
            )

            diffusion_model = fit_innovations(
                residuals=residuals,
                method="diffusion",
                save=True,
                output_dir=(
                    output_dir
                    / forecast_model.lower()
                    / dgp_name
                    / "diffusion"
                ),
                **diffusion_kwargs,
            )

            samples = sample_innovations(
                innovation_model=diffusion_model,
                n_paths=n_samples,
                horizon=1,
                seed=seed,
            )

            samples = samples[:, 0, :]

            diffusion_models[forecast_model][dgp_name] = diffusion_model
            diffusion_samples[forecast_model][dgp_name] = samples

            save_array_npz(
                (
                    output_dir
                    / forecast_model.lower()
                    / dgp_name
                    / "diffusion_samples.npz"
                ),
                innovations=samples,
            )

            print(
                forecast_model,
                dgp_name,
                samples.shape,
            )

    return diffusion_models, diffusion_samples

def diffusion_diagnostics_table(
    residual_store,
    diffusion_samples,
    forecast_models,
    dgp_names,
):
    import pandas as pd

    from innovcal.innovations.diagnostics import summarize_innovations

    rows = []

    for forecast_model in forecast_models:
        for dgp_name in dgp_names:
            empirical = residual_store[forecast_model][dgp_name]
            generated = diffusion_samples[forecast_model][dgp_name]

            emp_diag = summarize_innovations(empirical)
            gen_diag = summarize_innovations(generated)

            for j in range(empirical.shape[1]):
                rows.append(
                    {
                        "forecast_model": forecast_model,
                        "dgp": dgp_name,
                        "series": j + 1,
                        "source": "empirical_residuals",
                        "mean": emp_diag["mean"][j],
                        "std": emp_diag["std"][j],
                        "skewness": emp_diag["skewness"][j],
                        "kurtosis": emp_diag["kurtosis"][j],
                        "excess_kurtosis": emp_diag["excess_kurtosis"][j],
                        "jb_pvalue": emp_diag["jarque_bera_pvalue"][j],
                    }
                )

                rows.append(
                    {
                        "forecast_model": forecast_model,
                        "dgp": dgp_name,
                        "series": j + 1,
                        "source": "diffusion",
                        "mean": gen_diag["mean"][j],
                        "std": gen_diag["std"][j],
                        "skewness": gen_diag["skewness"][j],
                        "kurtosis": gen_diag["kurtosis"][j],
                        "excess_kurtosis": gen_diag["excess_kurtosis"][j],
                        "jb_pvalue": gen_diag["jarque_bera_pvalue"][j],
                    }
                )

    return pd.DataFrame(rows)

def diffusion_diagnostics_summary(
    diffusion_diag_df,
):
    return (
        diffusion_diag_df
        .groupby(
            [
                "forecast_model",
                "dgp",
                "source",
            ]
        )[
            [
                "mean",
                "std",
                "skewness",
                "kurtosis",
                "excess_kurtosis",
            ]
        ]
        .mean()
        .reset_index()
    )

def diffusion_quantile_recovery_table(
    residual_store,
    diffusion_samples,
    forecast_models,
    dgp_names,
    quantiles,
):
    import numpy as np
    import pandas as pd

    q_rows = []

    for forecast_model in forecast_models:
        for dgp_name in dgp_names:

            empirical = residual_store[
                forecast_model
            ][
                dgp_name
            ]

            generated = diffusion_samples[
                forecast_model
            ][
                dgp_name
            ]

            for j in range(empirical.shape[1]):

                emp_q = np.quantile(
                    empirical[:, j],
                    quantiles,
                )

                gen_q = np.quantile(
                    generated[:, j],
                    quantiles,
                )

                for q, e, g in zip(
                    quantiles,
                    emp_q,
                    gen_q,
                ):
                    q_rows.append(
                        {
                            "forecast_model": forecast_model,
                            "dgp": dgp_name,
                            "series": j + 1,
                            "quantile": q,
                            "empirical": e,
                            "diffusion": g,
                            "absolute_error": abs(e - g),
                        }
                    )

    return pd.DataFrame(q_rows)

def diffusion_moment_error_table(
    residual_store,
    diffusion_samples,
    forecast_models,
    dgp_names,
):
    import numpy as np
    import pandas as pd
    import scipy.stats as stats

    moment_rows = []

    for forecast_model in forecast_models:
        for dgp_name in dgp_names:

            empirical = residual_store[
                forecast_model
            ][
                dgp_name
            ]

            generated = diffusion_samples[
                forecast_model
            ][
                dgp_name
            ]

            emp_mean = empirical.mean(axis=0)
            gen_mean = generated.mean(axis=0)

            emp_cov = np.cov(
                empirical,
                rowvar=False,
            )

            gen_cov = np.cov(
                generated,
                rowvar=False,
            )

            emp_skew = stats.skew(
                empirical,
                axis=0,
            )

            gen_skew = stats.skew(
                generated,
                axis=0,
            )

            emp_kurt = stats.kurtosis(
                empirical,
                axis=0,
                fisher=False,
            )

            gen_kurt = stats.kurtosis(
                generated,
                axis=0,
                fisher=False,
            )

            moment_rows.append(
                {
                    "forecast_model": forecast_model,
                    "dgp": dgp_name,
                    "mean_l2_error":
                        np.linalg.norm(
                            emp_mean - gen_mean
                        ),
                    "cov_frobenius_error":
                        np.linalg.norm(
                            emp_cov - gen_cov,
                            ord="fro",
                        ),
                    "skew_l2_error":
                        np.linalg.norm(
                            emp_skew - gen_skew
                        ),
                    "kurtosis_l2_error":
                        np.linalg.norm(
                            emp_kurt - gen_kurt
                        ),
                }
            )

    return pd.DataFrame(
        moment_rows
    )
