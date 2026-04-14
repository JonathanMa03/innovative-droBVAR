from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, multivariate_normal, skew


def summarize_forecast_draws(
    forecast_draws: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Summarize Monte Carlo forecast draws.

    Parameters
    ----------
    forecast_draws : np.ndarray
        Array of shape (n_draws, h, k).
    alpha : float, default=0.05
        Tail probability for predictive intervals.

    Returns
    -------
    dict
        Keys:
        - "mean": shape (h, k)
        - "lower": shape (h, k)
        - "upper": shape (h, k)
    """
    forecast_draws = np.asarray(forecast_draws, dtype=float)

    if forecast_draws.ndim != 3:
        raise ValueError(
            f"forecast_draws must have shape (n_draws, h, k), got {forecast_draws.shape}"
        )

    mean_fcst = forecast_draws.mean(axis=0)
    lower_fcst = np.quantile(forecast_draws, alpha / 2, axis=0)
    upper_fcst = np.quantile(forecast_draws, 1 - alpha / 2, axis=0)

    return {
        "mean": mean_fcst,
        "lower": lower_fcst,
        "upper": upper_fcst,
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root mean squared error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    return float(np.mean(np.abs(y_true - y_pred)))


def predictive_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Empirical predictive interval coverage.

    Parameters
    ----------
    y_true : np.ndarray
    lower : np.ndarray
    upper : np.ndarray

    Returns
    -------
    float
        Fraction covered.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if not (y_true.shape == lower.shape == upper.shape):
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, lower={lower.shape}, upper={upper.shape}"
        )

    covered = (lower <= y_true) & (y_true <= upper)
    return float(np.mean(covered))


def log_predictive_density_gaussian(
    y_true: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    jitter: float = 1e-8,
) -> float:
    """
    Log predictive density under a Gaussian predictive distribution.

    Parameters
    ----------
    y_true : np.ndarray, shape (k,)
    mean : np.ndarray, shape (k,)
    cov : np.ndarray, shape (k, k)
    jitter : float, default=1e-8
        Diagonal stabilization.

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if y_true.ndim != 1 or mean.ndim != 1:
        raise ValueError("y_true and mean must be 1D arrays.")
    if cov.shape != (len(y_true), len(y_true)):
        raise ValueError(
            f"cov must have shape ({len(y_true)}, {len(y_true)}), got {cov.shape}"
        )

    cov = 0.5 * (cov + cov.T) + jitter * np.eye(cov.shape[0])
    return float(multivariate_normal.logpdf(y_true, mean=mean, cov=cov))


def covariance_frobenius_error(
    X_empirical: np.ndarray,
    X_generated: np.ndarray,
) -> float:
    """
    Frobenius norm error between covariance matrices.
    """
    X_empirical = np.asarray(X_empirical, dtype=float)
    X_generated = np.asarray(X_generated, dtype=float)

    if X_empirical.ndim != 2 or X_generated.ndim != 2:
        raise ValueError("Inputs must be 2D arrays.")

    emp_cov = np.cov(X_empirical.T)
    gen_cov = np.cov(X_generated.T)

    return float(np.linalg.norm(emp_cov - gen_cov, ord="fro"))


def summarize_distribution(
    X: np.ndarray,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Summarize marginals of a multivariate sample.

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
    columns : sequence of str, optional

    Returns
    -------
    pd.DataFrame
        Indexed by variable name or integer.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n, k = X.shape

    if columns is None:
        index = [f"x{j+1}" for j in range(k)]
    else:
        if len(columns) != k:
            raise ValueError(f"Expected {k} column names, got {len(columns)}")
        index = list(columns)

    return pd.DataFrame(
        {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0, ddof=1),
            "skew": [skew(X[:, j]) for j in range(k)],
            "kurtosis": [kurtosis(X[:, j], fisher=False) for j in range(k)],
        },
        index=index,
    )


def compare_distribution_summaries(
    X_generated: np.ndarray,
    X_empirical: np.ndarray,
    columns: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compare generated sample against empirical sample.

    Returns
    -------
    metrics : dict
        Aggregate absolute errors and covariance Frobenius error.
    empirical_summary : pd.DataFrame
    generated_summary : pd.DataFrame
    empirical_cov : np.ndarray
    generated_cov : np.ndarray
    """
    empirical_summary = summarize_distribution(X_empirical, columns=columns)
    generated_summary = summarize_distribution(X_generated, columns=columns)

    empirical_cov = np.cov(np.asarray(X_empirical, dtype=float).T)
    generated_cov = np.cov(np.asarray(X_generated, dtype=float).T)

    metrics = {
        "mean_abs_mean_error": float(
            np.mean(np.abs(generated_summary["mean"] - empirical_summary["mean"]))
        ),
        "mean_abs_std_error": float(
            np.mean(np.abs(generated_summary["std"] - empirical_summary["std"]))
        ),
        "mean_abs_skew_error": float(
            np.mean(np.abs(generated_summary["skew"] - empirical_summary["skew"]))
        ),
        "mean_abs_kurtosis_error": float(
            np.mean(np.abs(generated_summary["kurtosis"] - empirical_summary["kurtosis"]))
        ),
        "cov_frobenius_error": float(
            np.linalg.norm(empirical_cov - generated_cov, ord="fro")
        ),
    }

    return metrics, empirical_summary, generated_summary, empirical_cov, generated_cov


def final_horizon_summary_row(
    model_name: str,
    summary_obj: Dict[str, np.ndarray],
    horizon: int,
    variable_names: Sequence[str],
    variable_index: int = 0,
) -> Dict[str, float | int | str]:
    """
    Build one summary row at a given forecast horizon.

    Parameters
    ----------
    model_name : str
    summary_obj : dict
        Output of summarize_forecast_draws.
    horizon : int
        1-based forecast horizon.
    variable_names : sequence of str
    variable_index : int, default=0

    Returns
    -------
    dict
    """
    idx = horizon - 1
    lower = float(summary_obj["lower"][idx, variable_index])
    mean = float(summary_obj["mean"][idx, variable_index])
    upper = float(summary_obj["upper"][idx, variable_index])

    return {
        "model": model_name,
        "horizon": horizon,
        "variable": variable_names[variable_index],
        "mean": mean,
        "lower_95": lower,
        "upper_95": upper,
        "interval_width": upper - lower,
    }


def final_horizon_summary_all_variables(
    model_name: str,
    summary_obj: Dict[str, np.ndarray],
    horizon: int,
    variable_names: Sequence[str],
) -> pd.DataFrame:
    """
    Final-horizon summary across all variables.
    """
    idx = horizon - 1
    rows = []

    for j, name in enumerate(variable_names):
        lower = float(summary_obj["lower"][idx, j])
        mean = float(summary_obj["mean"][idx, j])
        upper = float(summary_obj["upper"][idx, j])

        rows.append(
            {
                "model": model_name,
                "variable": name,
                "mean": mean,
                "lower_95": lower,
                "upper_95": upper,
                "interval_width": upper - lower,
            }
        )

    return pd.DataFrame(rows)