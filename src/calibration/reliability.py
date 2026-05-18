import numpy as np
import pandas as pd

from src.calibration.ece import calibration_curve_data
from src.calibration.pit import pit_values_multivariate_marginal, pit_deviation_from_uniform


def reliability_summary(
    forecast_paths: np.ndarray,
    y_true: np.ndarray,
    nominal_levels: list[float] | tuple[float, ...] = (
        0.5, 0.8, 0.9,
    ),
) -> dict:
    """
    Combine interval calibration and PIT diagnostics.
    """
    curve = calibration_curve_data(
        forecast_paths=forecast_paths,
        y_true=y_true,
        nominal_levels=nominal_levels,
    )

    pit = pit_values_multivariate_marginal(
        forecast_paths=forecast_paths,
        y_true=y_true,
    )

    return {
        "ece": curve["ece"],
        "nominal": curve["nominal"],
        "empirical": curve["empirical"],
        "abs_error": curve["abs_error"],
        "pit_values": pit,
        "pit_deviation": pit_deviation_from_uniform(pit),
    }


def reliability_table(
    forecast_paths: np.ndarray,
    y_true: np.ndarray,
    model_name: str,
    dgp_name: str,
    nominal_levels: list[float] | tuple[float, ...] = (
        0.5, 0.8, 0.9,
    ),
) -> pd.DataFrame:
    """
    Return interval calibration table for one model/DGP pair.
    """
    curve = calibration_curve_data(
        forecast_paths=forecast_paths,
        y_true=y_true,
        nominal_levels=nominal_levels,
    )

    return pd.DataFrame({
        "dgp": dgp_name,
        "model": model_name,
        "nominal": curve["nominal"],
        "empirical": curve["empirical"],
        "abs_error": curve["abs_error"],
        "ece": curve["ece"],
    })