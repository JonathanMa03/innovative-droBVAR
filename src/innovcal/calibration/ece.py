import numpy as np

from innovcal.forecasting.intervals import prediction_interval
from innovcal.calibration.coverage import average_coverage


def expected_calibration_error_intervals(
    forecast_paths: np.ndarray,
    y_true: np.ndarray,
    nominal_levels: list[float] | tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict:
    """
    Compute interval-based expected calibration error.

    For each nominal level, build central prediction intervals and compare
    empirical coverage to nominal coverage.
    """
    rows = []
    weighted_abs_errors = []

    for level in nominal_levels:
        alpha = 1.0 - level
        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0

        lower, upper = prediction_interval(
            forecast_paths,
            lower_q=lower_q,
            upper_q=upper_q,
        )

        empirical = average_coverage(
            y_true=y_true,
            lower=lower,
            upper=upper,
        )

        abs_error = abs(empirical - level)

        rows.append({
            "nominal": float(level),
            "empirical": float(empirical),
            "abs_error": float(abs_error),
            "lower_q": float(lower_q),
            "upper_q": float(upper_q),
        })

        weighted_abs_errors.append(abs_error)

    ece = float(np.mean(weighted_abs_errors))

    return {
        "ece": ece,
        "by_level": rows,
    }


def calibration_curve_data(
    forecast_paths: np.ndarray,
    y_true: np.ndarray,
    nominal_levels: list[float] | tuple[float, ...] = (
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9,
    ),
) -> dict:
    result = expected_calibration_error_intervals(
        forecast_paths=forecast_paths,
        y_true=y_true,
        nominal_levels=nominal_levels,
    )

    return {
        "nominal": np.array([row["nominal"] for row in result["by_level"]]),
        "empirical": np.array([row["empirical"] for row in result["by_level"]]),
        "abs_error": np.array([row["abs_error"] for row in result["by_level"]]),
        "ece": result["ece"],
    }