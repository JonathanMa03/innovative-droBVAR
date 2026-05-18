import numpy as np

from src.evaluation.crps import mean_crps_marginal
from src.evaluation.energy_score import mean_energy_score


def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> np.ndarray:
    """
    Interval score for central (1 - alpha) prediction intervals.

    Lower is better.

    Score = width
            + 2/alpha * (lower - y) * 1{y < lower}
            + 2/alpha * (y - upper) * 1{y > upper}
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    width = upper - lower

    lower_penalty = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    upper_penalty = (2.0 / alpha) * (y_true - upper) * (y_true > upper)

    return width + lower_penalty + upper_penalty


def mean_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> float:
    return float(
        np.mean(
            interval_score(
                y_true=y_true,
                lower=lower,
                upper=upper,
                alpha=alpha,
            )
        )
    )


def summarize_scoring_rules(
    forecast_paths: np.ndarray,
    y_true: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    alpha: float = 0.10,
) -> dict:
    """
    Compute core probabilistic forecast scores.
    """
    scores = {
        "energy_score": mean_energy_score(
            forecast_paths=forecast_paths,
            y_true=y_true,
        ),
        "crps": mean_crps_marginal(
            forecast_paths=forecast_paths,
            y_true=y_true,
        ),
    }

    if lower is not None and upper is not None:
        scores["interval_score"] = mean_interval_score(
            y_true=y_true,
            lower=lower,
            upper=upper,
            alpha=alpha,
        )

    return scores