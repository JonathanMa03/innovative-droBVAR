import numpy as np

from src.dro.perturbations import perturb_innovation_paths
from src.dro.reweighting import reweight_forecast_paths


def terminal_loss(
    forecast_paths: np.ndarray,
    variable: int = 0,
    direction: str = "lower",
) -> np.ndarray:
    """
    Path loss based on terminal forecast value.

    direction='lower' treats low terminal values as bad.
    direction='upper' treats high terminal values as bad.
    """
    terminal = forecast_paths[:, -1, variable]

    if direction == "lower":
        return -terminal

    if direction == "upper":
        return terminal

    raise ValueError("direction must be either 'lower' or 'upper'.")


def max_drawdown_loss(
    forecast_paths: np.ndarray,
    variable: int = 0,
) -> np.ndarray:
    """
    Simple pathwise drawdown loss.
    """
    x = forecast_paths[:, :, variable]

    running_max = np.maximum.accumulate(
        x,
        axis=1,
    )

    drawdowns = running_max - x

    return np.max(drawdowns, axis=1)


def tail_exceedance_loss(
    forecast_paths: np.ndarray,
    threshold: float,
    variable: int = 0,
    direction: str = "upper",
) -> np.ndarray:
    """
    Loss based on tail exceedance over horizon.
    """
    x = forecast_paths[:, :, variable]

    if direction == "upper":
        exceedance = np.maximum(x - threshold, 0.0)

    elif direction == "lower":
        exceedance = np.maximum(threshold - x, 0.0)

    else:
        raise ValueError("direction must be either 'upper' or 'lower'.")

    return np.max(exceedance, axis=1)


def stress_innovation_paths(
    innovation_paths: np.ndarray,
    epsilon: float,
    method: str = "scale",
    seed: int | None = None,
) -> np.ndarray:
    return perturb_innovation_paths(
        innovation_paths=innovation_paths,
        method=method,
        epsilon=epsilon,
        seed=seed,
    )


def summarize_stress_losses(
    forecast_paths: np.ndarray,
    loss_type: str = "terminal",
    variable: int = 0,
    **kwargs,
) -> dict:
    if loss_type == "terminal":
        losses = terminal_loss(
            forecast_paths,
            variable=variable,
            direction=kwargs.get("direction", "lower"),
        )

    elif loss_type == "drawdown":
        losses = max_drawdown_loss(
            forecast_paths,
            variable=variable,
        )

    elif loss_type == "tail_exceedance":
        losses = tail_exceedance_loss(
            forecast_paths,
            threshold=kwargs["threshold"],
            variable=variable,
            direction=kwargs.get("direction", "upper"),
        )

    else:
        raise ValueError("loss_type must be one of: 'terminal', 'drawdown', 'tail_exceedance'.")

    return {
        "mean_loss": float(np.mean(losses)),
        "median_loss": float(np.median(losses)),
        "q90_loss": float(np.quantile(losses, 0.90)),
        "q95_loss": float(np.quantile(losses, 0.95)),
        "max_loss": float(np.max(losses)),
        "losses": losses,
    }


def robust_reweighting_summary(
    forecast_paths: np.ndarray,
    eta: float,
    loss_type: str = "terminal",
    variable: int = 0,
    **kwargs,
) -> dict:
    stress_summary = summarize_stress_losses(
        forecast_paths=forecast_paths,
        loss_type=loss_type,
        variable=variable,
        **kwargs,
    )

    reweight_summary = reweight_forecast_paths(
        losses=stress_summary["losses"],
        eta=eta,
    )

    return {
        **stress_summary,
        **reweight_summary,
    }