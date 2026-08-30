import numpy as np
import torch


@torch.no_grad()
def rnn_predict_distribution(
    model,
    context: np.ndarray,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict Gaussian mean and scale from one context window.

    Parameters
    ----------
    context:
        Shape (context_length, k)

    Returns
    -------
    mean:
        Shape (prediction_length, k)

    scale:
        Shape (prediction_length, k)
    """
    model.eval()

    x = torch.tensor(
        context,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    mean, scale = model(x)

    return (
        mean.squeeze(0).cpu().numpy(),
        scale.squeeze(0).cpu().numpy(),
    )


def sample_rnn_forecast_paths(
    mean: np.ndarray,
    scale: np.ndarray,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample forecast paths from Gaussian predictive output.

    Returns
    -------
    paths:
        Shape (n_paths, horizon, k)
    """
    rng = np.random.default_rng(seed)

    paths = rng.normal(
        loc=mean[None, :, :],
        scale=scale[None, :, :],
        size=(n_paths, *mean.shape),
    )

    return paths


def forecast_rnn_paths(
    model,
    context: np.ndarray,
    n_paths: int = 1000,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> dict:
    """
    Generate probabilistic forecast paths from a trained RNN model.
    """
    mean, scale = rnn_predict_distribution(
        model=model,
        context=context,
        device=device,
    )

    paths = sample_rnn_forecast_paths(
        mean=mean,
        scale=scale,
        n_paths=n_paths,
        seed=seed,
    )

    return {
        "mean": mean,
        "scale": scale,
        "forecast_paths": paths,
        "paths": paths,  # backward-compatible alias
    }


# Backward-compatible aliases
deepar_predict_distribution = rnn_predict_distribution
sample_deepar_forecast_paths = sample_rnn_forecast_paths
forecast_deepar_paths = forecast_rnn_paths