import numpy as np
import torch

from src.diffusion.reverse import p_sample


@torch.no_grad()
def sample_ddpm(
    model,
    n_samples: int,
    input_dim: int,
    schedule: dict,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Generate samples from trained DDPM model.

    Returns tensor with shape (n_samples, input_dim).
    """
    model.eval()

    x = torch.randn(
        n_samples,
        input_dim,
        device=device,
    )

    timesteps = schedule["timesteps"]

    for step in reversed(range(timesteps)):
        t = torch.full(
            (n_samples,),
            step,
            device=device,
            dtype=torch.long,
        )

        x = p_sample(
            model=model,
            x_t=x,
            t=t,
            schedule=schedule,
        )

    return x


def sample_diffusion_innovations(
    model,
    n_paths: int,
    horizon: int,
    input_dim: int,
    schedule: dict,
    device: str | torch.device = "cpu",
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate forecast innovation paths from a trained diffusion model.

    If mean/std are provided, generated standardized samples are mapped
    back to the original residual scale.
    """
    n_samples = n_paths * horizon

    samples = sample_ddpm(
        model=model,
        n_samples=n_samples,
        input_dim=input_dim,
        schedule=schedule,
        device=device,
    )

    samples_np = samples.detach().cpu().numpy()

    if mean is not None and std is not None:
        samples_np = samples_np * std + mean

    return samples_np.reshape(
        n_paths,
        horizon,
        input_dim,
    )