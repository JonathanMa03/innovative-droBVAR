import torch
import torch.nn.functional as F

from src.diffusion.forward import q_sample


def ddpm_noise_prediction_loss(
    model,
    x_0: torch.Tensor,
    schedule: dict,
) -> torch.Tensor:
    """
    Standard DDPM noise-prediction objective.
    """
    batch_size = x_0.shape[0]
    timesteps = schedule["timesteps"]

    t = torch.randint(
        low=0,
        high=timesteps,
        size=(batch_size,),
        device=x_0.device,
    )

    x_t, noise = q_sample(
        x_0=x_0,
        t=t,
        schedule=schedule,
    )

    noise_pred = model(
        x_t=x_t,
        t=t,
    )

    return F.mse_loss(
        noise_pred,
        noise,
    )