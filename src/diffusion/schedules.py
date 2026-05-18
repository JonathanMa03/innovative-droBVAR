import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if timesteps < 1:
        raise ValueError("timesteps must be positive.")

    return torch.linspace(
        beta_start,
        beta_end,
        timesteps,
        device=device,
    )


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    steps = timesteps + 1

    x = torch.linspace(
        0,
        timesteps,
        steps,
        device=device,
    )

    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5
    ) ** 2

    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (
        alphas_cumprod[1:] / alphas_cumprod[:-1]
    )

    return torch.clamp(betas, min=1e-5, max=0.999)


def make_ddpm_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    schedule_type: str = "linear",
    device: str | torch.device = "cpu",
) -> dict:
    if schedule_type == "linear":
        betas = linear_beta_schedule(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
        )
    elif schedule_type == "cosine":
        betas = cosine_beta_schedule(
            timesteps=timesteps,
            device=device,
        )
    else:
        raise ValueError("schedule_type must be either 'linear' or 'cosine'.")

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        "timesteps": timesteps,
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": torch.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
    }