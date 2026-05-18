import torch

from src.diffusion.forward import extract_schedule_values


@torch.no_grad()
def p_sample(
    model,
    x_t: torch.Tensor,
    t: torch.Tensor,
    schedule: dict,
) -> torch.Tensor:
    """
    One reverse DDPM sampling step.
    """
    betas_t = extract_schedule_values(
        schedule["betas"],
        t,
        x_t.shape,
    )

    alphas_t = extract_schedule_values(
        schedule["alphas"],
        t,
        x_t.shape,
    )

    alpha_bars_t = extract_schedule_values(
        schedule["alpha_bars"],
        t,
        x_t.shape,
    )

    sqrt_one_minus_alpha_bars_t = extract_schedule_values(
        schedule["sqrt_one_minus_alpha_bars"],
        t,
        x_t.shape,
    )

    noise_pred = model(
        x_t=x_t,
        t=t,
    )

    mean = (
        1.0 / torch.sqrt(alphas_t)
        * (
            x_t
            - (betas_t / sqrt_one_minus_alpha_bars_t) * noise_pred
        )
    )

    noise = torch.randn_like(x_t)

    nonzero_mask = (
        (t != 0)
        .float()
        .reshape(x_t.shape[0], *((1,) * (len(x_t.shape) - 1)))
    )

    return mean + nonzero_mask * torch.sqrt(betas_t) * noise