import torch


def extract_schedule_values(
    values: torch.Tensor,
    t: torch.Tensor,
    x_shape: tuple,
) -> torch.Tensor:
    """
    Gather schedule values at time indices t and reshape for broadcasting.
    """
    out = values.gather(
        dim=0,
        index=t,
    )

    return out.reshape(
        t.shape[0],
        *((1,) * (len(x_shape) - 1)),
    )


def q_sample(
    x_0: torch.Tensor,
    t: torch.Tensor,
    schedule: dict,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward diffusion:

        x_t = sqrt(alpha_bar_t) x_0
              + sqrt(1 - alpha_bar_t) epsilon
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alpha_bar_t = extract_schedule_values(
        schedule["sqrt_alpha_bars"],
        t,
        x_0.shape,
    )

    sqrt_one_minus_alpha_bar_t = extract_schedule_values(
        schedule["sqrt_one_minus_alpha_bars"],
        t,
        x_0.shape,
    )

    x_t = (
        sqrt_alpha_bar_t * x_0
        + sqrt_one_minus_alpha_bar_t * noise
    )

    return x_t, noise