"""Joint Gaussian parameterization used by all CA-RNN variants."""

import torch
from torch.nn import functional as F


def vector_to_cholesky(
    raw: torch.Tensor, dimension: int, min_scale: float = 1e-4
) -> torch.Tensor:
    """Map unconstrained vectors to valid lower-triangular scale matrices."""
    expected = dimension * (dimension + 1) // 2
    if raw.shape[-1] != expected:
        raise ValueError(f"expected {expected} Cholesky parameters")
    result = raw.new_zeros(*raw.shape[:-1], dimension, dimension)
    rows, cols = torch.tril_indices(dimension, dimension, device=raw.device)
    result[..., rows, cols] = raw
    diagonal = torch.arange(dimension, device=raw.device)
    result[..., diagonal, diagonal] = (
        F.softplus(result[..., diagonal, diagonal]) + min_scale
    )
    return result


def joint_gaussian(mean: torch.Tensor, scale_tril: torch.Tensor):
    """Return a multivariate Normal with a Cholesky scale parameter."""
    return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)
