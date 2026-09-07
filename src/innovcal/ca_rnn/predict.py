"""Posterior predictive summaries and samples."""

import torch

from innovcal.ca_rnn.distributions import joint_gaussian


@torch.no_grad()
def predict_distribution(model, context: torch.Tensor, n_weight_samples: int = 50):
    """Return predictive means and Cholesky factors, retaining Bayesian draws."""
    model.eval()
    device = next(model.parameters()).device
    context = context.to(device)
    if getattr(model, "is_bayesian", False):
        outputs = [model(context, sample=True) for _ in range(n_weight_samples)]
    else:
        outputs = [model(context)]
    means = torch.stack([output[0] for output in outputs])
    scales = torch.stack([output[1] for output in outputs])
    return means, scales


@torch.no_grad()
def sample_forecasts(model, context: torch.Tensor, n_samples: int = 500) -> torch.Tensor:
    """Sample the predictive mixture; output shape is (samples, batch, dimension)."""
    model.eval()
    device = next(model.parameters()).device
    context = context.to(device)
    draws = []
    for _ in range(n_samples):
        if getattr(model, "is_bayesian", False):
            mean, scale = model(context, sample=True)
        else:
            mean, scale = model(context)
        epsilon = torch.randn_like(mean)
        draws.append(mean + torch.einsum("bij,bj->bi", scale, epsilon))
    return torch.stack(draws)


@torch.no_grad()
def predictive_nll(
    model,
    context: torch.Tensor,
    target: torch.Tensor,
    n_weight_samples: int = 50,
) -> float:
    """Evaluate NLL of a model or Monte Carlo Bayesian predictive mixture."""
    model.eval()
    device = next(model.parameters()).device
    context, target = context.to(device), target.to(device)
    draws = n_weight_samples if getattr(model, "is_bayesian", False) else 1
    log_probabilities = []
    for _ in range(draws):
        if getattr(model, "is_bayesian", False):
            mean, scale = model(context, sample=True)
        else:
            mean, scale = model(context)
        log_probabilities.append(joint_gaussian(mean, scale).log_prob(target))
    log_probabilities = torch.stack(log_probabilities)
    mixture_log_probability = torch.logsumexp(log_probabilities, dim=0) - torch.log(
        torch.as_tensor(draws, device=device, dtype=log_probabilities.dtype)
    )
    return float(-mixture_log_probability.mean().item())
