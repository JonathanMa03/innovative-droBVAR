"""Variational layers for the Bayesian CA-RNN comparison."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _normal_kl(mu: torch.Tensor, sigma: torch.Tensor, prior_scale: float) -> torch.Tensor:
    return (
        torch.log(torch.as_tensor(prior_scale, device=mu.device) / sigma)
        + (sigma.square() + mu.square()) / (2.0 * prior_scale**2)
        - 0.5
    ).sum()


class BayesianParameter(nn.Module):
    """Mean-field Gaussian parameter with a zero-mean Gaussian prior."""

    def __init__(self, shape: tuple[int, ...], prior_scale: float = 1.0) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.empty(shape))
        self.rho = nn.Parameter(torch.full(shape, -4.0))
        self.prior_scale = float(prior_scale)
        nn.init.normal_(self.mu, std=1.0 / math.sqrt(max(1, shape[-1])))

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.rho) + 1e-8

    def sample(self) -> torch.Tensor:
        return self.mu + self.sigma * torch.randn_like(self.mu)

    def kl_divergence(self) -> torch.Tensor:
        return _normal_kl(self.mu, self.sigma, self.prior_scale)


class BayesianLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, prior_scale: float = 1.0):
        super().__init__()
        self.weight = BayesianParameter((output_dim, input_dim), prior_scale)
        self.bias = BayesianParameter((output_dim,), prior_scale)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        weight = self.weight.sample() if sample else self.weight.mu
        bias = self.bias.sample() if sample else self.bias.mu
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        return self.weight.kl_divergence() + self.bias.kl_divergence()


class BayesianGRUCell(nn.Module):
    """A mean-field variational GRU cell with one weight draw per sequence."""

    def __init__(self, input_dim: int, hidden_dim: int, prior_scale: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight_ih = BayesianParameter((3 * hidden_dim, input_dim), prior_scale)
        self.weight_hh = BayesianParameter((3 * hidden_dim, hidden_dim), prior_scale)
        self.bias_ih = BayesianParameter((3 * hidden_dim,), prior_scale)
        self.bias_hh = BayesianParameter((3 * hidden_dim,), prior_scale)

    def sampled_parameters(self, sample: bool = True) -> tuple[torch.Tensor, ...]:
        parameters = (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        return tuple(parameter.sample() if sample else parameter.mu for parameter in parameters)

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor, parameters: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        weight_ih, weight_hh, bias_ih, bias_hh = parameters
        input_gates = F.linear(x, weight_ih, bias_ih)
        hidden_gates = F.linear(hidden, weight_hh, bias_hh)
        input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_new + reset * hidden_new)
        return candidate + update * (hidden - candidate)

    def kl_divergence(self) -> torch.Tensor:
        return sum(
            parameter.kl_divergence()
            for parameter in (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        )
