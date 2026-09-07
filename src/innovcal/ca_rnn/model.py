"""Deterministic and Bayesian multivariate probabilistic RNNs."""

import torch
from torch import nn

from innovcal.ca_rnn.config import CARNNConfig
from innovcal.ca_rnn.distributions import vector_to_cholesky
from innovcal.ca_rnn.layers import BayesianGRUCell, BayesianLinear


class CARNN(nn.Module):
    """GRU with a full-covariance one-step joint Gaussian predictive head."""

    is_bayesian = False

    def __init__(self, config: CARNNConfig):
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            config.input_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        parameter_count = config.input_dim * (config.input_dim + 1) // 2
        self.mean_head = nn.Linear(config.hidden_dim, config.input_dim)
        self.scale_head = nn.Linear(config.hidden_dim, parameter_count)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3 or context.shape[-1] != self.config.input_dim:
            raise ValueError("context must have shape (batch, history, input_dim)")
        output, _ = self.rnn(context)
        hidden = output[:, -1]
        mean = self.mean_head(hidden)
        scale = vector_to_cholesky(
            self.scale_head(hidden), self.config.input_dim, self.config.min_scale
        )
        return mean, scale

    def kl_divergence(self) -> torch.Tensor:
        return next(self.parameters()).new_zeros(())


class BayesianCARNN(nn.Module):
    """Mean-field Bayesian GRU and predictive head for the CA-BRNN comparison."""

    is_bayesian = True

    def __init__(self, config: CARNNConfig, prior_scale: float = 1.0):
        super().__init__()
        if config.num_layers != 1:
            raise ValueError("BayesianCARNN currently supports one recurrent layer")
        self.config = config
        self.cell = BayesianGRUCell(config.input_dim, config.hidden_dim, prior_scale)
        parameter_count = config.input_dim * (config.input_dim + 1) // 2
        self.mean_head = BayesianLinear(config.hidden_dim, config.input_dim, prior_scale)
        self.scale_head = BayesianLinear(config.hidden_dim, parameter_count, prior_scale)

    def forward(
        self, context: torch.Tensor, sample: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3 or context.shape[-1] != self.config.input_dim:
            raise ValueError("context must have shape (batch, history, input_dim)")
        if sample is None:
            sample = self.training
        hidden = context.new_zeros(context.shape[0], self.config.hidden_dim)
        parameters = self.cell.sampled_parameters(sample)
        for step in range(context.shape[1]):
            hidden = self.cell(context[:, step], hidden, parameters)
        mean = self.mean_head(hidden, sample)
        raw_scale = self.scale_head(hidden, sample)
        scale = vector_to_cholesky(
            raw_scale, self.config.input_dim, self.config.min_scale
        )
        return mean, scale

    def kl_divergence(self) -> torch.Tensor:
        return (
            self.cell.kl_divergence()
            + self.mean_head.kl_divergence()
            + self.scale_head.kl_divergence()
        )
