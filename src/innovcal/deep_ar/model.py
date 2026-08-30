import torch
from torch import nn
import torch.nn.functional as F


class ProbabilisticRNN(nn.Module):
    """
    Minimal multivariate probabilistic RNN forecaster.

    Uses a GRU encoder and outputs Gaussian parameters for each
    future time step and variable.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        prediction_length: int = 20,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.mean_head = nn.Linear(
            hidden_dim,
            prediction_length * input_dim,
        )

        self.scale_head = nn.Linear(
            hidden_dim,
            prediction_length * input_dim,
        )

    def forward(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        context:
            Shape (batch, context_length, input_dim)

        Returns
        -------
        mean:
            Shape (batch, prediction_length, input_dim)

        scale:
            Shape (batch, prediction_length, input_dim)
        """
        _, hidden = self.rnn(context)

        final_hidden = hidden[-1]

        mean = self.mean_head(final_hidden)
        scale_raw = self.scale_head(final_hidden)

        mean = mean.view(
            -1,
            self.prediction_length,
            self.input_dim,
        )

        scale = F.softplus(scale_raw).view(
            -1,
            self.prediction_length,
            self.input_dim,
        )

        scale = scale + 1e-5

        return mean, scale

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """
        Sample probabilistic forecast trajectories.

        Parameters
        ----------
        context:
            Shape (batch, context_length, input_dim)

        n_samples:
            Number of sampled trajectories.

        Returns
        -------
        samples:
            Shape (n_samples, batch, prediction_length, input_dim)
        """
        mean, scale = self.forward(context)

        dist = torch.distributions.Normal(
            mean,
            scale,
        )

        samples = dist.sample(
            (n_samples,)
        )

        return samples

    @torch.no_grad()
    def predict_mean(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return forecast mean.

        Output shape:
            (batch, prediction_length, input_dim)
        """
        mean, _ = self.forward(context)
        return mean

    @torch.no_grad()
    def predict_scale(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return forecast scale.

        Output shape:
            (batch, prediction_length, input_dim)
        """
        _, scale = self.forward(context)
        return scale


class DeepAR(ProbabilisticRNN):
    """
    Backward-compatible alias.

    This model is DeepAR-style but implemented as a direct
    multi-step probabilistic GRU forecaster.
    """
    pass


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian negative log likelihood.
    """
    dist = torch.distributions.Normal(
        mean,
        scale,
    )

    return -dist.log_prob(target).mean()