import math
import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        t:
            Shape (batch,)
        """
        half_dim = self.embedding_dim // 2

        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(
                0,
                half_dim,
                device=t.device,
                dtype=torch.float32,
            )
            / max(half_dim - 1, 1)
        )

        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)

        emb = torch.cat(
            [torch.sin(args), torch.cos(args)],
            dim=1,
        )

        if self.embedding_dim % 2 == 1:
            emb = torch.cat(
                [emb, torch.zeros_like(emb[:, :1])],
                dim=1,
            )

        return emb


class ResidualDenoisingMLP(nn.Module):
    """
    Simple DDPM noise-prediction network for innovation vectors.

    Input:
        x_t with shape (batch, input_dim)
        t with shape (batch,)

    Output:
        predicted noise with shape (batch, input_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        time_embedding_dim: int = 32,
        num_hidden_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.time_embedding = SinusoidalTimeEmbedding(
            embedding_dim=time_embedding_dim,
        )

        layers = []

        in_dim = input_dim + time_embedding_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embedding(t)

        x = torch.cat(
            [x_t, t_emb],
            dim=1,
        )

        return self.net(x)