"""Context-conditioned diffusion for standardized VAR innovations."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from innovcal.diffusion.forward import extract_schedule_values, q_sample
from innovcal.diffusion.networks import SinusoidalTimeEmbedding


class ConditionalResidualDenoiser(nn.Module):
    """Predict diffusion noise using the noisy shock and observable context."""

    def __init__(self, input_dim: int, context_dim: int, hidden_dim: int = 128,
                 time_embedding_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + context_dim + time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        if context is None:
            raise ValueError("ConditionalResidualDenoiser requires context.")
        return self.net(torch.cat([x_t, self.time_embedding(t), context], dim=1))


def train_conditional_diffusion(
    model: nn.Module,
    targets: np.ndarray,
    contexts: np.ndarray,
    schedule: dict,
    validation_targets: np.ndarray | None = None,
    validation_contexts: np.ndarray | None = None,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 5e-4,
    patience: int | None = 40,
    device: str | torch.device = "cpu",
    seed: int = 123,
    verbose: bool = False,
) -> dict:
    """Train with chronological validation and restore the best checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)
    model.to(device)
    dataset = TensorDataset(
        torch.as_tensor(targets, dtype=torch.float32),
        torch.as_tensor(contexts, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "validation_loss": [], "best_epoch": None,
               "best_validation_loss": None}

    val_x = val_c = val_t = val_noise = None
    if validation_targets is not None and len(validation_targets):
        val_x = torch.as_tensor(validation_targets, dtype=torch.float32, device=device)
        val_c = torch.as_tensor(validation_contexts, dtype=torch.float32, device=device)
        val_generator = torch.Generator(device=device).manual_seed(seed + 1)
        val_t = torch.randint(0, schedule["timesteps"], (len(val_x),),
                              generator=val_generator, device=device)
        val_noise = torch.randn(val_x.shape, generator=val_generator, device=device)

    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x_0, context in loader:
            x_0, context = x_0.to(device), context.to(device)
            t = torch.randint(0, schedule["timesteps"], (len(x_0),), device=device)
            x_t, noise = q_sample(x_0, t, schedule)
            loss = F.mse_loss(model(x_t, t, context), noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history["loss"].append(float(np.mean(losses)))

        if val_x is not None:
            model.eval()
            with torch.no_grad():
                noisy, noise = q_sample(val_x, val_t, schedule, noise=val_noise)
                val_loss = F.mse_loss(model(noisy, val_t, val_c), noise).item()
            history["validation_loss"].append(val_loss)
            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                best_state = copy.deepcopy(model.state_dict())
                history["best_epoch"] = epoch
                history["best_validation_loss"] = val_loss
            else:
                stale += 1
        if verbose and (epoch == 1 or epoch % 25 == 0):
            suffix = "" if val_x is None else f" | val = {val_loss:.6f}"
            print(f"Epoch {epoch:04d} | loss = {history['loss'][-1]:.6f}{suffix}")
        if val_x is not None and patience is not None and stale >= patience:
            history["stopped_epoch"] = epoch
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def sample_conditional_ddpm(model: nn.Module, contexts: np.ndarray, schedule: dict,
                            device: str | torch.device = "cpu") -> np.ndarray:
    """Draw one standardized joint shock per supplied context row."""
    device = torch.device(device)
    context = torch.as_tensor(contexts, dtype=torch.float32, device=device)
    x = torch.randn(len(context), model.input_dim, device=device)
    model.eval()
    for step in reversed(range(schedule["timesteps"])):
        t = torch.full((len(x),), step, dtype=torch.long, device=device)
        beta = extract_schedule_values(schedule["betas"], t, x.shape)
        alpha = extract_schedule_values(schedule["alphas"], t, x.shape)
        alpha_bar = extract_schedule_values(schedule["alpha_bars"], t, x.shape)
        predicted = model(x, t, context)
        mean = (x - beta * predicted / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
        if step:
            mean = mean + torch.sqrt(beta) * torch.randn_like(x)
        x = mean
    return x.cpu().numpy()
