from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.diffusion.losses import ddpm_noise_prediction_loss


def make_residual_dataloader(
    residuals: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    x = torch.tensor(
        residuals,
        dtype=torch.float32,
    )

    dataset = TensorDataset(x)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_diffusion_model(
    model,
    residuals: np.ndarray,
    schedule: dict,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict:
    model = model.to(device)

    dataloader = make_residual_dataloader(
        residuals=residuals,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    history = {
        "loss": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for (batch,) in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()

            loss = ddpm_noise_prediction_loss(
                model=model,
                x_0=batch,
                schedule=schedule,
            )

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        history["loss"].append(avg_loss)

        if verbose and (
            epoch == 1
            or epoch % 25 == 0
            or epoch == epochs
        ):
            print(f"Epoch {epoch:04d} | loss = {avg_loss:.6f}")

    return history


def save_diffusion_model(
    model,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_diffusion_model(
    model,
    path: str | Path,
    device: str | torch.device = "cpu",
):
    state = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model