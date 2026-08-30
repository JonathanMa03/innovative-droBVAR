from pathlib import Path
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from innovcal.diffusion.losses import ddpm_noise_prediction_loss
from innovcal.diffusion.forward import q_sample


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
    validation_residuals: np.ndarray | None = None,
    early_stopping_patience: int | None = None,
    validation_seed: int = 123,
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
        "validation_loss": [],
        "best_epoch": None,
        "best_validation_loss": None,
    }

    validation_batch = None
    validation_t = None
    validation_noise = None
    if validation_residuals is not None:
        validation_batch = torch.tensor(
            validation_residuals,
            dtype=torch.float32,
            device=device,
        )
        generator = torch.Generator(device=device).manual_seed(validation_seed)
        validation_t = torch.randint(
            0,
            schedule["timesteps"],
            (len(validation_batch),),
            generator=generator,
            device=device,
        )
        validation_noise = torch.randn(
            validation_batch.shape,
            generator=generator,
            device=device,
        )

    best_state = None
    best_loss = float("inf")
    epochs_without_improvement = 0

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

        if validation_batch is not None:
            model.eval()
            with torch.no_grad():
                validation_noisy, target_noise = q_sample(
                    validation_batch,
                    validation_t,
                    schedule,
                    noise=validation_noise,
                )
                predicted_noise = model(validation_noisy, validation_t)
                validation_loss = torch.mean(
                    (predicted_noise - target_noise) ** 2
                ).item()
            history["validation_loss"].append(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                history["best_epoch"] = epoch
                history["best_validation_loss"] = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if verbose and (
            epoch == 1
            or epoch % 25 == 0
            or epoch == epochs
        ):
            message = f"Epoch {epoch:04d} | loss = {avg_loss:.6f}"
            if validation_batch is not None:
                message += f" | val = {validation_loss:.6f}"
            print(message)

        if (
            validation_batch is not None
            and early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            history["stopped_epoch"] = epoch
            break

    if best_state is not None:
        model.load_state_dict(best_state)

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
