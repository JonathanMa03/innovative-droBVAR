from pathlib import Path
import torch

from src.deep_ar.model import gaussian_nll


def train_deepar(
    model,
    dataloader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict:
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    history = {
        "loss": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            mean, scale = model(context)
            loss = gaussian_nll(
                target=target,
                mean=mean,
                scale=scale,
            )

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        history["loss"].append(avg_loss)

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"Epoch {epoch:04d} | loss = {avg_loss:.6f}")

    return history


def save_deepar_model(
    model,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_deepar_model(
    model,
    path: str | Path,
    device: str | torch.device = "cpu",
):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model