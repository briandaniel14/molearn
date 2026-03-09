from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, optim

from molearn.models.latent_autoencoder import LatentAutoencoder


def rmsd_loss(x: Tensor, x_recon: Tensor) -> Tensor:
    return torch.sqrt(torch.mean((x - x_recon) ** 2))


@dataclass
class TrainConfig:
    model: LatentAutoencoder
    encoded_data: Tensor
    loss_func: Callable[[Tensor, Tensor], Tensor] = rmsd_loss
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    verbose: bool = False


def train_loop(c: TrainConfig) -> LatentAutoencoder:
    dataset = torch.utils.data.TensorDataset(c.encoded_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=c.batch_size, shuffle=True)

    optimizer = optim.Adam(c.model.parameters(), lr=c.lr)
    c.model.train()

    for epoch in range(c.epochs):
        epoch_loss: float = 0.0
        for (batch,) in loader:
            x_recon, z = c.model(batch)
            loss = c.loss_func(batch, x_recon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0 and c.verbose:
            print(
                f"Epoch {epoch + 1}/{c.epochs} — Loss: {epoch_loss / len(loader):.6f}"
            )

    return c.model
