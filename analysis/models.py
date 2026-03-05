from collections.abc import Callable
from typing import dataclass

import torch
from torch import Tensor, nn, optim


class LatentAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        # =================================================
        # Encoder Block: input_dim -> 2
        # =================================================

        encoder_layers: list[nn.Module] = []
        in_dim: int = input_dim
        for i in range(num_layers):
            out_dim: int = hidden_dim if i < num_layers - 1 else latent_dim
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                encoder_layers.append(nn.LayerNorm(out_dim))
                encoder_layers.append(
                    nn.Mish()
                )  # smooth activation > ReLU for reconstruction
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # =================================================
        # Decoder Block: 2 -> input_dim
        # =================================================

        decoder_layers: list[nn.Module] = []
        in_dim = latent_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else input_dim
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                decoder_layers.append(nn.LayerNorm(out_dim))
                decoder_layers.append(nn.Mish())
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z: torch.Tensor = self.encode(x)
        x_recon: torch.Tensor = self.decode(z)
        return x_recon, z


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
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{c.epochs} — Loss: {epoch_loss / len(loader):.6f}"
            )

    return c.model
