import torch
from torch import nn


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

        # Encoder: input_dim -> 2
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

        # Decoder: 2 -> input_dim
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
