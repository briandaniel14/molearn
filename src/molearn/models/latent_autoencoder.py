import torch
from torch import nn


class LatentAutoencoder(nn.Module):
    """MLP autoencoder that compresses a high-dimensional latent code to a lower-dimensional representation.

    Architecture: Linear → (LayerNorm → Mish → Dropout) × (num_layers - 1) → Linear

    Used both as a standalone post-hoc dimensionality reducer (trained offline on CNN latent
    codes) and as the inline bottleneck inside :class:`~molearn.models.cnn_ae_2d_online.AutoEncoder2D`.
    Using the same class in both contexts guarantees architectural consistency when comparing methods.

    :param int input_dim: dimensionality of the input latent code
    :param int latent_dim: dimensionality of the compressed representation (default 2)
    :param int hidden_dim: width of hidden layers (default 128)
    :param int num_layers: total number of linear layers in each of the encoder and decoder (default 4)
    :param float dropout: dropout probability (default 0.05)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        # --- Encoder: input_dim → latent_dim ---------------------------------
        encoder_layers: list[nn.Module] = []
        in_d = input_dim
        for i in range(num_layers):
            out_d = hidden_dim if i < num_layers - 1 else latent_dim
            encoder_layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                encoder_layers.append(nn.LayerNorm(out_d))
                encoder_layers.append(nn.Mish())
                encoder_layers.append(nn.Dropout(dropout))
            in_d = out_d
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder: latent_dim → input_dim ---------------------------------
        decoder_layers: list[nn.Module] = []
        in_d = latent_dim
        for i in range(num_layers):
            out_d = hidden_dim if i < num_layers - 1 else input_dim
            decoder_layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                decoder_layers.append(nn.LayerNorm(out_d))
                decoder_layers.append(nn.Mish())
                decoder_layers.append(nn.Dropout(dropout))
            in_d = out_d
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, input_dim)`` → ``(batch, latent_dim)``"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``(batch, latent_dim)`` → ``(batch, input_dim)``"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(x_recon, z)``."""
        z = self.encode(x)
        return self.decode(z), z
