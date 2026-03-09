# Copyright (c) 2022 Samuel C. Musson
#
# Molightning is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# Molightning is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with molightning ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
import math

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, f):
        super().__init__()

        conv_block = [
            nn.Conv1d(f, f, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(f),
            nn.ReLU(inplace=True),
            nn.Conv1d(f, f, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(f),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ToND(nn.Module):
    def __init__(self, latent_z, n=2):
        super().__init__()
        self.latent_z = latent_z
        self.n = n
        self.proj = nn.Linear(n * latent_z, n)

    def forward(self, x):
        z = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(self.n, 1))
        z = torch.sigmoid(z)
        z = z.view(z.size(0), -1)
        z = self.proj(z)
        return z


class FromND(nn.Module):
    def __init__(self, latent_z, n=2, out_length=26):
        super().__init__()
        self.latent_z = latent_z
        self.out_length = out_length
        self.f = nn.Linear(n, self.out_length * latent_z)

    def forward(self, x):
        batch = x.size(0)
        x = self.f(x)
        x = x.view(batch, self.latent_z, self.out_length)
        return x


def _compute_out_length(n_atoms, depth):
    """Compute the decoder's initial spatial size based on target atom count and depth.

    The decoder has (depth + 3) transpose convolutions, each doubling spatial size.
    Therefore: final_atoms = out_length * 2^(depth+3)
    """
    return math.ceil(n_atoms / (2 ** (depth + 3)))


class AutoEncoder(nn.Module):
    """
    This is the autoencoder used in our `Ramaswamy 2021 paper <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052>`_.
    It is largely superseded by :func:`molearn.models.foldingnet.AutoEncoder`.
    """

    def __init__(
        self,
        n_atoms,
        init_z=32,
        latent_z=1,
        latent_dim=2,
        depth=4,
        m=1.5,
        r=0,
        droprate=None,
    ):
        """
        :param int init_z: number of channels in first layer
        :param int latent_z: number of latent channels
        :param int latent_dim: number of latent dimensions
        :param int depth: number of layers
        :param float m: scaling factor, dictating number of channels in subsequent layers
        :param int r: number of residual blocks between layers
        :param float droprate: dropout rate
        :param int n_atoms: number of atoms in target structure. If provided, decoder output is
            sized appropriately and sliced to exact atom count. If None, uses legacy out_length=26.
        """

        super().__init__()

        # Compute decoder spatial size based on target atoms
        self.n_atoms = n_atoms
        self.out_length = _compute_out_length(self.n_atoms, depth)

        # ============================================================
        # encoder block
        # ============================================================

        eb = nn.ModuleList()
        eb.append(nn.Conv1d(3, init_z, 4, 2, 1, bias=False))
        eb.append(nn.BatchNorm1d(init_z))

        if droprate is not None:
            eb.append(nn.Dropout(p=droprate))

        eb.append(nn.ReLU(inplace=True))

        for i in range(depth):
            eb.append(
                nn.Conv1d(
                    int(init_z * m**i), int(init_z * m ** (i + 1)), 4, 2, 1, bias=False
                )
            )
            eb.append(nn.BatchNorm1d(int(init_z * m ** (i + 1))))

            if droprate is not None:
                eb.append(nn.Dropout(p=droprate))

            eb.append(nn.ReLU(inplace=True))

            for _ in range(r):
                eb.append(ResidualBlock(int(init_z * m ** (i + 1))))

        eb.append(nn.Conv1d(int(init_z * m**depth), latent_z, 4, 2, 1, bias=False))
        eb.append(
            ToND(latent_z=latent_z, n=latent_dim)
        )  # To enable architectures without 2D layers

        self.encoder = eb

        # ============================================================
        # decoder block
        # ============================================================

        db = nn.ModuleList()
        db.append(FromND(latent_z=latent_z, n=latent_dim, out_length=self.out_length))

        db.append(
            nn.ConvTranspose1d(
                latent_z, int(init_z * m ** (depth + 1)), 4, 2, 1, bias=False
            )
        )

        db.append(nn.BatchNorm1d(int(init_z * m ** (depth + 1))))

        if droprate is not None:
            db.append(nn.Dropout(p=droprate))

        db.append(nn.ReLU(inplace=True))

        for i in reversed(range(depth + 1)):
            db.append(
                nn.ConvTranspose1d(
                    int(init_z * m ** (i + 1)), int(init_z * m**i), 4, 2, 1, bias=False
                )
            )
            db.append(nn.BatchNorm1d(int(init_z * m**i)))

            if droprate is not None:
                db.append(nn.Dropout(p=droprate))

            db.append(nn.ReLU(inplace=True))

            for _ in range(r):
                db.append(ResidualBlock(int(init_z * m**i)))

        db.append(nn.ConvTranspose1d(int(init_z * m ** (i)), 3, 4, 2, 1))
        self.decoder = db

    def encode(self, x):
        """Encode coordinates shaped ``(batch, atoms, 3)`` or ``(batch, 3, atoms)``."""

        if x.shape[2] == 3 and x.shape[1] != 3:  # noqa: PLR2004
            x = x.permute(0, 2, 1)

        for m in self.encoder:
            x = m(x.unsqueeze(-1)) if isinstance(m, ToND) else m(x)

        return x

    def decode(self, x):
        """Decode the latent representation back to ``(batch, atoms, 3)`` coordinates."""

        for m in self.decoder:
            x = m(x)

        x = x.permute(0, 2, 1)

        # Slice to exact atom count if n_atoms was specified
        if self.n_atoms is not None:
            x = x[:, : self.n_atoms, :]

        return x

    def forward(self, x):
        """Full autoencoder pass with input/output shaped ``(batch, atoms, 3)``."""

        z = self.encode(x)
        return self.decode(z)


class AutoEncoder2D(nn.Module):
    """
    End-to-end autoencoder with a 2-D hyperlatent bottleneck.

    Data flow per conformation::

        (batch, atoms, 3)
          → CNN encoder  → (batch, latent_dim)
          → MLP encoder  → (batch, 2)
          → MLP decoder  → (batch, latent_dim)
          → CNN decoder  → (batch, atoms, 3)

    The CNN encoder / decoder reuse the architecture of :class:`AutoEncoder`.
    The MLP bottleneck (Linear → LayerNorm → Mish → Dropout) compresses the
    CNN latent representation to 2-D and back, and is trained jointly with the
    CNN so the entire pipeline is optimised end-to-end.

    :param int init_z: number of channels in first CNN layer
    :param int latent_z: number of latent CNN channels
    :param int latent_dim: dimensionality of the CNN latent space (3–13 typical)
    :param int depth: number of CNN layers
    :param float m: channel scaling factor for successive CNN layers
    :param int r: number of residual blocks between CNN layers
    :param float droprate: dropout rate for CNN layers (None to disable)
    :param int n_atoms: number of atoms; if provided, decoder output is sliced to exact count
    :param int hyperlatent_dim: dimensionality of the 2-D bottleneck (default 2)
    :param int hidden_dim: width of MLP hidden layers
    :param int num_mlp_layers: depth of MLP encoder and decoder
    :param float mlp_dropout: dropout rate for MLP layers
    """

    def __init__(
        self,
        n_atoms,
        init_z=32,
        latent_z=1,
        latent_dim=7,
        depth=4,
        m=1.5,
        r=0,
        droprate=None,
        hyperlatent_dim=2,
        hidden_dim=128,
        num_mlp_layers=4,
        mlp_dropout=0.05,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hyperlatent_dim = hyperlatent_dim

        # --- CNN encoder & decoder (reuse AutoEncoder architecture) ------

        _cnn = AutoEncoder(
            n_atoms=n_atoms,
            init_z=init_z,
            latent_z=latent_z,
            latent_dim=latent_dim,
            depth=depth,
            m=m,
            r=r,
            droprate=droprate,
        )

        self.cnn_encoder = _cnn.encoder
        self.cnn_decoder = _cnn.decoder
        self.out_length = _cnn.out_length
        self.n_atoms = n_atoms

        # --- MLP encoder: latent_dim → hyperlatent_dim -------------------

        enc_layers = []
        in_d = latent_dim

        for i in range(num_mlp_layers):
            out_d = hidden_dim if i < num_mlp_layers - 1 else hyperlatent_dim
            enc_layers.append(nn.Linear(in_d, out_d))

            if i < num_mlp_layers - 1:
                enc_layers.append(nn.LayerNorm(out_d))
                enc_layers.append(nn.Mish())
                enc_layers.append(nn.Dropout(mlp_dropout))

            in_d = out_d

        self.mlp_encoder = nn.Sequential(*enc_layers)

        # --- MLP decoder: hyperlatent_dim → latent_dim -------------------

        dec_layers = []
        in_d = hyperlatent_dim

        for i in range(num_mlp_layers):
            out_d = hidden_dim if i < num_mlp_layers - 1 else latent_dim
            dec_layers.append(nn.Linear(in_d, out_d))

            if i < num_mlp_layers - 1:
                dec_layers.append(nn.LayerNorm(out_d))
                dec_layers.append(nn.Mish())
                dec_layers.append(nn.Dropout(mlp_dropout))

            in_d = out_d

        self.mlp_decoder = nn.Sequential(*dec_layers)

    # --- CNN helpers (same logic as AutoEncoder.encode / .decode) -----

    def _cnn_encode(self, x):
        """(batch, atoms, 3) -> (batch, latent_dim)"""

        if x.shape[2] == 3 and x.shape[1] != 3:  # noqa: PLR2004
            x = x.permute(0, 2, 1)

        for layer in self.cnn_encoder:
            x = layer(x.unsqueeze(-1)) if isinstance(layer, ToND) else layer(x)

        return x

    def _cnn_decode(self, x):
        """(batch, latent_dim) → (batch, atoms, 3)"""
        for layer in self.cnn_decoder:
            x = layer(x)
        x = x.permute(0, 2, 1)
        if self.n_atoms is not None:
            x = x[:, : self.n_atoms, :]
        return x

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def encode(self, x):
        """Full encode: (batch, atoms, 3) → (batch, hyperlatent_dim)."""
        return self.mlp_encoder(self._cnn_encode(x))

    def decode(self, z):
        """Full decode: (batch, hyperlatent_dim) → (batch, atoms, 3)."""
        return self._cnn_decode(self.mlp_decoder(z))

    def encode_latent(self, x):
        """CNN-only encode: (batch, atoms, 3) → (batch, latent_dim)."""
        return self._cnn_encode(x)

    def decode_latent(self, z):
        """CNN-only decode: (batch, latent_dim) → (batch, atoms, 3)."""
        return self._cnn_decode(z)

    def forward(self, x):
        """End-to-end pass.

        :param x: input coordinates, shape ``(batch, atoms, 3)``
        :returns: ``(x_recon, z_hyper)`` where *x_recon* has the same shape
            as *x* and *z_hyper* is the 2-D hyperlatent code ``(batch, 2)``
        """
        z_latent = self._cnn_encode(x)
        z_hyper = self.mlp_encoder(z_latent)
        z_latent_recon = self.mlp_decoder(z_hyper)
        x_recon = self._cnn_decode(z_latent_recon)
        return x_recon, z_hyper
