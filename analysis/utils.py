from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch import Tensor, optim

from molearn.analysis.analyser import MolearnAnalysis
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


def get_inversion_ratios(ma: MolearnAnalysis, plot_data: list) -> list[float, ...]:
    """Return the fraction of decoded structures with zero chirality inversions for each entry in plot_data."""
    ratios = []

    for key, *_ in plot_data:
        decoded = ma.get_inversions(key)["decoded_inversions"]
        ratios.append(float(np.sum(decoded == 0) / len(decoded)))

    return ratios


def get_wasserstein_distances(
    ma: MolearnAnalysis, plot_data: list, bond_types: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Return Wasserstein distances between dataset and decoded bond length distributions.

    Returns a dict keyed by dataset key, each mapping bond type -> distance.
    """
    results: dict[str, dict[str, float]] = {}

    for key, *_ in plot_data:
        bl = ma.get_bondlengths(key)
        dataset_bl = bl["dataset_bondlen"]
        decoded_bl = bl["decoded_bondlen"]

        types = bond_types if bond_types is not None else list(dataset_bl.keys())
        results[key] = {}
        for bt in types:
            results[key][bt] = float(
                wasserstein_distance(dataset_bl[bt].flatten(), decoded_bl[bt].flatten())
            )

    return results
