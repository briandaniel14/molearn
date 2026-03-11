import gc
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from torch import Tensor, optim

from molearn.analysis.analyser import MolearnAnalysis
from molearn.models.latent_autoencoder import LatentAutoencoder


def rmsd(x: Tensor, x_recon: Tensor) -> Tensor:
    return torch.sqrt(torch.mean((x - x_recon) ** 2))


# ============================================================================
# Loss functions
# ============================================================================


def latent_rmsd_loss(enc_batch: Tensor, enc_recon: Tensor, raw_batch: Tensor) -> Tensor:
    return rmsd(enc_batch, enc_recon)


def latent_pairwise_stress_loss(
    enc_batch: Tensor, enc_recon: Tensor, raw_batch: Tensor
) -> Tensor:
    flat_in = enc_batch.reshape(enc_batch.size(0), -1)
    flat_out = enc_recon.reshape(enc_recon.size(0), -1)

    d_in = torch.cdist(flat_in, flat_in)
    d_out = torch.cdist(flat_out, flat_out)

    mask = torch.triu(torch.ones_like(d_in), diagonal=1).bool()
    return ((d_in[mask] - d_out[mask]) ** 2).sum() / (d_in[mask] ** 2 + 1e-8).sum()


def raw_rmsd_loss(ma: MolearnAnalysis):
    for p in ma.network.parameters():
        p.requires_grad_(False)

    def loss_fn(enc_batch: Tensor, enc_recon: Tensor, raw_batch: Tensor) -> Tensor:
        raw_recon = ma.network.decode(enc_recon)
        return rmsd(raw_batch, raw_recon)

    return loss_fn


def raw_pairwise_stress_loss(ma: MolearnAnalysis):
    for p in ma.network.parameters():
        p.requires_grad_(False)

    def loss_fn(enc_batch: Tensor, enc_recon: Tensor, raw_batch: Tensor) -> Tensor:
        raw_recon = ma.network.decode(enc_recon)

        flat_in = raw_batch.reshape(raw_batch.size(0), -1)
        flat_out = raw_recon.reshape(raw_recon.size(0), -1)

        d_in = torch.cdist(flat_in, flat_in)
        d_out = torch.cdist(flat_out, flat_out)

        mask = torch.triu(torch.ones_like(d_in), diagonal=1).bool()
        return ((d_in[mask] - d_out[mask]) ** 2).sum() / (d_in[mask] ** 2 + 1e-8).sum()

    return loss_fn


# ============================================================================
# Train functions
# ============================================================================


@dataclass
class TrainConfig:
    model: LatentAutoencoder
    ma: MolearnAnalysis
    key: str
    loss_func: Callable[[Tensor, Tensor, Tensor], Tensor]
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    verbose: bool = False


def train_loop(c: TrainConfig) -> LatentAutoencoder:
    encoded = c.ma.get_encoded(key=c.key)
    raw = c.ma.get_dataset(key=c.key)
    dataset = torch.utils.data.TensorDataset(encoded, raw)

    loader = torch.utils.data.DataLoader(dataset, batch_size=c.batch_size, shuffle=True)

    optimizer = optim.Adam(c.model.parameters(), lr=c.lr)
    c.model.train()

    for epoch in range(c.epochs):
        epoch_loss: float = 0.0
        for enc_batch, raw_batch in loader:
            enc_recon, z = c.model(enc_batch)
            loss = c.loss_func(enc_batch, enc_recon, raw_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0 and c.verbose:
            print(
                f"Epoch {epoch + 1}/{c.epochs} — Loss: {epoch_loss / len(loader):.6f}"
            )

    return c.model


# ============================================================================
# Scoring statistics
# ============================================================================


def get_inversion_ratios(ma: MolearnAnalysis, plot_data: list) -> list[float, float]:
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


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    sx, sy = x.var(), y.var()
    sxy = np.mean((x - mx) * (y - my))
    return float(2 * sxy / (sx + sy + (mx - my) ** 2))


def get_all_stats(
    mas,
    variables,
    runs,
    keys,
    random_state,
    max_pairs=5000,
) -> list:
    rows = []

    for var in variables:
        for run in runs:
            ma = mas[run][var]

            for key in keys:
                dataset = ma.get_dataset(key, scale=True)  # triggers cache
                decoded = ma.get_decoded(key, scale=True)  # encode + decode once

                # --- RMSD ---
                errors = ma.get_error(key)  # uses cached decoded
                median_rmsd = float(np.median(errors))
                mean_rmsd = float(np.mean(errors))
                std_rmsd = float(np.std(errors))

                # --- Pairwise RMSD CCC ---
                n_atoms = dataset.shape[1]
                flat_ds = dataset.reshape(dataset.shape[0], -1).numpy()
                flat_dc = decoded.reshape(decoded.shape[0], -1).numpy()
                pw_ds = pdist(flat_ds, "euclidean") / np.sqrt(n_atoms)
                pw_dc = pdist(flat_dc, "euclidean") / np.sqrt(n_atoms)
                if len(pw_ds) > max_pairs:
                    rng = np.random.default_rng(random_state)
                    idx = rng.choice(len(pw_ds), size=max_pairs, replace=False)
                    pw_ds_sub, pw_dc_sub = pw_ds[idx], pw_dc[idx]
                else:
                    pw_ds_sub, pw_dc_sub = pw_ds, pw_dc
                ccc = concordance_correlation(pw_ds_sub, pw_dc_sub)

                # --- Inversion ratio ---
                inv = ma.get_inversions(key)["decoded_inversions"]
                inversion_ratio = float(np.sum(inv == 0) / len(inv))

                # --- Wasserstein distances (per bond type) ---
                bl = ma.get_bondlengths(key)
                dataset_bl = bl["dataset_bondlen"]
                decoded_bl = bl["decoded_bondlen"]
                wd = {
                    bt: float(
                        wasserstein_distance(
                            dataset_bl[bt].flatten(), decoded_bl[bt].flatten()
                        )
                    )
                    for bt in dataset_bl
                }

                rows.append(
                    {
                        "var": var,
                        "run": run,
                        "key": key,
                        "median_rmsd": median_rmsd,
                        "mean_rmsd": mean_rmsd,
                        "std_rmsd": std_rmsd,
                        "ccc": ccc,
                        "inversion_ratio": inversion_ratio,
                        **{f"wd_{bt}": v for bt, v in wd.items()},
                    }
                )

            # free GPU/CPU memory before next model
            ma._decoded.clear()
            ma._encoded.clear()
            gc.collect()
            torch.cuda.empty_cache()

    return rows


# ===============================================
# hyperlatent molearn analysis for plotting
# ===============================================


def make_hyperlatent_ma(
    base_ma: MolearnAnalysis,
    mapping: Callable[[np.ndarray], np.ndarray],
    inverse_mapping: Callable[[np.ndarray], np.ndarray],
    keys: list[str] = ("train_both", "test_trans"),
) -> MolearnAnalysis:
    hyper_ma = MolearnAnalysis()
    hyper_ma.batch_size = base_ma.batch_size
    hyper_ma.processes = base_ma.processes
    hyper_ma.device = base_ma.device
    hyper_ma.network = base_ma.network
    hyper_ma.mol = base_ma.mol
    hyper_ma.meanval = base_ma.meanval
    hyper_ma.stdval = base_ma.stdval
    hyper_ma.n_atoms = base_ma.n_atoms
    hyper_ma.atoms = base_ma.atoms
    hyper_ma._datasets = base_ma._datasets

    with torch.no_grad():
        for key in keys:
            encoded = base_ma.get_encoded(key).numpy()
            projected = mapping(encoded)  # D → 2
            recon = inverse_mapping(projected)  # 2 → D
            hyper_ma.set_encoded(key, recon)

    return hyper_ma
