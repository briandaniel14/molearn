import copy
import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import umap
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from torch import Tensor, optim

from molearn.analysis.analyser import MolearnAnalysis
from molearn.models.latent_autoencoder import LatentAutoencoder

RANDOM_STATE = 42


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


def composite_loss(ma, weights):
    latent_rmsd = latent_rmsd_loss
    latent_stress = latent_pairwise_stress_loss
    raw_rmsd = raw_rmsd_loss(ma)
    raw_stress = raw_pairwise_stress_loss(ma)

    def loss_fn(enc_batch, enc_recon, raw_batch):
        return (
            weights[0] * latent_rmsd(enc_batch, enc_recon, raw_batch)
            + weights[1] * latent_stress(enc_batch, enc_recon, raw_batch)
            + weights[2] * raw_rmsd(enc_batch, enc_recon, raw_batch)
            + weights[3] * raw_stress(enc_batch, enc_recon, raw_batch)
        )

    return loss_fn


# ============================================================================
# Train functions
# ============================================================================


@dataclass
class TrainConfig:
    model: LatentAutoencoder
    ma: MolearnAnalysis
    keys: list[str]
    loss_func: Callable[[Tensor, Tensor, Tensor], Tensor]
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    verbose: bool = False
    val_split: float = 0.1


def train_loop(c: TrainConfig) -> LatentAutoencoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    c.model = c.model.to(device)

    encoded_all = torch.vstack([c.ma.get_encoded(key=key) for key in c.keys]).to(device)
    raw_all = torch.vstack([c.ma.get_dataset(key=key) for key in c.keys]).to(device)

    dataset = torch.utils.data.TensorDataset(encoded_all, raw_all)

    n_total = len(dataset)
    n_val = int(n_total * c.val_split)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=c.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=c.batch_size, shuffle=False
    )

    optimizer = optim.Adam(c.model.parameters(), lr=c.lr)
    c.model.train()

    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(c.model.state_dict())

    for epoch in range(c.epochs):
        epoch_loss: float = 0.0

        c.model.train()

        for enc_batch, raw_batch in train_loader:
            enc_batch = enc_batch.to(device)  # noqa: PLW2901
            raw_batch = raw_batch.to(device)  # noqa: PLW2901
            enc_recon, z = c.model(enc_batch)

            loss = c.loss_func(enc_batch, enc_recon, raw_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = 0.0
        c.model.eval()

        with torch.no_grad():
            for enc_batch, raw_batch in val_loader:
                enc_batch = enc_batch.to(device)  # noqa: PLW2901
                raw_batch = raw_batch.to(device)  # noqa: PLW2901
                enc_recon, z = c.model(enc_batch)
                loss = c.loss_func(enc_batch, enc_recon, raw_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(c.model.state_dict())

        if (epoch + 1) % 10 == 0 and c.verbose:
            print(f"Epoch {epoch + 1}/{c.epochs} — Best val loss: {best_val_loss:.6f}")

        c.model.load_state_dict(best_state_dict)

    return c.model, best_val_loss


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
    varis,
    runs,
    keys,
    save_dir: Path,
    random_state,
    max_pairs=5000,
) -> list:
    rows = []

    for var in varis:
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

            ma._decoded.clear()
            ma._encoded.clear()
            gc.collect()
            torch.cuda.empty_cache()

    df_stats = pd.DataFrame(rows)
    df_stats.sort_values(["var", "run", "key"]).to_csv(save_dir, index=False)

    return df_stats


# ===============================================
# hyperlatent molearn analysis for plotting
# ===============================================


def make_hyperlatent_ma(
    base_ma: MolearnAnalysis,
    encoded_by_keys: dict[str, Tensor],
    mapping: Callable[[np.ndarray], np.ndarray],
    inverse_mapping: Callable[[np.ndarray], np.ndarray],
    keys: list[str] = ("train_open", "train_closed", "train_both", "test_trans"),
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
            encoded = encoded_by_keys[key].numpy()
            projected = mapping(encoded)  # D → 2
            recon = inverse_mapping(projected)  # 2 → D
            hyper_ma.set_encoded(key, recon)

    return hyper_ma


def make_hyperlatent_pca_mas(
    mas, runs, varis, keys
) -> dict[int, dict[int, MolearnAnalysis]]:
    pca_mas: dict[int, dict[int, MolearnAnalysis]] = {}
    mappings: dict[int, dict[int, Callable]] = {}
    inverse_mappings: dict[int, dict[int, Callable]] = {}

    for run in runs:
        pca_mas[run] = {}
        mappings[run] = {}
        inverse_mappings[run] = {}

        for var in varis:
            encoded: dict[str, Tensor] = {}
            ma = mas[run][var]

            for key in keys:
                encoded[key] = ma.get_encoded(key=key)

            encoded_all = np.vstack([encoded[key] for key in keys])

            pca = PCA(n_components=2)
            pca.fit(encoded_all)

            pca_mas[run][var] = make_hyperlatent_ma(
                base_ma=mas[run][var],
                encoded_by_keys=encoded,
                mapping=pca.transform,
                inverse_mapping=pca.inverse_transform,
                keys=keys,
            )

            mappings[run][var] = pca.transform
            inverse_mappings[run][var] = pca.inverse_transform

            ma._decoded.clear()
            ma._encoded.clear()
            gc.collect()
            torch.cuda.empty_cache()

    return pca_mas, mappings, inverse_mappings


def make_hyperlatent_umap_mas(
    mas,
    runs,
    varis,
    keys,
    n_components=2,
    n_neighbors=12,
    min_dist=0.1,
    random_state=RANDOM_STATE,
) -> dict[int, dict[int, MolearnAnalysis]]:
    umap_mas: dict[int, dict[int, MolearnAnalysis]] = {}
    mappings: dict[int, dict[int, Callable]] = {}
    inverse_mappings: dict[int, dict[int, Callable]] = {}

    for run in runs:
        umap_mas[run] = {}
        mappings[run] = {}
        inverse_mappings[run] = {}

        for var in varis:
            encoded: dict[str, Tensor] = {}
            ma = mas[run][var]

            for key in keys:
                encoded[key] = ma.get_encoded(key=key)

            encoded_all = np.vstack([encoded[key] for key in keys])

            reducer = umap.UMAP(
                n_components=2, random_state=RANDOM_STATE, n_neighbors=12, min_dist=0.1
            )
            reducer.fit(encoded_all)

            umap_mas[run][var] = make_hyperlatent_ma(
                base_ma=mas[run][var],
                encoded_by_keys=encoded,
                mapping=reducer.transform,
                inverse_mapping=reducer.inverse_transform,
                keys=keys,
            )

            mappings[run][var] = reducer.transform
            inverse_mappings[run][var] = reducer.inverse_transform

            ma._decoded.clear()
            ma._encoded.clear()
            gc.collect()
            torch.cuda.empty_cache()

    return umap_mas, mappings, inverse_mappings


def make_hyperlatent_mlp_mas(
    mas: dict[int, dict[int, MolearnAnalysis]],
    runs: list[int],
    varis: list[int],
    keys: list[str],
    train_config: TrainConfig,
) -> dict[int, dict[int, MolearnAnalysis]]:
    mlp_mas: dict[int, dict[int, MolearnAnalysis]] = {}
    mappings: dict[int, dict[int, Callable]] = {}
    inverse_mappings: dict[int, dict[int, Callable]] = {}
    val_loss: dict[int, dict[int, Callable]] = {}

    for run in runs:
        mlp_mas[run] = {}
        mappings[run] = {}
        inverse_mappings[run] = {}
        val_loss[run] = {}

        for var in varis:
            encoded: dict[str, Tensor] = {}
            ma = mas[run][var]

            for key in keys:
                encoded[key] = ma.get_encoded(key=key)

            model, val_loss[run][var] = train_loop(train_config)

            def ae_encode(x: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    return model.encode(torch.tensor(x, dtype=torch.float32)).numpy()  # noqa: B023

            mappings[run][var] = ae_encode

            def ae_decode(x: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    return model.decode(torch.tensor(x, dtype=torch.float32)).numpy()  # noqa: B023

            inverse_mappings[run][var] = ae_decode

            mlp_mas[run][var] = make_hyperlatent_ma(
                base_ma=mas[run][var],
                encoded_by_keys=encoded,
                mapping=ae_encode,
                inverse_mapping=ae_decode,
                keys=keys,
            )

            ma._decoded.clear()
            ma._encoded.clear()
            gc.collect()
            torch.cuda.empty_cache()

    return mlp_mas, mappings, inverse_mappings, val_loss
