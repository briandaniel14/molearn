from __future__ import annotations

import gc
import os
from collections.abc import Callable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist

from molearn.analysis.analyser import MolearnAnalysis
from molearn.analysis.path import get_path, get_point_index
from molearn.analysis.plot import (
    plot_bondlength_hist,
    plot_inversion_hist,
    plot_rmsd_hist,
)


def _latent_edge(values: np.ndarray) -> np.ndarray:
    return np.append(values, (2 * values[-1] - values[-2]))


def plot_dope_mesh(  # noqa: PLR0913 PLR0915
    ma: MolearnAnalysis,  # noqa: N803
    encoded_datasets: Sequence[tuple[np.ndarray, str, str]],
    mapping: Callable[[np.ndarray], np.ndarray],
    inverse_mapping: Callable[[np.ndarray], np.ndarray],
    truncate_at: float,
    *,
    show: bool = True,
    n_samples: int = 20,
    margin: float = 0.1,
    refine: bool = True,
    cmap: str = "viridis",
    title: str | None = None,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    fname: str | None = None,
    plot_crystals: bool = False,
    # --- interpolation paths ---
    plot_linear: bool = False,
    plot_astar: bool = False,
    start_encoded: np.ndarray | None = None,
    end_encoded: np.ndarray | None = None,
    n_interp: int = 50,
    **savefig_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot a 2D DOPE-score mesh with optional linear / A* paths."""
    need_paths = plot_linear or plot_astar

    # === project all datasets into 2D to determine grid bounds =========
    all_2d = np.vstack([mapping(enc) for enc, _, _ in encoded_datasets])

    d1_min, d1_max = all_2d[:, 0].min(), all_2d[:, 0].max()
    d2_min, d2_max = all_2d[:, 1].min(), all_2d[:, 1].max()
    d1_range = d1_max - d1_min
    d2_range = d2_max - d2_min

    d1_vals = np.linspace(
        d1_min - margin * d1_range, d1_max + margin * d1_range, n_samples
    )
    d2_vals = np.linspace(
        d2_min - margin * d2_range, d2_max + margin * d2_range, n_samples
    )

    d1_grid, d2_grid = np.meshgrid(d1_vals, d2_vals)
    grid_2d = np.column_stack([d1_grid.ravel(), d2_grid.ravel()])

    # === inverse-map back to latent space and decode ===================

    latent_grid = inverse_mapping(grid_2d)

    print(f"Decoding {n_samples}x{n_samples} = {n_samples**2} grid points...")
    with torch.no_grad():
        latent_tensor = torch.tensor(latent_grid, dtype=torch.float32)
        decoded = ma.network.decode(latent_tensor).cpu().numpy()

    decoded_scaled = decoded * ma.stdval + ma.meanval

    print("Computing DOPE scores...")
    dope_grid = ma.get_all_dope_score(decoded_scaled, refine=refine)
    dope_surface = np.asarray(dope_grid).reshape(n_samples, n_samples)
    # Shift dope_surface so minimum is 0 for pathfinding
    dope_min = dope_surface.min()
    dope_surface_shifted = dope_surface - dope_min

    # === plot ======================================

    cmap_obj = mpl.cm.get_cmap(name=cmap)
    cmap_obj.set_over(cmap_obj(1.0))

    fig, ax = plt.subplots(figsize=(10, 6))

    mesh = ax.pcolormesh(
        _latent_edge(d1_vals),
        _latent_edge(d2_vals),
        dope_surface,
        vmin=dope_surface.min(),
        vmax=truncate_at,
        cmap=cmap_obj,
        shading="auto",
    )

    ax.set_xlim(d1_vals.min(), d1_vals.max())
    ax.set_ylim(d2_vals.min(), d2_vals.max())

    # === overlay datasets =============================

    legend_handles = []
    for encoded, label, colour in encoded_datasets:
        coords_2d = mapping(encoded)
        ax.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=colour,
            s=5,
            alpha=0.6,
            label=label,
        )
        legend_handles.append(
            mpl.lines.Line2D(
                [0],
                [0],
                color=colour,
                linestyle="",
                marker="o",
                markersize=8,
                label=label,
            )
        )

    # === interpolation paths ===========================

    def get_dope_at_point(pt):
        x_idx = np.argmin(np.abs(d1_vals - pt[0]))
        y_idx = np.argmin(np.abs(d2_vals - pt[1]))
        return dope_surface[x_idx, y_idx]

    paths = {}

    if need_paths:
        start_2d = mapping(start_encoded.reshape(1, -1)).squeeze()
        end_2d = mapping(end_encoded.reshape(1, -1)).squeeze()
        dope_start = get_dope_at_point(start_2d)
        dope_end = get_dope_at_point(end_2d)

        if plot_astar:
            idx_start = get_point_index(start_2d, d1_vals, d2_vals)
            idx_end = get_point_index(end_2d, d1_vals, d2_vals)

            astar_coords, cost_so_far_shifted = get_path(
                idx_start, idx_end, dope_surface_shifted, d1_vals, d2_vals, smooth=3
            )

            astar_coords = np.vstack([start_2d, astar_coords, end_2d])
            # Plot A* path as markers at grid centers, and lines between them
            ax.plot(
                astar_coords[:, 0],
                astar_coords[:, 1],
                color="cyan",
                linewidth=2,
                linestyle="--",
                label="A*",
                alpha=0.9,
                zorder=100,
            )
            # Overlay path nodes as dots at grid centers
            ax.scatter(
                astar_coords[:, 0],
                astar_coords[:, 1],
                color="cyan",
                s=40,
                marker="o",
                edgecolor="black",
                linewidth=0.8,
                zorder=101,
                label=None,
            )
            legend_handles.append(
                mpl.lines.Line2D(
                    [0], [0], color="cyan", linewidth=2, linestyle="--", label="A*"
                )
            )

            astar_latent = inverse_mapping(astar_coords)

            with torch.no_grad():
                decoded_astar = (
                    ma.network.decode(torch.tensor(astar_latent, dtype=torch.float32))
                    .cpu()
                    .numpy()
                )

            decoded_astar_scaled = decoded_astar * ma.stdval + ma.meanval
            dope_scores_astar = np.concatenate(
                [[dope_start], cost_so_far_shifted + dope_min, [dope_end]]
            )

            paths["astar"] = {
                "latent": astar_latent,
                "dope": np.asarray(dope_scores_astar),
                "hyper_latent": astar_coords,
                "decoded_scaled": decoded_astar_scaled,
                "decoded": decoded_astar,
            }

        if plot_linear:
            if plot_astar:
                n_interp = len(astar_coords)

            alphas = np.linspace(0, 1, n_interp)
            linear_2d = np.array([(1 - a) * start_2d + a * end_2d for a in alphas])
            ax.plot(
                linear_2d[:, 0],
                linear_2d[:, 1],
                color="red",
                linewidth=2,
                label="Linear",
                alpha=0.9,
                zorder=100,
            )
            legend_handles.append(
                mpl.lines.Line2D([0], [0], color="red", linewidth=2, label="Linear")
            )

            linear_2d = np.vstack([start_2d, linear_2d, end_2d])

            linear_latent = inverse_mapping(linear_2d)

            with torch.no_grad():
                decoded_linear = (
                    ma.network.decode(torch.tensor(linear_latent, dtype=torch.float32))
                    .cpu()
                    .numpy()
                )

            decoded_linear_scaled = decoded_linear * ma.stdval + ma.meanval
            dope_scores_linear = ma.get_all_dope_score(
                decoded_linear_scaled, refine=refine
            )
            np.concatenate([[dope_start], dope_scores_linear, [dope_end]])

            paths["linear"] = {
                "latent": linear_latent,
                "dope": np.asarray(dope_scores_linear),
                "hyper_latent": linear_2d,
                "decoded_scaled": decoded_linear_scaled,
                "decoded": decoded_linear,
            }

        # shared start/end markers
        ax.plot(
            *start_2d,
            marker="o",
            color="white",
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=101,
        )
        ax.plot(
            *end_2d,
            marker="s",
            color="white",
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=101,
        )

    # === plotting crystal transitions ===================================

    if plot_crystals:
        crystal_datasets = [
            (ma.get_encoded("5A5E").numpy(), "5A5E", "gold"),
            (ma.get_encoded("5A5F").numpy(), "5A5F", "gold"),
        ]

        for encoded, label, colour in crystal_datasets:
            coords_2d = mapping(encoded)
            ax.scatter(
                coords_2d[:, 0],
                coords_2d[:, 1],
                c=colour,
                s=120,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                zorder=102,
                label=label,
            )
            legend_handles.append(
                mpl.lines.Line2D(
                    [0],
                    [0],
                    color=colour,
                    linestyle="",
                    marker="*",
                    markersize=12,
                    label=label,
                )
            )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "DOPE Score Surface")
    ax.grid(False)
    ax.set_aspect("equal")

    cbar_ax = fig.add_axes(
        [
            ax.get_position().x1 + 0.02,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    cb = fig.colorbar(mesh, cax=cbar_ax)
    cb.ax.tick_params(left=False, right=True)
    cb.ax.set_ylabel("DOPE score")

    if fname is not None:
        plt.savefig(fname, **savefig_kwargs)
    if show:
        plt.show()

    return dope_surface, d1_vals, d2_vals, fig, ax, paths


def plot_dope_interp_scores(scores: np.ndarray, title: str, outdir) -> None:
    """
    Plot a simple line graph showing how the DOPE scores change along a path.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(scores, marker="o", linestyle="-", color="blue")
    plt.xlabel("Path Index")
    plt.ylabel("DOPE Score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{outdir}/{title}.png")


def plot_path_on_mesh(
    path: list[np.ndarray],
    mapping: Callable[[np.ndarray], np.ndarray],
    ax: mpl.axes.Axes | None = None,
    color: str = "red",
    linewidth: float = 2.0,
    marker_size: int = 8,
    label: str = "Path",
) -> mpl.axes.Axes:
    """Plot a path through latent space on the current DOPE mesh.

    Call this immediately after plot_dope_mesh to overlay the path.

    Parameters
    ----------
    path : list[np.ndarray]
        List of latent codes along the path, each shape (latent_dim,)
    mapping : callable
        (N, latent_dim) -> (N, 2) function (e.g. pca.transform, reducer.transform)
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, uses current axes.
    color : str
        Color for the path line
    linewidth : float
        Width of the path line
    marker_size : int
        Size of start/end markers
    label : str
        Label for the path in the legend

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object
    """
    if not ax:
        ax = plt.gca()

    # Map path points to 2D
    path_array = np.vstack(path)
    path_2d = mapping(path_array)

    # Plot the path as a line
    ax.plot(
        path_2d[:, 0],
        path_2d[:, 1],
        color=color,
        linewidth=linewidth,
        label=label,
        alpha=0.8,
        zorder=100,
    )

    # Mark start and end
    ax.plot(
        path_2d[0, 0],
        path_2d[0, 1],
        marker="o",
        color=color,
        markersize=marker_size,
        markeredgecolor="black",
        markeredgewidth=1.5,
        zorder=101,
    )
    ax.plot(
        path_2d[-1, 0],
        path_2d[-1, 1],
        marker="s",
        color=color,
        markersize=marker_size,
        markeredgecolor="black",
        markeredgewidth=1.5,
        zorder=101,
    )

    # Update legend if there's existing one
    if ax.get_legend():
        ax.legend(loc="upper right")

    return ax


def plot_pairwise_rmsd(
    ma: MolearnAnalysis,
    key: str,
    show_plot: bool,
    latent_dim: str,
    *,
    max_pairs: int = 5000,
    seed: int = 42,
    title: str | None = None,
    fname: str | None = None,
    show=True,
    **savefig_kwargs,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Scatter plot of dataset vs decoded pairwise RMSD with CCC.

    For N conformations there are N(N-1)/2 pairs. When this exceeds
    *max_pairs*, a random subsample of pairs is drawn for plotting.

    Returns (ccc, dataset_pw_rmsd, decoded_pw_rmsd).
    """
    dataset = ma.get_dataset(key, scale=True)  # (N, atoms, 3)
    decoded = ma.get_decoded(key, scale=True)  # (N, atoms, 3)
    n_atoms = dataset.shape[1]

    flat_ds = dataset.reshape(dataset.shape[0], -1).numpy()  # (N, atoms*3)
    flat_dc = decoded.reshape(decoded.shape[0], -1).numpy()

    # pairwise L2 distances, normalised to per-atom RMSD

    pw_ds = pdist(flat_ds, "euclidean") / np.sqrt(n_atoms)
    pw_dc = pdist(flat_dc, "euclidean") / np.sqrt(n_atoms)

    # subsample if too many pairs
    if len(pw_ds) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pw_ds), size=max_pairs, replace=False)
        pw_ds_plot = pw_ds[idx]
        pw_dc_plot = pw_dc[idx]
    else:
        pw_ds_plot = pw_ds
        pw_dc_plot = pw_dc

    ccc = _concordance_correlation(pw_ds_plot, pw_dc_plot)

    # plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pw_ds_plot, pw_dc_plot, s=1, alpha=0.3, rasterized=True)

    lim = max(pw_ds_plot.max(), pw_dc_plot.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r-", linewidth=1)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Dataset pairwise RMSD (Å)")
    ax.set_ylabel("Decoded pairwise RMSD (Å)")
    ax.set_aspect("equal")
    ax.text(0.05, 0.92, f"CCC={ccc:.3f}", transform=ax.transAxes, fontsize=12)

    if key == "test_trans":
        dataset_title = "Test"

    if key == "train_both":
        dataset_title = "Train"

    if key == "astar":
        dataset_title = "A*"

    if key == "linear":
        dataset_title = "Linear"

    ax.set_title(f"Pairwise RMSD {dataset_title}: Latent dim: {latent_dim}")

    plt.tight_layout()
    if fname:
        plt.savefig(fname, **savefig_kwargs)

    if show:
        plt.show()

    return ccc, pw_ds, pw_dc


def _concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    mx, my = x.mean(), y.mean()
    sx, sy = x.var(), y.var()
    sxy = np.mean((x - mx) * (y - my))
    return float(2 * sxy / (sx + sy + (mx - my) ** 2))


# ===============================================
# all base plots
# ===============================================


def all_base_plots(ma, keys, plot_data, latent_dim, run, save_dir, show) -> None:
    rmsd_plot_data = [
        ("train_both", "test_trans", "Train vs Transition", "train", "test"),
    ]

    os.makedirs(save_dir, exist_ok=True)

    plot_rmsd_hist(
        ma,
        plot_data=rmsd_plot_data,
        dpi=150,
        var=latent_dim,
        fname=f"{save_dir}/rmsd_r{run}_v{latent_dim}",
        show=show,
    )

    plot_bondlength_hist(
        ma,
        plot_data=plot_data,
        bins=300,
        dpi=150,
        bond_types=["N-CA"],
        latent_dim=latent_dim,
        fname=f"{save_dir}/wd_r{run}_v{latent_dim}",
        show=show,
    )

    plot_inversion_hist(
        ma,
        plot_data=plot_data,
        latent_dim=latent_dim,
        show_plot=True,
        fname=f"{save_dir}/inversion_r{run}_v{latent_dim}",
        show=show,
    )

    for key in ["train_both", "test_trans"]:
        plot_pairwise_rmsd(
            ma,
            key=key,
            latent_dim=latent_dim,
            show_plot=True,
            title=f"Pairwise RMSD — Latent Dim {latent_dim}",
            max_pairs=5000,
            fname=f"{save_dir}/ccc_{key}_r{run}_v{latent_dim}",
            show=show,
        )

    gc.collect()
    torch.cuda.empty_cache()
