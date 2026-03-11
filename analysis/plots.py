from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist

from molearn.analysis.analyser import MolearnAnalysis


def _latent_edge(values: np.ndarray) -> np.ndarray:
    return np.append(values, (2 * values[-1] - values[-2]))


def plot_dope_mesh(  # noqa: PLR0913
    MA: MolearnAnalysis,  # noqa: N803
    encoded_datasets: Sequence[tuple[np.ndarray, str, str]],
    mapping: Callable[[np.ndarray], np.ndarray],
    inverse_mapping: Callable[[np.ndarray], np.ndarray],
    *,
    n_samples: int = 20,
    margin: float = 0.1,
    refine: bool = True,
    cmap: str = "viridis",
    truncate_percentile: float = 80,
    truncate_at: float | None = None,
    title: str | None = None,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    fname: str | None = None,
    **savefig_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot a 2D DOPE-score mesh for an arbitrary latent -> 2D mapping.

    Parameters
    ----------
    MA : MolearnAnalysis
        Analysis object with a network set and at least one dataset loaded
        (used for decoding grid points and computing DOPE scores).
    encoded_datasets : sequence of (encoded, label, colour)
        Each entry is a tuple of:
        - **encoded** – latent codes, shape ``(N, latent_dim)`` (numpy array).
        - **label** – legend label for this dataset.
        - **colour** – matplotlib colour string.
    mapping : (N, latent_dim) -> (N, 2)
        Forward mapping from latent space to the 2D visualisation space
        (e.g. ``pca.transform``, ``reducer.transform``).
    inverse_mapping : (N, 2) -> (N, latent_dim)
        Inverse mapping from 2D back to the full latent space
        (e.g. ``pca.inverse_transform``, ``reducer.inverse_transform``).
    n_samples : int
        Grid resolution per axis.
    margin : float
        Fractional padding around data extent (0.1 = 10 %).
    refine : bool
        Refine decoded structures before DOPE scoring.
    cmap : str
        Matplotlib colormap name.
    truncate_percentile : float
        Percentile of DOPE values used as the colour-scale upper bound
        (ignored when *truncate_at* is set explicitly).
    truncate_at : float or None
        Explicit upper bound for the colour scale.
    title : str or None
        Plot title.  When ``None`` a generic title is used.
    xlabel, ylabel : str
        Axis labels.
    fname : str or None
        If given, save the figure to this path.
    **savefig_kwargs
        Extra keyword arguments forwarded to ``plt.savefig``.

    Returns
    -------
    dope_surface : np.ndarray, shape (n_samples, n_samples)
    dim1_vals : np.ndarray, shape (n_samples,)
    dim2_vals : np.ndarray, shape (n_samples,)
    """
    # --- project all datasets into 2D to determine grid bounds -----------
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

    # --- inverse-map back to latent space and decode ---------------------
    latent_grid = inverse_mapping(grid_2d)

    print(f"Decoding {n_samples}x{n_samples} = {n_samples**2} grid points...")
    with torch.no_grad():
        latent_tensor = torch.tensor(latent_grid, dtype=torch.float32)
        decoded = MA.network.decode(latent_tensor).cpu().numpy()

    decoded_scaled = decoded * MA.stdval + MA.meanval

    print("Computing DOPE scores...")
    dope_grid = MA.get_all_dope_score(decoded_scaled, refine=refine)
    dope_surface = np.asarray(dope_grid).reshape(n_samples, n_samples)

    # --- colour-scale clipping -------------------------------------------
    if truncate_at is None:
        truncate_at = float(np.percentile(dope_surface, truncate_percentile))

    # --- plot ------------------------------------------------------------
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

    # --- overlay datasets ------------------------------------------------
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
    plt.show()

    return dope_surface, d1_vals, d2_vals


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
    *,
    max_pairs: int = 5000,
    seed: int = 42,
    title: str | None = None,
    fname: str | None = None,
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

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, **savefig_kwargs)

    if show_plot:
        plt.show()

    return ccc, pw_ds, pw_dc


def _concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    mx, my = x.mean(), y.mean()
    sx, sy = x.var(), y.var()
    sxy = np.mean((x - mx) * (y - my))
    return float(2 * sxy / (sx + sy + (mx - my) ** 2))
