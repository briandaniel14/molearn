#! To be run from scripts/molearn

if __name__ == "__main__":

    import sys
    import os
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import pandas as pd

    sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
    from molearn.data import PDBData
    from molearn.models.CNN_autoencoder import AutoEncoder
    from molearn.analysis import MolearnAnalysis
    from molearn.analysis.plot import *

    import torch


    atoms = ["N", "CA", "C", "O", 'CB']

    mean = json.load(open("../data/data_statistics.json"))["mean"]
    std = json.load(open("../data/data_statistics.json"))["std"]

    data_closed = PDBData(
        filename=["../data/MurD_closed.pdb"],
        atoms=atoms,
        fix_terminal=True,
    )
    _ = data_closed.prepare_dataset(mean=mean, std=std)

    data_open = PDBData(
        filename=["../data/MurD_open.pdb"],
        atoms=atoms,
        fix_terminal=True,
    )
    _ = data_open.prepare_dataset(mean=mean, std=std)

    data_test = PDBData(
        filename=["../data/MurD_closed_apo.pdb"],
        atoms=atoms,
        fix_terminal=True,
    )
    _ = data_test.prepare_dataset(mean=mean, std=std)

    log_dfs = {}
    min_valid_losses = {}
    min_valid_mse_losses = {}
    models = {}

    base_path = r"../results/20251128_172209/cnn_multi_dim"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for n in range(2, 11):

        log_dfs[n] = pd.read_csv(Path(f"{base_path}/{n}/log.dat"))
        best_epoch = log_dfs[n]['valid_loss'].idxmin()
        min_valid_losses[n] = min(log_dfs[n].valid_loss)
        min_valid_mse_losses[n] = min(log_dfs[n].valid_mse_loss)

        checkpoint_path = Path(f"{base_path}/{n}/checkpoint_converged.ckpt")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model = AutoEncoder(latent_z=n)
        model.load_state_dict(checkpoint['model_state_dict'])

        models[n] = model

    MA = MolearnAnalysis()
    MA.batch_size = 16
    MA.processes = 4

    MA.set_dataset(data=data_open, key="train_open")
    MA.set_dataset(data=data_closed, key="train_closed")
    MA.set_dataset(data=data_test, key="test_trans")

    MA.set_network(models[2])

        # We first setup a latent grid whose boundaries encompass all datasets.
    if "grid" not in MA._encoded:
        grid_key = MA.setup_grid(samples=15)
        print(f"Latent grid '{grid_key}' initialised with {MA.n_samples} samples per axis.")
    else:
        grid_key = "grid"
        print("Re-using previously initialised latent grid.")

    # Next, we compute the DOPE surface over the latent grid.
    dope_surface, xvals, yvals = MA.scan_dope(refine=False)
    surface_clip = np.percentile(dope_surface, 80)

    dope_plot_data = [
        ("train_open", "Train Open", "#FDBFCA", "scatter"),
        ("train_closed", "Train Closed", "#AFC2DC", "scatter"),
        ("test_trans", "Test Transition", "#7FB069", "scatter"),
    ]

    plot_dope_surface(
        MA,
        refine=False,
        truncate_at=surface_clip,
        plot_data=dope_plot_data,
        fname='../figures/dope_surface_refined.png',
        cmap="viridis",
        bbox_inches="tight",
    )

