import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
from molearn.data import PDBData
from molearn.models.CNN_autoencoder import AutoEncoder
from molearn.trainers import OpenMM_Physics_Trainer
from molearn.utils import convert_dcd_to_pdb


def main():
    ##### Create PDB files #####

    convert_dcd_to_pdb(
        "../data/cleaned_aligned_structure.pdb",
        "../data/aligned_murd_closed_npt_prod_downsampled.dcd",
        "../data/full_murd_closed.pdb",
    )

    convert_dcd_to_pdb(
        "../data/cleaned_aligned_structure.pdb",
        "../data/aligned_murd_open_npt_prod_downsampled.dcd",
        "../data/full_murd_open.pdb",
    )

    for n in [1, 2, 3, 4, 5]:
        latent_dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        for latent_dim in latent_dims:
            ##### Load Data #####
            data = PDBData()
            data.import_pdb(
                ["../data/full_murd_open.pdb", "../data/full_murd_closed.pdb"]
            )
            data.fix_terminal()
            data.atomselect(atoms=["N", "CA", "CB", "C", "O"])
            _ = data.prepare_dataset()
            # data.write_statistics("data_statistics_full.json") # Save mean and std for analysis later

            ##### Prepare Trainer #####
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trainer = OpenMM_Physics_Trainer(device=device)

            trainer.set_data(
                data,
                batch_size=16,
                validation_split=0.1,
                manual_seed=25,
                save_indices=False,  # If True, the training/validation split indices will be saved to disk
            )

            trainer.prepare_physics(remove_NB=True)

            print(f"running CNN autoencoder model with latent_dim: {latent_dim}")
            torch.manual_seed(0)
            model = AutoEncoder(latent_dim=latent_dim, n_atoms=data.dataset.shape[1])
            trainer.set_autoencoder(model, out_points=data.dataset.shape[1])
            trainer.prepare_optimiser()

            ##### Training Loop #####
            # Keep training until loss does not improve for 16 consecutive epochs
            output_dir = os.getenv("OUTPUT_SCRATCH", "./cnn_multi_dim")

            # Train for 8 epochs with physics_inter_weight=0
            trainer.run(
                8,
                log_filename="log.dat",
                log_folder=f"{output_dir}/{n}/{latent_dim}/initial",
                checkpoint_folder=f"{output_dir}/{n}/{latent_dim}/initial",
            )

            # Compute scale and update
            physics_inter_weight = trainer.get_scale(
                ref_loss=trainer.results_epoch["mse_loss"],
                tar_loss=trainer.results_epoch["inter_physics_loss"],
                scale_scale=0.1,
            )

            trainer.update_hyperparameters(physics_inter_weight=physics_inter_weight)

            fit_results = trainer.run_until_converge(
                patience=10,
                log_filename="log.dat",
                log_folder=f"{output_dir}/{n}/{latent_dim}",
                checkpoint_folder=f"{output_dir}/{n}/{latent_dim}",
                verbose=True,
            )

            # Convert log file to 3 significant figures
            log_path = Path(f"{output_dir}/{n}/{latent_dim}/log.dat")
            if log_path.exists():
                df = pd.read_csv(log_path)
                for col in df.columns:
                    if col != "epoch" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].apply(
                            lambda x: float(f"{x:.2e}") if pd.notna(x) else x
                        )
                df.to_csv(log_path, index=False, float_format="%.2e")

            del model
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(fit_results)


if __name__ == "__main__":
    main()
