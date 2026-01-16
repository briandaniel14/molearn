import sys
import os
import torch
import pandas as pd

from pathlib import Path

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
from molearn.data import PDBData
from molearn.trainers import OpenMM_Physics_Trainer
from molearn.models.CNN_autoencoder import AutoEncoder
from molearn.utils import convert_dcd_to_pdb




def main():

    ##### Create PDB files #####

    convert_dcd_to_pdb(
    "../data/cleaned_aligned_structure.pdb",
    "../data/aligned_murd_closed_npt_prod_downsampled.dcd",
    "../data/full_MurD_closed.pdb"
    )

    convert_dcd_to_pdb(
    "../data/cleaned_aligned_structure.pdb",
    "../data/aligned_murd_open_npt_prod_downsampled.dcd",
    "../data/full_MurD_open.pdb"
    )

    dims = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for d in dims:    

        ##### Load Data #####
        data = PDBData()
        data.import_pdb(
            ["../data/full_MurD_open.pdb", "../data/full_MurD_closed.pdb"]
        )
        data.fix_terminal()
        data.atomselect(atoms=["N", "CA", "CB", "C", "O"])
        dataset = data.prepare_dataset()
        #data.write_statistics("data_statistics_full.json") # Save mean and std for analysis later

        ##### Prepare Trainer #####
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = OpenMM_Physics_Trainer(device=device)

        trainer.set_data(data, 
                        batch_size=16, 
                        validation_split=0.1, 
                        manual_seed=25,
                        save_indices=False     # If True, the training/validation split indices will be saved to disk
                        )
        
        trainer.prepare_physics(remove_NB=True)

        print(f'running CNN autoencoder model with latent dim: {d}')
        torch.manual_seed(0)
        model = AutoEncoder(latent_dim=d)
        trainer.set_autoencoder(model, out_points=data.dataset.shape[1])
        trainer.prepare_optimiser()

        ##### Training Loop #####
        # Keep training until loss does not improve for 16 consecutive epochs
        OUTPUT_DIR = os.getenv('OUTPUT_SCRATCH', './cnn_multi_dim')

        fit_results = trainer.run_until_converge(
            patience=16,
            log_filename=f"log.dat",
            log_folder= f"{OUTPUT_DIR}/{d}",
            checkpoint_folder= f"{OUTPUT_DIR}/{d}",
            verbose=True,
        )

        # Convert log file to 3 significant figures
        log_path = Path(f"{OUTPUT_DIR}/{d}/log.dat")
        if log_path.exists():
            df = pd.read_csv(log_path)
            for col in df.columns:
                if col != 'epoch' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].apply(lambda x: float(f"{x:.2e}") if pd.notna(x) else x)
            df.to_csv(log_path, index=False, float_format='%.2e')

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(fit_results)


if __name__ == "__main__":
    main()
