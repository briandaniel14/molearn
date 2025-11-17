import sys
import os

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
from molearn.data import PDBData
from molearn.trainers import OpenMM_Physics_Trainer
from molearn.models.CNN_autoencoder import AutoEncoder
import torch


def main():
    ##### Load Data #####
    data = PDBData()
    data.import_pdb(
        ["./data/MurD_open.pdb", "./data/MurD_closed.pdb"]
    )
    data.fix_terminal()
    data.atomselect(atoms=["N", "CA", "CB", "C", "O"])
    dataset = data.prepare_dataset()
    data.write_statistics("data_statistics.json") # Save mean and std for analysis later

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

    dims = [2, 3]
    
    for d in dims:
        torch.manual_seed(0)
        model = AutoEncoder()
        trainer.set_autoencoder(model, out_points=data.dataset.shape[1])
        trainer.prepare_optimiser()

        ##### Training Loop #####
        # Keep training until loss does not improve for 16 consecutive epochs

        fit_results = trainer.run(
            epochs=1,
            log_filename=f"log.dat{d}",
            log_folder= f"foldingnet_checkpoints{d}",
            checkpoint_folder= f"foldingnet_checkpoints{d}",
            verbose=True,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(fit_results)


if __name__ == "__main__":
    main()
