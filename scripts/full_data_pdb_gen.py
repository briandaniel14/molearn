import sys, os

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))

import MDAnalysis as mda

def convert_dcd_to_pdb(pdb_file, dcd_file, output_file):
    u = mda.Universe(pdb_file, dcd_file)
    u.atoms.write(output_file, frames='all')
    print(f"Wrote {len(u.trajectory)} frames to {output_file}")


def concatenate_pdbs(pdb_files, output_file):
    """Concatenate multiple PDB files into a single file"""
    u_first = mda.Universe(pdb_files[0])
    
    with mda.Writer(output_file, n_atoms=u_first.atoms.n_atoms) as W:
        # Write frames from first file
        for ts in u_first.trajectory:
            W.write(u_first.atoms)
        
        # Write frames from other files
        for pdb_file in pdb_files[1:]:
            u = mda.Universe(pdb_file)
            for ts in u.trajectory:
                W.write(u.atoms)
    
    print(f"Concatenated {len(pdb_files)} files to {output_file}")

ls = ["open", "closed"]

if __name__ == "__main__":

    # for l in ls:
    #     convert_dcd_to_pdb(
    #     "../data/cleaned_aligned_structure.pdb",
    #     f"../data/aligned_murd_closed_open_{l}_npt_prod_downsampled.dcd",
    #     f"../data/full_murd_open_closed_{l}.pdb"
    #     )

    pdb_files = [f"../data/full_murd_{l}.pdb" for l in ls]
    concatenate_pdbs(pdb_files, "../data/full_murd_open_closed_all.pdb")
    print("Concatenation complete!")